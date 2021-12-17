import os
import pdb
import time
from collections import defaultdict

import torch
from torch.nn.functional import one_hot

import model as m


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../../data/processed")


def train1(data_path=DATA_DIR):

    # Load a dataset
    data = m.LengthGroupedDataset(os.path.join(data_path, "sentences.index"))
    data.read()
    dictionary = m.Dictionary(os.path.join(data_path, "tokens.dict"))

    # Initialize model
    embedding_dimension = 500
    embedding = m.EmbeddingLayer(len(dictionary), embedding_dimension)
    model = LanguageModel(embedding)

    # Draw a positive batch from the dataset
    for tokens_batch, heads_batch, relations_batch in data:

        sample_parses = model.sample_parses(tokens_batch)
        parse_energy = embedding.parse_energy(tokens_batch, sample_parses)

        # resample tokens and get a negative sample and measure its energy
        # and calculate loss...


def train_w2v(data_path=DATA_DIR):

    lr = 1e-2
    num_epochs = 5
    batch_size = 1000

    # Load a dataset
    #data = m.LengthGroupedDataset(os.path.join(data_path, "sentences.index"))
    sentences_path = os.path.join(data_path, "sentences.index")
    data = m.PaddedDataset(
        sentences_path, padding=-1, batch_size=batch_size, min_length=3)
    data.read()
    dictionary = m.Dictionary(os.path.join(data_path, "tokens.dict"))

    # Initialize model
    embedding_dimension = 500
    embedding = m.EmbeddingLayer(len(dictionary), embedding_dimension)
    model = Word2VecModel(embedding, data.Px)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=lr)

    # Draw a positive batch from the dataset
    m.timer.start()
    for epoch in range(num_epochs):
        epoch_loss = torch.tensor(0.)

        for tokens_batch, _, _  in data:

            # Don't allow 1-token sentences (which have length 2 due to <ROOT>)
            if tokens_batch.shape[1] <= 2:
                continue

            with torch.no_grad():
                m.timer.log("setup")
                positive_heads = model.sample_parses(tokens_batch)
                negative_heads = model.sample_tokens(
                    tokens_batch, positive_heads)
                m.timer.log("sample")

            mask = (tokens_batch == -1)
            mask[:,0] = True

            loss = -torch.where(
                mask,
                torch.tensor([[0]], dtype=torch.float),
                embedding.link_energy(tokens_batch, positive_heads) -
                embedding.link_energy(tokens_batch, negative_heads) 
            ).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.timer.log("grad")

            with torch.no_grad():
                epoch_loss += loss/(tokens_batch.shape[0]*tokens_batch.shape[1])

        print(m.timer.repr())

        words = ["recently", "ran", "Paris", "debt", "doctor", "is"]
        ids = dictionary.get_ids(words)
        with torch.no_grad():
            heads = dictionary.get_tokens([
                embedding.head_match(idx) for idx in ids])
            similar = dictionary.get_tokens([
                embedding.similar(idx) for idx in ids])

        print("\n")
        print(embedding.U[ids[0],:3], embedding.Ubias[ids[0]])
        print(epoch_loss)
        print(words)
        print(heads)
        print(similar)

    print(m.timer.repr())
    m.timer.write("timer", "w2v-padding-batch1000")
    return model



def print_tree(headlist, curnode=0, depth=0):
    print('  ' * depth + str(curnode))
    children = [
        i for i, val in enumerate(headlist) 
        if val == curnode
    ]
    for child in children:

        # Node 0 is root and is self-linked.  Skip it as a child.
        if child == 0:
            continue

        print_tree(headlist, curnode=child, depth=depth+1)



class Word2VecModel:

    def __init__(self, embedding_layer, Px):
        self.embedding = embedding_layer
        self.kernel = [torch.tensor(v) for v in [0,5,4,3,2,1]]
        self.set_Px(Px)
        self.unigram_sampler = torch.distributions.Categorical(self.Px)


    def set_Px(self, Px):
        self.Px = Px
        # We never sample the <ROOT> token.
        self.Px[0] = 0


    def get_head_probs(self, num_tokens):
        head_probs = torch.zeros((num_tokens, num_tokens))
        for i,v in enumerate(self.kernel):
            # Handle sentences shorter than kernel properly.
            if i > num_tokens:
                break
            diag = torch.full((num_tokens-i,), v).diagflat(i)
            head_probs += diag
        head_probs += head_probs.T.clone()
        head_probs[:,0] = 0
        head_probs[0] = one_hot(torch.tensor(0), num_tokens)
        return head_probs


    def sample_parses(self, tokens_batch):
        num_sentences, num_tokens = tokens_batch.shape

        # Construct a head selector, which provides the probability
        # of a token in position i selecting a token in position j as it's
        # head.  For w2v, this is like a convolution; choosing tokens nearby
        # is most likely and is zero beyond the kernel.
        head_probs = self.get_head_probs(num_tokens)
        head_probs.unsqueeze(0).expand(500,-1,-1)

        # Tokens should not choose padding as head.
        head_probs = torch.where(
            tokens_batch.unsqueeze(1)==-1,
            torch.tensor([[[0]]], dtype=torch.float),
            head_probs.unsqueeze(0)
        )

        # Padding should always choose <ROOT> as head.
        head_probs = torch.where(
            tokens_batch.unsqueeze(2)==-1,
            one_hot(torch.tensor(0), num_tokens).to(torch.float).unsqueeze(0),
            head_probs
        )

        head_selector = torch.distributions.Categorical(head_probs)

        # Sample pointers that indicate in each sentence and at each token
        # position, what position was selected to act as that token's head
        head_pointers = torch.where(
            tokens_batch == -1,
            torch.tensor([[0]]),
            head_selector.sample()
        )

        # Convert head pointers into actual token ids (was positions).
        heads = tokens_batch.gather(dim=1, index=head_pointers)

        return heads


    def sample_tokens(self, tokens_batch, prev_heads):
        num_sentences, num_tokens = tokens_batch.shape

        # Sample 1 head for each word in each sentence.
        next_heads = self.unigram_sampler.sample(tokens_batch.shape)

        # Calculate link energies for initial heads batch and for the next.
        prev_weights = self.embedding.link_energy(tokens_batch, prev_heads)
        next_weights = self.embedding.link_energy(tokens_batch, next_heads)

        proposal_weights = (self.Px[prev_heads]/self.Px[next_heads])

        accept_score = proposal_weights * torch.exp(next_weights - prev_weights)
        reject_score = torch.rand(tokens_batch.shape)
        accept_proposal = accept_score > reject_score

        # <ROOT> and padding always reject, keeping <ROOT> as their head
        accept_proposal = torch.where(
            tokens_batch == -1,
            torch.tensor([[False]]),
            accept_proposal
        )
        accept_proposal[:,0] = False

        next_heads = torch.where(accept_proposal, next_heads, prev_heads)
        return next_heads



class LanguageModel:

    def __init__(self, embedding_layer): 
        self.embedding = embedding_layer

    def sample_parses(self, tokens_batch):
        #return m.sp.sample_parses(self, tokens_batch)
        #return m.sp.sample_parses_new(self, tokens_batch)
        sampler = m.sp.ContentionResampler()
        return sampler.sample(tokens_batch, self.embedding)

    def gibbs_step_tokens():
        # Resample some of the words
        pass



def view_dataset(data_path=DATA_DIR):
    data = m.SentenceDataset(os.path.join(data_path, "sentences.index"))
    data.read()
    token_dictionary = m.Dictionary(os.path.join(data_path, "tokens.dict"))
    relation_dictionary = m.Dictionary(os.path.join(data_path, "relations.dict"))

    pdb.set_trace()
    for tokens_batch, heads_batch, relations_batch in data:
        for i in range(tokens_batch.shape[0]):
            token_ids = tokens_batch[i,:]
            tokens = token_dictionary.get_tokens(token_ids)
            relation_ids = relations_batch[i,:]
            relations = relation_dictionary.get_tokens(relation_ids)
            print(tokens)
            print(heads_batch[i,:])
            print(relations)
            pdb.set_trace()




if __name__ == "__main__":
    sd = m.SentenceDataset("../data/processed/en_ewt-ud-train/sentences.index")
    sd.read()
