import pdb
import model as m
import torch
from torch.nn.functional import one_hot


class EdgeModel:

    def __init__(self, vocab_size, embed_dim, Px):
        self.embedding = m.EmbeddingLayer(vocab_size, embed_dim)
        self.token_sampler = m.sp.ParityTokenSampler(Px)
        self.parse_sampler = m.sp.CycleProofRootingParseSampler()

    @staticmethod
    def load(path, Px):
        params = torch.load(path)
        vocab_size, embed_dim = params['U'].shape
        model = EdgeModel(vocab_size, embed_dim, Px)
        model.embedding.U = params['U']
        model.embedding.V = params['V']
        model.embedding.Ubias = params['Ubias']
        model.embedding.Vbias = params['Vbias']
        return model

    def sample_parses(self, tokens_batch, mask, start_temp=1, temp_step=0.001):
        return self.parse_sampler.sample_parses(
            tokens_batch, self.embedding, mask, start_temp, temp_step)

    def sample_tokens(self, tokens_batch, head_ptrs, mask):
        return self.token_sampler.sample_tokens(
            tokens_batch, head_ptrs, self.embedding, mask)


    # TODO: Should this just be an actual subclass of Model?
    def parameters(self):
        return self.embedding.parameters()


    def get_loss(self, tokens_batch, mask):

        num_sentences, num_tokens = tokens_batch.shape

        with torch.no_grad():
            head_ptrs = self.sample_parses(tokens_batch, mask)
            heads = tokens_batch.gather(dim=1, index=head_ptrs)
            mutated_tokens = self.sample_tokens(tokens_batch, head_ptrs, mask)
            head_ptrs_exp = head_ptrs.unsqueeze(1).expand(
                num_sentences, 2, num_tokens)
            mutated_heads = mutated_tokens.gather(dim=2, index=head_ptrs_exp)

        positive_energy = self.embedding.link_energy(tokens_batch, heads)
        negative_energy = self.embedding.link_energy(
                mutated_tokens, mutated_heads).sum(dim=1) / 2

        loss = -torch.where(
            mask, 
            torch.tensor([[0]], dtype=torch.float),
            positive_energy - negative_energy
        ).sum()

        return loss



class FlatModel:

    def __init__(self, vocab_size, embed_dim, Px):
        self.embedding = m.EmbeddingLayer(vocab_size, embed_dim)
        self.kernel = [torch.tensor(v) for v in [0,5,4,3,2,1]]
        self.set_Px(Px)
        self.unigram_sampler = torch.distributions.Categorical(self.Px)


    def set_Px(self, Px):
        self.Px = Px
        # We never sample the <ROOT> token.
        self.Px[0] = 0


    def parameters(self):
        return self.embedding.parameters()


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
        #heads = tokens_batch.gather(dim=1, index=head_pointers)

        return head_pointers


    def sample_tokens(self, tokens_batch, prev_head_ptrs):
        num_sentences, num_tokens = tokens_batch.shape
        prev_heads = tokens_batch.gather(dim=1, index=prev_head_ptrs)

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


    def get_loss(self, tokens_batch, mask):
        with torch.no_grad():
            positive_head_ptrs = self.sample_parses(tokens_batch)
            negative_heads = self.sample_tokens(
                tokens_batch, positive_head_ptrs)
            positive_heads = tokens_batch.gather(
                dim=1, index=positive_head_ptrs)

        # Mask off ROOT, we don't want it's energy in the model.
        mask[:,0] = True
        loss = -torch.where(
            mask,
            torch.tensor([[0]], dtype=torch.float),
            self.embedding.link_energy(tokens_batch, positive_heads, mask) -
            self.embedding.link_energy(tokens_batch, negative_heads, mask) 
        ).sum()
        return loss

