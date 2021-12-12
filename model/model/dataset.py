from collections import defaultdict, Counter
import pdb

import torch
from torch.utils.data import Dataset

import model as m



def read_conllu(path):
    with open(path, 'r') as infile:
        return read_conllu_(infile)


def read_conllu_(infile):

    Nx = Counter()

    # We'll need to sort sentences together by length in order minimize 
    # padding.  Use a defaultdict to index lists of sentences by length.
    token_chunks = defaultdict(list)
    head_chunks = defaultdict(list)
    relation_chunks = defaultdict(list)
    for line in infile:

        # split into three parts
        tokens, heads, relations = line.split(";")

        # Extract integer ids for each symbol in the three parts
        tokens = [int(tokenId) for tokenId in tokens.split(",")]
        heads = [int(headPos) for headPos in heads.split(",")]
        relations = [int(relationId) for relationId in relations.split(",")]

        # Keep count of number of times every token appears
        for token in tokens:
            Nx[token] += 1

        # The tokens, heads, and relations are parallel arrays with each
        # entry describing a words relationship to the sentence.
        # gather the information on a per-word basis:
        token_chunks[len(tokens)].append(tokens)
        head_chunks[len(heads)].append(heads)
        relation_chunks[len(relations)].append(relations)

    # Convert token counts to tensor of unigram probabilities
    Nx_token_ids = torch.tensor(list(Nx.keys()))
    Nx_counts = torch.tensor(list(Nx.values()))
    vocabulary_size = max(Nx.keys()) + 1 # this is better than len() since
    # some tokens that were in in our dictionary may not appear in data.
    Px = torch.zeros(vocabulary_size, dtype=torch.long)
    Px[Nx_token_ids] = Nx_counts
    Px = Px / Px.sum()

    return token_chunks, head_chunks, relation_chunks, Px



class LengthGroupedDataset(Dataset):

    def __init__(self, path):
        self.path = path


    def read(self):
        token_chunks, head_chunks, relation_chunks, Px = read_conllu(self.path)
        self.Px = Px

        # Tensorify the chunks grouping tokens, heads, and labels together
        # based on sentence length
        self.data = [
            (
                torch.tensor(token_chunks[length]),
                torch.tensor(head_chunks[length]), 
                torch.tensor(relation_chunks[length])
            )
            for length in token_chunks.keys()
        ]


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


class PaddedDataset(Dataset):


    def __init__(self, path, padding, batch_size=500, min_length=0):
        self.path = path
        self.padding = padding
        self.batch_size = batch_size
        self.min_length = min_length


    def read(self):
        token_chunks, head_chunks, relation_chunks, Px = read_conllu(self.path)
        self.Px = Px
        self.data = []

        tokens_batch, heads_batch, relations_batch = [],[],[]
        for length in sorted(token_chunks.keys()):
            for i in range(len(token_chunks[length])):
                if length < self.min_length:
                    continue

                tokens_batch.append(token_chunks[length][i])
                heads_batch.append(head_chunks[length][i])
                relations_batch.append(relation_chunks[length][i])

                if len(tokens_batch) == self.batch_size:
                    self.pad_append(
                        length, tokens_batch, heads_batch, relations_batch)
                    tokens_batch, heads_batch, relations_batch = [], [], []

        if len(tokens_batch) > 0:
            self.pad_append(
                length, tokens_batch, heads_batch, relations_batch)


    def pad_append(self, length, tokens_batch, heads_batch, relations_batch):
        self.data.append((
            torch.tensor(self.pad(length, tokens_batch)),
            torch.tensor(self.pad(length, heads_batch)),
            torch.tensor(self.pad(length, relations_batch))
        ))


    def pad(self, length, symbols_batch):
        return [
            symbols + [self.padding] * (length - len(symbols))
            for symbols in symbols_batch
        ]


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


