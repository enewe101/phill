import os
import pdb
import math
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset
import model as m


def parse_sentences_by_length(path):
    with open(path, 'r') as infile:
        return parse_sentences_by_length_(infile)


def parse_sentences_by_length_(lines_generator):
    """
    lines_generator could be a file open for reading, or list of strings.
    anything that is iterable and yields strings.
    """

    # We'll need to sort sentences together by length in order minimize 
    # padding.  Use a defaultdict to index lists of sentences by length.
    token_chunks = defaultdict(list)
    head_chunks = defaultdict(list)
    relation_chunks = defaultdict(list)
    for line in lines_generator:

        tokens, heads, relations = parse_line(line)

        # The tokens, heads, and relations are parallel arrays with each
        # entry describing a words relationship to the sentence.
        # gather the information on a per-word basis:
        token_chunks[len(tokens)].append(tokens)
        head_chunks[len(heads)].append(heads)
        relation_chunks[len(relations)].append(relations)

    return token_chunks, head_chunks, relation_chunks


class LengthGroupedDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dictionary = None
        self.read()


    def tokens_path(self):
        return os.path.join(self.path, "tokens.dict")

    def sentences_path(self):
        return os.path.join(self.path, "sentences.index")

    def read(self):

        self.dictionary = m.Dictionary(self.tokens_path())
        chunks = parse_sentences_by_length(self.sentences_path())
        token_chunks, head_chunks, relation_chunks = chunks

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

    def __init__(self, path, padding=0, batch_size=500, min_length=0):
        self.path = path
        self.padding = padding
        self.batch_size = batch_size
        self.min_length = min_length
        self.dictionary = None
        self.read()

    def tokens_path(self):
        return os.path.join(self.path, "tokens.dict")

    def sentences_path(self):
        return os.path.join(self.path, "sentences.index")

    def read(self):

        self.dictionary = m.Dictionary(self.tokens_path())
        chunks = parse_sentences_by_length(self.sentences_path())
        token_chunks, head_chunks, relation_chunks = chunks
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
        padding_mask = self.get_padding_mask(length, tokens_batch)
        self.data.append((
            torch.tensor(self.pad(length, tokens_batch)),
            torch.tensor(self.pad(length, heads_batch)),
            torch.tensor(self.pad(length, relations_batch)),
            torch.tensor(self.get_padding_mask(length, tokens_batch))
        ))


    def get_padding_mask(self, length, symbols_batch):
        return [
            [False] * len(symbols) + [True] * (length - len(symbols))
            for symbols in symbols_batch
        ]


    def pad(self, length, symbols_batch):
        return [
            symbols + [self.padding] * (length - len(symbols))
            for symbols in symbols_batch
        ]


    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


class PaddedDatasetParallel(Dataset):

    def __init__(self,
            path,
            padding=0,
            min_length=0,
            has_heads=False,
            has_relations=False,
            approx_chunk_size=100*m.const.KB
    ):

        self.path = path
        self.padding = padding
        self.min_length = min_length
        self.dictionary = m.Dictionary(self.tokens_path())
        self.Px = self.calculate_Px()
        self.has_heads = has_heads
        self.has_relations = has_relations

        self.relations_dictionary = None
        if has_relations:
            self.relations_dictionary = m.Dictionary(self.relations_path())

        # work out number of chunks needed to give approx desired chunk_size.
        total_bytes = os.path.getsize(self.sentences_path())
        self.num_chunks = math.ceil(total_bytes / approx_chunk_size)

    def calculate_Px(self):
        Nx = torch.tensor(self.dictionary.counts)
        return Nx / Nx.sum()

    def tokens_path(self):
        return os.path.join(self.path, "tokens.dict")

    def relations_path(self):
        return os.path.join(self.path, "relations.dict")

    def sentences_path(self):
        return os.path.join(self.path, "sentences.index")


    def read(self, chunk_num):
        file_chunk = m.file.open_chunk(
            self.sentences_path(), chunk_num, self.num_chunks)
        return self.parse_pad_tensorfy(file_chunk)


    def parse_pad_tensorfy(self, lines):

        # parse the chunks into lists of variably-sized lists of symbols.
        parsed_batch = self.parse(lines)
        tokens_batch, head_ptrs_batch, relations_batch, max_length = parsed_batch

        # Add padding and tensorfy the data.
        pad_tokens_batch = self.pad_tensorfy(tokens_batch, max_length)
        mask = self.get_mask(tokens_batch, max_length)
        pad_head_ptrs_batch = None
        if self.has_heads:
            pad_head_ptrs_batch = self.pad_tensorfy(
                head_ptrs_batch, max_length)
        pad_relations_batch = None
        if self.has_relations:
            pad_relations_batch = self.pad_tensorfy(
                relations_batch, max_length)

        return pad_tokens_batch, pad_head_ptrs_batch, pad_relations_batch, mask


    def parse(self, lines):
        tokens_batch = []
        head_ptrs_batch = [] if self.has_heads else None
        relations_batch = [] if self.has_relations else None
        max_length = 0
        for line in lines:
            tokens, head_ptrs, relations = self.parse_line(line)
            length = len(tokens)
            if length < self.min_length:
                continue
            max_length = max(max_length, length)
            tokens_batch.append(tokens)
            if self.has_heads:
                head_ptrs_batch.append(head_ptrs)
            if self.has_relations:
                relations_batch.append(relations)
        return tokens_batch, head_ptrs_batch, relations_batch, max_length


    def get_mask(self, batch, pad_to):
        mask = torch.ones((len(batch), pad_to), dtype=torch.bool)
        for i in range(len(batch)):
            mask[i][:len(batch[i])] = False
        return mask


    def pad_tensorfy(self, batch, pad_to):
        pad_batch = torch.empty((len(batch), pad_to), dtype=torch.long)
        for i in range(len(batch)):
            pad_length_needed = pad_to - len(batch[i])
            pad_needed = [self.padding] * pad_length_needed
            pad_batch[i] = torch.tensor(batch[i] + pad_needed)
        return pad_batch


    def parse_line(self, line):

        parse = line.split(";")
        tokens = parse[0]
        tokens = [int(tokenId) for tokenId in tokens.split(",")]

        heads = None
        if self.has_heads:
            heads = parse[1]
            heads = [int(headPos) for headPos in heads.split(",")]

        relations = None
        if self.has_relations:
            relations = parse[2]
            relations = [int(relationId) for relationId in relations.split(",")]

        return tokens, heads, relations


    def __getitem__(self, idx):
        try:
            return self.read(idx)
        except ValueError:
            raise IndexError(f"No such index in dataset: {idx}")


    def __len__(self):
        return self.num_chunks


def view_dataset(
    data_path=m.const.DEFAULT_GOLD_DATA_DIR,
    has_heads=True,
    has_relations=True
):
    data = m.PaddedDatasetParallel(
        data_path,
        approx_chunk_size=1024,
        has_heads=has_heads,
        has_relations=has_relations
    )
    for tokens_batch, head_ptrs_batch, relations_batch, mask in data:
        for i in range(tokens_batch.shape[0]):
            token_ids = tokens_batch[i,:]
            tokens = data.dictionary.get_tokens(token_ids)
            print(tokens)
            if has_heads:
                heads = data.dictionary.get_tokens(token_ids[head_ptrs_batch[i]])
                print(heads)
            if has_relations:
                relations = data.relations_dictionary.get_tokens(
                    relations_batch[i])
                print(relations)
            pdb.set_trace()


