import os
import pdb
import math
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset
import model as m



class PaddedDatasetParallel(Dataset):

    def __init__(self,
            path,
            padding=0,
            min_length=0,
            max_length=140,
            has_heads=False,
            has_relations=False,
            approx_chunk_size=100*m.const.KB,
            vocab_limit=None
    ):

        self.path = path
        self.padding = padding
        self.min_length = min_length
        self.max_length = max_length
        self.has_heads = has_heads
        self.has_relations = has_relations
        self.vocab_limit = vocab_limit

        # Build the dictionaries
        self.dictionary = m.Dictionary(
            self.tokens_path(), vocab_limit=vocab_limit)
        self.Nx = self.calculate_Nx()
        self.relations_dictionary = None
        if has_relations:
            self.relations_dictionary = m.Dictionary(self.relations_path())

        # work out number of chunks needed to give approx desired chunk_size.
        total_bytes = os.path.getsize(self.sentences_path())
        self.num_chunks = math.ceil(total_bytes / approx_chunk_size)


    def calculate_Nx(self):

        Nx = torch.tensor(self.dictionary.counts)

        # If vocab_limit is an integer, then we'll slice out only frequencies
        # up to vocab_limit.  Attribute cut words to counts of "<UNK>".
        if self.vocab_limit is not None:
            Nx_limited = Nx[:self.vocab_limit]
            Nx_cut_weight = Nx.sum() - Nx_limited.sum()
            Nx_limited[self.dictionary.get_id("<UNK>")] += Nx_cut_weight
            return Nx_limited

        return Nx

    def tokens_path(self):
        return os.path.join(self.path, "tokens.dict")

    def relations_path(self):
        return os.path.join(self.path, "relations.dict")

    def sentences_path(self):
        return os.path.join(self.path, "sentences.index")


    def read(self, chunk_num):
        file_chunk = m.file.open_chunk(
            self.sentences_path(), chunk_num, self.num_chunks)
        return self.prepare_chunk(file_chunk)


    def prepare_chunk(self, lines):
        """
        Given an iterable source of lines (e.g. a file or file chunk)
        parse the sentence data therein, and generate tensorfied batches
        of sentence data.
        """
        # parse the chunks into lists of variably-sized lists of symbols.
        parsed_batch = self.parse(lines)

        # Split the batch based on sentence length.
        short_batch, long_batch = self.split_batch(parsed_batch)

        # Tensorify the sub-batches, and create a mask.
        finished_batches = []
        for batch in (short_batch, long_batch):
            if len(batch[0]) == 0:
                mask = []
                finished_batches.append(batch + [mask])
                continue

            # Assumes sorted by increasing length.
            max_length = len(batch[0][-1])
            finished_batch = [
                self.pad_tensorfy(symbol_batch, max_length) 
                for symbol_batch in batch
            ]
            finished_batch.append(self.get_mask(batch[0], max_length))
            finished_batches.append(finished_batch)

        return finished_batches


    def sort_split(self, symbols_batch, sorter=None, split_at=0.95):
        """
        Given a symbols_batch, sort it by length, or by `sorter` if provided,
        and then split the batch into short and long, by splitting at the
        95th percentile of sentence lengths.  The symbols batch will be returned
        length-sorted or sorted by `sorter`.  Sorter is a permutation, a list
        of indices, whose ith position holds the index of the sentence that
        shoud appear in position i.
        """
        cutoff = int(math.ceil(0.95 * len(symbols_batch)))
        if sorter is None:
            lengths = [
                (len(tokens), i) for i, tokens in enumerate(symbols_batch)]
            lengths.sort()
            sorter = [i for length, i in lengths]
        sorted_symbols_batch = [symbols_batch[i] for i in sorter]
        split_symbol_batch = (
            sorted_symbols_batch[:cutoff],
            sorted_symbols_batch[cutoff:]
        )
        return split_symbol_batch, sorter



    def split_batch(self, batch):
        """
        Given batch, a tuple of various types of symbol-batches, generate
        two batches, by breaking each of the symbol-batches in two.
        Don't split the symbol batches evenly, sort them by 
        sentence-length, and then split into short and long, where long is the
        top 5% by length of sentences, and short has the bulk of the sentences.
        """
        # Split each of the symbols batches based on sentence length, and 
        # group them into separate batches.
        sorter = None
        short_batch, long_batch = [],[]
        for symbols_batch in batch:
            if symbols_batch is None:
                short_batch.append([])
                long_batch.append([])
            else:
                split_batch, sorter = self.sort_split(symbols_batch, sorter)
                short_batch.append(split_batch[0])
                long_batch.append(split_batch[1])
        return (short_batch, long_batch)


    def parse(self, lines):
        """
        Given an iterable of lines (e.g. a file or file chunk), parse the 
        lines into a series of three symbol-batches.
        """
        tokens_batch = []
        head_ptrs_batch = [] if self.has_heads else None
        relations_batch = [] if self.has_relations else None
        for line in lines:
            tokens, head_ptrs, relations = self.parse_line(line)
            length = len(tokens)
            if length < self.min_length or length > self.max_length:
                continue
            tokens_batch.append(tokens)
            if self.has_heads:
                head_ptrs_batch.append(head_ptrs)
            if self.has_relations:
                relations_batch.append(relations)
        return tokens_batch, head_ptrs_batch, relations_batch


    def get_mask(self, batch, pad_to):
        """
        Given an example symbol_batch having sentences of different length,
        generate the bolean mask which will be true wherever padding cells
        are added to tensorfy batch by padding up to pad_to.
        """
        mask = torch.ones((len(batch), pad_to), dtype=torch.bool)
        for i in range(len(batch)):
            mask[i][:len(batch[i])] = False
        return mask


    def pad_tensorfy(self, batch, pad_to):
        """
        Given a symbol batch, convert it into a tensor, by padding every 
        sentence up to pad_to length.
        """
        pad_batch = torch.empty((len(batch), pad_to), dtype=torch.long)
        for i in range(len(batch)):
            pad_length_needed = pad_to - len(batch[i])
            pad_needed = [self.padding] * pad_length_needed
            pad_batch[i] = torch.tensor(batch[i] + pad_needed)
        return pad_batch


    def parse_line(self, line):
        """
        Given a single line of text (a string), parse out the token ids, 
        head-assignments, and relation ids.  Token ids outside of vocab_limit
        are converted to the id corresponding to the "<UNK>" token.
        """
        parsed_line = line.split(";")
        tokens = [int(t) for t in parsed_line[0].split(",")]

        # If vocab_limit is set, convert high token_ids to the id for <UNK>.
        if self.vocab_limit is not None:
            unk_id = self.dictionary.get_id("<UNK>")
            tokens = [t if t < self.vocab_limit else unk_id for t in tokens]

        heads = None
        if self.has_heads:
            heads = [int(headPos) for headPos in parsed_line[1].split(",")]

        relations = None
        if self.has_relations:
            relations = [
                int(relationId) for relationId in parsed_line[2].split(",")]

        return tokens, heads, relations


    def __getitem__(self, idx):
        try:
            return self.read(idx), idx
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


