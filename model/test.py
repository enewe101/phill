import os
import sys
import pdb
import time
import random
import itertools as it
from unittest import TestCase, main, skip
from collections import Counter

import numpy as np
import torch

import model as m


PADDING = 0
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test-data")

def seed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DatasetTest(TestCase):

    def read_expected_sentences(self, path):
        expected_sentences = set()
        with open(path) as f:
            for line in f:
                if line.strip() == "":
                    continue
                expected_sentences.add(line.strip())
        return expected_sentences


    def test_length_grouped_dataset(self):
        path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/sentences.index")
        dataset = m.LengthGroupedDataset(path)
        dataset.read()

        found_sentences = set()
        for tokens_batch, heads_batch, relations_batch in dataset:
            for i in range(tokens_batch.shape[0]):
                found_sentences.add(";".join([
                    ",".join([str(t) for t in symbols.tolist()])
                    for symbols in 
                    (tokens_batch[i], heads_batch[i], relations_batch[i])
                ]))

        expected_sentences = self.read_expected_sentences(path)
        self.assertTrue(found_sentences == expected_sentences)


    def test_padded_dataset(self):
        path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/sentences.index")
        batch_size = 7
        dataset = m.PaddedDataset(path, PADDING, batch_size)
        dataset.read()

        found_sentences = set()
        for items  in dataset:

            # Wherever mask is 1, the batches should be padding.
            tokens_batch, head_ptrs_batch, relations_batch, mask_batch = items
            torch.all(tokens_batch[mask_batch] == PADDING)
            torch.all(head_ptrs_batch[mask_batch] == PADDING)
            torch.all(relations_batch[mask_batch] == PADDING)

            # Wherever mask is 0, the batches should contain the expected data.
            for i in range(tokens_batch.shape[0]):
                mask = mask_batch[i].tolist()
                tokens = tokens_batch[i].tolist()
                heads = head_ptrs_batch[i].tolist()
                relations = relations_batch[i].tolist()
                while mask[-1]:
                    mask.pop()
                    tokens.pop()
                    heads.pop()
                    relations.pop()

                found_sentences.add(";".join([
                    ",".join([str(t) for t in symbols])
                    for symbols in (tokens, heads, relations)
                ]))

        expected_sentences = self.read_expected_sentences(path)
        self.assertTrue(found_sentences == expected_sentences)


class EmbeddingTest(TestCase):

    def test_forward(self):
        embedding = m.EmbeddingLayer(5,4)

        tokens = torch.tensor([[1,3,4],[2,0,1]])
        heads = torch.tensor([[0,3,2],[3,4,1]])
        found_energy = embedding.forward(tokens, heads)

        U = embedding.U[tokens]
        V = embedding.V[heads]
        ub = embedding.Ubias[tokens]
        vb = embedding.Vbias[heads]

        expected_energy = (
            (U * V).sum(dim=2, keepdim=True) 
            + ub + vb
        ).sum()

        self.assertTrue(torch.equal(found_energy, expected_energy))


class Word2VecModelTest(TestCase):

    def test_get_head_probs(self):

        Px = torch.tensor([1/100] * 100)
        embedding = m.EmbeddingLayer(100,4)
        model = m.train.Word2VecModel(embedding, Px) 
        tokens_batch = torch.tensor([
            [0,90, 91,92,93,94,95,96,97,98], 
            [0,80,81,82,83,84,85,86,87,88]
        ])
        num_sentences, num_tokens = tokens_batch.shape

        # Test the function.
        found_probs = model.get_head_probs(num_tokens)

        # Calculate the expected value.
        kernel = torch.tensor([0,5,4,3,2,1])
        expected_probs = self.calculate_expected_head_probs(kernel, num_tokens)

        self.assertTrue(torch.equal(found_probs, expected_probs))


    def calculate_expected_head_probs(self, kernel, num_tokens):

        conv = torch.zeros(num_tokens)
        truncated_size = min(num_tokens, kernel.shape[0])
        conv[:truncated_size] += kernel[:truncated_size]

        expected_probs = torch.zeros(num_tokens, num_tokens)
        for i in range(num_tokens):
            expected_probs[i] = conv
            conv = conv.roll(1)
            conv[0] = 0
        expected_probs += expected_probs.T.clone()

        expected_probs[:,0] = 0
        expected_probs[0] = torch.nn.functional.one_hot(
            torch.tensor(0),num_tokens)
        return expected_probs


    def test_sample_parses(self):

        seed_random(0)

        Px = torch.tensor([1/100] * 100)
        embedding = m.EmbeddingLayer(100,4)
        model = m.train.Word2VecModel(embedding, Px) 
        tokens_batch = torch.tensor([[0,90, 91,92,93,94,95,96,97,98]])
        tokens_batch = tokens_batch.expand((2000,-1))
        num_sentences, num_tokens = tokens_batch.shape

        # Test the function.
        heads = model.sample_parses(tokens_batch)

        # We will check which tokens chose <ROOT> as head.
        ROOT = 0
        zero_heads = (heads == ROOT)

        # Every <ROOT> chooses itself as head.
        for i in range(num_sentences):
            self.assertTrue(zero_heads[i,ROOT])

        # No others choose <ROOT> as head
        self.assertTrue(zero_heads.sum() == num_sentences)

        # Now check how far away each token's chosen head is.  The conv
        # tensor below sets the relative probability of choosing a head at
        # various distances.  
        # Note that the expected probability is slightly different than 
        # the relative probabilities in conv because tokens near the ends of
        # sentences have truncated selection probabilities.
        conv = torch.tensor([0,5,4,3,2,1])
        expected_prob = torch.tensor([
            0.0000, 0.3967, 0.2724, 0.1847, 0.1018, 0.0444])

        # We can check actual distances by subtracting the ids of tokens and
        # their heads (because the ids in token_batch are consecutive).  
        dist_counts = torch.zeros(conv.shape[0])
        distances = (heads - tokens_batch).abs()
        for i in range(num_sentences):
            for j in range(num_tokens):
                # A head should never be selected beyond the support of the
                # kernel.  The following line would trigger an index error if so.
                dist_counts[distances[i,j]] += 1

        # Adjust counts to exclude <ROOT> choosing itself in each sentence.
        dist_counts[0] -= num_sentences

        # Confirm frequency of distances are close to expected probability.
        found_prob = dist_counts / dist_counts.sum()
        self.assertTrue(torch.allclose(found_prob, expected_prob, atol=1e-02))


    def test_sample_tokens(self):

        seed_random(0)

        vocab = 10
        num_sentences = 10000
        Px = torch.tensor([1/vocab] * vocab)
        embedding = m.EmbeddingLayer(vocab,4)
        model = m.train.Word2VecModel(embedding, Px) 
        num_metropolis_hastings_steps = 20

        # By using the full vocabulary consecutively in one sentence, it will
        # be easier to count head selection frequencies later.
        num_tokens = vocab
        tokens_batch = torch.tensor([range(num_tokens)])
        tokens_batch = tokens_batch.expand((num_sentences,-1))
        num_sentences, num_tokens = tokens_batch.shape

        kernel = torch.tensor([0,5,4,3,2,1])
        probs = self.calculate_expected_head_probs(kernel, num_tokens)
        sampler = torch.distributions.Categorical(probs)
        samples = sampler.sample((num_sentences,))

        heads = tokens_batch.gather(dim=1,index=samples)
        new_heads = heads.clone()
        for i in range(num_metropolis_hastings_steps):
            new_heads = model.sample_tokens(tokens_batch, new_heads)

        # We're going to check which tokens chose <ROOT> as head.
        chose_root = (new_heads == 0)
        root_mask = torch.nn.functional.one_hot(
            torch.tensor(0),new_heads.shape[1]).unsqueeze(0)

        # non-<ROOT> tokens never choose <ROOT> as head.
        non_root_chose_root = ((1 - root_mask) * chose_root).sum()
        self.assertEqual(non_root_chose_root, 0)

        # <ROOT> always chooses root as head
        root_chose_root = (root_mask * chose_root).sum()
        self.assertEqual(root_chose_root, num_sentences)

        # non-<ROOT> tokens choose heads according to model probability
        # Count how often token j is chosen as head by token i
        counts = torch.zeros(vocab, vocab)
        token_position_index = torch.tensor(range(num_tokens))
        for i in range(num_sentences):
            counts[token_position_index,new_heads[i]] += 1
        # Normalize each row to make this a probability distribution over
        # head choice conditioned on token.
        counts = counts / counts.sum(dim=1, keepdim=True)

        # Calculate the expected probability of each token chosing each head
        # We're assuming that sentence_link_energy works
        energy = embedding.sentence_link_energy(
            tokens_batch[0].unsqueeze(0),mask=False).squeeze(0)
        probs = torch.exp(energy)

        # We're expecting root to always choose root, and other tokens to never
        # choose root
        probs[:,0] = 0
        probs[0] = torch.nn.functional.one_hot(torch.tensor(0), num_tokens)

        # The model will not be normalized, so we must normalize probs like 
        # counts
        probs = probs / probs.sum(dim=1, keepdim=True)

        self.assertTrue(torch.allclose(counts, probs, atol=1e-2))


class ParityTokenResamplerTest(TestCase):

    def test_get_node_parity(self):
        """
        The function should accept a (*, sentence_length)-shaped batch of
        heads_ptr vectors, containing trees, and it should return a 
        (*, 2, sentence_length)-shaped batch of "node-parities".

        The parity of a node i is whether it takes an even or odd number of
        hops to get from i to ROOT.
        """

        # We'll use a couple real sentences with annotated parse structure
        sentence_path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/sentences.index")
        dataset = m.PaddedDataset(sentence_path, PADDING)
        dataset.read()
        dictionary_path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/tokens.dict")
        dictionary = m.Dictionary(dictionary_path)

        # Build the sampler, and read the batch from the dataset.  Adjust
        # padding
        sampler = m.sp.ParityTokenResampler(dataset.Px)
        tokens_batch, head_ptrs_batch, relations_batch = dataset[0]
        mask = (head_ptrs_batch == -1)
        head_ptrs_batch[mask] = 0

        # Test the target function
        found_node_parities = sampler.get_node_parity(head_ptrs_batch)

        # These were verified by hand.
        expected_node_parities = torch.tensor([
            [
                False, True, False, False, False, True, True, True, False,
                False, False, False, True, False, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True
            ],[
                False, False, True, True, True, True, False, False, True,
                False, False, False, True, False, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True
            ],[
                False, False, False, True, False, False, True, True, True,
                False, False, False, True, True, True, False, False, True,
                True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True
            ],[
                False, True, False, False, True, True, True, True, True, True,
                True, False, False, False, False, True, False, True, True,
                True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True
            ],[
                False, True, False, False, True, False, False, False, False,
                True, False, True, True, False, True, True, False, False, True,
                True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True
            ],[
                False, False, True, False, False, False, False, True, False,
                False, True, False, False, True, False, False, True, False,
                False, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True
            ],[
                False, False, True, False, True, False, True, True, True,
                False, False, False, False, True, True, True, True, False,
                True, False, False, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True
            ],[
                False, True, False, False, False, False, True, False, True,
                False, False, False, False, False, True, False, False, False,
                True, True, True, False, False, True, True, False, False,
                False, True, False, True, True, True, True, True, True, True
            ],[
                False, True, False, True, True, True, False, False, True,
                False, False, True, True, False, True, True, False, False,
                False, False, False, True, True, False, False, True, False,
                True, True, False, False, False, True, True, False, False,
                True
            ],[
                False, True, False, False, True, False, True, True, True, True,
                False, False, False, True, True, False, False, True, True,
                False, False, True, False, False, False, False, True, True,
                True, True, True, False, False, False, False, True, False
            ]
        ])

        self.assertTrue(torch.equal(
            found_node_parities, expected_node_parities))

        # Now add another batching dimension.  The function should simply 
        # treat the final dimension as the dimension indexed by tree structure
        # and should treat all others simply as batching.  So, the expected
        # result will simply be similarly reshaped.
        rebatched_heads = head_ptrs_batch.unsqueeze(0).expand(5,-1,-1)
        found_node_parities = sampler.get_node_parity(rebatched_heads)
        expected_node_parities = expected_node_parities.unsqueeze(0).expand(
            5,-1,-1)
        self.assertTrue(torch.equal(found_node_parities, expected_node_parities))
        

    #@skip("Long test")
    def test_parity_token_resampler(self):

        seed_random(0)
        num_sentences = 50000
        num_resampling_steps = 40
        num_batches = 10

        # We'll use a real sentence, with it's annotated parse structure
        # as a starting point of testing the token resampling function.
        sentence_path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/sentences.index")
        dataset = m.PaddedDataset(sentence_path, PADDING)
        dataset.read()
        dictionary_path = os.path.join(
            TEST_DATA_PATH, "ten-sentence-dataset/tokens.dict")
        dictionary = m.Dictionary(dictionary_path)
        embedding = m.EmbeddingLayer(len(dictionary),4)

        # For this test, we will just use sentence 7.  It's actually the first
        # sentence in the dataset; a quirk of dataset loading puts it in pos 7.
        tokens_batch, head_ptrs_batch, _, mask = dataset[0]
        _, num_tokens = tokens_batch.shape
        tokens_batch = tokens_batch[7].unsqueeze(0).expand(
            num_sentences,-1).clone()
        head_ptrs_batch = head_ptrs_batch[7].unsqueeze(0).expand(
            num_sentences,-1).clone()
        mask = mask[7].unsqueeze(0).expand(num_sentences,-1)

        # For the sentence selected, we only actually need the first 26 tokens.
        # Sampling a smaller vocabulary gives faster convergence.
        restricted_vocab = 145
        Px_restricted_vocab = dataset.Px[:restricted_vocab]
        token_sampler = m.sp.ParityTokenResampler(Px_restricted_vocab)

        # Continually resample the sentence.  Every resampling generates
        # two mutated sentences, one with "even" tokens mutated and one with
        # "odd" token mutated (based on parity of number of steps to get to 
        # ROOT).  Keep re-submitting the even one for mutation so that it
        # the probability of sampling tokens at the sentence head (odd)
        # should approach model probability dictated by energies.
        # Do this in multiple batches to generate many samples without
        # clogging memory.
        watched_index = 1
        counts = torch.zeros(restricted_vocab)
        torch.set_printoptions(sci_mode=False)
        for batch in range(num_batches):
            sys.stdout.write(str(batch)+":")
            mutated_tokens = tokens_batch.unsqueeze(1).expand(-1,2,-1).clone()
            for i in range(num_resampling_steps):
                sys.stdout.write(str(i))
                sys.stdout.flush()
                mutated_tokens = token_sampler.sample_tokens(
                    mutated_tokens[:,1,:], head_ptrs_batch, embedding, mask)
                sys.stdout.write("\b"*len(str(i)))
            # Count how often token j is chosen as head by token i
            for i in range(num_sentences):
                counts[mutated_tokens[i,1,watched_index]] += 1
            sys.stdout.write("\b"*(len(str(batch))+1))

        # Get the probability of choosing any given token at
        # watched_index.
        found_probs = counts / counts.sum()

        # We will look at the rate of occurrence of mutations in specific
        # locations.  Consider the token at watched_index.  The
        # distribution of mutants in that position should be proportional to
        # the exp(sum of energies of its head and subordinate relations).
        heads = tokens_batch[0].gather(dim=0, index=head_ptrs_batch[0])
        pos_head = heads[watched_index].unsqueeze(0)
        pos_subs_loc = heads==tokens_batch[0, watched_index]
        pos_subs_loc[mask[0]] = 0
        pos_subs_loc = pos_subs_loc.nonzero().squeeze(1)
        pos_subs = tokens_batch[0, pos_subs_loc]

        # Calculate the total link energy when the watched_pos token is mutated
        # into each of the possible words in the vocabulary.
        with torch.no_grad():
            sub_energy = (
                embedding.V[:restricted_vocab] @ embedding.U[pos_subs].T
                + embedding.Vbias[:restricted_vocab] 
                + embedding.Ubias[pos_subs].T
            ).sum(dim=1)
            head_energy = (
                embedding.U[:restricted_vocab] @ embedding.V[pos_head].T
                + embedding.Ubias[:restricted_vocab] 
                + embedding.Vbias[pos_head].T
            ).squeeze(1)
        mutation_energy = sub_energy + head_energy
        weights = torch.exp(mutation_energy)
        expected_probs = weights / weights.sum()

        print("expected_probs, found_probs")
        print_probs = torch.zeros(3, restricted_vocab)
        print_probs[0] = expected_probs
        print_probs[1] = found_probs
        print_probs[2] = token_sampler.Px
        print(print_probs.T)
        pdb.set_trace()
        self.assertTrue(torch.allclose(found_probs, expected_probs, atol=4e-3))


class ContentionSamplerTest(TestCase):

    def test_has_cycle(self):
        heads = torch.tensor([
            [0,2,3,4,0],
            [0,2,3,2,0],
            [0,2,3,1,1],
            [0,3,4,2,1]
        ])
        expected_cycle = torch.tensor([
            [0,0,0,0,0],
            [0,0,1,1,0],
            [0,1,1,1,0],
            [0,1,1,1,1]
        ])

        sampler = m.sp.Contention()
        found_cycle = sampler.has_cycle(heads)

        self.assertTrue(torch.all(found_cycle == expected_cycle))


    def test_is_multiple_root(self):
        heads = torch.tensor([
            [0,2,3,4,0],
            [0,0,3,4,0],
            [0,2,0,0,1],
            [0,0,0,0,2],
            [0,0,0,0,0],
        ])
        expected_multiple_root = torch.tensor([
            [0,0,0,0,0],
            [0,1,0,0,1],
            [0,0,1,1,0],
            [0,1,1,1,0],
            [0,1,1,1,1],
        ])

        sampler = m.sp.Contention()
        found_multiple_root = sampler.is_multiple_root(heads)

        self.assertTrue(torch.all(found_multiple_root == expected_multiple_root))


class QuadTreeCounterTest(TestCase):
    NUM_FOUR_TREES = 64
    def test_tree_coding(self):
        # NUM_FOUR_TREES number of unique tree codes were generated.
        counter = QuadTreeCounter()
        self.assertTrue(len(counter.trees) == self.NUM_FOUR_TREES)
        self.assertTrue(len(set(counter.trees)) == self.NUM_FOUR_TREES)

    def test_tree_allocation(self):
        counter = QuadTreeCounter()

        # Each of the 64 indices into self.trees should be in exactly one tree
        # shape list (self.el_trees, ..j_trees, ..h_trees, ..m_trees).
        index_sightings = Counter()
        for indices in counter.tree_shape_indices:
            index_sightings.update(indices)

        # All 64 trees are assigned, all of which were assigned only once:
        self.assertTrue(len(index_sightings)==self.NUM_FOUR_TREES)
        self.assertTrue(all([val==1 for val in index_sightings.values()]))
        self.assertTrue(
            sorted(index_sightings.keys()) == list(range(self.NUM_FOUR_TREES)))

    def test_counter_probs(self):
        counter = QuadTreeCounter()

        # Here we will assuming counter.trees is correct.
        # Fill a tree counter with arbitrary values for each tree, and 
        # Maintain an external counter using counter.trees as an index.
        expected_counts = Counter()
        for i, tree in enumerate(counter.trees):
            expected_counts[tree] += i
            counter.add(tree, i)

        # Test the target function.
        found_probs = counter.get_probs()

        # Calculate the probabilities that we expect by looking in our external
        # counter.  These probs were also verified manually.
        expected_probs = torch.zeros(self.NUM_FOUR_TREES)
        total_trees = sum(expected_counts.values())
        for i, tree in enumerate(counter.trees):
            expected_probs[i] = expected_counts[tree] / total_trees

        self.assertTrue(torch.allclose(found_probs, expected_probs))


class QuadTreeCounter:

    # To compare counts and probabilities for trees on four nodes we need to
    # assign a unique, orderable key to each tree.  To do this, we generate all
    # possible trees on four nodes, and assign each the tuple (a,b,c,d) in
    # which, as usual, the ith position contains a pointer to the index of the
    # ith nodes head.  Note that there are only four possible shapes for trees
    # of four nodes: (I name them according to the letter they look like):
    #
    #    el:  |    j:  |    h: |    m:  |
    #         |       | |      |       |||
    #         |         |     | |
    #         |  
    #
    # Trees of the same shape may differ by permutation of the node ids.  So,
    # generate all trees by looking at all permutations on each shape.
    # Multiple permutations can generate the same tree, but the unique key
    # assignment prevents duplication.

    EL = 0
    J = 1
    H = 2
    M = 3

    def __init__(self):
        self.tree_counter = None
        self.trees = None
        self.seen_trees = None
        self.tree_shape_indices = None
        self.el_trees = None
        self.j_trees = None
        self.h_trees = None
        self.m_trees = None
        self.tree_structures = None
        self.setup()

    def get_probs(self):
        """
        Return the probabilities found for the trees (which are a list of
        indices into the self.trees list)
        """
        # Compute all probabilities afresh.
        num_sentences = sum(self.tree_counter.values())
        found_probs = torch.zeros(len(self.trees))
        for i, tree in enumerate(self.trees):
            found_probs[i] = (
                self.tree_counter[tuple(tree)]
                / num_sentences
            )
        return found_probs

    def __len__(self):
        """Number of trees (always 64)"""
        return len(self.tree_counter)

    def add(self, tree_code, amount=1):
        """ Increment count for tree_code by amount.  """
        # Update main counter.
        self.tree_counter[tree_code] += amount

    def build_tree(self, perm, tree_type, tree_number):

        # Get structure and index accumulator for this tree type.
        heads = self.tree_structures[tree_type]
        tree_indices = self.tree_shape_indices[tree_type]

        # Build the tree based on the permutation and tree structure given.
        tree_code = [0] * 5
        tree_code[perm[0]] = heads[0]
        tree_code[perm[1]] = perm[heads[1]]
        tree_code[perm[2]] = perm[heads[2]]
        tree_code[perm[3]] = perm[heads[3]]
        tree_code = tuple(tree_code)

        # Add the tree if its new.  
        added_tree = 0
        if tree_code not in self.seen_trees:
            self.seen_trees.add(tree_code)
            self.trees.append(tree_code)
            tree_indices.append(tree_number)
            added_tree = 1

        # Return 1 if a new tree was made, 0 if not.
        return added_tree

    def setup(self):

        # Head-pointer-encoded structure of four shapes of four-trees
        self.tree_structures = [
            [0,0,1,2], # el_heads
            [0,0,0,2], # j_heads
            [0,0,1,1], # h_heads 
            [0,0,0,0]  # m_heads
        ]

        # Initialize some accumulators for trees and tree indices.
        self.seen_trees = set()
        self.trees = []
        self.el_trees = []
        self.j_trees = []
        self.h_trees = []
        self.m_trees = []
        self.tree_shape_indices = (
            self.el_trees, self.j_trees, self.h_trees, self.m_trees)

        # Apply each permutation of four nodes to each tree structure.  Generate
        # the code for that tree, and add it (provided it is not a duplicate).
        # Every tree is assigned a unique number, which is the index of 
        # that tree code in self.trees.
        tree_number = 0
        for perm in it.permutations([1,2,3,4]):
            for tree_type in [self.EL, self.J, self.H, self.M]:
                tree_number += self.build_tree(perm, tree_type, tree_number)

        # Make a counter indexed by the tuple encoding each tree.
        self.tree_counter = {tree:0 for tree in self.trees}


class ParseSamplingMeasureTests(TestCase):

    def test_contention_uniform(self):
        sampler = m.sp.Contention()
        self.uniform_test(sampler)

    def test_cpwr2_uniform(self):
        sampler = m.sp.CycleProofRerooting2()
        self.uniform_test(sampler)

    def test_cycle_proof_walk_with_rerooting_uniform(self):
        sampler = m.sp.CycleProofRerooting()
        self.uniform_test(sampler)

    def test_root_reset_uniform(self):
        sampler = m.sp.RootReset()
        self.uniform_test(sampler)

    def uniform_test(self, sampler):

        seed_random(0)

        start = time.time()
        vocab = 5
        num_sentences = 200000

        # Make all embeddingn parameters equal to constant 0.1 so that
        # every tree should be equally likely.
        with torch.no_grad():
            embedding = m.EmbeddingLayer(vocab,4)
            embedding.U[:,:] = 0.1
            embedding.V[:,:] = 0.1
            embedding.Ubias[:,:] = 0.1
            embedding.Vbias[:,:] = 0.1

        tokens_batch = torch.tensor([[0,1,2,3,4]]).expand(num_sentences,-1)

        # First we will make a set of all possible rooted trees constructed
        # with four labelled nodes.  
        tree_counter = QuadTreeCounter()

        # Count the occurrences of each tree generated by the model
        trees = sampler.sample(tokens_batch, embedding)
        for i in range(trees.shape[0]):
            tree = tuple(trees[i].tolist())
            tree_counter.add(tree)
        frequencies = torch.tensor([
            val / (num_sentences)
            for val in tree_counter.tree_counter.values()
        ])

        # The frequencies should be uniform and all close to 1 / num_trees:
        expected_frequency = torch.tensor([1/len(tree_counter)])
        self.assertTrue(torch.allclose(
            frequencies,
            expected_frequency,
            atol=0.0035
        ))

    def test_contention_nonuniform(self):
        sampler = m.sp.Contention()
        self.nonuniform_test(sampler)
        
    def test_cycle_proof_rerooting_nonuniform(self):
        sampler = m.sp.CycleProofRerooting()
        self.nonuniform_test(sampler)

    def test_cpwr2_nonuniform(self):
        sampler = m.sp.CycleProofRerooting2()
        self.nonuniform_test(sampler)

    def test_root_reset_nonuniform(self):
        sampler = m.sp.RootReset()
        self.nonuniform_test(sampler)


    def get_tree_probs(self, trees, embedding, tokens_batch):
        trees = torch.tensor(trees)
        energies = embedding.link_energy(
            tokens_batch.expand(len(trees), -1),
            trees
        )
        # energies[:,0] = 0 # It's not clear if we should do this or not
        expected_probs = torch.exp(energies.sum(dim=1))
        expected_probs = expected_probs / expected_probs.sum()

        return expected_probs


    def nonuniform_test(self, sampler):

        seed_random(0)

        start = time.time()
        vocab = 5
        num_sentences = 100000

        # Embedding layer random.  Each tree should have a probability 
        # proportional to its energy
        embedding = m.EmbeddingLayer(vocab,4)

        tokens_batch = torch.tensor([[0,1,2,3,4]])

        # First we will make a set of all possible rooted trees constructed
        # with four labelled nodes.  
        tree_counter = QuadTreeCounter()

        # Generate trees from the model, and count them.
        start = time.time()
        found_trees = sampler.sample(
            tokens_batch.expand(num_sentences, -1), embedding)
        elapsed = time.time() - start
        for i in range(found_trees.shape[0]):
            tree = tuple(found_trees[i].tolist())
            tree_counter.add(tree)

        # Determine the probability of finding each tree.  Provide a breakdown
        # for different topologies of tree (el, j, j, m).
        found_probs = tree_counter.get_probs()
        found_el_probs = found_probs[torch.tensor(tree_counter.el_trees)]
        found_j_probs = found_probs[torch.tensor(tree_counter.j_trees)]
        found_h_probs = found_probs[torch.tensor(tree_counter.h_trees)]
        found_m_probs = found_probs[torch.tensor(tree_counter.m_trees)]

        # TODO: factor this out as "get_tree_probs" and apply it to all
        # tree sub counter types.  Also convert sub-counts to sub found_probs.
        expected_probs = self.get_tree_probs(
            tree_counter.trees, embedding, tokens_batch)
        expected_el_probs = expected_probs[torch.tensor(tree_counter.el_trees)]
        expected_j_probs = expected_probs[torch.tensor(tree_counter.j_trees)]
        expected_h_probs = expected_probs[torch.tensor(tree_counter.h_trees)]
        expected_m_probs = expected_probs[torch.tensor(tree_counter.m_trees)]

        # The frequencies should be uniform and all close to 1 / num_trees:
        #torch.set_printoptions(sci_mode=False)
        #print("Elapsed:", elapsed)
        #print((found_probs - expected_probs).abs().mean())
        self.assertTrue(torch.allclose(
            found_probs,
            expected_probs,
            atol=0.005
        ))


class EnergyTest(TestCase):


    def test_link_energy_1(self):
        """
        Given a list of vectors and covectors, calculate the total energy,
        which is the sum of all of the inner products of vectors and covectors,
        including biases.
        """
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[2,0,3], [1,3,4]])
        heads_batch = torch.tensor([[1,1,1], [2,2,2]])
        num_sentences, num_tokens = tokens_batch.shape

        # Test the function.
        found_energy = embedding.link_energy(tokens_batch, heads_batch)

        # Calculate the expected value
        U = embedding.U[tokens_batch]
        V = embedding.V[heads_batch]
        ub = embedding.Ubias[tokens_batch]
        vb = embedding.Vbias[heads_batch]

        expected_energy = ((U*V).sum(dim=2, keepdim=True) + ub + vb).squeeze(2)

        self.assertTrue(torch.equal(found_energy, expected_energy))


    def test_link_energy_mask(self):
        """
        Given a list of vectors and covectors, calculate the total energy,
        which is the sum of all of the inner products of vectors and covectors,
        including biases.
        """
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[2,0,3], [1,3,4]])
        heads_batch = torch.tensor([[1,1,1], [2,2,2]])
        num_sentences, num_tokens = tokens_batch.shape
        mask = torch.tensor([[0,0,1],[0,0,1]], dtype=torch.bool)

        # Test the function.
        found_energy = embedding.link_energy(tokens_batch, heads_batch, mask)

        # Calculate the expected value
        # Set masked values to zero
        U = embedding.U[tokens_batch]
        V = embedding.V[heads_batch]
        ub = embedding.Ubias[tokens_batch]
        vb = embedding.Vbias[heads_batch]

        # Calculate energy, then set energy for masked values to zero
        expected_energy = (
            (U*V).sum(dim=2, keepdim=True) + ub + vb).squeeze(2)
        expected_energy[:,2] = 0

        self.assertTrue(torch.equal(found_energy, expected_energy))


    def test_link_energy_nested_batches(self):
        """
        Similar to test_link_energy, but here we are checking whether 
        the function can handle nesting of batches.  It should just consider
        tokens_batch and heads_batch to be parallel tensors containing
        indices for covectors and vectors, and should compute pairwise energies,
        returning a tensor of energies having the same shape as both inputs.
        """
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([
            [
                [2,0,3], 
                [4,4,4]
            ],[
                [2,0,3],
                [2,0,3]
            ]
        ])
        heads_batch = torch.tensor([
            [
                [1,1,1], 
                [2,2,2]
            ],[
                [1,1,1], 
                [3,3,3]
            ]
        ])
        num_sentences, num_mutations, num_tokens = tokens_batch.shape

        # Test the function.
        found_energy = embedding.link_energy(tokens_batch, heads_batch)

        # Calculate the expected value
        U = embedding.U[tokens_batch]
        V = embedding.V[heads_batch]
        ub = embedding.Ubias[tokens_batch]
        vb = embedding.Vbias[heads_batch]

        expected_energy = ((U*V).sum(dim=3, keepdim=True) + ub + vb).squeeze(3)
        self.assertTrue(torch.equal(found_energy, expected_energy))


    def test_link_energy_requires_same_shape(self):
        """
        The embedding.link_energy function should reject input where 
        tokens_batch and heads_batch do not have the same shape.
        """
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[2,0,3], [1,3,4]])
        heads_batch = torch.tensor([[1,1,1]])
        num_sentences, num_tokens = tokens_batch.shape

        with self.assertRaises(ValueError):
            embedding.link_energy(tokens_batch, heads_batch)


    def test_sentence_link_energy(self):
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[2,0,3], [1,3,4]])
        num_sentences, num_tokens = tokens_batch.shape

        # We're testing the function that calculates token-token link energies.
        found_energy = embedding.sentence_link_energy(
            tokens_batch, mask=False)

        # What were we expecting?  It should be the matrix product of the 
        # vectors and covectors plus the biases.  The covectors and covector-
        # biases need to be transposed so that we can use matrix multiplication.
        U = embedding.U[tokens_batch]
        V_t = embedding.V[tokens_batch].transpose(1,2)
        ub = embedding.Ubias[tokens_batch]
        vb_t = embedding.Vbias[tokens_batch].transpose(1,2)
        expected_energy = U @ V_t + ub + vb_t

        self.assertTrue(torch.equal(found_energy, expected_energy))

        # Now try calculating energy where root always self-links and 
        # no other token chooses itself.
        found_energy = embedding.sentence_link_energy(
            tokens_batch, mask=True)
        diagonal = torch.zeros(
            (num_tokens, num_tokens)).fill_diagonal_(-torch.inf)
        expected_energy += diagonal
        root_row = torch.full((num_tokens,), -torch.inf)
        root_row[0] = 0
        expected_energy[:,0] = root_row

        self.assertTrue(torch.equal(found_energy, expected_energy))




if __name__ == "__main__":
    main()


