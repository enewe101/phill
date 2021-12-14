import os
import pdb
import time
import random
import itertools as it
from unittest import TestCase, main
from collections import Counter

import numpy as np
import torch

import model as m


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
            os.path.dirname(__file__), "test-data/sentences.index")
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
            os.path.dirname(__file__), "test-data/sentences.index")
        padding = -1
        batch_size = 7
        dataset = m.PaddedDataset(path, padding, batch_size)
        dataset.read()

        found_sentences = set()
        for tokens_batch, heads_batch, relations_batch in dataset:
            for i in range(tokens_batch.shape[0]):
                tokens = tokens_batch[i].tolist()
                heads = heads_batch[i].tolist()
                relations = relations_batch[i].tolist()
                while tokens[-1] == padding:
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




class ParseSamplingMeasureTests(TestCase):

    def make_tree_counter(self):

        # Construct a counter for each possible rooted, labelled tree
        # on four nodes.  An easy way to find all trees is to encode
        # a tree as a tuple where position i of the tuple contains the 
        # the label of the head chosen by the ith node.  This provides
        # a unique encoding for each tree so we can easily avoid duplication.

        # Notice that there are four un-labelled tree structures, which we 
        # name based on resemblance to letters:
        #    el:  |    j:  |    h: |    m:  |
        #         |       | |      |       |||
        #         |         |     | |
        #         |  
        #
        # By checking each permutation of labels over each structure,
        # and encoding the tree in a tuple, we can find all labelled trees.
        # Not all label permutations over a given unlabelled structure yield
        # unique trees, but our encoding is unique, so this duplication is 
        # easily eliminated.
        trees = set()
        for perm in it.permutations([1,2,3,4]):
            # Apply each permutation to each unlabelled structure, and then
            # generate the unique encoding for that tree, and add it to a set.
            el_tree = [0] * 5
            el_tree[perm[0]] = 0
            el_tree[perm[1]] = perm[0]
            el_tree[perm[2]] = perm[1]
            el_tree[perm[3]] = perm[2]
            trees.add(tuple(el_tree))

            j_tree = [0] * 5
            j_tree[perm[0]] = 0
            j_tree[perm[1]] = perm[0]
            j_tree[perm[2]] = perm[0]
            j_tree[perm[3]] = perm[2]
            trees.add(tuple(j_tree))

            h_tree = [0] * 5
            h_tree[perm[0]] = 0
            h_tree[perm[1]] = perm[0]
            h_tree[perm[2]] = perm[1]
            h_tree[perm[3]] = perm[1]
            trees.add(tuple(h_tree))

            m_tree = [0] * 5
            m_tree[perm[0]] = 0
            m_tree[perm[1]] = perm[0]
            m_tree[perm[2]] = perm[0]
            m_tree[perm[3]] = perm[0]
            trees.add(tuple(m_tree))

        # Make a counter indexed by the tuple encoding each tree.
        return {tree:0 for tree in trees}


    def test_contention_uniform(self):
        sampler = m.sp.Contention()
        self.sample_parses_uniform_test(sampler)

    def test_cycle_proof_walk_with_rerooting_uniform(self):
        sampler = m.sp.CycleProofRerooting()
        self.sample_parses_uniform_test(sampler)

    def test_root_reset_uniform(self):
        sampler = m.sp.RootReset()
        self.sample_parses_uniform_test(sampler)


    def sample_parses_uniform_test(self, sampler):

        seed_random(0)

        start = time.time()
        vocab = 5
        num_sentences = 120000

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
        tree_counter = self.make_tree_counter()

        # Count the occurrences of each tree generated by the model
        trees = sampler.sample(tokens_batch, embedding)
        for i in range(trees.shape[0]):
            tree = tuple(trees[i].tolist())
            tree_counter[tree] += 1
        frequencies = torch.tensor([
            val / (num_sentences)
            for val in tree_counter.values()
        ])

        # The frequencies should be uniform and all close to 1 / num_trees:
        pdb.set_trace()
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
        tree_counter = self.make_tree_counter()

        expected_trees = torch.tensor(sorted(list(tree_counter.keys())))
        energies = embedding.link_energy(
            tokens_batch.expand(len(expected_trees), -1),
            expected_trees
        )
        # energies[:,0] = 0 # It's not clear if we should do this or not
        expected_probs = torch.exp(energies.sum(dim=1))
        expected_probs = expected_probs / expected_probs.sum()

        # Count the occurrences of each tree generated by the model
        start = time.time()
        trees = sampler.sample(tokens_batch.expand(num_sentences, -1), embedding)
        elapsed = time.time() - start
        for i in range(trees.shape[0]):
            tree = tuple(trees[i].tolist())
            tree_counter[tree] += 1

        found_probs = torch.zeros(len(tree_counter))
        for i, tree in enumerate(expected_trees.tolist()):
            found_probs[i] = tree_counter[tuple(tree)] / num_sentences

        # The frequencies should be uniform and all close to 1 / num_trees:
        torch.set_printoptions(sci_mode=False)
        print("Elapsed:", elapsed)
        print((found_probs - expected_probs).abs().mean())
        pdb.set_trace()
        self.assertTrue(torch.allclose(
            found_probs,
            expected_probs,
            atol=0.005
        ))







class EnergyTest(TestCase):


    def test_link_energy(self):
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


    def test_sentence_link_energy(self):
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[2,0,3], [1,3,4]])
        num_sentences, num_tokens = tokens_batch.shape

        # We're testing the function that calculates token-token link energies.
        found_energy = embedding.sentence_link_energy(
            tokens_batch, mask=False)

        # What were we expecting?  It should be the matrix product of the 
        # vectors and covectors plus the biases.  The covectors and covector-
        # biases need to be transposed.
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


    def test_parse_link_energy(self):
        embedding = m.EmbeddingLayer(5,4)
        tokens_batch = torch.tensor([[0,1,2], [0,3,4]])
        num_sentences, num_tokens = tokens_batch.shape
        parse_tree_batch = torch.tensor([
            [[1,0,0],
            [0,0,1],
            [1,0,0]],

            [[1,0,0],
            [1,0,0],
            [0,1,0]]
        ])

        # Test the function.
        found_energy = embedding.parse_link_energy(tokens_batch,parse_tree_batch)

        # Calculate the expected value.
        heads_batch = (parse_tree_batch @ tokens_batch.unsqueeze(2)).squeeze(2)
        U = embedding.U[tokens_batch]
        V = embedding.V[heads_batch]
        ub = embedding.Ubias[tokens_batch]
        vb = embedding.Vbias[heads_batch]
        expected_energy = ((U*V).sum(dim=2, keepdim=True) + ub + vb).squeeze(2)
        self.assertTrue(torch.equal(found_energy, expected_energy))



if __name__ == "__main__":
    main()


