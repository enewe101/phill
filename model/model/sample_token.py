import torch
import pdb


class SimpleTokenSampler():

    def __init__(self, Nx):
        self.Nx = Nx
        self.Nx[0] = 0
        self.unigram_sampler = torch.distributions.Categorical(self.Nx)

    def sample_proposal(self, shape):
        return self.unigram_sampler.sample(shape)

    def get_proposal_weights(self, prev_heads, next_heads):
        return (self.Nx[prev_heads]/self.Nx[next_heads])

    def sample_tokens(self, tokens_batch, prev_head_ptrs, embedding, mask):
        num_sentences, num_tokens = tokens_batch.shape
        prev_heads = tokens_batch.gather(dim=1, index=prev_head_ptrs)

        # Sample 1 head for each word in each sentence.
        next_heads = self.sample_proposal(tokens_batch.shape)

        # Calculate link energies for initial heads batch and for the next.
        prev_weights = embedding.link_energy(tokens_batch, prev_heads)
        next_weights = embedding.link_energy(tokens_batch, next_heads)

        proposal_weights = self.get_proposal_weights(prev_heads, next_heads)

        accept_score = proposal_weights * torch.exp(next_weights - prev_weights)
        reject_score = torch.rand(tokens_batch.shape)
        accept_proposal = accept_score > reject_score

        ## <ROOT> and padding always reject, keeping <ROOT> as their head
        accept_proposal[mask] = False
        accept_proposal[:,0] = False

        next_heads = torch.where(accept_proposal, next_heads, prev_heads)
        return next_heads



class RebasedTokenSampler(SimpleTokenSampler):

    def get_proposal_weights(self, prev_heads, next_heads):
        return (self.Nx[prev_heads]/self.Nx[next_heads])**2

    #def sample_proposal(self, shape):
    #    return torch.randint(1, self.Nx.shape[0], shape)



class ParityTokenSampler:

    MIN_DEPTH = 7

    def __init__(self, Px):
        self.Px = Px
        self.unigram_sampler = torch.distributions.Categorical(self.Px)


    def get_node_parity(self, head_ptrs):
        """ 
        Accepts a (*, sentence_length)-shaped batch of heads_ptr vectors,
        containing trees, and returns a (*, 2, sentence_length)-shaped batch of
        "node-parities".

        The parity of a node i True if it is an odd number of hops from ROOT
        and FALSE if it is an even number of hops.  ROOTs parity is False.
        """
        # Partition nodes into two groups based on whether they are an even or
        # odd number of steps from ROOT.  Continually follow links toward the
        # root by recursively indexing head_ptrs into itself.  At each step,
        # toggle the parity nodes that have reached ROOT.  The final reflects
        # the parity of the step on each node reached ROOT.  Stop on an even
        # parity step such that, by convention ROOT is considered even.  Do at
        # least MIN_DEPTH steps (typical minimum parse tree depth),
        node_parity = torch.zeros(head_ptrs.shape, dtype=torch.bool)
        reachable = head_ptrs
        step_parity = True
        num_steps = 0
        while num_steps < self.MIN_DEPTH or step_parity or torch.any(reachable):
            node_parity = node_parity ^ (reachable == 0)
            reachable = head_ptrs.gather(dim=-1, index=reachable)
            num_steps += 1
            step_parity = step_parity ^ True

        # ROOT's self-loop causes its parity to be wrong. Fix that. 
        node_parity[...,0] = False

        return node_parity


    def sample_tokens(self, tokens_batch, head_ptrs, embedding, mask):
        num_sentences, num_tokens = tokens_batch.shape

        # Randomly nodes from the unigram distribution (except at mask).
        mutants_batch = self.unigram_sampler.sample(tokens_batch.shape)
        mutants_batch[mask] = 0 

        # Gather the original head tokens, as well as mutated head tokens.
        mutant_heads = mutants_batch.gather(dim=1, index=head_ptrs)
        heads = tokens_batch.gather(dim=1, index=head_ptrs)

        # Calculate the total energy of sentences for three cases:
        # 1) Where heads and subordinates are not mutated
        # 2) Where heads are mutated.
        # 3) Where subordinates are mutated.
        normal_energy = embedding.link_energy(
            tokens_batch, heads, mask)
        mutant_head_energy = embedding.link_energy(
            tokens_batch, mutant_heads, mask)
        mutant_sub_energy = embedding.link_energy(
            mutants_batch, heads, mask)

        # Calculate the energy for each possible point mutation.
        # At each position i, take the energy from either 
        # (a) mutant_head_energy, (b) mutant_sub_energy, or (c) normal_energy,
        # depending on whether that point i (a) has the point mutation as its 
        # head, (b) is the point mutation (and from the perspective of
        # calculating energy, it is a mutated subordinate of an unmutated head)
        # (c) is neither the point mutation nor a subordinate of it.
        sub_mutation = torch.eye(num_tokens, dtype=torch.bool)
        sub_mutation = sub_mutation.unsqueeze(0).expand(num_sentences, -1, -1)
        head_mutation = sub_mutation.gather(
            dim=2, index=head_ptrs.unsqueeze(1).expand(-1, num_tokens, -1))

        # Construct the link energy for each possible point mutation by
        # selecting values from the three energies already calculated,
        # depending on whether there is a point mutation at i (take
        # mutant_sub_energy), a point mutation at i's head (take
        # mutant_head_energy), or neither (take normal_energy).
        mutation_energy = torch.empty(num_sentences, num_tokens, num_tokens)
        mutation_energy = torch.where(
            (sub_mutation.logical_or(head_mutation)).logical_not(),
            normal_energy.unsqueeze(1).expand(-1, num_tokens, -1),
            mutation_energy
        )
        # At locations that are mutated, take the mutant sub energy.
        mutation_energy = torch.where(
            sub_mutation,
            mutant_sub_energy.unsqueeze(1).expand(-1, num_tokens, -1),
            mutation_energy
        )
        # At locations whose head was mutated, take the mutant head energy.
        mutation_energy = torch.where(
            head_mutation,
            mutant_head_energy.unsqueeze(1).expand(-1, num_tokens, -1),
            mutation_energy
        )

        # Now calculate total sentence energy for the normal sentences
        # and for each possible point mutation of normal sentences.
        normal_energy = normal_energy.sum(dim=1)
        mutation_energy = mutation_energy.sum(dim=2)

        # Calculate the metropolis-hastings acceptance score, then either
        # accept or reject each mutation.
        proposal_weights = self.Px[tokens_batch] / self.Px[mutants_batch]

        accept_score = proposal_weights * torch.exp(
            mutation_energy
            - normal_energy.unsqueeze(1).expand(-1, num_tokens)
        )

        reject_score = torch.rand(tokens_batch.shape)
        do_accept = accept_score > reject_score

        # Collapse all mutations into two sentences: one taking all mutations
        # in tokens that are an even number of steps from the root, and one
        # taking all mutations in tokens that are an odd number of steps from
        # the root.  This allows all parameters involved in a sentence
        # to receive negative sampling signal.
        node_parity = self.get_node_parity(head_ptrs)
        do_accept = do_accept
        tokens_mutated = torch.empty(
            (num_sentences, 2, num_tokens), dtype=torch.int64)
        tokens_mutated[:,0,:] = torch.where(
            do_accept.logical_and(node_parity==0), mutants_batch, tokens_batch,
        )
        tokens_mutated[:,1,:] = torch.where(
            do_accept.logical_and(node_parity), mutants_batch, tokens_batch,
        )

        return tokens_mutated

