import torch
from torch.nn.functional import one_hot
import pdb
import model as m



"""tensor([0.0154, 0.0155, 0.0152, 0.0159, 0.0159, 0.0160, 0.0157, 0.0155, 0.0155,
        0.0158, 0.0156, 0.0156, 0.0152, 0.0159, 0.0157, 0.0158, 0.0156, 0.0155,
        0.0152, 0.0151, 0.0159, 0.0153, 0.0162, 0.0158, 0.0155, 0.0156, 0.0159,
        0.0153, 0.0156, 0.0157, 0.0160, 0.0162, 0.0154, 0.0159, 0.0156, 0.0161,
        0.0150, 0.0153, 0.0154, 0.0160, 0.0155, 0.0155, 0.0155, 0.0157, 0.0155,
        0.0160, 0.0154, 0.0157, 0.0161, 0.0156, 0.0156, 0.0157, 0.0153, 0.0155,
        0.0157, 0.0161, 0.0163, 0.0153, 0.0158, 0.0147, 0.0160, 0.0152, 0.0155,
        0.0157])"""

class Contention():

    def sample(self, tokens_batch, embedding, mask=False):
        """
        The approach here is to randomly select all heads, and then resolve any
        issues by reselecting heads for all tokens involved in said issues.
        There are two possible issues: cycles, and having more than one node that
        selects <ROOT>.
        """

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        head_selector = torch.distributions.Categorical(torch.exp(energy))

        heads = head_selector.sample()
        i = 0
        while True:
            i += 1

            # Find contentions: cycles or multiple roots.
            has_cycle = self.has_cycle(heads)
            is_multiple_root = self.is_multiple_root(heads, mask)

            # Resample nodes involved in contentions.
            has_contention = has_cycle.logical_or(is_multiple_root)

            # Break once all sentences have no contentions.
            if not has_contention.any():
                break

            # Try selecting only one contentious element per sentence.
            # (For sentences without contention, harmlessly select <ROOT>.
            contenders = has_contention
            #contenders = self.keep_one_contender(has_contention)

            heads_sample = head_selector.sample()
            heads = torch.where(contenders, heads_sample, heads)

        return heads


    def keep_one_contender(self, has_contention):
        select_contender = torch.where(
            has_contention.sum(dim=1,keepdim=True).to(torch.bool),
            has_contention,
            one_hot(
                torch.tensor(0), has_contention.shape[1]
            ).to(torch.bool).unsqueeze(0)
        )
        contenders = torch.distributions.Categorical(select_contender).sample()
        contenders = one_hot(contenders, has_contention.shape[1]).to(torch.bool)
        return contenders


    def is_multiple_root(self, heads, mask):
        # Get the non-PADDING tokens that have ROOT as head.
        rooted = (heads == 0).logical_and(mask.logical_not())
        # Eliminate ROOT itself
        rooted[:,0] = False
        # Determine which sentences have more than one root.
        has_multiple_roots = rooted.sum(dim=1, keepdim=True) > 1
        return torch.where(has_multiple_roots, rooted, False)

    def has_cycle(self, heads):
        """
        A node is on a loop if it can reach itself by taking enough hops
        along directed edges.  Recursively index heads with itself, to 
        visit all ancestors of each node, and look for a node itself among
        it's ancestors.  If seen, this node is in a cycle.
        """
        max_steps = heads.shape[1]
        has_cycle = torch.full(heads.shape, False)
        self_head = torch.arange(
            heads.shape[1]).unsqueeze(0).expand(heads.shape)
        ancestors = heads.clone()
        for i in range(max_steps):
            has_cycle = has_cycle.logical_or(ancestors == self_head)
            ancestors = heads.gather(dim=1, index=ancestors)

        # ROOT is trivially cycled due to self-link, but we don't care about it.
        has_cycle[:,0] = 0
        return has_cycle
        

        

class ParityTokenResampler:

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


    def sample_tokens(self, tokens_batch, head_ptrs, embedding, mask=False):
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
        with torch.no_grad():
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


class CycleProofRooting:

    def sample(self, tokens_batch, embedding, mask=False):

        num_sentences, num_tokens = tokens_batch.shape

        # Track which nodes are "rooted" (have a path to ROOT), and which nodes 
        # are "rooting" (part of on an actively growing branch).
        rooted = torch.zeros_like(tokens_batch, dtype=torch.bool)
        rooted[mask] = True
        rooted[:,0] = True
        fully_rooted = torch.zeros((num_sentences,1), dtype=torch.bool)
        rooting = torch.zeros_like(tokens_batch, dtype=torch.bool)

        # Encode the partial parse structure as a choice of head by each token.
        head_ptrs = torch.full_like(tokens_batch, -1)
        head_ptrs[mask] = 0
        head_ptrs[:,0] = 0
        # Heads are chosen randomly with probability based on link energy.
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        head_probs = torch.exp(energy)

        # Track the head of the sentence (the token whose head is ROOT)
        # Default value 0 indicates that sentence head is not yet established.
        sentence_head = torch.zeros((num_sentences,1), dtype=torch.int64)

        # Each iteration, one node per sentence is considered "active".
        ptr_sampler = torch.distributions.Categorical(rooted.logical_not())
        ptr = ptr_sampler.sample().unsqueeze(1)
        # In each sentence, the "active" node chosen is considered in "rooting"  
        per_sentence = torch.arange(num_sentences)
        rooting[per_sentence,ptr.squeeze(1)] = True
        while True:

            this_head_ptr = torch.distributions.Categorical(
                head_probs[per_sentence, ptr.squeeze(1)]).sample().unsqueeze(1)
            head_ptrs[per_sentence, ptr.squeeze(1)] = this_head_ptr.squeeze(1)

            # Updates to our tree depend on whether we chose a rooted token
            # (a token with path to ROOT), and on whether we chose ROOT itself.
            chose_rooted = rooted.gather(dim=1, index=this_head_ptr)
            chose_ROOT = (this_head_ptr == 0)

            # Update the rooted / rooting status of nodes.
            new_rooting = self.update_rooting(
                rooting, rooted, chose_rooted, chose_ROOT, fully_rooted, mask)
            new_rooted = self.update_rooted(
                rooting, rooted, chose_rooted, chose_ROOT, fully_rooted, mask)
            rooting = new_rooting
            rooted = new_rooted
            fully_rooted = rooted.all(dim=1, keepdim=True)

            # As soon as we achieve full rooting of all sentences, we can 
            # stop.  Otherwise, we'll move the pointer and reiterate.
            if fully_rooted.sum() == num_sentences:
                break

            # Move the pointer and mark the node identified by ptr as rooting.
            new_ptr = self.update_ptr(
                rooted, chose_rooted, chose_ROOT, fully_rooted, this_head_ptr,
                sentence_head, 
            )
            rooting[per_sentence, new_ptr.squeeze(1)] = True
            rooting[:,0] = False

            # Update the sentence head.
            sentence_head = torch.where(chose_ROOT, ptr, sentence_head)

            ptr = new_ptr

        return head_ptrs


    def update_ptr(
        self, rooted, chose_rooted, chose_ROOT, fully_rooted, head_ptr,
        sentence_head
    ):
        # In some cases we move ptr to a randomly selected unrooted token.
        # Unrooted tokens are eligible for this random selection, but for
        # sentences where all tokens are rooted, harmlessly set the probability
        # of sampling ROOT to 1.  This avoids a zero-support sampling
        # error due for sentences with no unrooted tokens.
        eligible = rooted.logical_not()
        eligible = torch.where(
            fully_rooted,
            one_hot(torch.tensor(0), rooted.shape[1]).to(torch.bool),
            eligible
        )

        # As a default case assume ptr is moved to a randomly sampled token.
        ptr_sampler = torch.distributions.Categorical(eligible)
        ptr = ptr_sampler.sample().unsqueeze(1)

        # If we chose ROOT and there was a prior sentence head, make it ptr
        # that sentence head has been "bumped" and needs to choose a new head.
        prior_root_bumped = chose_ROOT.logical_and(sentence_head>0)
        ptr = torch.where(prior_root_bumped, sentence_head, ptr)

        # If we chose an unrooted token, then ptr moves to it.
        ptr = torch.where(
            chose_rooted.logical_not(), head_ptr, ptr)

        # Keep ptr on ROOT whenever the sentence is fully rooted.
        # This will stop updates to the sentence structure, since ROOT always
        # selects itself as head (preserving the default self-link that it has).
        #
        # TODO: this line is redundant due to how eligible is calculated.
        #       remove!
        ptr = torch.where(fully_rooted, 0, ptr)

        return ptr


    def update_rooting(
        self, rooting, rooted, chose_rooted, chose_ROOT, fully_rooted, mask
    ):
        # If you chose a rooted token, then rooting gets cleared.
        new_rooting = torch.where(chose_rooted, False, rooting)
        # If you chose ROOT, then rooting gets what was rooted.
        new_rooting = torch.where(chose_ROOT, rooted, new_rooting)
        # If you were fully rooted, then rooting is cleared.
        new_rooting = torch.where(fully_rooted, False, new_rooting)
        # ROOT is never rooting.
        new_rooting[:,0] = False
        # Padding is never rooting.
        new_rooting[mask] = False
        return new_rooting


    def update_rooted(
        self, rooting, rooted, chose_rooted, chose_ROOT, fully_rooted, mask
    ):
        # If you chose ROOT, then clear what was previously rooted.
        new_rooted = torch.where(chose_ROOT, False, rooted)
        # If you chose a rooted token, then add rooting to rooted.
        new_rooted = torch.where(
            chose_rooted, new_rooted.logical_or(rooting), new_rooted)
        # But overriding, if you were fully rooted, always stay that way
        new_rooted = torch.where(fully_rooted, rooted, new_rooted)
        # ROOT is always rooted.
        new_rooted[:,0] = True
        # Padding is always rooted.
        new_rooted[mask] = True
        return new_rooted



