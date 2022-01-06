import torch
from torch.nn.functional import one_hot
import pdb
import model as m



class ContentionParseSampler():

    def sample_parses(
        self, tokens_batch, embedding, mask=False, start_temp=1, temp_step=0.000
    ):
        """
        The approach here is to randomly select all heads, and then resolve any
        issues by reselecting heads for all tokens involved in said issues.
        There are two possible issues: cycles, and having more than one node that
        selects <ROOT>.
        """

        num_sentences, num_tokens = tokens_batch.shape
        temp = start_temp
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        contenders = torch.tensor([[True]]).expand(tokens_batch.shape)
        heads = torch.zeros(tokens_batch.shape, dtype=torch.long)
        while True:

            head_selector = torch.distributions.Categorical(
                torch.exp(energy/temp))
            temp += temp_step
            heads_sample = head_selector.sample()
            heads = torch.where(contenders, heads_sample, heads)

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


class CycleProofRootingParseSampler:

    def sample_parses(
        self, tokens_batch, embedding, mask, start_temp=1, temp_step=0.1
    ):

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

        # Track the head of the sentence (the token whose head is ROOT)
        # Default value 0 indicates that sentence head is not yet established.
        sentence_head = torch.zeros((num_sentences,1), dtype=torch.int64)

        # Each iteration, one node per sentence is considered "active".
        ptr_sampler = torch.distributions.Categorical(rooted.logical_not())
        ptr = ptr_sampler.sample().unsqueeze(1)
        # In each sentence, the "active" node chosen is considered in "rooting"  
        per_sentence = torch.arange(num_sentences)
        rooting[per_sentence,ptr.squeeze(1)] = True
        temp = start_temp - temp_step
        while True:

            temp += temp_step

            head_probs = torch.exp(energy / temp)
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
        new_rooted = torch.where(chose_root, False, rooted)
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




class WalkSampler:

    def sample_parses(
        self, tokens_batch, embedding, mask, start_temp=1, temp_step=0
    ):

        num_sentences, num_tokens = tokens_batch.shape

        # Track which nodes have been seen.
        seen = torch.zeros_like(tokens_batch, dtype=torch.bool)
        seen[mask] = True
        seen[:,0] = True

        # Encode the partial parse structure as a choice of head by each token.
        head_ptrs = torch.full_like(tokens_batch, -1)
        head_ptrs[mask] = 0
        head_ptrs[:,0] = 0

        # Heads are chosen randomly with probability based on link energy.
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        # Tokens never choose root in this sampler.  Instead root is the 
        # token first landed on whenever all tokens have been seen.
        energy[:,:,0] = -torch.inf
        energy[:,0,0] = 0
        print(energy[0])

        # Each iteration, one node per sentence is considered "active".
        # Choose from one of the non-ROOT non-PAD tokens.
        ptr_sampler = torch.distributions.Categorical(seen.logical_not())
        ptr = ptr_sampler.sample().unsqueeze(1)
        per_sentence = torch.arange(num_sentences)
        seen[per_sentence, ptr.squeeze(1)] = True

        # In each sentence, the "active" node chosen is considered in "rooting"  
        temp = start_temp - temp_step
        i = 0
        while i < 5*num_tokens or not seen.all():

            i += 1
            temp += temp_step

            # Have each pointed-to token select a head.
            head_probs = torch.exp(energy / temp)
            this_head_ptr = torch.distributions.Categorical(
                head_probs[per_sentence, ptr.squeeze(1)]).sample().unsqueeze(1)

            # Record the head selection made by the pointed-to token.
            head_ptrs[per_sentence, ptr.squeeze(1)] = this_head_ptr.squeeze(1)
            # The head is root until (and unless) next iteration occurs
            head_ptrs[per_sentence, this_head_ptr.squeeze(1)] = 0

            # Move the pointer to the newly selected head.
            ptr = this_head_ptr
            seen[per_sentence, ptr.squeeze(1)] = True

        print(f"{i} turns to parse")

        assert not torch.any(head_ptrs == -1), "Invalid parse exists."
        return head_ptrs


class SimpleRandomTree:

    def init_state(self, num_tokens, num_sentences, mask):
        tree = torch.full((num_sentences, num_tokens,),-1, dtype=torch.int64)
        tree[:,0] = 0
        tree[mask] = 0
        intree = torch.zeros((num_sentences, num_tokens,), dtype=torch.bool)
        intree[:,0] = True
        intree[mask] = True
        leaf_ptr = torch.distributions.Categorical(
            intree.logical_not()).sample()
        head_ptr = leaf_ptr.clone()
        has_root = torch.zeros((num_sentences,), dtype=torch.bool)
        return tree, intree, leaf_ptr, head_ptr, has_root


    def sample_parses(self, tokens, embedding, mask):

        num_sentences, num_tokens = tokens.shape

        # Adjust the probabilities to make p~
        probs = torch.exp(embedding.sentence_link_energy(tokens, mask))
        probs[:,:,0] = 1
        probs = self.get_tilde_probs(probs, mask)
        epsilon = torch.full((num_sentences, 1), 1.0)
        probs[:,:,0] = epsilon

        # Various pointers and booleans track the state of tokens and sentences.
        state = self.init_state(num_tokens, num_sentences, mask)
        tree, intree, leaf_ptr, head_ptr, has_root = state
        done = intree.all(dim=1)

        per_sentence = torch.arange(num_sentences)
        i = 0
        while not done.all():

            # In each sentence, the token head_ptr choses a next head.
            next_head_ptr = torch.distributions.Categorical(
                probs[per_sentence, head_ptr]).sample()
            tree[per_sentence, head_ptr] = next_head_ptr
            head_ptr = next_head_ptr

            # If multiple tokens in a sentence choose root, restart parsing of
            # that sentence (reinitialize the state data associated to it).
            chose_root = (next_head_ptr == 0)
            needs_reset = chose_root.logical_and(
                has_root).logical_and(done.logical_not())
            has_root[chose_root] = True
            if needs_reset.any():
                epsilon[needs_reset] = epsilon[needs_reset] / 2
                probs[needs_reset,:,0] = epsilon[needs_reset]
                tree[needs_reset] = torch.full(
                    (needs_reset.sum(), num_tokens), -1)
                tree[:,0] = 0
                tree[mask] = 0
                intree[needs_reset] = False
                intree[:,0] = True
                intree[mask] = True
                leaf_ptr[needs_reset] = torch.distributions.Categorical(
                    intree[needs_reset].logical_not()).sample()
                head_ptr[needs_reset] = leaf_ptr[needs_reset]
                has_root[needs_reset] = False

            # Whenever a next head is chosen that is intree, mark everything
            # from leaf_ptr to next_head_ptr as intree.  Some sentences will
            # have longer paths than others, but ones that reach ROOT first will
            # cycle harmlessly there.
            anchored = intree[per_sentence, next_head_ptr]
            if anchored.any():
                while not intree[anchored, leaf_ptr[anchored]].all():
                    intree[anchored, leaf_ptr[anchored]] = True
                    leaf_ptr[anchored] = tree[anchored, leaf_ptr[anchored]]

                done = intree.all(dim=1)
                needs_ptr_reset = anchored.logical_and(done.logical_not())
                leaf_ptr[needs_ptr_reset] = torch.distributions.Categorical(
                    intree[needs_ptr_reset].logical_not()).sample()
                head_ptr[needs_ptr_reset] = leaf_ptr[needs_ptr_reset]
                # Any sentence that has a viable tree is forced into the
                # absorbing state.
                leaf_ptr[done] = 0
                head_ptr[done] = 0

        assert not (tree == -1).any(), "parse invalid."
        return tree

            
    def get_tilde_probs(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        needed_self_weight = max_out_weight - out_weights
        probs = probs + torch.diag_embed(needed_self_weight, dim1=1, dim2=2)
        probs[:,0,0] = 1
        probs[mask.unsqueeze(1).expand(probs.shape)] = 0
        return probs


class RandomTree:
    """
    As listed on p.204 of Propp and Wilson's "How to get a perfectly random 
    sample from a generic markov chain and generate a random spanning tree of
    a directed graph", implemented to work on multiple sentences in parallel.
    """

    def sample_parses(
        self, tokens_batch, embedding, mask, start_temp=1, temp_step=0.1
    ):

        num_sentences, num_tokens = tokens_batch.shape

        # Track which nodes are "intree" (have a path to ROOT), and which nodes 
        # are "rooting" (part of on an actively growing branch).
        intree = torch.zeros_like(tokens_batch, dtype=torch.bool)
        intree[mask] = True
        intree[:,0] = True

        # Encode the partial parse structure as a choice of head by each token.
        head_ptrs = torch.full_like(tokens_batch, -1)
        head_ptrs[mask] = 0
        head_ptrs[:,0] = 0

        # Heads are chosen randomly with probability based on link energy.
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        # The probability that any given node chooses root is the same for
        # all tokens in a sentence, given by epsilon.  Although all tokens in 
        # the sentence have the same epsilon, different sentences have
        # different epsilons.
        epsilon = torch.full((num_sentences, 1), 1.0)

        # Keep track of whether a given sentence has root.
        has_root = torch.zeros((num_sentences,1), dtype=torch.bool)
        done = torch.zeros((num_sentences, 1), dtype=torch.bool)

        # We continually take "jaunts" of several steps starting from a sub_ptr.
        # During the juant, our position is stored in head_ptr.
        # To start with, pick a random sub_ptr (and this is where head_ptr
        # starts too.
        ptr_sampler = torch.distributions.Categorical(intree.logical_not())
        ptr = ptr_sampler.sample().unsqueeze(1)
        head_ptr = ptr.clone()

        # In each sentence, the "active" node chosen is considered in "rooting"  
        per_sentence = torch.arange(num_sentences)
        temp = start_temp - temp_step

        while not done.all():

            temp += temp_step

            head_probs = self.get_tilde_probs(energy / temp, epsilon, mask)

            next_head_ptr = torch.distributions.Categorical(
                head_probs[per_sentence, head_ptr.squeeze(1)]
            ).sample().unsqueeze(1)
            head_ptrs[per_sentence, head_ptr.squeeze(1)] = (
                next_head_ptr.squeeze(1))

            # Is any head_ptr pointing to a node that is intree?
            chose_intree = intree.gather(dim=1, index=next_head_ptr)
            intree = self.update_intree(intree, chose_intree, ptr, head_ptrs)

            # Did any sentence choose ROOT for a second time?
            # If so, reset intree (start building tree from scratch) and 
            # cut the chance of selecting root in half.
            chose_root = (next_head_ptr == 0)
            double_root = chose_root.logical_and(
                has_root).logical_and(done.logical_not()).squeeze(1)
            has_root = has_root.logical_or(chose_root)
            has_root[double_root] = False
            intree[double_root] = False
            intree[double_root] = mask[double_root]
            intree[:,0] = True
            epsilon[double_root] = epsilon[double_root] / 2
            energy[:,:,0] = epsilon

            # if done, harmlessly set ptr to ROOT.
            done = intree.all(dim=1, keepdim=True)
            head_ptr[done] = 0

            # If chose_intree and not done, sample a new pointer
            needs_ptr_reset = (
                done.logical_not().logical_and(chose_intree).squeeze(1))
            ptr_sampler = torch.distributions.Categorical(
                intree[needs_ptr_reset].logical_not())
            ptr[needs_ptr_reset] = ptr_sampler.sample().unsqueeze(1)
            head_ptr[needs_ptr_reset] = ptr[needs_ptr_reset]

            # Otherwise set the new head_ptr to next_head_ptr
            proceed = done.logical_not().logical_and(chose_intree.logical_not())
            head_ptr[proceed] = next_head_ptr[proceed]

        return head_ptrs


    def get_tilde_probs(self, energy, epsilon, mask):
        probs = torch.exp(energy)
        probs[:,:,0] = epsilon
        out_weight = probs.sum(dim=2)
        max_out_weight = out_weight.max(dim=1, keepdim=True).values
        add_weight = max_out_weight - out_weight
        probs = probs + torch.diag_embed(add_weight, dim1=1, dim2=2)
        probs[mask.unsqueeze(1).expand(probs.shape)] = 0
        probs[:,0,0] = 1
        return probs


    def update_intree(self, intree, chose_intree, ptr, head_ptrs):
        # Walk from ptr to intree, marking tokens as intree.
        # Once a sentence reaches ROOT, it stays there because root is it's own
        # head, harmlessly marking ROOT as intree (it's default).  Once all
        # this_ptrs have reach an token that is already intree, stop updating.
        chose_intree_index = chose_intree.squeeze(1)
        this_ptr = ptr[chose_intree_index]
        traced = intree[chose_intree_index].gather(dim=1, index=this_ptr)
        while not traced.all():
            intree[chose_intree_index, this_ptr.squeeze(1)] = True
            this_ptr = head_ptrs[chose_intree_index].gather(
                dim=1, index=this_ptr)
            traced = intree[
                chose_intree.squeeze(1)].gather(dim=1, index=this_ptr)
        return intree


