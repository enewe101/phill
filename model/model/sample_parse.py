import torch
from torch.nn.functional import one_hot
import pdb
import model as m



class Contention():

    def sample(self, tokens_batch, embedding):
        """
        The approach here is to randomly select all heads, and then resolve any
        issues by reselecting heads for all tokens involved in said issues.
        There are two possible issues: cycles, and having more than one node that
        selects <ROOT>.
        """

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch)
        head_selector = torch.distributions.Categorical(torch.exp(energy))

        heads = head_selector.sample()
        i = 0
        while True:
            i += 1

            # Find contentions: cycles or multiple roots.
            has_cycle = self.has_cycle(heads)
            is_multiple_root = self.is_multiple_root(heads)

            # Break once all sentences have no contentions.
            if (has_cycle.sum() + is_multiple_root.sum()) == 0:
                break

            # Resample nodes involved in contentions.
            has_contention = has_cycle.logical_or(is_multiple_root)

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


    def is_multiple_root(self, heads):
        # <ROOT> is self-linked, so having multiple non-<ROOT> nodes linked
        # to <ROOT> occurs when we have 3 or more nodes linked to <ROOT>.
        has_multiple_roots = ((heads == 0).sum(dim=1) > 2).unsqueeze(1)
        is_rooted = (heads == 0)
        is_rooted[:,0] = False
        return torch.where(has_multiple_roots, is_rooted, False)


    def has_cycle(self, heads):
        """
        A node is on a loop if it can reach itself by taking enough hops
        along directed edges.  To find the nodes that participate in cycles,
        continually "make hops" by taking higher and higher powers of the 
        adjacency matrix (built from ``heads``).
        I.e., A tells you what node j can be reached from i in one hop, while
        A^2 tells nodes j reachable in two hops from i.  
        if you take A + A^2 + A^3 + ... A^n, where n is the number of tokens
        in the sentence, then even the longest loop will visit all its members.
        So, the diagonal of sum_n(A^n) tells you which nodes participate in 
        loops.
        """
        max_steps = heads.shape[1]
        adjacency = torch.nn.functional.one_hot(
            heads, heads.shape[1]).to(torch.int)
        seen_heads = adjacency
        for i in range(max_steps):
            seen_heads = (
                seen_heads.logical_or(seen_heads @ adjacency).to(torch.int))

        has_cycle = seen_heads.diagonal(dim1=1, dim2=2)
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


"""
tensor([0.0165, 0.0149, 0.0167, 0.0147, 0.0161, 0.0144, 0.0165, 0.0145, 0.0160,
        0.0163, 0.0145, 0.0163, 0.0148, 0.0174, 0.0159, 0.0145, 0.0162, 0.0148,
        0.0167, 0.0140, 0.0161, 0.0144, 0.0163, 0.0148, 0.0163, 0.0168, 0.0143,
        0.0166, 0.0148, 0.0162, 0.0161, 0.0146, 0.0165, 0.0144, 0.0166, 0.0142,
        0.0164, 0.0148, 0.0165, 0.0145, 0.0161, 0.0164, 0.0149, 0.0165, 0.0144,
        0.0167, 0.0168, 0.0146, 0.0166, 0.0152, 0.0163, 0.0148, 0.0156, 0.0143,
        0.0163, 0.0147, 0.0166, 0.0166, 0.0145, 0.0171, 0.0147, 0.0165, 0.0165,
        0.0148])"""

class CycleProofRerooting2:

    def sample(self, tokens_batch, embedding, mask=False):

        num_sentences, num_tokens = tokens_batch.shape

        # Track which nodes are "rooted" (have a path to ROOT), and which nodes 
        # are "rooting" (part of on an actively growing branch).
        # PLAN: padding should always be rooted, never rooting.
        #       padding's head should always ROOT.
        #       energy for padding as head is always zero, with padding as
        #       sub, zero except for with ROOT.
        #       pointer should never be padding.
        #       sentence head can never be root
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
        head_sampler = torch.distributions.Categorical(torch.exp(energy))

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

            # Sample a head for each ptr, and update tree stored in heads. 
            # TODO: Right now this samples a head for every token in every 
            #       sentence, but we only use the sampled head for ptr.
            #       Is it faster to make a new sampler, just for ptr on each
            #       round?
            #
            head_ptr_sample = head_sampler.sample()
            this_head_ptr = head_ptr_sample.gather(dim=1, index=ptr)
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


class CycleProofRerooting:

    def sample(self, tokens_batch, embedding, mask=False):

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch)
        head_selector = torch.distributions.Categorical(torch.exp(energy))

        heads_adj = torch.zeros(
            (tokens_batch.shape[0], tokens_batch.shape[1],tokens_batch.shape[1]),
            dtype=torch.long
        )
        heads_adj[:,0,0] = 1
        heads = torch.full(tokens_batch.shape, -1)
        heads[:,0] = 0

        # Track which tokens are on a branch attached to <ROOT>.
        rooted = torch.zeros(tokens_batch.shape, dtype=torch.bool)
        rooted[:,0] = True
        # Track which tokens are on a branch actively selecting its head.
        rooting = torch.zeros(tokens_batch.shape, dtype=torch.bool)

        # Pointer is the active node currently selecting its head.
        pointer_selector = torch.distributions.Categorical(rooted.logical_not())
        pointer = pointer_selector.sample().unsqueeze(1)

        # The head of the sentence is whatever attaches to <ROOT>.  Only one
        # is allowed when all is said and done.  New attachments to <ROOT>
        # bump old ones.
        sentence_head = torch.zeros(tokens_batch.shape[0], dtype=torch.long)

        # Keep track of whether next_head is <ROOT>.
        chose_root = torch.zeros(tokens_batch.shape[0], dtype=torch.bool)

        sentence_index = torch.arange(
            tokens_batch.shape[0], dtype=torch.long)
        num_turns = torch.tensor(0)
        while True:

            num_turns += 1

            head_sample = head_selector.sample()
            next_head = head_sample.gather(dim=1, index=pointer).squeeze(1)

            # 1: token chose an unrooted token
            # 2: token chose rooted token
            # 3: token chose <ROOT> for the first time
            # 4: token chose <ROOT> for the nth time
            # 5: root chose <ROOT> and we are fully rooted

            heads[sentence_index,pointer.squeeze()] = next_head

            # If we chose a new head for the sentence, record that.
            chose_root = (next_head.unsqueeze(1) == 0)

            # Mark this token as "rooting"
            rooting[sentence_index, pointer.squeeze()] = True

            fully_rooted = rooted.logical_not().sum(dim=1) == 0
            new_rooted = (
                # If next_head was rooted, everything rooting becomes rooted
                rooted[sentence_index, next_head].unsqueeze(1) 
                * rooting  + 
                # If we did not chose a new sentence head, preserve rooted
                chose_root.logical_not() * rooted + 
                # Or if we are fully rooted, then also preserve rooted
                fully_rooted.unsqueeze(1) * rooted
                # Otherwise next_head must be <ROOT> and there is a contest
                # with a previous sentence head
            )
            # Regardlesss, <ROOT> (position 0) is always rooted.
            new_rooted[:,0] = True

            # If next_head is root then the previous sentence head is 
            # bumped, so it and its children are once again rooting.
            # If we chose an unrooted token, then we should just add that
            # to rooting.  If neither condition holds (I.e. we are joining
            # an already-rooted sub-tree) then rooting just resets.
            rooting = (
                # If chose <ROOT>, previously rooted becomes rooting (bumped)
                # Except if we have fully rooted the sentence
                chose_root * fully_rooted.logical_not().unsqueeze(1) * rooted + 
                # If chose a rooted token, rooting becomes rooted.
                ~rooted[sentence_index, next_head].unsqueeze(1) * rooting
            )
            # <ROOT> is never rooting (it is always already rooted)
            rooting[:,0] = False

            # this will either be empty or it will contain
            # the children of the previous head.
            rooted = new_rooted

            # Update fully rooted again, having recalculated rooted and rooting
            fully_rooted = rooted.logical_not().sum(dim=1) == 0



            # Break if fully rooted.
            if fully_rooted.sum() == tokens_batch.shape[0]:
                break




            # if next_head was rooted, sample an unrooted pointer, otherwise
            # use next_head
            make_root_absorbing = fully_rooted.unsqueeze(1) * one_hot(
                torch.tensor(0), tokens_batch.shape[1])
            pointer_probabilities =  rooted.logical_not() + make_root_absorbing
            pointer_selector = torch.distributions.Categorical(
                pointer_probabilities
            )

            do_sample = (
                chose_root.logical_not()
                * rooted[sentence_index, next_head].unsqueeze(1)
                + chose_root * (sentence_head == 0).unsqueeze(1)
            )

            new_pointer = (
                do_sample.squeeze(1) * pointer_selector.sample()
                + next_head * (
                    rooted.logical_not()[sentence_index, next_head])
                + chose_root.squeeze(1) * sentence_head
            ).unsqueeze(1)

            new_pointer =(
                fully_rooted.logical_not().unsqueeze(1) * new_pointer)

            # Points to the current sentence head
            sentence_head = (
                chose_root * pointer 
                + chose_root.logical_not() * sentence_head.unsqueeze(1)
            ).squeeze(1)

            pointer = new_pointer

        #energies = embedding.link_energy(tokens_batch, heads)

        return heads



class RootReset:

    def sample(self, tokens_batch, embedding):

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch)
        head_ptr_selector = torch.distributions.Categorical(torch.exp(energy))
        inbound_weight = torch.exp(energy).transpose(1,2).sum(dim=2)
        inbound_weight[:,0] = 0
        ptr_resetter = torch.distributions.Categorical(inbound_weight)

        head_ptr = torch.zeros(tokens_batch.shape, dtype=torch.long)
        touched = torch.zeros(tokens_batch.shape, dtype=torch.bool)
        finished = torch.zeros(tokens_batch.shape, dtype=torch.bool)

        per_sentence = torch.arange(num_sentences)
        ptr = torch.randint(1,tokens_batch.shape[1], (tokens_batch.shape[0],))
        while True:
            touched[per_sentence, ptr] = True

            if touched[:,0].sum() == tokens_batch.shape[0]:
                break

            head_ptr_sample = head_ptr_selector.sample()
            head_ptr_next = head_ptr_sample[per_sentence, ptr]

            # If we select root as head before all is touched, we'll reject
            # it (won't record it in head).
            not_all_touched = touched.sum(dim=1) < (tokens_batch.shape[1] - 1)
            head_is_root = (head_ptr_next == 0)
            reject_and_reset = not_all_touched * head_is_root

            # "Reject" means "don't update" head
            head_ptr[per_sentence, ptr] = torch.where(
                reject_and_reset, 
                head_ptr[per_sentence, ptr], 
                head_ptr_next
            )

            # "Reset" means ptr does not count as touched because head was
            # rejected.
            touched[per_sentence, ptr] = torch.where(
                reject_and_reset, False, True
            )

            # If we accepted the head, it becomes the new pointer.
            # If we rejected the head, we sample a new pointer.
            #ptr_reset = torch.randint(
            #    1,tokens_batch.shape[1], (tokens_batch.shape[0],))
            ptr_weight = touched.logical_not() 
            ptr_weight[:,0] = touched.sum(dim=1) >= (tokens_batch.shape[1]-1)

            ptr_new = torch.where(
                reject_and_reset, 
                ptr_reset,
                head_ptr[per_sentence, ptr]
            )
            ptr = ptr_new

        head = tokens_batch.gather(dim=1, index=head_ptr)
        return head


