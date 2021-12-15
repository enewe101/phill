import torch
from torch.nn.functional import one_hot
import pdb


        # One way to sample only trees is to do as follows
        #   1. sample one edge.
        #   2. always consider the "cursor" to point to the head in the most 
        #       recently selected edge.
        #   3. Sample from the cursor's possible choices of head.
        #   4. Repeat 3 until we select the root.  Mark all tokens attached
        #       to an edge so far as "rooted"
        #   5. Allow every unrooted token to sample a new head.
        #   6. Mark any token that takes a rooted token as its head as rooted.
        #   7. Continue until all tokens are rooted.

        #Sample from trees.  Cannot express conflict or crowding between
        #tokens as they select their head.
        # 1. calculate the vector-covector energies
        # 2. sample one edge with head root. the subordinate is "rooted".
        # 3. sample one unrooted subordinate having rooted head.
        # 4. repeat until there are no unrooted subordinates.


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

        print(num_sentences, "sentences parsed in",i,"turns.")
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
        iteratively move token i's head-pointer to that of it's head.
        The pointer will eventually either cycle or absorb to zero.
        The pointer advances 2^i steps in each iteration, so
        even a structure of length n will root in log n steps.
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
        



class CycleProofRerooting2:

    def sample(self, tokens_batch, embedding):

        num_sentences, num_tokens = tokens_batch.shape

        # Track which nodes are "rooted" (have a path to ROOT), and which nodes 
        # are "rooting" (part of on an actively growing branch).
        rooted = torch.zeros_like(tokens_batch, dtype=torch.bool)
        rooted[:,0] = True
        fully_rooted = torch.zeros((num_sentences,1), dtype=torch.bool)
        rooting = torch.zeros_like(tokens_batch, dtype=torch.bool)

        # Encode the partial parse structure as a choice of head by each token.
        heads = torch.full_like(tokens_batch, -1)
        heads[:,0] = 0
        # Heads are chosen randomly with probability based on link energy.
        energy = embedding.sentence_link_energy(tokens_batch)
        head_sampler = torch.distributions.Categorical(torch.exp(energy))

        # Track the head of the sentence (the token whose head is ROOT)
        # Default value 0 indicates no sentence head is yet established.
        sentence_head = torch.zeros((num_sentences,1), dtype=torch.int64)

        # Each iteration, one node per sentence (ptr) actively selects a head.
        # This node, and its ancestors are "rooting".
        ptr_sampler = torch.distributions.Categorical(rooted.logical_not())
        ptr = ptr_sampler.sample().unsqueeze(1)
        per_sentence = torch.arange(num_sentences)
        rooting[per_sentence,ptr.squeeze(1)] = True
        i = 0
        while True:
            i += 1
            #print(i)

            # Sample a head for the node identified by ptr.
            head_sample = head_sampler.sample()
            head_choice = head_sample.gather(dim=1, index=ptr)
            heads[per_sentence, ptr.squeeze(1)] = head_choice.squeeze(1)
            #print("ptr");print(ptr)
            #print("heads");print(heads)

            # Updates to our tree depend on whether we chose a rooted token
            # (a token with path to ROOT), and whether we chose ROOT itself.
            chose_rooted = rooted.gather(dim=1, index=head_choice)
            chose_ROOT = (head_choice == 0)

            # Update the rooted / rooting status of nodes.
            new_rooting = self.update_rooting(
                rooting, rooted, chose_rooted, chose_ROOT, fully_rooted)
            new_rooted = self.update_rooted(
                rooting, rooted, chose_rooted, chose_ROOT, fully_rooted)
            rooting = new_rooting
            rooted = new_rooted
            fully_rooted = rooted.all(dim=1, keepdim=True)

            # As soon as we achieve full rooting of all sentences, we can 
            # stop.  Otherwise, we'll move the pointer and reiterate.
            if fully_rooted.sum() == num_sentences:
                print("fully rooted after", i, "turns.")
                break

            # Move the pointer and mark the node identified by ptr as rooting.
            new_ptr = self.update_ptr(
                rooted, chose_rooted, chose_ROOT, fully_rooted, head_choice,
                sentence_head, 
            )
            rooting[per_sentence, new_ptr.squeeze(1)] = True
            rooting[:,0] = False
            #print("rooting");print(rooting)
            #print("rooted");print(rooted)
            #print("new_ptr");print(new_ptr)

            # Update the sentence head.
            sentence_head = torch.where(chose_ROOT, ptr, sentence_head)

            #pdb.set_trace()
            ptr = new_ptr

        return heads


    def update_ptr(
        self, rooted, chose_rooted, chose_ROOT, fully_rooted, head_choice,
        sentence_head
    ):
        # In some cases we move ptr to a randomly selected unrooted token.
        # Unrooted tokens are eligible for this random selection, but for
        # sentences where all tokens are rooted, harmlessly set the probability
        # of sampling ROOT to 1.  This avoids a zero-support sampling
        # error due for sentences with unrooted tokens.
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
            chose_rooted.logical_not(), head_choice, ptr)

        # Keep ptr on ROOT whenever the sentence is fully rooted.
        # This will stop updates to the sentence structure, since ROOT always
        # selects itself as head (preserving the default self-link that it has).
        ptr = torch.where(fully_rooted, 0, ptr)

        return ptr


    def update_rooting(
        self, rooting, rooted, chose_rooted, chose_ROOT, fully_rooted
    ):
        # If you chose a rooted token, then rooting gets cleared
        new_rooting = torch.where(chose_rooted, False, rooting)
        # If you chose ROOT, then rooting gets what was rooted
        new_rooting = torch.where(chose_ROOT, rooted, new_rooting)
        # If you were fully rooted, then rooting is cleared.
        new_rooting = torch.where(fully_rooted, False, new_rooting)
        # ROOT is never rooting
        new_rooting[:,0] = False
        return new_rooting


    def update_rooted(
        self, rooting, rooted, chose_rooted, chose_ROOT, fully_rooted
    ):
        # If you chose ROOT, then clear what was previously rooted
        new_rooted = torch.where(chose_ROOT, False, rooted)
        # If you chose a rooted token, then add rooting to rooted
        new_rooted = torch.where(
            chose_rooted, new_rooted.logical_or(rooting), new_rooted)
        # But overriding, if you were fully rooted, always stay that way
        new_rooted = torch.where(fully_rooted, rooted, new_rooted)
        # ROOT is always rooted
        new_rooted[:,0] = True
        return new_rooted


class CycleProofRerooting:

    def sample(self, tokens_batch, embedding):

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

            #if(chose_root):
            #    print("do sample:", do_sample)
            #    pdb.set_trace()

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

        #ptr = ptr_resetter.sample()
        per_sentence = torch.arange(num_sentences)
        ptr = torch.randint(1,tokens_batch.shape[1], (tokens_batch.shape[0],))
        while True:
            touched[per_sentence, ptr] = True

            if touched[:,0].sum() == tokens_batch.shape[0]:
                break

            head_ptr_sample = head_ptr_selector.sample()
            head_ptr_next = head_ptr_sample[per_sentence, ptr]
            print(ptr[0], "=>", head_ptr_next[0])

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
            try:
                ptr_reset = torch.distributions.Categorical(ptr_weight).sample()
            except:
                pdb.set_trace()

            ptr_new = torch.where(
                reject_and_reset, 
                ptr_reset,
                head_ptr[per_sentence, ptr]
            )
            ptr = ptr_new

        head = tokens_batch.gather(dim=1, index=head_ptr)
        return head
