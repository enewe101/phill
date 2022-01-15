import torch
from torch.nn.functional import one_hot
import pdb
import model as m
import time

ROOT_MASS = 0.5


class ConvSampler:
    
    default_kernel = [0,5,4,3,2,1]
    def __init__(self, kernel=None):
        self.kernel = self.default_kernel if kernel is None else kernel


    def get_head_probs(self, tokens_batch, mask):
        num_sentences, num_tokens = tokens_batch.shape

        # First we make the probability convolution for one sentence, ignoring
        # the mask.
        head_probs = torch.zeros((num_tokens, num_tokens))
        for i,v in enumerate(self.kernel):
            # Handle sentences shorter than kernel properly.
            if i > num_tokens:
                break
            diag = torch.full((num_tokens-i,), v).diagflat(i)
            head_probs += diag
        head_probs += head_probs.T.clone()
        head_probs[:,0] = 0
        head_probs[0] = one_hot(torch.tensor(0), num_tokens)

        # Now copy this convolution for all sentences, and we can then apply
        # sentence-specific masking.
        head_probs = head_probs.unsqueeze(0).expand(
            (num_sentences,num_tokens,num_tokens)).clone()

        # Tokens should not choose padding as head.
        head_probs[mask.unsqueeze(1).expand(head_probs.shape)] = 0
        # Padding never chooses non-<ROOT> tokens as head.
        head_probs[mask.unsqueeze(2).expand(head_probs.shape)] = 0
        # Padding always chooses <ROOT>.
        head_probs[:,:,0][mask] = 1

        return head_probs

    def sample_parses(self, tokens_batch, mask):
        head_probs = self.get_head_probs(tokens_batch, mask)
        head_selector = torch.distributions.Categorical(head_probs)
        head_ptrs = head_selector.sample()
        return head_ptrs



class RebasedConvSampler(ConvSampler):

    def __init__(self, Nx, kernel=None):
        super().__init__(kernel)
        self.Nx = Nx

    def get_head_probs(self, tokens_batch, mask):
        num_sentences, num_tokens = tokens_batch.shape
        head_probs = super().get_head_probs(tokens_batch, mask)
        unigram_counts = self.Nx.unsqueeze(0).expand(
            (num_sentences, -1)).gather(dim=1, index=tokens_batch
        ).unsqueeze(1).expand(head_probs.shape)
        return head_probs



class ContractionRandomTree():

    energy_epsilon_adjust = -0.09
    def sample_parses(s, tokens_batch, embedding, mask):
        heads = s.contract(tokens_batch, embedding, mask)
        trees = s.extract()
        return trees


    def extract(s):

        s.to_dismantle = torch.full((s.num_sentences, s.num_tokens), -1)

        # Begin by dismantling around <ROOT>.  This will enqueue super nodes
        # to be dismantled.
        s.dismantle(torch.zeros((s.num_sentences,)))

        # Now dismantle super_nodes that got queued.
        active = (s.to_dismantle[:,0] != -1)
        while active.any():

            # Pop an element per sentence to dismantle.
            ptr = s.to_dismantle[:,0]
            s.to_dismantle = s.to_dismantle[:,1:]

            # The outgoing arc from ptr is kept in the final tree.
            src_head = torch.full((s.num_sentences, 2), -1)
            head_ptr = ptr[active].unsqueeze(1).unsqueeze(2).expand(-1,-1,2)
            src_head = s.heads[active].gather(index=head_ptr, dim=1).squeeze(1)
            src = torch.full((s.num_sentences,) , -1)
            src[active] = src_head[:,0]
            s.heads[active, src[active]] = src_head  # inserts into tree.

            # Dismantle src (this may add super nodes to s.to_dismantle
            s.dismantle(src)

            # Where is there still work to do?
            active = (s.to_dismantle[:,0] != -1)

        return s.heads[:,:s.num_tokens,1]


    def dismantle(s, ptr):

        # Pointer is active if it is not -1 and if it's parent is not -1
        ptr_active = (ptr != -1)
        parent = s.parent[ptr_active, ptr[ptr_active]]
        ptr_active[ptr_active.clone()] = (parent != -1)
        j = 0
        while ptr_active.any():
            print("j", j)
            j += 1

            # Find siblings of 
            # Get next sibling.
            sibling = torch.zeros((s.num_sentences,2*s.num_tokens), dtype=bool)
            sibling[ptr_active] = s.children[ptr_active, parent]

            # Unset each sibling's parent
            s.parent[sibling] = -1

            # Get the siblings that have children.
            has_children = s.children.any(dim=2)
            sib_has_children = sibling.logical_and(has_children)

            # Add siblings with children to the dismantling "queue".
            start_index = s.next_available_slot().unsqueeze(1)
            end_index = start_index + sib_has_children.sum(dim=1).unsqueeze(1)
            position = torch.arange(2*s.num_tokens)
            position = position.unsqueeze(0).expand((
                s.num_sentences, 2*s.num_tokens))
            allocation = (
                position[:,:s.to_dismantle.shape[1]] >= start_index
            ).logical_and(
                position[:,:s.to_dismantle.shape[1]] < end_index
            )
            s.to_dismantle[allocation] = position[sib_has_children]

            pdb.set_trace()

            # Move pointer to parent.  Apply same activity criterion as before.
            ptr[ptr_active] = s.parent[ptr_active, ptr[ptr_active]]
            ptr_active = (ptr != -1)
            parent = s.parent[ptr_active, ptr[ptr_active]]
            ptr_active[ptr_active.clone()] = (parent != -1)


    def next_available_slot(s):
        return s.to_dismantle.shape[1] - (s.to_dismantle==-1).sum(dim=1)


    def get_preroot(s):
        preroot = (s.heads[:,:,1] == 0)
        preroot[:,0] = False
        preroot = preroot.to(torch.int).argmax(dim=1)
        return preroot


    def contract(s, tokens_batch, embedding, mask):
        s.num_sentences, s.num_tokens = tokens_batch.shape
        s.per_sentence = torch.arange(s.num_sentences)

        s.energy = embedding.sentence_link_energy(tokens_batch, mask)
        s.energy[:,:,0] = torch.log(s.init_epsilon(torch.exp(s.energy), mask))
        s.energy[mask,:] = -torch.inf
        s.heads = torch.full((s.num_sentences, 2*s.num_tokens, 2), -1)
        s.mask = mask
        s.ex_mask = torch.cat(
            (mask, torch.zeros(mask.shape, dtype=torch.bool)), dim=1)
        s.is_done = torch.zeros((s.num_sentences,), dtype=torch.bool)
        s.num_real_tokens = s.num_tokens - s.mask.sum(dim=1) - 1
        s.children = torch.zeros(
            (s.num_sentences, 2*s.num_tokens, 2*s.num_tokens), dtype=torch.bool)
        s.cycle_heads = torch.full((s.num_sentences, 2*s.num_tokens), -1)
        s.parent = torch.full((s.num_sentences, 2*s.num_tokens), -1)
        s.great_parent = torch.arange(2*s.num_tokens).unsqueeze(0).expand(
            (s.num_sentences, 2*s.num_tokens)).clone()

        s.descendents = torch.zeros(
            (s.num_sentences, 2*s.num_tokens, s.num_tokens), dtype=torch.bool)
        s.descendents[:, :s.num_tokens, :s.num_tokens] = torch.eye(
            s.num_tokens, dtype=torch.bool).unsqueeze(0).expand(
                (s.num_sentences, s.num_tokens, s.num_tokens))

        i = 0
        s.super_ptrs = torch.ones(s.num_sentences, dtype=torch.long)
        while not s.is_done.all():
            i += 1

            # TODO: all mutation ops should dodge sentences that are done.

            # Each (super)node given by super_ptrs selects a head.
            s.heads[s.per_sentence, s.super_ptrs, :] = s.pick_head(s.super_ptrs)
            src = s.heads[s.per_sentence,s.super_ptrs,0]
            head = s.heads[s.per_sentence,s.super_ptrs,1]

            # head is a simple node.  Get widest super_node that contains it.
            super_head = s.get_root_cycle(head)
            s.cycle_heads[s.per_sentence,s.super_ptrs] = super_head

            # If we select an already-seen node, we have made a cycle.
            # A node is already seen if its head has a head.
            head_of_head = s.heads.gather(
                dim=1, index=head.unsqueeze(1).unsqueeze(2)
            ).squeeze(2).squeeze(1)
            no_cycle = (head_of_head == -1).logical_or(s.super_ptrs==0)
            has_cycle = no_cycle.logical_not()

            # If we have not made a cycle, simply move ahead to the new node.
            s.super_ptrs[no_cycle] = head[no_cycle]

            # If cycle, contract cycle and set ptr to this super node.
            if has_cycle.any():
                s.super_ptrs[has_cycle] = s.contract_cycle(has_cycle)

                # Sentence is done when a super node subsumes all simple tokens.
                cycle_descendents = (
                    s.descendents[has_cycle, s.super_ptrs[has_cycle]])
                num_cycle_descendents = cycle_descendents.sum(dim=1)
                num_real_tokens = s.num_tokens - s.mask[has_cycle].sum(dim=1)
                s.is_done[has_cycle] = (num_cycle_descendents == num_real_tokens)

            if i % 100 == 0:
                print("\nloc:", has_cycle.any(dim=1).to(torch.int).argmax())
                print("cycles:", has_cycle.any(dim=1).sum())

        return s.heads


    def get_root_cycle(s, heads):
        root_cycle = heads.unsqueeze(1)
        next_parent = s.parent.gather(dim=1, index=heads.unsqueeze(1))
        has_parent = (next_parent != -1).squeeze(1)
        while has_parent.any():
            root_cycle[has_parent] = next_parent[has_parent]
            next_parent[has_parent] = s.parent[has_parent].gather(
                dim=1, index=next_parent[has_parent])
            has_parent = (next_parent != -1).squeeze(1)
        return root_cycle.squeeze(1)


    def contract_cycle(s, has_cycle):
        active_cycle = has_cycle.clone()
        cycle_ptr = torch.full((s.num_sentences,), -1)
        cycle_ptr[active_cycle] = s.new_super_node(active_cycle)
        super_ptrs = s.super_ptrs.clone()
        super_ptrs[active_cycle.logical_not()] = -1

        while active_cycle.any():

            # Current (super)node has contracting cycle as new parent
            s.parent[active_cycle, super_ptrs[active_cycle]] = (
                cycle_ptr[active_cycle])
            # Current (super)node is child to the contracting cycle
            s.children[
                active_cycle,
                cycle_ptr[active_cycle],
                super_ptrs[active_cycle]] = (True)
            # Contracting cycle inherits descendents of current (super)node 
            descendents = s.descendents[active_cycle, super_ptrs[active_cycle]]
            s.descendents[active_cycle, cycle_ptr[active_cycle]] = (
                s.descendents[active_cycle, cycle_ptr[active_cycle]].logical_or(
                    descendents))

            # Current (super)node (i.e. all its descendents) have out-energy
            # deducted by the current (super)node's in-cycle energy.
            ptrs = s.heads[active_cycle, super_ptrs[active_cycle], 0]
            heads = s.heads[active_cycle, super_ptrs[active_cycle], 1]
            in_cycle_energy = s.energy[active_cycle, ptrs, heads]
            in_cycle_energy = in_cycle_energy.unsqueeze(1).unsqueeze(2).expand(
                ptrs.shape[0], s.num_tokens, s.num_tokens)
            in_cycle_energy = torch.where(
                descendents.unsqueeze(2), in_cycle_energy, torch.tensor(0.0))
            s.energy[active_cycle,:,:] -= in_cycle_energy

            # Move the pointer along
            super_ptrs[active_cycle] = s.cycle_heads[active_cycle,:].gather(
                dim=1, index=super_ptrs[active_cycle].unsqueeze(1)).squeeze(1)

            # Once super_ptrs has no parent, the cycle is resolved
            old_has_cycle = active_cycle.clone()
            active_cycle[old_has_cycle] = (
                s.parent[old_has_cycle, super_ptrs[old_has_cycle]] == -1)

        # Nodes within the contracted-cycle super-node cannot select eachother.
        cycle_descendents = torch.zeros(
            (s.num_sentences, s.num_tokens), dtype=torch.bool)
        cycle_descendents[has_cycle] = (
            s.descendents[has_cycle, cycle_ptr[has_cycle]])
        within_cycle = cycle_descendents.unsqueeze(1).logical_and(
            cycle_descendents.unsqueeze(2))
        s.energy[within_cycle] = -torch.inf

        return cycle_ptr[has_cycle]


    def new_super_node(s, mask):
        offset = (s.heads[mask, s.num_tokens:, 0] != -1).sum(dim=1)
        available_super_ptr = s.num_tokens + offset
        return available_super_ptr


    def pick_head(s, super_ptrs):

        # Get all simple source nodes (super_ptrs may point to super-nodes)
        source_nodes = s.descendents[s.per_sentence, super_ptrs]
        # We'll create the energy for picking by this (super)node
        pick_energy = s.energy.clone()
        # Other nodes beyond this (super)node aren't picking, so no energy there.
        pick_energy[source_nodes.logical_not(),:] = -torch.inf

        # We can't pick <ROOT> unless it is the only node remaining.
        only_root_remaining = (source_nodes.sum(dim=1) == s.num_real_tokens)
        pick_energy[only_root_remaining.logical_not(),:,0]= -torch.inf

        # If super_ptr is <ROOT>, then arbitrarilly choose token 1, which will
        # trigger a final contraction
        pick_energy[super_ptrs==0,0,1] = 0

        # We're sampling accross potentially multiple simple-nodes that make
        # up this super-node.  Flatten the energy array for each sentence.
        pdb.set_trace()
        pick_energy = pick_energy.flatten(start_dim=1, end_dim=2)
        picked_raw = torch.distributions.Categorical(pick_energy.exp()).sample()
        heads = torch.empty((s.num_sentences, 2), dtype=torch.long)
        # Recover specific source and target from the flattened position.
        heads[:,0] = (picked_raw / s.num_tokens).floor() # simple-node source
        heads[:,1] = (picked_raw % s.num_tokens) # simple-node target
        return heads


    def init_epsilon(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        num_sentences, num_tokens = mask.shape
        unpadded_num_tokens = num_tokens - mask.sum(dim=1, keepdim=True)
        epsilon = (max_out_weight / unpadded_num_tokens * ROOT_MASS) / 10
        return epsilon



class CycleEscapeRandomTree():

    energy_epsilon_adjust = -0.09
    def sample_parses(
        self, tokens_batch, embedding, mask=False):
        """
        The approach here is to randomly select all heads, and then resolve any
        issues by reselecting heads for all tokens involved in said issues.
        There are two possible issues: cycles, and having more than one node that
        selects <ROOT>.
        """

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        energy[:,:,0] = torch.log(self.init_epsilon(torch.exp(energy), mask))

        contenders = torch.ones_like(tokens_batch, dtype=torch.bool)
        has_cycle = torch.tensor(True) # temporary to pass while condition.
        heads = torch.zeros(tokens_batch.shape, dtype=torch.long)
        i = 0
        while has_cycle.any() or contenders.any():
            i += 1

            head_selector = torch.distributions.Categorical(torch.exp(energy))
            heads_sample = head_selector.sample()
            heads = torch.where(contenders, heads_sample, heads)

            # Parses that contain multiple roots should be restarted.
            has_multiple_roots = self.has_multiple_roots(heads, mask)
            heads[has_multiple_roots,:] = 0
            contenders[has_multiple_roots,:] = True
            energy[has_multiple_roots,:,0] += self.energy_epsilon_adjust
            # Drop dim1, it's unnecessasry.
            contenders = has_multiple_roots.unsqueeze(1).expand(
                heads.shape).clone()

            # Find a cycle (up to one per sentence).
            has_cycle, cycle_lengths = self.mark_one_cycle(heads)
            # Resolve a cycle.
            cycle_break_ptr, cycle_break_head_ptr = self.get_cycle_break(
                energy, heads, has_cycle)
            per_sentence = torch.arange(heads.shape[0])
            heads[per_sentence, cycle_break_ptr] = cycle_break_head_ptr

            if i % 100 == 0:
                avg_cycle_length = (
                    cycle_lengths.sum() / has_cycle.any(dim=1).sum())
                print("\ncycles:", has_cycle.any(dim=1).sum())
                print("avg cycle length:", avg_cycle_length)
                print("multiple_roots:", has_multiple_roots.sum(), "\n")

        return heads


    def get_cycle_break(self, energy, heads, has_cycle):
        # Subtract each token's in-cycle energy from each of that token's 
        # other energies.
        in_cycle_energy = energy.gather(dim=2, index=heads.unsqueeze(2))
        cycle_breaking_energy = energy - in_cycle_energy
        out_cycle = has_cycle.logical_not().unsqueeze(2).expand(energy.shape)
        cycle_breaking_energy[out_cycle] = -torch.inf
        in_cycle = has_cycle.unsqueeze(1).logical_and(has_cycle.unsqueeze(2))
        cycle_breaking_energy[in_cycle] = -torch.inf

        non_cyclic = has_cycle.any(dim=1).logical_not()
        cycle_breaking_energy = cycle_breaking_energy.flatten(
            start_dim=1, end_dim=2)
        cycle_breaking_energy[non_cyclic,:] = -torch.inf
        cycle_breaking_energy[non_cyclic,0] = 0
        cycle_breaking_probs = torch.exp(cycle_breaking_energy)
        cycle_break = torch.distributions.Categorical(
            cycle_breaking_probs).sample()
        num_tokens = heads.shape[1]
        cycle_break_ptr = (cycle_break / num_tokens).to(torch.int64)
        cycle_break_head_ptr = cycle_break % num_tokens
        return cycle_break_ptr, cycle_break_head_ptr


    def init_epsilon(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        num_sentences, num_tokens = mask.shape
        unpadded_num_tokens = num_tokens - mask.sum(dim=1, keepdim=True)
        epsilon = max_out_weight / unpadded_num_tokens * ROOT_MASS
        return epsilon


    def has_multiple_roots(self, heads, mask):
        # Get the non-PADDING tokens that have ROOT as head.
        rooted = (heads == 0).logical_and(mask.logical_not())
        # Eliminate ROOT itself
        rooted[:,0] = False
        # Determine which sentences have more than one root.
        has_multiple_roots = rooted.sum(dim=1) > 1
        return has_multiple_roots


    def mark_one_cycle(self, heads):
        """
        A node is on a loop if it can reach itself by taking enough hops
        along directed edges.  Recursively index heads with itself, to 
        visit all ancestors of each node, and look for a node itself among
        it's ancestors.  If seen, this node is in a cycle.

        We only want to mark the tokens of one cycle.  If there is more than 
        one cycle, each cycle will have a unique minimum ancestor seen
        during the walk through ancestors.  Use this fact to single out the
        tokens of at most one cycle, and return a mask that is hot at those
        tokens.
        """
        max_steps = heads.shape[1]
        has_cycle = torch.full(heads.shape, False)
        self_head = torch.arange(
            heads.shape[1]).unsqueeze(0).expand(heads.shape)
        ancestors = heads.clone()
        cycle_lengths = torch.zeros(heads.shape[0], dtype=torch.int)

        # At each token position, watch for the minimum value of ancestor.
        # The positions of each cycle will have a common minimum ancestor that is
        # unique to that cycle.  Using this, we can select the tokens of exactly 
        # *one* (or zero) cycle to be returned for resolution, as required.
        min_ancestor = ancestors
        for i in range(max_steps):
            min_ancestor = torch.minimum(min_ancestor, ancestors)
            found_cycle = ancestors == self_head
            found_cycle[:,0] = False
            found_new_cycle = found_cycle.any(dim=1).logical_and(
                has_cycle.any(dim=1).logical_not())
            cycle_lengths[found_new_cycle] = i
            has_cycle = has_cycle.logical_or(found_cycle)
            ancestors = heads.gather(dim=1, index=ancestors)

        # ROOT is trivially cycled due to self-link, but we don't want it.
        has_cycle[:,0] = 0

        # Right now the all positions involved in cycles are marked.  We want
        # only one per sentence.  Use the *minimum* minimum ancestor found
        # to be part of a cycle in each sentence to select at most one cycle
        # per sentence.  

        # To find this minimum, first, give all non-cycling positions a max
        # value.  That makes the minimum value we seek equal to the straight
        # minimum across each whole sentence.
        min_ancestor_selected = min_ancestor.clone()
        min_ancestor_selected[has_cycle.logical_not()] = ancestors.shape[1]
        min_ancestor_selected = min_ancestor_selected.min(
            dim=1, keepdim=True).values
        # The nodes in the selected cycle are those that are both in a cycle
        # and saw the selected minimum ancestor value.
        has_cycle_selected = has_cycle.logical_and(
            min_ancestor == min_ancestor_selected)

        return has_cycle_selected, cycle_lengths


class ContentionRandomTree():

    def sample_parses(
        self, tokens_batch, embedding, mask=False):
        """
        The approach here is to randomly select all heads, and then resolve any
        issues by reselecting heads for all tokens involved in said issues.
        There are two possible issues: cycles, and having more than one node that
        selects <ROOT>.
        """

        num_sentences, num_tokens = tokens_batch.shape
        energy = embedding.sentence_link_energy(tokens_batch, mask)
        probs = torch.exp(energy)
        probs[:,:,0] = self.init_epsilon(probs, mask)
        #probs[:,:,0] = 1
        contenders = torch.ones_like(tokens_batch, dtype=torch.bool)
        heads = torch.zeros(tokens_batch.shape, dtype=torch.long)
        i = 0
        while contenders.any():
            i += 1

            head_selector = torch.distributions.Categorical(probs)
            heads_sample = head_selector.sample()
            heads = torch.where(contenders, heads_sample, heads)

            # Parses that contain multiple roots should be restarted.
            has_multiple_roots = self.has_multiple_roots(heads, mask)
            heads[has_multiple_roots,:] = 0
            contenders[has_multiple_roots,:] = True
            probs[has_multiple_roots,:,0] = probs[has_multiple_roots,:,0] / 1.1

            # Find cycles.  Those tokens need to re-select their head.
            has_cycle, cycle_lengths = self.has_cycle(heads)
            contenders = has_cycle.logical_or(has_multiple_roots.unsqueeze(1))

            if i % 100 == 0:
                avg_cycle_length = cycle_lengths.sum() / has_cycle.any(dim=1).sum()
                print("\ncycles:", contenders.any(dim=1).sum())
                print("avg cycle length:", avg_cycle_length)
                print("multiple_roots:", has_multiple_roots.sum(), "\n")

        return heads

    def init_epsilon(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        num_sentences, num_tokens = mask.shape
        unpadded_num_tokens = num_tokens - mask.sum(dim=1, keepdim=True)
        epsilon = max_out_weight / unpadded_num_tokens * ROOT_MASS
        return epsilon


    def has_multiple_roots(self, heads, mask):
        # Get the non-PADDING tokens that have ROOT as head.
        rooted = (heads == 0).logical_and(mask.logical_not())
        # Eliminate ROOT itself
        rooted[:,0] = False
        # Determine which sentences have more than one root.
        has_multiple_roots = rooted.sum(dim=1) > 1
        return has_multiple_roots


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
        cycle_lengths = torch.zeros(heads.shape[0], dtype=torch.int)
        for i in range(max_steps):
            found_cycle = ancestors == self_head
            found_cycle[:,0] = False
            found_new_cycle = found_cycle.any(dim=1).logical_and(
                has_cycle.any(dim=1).logical_not())
            cycle_lengths[found_new_cycle] = i
            has_cycle = has_cycle.logical_or(found_cycle)
            ancestors = heads.gather(dim=1, index=ancestors)

        # ROOT is trivially cycled due to self-link, but we don't care about it.
        has_cycle[:,0] = 0
        return has_cycle, cycle_lengths
        




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


class ConvRandomTree:

    kernel = [torch.tensor(v) for v in [0,5,4,3,2,1]]

    def sample_parses(self, tokens, embedding, mask):
        probs, epsilon = self.get_probs(tokens, embedding, mask)
        return m.random_tree(tokens, probs, epsilon, mask)


    def get_probs(self, tokens, embedding, mask):
        num_sentences, num_tokens = tokens.shape
        probs_token, epsilon = self.get_probs_tokens(tokens, embedding, mask)
        probs_position = self.get_probs_positions(num_tokens)
        probs = probs_token * probs_position
        return probs, epsilon


    def get_probs_tokens(self, tokens, embedding, mask):
        probs = torch.exp(embedding.sentence_link_energy(tokens, mask))
        probs[:,:,0] = 1
        epsilon = self.init_epsilon(probs, mask)
        probs[:,:,0] = epsilon
        return probs, epsilon


    def init_epsilon(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        num_sentences, num_tokens = mask.shape
        unpadded_num_tokens = num_tokens - mask.sum(dim=1, keepdim=True)
        epsilon = max_out_weight / unpadded_num_tokens * ROOT_MASS
        return epsilon


    def get_probs_positions(self, num_tokens):
        head_probs = torch.zeros((num_tokens, num_tokens))
        for i,v in enumerate(self.kernel):
            # Handle sentences shorter than kernel properly.
            if i > num_tokens:
                break
            diag = torch.full((num_tokens-i,), v).diagflat(i)
            head_probs += diag
        head_probs += head_probs.T.clone()
        head_probs[0] = 0
        head_probs[:,0] = 1
        return head_probs



class SimpleRandomTree():

    def sample_parses(self, tokens, embedding, mask):
        # Adjust the probabilities to make p.
        probs = torch.exp(embedding.sentence_link_energy(tokens, mask))
        probs[:,:,0] = 1
        #probs = self.apply_self_energy(probs, mask)
        epsilon = self.init_epsilon(probs, mask)
        probs[:,:,0] = epsilon
        return m.random_tree(tokens, probs, epsilon, mask)


    def init_epsilon(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        num_sentences, num_tokens = mask.shape
        unpadded_num_tokens = num_tokens - mask.sum(dim=1, keepdim=True)
        epsilon = max_out_weight / unpadded_num_tokens * ROOT_MASS
        return epsilon

            
    def apply_self_energy(self, probs, mask):
        out_weights = probs.sum(dim=2)
        max_out_weight = out_weights.max(dim=1, keepdim=True).values
        needed_self_weight = max_out_weight - out_weights
        probs = probs + torch.diag_embed(needed_self_weight, dim1=1, dim2=2)
        probs[:,0,0] = 1
        probs[mask.unsqueeze(1).expand(probs.shape)] = 0
        return probs



