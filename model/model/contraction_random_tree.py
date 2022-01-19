import pdb
import torch

ROOT_MASS = 0.5


class ScalarContractionRandomTree():

    def sample(self, energy, mask):


class ContractionRandomTree():

    energy_epsilon_adjust = -0.09

    def sample(s, energy, mask):
        heads = s.contract(energy, mask)
        rooted_trees = s.extract()
        selected_trees = s.select_root(rooted_trees, mask)
        return selected_trees


    def init(s, energy, mask):
        s.num_sentences, s.num_tokens, _ = energy.shape
        s.energy = energy.clone() # We'll mutate this during cycle resolution.
        s.energy_ = energy  # keep a reference to the pristine energy.
        s.heads = torch.full((s.num_sentences, 2*s.num_tokens, 2), -1)
        s.num_real_tokens = s.num_tokens - mask.sum(dim=1)
        s.children = torch.zeros(
            (s.num_sentences, 2*s.num_tokens, 2*s.num_tokens), dtype=torch.bool)
        s.cycle_heads = torch.full((s.num_sentences, 2*s.num_tokens), -1)
        s.parent = torch.full((s.num_sentences, 2*s.num_tokens), -1)
        s.descendents = torch.zeros(
            (s.num_sentences, 2*s.num_tokens, s.num_tokens), dtype=torch.bool)
        s.descendents[:, :s.num_tokens, :s.num_tokens] = torch.eye(
            s.num_tokens, dtype=torch.bool).unsqueeze(0).expand(
                (s.num_sentences, s.num_tokens, s.num_tokens))


    def contract(s, energy, mask):
        s.init(energy, mask)
        s.super_ptrs = torch.distributions.Categorical(
            mask.logical_not()).sample()
        #s.super_ptrs = torch.ones(s.num_sentences, dtype=torch.long)
        active = torch.ones((s.num_sentences,), dtype=torch.bool)
        while active.any():

            # Each (super)node given by super_ptrs selects a head.
            s.heads[active,s.super_ptrs[active], :] = s.pick_head(active)
            head = s.heads[active, s.super_ptrs[active], 1]

            # head is a simple node.  Get widest super_node that contains it.
            super_head = s.get_root_cycle(head, active)
            s.cycle_heads[active, s.super_ptrs[active]] = super_head

            # Check for cycles.  has_cycle and no_cycle are not exact
            # compliments.
            has_cycle, no_cycle = s.has_cycle(head, active)

            # If we have not made a cycle, simply move ahead to the new node.
            s.super_ptrs[no_cycle] = head[no_cycle[active]]

            # If cycle, contract cycle and set ptr to this super node.
            if has_cycle.any():
                s.super_ptrs[has_cycle] = s.contract_cycle(has_cycle)
                active[has_cycle] = s.check_active(has_cycle)

        return s.heads


    def check_active(s, has_cycle):
        # Sentence is done when a super node subsumes all simple tokens.
        cycle_descendents = (
            s.descendents[has_cycle, s.super_ptrs[has_cycle]])
        num_cycle_descendents = cycle_descendents.sum(dim=1)

        # This is ultimately what makes us eventually break.  A sentence
        # TODO: this seems to assume that the graph is strongly
        # connected, no transients.
        in_cycle_active = num_cycle_descendents < s.num_real_tokens[has_cycle]
        return in_cycle_active



    def has_cycle(s, head, active):
        # If we select an already-seen node as head, we have made a cycle.
        # The head selected has been seen if it has it's own head.
        head_of_head = s.heads[active,:,1].gather(
            dim=1, index=head.unsqueeze(1)).squeeze(1)
        no_cycle = active.clone()
        no_cycle[active] = (head_of_head == -1)
        has_cycle = active.clone()
        has_cycle[active] = no_cycle[active].logical_not()
        return has_cycle, no_cycle


    def get_root_cycle(s, heads, active):
        root_cycle = heads.unsqueeze(1).clone()
        next_parent = s.parent[active].gather(dim=1, index=heads.unsqueeze(1))
        has_parent = (next_parent != -1).squeeze(1)
        while has_parent.any():
            root_cycle[has_parent] = next_parent[has_parent]
            next_parent[has_parent] = s.parent[active][has_parent].gather(
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


    def new_super_node(s, active):
        offset = (s.heads[active, s.num_tokens:, 0] != -1).sum(dim=1)
        available_super_ptr = s.num_tokens + offset
        return available_super_ptr


    def pick_head(s, active):

        # Get all simple source nodes (super_ptrs may point to super-nodes)
        source_nodes = s.descendents[active, s.super_ptrs[active]]
        # We'll create the energy for picking by this (super)node
        pick_energy = s.energy[active].clone()
        # Other nodes beyond this (super)node aren't picking, so no energy there.
        pick_energy[source_nodes.logical_not(),:] = -torch.inf

        # We can't pick <ROOT> unless it is the only node remaining.
        #only_root_remaining = (
        #    (source_nodes.sum(dim=1)+1) == s.num_real_tokens[active])
        #pick_energy[only_root_remaining.logical_not(),:,0]= -torch.inf

        # If super_ptr is <ROOT>, then arbitrarilly choose token 1, which will
        # trigger a final contraction
        #pick_energy[s.super_ptrs[active]==0,0,1] = 0

        # We're sampling accross potentially multiple simple-nodes that make
        # up this super-node.  Flatten the energy array for each sentence.
        pick_energy = pick_energy.flatten(start_dim=1, end_dim=2)
        picked_raw = torch.distributions.Categorical(pick_energy.exp()).sample()
        heads = torch.empty((active.sum(), 2), dtype=torch.long)
        # Recover specific source and target from the flattened position.
        heads[:,0] = (picked_raw / s.num_tokens).floor() # simple-node source
        heads[:,1] = (picked_raw % s.num_tokens) # simple-node target
        return heads


    def extract(s):
        s.energy = s.energy_    # recover the pristine energy.

        # We'll build a different tree for each possible sentence root.
        # Thus trees[sent, token] will contain a tree for sent rooted at token.
        trees = torch.full(s.energy.shape, -1)

        # Start with root 0, we'll iterate.  Sentence stays active if it
        # has a real token (not mask) at position `root`.
        root = 0
        active = (root < s.num_real_tokens)
        while active.any():

            # Get the trees rooted at `root`.
            roots = torch.full((s.num_sentences,), -1)
            roots[active] = root
            trees[active, root] = s.extract_for_root(roots)

            # Proceed to the next root (if any sentences are still active)
            root += 1
            active = root < s.num_real_tokens

        return trees


    def select_root(s, trees, mask):
        tree_energies = torch.full(trees.shape, -torch.inf)
        root = 0
        active = (root < s.num_real_tokens)
        while active.any():
            heads = trees[active, root]

            # ROOT and PAD appear as -1s.  Set them harmlessly to 0 so that
            # indexing does not error.  We'll kill this spurious energy below.
            head_mask = (heads == -1)
            heads_masked = heads.clone()
            heads_masked[head_mask] = 0
            energy = s.energy[active].gather(
                dim=2, index=heads_masked.unsqueeze(2))

            # Now kill energy for ROOT and PAD.
            energy[head_mask] = 0
            # Store the energy for sentences parsed on this root.
            tree_energies[active, root] = energy.squeeze(2)

            # Advance root, and check which sentences are still active.
            root += 1
            active = (root < s.num_real_tokens)

        # Now sample
        probs = tree_energies.sum(dim=2).exp()
        pdb.set_trace()
        selected_tree_ptr = torch.distributions.Categorical(probs).sample()
        per_sentence = torch.arange(s.num_sentences)
        selected_trees = trees[per_sentence,selected_tree_ptr]

        return selected_trees


    def extract_for_root(s, roots):

        # We will be mutating `s.parent`, keep a pristine copy to restore later.
        s.parent_ = s.parent.clone()
        heads = s.heads.clone()

        # As we process sentences they become inactive.  Remember which sentences
        # were active to begin with.
        was_active = (roots != -1)

        # Begin by dismantling around <ROOT>.  This will enqueue other super 
        # nodes in `to_dismantle`.
        to_dismantle = torch.full((s.num_sentences, s.num_tokens), -1)
        s.dismantle(to_dismantle, roots)
        active = (to_dismantle[:,0] != -1)

        # Dismantle super_nodes queued by side effect of s.dismantle()
        while active.any():

            # Pop an element per sentence to dismantle.
            ptr = to_dismantle[:,0]
            to_dismantle = to_dismantle[:,1:]

            # The outgoing arc from ptr is kept in the final tree.
            src_head = torch.full((s.num_sentences, 2), -1)
            head_ptr = ptr[active].unsqueeze(1).unsqueeze(2).expand(-1,-1,2)
            src_head = heads[active].gather(index=head_ptr, dim=1).squeeze(1)
            src = torch.full((s.num_sentences,) , -1)
            src[active] = src_head[:,0]
            heads[active, src[active]] = src_head  # inserts into tree.

            # Dismantle src (this may add super nodes to to_dismantle
            s.dismantle(to_dismantle, src)

            # Where is there still work to do?
            active = (to_dismantle[:,0] != -1)

        # We do not keep the head chosen by <ROOT>, because it is the root.
        result = heads[:,:s.num_tokens,1]

        # Designate the root token with a -1 in the heads array.
        result[was_active, roots] = -1

        # Restore pristine parents.
        s.parent = s.parent_

        return result


    def dismantle(s, to_dismantle, ptr):
        ptr = ptr.clone()

        # Pointer is active if it is not -1 and if it's parent is not -1
        ptr_active = (ptr != -1)
        active_parent = s.parent[ptr_active, ptr[ptr_active]]
        ptr_active[ptr_active.clone()] = (active_parent !=  -1)
        active_parent = s.parent[ptr_active, ptr[ptr_active]]
        while ptr_active.any():

            # Find siblings of 
            # Get next sibling.
            sibling = torch.zeros((s.num_sentences,2*s.num_tokens), dtype=bool)
            sibling[ptr_active] = s.children[ptr_active, active_parent]

            # Unset each sibling's parent
            s.parent[sibling] = -1

            # Get the siblings that have children.
            has_children = s.children.any(dim=2)
            sib_has_children = sibling.logical_and(has_children)
            # But don't include ptr itself.
            sib_has_children[ptr_active, ptr[ptr_active]] = False

            # Add siblings with children to the dismantling "queue".
            start_index = s.next_available_slot(to_dismantle).unsqueeze(1)
            end_index = start_index + sib_has_children.sum(dim=1).unsqueeze(1)
            position = torch.arange(2*s.num_tokens)
            position = position.unsqueeze(0).expand((
                s.num_sentences, 2*s.num_tokens))
            allocation = (
                position[:,:to_dismantle.shape[1]] >= start_index
            ).logical_and(
                position[:,:to_dismantle.shape[1]] < end_index
            )
            to_dismantle[allocation] = position[sib_has_children]

            # Move pointer to parent.  Apply same activity criterion as before.
            ptr[ptr_active] = active_parent
            ptr_active = (ptr != -1)
            active_parent = s.parent[ptr_active, ptr[ptr_active]]
            ptr_active[ptr_active.clone()] = (active_parent != -1)
            active_parent = s.parent[ptr_active, ptr[ptr_active]]


    def next_available_slot(s, to_dismantle):
        return to_dismantle.shape[1] - (to_dismantle==-1).sum(dim=1)

