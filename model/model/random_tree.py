import model as m
import torch


def init_state(num_tokens, num_sentences, mask):
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



def random_tree(tokens, probs, epsilon, mask):

    # Various pointers and booleans track the state of tokens and sentences.
    m.timer.start()
    num_sentences, num_tokens = tokens.shape
    state = init_state(num_tokens, num_sentences, mask)
    tree, intree, leaf_ptr, head_ptr, has_root = state
    done = intree.all(dim=1)

    per_sentence = torch.arange(num_sentences)
    i = 0
    m.timer.log("init")
    while not done.all():

        # In each sentence, the token head_ptr choses a next head.
        next_head_ptr = torch.distributions.Categorical(
            probs[per_sentence, head_ptr]).sample()
        tree[per_sentence, head_ptr] = next_head_ptr
        head_ptr = next_head_ptr
        m.timer.log("sample-head")

        # If multiple tokens in a sentence choose root, restart parsing of
        # that sentence (reinitialize the state data associated to it).
        chose_root = (next_head_ptr == 0)
        needs_reset = chose_root.logical_and(
            has_root).logical_and(done.logical_not())
        has_root[chose_root] = True
        if needs_reset.any():
            epsilon[needs_reset] = epsilon[needs_reset] / 1.5
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
        m.timer.log("handle-root")

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
        m.timer.log("handle-anchor")

    assert not (tree == -1).any(), "parse invalid."
    return tree

