import os
import sys
import pdb
import time
from collections import defaultdict

import torch

import model as m


def train_flat():

    lr = 1e-2
    num_epochs = 100
    batch_size = 100
    embed_dim = 300

    # TODO: Can padding be zero like elsewhere?
    # Get the data, model, and optimizer.
    data = m.PaddedDataset(
        m.const.DEFAULT_GOLD_DATA_DIR, padding=-1,
        batch_size=batch_size, min_length=3
    )
    model = m.FlatModel(len(data.dictionary), embed_dim, data.Px)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Train the model on the data with the optimizer.
    train_model(model, data, optimizer, num_epochs)

    pdb.set_trace()
    return model


def train_edge(load_existing_path=None):

    lr = 1e-2
    num_epochs = 100
    batch_size = 100
    embed_dim = 200
    vocab_limit = 50000

    # TODO: Can padding be zero like elsewhere?
    # Get the data, model, and optimizer.
    data = m.PaddedDatasetParallel(
        m.const.WIKI_DATA_PATH,
        padding=0,
        min_length=0,
        has_heads=False,
        has_relations=False,
        approx_chunk_size=10*m.const.KB,
        vocab_limit=vocab_limit
    )
    pdb.set_trace()

    if load_existing_path is not None:
        model = m.EdgeModel.load(load_existing_path, data.Px)
    else:
        model = m.EdgeModel(len(data.dictionary), embed_dim, data.Px)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_model(model, data, optimizer, num_epochs)

    pdb.set_trace()
    return model


def view_edge():

    data = m.PaddedDataset(m.const.DEFAULT_GOLD_DATA_DIR, min_length=8)
    tokens_batch, head_ptrs_batch, relations_batch, mask_batch = data[0]

    params_subpath = "../test-data/model-params"
    params_dir = os.path.join(m.const.SCRIPT_DIR, params_subpath)
    model_up_path = os.path.join(params_dir, "edge-params.pt")
    model_up = m.EdgeModel.load(model_up_path, data.Px)
    model_head_ptrs_batch_up = model_up.sample_parses(
        tokens_batch, mask_batch, start_temp=1, temp_step=0.001)

    model_down_path = os.path.join(params_dir, "edge-params200.pt")
    model_down = m.EdgeModel.load(model_down_path, data.Px)
    model_head_ptrs_batch_down = model_down.sample_parses(
        tokens_batch, mask_batch, start_temp=1, temp_step=0.001)

    tokens_batch = remove_padding(tokens_batch, mask_batch)
    model_head_ptrs_batch_up = remove_padding(
        model_head_ptrs_batch_up, mask_batch)
    model_head_ptrs_batch_down = remove_padding(
        model_head_ptrs_batch_down, mask_batch)

    head_ptrs_batch = remove_padding(head_ptrs_batch, mask_batch)

    m.viz.print_trees(
        tokens_batch, 
        model_head_ptrs_batch_up,
        model_head_ptrs_batch_down,
        #head_ptrs_batch,
        data.dictionary,
        out_path=m.const.HTML_DIR
    )


def remove_padding(padded_batch, mask_batch):
    padded_batch = padded_batch.tolist()
    mask_batch = mask_batch.tolist()
    for i in range(len(padded_batch)):
        mask_size = sum(mask_batch[i])
        if mask_size > 0:
            padded_batch[i] = padded_batch[i][:-mask_size]
    return padded_batch


def train_model(model, data, optimizer, num_epochs):

    for epoch in range(num_epochs):
        print("epoch:", epoch)
        epoch_loss = torch.tensor(0.)

        for batch_num, (tokens_batch, _, _, mask) in enumerate(data):
            sys.stdout.write("\b" * 10 + " " * 10 + "\b" * 10)
            sys.stdout.write(str(batch_num))
            sys.stdout.write(":"+str(tokens_batch.shape[1]))
            sys.stdout.flush()

            loss = model.get_loss(tokens_batch, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss/(tokens_batch.shape[0]*tokens_batch.shape[1])

            if batch_num % 10 == 0:
                with torch.no_grad():
                    print_model(model, data.dictionary)

        print(epoch_loss)


def print_model(model, dictionary, start_temp=1, temp_step=0.001):
    words = [
        "apparently", "would", "gave", "said",
        "lawyer", "the", "fast", "Washington"
    ]
    for word in words:
        word_id = dictionary.get_id(word)
        heads = dictionary.get_tokens(model.embedding.head_match(word_id, 5))
        similars = dictionary.get_tokens(model.embedding.similar(word_id, 5))
        print(word, "|", " ".join(heads), "|", " ".join(similars))

    display_parse(
        "<ROOT> I gave the documents to my lawyer .".split(),
        dictionary, model, start_temp, temp_step)
    display_parse(
        "<ROOT> Could you buy some pet food ?".split(),
        dictionary, model, start_temp, temp_step)
    display_parse(
        "<ROOT> Before law school , she studied computer science at MIT".split(),
        dictionary, model, start_temp, temp_step)


def display_parse(sentence, dictionary, model, start_temp, temp_step):
    ids = torch.tensor([dictionary.get_ids(sentence)])
    mask = torch.tensor([[False]*len(sentence)])
    heads = model.sample_parses(ids, mask, start_temp, temp_step)
    print_tree(heads[0].tolist(), sentence)




def get_run_code(batch_size, num_epochs, embedding_dimension, corpus):
    return (
        f"w2v"
        f"-b{batch_size}"
        f"-e{num_epochs}"
        f"-d{embedding_dimension}"
        f"-{corpus}"
    )


def print_tree(headlist, tokens=None, curnode=0, depth=0):
    string = str(curnode) if tokens is None else tokens[curnode]
    print('  ' * depth + string)
    children = [
        i for i, val in enumerate(headlist) 
        if val == curnode
    ]
    for child in children:

        # Node 0 is root and is self-linked.  Skip it as a child.
        if child == 0:
            continue

        print_tree(headlist, tokens, curnode=child, depth=depth+1)


if __name__ == "__main__":
    view_edge()


