import os
import sys
import pdb
import time
import random
from collections import defaultdict

import torch

import model as m


def train_flat():

    lr = 1e-3
    num_epochs = 100
    batch_size = 100
    embed_dim = 300
    vocab_limit = 250000

    # TODO: Can padding be zero like elsewhere?
    # Get the data, model, and optimizer.
    data = m.PaddedDatasetParallel(
        #m.const.DEFAULT_GOLD_DATA_DIR,
        m.const.WIKI_DATA_PATH,
        padding=-1,
        min_length=3,
        max_length=140,
        has_heads=False,
        has_relations=False,
        approx_chunk_size=100*m.const.KB,
        vocab_limit=vocab_limit
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        shuffle=True,
        #num_workers=2,
        pin_memory=True
    )

    model = m.FlatModel(len(data.dictionary), embed_dim, data.Nx)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = Scheduler()

    # Train the model on the data with the optimizer.
    train_model(model, dataloader, optimizer, scheduler, num_epochs)

    pdb.set_trace()
    return model


def train_edge(load_existing_path=None):

    lr = 1e-3
    num_epochs = 100
    batch_size = 100
    embed_dim = 300
    vocab_limit = 250000

    # TODO: Can padding be zero like elsewhere?
    # Get the data, model, and optimizer.
    data = m.PaddedDatasetParallel(
        #m.const.DEFAULT_GOLD_DATA_DIR,
        m.const.WIKI_DATA_PATH,
        padding=0,
        min_length=3,
        max_length=140,
        has_heads=False,
        has_relations=False,
        approx_chunk_size=1*m.const.KB,
        vocab_limit=vocab_limit
    )
    batch = data[0]

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        shuffle=True,
        #num_workers=2,
        pin_memory=True
    )

    if load_existing_path is not None:
        model = m.EdgeModel.load(load_existing_path, data.Nx/data.Nx.sum())
    else:
        model = m.EdgeModel(
            len(data.dictionary), embed_dim, data.Nx/data.Nx.sum())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = schedule_temperature(model, 20, 1, 3000)
    train_model(model, dataloader, optimizer, scheduler, num_epochs)

    pdb.set_trace()
    return model


def schedule_temperature(model, init, final, after, scheduler=None):
    if scheduler is None:
        scheduler = Scheduler()
    def set_temp(batch_num): 
        print(f"setting temperature to {init}")
        model.start_temp = init
    def update_temp(batch_num):
        print(f"setting temperature to {final}")
        model.start_temp = final
    scheduler.after(0, set_temp)
    scheduler.after(after, update_temp)
    return scheduler


def view_edge():

    data = m.PaddedDataset(m.const.DEFAULT_GOLD_DATA_DIR, min_length=8)
    tokens_batch, head_ptrs_batch, relations_batch, mask_batch = data[0]

    params_subpath = "../test-data/model-params"
    params_dir = os.path.join(m.const.SCRIPT_DIR, params_subpath)
    model_up_path = os.path.join(params_dir, "edge-params.pt")
    model_up = m.EdgeModel.load(model_up_path, data.Nx/data.Nx.sum())
    model_head_ptrs_batch_up = model_up.sample_parses(
        tokens_batch, mask_batch, start_temp=1, temp_step=0.001)

    model_down_path = os.path.join(params_dir, "edge-params200.pt")
    model_down = m.EdgeModel.load(model_down_path, data.Nx/data.Nx.sum())
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


class Scheduler:

    def __init__(self):
        self._after = defaultdict(list)
        self._every = defaultdict(list)
        self.batch_num = 0

    def tick(self):
        for func in self._after[self.batch_num]:
            func(self.batch_num)
        for frequency, func in self._every:
            if self.batch_num % frequency == 0:
                func(self.batch_num)
        self.batch_num += 1

    def after(self, batch_num, update_function):
        self._after[batch_num].append(update_function)

    def every(self, num_batches, update_function):
        self._every.append((batch_num, update_function))


def train_model(model, dataloader, optimizer, scheduler, num_epochs):

    for epoch in range(num_epochs):
        print("epoch:", epoch)
        epoch_loss = torch.tensor(0.)

        permutation = list(range(len(dataloader.dataset)))
        random.shuffle(permutation)
        start = time.time()
        num_sentences_processed = 0
        num_batches = len(dataloader)
        for batch_num, batch in enumerate(dataloader):

            scheduler.tick()
            tokens_batch, _, _, mask, batch_idx = batch
            batch_idx = batch_idx.item()

            for tokens_chunk, mask_chunk in zip(tokens_batch, mask):

                tokens_chunk = tokens_chunk.squeeze(0)
                mask_chunk = mask_chunk.squeeze(0)

                sys.stdout.write("\b" * 60 + " " * 60 + "\b" * 60)
                sys.stdout.write(f"{epoch}:{batch_num}/{num_batches}")
                sys.stdout.write(":"+str(tokens_chunk.shape[1]))

                elapsed = time.time() - start
                sent_per_sec = num_sentences_processed / elapsed
                sys.stdout.write(f"\t{sent_per_sec:.2f} sentences per second")
                num_sentences_processed += tokens_chunk.shape[0] 
                sys.stdout.flush()

                loss = model.get_loss(tokens_chunk, mask_chunk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    num_edges = (
                        tokens_chunk.shape[0] * tokens_chunk.shape[1]
                        - mask_chunk.sum()
                    )
                    epoch_loss += loss/(num_edges)

            if batch_num % 4 == 0:
                with torch.no_grad():
                    print(loss)
                    print_model(model, dataloader.dataset.dictionary)

        print(epoch_loss)


def print_model(model, dictionary):
    words = [
        "chocolate", "Chicago", "challenge", "night",
        "cat", "burn", "fortified", "strange"
    ]
    for word in words:
        word_id = dictionary.get_id(word)
        heads = dictionary.get_tokens(model.embedding.head_match(word_id, 5))
        similars = dictionary.get_tokens(model.embedding.similar(word_id, 5))
        print(word, "|", " ".join(heads), "|", " ".join(similars))

    display_parse(
        "<ROOT> A Washington lobbyist took the opportunity .".split(),
        dictionary, model)
    display_parse(
        "<ROOT> Before I go , could we talk ?".split(),
        dictionary, model)
    display_parse(
        "<ROOT> There were three reasons she left Tokyo .".split(),
        dictionary, model)


def display_parse(sentence, dictionary, model):
    ids = torch.tensor([dictionary.get_ids(sentence)])
    mask = torch.tensor([[False]*len(sentence)])
    heads = model.sample_parses(ids, mask)
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


