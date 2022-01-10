import pdb
import model as m
import torch
from torch.nn.functional import one_hot


class EdgeModel:

    def __init__(self, vocab_size, embed_dim, Px):
        self.embedding = m.EmbeddingLayer(vocab_size, embed_dim)
        self.token_sampler = m.st.ParityTokenSampler(Px)
        self.parse_sampler = m.sp.ConvRandomTree()
        self.start_temp = 1
        self.temp_step = 0.01


    @staticmethod
    def load(path, Px):
        params = torch.load(path)
        vocab_size, embed_dim = params['U'].shape
        model = EdgeModel(vocab_size, embed_dim, Px)
        model.embedding.U = params['U']
        model.embedding.V = params['V']
        model.embedding.Ubias = params['Ubias']
        model.embedding.Vbias = params['Vbias']
        return model


    def sample_parses(self, tokens_batch, mask):
        return self.parse_sampler.sample_parses(
            tokens_batch, self.embedding, mask)#, self.start_temp, self.temp_step)


    def sample_tokens(self, tokens_batch, head_ptrs, mask):
        return self.token_sampler.sample_tokens(
            tokens_batch, head_ptrs, self.embedding, mask)


    # TODO: Should this just be an actual subclass of Model?
    def parameters(self):
        return self.embedding.parameters()


    def get_loss(self, tokens_batch, mask):

        num_sentences, num_tokens = tokens_batch.shape

        with torch.no_grad():
            head_ptrs = self.sample_parses(tokens_batch, mask)
            heads = tokens_batch.gather(dim=1, index=head_ptrs)
            mutated_tokens = self.sample_tokens(tokens_batch, head_ptrs, mask)
            head_ptrs_exp = head_ptrs.unsqueeze(1).expand(
                num_sentences, 2, num_tokens)
            mutated_heads = mutated_tokens.gather(dim=2, index=head_ptrs_exp)

        positive_energy = self.embedding.link_energy(tokens_batch, heads)
        negative_energy = self.embedding.link_energy(
                mutated_tokens, mutated_heads).sum(dim=1) / 2

        loss = -torch.where(
            mask, 
            torch.tensor([[0]], dtype=torch.float),
            positive_energy - negative_energy
        ).sum()

        return loss



class FlatModel:

    def __init__(self, vocab_size, embed_dim, Nx):
        bias = torch.log(Nx / Nx.sum()).unsqueeze(1).clamp(min=-10)
        self.embedding = m.EmbeddingLayer(vocab_size, embed_dim, bias=None)
        self.token_sampler = m.st.SimpleTokenSampler(Nx)
        self.parse_sampler = m.sp.ConvSampler()


    def parameters(self):
        return self.embedding.parameters()


    def sample_parses(self, tokens_batch, mask):
        return self.parse_sampler.sample_parses(tokens_batch, mask)


    def sample_tokens(self, tokens_batch, prev_head_ptrs, mask):
        return self.token_sampler.sample_tokens(
            tokens_batch, prev_head_ptrs, self.embedding, mask)


    def get_loss(self, tokens_batch, mask):
        with torch.no_grad():
            positive_head_ptrs = self.sample_parses(tokens_batch, mask)
            negative_heads = self.sample_tokens(
                tokens_batch, positive_head_ptrs, mask)
            positive_heads = tokens_batch.gather(
                dim=1, index=positive_head_ptrs)

        # Mask off ROOT, we don't want it's energy in the model.
        mask[:,0] = True
        loss = -torch.where(
            mask,
            torch.tensor([[0]], dtype=torch.float),
            self.embedding.link_energy(tokens_batch, positive_heads, mask) -
            self.embedding.link_energy(tokens_batch, negative_heads, mask) 
        ).sum()
        return loss



class RebasedFlatModel:

    def __init__(self, vocab_size, embed_dim, Nx):
        bias = torch.log(Nx / Nx.sum()).unsqueeze(1).clamp(min=-10)
        self.embedding = m.EmbeddingLayer(vocab_size, embed_dim, bias=None)
        self.token_sampler = m.st.RebasedTokenSampler(Nx)
        self.parse_sampler = m.sp.RebasedConvSampler(Nx)


    def parameters(self):
        return self.embedding.parameters()


    def sample_parses(self, tokens_batch, mask):
        return self.parse_sampler.sample_parses(tokens_batch, mask)


    def sample_tokens(self, tokens_batch, prev_head_ptrs, mask):
        return self.token_sampler.sample_tokens(
            tokens_batch, prev_head_ptrs, self.embedding, mask)


    def get_loss(self, tokens_batch, mask):
        with torch.no_grad():
            positive_head_ptrs = self.sample_parses(tokens_batch, mask)
            negative_heads = self.sample_tokens(
                tokens_batch, positive_head_ptrs, mask)
            positive_heads = tokens_batch.gather(
                dim=1, index=positive_head_ptrs)

        # Mask off ROOT, we don't want it's energy in the model.
        mask[:,0] = True
        loss = -torch.where(
            mask,
            torch.tensor([[0]], dtype=torch.float),
            self.embedding.link_energy(tokens_batch, positive_heads, mask) -
            self.embedding.link_energy(tokens_batch, negative_heads, mask) 
        ).sum()
        return loss
