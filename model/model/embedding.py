import pdb
import torch
from torch import nn


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embedding_dimension):
        super().__init__()
        self.U = nn.Parameter(
            torch.randn(vocab_size, embedding_dimension)
            *1/torch.sqrt(torch.tensor(embedding_dimension))
        )
        self.Ubias = nn.Parameter(torch.randn(vocab_size, 1))
        self.V = nn.Parameter(
            torch.randn(vocab_size, embedding_dimension)
            *1/torch.sqrt(torch.tensor(embedding_dimension))
        )
        self.Vbias = nn.Parameter(torch.randn(vocab_size, 1))


    def head_match(self, token_id, k=5):
        return (self.V @ self.U[token_id]).topk(k).indices.tolist()


    def similar(self, token_id, k=5):
        similarity = (self.U @ self.U[token_id])
        most_similar = similarity.topk(k+1).indices.tolist()
        most_similar_nonidentical = [
            elm for elm in most_similar 
            if elm != token_id
        ][:k]
        return most_similar_nonidentical


    def batch_energy(self, tokens_batch, heads_batch):
        return self.link_energy(tokens_batch, heads_batch).sum()


    def link_energy(self, tokens_batch, heads_batch, mask=False, temp=1):
        """
        Generates the token-link energies between pairs of tokens, one from
        tokens, one from heads, which are parallel tensors.  The tensors must
        have the same shape.
        """

        last_dim = len(tokens_batch.shape)

        energies = (torch.sum(
            self.U[tokens_batch] * self.V[heads_batch],
            dim=last_dim, keepdim=True
        ) + self.Ubias[tokens_batch] + self.Vbias[heads_batch]).squeeze(last_dim)

        energies[mask] = 0
        return energies / temp


    def sentence_link_energy(self, tokens_batch, mask=None, temp=1):
        num_sentences, num_tokens = tokens_batch.shape

        U = self.U[tokens_batch]
        V_T = self.V[tokens_batch].transpose(1,2)
        ub = self.Ubias[tokens_batch]
        vb_T = self.Vbias[tokens_batch].transpose(1,2)
        energy = U @ V_T + ub + vb_T

        # Tokens never head themselves.  Put -inf on the diagonal, except at
        # ROOT: ROOT should always head itself (on Root row, all else is -inf).
        diag_mask = torch.zeros(
            (num_tokens, num_tokens)).fill_diagonal_(-torch.inf)
        energy += diag_mask.unsqueeze(0)
        root_row = torch.full((num_tokens,), -torch.inf)
        root_row[0] = 0
        energy[:,0] = root_row

        if mask is not None:
            # Tokens cannot choose padding as a head
            mask_head_choice = mask.unsqueeze(1).expand((-1,num_tokens,-1))
            energy[mask_head_choice] = -torch.inf
            # Padding cannot choose non-ROOT tokens as a head
            mask_sub_choice = mask.unsqueeze(2).expand((-1,-1,num_tokens))
            energy[mask_sub_choice] = -torch.inf
            # Padding always chooses ROOT as a head
            energy[:,:,0][mask] = 0

        return energy / temp


    def forward(self, tokens_batch, heads_batch):
        return self.batch_energy(tokens_batch, heads_batch)


