import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-6

        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        # TODO
        # Compute mean
        mean = x.mean(
                    dim=-1,
                    keepdim=True
                )
        # Compute variance
        var = x.var(
                    dim=-1,
                    keepdim=True,
                    unbiased=False
                )
        # Normalize
        norm_x = (
                    (x - mean) /
                    torch.sqrt(var + self.eps)
                )
        # Apply scale and shift
        return self.scale * norm_x + self.shift
    
        raise NotImplementedError()