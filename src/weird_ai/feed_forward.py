import torch
import torch.nn as nn


class GELU(nn.Module):

    def forward(self, x):

        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(
                    torch.tensor(
                        2.0 / torch.pi,
                        device=x.device
                    )
                ) *
                (
                    x +
                    0.044715 * torch.pow(x, 3)
                )
            )
        )


class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(

            # Expands embedding dimension
            nn.Linear(
                emb_dim,
                4 * emb_dim
            ),

            GELU(),

            # Reduces embedding dimension back down
            nn.Linear(
                4 * emb_dim,
                emb_dim
            )
        )

    def forward(self, x):
        return self.layers(x)