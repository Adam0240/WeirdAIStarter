from torch import nn as nn
from .layer_norm import LayerNorm
from .attention import SelfAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self,
        emb_dim,
        dropout,
        qkv_bias=False):
        super().__init__()

    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    # using 
    #  - LayerNorm
        self.norm1 = LayerNorm(
            emb_dim=emb_dim
        )

        self.norm2 = LayerNorm(
            emb_dim=emb_dim
        )

    #  - SelfAttention from previous assignment
        self.att = SelfAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            qkv_bias=qkv_bias
        )

    #  - FeedForward
        self.ff = FeedForward(
            emb_dim=emb_dim
        )

    #  - Residual connections
        self.drop_shortcut = nn.Dropout(
                dropout
            )


    def forward(self, x):

        # Residual connection for self-attention
        shortcut = x

        x = self.norm1(x)
        x, attn_weights = self.att(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        # Residual connection for feed forward
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        return x    