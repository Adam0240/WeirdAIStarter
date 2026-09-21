import torch
import torch.nn as nn


import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .layer_norm import LayerNorm


class WeirdAIModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        context_length=128,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length

        # Convert token IDs into embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            emb_dim
        )

        # Give the model information about token position
        self.position_embedding = nn.Embedding(
            context_length,
            emb_dim
        )

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=emb_dim,
                    dropout=dropout,
                    context_length=context_length
)
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = LayerNorm(
            emb_dim=emb_dim
        )

        # Convert embeddings into vocabulary logits
        self.output_head = nn.Linear(
            emb_dim,
            vocab_size,
            bias=False
        )

    def forward(self, x):
        batch_size, sequence_length = x.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        # Token embeddings
        token_embeddings = self.token_embedding(x)

        # Position embeddings
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token and position information
        x = token_embeddings + position_embeddings

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Produce logits for every token position
        logits = self.output_head(x)

        return logits