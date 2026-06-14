from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        max_positions: int = 128,
    ) -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_positions, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.head = nn.Linear(embedding_dim, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _batch, seq_len = token_ids.shape
        embedded = self.embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = embedded + self.pos_embedding(positions)
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=self._causal_mask(seq_len, token_ids.device),
        )
        x = x + self.ln1(attn_out)
        x = x + self.ln2(self.ffn(x))
        return self.head(x[:, -1, :])
