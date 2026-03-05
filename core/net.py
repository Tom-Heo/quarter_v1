from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import NUM_FEATURES, NUM_TARGET_FEATURES, SEQ_LEN, TARGET_LEN

from .block import EmbeddingBlock, QuarterBlock

_NUM_BLOCKS = 32
_MAX_SEQ_LEN = SEQ_LEN + TARGET_LEN


class QuarterNet(nn.Module):
    """
    Input:  (B, SEQ_LEN, NUM_FEATURES)
    Output: (B, TARGET_LEN, NUM_TARGET_FEATURES)
    """

    def __init__(
        self,
        features: int = NUM_FEATURES,
        target_features: int = NUM_TARGET_FEATURES,
        target_len: int = TARGET_LEN,
        d_model: int = 2048,
        num_heads: int = 32,
        num_blocks: int = _NUM_BLOCKS,
        max_seq_len: int = _MAX_SEQ_LEN,
    ):
        super().__init__()
        self.target_len = target_len

        self.embedding = EmbeddingBlock(features, d_model)
        self.blocks = nn.ModuleList(
            QuarterBlock(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len)
            for _ in range(num_blocks)
        )

        self.cls_tokens = nn.Parameter(torch.zeros(1, target_len, d_model))
        nn.init.normal_(self.cls_tokens, std=0.02)
        self.head = nn.Linear(d_model, target_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = x[:, -self.target_len :, :]
        return self.head(x)
