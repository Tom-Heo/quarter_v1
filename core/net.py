from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .block import EmbeddingBlock, QuarterBlock


class QuarterNet(nn.Module):
    """
    Input: (B, S, Features)
    Output: (B, S, 5)
    """

    def __init__(self, features: int = 7, d_model: int = 4096):
        super().__init__()
        self.embedding = EmbeddingBlock(features, d_model)
        self.quarter1 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter2 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter3 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter4 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter5 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter6 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter7 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter8 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter9 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter10 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter11 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter12 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter13 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter14 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter15 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter16 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter17 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter18 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter19 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter20 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter21 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter22 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter23 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter24 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter25 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter26 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter27 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter28 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter29 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter30 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter31 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)
        self.quarter32 = QuarterBlock(d_model=d_model, num_heads=32, max_seq_len=9696)

        self._blocks = [
            self.quarter1, self.quarter2, self.quarter3, self.quarter4,
            self.quarter5, self.quarter6, self.quarter7, self.quarter8,
            self.quarter9, self.quarter10, self.quarter11, self.quarter12,
            self.quarter13, self.quarter14, self.quarter15, self.quarter16,
            self.quarter17, self.quarter18, self.quarter19, self.quarter20,
            self.quarter21, self.quarter22, self.quarter23, self.quarter24,
            self.quarter25, self.quarter26, self.quarter27, self.quarter28,
            self.quarter29, self.quarter30, self.quarter31, self.quarter32,
        ]

        self.cls_tokens = nn.Parameter(torch.zeros(1, 96, d_model))
        nn.init.normal_(self.cls_tokens, std=0.02)
        self.head = nn.Linear(d_model, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        for block in self._blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = x[:, -96:, :]
        return self.head(x)
