from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import NUM_CLS_TOKENS, NUM_DIRECTION_OUTPUTS, NUM_FEATURES, SEQ_LEN

from .block import EmbeddingBlock, QuarterBlock, FFNBlock

_NUM_BLOCKS = 64
_MAX_SEQ_LEN = SEQ_LEN + NUM_CLS_TOKENS


class QuarterNet(nn.Module):
    """
    Input:  (B, SEQ_LEN, NUM_FEATURES)
    Output: (B,)
    """

    def __init__(
        self,
        features: int = NUM_FEATURES,
        direction_outputs: int = NUM_DIRECTION_OUTPUTS,
        num_cls_tokens: int = NUM_CLS_TOKENS,
        d_model: int = 2048,
        num_heads: int = 16,
        num_blocks: int = _NUM_BLOCKS,
        max_seq_len: int = _MAX_SEQ_LEN,
    ):
        super().__init__()
        if num_cls_tokens != 1:
            raise ValueError("QuarterNet은 현재 단일 CLS 토큰만 지원합니다.")
        if direction_outputs != 1:
            raise ValueError("QuarterNet은 현재 단일 방향 logit만 지원합니다.")

        self.num_cls_tokens = num_cls_tokens

        self.embedding1 = EmbeddingBlock(features, d_model)
        self.embedding2 = FFNBlock(d_model)
        self.blocks = nn.ModuleList(
            QuarterBlock(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len)
            for _ in range(num_blocks)
        )

        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, d_model))
        nn.init.normal_(self.cls_tokens, std=0.02)
        self.head1 = FFNBlock(d_model)
        self.head2 = nn.Linear(d_model, direction_outputs, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding1(x)
        x = self.embedding2(x)
        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        x = x[:, -self.num_cls_tokens :, :]
        x = self.head1(x)
        x = self.head2(x)
        return x.squeeze(-1).squeeze(-1)
