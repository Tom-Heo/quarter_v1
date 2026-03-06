from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from config import NUM_CLS_TOKENS, NUM_DIRECTION_OUTPUTS, NUM_FEATURES, SEQ_LEN

from .block import EmbeddingBlock, QuarterBlock, FFNBlock

_NUM_BLOCKS = 64
_MAX_SEQ_LEN = SEQ_LEN + NUM_CLS_TOKENS
DEBUG_NUMERICS = True


def _tensor_finite_summary(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach()
    finite_mask = torch.isfinite(t)
    finite_count = int(finite_mask.sum().item())
    total_count = t.numel()
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    summary = (
        f"{name} | shape={tuple(t.shape)} | dtype={t.dtype} | "
        f"finite {finite_count}/{total_count} | nan {nan_count} | inf {inf_count}"
    )
    if finite_count > 0:
        finite_values = t[finite_mask]
        summary += (
            f" | min {finite_values.min().item():.6e}"
            f" | max {finite_values.max().item():.6e}"
        )
    return summary


def _ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not DEBUG_NUMERICS:
        return
    if torch.isfinite(tensor).all().item():
        return
    raise RuntimeError(_tensor_finite_summary(name, tensor))


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
        d_model: int = 1024,
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
        _ensure_finite("QuarterNet.input", x)

        try:
            x = self.embedding1(x)
        except RuntimeError as exc:
            raise RuntimeError(f"QuarterNet.embedding1 failed: {exc}") from exc
        _ensure_finite("QuarterNet.embedding1", x)

        try:
            x = self.embedding2(x)
        except RuntimeError as exc:
            raise RuntimeError(f"QuarterNet.embedding2 failed: {exc}") from exc
        _ensure_finite("QuarterNet.embedding2", x)

        cls = self.cls_tokens.expand(x.size(0), -1, -1)
        _ensure_finite("QuarterNet.cls_tokens", cls)
        x = torch.cat([x, cls], dim=1)
        _ensure_finite("QuarterNet.concat", x)

        for idx, block in enumerate(self.blocks):
            try:
                x = checkpoint(block, x, use_reentrant=False)
            except RuntimeError as exc:
                raise RuntimeError(f"QuarterNet.block[{idx}] failed: {exc}") from exc
            _ensure_finite(f"QuarterNet.block[{idx}].output", x)

        x = x[:, -self.num_cls_tokens :, :]
        _ensure_finite("QuarterNet.cls_slice", x)

        try:
            x = self.head1(x)
        except RuntimeError as exc:
            raise RuntimeError(f"QuarterNet.head1 failed: {exc}") from exc
        _ensure_finite("QuarterNet.head1", x)

        x = self.head2(x)
        _ensure_finite("QuarterNet.head2", x)

        x = x.squeeze(-1).squeeze(-1)
        _ensure_finite("QuarterNet.logits", x)
        return x
