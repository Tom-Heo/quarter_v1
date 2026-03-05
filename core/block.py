import torch
import torch.nn as nn
import torch.nn.functional as F
from .heo import Heo


class FFNBlock(nn.Module):
    """
    Input: (B, S, D)
    Output: (B, S, D)
    """

    def __init__(self, d_model: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_model * 2

        self.up_proj = nn.Linear(d_model, self.d_hidden * 3, bias=False)
        self.gate1 = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.45)
        self.gate2 = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.45)
        self.feature = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.0)
        self.down_proj = nn.Linear(self.d_hidden, self.d_model, bias=False)
        self.residual_gate = Heo.HeoGate(self.d_model, lr_scale=self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.up_proj(x)

        gate1, gate2, feature = x.chunk(3, dim=2)
        gate1 = self.gate1(gate1)
        gate2 = self.gate2(gate2)
        feature = self.feature(feature)
        x = gate1 * gate2 * feature
        x = torch.sign(x) * (torch.abs(x) + (1e-8)).pow(1.0 / 3.0)
        x = self.down_proj(x)
        return self.residual_gate(x, residual)


class RoPE(nn.Module):
    """
    Input q, k: (B, S, H, D_h)
    Output q_rot, k_rot: (B, S, H, D_h)
    """

    def __init__(
        self, head_dim: int = 128, max_seq_len: int = 9600, base: float = 10000.0
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        S = q.size(1)
        cos = self.cos_cached[:S].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:S].unsqueeze(0).unsqueeze(2)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class AttentionBlock(nn.Module):
    """
    Input: (B, S, D)
    Output: (B, S, D)
    """

    def __init__(
        self, d_model: int = 2048, num_heads: int = 32, max_seq_len: int = 9600
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model({d_model}) must be divisible by num_heads({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim, max_seq_len)

        self.residual_gate = Heo.HeoGate(self.d_model, lr_scale=self.d_model)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, S, D = x.shape
        residual = x

        qkv = self.qkv_proj(x)

        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)

        q_rot, k_rot = self.rope(q, k)

        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, S, D)

        out = self.o_proj(attn_out)
        return self.residual_gate(out, residual)


class QuarterBlock(nn.Module):
    """
    Input: (B, S, D)
    Output: (B, S, D)
    """

    def __init__(
        self, d_model: int = 2048, num_heads: int = 32, max_seq_len: int = 9600
    ):
        super().__init__()
        self.attn = AttentionBlock(d_model, num_heads, max_seq_len)
        self.ffn = FFNBlock(d_model)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        x = self.attn(x, is_causal=is_causal)
        x = self.ffn(x)
        return x


class EmbeddingBlock(nn.Module):
    """
    Input: (B, S, Features)
    Output: (B, S, D)
    """

    def __init__(self, features: int = 9, d_model: int = 2048):
        super().__init__()

        self.features = features
        self.d_model = d_model
        self.d_hidden = d_model * 2

        self.up_proj = nn.Linear(self.features, self.d_hidden * 3, bias=False)
        self.gate1 = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.45)
        self.gate2 = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.45)
        self.feature = Heo.HeLUseq(self.d_hidden, lr_scale=self.d_model, std=0.0)
        self.down_proj = nn.Linear(self.d_hidden, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.up_proj(x)

        gate1, gate2, feature = x.chunk(3, dim=2)
        gate1 = self.gate1(gate1)
        gate2 = self.gate2(gate2)
        feature = self.feature(feature)
        x = gate1 * gate2 * feature
        x = torch.sign(x) * (torch.abs(x) + (1e-8)).pow(1.0 / 3.0)
        x = self.down_proj(x)
        return x
