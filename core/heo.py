from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from collections import defaultdict
from math import sqrt
import math


class Heo:
    class HeLU(nn.Module):
        """
        원본 HeLU: last-dim 기반 (..., dim) 입력용
        """

        def __init__(self, dim: int, std: float = 0.45):
            super().__init__()
            self.std = std
            self.alpha = nn.Parameter(torch.full((dim,), 1.0))
            self.beta = nn.Parameter(torch.full((dim,), -1.0))
            self.redweight = nn.Parameter(
                torch.empty(dim).normal_(mean=0.0, std=self.std)
            )
            self.blueweight = nn.Parameter(
                torch.empty(dim).normal_(mean=0.0, std=self.std)
            )
            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor):
            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = torch.tanh(sqrt(3.0) * self.redweight) + 1.0
            blue = torch.tanh(sqrt(3.0) * self.blueweight) + 1.0
            redx = rgx * red
            bluex = bgx * blue
            x = redx + bluex
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLU2d(nn.Module):
        """
        입력: (N,C,H,W)
        """

        def __init__(self, channels: int, lr_scale: float, std: float = 0.45):
            super().__init__()
            c = int(channels)
            self.channels = c
            self.lr_scale = lr_scale
            self.std = std

            self.alpha = nn.Parameter(torch.full((c,), 1.0))
            self.beta = nn.Parameter(torch.full((c,), -1.0))
            self.redweight = nn.Parameter(
                torch.empty(c).normal_(mean=0.0, std=self.std)
            )
            self.blueweight = nn.Parameter(
                torch.empty(c).normal_(mean=0.0, std=self.std)
            )

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(
                    f"HeLU2d expects NCHW 4D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(1) != self.channels:
                raise ValueError(
                    f"HeLU2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (C,) -> (1,C,1,1) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLUseq(nn.Module):
        """
        입력: (B, S, D)
        """

        def __init__(self, d_model: int, lr_scale: float, std: float = 0.45):
            super().__init__()
            d = int(d_model)
            self.d_model = d
            self.lr_scale = lr_scale
            self.std = std

            self.alpha = nn.Parameter(torch.full((d,), 1.0))
            self.beta = nn.Parameter(torch.full((d,), -1.0))
            self.redweight = nn.Parameter(
                torch.empty(d).normal_(mean=0.0, std=self.std)
            )
            self.blueweight = nn.Parameter(
                torch.empty(d).normal_(mean=0.0, std=self.std)
            )

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 3:
                raise ValueError(
                    f"HeLUSeq expects (B,S,D) 3D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(2) != self.d_model:
                raise ValueError(
                    f"HeLUSeq d_model mismatch: expected D={self.d_model}, got D={x.size(2)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (D,) -> (1,1,D) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, 1, -1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, 1, -1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, 1, -1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, 1, -1)
            y = (alpha * x + beta * raw) / 2
            return y

    class Heoptimizer(AdamW):
        """
        HeLU2d / HeoGate2d 모듈의 .lr_scale 속성과 base lr을 조합하여
        파라미터 그룹을 구성하고, Weight Decay를 해제하는 맞춤형 옵티마이저입니다.
        """

        def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            **kwargs,
        ):
            heo_param_ids = set()
            helu_by_lr = defaultdict(list)

            for module in model.modules():
                if isinstance(
                    module, (Heo.HeLU2d, Heo.HeLUseq, Heo.HeoGate, Heo.HeoGate2d)
                ):
                    effective_lr = lr * module.lr_scale
                    for param in module.parameters(recurse=False):
                        if param.requires_grad:
                            helu_by_lr[effective_lr].append(param)
                            heo_param_ids.add(id(param))

            base_params = [
                p
                for p in model.parameters()
                if p.requires_grad and id(p) not in heo_param_ids
            ]

            param_groups = [
                {"params": base_params, "lr": lr, "weight_decay": weight_decay}
            ]

            for heo_lr, params in helu_by_lr.items():
                param_groups.append(
                    {
                        "params": params,
                        "lr": heo_lr,
                        "weight_decay": 0.0,
                    }
                )

            super().__init__(param_groups, **kwargs)

    class HeoGate(nn.Module):
        def __init__(self, dim: int, lr_scale: float):
            super().__init__()
            self.lr_scale = lr_scale

            self.alpha = nn.Parameter(torch.full((dim,), 0.0))
            self.beta = nn.Parameter(torch.full((dim,), 0.0))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            return (alpha * x + beta * raw) / 2

    class HeoGate2d(nn.Module):
        def __init__(self, channels: int, lr_scale: float):
            super().__init__()
            c = int(channels)
            self.channels = c
            self.lr_scale = lr_scale

            self.alpha = nn.Parameter(torch.full((c,), 0.0))
            self.beta = nn.Parameter(torch.full((c,), 0.0))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            if x.dim() != 4 or x.size(1) != self.channels:
                raise ValueError(
                    f"HeoGate2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            return (alpha * x + beta * raw) / 2

    class HeoLoss(nn.Module):
        def __init__(self, epsilon=1 / (math.e - 1), feature_weights=None):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8
            if feature_weights is not None:
                self.register_buffer(
                    "weights", torch.tensor(feature_weights, dtype=torch.float32)
                )
            else:
                self.weights = None

        def forward(self, pred, target):
            pred = pred.float()
            target = target.float()

            diff = pred - target
            abs_diff = diff.abs()

            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            sharp_loss = torch.log(1 + 10000.0 * charbonnier / self.epsilon) / 10000.0

            loss = torch.where(abs_diff <= 0.0001, sharp_loss, abs_diff)

            if self.weights is not None:
                w = self.weights.view(1, 1, -1)
                return (loss * w).sum() / (w.sum() * loss.shape[0] * loss.shape[1])

            return loss.mean()
