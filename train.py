from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    BINANCE_SYMBOL,
    CHECKPOINT_DIR,
    DATASET_DIR,
    EVAL_DATASET_END,
    EVAL_DATASET_START,
    LOSS_WEIGHTS,
    TRAIN_DATASET_END,
    TRAIN_DATASET_START,
)
from core.heo import Heo
from core.net import QuarterNet
from data.dataset import QuarterDataset, build_dataset_pipeline, is_legacy_hdf5

_korean_fonts = ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
for _f in _korean_fonts:
    if any(
        _f.lower() in f.name.lower()
        for f in matplotlib.font_manager.fontManager.ttflist
    ):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ── 학습 상수 ────────────────────────────────────────────────────────

EMA_DECAY = 0.999
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_GAMMA = 0.999998
WARMUP_START_FACTOR = 1e-7
EVAL_INTERVAL = 256
LOG_INTERVAL = 8
OUTPUT_DIR = "outputs"


# ── EMA ──────────────────────────────────────────────────────────────


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ── 로깅 ─────────────────────────────────────────────────────────────

LOG_DIR = "logs"

logger = logging.getLogger("quarter")


def _setup_logging() -> None:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{ts}.log"

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"로그 파일: {log_file}")


def _log(msg: str):
    logger.info(msg)


def _log_banner(
    mode: str,
    device: torch.device,
    n_params: int,
    n_train: int,
    n_eval: int,
    warmup_steps: int,
):
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    sep = "═" * 58
    banner = (
        f"\n{sep}\n"
        f"  QuarterNet 학습\n"
        f"{sep}\n"
        f"  시각        │ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  모드        │ {mode}\n"
        f"  디바이스    │ {gpu}\n"
        f"  파라미터    │ {n_params:,}\n"
        f"  학습 데이터 │ {n_train:,} 샘플\n"
        f"  평가 데이터 │ {n_eval:,} 샘플\n"
        f"  배치 크기   │ {BATCH_SIZE}\n"
        f"  학습률      │ {LEARNING_RATE:.2e}\n"
        f"  EMA 계수    │ {EMA_DECAY}\n"
        f"  스케줄러    │ ExponentialLR (γ={SCHEDULER_GAMMA})\n"
        f"  워밍업      │ {warmup_steps:,} 스텝 (1 에폭)\n"
        f"{sep}"
    )
    for line in banner.splitlines():
        logger.info(line)


def _log_eval(
    global_step: int,
    eval_loss: float,
    best_eval_loss: float,
    is_best: bool,
):
    mark = " ★ 갱신" if is_best else ""
    header = f"── 평가 (스텝 {global_step:,}) "
    header += "─" * max(0, 54 - len(header))
    _log(header)
    _log(f"  Eval Loss   │ {eval_loss:.6f}")
    _log(f"  Best Loss   │ {best_eval_loss:.6f}{mark}")
    _log(f"  last.pt     │ 저장 완료")
    if is_best:
        _log("  best.pt     │ 저장 완료")
    _log("─" * 54)


# ── 데이터셋 준비 ────────────────────────────────────────────────────


def _ensure_datasets() -> tuple[Path, Path]:
    ds_dir = Path(DATASET_DIR)
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_path = (
        ds_dir / f"{BINANCE_SYMBOL}_{TRAIN_DATASET_START}_{TRAIN_DATASET_END}.h5"
    )
    eval_path = ds_dir / f"{BINANCE_SYMBOL}_{EVAL_DATASET_START}_{EVAL_DATASET_END}.h5"

    for p in [train_path, eval_path]:
        if is_legacy_hdf5(str(p)):
            _log(f"구 스키마 감지 → 삭제: {p}")
            p.unlink()

    if not train_path.exists():
        _log(f"학습 데이터셋 생성 중: {train_path}")
        build_dataset_pipeline(TRAIN_DATASET_START, TRAIN_DATASET_END, str(train_path))

    if not eval_path.exists():
        _log(f"평가 데이터셋 생성 중: {eval_path}")
        build_dataset_pipeline(EVAL_DATASET_START, EVAL_DATASET_END, str(eval_path))

    return train_path, eval_path


# ── 체크포인트 ────────────────────────────────────────────────────────


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    global_step: int,
    epoch: int,
    best_eval_loss: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_eval_loss": best_eval_loss,
        },
        path,
    )


def _save_export(path: Path, model: nn.Module):
    """EMA 적용 상태에서 호출. 추론 전용 가중치만 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


# ── 평가 ─────────────────────────────────────────────────────────────


def _evaluate(
    model: nn.Module,
    eval_dataset: QuarterDataset,
    criterion: nn.Module,
    device: torch.device,
    n_samples: int = 1,
) -> float:
    model.eval()
    indices = torch.randint(0, len(eval_dataset), (n_samples,))
    pairs = [eval_dataset[i] for i in indices]
    x = torch.stack([p[0] for p in pairs]).to(device)
    y = torch.stack([p[1] for p in pairs]).to(device)
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, y)
    return loss.item()


# ── 캔들차트 시각화 ──────────────────────────────────────────────────


def _reconstruct_ohlc(log_returns: np.ndarray, base: float = 100.0) -> pd.DataFrame:
    """(T, 5) = [lnCO, lnHO, lnLO, lnCH, lnCL] → OHLC DataFrame"""
    T = log_returns.shape[0]
    opens = np.empty(T)
    highs = np.empty(T)
    lows = np.empty(T)
    closes = np.empty(T)

    opens[0] = base
    for t in range(T):
        o = opens[t]
        closes[t] = o * np.exp(log_returns[t, 0])
        highs[t] = o * np.exp(log_returns[t, 1])
        lows[t] = o * np.exp(log_returns[t, 2])
        if t + 1 < T:
            opens[t + 1] = closes[t]

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
    )


def _draw_candles(ax: plt.Axes, df: pd.DataFrame, label: str, tag_color: str):
    n = len(df)
    up_c, dn_c = "#22ab94", "#f23645"
    body_w = 0.6

    for i in range(n):
        o, h, l, c = df.iloc[i][["Open", "High", "Low", "Close"]]
        color = up_c if c >= o else dn_c
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, solid_capstyle="round")
        body_lo = min(o, c)
        body_h = abs(c - o) or (h - l) * 0.005 or 0.01
        ax.add_patch(
            plt.Rectangle(
                (i - body_w / 2, body_lo),
                body_w,
                body_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
            )
        )

    y_lo, y_hi = df["Low"].min(), df["High"].max()
    margin = (y_hi - y_lo) * 0.06 or 0.1
    ax.set_xlim(-1, n)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#cccccc")

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#e0e0e0", alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", labelsize=8, colors="#888888")
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_ylabel("가격", fontsize=9, color="#888888", labelpad=8)

    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=tag_color,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=tag_color,
            alpha=0.85,
            linewidth=1.2,
        ),
    )


def _visualize(
    model: nn.Module,
    eval_dataset: QuarterDataset,
    device: torch.device,
    global_step: int,
    output_dir: Path,
) -> Path:
    """EMA 적용 상태에서 호출."""
    idx = torch.randint(0, len(eval_dataset), (1,)).item()
    x, y_true = eval_dataset[idx]
    with torch.no_grad():
        y_pred = model(x.unsqueeze(0).to(device))

    gt_df = _reconstruct_ohlc(y_true.numpy())
    pred_df = _reconstruct_ohlc(y_pred.squeeze(0).cpu().numpy())

    fig = plt.figure(figsize=(22, 8.5), facecolor="#ffffff")
    gs = fig.add_gridspec(
        1,
        2,
        wspace=0.15,
        left=0.05,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])

    for ax in (ax_pred, ax_gt):
        ax.set_facecolor("#fafafa")

    _draw_candles(ax_pred, pred_df, "Prediction", "#1565c0")
    _draw_candles(ax_gt, gt_df, "Ground Truth", "#555555")

    fig.suptitle(
        f"Step {global_step:,}",
        fontsize=11,
        color="#999999",
        y=0.96,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"step_{global_step:06d}.png"
    fig.savefig(
        str(save_path),
        dpi=150,
        bbox_inches="tight",
        facecolor="#ffffff",
        pad_inches=0.3,
    )
    plt.close(fig)
    return save_path


# ── 메인 ─────────────────────────────────────────────────────────────


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="QuarterNet 학습")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--restart", action="store_true", help="처음부터 새로 학습")
    group.add_argument(
        "--resume", action="store_true", help="체크포인트에서 이어서 학습"
    )
    args = parser.parse_args()

    do_resume = not args.restart

    # ── 디바이스 ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        _log("경고: CUDA를 사용할 수 없습니다. CPU로 학습합니다.")

    # ── 데이터셋 ──────────────────────────────────────────────────────
    train_path, eval_path = _ensure_datasets()
    train_dataset = QuarterDataset(str(train_path))
    eval_dataset = QuarterDataset(str(eval_path))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    # ── 모델 ──────────────────────────────────────────────────────────
    model = QuarterNet().to(device)

    # ── 옵티마이저 ────────────────────────────────────────────────────
    optimizer = Heo.Heoptimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ── EMA ───────────────────────────────────────────────────────────
    ema = EMA(model, decay=EMA_DECAY)

    # ── 스케줄러 (1 에폭 warmup → ExponentialLR) ─────────────────────
    warmup_steps = len(train_dataset)
    warmup_sched = LinearLR(
        optimizer, start_factor=WARMUP_START_FACTOR, total_iters=warmup_steps
    )
    exp_sched = ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)
    scheduler = SequentialLR(
        optimizer, [warmup_sched, exp_sched], milestones=[warmup_steps]
    )

    # ── 손실 함수 ─────────────────────────────────────────────────────
    criterion = Heo.HeoLoss(feature_weights=LOSS_WEIGHTS).to(device)

    # ── 상태 초기화 ───────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    best_eval_loss = float("inf")

    # ── 체크포인트 로드 ───────────────────────────────────────────────
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_pt = ckpt_dir / "last.pt"

    if do_resume and last_pt.exists():
        _log("체크포인트 로드 중...")
        ckpt = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        best_eval_loss = ckpt["best_eval_loss"]
        _log(
            f"체크포인트 로드 완료 │ 에폭 {start_epoch} │ "
            f"스텝 {global_step:,} │ Best Loss {best_eval_loss:.6f}"
        )
    elif do_resume:
        _log("체크포인트 없음 │ 처음부터 학습을 시작합니다")
    else:
        _log("재시작 모드 │ 처음부터 학습을 시작합니다")

    # ── 배너 ──────────────────────────────────────────────────────────
    mode_str = "재개 (--resume)" if do_resume else "재시작 (--restart)"
    n_params = sum(p.numel() for p in model.parameters())
    _log_banner(
        mode_str,
        device,
        n_params,
        len(train_dataset),
        len(eval_dataset),
        warmup_steps,
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────
    output_dir = Path(OUTPUT_DIR)
    epoch = start_epoch
    interval_start = time.time()

    try:
        while True:
            epoch += 1
            model.train()

            for step_in_epoch, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ema.update()
                scheduler.step()
                global_step += 1

                # ── 학습 로그 · 시각화 ────────────────────────────────
                if global_step % LOG_INTERVAL == 0:
                    elapsed = time.time() - interval_start
                    speed = LOG_INTERVAL / max(elapsed, 1e-6)
                    lr = optimizer.param_groups[0]["lr"]
                    _log(
                        f"[학습] 에폭 {epoch} │ "
                        f"스텝 {step_in_epoch}/{len(train_loader)} │ "
                        f"전체 {global_step:,} │ "
                        f"Loss {loss.item():.6f} │ "
                        f"LR {lr:.2e} │ "
                        f"{speed:.2f} step/s"
                    )

                    ema.apply_shadow()
                    model.eval()
                    _visualize(model, eval_dataset, device, global_step, output_dir)
                    ema.restore()
                    model.train()

                    interval_start = time.time()

                # ── 평가 · 체크포인트 ─────────────────────────────────
                if global_step % EVAL_INTERVAL == 0:
                    ema.apply_shadow()
                    model.eval()

                    eval_loss = _evaluate(model, eval_dataset, criterion, device)

                    is_best = eval_loss < best_eval_loss
                    if is_best:
                        best_eval_loss = eval_loss
                        _save_export(ckpt_dir / "best_export.pt", model)

                    ema.restore()

                    _save_checkpoint(
                        last_pt,
                        model,
                        optimizer,
                        scheduler,
                        ema,
                        global_step,
                        epoch,
                        best_eval_loss,
                    )
                    if is_best:
                        _save_checkpoint(
                            ckpt_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            ema,
                            global_step,
                            epoch,
                            best_eval_loss,
                        )

                    _log_eval(
                        global_step,
                        eval_loss,
                        best_eval_loss,
                        is_best,
                    )

                    model.train()
                    interval_start = time.time()

            _log(f"에폭 {epoch} 완료 │ 전체 스텝 {global_step:,}")

    except KeyboardInterrupt:
        _log("학습 중단 │ 마지막 체크포인트까지의 진행이 저장되어 있습니다")
    finally:
        train_dataset.close()
        eval_dataset.close()


if __name__ == "__main__":
    main()
