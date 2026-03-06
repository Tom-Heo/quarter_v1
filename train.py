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
EVAL_SAMPLES = 256
LOG_INTERVAL = 8
OUTPUT_DIR = "outputs"
CKPT_TASK = "direction_binary_cls1"
CHECK_NUMERICS = True


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
    eval_samples: int,
):
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    sep = "═" * 58
    banner = (
        f"\n{sep}\n"
        f"  QuarterNet 방향 분류 학습\n"
        f"{sep}\n"
        f"  시각        │ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  모드        │ {mode}\n"
        f"  태스크      │ 미래 96개 lnCO 누적 방향 분류\n"
        f"  디바이스    │ {gpu}\n"
        f"  파라미터    │ {n_params:,}\n"
        f"  학습 데이터 │ {n_train:,} 샘플\n"
        f"  평가 데이터 │ {n_eval:,} 샘플\n"
        f"  평가 샘플   │ {min(n_eval, eval_samples):,}개 랜덤 샘플\n"
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
    eval_accuracy: float,
    best_eval_accuracy: float,
    is_best: bool,
):
    mark = " ★ 갱신" if is_best else ""
    header = f"── 평가 (스텝 {global_step:,}) "
    header += "─" * max(0, 54 - len(header))
    _log(header)
    _log(f"  Eval BCE    │ {eval_loss:.6f}")
    _log(f"  Eval Acc    │ {eval_accuracy:.2f}%")
    _log(f"  Best Acc    │ {best_eval_accuracy:.2f}%{mark}")
    _log(f"  last.pt     │ 저장 완료")
    _log("  last_export.pt │ 저장 완료")
    if is_best:
        _log("  best.pt     │ 저장 완료")
        _log("  best_export.pt │ 저장 완료")
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
    best_eval_accuracy: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": CKPT_TASK,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_eval_accuracy": best_eval_accuracy,
        },
        path,
    )


def _save_export(path: Path, model: nn.Module):
    """EMA 적용 상태에서 호출. 추론 전용 가중치만 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp_path)
    tmp_path.replace(path)


def _direction_targets(targets: torch.Tensor) -> torch.Tensor:
    return (targets[..., 0].sum(dim=-1) > 0).float()


def _direction_predictions(logits: torch.Tensor) -> torch.Tensor:
    return logits > 0


def _tensor_finite_summary(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach()
    finite_mask = torch.isfinite(t)
    finite_count = int(finite_mask.sum().item())
    total_count = t.numel()
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    summary = (
        f"{name} │ shape={tuple(t.shape)} │ dtype={t.dtype} │ "
        f"finite {finite_count}/{total_count} │ nan {nan_count} │ inf {inf_count}"
    )
    if finite_count > 0:
        finite_values = t[finite_mask]
        summary += (
            f" │ min {finite_values.min().item():.6e}"
            f" │ max {finite_values.max().item():.6e}"
        )
    return summary


def _assert_finite_tensors(
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    **tensors: torch.Tensor,
) -> None:
    if not CHECK_NUMERICS:
        return

    bad_names = [
        name
        for name, tensor in tensors.items()
        if not torch.isfinite(tensor).all().item()
    ]
    if not bad_names:
        return

    _log(
        f"[디버그] 비정상 수치 감지 │ 에폭 {epoch} │ "
        f"스텝 {step_in_epoch} │ 전체 {global_step:,} │ {', '.join(bad_names)}"
    )
    for name, tensor in tensors.items():
        _log(f"  {_tensor_finite_summary(name, tensor)}")

    raise RuntimeError(f"Non-finite tensor detected: {', '.join(bad_names)}")


def _load_resume_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    device: torch.device,
) -> tuple[bool, int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("task") != CKPT_TASK:
        _log(
            "체크포인트 구조 불일치 │ 기존 회귀 체크포인트이거나 다른 태스크입니다. "
            "현재 설정으로 새 학습을 시작합니다."
        )
        return False, 0, 0, float("-inf")

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        _log(f"체크포인트 로드 실패 │ 현재 모델 구조와 호환되지 않아 재시작합니다. ({exc})")
        return False, 0, 0, float("-inf")

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    ema.load_state_dict(ckpt["ema"])

    global_step = int(ckpt["global_step"])
    start_epoch = int(ckpt["epoch"])
    best_eval_accuracy = float(ckpt.get("best_eval_accuracy", float("-inf")))
    return True, global_step, start_epoch, best_eval_accuracy


# ── 평가 ─────────────────────────────────────────────────────────────


def _evaluate(
    model: nn.Module,
    eval_dataset: QuarterDataset,
    criterion: nn.Module,
    device: torch.device,
    n_samples: int = EVAL_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, float]:
    model.eval()
    sample_count = min(n_samples, len(eval_dataset))
    indices = torch.randperm(len(eval_dataset))[:sample_count]

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            batch_indices = indices[start : start + batch_size].tolist()
            pairs = [eval_dataset[i] for i in batch_indices]
            x = torch.stack([p[0] for p in pairs]).float().to(device)
            y = torch.stack([p[1] for p in pairs]).float().to(device)

            logits = model(x)
            labels = _direction_targets(y)
            loss = criterion(logits, labels)
            preds = _direction_predictions(logits)

            batch_size_now = len(batch_indices)
            total_loss += loss.item() * batch_size_now
            total_correct += int((preds == (labels > 0.5)).sum().item())
            total_seen += batch_size_now

    avg_loss = total_loss / max(total_seen, 1)
    accuracy = 100.0 * total_correct / max(total_seen, 1)
    return avg_loss, accuracy


# ── 방향 시각화 ──────────────────────────────────────────────────────


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
        logit = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()

    true_up = bool(_direction_targets(y_true.unsqueeze(0)).item())
    pred_up = bool(_direction_predictions(logit).item())
    prob_up = torch.sigmoid(logit).item()
    cum_lnco = np.cumsum(y_true[:, 0].numpy())
    steps = np.arange(1, len(cum_lnco) + 1)
    curve_color = "#22ab94" if true_up else "#f23645"

    fig = plt.figure(figsize=(18, 5.8), facecolor="#ffffff")
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.5, 1.0],
        wspace=0.12,
        left=0.05,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_meta = fig.add_subplot(gs[0, 1])

    ax_curve.set_facecolor("#fafafa")
    ax_curve.plot(steps, cum_lnco, color=curve_color, linewidth=2.2)
    ax_curve.axhline(0.0, color="#9e9e9e", linewidth=1.0, linestyle="--")
    ax_curve.fill_between(steps, cum_lnco, 0.0, color=curve_color, alpha=0.12)
    ax_curve.set_xlim(1, len(steps))
    ax_curve.set_xlabel("미래 스텝", fontsize=10, color="#666666")
    ax_curve.set_ylabel("누적 lnCO", fontsize=10, color="#666666")
    ax_curve.set_title("Ground Truth 누적 방향 경로", fontsize=13, color="#333333")
    ax_curve.grid(True, linestyle=":", linewidth=0.6, color="#dcdcdc", alpha=0.8)
    for sp in ("top", "right"):
        ax_curve.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_curve.spines[sp].set_color("#cccccc")
    ax_curve.tick_params(axis="both", labelsize=9, colors="#777777")

    ax_meta.set_facecolor("#fafafa")
    ax_meta.axis("off")
    ax_meta.text(
        0.05,
        0.88,
        "Direction Summary",
        fontsize=16,
        fontweight="bold",
        color="#222222",
        ha="left",
    )
    ax_meta.text(
        0.05,
        0.68,
        f"Prediction : {'UP' if pred_up else 'DOWN'}\n"
        f"P(up)      : {prob_up:.4f}\n"
        f"True Label  : {'UP' if true_up else 'DOWN'}\n"
        f"Sample IDX  : {idx:,}",
        fontsize=12,
        color="#444444",
        ha="left",
        linespacing=1.7,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#d0d0d0",
            linewidth=1.0,
        ),
    )
    ax_meta.text(
        0.05,
        0.28,
        f"Final cumulative lnCO : {cum_lnco[-1]:+.6f}",
        fontsize=11,
        color=curve_color,
        ha="left",
    )

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

    parser = argparse.ArgumentParser(description="QuarterNet 방향 분류 학습")
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
    criterion = nn.BCEWithLogitsLoss().to(device)

    # ── 상태 초기화 ───────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    best_eval_accuracy = float("-inf")
    resumed = False

    # ── 체크포인트 로드 ───────────────────────────────────────────────
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_pt = ckpt_dir / "last.pt"

    if do_resume and last_pt.exists():
        _log("체크포인트 로드 중...")
        resumed, global_step, start_epoch, best_eval_accuracy = _load_resume_state(
            last_pt,
            model,
            optimizer,
            scheduler,
            ema,
            device,
        )
        if resumed:
            _log(
                f"체크포인트 로드 완료 │ 에폭 {start_epoch} │ "
                f"스텝 {global_step:,} │ Best Acc {best_eval_accuracy:.2f}%"
            )
    elif do_resume:
        _log("체크포인트 없음 │ 처음부터 학습을 시작합니다")
    else:
        _log("재시작 모드 │ 처음부터 학습을 시작합니다")

    # ── 배너 ──────────────────────────────────────────────────────────
    if do_resume and resumed:
        mode_str = "재개 (--resume)"
    elif do_resume:
        mode_str = "재개 요청 → 호환 불가로 재시작"
    else:
        mode_str = "재시작 (--restart)"
    n_params = sum(p.numel() for p in model.parameters())
    _log_banner(
        mode_str,
        device,
        n_params,
        len(train_dataset),
        len(eval_dataset),
        warmup_steps,
        EVAL_SAMPLES,
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
                next_step = global_step + 1
                _assert_finite_tensors(epoch, step_in_epoch, next_step, x=x, y=y)
                logits = model(x)
                _assert_finite_tensors(
                    epoch, step_in_epoch, next_step, x=x, y=y, logits=logits
                )
                labels = _direction_targets(y)
                _assert_finite_tensors(
                    epoch, step_in_epoch, next_step, y=y, logits=logits, labels=labels
                )
                loss = criterion(logits, labels)
                _assert_finite_tensors(
                    epoch,
                    step_in_epoch,
                    next_step,
                    x=x,
                    y=y,
                    logits=logits,
                    labels=labels,
                    loss=loss,
                )

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
                    train_acc = (
                        (_direction_predictions(logits) == (labels > 0.5))
                        .float()
                        .mean()
                        .item()
                        * 100.0
                    )
                    _log(
                        f"[학습] 에폭 {epoch} │ "
                        f"스텝 {step_in_epoch}/{len(train_loader)} │ "
                        f"전체 {global_step:,} │ "
                        f"BCE {loss.item():.6f} │ "
                        f"Acc {train_acc:.2f}% │ "
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

                    eval_loss, eval_accuracy = _evaluate(
                        model,
                        eval_dataset,
                        criterion,
                        device,
                    )
                    _save_export(ckpt_dir / "last_export.pt", model)

                    is_best = eval_accuracy > best_eval_accuracy
                    if is_best:
                        best_eval_accuracy = eval_accuracy
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
                        best_eval_accuracy,
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
                            best_eval_accuracy,
                        )

                    _log_eval(
                        global_step,
                        eval_loss,
                        eval_accuracy,
                        best_eval_accuracy,
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
