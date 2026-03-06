from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from config import (
    BINANCE_SYMBOL,
    CHECKPOINT_DIR,
    DATASET_DIR,
    EVAL_DATASET_END,
    EVAL_DATASET_START,
)
from core.net import QuarterNet
from data.dataset import QuarterDataset, build_dataset_pipeline, is_legacy_hdf5

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_SAMPLES = 100
DEFAULT_BATCH_SIZE = 1
PROGRESS_LOG_STEPS = 10

logger = logging.getLogger("quarter.eval")
LogFn = Callable[[str], None]


def _setup_logging() -> Path:
    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{ts}.log"

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"로그 파일: {log_file}")
    return log_file


def _log(msg: str) -> None:
    logger.info(msg)


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_path(checkpoint_path: Path) -> tuple[Path, str | None]:
    if checkpoint_path.exists():
        return checkpoint_path, None

    if checkpoint_path.name == "last_export.pt":
        fallback_path = checkpoint_path.with_name("best_export.pt")
        if fallback_path.exists():
            return fallback_path, "last_export.pt 없음 -> best_export.pt 사용"

    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
        "먼저 train.py를 실행해 export 체크포인트를 생성하거나 "
        "--checkpoint 경로를 확인해 주세요."
    )


def _ensure_eval_dataset(log_fn: LogFn | None = None) -> Path:
    ds_dir = _resolve_project_path(Path(DATASET_DIR))
    ds_dir.mkdir(parents=True, exist_ok=True)

    eval_path = ds_dir / f"{BINANCE_SYMBOL}_{EVAL_DATASET_START}_{EVAL_DATASET_END}.h5"
    if is_legacy_hdf5(str(eval_path)):
        if log_fn is not None:
            log_fn(f"구 스키마 감지 -> 삭제: {eval_path}")
        eval_path.unlink()

    if not eval_path.exists():
        if log_fn is not None:
            log_fn(f"평가 데이터셋 생성 시작 │ {eval_path}")
        build_dataset_pipeline(
            EVAL_DATASET_START,
            EVAL_DATASET_END,
            str(eval_path),
            log_fn=log_fn,
        )
    elif log_fn is not None:
        log_fn(f"평가 데이터셋 재사용 │ {eval_path}")

    return eval_path


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[QuarterNet, int]:
    _log(f"모델 로드 시작 │ {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError("지원하지 않는 체크포인트 형식입니다.")
    if {"model", "optimizer", "scheduler", "ema"}.issubset(state_dict.keys()):
        raise ValueError(
            "학습용 전체 체크포인트를 지정했습니다. "
            "eval.py에는 last_export.pt 또는 best_export.pt를 사용해 주세요."
        )

    model = QuarterNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    _log(f"모델 로드 완료 │ 파라미터 {n_params:,}")
    return model, n_params


def _sample_indices(
    dataset_len: int,
    n_samples: int,
    seed: int | None,
) -> torch.Tensor:
    if dataset_len <= 0:
        raise ValueError("eval_dataset이 비어 있습니다.")

    sample_count = min(n_samples, dataset_len)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.randperm(dataset_len, generator=generator)[:sample_count]


def _is_upward(log_returns: torch.Tensor) -> torch.Tensor:
    # C_95 / O_0 = exp(sum(lnCO_0..95)) 이므로 부호 비교만 하면 된다.
    return log_returns[..., 0].sum(dim=-1) > 0


def _log_banner(
    device: torch.device,
    checkpoint_path: Path,
    eval_path: Path,
    dataset: QuarterDataset,
    sample_count: int,
    batch_size: int,
    seed: int | None,
    n_params: int,
    fallback_note: str | None,
) -> None:
    sep = "═" * 58
    banner = (
        f"\n{sep}\n"
        f"  QuarterNet 평가\n"
        f"{sep}\n"
        f"  시각        │ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  디바이스    │ {_device_name(device)}\n"
        f"  체크포인트  │ {checkpoint_path.name}\n"
        f"  경로        │ {checkpoint_path}\n"
        f"  Fallback    │ {fallback_note or '없음'}\n"
        f"  데이터셋    │ {eval_path}\n"
        f"  전체 샘플   │ {len(dataset):,}\n"
        f"  평가 샘플   │ {sample_count:,}\n"
        f"  배치 크기   │ {batch_size}\n"
        f"  시드        │ {seed if seed is not None else '없음'}\n"
        f"  파라미터    │ {n_params:,}\n"
        f"  입력 길이   │ {dataset.seq_len:,}\n"
        f"  타깃 길이   │ {dataset.target_len:,}\n"
        f"  스트라이드  │ {dataset.stride}\n"
        f"{sep}"
    )
    for line in banner.splitlines():
        logger.info(line)


def _log_result(
    checkpoint_path: Path,
    correct: int,
    total: int,
    elapsed: float,
) -> None:
    accuracy = 100.0 * correct / max(total, 1)
    header = "── 방향성 평가 결과 "
    header += "─" * max(0, 54 - len(header))
    _log(header)
    _log(f"  Accuracy    │ {accuracy:.2f}%")
    _log(f"  Correct     │ {correct:,}/{total:,}")
    _log(f"  Checkpoint  │ {checkpoint_path}")
    _log(f"  Elapsed     │ {elapsed:.2f}s")
    _log("─" * 54)


def _evaluate_direction_accuracy(
    model: QuarterNet,
    dataset: QuarterDataset,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[int, int, float]:
    total = int(indices.numel())
    correct = 0
    total_batches = (total + batch_size - 1) // batch_size
    log_interval = max(1, total_batches // PROGRESS_LOG_STEPS)
    eval_start = time.time()

    _log(
        f"[평가] 시작 │ 샘플 {total:,} │ 배치 {total_batches:,}개 │ "
        f"batch_size {batch_size}"
    )

    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
            batch_indices = indices[start : start + batch_size].tolist()
            pairs = [dataset[idx] for idx in batch_indices]

            x = torch.stack([pair[0] for pair in pairs]).float().to(device)
            y = torch.stack([pair[1] for pair in pairs]).float().to(device)

            pred = model(x)
            pred_dir = _is_upward(pred)
            true_dir = _is_upward(y)

            correct += int((pred_dir == true_dir).sum().item())
            processed = min(start + len(batch_indices), total)

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                elapsed = time.time() - eval_start
                speed = processed / max(elapsed, 1e-6)
                eta = (total - processed) / max(speed, 1e-6)
                running_acc = 100.0 * correct / max(processed, 1)
                _log(
                    f"[평가] 배치 {batch_idx:,}/{total_batches:,} │ "
                    f"샘플 {processed:,}/{total:,} │ "
                    f"정확도 {running_acc:.2f}% │ "
                    f"{speed:.2f} sample/s │ ETA {eta:.1f}s"
                )

    elapsed = time.time() - eval_start
    return correct, total, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="QuarterNet 방향성 평가")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(CHECKPOINT_DIR) / "last_export.pt",
        help="EMA 추론용 가중치 파일 경로 (기본: last_export.pt, 없으면 best_export.pt 사용)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="평가에 사용할 랜덤 샘플 수",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="추론 배치 크기",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="샘플링 재현용 시드",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="예: cuda, cuda:0, cpu",
    )
    args = parser.parse_args()

    _setup_logging()
    dataset: QuarterDataset | None = None

    try:
        if args.samples <= 0:
            raise ValueError("--samples는 1 이상이어야 합니다.")
        if args.batch_size <= 0:
            raise ValueError("--batch-size는 1 이상이어야 합니다.")

        _log("평가 스크립트 시작")

        device = _resolve_device(args.device)
        requested_checkpoint = _resolve_project_path(args.checkpoint)
        checkpoint_path, fallback_note = _resolve_checkpoint_path(requested_checkpoint)

        if fallback_note is not None:
            _log(f"체크포인트 대체 │ {fallback_note} │ {checkpoint_path}")
        else:
            _log(f"체크포인트 확인 │ {checkpoint_path}")

        eval_path = _ensure_eval_dataset(log_fn=_log)
        dataset = QuarterDataset(str(eval_path))
        _log(
            f"평가 데이터 로드 완료 │ {len(dataset):,} 샘플 │ "
            f"seq {dataset.seq_len:,} │ target {dataset.target_len:,} │ "
            f"stride {dataset.stride}"
        )

        indices = _sample_indices(len(dataset), args.samples, args.seed)
        sample_count = int(indices.numel())
        _log(
            f"샘플링 완료 │ 요청 {args.samples:,} │ 사용 {sample_count:,} │ "
            f"seed {args.seed if args.seed is not None else '없음'}"
        )

        model, n_params = _load_model(checkpoint_path, device)
        _log_banner(
            device=device,
            checkpoint_path=checkpoint_path,
            eval_path=eval_path,
            dataset=dataset,
            sample_count=sample_count,
            batch_size=args.batch_size,
            seed=args.seed,
            n_params=n_params,
            fallback_note=fallback_note,
        )

        correct, total, elapsed = _evaluate_direction_accuracy(
            model=model,
            dataset=dataset,
            indices=indices,
            device=device,
            batch_size=args.batch_size,
        )
        _log_result(checkpoint_path, correct, total, elapsed)
    except KeyboardInterrupt:
        _log("평가 중단 │ 사용자 요청으로 중단되었습니다.")
        raise SystemExit(130)
    except (FileNotFoundError, ValueError) as exc:
        _log(f"평가 실패 │ {exc}")
        raise SystemExit(1)
    except Exception:
        logger.exception("예상치 못한 오류로 평가가 실패했습니다.")
        raise SystemExit(1)
    finally:
        if dataset is not None:
            dataset.close()
            _log("평가 데이터셋 닫기 완료")


if __name__ == "__main__":
    main()
