from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_SAMPLES = 100
DEFAULT_BATCH_SIZE = 1


def _resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    if checkpoint_path.name == "last_export.pt":
        fallback_path = checkpoint_path.with_name("best_export.pt")
        if fallback_path.exists():
            print(
                "last_export.pt가 없어 best_export.pt를 사용합니다: "
                f"{fallback_path}"
            )
            return fallback_path

    raise FileNotFoundError(
        f"체크포인트를 찾을 수 없습니다: {checkpoint_path}\n"
        "먼저 train.py를 실행해 export 체크포인트를 생성하거나 "
        "--checkpoint 경로를 확인해 주세요."
    )


def _ensure_eval_dataset() -> Path:
    ds_dir = _resolve_project_path(Path(DATASET_DIR))
    ds_dir.mkdir(parents=True, exist_ok=True)

    eval_path = ds_dir / f"{BINANCE_SYMBOL}_{EVAL_DATASET_START}_{EVAL_DATASET_END}.h5"
    if is_legacy_hdf5(str(eval_path)):
        print(f"구 스키마 감지 -> 삭제: {eval_path}")
        eval_path.unlink()

    if not eval_path.exists():
        print(f"평가 데이터셋 생성 중: {eval_path}")
        build_dataset_pipeline(EVAL_DATASET_START, EVAL_DATASET_END, str(eval_path))

    return eval_path


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint_path: Path, device: torch.device) -> QuarterNet:
    model = QuarterNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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


def _evaluate_direction_accuracy(
    model: QuarterNet,
    dataset: QuarterDataset,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[int, int]:
    total = int(indices.numel())
    correct = 0

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_indices = indices[start : start + batch_size].tolist()
            pairs = [dataset[idx] for idx in batch_indices]

            x = torch.stack([pair[0] for pair in pairs]).float().to(device)
            y = torch.stack([pair[1] for pair in pairs]).float().to(device)

            pred = model(x)
            pred_dir = _is_upward(pred)
            true_dir = _is_upward(y)

            correct += int((pred_dir == true_dir).sum().item())

    return correct, total


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

    if args.samples <= 0:
        raise ValueError("--samples는 1 이상이어야 합니다.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint_path(_resolve_project_path(args.checkpoint))
    eval_path = _ensure_eval_dataset()
    dataset = QuarterDataset(str(eval_path))

    try:
        indices = _sample_indices(len(dataset), args.samples, args.seed)
        model = _load_model(checkpoint_path, device)
        correct, total = _evaluate_direction_accuracy(
            model=model,
            dataset=dataset,
            indices=indices,
            device=device,
            batch_size=args.batch_size,
        )
    finally:
        dataset.close()

    accuracy = 100.0 * correct / max(total, 1)
    print(f"checkpoint: {checkpoint_path}")
    print(f"eval_dataset: {eval_path}")
    print(f"device: {device}")
    print(f"direction_accuracy: {correct}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
