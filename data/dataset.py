from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    CLIP_BOUND,
    DATASET_SCHEMA_VERSION,
    EPSILON,
    FEATURES,
    NUM_FEATURES,
    NUM_TARGET_FEATURES,
    OHLC_LOG_RATIO_CLIP,
    SEQ_LEN,
    STRIDE,
    TARGET_FEATURES,
    TARGET_LEN,
)

LogFn = Callable[[str], None]
EXTREME_ZSCORE_THRESHOLD = 50.0
REPORT_TOP_K = 5


def _emit(log_fn: LogFn | None, msg: str) -> None:
    if log_fn is None:
        print(msg)
    else:
        log_fn(msg)


def _format_row_label(row_labels: pd.Index | None, row_idx: int) -> str:
    if row_labels is None:
        return str(row_idx)
    return str(row_labels[row_idx])


def _clip_logged_values(
    name: str,
    values: np.ndarray,
    clip_abs: float,
    log_fn: LogFn | None = None,
) -> np.ndarray:
    clipped = np.clip(values, -clip_abs, clip_abs)
    clipped_count = int(np.count_nonzero(np.abs(values) > clip_abs))
    if clipped_count > 0:
        _emit(
            log_fn,
            f"[데이터셋] OHLC log-ratio clip │ {name} │ "
            f"clipped {clipped_count:,}/{values.size:,} │ "
            f"raw min {values.min():+.6e} │ raw max {values.max():+.6e}",
        )
    return clipped


def _matrix_finite_summary(name: str, values: np.ndarray) -> str:
    finite_mask = np.isfinite(values)
    finite_count = int(finite_mask.sum())
    total_count = int(values.size)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    summary = (
        f"{name} │ shape={values.shape} │ "
        f"finite {finite_count}/{total_count} │ nan {nan_count} │ inf {inf_count}"
    )
    if finite_count > 0:
        finite_values = values[finite_mask]
        summary += (
            f" │ min {finite_values.min():.6e}"
            f" │ max {finite_values.max():.6e}"
        )
    return summary


def _collect_abs_entries(
    values: np.ndarray,
    columns: list[str],
) -> list[tuple[str, float, float, int]]:
    entries: list[tuple[str, float, float, int]] = []
    for col_idx, col_name in enumerate(columns):
        col = values[:, col_idx]
        finite_idx = np.flatnonzero(np.isfinite(col))
        if finite_idx.size == 0:
            continue
        local_idx = int(np.abs(col[finite_idx]).argmax())
        row_idx = int(finite_idx[local_idx])
        value = float(col[row_idx])
        entries.append((col_name, value, abs(value), row_idx))
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries


def _format_abs_entries(
    entries: list[tuple[str, float, float, int]],
    row_labels: pd.Index | None = None,
    top_k: int = REPORT_TOP_K,
) -> list[str]:
    return [
        (
            f"{col_name} {value:+.6e}"
            f" │ abs {abs_value:.6e}"
            f" │ row {_format_row_label(row_labels, row_idx)}"
        )
        for col_name, value, abs_value, row_idx in entries[:top_k]
    ]


def _collect_nonfinite_entries(
    values: np.ndarray,
    columns: list[str],
) -> list[tuple[str, int, int, int, int]]:
    entries: list[tuple[str, int, int, int, int]] = []
    for col_idx, col_name in enumerate(columns):
        col = values[:, col_idx]
        bad_idx = np.flatnonzero(~np.isfinite(col))
        if bad_idx.size == 0:
            continue
        entries.append(
            (
                col_name,
                int(bad_idx.size),
                int(np.isnan(col).sum()),
                int(np.isinf(col).sum()),
                int(bad_idx[0]),
            )
        )
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries


def _validate_named_vector(
    name: str,
    values: np.ndarray,
    columns: list[str],
    log_fn: LogFn | None = None,
    *,
    positive: bool = False,
) -> None:
    if values.ndim != 1 or values.shape[0] != len(columns):
        raise ValueError(f"{name} shape mismatch: {values.shape} vs {len(columns)} columns")
    if not np.isfinite(values).all():
        bad = [columns[i] for i, value in enumerate(values) if not np.isfinite(value)]
        raise ValueError(f"{name} contains non-finite values: {', '.join(bad[:REPORT_TOP_K])}")
    if positive and np.any(values <= 0):
        bad = [columns[i] for i, value in enumerate(values) if value <= 0]
        raise ValueError(f"{name} must be positive: {', '.join(bad[:REPORT_TOP_K])}")
    _emit(
        log_fn,
        f"[데이터셋] 검증 │ {name} │ min {values.min():.6e} │ max {values.max():.6e}",
    )


def _validate_named_matrix(
    name: str,
    values: np.ndarray,
    columns: list[str],
    log_fn: LogFn | None = None,
    *,
    row_labels: pd.Index | None = None,
    extreme_abs_threshold: float | None = None,
    report_top_abs: bool = False,
) -> None:
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"{name} shape mismatch: {values.shape} vs {len(columns)} columns")

    _emit(log_fn, f"[데이터셋] 검증 │ {_matrix_finite_summary(name, values)}")

    nonfinite_entries = _collect_nonfinite_entries(values, columns)
    if nonfinite_entries:
        reports = [
            (
                f"{col_name} │ bad {bad_count}"
                f" │ nan {nan_count}"
                f" │ inf {inf_count}"
                f" │ first {_format_row_label(row_labels, first_row)}"
            )
            for col_name, bad_count, nan_count, inf_count, first_row in nonfinite_entries[:REPORT_TOP_K]
        ]
        for report in reports:
            _emit(log_fn, f"[데이터셋] 비정상 {name} │ {report}")
        raise ValueError(f"{name} contains non-finite values.")

    abs_entries = _collect_abs_entries(values, columns)
    if report_top_abs and abs_entries:
        _emit(
            log_fn,
            f"[데이터셋] {name} 절댓값 상위 │ "
            + " │ ".join(_format_abs_entries(abs_entries, row_labels=row_labels)),
        )

    if extreme_abs_threshold is None:
        return

    extreme_entries = [
        entry for entry in abs_entries if entry[2] >= extreme_abs_threshold
    ]
    if not extreme_entries:
        return

    _emit(
        log_fn,
        f"[데이터셋] 경고 │ {name} 절댓값이 {extreme_abs_threshold:.1f} 이상인 열이 있습니다.",
    )
    _emit(
        log_fn,
        f"[데이터셋] {name} 극단치 │ "
        + " │ ".join(_format_abs_entries(extreme_entries, row_labels=row_labels)),
    )


def validate_hdf5(path: str, log_fn: LogFn | None = None) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    _emit(log_fn, f"[데이터셋] HDF5 검증 시작 │ {p}")
    with h5py.File(str(p), "r") as f:
        if "features" not in f or "targets" not in f:
            raise ValueError(f"HDF5 schema mismatch: {p}")

        features = f["features"][:]
        targets = f["targets"][:]
        mean = np.asarray(f.attrs["mean"])
        std = np.asarray(f.attrs["std"])
        stored_features = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in f.attrs.get("features", [])
        ]
        stored_targets = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in f.attrs.get("target_features", [])
        ]
        stored_version = int(f.attrs.get("schema_version", 0))

    if stored_features and stored_features != FEATURES:
        raise ValueError(f"HDF5 feature schema mismatch: {p}")
    if stored_targets and stored_targets != TARGET_FEATURES:
        raise ValueError(f"HDF5 target schema mismatch: {p}")
    if stored_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"HDF5 schema version mismatch: {p} "
            f"(stored={stored_version}, expected={DATASET_SCHEMA_VERSION})"
        )

    _validate_named_vector("hdf5.mean", mean, FEATURES, log_fn)
    _validate_named_vector("hdf5.std", std, FEATURES, log_fn, positive=True)
    _validate_named_matrix(
        "hdf5.features",
        features,
        FEATURES,
        log_fn,
        extreme_abs_threshold=EXTREME_ZSCORE_THRESHOLD,
        report_top_abs=True,
    )
    _validate_named_matrix(
        "hdf5.targets",
        targets,
        TARGET_FEATURES,
        log_fn,
        report_top_abs=True,
    )
    _emit(log_fn, f"[데이터셋] HDF5 검증 완료 │ {p}")


def build_aligned_df(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    basis: pd.DataFrame,
) -> pd.DataFrame:
    df = klines.set_index("timestamp")

    for sub_df, col in [
        (funding, "funding_rate"),
        (basis, "basis"),
    ]:
        s = sub_df.set_index("timestamp")[col]
        s = s[~s.index.duplicated(keep="first")]
        df = df.join(s, how="left")

    df = df.ffill()
    df = df.dropna()
    return df


def compute_features(
    df: pd.DataFrame,
    log_fn: LogFn | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    V = df["volume"].values
    T = df["trades"].values
    TBV = df["taker_buy_vol"].values

    # Keep lnCO raw for direction labels, but clip wick-related log ratios used as features.
    with np.errstate(divide="ignore", invalid="ignore"):
        lnHO_raw = np.log(H / O)
        lnLO_raw = np.log(L / O)
        lnCO = np.log(C / O)

    lnCH_raw = lnCO - lnHO_raw
    lnCL_raw = lnCO - lnLO_raw
    lnHL_raw = lnHO_raw - lnLO_raw

    lnHO = _clip_logged_values("lnHO", lnHO_raw, OHLC_LOG_RATIO_CLIP, log_fn)
    lnLO = _clip_logged_values("lnLO", lnLO_raw, OHLC_LOG_RATIO_CLIP, log_fn)
    lnCH = _clip_logged_values("lnCH", lnCH_raw, OHLC_LOG_RATIO_CLIP, log_fn)
    lnCL = _clip_logged_values("lnCL", lnCL_raw, OHLC_LOG_RATIO_CLIP, log_fn)
    lnHL = _clip_logged_values("lnHL", lnHL_raw, OHLC_LOG_RATIO_CLIP, log_fn)

    body = lnCO
    body_safe = np.where(
        np.abs(body) < EPSILON,
        np.sign(body + 1e-12) * EPSILON,
        body,
    )

    f1 = body
    f2 = np.clip(lnHO * lnLO * lnCH * lnCL / body_safe**3, -CLIP_BOUND, CLIP_BOUND)
    f3 = np.clip(lnHO * lnCH / body_safe, -CLIP_BOUND, CLIP_BOUND)
    f4 = np.clip(lnLO * lnCL / body_safe, -CLIP_BOUND, CLIP_BOUND)

    targets = np.column_stack([lnCO, lnHO_raw, lnLO_raw, lnCH_raw, lnCL_raw])

    def _log_return(arr: np.ndarray) -> np.ndarray:
        prev = arr[:-1].copy()
        prev = np.where(np.abs(prev) < EPSILON, EPSILON, prev)
        ratio = arr[1:] / prev
        ratio = np.where(~np.isfinite(ratio) | (ratio <= 0), EPSILON, ratio)
        return np.clip(np.log(ratio), -CLIP_BOUND, CLIP_BOUND)

    log_V = _log_return(V)
    log_T = _log_return(T)
    taker_buy_ratio = np.where(V > EPSILON, TBV / V, 0.5)

    hour = df.index.hour.values.astype(np.float64)
    dow = df.index.dayofweek.values.astype(np.float64)
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_dow = np.sin(2 * np.pi * dow / 7)
    cos_dow = np.cos(2 * np.pi * dow / 7)

    doy = df.index.dayofyear.values.astype(np.float64)
    sin_year = np.sin(2 * np.pi * doy / 365.25)
    cos_year = np.cos(2 * np.pi * doy / 365.25)

    minute = df.index.minute.values.astype(np.float64)
    fund_pos = ((hour * 60 + minute) % 480) / 480.0
    sin_fund = np.sin(2 * np.pi * fund_pos)
    cos_fund = np.cos(2 * np.pi * fund_pos)

    funding = df["funding_rate"].values[1:]
    basis_raw = df["basis"].values[1:]

    f1 = f1[1:]
    f2 = f2[1:]
    f3 = f3[1:]
    f4 = f4[1:]
    lnHO = lnHO[1:]
    lnLO = lnLO[1:]
    lnCH = lnCH[1:]
    lnCL = lnCL[1:]
    lnHL = lnHL[1:]
    taker_buy_ratio = taker_buy_ratio[1:]
    sin_hour = sin_hour[1:]
    cos_hour = cos_hour[1:]
    sin_dow = sin_dow[1:]
    cos_dow = cos_dow[1:]
    sin_year = sin_year[1:]
    cos_year = cos_year[1:]
    sin_fund = sin_fund[1:]
    cos_fund = cos_fund[1:]
    targets = targets[1:]

    features = np.column_stack([
        f1, f2, f3, f4,
        lnHO, lnLO, lnCH, lnCL,
        lnHL,
        log_V,
        log_T,
        taker_buy_ratio,
        funding, basis_raw,
        sin_hour, cos_hour,
        sin_dow, cos_dow,
        sin_year, cos_year,
        sin_fund, cos_fund,
    ]).astype(np.float32)

    assert features.shape[1] == NUM_FEATURES
    assert targets.shape[1] == NUM_TARGET_FEATURES
    return features, targets.astype(np.float32)


def create_hdf5(
    features: np.ndarray,
    targets: np.ndarray,
    path: str,
    seq_len: int = SEQ_LEN,
    target_len: int = TARGET_LEN,
    stride: int = STRIDE,
    log_fn: LogFn | None = None,
    row_labels: pd.Index | None = None,
) -> None:
    T = features.shape[0]

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    features = (features - mean) / std

    _validate_named_vector("feature.mean", mean, FEATURES, log_fn)
    _validate_named_vector("feature.std", std, FEATURES, log_fn, positive=True)
    _validate_named_matrix(
        "normalized.features",
        features,
        FEATURES,
        log_fn,
        row_labels=row_labels,
        extreme_abs_threshold=EXTREME_ZSCORE_THRESHOLD,
        report_top_abs=True,
    )
    _validate_named_matrix(
        "normalized.targets",
        targets,
        TARGET_FEATURES,
        log_fn,
        row_labels=row_labels,
        report_top_abs=True,
    )

    n_samples = (T - seq_len - target_len) // stride + 1
    if n_samples <= 0:
        raise ValueError(
            f"Time series length {T} is shorter than "
            f"seq_len({seq_len}) + target_len({target_len})"
        )

    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features, dtype=np.float32)
        f.create_dataset("targets", data=targets, dtype=np.float32)

        f.attrs["features"] = FEATURES
        f.attrs["target_features"] = TARGET_FEATURES
        f.attrs["schema_version"] = DATASET_SCHEMA_VERSION
        f.attrs["mean"] = mean
        f.attrs["std"] = std
        f.attrs["seq_len"] = seq_len
        f.attrs["target_len"] = target_len
        f.attrs["stride"] = stride
        f.attrs["n_samples"] = n_samples


def _to_ms(date_str: str) -> int:
    from datetime import datetime, timezone

    return int(
        datetime.fromisoformat(date_str)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )


def build_dataset_pipeline(
    start_date: str,
    end_date: str,
    out_path: str,
    log_fn: LogFn | None = None,
) -> None:
    from .apicalling import BinanceFetcher

    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)

    fetcher = BinanceFetcher(log_fn=log_fn)

    _emit(log_fn, f"[데이터셋] [1/3] klines 수집 중 ({start_date} ~ {end_date})")
    klines = fetcher.fetch_klines(start_ms, end_ms, label="klines")

    _emit(log_fn, "[데이터셋] [2/3] 펀딩비 수집 중")
    funding = fetcher.fetch_funding_rate(start_ms, end_ms, label="펀딩비")

    _emit(log_fn, "[데이터셋] [3/3] 베이시스 수집 중")
    basis = fetcher.fetch_basis(start_ms, end_ms, label="베이시스")

    _emit(log_fn, "[데이터셋] 데이터 정렬 중 ...")
    df = build_aligned_df(klines, funding, basis)
    _emit(log_fn, f"[데이터셋] 정렬 완료 │ {len(df):,}행")

    _emit(log_fn, "[데이터셋] 피처 엔지니어링 중 ...")
    features, targets = compute_features(df, log_fn=log_fn)
    _emit(log_fn, f"[데이터셋] 피처 {features.shape} │ 타깃 {targets.shape}")
    row_labels = df.index[1:]
    _validate_named_matrix(
        "raw.features",
        features,
        FEATURES,
        log_fn,
        row_labels=row_labels,
        report_top_abs=True,
    )
    _validate_named_matrix(
        "raw.targets",
        targets,
        TARGET_FEATURES,
        log_fn,
        row_labels=row_labels,
        report_top_abs=True,
    )

    _emit(log_fn, f"[데이터셋] HDF5 저장 중 │ {out_path}")
    create_hdf5(features, targets, out_path, log_fn=log_fn, row_labels=row_labels)
    _emit(log_fn, "[데이터셋] 빌드 완료.")


def is_legacy_hdf5(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with h5py.File(str(p), "r") as f:
        if "features" not in f:
            return True
        stored = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in f.attrs.get("features", [])
        ]
        stored_targets = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in f.attrs.get("target_features", [])
        ]
        stored_version = int(f.attrs.get("schema_version", 0))
        return (
            stored != FEATURES
            or stored_targets != TARGET_FEATURES
            or stored_version != DATASET_SCHEMA_VERSION
        )


class QuarterDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.n_samples = int(f.attrs["n_samples"])
            self.seq_len = int(f.attrs["seq_len"])
            self.target_len = int(f.attrs["target_len"])
            self.stride = int(f.attrs["stride"])
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        start = idx * self.stride
        x = torch.from_numpy(self._h5["features"][start : start + self.seq_len].copy())
        t_start = start + self.seq_len
        y = torch.from_numpy(self._h5["targets"][t_start : t_start + self.target_len].copy())
        return x, y

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


if __name__ == "__main__":
    from config import (
        BINANCE_SYMBOL,
        DATASET_DIR,
        EVAL_DATASET_END,
        EVAL_DATASET_START,
        TRAIN_DATASET_END,
        TRAIN_DATASET_START,
    )

    out_dir = Path(DATASET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{BINANCE_SYMBOL}_{TRAIN_DATASET_START}_{TRAIN_DATASET_END}.h5"
    eval_path = out_dir / f"{BINANCE_SYMBOL}_{EVAL_DATASET_START}_{EVAL_DATASET_END}.h5"

    for p in [train_path, eval_path]:
        if is_legacy_hdf5(str(p)):
            print(f"  구 스키마 감지 → 삭제: {p}")
            p.unlink()

    if not train_path.exists():
        print(f"Building train dataset: {train_path}")
        build_dataset_pipeline(TRAIN_DATASET_START, TRAIN_DATASET_END, str(train_path))

    if not eval_path.exists():
        print(f"Building eval dataset: {eval_path}")
        build_dataset_pipeline(EVAL_DATASET_START, EVAL_DATASET_END, str(eval_path))
