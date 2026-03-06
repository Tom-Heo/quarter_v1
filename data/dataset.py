from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    CLIP_BOUND,
    EPSILON,
    FEATURES,
    NUM_FEATURES,
    NUM_TARGET_FEATURES,
    SEQ_LEN,
    STRIDE,
    TARGET_FEATURES,
    TARGET_LEN,
)


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
) -> tuple[np.ndarray, np.ndarray]:
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    V = df["volume"].values
    T = df["trades"].values
    TBV = df["taker_buy_vol"].values

    lnHO = np.log(H / O)
    lnLO = np.log(L / O)
    lnCO = np.log(C / O)
    lnCH = np.log(C / H)
    lnCL = np.log(C / L)
    lnHL = np.log(H / L)

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

    targets = np.column_stack([lnCO, lnHO, lnLO, lnCH, lnCL])

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
) -> None:
    T = features.shape[0]

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    features = (features - mean) / std

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


def build_dataset_pipeline(start_date: str, end_date: str, out_path: str) -> None:
    from .apicalling import BinanceFetcher

    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)

    fetcher = BinanceFetcher()

    print(f"[1/3] klines 수집 중 ({start_date} ~ {end_date})")
    klines = fetcher.fetch_klines(start_ms, end_ms, label="klines")

    print(f"[2/3] 펀딩비 수집 중")
    funding = fetcher.fetch_funding_rate(start_ms, end_ms, label="펀딩비")

    print(f"[3/3] 베이시스 수집 중")
    basis = fetcher.fetch_basis(start_ms, end_ms, label="베이시스")

    print("데이터 정렬 중 ...")
    df = build_aligned_df(klines, funding, basis)
    print(f"  정렬 완료: {len(df):,}행")

    print("피처 엔지니어링 중 ...")
    features, targets = compute_features(df)
    print(f"  피처 {features.shape}, 타겟 {targets.shape}")

    print(f"HDF5 저장 중 → {out_path}")
    create_hdf5(features, targets, out_path)
    print("데이터셋 빌드 완료.")


def is_legacy_hdf5(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with h5py.File(str(p), "r") as f:
        if "features" not in f:
            return True
        stored = list(f.attrs.get("features", []))
        return stored != FEATURES


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
