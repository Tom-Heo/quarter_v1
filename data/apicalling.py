from __future__ import annotations

import time

import pandas as pd
import requests

from config import (
    BINANCE_DATA_LIMIT,
    BINANCE_DATA_SLEEP,
    BINANCE_FUNDING_LIMIT,
    BINANCE_INTERVAL,
    BINANCE_KLINE_LIMIT,
    BINANCE_PAIR,
    BINANCE_SLEEP,
    BINANCE_SYMBOL,
)


class BinanceFetcher:
    FAPI = "https://fapi.binance.com"
    DATA = f"{FAPI}/futures/data"
    _MAX_RETRIES = 5

    def _request_with_retry(self, url: str, params: dict) -> list:
        for attempt in range(self._MAX_RETRIES):
            resp = requests.get(url, params=params)
            if resp.status_code in (418, 429):
                wait = int(resp.headers.get("Retry-After", 2**attempt * 5))
                print(f"\n  [rate-limit {resp.status_code}] {wait}s 대기 후 재시도 ({attempt + 1}/{self._MAX_RETRIES})")
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                print(f"\n  [400 Bad Request] 해당 구간 데이터 없음 — 건너뜀")
                return []
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()

    def fetch_klines(
        self,
        start_ms: int,
        end_ms: int,
        symbol: str = BINANCE_SYMBOL,
        interval: str = BINANCE_INTERVAL,
        label: str = "",
    ) -> pd.DataFrame:
        rows: list = []
        origin_start = start_ms
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": BINANCE_KLINE_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        while True:
            data = self._request_with_retry(f"{self.FAPI}/fapi/v1/klines", params)
            if not data:
                break
            rows.extend(data)
            next_start = data[-1][0] + 1
            if label:
                pct = min((next_start - origin_start) / max(end_ms - origin_start, 1) * 100, 100.0)
                print(f"\r  {label} {pct:5.1f}% ({len(rows):,}건)", end="", flush=True)
            if next_start > end_ms:
                break
            params["startTime"] = next_start
            time.sleep(BINANCE_SLEEP)
        if label:
            print(f"\r  {label} 100.0% ({len(rows):,}건)", flush=True)

        cols = [
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_vol", "trades",
            "taker_buy_vol", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ("open", "high", "low", "close", "volume", "trades", "taker_buy_vol"):
            df[c] = df[c].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_vol"]]

    def fetch_funding_rate(
        self,
        start_ms: int,
        end_ms: int,
        symbol: str = BINANCE_SYMBOL,
        label: str = "",
    ) -> pd.DataFrame:
        rows: list = []
        origin_start = start_ms
        params = {
            "symbol": symbol,
            "limit": BINANCE_FUNDING_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        while True:
            data = self._request_with_retry(f"{self.FAPI}/fapi/v1/fundingRate", params)
            if not data:
                break
            rows.extend(data)
            next_start = data[-1]["fundingTime"] + 1
            if label:
                pct = min((next_start - origin_start) / max(end_ms - origin_start, 1) * 100, 100.0)
                print(f"\r  {label} {pct:5.1f}% ({len(rows):,}건)", end="", flush=True)
            if next_start > end_ms:
                break
            params["startTime"] = next_start
            time.sleep(BINANCE_SLEEP)
        if label:
            print(f"\r  {label} 100.0% ({len(rows):,}건)", flush=True)

        if not rows:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        return df[["timestamp", "funding_rate"]]

    def _fetch_data_endpoint(
        self,
        url: str,
        value_col: str,
        out_col: str,
        start_ms: int,
        end_ms: int,
        extra_params: dict | None = None,
        use_symbol: bool = True,
        label: str = "",
    ) -> pd.DataFrame:
        rows: list = []
        origin_start = start_ms
        params = {
            "period": BINANCE_INTERVAL,
            "limit": BINANCE_DATA_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        if use_symbol:
            params["symbol"] = BINANCE_SYMBOL
        if extra_params:
            params.update(extra_params)

        while True:
            data = self._request_with_retry(url, params)
            if not data:
                break
            rows.extend(data)
            next_start = data[-1]["timestamp"] + 1
            if label:
                pct = min((next_start - origin_start) / max(end_ms - origin_start, 1) * 100, 100.0)
                print(f"\r  {label} {pct:5.1f}% ({len(rows):,}건)", end="", flush=True)
            if next_start > end_ms:
                break
            params["startTime"] = next_start
            time.sleep(BINANCE_DATA_SLEEP)
        if label:
            print(f"\r  {label} 100.0% ({len(rows):,}건)", flush=True)

        if not rows:
            return pd.DataFrame(columns=["timestamp", out_col])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df[out_col] = df[value_col].astype(float)
        return df[["timestamp", out_col]]

    def fetch_basis(self, start_ms: int, end_ms: int, label: str = "") -> pd.DataFrame:
        return self._fetch_data_endpoint(
            url=f"{self.DATA}/basis",
            value_col="basisRate",
            out_col="basis",
            start_ms=start_ms,
            end_ms=end_ms,
            extra_params={
                "pair": BINANCE_PAIR,
                "contractType": "CURRENT_QUARTER",
            },
            use_symbol=False,
            label=label,
        )

    def fetch_open_interest(self, start_ms: int, end_ms: int, label: str = "") -> pd.DataFrame:
        return self._fetch_data_endpoint(
            url=f"{self.DATA}/openInterestHist",
            value_col="sumOpenInterest",
            out_col="open_interest",
            start_ms=start_ms,
            end_ms=end_ms,
            label=label,
        )

    def fetch_long_short_ratio(self, start_ms: int, end_ms: int, label: str = "") -> pd.DataFrame:
        return self._fetch_data_endpoint(
            url=f"{self.DATA}/globalLongShortAccountRatio",
            value_col="longShortRatio",
            out_col="long_short_ratio",
            start_ms=start_ms,
            end_ms=end_ms,
            label=label,
        )

