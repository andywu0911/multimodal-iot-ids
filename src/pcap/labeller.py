"""封包級標籤索引：CSV → (timestamp, frame length) → label。

標籤對齊統一在此。official CSV 沒有封包 id，
用「frame.time（微秒精度）+ frame.len」作為 join key —— 在 air-quality-1
上驗證過 55,052 列 CSV 對到 55,070 個 IPv4 封包，match rate 99.97%
（未對到的是罕見的 timestamp+len 完全重複而 CSV 只留первое occurrence 的情況）。

不用五元組 join：GRE 隧道封包的官方 CSV 只保留「外層」ip.src/ip.dst 單一值，
但本專案的 flow builder 用剝殼後的「內層」五元組建 flow，兩者天然對不上。
timestamp+len 與封包內容無關，兩種視角都適用。

CSV 陷阱：第一欄 frame.time 是含逗號的引號字串。pandas.read_csv
本來就照 RFC4180 處理引號，不受影響；陷阱只坑「naive str.split(',')」的手刻解析器。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config

_CSV_COLUMNS = ["frame.time", "frame.len", "label"]


def _parse_ts_us(frame_time: pd.Series) -> pd.Series:
    """'Jan 14, 2025 18:40:22.447710000 GMT' -> 微秒 epoch（int64）。

    pcap 本身是微秒精度（magic d4c3b2a1），frame.time 的奈秒欄位固定補零到
    9 位，只取前 6 位即可還原微秒，不需要奈秒精度的 datetime 解析。
    """
    s = frame_time.str.replace(" GMT", "", regex=False)
    date_part = s.str.split(".").str[0]
    frac_part = s.str.split(".").str[1]
    dt = pd.to_datetime(date_part, format="%b %d, %Y %H:%M:%S")
    sec = dt.astype("int64") // 1_000_000_000
    us = frac_part.str[:6].astype("int64")
    return sec * 1_000_000 + us


def build_label_index(device: str) -> pd.DataFrame:
    """讀取（或建立快取）某裝置的封包級標籤索引，欄位：ts_us, length, label。

    快取到 config.LABEL_INDEX_DIR/<device>.parquet，之後不再重新解析 CSV。
    """
    cache_path = config.LABEL_INDEX_DIR / f"{device}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = config.OFFICIAL_CSV_ROOT / f"{config.PCAP_PREFIX}{device}.csv"
    df = pd.read_csv(csv_path, usecols=_CSV_COLUMNS)
    df = df.rename(columns={"frame.len": "length"})
    df["ts_us"] = _parse_ts_us(df["frame.time"])
    df = df.drop_duplicates(subset=["ts_us", "length"], keep="first")
    out = df[["ts_us", "length", "label"]].reset_index(drop=True)

    out.to_parquet(cache_path, index=False)
    return out


@dataclass
class PacketLabeller:
    """單一裝置的 (ts_us, length) -> label 查表，供 flow builder 逐封包查詢。"""

    device: str
    _index: dict

    @classmethod
    def for_device(cls, device: str) -> "PacketLabeller":
        df = build_label_index(device)
        idx = dict(zip(zip(df["ts_us"], df["length"]), df["label"]))
        return cls(device=device, _index=idx)

    def lookup(self, ts_seconds: float, length: int) -> str | None:
        ts_us = round(ts_seconds * 1_000_000)
        return self._index.get((ts_us, length))
