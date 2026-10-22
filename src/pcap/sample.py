"""Flow 層抽樣。共用邏輯統一在此，禁止各腳本各自實作。

抽樣策略：每個原始標籤（L2）設 flow 數上限，用單一規則同時涵蓋「多數類設
上限、稀有類 100% 保留」——稀有類本來就低於上限，套用同一規則後原樣通過，
不需要為多數類/稀有類寫兩套邏輯。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def subsample_by_raw_label(
    df: pd.DataFrame,
    max_per_label: int = config.MAX_FLOWS_PER_RAW_LABEL,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """對每個 raw_label（含 None／未對到標籤的 flow）各自抽樣到至多 max_per_label 條。"""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("raw_label", dropna=False, sort=False):
        if len(group) > max_per_label:
            keep = rng.choice(group.index.to_numpy(), size=max_per_label, replace=False)
            parts.append(group.loc[np.sort(keep)])
        else:
            parts.append(group)
    return pd.concat(parts).sort_index().reset_index(drop=True)


def cap_windows_per_connection(
    df: pd.DataFrame,
    cap: int = config.WINDOW_CAP_PER_CONN,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """每條連線（conn_id）至多保留 cap 個觀測窗，超過時均勻隨機抽。

    長連線（如 CoAP Amplification 的單一連線有 1,375 個窗）若不設上限，會讓
    一條連線主導整個類別的樣本與評估指標；上限讓樣本數更貼近獨立事件的尺度。
    必須在 subsample_by_raw_label 之前套用。
    """
    rng = np.random.default_rng(seed)
    sizes = df.groupby("conn_id")["conn_id"].transform("size").to_numpy()
    keep = sizes <= cap
    for _, idx in df.index[~keep].to_series().groupby(df.loc[~keep, "conn_id"]):
        chosen = rng.choice(idx.to_numpy(), size=cap, replace=False)
        keep[df.index.get_indexer(chosen)] = True
    return df[keep].reset_index(drop=True)
