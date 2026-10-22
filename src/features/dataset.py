"""特徵產物的建構與載入。組 manifest 與三個模態一律走
build_feature_artifacts()，訓練資料一律走 build_splits()，禁止各腳本各自 join。

產物（data/processed/features/<subset>/，列順序彼此對齊）：
  manifest.parquet   觀測窗 id、連線 id、標籤、split、split_mode
  f_stats.parquet    View F 統計向量
  f_seq.npy          View F 封包序列 (n, SEQ_LEN, 9) float32
  f_seq_len.npy      每列實際封包數 (n,) int64
  p.npy              View P token (n, PAYLOAD_BYTES) int16
  c.parquet          View C 主機情境向量
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features.view_c import build_view_c
from src.features.view_f import build_view_f_seq, build_view_f_stats
from src.features.view_p import build_view_p
from src.labels import BENIGN_LABEL, L1_TO_ID

MANIFEST_COLS = ["flow_id", "conn_id", "window_idx", "device", "scenario", "start_ts", "end_ts",
                 "raw_label", "l1_label", "l2_label", "has_unknown_label", "split", "split_mode"]
MODALITIES = ("f", "p", "c")


def build_feature_artifacts(df: pd.DataFrame) -> dict:
    """從含逐封包陣列與 ctx_* 欄位的觀測窗 df 算出全部產物（純記憶體運算，不寫檔）。"""
    seq, seq_len = build_view_f_seq(df)
    return {
        "manifest": df[[c for c in MANIFEST_COLS if c in df.columns]].copy(),
        "f_stats": build_view_f_stats(df),
        "f_seq": seq,
        "f_seq_len": seq_len,
        "p": build_view_p(df),
        "c": build_view_c(df),
    }


def save_feature_artifacts(art: dict, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    art["manifest"].to_parquet(out_dir / "manifest.parquet")
    art["f_stats"].to_parquet(out_dir / "f_stats.parquet")
    art["c"].to_parquet(out_dir / "c.parquet")
    for k in ("f_seq", "f_seq_len", "p"):
        np.save(out_dir / f"{k}.npy", art[k])


def load_feature_artifacts(subset: str) -> dict:
    d = config.FEATURES_DIR / subset
    return {
        "manifest": pd.read_parquet(d / "manifest.parquet"),
        "f_stats": pd.read_parquet(d / "f_stats.parquet").to_numpy(dtype=np.float32),
        "f_seq": np.load(d / "f_seq.npy"),
        "f_seq_len": np.load(d / "f_seq_len.npy"),
        "p": np.load(d / "p.npy"),
        "c": pd.read_parquet(d / "c.parquet").to_numpy(dtype=np.float32),
    }


def build_splits(subset: str, unknown_policy: str = "exclude", art: dict | None = None) -> dict:
    """回傳 {'train'/'val'/'test': {'f_stats','f_seq','f_seq_len','p','c','y','y_det','manifest'}}。

    y 是 L1 六類 id，y_det 是偵測頭的二元標籤（非 Benign = 1）。未對到標籤的窗剔除。
    unknown_policy：'exclude'（預設）剔除 Unknown；'as_benign' 併入 Benign（T2 敏感度）。
    art：可選，直接傳入已載入/當場算好的產物（例如 LODO 重新切分時）。
    """
    art = art if art is not None else load_feature_artifacts(subset)
    manifest = art["manifest"].copy()
    if unknown_policy == "as_benign":
        manifest.loc[manifest["raw_label"] == "Unknown", "l1_label"] = BENIGN_LABEL
    elif unknown_policy != "exclude":
        raise ValueError(f"unknown_policy must be 'exclude' or 'as_benign', got {unknown_policy!r}")

    keep = manifest["l1_label"].notna().to_numpy()
    manifest = manifest[keep].reset_index(drop=True)
    arrays = {k: art[k][keep] for k in ("f_stats", "f_seq", "f_seq_len", "p", "c")}
    y = manifest["l1_label"].map(L1_TO_ID).to_numpy(dtype=np.int64)
    y_det = (manifest["l1_label"] != BENIGN_LABEL).to_numpy(dtype=np.float32)

    splits = {}
    for name in ("train", "val", "test"):
        idx = (manifest["split"] == name).to_numpy()
        splits[name] = {k: v[idx] for k, v in arrays.items()}
        splits[name].update(y=y[idx], y_det=y_det[idx], manifest=manifest[idx].reset_index(drop=True))
    return splits
