"""建構三模態特徵產物。CLI: python scripts/build_features.py --subset S3

讀 <subset>_split.parquet，輸出到 data/processed/features/<subset>/（產物清單見
src/features/dataset.py）。下游用列索引對齊即可，不需要每次重新 join。
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from src.features.dataset import build_feature_artifacts, save_feature_artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.SUBSETS))
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(config.FLOW_DIR / f"{args.subset}_split.parquet").reset_index(drop=True)
    art = build_feature_artifacts(df)
    out_dir = config.FEATURES_DIR / args.subset
    save_feature_artifacts(art, out_dir)
    for k, v in art.items():
        print(f"[build_features] {k}: {getattr(v, 'shape', None)}")
    print(f"[build_features] -> {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
