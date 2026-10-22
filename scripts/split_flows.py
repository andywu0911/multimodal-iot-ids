"""切分流程 CLI。python scripts/split_flows.py --subset S3

讀 <subset>_sampled.parquet，套用主評估流程 connection_group_split（以連線為單位，
見 src/pcap/split.py），輸出新增 split / split_mode 欄位的 <subset>_split.parquet。
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from src.pcap.split import connection_group_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.SUBSETS))
    args = ap.parse_args()

    in_path = config.FLOW_DIR / f"{args.subset}_sampled.parquet"
    out_path = config.FLOW_DIR / f"{args.subset}_split.parquet"

    out = connection_group_split(pd.read_parquet(in_path))
    out.to_parquet(out_path, index=False)

    print(f"[split_flows] {in_path.name} -> {out_path.name}  n={len(out)}")
    print("\n[raw_label x split]  windows (connections)")
    win = out.groupby(["raw_label", "split"]).size().unstack("split").fillna(0).astype(int)
    con = out.groupby(["raw_label", "split"])["conn_id"].nunique().unstack("split").fillna(0).astype(int)
    cols = ["train", "val", "test"]
    print((win[cols].astype(str) + " (" + con[cols].astype(str) + ")").to_string())
    within = out.loc[out["split_mode"] == "within_connection", "raw_label"].unique()
    print(f"\n[split_mode] within_connection 群組：{sorted(map(str, within)) or '無'}")


if __name__ == "__main__":
    main()
