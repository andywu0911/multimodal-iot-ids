"""觀測窗下採樣 + 主機情境。CLI: python scripts/sample_flows.py --subset S3

讀 build_flows.py 的全量 <subset>.parquet：
1. 每條連線至多 WINDOW_CAP_PER_CONN 個窗（src/pcap/sample.py）
2. 每個原始標籤至多 MAX_FLOWS_PER_RAW_LABEL 個窗
3. 對抽中的窗，用「全量」歷史計算主機情境（src/pcap/context.py），存成 ctx_* 欄位
輸出 <subset>_sampled.parquet。全量版不動，換上限只需重跑此腳本。
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from src.pcap.context import compute_host_context
from src.pcap.sample import cap_windows_per_connection, subsample_by_raw_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.SUBSETS))
    ap.add_argument("--max-per-label", type=int, default=config.MAX_FLOWS_PER_RAW_LABEL)
    ap.add_argument("--window-cap", type=int, default=config.WINDOW_CAP_PER_CONN)
    args = ap.parse_args()

    in_path = config.FLOW_DIR / f"{args.subset}.parquet"
    out_path = config.FLOW_DIR / f"{args.subset}_sampled.parquet"
    t0 = time.time()

    df = pd.read_parquet(in_path)
    print(f"[sample_flows] input windows={len(df)}  connections={df['conn_id'].nunique()}")

    capped = cap_windows_per_connection(df, cap=args.window_cap)
    sampled = subsample_by_raw_label(capped, max_per_label=args.max_per_label)
    print(f"[sample_flows] after window cap={len(capped)}  after label cap={len(sampled)}")

    ctx = compute_host_context(df, sampled["flow_id"])
    sampled = sampled.join(ctx.add_prefix("ctx_"), on="flow_id")
    sampled.to_parquet(out_path, index=False)

    print(f"[sample_flows] output windows={len(sampled)}  ({time.time() - t0:.0f}s) -> {out_path}")
    print("\n[raw_label: windows / connections]")
    print(sampled.groupby("raw_label", dropna=False)
          .agg(windows=("flow_id", "size"), connections=("conn_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
