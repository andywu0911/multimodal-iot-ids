"""E2：捷徑 probe CLI。python scripts/e2_shortcut_probe.py --subset S2

讀 split_flows.py 產出的 <subset>_split.parquet，對 L1 攻擊類別層標籤分別跑
IP / MAC / 埠號 / checksum（對照組）四組 probe，並與多數類基準對照。
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config
from src.audit.shortcut_probe import run_all_probes
from src.provenance import snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.SUBSETS))
    ap.add_argument("--label", default="l1_label", choices=["l1_label", "l2_label"])
    args = ap.parse_args()

    in_path = config.FLOW_DIR / f"{args.subset}_split.parquet"
    df = pd.read_parquet(in_path)

    results = run_all_probes(df, label_col=args.label)

    print(f"[e2_shortcut_probe] subset={args.subset} label={args.label}\n")
    header = f"{'probe':18s} {'accuracy':>10s} {'macro_f1':>10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['probe']:18s} {r['accuracy']:10.4f} {r['macro_f1']:10.4f}")

    out_path = config.RESULTS_DIR / f"e2_shortcut_probe_{args.subset}_{args.label}.json"
    out_path.write_text(json.dumps({
        "experiment": "E2_shortcut_probe",
        "subset": args.subset,
        "label_col": args.label,
        "seed": config.RANDOM_SEED,
        "provenance": snapshot(in_path),
        "results": results,
    }, indent=2, default=str))
    print(f"\n[e2_shortcut_probe] -> {out_path}")


if __name__ == "__main__":
    main()
