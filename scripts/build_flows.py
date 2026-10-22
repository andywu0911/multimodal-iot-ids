"""pcap -> flow parquet。CLI: python scripts/build_flows.py --subset S1

一次性全量解析，之後所有實驗只吃 config.FLOW_DIR 底下的 parquet
（30 GB 的 pcap 不再碰）。輸出檔名：<subset>.parquet。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config
from src.pcap.flow import add_window_index, build_flows_for_device_scenario, find_pcaps
from src.pcap.labeller import PacketLabeller


def devices_for_subset(subset: str) -> list[str]:
    devices = config.SUBSETS[subset]["devices"]
    if devices is not None:
        return devices
    # S3: 全部 78 台，從 benign pcap 目錄枚舉
    import glob
    benign_dir = config.PCAP_ROOT / config.SCENARIOS["benign"]
    names = []
    for p in sorted(glob.glob(str(benign_dir / f"{config.PCAP_PREFIX}*_0-0_to_*.pcap"))):
        stem = Path(p).stem
        dev = stem[len(config.PCAP_PREFIX):].split("_0-0_to_")[0]
        names.append(dev)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=list(config.SUBSETS))
    args = ap.parse_args()

    subset = config.SUBSETS[args.subset]
    devices = devices_for_subset(args.subset)
    scenarios = subset["scenarios"]

    print(f"[build_flows] subset={args.subset} devices={len(devices)} scenarios={scenarios}")

    all_rows: list[dict] = []
    evictions: list[dict] = []
    total_parse_errors = total_non_ipv4 = 0
    t0 = time.time()
    for device in devices:
        labeller = PacketLabeller.for_device(device)
        for scenario in scenarios:
            if not find_pcaps(device, scenario):
                continue
            rows, eviction_events, parse_stats = build_flows_for_device_scenario(
                device, scenario, labeller=labeller)
            flag = f"  ⚠ eviction x{eviction_events}" if eviction_events else ""
            print(f"  {device:22s} {scenario:20s} flows={len(rows):7d}{flag}")
            all_rows.extend(rows)
            if eviction_events:
                evictions.append({"device": device, "scenario": scenario, "eviction_events": eviction_events})
            total_parse_errors += parse_stats["n_parse_errors"]
            total_non_ipv4 += parse_stats["n_non_ipv4"]

    df = add_window_index(pd.DataFrame(all_rows))
    out_path = config.FLOW_DIR / f"{args.subset}.parquet"
    df.to_parquet(out_path, index=False)
    dt = time.time() - t0
    print(f"[build_flows] total flows={len(df)}  elapsed={dt:.1f}s  -> {out_path}")

    # MAX_OPEN_FLOWS 保護機制若被觸發，代表某些 flow 被強制提前關閉、統計量
    # 可能失真——不要默默丟掉，記錄下來供論文方法論限制交代。
    eviction_path = config.RESULTS_DIR / f"build_flows_eviction_{args.subset}.json"
    eviction_path.write_text(json.dumps({
        "subset": args.subset, "total_eviction_events": sum(e["eviction_events"] for e in evictions),
        "affected": evictions,
    }, indent=2))
    if evictions:
        print(f"[build_flows] ⚠ MAX_OPEN_FLOWS 保護機制觸發 {sum(e['eviction_events'] for e in evictions)} 次，"
              f"影響 {len(evictions)} 組 (device, scenario)，明細見 {eviction_path}")
    else:
        print(f"[build_flows] 無 eviction 事件 -> {eviction_path}")

    # 解析失敗 / 非 IPv4 封包計數——完整性檢查得到「截斷封包 0、損壞檔 0」，
    # 這是讓那句話有 telemetry 佐證，而不是單純斷言（見 src/pcap/flow.py 的
    # iter_packets stats 參數）。
    parse_stats_path = config.RESULTS_DIR / f"build_flows_parse_stats_{args.subset}.json"
    parse_stats_path.write_text(json.dumps({
        "subset": args.subset,
        "n_parse_errors": total_parse_errors,
        "n_non_ipv4": total_non_ipv4,
    }, indent=2))
    print(f"[build_flows] parse_errors={total_parse_errors} non_ipv4={total_non_ipv4} -> {parse_stats_path}")

    print(f"[build_flows] connections={df['conn_id'].nunique()}  "
          f"max windows/conn={df.groupby('conn_id').size().max()}")
    print("\n[raw_label distribution]")
    print(df["raw_label"].value_counts(dropna=False).to_string())
    print("\n[l1_label distribution]")
    print(df["l1_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
