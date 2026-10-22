"""E5 部署效能量測。python scripts/efficiency.py --checkpoint models/S3_fpc_gmu.pt

量三件事，寫入 results/efficiency_<checkpoint 名>.json：
1. 模型規模：參數量、checkpoint 檔案大小。
2. 模型推論（只含前向運算）：CPU 與 MPS 上，batch=1 的單窗延遲（中位數、p95）
   與 batch 16/64/256/1024 的吞吐量（窗/秒）。須在機器閒置時量（訓練會搶 CPU/GPU）。輸入取自 S3 的特徵產物，不使用任何標籤。
3. 端到端（pcap → 報告前的事件）：各階段耗時、每秒處理封包數與觀測窗數，
   走的是 scripts/predict.py 同一條路徑（src/inference.py）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

import config
from src.features.dataset import load_feature_artifacts
from src.inference import run
from src.models.train import load_checkpoint, to_tensors
from src.pcap.flow import find_pcaps


def _time_model(model, tensors: dict, batch: int, repeats: int) -> list[float]:
    n = tensors["f_stats"].shape[0]
    times = []
    with torch.no_grad():
        for r in range(repeats):
            i = (r * batch) % max(n - batch, 1)
            b = {k: v[i:i + batch] for k, v in tensors.items()}
            t0 = time.perf_counter()
            model(b)
            if tensors["f_stats"].device.type == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--e2e-device", default="air-quality-1")
    ap.add_argument("--e2e-scenario", default="mirai-infection")
    args = ap.parse_args()

    art = load_feature_artifacts(args.subset)
    inputs = {k: art[k][:5000] for k in ("f_stats", "f_seq", "f_seq_len", "p", "c")}
    out = {"platform": {"machine": platform.machine(), "processor": platform.processor(),
                        "torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
           "checkpoint": args.checkpoint,
           "checkpoint_mb": os.path.getsize(args.checkpoint) / 1e6}

    devices = [torch.device("cpu")] + ([torch.device("mps")] if torch.backends.mps.is_available() else [])
    for dev in devices:
        ck = load_checkpoint(args.checkpoint, dev)
        out["n_params"] = int(sum(p.numel() for p in ck["model"].parameters()))
        t = to_tensors(inputs, ck["scalers"], dev)
        _time_model(ck["model"], t, 64, 5)   # 暖機
        lat = np.array(_time_model(ck["model"], t, 1, 300)) * 1000
        sweep = {bs: float(bs / np.median(_time_model(ck["model"], t, bs, 20))) for bs in (16, 64, 256, 1024)}
        out[f"model_{dev.type}"] = {"latency_ms_median": float(np.median(lat)),
                                    "latency_ms_p95": float(np.percentile(lat, 95)),
                                    "throughput_windows_per_s_by_batch": sweep}
        print(f"[efficiency] {dev.type}: {out[f'model_{dev.type}']}")

    pcaps = find_pcaps(args.e2e_device, args.e2e_scenario)
    ck = load_checkpoint(args.checkpoint, torch.device("cpu"))
    pred, events, t = run(pcaps, args.e2e_device, ck)
    stages = {k: v for k, v in t.items() if not k.startswith("n_")}
    total, n_pkts = sum(stages.values()), t["n_packets"]
    out["end_to_end_cpu"] = {"pcaps": [Path(p).name for p in pcaps], "n_packets": n_pkts,
                             "n_windows": t["n_windows"], "n_events": int(len(events)),
                             "stage_seconds": stages,
                             "total_seconds": total, "packets_per_s": n_pkts / total,
                             "windows_per_s": t["n_windows"] / total}
    print(f"[efficiency] end-to-end: {out['end_to_end_cpu']}")

    path = config.RESULTS_DIR / f"efficiency_{Path(args.checkpoint).stem}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[efficiency] -> {path}")


if __name__ == "__main__":
    main()
