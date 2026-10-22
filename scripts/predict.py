"""Demo：一條指令從 pcap 產出資安事件通報。

python scripts/predict.py --pcap capture.pcap --checkpoint models/S3_fpc_gmu.pt --out report.md

流程（src/inference.py）：pcap → 觀測窗 → 三模態特徵 → 模型 → L3 事件 → L4 Markdown。
部署情境沒有官方標籤，也不需要；--device 只是報告裡的裝置名稱，不進任何特徵。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.inference import run
from src.models.train import load_checkpoint
from src.report.template import render_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None, help="報告中的裝置名稱，預設取第一個 pcap 的檔名")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--cpu", action="store_true", help="強制用 CPU 推論")
    args = ap.parse_args()

    device_name = args.device or Path(args.pcap[0]).stem
    ckpt = load_checkpoint(args.checkpoint, torch.device("cpu") if args.cpu else None)
    pred, events, t = run(args.pcap, device_name, ckpt)

    report = render_report(events.to_dict("records"), f"{device_name} 資安事件通報（模型預測）")
    Path(args.out).write_text(report)
    total = sum(v for k, v in t.items() if not k.startswith("n_"))
    print(f"[predict] windows={t['n_windows']}  attack windows={int((pred['pred_l1_label'] != 'Benign').sum())}  "
          f"events={len(events)}")
    print(f"[predict] time: parse={t['parse_windows']:.2f}s features={t['features']:.2f}s "
          f"model={t['model']:.2f}s events={t['events']:.2f}s total={total:.2f}s")
    print(f"[predict] -> {args.out}")


if __name__ == "__main__":
    main()
