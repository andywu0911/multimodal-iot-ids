"""L3 事件聚合 CLI。python scripts/build_events.py --subset S3 --checkpoint models/S3_fpc_gmu.pt

讀已訓練模型（scripts/train.py --checkpoint）做推論，依裝置聚合成事件
（src/events/aggregate.py，idle/active timeout 邊界），同時用真實標籤聚合一份
對照組，比較兩者的事件層級一致性。不重新訓練：mps 非決定性會讓「同設定」
重訓出不同模型，論文數字一律來自同一份 checkpoint。

--splits 決定推論範圍，輸出檔名帶上這個 scope：
  test（預設，論文引用這組；屬最終評估，評估流程固定後才跑）
  all  train+val+test 合併，屬 in-sample，只當 L4 報告的輸出樣貌展示。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from src.events.aggregate import aggregate_events, match_events
from src.features.dataset import build_splits
from src.labels import L1_CLASSES
from src.models.train import load_checkpoint, predict, to_tensors
from src.provenance import snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", choices=["test", "all"], default="test")
    args = ap.parse_args()

    split_names = ("test",) if args.splits == "test" else ("train", "val", "test")
    ckpt = load_checkpoint(args.checkpoint)
    splits = build_splits(args.subset)
    manifests, preds = [], []
    for name in split_names:
        out = predict(ckpt["model"], to_tensors(splits[name], ckpt["scalers"], ckpt["device"]))
        manifests.append(splits[name]["manifest"])
        preds.append(out["tactic_prob"].argmax(axis=1))

    manifest = pd.concat(manifests, ignore_index=True)
    pred_ids = np.concatenate(preds)
    manifest["pred_l1_label"] = [L1_CLASSES[i] for i in pred_ids]

    pred_events = aggregate_events(manifest, label_col="pred_l1_label")
    true_events = aggregate_events(manifest, label_col="l1_label")

    cfg = ckpt["config"]
    print(f"[build_events] subset={args.subset} model={cfg['modalities']}/{cfg['fusion']} splits={args.splits}")
    print(f"  flows(labeled)={len(manifest)}  pred_events={len(pred_events)}  true_events={len(true_events)}")
    print(f"  devices w/ predicted events: {sorted(pred_events['device'].unique())}")

    # 事件層級一致性：邊界各自獨立（各自的 idle/active timeout），改用時間
    # 區間重疊配對，而非固定窗那版的 (device, window_start) 精確匹配。
    matched_df = match_events(true_events, pred_events)
    n_true = len(true_events)
    n_matched = int((matched_df["outcome"] == "matched").sum())
    n_mismatched = int((matched_df["outcome"] == "mismatched").sum())
    n_missed = int((matched_df["outcome"] == "missed").sum())
    n_false = int((matched_df["outcome"] == "false_alarm").sum())
    print(f"  event-level severity exact-match={n_matched}/{len(matched_df)}  "
          f"mismatched(重疊但嚴重度不同)={n_mismatched}  "
          f"missed(true有pred無)={n_missed}  false_alarm(true無pred有)={n_false}  n_true_events={n_true}")

    # 指標寫進 results/，論文不必再從 stdout 抄數字。
    metrics = {
        "experiment": "L3_event_aggregation",
        "settings": {"subset": args.subset, "modalities": cfg["modalities"], "fusion": cfg["fusion"],
                      "splits": args.splits, "checkpoint": args.checkpoint,
                      "idle_timeout": float(config.IDLE_TIMEOUT),
                      "active_timeout": float(config.ACTIVE_TIMEOUT),
                      "in_sample": args.splits == "all"},
        "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"),
        "n_flows_labeled": len(manifest),
        "n_true_events": n_true, "n_pred_events": len(pred_events),
        "severity_exact_match": n_matched, "n_windows_union": len(matched_df),
        "mismatched": n_mismatched, "missed": n_missed, "false_alarm": n_false,
    }
    metrics_path = config.RESULTS_DIR / f"l3_events_{args.subset}_{args.splits}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[build_events] metrics -> {metrics_path}")

    out_dir = config.DATA_PROCESSED / "events" / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_events.to_json(out_dir / f"pred_events_{args.splits}.json", orient="records", indent=2)
    true_events.to_json(out_dir / f"true_events_{args.splits}.json", orient="records", indent=2)
    print(f"[build_events] -> {out_dir}/{{pred,true}}_events_{args.splits}.json")


if __name__ == "__main__":
    main()
