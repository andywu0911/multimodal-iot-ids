"""E2 對照組：Kitsune 的 KitNET 異常偵測（只比偵測層）。
python scripts/kitsune_baseline.py --subset S3 [--evaluate-test]

只用 train 的良性窗訓練（無監督），異常分數 = 輸出層 RMSE。門檻取 val 良性窗分數的
第 99 百分位（目標 FPR 約 1%），不看 val 的攻擊窗，也不看 test。
報 ROC-AUC、PR-AUC（不依門檻），與門檻下的 precision / recall / F1 / FPR。
超參數沿用原論文（每群至多 10 維、隱藏層比例 0.75），不調。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

import config
from src.features.dataset import build_splits
from src.models.baselines import KitNET
from src.provenance import snapshot

THRESHOLD_QUANTILE = 0.99


def _x(split: dict) -> np.ndarray:
    return np.hstack([split["f_stats"], split["c"]]).astype(np.float64)


def _metrics(scores: np.ndarray, y_det: np.ndarray, threshold: float) -> dict:
    pred = (scores > threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_det, pred, average="binary", zero_division=0)
    return {"roc_auc": float(roc_auc_score(y_det, scores)), "pr_auc": float(average_precision_score(y_det, scores)),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "fpr": float(pred[y_det == 0].mean()), "n_benign": int((y_det == 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.EVAL_SEEDS))
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    s = build_splits(args.subset)
    tr, va, te = s["train"], s["val"], s["test"]
    X_benign = _x(tr)[tr["y_det"] == 0]
    runs = []
    for seed in args.seeds:
        model = KitNET(seed=seed).fit(X_benign)
        val_scores = model.score(_x(va))
        threshold = float(np.quantile(val_scores[va["y_det"] == 0], THRESHOLD_QUANTILE))
        run = {"seed": seed, "n_clusters": len(model.groups), "threshold": threshold,
               "val": _metrics(val_scores, va["y_det"], threshold), "test": None}
        if args.evaluate_test:
            run["test"] = _metrics(model.score(_x(te)), te["y_det"], threshold)
        runs.append(run)
        msg = f"[kitsune] seed={seed} clusters={run['n_clusters']} val AUC={run['val']['roc_auc']:.4f} " \
              f"F1={run['val']['f1']:.4f} FPR={run['val']['fpr']:.4f}"
        if run["test"]:
            msg += f" | test AUC={run['test']['roc_auc']:.4f} F1={run['test']['f1']:.4f} FPR={run['test']['fpr']:.4f}"
        print(msg, flush=True)

    out = {"experiment": "E2_kitsune_baseline", "settings": vars(args),
           "threshold_quantile": THRESHOLD_QUANTILE, "n_train_benign": int(len(X_benign)),
           "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"), "runs": runs}
    path = config.RESULTS_DIR / f"kitsune_baseline_{args.subset}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[kitsune] -> {path}")


if __name__ == "__main__":
    main()
