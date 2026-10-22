"""E4 跨裝置泛化：留一裝置（leave-one-device-out）。
python scripts/lodo_eval.py --subset S3 [--devices ...] [--with-baseline]

8 台攻擊裝置輪流留出當 test，其餘 77 台（含 70 台純良性）當訓練池。每台只評估
「留出裝置有、訓練池也有」的類別：訓練池沒有的類別（如留出 combined-cycle-1 時的
CoAP Amplification、UDP Scan）是零樣本情境，另列不併入 macro F1。

LODO 是跨裝置的外部效度檢驗，不是調參用的：模型設定沿用主評估流程固定後的版本，
每台裝置都是一次最終評估。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import config
from src.features.dataset import build_splits, load_feature_artifacts
from src.labels import L1_CLASSES, L1_TO_ID
from src.models.train import class_weights, predict, to_tensors, train_and_evaluate
from src.pcap.split import device_holdout_split
from src.provenance import snapshot


def _common_macro_f1(y, pred, train_classes) -> tuple[float, list[str]]:
    common = sorted(set(np.unique(y)) & set(train_classes))
    return float(f1_score(y, pred, labels=common, average="macro", zero_division=0)), [L1_CLASSES[c] for c in common]


def _per_class(y, pred) -> dict:
    out = {}
    for c in np.unique(y):
        m = y == c
        out[L1_CLASSES[c]] = {"n": int(m.sum()), "recall": float((pred[m] == c).mean()),
                              "top_confusion": L1_CLASSES[int(np.bincount(pred[m], minlength=6).argmax())]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--modalities", default="fpc")
    ap.add_argument("--fusion", default="gmu")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--devices", nargs="+", default=list(config.ATTACKED_DEVICES))
    ap.add_argument("--with-baseline", action="store_true", help="同時跑 XGBoost(F+C) 對照")
    args = ap.parse_args()

    art = load_feature_artifacts(args.subset)
    base_manifest = art["manifest"]
    results = []
    for dev in args.devices:
        art["manifest"] = device_holdout_split(base_manifest, dev)
        splits = build_splits(args.subset, art=art)
        te, tr = splits["test"], splits["train"]
        train_classes = np.unique(tr["y"])
        r = train_and_evaluate(args.subset, args.modalities, args.fusion, seed=args.seed,
                               splits=splits, verbose=False)
        pred = predict(r["model"], to_tensors(te, r["scalers"], r["device"]))["tactic_prob"].argmax(1)
        f1, common = _common_macro_f1(te["y"], pred, train_classes)
        entry = {"device": dev, "n_test": int(len(te["y"])), "common_classes": common,
                 "zero_shot_classes": [L1_CLASSES[c] for c in np.unique(te["y"]) if c not in train_classes],
                 "model": {"macro_f1_common": f1, "per_class": _per_class(te["y"], pred)}}
        if args.with_baseline:
            x = lambda s: np.hstack([s["f_stats"], s["c"]])  # noqa: E731
            xgb = XGBClassifier(objective="multi:softprob", num_class=len(L1_CLASSES), tree_method="hist",
                                n_estimators=300, max_depth=6, learning_rate=0.1,
                                random_state=args.seed, n_jobs=-1, verbosity=0)
            xgb.fit(x(tr), tr["y"], sample_weight=class_weights(tr["y"])[tr["y"]])
            bp = xgb.predict(x(te))
            entry["xgboost_fc"] = {"macro_f1_common": _common_macro_f1(te["y"], bp, train_classes)[0],
                                   "per_class": _per_class(te["y"], bp)}
        results.append(entry)
        msg = f"[lodo] {dev:22s} n={entry['n_test']:6d} macroF1(common)={f1:.4f}"
        if args.with_baseline:
            msg += f"  xgb={entry['xgboost_fc']['macro_f1_common']:.4f}"
        print(msg + f"  zero-shot={entry['zero_shot_classes']}", flush=True)

    out = {"experiment": "E4_leave_one_device_out", "settings": vars(args),
           "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"),
           "benign_label_id": L1_TO_ID["Benign"], "devices": results}
    path = config.RESULTS_DIR / f"lodo_{args.subset}_{args.modalities}_{args.fusion}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[lodo] -> {path}")


if __name__ == "__main__":
    main()
