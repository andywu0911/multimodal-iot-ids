"""特徵的裝置身分 probe。python scripts/identity_probe.py --subset S3

用主模型的表格特徵（F 統計、C 主機情境、F+C）預測觀測窗來自哪台裝置
（src/audit/shortcut_probe.py 的 feature_identity_probe），兩個族群：
  benign   全部裝置的良性窗（裝置在正常狀態下的指紋）
  attack   8 台攻擊裝置的攻擊窗（跨裝置泛化 LODO 面對的情境）
以主評估流程的 train 訓練、test 評估。這是對特徵的審計，不涉及模型選擇。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from src.audit.shortcut_probe import feature_identity_probe
from src.features.dataset import build_splits
from src.provenance import snapshot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    args = ap.parse_args()

    s = build_splits(args.subset)
    feats = {"f": lambda d: d["f_stats"], "c": lambda d: d["c"],
             "fc": lambda d: np.hstack([d["f_stats"], d["c"]])}
    populations = {"benign": lambda d: d["y_det"] == 0, "attack": lambda d: d["y_det"] == 1}

    results = []
    for pop, sel in populations.items():
        tr, te = sel(s["train"]), sel(s["test"])
        dev_tr = s["train"]["manifest"]["device"].to_numpy()[tr]
        dev_te = s["test"]["manifest"]["device"].to_numpy()[te]
        for name, get in feats.items():
            r = feature_identity_probe(get(s["train"])[tr], dev_tr, get(s["test"])[te], dev_te)
            r.update(population=pop, features=name)
            results.append(r)
            print(f"[identity_probe] {pop:6s} {name:2s} devices={r['n_devices']:2d} "
                  f"acc={r['accuracy']:.3f} (majority {r['majority_accuracy']:.3f}) macroF1={r['macro_f1']:.3f}")

    path = config.RESULTS_DIR / f"identity_probe_{args.subset}.json"
    path.write_text(json.dumps({"experiment": "feature_identity_probe", "settings": vars(args),
                                "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"),
                                "results": results}, indent=2))
    print(f"[identity_probe] -> {path}")


if __name__ == "__main__":
    main()
