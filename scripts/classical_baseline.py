"""E2 傳統機器學習對照組：XGBoost / RandomForest 吃表格型特徵。
python scripts/classical_baseline.py --subset S3 [--evaluate-test]

兩組特徵：
  f   View F 統計向量（36 維）
  fc  View F 統計向量 + View C 主機情境（78 維）
fc 這組是關鍵公平性設計：情境特徵本身是表格型、樹模型也吃得下，若只讓神經網路
拿到情境特徵，就會把「情境特徵的效益」誤算成「深度學習的效益」。樹模型吃不下的
是 View F 的封包序列與 View P 的 byte 序列，那才是多模態模型的貢獻所在。

公平性：同一份切分、同一套 Class-Balanced 類別權重（src/models/train.py 的
class_weights）、同一份指標定義（metrics_from_predictions）；超參數在小型網格上
以 val 攻擊分類 macro F1 挑選（seed 42），再以 EVAL_SEEDS 重跑報 mean ± std。
偵測頭的分數取 1 − P(Benign)。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import config
from src.features.dataset import build_splits
from src.labels import L1_CLASSES, L1_TO_ID
from src.models.train import class_weights, metrics_from_predictions
from src.provenance import snapshot

GRIDS = {
    "xgboost": [{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
                {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05}],
    "random_forest": [{"n_estimators": 300, "max_depth": None},
                      {"n_estimators": 600, "max_depth": 20}],
}
BENIGN_ID = L1_TO_ID["Benign"]


def _make(name: str, params: dict, seed: int):
    if name == "xgboost":
        return XGBClassifier(objective="multi:softprob", num_class=len(L1_CLASSES), tree_method="hist",
                             random_state=seed, n_jobs=-1, verbosity=0, **params)
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


def _x(split: dict, features: str) -> np.ndarray:
    return split["f_stats"] if features == "f" else np.hstack([split["f_stats"], split["c"]])


def _predict(model, split, features) -> dict:
    prob = model.predict_proba(_x(split, features))
    return {"tactic_prob": prob, "det_prob": 1.0 - prob[:, BENIGN_ID], "gates": None}


def run(name: str, features: str, splits: dict, seeds, evaluate_test: bool) -> dict:
    tr, va = splits["train"], splits["val"]
    sw = class_weights(tr["y"])[tr["y"]]
    best = None
    for params in GRIDS[name]:
        m = _make(name, params, config.RANDOM_SEED).fit(_x(tr, features), tr["y"], sample_weight=sw)
        f1 = f1_score(va["y"], m.predict(_x(va, features)), average="macro",
                      labels=list(range(len(L1_CLASSES))), zero_division=0)
        if best is None or f1 > best[1]:
            best = (params, f1)
    runs = []
    for seed in seeds:
        m = _make(name, best[0], seed).fit(_x(tr, features), tr["y"], sample_weight=sw)
        r = {"seed": seed, "val": metrics_from_predictions(_predict(m, va, features), va["y"], va["y_det"],
                                                           va["manifest"]), "test": None}
        if evaluate_test:
            te = splits["test"]
            r["test"] = metrics_from_predictions(_predict(m, te, features), te["y"], te["y_det"], te["manifest"])
        runs.append(r)
    tac = [r["val"]["tactic"]["macro_f1"] for r in runs]
    print(f"  [{name}/{features}] params={best[0]} val tactic F1={np.mean(tac):.4f}±{np.std(tac):.4f}", flush=True)
    return {"model": name, "features": features, "best_params": best[0], "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.EVAL_SEEDS))
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    splits = build_splits(args.subset)
    results = {
        "experiment": "E2_classical_baselines", "settings": vars(args),
        "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"),
        "models": [run(name, feats, splits, args.seeds, args.evaluate_test)
                   for name in GRIDS for feats in ("f", "fc")],
    }
    out = config.RESULTS_DIR / f"classical_baseline_{args.subset}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"[classical_baseline] -> {out}")


if __name__ == "__main__":
    main()
