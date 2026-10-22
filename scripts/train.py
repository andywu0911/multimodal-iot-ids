"""多模態模型訓練 CLI（E2 主模型、E3 模態/融合消融共用）。

python scripts/train.py --modalities fpc --fusion gmu                  # 主模型，EVAL_SEEDS 全跑
python scripts/train.py --modalities fc --fusion gmu --seeds 42        # 消融，單 seed
python scripts/train.py ... --checkpoint                               # 存第一個 seed 的模型給下游用
python scripts/train.py ... --evaluate-test                            # 固定後的最終實驗才加

結果寫入 results/train_<subset>_<modalities>_<fusion>.json：每個 seed 的指標加上
val（與 test，若有評估）攻擊分類 macro F1、偵測 F1 的 mean ± std。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from src.features.dataset import build_splits
from src.models.train import train_and_evaluate
from src.provenance import snapshot


def _summary(runs: list[dict], split: str) -> dict | None:
    if runs[0][split] is None:
        return None
    tac = [r[split]["tactic"]["macro_f1"] for r in runs]
    det = [r[split]["detection"]["f1"] for r in runs]
    return {"tactic_macro_f1_mean": float(np.mean(tac)), "tactic_macro_f1_std": float(np.std(tac)),
            "detection_f1_mean": float(np.mean(det)), "detection_f1_std": float(np.std(det))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--modalities", default="fpc")
    ap.add_argument("--fusion", choices=["gmu", "concat"], default="gmu")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.EVAL_SEEDS))
    ap.add_argument("--model", choices=["multimodal", "fsnet"], default="multimodal",
                    help="fsnet = E2 的 FS-Net 對照組（只吃封包序列，--modalities/--fusion 不適用）")
    ap.add_argument("--unknown-policy", choices=["exclude", "as_benign"], default="exclude")
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    tag = f"{args.subset}_{args.modalities}_{args.fusion}" if args.model == "multimodal" else f"{args.subset}_fsnet"
    if args.unknown_policy != "exclude":
        tag += f"_unknown-{args.unknown_policy}"
    splits = build_splits(args.subset, unknown_policy=args.unknown_policy)

    runs = []
    for i, seed in enumerate(args.seeds):
        ckpt = config.CHECKPOINTS_DIR / f"{tag}.pt" if (args.checkpoint and i == 0) else None
        r = train_and_evaluate(args.subset, args.modalities, args.fusion, seed=seed, splits=splits,
                               evaluate_test=args.evaluate_test, checkpoint_path=ckpt, verbose=False,
                               model_name=args.model)
        runs.append({"seed": seed, "best_epoch": r["best_epoch"], "val": r["val"], "test": r["test"]})
        msg = f"[train] {tag} seed={seed} val tactic F1={r['val']['tactic']['macro_f1']:.4f} " \
              f"det F1={r['val']['detection']['f1']:.4f}"
        if r["test"] is not None:
            msg += f" | test tactic F1={r['test']['tactic']['macro_f1']:.4f}"
        print(msg, flush=True)

    out = {
        "settings": vars(args), "model_config": r["config"],
        "provenance": snapshot(config.FEATURES_DIR / args.subset / "manifest.parquet"),
        "n": {s: int(len(splits[s]["y"])) for s in ("train", "val", "test")},
        "summary": {"val": _summary(runs, "val"), "test": _summary(runs, "test")},
        "runs": runs,
    }
    path = config.RESULTS_DIR / f"train_{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[train] summary val={out['summary']['val']}")
    print(f"[train] -> {path}")


if __name__ == "__main__":
    main()
