"""L2 訓練與評估的共用邏輯。訓練流程統一在此，各腳本一律 import。

測試集紀律：train_and_evaluate() 預設只評估 val（evaluate_test=False）。
開發、調參、模型選擇都只看 val；評估流程與超參數固定後的最終實驗才傳 True。
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import config
from src.features.dataset import build_splits
from src.features.view_f import SEQ_CHANNELS
from src.labels import L1_CLASSES
from src.models.baselines import FSNetBaseline
from src.models.fusion import MultimodalIDS

CC_RAW_LABELS = ("Mirai C&C Communication", "Merlin C&C Communication")

# CPU 推論時，batch ≥ 16 的 Conv1d 會被 PyTorch 導向 NNPACK，在 Apple Silicon 上實測佔掉
# 97% 時間、吞吐量只剩 262 窗/秒；關掉後為 ~3,400 窗/秒，輸出差異 < 1e-6。MPS 不受影響。
torch.backends.nnpack.set_flags(False)


class FeatureScaler:
    """只存 mean_/scale_ 兩個陣列的標準化器；checkpoint 不 pickle sklearn 物件，
    load_checkpoint() 才能用 weights_only=True 安全載入。"""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, X: np.ndarray) -> "FeatureScaler":
        sk = StandardScaler().fit(X)
        return cls(sk.mean_, sk.scale_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_).astype(np.float32)


def class_weights(y_train: np.ndarray, n_classes: int = len(L1_CLASSES),
                  beta: float = config.CB_BETA) -> np.ndarray:
    """Class-Balanced 權重（Cui et al., CVPR 2019）：w_c ∝ (1-β)/(1-β^{n_c})，正規化成總和 = 類別數。

    神經網路與樹模型對照組共用同一份，比較才公平。"""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    eff = (1.0 - np.power(beta, np.maximum(counts, 1))) / (1.0 - beta)
    w = 1.0 / eff
    return (w / w.sum() * n_classes).astype(np.float32)


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_tensors(split: dict, scalers: dict, device) -> dict:
    return {
        "f_stats": torch.tensor(scalers["f_stats"].transform(split["f_stats"]), device=device),
        "f_seq": torch.tensor(split["f_seq"], dtype=torch.float32, device=device),
        "f_seq_len": torch.tensor(split["f_seq_len"], dtype=torch.long, device=device),
        "p": torch.tensor(split["p"].astype(np.int64), device=device),
        "c": torch.tensor(scalers["c"].transform(split["c"]), device=device),
    }


def _slice(t: dict, idx) -> dict:
    return {k: v[idx] for k, v in t.items()}


def predict(model, tensors: dict, batch_size: int = 4096) -> dict:
    """回傳 {'tactic_prob' (n,6), 'det_prob' (n,), 'gates' (n,k) 或 None}。"""
    model.eval()
    n = tensors["f_stats"].shape[0]
    tac, det, gates = [], [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            (logits, det_logit), g = model(_slice(tensors, slice(i, i + batch_size)), return_gates=True)
            tac.append(torch.softmax(logits, dim=1).cpu().numpy())
            det.append(torch.sigmoid(det_logit).cpu().numpy())
            if g is not None:
                gates.append(g.cpu().numpy())
    return {"tactic_prob": np.concatenate(tac), "det_prob": np.concatenate(det),
            "gates": np.concatenate(gates) if gates else None}


def cc_recall_breakdown(pred: np.ndarray, y: np.ndarray, manifest) -> dict:
    """明文 Mirai C&C 與 TLS Merlin C&C 各自被正確分到 Command & Control 的比例與連線數。"""
    raw = manifest["raw_label"].to_numpy()
    out = {}
    for lab in CC_RAW_LABELS:
        m = raw == lab
        out[lab] = None if not m.any() else {
            "n_windows": int(m.sum()), "n_connections": int(manifest.loc[m, "conn_id"].nunique()),
            "recall_to_CC": float((pred[m] == y[m]).mean())}
    return out


def metrics_from_predictions(out: dict, y: np.ndarray, y_det: np.ndarray, manifest) -> dict:
    pred = out["tactic_prob"].argmax(axis=1)
    labels = list(range(len(L1_CLASSES)))
    det_pred = (out["det_prob"] >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_det, det_pred, average="binary", zero_division=0)
    benign = y_det == 0
    m = {
        "tactic": {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
            "report": classification_report(y, pred, labels=labels, target_names=L1_CLASSES,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "cc_breakdown": cc_recall_breakdown(pred, y, manifest),
        },
        "detection": {
            "precision": p, "recall": r, "f1": f,
            "fpr": float(det_pred[benign].mean()) if benign.any() else None,
            "n_benign": int(benign.sum()),
            "roc_auc": roc_auc_score(y_det, out["det_prob"]) if len(set(y_det)) > 1 else None,
            "pr_auc": average_precision_score(y_det, out["det_prob"]) if len(set(y_det)) > 1 else None,
        },
    }
    if out["gates"] is not None:
        m["gates_mean"] = out["gates"].mean(axis=0).tolist()
    return m


def train_and_evaluate(
    subset: str = "S3", modalities: str = "fpc", fusion: str = "gmu",
    embed_dim: int = config.EMBED_DIM, epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
    lambda_det: float = config.LAMBDA_DET, seed: int = config.RANDOM_SEED,
    unknown_policy: str = "exclude", splits: dict | None = None,
    evaluate_test: bool = False, checkpoint_path=None, verbose: bool = True,
    model_name: str = "multimodal",
) -> dict:
    """訓練一個模態組合，以 val 攻擊分類 macro F1 選最佳 epoch，回傳模型、scaler 與指標。

    model_name='fsnet' 改訓練 E2 的 FS-Net 對照組（src/models/baselines.py），其餘流程、
    損失權重、模型選擇與指標完全相同；該模型的重建損失經由 model.aux_loss 加入總損失。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    splits = splits if splits is not None else build_splits(subset, unknown_policy=unknown_policy)
    tr, va = splits["train"], splits["val"]

    scalers = {"f_stats": FeatureScaler.fit(tr["f_stats"]), "c": FeatureScaler.fit(tr["c"])}
    t_tr, t_va = to_tensors(tr, scalers, device), to_tensors(va, scalers, device)
    y_tr = torch.tensor(tr["y"], device=device)
    yd_tr = torch.tensor(tr["y_det"], device=device)

    if model_name == "fsnet":
        model = FSNetBaseline(len(SEQ_CHANNELS), n_classes=len(L1_CLASSES)).to(device)
    else:
        model = MultimodalIDS(modalities, fusion, stats_dim=tr["f_stats"].shape[1],
                              seq_channels=len(SEQ_CHANNELS), ctx_dim=tr["c"].shape[1],
                              n_classes=len(L1_CLASSES), embed_dim=embed_dim).to(device)
    w = torch.tensor(class_weights(tr["y"]), device=device)
    n_pos = float(tr["y_det"].sum())
    pos_weight = torch.tensor((len(tr["y_det"]) - n_pos) / max(n_pos, 1.0), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n, t0 = len(tr["y"]), time.time()
    best_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, det_logit = model(_slice(t_tr, idx))
            loss = (F.cross_entropy(logits, y_tr[idx], weight=w)
                    + lambda_det * F.binary_cross_entropy_with_logits(det_logit, yd_tr[idx], pos_weight=pos_weight))
            if getattr(model, "aux_loss", None) is not None:
                loss = loss + model.aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_f1 = f1_score(va["y"], predict(model, t_va)["tactic_prob"].argmax(1), average="macro",
                          labels=list(range(len(L1_CLASSES))), zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:2d} val_macroF1={val_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)

    result = {
        "model": model, "scalers": scalers, "device": device, "splits": splits,
        "config": {"model_name": model_name, "modalities": model.modalities, "fusion": fusion,
                   "embed_dim": embed_dim,
                   "stats_dim": tr["f_stats"].shape[1], "ctx_dim": tr["c"].shape[1],
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "lambda_det": lambda_det, "seed": seed, "cb_beta": config.CB_BETA},
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "val": metrics_from_predictions(predict(model, t_va), va["y"], va["y_det"], va["manifest"]),
        "test": None,
    }
    if evaluate_test:
        te = splits["test"]
        result["test"] = metrics_from_predictions(predict(model, to_tensors(te, scalers, device)),
                                                  te["y"], te["y_det"], te["manifest"])
    if checkpoint_path is not None:
        save_checkpoint(result, checkpoint_path)
    return result


def save_checkpoint(result: dict, path) -> None:
    """mps 在固定 seed 下仍非決定性，下游腳本一律讀同一份 checkpoint，不各自重訓。"""
    torch.save({
        "format_version": 3,
        "state_dict": result["model"].state_dict(),
        "config": result["config"],
        "scalers": {k: {"mean": torch.tensor(s.mean_, dtype=torch.float32),
                        "scale": torch.tensor(s.scale_, dtype=torch.float32)}
                    for k, s in result["scalers"].items()},
    }, path)


def load_checkpoint(path, device: torch.device | None = None) -> dict:
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = MultimodalIDS(cfg["modalities"], cfg["fusion"], stats_dim=cfg["stats_dim"],
                          seq_channels=len(SEQ_CHANNELS), ctx_dim=cfg["ctx_dim"],
                          n_classes=len(L1_CLASSES), embed_dim=cfg["embed_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scalers = {k: FeatureScaler(v["mean"].numpy(), v["scale"].numpy()) for k, v in ckpt["scalers"].items()}
    return {"model": model, "scalers": scalers, "config": cfg, "device": device}

