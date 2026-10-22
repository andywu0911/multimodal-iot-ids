"""E1：官方 baseline 重現。CLI: python scripts/e1_baseline.py

目標對照（Belarbi et al. 2025, arXiv:2502.03134，Table 4，FL-DNN）：
    Accuracy 0.89；Infection F1 0.00；Normal recall 0.52。

方法論對齊官方論文：
  - 封包級（非本專案自己的 flow 級），直接讀 gotham2025/processed/*.csv
    的官方 23 個 tshark 欄位（frame.time ... udp.dstport + label，見論文 Table 2）。
    這些欄位包含 eth/ip 位址與 checksum —— 在本專案自己的模型中這些是被
    隔離的捷徑特徵，但這裡是刻意「原樣重現官方
    baseline」，不是我們自己的主模型，所以不適用該隔離規則。
  - 6 類標籤：直接借用 src/labels.py 的 RAW_TO_L1（我們的 L1 攻擊類別層），
    因為官方論文的 6 類（Normal/Network Scanning/Brute Force/Infection/
    C&C Communication/DoS）與我們的 L1（Benign/Reconnaissance/
    Credential Access/Tool Transfer/Command & Control/Impact）在
    label→class 的分組上是同一件事，只是命名不同，見 L1_TO_PAPER_NAME。
  - DNN、標準化數值特徵、類別特徵編碼為數值 —— 對應論文 Fig. 7 的前處理描述。

已知簡化（論文原設計 vs 本重現，皆為刻意取捨，非疏漏）：
  1. 論文用 78 台裝置、10 輪 FL、每輪抽樣 13 台、每台本地 80/20 切分後
     FedAvg 聚合；此處改為集中式訓練，裝置範圍限縮為 S2 的 6 台非攝影機
     攻擊裝置（config.SUBSETS["S2"]["devices"]），每台各自隨機 80/20 切分
     後合併——不做真正的聯邦輪次與跨裝置非 IID 聚合。
  2. 只取 78 台裝置的一個子集，會讓 Benign 相對稀有類別的比例遠高於官方
     全量 78 台的設定（官方用了 70 台純良性裝置稀釋整體）；因此本重現的
     Infection/Normal 數字預期會與論文有落差，這是可預期的方法論差異，
     不是重現失敗。
  3. 論文正文只用文字描述前處理步驟（移除不必要特徵、標準化數值、類別
     編碼為數值），未公開確切程式碼；本腳本的類別特徵一律用 pandas
     factorize（ordinal code），數值特徵一律標準化，是對該描述的
     best-effort 具體化。
  4. 為了讓封包級全量資料在單機上可訓練，對每台裝置的 CSV 做均勻隨機列
     抽樣（config 內 E1_SAMPLE_FRAC），維持原始類別比例不變，只縮小絕對量。

--drop-shortcuts（C1 補證）：
本重現的 0.9997 一直只被拿來跟論文的 0.89 比較，但那個落差有兩個混淆因子
（集中式訓練 vs FL、6 台裝置 vs 78 台，見上方已知簡化 1、2），捷徑特徵
（eth/ip 身分欄位、checksum、埠號）的貢獻從未被單獨量測過——E2/E9 量的是
本專案自己的 flow 特徵與自家模型，都沒有碰過官方這 22 個欄位。
加這個旗標，同資料、同模型、同訓練法、同切分，只拿掉
SHORTCUT_CATEGORICAL_COLS/SHORTCUT_NUMERIC_COLS 這組欄位，得到一個乾淨的
反事實對照：「官方特徵集拿掉身分/埠號捷徑後掉多少」，才是 C1b
（「官方特徵集確實被捷徑污染」）唯一的直接證據。輸出另存
e1_baseline_no_shortcuts.json，不覆蓋原本含捷徑那份。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import config
from src.labels import RAW_TO_L1, UNKNOWN_LABEL
from src.pcap.labeller import _parse_ts_us
from src.provenance import snapshot

E1_DEVICES = config.SUBSETS["S2"]["devices"]

NUMERIC_COLS = [
    "frame.len", "ip.ttl", "ip.checksum", "ip.tos",
    "tcp.srcport", "tcp.dstport", "tcp.window_size_value",
    "tcp.window_size_scalefactor", "tcp.checksum", "tcp.pdu.size",
    "udp.srcport", "udp.dstport",
]
CATEGORICAL_COLS = [
    "frame.protocols", "eth.src", "eth.dst", "ip.dst", "ip.src",
    "ip.flags", "ip.proto", "tcp.flags", "tcp.options",
]
ALL_COLS = ["frame.time"] + NUMERIC_COLS + CATEGORICAL_COLS + ["label"]

# --drop-shortcuts 拿掉的欄位：身分（eth/ip 位址）+ checksum + 埠號，對應
# 「捷徑特徵隔離」原則的官方欄位版本。frame.time 刻意保留
# ——時間洩漏（§13 已知限制 6）是另一個獨立的混淆因子，這個消融只想單獨
# 隔離「身分/埠號捷徑」的貢獻，兩個問題混在一起會讓結果無法解讀。
SHORTCUT_CATEGORICAL_COLS = ["eth.src", "eth.dst", "ip.dst", "ip.src"]
SHORTCUT_NUMERIC_COLS = [
    "ip.checksum", "tcp.checksum",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
]

L1_TO_PAPER_NAME = {
    "Benign": "Normal",
    "Reconnaissance": "Network Scanning",
    "Credential Access": "Brute Force",
    "Tool Transfer": "Infection",
    "Command & Control": "C&C Communication",
    "Impact": "DoS",
}
PAPER_CLASSES = list(L1_TO_PAPER_NAME.values())
CLASS_TO_ID = {c: i for i, c in enumerate(PAPER_CLASSES)}


def _hex_or_float(series: pd.Series) -> pd.Series:
    def parse(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str) and v.startswith("0x"):
            return float(int(v, 16))
        return float(v)
    return series.map(parse)


def load_device(device: str, frac: float, seed: int) -> pd.DataFrame:
    csv_path = config.OFFICIAL_CSV_ROOT / f"{config.PCAP_PREFIX}{device}.csv"
    df = pd.read_csv(csv_path, usecols=ALL_COLS)
    df = df.sample(frac=frac, random_state=seed)

    df["l1_label"] = df["label"].map(RAW_TO_L1)
    df = df[df["label"] != UNKNOWN_LABEL]
    df = df.dropna(subset=["l1_label"])
    df["paper_label"] = df["l1_label"].map(L1_TO_PAPER_NAME)
    df["device"] = device
    return df


def fit_categorical_maps(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, dict]:
    """在 train 上 fit 類別碼表，test 必須沿用同一份，否則同一類別在兩邊會被
    編成不同數字，等於把身分特徵變成雜訊。"""
    maps = {}
    for c in categorical_cols:
        cats = pd.Categorical(df[c].fillna("__na__"))
        maps[c] = {v: i for i, v in enumerate(cats.categories)}
    return maps


def build_features(df: pd.DataFrame, cat_maps: dict[str, dict],
                    numeric_cols: list[str], categorical_cols: list[str],
                    ) -> tuple[np.ndarray, np.ndarray]:
    num = df[[c for c in numeric_cols if c not in ("ip.checksum", "tcp.checksum")]].copy()
    if "ip.checksum" in numeric_cols:
        num["ip.checksum"] = _hex_or_float(df["ip.checksum"])
    if "tcp.checksum" in numeric_cols:
        num["tcp.checksum"] = _hex_or_float(df["tcp.checksum"])
    num["frame.time_us"] = _parse_ts_us(df["frame.time"])
    for c in num.columns:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.fillna(0.0)

    cat = pd.DataFrame(index=df.index)
    for c in categorical_cols:
        lookup = cat_maps[c]
        cat[c] = df[c].fillna("__na__").map(lambda v: lookup.get(v, -1)).astype(np.float32)

    X = pd.concat([num, cat], axis=1).to_numpy(dtype=np.float32)
    y = df["paper_label"].map(CLASS_TO_ID).to_numpy(dtype=np.int64)
    return X, y


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=config.E1_SAMPLE_FRAC)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop-shortcuts", action="store_true",
                     help="C1 補證：拿掉 SHORTCUT_CATEGORICAL_COLS/SHORTCUT_NUMERIC_COLS"
                          "（eth/ip 身分、checksum、埠號），其餘設定完全不變，量測官方特徵集"
                          "被身分捷徑污染的貢獻。輸出另存 e1_baseline_no_shortcuts.json。")
    args = ap.parse_args()

    numeric_cols = list(NUMERIC_COLS)
    categorical_cols = list(CATEGORICAL_COLS)
    if args.drop_shortcuts:
        numeric_cols = [c for c in numeric_cols if c not in SHORTCUT_NUMERIC_COLS]
        categorical_cols = [c for c in categorical_cols if c not in SHORTCUT_CATEGORICAL_COLS]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[e1_baseline] devices={E1_DEVICES} frac={args.frac} drop_shortcuts={args.drop_shortcuts} "
          f"torch_device={device}")

    t0 = time.time()
    train_parts, test_parts = [], []
    for dev in E1_DEVICES:
        df = load_device(dev, args.frac, config.RANDOM_SEED)
        df = df.sample(frac=1.0, random_state=config.RANDOM_SEED)  # shuffle
        n_test = int(len(df) * 0.2)
        test_parts.append(df.iloc[:n_test])
        train_parts.append(df.iloc[n_test:])
        print(f"  {dev:22s} rows={len(df):8d} (loaded {time.time() - t0:.1f}s)")

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)
    print(f"[e1_baseline] train={len(train_df)} test={len(test_df)}  load+concat {time.time() - t0:.1f}s")

    cat_maps = fit_categorical_maps(train_df, categorical_cols)
    X_train, y_train = build_features(train_df, cat_maps, numeric_cols, categorical_cols)
    X_test, y_test = build_features(test_df, cat_maps, numeric_cols, categorical_cols)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLP(X_train.shape[1], len(PAPER_CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    n = Xt.shape[0]

    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:2d}  loss={total_loss / n:.4f}  ({time.time() - t0:.1f}s)")

    model.eval()
    with torch.no_grad():
        Xe = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds = model(Xe).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=list(range(len(PAPER_CLASSES))),
        target_names=PAPER_CLASSES, output_dict=True, zero_division=0,
    )
    print(f"\n[e1_baseline] accuracy={acc:.4f}\n")
    print(classification_report(
        y_test, preds, labels=list(range(len(PAPER_CLASSES))),
        target_names=PAPER_CLASSES, zero_division=0,
    ))

    out = {
        "experiment": "E1_baseline_reproduction" if not args.drop_shortcuts
                       else "E1_baseline_reproduction_no_shortcuts",
        "target_paper": "Belarbi et al. 2025, arXiv:2502.03134, Table 4",
        "target_metrics": {"accuracy": 0.89, "Infection_f1": 0.00, "Normal_recall": 0.52},
        "settings": {
            "devices": E1_DEVICES, "sample_frac": args.frac,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seed": config.RANDOM_SEED, "n_train": len(train_df), "n_test": len(test_df),
            "drop_shortcuts": args.drop_shortcuts,
            "dropped_columns": (SHORTCUT_CATEGORICAL_COLS + SHORTCUT_NUMERIC_COLS
                                 if args.drop_shortcuts else []),
            "n_features": X_train.shape[1],
        },
        "provenance": snapshot(),
        "accuracy": acc,
        "classification_report": report,
    }
    out_name = "e1_baseline_no_shortcuts.json" if args.drop_shortcuts else "e1_baseline.json"
    out_path = config.RESULTS_DIR / out_name
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e1_baseline] -> {out_path}")


if __name__ == "__main__":
    main()
