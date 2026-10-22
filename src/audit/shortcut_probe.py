"""E2：捷徑特徵 probe。捷徑特徵（IP/MAC/埠號/checksum）
絕對不可進入主模型特徵集，只允許出現在這裡。

目的：分別只用「IP」「MAC」「埠號」「checksum（對照組）」四組欄位訓練一個
簡單分類器，看這些本身跟裝置身分／協定慣例綁死的欄位，單獨能把 L1 攻擊類別層
標籤解釋掉多少。checksum 是對照組——理論上跟標籤無關，預期接近多數類基準；
如果 checksum 也異常準，代表 probe 方法本身有漏洞（例如切分洩漏），要先查
那個而不是相信 IP/MAC 的數字。

類別特徵編碼：sparse one-hot（`OneHotEncoder(handle_unknown="ignore")`），
不是把類別手動編碼成單一維度的 ordinal code。原因：probe 要衡量的是「這些
欄位單獨能洩漏多少標籤資訊」的上界，one-hot 讓每個類別各自有一個維度，樹
一次分裂就能直接隔離任一類別；ordinal 編碼把所有類別擠進同一個維度，樹得
用一串門檻切分才能隔離不相鄰的類別，人為限制了樹能利用的資訊、可能低估
捷徑污染的嚴重度——這對 C1 的證據強度是反向偏誤。dst_ip 有 5,000+ 相異值，
超過 HistGradientBoostingClassifier 原生類別支援的 255 上限，故用
DecisionTreeClassifier（原生接受 sparse 輸入，不受此限制）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import config

CATEGORICAL_PROBES = {
    "ip": ["src_ip", "dst_ip"],
    "mac": ["src_mac", "dst_mac"],
}
NUMERIC_PROBES = {
    "port": ["src_port", "dst_port"],
    "checksum": ["ip_checksum", "tcp_checksum"],   # 對照組
}


def _encode_categorical(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """sparse one-hot：在 train 上 fit，test 用同一份編碼器——train 沒見過的
    類別（handle_unknown="ignore"）在 test 全部維度為 0，等同舊版的 -1
    sentinel（「模型沒學過這個值」），但不會把不相鄰的類別硬塞進同一個
    有序維度。"""
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    X_train = enc.fit_transform(train[cols].fillna("__na__"))
    X_test = enc.transform(test[cols].fillna("__na__"))
    return X_train, X_test


def _encode_numeric(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_num = train[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    test_num = test[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return train_num, test_num


def run_probe(train: pd.DataFrame, test: pd.DataFrame, name: str, cols: list[str],
              label_col: str = "l1_label", seed: int = config.RANDOM_SEED) -> dict:
    if name in CATEGORICAL_PROBES:
        X_train, X_test = _encode_categorical(train, test, cols)
    else:
        X_train, X_test = _encode_numeric(train, test, cols)

    y_train = train[label_col].to_numpy()
    y_test = test[label_col].to_numpy()

    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        "probe": name,
        "columns": cols,
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }


def majority_baseline(train: pd.DataFrame, test: pd.DataFrame, label_col: str = "l1_label") -> dict:
    majority = train[label_col].value_counts().idxmax()
    preds = np.full(len(test), majority)
    y_test = test[label_col].to_numpy()
    return {
        "probe": "majority_baseline",
        "columns": [],
        "majority_class": majority,
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
    }


def run_all_probes(df: pd.DataFrame, label_col: str = "l1_label") -> list[dict]:
    df = df[df[label_col].notna()]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    results = [majority_baseline(train, test, label_col)]
    for name, cols in {**CATEGORICAL_PROBES, **NUMERIC_PROBES}.items():
        results.append(run_probe(train, test, name, cols, label_col))
    return results


def feature_identity_probe(X_train: np.ndarray, dev_train: np.ndarray, X_test: np.ndarray,
                           dev_test: np.ndarray, seed: int = config.RANDOM_SEED) -> dict:
    """反向 probe：主模型的特徵本身能不能認出「這個窗來自哪台裝置」。

    規範第 2 條把身分欄位擋在特徵集之外，但行為特徵仍可能間接帶有裝置指紋
    （例如某台裝置固定的流量規模）。這裡用 RandomForest 從特徵預測裝置，準確率
    越高於多數類基準，代表特徵裡的裝置指紋越強，跨裝置泛化（LODO）越可能受影響。
    """
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X_train, dev_train)
    pred = clf.predict(X_test)
    majority = pd.Series(dev_train).value_counts().idxmax()
    return {
        "n_devices": int(len(np.unique(dev_train))), "n_train": int(len(dev_train)), "n_test": int(len(dev_test)),
        "accuracy": float(accuracy_score(dev_test, pred)),
        "macro_f1": float(f1_score(dev_test, pred, average="macro", zero_division=0)),
        "majority_accuracy": float((dev_test == majority).mean()),
    }
