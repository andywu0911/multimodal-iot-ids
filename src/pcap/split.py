"""切分流程。共用邏輯統一在此，禁止各腳本各自實作。

主評估流程是 connection_group_split（以連線為單位）；chronological_split 是它
處理少連線群組時的內部工具，也保留給附錄重現 v1 的 P1 對照。

兩者都逐「情境 + 標籤」時序切分 60/20/20，再合併。絕對不可做全域時間切分——
攻擊情境是依序批次執行的（見 config.SCENARIO_DATES），全域切分會把整個情境
（如 mirai-dos）丟進單一子集。

只按情境切（不細分類別）曾經在 S2 上實測出退化案例：情境內攻擊行為常集中
爆發在某個時段（例如 Merlin UDP Flooding 全部落在該情境前 60%、CoAP
Amplification 全部落在後 40%），導致該類別在切分後直接從 train 或 test
消失，F1/recall 完全算不出來。改成逐「情境 + 原始標籤」切分：同一情境內，
每個原始標籤各自依時間排序後才切 60/20/20——仍保留「同一類別不會用未來
預測過去」的時序因果性，但不再讓類別的全部樣本被單一時段吞掉。

chronological_split 以觀測窗為單位，同一連線的多個窗可能同時落進 train 與
test（近重複洩漏），因此只在連線數不足、連線分組在定義上不可行時使用。
"""
from __future__ import annotations

import pandas as pd

import config
from src.labels import flow_label


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = config.SPLIT_TRAIN_FRAC,
    val_frac: float = config.SPLIT_VAL_FRAC,
) -> pd.DataFrame:
    """回傳新增 'split' 欄位（train/val/test）的 df，逐 (scenario, raw_label) 各自依時間排序切分。"""
    parts = []
    for _, group in df.groupby(["scenario", "raw_label"], dropna=False, sort=False):
        g = group.sort_values("start_ts").copy()
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        g["split"] = (["train"] * n_train + ["val"] * n_val
                       + ["test"] * (n - n_train - n_val))
        parts.append(g)
    return pd.concat(parts).sort_index().reset_index(drop=True)


# ------------------------------------------------ E4：留一裝置（LODO）
def device_holdout_split(df: pd.DataFrame, holdout_device: str,
                         val_frac: float = config.SPLIT_VAL_FRAC) -> pd.DataFrame:
    """留出一台裝置當 test（它的全部觀測窗，任何時間點都不進 train/val）；
    其餘裝置的訓練池沿用主評估流程的連線分組切分，只切 train/val（不留 test），
    主評估流程強制保留的那 1 條 test 連線併入 val，維持時序因果性。
    只能評估留出裝置與訓練池共同涵蓋的類別（效度威脅 T3）。"""
    is_holdout = (df["device"] == holdout_device).to_numpy()
    pool = connection_group_split(df.loc[~is_holdout], train_frac=1 - val_frac, val_frac=val_frac)
    pool["split"] = pool["split"].replace("test", "val")
    test = df.loc[is_holdout].copy()
    test["split"], test["split_mode"] = "test", "device_holdout"
    pool.index, test.index = df.index[~is_holdout], df.index[is_holdout]
    return pd.concat([pool, test]).sort_index()


# ------------------------------------------------ 主評估流程：連線分組切分
def connection_label(df: pd.DataFrame) -> pd.Series:
    """每條連線的代表標籤：套用與 flow 標籤相同的攻擊類別嚴重度優先序
    （src/labels.py 的 flow_label），讓同一連線的所有觀測窗落進同一個
    (情境, 標籤) 群組，連線不會因為各窗標籤不同而被拆到不同 split。"""
    return df.groupby("conn_id")["raw_label"].agg(lambda s: flow_label(set(s.dropna())))


def connection_group_split(
    df: pd.DataFrame,
    train_frac: float = config.SPLIT_TRAIN_FRAC,
    val_frac: float = config.SPLIT_VAL_FRAC,
    min_conns: int = config.MIN_CONNS_FOR_GROUP_SPLIT,
) -> pd.DataFrame:
    """主評估流程：逐 (情境, 連線標籤) 以「連線」為單位時序切分 60/20/20。

    K 把長連線切成多個觀測窗，窗之間共用同一段 session、高度相關；以窗為單位
    切分會讓同一連線同時出現在 train 與 test（v1 的 P1 實測 macro F1 因此虛高
    0.19）。這裡把整條連線一起指派，並保留時序因果性（train 連線的起始時間
    早於 val 早於 test）。

    連線數 ≥ min_conns 的群組：val 與 test 各至少 1 條連線。v1 用無條件捨去，
    3 條連線的 Merlin/Mirai C&C 會得到 1/0/2，val 完全沒有這兩類，模型選擇
    對它們是盲的。

    連線數 < min_conns 的群組（如 CoAP Amplification 全域 1 條連線）：連線分組
    在定義上無法同時出現在 train 與 test，改在連線內依時間切分觀測窗。
    split_mode 欄位標記每列用了哪種方式，論文須揭露。
    """
    df = df.copy()
    df["_conn_label"] = df["conn_id"].map(connection_label(df))
    df["split_mode"] = "connection"
    parts = []
    for _, group in df.groupby(["scenario", "_conn_label"], dropna=False, sort=False):
        conn_start = group.groupby("conn_id")["start_ts"].min().sort_values()
        n = len(conn_start)
        if n < min_conns:
            g = chronological_split(group, train_frac=train_frac, val_frac=val_frac)
            g["split_mode"] = "within_connection"
            parts.append(g)
            continue
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * (1 - train_frac - val_frac)))
        n_train = n - n_val - n_test
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        g = group.copy()
        g["split"] = g["conn_id"].map(dict(zip(conn_start.index, labels)))
        parts.append(g)
    out = pd.concat(parts).sort_index().reset_index(drop=True)
    return out.drop(columns=["_conn_label"])


def n_connections_by_label(df: pd.DataFrame, label_col: str = "raw_label") -> pd.Series:
    """每個標籤有多少條「連線」（而非多少個觀測窗）——連線分組切分下真正的
    有效樣本量，供論文表格與觀測窗數字並排對照。"""
    return df.groupby(label_col)["conn_id"].nunique().sort_values(ascending=False)
