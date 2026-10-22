"""chronological_split 與主評估流程 connection_group_split 的核心不變式。

這兩個函式的 bug 不會拋例外——只會安靜地讓 test 集偷看到 train 的資訊
（時序洩漏／連線洩漏），跑起來一切正常、數字甚至會更好看，是最需要測試
釘死的地方。
"""
import pandas as pd

from src.pcap.split import chronological_split, connection_group_split, device_holdout_split


def test_chronological_split_no_time_leakage():
    """同一個 (scenario, raw_label) 內，train 的全部時間戳必須早於 val，
    val 必須早於 test——這是「同一類別不用未來預測過去」的直接體現。"""
    n = 100
    df = pd.DataFrame({
        "scenario": ["s1"] * n,
        "raw_label": ["Benign"] * n,
        "start_ts": list(range(n)),  # 嚴格遞增，避免同時間戳造成的邊界模糊
    })
    out = chronological_split(df, train_frac=0.6, val_frac=0.2)

    train_ts = out.loc[out["split"] == "train", "start_ts"]
    val_ts = out.loc[out["split"] == "val", "start_ts"]
    test_ts = out.loc[out["split"] == "test", "start_ts"]

    assert train_ts.max() < val_ts.min()
    assert val_ts.max() < test_ts.min()
    assert len(train_ts) + len(val_ts) + len(test_ts) == n


def test_chronological_split_independent_per_group():
    """不同 (scenario, raw_label) 各自獨立切分，不會互相污染切點。"""
    df = pd.DataFrame({
        "scenario": ["s1"] * 10 + ["s2"] * 10,
        "raw_label": ["A"] * 10 + ["B"] * 10,
        "start_ts": list(range(10)) + list(range(100, 110)),
    })
    out = chronological_split(df, train_frac=0.6, val_frac=0.2)
    for (_, _), g in out.groupby(["scenario", "raw_label"]):
        assert len(g) == 10
        assert (g["split"] == "train").sum() == 6
        assert (g["split"] == "val").sum() == 2
        assert (g["split"] == "test").sum() == 2


def _make_connection_rows(conn_id: int, n_windows: int, base_ts: float,
                           scenario="s1", raw_label="Benign") -> list[dict]:
    """模擬一條連線被 K 切成多個觀測窗：共用 conn_id，時間緊鄰，
    整條連線的時間範圍跟其他連線不重疊。"""
    return [{"conn_id": f"c{conn_id}", "scenario": scenario, "raw_label": raw_label,
             "start_ts": base_ts + i * 0.01} for i in range(n_windows)]


def _conns(n_conns, n_windows=3, **kw):
    rows = []
    for c in range(n_conns):
        rows.extend(_make_connection_rows(c, n_windows, base_ts=c * 10.0, **kw))
    return pd.DataFrame(rows)


def test_connection_group_split_no_connection_spans_splits():
    """核心不變式：同一連線的全部觀測窗必須落在同一個 split。"""
    out = connection_group_split(_conns(10))
    assert (out.groupby("conn_id")["split"].nunique() == 1).all(), "有連線跨越了多個 split"


def test_connection_group_split_preserves_chronological_order():
    """連線層級的時序因果性：train 連線早於 val，val 早於 test。"""
    out = connection_group_split(_conns(10, n_windows=2))
    ts = {s: out.loc[out["split"] == s, "start_ts"] for s in ("train", "val", "test")}
    assert ts["train"].max() < ts["val"].min()
    assert ts["val"].max() < ts["test"].min()


def test_connection_group_split_three_connections_fill_every_split():
    """v1 的缺陷：3 條連線用無條件捨去會得到 1/0/2，val 沒有這個類別。
    連線數 ≥ MIN_CONNS_FOR_GROUP_SPLIT 時每個 split 都要至少 1 條連線。"""
    out = connection_group_split(_conns(3), min_conns=3)
    per_split = out.groupby("split")["conn_id"].nunique()
    assert per_split.reindex(["train", "val", "test"]).fillna(0).min() >= 1
    assert (out["split_mode"] == "connection").all()


def test_connection_group_split_few_connections_fall_back_within_connection():
    """單一連線的類別（如 CoAP Amplification）改在連線內依時間切分，並標記 split_mode。"""
    out = connection_group_split(_conns(1, n_windows=10), min_conns=3)
    assert set(out["split"]) == {"train", "val", "test"}
    assert (out["split_mode"] == "within_connection").all()
    ts = {s: out.loc[out["split"] == s, "start_ts"] for s in ("train", "val", "test")}
    assert ts["train"].max() < ts["val"].min() <= ts["val"].max() < ts["test"].min()


def test_connection_group_split_uses_connection_level_label():
    """一條連線的窗可能帶不同標籤；整條連線依優先序歸入同一群組，不會被拆開。"""
    rows = []
    for c in range(6):
        for i, lab in enumerate(["Benign", "Telnet Brute Force", "Benign"]):
            rows.append({"conn_id": f"c{c}", "scenario": "s1", "raw_label": lab,
                         "start_ts": c * 10.0 + i})
    out = connection_group_split(pd.DataFrame(rows))
    assert (out.groupby("conn_id")["split"].nunique() == 1).all()


def test_device_holdout_split_isolates_device():
    """LODO：留出裝置的窗只能在 test，訓練池的窗不能落進 test。"""
    df = _conns(6)
    df["device"] = ["held"] * 3 + ["other"] * (len(df) - 3)
    out = device_holdout_split(df, "held")
    assert set(out.loc[out["device"] == "held", "split"]) == {"test"}
    assert "test" not in set(out.loc[out["device"] != "held", "split"])
