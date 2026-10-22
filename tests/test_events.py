"""src/events/aggregate.py 兩個容易錯、錯了也不會拋例外的地方：
1. aggregate_events() 的 idle/active timeout 邊界——跟 flow 聚合同一種問題，
   間隔或總長度算錯只會讓事件被錯誤合併或錯誤拆開，不會有任何錯誤訊息。
2. match_events() 的區間重疊雙指標掃描——兩個事件的 window_end 剛好相等時
   只推進一個指標，會讓同一個 pred 事件被拿去跟多個 true 事件重複配對
   （這是實作時真的發生過的 bug：自己跟自己比對本該 100% matched，
   卻因為這個 tie-breaking 錯誤算出多出將近一倍的配對數）。
"""
import pandas as pd

from src.events.aggregate import aggregate_events, match_events


def _flow(device, start_ts, end_ts, label):
    return {"device": device, "start_ts": start_ts, "end_ts": end_ts, "l1_label": label}


def test_aggregate_events_idle_timeout_splits_session():
    """間隔超過 idle_timeout 必須起新事件，間隔以內必須併進同一事件。"""
    flows = pd.DataFrame([
        _flow("d1", 0, 1, "Impact"),
        _flow("d1", 5, 6, "Impact"),        # gap=4 <= idle(10) -> 同一事件
        _flow("d1", 20, 21, "Impact"),       # gap=14 > idle(10) -> 新事件
    ])
    events = aggregate_events(flows, "l1_label", idle_timeout=10, active_timeout=1000)
    assert len(events) == 2
    assert events.iloc[0]["n_flows"] == 2
    assert events.iloc[1]["n_flows"] == 1


def test_aggregate_events_active_timeout_caps_long_session():
    """即使從未間斷（每條 flow 間隔都在 idle_timeout 內），總長度超過
    active_timeout 也要強制切出新事件——避免 mirai-dos 這種持續數小時的
    洪水變成一個沒有時間軸意義的超長事件。"""
    flows = pd.DataFrame([_flow("d1", t, t + 1, "Impact") for t in range(0, 30, 5)])
    events = aggregate_events(flows, "l1_label", idle_timeout=100, active_timeout=12)
    assert len(events) > 1, "全部間隔都在 idle_timeout 內，只有 active_timeout 能切出多個事件"


def test_aggregate_events_excludes_benign_and_ranks_severity():
    flows = pd.DataFrame([
        _flow("d1", 0, 1, "Benign"),
        _flow("d1", 1, 2, "Reconnaissance"),
        _flow("d1", 2, 3, "Impact"),
    ])
    events = aggregate_events(flows, "l1_label", idle_timeout=10, active_timeout=1000)
    assert len(events) == 1
    assert events.iloc[0]["n_flows"] == 2, "Benign flow 不能被計入事件"
    assert events.iloc[0]["severity"] == "Impact", "嚴重度應取窗內最高攻擊類別層級"


def test_match_events_self_match_is_100_percent():
    """自己跟自己比對必須全部 matched，且列數要等於事件數——這條釘住
    window_end 相等時的 tie-breaking。"""
    flows = pd.DataFrame([
        _flow("d1", 0, 1, "Impact"),
        _flow("d1", 100, 101, "Reconnaissance"),
        _flow("d2", 0, 1, "Credential Access"),
    ])
    events = aggregate_events(flows, "l1_label", idle_timeout=10, active_timeout=1000)
    matched = match_events(events, events)
    assert len(matched) == len(events)
    assert (matched["outcome"] == "matched").all()


def test_match_events_missed_false_alarm_mismatched():
    true_events = pd.DataFrame([
        {"device": "d1", "window_start": 0, "window_end": 1, "severity": "Impact"},
        {"device": "d1", "window_start": 100, "window_end": 101, "severity": "Reconnaissance"},
    ])
    pred_events = pd.DataFrame([
        {"device": "d1", "window_start": 0, "window_end": 1, "severity": "Reconnaissance"},   # 重疊但嚴重度不同
        {"device": "d1", "window_start": 500, "window_end": 501, "severity": "Impact"},        # 誤報
    ])
    m = match_events(true_events, pred_events)
    outcomes = sorted(m["outcome"])
    assert outcomes == ["false_alarm", "mismatched", "missed"]
