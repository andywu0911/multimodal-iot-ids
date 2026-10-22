"""L3 事件聚合。flow 預測 -> 依裝置聚合成資安事件。
輸出：受影響裝置、涵蓋的 L1 攻擊類別集合、時間軸、嚴重度。不做完整攻擊鏈階段定位。

事件邊界採 idle/active timeout（跟 src/pcap/flow.py 的 flow 邊界同一套機制，
只是套用在「裝置 x 非 Benign flow 序列」而非「五元組 x 封包序列」）：同裝置
連續非 Benign flow，只要跟上一條的間隔 <= idle_timeout 就併入同一事件；間隔
一拉開就起新事件。另設 active_timeout 上限，避免像 mirai-dos 這種持續數小時
不間斷的洪水把整段時間壓成一個沒有時間軸意義的超長事件。

不用固定時鐘窗（如 device x 300s bucket）的理由：固定窗會把一個持續攻擊硬切
成兩個事件（攻擊剛好跨過窗界），也可能把兩起不相關的行為湊進同一窗——這是
alert-correlation 文獻裡 fixed sliding window 的已知缺陷（Debar & Wespi 2001
談的 duplicate/consequence 關聯、以及 SIEM 常見的 window-boundary 問題）。
idle/active timeout 是這個問題最小必要的修法，且跟 L0 用同一套機制，不引入
攻擊語意關聯（attack graph）等超出範圍的複雜度。

共用邏輯統一在此，禁止各腳本各自實作。
"""
from __future__ import annotations

import pandas as pd

import config
from src.labels import TACTIC_SEVERITY_ORDER, BENIGN_LABEL

_SEVERITY_RANK = {t: i for i, t in enumerate(TACTIC_SEVERITY_ORDER)}

EVENT_COLUMNS = [
    "device", "window_start", "window_end", "tactics", "severity",
    "n_flows", "flow_counts_by_tactic",
]


def _severity(tactics: set[str]) -> str | None:
    """tactics 集合中攻擊類別嚴重度最高者（Impact 最高，見 TACTIC_SEVERITY_ORDER）。
    空集合（全 Benign）回傳 None。"""
    present = [t for t in tactics if t in _SEVERITY_RANK]
    if not present:
        return None
    return min(present, key=lambda t: _SEVERITY_RANK[t])


def aggregate_events(
    flows: pd.DataFrame, label_col: str,
    idle_timeout: float = config.IDLE_TIMEOUT,
    active_timeout: float = config.ACTIVE_TIMEOUT,
) -> pd.DataFrame:
    """flows 需含欄位：device, start_ts, end_ts, <label_col>（L1 標籤，字串）。

    label_col 可指向真實標籤或模型預測標籤——用同一函式聚合兩者，才能公平
    比較「用預測建出的事件」跟「用真值建出的事件」差多少。Benign flow 一律
    排除；同裝置的連續非 Benign flow 依 idle/active timeout 併成事件（見模組
    docstring）。
    """
    df = flows[flows[label_col] != BENIGN_LABEL].copy()
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = df.sort_values("start_ts")

    rows = []
    for device, g in df.groupby("device", sort=False):
        session_start = session_last_end = None
        event_rows: list[dict] = []

        def flush():
            if not event_rows:
                return
            eg = pd.DataFrame(event_rows)
            tactics = set(eg[label_col].unique())
            rows.append({
                "device": device,
                "window_start": float(eg["start_ts"].min()),
                "window_end": float(eg["end_ts"].max()),
                "tactics": sorted(tactics, key=lambda t: _SEVERITY_RANK.get(t, len(_SEVERITY_RANK))),
                "severity": _severity(tactics),
                "n_flows": len(eg),
                "flow_counts_by_tactic": eg[label_col].value_counts().to_dict(),
            })

        for _, flow in g.iterrows():
            if (session_start is not None
                    and flow["start_ts"] - session_last_end <= idle_timeout
                    and flow["start_ts"] - session_start <= active_timeout):
                event_rows.append(flow)
            else:
                flush()
                event_rows = [flow]
                session_start = flow["start_ts"]
            session_last_end = max(session_last_end or flow["end_ts"], flow["end_ts"])
        flush()

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return events.sort_values(["device", "window_start"]).reset_index(drop=True)


def match_events(true_events: pd.DataFrame, pred_events: pd.DataFrame) -> pd.DataFrame:
    """真值/預測事件的邊界各自獨立（各自的 idle/active timeout 依各自的標籤序列
    計算），不再保證同一 window_start 對得上，因此改用時間區間重疊配對，而非
    固定窗那版的 (device, window_start) 精確匹配。

    每個裝置內，true/pred 事件各自不重疊（同一函式聚合出來的邊界互斥），因此
    用雙指標掃描即可：區間有重疊就配對比較嚴重度，無重疊的 true 事件算漏報，
    無重疊的 pred 事件算誤報。

    回傳每列一組配對結果：device、true/pred 的 window、true/pred 的 severity、
    match（bool，重疊配對且嚴重度相同）、outcome（"matched"/"missed"/"false_alarm"）。
    """
    rows = []
    devices = set(true_events["device"]) | set(pred_events["device"])
    for device in devices:
        t = true_events[true_events.device == device].sort_values("window_start").to_dict("records")
        p = pred_events[pred_events.device == device].sort_values("window_start").to_dict("records")
        i = j = 0
        while i < len(t) and j < len(p):
            te, pe = t[i], p[j]
            if te["window_end"] < pe["window_start"]:
                rows.append({"device": device, "true_severity": te["severity"], "pred_severity": None,
                             "outcome": "missed"})
                i += 1
            elif pe["window_end"] < te["window_start"]:
                rows.append({"device": device, "true_severity": None, "pred_severity": pe["severity"],
                             "outcome": "false_alarm"})
                j += 1
            else:
                match = te["severity"] == pe["severity"]
                rows.append({"device": device, "true_severity": te["severity"],
                             "pred_severity": pe["severity"], "outcome": "matched" if match else "mismatched"})
                if te["window_end"] == pe["window_end"]:
                    i += 1
                    j += 1
                elif te["window_end"] < pe["window_end"]:
                    i += 1
                else:
                    j += 1
        for te in t[i:]:
            rows.append({"device": device, "true_severity": te["severity"], "pred_severity": None,
                         "outcome": "missed"})
        for pe in p[j:]:
            rows.append({"device": device, "true_severity": None, "pred_severity": pe["severity"],
                         "outcome": "false_alarm"})
    return pd.DataFrame(rows, columns=["device", "true_severity", "pred_severity", "outcome"])
