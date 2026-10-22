"""L4-lite：固定模板的事件通報產生器。只做「L3 事件 JSON → 人看得懂的文字」
這一步，純模板串接，不含檢索、不含生成式模型。

共用邏輯統一在此，禁止各腳本各自實作。
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.labels import L1_TO_ATTACK

SEVERITY_RANK_LABEL = {
    "Impact": "高",
    "Command & Control": "高",
    "Credential Access": "中",
    "Tool Transfer": "中",
    "Reconnaissance": "低",
}


def _fmt_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_event(event: dict) -> str:
    """把 aggregate_events() 產出的單筆事件 dict 轉成一段 Markdown 通報文字。"""
    device = event["device"]
    severity = event["severity"]
    level = SEVERITY_RANK_LABEL.get(severity, "未知")
    tactics = event["tactics"]
    counts = event["flow_counts_by_tactic"]

    rows = "\n".join(
        f"| {t} | {L1_TO_ATTACK.get(t, '—')} | {counts.get(t, 0)} |"
        for t in tactics
    )

    return (
        f"### 事件：{device}\n"
        f"- 時間範圍：{_fmt_ts(event['window_start'])} – {_fmt_ts(event['window_end'])}\n"
        f"- 嚴重度：**{severity}**（風險等級：{level}）\n"
        f"- 涉及片段數：{event['n_flows']}\n\n"
        f"| 攻擊類別 | ATT&CK 技術 | 片段數 |\n"
        f"|---|---|---:|\n"
        f"{rows}\n"
    )


def render_report(events: list[dict], title: str) -> str:
    """把多筆事件組成一份完整報告（含摘要表），仍是純模板串接。"""
    if not events:
        return f"# {title}\n\n（此區間無非 Benign 事件）\n"

    by_device: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for e in events:
        by_device[e["device"]] = by_device.get(e["device"], 0) + 1
        by_severity[e["severity"]] = by_severity.get(e["severity"], 0) + 1

    summary_rows = "\n".join(f"| {d} | {n} |" for d, n in sorted(by_device.items()))
    severity_rows = "\n".join(
        f"| {s} | {SEVERITY_RANK_LABEL.get(s, '未知')} | {n} |"
        for s, n in sorted(by_severity.items(), key=lambda kv: -kv[1])
    )

    header = (
        f"# {title}\n\n"
        f"共 {len(events)} 筆事件，涉及 {len(by_device)} 台裝置。\n\n"
        f"**依嚴重度統計**\n\n| 嚴重度 | 風險等級 | 事件數 |\n|---|---|---:|\n{severity_rows}\n\n"
        f"**依裝置統計**\n\n| 裝置 | 事件數 |\n|---|---:|\n{summary_rows}\n\n"
        f"---\n\n"
    )
    body = "\n".join(render_event(e) for e in events)
    return header + body
