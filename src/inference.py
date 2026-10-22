"""端到端推論：pcap → 觀測窗 → 三模態特徵 → 模型 → L3 事件。

scripts/predict.py（demo CLI）與 scripts/efficiency.py（E5 效能量測）共用這一份，
各階段與訓練時走同一套函式（flow.py / context.py / dataset.py / train.py），
不另寫推論專用的特徵程式碼，避免訓練與部署的特徵定義漂移。
"""
from __future__ import annotations

import time

import pandas as pd

from src.events.aggregate import aggregate_events
from src.features.dataset import build_feature_artifacts
from src.labels import L1_CLASSES
from src.models.train import predict, to_tensors
from src.pcap.context import compute_host_context
from src.pcap.flow import FlowBuilder, add_window_index


class _NoLabels:
    """部署時沒有官方標籤：FlowBuilder 的 labeller 介面一律回 None。"""

    @staticmethod
    def lookup(ts, length):
        return None


def windows_from_pcaps(pcaps: list[str], device: str, scenario: str = "live") -> pd.DataFrame:
    builder = FlowBuilder(device, scenario, _NoLabels())
    for p in pcaps:
        builder.feed_pcap(p)
    return add_window_index(pd.DataFrame(builder.flush()))


def run(pcaps: list[str], device: str, ckpt: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """回傳 (逐窗預測, 預測事件, 統計)：統計含各階段耗時秒數與 n_windows / n_packets。"""
    t = {}
    t0 = time.perf_counter()
    df = windows_from_pcaps(pcaps, device)
    t["parse_windows"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = df.join(compute_host_context(df, df["flow_id"]).add_prefix("ctx_"), on="flow_id")
    art = build_feature_artifacts(df)
    t["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    split = {"f_stats": art["f_stats"].to_numpy(dtype="float32"), "f_seq": art["f_seq"],
             "f_seq_len": art["f_seq_len"], "p": art["p"], "c": art["c"].to_numpy(dtype="float32")}
    out = predict(ckpt["model"], to_tensors(split, ckpt["scalers"], ckpt["device"]))
    t["model"] = time.perf_counter() - t0

    pred = art["manifest"].copy()
    pred["pred_l1_label"] = [L1_CLASSES[i] for i in out["tactic_prob"].argmax(axis=1)]
    pred["attack_score"] = out["det_prob"]
    t0 = time.perf_counter()
    events = aggregate_events(pred, label_col="pred_l1_label")
    t["events"] = time.perf_counter() - t0
    t["n_windows"] = len(pred)
    t["n_packets"] = int(df["n_pkts"].sum())
    return pred, events, t
