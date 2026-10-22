"""View C：主機情境向量。原始統計由 src/pcap/context.py 在抽樣階段算好
（sample_flows.py 存成 ctx_* 欄位），這裡只做尺度轉換：計數類取 log1p，
比例類保持 [0, 1]，另補兩個與主機規模無關的比值（每個窗對應幾個相異目的
位址 / 目的埠），讓「掃描打很多目標、洪水集中打一個」這種行為差異不依賴
特定裝置的流量大小。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def build_view_c(df: pd.DataFrame, windows=config.CONTEXT_WINDOWS) -> pd.DataFrame:
    ctx = df[[c for c in df.columns if c.startswith("ctx_")]].astype(np.float64)
    out = {}
    for c in ctx.columns:
        name = c[len("ctx_"):]
        out[name] = ctx[c] if name.endswith("_frac") else np.log1p(ctx[c])
    for w in windows:
        n = ctx[f"ctx_src_w{int(w)}_n_flows"].clip(lower=1)
        out[f"src_w{int(w)}_dip_per_flow"] = ctx[f"ctx_src_w{int(w)}_uniq_dip"] / n
        out[f"src_w{int(w)}_dport_per_flow"] = ctx[f"ctx_src_w{int(w)}_uniq_dport"] / n
    return pd.DataFrame(out, index=df.index)
