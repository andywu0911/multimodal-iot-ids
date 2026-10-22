"""主機情境（View C 的原始統計）。情境計算統一在此。

單一觀測窗看不出「這台主機正在做什麼」：Mirai TCP 洪水與 TCP 掃描都是大量
1–3 包的短窗，差別在主機層級——掃描打向大量不同目的位址、目的埠集中，
洪水集中打單一受害者。這裡對每個目標窗，彙整同一來源主機（與同一目的主機）
在它結束前 W 秒內「已結束」的全部觀測窗（含自己），多時間尺度 W 取自
config.CONTEXT_WINDOWS，概念沿用 Kitsune AfterImage（Mirsky et al., NDSS 2018）。

因果性：只用 end_ts ≤ 目標窗 end_ts 的窗，對應 IDS 在該窗結束時刻能取得的
資訊；未來的窗不影響結果（tests/test_context.py 釘住這個不變式）。

捷徑隔離：IP 與埠號只當分組鍵與「相異個數」的計數對象，數值本身不進特徵。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

CONTEXT_COLUMNS = ["flow_id", "device", "scenario", "proto", "src_ip", "dst_ip", "dst_port",
                   "end_ts", "n_pkts", "byte_count_fwd", "byte_count_bwd", "pkt_count_bwd"]


def _feature_names(windows) -> list[str]:
    names = []
    for w in windows:
        p = f"src_w{int(w)}_"
        names += [p + s for s in ("n_flows", "pkts", "bytes", "uniq_dip", "uniq_dport",
                                  "short_frac", "noreply_frac", "tcp_frac", "udp_frac")]
        q = f"dst_w{int(w)}_"
        names += [q + s for s in ("n_flows", "uniq_sip", "pkts")]
    return names


def _windowed(df: pd.DataFrame, host_col: str, target: np.ndarray, windows, fill) -> None:
    """對 host_col 分組、依 end_ts 排序後，逐目標窗以二分搜尋取出 (t-W, t] 區段並呼叫 fill。"""
    order = np.lexsort((df["end_ts"].to_numpy(), df[host_col].to_numpy()))
    host = df[host_col].to_numpy()[order]
    ts = df["end_ts"].to_numpy()[order]
    is_target = target[order]
    bounds = np.flatnonzero(np.diff(host)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(host)]):
        tgt = np.flatnonzero(is_target[s:e])
        if len(tgt) == 0:
            continue
        gts = ts[s:e]
        for j in tgt:
            hi = s + np.searchsorted(gts, gts[j], side="right")
            for w in windows:
                lo = s + np.searchsorted(gts, gts[j] - w, side="right")
                fill(order[s + j], order[lo:hi], w)


def compute_host_context(full: pd.DataFrame, target_ids, windows=config.CONTEXT_WINDOWS) -> pd.DataFrame:
    """回傳以 flow_id 為索引、欄位為 _feature_names(windows) 的原始情境統計。

    full：同一資料集的「全部」觀測窗（抽樣前），情境必須從完整歷史計算；
    target_ids：要輸出情境的觀測窗（通常是抽樣後的樣本）。
    """
    df = full[CONTEXT_COLUMNS].reset_index(drop=True)
    scope = df["device"] + "|" + df["scenario"] + "|"
    df["_src_host"] = pd.factorize(scope + df["src_ip"])[0]
    df["_dst_host"] = pd.factorize(scope + df["dst_ip"])[0]
    dip = pd.factorize(df["dst_ip"])[0]
    sip = pd.factorize(df["src_ip"])[0]
    dport = df["dst_port"].to_numpy()
    pkts = df["n_pkts"].to_numpy()
    nbytes = (df["byte_count_fwd"] + df["byte_count_bwd"]).to_numpy()
    short = (pkts <= 3).astype(np.float64)
    noreply = (df["pkt_count_bwd"].to_numpy() == 0).astype(np.float64)
    tcp = (df["proto"].to_numpy() == 6).astype(np.float64)
    udp = (df["proto"].to_numpy() == 17).astype(np.float64)

    target = df["flow_id"].isin(set(target_ids)).to_numpy()
    names = _feature_names(windows)
    col = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(df), len(names)), dtype=np.float64)

    def fill_src(i, sl, w):
        p = f"src_w{int(w)}_"
        o = out[i]
        o[col[p + "n_flows"]] = len(sl)
        o[col[p + "pkts"]] = pkts[sl].sum()
        o[col[p + "bytes"]] = nbytes[sl].sum()
        o[col[p + "uniq_dip"]] = len(np.unique(dip[sl]))
        o[col[p + "uniq_dport"]] = len(np.unique(dport[sl]))
        o[col[p + "short_frac"]] = short[sl].mean()
        o[col[p + "noreply_frac"]] = noreply[sl].mean()
        o[col[p + "tcp_frac"]] = tcp[sl].mean()
        o[col[p + "udp_frac"]] = udp[sl].mean()

    def fill_dst(i, sl, w):
        q = f"dst_w{int(w)}_"
        o = out[i]
        o[col[q + "n_flows"]] = len(sl)
        o[col[q + "uniq_sip"]] = len(np.unique(sip[sl]))
        o[col[q + "pkts"]] = pkts[sl].sum()

    _windowed(df, "_src_host", target, windows, fill_src)
    _windowed(df, "_dst_host", target, windows, fill_dst)
    return pd.DataFrame(out[target], columns=names, index=df.loc[target, "flow_id"].to_numpy())
