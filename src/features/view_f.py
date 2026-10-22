"""View F：觀測窗的流模態，兩個部分。

1. 統計向量（build_view_f_stats，36 維）：封包數、位元組、方向平衡、長度與 IAT
   統計、協定 one-hot、TCP flag 計數/比例。
2. 封包序列（build_view_f_seq，SEQ_LEN × 9）：前 N 包逐包的長度、方向、IAT、
   6 個 TCP flag，保留統計向量丟掉的「順序」。做法沿用加密流量分類以封包長度
   與方向序列建模的主流路線（FS-Net, Liu et al., INFOCOM 2019）。

捷徑隔離：埠號、IP、MAC、checksum、裝置名稱一律不進入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# dpkt TCP flag bits
_FLAG_BITS = {"fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08, "ack": 0x10, "urg": 0x20}

PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1
_EPS = 1e-9
# 單一封包的 flow duration=0（無法定義速率）：以最小合理計時解析度（1 ms）
# 當地板，避免除以 ~1e-9 時把 pkt_rate/byte_rate 炸成 1e9~1e14 量級的離群值。
_DURATION_FLOOR = 1e-3

FEATURE_COLS_BASE = [
    "n_pkts", "duration", "pkt_rate", "byte_rate",
    "fwd_pkts", "bwd_pkts", "fwd_bwd_pkt_balance",
    "fwd_bytes", "bwd_bytes", "fwd_bwd_byte_balance",
    "len_mean", "len_std", "len_min", "len_max", "len_median",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_median",
    "proto_is_tcp", "proto_is_udp", "proto_is_icmp", "proto_is_other",
] + [f"tcp_{name}_count" for name in _FLAG_BITS] + [f"tcp_{name}_ratio" for name in _FLAG_BITS]


def _stats(arr: np.ndarray) -> tuple[float, float, float, float, float]:
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max()), float(np.median(arr))


def _row_features(row: pd.Series) -> dict:
    lengths = np.asarray(row["pkt_lengths"], dtype=np.float64)
    iats = np.asarray(row["pkt_iat"], dtype=np.float64)
    iats = iats[1:] if iats.size > 1 else iats[:0]  # 第一個必為 0（無前一封包），排除
    flags = np.asarray(row["tcp_flags"], dtype=np.int64)

    len_mean, len_std, len_min, len_max, len_med = _stats(lengths)
    iat_mean, iat_std, iat_min, iat_max, iat_med = _stats(iats)

    duration = max(float(row["duration"]), _DURATION_FLOOR)
    n_pkts = int(row["n_pkts"])
    fwd_pkts, bwd_pkts = int(row["pkt_count_fwd"]), int(row["pkt_count_bwd"])
    fwd_bytes, bwd_bytes = int(row["byte_count_fwd"]), int(row["byte_count_bwd"])
    total_bytes = fwd_bytes + bwd_bytes

    feats = {
        "n_pkts": n_pkts,
        "duration": float(row["duration"]),
        "pkt_rate": n_pkts / duration,
        "byte_rate": total_bytes / duration,
        "fwd_pkts": fwd_pkts,
        "bwd_pkts": bwd_pkts,
        # 有界的方向平衡指標（[-1, 1]），取代原本會在 bwd=0 時炸到極大值的除法比例
        "fwd_bwd_pkt_balance": (fwd_pkts - bwd_pkts) / (fwd_pkts + bwd_pkts + _EPS),
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "fwd_bwd_byte_balance": (fwd_bytes - bwd_bytes) / (fwd_bytes + bwd_bytes + _EPS),
        "len_mean": len_mean, "len_std": len_std, "len_min": len_min,
        "len_max": len_max, "len_median": len_med,
        "iat_mean": iat_mean, "iat_std": iat_std, "iat_min": iat_min,
        "iat_max": iat_max, "iat_median": iat_med,
        "proto_is_tcp": float(row["proto"] == PROTO_TCP),
        "proto_is_udp": float(row["proto"] == PROTO_UDP),
        "proto_is_icmp": float(row["proto"] == PROTO_ICMP),
        "proto_is_other": float(row["proto"] not in (PROTO_TCP, PROTO_UDP, PROTO_ICMP)),
    }
    n_flag_pkts = max(len(flags), 1)
    for name, bit in _FLAG_BITS.items():
        count = int(np.sum((flags & bit) != 0)) if flags.size else 0
        feats[f"tcp_{name}_count"] = count
        feats[f"tcp_{name}_ratio"] = count / n_flag_pkts
    return feats


def build_view_f_stats(df: pd.DataFrame) -> pd.DataFrame:
    """回傳統計向量（純數值，不含 label/id），index 對齊 df。"""
    feats = df.apply(_row_features, axis=1, result_type="expand")[FEATURE_COLS_BASE]
    feats.index = df.index
    return feats


SEQ_CHANNELS = ["log_len", "dir", "log_iat_ms"] + [f"flag_{n}" for n in _FLAG_BITS]


def build_view_f_seq(df: pd.DataFrame, seq_len: int = config.SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (seq, lengths)：seq 為 (n, seq_len, 9) float32，不足 seq_len 的位置補 0；
    lengths 為每列實際封包數（上限 seq_len），供編碼器做 padding mask。

    dir：+1 = 發起方送出、-1 = 回應方送出。長度與 IAT 取 log1p 壓縮尺度差異。
    負 IAT 截為 0：少數 pcap 的封包時間戳前後顛倒（S3 抽樣後 37 個窗），是擷取假象，
    不截的話 IAT < −1 ms 時 log1p 會產生 NaN（實際發生在 1 個 CoAP 窗）。
    """
    n = len(df)
    seq = np.zeros((n, seq_len, len(SEQ_CHANNELS)), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    bits = np.array(list(_FLAG_BITS.values()), dtype=np.int64)
    for i, (ln, iat, d, fl) in enumerate(zip(df["pkt_lengths"], df["pkt_iat"], df["pkt_dir"], df["tcp_flags"])):
        k = min(len(ln), seq_len)
        lengths[i] = k
        if k == 0:
            continue
        seq[i, :k, 0] = np.log1p(np.asarray(ln[:k], dtype=np.float64))
        seq[i, :k, 1] = np.where(np.asarray(d[:k]) == 1, 1.0, -1.0)
        seq[i, :k, 2] = np.log1p(np.maximum(np.asarray(iat[:k], dtype=np.float64), 0.0) * 1000.0)
        seq[i, :k, 3:] = (np.asarray(fl[:k], dtype=np.int64)[:, None] & bits) != 0
    return seq, lengths
