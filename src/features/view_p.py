"""View P：連線開頭的 payload byte 序列。

每個觀測窗用所屬連線開頭的 PAYLOAD_BYTES 個 bytes（client -> server，flow.py
的 conn_payload，同連線各窗共用），byte 直接當 token。TLS handshake、Telnet
登入都只出現在連線開頭；v1 用各窗自己的開頭，長連線後段的窗只看得到密文中段。

身分遮罩（延伸到 payload）：payload 裡可能夾帶位址與主機名，
這些是身分資訊而非行為。以下位置的 byte 換成 MASK_TOKEN，保留長度與位置結構：
- 該窗 src/dst IPv4 的 4-byte 二進位表示
- 任何點分十進位 IPv4 字串（Telnet 指令、HTTP 內容常見）
- TLS ClientHello 的 SNI 主機名
- HTTP Host 標頭的值

vocab = 259（0-255 + PAD + EOS + MASK）。
"""
from __future__ import annotations

import re
import socket

import numpy as np
import pandas as pd

import config

PAD_TOKEN = 256
EOS_TOKEN = 257
MASK_TOKEN = 258
VOCAB_SIZE = 259

_DOTTED_IP = re.compile(rb"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
_HTTP_HOST = re.compile(rb"(?im)^host:[ \t]*([^\r\n]+)")


def _tls_sni_span(p: bytes) -> tuple[int, int] | None:
    """若 p 以 TLS ClientHello 開頭，回傳 SNI 主機名的 [start, end)；否則 None。"""
    try:
        if len(p) < 43 or p[0] != 0x16 or p[1] != 0x03 or p[5] != 0x01:
            return None
        i = 5 + 4 + 2 + 32                        # handshake 標頭、版本、random
        i += 1 + p[i]                              # session id
        i += 2 + int.from_bytes(p[i:i + 2], "big")  # cipher suites
        i += 1 + p[i]                              # compression methods
        ext_end = i + 2 + int.from_bytes(p[i:i + 2], "big")
        i += 2
        while i + 4 <= min(ext_end, len(p)):
            etype = int.from_bytes(p[i:i + 2], "big")
            elen = int.from_bytes(p[i + 2:i + 4], "big")
            if etype == 0:                         # server_name
                j = i + 4 + 2                      # 跳過 server_name_list 長度
                name_len = int.from_bytes(p[j + 1:j + 3], "big")
                start = j + 3
                return start, min(start + name_len, len(p))
            i += 4 + elen
    except IndexError:
        return None
    return None


def identity_mask(payload: bytes, ips=()) -> np.ndarray:
    """回傳與 payload 等長的 bool 陣列，True = 該 byte 屬於身分資訊。"""
    mask = np.zeros(len(payload), dtype=bool)
    for ip in ips:
        try:
            raw = socket.inet_aton(ip)
        except OSError:
            continue
        start = payload.find(raw)
        while start != -1:
            mask[start:start + 4] = True
            start = payload.find(raw, start + 1)
    for m in _DOTTED_IP.finditer(payload):
        mask[m.start():m.end()] = True
    for m in _HTTP_HOST.finditer(payload):
        mask[m.start(1):m.end(1)] = True
    span = _tls_sni_span(payload)
    if span is not None:
        mask[span[0]:span[1]] = True
    return mask


def _encode_payload(payload: bytes, length: int, ips=()) -> np.ndarray:
    seq = np.full(length, PAD_TOKEN, dtype=np.int16)
    n = min(len(payload), length)
    if n:
        seq[:n] = np.frombuffer(payload[:n], dtype=np.uint8)
        seq[:n][identity_mask(payload[:n], ips)] = MASK_TOKEN
    if n < length:
        seq[n] = EOS_TOKEN  # 比視窗短時明確標記結尾，也保證不會整列都是 PAD
    return seq


def build_view_p(df: pd.DataFrame, length: int = config.PAYLOAD_BYTES) -> np.ndarray:
    """回傳 (n, length) int16 token 矩陣。"""
    return np.stack([_encode_payload(p, length, (s, d))
                     for p, s, d in zip(df["conn_payload"], df["src_ip"], df["dst_ip"])])
