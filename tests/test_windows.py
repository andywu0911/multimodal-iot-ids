"""觀測窗與連線層級的不變式：錯了不會拋例外，只會安靜改變樣本與特徵的語意。

1. 連線開頭 payload 的承接：滿 K 切窗後，下一窗必須帶著連線開頭的 bytes
   （TLS handshake、Telnet 登入都只在開頭），但 idle timeout 後重開的
   新 session 不能繼承舊內容。
2. 每連線窗數上限：長連線不能主導樣本。
3. 主機情境的因果性：未來的窗不能影響現在的情境特徵。
"""
from types import SimpleNamespace

import pandas as pd

from src.pcap.context import compute_host_context
from src.pcap.flow import FlowBuilder, PacketRecord, add_window_index
from src.pcap.sample import cap_windows_per_connection


def _rec(ts, payload=b"", fwd=True):
    a, b = ("10.0.0.1", 40000), ("10.0.0.2", 443)
    (sip, sp), (dip, dp) = (a, b) if fwd else (b, a)
    return PacketRecord(ts=ts, length=60 + len(payload), proto=6, src_ip=sip, dst_ip=dip,
                        src_port=sp, dst_port=dp, src_mac="m1", dst_mac="m2",
                        ip_checksum=0, tcp_checksum=0, tcp_flags=0x18, payload=payload)


def _builder(**kw):
    labeller = SimpleNamespace(lookup=lambda ts, length: None)
    return FlowBuilder("dev", "scn", labeller, max_pkts=2, payload_bytes=8, **kw)


def test_conn_payload_carries_across_k_windows():
    fb = _builder()
    for i, p in enumerate([b"HELLO", b"", b"xx", b"yy"]):   # K=2 -> 兩個窗
        fb.add_packet(_rec(float(i), p))
    rows = sorted(fb.flush(), key=lambda r: r["start_ts"])
    assert len(rows) == 2
    assert rows[0]["conn_id"] == rows[1]["conn_id"]
    assert rows[0]["conn_payload"] == b"HELLO"
    assert rows[1]["payload"] == b"xxyy"                     # 窗自己的內容
    assert rows[1]["conn_payload"] == b"HELLOxxy"            # 連線開頭 + 補滿到 8 bytes


def test_conn_payload_not_carried_after_idle_gap():
    fb = _builder(idle_timeout=5.0)
    fb.add_packet(_rec(0.0, b"OLD"))
    fb.add_packet(_rec(1.0, b""))                            # 滿 K，準備承接
    fb.add_packet(_rec(100.0, b"NEW"))                       # 超過 idle 才回來 -> 新 session
    rows = sorted(fb.flush(), key=lambda r: r["start_ts"])
    assert rows[1]["conn_payload"] == b"NEW"


def test_window_index_and_cap_per_connection():
    df = pd.DataFrame({"conn_id": ["a"] * 30 + ["b"] * 3,
                       "start_ts": list(range(30)) + [0, 1, 2]})
    df = add_window_index(df)
    assert df.loc[df.conn_id == "a", "window_idx"].tolist() == list(range(30))
    capped = cap_windows_per_connection(df, cap=5)
    assert capped.groupby("conn_id").size().to_dict() == {"a": 5, "b": 3}


def _ctx_frame(n_future):
    rows = [dict(flow_id=f"f{i}", device="d", scenario="s", proto=6, src_ip="1.1.1.1",
                 dst_ip=f"9.9.9.{i}", dst_port=23, end_ts=float(i), n_pkts=1,
                 byte_count_fwd=60, byte_count_bwd=0, pkt_count_bwd=0) for i in range(5)]
    rows += [dict(flow_id=f"g{i}", device="d", scenario="s", proto=17, src_ip="1.1.1.1",
                  dst_ip="8.8.8.8", dst_port=53, end_ts=10.0 + i, n_pkts=50,
                  byte_count_fwd=5000, byte_count_bwd=5000, pkt_count_bwd=25) for i in range(n_future)]
    return pd.DataFrame(rows)


def test_host_context_is_causal():
    base = compute_host_context(_ctx_frame(0), ["f4"], windows=(60.0,))
    with_future = compute_host_context(_ctx_frame(20), ["f4"], windows=(60.0,))
    pd.testing.assert_frame_equal(base, with_future)
    assert base.loc["f4", "src_w60_n_flows"] == 5
    assert base.loc["f4", "src_w60_uniq_dip"] == 5
    assert base.loc["f4", "src_w60_uniq_dport"] == 1
