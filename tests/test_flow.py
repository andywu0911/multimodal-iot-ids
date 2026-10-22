"""src/pcap/flow.py 幾個容易錯、錯了也不會拋例外的地方：
1. _flow_key() 的方向對稱性——a->b 與 b->a 兩個封包必須歸到同一條 flow，
   這是整個 flow 聚合邏輯的地基，錯了會讓每條連線的雙向封包各自散成兩條
   flow，統計量全部跑掉但不會有任何錯誤訊息。
2. _unwrap_gre() 的巢狀 GRE 剝殼——Mirai GRE Flooding 若剝殼失敗會被整個
   聚合成一條巨大 flow（見 src/pcap/flow.py 開頭的 docstring），同樣是
   靜默失敗。
3. iter_packets() 的 stats 計數——解析失敗與非 IPv4 封包本來就會被靜默跳過
   （完整性檢查得到「截斷封包 0、損壞檔 0」，這個計數器是唯一能驗證這句話
   的地方），計數邏輯本身若算錯也不會有任何錯誤訊息。
"""
from pathlib import Path
from types import SimpleNamespace

import dpkt

from src.pcap.flow import _flow_key, _unwrap_gre, iter_packets


def _pkt(src_ip, src_port, dst_ip, dst_port, proto=6):
    return SimpleNamespace(src_ip=src_ip, src_port=src_port,
                            dst_ip=dst_ip, dst_port=dst_port, proto=proto)


def test_flow_key_symmetric_5tuple():
    fwd = _pkt("10.0.0.1", 1234, "10.0.0.2", 80)
    bwd = _pkt("10.0.0.2", 80, "10.0.0.1", 1234)
    key_fwd, is_fwd_fwd = _flow_key(fwd, "5tuple")
    key_bwd, is_fwd_bwd = _flow_key(bwd, "5tuple")

    assert key_fwd == key_bwd, "a->b 與 b->a 必須歸到同一條 flow"
    assert is_fwd_fwd != is_fwd_bwd, "兩個方向的 is_forward 必須互斥"


def test_flow_key_symmetric_hostpair():
    """hostpair 模式（埠號降級為特徵）同樣要維持方向對稱性。"""
    fwd = _pkt("10.0.0.1", 1234, "10.0.0.2", 80)
    bwd = _pkt("10.0.0.2", 999, "10.0.0.1", 1234)  # 埠號不對稱也不影響 hostpair 分組
    key_fwd, _ = _flow_key(fwd, "hostpair")
    key_bwd, _ = _flow_key(bwd, "hostpair")
    assert key_fwd == key_bwd


def test_flow_key_different_connections_differ():
    a = _pkt("10.0.0.1", 1234, "10.0.0.2", 80)
    b = _pkt("10.0.0.1", 1234, "10.0.0.3", 80)  # 不同目的地，應為不同連線
    key_a, _ = _flow_key(a, "5tuple")
    key_b, _ = _flow_key(b, "5tuple")
    assert key_a != key_b


def _build_gre_in_ip(inner_src="10.0.0.5", inner_dst="10.0.0.6") -> dpkt.ip.IP:
    """組一個 outer_ip(proto=GRE) -> GRE -> inner_ip(proto=UDP) 的封包，
    序列化再重新 parse，模擬 iter_packets() 從 wire bytes 讀進來的路徑
    （不能只在記憶體裡串物件，dpkt 的 .data 型別是 parse 時才決定的）。"""
    import socket

    inner_ip = dpkt.ip.IP(
        src=socket.inet_aton(inner_src), dst=socket.inet_aton(inner_dst),
        p=dpkt.ip.IP_PROTO_UDP,
        data=dpkt.udp.UDP(sport=1111, dport=2222, data=b"payload"),
    )
    inner_ip.len = len(inner_ip)

    gre = dpkt.gre.GRE(flags=0, p=0x0800, data=bytes(inner_ip))

    outer_ip = dpkt.ip.IP(
        src=socket.inet_aton("192.168.1.1"), dst=socket.inet_aton("192.168.1.2"),
        p=dpkt.ip.IP_PROTO_GRE, data=bytes(gre),
    )
    outer_ip.len = len(outer_ip)
    return dpkt.ip.IP(bytes(outer_ip))  # 重新 parse


def test_unwrap_gre_reaches_inner_ip():
    outer = _build_gre_in_ip()
    inner = _unwrap_gre(outer)
    assert inner.p == dpkt.ip.IP_PROTO_UDP
    assert isinstance(inner.data, dpkt.udp.UDP)


def test_unwrap_gre_noop_on_non_gre_packet():
    """非 GRE 封包要原樣回傳，不能誤剝殼。"""
    import socket
    ip = dpkt.ip.IP(
        src=socket.inet_aton("10.0.0.1"), dst=socket.inet_aton("10.0.0.2"),
        p=dpkt.ip.IP_PROTO_UDP, data=dpkt.udp.UDP(sport=1, dport=2, data=b"x"),
    )
    ip.len = len(ip)
    reparsed = dpkt.ip.IP(bytes(ip))
    out = _unwrap_gre(reparsed)
    assert out is reparsed


def _write_pcap(path: Path, packets: list[bytes]) -> None:
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for i, buf in enumerate(packets):
            writer.writepkt(buf, ts=float(i))


def test_iter_packets_stats_counts_parse_errors_and_non_ipv4(tmp_path):
    """一個好封包 + 一個太短無法解析的 buffer + 一個 IPv6 封包，驗證
    stats 計數跟實際 yield 出來的封包數一致——這是「截斷封包 0、
    損壞檔 0」唯一能被驗證的地方，計數邏輯本身要有測試釘住。"""
    import socket

    good = dpkt.ethernet.Ethernet(
        src=b"\x01" * 6, dst=b"\x02" * 6, type=dpkt.ethernet.ETH_TYPE_IP,
        data=dpkt.ip.IP(src=socket.inet_aton("10.0.0.1"), dst=socket.inet_aton("10.0.0.2"),
                         p=dpkt.ip.IP_PROTO_UDP, data=dpkt.udp.UDP(sport=1, dport=2, data=b"x")),
    )
    too_short = b"\x00" * 4  # 不足 14 bytes 的 Ethernet header，觸發解析例外
    ipv6 = dpkt.ethernet.Ethernet(
        src=b"\x03" * 6, dst=b"\x04" * 6, type=dpkt.ethernet.ETH_TYPE_IP6,
        data=dpkt.ip6.IP6(src=b"\x00" * 16, dst=b"\x00" * 16, data=b"y"),
    )

    pcap_path = tmp_path / "mixed.pcap"
    _write_pcap(pcap_path, [bytes(good), too_short, bytes(ipv6)])

    stats: dict = {}
    packets = list(iter_packets(str(pcap_path), stats=stats))

    assert len(packets) == 1, "只有那個合法的 IPv4 封包應該被 yield"
    assert stats["n_parse_errors"] == 1
    assert stats["n_non_ipv4"] == 1


def test_iter_packets_stats_none_by_default():
    """不傳 stats 就不計數、也不能拋例外（直接呼叫 iter_packets() 不帶 stats 的舊用法）。"""
    # 不需要真的讀檔——只驗證函式簽章與預設值不會強制要求 stats。
    import inspect
    sig = inspect.signature(iter_packets)
    assert sig.parameters["stats"].default is None
