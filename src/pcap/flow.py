"""pcap -> flow 聚合。flow 聚合統一在此。

輸出是「一次性全量解析」的中間產物：每一列是一個「觀測窗」（同一連線上最多
K 個封包的片段），存最終統計與逐封包的長度 / IAT / 方向 / TCP flag 陣列，
讓 src/features/ 之後可以在不重碰 pcap 的前提下算出任何衍生統計。

兩層單位：
- 連線（conn_id）：(device, scenario, canonical 5-tuple)，切分與評估的獨立單位。
- 觀測窗（flow_id）：連線上的片段，滿 K 包、idle 或 active timeout 即結束。
  同一連線因滿 K 而接續開的下一窗，會繼承連線開頭的 payload（conn_payload），
  對應 IDS 實務上「追蹤連線狀態」會看到的內容；timeout 後重開則視為新 session。

GRE 剝殼（必須）：dpkt 對 ip.proto=47 會自動把 .data 解成 dpkt.gre.GRE，
其 .data 若為內層 dpkt.ip.IP 則再往下一層才是真正的五元組。實測
Mirai GRE Flooding 的內層目的位址逐封包隨機（偽造的外部 IP），
不剝殼會把整個攻擊聚合成一條巨大 flow。
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field

import dpkt

import config
from src.pcap.labeller import PacketLabeller
from src.labels import flow_label, to_l1, to_l2, UNKNOWN_LABEL


@dataclass
class PacketRecord:
    ts: float
    length: int
    proto: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    src_mac: str
    dst_mac: str
    ip_checksum: int
    tcp_checksum: int | None
    tcp_flags: int
    payload: bytes


def _mac_str(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw)


def _ip_str(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _unwrap_gre(ip_pkt: dpkt.ip.IP) -> dpkt.ip.IP:
    """若封包是 GRE 隧道（可能巢狀），剝到最內層 IP。非 GRE 就原樣回傳。"""
    depth = 0
    while ip_pkt.p == dpkt.ip.IP_PROTO_GRE and depth < 4:
        gre = ip_pkt.data
        if not isinstance(gre, dpkt.gre.GRE):
            break
        inner = gre.data
        if not isinstance(inner, dpkt.ip.IP):
            break
        ip_pkt = inner
        depth += 1
    return ip_pkt


def iter_packets(pcap_path: str, stats: dict | None = None):
    """走訪單一 pcap，yield 已剝殼、已解析的 PacketRecord（跳過非 IPv4 封包）。

    stats：可選，傳入一個 dict 就地累計 'n_parse_errors'（Ethernet 層解析
    失敗）與 'n_non_ipv4'（IPv6 等非 IPv4 封包，被跳過）的計數。資料集完整性
    檢查得到「截斷封包 0、損壞檔 0」，這兩個計數器是讓這句話有 telemetry 佐證、
    而不是純粹斷言——之前這兩種情況都被靜默跳過，無法驗證。預設 None
    （不計數，呼叫端不需要就沒有額外開銷）。"""
    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            length = len(buf)
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                if stats is not None:
                    stats["n_parse_errors"] = stats.get("n_parse_errors", 0) + 1
                continue
            ip_pkt = eth.data
            if not isinstance(ip_pkt, dpkt.ip.IP):
                if stats is not None:
                    stats["n_non_ipv4"] = stats.get("n_non_ipv4", 0) + 1
                continue

            src_mac = _mac_str(eth.src)
            dst_mac = _mac_str(eth.dst)

            ip_pkt = _unwrap_gre(ip_pkt)
            transport = ip_pkt.data

            src_port = dst_port = 0
            tcp_flags = 0
            tcp_checksum = None
            payload = b""
            proto = ip_pkt.p

            if isinstance(transport, dpkt.tcp.TCP):
                src_port, dst_port = transport.sport, transport.dport
                tcp_flags = transport.flags
                tcp_checksum = transport.sum
                payload = bytes(transport.data)
            elif isinstance(transport, dpkt.udp.UDP):
                src_port, dst_port = transport.sport, transport.dport
                payload = bytes(transport.data)
            elif isinstance(transport, (dpkt.icmp.ICMP,)):
                payload = bytes(getattr(transport, "data", b"") or b"")
            # 其餘協定（igmp 等）：不解析 payload/port，仍保留 5-tuple 骨架

            yield PacketRecord(
                ts=ts,
                length=length,
                proto=proto,
                src_ip=_ip_str(ip_pkt.src),
                dst_ip=_ip_str(ip_pkt.dst),
                src_port=src_port,
                dst_port=dst_port,
                src_mac=src_mac,
                dst_mac=dst_mac,
                ip_checksum=ip_pkt.sum,
                tcp_checksum=tcp_checksum,
                tcp_flags=tcp_flags,
                payload=payload,
            )


def _flow_key(pkt: PacketRecord, key_mode: str):
    """回傳 (無方向性的 canonical key, 這個封包是否為 a->b 的『正向』)。"""
    a = (pkt.src_ip, pkt.src_port)
    b = (pkt.dst_ip, pkt.dst_port)
    if key_mode == "hostpair":
        a, b = (pkt.src_ip,), (pkt.dst_ip,)
    forward_tuple = (pkt.proto, a, b)
    backward_tuple = (pkt.proto, b, a)
    if forward_tuple <= backward_tuple:
        return forward_tuple, True
    return backward_tuple, False


def conn_id_str(device: str, scenario: str, key: tuple) -> str:
    """連線的唯一識別字串。key 是 _flow_key() 的 canonical key，方向無關。"""
    proto, a, b = key
    return f"{device}|{scenario}|{proto}|{'/'.join(map(str, a))}|{'/'.join(map(str, b))}"


def add_window_index(df):
    """每個觀測窗在所屬連線中的序號（依 start_ts，從 0 起算）。"""
    order = df.sort_values(["conn_id", "start_ts"], kind="stable")
    df = df.copy()
    df.loc[order.index, "window_idx"] = order.groupby("conn_id").cumcount().to_numpy()
    df["window_idx"] = df["window_idx"].astype("int64")
    return df


@dataclass
class _FlowState:
    key: tuple
    first_pkt: PacketRecord
    initiator_fwd: bool  # 這條 flow 的 key 是否與第一個封包同向
    start_ts: float
    last_ts: float
    n_pkts: int = 0
    pkt_count_fwd: int = 0
    pkt_count_bwd: int = 0
    byte_count_fwd: int = 0
    byte_count_bwd: int = 0
    pkt_lengths: list = field(default_factory=list)
    pkt_iat: list = field(default_factory=list)
    pkt_dir: list = field(default_factory=list)
    tcp_flags: list = field(default_factory=list)
    payload: bytearray = field(default_factory=bytearray)
    conn_payload: bytearray = field(default_factory=bytearray)
    inherited: bool = False   # conn_payload 是否承接自同一連線的前一窗
    raw_labels: set = field(default_factory=set)

    def add(self, pkt: PacketRecord, is_forward: bool, labeller: PacketLabeller,
            payload_bytes: int) -> None:
        # is_forward 是相對「這個封包與 key 同向」；換算成相對 flow 發起者的方向
        fwd = is_forward == self.initiator_fwd
        if fwd:
            self.pkt_count_fwd += 1
            self.byte_count_fwd += pkt.length
        else:
            self.pkt_count_bwd += 1
            self.byte_count_bwd += pkt.length

        label = labeller.lookup(pkt.ts, pkt.length)
        if label is not None:
            self.raw_labels.add(label)

        iat = pkt.ts - self.last_ts if self.n_pkts > 0 else 0.0
        self.pkt_lengths.append(pkt.length)
        self.pkt_iat.append(iat)
        self.pkt_dir.append(1 if fwd else 0)
        self.tcp_flags.append(pkt.tcp_flags)
        if fwd and len(self.payload) < payload_bytes:
            remaining = payload_bytes - len(self.payload)
            self.payload.extend(pkt.payload[:remaining])
        if fwd and len(self.conn_payload) < payload_bytes:
            remaining = payload_bytes - len(self.conn_payload)
            self.conn_payload.extend(pkt.payload[:remaining])
        self.n_pkts += 1

        self.last_ts = pkt.ts

    def to_row(self, device: str, scenario: str, key_mode: str, flow_id: str) -> dict:
        raw_label = flow_label(self.raw_labels)
        payload = bytes(self.payload)
        # 沒承接時兩者內容相同，共用同一個 bytes 物件，全量 S3 不必多存一份
        conn_payload = bytes(self.conn_payload) if self.inherited else payload
        return {
            "flow_id": flow_id,
            "conn_id": conn_id_str(device, scenario, self.key),
            "device": device,
            "scenario": scenario,
            "flow_key_mode": key_mode,
            "proto": self.first_pkt.proto,
            "src_ip": self.first_pkt.src_ip,
            "dst_ip": self.first_pkt.dst_ip,
            "src_port": self.first_pkt.src_port,
            "dst_port": self.first_pkt.dst_port,
            "src_mac": self.first_pkt.src_mac,
            "dst_mac": self.first_pkt.dst_mac,
            "ip_checksum": self.first_pkt.ip_checksum,
            "tcp_checksum": self.first_pkt.tcp_checksum,
            "start_ts": self.start_ts,
            "end_ts": self.last_ts,
            "duration": self.last_ts - self.start_ts,
            "n_pkts": self.n_pkts,
            "pkt_count_fwd": self.pkt_count_fwd,
            "pkt_count_bwd": self.pkt_count_bwd,
            "byte_count_fwd": self.byte_count_fwd,
            "byte_count_bwd": self.byte_count_bwd,
            "pkt_lengths": list(self.pkt_lengths),
            "pkt_iat": list(self.pkt_iat),
            "pkt_dir": list(self.pkt_dir),
            "tcp_flags": list(self.tcp_flags),
            "payload": payload,
            "conn_payload": conn_payload,
            "raw_label": raw_label,
            "l1_label": to_l1(raw_label) if raw_label else None,
            "l2_label": to_l2(raw_label) if raw_label else None,
            "has_unknown_label": UNKNOWN_LABEL in self.raw_labels,
        }


class FlowBuilder:
    """單一 (device, scenario) 的 flow 聚合器。逐 pcap 檔案 feed，最後 flush。"""

    def __init__(self, device: str, scenario: str, labeller: PacketLabeller,
                 key_mode: str = config.FLOW_KEY_MODE,
                 idle_timeout: float = config.IDLE_TIMEOUT,
                 active_timeout: float = config.ACTIVE_TIMEOUT,
                 max_pkts: int = config.MAX_PKTS_PER_FLOW,
                 payload_bytes: int = config.PAYLOAD_BYTES,
                 max_open_flows: int = config.MAX_OPEN_FLOWS):
        self.device = device
        self.scenario = scenario
        self.labeller = labeller
        self.key_mode = key_mode
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.max_pkts = max_pkts
        self.payload_bytes = payload_bytes
        self.max_open_flows = max_open_flows

        self._open: dict[tuple, _FlowState] = {}
        # 因滿 K 而結束的窗，把連線開頭的 payload 交給同一連線的下一窗
        self._carry_payload: dict[tuple, tuple[bytes, float]] = {}
        self._rows: list[dict] = []
        self._next_id = 0
        self.eviction_events = 0
        self.parse_stats: dict = {"n_parse_errors": 0, "n_non_ipv4": 0}

    def _new_flow_id(self) -> str:
        self._next_id += 1
        return f"{self.device}:{self.scenario}:{self._next_id}"

    def _close(self, key: tuple, carry: bool = False) -> None:
        state = self._open.pop(key)
        if carry:
            self._carry_payload[key] = (bytes(state.conn_payload), state.last_ts)
        self._rows.append(state.to_row(self.device, self.scenario, self.key_mode,
                                        self._new_flow_id()))

    def add_packet(self, pkt: PacketRecord) -> None:
        key, is_forward = _flow_key(pkt, self.key_mode)
        state = self._open.get(key)
        if state is not None:
            if (pkt.ts - state.last_ts > self.idle_timeout
                    or pkt.ts - state.start_ts > self.active_timeout):
                self._close(key)
                state = None
        if state is None:
            state = _FlowState(key=key, first_pkt=pkt, initiator_fwd=is_forward,
                                start_ts=pkt.ts, last_ts=pkt.ts)
            carried = self._carry_payload.pop(key, None)
            if carried is not None and pkt.ts - carried[1] <= self.idle_timeout:
                state.conn_payload.extend(carried[0])
                state.inherited = True
            self._open[key] = state
        state.add(pkt, is_forward, self.labeller, self.payload_bytes)

        # K 是觀測窗邊界：極端高速率的單一連線（如 CoAP Amplification 15 秒內
        # 274,837 包）若不切窗只會有 1 個樣本。滿 K 就結束這個窗，同一連線的
        # 後續封包開下一窗並繼承連線開頭的 payload；這些窗共用 conn_id，
        # 切分時整條連線一起指派（src/pcap/split.py），不會造成近重複洩漏。
        if key in self._open and self._open[key].n_pkts >= self.max_pkts:
            self._close(key, carry=True)

        if len(self._open) > self.max_open_flows:
            self._evict_oldest()

    def _evict_oldest(self, fraction: float = 0.25) -> None:
        n_evict = max(1, int(len(self._open) * fraction))
        oldest_keys = sorted(self._open, key=lambda k: self._open[k].last_ts)[:n_evict]
        for key in oldest_keys:
            self._close(key)
        self.eviction_events += 1

    def feed_pcap(self, pcap_path: str) -> None:
        for pkt in iter_packets(pcap_path, stats=self.parse_stats):
            self.add_packet(pkt)

    def flush(self) -> list[dict]:
        for key in list(self._open):
            self._close(key)
        self._carry_payload.clear()
        return self._rows


def find_pcaps(device: str, scenario: str) -> list[str]:
    scenario_dir = config.PCAP_ROOT / config.SCENARIOS[scenario]
    pattern = str(scenario_dir / f"{config.PCAP_PREFIX}{device}_*.pcap")
    return sorted(glob.glob(pattern))


def build_flows_for_device_scenario(device: str, scenario: str,
                                     labeller: PacketLabeller | None = None,
                                     **builder_kwargs) -> tuple[list[dict], int, dict]:
    """單一裝置 x 單一情境的 flow 抽取。裝置在此情境沒有 pcap 就回傳空列表。

    回傳 (rows, eviction_events, parse_stats)：
    - eviction_events 是 MAX_OPEN_FLOWS 保護機制被觸發的次數，非 0 代表這個
      (device, scenario) 有 flow 被強制提前關閉，統計量可能失真，呼叫端應該
      記錄下來而不是默默丟掉（見 scripts/build_flows.py）。
    - parse_stats 是 {'n_parse_errors', 'n_non_ipv4'}（見 iter_packets），
      讓「截斷封包 0、損壞檔 0」這句話有 telemetry 佐證，而不是
      單純斷言——之前這兩種情況都被靜默跳過，無從驗證。
    """
    pcaps = find_pcaps(device, scenario)
    if not pcaps:
        return [], 0, {"n_parse_errors": 0, "n_non_ipv4": 0}
    if labeller is None:
        labeller = PacketLabeller.for_device(device)
    builder = FlowBuilder(device, scenario, labeller, **builder_kwargs)
    for p in pcaps:
        builder.feed_pcap(p)
    return builder.flush(), builder.eviction_events, builder.parse_stats
