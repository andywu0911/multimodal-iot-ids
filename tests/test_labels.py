"""flow_label() 優先序的行為驗證。flow 標籤採「攻擊類別嚴重度分層，
同層再以全域封包數由少到多 tie-break」，不是多數決——這條規則錯了不會拋出
例外，只會默默把稀有標籤洗掉，所以特別需要測試釘死。
"""
from src.labels import flow_label


def test_higher_severity_tactic_wins():
    """Impact 的攻擊類別層級高於 Credential Access，即使 Credential Access
    (Telnet Brute Force) 全域封包數比較少也一樣。"""
    assert flow_label({"Telnet Brute Force", "Mirai UDP Flooding"}) == "Mirai UDP Flooding"


def test_same_tactic_tiebreak_by_global_rarity():
    """同一攻擊類別層級（Command & Control）內，tie-break 是全域封包數由少到多——
    Mirai C&C（1,074 封包）比 Merlin C&C（29,356 封包）稀有，應該勝出。"""
    assert flow_label({"Mirai C&C Communication", "Merlin C&C Communication"}) == "Mirai C&C Communication"


def test_any_attack_label_beats_benign():
    assert flow_label({"Benign", "TCP Scan"}) == "TCP Scan"


def test_unknown_beats_benign_but_loses_to_attack_labels():
    """Unknown 只贏過 Benign，輸給任何真正的攻擊標籤。"""
    assert flow_label({"Unknown", "Benign"}) == "Unknown"
    assert flow_label({"Unknown", "TCP Scan"}) == "TCP Scan"


def test_empty_input_returns_none():
    assert flow_label([]) is None


def test_single_label_passthrough():
    assert flow_label({"Benign"}) == "Benign"
    assert flow_label({"Reporting"}) == "Reporting"
