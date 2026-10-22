"""標籤定義、階層映射、flow 標籤優先序。

標籤對齊與階層映射統一在此，任何腳本一律 import。
封包數取自已驗證的實測分布（35,134,281 列），勿重新量測。
"""
from __future__ import annotations

# ------------------------------------------------ 原始 18 類（官方 CSV 的 label 欄）
# 論文宣稱 10 類，實際 18 類；Remote Command/Code Execution 與
# Periodic C&C Communication 三者不存在。
RAW_LABEL_COUNTS = {
    "Benign": 12_256_883,
    "Mirai UDP Flooding": 8_897_895,
    "Mirai TCP Flooding": 6_548_173,
    "Mirai GRE Flooding": 5_911_401,
    "TCP Scan": 737_764,
    "CoAP Amplification": 274_837,
    "Telnet Brute Force": 227_649,
    "Merlin TCP Flooding": 120_000,
    "Merlin ICMP Flooding": 57_580,
    "Merlin UDP Flooding": 29_996,
    "Merlin C&C Communication": 29_356,
    "Ingress Tool Transfer": 21_587,
    "Unknown": 7_670,
    "File Download": 7_196,
    "UDP Scan": 4_242,
    "Mirai C&C Communication": 1_074,
    "C&C Communication": 528,
    "Reporting": 450,
}
RAW_LABELS = list(RAW_LABEL_COUNTS)
RAW_LABEL_TO_ID = {lab: i for i, lab in enumerate(RAW_LABELS)}

UNKNOWN_LABEL = "Unknown"   # 標記雜訊的直接證據；預設排除但需做敏感度分析
BENIGN_LABEL = "Benign"

# ------------------------------------------------ Level 1：攻擊類別層 6 類（主任務）
# 原為 7 類；Exfiltration（Reporting，僅 450 個封包）flow 數過低（S2 上 39 條，
# 低於 W1 檢查點的 50 條門檻），決定併入 Tool Transfer —— 兩者在攻擊鏈上位置
# 相近（取得工具 → 回報）。ATT&CK 對應仍保留 Reporting -> T1041 的原始語意
# （見 RAW_TO_ATTACK），只有 L1 分類任務層級合併。
L1_CLASSES = [
    "Benign", "Reconnaissance", "Credential Access",
    "Tool Transfer", "Command & Control", "Impact",
]
L1_TO_ID = {c: i for i, c in enumerate(L1_CLASSES)}

RAW_TO_L1 = {
    "Benign": "Benign",
    "TCP Scan": "Reconnaissance",
    "UDP Scan": "Reconnaissance",
    "Telnet Brute Force": "Credential Access",
    "Ingress Tool Transfer": "Tool Transfer",
    "File Download": "Tool Transfer",
    "Reporting": "Tool Transfer",
    "Mirai C&C Communication": "Command & Control",
    "Merlin C&C Communication": "Command & Control",
    "C&C Communication": "Command & Control",
    "Mirai UDP Flooding": "Impact",
    "Mirai TCP Flooding": "Impact",
    "Mirai GRE Flooding": "Impact",
    "Merlin TCP Flooding": "Impact",
    "Merlin ICMP Flooding": "Impact",
    "Merlin UDP Flooding": "Impact",
    "CoAP Amplification": "Impact",
    # Unknown 不映射；預設排除
}

# ------------------------------------------------ Level 2：技術層 17 類（輔助頭）
L2_CLASSES = [lab for lab in RAW_LABELS if lab != UNKNOWN_LABEL]
L2_TO_ID = {c: i for i, c in enumerate(L2_CLASSES)}

# ------------------------------------------------ ATT&CK 對應（L4 報告層）
L1_TO_ATTACK = {
    "Reconnaissance": "T1046",
    "Credential Access": "T1110",
    "Tool Transfer": "T1105",
    "Command & Control": "T1071",
    "Impact": "T1498",
}
# 原始標籤層級覆寫：即使 L1 已合併，L4 報告仍要能講出正確的 ATT&CK 技術語意。
RAW_TO_ATTACK = {
    "CoAP Amplification": "T1498.002",
    "Reporting": "T1041",   # L1 併入 Tool Transfer，但攻擊目的仍是 Exfiltration
}

# ------------------------------------------------ flow 標籤優先序
# flow 標籤採優先序，不用多數決。
#
# 第一版用「全域封包數由少到多」排序，稀有者優先 —— 這保護了 Reporting／C2
# 這類稀少標籤不被 Benign 多數決洗掉，但實測發現副作用：CoAP Amplification
# （274,837 封包，combined-cycle-1 唯一的完整 Impact 類攻擊）跟 UDP Scan
# （4,242 封包）共用同一個 flow 時間窗時，反而被數量更少、攻擊類別層級更輕的
# UDP Scan 蓋過（S2 上 2,191 條 flow 判成 UDP Scan，只剩 1 條保住
# CoAP Amplification）。
#
# 改為「攻擊類別嚴重度分層」：先比 L1 攻擊類別層級（Impact 最高、Reconnaissance
# 最低），同層級才用封包數由少到多當 tie-break（沿用第一版邏輯，保護同層級
# 內的稀有標籤，例如同屬 Command & Control 的 Mirai C&C 對上 Merlin C&C）。
# Unknown 僅贏過 Benign，維持不變。
TACTIC_SEVERITY_ORDER = [
    "Impact", "Command & Control", "Credential Access",
    "Tool Transfer", "Reconnaissance",
]
_TACTIC_RANK = {t: i for i, t in enumerate(TACTIC_SEVERITY_ORDER)}

_ATTACK_LABELS = [lab for lab in RAW_LABELS
                  if lab not in (BENIGN_LABEL, UNKNOWN_LABEL)]
LABEL_PRIORITY = (
    sorted(_ATTACK_LABELS,
           key=lambda lab: (_TACTIC_RANK[RAW_TO_L1[lab]], RAW_LABEL_COUNTS[lab]))
    + [UNKNOWN_LABEL, BENIGN_LABEL]
)
_PRIORITY_RANK = {lab: i for i, lab in enumerate(LABEL_PRIORITY)}


def flow_label(packet_labels) -> str | None:
    """由一條 flow 內所有封包的標籤決定 flow 標籤（優先序，非多數決）。

    packet_labels 可為 list / set / Counter 的鍵；未對到標籤的封包請先剔除。
    """
    best, best_rank = None, len(_PRIORITY_RANK)
    for lab in packet_labels:
        r = _PRIORITY_RANK.get(lab)
        if r is not None and r < best_rank:
            best, best_rank = lab, r
    return best


def to_l1(raw_label: str) -> str | None:
    return RAW_TO_L1.get(raw_label)


def to_l2(raw_label: str) -> str | None:
    return raw_label if raw_label in L2_TO_ID else None
