"""專案唯一的設定來源。

所有路徑、標籤、閥值、模型名稱一律定義在此，
禁止在任何模組寫死。資料集位置可用環境變數 GOTHAM_ROOT 覆寫。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路徑
PROJECT_ROOT = Path(__file__).resolve().parent

GOTHAM_ROOT = Path(os.environ.get("GOTHAM_ROOT", PROJECT_ROOT / "data/raw/gotham2025"))
PCAP_ROOT = GOTHAM_ROOT / "raw"                 # 上游原始佈局，勿更動
OFFICIAL_CSV_ROOT = GOTHAM_ROOT / "processed"   # 官方 tshark 封包級 CSV

DATA_INTERIM = PROJECT_ROOT / "data/interim"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
LABEL_INDEX_DIR = DATA_INTERIM / "label_index"   # 每裝置的 (ts, len) -> label 索引快取
FLOW_DIR = DATA_PROCESSED / "flows"
FEATURES_DIR = DATA_PROCESSED / "features"   # W2：View A/B 特徵產物
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "models"    # 訓練好的模型權重（單一事實來源，見 src/models/train.py）

for _d in (LABEL_INDEX_DIR, FLOW_DIR, FEATURES_DIR, RESULTS_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 可重現性
RANDOM_SEED = 42

# ---------------------------------------------------------------- 情境
# 情境名稱 -> pcap 目錄（相對 PCAP_ROOT）
SCENARIOS = {
    "benign": "benign",
    "network-scanning": "malicious/network-scanning",
    "coap-amplificator": "malicious/coap-amplificator",   # 目錄名非 README 寫的 coap-amplification
    "merlin": "malicious/merlin",
    "mirai-infection": "malicious/mirai-infection",
    "mirai-dos": "malicious/mirai-dos",
}

# 實測：每個情境是在不同「日期」批次執行的（2026-08-24 由 pcap 首包時間驗證）。
# 這是 P1 逐情境時序切分的天然邊界，也讓 CSV 區塊能無歧義對回情境。
SCENARIO_DATES = {
    "benign": "2025-01-14",
    "network-scanning": "2025-01-14",   # 23:37 起，與 benign 同日但不重疊
    "coap-amplificator": "2025-01-15",
    "merlin": "2025-01-17",
    "mirai-infection": "2025-01-18",
    "mirai-dos": "2025-01-25",
}

PCAP_PREFIX = "iotsim-"   # 檔名格式：iotsim-<device>_0-0_to_OpenvSwitch-<n>_<m>-0.pcap

# ---------------------------------------------------------------- 裝置
# 8 / 78 台裝置有攻擊流量（其餘 70 台純良性）
ATTACKED_DEVICES = [
    "air-quality-1",
    "building-monitor-1",
    "city-power-1",
    "combined-cycle-1",
    "combined-cycle-10",
    "domotic-monitor-1",
    "ip-camera-museum-1",
    "ip-camera-street-1",
]
CAMERA_DEVICES = ["ip-camera-museum-1", "ip-camera-street-1"]

# ---------------------------------------------------------------- 三階子集
# 抽樣抽在 flow 層，不是丟棄 pcap；子集只是為了縮短開發迭代。
SUBSETS = {
    # S1 Dev（~33 MB）：寫解析器、除錯
    "S1": {
        "devices": ["air-quality-1", "combined-cycle-10"],
        "scenarios": list(SCENARIOS),
    },
    # S2 Rare-complete：含全部稀有類別，W1–W2 主力。
    # mirai-dos 必須列入：Mirai C&C Communication（E4 旗艦實驗的一半）實測落在
    # 2025-01-25，屬 mirai-dos 情境的時間窗，不在 mirai-infection（01-18）裡；
    # 少了它 S2 就沒有明文 Telnet C2 樣本，E4 對照做不了。
    "S2": {
        "devices": [d for d in ATTACKED_DEVICES if d not in CAMERA_DEVICES],
        "scenarios": ["benign", "network-scanning", "coap-amplificator",
                      "merlin", "mirai-infection", "mirai-dos"],
    },
    # S3 Full（30 GB）：W3 最終實驗
    "S3": {"devices": None, "scenarios": list(SCENARIOS)},
}

# ---------------------------------------------------------------- L0 flow 參數
# flow key 模式：
#   "5tuple"   —— 文獻相容（CICFlowMeter 式）。實測對 Mirai 洪水會退化成 1.0~1.8 封包/流。
#   "hostpair" —— (src_ip, dst_ip, proto) 雙向；埠號降級為特徵，同時結構性地移除埠號捷徑。
FLOW_KEY_MODE = "5tuple"
FLOW_KEY_MODES = ("5tuple", "hostpair")

IDLE_TIMEOUT = 60.0        # 秒
ACTIVE_TIMEOUT = 300.0     # 秒
MAX_PKTS_PER_FLOW = 200    # K：觀測窗邊界（滿 K 包即結束該窗，同一連線續開下一窗）。
                            # 連線（conn_id）才是切分與評估的獨立單位；窗只是訓練/推論
                            # 的觀測片段。CoAP Amplification 單一連線 15 秒內 274,837 包，
                            # 不切窗就只有 1 個樣本、無法訓練。
PAYLOAD_BYTES = 512        # View P：連線開頭 512 bytes（client -> server），同連線各窗共用
WINDOW_CAP_PER_CONN = 20   # 每條連線最多保留幾個觀測窗，避免單一長連線主導某類別的樣本與指標

# ---------------------------------------------------------------- 主機情境（View C）
# 同一來源主機在觀測窗結束時刻往回看 W 秒內「已結束」的觀測窗統計（因果：只用過去）。
# 做法沿用 Kitsune AfterImage 的多時間尺度概念（Mirsky et al., NDSS 2018）。
CONTEXT_WINDOWS = (1.0, 10.0, 60.0)   # 秒

# View F 的封包序列長度（前 N 包的長度/方向/IAT/flags）
SEQ_LEN = 32

# 記憶體護欄：5tuple 模式下 mirai-dos 會產生數百萬條並存的單封包 flow。
# 超過上限時強制關閉最久未活動的 25%，並在報告中記錄觸發次數。
MAX_OPEN_FLOWS = 2_000_000

# 內層隨機性特徵的取樣上限（避免對每條 flow 存下百萬個 tuple）
INNER_CARD_CAP = 4096

# ---------------------------------------------------------------- 抽樣（flow 層）
# 對每個「原始標籤」（L2，非 L1 大類）設同一個 flow 數上限，用單一規則同時
# 實作「多數類設上限、稀有類 100% 保留」：稀有類本來就低於門檻，套用後不受
# 影響，不需要為多數類/稀有類寫兩套邏輯。見 src/pcap/sample.py。
MAX_FLOWS_PER_RAW_LABEL = 5000

# ---------------------------------------------------------------- E1 官方 baseline 重現
# 官方 baseline 是封包級全量（3,500 萬列），單機訓練吃不下，對每台裝置的 CSV
# 做均勻隨機列抽樣後再訓練——維持原始類別比例不變，只縮小絕對量。
E1_SAMPLE_FRAC = 0.10

# ---------------------------------------------------------------- 切分流程（P1）
# 逐情境時序切分 60/20/20，再合併。絕對不可做全域時間切分——攻擊情境依序
# 批次執行（見 SCENARIO_DATES），全域切分會把整個情境丟進單一子集。
SPLIT_TRAIN_FRAC = 0.6
SPLIT_VAL_FRAC = 0.2   # test = 1 - TRAIN - VAL
# 主評估流程（連線分組切分）：一個 (情境, 連線標籤) 群組至少要有這麼多條連線，才能
# 保證 train/val/test 各分到 ≥1 條。不足時（如 CoAP Amplification 全域 1 條連線）
# 改為在連線內依時間切分，並在 split_mode 欄位標記。
MIN_CONNS_FOR_GROUP_SPLIT = 3

# ---------------------------------------------------------------- L2 訓練
EMBED_DIM = 64
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
CB_BETA = 0.999      # Class-Balanced Loss 的 β（Cui et al., CVPR 2019），有效樣本數 (1-β^n)/(1-β)
LAMBDA_DET = 0.5     # 總損失 = 分類頭 CB-CE + LAMBDA_DET × 偵測頭 BCE
EVAL_SEEDS = (42, 43, 44, 45, 46)   # 多 seed 報 mean ± std（mps 在固定 seed 下仍非決定性）

# ---------------------------------------------------------------- 捷徑特徵隔離
# 核心學術主張。這些欄位會留在 interim parquet 供
# E2 probe 與 P2 切分使用，但絕對不可進入主模型特徵集。
SHORTCUT_COLUMNS = [
    "device", "src_ip", "dst_ip", "src_mac", "dst_mac",
    "src_port", "dst_port",           # 埠號幾乎等於標籤本身，一律視為捷徑
    "ip_checksum", "tcp_checksum",
]
# View A 要做兩版：含埠 / 不含埠
VIEW_A_PORT_VARIANTS = ("with_port", "no_port")
