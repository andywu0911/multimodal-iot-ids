"""實驗結果的可重現性快照。指標寫入 results/ 時
「附設定快照」——這裡把它具體化成「哪個 git commit、吃了哪份輸入檔、
config.py 當時長什麼樣子」，讓論文引用的每個數字都能對回產生它的那次執行。

用法：在寫出 results JSON 之前呼叫 snapshot()，把回傳值塞進輸出 dict
的 "provenance" 欄位（純新增鍵，不影響既有讀者）。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import config


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    """True 代表跑這次實驗時工作目錄有未 commit 的變更——若為 True，
    git_sha 只能定位到「最接近」的程式碼版本，不保證逐位元組一致。"""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _config_snapshot() -> dict:
    """config.py 裡所有大寫常數，Path 物件轉字串、dict/list 遞迴轉安全型別。"""
    def _safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {str(k): _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    return {k: _safe(v) for k, v in vars(config).items()
            if k.isupper() and not k.startswith("_")}


def file_sha256(path) -> str | None:
    """輸入檔（如 <subset>_split.parquet）的雜湊，前 16 hex 字元即可分辨版本，
    不需要完整 64 字元。檔案不存在時回傳 None，不中斷呼叫端。"""
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def snapshot(input_path=None) -> dict:
    """回傳可直接塞進 results JSON 的 provenance dict。

    input_path：可選，這次實驗讀的主要輸入檔（如 <subset>_split.parquet），
    會記錄其 sha256 前 16 碼——用來確認兩次「同 subset」的執行讀的是不是
    同一份中間產物（例如重跑過 sample_flows.py 後這個值會變）。
    """
    out = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "config": _config_snapshot(),
    }
    if input_path is not None:
        out["input_sha256_16"] = file_sha256(input_path)
        out["input_path"] = str(input_path)
    return out
