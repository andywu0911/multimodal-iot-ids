"""L4-lite 報告產生 CLI。python scripts/render_reports.py --subset S3 --source pred

讀 scripts/build_events.py 的輸出（<subset>/{pred,true}_events_{test,all}.json），
用 src/report/template.py 的固定模板轉成 Markdown 通報，存到 results/reports/。
不含 RAG 檢索、不含 LLM 生成，純模板串接。

--scope 要跟 build_events.py 的 --splits 對齊：test 是模型沒看過的資料（論文
引用這組）；all 是 train+val+test，事件數較多、適合當附錄的輸出樣本展示，
但屬 in-sample，不可當成效能證據。
"""
from __future__ import annotations

import argparse
import json

import config
from src.report.template import render_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="S3")
    ap.add_argument("--source", choices=["pred", "true"], default="pred")
    ap.add_argument("--scope", choices=["test", "all"], default="test",
                     help="對應 build_events.py 的 --splits，預設 test。")
    args = ap.parse_args()

    in_path = (config.DATA_PROCESSED / "events" / args.subset
               / f"{args.source}_events_{args.scope}.json")
    events = json.loads(in_path.read_text())

    scope_zh = "test split" if args.scope == "test" else "train+val+test，in-sample"
    title = (f"{args.subset} 資安事件通報"
             f"（{'模型預測' if args.source == 'pred' else '真實標籤'}，{scope_zh}）")
    report = render_report(events, title)

    out_dir = config.RESULTS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.subset}_{args.source}_report_{args.scope}.md"
    out_path.write_text(report)
    print(f"[render_reports] {len(events)} events -> {out_path}")


if __name__ == "__main__":
    main()
