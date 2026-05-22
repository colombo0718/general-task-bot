"""
codex_exam_runner.py — 跑題本對 gtb_codex.py 的 /sim 後門

跟 exam_runner.py 不同：codex /sim 回的是純文字 reply、沒有 intent/values
結構，所以本 runner 只記 reply + elapsed + tags，不做 intent 比對；
expected_intent 只當註記、人後續用 grep 看。

Sequential（codex 同 SID 不支援並發 resume），每筆 incremental 寫檔
（中斷也能看到目前進度）。

用法：
    py -3 codex_exam_runner.py <exam.json> --oaid @526fdbzo
    py -3 codex_exam_runner.py <exam.json> --oaid @526fdbzo \\
        --target http://127.0.0.1:6010 --timeout 120 \\
        --out path/to/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("exam_path")
    ap.add_argument("--oaid", required=True)
    ap.add_argument("--target", default="http://127.0.0.1:6010")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=1, help="resume from item id N (1-based)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 題、0=全跑")
    args = ap.parse_args()

    exam = json.loads(Path(args.exam_path).read_text(encoding="utf-8"))
    items = exam["items"]
    if args.start > 1:
        items = [it for it in items if it["id"] >= args.start]
    if args.limit > 0:
        items = items[: args.limit]

    sim_url = f"{args.target}/sim/{args.oaid}"
    started_at = time.strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out)
        if args.out
        else Path(args.exam_path).parent / "reports" / f"codex_run_{started_at}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[runner] exam={args.exam_path} items={len(items)} oaid={args.oaid}")
    print(f"[runner] sim={sim_url} timeout={args.timeout}s")
    print(f"[runner] out={out_path}")
    print()

    results: list[dict] = []
    t_start = time.time()
    for i, item in enumerate(items, 1):
        user_text = item["user_text"]
        expected_intent = item.get("expected_intent")
        t0 = time.time()
        ok = False
        reply_clean = ""
        tag_names: list[str] = []
        err: str | None = None
        try:
            r = requests.post(sim_url, json={"user_text": user_text}, timeout=args.timeout)
            resp = r.json()
            ok = bool(resp.get("ok"))
            reply_clean = resp.get("reply_clean") or resp.get("reply_raw") or ""
            tags = resp.get("tags") or []
            tag_names = [t[0] if isinstance(t, list) and t else str(t) for t in tags]
            if not ok:
                err = resp.get("error") or "ok=false"
        except Exception as e:
            err = str(e)
        elapsed = time.time() - t0

        result = {
            "id": item["id"],
            "user_text": user_text,
            "expected_intent": expected_intent,
            "ok": ok,
            "elapsed": round(elapsed, 1),
            "reply_chars": len(reply_clean),
            "reply_preview": reply_clean[:300],
            "reply_full": reply_clean,
            "tag_names": tag_names,
            "err": err,
        }
        results.append(result)

        out_path.write_text(
            json.dumps(
                {
                    "exam": args.exam_path,
                    "oaid": args.oaid,
                    "started_at": started_at,
                    "progress": f"{i}/{len(items)}",
                    "elapsed_total": round(time.time() - t_start, 1),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        marker = "OK " if ok else "ERR"
        tag_str = f" [{','.join(tag_names)}]" if tag_names else ""
        preview = reply_clean[:60].replace("\n", " ")
        print(
            f"[{i:3d}/{len(items)}] {marker} {elapsed:5.1f}s  "
            f"{(expected_intent or '?'):20s}  {preview}{tag_str}",
            flush=True,
        )

    total = time.time() - t_start
    tag_counter: Counter = Counter()
    for r in results:
        tag_counter.update(r["tag_names"])
    intent_stats: dict[str, dict] = {}
    for intent in {r["expected_intent"] for r in results if r["expected_intent"]}:
        rs = [r for r in results if r["expected_intent"] == intent]
        intent_stats[intent] = {
            "n": len(rs),
            "avg_elapsed": round(sum(r["elapsed"] for r in rs) / len(rs), 1),
            "avg_chars": round(sum(r["reply_chars"] for r in rs) / len(rs)),
            "err_count": sum(1 for r in rs if not r["ok"]),
        }

    summary = {
        "total_items": len(results),
        "total_elapsed_s": round(total, 1),
        "avg_per_item_s": round(total / max(len(results), 1), 1),
        "err_count": sum(1 for r in results if not r["ok"]),
        "tags_emitted": dict(tag_counter),
        "intent_stats": intent_stats,
    }

    final = {
        "exam": args.exam_path,
        "oaid": args.oaid,
        "started_at": started_at,
        "finished_at": time.strftime("%Y%m%d_%H%M%S"),
        "summary": summary,
        "results": results,
    }
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
