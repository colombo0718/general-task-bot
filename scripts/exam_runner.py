"""
exam_runner.py — 並發批次跑 GTB /sim 後門題本

用法：
    py -3 exam_runner.py <題本路徑>
    py -3 exam_runner.py <題本路徑> --target http://localhost:6010
    py -3 exam_runner.py <題本路徑> --concurrency 5 --timeout 90

預設行為（從題本 JSON 自動推）：
    題本路徑：CLI 第一參數
    target  ：http://localhost:6010/sim/<oaid from 題本 _meta>
    並發    ：題本內 "concurrency" 欄位
    輸出    ：<題本所在資料夾>/reports/<YYYY-MM-DD_HHMM>_<題本檔名>/
"""
import argparse
import concurrent.futures
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests


def _camel_to_snake(s):
    """adjustPoints -> adjust_points；用於 agent 模式 expected (camelCase) vs actual (snake_case task_id) 對齊"""
    if not s:
        return s
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def run_one(item, target, timeout, mode=None, agent_provider=None, agent_model=None):
    t0 = time.time()
    try:
        body = {
            "user_id": item.get("user_id", "Uexam"),
            "display_name": item.get("display_name", item.get("user_id", "exam")),
            "user_text": item["user_text"],
        }
        if mode:
            body["mode"] = mode
        if agent_provider:
            body["agent_provider"] = agent_provider
        if agent_model:
            body["agent_model"] = agent_model
        r = requests.post(target, json=body, timeout=timeout)
        resp = r.json()
        wire_ms = int((time.time() - t0) * 1000)

        # 比對策略：
        # - intent：完全相等；agent 模式 actual=snake_case 自動跟 expected=camelCase 對齊
        # - values：寬鬆——expected_values 是 actual_values 的子集（key 對 + value 一致）
        actual_intent = resp.get("intent")
        expected_intent = item.get("expected_intent")
        if mode == "agent" and expected_intent:
            intent_match = actual_intent == _camel_to_snake(expected_intent)
        else:
            intent_match = actual_intent == expected_intent

        actual_values = resp.get("values", {}) or {}
        expected_values = item.get("expected_values", {}) or {}
        values_match = all(
            str(actual_values.get(k, "")).strip() == str(v).strip()
            for k, v in expected_values.items()
        )

        return {
            "id": item["id"],
            "user_text": item["user_text"],
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_match": intent_match,
            "expected_values": expected_values,
            "actual_values": actual_values,
            "values_match": values_match,
            "intent_chain": resp.get("intent_chain"),
            "task_id": resp.get("task_id"),
            "cmd_url": resp.get("cmd_url"),
            "missing": resp.get("missing"),
            "sim_timing_ms": resp.get("timing_ms"),
            "wire_latency_ms": wire_ms,
            "errors": resp.get("errors", []),
        }
    except Exception as e:
        return {
            "id": item["id"],
            "user_text": item["user_text"],
            "expected_intent": item.get("expected_intent"),
            "actual_intent": None,
            "intent_match": False,
            "expected_values": item.get("expected_values", {}),
            "actual_values": {},
            "values_match": False,
            "exception": f"{type(e).__name__}: {e}",
        }


def percentile(sorted_arr, p):
    if not sorted_arr:
        return 0
    idx = min(len(sorted_arr) - 1, int(len(sorted_arr) * p / 100))
    return sorted_arr[idx]


def main():
    parser = argparse.ArgumentParser(description="GTB /sim 後門題本 runner")
    parser.add_argument("exam", help="path to exam JSON")
    parser.add_argument("--target", default="http://localhost:6010", help="dev /sim 服務 host (default: http://localhost:6010)")
    parser.add_argument("--concurrency", type=int, default=None, help="覆寫題本內 concurrency")
    parser.add_argument("--timeout", type=int, default=60, help="per-request timeout 秒 (default: 60)")
    parser.add_argument("--out", default=None, help="覆寫 output 資料夾")
    parser.add_argument("--per-item-sleep", type=float, default=0.0,
                        help="每題完成後睡 N 秒再下一題 (避開 free-tier rate limit；建議 LLM mode 設 3-5)")
    parser.add_argument("--mode", choices=["llm", "agent"], default=None,
                        help="走 /sim 的 LLM cascade 模式還是 agent 暴力雙問模式 (default: llm 端的預設)")
    parser.add_argument("--agent-provider", default=None,
                        help="agent 模式覆寫 provider (e.g. home_ollama)，預設 codex_cli")
    parser.add_argument("--agent-model", default=None,
                        help="agent 模式覆寫 model (e.g. gemma3:4b)，預設 gpt-5.4")
    args = parser.parse_args()

    exam_path = Path(args.exam).resolve()
    exam = json.loads(exam_path.read_text(encoding="utf-8"))
    items = exam["items"]
    oaid = exam.get("oaid", "@test_oa")
    concurrency = args.concurrency or exam.get("concurrency", 5)
    target = f"{args.target.rstrip('/')}/sim/{oaid}"

    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_dir = exam_path.parent / "reports" / f"{ts}_{exam_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG]")
    print(f"  exam        : {exam_path}")
    print(f"  target      : {target}")
    print(f"  items       : {len(items)}")
    print(f"  concurrency : {concurrency}")
    print(f"  output      : {out_dir}")
    print()

    # concurrency=1 + per_item_sleep>0 走純序列分支（保證一題完整答完再 sleep N 秒才打下一題）
    start_all = time.time()
    results = [None] * len(items)
    if concurrency == 1 and args.per_item_sleep > 0:
        print(f"  per-item-sleep : {args.per_item_sleep}s (sequential mode)")
        print()
        for idx, item in enumerate(items):
            results[idx] = run_one(item, target, args.timeout, mode=args.mode, agent_provider=args.agent_provider, agent_model=args.agent_model)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(items):
                print(f"  progress: {idx + 1}/{len(items)}")
            if idx + 1 < len(items):
                time.sleep(args.per_item_sleep)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(run_one, item, target, args.timeout, args.mode, args.agent_provider, args.agent_model): idx for idx, item in enumerate(items)}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  progress: {done}/{len(items)}")
    total_sec = time.time() - start_all

    # 寫 results.json
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps({
            "exam": str(exam_path),
            "target": target,
            "concurrency": concurrency,
            "total_sec": total_sec,
            "items": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 統計
    total = len(results)
    intent_pass = sum(1 for r in results if r.get("intent_match"))
    values_pass = sum(1 for r in results if r.get("values_match"))
    both_pass = sum(1 for r in results if r.get("intent_match") and r.get("values_match"))

    latencies = sorted(r["sim_timing_ms"] for r in results if r.get("sim_timing_ms") is not None)
    avg_lat = sum(latencies) // max(len(latencies), 1)

    # tier depth = intent_chain 長度
    tier_dist = Counter()
    for r in results:
        chain = r.get("intent_chain") or []
        depth = len(chain) if chain else 0
        if depth == 0:
            tier_dist["error/empty"] += 1
        else:
            tier_dist[f"Tier{depth}"] += 1

    fails = [r for r in results if not (r.get("intent_match") and r.get("values_match"))]

    # 寫 summary.md
    summary_lines = [
        f"# Exam 報告：{exam_path.name}",
        "",
        f"- 跑題時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 題本：`{exam_path}`",
        f"- target：`{target}`",
        f"- 並發：{concurrency}",
        f"- 總題數：{total}",
        f"- 總耗時：{total_sec:.1f} 秒",
        "",
        "## 通過率",
        "",
        "| 指標 | 通過 | 比率 |",
        "|------|------|------|",
        f"| Intent 正確 | {intent_pass}/{total} | {intent_pass/total*100:.1f}% |",
        f"| Values 正確 | {values_pass}/{total} | {values_pass/total*100:.1f}% |",
        f"| **Intent + Values 都對** | **{both_pass}/{total}** | **{both_pass/total*100:.1f}%** |",
        "",
        "## Tier 命中分布",
        "",
        "| Tier | 筆數 |",
        "|------|------|",
    ]
    for t, c in sorted(tier_dist.items()):
        summary_lines.append(f"| {t} | {c} |")
    summary_lines += [
        "",
        "## Latency（/sim 內部耗時，毫秒）",
        "",
        "| 統計 | ms |",
        "|------|------|",
        f"| 平均 | {avg_lat} |",
        f"| p50 | {percentile(latencies, 50)} |",
        f"| p95 | {percentile(latencies, 95)} |",
        f"| p99 | {percentile(latencies, 99)} |",
        "",
        f"## Fail Cases（共 {len(fails)} 筆，前 30 筆）",
        "",
        "| ID | user_text | 預期 intent | 實際 intent | 預期 values | 實際 values |",
        "|----|-----------|-----------|-----------|-----------|-----------|",
    ]
    for r in fails[:30]:
        ev = json.dumps(r.get("expected_values", {}), ensure_ascii=False)
        av = json.dumps(r.get("actual_values", {}), ensure_ascii=False)
        ut = r["user_text"].replace("|", "\\|")
        summary_lines.append(
            f"| {r['id']} | {ut[:30]} | {r.get('expected_intent')} | {r.get('actual_intent')} | `{ev[:50]}` | `{av[:50]}` |"
        )
    if len(fails) > 30:
        summary_lines.append("")
        summary_lines.append(f"... 還有 {len(fails) - 30} 筆 fail，詳見 results.json")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print("=== 跑完 ===")
    print(f"  intent 正確: {intent_pass}/{total} ({intent_pass/total*100:.1f}%)")
    print(f"  values 正確: {values_pass}/{total} ({values_pass/total*100:.1f}%)")
    print(f"  兩者都對:   {both_pass}/{total} ({both_pass/total*100:.1f}%)")
    print(f"  耗時: {total_sec:.1f}s")
    print()
    print(f"  results: {results_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
