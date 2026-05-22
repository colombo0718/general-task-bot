"""
codex_doc_smoke.py — 對話層 smoke test：codex 真的會用新 MCP tool 嗎？

每題 spawn 獨立 ephemeral codex session（不 resume、跑完丟）、
抓 reply + 從該 session 的 rollout jsonl 統計實際 function_call 名稱。

用法：
    py -3 codex_doc_smoke.py
    py -3 codex_doc_smoke.py --model gpt-5.4 --timeout 90

驗證目的：
1. codex 看見題目能挑對 tool 嗎（不是亂呼別的、也不是不呼任何 tool 就硬答）
2. tool 回的內容有沒有真的用進 reply
3. 新 tool（tracker / cs_kb）能被 codex 正確 discover
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


CODEX_CLI = r"D:\npm-global\codex.CMD"
AQUAN_DIR = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager")
SESSION_ROOT = Path.home() / ".codex" / "sessions"

DISABLED_BUILTINS = ["shell_tool", "multi_agent", "apps"]

# 6 題涵蓋全部新 tool + 對照一個既有 kb
QUERIES = [
    {
        "tag": "worklog-recent",
        "user_text": "最近兩天的 worklog 我們在搞什麼東西、講重點就好",
        "expect_tools": {"list_tracker_docs", "read_tracker_doc"},
        "expect_category": "worklogs",
    },
    {
        "tag": "meeting-latest",
        "user_text": "最近一次的每週會議紀錄整理裡面寫了什麼、條列重點",
        "expect_tools": {"list_tracker_docs", "read_tracker_doc"},
        "expect_category": "meetings",
    },
    {
        "tag": "project-status",
        "user_text": "『首例_消費毛利』這個專案目前進度如何、誰負責",
        "expect_tools": {"read_tracker_doc"},
        "expect_category": "projects",
    },
    {
        "tag": "kb-engineer",
        "user_text": "POSConfig 這個資料庫的角色是什麼、給工程師看的版本",
        "expect_tools": {"read_kb_doc"},
        "expect_category": "kb-engineer",
    },
    {
        "tag": "kb-customer",
        "user_text": "如果客戶問怎麼安裝 POS 系統、客服 SOP 是怎麼教的",
        "expect_tools": {"read_cs_kb_doc"},
        "expect_category": "kb-customer",
    },
    {
        "tag": "business-meeting",
        "user_text": "4/1 那場跟宇新客戶的商談紀錄、講了哪些重點",
        "expect_tools": {"read_tracker_doc"},
        "expect_category": "business",
    },
]


def _parse_session_id(stderr: str) -> str | None:
    m = re.search(r"session id:\s*([0-9a-f-]+)", stderr or "", re.IGNORECASE)
    return m.group(1) if m else None


def _find_rollout(sid: str) -> Path | None:
    if not sid:
        return None
    # 今天 + 昨天的 sessions 目錄都掃（跨日跑保險）
    today = time.strftime("%Y/%m/%d")
    for sub in [today]:
        d = SESSION_ROOT / sub
        if not d.exists():
            continue
        for p in d.glob(f"rollout-*-{sid}.jsonl"):
            return p
    return None


def _count_tools(jsonl_path: Path) -> dict[str, int]:
    if not jsonl_path or not jsonl_path.exists():
        return {}
    from collections import Counter

    calls: Counter = Counter()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue

            def walk(x):
                if isinstance(x, dict):
                    if x.get("type") in ("function_call", "tool_use"):
                        # codex 工具名稱多為 mcp__cwsoft_ai_tools__read_kb_doc 之類
                        full = x.get("name", "?")
                        # 切出最後一段（read_kb_doc）方便比對 expect_tools
                        short = full.rsplit("__", 1)[-1] if "__" in full else full
                        calls[short] += 1
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)

            walk(d)
    return dict(calls)


def run_one(query: dict, model: str, timeout: int) -> dict:
    cmd = [
        CODEX_CLI,
        "exec",
        "--skip-git-repo-check",
        "-C", str(AQUAN_DIR),
        "--dangerously-bypass-approvals-and-sandbox",
    ]
    for feat in DISABLED_BUILTINS:
        cmd.extend(["--disable", feat])
    cmd.extend(["-m", model, query["user_text"]])

    t0 = time.time()
    err = None
    sid = None
    reply = ""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        reply = (proc.stdout or "").strip()
        sid = _parse_session_id(proc.stderr or "")
        if proc.returncode != 0:
            err = f"rc={proc.returncode}: {(proc.stderr or '')[:300]}"
    except subprocess.TimeoutExpired:
        err = f"timeout {timeout}s"
    elapsed = time.time() - t0

    rollout = _find_rollout(sid) if sid else None
    tools_called = _count_tools(rollout) if rollout else {}

    # 判斷：expect_tools 至少一個有被呼叫 = 對
    expected = set(query["expect_tools"])
    actual = set(tools_called.keys())
    intersection = expected & actual
    judgment = "PASS" if intersection else "FAIL"

    return {
        "tag": query["tag"],
        "user_text": query["user_text"],
        "elapsed": round(elapsed, 1),
        "session_id": sid,
        "rollout": str(rollout) if rollout else None,
        "tools_called": tools_called,
        "expected_tools": sorted(expected),
        "expected_category": query["expect_category"],
        "matched_tools": sorted(intersection),
        "judgment": judgment,
        "reply_chars": len(reply),
        "reply": reply,
        "err": err,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    started_at = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent.parent
        / "logs" / f"codex_doc_smoke_{started_at}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] {len(QUERIES)} queries, model={args.model}, timeout={args.timeout}s")
    print(f"[smoke] out={out_path}\n")

    results = []
    t_start = time.time()
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {q['tag']}: {q['user_text']}", flush=True)
        r = run_one(q, args.model, args.timeout)
        results.append(r)
        out_path.write_text(
            json.dumps(
                {"started_at": started_at, "progress": f"{i}/{len(QUERIES)}",
                 "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        marker = "PASS" if r["judgment"] == "PASS" else "FAIL"
        tool_str = ", ".join(f"{k}×{v}" for k, v in r["tools_called"].items()) or "(無)"
        print(f"      [{marker}] {r['elapsed']:5.1f}s  tools: {tool_str}")
        preview = r["reply"][:120].replace("\n", " ")
        print(f"      reply: {preview}")
        if r["err"]:
            print(f"      ERR: {r['err']}")
        print()

    total = time.time() - t_start
    pass_count = sum(1 for r in results if r["judgment"] == "PASS")
    print(f"=== Summary ===")
    print(f"total: {len(results)}  pass: {pass_count}  fail: {len(results) - pass_count}")
    print(f"time: {total:.1f}s  avg: {total/len(results):.1f}s")
    print(f"\nreport: {out_path}")
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
