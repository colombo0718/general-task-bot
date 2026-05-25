"""
todo_worker.py — 排程任務 worker（v2：MCP-aware dispatcher）

跟 v1 差別：
- v1：到時間 GET url（todo_items.url 欄、跟 sqlgate cascade 配套）
- v2：到時間 `import tool_module` + `getattr(tool_func)` + `func(**args)`、直接 Python call

CHECK_INTERVAL 1 小時 → 30 秒（老闆需求 5/31 23:59:59 觸發、1 小時粒度不夠）

對應計畫書：general-task-bot/docs/mcp_todo_worker_開發計畫.md § 5
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from todo_list import TODO_DB_PATH, init_todo_db


CHECK_INTERVAL = 30  # 秒、v1 是 60*60 (1 小時)

NOTEBOOK_DIR = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\notebooks")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fetch_pending_due(conn) -> list:
    """撈 state='pending' 且 run_at <= now 的 rows"""
    cur = conn.execute("""
        SELECT id, user_id, run_at, tool_module, tool_func, tool_args_json, note
        FROM todo_items
        WHERE state = 'pending' AND run_at <= ?
        ORDER BY run_at, id
    """, (now_iso(),))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def dispatch_one(row: dict) -> dict:
    """執行一筆 row、回原 function 的 dict 結果（不抓 exception、讓 caller try/except）"""
    mod = importlib.import_module(row["tool_module"])
    func = getattr(mod, row["tool_func"])
    args = json.loads(row["tool_args_json"])
    return func(**args)


def mark_executed(conn, schedule_id: int, result: dict) -> None:
    conn.execute("""
        UPDATE todo_items
        SET state = 'executed', executed_at = ?, execution_result = ?
        WHERE id = ?
    """, (now_iso(), json.dumps(result, ensure_ascii=False, default=str), schedule_id))
    conn.commit()


def mark_failed(conn, schedule_id: int, err_msg: str) -> None:
    conn.execute("""
        UPDATE todo_items
        SET state = 'failed', executed_at = ?, execution_result = ?
        WHERE id = ?
    """, (now_iso(), err_msg[:2000], schedule_id))
    conn.commit()


def write_action_notebook(row: dict, result: dict, ok: bool) -> None:
    """寫一條 [ACTION] 進共享 notebook、留 audit"""
    try:
        NOTEBOOK_DIR.mkdir(exist_ok=True)
        now = datetime.now()
        nb = NOTEBOOK_DIR / f"notebook_{now.strftime('%Y%m%d')}.md"
        ts = now.strftime("%H:%M")
        args_short = row["tool_args_json"][:100]
        note_part = f"、原排程 note: {row['note']}" if row["note"] else ""
        status = "已執行" if ok else "執行失敗"
        result_short = json.dumps(result, ensure_ascii=False, default=str)[:200]
        line = (f"- {ts} [worker:scheduled] [ACTION] {row['tool_func']}({args_short}) "
                f"{status}（排程觸發、schedule_id={row['id']}{note_part}、結果: {result_short}）\n")

        new_file = not nb.exists()
        with open(nb, "a", encoding="utf-8") as f:
            if new_file:
                f.write(f"# Notebook {now.strftime('%Y-%m-%d')}\n\n")
            f.write(line)
    except Exception as e:
        print(f"[WORKER] notebook write failed (non-fatal): {e}")


def run_worker():
    print(f"[WORKER] started、db={TODO_DB_PATH}、interval={CHECK_INTERVAL}s")
    init_todo_db()

    while True:
        try:
            conn = sqlite3.connect(TODO_DB_PATH)
            try:
                rows = fetch_pending_due(conn)
                if rows:
                    print(f"[WORKER] {now_iso()} found {len(rows)} due item(s)")
                for row in rows:
                    sid = row["id"]
                    print(f"[WORKER] dispatching id={sid} {row['tool_module']}.{row['tool_func']}({row['tool_args_json'][:80]})")
                    try:
                        result = dispatch_one(row)
                        # 約定原 function 回 {"ok": True/False, ...}
                        ok = bool(result.get("ok")) if isinstance(result, dict) else True
                        if ok:
                            mark_executed(conn, sid, result)
                            print(f"[WORKER] id={sid} OK")
                        else:
                            err = result.get("error", "原 function 回 ok=False")
                            mark_failed(conn, sid, f"function returned ok=False: {err}")
                            print(f"[WORKER] id={sid} func ok=False: {err}")
                        write_action_notebook(row, result, ok)
                    except Exception as e:
                        tb = traceback.format_exc()
                        err_msg = f"{type(e).__name__}: {e}\n{tb}"
                        mark_failed(conn, sid, err_msg)
                        print(f"[WORKER] id={sid} EXCEPTION: {e}")
                        write_action_notebook(row, {"ok": False, "error": str(e)}, False)
            finally:
                conn.close()
        except Exception as e:
            print(f"[WORKER] poll loop error (non-fatal): {e}")
            traceback.print_exc()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # 讓 dispatch 能 import cwsoft_sqlserver_mcp.server
    sys.path.insert(0, r"C:\Users\pos\cwsoft-sqlserver-mcp\src")
    run_worker()
