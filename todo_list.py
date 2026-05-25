"""
todo_list.py — 排程任務 SQLite storage（v2: MCP-aware schema）

跟 v1 差別：
- v1 schema 只存 url（給舊 GTB cascade 用、worker 到時間 GET URL）
- v2 schema 存 tool_module / tool_func / tool_args_json（worker 到時間 import + call MCP function）

v1 schema 不留向下相容（colombo 5/24 拍板「不會走回頭路用舊版 GTB」、5/25 動工切 v2）。

寫入由 cwsoft_sqlserver_mcp/scheduling.py 的 @schedulable decorator 處理、
本模組只提供 path + init schema + worker 端 dispatch helpers。

對應計畫書：general-task-bot/docs/mcp_todo_worker_開發計畫.md § 3
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

TODO_DB_PATH = str(Path(__file__).resolve().parent / "todo_list.db")


SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS todo_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    run_at           TEXT NOT NULL,
    tool_module      TEXT NOT NULL,
    tool_func        TEXT NOT NULL,
    tool_args_json   TEXT NOT NULL,
    user_text        TEXT NOT NULL DEFAULT '',
    note             TEXT NOT NULL DEFAULT '',
    state            TEXT NOT NULL DEFAULT 'pending',
    executed_at      TEXT,
    execution_result TEXT
)
"""

INDEX_V2 = """
CREATE INDEX IF NOT EXISTS idx_todo_pending_due
ON todo_items(state, run_at)
"""


def init_todo_db(db_path: str = TODO_DB_PATH) -> None:
    """確保 todo_list.db 跟 v2 schema 存在。

    若舊 v1 schema 還在（含 'url' 欄、無 'tool_module' 欄）、會 DROP 重建——v1 內容不保留。
    舊 db 備份在外部處理（cp .bak.<時間戳>）、本函式不負責備份。
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='todo_items'"
        ).fetchone()
        existing_sql = (existing[0] if existing else "") or ""
        if "url" in existing_sql and "tool_module" not in existing_sql:
            print("[init_todo_db] 偵測到 v1 schema（含 url 欄）、DROP 重建為 v2")
            cur.execute("DROP TABLE todo_items")
        cur.execute(SCHEMA_V2)
        cur.execute(INDEX_V2)
        conn.commit()
    finally:
        conn.close()
