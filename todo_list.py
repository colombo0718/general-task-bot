import sqlite3
from datetime import datetime

TODO_DB_PATH = "todo_list.db"

def init_todo_db(db_path: str = TODO_DB_PATH):
    """確保 todo_list.db 與 todo_items 表存在"""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS todo_items (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT NOT NULL,
            run_at    TEXT NOT NULL,
            url       TEXT NOT NULL,
            user_text TEXT NOT NULL,
            state     TEXT NOT NULL DEFAULT 'pending'
        )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_todo_item(user_id: str, run_at: str, url: str, user_text: str,
                     db_path: str = TODO_DB_PATH) -> int:
    """
    新增一筆排程任務，回傳新插入的 id
    state 預設為 pending（未完成）
    """
    init_todo_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO todo_items (user_id, run_at, url, user_text, state)
            VALUES (?, ?, ?, ?, 'pending')
        """, (user_id, run_at, url, user_text))

        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def view_todo_list(user_id: str, db_path: str = TODO_DB_PATH) -> list[tuple[int, str]]:
    """
    回傳使用者的待辦清單：[(id, user_text), ...]
    只列 state='pending'（未完成）的任務
    """
    init_todo_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_text
            FROM todo_items
            WHERE user_id = ?
              AND state = 'pending'
            ORDER BY run_at ASC
        """, (user_id,))
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()