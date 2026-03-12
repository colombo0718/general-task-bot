# todo_worker.py
import time
import sqlite3
from datetime import datetime

import requests

from todo_list import TODO_DB_PATH, init_todo_db

CHECK_INTERVAL = 60 * 60  # 每 1 小時檢查一次


def now_iso() -> str:
    # 需確保 run_at 也用可比較的 ISO 格式（YYYY-MM-DDTHH:MM:SS）
    return datetime.now().isoformat(timespec="seconds")


def execute_command(method: str, url: str, user_id: str) -> bool:
    """
    極簡執行器：只負責打 HTTP
    - method: "GET" / "POST"
    - url: 任務 URL
    - user_id: 目前僅用於 log
    """
    try:
        print(f"[EXEC] {method} {url} (user={user_id})")

        if method.upper() == "POST":
            r = requests.post(url, timeout=30)
        else:
            r = requests.get(url, timeout=30)

        print(f"[DONE] status={r.status_code}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def run_worker():
    print("[WORKER] started")
    init_todo_db()

    while True:
        conn = sqlite3.connect(TODO_DB_PATH)
        cur = conn.cursor()

        cur.execute("""
            SELECT id, user_id, url
            FROM todo_items
            WHERE state = 'pending'
              AND run_at <= ?
            ORDER BY id ASC
        """, (now_iso(),))

        rows = cur.fetchall()

        if rows:
            print(f"[WORKER] found {len(rows)} task(s)")

        for task_id, user_id, url in rows:
            print(f"[WORKER] executing id={task_id}")

            ok = execute_command("POST", url, user_id)

            if ok:
                cur.execute("""
                    UPDATE todo_items
                    SET state = 'done'
                    WHERE id = ?
                """, (task_id,))
                conn.commit()
                print(f"[WORKER] done id={task_id}")

        conn.close()

        print("[WORKER] sleep 1 hour\n")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_worker()
