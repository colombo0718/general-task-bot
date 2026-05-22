"""
db_helper.py — GTB 通用資料庫層

每個專案的 database/ 下：
  config.db              ← user registry (oaid + user_id → conv db path)
  conv_{user_id}.db      ← 每個 user 的對話 DB（4 張表）
  todo_list.db           ← 排程執行佇列（todo_worker 專用，不動）
"""

import os, re, sqlite3, json, hashlib, time
from datetime import datetime


# ── config.db ─────────────────────────────────────────────────────────────────

def init_config_db(config_db_path: str):
    conn = sqlite3.connect(config_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            oaid         TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            db_name      TEXT NOT NULL,
            display_name TEXT,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen    DATETIME,
            UNIQUE(oaid, user_id)
        )
    """)
    conn.commit()
    conn.close()


def get_or_create_conv_db(config_db_path: str, oaid: str, user_id: str,
                           display_name: str, db_dir: str) -> str:
    """回傳該 user 的 conv DB 完整路徑，不存在時自動建立 registry 紀錄。

    db_dir 是放置 conv_*.db 的目錄絕對路徑（正式版用 <project>/database，
    壓測 / dev 版可指向 <project>/database_dev）。
    """
    db_name = f"conv_{_safe(user_id)}.db"
    conv_path = os.path.join(db_dir, db_name)
    now = _now()

    conn = sqlite3.connect(config_db_path)
    conn.execute("""
        INSERT INTO users (oaid, user_id, db_name, display_name, created_at, last_seen)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(oaid, user_id) DO UPDATE SET
            display_name = excluded.display_name,
            last_seen    = excluded.last_seen
    """, (oaid, user_id, db_name, display_name, now, now))
    conn.commit()
    conn.close()

    init_conv_db(conv_path)
    return conv_path


# ── conv_{user_id}.db ─────────────────────────────────────────────────────────

def init_conv_db(conv_db_path: str):
    conn = sqlite3.connect(conv_db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS message_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            direction    TEXT NOT NULL,
            content      TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            role         TEXT DEFAULT 'unknown',
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clarification_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL,
            field_name    TEXT NOT NULL,
            question      TEXT NOT NULL,
            options       TEXT NOT NULL,
            confidence_in INTEGER,
            context_json  TEXT,
            user_input    TEXT,
            resolved_value TEXT,
            state         TEXT DEFAULT 'pending',
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at   DATETIME
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name      TEXT NOT NULL,
            raw_input       TEXT NOT NULL,
            corrected_value TEXT NOT NULL,
            source          TEXT DEFAULT 'clarification',
            use_count       INTEGER DEFAULT 0,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_corrections_field_raw
        ON corrections(field_name, raw_input)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            task_id      TEXT NOT NULL,
            field_values TEXT NOT NULL,
            url          TEXT NOT NULL,
            run_at       TEXT DEFAULT 'now',
            todo_ref_id  INTEGER,
            state        TEXT DEFAULT 'queued',
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            executed_at  DATETIME
        )
    """)

    conn.commit()
    conn.close()


# ── session_id ────────────────────────────────────────────────────────────────

def new_session_id(user_id: str) -> str:
    return hashlib.md5(f"{user_id}:{time.time()}".encode()).hexdigest()[:16]


# ── message_log ───────────────────────────────────────────────────────────────

def log_message(conv_db_path: str, session_id: str, direction: str,
                content: str, content_type: str = "text", role: str = "unknown"):
    _write(conv_db_path,
           "INSERT INTO message_log (session_id, direction, content, content_type, role) VALUES (?,?,?,?,?)",
           (session_id, direction, content, content_type, role))


# ── clarification_log ─────────────────────────────────────────────────────────

def save_clarification(conv_db_path: str, session_id: str, field_name: str,
                        question: str, options: list, confidence_in: int = None,
                        context: dict = None):
    """context 存放繼續流程所需的完整快照：task_id, values, run_at 等。"""
    _write(conv_db_path,
           "INSERT INTO clarification_log (session_id, field_name, question, options, confidence_in, context_json) VALUES (?,?,?,?,?,?)",
           (session_id, field_name, question, json.dumps(options, ensure_ascii=False),
            confidence_in, json.dumps(context, ensure_ascii=False) if context else None))


CLARIFICATION_TIMEOUT_SECONDS = 300  # 5 分鐘內沒回應就自動作廢

def get_pending_clarification(conv_db_path: str) -> dict | None:
    """回傳最新一筆 state='pending' 且未逾時的澄清紀錄，或 None。逾時的自動 cancel。"""
    conn = sqlite3.connect(conv_db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM clarification_log WHERE state='pending' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        conn.close()
        return None
    d = dict(row)
    created = datetime.strptime(d["created_at"], "%Y-%m-%d %H:%M:%S")
    age = (datetime.now() - created).total_seconds()
    if age > CLARIFICATION_TIMEOUT_SECONDS:
        conn.execute(
            "UPDATE clarification_log SET state='cancelled', resolved_at=? WHERE id=?",
            (_now(), d["id"])
        )
        conn.commit()
        conn.close()
        return None
    conn.close()
    d["options"] = json.loads(d["options"] or "[]")
    d["context"] = json.loads(d["context_json"] or "{}") if d.get("context_json") else {}
    return d


def resolve_clarification(conv_db_path: str, clarification_id: int,
                           user_input: str, resolved_value: str):
    _write(conv_db_path,
           "UPDATE clarification_log SET state='resolved', user_input=?, resolved_value=?, resolved_at=? WHERE id=?",
           (user_input, resolved_value, _now(), clarification_id))


def cancel_clarification(conv_db_path: str, clarification_id: int):
    _write(conv_db_path,
           "UPDATE clarification_log SET state='cancelled', resolved_at=? WHERE id=?",
           (_now(), clarification_id))


# ── corrections ───────────────────────────────────────────────────────────────

def save_correction(conv_db_path: str, field_name: str, raw_input: str, corrected_value: str):
    conn = sqlite3.connect(conv_db_path)
    existing = conn.execute(
        "SELECT id FROM corrections WHERE field_name=? AND raw_input=?",
        (field_name, raw_input)
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE corrections SET corrected_value=?, use_count=use_count+1 WHERE id=?",
            (corrected_value, existing[0])
        )
    else:
        conn.execute(
            "INSERT INTO corrections (field_name, raw_input, corrected_value) VALUES (?,?,?)",
            (field_name, raw_input, corrected_value)
        )
    conn.commit()
    conn.close()


def lookup_correction(conv_db_path: str, field_name: str, raw_input: str) -> str | None:
    """回傳已知的勘誤值，找不到回傳 None。"""
    conn = sqlite3.connect(conv_db_path)
    row = conn.execute(
        "SELECT corrected_value FROM corrections WHERE field_name=? AND raw_input=? ORDER BY use_count DESC LIMIT 1",
        (field_name, raw_input)
    ).fetchone()
    conn.close()
    return row[0] if row else None


# ── decisions ─────────────────────────────────────────────────────────────────

def save_decision(conv_db_path: str, session_id: str, task_id: str,
                  field_values: dict, url: str, run_at: str = "now",
                  state: str = "queued") -> int:
    conn = sqlite3.connect(conv_db_path)
    cur = conn.execute(
        "INSERT INTO decisions (session_id, task_id, field_values, url, run_at, state) VALUES (?,?,?,?,?,?)",
        (session_id, task_id, json.dumps(field_values, ensure_ascii=False), url, run_at, state)
    )
    decision_id = cur.lastrowid
    conn.commit()
    conn.close()
    return decision_id


def get_pending_decision(conv_db_path: str) -> dict | None:
    """回傳最新一筆 state='awaiting_confirmation' 的決議，或 None。"""
    conn = sqlite3.connect(conv_db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM decisions WHERE state='awaiting_confirmation' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["field_values"] = json.loads(d["field_values"] or "{}")
    return d


def update_decision_state(conv_db_path: str, decision_id: int, state: str,
                           todo_ref_id: int = None):
    if todo_ref_id is not None:
        _write(conv_db_path,
               "UPDATE decisions SET state=?, todo_ref_id=?, executed_at=? WHERE id=?",
               (state, todo_ref_id, _now(), decision_id))
    else:
        _write(conv_db_path,
               "UPDATE decisions SET state=?, executed_at=? WHERE id=?",
               (state, _now(), decision_id))


# ── 內部工具 ──────────────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write(db_path: str, sql: str, params: tuple):
    conn = sqlite3.connect(db_path)
    conn.execute(sql, params)
    conn.commit()
    conn.close()
