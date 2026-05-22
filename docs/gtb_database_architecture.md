# GTB 通用資料庫架構

> 適用版本：gtb.py（新版多專案架構）
> 設計日期：2026-04-27

---

## 設計原則

- 每個專案（如 cwsoft-aquan-manager）的 `database/` 資料夾自帶所有 DB
- 一個專案可接多個 LINE OA，每個 OA 可有多個 user
- User 資料完全隔離，一人一個 conv DB
- `todo_list.db` 獨立，供 todo_worker 統一掃描，不依賴 conv DB

---

## 目錄結構

```
database/
  config.db               ← User registry（oaid、userid、db 對應表）
  todo_list.db            ← 排程執行佇列（todo_worker 專用，現有）
  conv_{userid}.db        ← 每個 user 的對話資料庫（動態產生）
```

---

## config.db

記錄每個 user 的基本資訊與對應的 conv DB 路徑。
每次 webhook 收到訊息，先查此表找到該 user 的 conv DB 再繼續處理。

```sql
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    oaid        TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    db_name     TEXT NOT NULL,          -- conv_{user_id}
    display_name TEXT,                  -- LINE 顯示名稱（選填）
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen   DATETIME,
    UNIQUE(oaid, user_id)
);
```

> LINE 的 user_id 在同一個 OA 內是唯一的，(oaid, user_id) 作為複合唯一鍵。

---

## conv_{userid}.db

每個 user 獨立一個 SQLite 檔，內含四張表。

---

### 1. message_log（流水）

記錄所有進出訊息，append-only，不做狀態判斷。

```sql
CREATE TABLE message_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,                  -- 同一輪任務共用，可 NULL
    direction    TEXT NOT NULL,         -- 'in' | 'out'
    content      TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',  -- 'text' | 'image'
    role         TEXT DEFAULT 'unknown',
    -- in:  'claim' | 'answer' | 'confirm' | 'cancel' | 'query' | 'unknown'
    -- out: 'result' | 'clarify' | 'confirm_request' | 'draft' | 'error'
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

### 2. clarification_log（問答）

信心值不足時暫停主流程，等 user 回覆確認欄位值。
`state = 'pending'` 同時作為當前對話狀態（取代 in-memory `todo_command`）。

```sql
CREATE TABLE clarification_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL,
    field_name       TEXT NOT NULL,     -- 哪個欄位在問（如 identify_name）
    question         TEXT NOT NULL,     -- 發給 user 的問題文字
    options          TEXT NOT NULL,     -- JSON array，top-3 候選
    confidence_in    INTEGER,           -- 觸發時的信心值（0~100）
    context_json     TEXT,              -- JSON：task_id / values / run_at / raw_input（繼續流程用）
    user_input       TEXT,              -- user 實際回覆（按鈕或手打）
    resolved_value   TEXT,              -- 最終確認值
    state            TEXT DEFAULT 'pending',  -- 'pending' | 'resolved' | 'cancelled'
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at      DATETIME
);
```

**信心門檻**：預設 70 分以下觸發問答。

---

### 3. corrections（勘誤）

累積 user 確認過的正確對應，供下次萃取前優先比對，不再走相似度計算。

```sql
CREATE TABLE corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    field_name      TEXT NOT NULL,      -- 哪個欄位（如 identify_name）
    raw_input       TEXT NOT NULL,      -- LLM 萃取的原始值 / 語音誤識詞
    corrected_value TEXT NOT NULL,      -- user 確認的正確值
    source          TEXT DEFAULT 'clarification',  -- 'clarification' | 'manual'
    use_count       INTEGER DEFAULT 0,  -- 命中次數，評估效果用
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_corrections_field_raw ON corrections(field_name, raw_input);
```

---

### 4. decisions（決議）

記錄「意圖已確認、即將或已執行」的完整資訊。
`todo_list.db` 只是執行佇列，決議表才是完整的歷史紀錄。

```sql
CREATE TABLE decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    task_id      TEXT NOT NULL,         -- GTB task id
    field_values TEXT NOT NULL,         -- JSON，所有欄位最終確認值
    url          TEXT NOT NULL,         -- build_command 產生的 API URL
    run_at       TEXT DEFAULT 'now',    -- 'now' 或 ISO 8601 datetime
    todo_ref_id  INTEGER,               -- 若排程，對應 todo_list.db 的 rowid
    state        TEXT DEFAULT 'queued', -- 'awaiting_confirmation' | 'queued' | 'executed' | 'cancelled'
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    executed_at  DATETIME
);
```

---

## todo_list.db（現有，不動 schema）

由 todo_worker 統一掃描，結構維持現有設計。
決議表透過 `todo_ref_id` 單向關聯到此表，worker 不需要知道 conv DB 的存在。

---

## 訊息處理流程與 DB 對應

```
LINE webhook 收到訊息
  │
  ├─ 1. 查 config.db → 取得 conv DB 路徑（若無則建立新 user）
  ├─ 2. 寫入 message_log（direction='in'）
  │
  ├─ 3. 查 clarification_log WHERE state='pending'
  │      └─ 有 → 進入欄位澄清分支
  │              ├─ 解析 user 回覆（按鈕 1/2/3 或手打）
  │              ├─ 更新 clarification_log.state = 'resolved'
  │              ├─ 寫入 corrections（學習這次對應）
  │              └─ 繼續 build_command → 寫入 decisions
  │
  ├─ 4. 查 decisions WHERE state='awaiting_confirmation'
  │      └─ 有 → 進入 human_check 確認分支（取代舊 todo_command）
  │              ├─ user 說「好/確認」→ decisions.state = 'queued' → 執行
  │              └─ user 說「取消」  → decisions.state = 'cancelled'
  │
  ├─ 5. 無 pending → 正常主流程
  │      ├─ identify_needs → identify_name（含信心值）
  │      ├─ gather_fields
  │      │    └─ 信心值 < 70 → 寫入 clarification_log（state='pending'）
  │      │                      → reply_quick_reply 發問答給 user（暫停）
  │      │    └─ 信心值 >= 70 → 繼續
  │      ├─ build_command
  │      └─ human_check?
  │           ├─ false → 寫入 decisions(queued) → 執行 → decisions(executed)
  │           └─ true  → 寫入 decisions(awaiting_confirmation) → 等 user 確認
  │                        ├─ run_at='now'  → 確認後直接執行
  │                        └─ run_at=未來  → 確認後寫入 todo_list.db
  │
  └─ 6. bot 回覆 → 寫入 message_log（direction='out'）
```

> **session_id 邊界**：一個任務從 webhook 進來到 decisions 完成為一個 session。
> clarification pending 超過設定時間未回覆，自動改為 `cancelled`。

---

## session_id 產生規則

同一輪任務（從收到訊息到決議完成）共用一個 session_id，跨表可 JOIN。

```python
import hashlib, time
session_id = hashlib.md5(f"{user_id}:{time.time()}".encode()).hexdigest()[:16]
```

---

## 各表職責一句話總結

| 表 | 職責 |
|----|------|
| `config.db / users` | 知道每個 user 的 conv DB 在哪裡 |
| `message_log` | 原始流水，什麼都記，不做判斷 |
| `clarification_log` | 信心不足時的問答過程 + 當前對話狀態 |
| `corrections` | 從問答學到的勘誤對照，下次直接用 |
| `decisions` | 已確認的指令完整紀錄 + human_check 等待確認狀態（取代 todo_command） |
| `todo_list.db` | 純執行佇列，worker 只認這個 |
