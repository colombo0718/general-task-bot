# gtb_dev.py 對話流程與資料庫調用說明

> 適用版本：gtb_dev.py（含 db_helper.py / llm_clients.py）
> 整理日期：2026-04-27

---

## 一、啟動初始化

```
python gtb_dev.py --conf cs --port 6001
```

| 動作 | 說明 |
|------|------|
| 讀取 `config/prompts{suffix}.ini` + `prompts_system.ini` | 載入所有 extractor prompt |
| 讀取 `config/mission{suffix}.json` | 載入任務定義（分類樹 + 欄位規格） |
| `init_todo_db(DB_PATH)` | 建立 `database/todo_list.db`（排程） |
| `db_helper.init_config_db(CONFIG_DB_PATH)` | 建立 `database/config.db`（user registry） |
| `start_todo_worker(DB_PATH)` | 啟動背景 thread，每 60s 掃排程任務 |
| `app.run(..., use_reloader=False)` | 啟動 Flask，關閉 auto-reload |

---

## 二、每則訊息的完整流程

### 入口：`POST /callback/<oaid>`

```
LINE 傳訊息
  │
  ├─ 驗證 X-Line-Signature（HMAC-SHA256）
  │    └─ 失敗 → abort(400)
  │
  ├─ 過濾 event
  │    └─ msg_type 不是 text / image → 跳過
  │
  ├─ 讀取 LINE 使用者資料（display_name）
  │    └─ group / room / 1-on-1 三種 API 分支
  │
  ├─── 影子模式（--mode shadow）─────────────────→ 見第三節
  │
  └─── 一般模式（預設）─────────────────────────→ 見第四節
```

---

## 三、影子模式分支（Shadow Mode）

目的：**只觀測、不回覆**，儲存使用者行為供分析。

```
DB 調用：
  shadow.db / users         ← upsert_user()
  shadow.db / messages_{uid} ← save_shadow_message()

LLM 調用：
  1. run_extractor(classify_tree prompt_key)  → 分類意圖
  2. gather_fields(task_id, ...)              → 抽取欄位（信心值不觸發澄清）
  3. execute_command("GET", cmd_url)          → HTTP GET 取得 AI 草稿

回覆：不回覆（continue 跳過 reply_text）
```

---

## 四、一般模式完整流程

### 4-0 取得 / 建立 conv_db

```python
conv_db = db_helper.get_or_create_conv_db(
    CONFIG_DB_PATH, oaid, user_id, display_name, PROJECT_DIR
)
```

| 資料庫 | 表 | 動作 |
|--------|-----|------|
| `config.db` | `users` | UPSERT（oaid, user_id 為複合唯一鍵）|
| `conv_{uid}.db` | — | 若不存在則 init_conv_db()，建立四張表 |

---

### 4-1 澄清分支（待使用者選擇客戶名稱）

```python
pending_clar = db_helper.get_pending_clarification(conv_db)
```

**`get_pending_clarification` 內部邏輯：**

```
clarification_log WHERE state='pending' ORDER BY id DESC LIMIT 1
  │
  ├─ 找不到 → 回傳 None
  ├─ 找到，但建立超過 5 分鐘 → UPDATE state='cancelled'，回傳 None
  └─ 找到且未逾時 → 回傳 dict（含 options、context）
```

#### 有 pending_clar → 進入澄清分支

```
DB 調用順序：

1. log_message(conv_db, session_id, "in", user_text, content_type, "answer")
   └─ message_log INSERT（此訊息是澄清問答的答案）

2. 解析使用者輸入：
   "1"/"2"/"3" → options[idx]
   直接在 options 中 → 直接採用
   其他自由輸入 → 直接採用（resolved = stripped）

3. resolve_clarification(conv_db, pending_clar["id"], user_text, resolved)
   └─ clarification_log UPDATE state='resolved', user_input=, resolved_value=

4. save_correction(conv_db, field, raw_input, resolved)
   └─ corrections INSERT 或 UPDATE use_count（學習這次對應）

5. 從 context_json 取回任務狀態：
   values = ctx["values"]   # 澄清前的其他欄位值
   task_id = ctx["task_id"]
   run_at  = ctx["run_at"]
   values[field] = resolved  # 填入本次確認值

6. build_command(action, values) → cmd_url（純計算，不用 DB）

7. save_decision(conv_db, session_id, task_id, values, cmd_url, run_at, state=...)
   └─ decisions INSERT
   ├─ human_check=true/auto → state='awaiting_confirmation'
   └─ human_check=false    → state='queued'

8. execute_command("GET", cmd_url, user_id) → HTTP GET（不用 DB）

9. reply_text(line_bot_api, ...) → 回覆 LINE

10. log_message(conv_db, session_id, "out", reply, "text", out_role)
    └─ message_log INSERT
    out_role = "confirm_request" (human_check) 或 "result"

→ continue（進下一則訊息）
```

---

### 4-2 一般 session 起點

若無 pending_clar，建立新 session：

```python
session_id = db_helper.new_session_id(user_id)
# md5(user_id + timestamp)[:16]，純計算

db_helper.log_message(conv_db, session_id, "in", user_text, content_type, "claim")
# message_log INSERT，role="claim"（主動發起的訊息）
```

---

### 4-3 決議確認分支（human_check 等待回覆）

```python
pending_dec = db_helper.get_pending_decision(conv_db)
# decisions WHERE state='awaiting_confirmation' ORDER BY id DESC LIMIT 1
```

#### 有 pending_dec → 使用者在回覆是否執行上一個指令

```
LLM 調用：
  run_extractor("extracted_ronot", user_text) → "true" / "false" / "null"

分支：
  extracted_ronot = "true"
  OR (extracted_ronot = "null" AND human_check = "auto")
  → 確認執行
      run_at = "now"  → execute_command("POST", url)
                         update_decision_state(conv_db, id, "executed")
                         └─ decisions UPDATE executed_at=now
      run_at = 未來   → insert_todo_item(todo_list.db)
                         update_decision_state(conv_db, id, "queued", todo_ref_id=todo_id)
                         └─ decisions UPDATE state='queued', todo_ref_id=

  extracted_ronot = "false"
  OR (extracted_ronot = "null" AND human_check = "true")
  → 取消執行
      update_decision_state(conv_db, id, "cancelled")
      └─ decisions UPDATE state='cancelled'

最後：
  log_message(conv_db, session_id, "out", reply, "text", "result")
  return "OK", 200（直接結束，不走主流程）
```

---

### 4-4 查詢待辦清單

```python
view_todos = run_extractor("extract_view_todos", user_text)
# LLM 判斷是否「查詢待辦」
```

若 `view_todos = "true"` → `view_todo_list(user_id, db_path=DB_PATH)` → 回覆 + `log_message` → return

---

### 4-5 主流程（新任務）

```
LLM 調用：
  1. run_extractor("extract_run_at", user_text)   → "now" 或 ISO datetime
  2. run_extractor(classify_tree prompt_key)       → 分類意圖
  3. gather_fields(task_id, mission_data, user_text)
       └─ 對每個欄位 run_extractor(prompt_key, ...)
       └─ 若 meta.with_confidence=true → 解析 JSON { value, confidence }
       └─ 若 meta.match_pool → extract_top_matching_names(val, candidates)
            ├─ 最高分 >= 0.8 且 LLM 信心 >= 70 → 直接採用
            └─ 否則 → clarification_needed = { field, raw_input, top_candidates }

分支 A：缺少必填欄位（missing）
  log_message("out", ..., "error") → reply → continue

分支 B：信心不足，需澄清（clarification_needed）

  DB 調用：
  1. save_clarification(conv_db, session_id, field, question, options,
                        confidence_in=conf_in, context=context)
     └─ clarification_log INSERT state='pending'
     context = { task_id, values, run_at, raw_input }（重啟後繼續用）

  2. reply_quick_reply(line_bot_api, ..., question, options)
     └─ LINE 發出 Quick Reply 按鈕

  3. log_message(conv_db, session_id, "out", question, "text", "clarify")
     └─ message_log INSERT

  → continue（等下一則訊息走 4-1 澄清分支）

分支 C：正常完成萃取

  build_command(action, values) → cmd_url（純計算）

  DB 調用（依 human_check 值）：
  ├─ human_check = "true"  → save_decision(..., state="awaiting_confirmation")
  ├─ human_check = "false" → save_decision(..., state="queued")
  └─ human_check = "auto"  → save_decision(..., state="awaiting_confirmation")
     └─ decisions INSERT

  execute_command("GET", cmd_url, user_id) → HTTP GET（預覽結果，不動 DB）

  reply_text(line_bot_api, ...) → 回覆 LINE

  log_message(conv_db, session_id, "out", reply, "text", out_role)
  └─ message_log INSERT
  out_role = "confirm_request" (human_check) 或 "result"
```

---

## 五、資料庫調用總覽

| 時機 | 表 | 操作 | role / state |
|------|-----|------|-------------|
| 收到任何訊息（一般模式） | `config.db / users` | UPSERT | — |
| 收到任何訊息（一般模式） | `clarification_log` | SELECT（檢查 pending） | — |
| 澄清逾時（>5 分鐘） | `clarification_log` | UPDATE | cancelled |
| 澄清分支：收到答案 | `message_log` | INSERT | in / answer |
| 澄清分支：確認答案 | `clarification_log` | UPDATE | resolved |
| 澄清分支：學習對應 | `corrections` | INSERT / UPDATE use_count | — |
| 澄清分支：建立決議 | `decisions` | INSERT | awaiting_confirmation / queued |
| 澄清分支：回覆使用者 | `message_log` | INSERT | out / confirm_request or result |
| 新 session 起點 | `message_log` | INSERT | in / claim |
| 決議確認分支 | `decisions` | SELECT（檢查 awaiting_confirmation） | — |
| 決議確認→確認執行（now） | `decisions` | UPDATE | executed |
| 決議確認→確認執行（排程） | `decisions` | UPDATE | queued + todo_ref_id |
| 決議確認→取消 | `decisions` | UPDATE | cancelled |
| 決議確認→回覆使用者 | `message_log` | INSERT | out / result |
| 主流程→需澄清 | `clarification_log` | INSERT | pending |
| 主流程→需澄清→回覆 | `message_log` | INSERT | out / clarify |
| 主流程→正常完成 | `decisions` | INSERT | awaiting_confirmation / queued |
| 主流程→正常完成→回覆 | `message_log` | INSERT | out / confirm_request or result |

---

## 六、session_id 的生命週期

```
一個任務 = 一個 session_id
  │
  ├─ 新訊息進來（claim）
  │    session_id = md5(user_id + timestamp)[:16]
  │
  ├─ 若觸發澄清
  │    → clarification_log.session_id = 此 session_id（pending）
  │    → 等待使用者回答（可能跨幾則訊息）
  │    → 澄清分支用 pending_clar["session_id"] 接回同一個 session
  │
  └─ 任務完成
       decisions.session_id = 此 session_id
       message_log 的 in / out 都綁同一個 session_id
       → 可用 session_id JOIN 三張表重建完整對話紀錄
```

---

## 七、LLM 調用次數（每則訊息最多）

| 調用 | Extractor | 目的 |
|------|-----------|------|
| 1 | `extracted_ronot`（有 pending_dec 時） | 判斷確認/取消 |
| 2 | `extract_view_todos` | 判斷是否查待辦 |
| 3 | `extract_run_at` | 萃取執行時間 |
| 4 | `classify_tree.prompt_key` | 分類意圖 |
| 5~N | 各欄位 `prompt_key` | gather_fields 萃取欄位 |

澄清分支不重新跑 LLM，直接用 `context_json` 恢復任務狀態。
