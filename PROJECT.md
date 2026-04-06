# PROJECT.md — general-task-bot (GTB)

## 一句話定位

LINE webhook Flask 服務，透過 `mission*.json` + `prompts*.ini` 設定檔驅動，
將自然語言訊息轉換為後端 API 呼叫，支援人工確認與排程延後執行。

---

## 部署方式

```bash
python main.py pos      # POS 模式，port 預設 6000
python main.py pos 6001 # 指定 port
python main.py cs       # CS 客服模式
python main.py store    # 門市模式
python main.py          # 通用記帳模式
```

- 無 CI/CD，直接在機器上跑
- Port 優先順序：env `PORT` > argv[2] > 6000
- LINE webhook 路由：`/callback/<oaid>`，oaid 對應 `oa_registry.json`

---

## 關鍵檔案

| 檔案 | 說明 |
|------|------|
| `main.py` | Flask 主程式，所有流程都在這裡 |
| `mission_pos.json` | POS 任務定義（分類樹 + 20 種任務） |
| `prompts_pos.ini` | POS 萃取器 prompt（23 個） |
| `prompts_system.ini` | 跨模式共用的 prompt（confirm、run_at 等） |
| `oa_registry.json` | LINE OA 帳號與 token 對應表（不進 git） |
| `customerlist.txt` | 客戶名稱候選清單，每次收到訊息自動重建 |
| `todo_list.db` | SQLite，排程任務 |
| `todo_list.py` | SQLite 操作 helper |
| `todo_worker.py` | 背景執行排程任務的 worker |
| `generate_customerlist_simple.py` | 從 SQL Server 重建 customerlist.txt |

---

## 架構核心：兩類設定檔驅動

```
mission*.json  → 定義「有哪些任務」以及「每個任務要哪些欄位」
prompts*.ini   → 定義「每個欄位怎麼用 LLM 萃取」
```

新增功能優先改設定檔，Python 只在框架本身需要新能力時才動。

### 設定檔配對

| 啟動參數 | prompts 檔 | mission 檔 |
|----------|-----------|------------|
| （無）   | prompts.ini | mission.json |
| pos      | prompts_pos.ini | mission_pos.json |
| store    | prompts_store.ini | mission_store.json |
| cs       | prompts_cs.ini | mission_cs.json |

---

## 一次訊息的完整流程

```
LINE webhook
  │
  ├─ 0. [若有] pending_clarification → 等待使用者選擇欄位值   ← 規劃中
  ├─ 1. extracted_ronot → 確認 / 取消前一個 todo_command
  ├─ 2. extract_view_todos → 查待辦清單
  ├─ 3. extract_run_at → 是否延後執行
  ├─ 4. identify_needs → 分類意圖，路由到 task_id
  ├─ 5. gather_fields → 對每個欄位呼叫 run_extractor
  │      └─ 有 match_pool → 拼音 + 字串相似度比對客戶名稱
  ├─ 6. build_command → 填入 url_template 產生 API URL
  ├─ 7. human_check
  │      ├─ false → 直接執行
  │      ├─ true  → 存入 todo_command，等使用者確認（預設取消）
  │      └─ auto  → 存入 todo_command，等使用者確認（預設執行）
  └─ 8. execute_command GET → 回覆預覽結果
```

---

## LLM 設定

- 目前使用：Groq / `llama-3.3-70b-versatile`
- 支援切換：`PROVIDER = "groq" | "openrouter" | "hf"`
- 統一入口：`llm_complete(user_text)` → `run_extractor()`
- API Key 放 `.env`：`GROQ_API_KEY`、`OPENROUTER_API_KEY`、`HF_TOKEN`

---

## 已知的坑與設計限制

### `todo_command` 不持久化
```python
todo_command = {}  # 純記憶體 dict
```
重啟服務後，所有待確認的指令都消失。目前接受此限制。

### 每則訊息跑一次 subprocess
```python
subprocess.run([sys.executable, "generate_customerlist_simple.py"], check=True)
```
每次收到訊息就重建 customerlist.txt，是暫時 workaround，需要 SQL Server 連線。

### human_check=true 時 GET 仍會被呼叫
設計上是讓使用者在確認前先看到 GET 預覽結果，屬於有意為之。

### extractor prompt 嚴格格式約束
每個 extractor 只能輸出特定格式（enum / 數字 / 日期），不得自由發揮。
改 prompt 時必須保留 `【輸出格式】` 那一行的正則限制。

---

## 客戶名稱比對邏輯

1. LLM 從訊息萃取原始名稱（`identify_name`）
2. 對比 `customerlist.txt` 候選清單
3. 中文走**拼音相似度**（pypinyin），英文走字串相似度
4. 門檻 0.8，超過才替換；否則保留原始萃取值
5. **規劃中**：信心不足時改為回傳 top-3 候選，發 LINE Quick Reply 詢問使用者

---

## 開發習慣

- 新任務優先改 `mission*.json` + `prompts*.ini`，不動 Python
- Python 只在框架需要新能力時才改（如新增澄清機制）
- Extractor 輸出合約不能破壞，runtime 直接用字串做判斷
- `oa_registry.json` 不進 git（含 LINE secret/token）
- 中文字串注意 encoding，統一 UTF-8
