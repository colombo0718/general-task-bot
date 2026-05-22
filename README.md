# General Task Bot（GTB）

設定驅動的 LINE Bot 任務引擎。接收 LINE 訊息，用 LLM 判斷意圖、萃取欄位，呼叫後端 API 完成任務。

---

## 設計理念

GTB 不把業務邏輯寫死在 Python 裡。新增一個任務，通常只需要改設定檔：

- `prompts*.ini` — 告訴 LLM 要萃取什麼
- `mission*.json` — 告訴系統怎麼分類意圖、需要哪些欄位、呼叫哪個 API

Python 主程式只是一個執行引擎，設定檔才是業務邏輯的所在。

---

## 兩個啟動器

### `main.py`（舊版）
設定檔與程式放在同一個資料夾，適合單一專案使用。

```bash
python main.py cs 6001
```

### `gtb.py`（新版，推薦）
從執行當下的目錄讀設定，框架與專案分離，支援多專案部署。

```bash
cd C:\path\to\your-project
python C:\path\to\general-task-bot\gtb.py --conf cs --port 6001
python C:\path\to\general-task-bot\gtb.py --conf cs --port 6003 --mode shadow
```

`gtb.py` 參數：

| 參數 | 說明 | 預設 |
|---|---|---|
| `--conf` | 設定檔後綴（cs / pos / store） | 無（讀預設檔）|
| `--port` | 監聽 port | 6000 |
| `--mode` | `normal`=正常回覆，`shadow`=只記錄不回覆 | normal |

---

## 訊息處理流程

```
LINE 訊息進來
  → 確認是否在回覆舊的待確認指令
  → 判斷是否查看排程清單
  → 萃取執行時間（run_at）
  → 意圖分類（classify_tree）
  → 欄位萃取（gather_fields）
      └─ 有 match_pool → 中文拼音 + 字串相似度比對候選名單
  → 組 API URL（build_command）
  → 依 human_check 決定：直接執行 / 等待確認 / auto
  → 執行 GET，回覆 LINE
```

---

## 設定檔說明

### `prompts*.ini`

每個 extractor 是一段嚴格限制輸出格式的 prompt，回傳值通常是：
- enum token（如 `queryKB`、`escalateSupport`）
- 單一數字
- ISO 8601 日期
- `true` / `false` / `null`

### `mission*.json`

```json
{
  "classify_tree": {
    "prompt_key": "identify_needs",
    "branch": [
      { "match": "queryKB", "task_id": "query_knowledge_base" },
      { "match": "null",    "task_id": "fallback_greeting" }
    ]
  },
  "tasks": {
    "query_knowledge_base": {
      "human_check": "false",
      "fields": {
        "sections":     { "prompt_key": "get_cs_sections", "required": true },
        "user_message": { "source": "raw", "required": true }
      },
      "action": {
        "method": "GET",
        "url_template": "https://example.com/api?sections={sections}&user_message={user_message}"
      }
    }
  }
}
```

`human_check` 三種值：
- `false` — 直接執行
- `true` — 等待使用者明確確認
- `auto` — 等待確認，但若回覆模糊則自動執行

欄位 `source: "raw"` — 直接帶入使用者原始訊息，不跑 LLM。

---

## 影子模式（shadow mode）

`--mode shadow` 啟動後，訊息不會回覆給使用者，只記錄到 `database/shadow.db`：

```
users 表            → user_id ↔ display_name
messages_{user_id}  → 訊息、意圖分類、AI 草稿
```

設計用途：讓 AI 在背後產生回覆草稿，客服人員審核後再自行發送，不讓 AI 直接對外回覆。

---

## 多專案部署

GTB 框架不動，每個專案自帶設定：

```
general-task-bot/            ← 框架（此 repo）
├── gtb.py
├── main.py
├── todo_list.py
├── prompts_system.ini
└── .env

your-project/                ← 各自的專案資料夾
├── config/
│   ├── prompts_cs.ini
│   └── mission_cs.json
├── database/
│   ├── todo_list.db
│   └── shadow.db
└── oa_registry.json
```

---

## 必要的本地設定

這個 repo clone 下來無法直接啟動，還需要：

| 檔案 | 說明 |
|---|---|
| `.env` | `GROQ_API_KEY`、`HF_TOKEN`、`OPENROUTER_API_KEY` |
| `oa_registry.json` | 各 LINE OA 的 channel_secret / channel_access_token |
| SQL Server 連線 | `generate_customerlist_simple.py` 需要 |

`oa_registry.json` 已被 `.gitignore` 排除。

---

## 文件

- [docs/gtb架構升級說明.md](docs/gtb架構升級說明.md) — main.py → gtb.py 差異說明
- [docs/mission_json_guide.md](docs/mission_json_guide.md) — mission.json 寫法
- [docs/prompts_ini_guide.md](docs/prompts_ini_guide.md) — prompts.ini 寫法
- [TODO.md](TODO.md) — 欄位澄清機制開發計畫
