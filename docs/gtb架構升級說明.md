# GTB 架構升級說明：main.py → gtb.py

---

## 升級動機

`main.py` 是原始框架，設定檔路徑寫死在同一個資料夾，無法支援多專案分離部署。
`gtb.py` 是新版啟動器，從執行目錄讀取設定，框架本身不動，各專案自帶設定。

---

## 核心差異

| 項目 | main.py | gtb.py |
|---|---|---|
| 啟動方式 | `python main.py cs 6001` | `python gtb.py --conf cs --port 6001 --mode shadow` |
| 參數解析 | `sys.argv` 位置參數 | `argparse` 具名參數 |
| 設定檔位置 | `{GTB_DIR}/prompts_cs.ini` | `{cwd}/config/prompts_cs.ini` |
| 任務定義位置 | `{GTB_DIR}/mission_cs.json` | `{cwd}/config/mission_cs.json` |
| 資料庫位置 | `{GTB_DIR}/todo_list.db` | `{cwd}/database/todo_list.db` |
| OA 憑證位置 | `{GTB_DIR}/oa_registry.json` | `{cwd}/oa_registry.json` |
| 客戶清單位置 | `{GTB_DIR}/customerlist.txt` | `{cwd}/customerlist.txt` |
| 更新客戶清單 | 強制每次執行 | 有找到腳本才執行 |
| 執行模式 | 只有正常模式 | normal / shadow |
| .env 載入 | 未載入 | 從 `{GTB_DIR}/.env` 自動載入 |

---

## 啟動參數

```
python gtb.py [--conf <後綴>] [--port <port>] [--mode <模式>]

--conf    設定檔後綴（cs / pos / store，省略則讀 prompts.ini / mission.json）
--port    監聽 port（預設 6000）
--mode    執行模式（normal=正常回覆，shadow=只記錄不回覆，預設 normal）
```

範例：
```powershell
# 正常模式
python $GTB --conf pos --port 6001

# 影子模式（記錄草稿，不回覆 LINE）
python $GTB --conf cs --port 6003 --mode shadow
```

---

## 多專案部署方式

每個專案自己的資料夾放設定與資料，GTB 框架不動：

```
general-task-bot/            ← 框架（不動）
├── gtb.py                   ← 啟動器
├── main.py                  ← 原版保留，舊 bot 繼續用
├── todo_list.py             ← 框架模組
├── prompts_system.ini       ← 系統共用 prompt
└── .env                     ← API 金鑰

cwsoft-ai-customer-service/  ← 客服專案
├── config/
│   ├── prompts_cs.ini
│   └── mission_cs.json
├── database/
│   ├── todo_list.db
│   └── shadow.db
└── oa_registry.json
```

---

## 影子模式（shadow mode）

`--mode shadow` 啟動後，訊息進來的處理流程：

```
收到 LINE 訊息
  → 更新 users 表（user_id ↔ display_name）
  → 意圖分類
  → 欄位萃取
  → 呼叫後端 API 取得 ai_draft
  → 寫入 shadow.db（messages_{user_id} 表）
  → 不回覆 LINE
```

正常模式照舊，完整的確認流程、排程、todo 都保留。

---

## gather_fields 新增 source=raw

`mission.json` 的欄位若設定 `"source": "raw"`，直接帶入使用者原始訊息，不跑 LLM：

```json
"user_message": {
  "source": "raw",
  "required": true
}
```

用途：將原始問題透過 API 傳給知識庫後端，讓後端搭配 .md 內容生成回覆。

---

## 保留 main.py 的原因

舊有的幾個 bot（pos / store / 記帳）仍用 `main.py` 運作，不受影響。
新專案一律用 `gtb.py`。兩者可以同時在不同 port 執行。
