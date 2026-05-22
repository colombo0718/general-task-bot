# CHANGELOG

重大里程碑與架構決策紀錄。小 bug fix 與文字調整見 git log。

---

## 2026-04-13｜GTB 通用化清理、信心值開發線與 Benchmark 啟動

### 變更內容
- 清理 `gtb.py` 中阿全（pos）專案特化殘留，改為以 mission/config 為主的通用讀取流程
- `match_pool` 由 mission 欄位接管，不再寫死特定檔名字串
- 新增 `gtb_dev.py` 作為開發版，避免直接衝擊正式 `gtb.py`
- `prompts_system.ini` 新增兩種輸出格式模板：
  - `format_value_only`
  - `format_with_confidence`
- 拆出 LLM 呼叫層 `llm_clients.py`，並新增 `scripts/llm_benchmark.py`
- `llm_benchmark.py` 支援多 provider/model 批次測試與平行化執行

### 架構決策
- 正式線維持穩定：先用 `gtb_dev.py` 驗證信心值輸出與相容性，再評估回併主線
- 信心值採「按欄位宣告」方向（`with_confidence=true/false`）以保留向下相容
- Benchmark 分兩層推進：
  1. 模型層（速度、格式正確率）
  2. GTB 任務層（意圖 + fields + 任務參數整體正確性）

### 相關文件
- [docs/gtb_信心值升級與benchmark規劃.md](docs/gtb_信心值升級與benchmark規劃.md)
- [docs/gtb_專案使用盤點.md](docs/gtb_專案使用盤點.md)
- [docs/gtb內嵌todo_worker改造方案.md](docs/gtb內嵌todo_worker改造方案.md)

---

## 2026-04-06｜gtb.py 上線，框架與專案分離

### 變更內容
- 新增 `gtb.py`：從執行當下目錄讀取 `config/`、`database/`、`oa_registry.json`
- 啟動參數從位置參數（`sys.argv`）改為具名參數（`argparse --conf / --port / --mode`）
- 新增 `--mode shadow`：訊息只記錄不回覆，AI 草稿存入 `database/shadow.db`
- `gather_fields` 新增 `source: "raw"` 欄位類型，直接帶入原始使用者訊息不跑 LLM
- `todo_list.py` 的所有函式加入 `db_path` 參數，支援多專案各自的資料庫路徑

### 架構決策
- `main.py` 保留不動，現有 bot（pos / store / 記帳）繼續使用
- 新專案一律使用 `gtb.py`，框架本身不含任何專案設定
- `.env` 統一放在 `general-task-bot/`，`gtb.py` 啟動時自動載入

### 相關文件
- [docs/gtb架構升級說明.md](docs/gtb架構升級說明.md)

---

## 2026-04-06｜全葳 AI 客服專案初始化（cwsoft-ai-customer-service）

### 變更內容
- `prompts_cs.ini` 與 `mission_cs.json` 移至獨立專案資料夾 `config/`
- 建立 `kbcs_server.py`（port 6004）：讀取 43 節 .md 知識庫 + Groq LLM 生成草稿
- 建立 `admin_server.py`（port 6005）：客服草稿檢視介面後端
- 建立 `admin/`：純 HTML/JS/CSS 前端，10 秒輪詢刷新，顯示 AI 草稿

### 架構決策
- 影子模式：AI 產生草稿 → 存 DB，客服在原本 LINE OA 後台回覆，不讓 AI 直接對外
- 每位使用者獨立一張訊息表（`messages_{user_id}`），方便後續按人查詢
- `kbcs/answer` 設計為標準 HTTP API，與 GTB 任務架構一致（action → url_template）

---

## 2026-04｜mission_cs.json 建立，cs 模式上線

### 變更內容
- 新增 `prompts_cs.ini`：意圖分類（queryKB / escalateSupport）、節次選擇（43 節目錄）
- 新增 `mission_cs.json`：三個任務（query_knowledge_base / escalate_support / fallback_greeting）
- 知識庫 43 節 .md 全部完成，通過 0-3 品質檢核五關

---

## 早期｜GTB 框架建立

### 核心設計確立
- 設定驅動架構：`prompts*.ini` + `mission*.json` 定義所有業務邏輯
- 三種 `human_check` 模式：false / true / auto
- 客戶名稱模糊比對：中文拼音相似度 + 英文字串相似度，門檻 0.8
- 排程延後執行：`extract_run_at` + SQLite `todo_list.db` + `todo_worker.py`
- 多 OA 支援：`oa_registry.json` 對應 `/callback/<oaid>` 路由
