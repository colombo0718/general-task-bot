# gtb.py（正式版）vs gtb_dev.py（開發版）盤點

> 日期：2026-05-12
> 用途：釐清開發版累積了哪些功能、跟正式版的差異在哪，未來決定哪些值得回流。
> 背景：開發版一段時間內塞太多東西，正式版最近又有一些小修補（5/11 補 prompt_template + 修中文亂碼），趁這個時間點對齊兩邊狀態。

---

## 規模

- gtb.py：745 行（正式版，prod 阿全在跑、cs-shadow service 跑 dev 版）
- gtb_dev.py：837 行（+92 行）
- `git diff --no-index`：**276 insertions / 184 deletions** — 實質差異遠大於行數差

---

## 一、開發版多塞的東西（gtb.py 沒有）

### A. 信心值機制（confidence-aware extraction）

| 元件 | 位置 | 作用 |
|---|---|---|
| `_pick_format_prompt(with_confidence)` | gtb_dev.py:230 | 依旗標選 LLM 輸出格式 prompt |
| `_parse_value_confidence(raw)` | gtb_dev.py:235 | 解析 `value \| confidence` 格式回傳 |
| `run_extractor(..., with_confidence=False)` | gtb_dev.py:259 | 多一個參數，可選擇要不要拿信心值 |
| `CONFIDENCE_THRESHOLD = 70` | gtb_dev.py:350 | 信心門檻常數 |
| `gather_fields` 多回 `field_confidence` | gtb_dev.py:352 | 每欄位的 LLM 信心值 |

### B. 客戶名澄清流程（top-N 比對 + Quick Reply）

| 元件 | 位置 | 作用 |
|---|---|---|
| `extract_top_matching_names(...)` | gtb_dev.py:325 | 除 best 還回 top 3 候選 |
| `QuickReply` / `QuickReplyItem` / `MessageAction` import | gtb_dev.py | LINE Quick Reply 三個 SDK 元件 |
| `reply_quick_reply(...)` | gtb_dev.py:213 | 發 Quick Reply 的工具函式 |
| 澄清分支 in callback | gtb_dev.py:629+ | 信心不足發 Quick Reply 給使用者選 1/2/3 |
| `gather_fields` 回 `clarification_needed` | gtb_dev.py:352 | 標記哪個欄位需澄清 |

### C. 對話歷史 + 決議狀態追蹤（db_helper）

| 元件 | 位置 | 作用 |
|---|---|---|
| `import db_helper` | gtb_dev.py | 抽出對話 / 決議的資料層 |
| `CONFIG_DB_PATH` | gtb_dev.py:110 | 額外的 config.db |
| `conv_<oaid>_<user_id>.db` | runtime 每組 OA×user 一個 | 對話歷史儲存 |
| `log_message` / `save_decision` / `save_clarification` / `save_correction` 等 | db_helper.py | 取代記憶體 `todo_command` dict |
| `get_pending_decision` / `get_pending_clarification` | db_helper.py | 跨訊息恢復狀態 |

### D. LLM 抽象層

| 元件 | 位置 | 作用 |
|---|---|---|
| `from llm_clients import call_provider` | gtb_dev.py | 統一 LLM 介面 |
| `llm_complete` | gtb_dev.py:200 | 3 行 delegate（正式版 23 行 inline 切換）|
| `llm_clients.py` | 528 行獨立模組 | 抽掉 hf/openrouter/groq/remote_worker/codex_cli/gemini_cli 邏輯 |

---

## 二、正式版有、開發版沒有的（被丟掉的）

| 元件 | 位置（gtb.py）| 備註 |
|---|---|---|
| `FlexMessage` import | line 29 | gtb_dev.py 移除（似乎沒在用）|
| `huggingface_hub.InferenceClient` import | line 35 | hf provider inline 用 |
| `generate_huggingface_reply` | line 201 | inline provider 函式 |
| `generate_openrouter_reply` | line 217 | inline provider 函式 |
| `generate_groq_reply` | line 234 | inline provider 函式 |
| `generate_remote_worker_reply` | line 249 | inline provider 函式 |
| **`llm_complete` 的 fallback to hf** | line 285-290 | 主 provider 失敗時 fallback 到 hf 模型；gtb_dev.py 失敗直接回 `"Error"` |
| `WERKZEUG_RUN_MAIN` 環境變數判斷 | line 736 | 避免 debug reloader 雙啟 todo_worker；gtb_dev.py 拿掉了 |

---

## 三、兩邊都有但實作差很多

| 函式 | gtb.py | gtb_dev.py |
|---|---|---|
| `run_extractor` | 組 3 個 parts：ref / template / 訊息（無 confidence）| 組 4 個 parts：fmt / ref / template / 訊息（有 confidence）|
| `gather_fields` | 回 `(values, missing, task_cfg)` | 回 `(values, missing, task_cfg, field_confidence, clarification_needed)` |
| `llm_complete` | 23 行 inline provider 切換 + fallback to hf | 3 行 delegate to `call_provider` |
| 一般模式 callback | 記憶體 `todo_command` dict 管 pending 狀態 | DB 管 session / pending / decision / clarification |
| 一般模式流程 | 線性：分類 → gather → 執行 | 分支：先看 pending_clarification → pending_decision → 新 session |

---

## 四、影子模式（兩邊都有）

兩邊**都有** `--mode shadow`，邏輯骨架一致：

| 流程步驟 | gtb.py | gtb_dev.py |
|---|---|---|
| `upsert_user` | ✓ | ✓ |
| 分類意圖（`run_extractor` 跑 `classify_tree`）| ✓ | ✓ + 可選信心值 |
| 走 `branches` 決定 task_id（含 null fallback）| ✓ | ✓ |
| `gather_fields` 抽欄位 | 回 3 個值 | 回 5 個值 |
| 缺欄位 → 跳過 AI 草稿 | ✓ | ✓ |
| `build_command` + `execute_command` → `ai_draft` | ✓ | ✓ |
| `save_shadow_message` 寫 `shadow.db` | ✓ | ✓ |
| `continue`（不回 LINE）| ✓ | ✓ |

**關鍵：**

1. 兩邊影子模式本質相同 — 跟一般模式走一樣流程，但結果寫 `shadow.db` 不回 LINE
2. gtb_dev.py 多的只是「信心值可選」— 需 `mission_*.json` 設 `with_confidence: true` 才會啟動
3. aquan-manager 哪天想試影子模式跑 prod，直接改 services.json 加 `--mode shadow` 就能跑

**小髒亂**：gtb_dev.py 影子模式段 `# 分類意圖` 註解重複出現兩次（複製貼上沒清乾淨）。

---

## 五、最近的小修補（5/11）

只動了**正式版**（gtb.py）：

| Commit | 內容 |
|---|---|
| `b3609cb` | 修中文亂碼（14 行 fallback / 註解，cp950→UTF-8 誤讀產生的「雿輻撓伐」型）+ 移 BOM + docstring 改 raw string |

**開發版**（gtb_dev.py）動了 1 行：

| Commit | 內容 |
|---|---|
| `e48fe8e` | `run_extractor` 補回漏掉的 `parts.append(prompt_template.strip())` — 害 cs-shadow 影子模式 LLM 一直回 null、所有 ai_draft 空白 |

**漂移檢查**：兩邊 `run_extractor` 內邏輯現在已對齊（除了 gtb_dev.py 多的 `with_confidence` 的 fmt_prompt 那段）。OK。

---

## 六、要不要回流 → 正式版

開發版塞了不少功能，但不全部都該回流。Risk vs value：

| 功能 | 回流價值 | 風險 | 建議 |
|---|---|---|---|
| **`llm_clients.call_provider` 抽象** | ★★★ 結構乾淨、4 個 generate_* 變 1 行、`huggingface_hub` 依賴可拿掉 | 低（純重構，行為不變）| **優先回流** |
| **信心值機制** | ★★ 抓 LLM 不確定，shadow 模式特別有用 | 中（要改 prompt 格式、callback 多一條分支）| 視 cs-shadow 觀察需求決定 |
| **客戶名澄清流程（Quick Reply）** | ★★ 提升 UX，但 prod 阿全很少需要澄清 | 中（要改 callback 流程）| 阿全用不太到，先擱 |
| **db_helper 對話歷史** | ★ 之後做客服分析有用 | 高（每 OA×user 一個 .db，磁碟用量增加；callback 流程大改）| 跟客服分析需求一起評估 |

**個人建議優先序：**

1. **先回流 `llm_clients` 抽象** — 純結構優化、最低風險、把 4 個 inline `generate_*_reply` + `huggingface_hub` 依賴整個拿掉
2. 順手清掉 gtb_dev.py 影子模式段「# 分類意圖」重複註解
3. 其他三項（信心值 / 澄清 / db_helper）等實際 prod 需求出來再評估，避免「為了對齊而對齊」

---

## 附：相關檔案 reference

- 本檔：`docs/gtb_正式版_開發版_盤點.md`
- 引擎本體：`gtb.py` / `gtb_dev.py`
- LLM 抽象層（只 dev 用）：`llm_clients.py`
- 對話歷史 / 決議資料層（只 dev 用）：`db_helper.py`
- 相關設計文件：
  - `docs/gtb_database_architecture.md`
  - `docs/gtb_dev_flow.md`
  - `docs/gtb_信心值升級與benchmark規劃.md`
  - `docs/shadow_mode_三個_bug_修正計畫.md`
