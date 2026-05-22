# gtb_dev.py 後門動工清單

> 日期：2026-05-12
> 目的：在開發版 gtb_dev.py 開後門，做兩種壓測 + 老闆歷史考題
> 協作：另一個 Claude session 同時在處理 **阿全測試版 config 檔**（cwsoft-aquan-manager 那邊），本檔聚焦 **gtb_dev.py + super-manager + 考題 runner** 這條線。

---

## 一、背景與目標

### 為什麼要開後門
- gtb_dev.py 目前唯一入口是 `/callback/<oaid>`，必須有真的 LINE OA webhook 才能觸發 → 沒辦法做壓測、也沒辦法跑批次考題
- 4/14 會議 §9「GTB 測試入口方向：脫離 LINE OA」已列為待跟進，但一直沒落地
- 5/12 推進原因：要讓阿全跑「正式版（接 LINE）+ 測試版（不接 LINE，純壓測）」雙版，測試版要能過老闆歷史訊息考題才能上線

### 設計原則：兩邊互動都是假的
- **向上對 LINE OA**：不驗 signature、不打任何 LINE API（含 reply / push / loading indicator）
- **向下對 downstream API**：不真打 sqlgate / autoQuotes 等服務，只記錄會打哪支 API（cmd_url）
- 後門只測：chatbot 跟人訊息互動的應答對不對 — 問對問題、組出正確 API 動作

### 雙版部署目標
| 用途 | 引擎 | 端口 | 接 LINE | 設定檔 | DB 目錄 |
|---|---|---|---|---|---|
| 阿全正式版（老闆日常用）| `gtb.py` | 6000 | 是（LINE OA `@708juxdz`）| `mission_pos.json` | `database/` |
| 阿全測試版（壓測 / 考題）| `gtb_dev.py` | 6010 | 否（後門入）| `mission_pos_dev.json` | `database_dev/` |
| cs-shadow（影子模式）| `gtb.py`（5/12 切回正式版）| 6003 | 從 prod cs OA mirror | `mission_cs.json` | `database/` |

---

## 二、已完成（在 gtb_dev.py master，**尚未 commit**）

### A. argparse + DB 路徑分流
- [x] `gtb_dev.py` 新增 `--db-dir` arg（default `database`，向後相容）
- [x] `DB_PATH` / `CONFIG_DB_PATH` / `SHADOW_DB_PATH` 全改用 `DB_DIR = PROJECT_DIR/<args.db_dir>`
- [x] boot log 多印 `DB_DIR`

### B. db_helper 簽名調整
- [x] `db_helper.get_or_create_conv_db` 第 5 參數 `project_dir` → `db_dir`，body 改 `os.path.join(db_dir, db_name)`（不再寫死 `database/` 子目錄）
- [x] `gtb_dev.py` 唯一 call site 同步改傳 `DB_DIR`

### C. /sim/<oaid> 後門 route
- [x] `from flask import` 加 `jsonify`
- [x] 新 route `POST /sim/<oaid>`，跳過 signature、不打 LINE、不真 call downstream
- [x] 回 JSON：`intent` / `task_id` / `values` / `missing` / `field_confidence` / `cmd_url` / `timing_ms` / `errors`

### D. 對 cs-shadow 影響評估
- [x] 確認 default `--db-dir database` 跟原行為等價
- [x] cs-shadow 5/12 已切回 `gtb.py`，gtb_dev.py 改動不影響它

### E. AST syntax 驗證
- [x] `py -3 -c "import ast; ast.parse(...)"` 對 gtb_dev.py / db_helper.py 都過

---

## 三、TODO（按依賴排序）

### Step 1 — 後門手動 smoke test ⭐ 第一步

跟阿全測試版部署無關，純驗 `/sim` 邏輯。

- [ ] 在某台 dev 機（或公司主機暫借一個 port）起 `gtb_dev.py --conf cs --port 6099 --db-dir database_dev`
  - 用 cs 設定當測試載體（簡單、無 match_pool）
  - `--db-dir database_dev` 確保不污染 prod cs 的 `database/`
- [ ] curl 戳：
  ```
  curl -X POST http://localhost:6099/sim/test_oa \
    -H "Content-Type: application/json" \
    -d '{"user_id":"Utest001","display_name":"測試客戶","user_text":"標籤機去哪設定"}'
  ```
- [ ] 驗證回 JSON：
  - `ok: true`
  - `intent: "queryKB"`
  - `task_id: "query_knowledge_base"`
  - `values.sections` 有抽到對應節次
  - `cmd_url` 含 `cwsoft.leaflune.org/kbcs/answer?...`
  - `timing_ms` 是合理數字
- [ ] 測 missing 場景（給不完整訊息）→ 確認 `missing` 有列出缺哪些欄位、`cmd_url=null`
- [ ] 測無效訊息（空字串）→ 確認回 400 + `errors`

### Step 2 — 阿全測試版 services.json entry

需等另一個 Claude 把 `mission_pos_dev.json` / `prompts_pos_dev.ini` 在 aquan repo 開好才做。

- [ ] `cwsoft-super-manager/services.json` 加 `aquan-manager-dev` entry：
  ```json
  {
    "name": "aquan-manager-dev (6010)",
    "cmd": ["py", "-3", "C:/Users/pos/Desktop/general-task-bot/gtb_dev.py",
            "--conf", "pos_dev", "--port", "6010",
            "--db-dir", "database_dev"],
    "cwd": "C:/Users/pos/Desktop/cwsoft-aquan-manager",
    "port": 6010,
    "health_path": "/ping",
    "health_type": "http",
    "startup_order": 4,
    "auto_restart": true
  }
  ```
- [ ] `cwsoft-aquan-manager/database_dev/` 目錄建立（gtb_dev.py 啟動會自動 makedirs，但先確認 .gitignore 排除）
- [ ] aquan-manager 的 `RUNTIME.md` 補測試版執行契約
- [ ] super-manager `PROJECT.md` 加 `aquan-manager-dev` 服務 card
- [ ] `nssm restart cwsoft-super-manager` 讓 services.json 重載
- [ ] dashboard 看 `aquan-manager-dev` 起得來、健康綠

### Step 3 — 考題 runner

- [ ] 新建 `general-task-bot/scripts/exam_runner.py`：
  - 讀 JSON 題本（單檔多題）
  - 並發 POST 到 `/sim/<oaid>`（concurrent 數量參數化）
  - 收 response、比對預期、量 latency
  - 產 markdown / json report
- [ ] 題本格式（v1 簡單版）：
  ```json
  {
    "oaid": "@708juxdz_dev",
    "concurrency": 10,
    "items": [
      {
        "id": 1,
        "user_id": "Uboss",
        "user_text": "幫零壹加100點",
        "expected_intent": "addPoints",
        "expected_task_id": "add_points",
        "expected_values": {"identify_name": "零壹通訊行", "points": "100"}
      }
    ]
  }
  ```
- [ ] 比對策略：
  - 必過：`intent` / `task_id` 一致
  - 嚴格：`values` 子集（題本只列 expected，不檢查其他欄位）
  - 寬鬆：`cmd_url` 含 expected substring（避免 query string 順序差異）
- [ ] report 欄位：通過率、平均 latency、p50/p95/p99、按 intent 分組準確率

### Step 4 — 老闆歷史訊息 → 題本

最費時、最有價值。可分子任務做。

- [ ] 從 `cwsoft-aquan-manager/boss_chat.txt` 抽出老闆訊息（6 個月）
- [ ] 半人工標 expected_intent / expected_task_id / expected_values
  - 一次標 100 條左右先做 alpha 題本
  - 標完後評估「人工標」要不要工具化（gemini --yolo 幫初篩）
- [ ] 存 `cwsoft-aquan-manager/benchmark/boss_exam_v1.json`
- [ ] 跑題：`py exam_runner.py --exam benchmark/boss_exam_v1.json --target http://localhost:6010/sim/@708juxdz_dev`
- [ ] 看 fail case → 看是 prompt 問題、mission 問題、還是 LLM 不穩

### Step 5 — Phase 2 + 3 commit + push

- [ ] gtb_dev.py + db_helper.py：commit「feat(gtb_dev): 加 /sim 後門 + --db-dir DB 路徑分流」
- [ ] super-manager services.json + PROJECT.md：commit「feat: 加 aquan-manager-dev 測試版服務（後門入）」
- [ ] aquan-manager（另一個 Claude 動）：commit dev config + RUNTIME.md
- [ ] 三個 repo 都 push（要 RDP 過去，SSH 卡 wincredman）

---

## 四、驗證紀錄

| 動作 | 預期 | 實際 | 日期 |
|---|---|---|---|
| AST syntax check | 2 檔都 OK | OK | 5/12 |
| cs-shadow 切回 gtb.py | running、shadow_messages 持續寫入 | 待用戶 LINE 測 | 5/12 |
| 後門 smoke test | 待做 | - | - |
| 阿全測試版啟動 | 待做 | - | - |
| 老闆考題 alpha 通過率 | TBD | - | - |

---

## 五、與其他 Claude 的協作邊界

### 對方（aquan repo Claude）負責
- `cwsoft-aquan-manager/config/mission_pos_dev.json` 從 `mission_pos.json` 複製 + 標註 dev 版本
- `cwsoft-aquan-manager/config/prompts_pos_dev.ini` 從 `prompts_pos.ini` 複製 + 標註 dev 版本
- aquan repo `.gitignore` 確認 `database_dev/` 也被排除（避免 commit conv_*.db / shadow.db）
- aquan repo `RUNTIME.md` 加上 dev 版條目（如果該 repo 有 RUNTIME.md 體系的話）

### 本檔負責（gtb-dev.py 這條線）
- gtb_dev.py / db_helper.py 後門實作（**已完成 Phase 1**）
- super-manager services.json 加 aquan-manager-dev entry
- super-manager PROJECT.md 補服務 card
- 考題 runner
- 題本準備（從 boss_chat.txt 抽）

### 交界檢查清單
動工前彼此先確認對方狀態：

- [ ] 對方有沒有把 `mission_pos_dev.json` / `prompts_pos_dev.ini` 開好？
- [ ] 對方的 dev config 有沒有開 `with_confidence: true`？
  - 若有 → 阿全測試版會走信心值路徑，跟 prod 行為不同；考題 expected 要對應調整
  - 若無 → 直接複製 prod 設定即可
- [ ] 對方的 dev config 對 `match_pool` 怎麼處理？
  - prod 阿全 mission_pos.json 用 `customerlist.txt` 當 match_pool
  - 測試版要不要共用同一個 customerlist？還是另外維護一份 dev 版？

### 可能的衝突點
- services.json 雙邊都不該動 super-manager 端（除非協調好）
- 對方動到 mission_pos.json（prod 設定）→ 影響真正在跑的阿全 → **不可**
- 都動 gtb_dev.py → 我已經改了，對方應該不需要動

---

## 六、設計小筆記

### 為什麼後門 dry-run 不真打 downstream API？
- 真打 sqlgate 會有副作用：扣點數、加分店、開客戶等都是 mutation
- 壓測 / 考題大量呼叫等於連環炸下游
- 考題重點是「組對 cmd_url」，不需要驗證 sqlgate SP 的正確性（那是 sqlgate 的測試範疇）

### 為什麼回應只記 cmd_url 不真 GET？
- 即使是 GET（read-only），打過去也會佔 sqlgate 的 connection / SQL Server 資源
- 大量並發 GET 一樣會干擾 prod 行為
- 想真的端對端測，另起一個 `--mode integration` 模式以後再加

### 為什麼 default `--db-dir database` 而不是 `database_dev`？
- 向後相容 — 既有 cs-shadow / 未來再起任何 gtb_dev.py 都不用改 services.json
- 「要分流就明確傳 `--db-dir database_dev`」這個語意比較清楚

### 為什麼 `/sim` 不走完整的 normal-mode 邏輯（pending_decision / clarification）？
- Phase 1 簡化：先驗 classification + extraction（90% 的考題只測這個）
- normal mode 的多訊息狀態流（先發 missing 補欄位、人工確認等）需要 multi-turn 設計
- 後續若考題要測 multi-turn，再擴展 `/sim` 支援 `session_state` 參數

---

## 七、相關檔案 reference

- 本檔：`docs/gtb_後門動工清單.md`
- 引擎本體：`gtb.py`（prod）/ `gtb_dev.py`（dev，已含後門）
- 對應盤點：`docs/gtb_正式版_開發版_盤點.md`
- 4/14 設計會議：[matrix-manager/meetings/2026-04-14-gtb-confidence-and-benchmark-roadmap.md §9](../../matrix-manager/meetings/2026-04-14-gtb-confidence-and-benchmark-roadmap.md)
- 老闆原始訴求：`cwsoft-aquan-manager/boss_chat.txt`
- aquan repo TODO：`cwsoft-aquan-manager/TODO.md`
