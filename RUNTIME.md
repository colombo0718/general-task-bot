# RUNTIME.md — general-task-bot

> **這份是什麼**
> 給 [`cwsoft-super-manager`](../cwsoft-super-manager/PROJECT.md) 端 Claude / 維護者看的「執行契約」 — 程式作者主張「我希望被這樣跑」。
>
> super-manager 的 `services.json` 應與本檔同步。**兩邊不一致時以本檔為準**（程式作者最懂自己）。
>
> 本 repo 主要是 chatbot 引擎（`gtb.py` / `gtb_dev.py`），被各 chatbot repo（aquan-manager / cs-shadow / aquan-manager-dev）以「散裝模式」import 使用 — 引擎本身不該被 super-manager 列管。
>
> **但這個 repo 內有一支獨立執行的 service：`todo_worker.py`** — 本檔規範它的執行契約。

---

## 服務角色

**todo-worker** — 阿全 codex 排程任務的觸發 worker。

每 30 秒 poll `database/todo_list.db`：
1. `SELECT FROM todo_items WHERE state='pending' AND run_at <= NOW()`
2. 對命中的 row：`importlib.import_module(tool_module)` + `getattr(tool_func)` + `func(**args)`
3. 將執行結果 + 時間戳記寫進 `cwsoft-aquan-manager/notebooks/notebook_YYYYMMDD.md`（audit）
4. UPDATE row state（done / failed）

是阿全「**到某個時間自動執行某個 MCP 工具**」這條 path 的觸發層。**worker 死掉 = 排程任務全部停**。

> 對應 worklog：[2026-05-25-aquan-prod-cutover-day5-mcp-todo-worker-and-schedulable-decorator.md](../cwsoft-project-tracker/worklogs/2026-05-25-aquan-prod-cutover-day5-mcp-todo-worker-and-schedulable-decorator.md)
> 對應設計文件：[docs/mcp_todo_worker_開發計畫.md](docs/mcp_todo_worker_開發計畫.md)

---

## 執行規格

| 欄位 | 值 |
|---|---|
| service name | `todo-worker` |
| **deployment_mode** | `resident` |
| cmd | `["py", "-3", "C:/Users/pos/Desktop/general-task-bot/todo_worker.py"]` |
| cwd | `C:/Users/pos/Desktop/general-task-bot` |
| port | `null`（背景 poller、不開 HTTP） |
| health_type | **`process`**（沒 HTTP endpoint，靠 `proc.poll() is None` 判斷活著）|
| health_path | 無 |
| startup_order | `5`（晚於 aquan-manager；worker 不會 race，靠 todo_list.db 取資料） |
| auto_restart | `true`（worker 死掉排程就停、必須拉回來） |

### 為什麼 health 用 `process`
todo_worker.py **沒 Flask、沒 HTTP endpoint**，是純 background poller。super-manager 的 `manager.py:check_health` 已支援 `health_type: "process"`（[`manager.py:73`](../cwsoft-super-manager/manager.py#L73)）— 用 `proc.poll() is None` 判斷子 process 還活著就算健康。

> 進階思路（沒採用）：health 查 `todo_list.db` 的 `executed_at` 最近時間。但問題是：worker 沒任務時不 fire、`executed_at` 永遠舊，無法區分「沒任務」vs「worker 死了」。所以 process check 才是最樸實可靠的。

---

## 依賴

### Python 環境
- `C:\Python313\python.exe`（跟 aquan-manager / cwsoft-sqlserver-mcp 同一支）
- import 環境要含：
  - `pyodbc`（透過 cwsoft-sqlserver-mcp 跟 SQL Server 交互）
  - `mcp`（function call 對應 MCP 工具）
  - 標準庫：`sqlite3` / `importlib` / `pathlib`

### 檔案路徑 hardcode
worker 內 `sys.path.insert(0, r'C:\Users\pos\cwsoft-sqlserver-mcp\src')`：
- 該路徑必須存在
- 該目錄含的 MCP module（adjust_points 等寫入類 + 讀取類）必須 import 得起來
- 跑同帳號（pos）有讀取權限

### 資料夾寫入權限
- `C:/Users/pos/Desktop/cwsoft-aquan-manager/database/`（讀寫 `todo_list.db`，跟 aquan-manager 共用同檔）
- `C:/Users/pos/Desktop/cwsoft-aquan-manager/notebooks/`（寫 `notebook_YYYYMMDD.md` audit log）

### 跟 aquan-manager 的關係
- 寫入端：aquan-manager 收到 LINE 訊息 → 解析 → 寫進 `todo_list.db`（state='pending'）
- 觸發端：**todo-worker**（本服務）→ 每 30 秒掃 pending due → 執行
- 兩者**透過 SQLite file 解耦**，沒 IPC 同步問題

---

## Log

stdout / stderr 走 super-manager 標準機制 → `cwsoft-super-manager/logs/todo-worker.log`。

> 注意：worklog 提到的「`C:/Users/pos/Desktop/general-task-bot/logs/todo_worker.log`」是 **manual 跑時** 用的路徑；super-manager 接管後 log **改走 super-manager logs 目錄**、跟其他 11 個服務一致。

---

## 已知 quirk / 觀察點

### 跟 aquan-manager 的 startup race
- aquan-manager 啟動才會初始化 `todo_list.db` schema（`init_todo_db()`）
- todo-worker 啟動時也 `init_todo_db()`（idempotent CREATE IF NOT EXISTS）
- 兩者誰先誰後都安全，但 `startup_order: 5` 排在 aquan-manager (`order=4`) 後較自然

### MCP module import 失敗時的退化
若 `cwsoft-sqlserver-mcp/src` 不存在或某 tool_module import 失敗：
- 該筆 row state 標 `failed`、`error` 欄寫進 traceback
- worker 自己不會死掉，繼續 poll 下一輪
- 但**整 batch 任務全部 fail** — 該排查 import path / Python env

### 30 秒 poll interval
worklog 提到「老闆需求 5/31 23:59:59 觸發、1 小時粒度不夠」改 30 秒。如果未來要再加密（5 秒 / 10 秒），改 worker 內 `CHECK_INTERVAL` 即可，不必動 super-manager。

---

## 變更紀錄

- 2026-05-25 day 5 上線（manual `Start-Process Hidden` 跑著、PID 142100）
- 2026-05-26 收進 super-manager 列管（services.json + 本檔）
