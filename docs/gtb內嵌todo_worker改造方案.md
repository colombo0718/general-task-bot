# GTB 內嵌 Todo Worker 改造方案

## 目的

目前 `gtb.py` 已經改成多專案分離架構：

- 每個 chatbot 專案各自擁有 `config/`
- 每個 chatbot 專案各自擁有 `database/`
- `gtb.py` 從目前執行目錄讀取專案設定與資料

但排程延遲執行仍依賴獨立的 `todo_worker.py`。  
這會造成一個落差：

- `gtb.py` 已經是 project-aware
- `todo_worker.py` 仍偏向單一全域 worker 思維

本次改造的目標是：

**讓 `gtb.py` 啟動後，自動附帶啟動該專案自己的 todo worker。**

也就是說，之後每個專案只需要啟動一次 `gtb.py`，就同時具備：

- LINE webhook 即時回應
- 延遲任務排程巡檢
- 到時自動執行待辦指令


## 改造原則

1. `todo worker` 不再要求使用者額外手動啟動獨立腳本
2. worker 必須跟著目前專案的 `PROJECT_DIR` 走
3. worker 必須使用目前專案的 `database/todo_list.db`
4. 先做最小可用改動，不先大改任務 schema
5. 保留 `todo_worker.py` 作為過渡檔案，先不刪除


## 目標行為

啟動：

```powershell
cd C:\Users\pos\Desktop\cwsoft-aquan-manager
python C:\Users\pos\Desktop\general-task-bot\gtb.py --conf pos --port 6001
```

預期效果：

1. `gtb.py` 啟動 Flask webhook
2. `gtb.py` 自動啟動背景 todo worker
3. worker 定時巡查 `database/todo_list.db`
4. 如果發現 `pending` 且時間已到的任務，就自動執行
5. 執行成功後把 `state` 更新為 `done`


## 實作方式

### 1. 在 `gtb.py` 內新增背景 worker loop

新增一組內部函式，建議形式：

- `now_iso()`
- `execute_scheduled_todo(...)`
- `run_todo_worker_loop(db_path, interval_seconds=60)`
- `start_todo_worker(db_path, interval_seconds=60)`

用途：

- 定時檢查 `todo_items`
- 找出 `state='pending'` 且 `run_at <= now`
- 逐筆執行
- 成功後更新 `state='done'`


### 2. worker 使用目前專案的 DB_PATH

直接沿用 `gtb.py` 既有的：

```python
DB_PATH = os.path.join(PROJECT_DIR, "database", "todo_list.db")
```

這樣每個 chatbot 專案都會自動巡自己的排程資料，不會互相干擾。


### 3. 在 `__main__` 中自動啟動 worker thread

在 Flask 啟動前，先呼叫：

```python
start_todo_worker(DB_PATH)
```

worker 使用 daemon thread 背景執行，不阻塞 Flask 主流程。


### 4. 避免 debug/reloader 重複啟動 worker

這是本次改造最重要的細節。

如果 Flask reloader 啟動兩次程序，worker 可能會重複巡同一份 DB，造成任務被重複執行。

因此需要加入保護，只在真正主進程啟 worker。  
可用做法擇一：

- 檢查 `WERKZEUG_RUN_MAIN`
- 關閉 reloader
- 或加上更明確的啟動守門條件

本次建議：

- 保留目前開發模式
- 但只在主進程中啟動 worker


## 本次先不改的部分

以下先不動，避免一次改太大：

1. 不先移除 `todo_worker.py`
2. 不先重構整個 `callback()` 主流程
3. 不先改 `todo_items` schema
4. 不先補完整 `method/body/confirm_method` 排程結構


## 已知限制

目前 `todo_list.db` 結構只有：

- `id`
- `user_id`
- `run_at`
- `url`
- `user_text`
- `state`

所以目前內嵌 worker 仍只能基於既有欄位工作。  
也就是說，現在仍維持既有的排程行為，不在這一版擴充成更完整的任務描述格式。


## 後續可再補強

下一階段可以考慮補這些：

1. 在 `todo_items` 新增 `method`
2. 在 `todo_items` 新增 `created_at`
3. 在 `todo_items` 新增 `last_error`
4. 將巡檢間隔改為可配置
5. 將排程執行日誌格式化，便於多專案追蹤


## 本次施工結論

本次改造的重點不是做新功能，而是把既有排程能力正式納入 `gtb.py` 的專案生命週期。

改完後的核心原則是：

**每個 chatbot 專案只要啟動一次 `gtb.py`，就自帶自己的 todo worker。**
