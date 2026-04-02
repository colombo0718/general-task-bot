# GTB TODO

---

## [PLAN] 欄位澄清機制（信心不足時主動詢問使用者）

### 背景
語音輸入常造成客戶名稱辨識模糊（例如「零壹通訊行」→「令一通訊行」）。
當 AI 萃取的欄位值信心不足時，應主動向使用者提出選擇題，而非直接帶錯誤值執行。

### 影響範圍
- `main.py`（核心改動，三個區塊）
- 不動：`prompts_pos.ini`、`mission_pos.json`、其他設定檔

---

### 改動一：`extract_best_matching_name` 改為回傳 top-N

**檔案**：`main.py`
**現況**：遍歷 `customerlist.txt` 計算相似度，只回傳最高分的一個名稱。
**目標**：改成 `extract_top_matching_names(keyword, candidates, top_n=3, threshold=0.8)`，
回傳 `[(name, score), ...]` 前三名。

- [ ] 新增 `extract_top_matching_names` 函式（保留舊函式或整合）
- [ ] `score[0] >= threshold` → 回傳最佳值，行為與現在相同
- [ ] `score[0] < threshold` → 回傳候選清單，供上層判斷

---

### 改動二：`gather_fields` 偵測信心不足並中斷

**檔案**：`main.py`
**現況**：`gather_fields` 呼叫 `extract_best_matching_name`，直接取單一值。
**目標**：有 `match_pool` 的欄位，若信心不足，回傳特殊標記讓 callback 知道需要詢問。

- [ ] `gather_fields` 回傳增加第四個值：`clarification_needed`
  ```python
  # 現在
  return values, missing, task_cfg
  # 改成
  return values, missing, task_cfg, clarification_needed
  # clarification_needed = None | {"field": "name", "options": ["A", "B", "C"]}
  ```
- [ ] 信心不足時：`values[field_name]` 先填入最高候選（供 URL 備用），同時設定 `clarification_needed`
- [ ] 只針對有 `match_pool` 的欄位觸發（目前只有 `identify_name`）

---

### 改動三：`todo_command` 新增 `pending_clarification` 狀態

**檔案**：`main.py`
**現況**：`todo_command[user_id]` 只有 `human_check=true/auto` 兩種等待狀態。
**目標**：新增第三種狀態，記錄「正在等使用者澄清哪個欄位」。

- [ ] `todo_command` 結構新增欄位：
  ```python
  {
      "state": "pending_clarification",  # 新增，原本為 None
      "task_id": task_id,
      "values": values,                  # 已萃取完的欄位
      "pending_field": "name",           # 哪個欄位待確認
      "options": ["A", "B", "C"],        # 選項清單
      # 原有欄位照舊
      "url": "...", "human_check": ..., "run_at": "now", "user_text": "..."
  }
  ```

---

### 改動四：`callback` 開頭新增澄清分支

**檔案**：`main.py`，`callback()` 函式，插在現有 `extracted_ronot` 判斷之前。
**目標**：使用者回答選項後，填入欄位值並繼續執行 `build_command` 以後的步驟。

- [ ] 偵測 `state == "pending_clarification"`
- [ ] 解析使用者回應：
  - 輸入 `"1"` / `"2"` / `"3"` → 對應 `options[0/1/2]`
  - 直接輸入選項文字（完全匹配）→ 直接用
  - 無法解析 → 重新發送選擇題
- [ ] 填入 `values[pending_field]`，清除 `pending_clarification` 狀態
- [ ] 繼續執行 `build_command` → `execute_command` 邏輯（不重跑分類與萃取）

---

### 改動五：澄清問題改用 LINE Quick Reply

**目標**：信心不足時，回覆改用按鈕選單，而非純文字。

- [ ] 新增 `reply_quick_reply(line_bot_api, reply_token, question, options)` 函式
- [ ] 使用 LINE `QuickReply` + `MessageAction` 按下即送出
- [ ] 格式範例：
  ```
  🤔 請問你說的是哪一間公司？
  [零壹通訊] [零一通訊行] [全壹科技]
  ```

---

### 優先順序

| 步驟 | 說明 | 相依 |
|------|------|------|
| 改動一 | top-N 相似度函式 | 無 |
| 改動二 | gather_fields 回傳 clarification_needed | 改動一 |
| 改動三 | todo_command 新狀態結構 | 無 |
| 改動四 | callback 澄清分支 | 改動二、三 |
| 改動五 | Quick Reply 回覆 | 改動四 |

### 暫不處理

- 其他欄位（數字、日期、布林值）的信心機制 — 模糊空間小，先觀察
- `identify_needs`（任務分類）的信心機制 — 之後視需求另開計畫
