# GTB TODO

## [TODO 2026-05-16] [FEATURE] gtb_codex.py 改 per-user session（同 OA 每個 LINE user_id 一條獨立 codex session）

### 緣起

目前 `gtb_codex.py` **整支 Flask 共用一個全域 `SESSION_ID`**——同 OA 不管誰打 LINE 進來，訊息都串進同一條 codex 對話。

阿全測式版（`@526fdbzo`）規劃要給 colombo + 彥偉、羿宏、士豪 4 個人用；目前 codex `ask_codex(user_text)` 只把訊息文字餵給 codex、**user_id 沒進 codex**（[gtb_codex.py:347-376](gtb_codex.py#L347-L376) `ask_codex` + [gtb_codex.py:408-425](gtb_codex.py#L408-L425) `/callback`）。

最危險情境：彥偉問「乙烯有幾點」、羿宏接著打「改 5 點」、codex 看上下文以為是彥偉要求 → 走 `adjust_points` 確認流程 → 跨人錯亂寫入。

CS（客服）後續上線時這是**絕對不能容忍**的——客戶與客戶之間絕對不能串味、否則隱私 / 商業資訊外洩。所以借這次三人試吃前先把架構鎖好、之後接上身分識別系統（[`mcp_principal_開發計畫`](docs/mcp_principal_開發計畫.md)）才順。

### 為什麼今晚不動

- 老闆同事不在、多人 LINE UX 真測要等明天
- 今天已動不少 shared state（重啟阿全 3 次、改 MCP server、改 AGENTS.md），再疊一個 session 重構是負擔
- 設計上有幾個決策點想跟 colombo 對一下（見下方）再動

### 預計設計

1. 全域單 `SESSION_ID` → 字典 `SESSIONS: dict[user_id, str]`
2. `.gtb_codex_session`（單一 UUID）→ `.gtb_codex_sessions/<user_id>.session` 或單一 `sessions.json` map
3. `/callback` 每次抓 `event.source.user_id`、傳給 `ask_codex(user_text, user_id)`
4. `ask_codex` 改：
   - 拿 user_id 對應的 sid、沒有 → 呼 `_mint_new_session()`、寫入 map
   - 有 → `codex exec resume <sid>` 照舊
5. AGENTS.md 不用改（codex 看到的就是「自己跟一個人對話」、不知道有別人）

### ⚠️ 跟既有 principal 架構的分歧（重要）

[`cwsoft-ai-mcp-and-principal-architecture.md`](../../cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md) 第 7-8 節寫的是**路線 A**：

> 全域單一 `SESSION_ID`、每收 LINE 訊息 → `lookup_principal(line_user_id)` → 注入 `setup_context(role=..., tenant_id=...)` 指令給 codex → codex 第一件事呼叫 setup_context 切 principal → 才進使用者本文

本 TODO 是**路線 B**（per-user session）：

> 每個 line_user_id 一條獨立 SESSION_ID。principal 在 session **mint 時**注入一次、boot prompt 就把該 user 的 role/tenant/db_name 寫進去、不需 setup_context per turn 切換。

對比：

| 維度 | 路線 A（既有 doc） | 路線 B（本 TODO） |
|---|---|---|
| codex session 數量 | 1 條/OA | N 條/OA（=用戶數）|
| MCP server `_session_principals` dict | N 把 key（每 LINE user 一把）、用 codex session_id keyed | 還是 N 把 key、但每把對應**整條** session 而非每 turn |
| `setup_context` 呼叫頻率 | **每 turn**（codex 收訊息第一件事）| **mint 一次**（或直接寫進 boot prompt） |
| 跨 user 對話污染 | 機制層**有**風險（codex history 混所有人）、靠 setup_context 切 + AGENTS.md 守 | 機制層**無**（codex history 只看到一個人）|
| 共學記憶 | 天然共享（彥偉教的羿宏看得到）| 各自獨立（共學靠 AGENTS.md 撐）|
| 每人首訊延遲 | 0（複用全域 session）| ~8s（mint）|
| 客服場景適配 | 不適（跨客戶絕不能共享 history）| 適 |
| 老闆+員工內部場景 | 適（共學有價值）| 適（隔離也合理、且機制更乾淨）|

**我的傾向**：客服必須 B、內部阿全 B 也比 A 乾淨（codex 不用在每 turn 認證身份切 context、減一層出錯空間）；A 的「共學」優勢用 AGENTS.md + 定期 colombo 手動萃取「教過的事」放進 AGENTS.md 就能撐住。

**對既有架構文件的影響**：本 TODO 落實後、`cwsoft-ai-mcp-and-principal-architecture.md` 第 7-8 節需要改寫——`setup_context` 從「per-turn 切 principal」降級為「per-session 初始化」、配合 per-user session 的天然隔離。第 9-12 節（多 server 策略、實作路徑、待解風險）大致不受影響。

### 決策點（明天跟 colombo 對）

- **路線 A vs B 拍板**——上表攤開來、確認走 B 並更新架構文件
- **mint 延遲**：每人首次 ~8s mint 是否要先 LINE 回個「初次見面、整理中…」訊息蓋過、避免 reply_token 過期
- **sid 容器**：JSON map 還是各別檔案。我傾向各別 `.session` 檔（鎖檔簡單、改一筆不會 race）
- **舊 sid 怎麼辦**：019e2fe9... 已經有 colombo 一下午的對話、應該 migrate 成 `<colombo_user_id>.session`、不要丟
- **principal 注入時機**：mint 時把 principal 直接寫進 boot prompt vs mint 後第一個 turn 呼一次 setup_context。我傾向前者（少一次 tool 呼叫、principal 跟 session 1-to-1 永遠綁死）

### 自驗計畫（不用同事）

`/sim/<oaid>` 後門已存在、payload 自組 user_id 就能驗 3 件事：
1. 新 user_id 第一次 → mint 新 sid、寫入 map、principal 注入 boot prompt
2. 同 user_id 再來 → resume 對 sid、principal 不重注（codex 已記得）
3. 跨 user_id → A 的上下文絕對不進 B 的 reply（拿 A session 的 jsonl + B 的 jsonl 對比、沒交集）

3 件 PASS 後再叫同事明天打 LINE 真測（驗 reply_token 過期 / 多人撞同時段這層）。

### 相關

- 設計文件待改：[`cwsoft-ai-mcp-and-principal-architecture.md`](../../cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md) §7-8（per-turn → per-session-mint）
- 今日 worklog 已記分歧背景：[`2026-05-16-gtb-codex-line-bot-and-mcp-architecture.md`](../../cwsoft-project-tracker/worklogs/2026-05-16-gtb-codex-line-bot-and-mcp-architecture.md) §「待跟進」L173
- 同步銜接：[下方 MCP principal TODO](#todo-2026-05-16-feature-mcp-principalpolicysetup_context-身份識別層) 的 principal lookup 跟本 TODO 的 user_id → session map 是同一個 user_id 軸、合一起做最省力
- 教訓：[`feedback_confirm_before_shared_state_changes`](../../.claude/projects/c--Users-pos-Desktop-general-task-bot/memory/feedback_confirm_before_shared_state_changes.md)

---

## [TODO 2026-05-16] [FEATURE] MCP principal/policy/setup_context 身份識別層

### 緣起

阿全 LINE bot（gtb_codex.py + cwsoft_ai_tools MCP）已上線、目前只給 colombo 自己用、所有 tool 對任何呼叫者都開放。
之後 cs_codex.py 客服 AI 會共用同一支 MCP——必須先建身份識別層、不然客戶能看到 `list_project_files` 之類內部資訊。

### 採用設計

3 層 RBAC：
1. `principals.json`：LINE user_id → {role, tenant_id, db_name, scopes}
2. `policy.py`：tool_name → 允許的 role 集合
3. MCP server `setup_context` tool + 每個 tool 開頭 `_check()`：gtb_codex.py 每收 LINE 訊息、prepend 指示要 codex 先呼 setup_context、後續 tool 都 check 拿到的 principal

詳見 [docs/mcp_principal_開發計畫.md](docs/mcp_principal_開發計畫.md)。

### 對應對話 / 規劃

- 規劃對話：5/16 與 Claude (claude-opus-4-7)
- 架構文件：[`cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md`](../../cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md)
- worklog：[`cwsoft-project-tracker/worklogs/2026-05-16-mcp-server-mvp-and-gtb-vs-codex-reflection-2.md`](../../cwsoft-project-tracker/worklogs/2026-05-16-mcp-server-mvp-and-gtb-vs-codex-reflection-2.md)

### 開工檢查清單

- [ ] 寫 `tools/cwsoft_ai_tools/policy.py`（POLICY 表 + can()）
- [ ] 寫 `tools/cwsoft_ai_tools/principal_state.py`（session_id → principal）
- [ ] 改 `tools/cwsoft_ai_tools/server.py`（加 setup_context tool + 每 tool 加 _check）
- [ ] 寫 `cwsoft-aquan-manager/principals.json`（colombo owner 一筆）
- [ ] 改 `gtb_codex.py`（lookup_principal + build_setup_instr + /callback hook）
- [ ] 改 `cwsoft-aquan-manager/AGENTS.md`（加「第一件事呼叫 setup_context」hard rule）
- [ ] 跑 4 case 驗收（owner 通 / vip forbidden / read_doc 同樣分流 / anon 全 forbidden）
- [ ] 寫一篇 worklog 收尾

### 估工

1-2 小時（plumbing 都建好了、只剩裝 ACL）。

---

## [DONE 2026-05-12] [BUG] match_pool 拼音比對：短 keyword 對長 candidate 排序錯

### 現象

`extract_top_matching_names` / `extract_best_matching_name` 用 `SequenceMatcher.ratio()` 算拼音相似度，公式 `2*M/(len1+len2)`，分母含 candidate 全長 → 短 keyword 對長 candidate 即使是完整前綴，分數仍被長度稀釋。

例：keyword="零壹" vs candidate="零壹通訊行" → ratio = 12/23 = 0.52，反而輸給「星威」 0.62。

### 採用修法

走 A 變體：**substring 命中時直接走 0.95~1.0 高優先**（不再用 base + bonus 加成）：

```python
def calculate_phonetic_similarity(name1, name2):
    p1 = "".join(lazy_pinyin(name1))
    p2 = "".join(lazy_pinyin(name2))
    if p1 and p1 in p2:
        coverage = len(p1) / max(len(p2), 1)
        return 0.95 + 0.05 * coverage   # substring 命中 → 高優先
    return SequenceMatcher(None, p1, p2).ratio()  # 模糊匹配走原邏輯
```

理由：substring 完整命中是更強的 match signal，比 longest common subsequence 更直接。
單純 `base + bonus` 救不回極端 case（「零壹」 lingyi vs「林一」 linyi 只差一個 g，base 0.91，bonus 加進去仍輸 substring case）。

### 修法套用

- [x] gtb.py `calculate_phonetic_similarity` + `calculate_word_similarity` 都套用
- [x] gtb_dev.py 同步（兩份 copy 都改）
- [x] 測試 5 個 case 全 PASS：原 bug case / 完整名稱 / 純英文 / 模糊（非 substring）/ 完全不相干

### 後續還可以做（沒做）

- [ ] 加單元測試到 repo（目前只是 inline 跑了測試）
- [ ] 考慮把 `extract_best_matching_name` 跟 `extract_top_matching_names` 整合成一份（避免兩份 drift）

---

## [PLAN] 高危險指令預警系統

### 目標
對高風險副作用任務建立額外保護，避免 bot 誤執行造成不可逆資料異動。

### 適用範圍
- 客戶下架
- 點數加減
- 簡訊點數調整
- 重建資料
- 還原測試庫
- 其他會刪資料、停用資料、卸載資料庫或批次修改資料的操作

### 規劃
- [ ] 在 mission/task 層新增高危險標記，例如 `risk_level: high`
- [ ] 高危險任務一律強制多一道確認，不可只靠一般 `human_check`
- [ ] 顯示清楚預警訊息，列出可能影響範圍與不可逆風險
- [ ] 確認語句升級，例如要求使用者回覆完整客戶名或指定確認碼
- [ ] 建立操作審計紀錄，保留使用者、時間、任務、參數、確認內容
- [ ] 成功與失敗都寫入可追查紀錄

---

## [PLAN] classify_tree 支援多步驟 tasks

### 目標
讓 `mission.json` 的分類樹可直接表達「一個意圖命中後，依序執行多個 task」，避免把流程藏進單一 task 內部。

### 規劃
- [ ] 在 `classify_tree.branch` 支援新寫法：`"tasks": ["task_a", "task_b"]`
- [ ] 保持舊寫法相容：若只有 `"task_id"`，仍視為單一步驟
- [ ] 執行層統一轉成 task list 後依序執行
- [ ] 文件補充：branch 負責決定執行序列，task 負責定義單一步驟

---

## [PLAN] 信心值輸出模式演進

### 目前決策
- [ ] 先採用 `with_confidence: true/false`（欄位級設定）
- [ ] `classify_tree` 強制走含信心值輸出

### 後續演進（保留）
- [ ] 若未來需要超過兩種輸出格式，再升級為 `answer_mode`
- [ ] 升級時保持向下相容：舊的 `with_confidence` 仍可被解析

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
