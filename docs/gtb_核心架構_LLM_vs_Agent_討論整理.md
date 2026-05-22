# GTB 核心架構選擇：LLM extractor vs Agent

> 整理日期：2026-05-12
> 觸發：老闆看到 codex CLI 在 terminal 互動的順暢度，想用它直接取代 GTB
> 結論：**GTB harness 框架對的，內核可演進；不換框架**

---

## 一、討論起點

老闆跟同事看到 codex CLI 在 terminal 講話就能跑事情很驚艷，提出「**直接用 codex CLI 取代 GTB**」的想法。

需要釐清：
1. codex CLI 為什麼讓老闆覺得「比 GTB 好」？（demo 印象 vs 服務化現實）
2. GTB 的設計方向（多 extractor + classify_tree）2026 年還對嗎？
3. 如果要改，是換 LLM 後端還是換整個架構？

---

## 二、關鍵認知差異：互動使用 vs 服務嵌入

老闆看 codex 是 **互動式使用**（人在 terminal 直接講話），這跟 **服務嵌入**（chatbot 接 LINE webhook 自動執行）是完全不同的工程：

| 維度 | 互動使用（terminal）| 服務嵌入（LINE bot）|
|------|---------|----------|
| 看到錯就喊停 | ✅ 人類即時審 | ❌ 客戶已收到錯誤回覆 / 錯誤 SQL 已執行 |
| 30 秒等待 | ✅ 沒問題 | ❌ LINE 那邊以為服務掛了 |
| 失敗 | 重來就好 | 要寫 retry / fallback / 監控 |
| 併發 | 一人一視窗 | N 個客戶同時打，要 lock / idempotency |
| 成本 | 一個人慢慢打 | 被洗版或攻擊一分鐘燒掉一個月預算 |
| 可追蹤 | 終端有 log 看 | 法遵/客訴需要審計鏈 |

類比：codex 在 terminal 好用 ≈ 「資深工程師在 SSMS 跑 query 很方便」，但**沒人會把 SSMS 直接接到客戶下單頁面**。中間那層服務化（驗證、審計、限流、人工確認）就是 GTB 在做的事，**不能因為核心換成 agent 就省掉**。

---

## 三、GTB 是什麼：harness engineering

GTB 本質上是 **LLM harness engineering**，跟業界主流方向一致：
- Claude Code
- Cursor / Devin
- LangGraph / DSPy
- function calling / tool use frameworks

都是把 LLM 包在「可配置、可觀測、可換卡」的 harness 裡，因為**裸 model 在生產環境不夠用**。

GTB 已具備的 harness 能力：
- 設定驅動（mission.json + prompts.ini）→ 新功能不用動 Python
- 多 OA / 多專案分離 → 同一個引擎跑多個 bot
- LLM 後端可換（llm_clients.py）→ 不綁 Groq
- 狀態機（todo / decisions / clarifications）→ 業務邏輯持久化
- shadow mode → 安全試行
- super-manager 接管 → 服務不掛

「**用 codex 直接取代 GTB**」= 把 harness 拆掉只留 model = 業界反方向。

---

## 四、技術可行性實測（2026-05-11）

當天為了解 Groq rate limit 問題，順便驗證了幾個 LLM provider 能不能直接當 GTB 的後端：

| Provider | 速度 | 格式穩定 | 適合 GTB 後端 | 備註 |
|----------|------|----------|--------------|------|
| **Groq Llama-3.3-70B** | ~1s | ✅ | 主力 | 免費但有 RPM 限制 |
| **HF Llama-3.3-70B Instruct** | ~1s | ✅ | ✅ Fallback 採用 | 免費、跟 Groq 同模型家族 |
| **OpenRouter gpt-4o-mini** | ~1.2s | ✅ | 可用（付費）| 帳戶有零頭餘額可用 |
| **codex_cli** | 8-15s | ❌ | ❌ 不適合 | Agent 設計，會「執行」prompt 而不是回傳完成結果 |
| **gemini_cli** | 15s | ✅ | ❌ 太慢 | CLI spawn overhead |
| **remote_worker (Ollama)** | 11s | ✅ | ❌ 太慢 | 本機 Ollama 推論慢 |
| **Gemini API** | 0.7s ~ rate limit | ✅ | ⚠️ 配額為 0 | API key 在沒啟用 billing 的 project |

**重要發現**：codex CLI 雖然強，但**不能直接當 GTB 的 LLM provider**——它是 agent 不是 completion API，會把 prompt 當任務執行，不會單純回傳格式化結果。

實際採用：**Groq 主力 + HF fallback**（已上線運作）。

---

## 五、選項對照

| 路線 | 改動範圍 | 優點 | 缺點 |
|------|---------|------|------|
| **A. 不變**（現狀繼續用 multi-extractor） | 0 | 穩定可運作 | 22 個 task 之後相鄰功能互相誤判（今天剛踩過 extendDueDate vs refreshCompanies） |
| **B. 換 LLM 後端但結構不變** | 小（換 provider） | 不影響業務邏輯 | 治標 — extractor 本質還是 22 選 1 |
| **C. 階梯式 classify**（樹狀分類）| 中（mission.json 結構 + gtb 加遞迴）| 維持 extractor 架構但縮小 LLM 心智負擔 | 仍需要 prompts 維護 |
| **D. GTB v2: agent-first inner core** | 大（重寫 classify+extract 層）| 一次 LLM call 解決、prompt 大幅瘦身、貼齊業界 | 工程量大、要驗證 agent 對嚴格格式的可靠度 |
| **E. 全用 codex CLI 取代 GTB** | 全砍重來 | 最潮 | 失去 harness 所有能力，老闆 demo 會掛 |

**今天（5/12）採行**：B（已做 Groq+HF fallback）+ 開始試 C（dev 環境 4 層階梯式分類器）。

**長期方向**：D — 但要等 C 驗證可行後再評估，並非今天該動的事。

**永遠不採**：E — 框架移除等於重新發明所有運維能力。

---

## 六、若走 D（GTB v2 agent-first）會是什麼樣

對齊 function calling / tool use 慣例：

### mission.json 演進
從「task → prompt_key 串接」變成「**tool catalog**」：

```json
{
  "tools": [
    {
      "name": "create_branch",
      "description": "在指定客戶 DB 新增分店",
      "parameters": {
        "name": {"type": "string", "match_pool": "customerlist.txt"},
        "shop_name": {"type": "string"}
      },
      "human_check": "true",
      "url_template": "..."
    },
    ...
  ]
}
```

### prompts.ini 退場 95%
現有 23 個 extractor（identify_needs / identify_name / get_*_date / ...）大部分不需要——agent 看 tool 描述自己抽參數。

只留 1 份精簡 system prompt 寫**領域知識**：
- 「加/減/查」後接數字是動作詞（邊界規則）
- 「有支手機」是公司名不是日常語句
- 民國年要轉西元
- 區碼對照表

### 框架外殼不變
- LINE webhook 入口
- OA 多租戶路由
- todo_list 排程
- decisions 狀態機
- super-manager 接管
- shadow mode
- audit log

**「LLM 策略」可以換，「框架外殼」還是要留**。

---

## 七、跟今天踩雷的對應

5/12 在 LINE 端跟 LLM 互動誤判很多次：
- adjust_points 跟 query_points 混淆（前者比後者頻率高 4 倍但 LLM 還是抽錯）
- extendDueDate 被歸成 refreshCompanies
- 「對」沒被列舉所以 extracted_ronot 回 null
- regex {2,7} 跟「最高優先 POSV3測試專用」(9 字) 互相矛盾

這些都不是「換 codex 就解決」的問題——而是 **prompt 工程在 22 選 1 規模下天然會撞牆**。換 codex agent 也會踩，只是踩的方式不一樣。

正確解法：
1. 短期：階梯式分類降低每層 LLM 心智負擔（C 路線，已開始做）
2. 中期：信心不足主動問（dev 已寫好 clarification flow，回寫 prod）
3. 長期：D 路線（agent-first inner core），但要 C 驗證後才評估

---

## 八、共識與待 revisit

### 已共識
- ✅ GTB harness 框架方向正確
- ✅ codex CLI 不適合直接當 LLM provider
- ✅ 「老闆要的功能」基本都跟「核心是 LLM 還是 agent」無關，跟「**框架外殼有沒有提供互動品質**」更相關
- ✅ Groq + HF fallback 是當前夠用的 LLM 配置
- ✅ 今天先做階梯式分類（C），不直接跳 v2 重寫（D）

### 待 revisit（驗證後再決定）
- 階梯式分類在 dev 環境驗證後，準確率有沒有顯著超過平面架構？如果只是邊際改善，是否該直接跳 D？
- D 路線真要做的時候，agent 對 prompts.ini 那些「嚴格格式約束」的可靠度是否夠？要先 spike 測試。
- codex 在某些 corner case（例如「複雜業務邏輯需要多步推理」）可能比 single LLM call 強，但這類 case 在阿全 LINE bot 場景出現多少？需要從 boss_chat 統計。

---

## 九、給老闆的解釋話術

如果老闆再提「為什麼不用 codex」，可以用：

> codex 在您看到的 demo 是「**互動使用**」——您看著它跑、有錯就喊停，這時候它很好用。
>
> 但接到 LINE 變成「**服務嵌入**」——客戶傳訊息，bot 自動執行，沒人在中間審。這時候 codex 那種「自由發揮」的特性反而是風險：
> - 它可能花 15 秒才回（客戶以為 LINE 掛了）
> - 它可能誤解意圖直接執行錯的 SQL（客戶資料受影響）
> - 它可能因為 prompt 略有歧義就回不一致的格式（後續處理炸掉）
>
> GTB 框架在 LLM 跟客戶之間多了一層**「審慎執行」**——驗證、確認、追蹤、限流。這層不是因為 LLM 不夠強，是因為**自動化服務本質上需要這層**。
>
> 內核我們可以換更聰明的（已經換了 Groq + HF），架構未來也可以演進到 agent-first 的單次調用，但**這層服務化的外殼您一定會需要**。

---

## 十、相關文件

- [check/targets.md](../check/targets.md) — 老闆對阿全/客服的外在表現期待（靶圖）
- [check/checklist.md](../check/checklist.md) — 跨技術核心的功能檢核清單
- [docs/gtb_dev_flow.md](gtb_dev_flow.md) — gtb_dev.py 對話與 DB 流程
- [docs/gtb_後門動工清單.md](gtb_後門動工清單.md) — 對方 Claude session 寫的後門 + 考題 runner 計畫
- [matrix-manager/meetings/2026-05-12-aquan-tier-classifier-prep-for-backdoor.md](../../matrix-manager/meetings/2026-05-12-aquan-tier-classifier-prep-for-backdoor.md) — aquan 端配合後門的階梯式分類器準備（C 路線實作）
