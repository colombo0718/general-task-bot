# CLAUDE.md — 通用工作規範

> 這份文件是跨專案通用的 AI 協作規範。
> 專案本身的架構、協定、技術細節寫在 `PROJECT.md`。

@PROJECT.md

---

## 檔案系統規範

每個專案應維護以下檔案，各司其職：

| 檔案 | 對象 | 寫什麼 |
|------|------|--------|
| `CLAUDE.md` | Claude（AI） | 通用工作規範（本檔）+ @PROJECT.md |
| `PROJECT.md` | Claude（AI） | 這個專案的架構、協定、已知坑、專案特有開發習慣 |
| `README.md` | 陌生人 | 專案介紹、安裝、使用方式 |
| `TODO.md` | 開發者 | 待辦、擱置功能、未來想法 |
| `CHANGELOG.md` | 開發者/使用者 | 重大改動里程碑、架構決策紀錄 |

### PROJECT.md 寫什麼
- 這個專案是什麼（一段話定位）
- 線上網址、部署方式
- 架構概覽（目錄結構、關鍵檔案）
- 通訊協定或 API 規格（不顯而易見的部分）
- 已知 bug / quirk / 例外處理
- 開發規範（commit 語言、避免大改的理由）
- 專案特有開發習慣（命名規則、工具使用偏好、格式慣例等）

### TODO.md 格式規則
- `[ ]` 待辦，`[x]` 完成
- 條目後附說明（why + 設計考量）
- 有依賴關係的加 `> 需等 xxx 完成`
- 用主題區段分組，不用時間順序
- 擱置的功能附擱置原因

### CHANGELOG.md 寫什麼
- 功能完整上線的里程碑（不是每個 commit）
- 架構層級的重大決策（換部署平台、協定改版）
- 破壞性變更（舊介面不相容）
- 重要會議或決策結果
- **不寫**：小 bug fix、文字調整（那是 git log 的事）

---

## Sub-agent 工具規範

本專案環境下可調用兩個 AI 小弟：

### Gemini（`gemini -p "..."`)
```bash
gemini -p "你的 prompt"                                    # 基本用法
gemini --include-directories "C:/path/outside" -p "..."   # 讀取 workspace 外的目錄
gemini --yolo -p "..."                                     # 自動批准所有工具調用
gemini -p "..." --output-format json                       # 結構化輸出，方便 Claude 解析
timeout 60 gemini -p "..."                                 # 加 timeout 保護，避免等太久
```

**適合交給 Gemini 的任務：**
- 需要 web search 的研究（查競品、查 API 文件、查最新規範）
- 批量生成相似內容（多篇文章、多個說明卡片）
- 讀取 workspace 外的目錄（用 `--include-directories`）
- 翻譯、改寫、潤稿

**不適合：**
- 需要深度理解本專案架構的改動（它沒有這段對話的 context）
- 需要跟使用者來回確認的決策性工作

### Codex（`codex exec "..."`)
```bash
codex exec "你的 prompt"    # 非互動模式
codex review                # code review 模式
```

**適合交給 Codex 的任務：**
- Code review（用 `codex review`）
- 分析某段程式的邏輯或潛在問題
- 生成符合現有 codebase 風格的程式碼片段

**注意：**
- 預設 sandbox 是 read-only，執行 shell 命令會被擋
- 模型為 gpt-5.4

### 分工原則
```
批量生成相似內容             → gemini（並行或順序）
需要 web search 的研究       → gemini
讀取 workspace 外目錄        → gemini --include-directories
Code review / 程式分析       → codex review
需要跟使用者確認的決策       → 不委派，自己處理
需要理解本專案 context 的改動 → 自己做
```

---

## Branch / 部署策略

```
功能開發    → dev 分支（或 feature/xxx）
穩定版本    → merge to master
部署        → 直接在目標機器 git pull 後重啟 main.py
```

