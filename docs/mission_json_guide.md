# mission.json 撰寫規則

## 頂層結構

```json
{
  "version": 1,
  "locale": "zh-TW",
  "classify_tree": { ... },
  "tasks": { ... }
}
```

| 欄位 | 說明 |
|------|------|
| `version` | 整數版本號，純供人工辨識，主程式不使用 |
| `locale` | 語系標記，目前固定 `"zh-TW"` |
| `classify_tree` | 定義如何將使用者訊息分類到對應任務 |
| `tasks` | 所有任務的定義集合 |

---

## `classify_tree`（意圖分類樹）

```json
"classify_tree": {
  "prompt_key": "<prompts.ini 中用來分類的 extractor key>",
  "branch": [
    { "match": "<extractor 可能回傳的值>", "task_id": "<tasks 中的 key>" },
    { "match": "<另一個值>",               "task_id": "<另一個 task>" },
    { "match": "null",                     "task_id": "<fallback task>" }
  ]
}
```

**規則：**
- `prompt_key` 必須對應 `[extractors]` 中存在的 key
- `branch` 是有序陣列，**第一個符合的 match 生效**
- 必須有一個 `"match": "null"` 的 fallback，對應分類失敗時的任務
- `match` 的值必須與 extractor 實際輸出的字串**完全一致（含大小寫）**

---

## `tasks`（任務定義）

```json
"tasks": {
  "<task_id>": {
    "description": "<人類可讀說明>",
    "human_check": "true" | "false" | "auto",
    "fields": {
      "<欄位名>": {
        "prompt_key":  "<extractor key>",
        "required":    true | false,
        "reference":   true,
        "match_pool":  "customerlist.txt"
      }
    },
    "action": {
      "method":       "GET" | "POST",
      "url_template": "https://..."
    }
  }
}
```

---

## `human_check` 三種模式

| 值 | 行為 | 適用情境 |
|----|------|---------|
| `"false"` | 直接執行，不等使用者確認 | 純查詢、唯讀操作 |
| `"true"` | GET 預覽後等確認，預設**取消** | 高風險寫入（刪除、下架、還原） |
| `"auto"` | GET 預覽後等確認，預設**執行** | 一般寫入（加點、開功能） |

使用者確認流程：
1. 主程式先用 GET 呼叫 API，將預覽結果回覆給使用者
2. 使用者回覆確認詞（好、OK、確認）→ 實際執行
3. 使用者回覆取消詞（不、取消）→ 放棄

---

## `fields` 欄位屬性說明

| 屬性 | 必填 | 型別 | 說明 |
|------|------|------|------|
| `prompt_key` | 是 | string | 對應 `[extractors]` 的 key，主程式呼叫 LLM 萃取此欄位 |
| `required` | 是 | boolean | `true` = 缺少此欄位就中止並回覆缺件；`false` = 允許為 null |
| `reference` | 否 | boolean | `true` = 將今日日期等 reference 資訊一併傳給 LLM |
| `match_pool` | 否 | string | 指定候選清單檔案，LLM 抽出的值會做模糊比對修正 |

### `reference` 的作用
設為 `true` 時，主程式會在呼叫 LLM 前加上以下區塊：

```
[REFERENCE]
today_date: 2025-04-01
currency: TWD
```

適合用在需要計算日期的欄位（如「昨天」「下個月」「月底」）。

### `match_pool` 的作用
LLM 萃取出值後，程式會用拼音相似度比對清單中最接近的名稱（門檻 0.8）。
目前支援：
- `"customerlist.txt"` — 客戶名稱清單（由腳本自動產生）
- 任意 `.txt` 檔（每行一個候選詞）

---

## `action.url_template` 規則

- 用 `{欄位名}` 插入 fields 中定義的欄位值
- 欄位名必須完全對應 `fields` 中的 key
- **`user_id` 由主程式自動附加**，不需要寫在 template 裡
- 未填入模板的欄位會自動附加到 query string（GET）或 request body（POST）

```
url_template: "https://example.com/api?name={name}&delta={delta}"

實際呼叫: https://example.com/api?name=以西&delta=5&user_id=Uabc123
```

---

## fallback task 寫法

分類失敗或使用者輸入無法對應任何任務時的處理節點。

```json
"fallback_greeting": {
  "description": "分類失敗時的友善提示",
  "human_check": "false",
  "fields": {},
  "action": {
    "method": "GET",
    "url_template": "https://example.com/ping"
  }
}
```

- `fields` 為空物件 `{}`
- `classify_tree.branch` 中必須有 `"match": "null"` 指向此 task

---

## 完整最小範例

```json
{
  "version": 1,
  "locale": "zh-TW",
  "classify_tree": {
    "prompt_key": "identify_expense",
    "branch": [
      { "match": "expenseRecord", "task_id": "record_expense" },
      { "match": "null",          "task_id": "fallback_greeting" }
    ]
  },
  "tasks": {
    "record_expense": {
      "description": "記錄一筆花銷流水",
      "human_check": "true",
      "fields": {
        "item":  { "prompt_key": "get_item",  "required": true },
        "price": { "prompt_key": "get_price", "required": true },
        "date":  { "prompt_key": "get_date",  "required": false, "reference": true }
      },
      "action": {
        "method": "POST",
        "url_template": "https://example.com/expense?item={item}&price={price}&date={date}"
      }
    },
    "fallback_greeting": {
      "description": "分類失敗時的友善提示",
      "human_check": "false",
      "fields": {},
      "action": {
        "method": "GET",
        "url_template": "https://example.com/ping"
      }
    }
  }
}
```

---

## 新增業務的完整流程

1. 在 `prompts.ini` 的 **分類 extractor** 新增對應的輸出值（task 名稱）
2. 在 `prompts.ini` 新增 **每個欄位的 extractor**
3. 在 `mission.json` 的 `classify_tree.branch` 加上 `{ "match": "<新值>", "task_id": "<新 task>" }`
4. 在 `mission.json` 的 `tasks` 新增 task，定義 `fields` 與 `action`

---

## 多個 mission.json 的命名規則

與 `prompts.ini` 對應，主程式啟動時依參數決定讀哪組：

```
python main.py          → mission.json
python main.py pos      → mission_pos.json
python main.py store    → mission_store.json
```
