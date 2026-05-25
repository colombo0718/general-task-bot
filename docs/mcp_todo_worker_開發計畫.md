# MCP 版 todo_worker 開發計畫

> 日期：2026-05-24 撰寫、5/25 起動工
> 對應 TODO：[`general-task-bot/TODO.md`](../TODO.md) `[TODO 2026-05-24] [FEATURE] MCP 版 todo_worker`
> 對應 worklog（起源）：[`2026-05-24-day4-...-mcp-scheduling-design.md`](../../cwsoft-project-tracker/worklogs/2026-05-24-aquan-prod-cutover-day4-boss-line-deferred-action-and-mcp-scheduling-design.md)

---

## 1. 緣起

5/24 上午 11:01 老闆 LINE 跟阿全互動「**我要在月底停用水源台哥大**」、阿全當下沒有排程 MCP tool、只能跟老闆說「**5/31 你再叫我**」並記在 codex session memory 裡。

問題：codex session memory **不是真排程**、Flask 重啟 / session 換 mint 就掉。老闆要的是「**真的 5/31 23:59:59 自動跑、不用我再叫**」。

→ 需要把 commitment **persist 到 db**、worker 到時間自動觸發 MCP function。

## 2. 設計決策

### 2.1 純 MCP 化、**不留 url_template 向下相容**

colombo 5/24 拍板：

> 「公司有訂閱 codex 商業版、額度給這些 AI 服務用。現在效果這麼好、不太可能走回頭路去用舊版 GTB。**todo worker 就改制成觸發 MCP 的吧**。」

含義：
- 舊 GTB cascade 路徑（extract_run_at → todo_items.url → todo_worker GET URL）**整段退役**
- todo_worker 純改成「import MCP function + call」
- todo_items schema 改寫、不留舊 `url` 欄
- Migration：先 dump 既有 pending 看有沒有要 port、有再決定

### 2.2 走 A1（純 Python import）、不走 A2（ephemeral codex）

排程觸發**無 AI 推理需求**——spec 已 frozen（哪個 function、哪些 args、什麼時間）、純執行。

| 維度 | A1 import | A2 ephemeral codex |
|---|---|---|
| 速度 | < 1s | 10-30s |
| 成本 | 0 | codex token |
| 失誤機率 | 極低（已 verified function、零 AI 中介） | 偏低、AI 仍可能漏 |

選 A1。AI 中介只在「**使用者→排程指令**」那步需要（codex 阿全跟使用者對話、決定排哪個 spec 進去）、一旦 spec 寫進 db、後續純工程。

### 2.3 codex 介面走 `@schedulable` decorator、不走 universal `schedule_action` meta-tool

5/24 對話 colombo 指出：universal `schedule_action(tool_module, tool_func, tool_args, run_at)` 是「**meta-MCP 包另一個 MCP**」、大包小、codex 用起來囉嗦、type safety 弱、容易拼錯。

對比三種介面：

| 設計 | codex 寫法（排程 5/31 軟下架 鑫盛/水源-台哥大）| 評價 |
|---|---|---|
| A. universal meta-tool | `schedule_action(tool_module="cwsoft_sqlserver_mcp.server", tool_func="close_branch", tool_args={"name":"鑫盛","branch_code":4}, run_at="2026-05-31T23:59:59")` | 大包小、4 個必填 abstract 欄、認知 overhead 高 |
| **B. 寫入 tool 自帶 `run_at` + `@schedulable` decorator**（**選這個**）| `close_branch(name="鑫盛", branch_code=4, run_at="2026-05-31T23:59:59")` | 一個 tool 兩種模式（有 run_at = 排程、沒 = 立即）、type safety 強、自然語意對應乾淨 |
| C. 每個寫入 tool 出 `_scheduled` 版 | `close_branch_scheduled(name="鑫盛", branch_code=4, run_at=...)` | 工具集翻倍、AGENTS.md 清單膨脹、重複命名 |

選 **B** 的理由：

1. **codex 認知最輕**——不必學「schedule_action」abstract 概念、就是「寫入類 tool 多了 optional `run_at` 參數」（cumulative 學習、沒新概念）
2. **type safety 強**——args 走 Python function signature 檢查、不是 JSON dict 任意鍵值
3. **自然語意對映乾淨**——使用者「現在做 / 之後做」 = 同一個 tool 的兩種模式
4. **notebook audit 直接**——`[ACTION] close_branch(..., run_at=...) 已排程` 比 `[ACTION] schedule_action(tool_func=close_branch, ...) 已排程` 直接得多
5. **pattern 可推廣**——`@schedulable` 是 cross-cutting concern 的標準解、未來其他類似橫切需求都可同樣套（`@auditable` 強制寫 audit log / `@rate_limited` 限流 / `@requires_confirmation` 統一確認流程）

代價：每個寫入類 tool 加幾行 boilerplate、用 decorator 集中、加新 tool 只需貼一行。

### 2.4 適用範圍：只「寫入類」、唯讀類不加

| tool 類型 | 加 `@schedulable` ? | 例子 |
|---|---|---|
| 寫入類（會改 DB） | ✓ 加 | adjust_points / adjust_sms_points / create_branch / close_branch / generate_quote |
| 唯讀類（純查詢） | ✗ 不加 | query_points / list_branches / read_doc / match_customer_name / readonly_query |

唯讀類沒有「排程查一次」的業務需求——要看就現在看、不會排到未來。

### 2.5 設計含義：MCP function = single source of truth

```python
@mcp.tool()
def close_branch(name: str, branch_code: int) -> dict:
    # ...純 Python 邏輯
```

`@mcp.tool()` 只是「對 MCP 協定註冊」、function 本身可從多種 entrypoint 呼叫：

| 觸發者 | 呼法 |
|---|---|
| codex agent | MCP 協定（JSON-RPC stdio） |
| **todo_worker（排程）** | **import + call**（本案要做的）|
| 批次腳本 | 同上 |
| 單元測試 | 同上 |
| 未來客服 AI | MCP 協定 |

寫一次、N 種觸發方式都能用。

---

## 3. Schema 改造

### Before（既有、`todo_list.py:9-23`）

```sql
CREATE TABLE todo_items (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    run_at    TEXT NOT NULL,
    url       TEXT NOT NULL,             -- ❌ 拿掉
    user_text TEXT NOT NULL,
    state     TEXT NOT NULL DEFAULT 'pending'
)
```

### After

```sql
CREATE TABLE todo_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,           -- 排程發起人（LINE user_id）
    created_at      TEXT NOT NULL,           -- 排程當下時間 ISO
    run_at          TEXT NOT NULL,           -- 排程觸發時間 ISO
    tool_module     TEXT NOT NULL,           -- e.g. "cwsoft_sqlserver_mcp.server"
    tool_func       TEXT NOT NULL,           -- e.g. "close_branch"
    tool_args_json  TEXT NOT NULL,           -- e.g. '{"name":"鑫盛","branch_code":4}'
    user_text       TEXT NOT NULL,           -- 原始使用者訊息（audit / debug 用）
    note            TEXT,                    -- 給人看的描述（如「月底下架 鑫盛/水源-台哥大」）
    state           TEXT NOT NULL DEFAULT 'pending',  -- pending / executed / failed / cancelled
    executed_at     TEXT,                    -- 實際執行時間（state ∈ {executed, failed} 時填）
    execution_result TEXT                    -- 執行結果 JSON / error 訊息
)
```

### Migration 步驟

1. **dump 既有 todo_items**：`SELECT * FROM todo_items WHERE state='pending'`
2. 若有 pending row：
   - 跟 colombo 確認舊 GTB 是否還有任何依賴
   - 確認可丟 → `DROP TABLE todo_items` + 重建
   - 不能丟 → 暫不動 schema、討論 port 策略（手動轉成新 schema）
3. 若 0 row pending：直接 `DROP TABLE todo_items` + 重建新 schema
4. 備份舊 db：`cp todo_list.db todo_list.db.bak.<時間戳>`、可救命

---

## 4. `@schedulable` decorator + 寫入類 tool 加 `run_at` 參數

放 `cwsoft-sqlserver-mcp/src/cwsoft_sqlserver_mcp/scheduling.py`（新 module、跟 db / server 平行）。

### 4.1 Decorator 實作

```python
# cwsoft_sqlserver_mcp/scheduling.py
import functools
import inspect
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

TODO_DB_PATH = Path(r"C:\Users\pos\Desktop\general-task-bot\todo_list.db")


def schedulable(func: Callable) -> Callable:
    """
    Decorator——讓寫入類 MCP function 自動支援 `run_at` 參數。

    傳了 run_at → 不執行、把 call 寫進 todo_list.db、worker 到時間 import + call
    沒傳 / None → 立即執行原 function

    使用方式（裝飾順序：@mcp.tool() 在外、@schedulable 在內）：

        @mcp.tool()
        @schedulable
        def close_branch(name: str, branch_code: int) -> dict:
            # 純執行邏輯、不必管 run_at
            ...
    """
    sig = inspect.signature(func)
    func_module = func.__module__
    func_name = func.__name__

    @functools.wraps(func)
    def wrapper(*args, run_at: str = None, note: str = "", **kwargs) -> Dict[str, Any]:
        if not run_at:
            # 立即執行
            return func(*args, **kwargs)

        # 排程：序列化 args + kwargs 進 db
        # 把 *args 對應到 signature 的位置參數、合成 kwargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        all_args = dict(bound.arguments)

        # 驗證：run_at 必須是未來時間
        try:
            run_at_dt = datetime.fromisoformat(run_at)
        except ValueError:
            return {"ok": False, "error": f"run_at 格式錯（要 ISO 8601、收到 {run_at!r}）"}
        if run_at_dt <= datetime.now():
            return {"ok": False, "error": f"run_at 必須是未來時間（收到 {run_at}）"}

        # 寫進 todo_list.db
        try:
            schedule_id = _insert_scheduled(
                tool_module=func_module,
                tool_func=func_name,
                tool_args=all_args,
                run_at=run_at,
                note=note,
            )
        except Exception as e:
            return {"ok": False, "error": f"排程寫入失敗: {e}"}

        return {
            "ok": True,
            "scheduled": True,
            "schedule_id": schedule_id,
            "run_at": run_at,
            "tool_func": func_name,
            "tool_args": all_args,
            "note": note,
        }

    return wrapper


def _insert_scheduled(
    tool_module: str,
    tool_func: str,
    tool_args: Dict[str, Any],
    run_at: str,
    note: str = "",
    user_id: str = "",
) -> int:
    """寫一筆 row 進 todo_items、回 schedule_id"""
    conn = sqlite3.connect(TODO_DB_PATH)
    try:
        cur = conn.execute("""
            INSERT INTO todo_items
              (user_id, created_at, run_at, tool_module, tool_func, tool_args_json, user_text, note, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            user_id,
            datetime.now().isoformat(timespec='seconds'),
            run_at,
            tool_module,
            tool_func,
            json.dumps(tool_args, ensure_ascii=False),
            "",  # user_text 由 codex 在外層補（可選）
            note,
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()
```

### 4.2 既有寫入類 tool 改造（5 個）

對每個既有寫入 tool 貼 `@schedulable` decorator：

```python
# cwsoft_sqlserver_mcp/server.py

from .scheduling import schedulable

@mcp.tool()
@schedulable
def adjust_points(name: str, delta: int) -> dict:
    # ... 原有立即執行邏輯不變

@mcp.tool()
@schedulable
def adjust_sms_points(name: str, amount: int) -> dict:
    # ... 原有

@mcp.tool()
@schedulable
def create_branch(name: str, shop_name: str) -> dict:
    # ...

@mcp.tool()
@schedulable
def close_branch(name: str, branch_code: int) -> dict:
    # ...

# tools/cwsoft_ai_tools/server.py

from cwsoft_sqlserver_mcp.scheduling import schedulable  # cross-package import

@mcp.tool()
@schedulable
def generate_quote(name: str, charge_months: int = None, ...) -> dict:
    # ...

@mcp.tool()
@schedulable
def generate_perip_quote(name: str, paper: int = 0, ...) -> dict:
    # ...
```

decorator 順序重要：`@mcp.tool()` 必須在**外面**（先註冊到 MCP）、`@schedulable` 在**裡面**（先包裝邏輯）、否則 MCP 看到的 signature 不對。

### 4.3 codex 介面

從 codex 視角、不必學新 tool、只是既有寫入 tool 多了 optional `run_at` 參數：

```python
# 立即執行
close_branch(name="鑫盛", branch_code=4)

# 排程到 5/31 23:59:59 執行
close_branch(name="鑫盛", branch_code=4, run_at="2026-05-31T23:59:59",
             note="月底下架 鑫盛/水源-台哥大")
```

回傳值差異：

```python
# 立即執行返回
{"ok": True, "executed": True, "name": "鑫盛", "shop_name": "...", "note": "已軟下架..."}

# 排程返回
{"ok": True, "scheduled": True, "schedule_id": 42, "run_at": "2026-05-31T23:59:59",
 "tool_func": "close_branch", "tool_args": {"name": "鑫盛", "branch_code": 4},
 "note": "月底下架..."}
```

codex 可從 `scheduled: True` vs `executed: True` 判斷該回給使用者什麼（「已排程」vs「已執行」）。

### 4.4 安全 / 邊界

- `run_at` 必須未來時間（decorator 內檢查）、不接過去
- `tool_args` 必須 JSON-serializable（decorator 透過 json.dumps 序列化、會 raise）
- 沒驗 tool_module 命名空間——因為 decorator 在 function 自己的 module 內注入、`func.__module__` 是定義 module、codex 無法手動指定
- worker import 時用 `func.__module__` + `func.__name__`、跟 server 註冊一致、安全

---

## 5. `todo_worker.py` dispatcher 改造

### 5.1 Polling interval：1 小時 → 30 秒

老闆需求 5/31 23:59:59、1 小時粒度誤差最多 1 小時、明顯不夠。

30 秒對 SQLite poll 沒壓力（SELECT WHERE state='pending' AND run_at<=now）、對排程精度夠。

實際 fire 時間：實際排 run_at + 0~30 秒抖動、可接受。

```python
CHECK_INTERVAL = 30  # OLD: 60 * 60 = 1 小時
```

### 5.2 Dispatcher 邏輯

```python
import importlib
import json
from datetime import datetime
import sqlite3

def fetch_pending_due(conn):
    """撈 run_at <= now 且 state='pending' 的 row"""
    now = datetime.now().isoformat(timespec='seconds')
    return conn.execute("""
        SELECT id, tool_module, tool_func, tool_args_json, user_id, user_text, note
        FROM todo_items
        WHERE state='pending' AND run_at <= ?
        ORDER BY run_at
    """, (now,)).fetchall()

def dispatch_one(row):
    """執行一筆排程 row、回傳 result dict / 拋 exception"""
    mod = importlib.import_module(row['tool_module'])
    func = getattr(mod, row['tool_func'])
    args = json.loads(row['tool_args_json'])
    return func(**args)

def mark_executed(conn, id, result):
    conn.execute("""
        UPDATE todo_items
        SET state='executed', executed_at=?, execution_result=?
        WHERE id=?
    """, (datetime.now().isoformat(timespec='seconds'), json.dumps(result, ensure_ascii=False), id))
    conn.commit()

def mark_failed(conn, id, err_msg):
    conn.execute("""
        UPDATE todo_items
        SET state='failed', executed_at=?, execution_result=?
        WHERE id=?
    """, (datetime.now().isoformat(timespec='seconds'), err_msg, id))
    conn.commit()

def run_worker():
    while True:
        try:
            conn = sqlite3.connect(TODO_DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                rows = fetch_pending_due(conn)
                if rows:
                    print(f"[WORKER] {datetime.now()} found {len(rows)} due item(s)")
                for row in rows:
                    print(f"[WORKER] dispatching id={row['id']} {row['tool_func']}({row['tool_args_json']})")
                    try:
                        result = dispatch_one(row)
                        mark_executed(conn, row['id'], result)
                        write_action_notebook(row, result)  # 寫進 notebook 留 audit
                        print(f"[WORKER] id={row['id']} OK")
                    except Exception as e:
                        mark_failed(conn, row['id'], f"{type(e).__name__}: {e}")
                        print(f"[WORKER] id={row['id']} FAILED: {e}")
            finally:
                conn.close()
        except Exception as e:
            print(f"[WORKER] poll loop error: {e}")
        time.sleep(CHECK_INTERVAL)
```

### 5.3 寫進 notebook 留 audit

每次 dispatcher fire 一條、要在 notebook 寫一條 `[ACTION] <tool>(<args>) 已執行（排程觸發、schedule_id=N）`、讓 colombo grep 時看到「這條不是即時人呼叫的、是 5/24 排好的」。

```python
def write_action_notebook(row, result):
    nb_dir = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\notebooks")
    nb_dir.mkdir(exist_ok=True)
    nb = nb_dir / f"notebook_{datetime.now().strftime('%Y%m%d')}.md"
    ts = datetime.now().strftime('%H:%M')
    args_short = row['tool_args_json'][:80]
    line = f"- {ts} [worker:scheduled] [ACTION] {row['tool_func']}({args_short}) 已執行（排程觸發、schedule_id={row['id']}、原排程時間 {row['note']}）\n"
    with open(nb, 'a', encoding='utf-8') as f:
        if not nb.exists() or nb.stat().st_size == 0:
            f.write(f"# Notebook {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(line)
```

---

## 6. AGENTS.md「排程指令流程」段

加進 [`cwsoft-aquan-manager/AGENTS.md`](../../cwsoft-aquan-manager/AGENTS.md)、放在「寫入類確認流程」之後：

### tool 清單**不必新增**

走 B 方案、沒有新 MCP tool 露出給 codex——既有寫入類 tool（adjust_points / close_branch 等）自然支援多一個 optional `run_at` 參數。AGENTS.md tool 清單**保持不變**、只在每個寫入 tool 的描述加註「可加 `run_at` 排程」。

### 新增「排程指令流程」段（草稿）

> ### 排程指令流程（**寫入類 tool 支援的 optional `run_at` 參數**）
>
> 所有寫入類 tool（adjust_points / adjust_sms_points / create_branch / close_branch / generate_quote / generate_perip_quote）**都接受 optional `run_at` 參數**：
>
> - **沒傳** = 立即執行（既有行為）
> - **傳了**（ISO 8601 字串、如 `"2026-05-31T23:59:59"`）= 排程到那時間自動執行
>
> #### 何時該用 run_at
>
> 使用者說「**X 時間做 Y**」這類**有時間延遲**的請求時、傳 run_at、不要當下執行。例：
>
> - 「現在加 3 點」→ `adjust_points(name=X, delta=3)`（沒 run_at）
> - 「月底加 3 點」→ `adjust_points(name=X, delta=3, run_at="2026-05-31T23:59:59")`
> - 「明天早上 9 點下架 Y」→ `close_branch(name=X, branch_code=Y, run_at="2026-05-25T09:00:00")`
>
> #### 流程（跟既有寫入確認流程結合）
>
> 1. 用對應 read tool 確認對象（如 `list_branches` 確認分店編號）
> 2. **時間明確化**——使用者說「月底」、確認是 `2026-05-31T23:59:59` 還是別意？要 confirm
>    - default 「月底」= 該月最後一天 23:59:59
>    - default 「下週一」= 下個禮拜一 09:00:00（上班時間開始）
>    - default 「明天」= 隔天 09:00:00
>    - 不確定一律問
> 3. **動作明確化**——要呼哪個 function、什麼 args、什麼 run_at？阿全自己組好 + 報給使用者看
> 4. **報「即將排程」確認**——「即將排程 2026-05-31 23:59:59 軟下架 鑫盛/水源-台哥大、確認嗎?」+ `[QUICK_REPLY: 確認排程|取消]`
> 5. **等使用者下一 turn 明確同意**才呼 tool 加 run_at
> 6. 結尾加 `[ACTION] close_branch(name=鑫盛, branch_code=4, run_at=2026-05-31T23:59:59) 已排程 schedule_id=N`
>
> #### 禁止
>
> - 跳過時間明確化、把含糊「月底 / 之後 / 過幾天」直接寫進 run_at
> - 同 turn 內既排程又另外即時執行同個動作（變兩筆）
> - run_at 設過去時間（decorator 會拒、不要試）
>
> #### 取消排程
>
> 使用者後悔 → 用 `readonly_query` 查 `todo_items WHERE state='pending'` 找 id、回報 colombo 手動 `UPDATE state='cancelled'`。`cancel_scheduled_action` MCP tool 暫不做、等需求 surface 再加。

---

## 7. 邊角問題 & 設計取捨

### 7.1 排程取消 / 修改

不在 v1 範圍。等真實使用觸發需求再加 `cancel_scheduled_action(schedule_id)` MCP tool。

### 7.2 失敗重試

不做。failure 寫 error、人 grep notebook / `state='failed'` row 介入。

理由：自動重試可能造成「應該只執行一次的動作執行多次」（例：close_branch 兩次、第二次回 already_closed、無害但會留 confusing log）。

### 7.3 觸發時通知

預設**只寫 notebook**、不主動 LINE push。

理由：自動 push 訊息預設有干擾（半夜 fire 一個 push 給老闆）、需要 explicit opt-in。等老闆要求「我要知道執行了」再加。

設計留 hook：`write_action_notebook` 之後加 `if row.notify_user: line_push(user_id, ...)` 一層即可。

### 7.4 機器重啟 / worker 當掉

- todo_list.db 是 persist SQLite、worker 重啟自動 resume
- worker 自己當掉的話、目前無 supervisor、要靠 cwsoft-super-manager 重啟
- 上工時要 verify super-manager 是否管 todo_worker、若沒管要加 entry

### 7.5 跨時區

全用 local time ISO（系統台北時區）、不轉 UTC。cwsoft 單時區、沒跨國協作。

### 7.6 codex session 端的「commitment」與 db 端「真排程」對齊

阿全跟使用者約完 5/31、**同 turn 內必須呼 schedule_action 把 commitment persist**——不能只靠 codex session memory 記。

AGENTS.md 規則要明寫：**「我會在 X 時間做 Y」這類承諾、必須立刻配 schedule_action 寫入 db**。不寫進去等於沒承諾。

### 7.7 兩個 codex（測試版 + 正式版）共用同一個 todo_list.db

OK、因為 todo_list.db 是 process-level 的 storage、worker poll 一份就好。排程是公司級的、不分 OA。

---

## 8. 測試計畫

### Unit

- schedule_action 寫進去 row 對（manual sqlite browser 看）
- worker dispatcher 跑 close_branch 觸發 OK（先排 5 分鐘後執行、驗 dispatch + verify + notebook 一條 龍）

### Integration

- **5/30 約一個近期測試**：5/30 下午用阿全排「5/30 17:00 close_branch(POSV3測試專用, 某編號)」、觀察 17:00 worker 是否真 fire
- **5/31 真實案例**：5/31 老闆要的 close_branch(鑫盛, 4) 在那天觸發

### Audit grep

- notebook 5/31 應該多一條 `[worker:scheduled] [ACTION] close_branch({"name":"鑫盛","branch_code":4}) 已執行（排程觸發、schedule_id=N、原排程時間 月底下架 鑫盛/水源-台哥大）`

---

## 9. 估工

| 步驟 | 估時 |
|---|---|
| dump 既有 todo_items + 評估 migration | 15 分 |
| 改 todo_list.py schema + 加 helper functions | 30 分 |
| 寫 `scheduling.py` 的 `@schedulable` decorator | 30 分 |
| 既有 5 個寫入類 tool 貼 decorator + import 調整 | 20 分 |
| 改 todo_worker.py dispatcher + interval + notebook 寫入 | 1 小時 |
| AGENTS.md 加排程流程段 + 寫入 tool 描述加 run_at 註 + 重啟 Flask | 30 分 |
| 全分身 ping sync（PowerShell iterate sessions）| 10 分 |
| Smoke test（5 分鐘後 fire 測試）| 30 分 |
| worklog day 5 | 30 分 |
| **合計** | **~4 小時** |

## 10. 上工 checklist（5/25 動工照走）

1. ✓ 動工前讀過 5/24 worklog + 本計畫
2. dump `todo_items WHERE state='pending'` → 評估 migration 策略
3. 備份 `cp todo_list.db todo_list.db.bak.<時間戳>`
4. 改 `todo_list.py` schema（DROP + CREATE 新版）
5. 寫 `cwsoft-sqlserver-mcp/src/cwsoft_sqlserver_mcp/scheduling.py`：`@schedulable` decorator + `_insert_scheduled` helper
6. 既有寫入類 tool 貼 decorator：
   - `cwsoft_sqlserver_mcp/server.py`: adjust_points / adjust_sms_points / create_branch / close_branch
   - `tools/cwsoft_ai_tools/server.py`: generate_quote / generate_perip_quote（要 cross-package import scheduling.py）
   - 順序：`@mcp.tool()` 在外、`@schedulable` 在內
7. 改 `todo_worker.py`：dispatcher import+call + notebook 寫入 + CHECK_INTERVAL 30s
8. 改 `cwsoft-aquan-manager/AGENTS.md`：加「排程指令流程」段 + 既有寫入 tool 描述加 run_at 註
9. 確認 todo_worker process 怎麼起（cwsoft-super-manager 管 / 手動）+ 重啟
10. 重啟 Flask（codex MCP 自動 spawn 新版 server.py、不必動）
11. 全分身 ping sync 重讀 AGENTS.md
12. Smoke test：/sim 排一個 5 分鐘後的測試動作（用 POSV3測試專用 DB、close_branch 某分店、帶 run_at）
13. 5 分鐘後 verify：notebook 多一條 [worker:scheduled] [ACTION]、todo_items 該 row state='executed'
14. 寫 day 5 worklog
15. push 兩個 repo

## 11. 風險 / 待解

- **舊 todo_items 內可能有 pending row**：上工先 dump、有的話跟 colombo 對齊處理
- **todo_worker process 是否被 cwsoft-super-manager 管**：要確認、否則改造後沒人重啟它
- **schedule_action 對話層流程的「時間明確化」對 codex 是難點**：使用者說「月底」是 5/31 23:59:59 還是 5/31 00:00:00 還是 5/31 工作時間結束？需要 AGENTS.md 寫清楚 default 解釋 + 一律 confirm
- **未來如果有大量排程（>100）**：30 秒 poll SELECT WHERE state='pending' 對 SQLite 沒壓力、但若 >10000 要加 index on (state, run_at)
- **worker 當掉的 supervisor 機制**：v1 不解、看 cwsoft-super-manager 怎麼處理
