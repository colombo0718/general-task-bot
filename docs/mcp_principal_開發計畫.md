# MCP Principal / Policy / setup_context 身份識別層 開發計畫

> 對應架構文件 [`cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md`](../../../cwsoft-project-tracker/docs/cwsoft-ai-mcp-and-principal-architecture.md) 的 step 3。
>
> 撰寫日期：2026-05-16
> 預估工：1-2 小時（plumbing 都建好了，只剩裝 ACL 機制）

---

## 1. 為什麼要做

目前（step 1+2 完成）：MCP server 有 2 個 tool，**任何 codex session 連上來都能無差別呼叫**。

只有阿全（colombo 自己用）情境沒事；之後 cs_codex.py（客服 LINE bot）共用同一支 MCP server 就漏餡——
LINE 客戶能看到 `list_project_files` 列出 cwsoft 內部專案結構、能讀 README/TODO.md 內容。

step 3 加身份識別層，讓**同一份 MCP 對不同 LINE 來源的呼叫者展現不同能力**，是上正式服務前必須先建好的基礎建設。

---

## 2. 設計原則（複習自架構文件）

1. **「能不能用」由 principal.role / scopes 決定**——查 POLICY 表
2. **「能看誰的」由 principal.tenant_id / db_name 鎖定**——從 principal 拿、不接 caller 帶參數
3. **「具體查什麼」由 caller 函式參數決定**——caller 只能在被授權範圍內細查

---

## 3. 檔案變更清單

### 新增

| 路徑 | 用途 | 預估行數 |
|---|---|---|
| `general-task-bot/tools/cwsoft_ai_tools/policy.py` | POLICY 表 + `can()` helper | ~30 |
| `general-task-bot/tools/cwsoft_ai_tools/principal_state.py` | session_id → principal 對應的 server-side 狀態 | ~20 |
| `cwsoft-aquan-manager/principals.json` | LINE user_id → principal 對應表（先放 colombo owner 一筆）| ~10 |

### 修改

| 路徑 | 變更 |
|---|---|
| `general-task-bot/tools/cwsoft_ai_tools/server.py` | 加 `setup_context` tool；既有 2 個 tool 開頭加 `_check(...)` |
| `general-task-bot/gtb_codex.py` | 加 `lookup_principal()` 函式；每收 LINE 訊息、prepend setup_context 指示給 codex |
| `cwsoft-aquan-manager/AGENTS.md` | 加 hard rule：「**收到使用者訊息第一件事就是呼叫 setup_context**」 |

---

## 4. 詳細設計

### 4.1 `policy.py`

```python
"""
RBAC 政策表：tool_name → 允許的 role 集合。

加新 tool 時，這份表跟 server.py 必須同步更新。
caller 沒在表裡的 role 一律拒。
"""

POLICY = {
    # 唯讀類（owner / staff 共用）
    "list_project_files":   {"owner", "staff"},
    "read_doc":             {"owner", "staff"},

    # setup_context 自身——所有 role 都該能呼，否則 bootstrap 不了
    "setup_context":        {"owner", "staff", "vip_customer", "basic_customer", "anon"},
}


def can(principal: dict | None, tool: str) -> bool:
    """principal 是 setup_context 寫進去的身份卡。"""
    if not principal:
        return False
    role = principal.get("role")
    return role in POLICY.get(tool, set())
```

### 4.2 `principal_state.py`

```python
"""
Server-side session_id → principal dict 對應。
MCP server 是長住程序、同時可能服務多個 codex session、
所以用 session_id 當 key 區隔（不能用全域變數）。

session_id 從哪拿：FastMCP 的 request context。
"""

import threading
from typing import Optional

_session_principals: dict[str, dict] = {}
_lock = threading.Lock()


def set_principal(session_id: str, principal: dict) -> None:
    with _lock:
        _session_principals[session_id] = principal


def get_principal(session_id: str) -> Optional[dict]:
    with _lock:
        return _session_principals.get(session_id)


def clear_principal(session_id: str) -> None:
    with _lock:
        _session_principals.pop(session_id, None)
```

`session_id` 怎麼拿——FastMCP 提供 `Context` 物件、可以從 tool 函式 signature 多接一個 `ctx: Context` 參數。

### 4.3 `server.py` 修改

```python
from mcp.server.fastmcp import FastMCP, Context
from .policy import POLICY, can
from .principal_state import set_principal, get_principal


@mcp.tool()
def setup_context(role: str, tenant_id: str = "", db_name: str = "",
                  scopes: list[str] = None, line_user_id: str = "",
                  ctx: Context = None) -> dict:
    """gtb_codex.py 在每個 LINE 訊息進入 codex 之前，要求 codex 第一件事呼叫這個 tool 載入身份。

    後續 tool 都會用 ctx.session_id 查 principal、檢查能不能呼叫。
    """
    sid = _session_id_from_ctx(ctx)
    principal = {
        "role": role,
        "tenant_id": tenant_id,
        "db_name": db_name,
        "scopes": scopes or [],
        "line_user_id": line_user_id,
        "session_id": sid,
    }
    set_principal(sid, principal)
    return {"ok": True, "loaded_role": role, "session_id": sid}


def _check(tool_name: str, ctx: Context) -> Optional[dict]:
    """每個 tool 進來第一行呼叫；若不允許回 error dict，否則回 None 表示通過。"""
    sid = _session_id_from_ctx(ctx)
    p = get_principal(sid)
    if not can(p, tool_name):
        return {
            "error": "forbidden",
            "tool": tool_name,
            "role": (p or {}).get("role"),
            "hint": "you may need to call setup_context first, or your role lacks permission",
        }
    return None


@mcp.tool()
def list_project_files(ctx: Context = None) -> dict:
    err = _check("list_project_files", ctx)
    if err:
        return err
    # ...原本邏輯
```

### 4.4 `gtb_codex.py` 修改

```python
PRINCIPALS_FILE = PROJECT_DIR / "principals.json"

def lookup_principal(line_user_id: str) -> dict:
    """從 principals.json 查身份；查不到當 anon。"""
    try:
        table = json.loads(PRINCIPALS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        table = {}
    return table.get(line_user_id, {
        "role": "anon", "tenant_id": "", "db_name": "", "scopes": [],
        "line_user_id": line_user_id,
    })


def build_setup_instr(principal: dict, user_text: str) -> str:
    """在使用者訊息前 prepend 一段指示給 codex、要它先呼叫 setup_context。"""
    return (
        "[SYSTEM] 在回答以下使用者訊息之前、請先呼叫 setup_context tool 並帶入這些參數：\n"
        f"  role={principal['role']!r},\n"
        f"  tenant_id={principal['tenant_id']!r},\n"
        f"  db_name={principal.get('db_name','')!r},\n"
        f"  scopes={principal.get('scopes', [])!r},\n"
        f"  line_user_id={principal['line_user_id']!r}\n"
        f"\n[USER] {user_text}"
    )


# /callback handler 改：
#   user_text = event.message.text
#   line_user_id = event.source.user_id
#   principal = lookup_principal(line_user_id)
#   wrapped = build_setup_instr(principal, user_text)
#   reply_text, meta = ask_codex(wrapped)
#   ...
```

### 4.5 `principals.json` v1

```json
{
  "U34e144c9bf7d30bc07c543a4ebae0df1": {
    "role": "owner",
    "tenant_id": "cwsoft",
    "db_name": "POSConfig",
    "scopes": ["all_dbs", "all_chats", "modify_customer", "promote_customer_tier"]
  }
}
```

只放 colombo 一筆。其他人進來預設 `anon`、policy 多半不放行、回 forbidden。

### 4.6 `AGENTS.md` 加 hard rule

在「Self-Report 標記字串」段之前插入新節：

```markdown
## 身份識別 setup（最重要、第一件事）

收到任何使用者訊息、**回答之前的第一件事**：
- 看訊息開頭有沒有 `[SYSTEM] ... setup_context ...` 的指示
- 有的話、立刻呼叫 `setup_context` tool 並帶入指定的 `role` / `tenant_id` 等參數
- 呼叫成功後、再正常處理 [USER] 後面的真實訊息

如果 setup_context 沒呼、後續 MCP tool 都會回 `forbidden`、你會做不了任何事。
```

---

## 5. Driver case：分 4 個情境驗收

預期行為：

| 情境 | 預期 |
|---|---|
| owner 從 LINE 問「列檔」 | codex 呼 setup_context(role=owner) → 呼 list_project_files → 通 → 回檔案清單 |
| vip_customer 從 LINE 問「列檔」 | codex 呼 setup_context(role=vip_customer) → 呼 list_project_files → 回 `{error:forbidden}` → codex 跟使用者說「你沒權限」 + `[WANT_NEW_TOOL]`/說明 |
| owner 從 LINE 問「讀 README」 | setup_context(owner) → read_doc(README.md) → 通 |
| anon 從 LINE 問「讀 README」 | setup_context(anon) → read_doc(README.md) → forbidden |

---

## 6. Smoke checklist

實作完成後逐項驗：

- [ ] `cwsoft_ai_tools.policy.can` 單元測試（owner/staff/vip/basic/anon × 各 tool）
- [ ] MCP server 重啟、direct CLI smoke：呼 setup_context → 呼 list_project_files → 應通
- [ ] 同上、不呼 setup_context 直接呼 list_project_files → 應 forbidden
- [ ] gtb_codex.py 重啟（fresh）、LINE smoke：colombo 帳號問「列檔」→ 通
- [ ] gtb_codex.py 加假 vip_customer 條目進 principals.json，模擬不同 LINE user_id 打 /sim → 應 forbidden
- [ ] 整理一篇 worklog 收尾

---

## 7. 風險 / 待解問題

- **codex 會乖乖呼 setup_context 嗎**：靠 AGENTS.md hard rule 強約束。最壞 codex 跳過 setup → 所有 tool forbidden → 對話完全卡死、會被立刻發現
- **session_id 從 ctx 拿穩定嗎**：FastMCP 文件要查清楚。stdio 一個連線一個 session、應該穩；之後若改 SSE/HTTP MCP 要重新驗
- **principals.json 沒建檔的人**：fallback 給 `anon` role，policy 大多不放、行為一致
- **之後 cs_codex.py 接同一支 MCP**：每個 LINE 客戶都要在 principals.json 登記過、不然只能 anon（這條跟客戶綁定流程要對接 cwsoft accountSystemServer）

---

## 8. 開工順序

1. 先寫 `policy.py` + `principal_state.py`（純資料/邏輯、不接 codex）
2. 改 `server.py` 加 setup_context、原 2 個 tool 加 `_check`
3. 直接從 codex CLI 手動測：呼 setup_context → 呼 list_project_files → 確認通；不呼 setup → 確認 forbidden
4. 寫 `principals.json`（先 colombo owner 一筆）
5. 改 `gtb_codex.py` 的 lookup_principal + build_setup_instr + /callback hook
6. 改 `AGENTS.md` 加 hard rule
7. 重啟 gtb_codex.py（--fresh，新 session 才會重新載 AGENTS.md）
8. LINE 真打、跑驗收 4 case

順序原則：tool side 先做、能直接 codex 手動驗 → 再做 orchestrator side、整合進 LINE flow。
