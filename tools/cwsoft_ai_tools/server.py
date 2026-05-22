"""
cwsoft_ai_tools — MVP MCP server。

Step 1 of the plan in docs/cwsoft-ai-mcp-and-principal-architecture.md:
- 30-50 行 + mcp SDK
- 1-2 個工具示範 codex 能 discover 能 call
- 還沒接 principal / policy（那是 step 3）

兩個工具：
    list_project_files()   列 PROJECT_ROOT 一層
    read_doc(name)         讀 PROJECT_ROOT 底下白名單檔案

PROJECT_ROOT 預設指 cwsoft-aquan-manager，可用環境變數 CWSOFT_AI_TOOLS_ROOT 覆寫。

啟動（stdio transport）：
    python -m tools.cwsoft_ai_tools.server
codex 會透過 ~/.codex/config.toml 自動 spawn，不用人手動跑。
"""
import os
from pathlib import Path
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP


PROJECT_ROOT = Path(
    os.getenv("CWSOFT_AI_TOOLS_ROOT", r"C:\Users\pos\Desktop\cwsoft-aquan-manager")
).resolve()

PROJECT_TRACKER_ROOT = Path(
    os.getenv("CWSOFT_PROJECT_TRACKER_ROOT", r"C:\Users\pos\cwsoft-project-tracker")
).resolve()

CS_KB_ROOT = Path(
    os.getenv("CWSOFT_CS_KB_ROOT", r"C:\Users\pos\Desktop\cwsoft-ai-customer-service\kb-customer")
).resolve()

# 客戶名單檔——預設 PROJECT_ROOT/customerlist.txt（= cwsoft-aquan-manager/customerlist.txt）
CUSTOMERLIST_PATH = Path(
    os.getenv("CWSOFT_CUSTOMERLIST_PATH", "")
).resolve() if os.getenv("CWSOFT_CUSTOMERLIST_PATH") else (PROJECT_ROOT / "customerlist.txt")

# autoQuotes API base（產報價單 PDF 用）
QUOTES_API_BASE = os.getenv("CWSOFT_QUOTES_API_BASE", "https://cwsoft.leaflune.org").rstrip("/")
QUOTES_API_TIMEOUT = int(os.getenv("CWSOFT_QUOTES_API_TIMEOUT", "30"))

# 白名單：read_doc 只認得這幾個檔名
DOC_WHITELIST = {
    "README.md", "TODO.md", "WISHLIST.md", "RUNTIME.md",
    "AGENTS.md", "PROJECT.md", "CLAUDE.md", "customerlist.txt",
}

# project-tracker 5 個資料軸：每個對應一個子資料夾 + 允許的副檔名
TRACKER_CATEGORIES = {
    "worklogs": {
        "dir": "worklogs", "exts": [".md"],
        "desc": "每日工作紀錄（claude 撰寫、格式 YYYY-MM-DD-<topic>.md）",
    },
    "meetings": {
        "dir": "meetings", "exts": [".md"],
        "desc": "每週會議紀錄整理（格式 會議記錄_YYYY-MM-DD.md）",
    },
    "minutes": {
        "dir": "minutes", "exts": [".txt"],
        "desc": "會議原始逐字稿（格式 逐字稿-YYYY-MMDD.txt、檔案大、優先讀 meetings 整理）",
    },
    "projects": {
        "dir": "projects", "exts": [".md"],
        "desc": "依專案分類的紀錄頁（檔名用專案中文名、如『小幫手.md』『零壹通訊行_集點系統.md』）",
    },
    "business": {
        "dir": "business", "exts": [".md"],
        "desc": "商談紀錄（格式 商談記錄_YYYY-MM-DD.md）",
    },
}


mcp = FastMCP("cwsoft-ai-tools")


@mcp.tool()
def list_project_files() -> dict:
    """列出目前 cwsoft 專案根目錄底下的檔案與資料夾（一層、不遞迴、不含隱藏項）。

    回傳結構：
        {"root": "<absolute path>", "dirs": [...], "files": [...]}

    用途：codex 想知道專案裡有什麼東西時呼叫。
    """
    if not PROJECT_ROOT.exists():
        return {"error": f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"}
    files, dirs = [], []
    for entry in sorted(PROJECT_ROOT.iterdir()):
        if entry.name.startswith("."):
            continue
        (dirs if entry.is_dir() else files).append(entry.name)
    return {"root": str(PROJECT_ROOT), "dirs": dirs, "files": files}


@mcp.tool()
def read_doc(name: str) -> dict:
    """讀取 PROJECT_ROOT 底下指定文件（限白名單）。

    參數：
        name: 檔名（如 'README.md'）。只接受白名單內的名字、不能含 / 或 \\ 或 ..

    回傳：
        {"name": ..., "size": <chars>, "content": <full text>}  成功
        {"error": "..."}                                         失敗
    """
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "invalid name: path separators are not allowed"}
    if name not in DOC_WHITELIST:
        return {
            "error": f"file '{name}' is not in whitelist",
            "allowed": sorted(DOC_WHITELIST),
        }
    path = PROJECT_ROOT / name
    if not path.exists():
        return {"error": f"file not found: {name}"}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"read failed: {e}"}
    return {"name": name, "size": len(text), "content": text}


@mcp.tool()
def list_kb_docs() -> dict:
    """列 cwsoft-ai-customer-service/kb-engineer/ 底下的工程師知識庫文件清單。

    這些文件是人手整理的 DB schema / SP 邏輯 / 排查場景說明、
    比直接查 DB 更快。建議 codex 想了解某 table/SP 時先讀這些。
    """
    kb = Path(r"C:\Users\pos\Desktop\cwsoft-ai-customer-service\kb-engineer")
    if not kb.exists():
        return {"error": f"kb-engineer dir not found: {kb}"}
    return {
        "kb_dir": str(kb),
        "files": sorted(p.name for p in kb.iterdir() if p.is_file() and p.suffix == ".md"),
    }


@mcp.tool()
def read_kb_doc(name: str) -> dict:
    """讀 cwsoft-ai-customer-service/kb-engineer/ 底下指定 markdown 文件。

    name: 純檔名（如 'POSConfig_資料庫說明.md'）、不能含 / 或 \\ 或 ..
    """
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "invalid name: path separators not allowed"}
    if not name.endswith(".md"):
        return {"error": "only .md files are allowed"}
    kb = Path(r"C:\Users\pos\Desktop\cwsoft-ai-customer-service\kb-engineer")
    path = kb / name
    if not path.exists():
        return {"error": f"not found: {name}"}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"read failed: {e}"}
    return {"name": name, "size": len(text), "content": text}


@mcp.tool()
def list_tracker_docs(category: str) -> dict:
    """列 cwsoft-project-tracker 某類別下的文件清單（檔名按名稱排序）。

    參數 category 必選一個：
        worklogs   每日工作紀錄（claude 撰寫、格式 YYYY-MM-DD-<topic>.md）
        meetings   每週會議紀錄整理（格式 會議記錄_YYYY-MM-DD.md）
        minutes    會議原始逐字稿（格式 逐字稿-YYYY-MMDD.txt、檔案大、優先讀 meetings 整理）
        projects   依專案分類的紀錄頁（用專案中文名）
        business   商談紀錄（格式 商談記錄_YYYY-MM-DD.md）

    回傳：{"category": ..., "dir": ..., "files": [...], "count": N}
    """
    if category not in TRACKER_CATEGORIES:
        return {
            "error": f"unknown category: {category}",
            "allowed": sorted(TRACKER_CATEGORIES),
        }
    cfg = TRACKER_CATEGORIES[category]
    target = PROJECT_TRACKER_ROOT / cfg["dir"]
    if not target.exists():
        return {"error": f"category dir not found: {target}"}
    files = sorted(
        p.name for p in target.iterdir()
        if p.is_file() and p.suffix in cfg["exts"]
    )
    return {
        "category": category,
        "dir": str(target),
        "desc": cfg["desc"],
        "files": files,
        "count": len(files),
    }


@mcp.tool()
def read_tracker_doc(category: str, name: str) -> dict:
    """讀 cwsoft-project-tracker 某類別下指定檔案。

    參數：
        category  worklogs / meetings / minutes / projects / business 五選一
        name      純檔名（含副檔名）、不能含 / \\ ..

    回傳：{"category": ..., "name": ..., "size": <chars>, "content": <full text>}
    """
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "invalid name: path separators not allowed"}
    if category not in TRACKER_CATEGORIES:
        return {
            "error": f"unknown category: {category}",
            "allowed": sorted(TRACKER_CATEGORIES),
        }
    cfg = TRACKER_CATEGORIES[category]
    if not any(name.endswith(ext) for ext in cfg["exts"]):
        return {"error": f"category '{category}' only accepts {cfg['exts']}"}
    path = PROJECT_TRACKER_ROOT / cfg["dir"] / name
    if not path.exists():
        return {"error": f"not found: {category}/{name}"}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"read failed: {e}"}
    return {
        "category": category,
        "name": name,
        "size": len(text),
        "content": text,
    }


@mcp.tool()
def list_cs_kb_docs() -> dict:
    """列 cwsoft-ai-customer-service/kb-customer/ 底下的【客服】知識庫文件清單。

    !! 注意 !!
    這是【未來客服 AI】要用的「對外、客戶看得到」POS 系統 SOP——
    阿全 read 時當「review 客服 SOP 內容」的角度、不要當 cwsoft 內部規範引用、
    也不要把這裡的口吻搬到回答 colombo 的對話裡。
    內部視角請優先用 list_kb_docs / read_kb_doc（工程師知識庫）。
    """
    if not CS_KB_ROOT.exists():
        return {"error": f"cs kb dir not found: {CS_KB_ROOT}"}
    files = sorted(
        p.name for p in CS_KB_ROOT.iterdir()
        if p.is_file() and p.suffix == ".md"
    )
    return {
        "kb_dir": str(CS_KB_ROOT),
        "kind": "customer-facing",
        "note": "面向客戶的 POS SOP；阿全當『review』、不當『規範』",
        "files": files,
        "count": len(files),
    }


def _phonetic_similarity(a: str, b: str) -> float:
    """拼音相似度（substring 命中走 0.95-1.0、否則 SequenceMatcher.ratio()）。"""
    from pypinyin import lazy_pinyin
    from difflib import SequenceMatcher
    pa = "".join(lazy_pinyin(a))
    pb = "".join(lazy_pinyin(b))
    if not pa or not pb:
        return 0.0
    if pa in pb:
        return 0.95 + 0.05 * (len(pa) / max(len(pb), 1))
    if pb in pa:
        return 0.95 + 0.05 * (len(pb) / max(len(pa), 1))
    return SequenceMatcher(None, pa, pb).ratio()


def _literal_similarity(a: str, b: str) -> float:
    """字面相似度（中文字直接比對、不轉拼音）。"""
    from difflib import SequenceMatcher
    if not a or not b:
        return 0.0
    if a in b:
        return 0.95 + 0.05 * (len(a) / max(len(b), 1))
    if b in a:
        return 0.95 + 0.05 * (len(b) / max(len(a), 1))
    return SequenceMatcher(None, a, b).ratio()


def _score_candidate(query: str, candidate: str) -> tuple[float, str]:
    """回 (score, match_type)。match_type ∈ {exact, literal, phonetic}。"""
    if query == candidate:
        return 1.0, "exact"
    lit = _literal_similarity(query, candidate)
    pho = _phonetic_similarity(query, candidate)
    if lit >= pho:
        return lit, "literal"
    return pho, "phonetic"


@mcp.tool()
def match_customer_name(query: str, top_n: int = 5) -> dict:
    """從客戶名單裡找最像的客戶名（拼音優先、處理語音輸入錯字）。

    !! 重要 !!
    使用者大多用 LINE 語音輸入、客戶名很容易聽錯（乙烯/乙希、林一/零壹、想想/相相）。
    任何涉及客戶的 tool（query_points、adjust_points、未來的查訂單 / 出帳單）
    **第一步都必須先呼叫本 tool**、拿真正的客戶名（= DB 名）再去呼後續 tool。

    判讀建議（依 top 第一筆 score）：
        ≥ 0.95  直接用、不再問
        0.7-0.95  跟使用者確認：「你是說 X 嗎?」
        < 0.7    請使用者重打或描述更多

    參數：
        query  使用者給的客戶名（可能語音轉錯字）
        top_n  回傳前 N 個候選（預設 5、上限 20）

    回傳：
        {"query": ..., "total_candidates": N,
         "top": [{"name": ..., "score": 0.xxx, "match_type": "phonetic"}, ...],
         "advice": "..."}
    """
    if not query or not query.strip():
        return {"error": "query 不能空"}
    if not CUSTOMERLIST_PATH.exists():
        return {"error": f"customerlist not found: {CUSTOMERLIST_PATH}"}
    try:
        from pypinyin import lazy_pinyin  # noqa: F401  imported in _phonetic_similarity
    except ImportError as e:
        return {"error": f"pypinyin 未安裝: {e}"}

    capped = max(1, min(int(top_n), 20))
    names = [
        line.strip()
        for line in CUSTOMERLIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for name in names:
        score, mtype = _score_candidate(query.strip(), name)
        if score > 0.3:  # 噪音門檻、過低不回
            rows.append({"name": name, "score": round(score, 3), "match_type": mtype})
    rows.sort(key=lambda r: -r["score"])

    top = rows[:capped]
    if not top:
        advice = "找不到任何 score > 0.3 的客戶、請使用者確認客戶名拼字"
    else:
        s = top[0]["score"]
        if s >= 0.95:
            advice = f"top1 score={s}、可直接使用 name={top[0]['name']!r}、不必再問使用者"
        elif s >= 0.7:
            advice = f"top1 score={s}、信心中等、建議跟使用者確認：「你是說 {top[0]['name']} 嗎?」"
        else:
            advice = f"top1 score={s}、信心不足、請使用者重打或描述更多細節"

    return {
        "query": query,
        "customerlist_path": str(CUSTOMERLIST_PATH),
        "total_candidates": len(names),
        "top": top,
        "advice": advice,
    }


@mcp.tool()
def read_cs_kb_doc(name: str) -> dict:
    """讀 cwsoft-ai-customer-service/kb-customer/ 底下指定 markdown 文件。

    name: 純檔名（如 '1-1_系統安裝與下載.md'）、不能含 / \\ ..
    """
    if "/" in name or "\\" in name or ".." in name:
        return {"error": "invalid name: path separators not allowed"}
    if not name.endswith(".md"):
        return {"error": "only .md files are allowed"}
    path = CS_KB_ROOT / name
    if not path.exists():
        return {"error": f"not found: {name}"}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"error": f"read failed: {e}"}
    return {
        "name": name,
        "kind": "customer-facing",
        "size": len(text),
        "content": text,
    }


@mcp.tool()
def generate_quote(
    name: str,
    charge_months: Optional[int] = None,
    include_tax: Optional[bool] = None,
    due_month: Optional[str] = None,
    unit_price: Optional[int] = None,
) -> dict:
    """產出**月租 POS 服務**報價單 PDF、回傳下載連結（呼叫 autoQuotes /api/quote）。

    !! 重要 !!
    這個是【月租 / 月費】類報價單。週邊產品（紙捲 / 碳帶 / 條碼機）用 `generate_perip_quote`。
    使用者口語常省略類型——「幫 X 出報價單」可能指任一種、不確定先反問。

    呼叫前**請先**：
    - 用 `match_customer_name` 校正客戶名（語音輸入常聽錯）

    參數（除 name 外都可選、不傳則用客戶 JSON 預設）：
        name           客戶名稱（= DB 名）
        charge_months  計費月數（例：6 / 12）
        include_tax    含稅 true / 未稅 false
        due_month      到期月份 `YYYY/MM`（例："2025/11"）
        unit_price     單價（元）

    回傳：
        {"ok": True, "name": ..., "link": "<pdf url>", "params_used": {...}}
        或失敗：{"ok": False, "error": "..."}
    """
    if not name or not name.strip():
        return {"ok": False, "error": "name 不能空"}
    params = {"name": name.strip()}
    if charge_months is not None:
        params["charge_months"] = int(charge_months)
    if include_tax is not None:
        params["price_includes_tax"] = "true" if include_tax else "false"
    if due_month:
        params["due_month"] = str(due_month).strip()
    if unit_price is not None:
        params["unit_price"] = int(unit_price)
    try:
        r = requests.get(f"{QUOTES_API_BASE}/api/quote",
                         params=params, timeout=QUOTES_API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "params_used": params}
    except requests.RequestException as e:
        return {"ok": False, "error": f"request failed: {e}", "params_used": params}
    except ValueError as e:
        return {"ok": False, "error": f"invalid JSON response: {e}", "params_used": params}
    if not data.get("ok"):
        return {"ok": False, "error": data.get("error", "unknown"), "params_used": params}
    return {
        "ok": True,
        "name": name,
        "link": data.get("link") or data.get("quote_path"),
        "params_used": params,
    }


@mcp.tool()
def generate_perip_quote(
    name: str,
    paper: int = 0,
    carbon: int = 0,
    machine: int = 0,
    include_tax: bool = False,
    month: Optional[str] = None,
) -> dict:
    """產出**週邊產品**報價單 PDF（紙捲 / 碳帶 / 條碼機）、回傳下載連結（呼叫 autoQuotes /api/perip）。

    !! 重要 !!
    這個是【週邊商品 / 紙捲 / 碳帶 / 條碼機】類報價單、不是月租。月租用 `generate_quote`。

    呼叫前**請先**用 `match_customer_name` 校正客戶名。

    參數：
        name         客戶名稱（= DB 名）
        paper        橫一刀紙卷 數量（單價 150 元、autoQuotes 端硬編）
        carbon       碳帶 數量（單價 75 元）
        machine      條碼機 TSC ttp-244ce 數量（單價 7800 元）
        include_tax  含稅 true / 未稅 false（預設未稅）
        month        檔名月份 `YYYYMM`（例："202512"）。不傳則用今日

    至少要有一個 paper / carbon / machine > 0、否則 API 會拒絕。
    """
    if not name or not name.strip():
        return {"ok": False, "error": "name 不能空"}
    paper_n = max(0, int(paper or 0))
    carbon_n = max(0, int(carbon or 0))
    machine_n = max(0, int(machine or 0))
    if paper_n + carbon_n + machine_n == 0:
        return {"ok": False, "error": "至少要有一個 paper / carbon / machine 數量大於 0"}
    params = {
        "name": name.strip(),
        "paper": paper_n,
        "carbon": carbon_n,
        "machine": machine_n,
        "price_includes_tax": "true" if include_tax else "false",
    }
    if month:
        params["month"] = str(month).strip()
    try:
        r = requests.get(f"{QUOTES_API_BASE}/api/perip",
                         params=params, timeout=QUOTES_API_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "params_used": params}
    except requests.RequestException as e:
        return {"ok": False, "error": f"request failed: {e}", "params_used": params}
    except ValueError as e:
        return {"ok": False, "error": f"invalid JSON response: {e}", "params_used": params}
    if not data.get("ok"):
        return {"ok": False, "error": data.get("error", "unknown"), "params_used": params}
    return {
        "ok": True,
        "name": name,
        "link": data.get("link"),
        "month": data.get("month"),
        "params_used": params,
    }


if __name__ == "__main__":
    mcp.run()
