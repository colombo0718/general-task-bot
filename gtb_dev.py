"""
gtb_dev.py - General Task Bot 開發版

從專案資料夾執行，自動讀取 config/ 與 database/

  cd C:\\Users\\pos\\Desktop\\cwsoft-ai-customer-service
  python C:\\Users\\pos\\Desktop\\general-task-bot\\gtb.py --conf cs --port 6001
  python C:\\Users\\pos\\Desktop\\general-task-bot\\gtb.py --conf cs --port 6001 --mode shadow
  選項
    --conf   設定檔後綴，如 pos_dev / cs_dev，對應 prompts{suffix}.ini / mission{suffix}.json
    --port   服務 port（預設 6000）
    --mode   執行模式：normal=正常模式，shadow=影子模式
"""


# ===== 基本匯入 =====
import os, sys, re, argparse, threading, time

GTB_DIR     = os.path.dirname(os.path.abspath(__file__))  # general-task-bot/ 所在位置
PROJECT_DIR = os.getcwd()                                 # 目前執行目錄，也就是專案根目錄
# 讓 Python 找得到 general-task-bot/ 裡的共用模組（例如 todo_list）
if GTB_DIR not in sys.path:
    sys.path.insert(0, GTB_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(GTB_DIR, ".env"))  # 從 general-task-bot/.env 載入 API 金鑰

from flask import Flask, request, abort, jsonify
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
    QuickReply, QuickReplyItem, MessageAction,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
import requests, sqlite3
from difflib import SequenceMatcher
from pypinyin import lazy_pinyin
import urllib.parse
import configparser
import json
from datetime import datetime
from todo_list import init_todo_db, insert_todo_item, view_todo_list
import db_helper
from llm_clients import call_provider

# Track B（Agent 模式）用：暴力雙問。
# 預設走 codex_cli + gpt-5.4，但 /sim 可以在 body 帶 agent_provider/agent_model 覆寫，
# 用來實驗「同 LLM 模型走 cascade vs 走 agent 一坨 mega-prompt」哪個準。
AGENT_PROVIDER = "codex_cli"
AGENT_MODEL    = "gpt-5.4"
AGENT_TIMEOUT  = 120


# ===== 啟動參數 =====

parser = argparse.ArgumentParser(description="General Task Bot")
parser.add_argument("--conf", default="", help="config suffix, such as cs / pos / store")
parser.add_argument("--port", default=6000, type=int, help="service port, default 6000")
parser.add_argument("--mode", default="normal", help="run mode: normal / shadow")
parser.add_argument("--db-dir", default="database", help="database directory name under cwd (e.g. database / database_dev)")
args = parser.parse_args()

suffix      = f"_{args.conf}" if args.conf else ""
PROMPTS_FILE = os.path.join(PROJECT_DIR, "config", f"prompts{suffix}.ini")
MISSION_FILE = os.path.join(PROJECT_DIR, "config", f"mission{suffix}.json")
SERVICE_PORT = args.port
SHADOW_MODE  = (args.mode == "shadow")
DB_DIR       = os.path.join(PROJECT_DIR, args.db_dir)
DEBUG_MODE   = True

print(f"[BOOT] PROJECT_DIR  = {PROJECT_DIR}")
print(f"[BOOT] PROMPTS_FILE = {PROMPTS_FILE}")
print(f"[BOOT] MISSION_FILE = {MISSION_FILE}")
print(f"[BOOT] DB_DIR       = {DB_DIR}")
print(f"[BOOT] PORT         = {SERVICE_PORT}")
print(f"[BOOT] MODE         = {'shadow' if SHADOW_MODE else 'normal'}")

if not os.path.exists(PROMPTS_FILE):
    raise FileNotFoundError(f"Missing prompts file: {PROMPTS_FILE}. Check config/ and --conf.")
if not os.path.exists(MISSION_FILE):
    raise FileNotFoundError(f"Missing mission file: {MISSION_FILE}. Check config/ and --conf.")


# ===== LLM 設定 =====

PROVIDER  = "home_ollama"
LLM_MODEL = "qwen3.5:2b"


# ===== 載入 prompts =====

SYSTEM_PROMPTS_FILE = os.path.join(GTB_DIR, "prompts_system.ini")

cfg = configparser.RawConfigParser()
cfg.read(SYSTEM_PROMPTS_FILE, encoding="utf-8")
cfg.read(PROMPTS_FILE, encoding="utf-8")
extractors = dict(cfg.items("extractors"))


# ===== 載入 mission.json =====

with open(MISSION_FILE, "r", encoding="utf-8") as f:
    mission_data = json.load(f)


# ===== Flask =====

app = Flask(__name__)


# ===== 排程資料庫 =====

DB_PATH = os.path.join(DB_DIR, "todo_list.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
init_todo_db(DB_PATH)

# ===== 對話資料庫 =====

CONFIG_DB_PATH = os.path.join(DB_DIR, "config.db")
db_helper.init_config_db(CONFIG_DB_PATH)

# ===== 影子模式資料庫 =====

SHADOW_DB_PATH = os.path.join(DB_DIR, "shadow.db")

def init_shadow_db():
    conn = sqlite3.connect(SHADOW_DB_PATH)
    cur  = conn.cursor()
    # 建立使用者基本資料表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id      TEXT PRIMARY KEY,
            display_name TEXT,
            first_seen   TEXT,
            last_seen    TEXT
        )
    """)
    conn.commit()
    conn.close()

def upsert_user(user_id: str, display_name: str):
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(SHADOW_DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO users (user_id, display_name, first_seen, last_seen)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            display_name = excluded.display_name,
            last_seen    = excluded.last_seen
    """, (user_id, display_name, now, now))
    conn.commit()
    conn.close()

def init_user_message_table(user_id: str):
    """為每個使用者建立自己的訊息紀錄表。"""
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", user_id)
    table   = f"messages_{safe_id}"
    conn    = sqlite3.connect(SHADOW_DB_PATH)
    cur     = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at  TEXT,
            oaid         TEXT,
            user_message TEXT,
            intent       TEXT,
            task_id      TEXT,
            ai_draft     TEXT,
            staff_reply  TEXT
        )
    """)
    conn.commit()
    conn.close()
    return table

def save_shadow_message(user_id: str, oaid: str, user_message: str,
                         intent: str, task_id: str, ai_draft: str):
    table = init_user_message_table(user_id)
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn  = sqlite3.connect(SHADOW_DB_PATH)
    cur   = conn.cursor()
    cur.execute(f"""
        INSERT INTO {table} (received_at, oaid, user_message, intent, task_id, ai_draft)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now, oaid, user_message, intent, task_id, ai_draft or ""))
    conn.commit()
    conn.close()

if SHADOW_MODE:
    init_shadow_db()


# ===== LINE OA registry =====

OA_REGISTRY_PATH = os.path.join(PROJECT_DIR, "oa_registry.json")
with open(OA_REGISTRY_PATH, "r", encoding="utf-8") as f:
    OA_REGISTRY = json.load(f)

def get_oa_secret_token(oaid: str):
    entry = OA_REGISTRY.get(oaid)
    if not entry:
        return None, None
    return entry["channel_secret"], entry["channel_access_token"]


# ===== LLM 呼叫 =====

def llm_complete(user_text: str) -> str:
    result = call_provider(PROVIDER, user_text, LLM_MODEL)
    return result.text if result.ok else "Error"


# ===== 共用小工具 =====

def reply_text(line_bot_api: MessagingApi, reply_token: str, text: str) -> None:
    safe_text = text[:4900]
    line_bot_api.reply_message(
        ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=safe_text)])
    )

def reply_quick_reply(line_bot_api: MessagingApi, reply_token: str, question: str, options: list) -> None:
    items = [
        QuickReplyItem(action=MessageAction(label=opt[:20], text=opt))
        for opt in options[:3]
    ]
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=question, quick_reply=QuickReply(items=items))]
        )
    )

reference_content = {
    "today_date": datetime.now().strftime("%Y-%m-%d"),
    "currency": "TWD"
}

def _pick_format_prompt(with_confidence: bool) -> str:
    if with_confidence:
        return (extractors.get("format_with_confidence") or "").strip()
    return (extractors.get("format_value_only") or "").strip()

def _parse_value_confidence(raw: str):
    text = (raw or "").strip()
    # 預設：向下相容（格式不對也不拋錯）
    fallback = {"value": text or "null", "confidence": 0}
    if not text:
        return {"value": "null", "confidence": 0}
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return fallback
        value = obj.get("value", "null")
        if value is None:
            value = "null"
        value = str(value).strip() or "null"
        conf = obj.get("confidence", 0)
        try:
            conf = int(float(conf))
        except Exception:
            conf = 0
        conf = max(0, min(100, conf))
        return {"value": value, "confidence": conf}
    except Exception:
        return fallback

def run_extractor(prompt_key: str, user_input: str, ref=None, with_confidence: bool = False) -> str:
    prompt_template = extractors.get(prompt_key, "")
    if not prompt_template:
        return "null"

    ref_block = ""
    if ref:
        if isinstance(ref, dict):
            lines = ["[REFERENCE]"]
            for k, v in ref.items():
                if isinstance(v, (list, tuple)):
                    v = "、".join(map(str, v))
                lines.append(f"{k}: {v}")
            ref_block = "\n".join(lines).strip()
        elif isinstance(ref, str):
            ref_block = ref.strip()

    parts = []
    fmt_prompt = _pick_format_prompt(with_confidence)
    if fmt_prompt:
        parts.append(fmt_prompt)
    if ref_block:
        parts.append(ref_block)
    parts.append(prompt_template.strip())
    parts.append(f"使用者訊息：{user_input or ''}")
    raw = llm_complete("\n\n".join(parts))
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    return (raw or "").strip()


def resolve_task_from_tree(tree: dict, user_text: str, intent_chain=None):
    """遞迴解析 classify_tree → (task_id, intent_chain)。

    兼容兩種結構：
    - 平面（cs / 早期 pos）：branch 含 task_id；"null" 是 catch-all
    - 階梯（aquan-dev mission_pos_dev.json）：branch 可含 subtree；"next" 表「不在此層，下一階分」
    LLM 沒命中任何 branch.match 時，fallback 到 "null"/"next" 分支。
    每層各自支援 with_confidence flag。
    """
    if intent_chain is None:
        intent_chain = []

    with_conf = bool(tree.get("with_confidence", False))
    raw       = run_extractor(tree["prompt_key"], user_text, with_confidence=with_conf)
    if with_conf:
        parsed = _parse_value_confidence(raw)
        intent = parsed["value"]
        intent_chain.append({"intent": intent, "confidence": parsed["confidence"]})
    else:
        intent = raw
        intent_chain.append(intent)

    matched_branch  = None
    fallback_branch = None
    for branch in tree["branch"]:
        if branch["match"] == intent:
            matched_branch = branch
            break
        if branch["match"] in ("null", "next"):
            fallback_branch = branch

    chosen = matched_branch or fallback_branch
    if chosen is None:
        return None, intent_chain
    if "subtree" in chosen:
        return resolve_task_from_tree(chosen["subtree"], user_text, intent_chain)
    return chosen.get("task_id"), intent_chain


def read_match_pool(pool_file: str):
    path = os.path.join(PROJECT_DIR, pool_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def calculate_phonetic_similarity(name1, name2):
    """拼音相似度 + substring 高優先
    原本 SequenceMatcher.ratio() = 2*M/(len1+len2)，分母含 candidate 全長 →
    短 keyword 對長 candidate 即使完整命中也會被長度稀釋
    （例：「零壹」 lingyi 對「零壹通訊行」 0.52；對「林一」 linyi 卻 0.91）。
    修法：**keyword pinyin 是 candidate pinyin 的 substring → 直接走 0.95~1.0 高優先**，
    覆蓋音近模糊匹配，因為 substring 完整命中是更強的 match signal。
    """
    p1 = "".join(lazy_pinyin(name1))
    p2 = "".join(lazy_pinyin(name2))
    if p1 and p1 in p2:
        coverage = len(p1) / max(len(p2), 1)
        return 0.95 + 0.05 * coverage
    return SequenceMatcher(None, p1, p2).ratio()

def calculate_word_similarity(name1, name2):
    """英文相似度 + substring 高優先（同 calculate_phonetic_similarity 邏輯）"""
    n1 = name1.lower()
    n2 = name2.lower()
    if n1 and n1 in n2:
        coverage = len(n1) / max(len(n2), 1)
        return 0.95 + 0.05 * coverage
    return SequenceMatcher(None, n1, n2).ratio()

def process_mixed_name(name):
    return ("".join(re.findall(r"[\u4e00-\u9fff]+", name)),
            "".join(re.findall(r"[a-zA-Z]+", name)))

def extract_best_matching_name(keyword, user_names, threshold=0.8):
    if keyword is None:
        return None
    best_match, highest = "null", 0.0
    for name in user_names:
        if keyword == name:
            return name
        cn, en = process_mixed_name(name)
        kc, ke = process_mixed_name(keyword)
        sim = 0.0
        if kc and cn:
            sim += calculate_phonetic_similarity(kc, cn)
        if ke and en:
            sim += calculate_word_similarity(ke, en)
        if sim > highest and sim >= threshold:
            highest, best_match = sim, name
    return best_match

def extract_top_matching_names(keyword, candidates, top_n=3, threshold=0.8):
    """回傳 top-N 候選與信心旗標。
    回傳：([(name, score), ...], is_confident)
    is_confident=True 代表最佳候選分數 >= threshold，可直接採用。
    """
    if not keyword or not candidates:
        return [], False
    scored = []
    for name in candidates:
        if keyword == name:
            return [(name, 1.0)], True
        cn, en = process_mixed_name(name)
        kc, ke = process_mixed_name(keyword)
        sim = 0.0
        if kc and cn:
            sim += calculate_phonetic_similarity(kc, cn)
        if ke and en:
            sim += calculate_word_similarity(ke, en)
        if sim > 0:
            scored.append((name, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]
    is_confident = bool(top and top[0][1] >= threshold)
    return top, is_confident

CONFIDENCE_THRESHOLD = 70  # LLM 信心值低於此分觸發澄清

def gather_fields(task_id: str, mission_data: dict, user_text: str):
    print("task", task_id)
    task_cfg   = mission_data.get("tasks", {}).get(task_id, {}) or {}
    field_defs = task_cfg.get("fields", {}) or {}
    values, missing, field_confidence = {}, [], {}
    clarification_needed = None  # {"field", "raw_input", "top_candidates"}

    for field_name, meta in field_defs.items():
        pk       = meta.get("prompt_key")
        required = bool(meta.get("required"))
        use_ref  = bool(meta.get("reference"))
        with_conf = bool(meta.get("with_confidence", False))

        if meta.get("source") == "raw":
            values[field_name] = user_text
            field_confidence[field_name] = None
            continue

        raw = run_extractor(pk, user_text, reference_content if use_ref else None, with_confidence=with_conf) if pk else ""
        if with_conf:
            parsed = _parse_value_confidence(raw)
            val  = parsed["value"]
            conf = parsed["confidence"]
            field_confidence[field_name] = conf
            print(field_name, val, f"(confidence={conf})")
        else:
            val  = (raw or "").strip()
            conf = 100
            field_confidence[field_name] = None
            print(field_name, val)
        values[field_name] = val

        pool_file = meta.get("match_pool")
        if pool_file and val and val.lower() != "null":
            candidates = read_match_pool(pool_file)
            if candidates:
                top, is_confident = extract_top_matching_names(val, candidates, top_n=3, threshold=0.8)
                llm_uncertain = with_conf and conf < CONFIDENCE_THRESHOLD

                if top and is_confident and not llm_uncertain:
                    # 信心足夠：直接採用最佳比對
                    values[field_name] = top[0][0]
                    print("matched:", field_name, values[field_name])
                elif top and clarification_needed is None:
                    # 信心不足：記錄第一個需要澄清的欄位
                    clarification_needed = {
                        "field": field_name,
                        "raw_input": val,
                        "top_candidates": top,
                    }
                    values[field_name] = top[0][0]  # 暫填最佳猜測
                    print("clarification needed:", field_name, val, "→", [n for n, _ in top])

        if required and values[field_name] == "null":
            missing.append(field_name)

    return values, missing, task_cfg, field_confidence, clarification_needed

# ===== Track B: Agent 模式（暴力雙問） =====
# 原本切碎成 N 個 extractor 餵 LLM 小小鳥的設計，agent 模式整合成 2 個大 prompt 餵 codex 老鷹：
#   Q1 哪個任務 → 把 mission JSON 所有 task description 拼進 context，回 task_id
#   Q2 怎麼做   → 把選中 task 的 fields schema 拼進 context，回 values dict
# 名稱對 customerlist 拼音相似度比對等後處理留在 Python 層。

def _parse_agent_json(text: str, key: str, default):
    """寬鬆 JSON parse：去掉 ``` fence、抓第一個 {...}、取指定 key。失敗回 default。"""
    if not text:
        return default
    s = text.strip()
    # 去 markdown code fence
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    # 抓第一個 {...} 區塊（含巢狀，最簡單抓法：最外層 brace）
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return default
    try:
        obj = json.loads(m.group(0))
        return obj.get(key, default)
    except Exception:
        return default

def _build_q1_classify_prompt(user_text: str) -> str:
    """Q1: 拼出「哪個任務」mega-prompt — 列出所有 task_id + description。"""
    lines = []
    for tid, tcfg in mission_data.get("tasks", {}).items():
        desc = (tcfg.get("description") or "").strip()
        lines.append(f"- {tid}: {desc}")
    today = reference_content.get("today_date", "")
    return (
        "你是任務分類器。從以下任務清單中，選一個最匹配使用者訊息的 task_id。\n\n"
        "[可用任務]\n"
        + "\n".join(lines)
        + f"\n\n[使用者訊息]\n{user_text}\n\n"
        f"[reference]\ntoday_date: {today}\n\n"
        "只輸出一行 JSON，不要任何前後說明或 markdown fence：\n"
        '{"task_id":"<從清單中選一個 task_id>"}\n'
        '若都不像，輸出 {"task_id":"fallback_greeting"}。\n'
    )

def _build_q2_extract_prompt(task_id: str, user_text: str) -> str:
    """Q2: 拼出「怎麼做」mega-prompt — 列出該 task 的所有欄位定義。"""
    task_cfg   = mission_data["tasks"].get(task_id) or {}
    fields_cfg = task_cfg.get("fields", {}) or {}
    today      = reference_content.get("today_date", "")
    if not fields_cfg:
        return ""
    lines = []
    for fname, fmeta in fields_cfg.items():
        req       = "必填" if fmeta.get("required") else "選填"
        pool      = fmeta.get("match_pool")
        ref       = bool(fmeta.get("reference"))
        hints = []
        if pool:
            hints.append(f"客戶名稱（之後系統會用 {pool} 做拼音模糊比對）")
        if ref:
            hints.append("可參考 today_date")
        suffix = f"（{' / '.join(hints)}）" if hints else ""
        lines.append(f"- {fname}（{req}）{suffix}")
    return (
        f"從使用者訊息中，抽出 task_id={task_id} 需要的所有欄位。\n\n"
        f"[任務說明]\n{task_cfg.get('description', '')}\n\n"
        "[欄位清單]\n"
        + "\n".join(lines)
        + f"\n\n[使用者訊息]\n{user_text}\n\n"
        f"[reference]\ntoday_date: {today}\n\n"
        "只輸出一行 JSON，不要任何前後說明或 markdown fence：\n"
        '{"values":{"field_name":"value", ...}}\n'
        '無法從訊息抽出的欄位填字串 "null"。'
    )

def agent_classify_and_extract(user_text: str, provider_override: str = None, model_override: str = None):
    """Track B 主流程：兩個 LLM 呼叫 + 名稱 match_pool 後處理。

    預設 provider/model 走 AGENT_PROVIDER / AGENT_MODEL（codex_cli + gpt-5.4）。
    可在 /sim body 帶 agent_provider / agent_model 覆寫，方便實驗
    「同 LLM 模型走 cascade vs 走 agent 一坨 mega-prompt」哪個準。

    Returns:
        task_id, values, missing, errors(list), debug(dict)
    """
    provider = provider_override or AGENT_PROVIDER
    model    = model_override    or AGENT_MODEL
    errors = []
    debug = {"provider": provider, "model": model, "q1": {}, "q2": {}}

    # reasoning 模型走 agent flow 時打開 think（mega-prompt 是它的舒適圈）；
    # **非 reasoning 模型送 think=True 會被 Ollama 直接 400**（gemma3/qwen2.5/llama3 等）；
    # think=False 才是普世安全。所以只對 reasoning 模型白名單開 think。
    model_l = (model or "").lower()
    think_capable = any(s in model_l for s in ("qwen3", "deepseek-r1", "gpt-oss"))
    agent_think = think_capable

    # ----- Q1: 分類 -----
    q1_prompt = _build_q1_classify_prompt(user_text)
    r1 = call_provider(provider, q1_prompt, model, timeout=AGENT_TIMEOUT, think=agent_think)
    debug["q1"] = {"ok": r1.ok, "latency_sec": round(r1.latency_sec, 2), "raw": r1.text[:300], "error": r1.error[:200]}
    if not r1.ok:
        errors.append(f"agent Q1 failed: {r1.error[:200]}")
        return None, {}, [], errors, debug

    task_id = _parse_agent_json(r1.text, "task_id", default="fallback_greeting")
    if task_id not in mission_data.get("tasks", {}):
        errors.append(f"agent Q1 returned unknown task_id: {task_id}")
        task_id = "fallback_greeting"

    task_cfg   = mission_data["tasks"][task_id]
    fields_cfg = task_cfg.get("fields", {}) or {}

    # 沒欄位的 task（如 fallback_greeting / refresh_companies）直接收工
    if not fields_cfg:
        return task_id, {}, [], errors, debug

    # ----- Q2: 抽欄位 -----
    q2_prompt = _build_q2_extract_prompt(task_id, user_text)
    r2 = call_provider(provider, q2_prompt, model, timeout=AGENT_TIMEOUT, think=agent_think)
    debug["q2"] = {"ok": r2.ok, "latency_sec": round(r2.latency_sec, 2), "raw": r2.text[:300], "error": r2.error[:200]}
    if not r2.ok:
        errors.append(f"agent Q2 failed: {r2.error[:200]}")
        return task_id, {}, [], errors, debug

    raw_values = _parse_agent_json(r2.text, "values", default={})
    if not isinstance(raw_values, dict):
        raw_values = {}

    # ----- 後處理：name match_pool（拼音相似度）+ missing 判定 -----
    customer_pool = read_match_pool("customerlist.txt")
    values, missing = {}, []
    for fname, fmeta in fields_cfg.items():
        raw = str(raw_values.get(fname, "null")).strip() or "null"
        if fmeta.get("match_pool") == "customerlist.txt" and raw and raw != "null":
            matched = extract_best_matching_name(raw, customer_pool, threshold=0.8)
            values[fname] = matched if matched and matched != "null" else raw
        else:
            values[fname] = raw
        if fmeta.get("required") and (values[fname] == "null" or not values[fname]):
            missing.append(fname)

    return task_id, values, missing, errors, debug


def build_command(action: dict, values: dict):
    url_tmpl = action.get("url_template", "")
    method   = (action.get("method", "GET") or "GET").upper()
    used     = set()

    def _repl(m):
        k = m.group(1)
        used.add(k)
        return urllib.parse.quote(str(values.get(k, "")), safe="")

    filled    = re.sub(r"\{(\w+)\}", _repl, url_tmpl)
    final_url = urllib.parse.unquote(filled, encoding="utf-8", errors="replace")

    leftover = {k: v for k, v in values.items() if k not in used}
    if method == "GET" and leftover:
        qs = urllib.parse.urlencode(leftover, doseq=True)
        final_url += ("&" if "?" in final_url else "?") + urllib.parse.unquote(qs, "utf-8", "replace")

    return final_url, {"url": final_url}

def execute_command(method: str, url: str, user_id: str, timeout: float = 10.0):
    sep       = "&" if "?" in url else "?"
    final_url = f"{url}{sep}user_id={user_id}"
    try:
        resp  = requests.request(method.upper(), final_url, timeout=timeout)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        data  = resp.json() if "application/json" in ctype else resp.text
        if isinstance(data, dict):
            link = data.get("link")
            info = data.get("info")
            if not data.get("ok") and info is None and link is None:
                info = data.get("error")
            if isinstance(info, (dict, list)):
                info = json.dumps(info, ensure_ascii=False)
            return link, info, data
        return None, str(data), data
    except Exception as e:
        return None, None, {"ok": False, "error": str(e)}


# ===== 內嵌 todo worker =====

TODO_CHECK_INTERVAL = 60

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def run_todo_worker_loop(db_path: str, interval_seconds: int = TODO_CHECK_INTERVAL):
    print(f"[TODO_WORKER] started db={db_path} interval={interval_seconds}s")

    while True:
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT id, user_id, url
                FROM todo_items
                WHERE state = 'pending'
                  AND run_at <= ?
                ORDER BY id ASC
            """, (now_iso(),))
            rows = cur.fetchall()

            if rows:
                print(f"[TODO_WORKER] found {len(rows)} task(s)")

            for task_id, user_id, url in rows:
                print(f"[TODO_WORKER] executing id={task_id}")
                _, _, raw = execute_command("POST", url, user_id, timeout=30.0)
                ok = bool(isinstance(raw, dict) and raw.get("ok"))

                if ok:
                    cur.execute("""
                        UPDATE todo_items
                        SET state = 'done'
                        WHERE id = ?
                    """, (task_id,))
                    conn.commit()
                    print(f"[TODO_WORKER] done id={task_id}")
                else:
                    err = raw.get("error") if isinstance(raw, dict) else str(raw)
                    print(f"[TODO_WORKER] failed id={task_id} error={err}")
        except Exception as e:
            print(f"[TODO_WORKER] loop error: {e}")
        finally:
            if conn is not None:
                conn.close()

        time.sleep(interval_seconds)

def start_todo_worker(db_path: str, interval_seconds: int = TODO_CHECK_INTERVAL):
    worker = threading.Thread(
        target=run_todo_worker_loop,
        args=(db_path, interval_seconds),
        name="gtb-todo-worker",
        daemon=True,
    )
    worker.start()
    return worker




# ===== LINE Webhook =====

@app.route("/callback/<oaid>", methods=["POST"])
def callback(oaid):
    signature = request.headers.get("X-Line-Signature", "")
    body      = request.get_data(as_text=True)

    channel_secret, channel_access_token = get_oa_secret_token(oaid)
    if not channel_secret:
        abort(404)

    parser_wb     = WebhookParser(channel_secret)
    configuration = Configuration(access_token=channel_access_token)

    try:
        events = parser_wb.parse(body, signature)
    except InvalidSignatureError:
        abort(400)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        for event in events:
            if not getattr(event, "message", None):
                continue
            msg_type = getattr(event.message, "type", "")
            if msg_type not in ("text", "image"):
                continue

            user_text = event.message.text.strip() if msg_type == "text" else "[圖片]"
            user_id   = event.source.user_id

            # 讀取 LINE 使用者資料（相容 1-on-1 / group / room）
            source_type = getattr(event.source, "type", "user")
            try:
                if source_type == "group":
                    profile = line_bot_api.get_group_member_profile(event.source.group_id, user_id)
                elif source_type == "room":
                    profile = line_bot_api.get_room_member_profile(event.source.room_id, user_id)
                else:
                    profile = line_bot_api.get_profile(user_id)
                display_name = profile.display_name
            except Exception:
                display_name = user_id

            print(f"[{args.mode.upper()}] oaid={oaid} user={display_name} msg={user_text}")
            # 傳送 LINE loading indicator（僅一般模式）

            if not SHADOW_MODE:
                try:
                    requests.post(
                        "https://api.line.me/v2/bot/chat/loading/start",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {channel_access_token}"
                        },
                        json={"chatId": user_id, "loadingSeconds": 20},
                        timeout=5
                    )
                except Exception:
                    pass  # loading 失敗不中斷主流程


            # ── 影子模式 ──────────────────────────────────────────────────────
            if SHADOW_MODE:
                upsert_user(user_id, display_name)
                # 分類意圖（支援階梯式 subtree 遞迴）
                task_id, intent_chain = resolve_task_from_tree(mission_data["classify_tree"], user_text)
                last_entry = intent_chain[-1] if intent_chain else ""
                extracted_intent = last_entry["intent"] if isinstance(last_entry, dict) else last_entry
                print("intent_chain:", intent_chain, "→ task_id:", task_id)
                # 抽取欄位
                values, missing, task_cfg, field_confidence, _ = gather_fields(task_id, mission_data, user_text)
                print("field_confidence:", field_confidence)
                # 生成 AI 草稿
                ai_draft = ""
                if not missing:
                    action  = task_cfg.get("action") or {}
                    cmd_url, _ = build_command(action, values)
                    _, info, _ = execute_command("GET", cmd_url, user_id)
                    ai_draft   = info or ""
                # 儲存影子訊息
                save_shadow_message(
                    user_id=user_id,
                    oaid=oaid,
                    user_message=user_text,
                    intent=extracted_intent,
                    task_id=task_id or "",
                    ai_draft=ai_draft
                )
                print(f"[SHADOW] 已儲存，ai_draft={ai_draft[:50]}...")
                continue  # 不回覆 LINE
            # ── 一般模式 ──────────────────────────────────────────────────────
            # ── 一般模式 ──────────────────────────────────────────────────────
            conv_db      = db_helper.get_or_create_conv_db(CONFIG_DB_PATH, oaid, user_id, display_name, DB_DIR)
            content_type = "image" if msg_type == "image" else "text"

            reply_lines = []
            def add(line: str):
                if line:
                    reply_lines.append(line)

            print("使用者名稱:", display_name)
            print("使用者 UID:", user_id)

            # ── 澄清分支：等待使用者選擇客戶名稱 ──────────────────────────────
            pending_clar = db_helper.get_pending_clarification(conv_db)
            if pending_clar:
                session_id = pending_clar["session_id"]
                db_helper.log_message(conv_db, session_id, "in", user_text, content_type, "answer")

                options   = pending_clar["options"]
                ctx       = pending_clar["context"]
                raw_input = ctx.get("raw_input", "")
                field     = pending_clar["field_name"]

                stripped = user_text.strip()
                if stripped in ("1", "2", "3"):
                    idx = int(stripped) - 1
                    resolved = options[idx] if idx < len(options) else stripped
                elif stripped in options:
                    resolved = stripped
                else:
                    resolved = stripped  # 使用者自由輸入，直接採用

                db_helper.resolve_clarification(conv_db, pending_clar["id"], user_text, resolved)
                db_helper.save_correction(conv_db, field, raw_input, resolved)

                values   = ctx.get("values", {})
                task_id  = ctx.get("task_id")
                run_at   = ctx.get("run_at", "now")
                values[field] = resolved
                task_cfg = mission_data.get("tasks", {}).get(task_id, {}) or {}

                add(f"已確認：{field} = {resolved}")
                action     = task_cfg.get("action") or {}
                cmd_url, _ = build_command(action, values)
                add(f"最終指令：{cmd_url}")

                human_check_val = (task_cfg.get("human_check") or "false").strip().lower()
                if human_check_val in ("true", "auto"):
                    add(f"執行模式：{'先詢問確認' if human_check_val == 'true' else '自動確認'}")
                    db_helper.save_decision(conv_db, session_id, task_id, values, cmd_url, run_at,
                                            state="awaiting_confirmation")
                else:
                    add("執行模式：直接執行")
                    db_helper.save_decision(conv_db, session_id, task_id, values, cmd_url, run_at,
                                            state="queued")

                link, info, _ = execute_command("GET", cmd_url, user_id)
                if link: add(f"下載連結：{link}")
                if info: add(f"查詢結果：{info}")

                out_role = "confirm_request" if human_check_val in ("true", "auto") else "result"
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", out_role)
                continue

            # ── 新的 session ───────────────────────────────────────────────────
            session_id = db_helper.new_session_id(user_id)
            db_helper.log_message(conv_db, session_id, "in", user_text, content_type, "claim")

            add(f"用戶身分：{display_name}<{user_id}>")
            add(f"訊息內容：{user_text}")

            # ── 決議確認分支（human_check 等待回覆）─────────────────────────
            pending_dec = db_helper.get_pending_decision(conv_db)
            if pending_dec:
                add(f"待定指令：{pending_dec['run_at']} 執行 {pending_dec['url']}")
                dec_task_cfg    = mission_data.get("tasks", {}).get(pending_dec["task_id"], {}) or {}
                human_check     = (dec_task_cfg.get("human_check") or "false").strip().lower()
                extracted_ronot = run_extractor("extracted_ronot", user_text)
                print("extracted_ronot:", extracted_ronot)

                if extracted_ronot == "true" or (extracted_ronot == "null" and human_check == "auto"):
                    add("待辦處理：確認執行")
                    if pending_dec["run_at"] == "now":
                        link, info, _ = execute_command("POST", pending_dec["url"], user_id)
                        db_helper.update_decision_state(conv_db, pending_dec["id"], "executed")
                        if link: add(f"下載連結：{link}")
                        if info: add(f"執行結果：{info}")
                    else:
                        todo_id = insert_todo_item(
                            user_id, pending_dec["run_at"], pending_dec["url"],
                            pending_dec["url"], db_path=DB_PATH
                        )
                        db_helper.update_decision_state(conv_db, pending_dec["id"], "queued", todo_ref_id=todo_id)
                        add(f"建立排程：#{todo_id} 將於 {pending_dec['run_at']} 執行")
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "result")
                    return "OK", 200

                elif extracted_ronot == "false" or (extracted_ronot == "null" and human_check == "true"):
                    add("待辦處理：取消執行")
                    db_helper.update_decision_state(conv_db, pending_dec["id"], "cancelled")
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "result")
                    return "OK", 200

                if extracted_ronot != "null":
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "result")
                    return "OK", 200

            view_todos = run_extractor("extract_view_todos", user_text)
            if view_todos == "true":
                rows = view_todo_list(user_id, db_path=DB_PATH)
                add("待辦清單：")
                for _id, txt in (rows or []):
                    add(f"#{_id} {txt}")
                if not rows:
                    add("目前沒有待辦項目")
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "result")
                return "OK", 200

            run_at = run_extractor("extract_run_at", user_text, reference_content)
            add(f"執行時間：{run_at}" if run_at != "null" else "執行時間：now")
            if run_at == "null":
                run_at = "now"

            # 分類意圖（支援階梯式 subtree 遞迴）
            task_id, intent_chain = resolve_task_from_tree(mission_data["classify_tree"], user_text)
            add(f"分類過程：{intent_chain}")
            print("intent_chain:", intent_chain, "→ task_id:", task_id)

            if not task_id:
                add("無法判斷對應任務")
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "error")
                continue

            add(f"任務判定：{task_id}")

            values, missing, task_cfg, field_confidence, clarification_needed = gather_fields(task_id, mission_data, user_text)
            add(f"抽取欄位：{values}")
            add(f"欄位信心值：{field_confidence}")

            if missing:
                add(f"缺少欄位：{', '.join(missing)}")
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", "error")
                continue

            # 信心不足 → 存到 clarification_log，發 Quick Reply
            if clarification_needed:
                field     = clarification_needed["field"]
                raw_input = clarification_needed["raw_input"]
                options   = [name for name, _ in clarification_needed["top_candidates"]]
                conf_in   = field_confidence.get(field)
                question  = f"請問你說的「{raw_input}」是哪一間公司？\n請回覆 1、2 或 3，或直接輸入正確名稱"
                context   = {"task_id": task_id, "values": values, "run_at": run_at, "raw_input": raw_input}
                db_helper.save_clarification(conv_db, session_id, field, question, options,
                                             confidence_in=conf_in, context=context)
                reply_quick_reply(line_bot_api, event.reply_token, question, options)
                db_helper.log_message(conv_db, session_id, "out", question, "text", "clarify")
                continue

            action     = task_cfg.get("action") or {}
            cmd_url, _ = build_command(action, values)
            add(f"最終指令：{cmd_url}")

            human_check = (task_cfg.get("human_check") or "false").strip().lower()

            if human_check == "true":
                add("執行模式：先詢問確認")
                db_helper.save_decision(conv_db, session_id, task_id, values, cmd_url, run_at,
                                        state="awaiting_confirmation")
            elif human_check == "false":
                add("執行模式：直接執行")
                db_helper.save_decision(conv_db, session_id, task_id, values, cmd_url, run_at,
                                        state="queued")
            elif human_check == "auto":
                add("執行模式：自動確認")
                db_helper.save_decision(conv_db, session_id, task_id, values, cmd_url, run_at,
                                        state="awaiting_confirmation")

            link, info, _ = execute_command("GET", cmd_url, user_id)
            if link: add(f"下載連結：{link}")
            if info: add(f"查詢結果：{info}")

            out_role = "confirm_request" if human_check in ("true", "auto") else "result"
            reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
            db_helper.log_message(conv_db, session_id, "out", "\n".join(reply_lines), "text", out_role)

    return "OK", 200


# ============================================================================
# 後門：壓測 / 老闆歷史考題用 dry-run 入口
# ============================================================================
# 設計原則：兩邊互動都是假的
#   向上對 LINE OA：不驗 signature、不打任何 LINE API（含 reply / push / loading）
#   向下對 downstream API：不真打 sqlgate / autoQuotes 等服務，只記錄會打哪支
# 後門只測：chatbot 跟人訊息互動的應答對不對 —— 能不能問出正確的問題、組出
# 正確的 API 動作（cmd_url）。實際 SQL / 報價單副作用不在測試範圍。

@app.route("/sim/<oaid>", methods=["POST"])
def sim(oaid):
    """模擬 LINE webhook event 的後門 dry-run 入口（壓測 / 考題用）。

    Payload (JSON):
        user_id      必填，模擬使用者 ID
        user_text    必填，模擬使用者訊息
        display_name 選填，預設 = user_id
        source_type  選填，user / group / room（預設 user，目前僅 echo）
        msg_type     選填，text / image（預設 text，僅 echo）

    回傳：
        ok / oaid / user_id / user_text
        intent           classify_tree 解出的最終（leaf-level）intent
        intent_chain     遞迴解 tree 的 intent 路徑（階梯式可見每層判斷；平面結構只有一個）
        task_id          對應 task
        values           gather_fields 抽出的欄位
        missing          缺的 required 欄位（若非空 → 「會問使用者補哪些」）
        field_confidence 各欄位信心值（mission 有開 with_confidence 才會有值）
        cmd_url          build_command 組出的 URL（「會打哪支 API」記錄，不真打）
        timing_ms        整支 handler 耗時（不含 client↔server 網路）
        errors           錯誤列表（若有）
    """
    t0 = time.time()
    result = {
        "ok": True,
        "oaid": oaid,
        "user_id": None,
        "user_text": "",
        "intent": None,
        "intent_chain": [],
        "task_id": None,
        "values": {},
        "missing": [],
        "field_confidence": {},
        "cmd_url": None,
        "errors": [],
    }
    try:
        payload = request.get_json(force=True, silent=True) or {}
        user_id      = payload.get("user_id") or "Usim_default"
        display_name = payload.get("display_name") or user_id
        user_text    = (payload.get("user_text") or "").strip()
        result["user_id"]   = user_id
        result["user_text"] = user_text

        if not user_text:
            result["ok"] = False
            result["errors"].append("user_text is empty")
            result["timing_ms"] = int((time.time() - t0) * 1000)
            return jsonify(result), 400

        mode = (payload.get("mode") or "llm").strip().lower()
        print(f"[SIM] oaid={oaid} user={display_name} mode={mode} msg={user_text}")

        # ----- Agent 模式：暴力雙問（Q1 task_id / Q2 values）→ 不走 cascade 也不走切碎 extractor -----
        if mode == "agent":
            agent_provider = (payload.get("agent_provider") or "").strip() or None
            agent_model    = (payload.get("agent_model") or "").strip() or None
            task_id, values, missing, agent_errors, agent_debug = agent_classify_and_extract(
                user_text, provider_override=agent_provider, model_override=agent_model
            )
            result["mode"]             = "agent"
            result["task_id"]          = task_id
            result["intent"]           = task_id  # agent 模式沒有 cascade，最終 intent == task_id
            result["intent_chain"]     = [task_id] if task_id else []
            result["values"]           = values
            result["missing"]          = missing
            result["field_confidence"] = {}
            result["agent_debug"]      = agent_debug
            if agent_errors:
                result["errors"].extend(agent_errors)
            if task_id and not missing:
                action = (mission_data["tasks"].get(task_id, {}).get("action") or {})
                cmd_url, _ = build_command(action, values)
                result["cmd_url"] = cmd_url
            result["timing_ms"] = int((time.time() - t0) * 1000)
            return jsonify(result), 200

        # ----- LLM 模式（原本路徑）：cascade tier 分類 + 切碎 extractor 抽欄位 -----
        result["mode"] = "llm"
        # 分類意圖（支援階梯式 subtree 遞迴；與 shadow / normal 一致）
        task_id, intent_chain = resolve_task_from_tree(mission_data["classify_tree"], user_text)
        last_entry = intent_chain[-1] if intent_chain else ""
        extracted_intent = last_entry["intent"] if isinstance(last_entry, dict) else last_entry
        result["intent"]       = extracted_intent
        result["intent_chain"] = intent_chain
        result["task_id"]      = task_id

        if not task_id:
            result["errors"].append("no task_id resolved")
            result["timing_ms"] = int((time.time() - t0) * 1000)
            return jsonify(result), 200

        # 抽欄位
        values, missing, task_cfg, field_confidence, _ = gather_fields(task_id, mission_data, user_text)
        result["values"]           = values
        result["missing"]          = missing
        result["field_confidence"] = field_confidence

        # 組 cmd_url（記錄「會打哪支 API」但不真打 — 向下對 downstream 也是假的）
        if not missing:
            action     = task_cfg.get("action") or {}
            cmd_url, _ = build_command(action, values)
            result["cmd_url"] = cmd_url

    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    result["timing_ms"] = int((time.time() - t0) * 1000)
    return jsonify(result), 200


if __name__ == "__main__":
    start_todo_worker(DB_PATH)
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=DEBUG_MODE, use_reloader=False)
