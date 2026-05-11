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

from flask import Flask, request, abort
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


# ===== 啟動參數 =====

parser = argparse.ArgumentParser(description="General Task Bot")
parser.add_argument("--conf", default="", help="config suffix, such as cs / pos / store")
parser.add_argument("--port", default=6000, type=int, help="service port, default 6000")
parser.add_argument("--mode", default="normal", help="run mode: normal / shadow")
args = parser.parse_args()

suffix      = f"_{args.conf}" if args.conf else ""
PROMPTS_FILE = os.path.join(PROJECT_DIR, "config", f"prompts{suffix}.ini")
MISSION_FILE = os.path.join(PROJECT_DIR, "config", f"mission{suffix}.json")
SERVICE_PORT = args.port
SHADOW_MODE  = (args.mode == "shadow")
DEBUG_MODE   = True

print(f"[BOOT] PROJECT_DIR  = {PROJECT_DIR}")
print(f"[BOOT] PROMPTS_FILE = {PROMPTS_FILE}")
print(f"[BOOT] MISSION_FILE = {MISSION_FILE}")
print(f"[BOOT] PORT         = {SERVICE_PORT}")
print(f"[BOOT] MODE         = {'shadow' if SHADOW_MODE else 'normal'}")

if not os.path.exists(PROMPTS_FILE):
    raise FileNotFoundError(f"Missing prompts file: {PROMPTS_FILE}. Check config/ and --conf.")
if not os.path.exists(MISSION_FILE):
    raise FileNotFoundError(f"Missing mission file: {MISSION_FILE}. Check config/ and --conf.")


# ===== LLM 設定 =====

PROVIDER  = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"


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

DB_PATH = os.path.join(PROJECT_DIR, "database", "todo_list.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
init_todo_db(DB_PATH)

# ===== 對話資料庫 =====

CONFIG_DB_PATH = os.path.join(PROJECT_DIR, "database", "config.db")
db_helper.init_config_db(CONFIG_DB_PATH)

# ===== 影子模式資料庫 =====

SHADOW_DB_PATH = os.path.join(PROJECT_DIR, "database", "shadow.db")

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

def read_match_pool(pool_file: str):
    path = os.path.join(PROJECT_DIR, pool_file)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def calculate_phonetic_similarity(name1, name2):
    return SequenceMatcher(None, "".join(lazy_pinyin(name1)), "".join(lazy_pinyin(name2))).ratio()

def calculate_word_similarity(name1, name2):
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

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
                # 分類意圖
                # 分類意圖
                classify_tree    = mission_data["classify_tree"]
                tree_with_conf = bool(classify_tree.get("with_confidence", False))
                extracted_intent_raw = run_extractor(classify_tree["prompt_key"], user_text, with_confidence=tree_with_conf)
                if tree_with_conf:
                    parsed_intent = _parse_value_confidence(extracted_intent_raw)
                    extracted_intent = parsed_intent["value"]
                    print("intent confidence:", parsed_intent["confidence"])
                else:
                    extracted_intent = extracted_intent_raw
                print("intent:", extracted_intent)

                task_id = None
                fallback_id = None
                for branch in classify_tree["branch"]:
                    if branch["match"] == extracted_intent:
                        task_id = branch["task_id"]
                        break
                    if branch["match"] == "null":
                        fallback_id = branch["task_id"]
                task_id = task_id or fallback_id
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
            conv_db      = db_helper.get_or_create_conv_db(CONFIG_DB_PATH, oaid, user_id, display_name, PROJECT_DIR)
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

            classify_tree    = mission_data["classify_tree"]
            tree_with_conf = bool(classify_tree.get("with_confidence", False))
            extracted_result_raw = run_extractor(classify_tree["prompt_key"], user_text, with_confidence=tree_with_conf)
            if tree_with_conf:
                parsed_result = _parse_value_confidence(extracted_result_raw)
                extracted_result = parsed_result["value"]
                add(f"分類信心值：{parsed_result['confidence']}")
            else:
                extracted_result = extracted_result_raw
            print("extracted_result:", extracted_result)

            task_id, fallback_id = None, None
            for branch in classify_tree["branch"]:
                if branch["match"] == extracted_result:
                    task_id = branch["task_id"]
                    break
                if branch["match"] == "null":
                    fallback_id = branch["task_id"]
            task_id = task_id or fallback_id

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


if __name__ == "__main__":
    start_todo_worker(DB_PATH)
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=DEBUG_MODE, use_reloader=False)
