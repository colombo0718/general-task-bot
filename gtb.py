r"""
gtb.py - General Task Bot 正式版

從專案資料夾執行，自動讀取 config/ 與 database/

  cd C:\Users\pos\Desktop\cwsoft-ai-customer-service
  python C:\Users\pos\Desktop\general-task-bot\gtb.py --conf cs --port 6001
  python C:\Users\pos\Desktop\general-task-bot\gtb.py --conf cs --port 6001 --mode shadow

  選項
    --conf   設定檔後綴，如 pos / cs / store，對應 prompts{suffix}.ini / mission{suffix}.json
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
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage, FlexMessage
from linebot.v3.webhook import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
import requests, sqlite3
from huggingface_hub import InferenceClient
from difflib import SequenceMatcher
from pypinyin import lazy_pinyin
import urllib.parse
import configparser
import json
from datetime import datetime
from todo_list import init_todo_db, insert_todo_item, view_todo_list


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


# ===== API 金鑰 =====

HF_TOKEN           = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY")
REMOTE_WORKER_BASE = os.environ.get("REMOTE_WORKER_BASE", "https://worker.leaflune.org")
REMOTE_WORKER_TOKEN = os.environ.get("REMOTE_WORKER_TOKEN")


# ===== 排程資料庫 =====

DB_PATH = os.path.join(PROJECT_DIR, "database", "todo_list.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
init_todo_db(DB_PATH)

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

client = InferenceClient(api_key=HF_TOKEN)

def generate_huggingface_reply(user_text: str, model: str = None, timeout: int = 45) -> str:
    m = model or LLM_MODEL
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": user_text}],
            timeout=timeout
        )
        msg  = completion.choices[0].message
        text = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_openrouter_reply(user_text: str, model: str = None, timeout: int = 45) -> str:
    m = model or LLM_MODEL
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": m, "messages": [{"role": "user", "content": user_text}], "temperature": 0.3},
            timeout=timeout
        )
        r.raise_for_status()
        text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_groq_reply(user_text: str, model: str = None, timeout: int = 60) -> str:
    m = model or LLM_MODEL
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": m, "messages": [{"role": "user", "content": user_text}], "temperature": 0.3},
            timeout=timeout
        )
        r.raise_for_status()
        text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

def generate_remote_worker_reply(user_text: str, model: str = None, timeout: int = 120) -> str:
    m = model or LLM_MODEL
    if not REMOTE_WORKER_TOKEN:
        return "Error"

    try:
        r = requests.post(
            f"{REMOTE_WORKER_BASE}/llm",
            headers={
                "Authorization": f"Bearer {REMOTE_WORKER_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"model": m, "prompt": user_text},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            return "Error"
        text = data.get("response", "") or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

def llm_complete(user_text: str) -> str:
    if PROVIDER == "remote_worker":
        result = generate_remote_worker_reply(user_text, model=LLM_MODEL)
    elif PROVIDER == "groq":
        result = generate_groq_reply(user_text, model=LLM_MODEL)
    elif PROVIDER == "openrouter":
        result = generate_openrouter_reply(user_text, model=LLM_MODEL)
    elif PROVIDER == "hf":
        result = generate_huggingface_reply(user_text, model=LLM_MODEL)
    else:
        result = generate_remote_worker_reply(user_text, model=LLM_MODEL)

    if result == "Error":
        print(f"[LLM] {PROVIDER} failed, fallback → hf")
        from llm_clients import call_provider as _call
        fb = _call("hf", user_text, "meta-llama/Llama-3.3-70B-Instruct")
        return fb.text if fb.ok else "null"

    return result


# ===== 共用小工具 =====

def reply_text(line_bot_api: MessagingApi, reply_token: str, text: str) -> None:
    safe_text = text[:4900]
    line_bot_api.reply_message(
        ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=safe_text)])
    )

reference_content = {
    "today_date": datetime.now().strftime("%Y-%m-%d"),
    "currency": "TWD"
}

def run_extractor(prompt_key: str, user_input: str, ref=None) -> str:
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

def gather_fields(task_id: str, mission_data: dict, user_text: str):
    print("task", task_id)
    task_cfg   = mission_data.get("tasks", {}).get(task_id, {}) or {}
    field_defs = task_cfg.get("fields", {}) or {}
    values, missing = {}, []

    for field_name, meta in field_defs.items():
        pk       = meta.get("prompt_key")
        required = bool(meta.get("required"))
        use_ref  = bool(meta.get("reference"))

        # source=raw 代表直接使用使用者原文，不經過 LLM 萃取
        if meta.get("source") == "raw":
            values[field_name] = user_text
            continue

        val = run_extractor(pk, user_text, reference_content if use_ref else None) if pk else ""
        print(field_name, val)
        values[field_name] = (val or "").strip()

        pool_file = meta.get("match_pool")
        if pool_file and values[field_name] and values[field_name].lower() != "null":
            candidates = read_match_pool(pool_file)
            if candidates:
                # ★ 第一優先：訊息含 pool 裡的客戶名 → 直接強制覆寫，跳過 LLM 不穩定
                #   依長度由長到短匹配，避免短名稱誤覆蓋長名稱（例 POSV3測試專用 vs POSV3）
                hit = None
                for cand in sorted(candidates, key=len, reverse=True):
                    if cand and cand in user_text:
                        hit = cand
                        break
                if hit:
                    values[field_name] = hit
                    print("hardcoded match:", field_name, hit)
                else:
                    best = extract_best_matching_name(values[field_name], candidates, threshold=0.8)
                    if best and best.lower() != "null":
                        values[field_name] = best
            print("new", field_name, values[field_name])

        if required and values[field_name] == "null":
            missing.append(field_name)

    return values, missing, task_cfg

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


todo_command = {}


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
                classify_tree    = mission_data["classify_tree"]
                extracted_intent = run_extractor(classify_tree["prompt_key"], user_text)
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
                values, missing, task_cfg = gather_fields(task_id, mission_data, user_text)

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
            reply_lines = []
            def add(line: str):
                if line:
                    reply_lines.append(line)

            if user_id not in todo_command:
                todo_command[user_id] = {"url": "null", "human_check": False, "run_at": "now", "user_text": ""}

            print("使用者名稱:", display_name)
            print("使用者 UID:", user_id)
            add(f"用戶身分：{display_name}<{user_id}>")
            add(f"待定指令：{todo_command[user_id]['run_at']} 執行 {todo_command[user_id]['url']}")
            add(f"訊息內容：{user_text}")

            human_check     = todo_command[user_id]["human_check"]
            extracted_ronot = run_extractor("extracted_ronot", user_text)
            print("extracted_ronot:", extracted_ronot)

            if extracted_ronot == "true" or (extracted_ronot == "null" and human_check == "auto"):
                add("待辦處理：確認執行")
                if todo_command[user_id]["run_at"] == "now":
                    link, info, _ = execute_command("POST", todo_command[user_id]["url"], user_id)
                    if link: add(f"下載連結：{link}")
                    if info: add(f"執行結果：{info}")
                else:
                    todo_id = insert_todo_item(
                        user_id,
                        todo_command[user_id]["run_at"],
                        todo_command[user_id]["url"],
                        todo_command[user_id]["user_text"],
                        db_path=DB_PATH
                    )
                    add(f"建立排程：#{todo_id} 將於 {todo_command[user_id]['run_at']} 執行 {todo_command[user_id]['url']}")
                todo_command[user_id] = {"url": "null", "human_check": False, "run_at": "now", "user_text": ""}

            elif extracted_ronot == "false" or (extracted_ronot == "null" and human_check == "true"):
                add("待辦處理：取消執行")
                todo_command[user_id] = {"url": "null", "human_check": False, "run_at": "now", "user_text": ""}

            if extracted_ronot != "null":
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
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
                return "OK", 200

            run_at = run_extractor("extract_run_at", user_text, reference_content)
            add(f"執行時間：{run_at}" if run_at != "null" else "執行時間：now")
            if run_at == "null":
                run_at = "now"

            classify_tree    = mission_data["classify_tree"]
            extracted_result = run_extractor(classify_tree["prompt_key"], user_text)
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
                continue

            add(f"任務判定：{task_id}")

            values, missing, task_cfg = gather_fields(task_id, mission_data, user_text)
            add(f"抽取欄位：{values}")

            if missing:
                add(f"缺少欄位：{', '.join(missing)}")
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                continue

            action   = task_cfg.get("action") or {}
            cmd_url, _ = build_command(action, values)
            add(f"最終指令：{cmd_url}")

            human_check = task_cfg.get("human_check", "false").strip().lower()

            if human_check == "true":
                add("執行模式：先詢問確認")
                todo_command[user_id] = {"url": cmd_url, "human_check": "true", "run_at": run_at, "user_text": user_text}
            elif human_check == "false":
                add("執行模式：直接執行")
            elif human_check == "auto":
                add("執行模式：自動確認")
                todo_command[user_id] = {"url": cmd_url, "human_check": "auto", "run_at": run_at, "user_text": user_text}

            link, info, _ = execute_command("GET", cmd_url, user_id)
            if link: add(f"下載連結：{link}")
            if info: add(f"查詢結果：{info}")

            reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))

    return "OK", 200


if __name__ == "__main__":
    should_start_worker = (
        not DEBUG_MODE
        or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    )
    if should_start_worker:
        start_todo_worker(DB_PATH)
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=DEBUG_MODE, use_reloader=False)
