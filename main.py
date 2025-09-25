from flask import Flask, request, abort
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhook import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from dotenv import load_dotenv
import os, requests,sqlite3
# from openai import OpenAI
from huggingface_hub import InferenceClient
from difflib import SequenceMatcher
from pypinyin import lazy_pinyin
import sys, re, os
import urllib.parse
import configparser
import json
from datetime import datetime

# ===== 檔案設定 =====
# # ----- 記帳小幫手 -----
# PROMPTS_FILE = "prompts.ini"
# MISSION_FILE = "mission.json"

# # ----- 全葳小幫手 -----
# PROMPTS_FILE = "prompts_pos.ini"
# MISSION_FILE = "mission_pos.json"

def _resolve_config_pair():
    """
    依 main.py 後面帶的第一個參數決定設定檔名稱。
    例：
      python main.py           -> prompts.ini / mission.json
      python main.py pos       -> prompts_pos.ini / mission_pos.json
    僅允許英數與底線，其他字元會被移除避免路徑風險。
    """
    suffix = ""
    if len(sys.argv) >= 2:
        raw = sys.argv[1]
        safe = re.sub(r"[^A-Za-z0-9_]+", "", raw)
        if safe:
            suffix = f"_{safe}"
    return f"prompts{suffix}.ini", f"mission{suffix}.json"

PROMPTS_FILE, MISSION_FILE = _resolve_config_pair()

print(f"[BOOT] Using config files -> PROMPTS_FILE={PROMPTS_FILE}, MISSION_FILE={MISSION_FILE}")
if not os.path.exists(PROMPTS_FILE):
    raise FileNotFoundError(f"找不到 {PROMPTS_FILE}（請確認檔名或參數是否正確）")
if not os.path.exists(MISSION_FILE):
    raise FileNotFoundError(f"找不到 {MISSION_FILE}（請確認檔名或參數是否正確）")

# ===== LLM 供應商與模型（先用簡單變數，之後再接 .env 也容易）=====
PROVIDER   = "hf"  # 可填: "groq" | "openrouter" | "hf"
# PROVIDER   = "openrouter"  # 可填: "groq" | "openrouter" | "hf"
PROVIDER   = "groq"  # 可填: "groq" | "openrouter" | "hf"

LLM_MODEL = "openai/gpt-oss-20b"  # 依供應商能用的模型填
LLM_MODEL = "openai/gpt-oss-20b"              # 開源 GPT-OSS
# LLM_MODEL = "deepseek-ai/DeepSeek-V3.1"       # DeepSeek Chat V3.1
# LLM_MODEL = "deepseek-ai/DeepSeek-R1"         # DeepSeek Reasoning 模型
# LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Meta LLaMA 3.1 8B Instruct
# LLM_MODEL = "meta-llama/llama-3.1-70b-instruct" # Meta LLaMA 3.1 70B Instruct
# LLM_MODEL = "gemma2-9b-it"                    # Google Gemma 2 9B Instruction
# LLM_MODEL = "qwen/qwen3-32b"                  # 阿里巴巴 Qwen3 32B
# LLM_MODEL = "NousResearch/Hermes-4-70B"       # Hermes-4 (70B)
# LLM_MODEL = "Cohere/command-r-plus"           # Cohere Command-R+
# LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Mistral MoE 模型
# LLM_MODEL = "groq/compound"  # Mistral MoE 模型
LLM_MODEL = "llama-3.1-8b-instant"  # Mistral MoE 模型
LLM_MODEL = "llama-3.3-70b-versatile"  # Mistral MoE 模型


# ===== 載入 prompts.ini =====
cfg = configparser.RawConfigParser()  # Raw 避免 % 插值
cfg.read(PROMPTS_FILE, encoding="utf-8") 
extractors = dict(cfg.items("extractors"))

# ===== 讀取 mission.json =====
with open(MISSION_FILE, "r", encoding="utf-8") as f:
    mission_data = json.load(f)

app = Flask(__name__)


# 讀環境變數
load_dotenv()

channel_secret = os.environ["LINE_CHANNEL_SECRET"]
channel_access_token = os.environ["LINE_CHANNEL_TOKEN"]

HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ===== Line OA 設定（可改用環境變數）=====
# channel_secret = "18bff72eac223d45b1e554d345d24e57"
# channel_access_token = "iN87wB2MJl/y7E45zi24vjbbQpEkXOBTce/B/3HCWkiJnIqLpJYjK4X9XDGhMGUtf7ml77JjTxUHk/H2uFdPrCpjhKF0iP9zLL5ssDDyU7Eyik338Zp6Lhn0PZG0hAsdSFwtEtStoybKO+0ats2nlgdB04t89/1O/w1cDnyilFU="
configuration = Configuration(access_token=channel_access_token)
parser = WebhookParser(channel_secret)

# ====== Hugging Face ======
client = InferenceClient(api_key=HF_TOKEN)
def generate_huggingface_reply(user_text: str, model: str = None, timeout: int = 45) -> str:
    m = model or "openai/gpt-oss-20b"
    try:
        completion = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": user_text}],
            timeout=timeout
        )
        msg = completion.choices[0].message
        text = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

# ====== OpenRouter ======
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
def generate_openrouter_reply(user_text: str, model: str = None, timeout: int = 45) -> str:
    m = model or "openai/gpt-oss-20b"
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": m,
                "messages": [{"role": "user", "content": user_text}],
                "temperature": 0.3,
            },
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"

# ====== Groq ======
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
def generate_groq_reply(user_text: str, model: str = None, timeout: int = 60) -> str:
    m = model or "openai/gpt-oss-20b"
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": m,
                "messages": [{"role": "user", "content": user_text}],
                "temperature": 0.3,
            },
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return text.strip() or "（沒有回覆）"
    except Exception:
        return "Error"


def llm_complete(user_text: str) -> str:
    if PROVIDER == "groq":
        return generate_groq_reply(user_text, model=LLM_MODEL)
    if PROVIDER == "openrouter":
        return generate_openrouter_reply(user_text, model=LLM_MODEL)
    if PROVIDER == "hf":
        return generate_huggingface_reply(user_text, model=LLM_MODEL)
    # 預設
    return generate_groq_reply(user_text, model=LLM_MODEL)

# ===== 封裝：回傳文字給 LINE =====
def reply_text(line_bot_api: MessagingApi, reply_token: str, text: str) -> None:
    # LINE 單則訊息長度上限約 5000 字；保險起見截到 4900
    safe_text = text[:4900]
    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=safe_text)]
        )
    )

# ===== 事前準備好提供給LLM的參考訊息 =====
reference_content = {
    "today_date": datetime.now().strftime("%Y-%m-%d"),
    "currency": "TWD"
}

# ===== 使用LLM萃取單一關鍵資訊
def run_extractor(prompt_key: str, user_input: str, reference_content=None) -> str:
    """
    依據 prompts.ini 的 [extractors] 取出對應提示詞，交給 LLM 回傳結果。
    不做關鍵字腳本、不做額外映射，完全仰賴提示詞的約束。
    """
    # 你前面已有：extractors = dict(cfg.items("extractors"))
    prompt_template = extractors.get(prompt_key, "")
    if not prompt_template:
        return "null"

    # 將 reference_content 正規化成可前置的一段文字
    ref_block = ""
    if reference_content:
        if isinstance(reference_content, dict):
            # 轉成簡單 [REFERENCE] 區塊
            lines = ["[REFERENCE]"]
            for k, v in reference_content.items():
                if isinstance(v, (list, tuple)):
                    v = "、".join(map(str, v))
                lines.append(f"{k}: {v}")
            ref_block = "\n".join(lines).strip()
        elif isinstance(reference_content, str):
            ref_block = reference_content.strip()

    # identify_needs 模板本身有格式約束；把使用者輸入接在後面即可
    parts = []
    if ref_block:
        parts.append(ref_block)
    parts.append(prompt_template.strip())
    parts.append(f"使用者輸入：{user_input or ''}")
    llm_input = "\n\n".join(parts)
    # llm_input = f"{prompt_template.strip()}\n\n使用者輸入：{user_input or ''}"
    # print(llm_input)


    raw = llm_complete(llm_input)   # ← 這裡改成統一入口


    # raw = generate_ollama_reply(llm_input)
    # raw = generate_openrouter_reply(llm_input)
    # raw = generate_groq_reply(llm_input)  # 保持你原本的 LLM 呼叫
    # print(raw)
    # 若回覆中夾帶 <think>…</think>，僅做最小清理
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    return (raw or "").strip()

def gather_fields(task_id: str, mission_data: dict, user_text: str):
    """
    依照 mission.json 的 tasks[task_id].fields
    對每個欄位用其 prompt_key 呼叫 run_extractor 來抽值。
    回傳：
      values: {欄位: 抽到的字串}
      missing: [缺少但 required=True 的欄位名]
      task_cfg: 該 task 的設定（給後續用）
    """
    print('task',task_id)
    tasks = mission_data.get("tasks", {})
    task_cfg = tasks.get(task_id, {}) or {}
    field_defs = task_cfg.get("fields", {}) or {}

    values = {}
    missing = []
    for field_name, meta in field_defs.items():
        pk = meta.get("prompt_key")
        required = bool(meta.get("required")) 

        use_ref = bool(meta.get("reference"))
        
        # val = run_extractor(pk, user_text) if pk else ""
        val = run_extractor(pk, user_text, reference_content if use_ref else None) if pk else ""
        print(field_name,val)
        values[field_name] = (val or "").strip()

        # === 新增：match_pool 後處理（最小改動） ===
        pool_file = meta.get("match_pool")
        if pool_file and values[field_name] and values[field_name].lower() != "null":
            # 讀候選清單（優先使用你現成的 reader；其餘走簡單 txt 逐行）
            if pool_file == "customerlist.txt":
                candidates = read_user_names()
            else:
                try:
                    with open(pool_file, "r", encoding="utf-8") as f:
                        candidates = [line.strip() for line in f if line.strip()]
                except FileNotFoundError:
                    candidates = []

            if candidates:
                # 門檻可先固定 0.8；之後若要可從 meta 讀 match_threshold
                best = extract_best_matching_name(values[field_name], candidates, threshold=0.8)
                if best and best.lower() != "null":
                    values[field_name] = best
        # === 新增結束 ===


        if required and values[field_name]=="null":
            missing.append(field_name)
        # print(missing)

    return values, missing, task_cfg


def build_command(action: dict, values: dict):
    """
    回傳：
      cmd_text: 純粹的指令文字（URL 或 SQL 語法）
      payload:  結構化資訊，後續要執行時可以直接用
    """
    a_type = action.get("type", "none")

    # ---- HTTP ----
    if a_type == "http":
        url_tmpl = action.get("url_template", "")
        method   = (action.get("method", "GET") or "GET").upper()

        used = set()
        def _repl(m):
            k = m.group(1)
            used.add(k)
            return urllib.parse.quote(str(values.get(k, "")), safe="")

        filled = re.sub(r"{(\w+)}", _repl, url_tmpl)
        leftover = {k: v for k, v in values.items() if k not in used}

        # 人類可讀的版本（IRI）
        display_url = urllib.parse.unquote(filled, encoding="utf-8", errors="replace")

        if method == "GET" and leftover:
            qs = urllib.parse.urlencode(leftover, doseq=True)
            display_show = display_url + ("&" if "?" in display_url else "?") + urllib.parse.unquote(qs, "utf-8", "replace")
            request_url  = filled      + ("&" if "?" in filled      else "?") + qs
            body = None
        else:
            display_show = display_url
            request_url  = filled
            body = leftover if leftover else None
        print(display_show)
        print(display_url)
        return display_url, {"type":"http", "method":method, "command":display_url, "body":body}

    # ---- SQL ----
    if a_type == "sql":
        sql_tmpl = action.get("template", "")
        style    = action.get("style", "qmark")
        fetch    = action.get("fetch", "none")

        sql, args = build_sql(sql_tmpl, values, style=style)

        # 產生 debug_sql（參數直接拼進 SQL，僅供驗證使用）
        def escape(v):
            if v is None:
                return "NULL"
            if isinstance(v, (int, float)):
                return str(v)
            return "'" + str(v).replace("'", "''") + "'"

        parts = sql.split("?")
        debug_sql = ""
        for p, a in zip(parts, list(args) + [""]):
            debug_sql += p + (escape(a) if a != "" else "")

        # 回傳雙版本
        return debug_sql, {
            "type": "sql",
            "exec_sql": sql,
            "args": args,
            "command": debug_sql,
            "fetch": fetch
        }

    # ---- NONE ----
    return "null", {"type":"none"}

def execute_simple_command(payload: dict, timeout: float = 10.0):
    """
    目前只處理 HTTP：
    - 以 payload['method'] 與 payload['command'] / ['request_url'] 送出請求
    - 回傳 (quote_path, raw_json 或 None)
    """
    if (payload or {}).get("type") != "http":
        return None, None

    method = (payload.get("method") or "GET").upper()
    # 若你之後在 payload 加了 request_url，就優先用；否則退回 command（你現在放的是 display_url）
    url = payload.get("request_url") or payload.get("command") or ""
    body = payload.get("body")  # 非 GET 時才可能用到

    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            # 夠簡單：用 JSON body；若要 form 可自行改成 data=body
            resp = requests.request(method, url, json=body, timeout=timeout)

        resp.raise_for_status()
        data = resp.json()  # 你的 API 會回 {"ok":true,"quote_path":"..."}
        return data.get("quote_path"), data

    except Exception as e:
        # 失敗就回 None；你也可以把錯誤字串回傳給上層 add()
        return None, {"error": str(e)}


# 小幫手：執行 action 並回傳可疊加的說明文字
def perform_action(action: dict, values: dict) -> str:
    try:
        if action.get("type") == "http":
            
            resp, data, filled_url = call_api(
                action.get("url_template", ""),
                values,
                method=action.get("method", "GET"),
                timeout=action.get("timeout", 15),
            )
            # print('filled_url',filled_url.encode("utf-8", "replace") )

            # filled_url = urllib.parse.unquote(filled_url, encoding="utf-8", errors="replace")

            # 組下載連結（expect_json + link_key）
            if action.get("expect_json") and isinstance(data, dict):
                link_key = action.get("link_key")
                link_base = action.get("link_base", "")
                allow_unicode = bool(action.get("allow_unicode_url"))
                if link_key and data.get(link_key):
                    raw_path = str(data.get(link_key)).lstrip("/")
                    pretty_path = (urllib.parse.unquote(raw_path)
                                    if allow_unicode else urllib.parse.quote(raw_path, safe="/:?&=%#"))
                    full_url = urllib.parse.urljoin(link_base, pretty_path) if link_base else pretty_path
                    return f"構築指令：{filled_url}\n下載連結：{full_url}"
            # 沒有 link_key 就回傳 API 回應
            short = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            return f"構築指令：{filled_url}\nAPI 回應：{short}"
        # 無動作
        # return "構築指令：無動作（action.type = none）"
        return "構築指令：null"
    except Exception as e:
        return f"構築指令：失敗（{e}）"


def call_api(url_template: str, params: dict, *, method: str = "GET", timeout: int = 10):
    """
    將 url_template 中的 {placeholder} 用 params 置換（自動 URL encode），
    剩餘沒用到的 params：GET 放 query string，POST 放 JSON body。
    回傳 (resp, data, filled_url)；data 會在是 JSON 時自動轉 dict。
    """
    used = set()
    print(url_template,params)
    def _repl(m):
        k = m.group(1)
        used.add(k)
        return urllib.parse.quote(str(params.get(k, "")), safe="")  # 只替換模板裡的佔位符

    filled_url = re.sub(r"\{(\w+)\}", _repl, url_template)
    filled_url = urllib.parse.unquote(filled_url, encoding="utf-8", errors="replace")
    print(filled_url)
    leftover = {k: v for k, v in params.items() if k not in used}
    
    if method.upper() == "GET":
        print(filled_url)
        resp = requests.get(filled_url, params=leftover, timeout=timeout)
    elif method.upper() == "POST":
        resp = requests.post(filled_url, json=leftover, timeout=timeout)
    else:
        raise ValueError("Unsupported method: " + method)

    ctype = (resp.headers.get("Content-Type") or "").lower()
    data = resp.json() if "application/json" in ctype else resp.text
    return resp, data, filled_url


def build_sql(sql_template: str, params: dict, *, style: str = "qmark"):
    """
    將 sql_template 中的 {placeholder} 轉成指定參數樣式，並回傳 (sql, args)。
    style:
      - "qmark"    ->  "?"            （sqlite3 / 一些 DB-API）
      - "named"    ->  ":name"        （sqlite3 支援、某些驅動）
      - "pyformat" ->  "%(name)s"     （psycopg2 / MySQLdb 等）
    只會替換模板中實際出現的鍵，並保持參數順序一致。
    """
    order = []  # 按出現順序記錄佔位鍵名

    def _repl(m):
        k = m.group(1)
        order.append(k)
        if style == "qmark":
            return "?"
        elif style == "named":
            return f":{k}"
        elif style == "pyformat":
            return f"%({k})s"
        else:
            raise ValueError("style must be 'qmark' | 'named' | 'pyformat'")

    sql = re.sub(r"{(\w+)}", _repl, sql_template)

    if style == "qmark":
        args = tuple(params[k] for k in order)
    elif style in ("named", "pyformat"):
        args = {k: params[k] for k in order}
    else:
        args = ()

    return sql, args

# 為新加入使用者建立資料庫
def init_user_db(user_id: str, base_dir="databases"):
    # 確保目錄存在
    os.makedirs(base_dir, exist_ok=True)

    # 以 user_id 命名 DB 檔案
    db_path = os.path.join(base_dir, f"{user_id}.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 初始化表結構（這裡先示範記帳用）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT NOT NULL,
        price REAL NOT NULL,
        date TEXT
    );
    """)


# ===== 暫存每位使用者的既有指令 =====
todo_command = {}  
# 結構: { user_id: {"type": "sql/http/none", "command": "實際的指令字串"} }

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        abort(400)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        for event in events:
            # 只處理文字訊息
            if getattr(event, "message", None) and getattr(event.message, "text", None):
                user_text = event.message.text.strip()
                print("message",user_text)

                reply_message='測試'

                # 逐段疊加回覆
                reply_lines = []
                def add(line: str):
                    if line:
                        reply_lines.append(line)

                user_id = event.source.user_id

                # 如果使用者沒有待辦，就初始化一筆
                if user_id not in todo_command:
                    todo_command[user_id] = {"command": "null","type":"none"}

                # 檢查此用戶的資料庫是否存在；沒有就建立
                db_path = os.path.join("databases", f"{user_id}.db")
                if not os.path.exists(db_path):
                    print(f"檢測到新加入用戶，正在建立新資料庫{db_path}")
                    init_user_db(user_id)  # 如果你的 init_db 需要 base_dir 參數：init_db(user_id, base_dir=DB_DIR)
                    # add(f"已為你建立專屬資料庫：{db_path}")
                    
                else:
                    # 可選：debug 顯示目前使用的 DB 路徑
                    print(f"使用資料庫：{db_path}")
                    pass

                # 呼叫 get_profile
                profile = line_bot_api.get_profile(user_id)

                display_name = profile.display_name
                picture_url = profile.picture_url
                status_msg = profile.status_message

                # print("使用者 UID:", user_id)
                print("使用者名稱:", display_name)
                print("使用者 UID:", user_id)
                # print(status_msg)
                add(f"用戶身分：{display_name}<{user_id}>")

                # 把既有的待辦指令也回覆出來
                print("待辦指令：",todo_command[user_id])
                add(f"待辦指令：{todo_command[user_id]['command']}")
                add(f"訊息內容：{user_text}")

                # 直接使用 classify_tree 做分類
                # 目前只能做單層分類
                classify_tree = mission_data["classify_tree"]
                extracted_result = run_extractor(classify_tree["prompt_key"], user_text)

                selected_task_id = None
                fallback_task_id = None
                for branch_rule in classify_tree["branch"]:
                    if branch_rule["match"] == extracted_result:
                        selected_task_id = branch_rule["task_id"]
                        break
                    if branch_rule["match"] == "null":
                        fallback_task_id = branch_rule["task_id"]
                # 找不到分類對應任務也屬於無分類
                if not selected_task_id:
                    selected_task_id = fallback_task_id

                task_id = selected_task_id or fallback_task_id

                

                # reply_message = f"分類完成，進入任務：{selected_task_id}"

                if not task_id:
                    add("任務分類：無法判斷任務。")
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    continue

                add(f"任務分類：{task_id}")

                # 2) 資訊萃取
                values, missing, task_cfg = gather_fields(task_id, mission_data, user_text)
                add(f"資訊萃取：{values}")

                # 缺件直接回覆，不再往下執行
                if missing:
                    add(f"資訊不足，缺少：{', '.join(missing)}")
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    continue

                # 3) 執行指令（action）
                action = (task_cfg.get("action") or {"type": "none"})
                # print(action['human_check'])
                
                # add(perform_action(action, values))

                cmd_text, payload = build_command(action, values)
                add(f"構築指令：{cmd_text}")
                # 順便暫存 payload
                todo_command[user_id] = payload

                if task_cfg.get("human_check") :
                    add(f"處理程序：等待確認")
                else :
                    add(f"處理程序：直接執行")
                    quote_path, raw = execute_simple_command(todo_command[user_id])
                    add(f"下載連結：{quote_path}")
                    todo_command[user_id] = {"command": "null","type":"none"}


                # 回覆至line
                reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))

    return "OK",200

# 讀取用戶列表檔案
def read_user_names():
    try:
        with open('customerlist.txt', 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return []
    
# 將名字轉換為拼音並計算相似度
def calculate_phonetic_similarity(name1, name2):
    # 將名字轉換為拼音
    pinyin1 = ''.join(lazy_pinyin(name1))
    pinyin2 = ''.join(lazy_pinyin(name2))
    # print(pinyin1,pinyin2,SequenceMatcher(None, pinyin1, pinyin2).ratio())
    # 計算拼音之間的相似度
    return SequenceMatcher(None, pinyin1, pinyin2).ratio()

# 計算英文名稱相似度
def calculate_word_similarity(name1, name2):
    # 將名字全部轉換為小寫
    name1 = name1.lower()
    name2 = name2.lower()
    # 計算拼音之間的相似度
    return SequenceMatcher(None, name1, name2).ratio()

# 對混合名稱進行處理
def process_mixed_name(name):
    chinese_part = ''.join(re.findall(r'[\u4e00-\u9fff]+', name)) # 匹配中文字符
    english_part = ''.join(re.findall(r'[a-zA-Z]+', name)) # 匹配英文字符
    return chinese_part, english_part

# 從用戶輸入中提取最相似的用戶名稱
def extract_best_matching_name(keyword, user_names, threshold=0.8):
    if keyword is None:
        print('找不到')
        return None
    best_match = "null"
    highest_similarity = 0.0
    # 遍歷用戶列表
    # print('對比用戶名稱相似度')
    for name in user_names:
        # 將中文及英文部分分開
        chinese_name, english_name = process_mixed_name(name)
        chinese_word, english_word = process_mixed_name(keyword)
        word_similarity = 0.0
        # print('aaa')
        if keyword == name:
            return name
        # 比對中文與英文相似度，相加後要是有相似度較高的就替換
        if chinese_word and chinese_name:
            word_similarity += calculate_phonetic_similarity(chinese_word, chinese_name)
        if english_word and english_name:
            word_similarity += calculate_word_similarity(english_word, english_name)
        # print(keyword,name,word_similarity)
        if word_similarity > highest_similarity and word_similarity >= threshold:
            highest_similarity = word_similarity
            best_match = name
    return best_match



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
