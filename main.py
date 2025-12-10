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

import subprocess

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
cfg.read("prompts_system.ini", encoding="utf-8") 
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
            print("new ",field_name,values[field_name])
        # === 新增結束 ===


        if required and values[field_name]=="null":
            missing.append(field_name)
        # print(missing)

    return values, missing, task_cfg


def build_command(action: dict, values: dict):
    """
    回傳：
      cmd_text: 人類可讀且實際會拿去請求的最終 URL
      payload:  {"method": "GET/POST", "url": <final_url>, "body": <dict 或 None>}
    備註：
      - 僅支援 HTTP：所有資料操作一律交給後端 API
      - 不再分 request_url / display_url；就用同一條 final_url
    """
    url_tmpl = action.get("url_template", "")
    method   = (action.get("method", "GET") or "GET").upper()

    used = set()
    def _repl(m):
        k = m.group(1)
        used.add(k)
        return urllib.parse.quote(str(values.get(k, "")), safe="")

    # 先把模板參數代入
    filled = re.sub(r"{(\w+)}", _repl, url_tmpl)

    # 人類可讀版本（同時作為實際請求的 URL 使用）
    final_url = urllib.parse.unquote(filled, encoding="utf-8", errors="replace")

    # 模板沒用到的欄位：GET → 併到 query string；其他 → 當 JSON body
    leftover = {k: v for k, v in values.items() if k not in used}
    if method == "GET" and leftover:
        qs = urllib.parse.urlencode(leftover, doseq=True)
        final_url = final_url + ("&" if "?" in final_url else "?") + urllib.parse.unquote(qs, "utf-8", "replace")
        body = None
    else:
        body = leftover if leftover else None

    # cmd_text 給你印在回覆裡看，payload 給下一步執行
    # return final_url, {"method": method, "url": final_url, "body": body}
    return final_url, {"url": final_url}


def execute_command(method: str, url: str, user_id: str, timeout: float = 10.0):
    """
    執行 HTTP API 指令（user_id 一律以 query 帶上，無 body）：
    - method: "GET" / "POST"（GET=預覽/查詢；POST=實際改動）
    - url:    API URL（不含 user_id）
    - user_id: 會自動附加為 ?user_id=xxx 或 &user_id=xxx
    回傳: (link, info, raw_json_or_text)；期望伺服器回 {"ok":..., "info":..., "link":...}
    """
    # 將 user_id 加到 query
    sep = "&" if "?" in url else "?"
    final_url = f"{url}{sep}user_id={user_id}"

    try:
        resp = requests.request(method.upper(), final_url, timeout=timeout)
        resp.raise_for_status()

        ctype = (resp.headers.get("Content-Type") or "").lower()
        data = resp.json() if "application/json" in ctype else resp.text

        if isinstance(data, dict):
            ok   = bool(data.get("ok", False))
            link = data.get("link")
            info = data.get("info")

            if not ok and info is None and link is None:
                info = data.get("error")

            if isinstance(info, (dict, list)):
                info = json.dumps(info, ensure_ascii=False)

            return link, info, data

        return None, str(data), data

    except Exception as e:
        return None, None, {"ok": False, "error": str(e)}

# ===== 暫存每位使用者的既有指令 =====
todo_command = {}  
# 結構: { user_id: {"url": "<api_url>", "human_check": "true"|"false"|"auto"} }

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, signature)
        print("signature:",signature)
    except InvalidSignatureError:
        abort(400)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        for event in events:
            # 只處理文字訊息
            if getattr(event, "message", None) and getattr(event.message, "text", None):
                user_text = event.message.text.strip()

                # 強插更新客戶列表的小程式，之後想辦法拿掉
                subprocess.run([sys.executable, "generate_customerlist_simple.py"], check=True)
                # print("run generate_customerlist_simple")

                print("message",user_text)

                # 逐段疊加回覆
                reply_lines = []
                def add(line: str):
                    if line:
                        reply_lines.append(line)

                user_id = event.source.user_id

                if user_id not in todo_command:
                    todo_command[user_id] = {"url": "null", "human_check": False}

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
                add(f"待辦指令：{todo_command[user_id]['url']}")
                add(f"訊息內容：{user_text}")

                human_check=todo_command[user_id]['human_check']
                extracted_ronot=run_extractor('extracted_ronot', user_text)
                print("extracted_ronot :",extracted_ronot)

                if extracted_ronot=="true" or  (extracted_ronot=="null" and human_check=='auto'):
                    add(f"待辦處理：確認執行")
                    link, info, raw = execute_command("POST",todo_command[user_id]['url'],user_id)
                    # print(link)
                    todo_command[user_id] = {"url": "null", "human_check": False}
                    if link:
                        add(f"下載連結：{link}")
                    if info:
                        add(f"執行結果：{info}")
                
                elif extracted_ronot=="false" or  (extracted_ronot=="null" and human_check=='true'):
                    add(f"待辦處理：取消待辦")
                    todo_command[user_id] = {"url": "null", "human_check": False}
                    
                
                if not extracted_ronot=="null":
                    reply_text(line_bot_api, event.reply_token, "\n".join(reply_lines))
                    return "OK",200

                # 直接使用 classify_tree 做分類
                # 目前只能做單層分類
                classify_tree = mission_data["classify_tree"]
                extracted_result = run_extractor(classify_tree["prompt_key"], user_text)
                print('extracted_result :',extracted_result)

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
                # action = (task_cfg.get("action") or {"type": "none"})
                action = (task_cfg.get("action") or {})
                # print(action['human_check'])
                
                # add(perform_action(action, values))

                cmd_url, payload = build_command(action, values)
                add(f"構築指令：{cmd_url}")

                human_check=task_cfg.get("human_check").strip().lower()
                print(human_check)

                if human_check=='true':
                    add("處理程序：等待確認，預設取消")
                    todo_command[user_id] = {"url": cmd_url, "human_check":'true'}

                elif human_check=='false': 
                    add("處理程序：直接執行")
                    # link, info, raw = execute_command("GET",cmd_url,user_id)
                    # if link:
                    #     add(f"下載連結：{link}")
                    # if info:
                    #     add(f"查詢結果：{info}")
                elif human_check=='auto':
                    add("處理程序：等待確認，預設執行")
                    todo_command[user_id] = {"url": cmd_url, "human_check":'auto'}

                link, info, raw = execute_command("GET",cmd_url,user_id)
                if link:
                    add(f"下載連結：{link}")
                if info:
                    add(f"查詢結果：{info}")
                

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
