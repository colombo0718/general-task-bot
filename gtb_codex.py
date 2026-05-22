"""
gtb_codex.py — LINE bot 直串「單一 codex session」的 prototype。

跟 gtb_dev.py 完全不同的架構：
- 啟動時 spawn 一個 codex session、capture session_id（不帶 --ephemeral，要落地）
- 每個 webhook event 用 `codex exec resume <SID>` 把 user_text 餵進去、拿 reply
- 沒有 mission JSON / sqlgate / DB / match_pool / cmd_url——純粹「老闆跟 codex 聊天」
- 全域單一 session：所有 LINE 使用者共用同一條對話脈絡（prototype 階段刻意如此）

驗證目標：codex 的 single-session 在跨 turn 能不能保留記憶（同樣的事不用講第二次）。

啟動：
    python gtb_codex.py --port 6010
    python gtb_codex.py --port 6010 --model gpt-5.4

LINE webhook 設：
    https://<your-tunnel>/callback/@526fdbzo

Dry-run smoke：
    curl -X POST http://localhost:6010/sim/@526fdbzo \
      -H "Content-Type: application/json; charset=utf-8" \
      --data-binary '{"user_text":"今天的密碼是 banana42 你記住"}'
    curl -X POST http://localhost:6010/sim/@526fdbzo \
      -H "Content-Type: application/json; charset=utf-8" \
      --data-binary '{"user_text":"剛剛的密碼是甚麼"}'
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, request, abort, jsonify

from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
    ShowLoadingAnimationRequest,
    QuickReply, QuickReplyItem, MessageAction,
)
from linebot.v3.webhook import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.getcwd()).resolve()  # 從哪個專案目錄起的 → 從那邊讀 oa_registry


# ===== CLI args =====
arg_parser = argparse.ArgumentParser(description="GTB single-session codex bot")
arg_parser.add_argument("--port", type=int, default=6010)
arg_parser.add_argument("--oa-registry", default=None,
                        help="預設讀 PROJECT_DIR/oa_registry.json")
arg_parser.add_argument("--codex-cli", default=os.getenv("CODEX_CLI_PATH", r"D:\npm-global\codex.CMD"))
arg_parser.add_argument("--model", default="gpt-5.4")
arg_parser.add_argument("--reply-timeout", type=int, default=120,
                        help="codex 單次 reply timeout 秒數 (default: 120)")
arg_parser.add_argument("--session-id", default=None,
                        help="顯式指定要 resume 的 codex session_id；不傳則讀 .gtb_codex_session 檔；都沒有才 mint 新的")
arg_parser.add_argument("--fresh", action="store_true",
                        help="忽略持久化的 session、強制 mint 新的")
arg_parser.add_argument("--codex-sandbox", default="danger-full-access",
                        choices=["read-only", "workspace-write", "danger-full-access"],
                        help="codex 沙箱層級。預設 danger-full-access 因為 Windows sandbox + MCP server "
                             "subprocess spawn 在 unelevated 模式下會 DLL init failed (0xC0000142)。"
                             "OS 沙箱關掉但 codex 同時 --disable shell_tool、看不到 shell 也叫不出網路工具，"
                             "實際能力 = 我們寫的 MCP tool 集合（capability-restricted 設計、見 docs/）")
arg_parser.add_argument("--disable-builtin", action="append",
                        default=["shell_tool", "multi_agent", "apps"],
                        help="要對 codex 關掉的內建 feature（重複用此 flag 加多個）。"
                             "預設關 shell_tool（codex 看不到 shell）+ multi_agent（不能開子 agent）。"
                             "github plugin、mempalace MCP 在 ~/.codex/config.toml 全域關掉。"
                             "web.run / apply_patch 沒有 feature flag、靠 boot prompt 約束 codex 不主動使用。")
args = arg_parser.parse_args()


# ===== OA registry =====
oa_reg_path = Path(args.oa_registry) if args.oa_registry else (PROJECT_DIR / "oa_registry.json")
if not oa_reg_path.exists():
    print(f"[BOOT] oa_registry not found: {oa_reg_path}")
    sys.exit(1)
OA_REG = json.loads(oa_reg_path.read_text(encoding="utf-8"))
print(f"[BOOT] OA registry loaded: {oa_reg_path}")
print(f"[BOOT]   {len(OA_REG)} OA: {list(OA_REG.keys())}")


# ===== module-level codex session state =====
# 全域單一 session—所有 LINE 使用者共用、跨 user 訊息也累積到同一條對話。
# 這是 prototype 為了驗證 codex single-session 跨 turn 記憶刻意如此設計。
#
# 持久化策略：
# - 啟動時把 session_id 寫進 PROJECT_DIR/.gtb_codex_session
# - 下次重啟讀那個檔、resume 同一 session（除非 --fresh 或 --session-id 覆寫）
# - /reset endpoint 清檔 + mint 新的
SESSION_ID = None
SESSION_FILE = PROJECT_DIR / ".gtb_codex_session"

# 完整雙方對話寫入：logs/<session_id 前 8 碼>_chat.log
# 一個 session 一個檔；session 換了（/reset 或 --fresh）就會開新檔
# 跟 codex 自己的 ~/.codex/sessions/.../rollout-*-<session_id>.jsonl 用同樣 prefix、好交叉查
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
_CHAT_LOG_LOCK = threading.Lock()

# Self-report 標記攔截：codex 用 [TAG] 標記自己想跟人類說的事、
# 後端在送 LINE 之前把標記行剝掉、按日寫進 notebooks/notebook_YYYYMMDD.md。
# 規範詳見 cwsoft-aquan-manager/AGENTS.md「Self-Report 標記字串」。
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
NOTEBOOK_DIR.mkdir(exist_ok=True)
KNOWN_TAGS = {"KB_FIX", "WANT_NEW_TOOL", "BUG_SIGHTED", "ACTION"}
_NOTEBOOK_LOCK = threading.Lock()


def _chat_log_path() -> Path:
    sid = SESSION_ID or "nosession"
    return LOGS_DIR / f"{sid[:8]}_chat.log"


def split_tags_from_reply(reply: str) -> tuple[str, list[tuple[str, str]]]:
    """從 codex 回覆中分離 [TAG] 行跟一般文字。

    Returns:
        (clean_text, [(tag, payload), ...])
    clean_text 就是要回 LINE 給使用者看的內容（已剝掉 KNOWN_TAGS 行）。
    未知 tag 不剝（codex 可能寫了我們不認識的東西、保留給使用者看比較安全）。
    """
    tags = []
    clean_lines = []
    for line in (reply or "").splitlines():
        stripped = line.strip()
        m = re.match(r"^\[(\w+)\]\s*(.*)$", stripped)
        if m and m.group(1) in KNOWN_TAGS:
            tags.append((m.group(1), m.group(2).strip()))
        else:
            clean_lines.append(line)
    clean = "\n".join(clean_lines).rstrip()
    return clean, tags


# LINE QuickReply 結構化標記攔截：codex 在回覆結尾加 [QUICK_REPLY: 選項1|選項2|...]、
# 後端攔截剝掉、構造 LINE QuickReply payload 附在訊息底部跳按鈕。
# 跟 notebook tags 同型機制、差別在下游：那邊寫檔、這邊組 LINE payload。
QUICKREPLY_PATTERN = re.compile(r"^\s*\[QUICK_REPLY:\s*(.+?)\s*\]\s*$")
QUICKREPLY_MAX_ITEMS = 13  # LINE API 上限
QUICKREPLY_MAX_LABEL = 20  # 每個選項 label 上限


def extract_quickreply_from_reply(reply: str) -> tuple[str, list[str] | None]:
    """從 codex 回覆中分離 [QUICK_REPLY: opt1|opt2|...] 標記行。

    Returns:
        (clean_text, options_list_or_None)
    若有 quickreply 標記、options 是字串 list；無則 None。
    支援 | 跟 , 兩種分隔符（codex 偶爾會用錯）。
    """
    options = None
    clean_lines = []
    for line in (reply or "").splitlines():
        m = QUICKREPLY_PATTERN.match(line.strip())
        if m and options is None:  # 只取第一個、後續忽略
            raw = m.group(1)
            opts = [o.strip() for o in re.split(r"[|,]", raw) if o.strip()]
            if opts:
                # LINE label 上限 20 字、超過截斷
                opts = [o[:QUICKREPLY_MAX_LABEL] for o in opts][:QUICKREPLY_MAX_ITEMS]
                options = opts
                continue
        clean_lines.append(line)
    return "\n".join(clean_lines).rstrip(), options


def build_text_message(text: str, options: list[str] | None) -> TextMessage:
    """組 TextMessage、若 options 不為空就附 QuickReply。"""
    if not options:
        return TextMessage(text=text)
    qr_items = [
        QuickReplyItem(action=MessageAction(label=opt, text=opt))
        for opt in options
    ]
    return TextMessage(text=text, quick_reply=QuickReply(items=qr_items))


def append_to_notebook(tags: list[tuple[str, str]], source: str = "") -> None:
    """把標記寫進當天 notebook。一行一筆、扁平條列。"""
    if not tags:
        return
    now = datetime.now()
    fname = f"notebook_{now.strftime('%Y%m%d')}.md"
    path = NOTEBOOK_DIR / fname
    ts = now.strftime("%H:%M")
    src_part = f"[{source}] " if source else ""
    with _NOTEBOOK_LOCK:
        try:
            new_file = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new_file:
                    f.write(f"# Notebook {now.strftime('%Y-%m-%d')}\n\n")
                for tag, payload in tags:
                    f.write(f"- {ts} {src_part}[{tag}] {payload}\n")
        except Exception as e:
            print(f"[NOTEBOOK] write failed: {e}")


def log_chat_turn(role: str, text: str, source: str = "", elapsed: float | None = None) -> None:
    """
    role:    'user' or 'bot'
    text:    對話內容（可多行）
    source:  LINE user_id 末 6 碼、或 'sim' 等來源標
    elapsed: 秒數（bot 回 reply 時填）
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_bits = [f"[{ts}]", role.upper()]
    if source:
        header_bits.append(source)
    if elapsed is not None:
        header_bits.append(f"({elapsed:.1f}s)")
    header = " ".join(header_bits)

    body = (text or "").rstrip()
    if "\n" in body:
        formatted = f"{header}\n" + "\n".join(f"    {line}" for line in body.split("\n"))
    else:
        formatted = f"{header}: {body}"

    with _CHAT_LOG_LOCK:
        try:
            with open(_chat_log_path(), "a", encoding="utf-8") as f:
                f.write(formatted + "\n\n")
        except Exception as e:
            print(f"[CHAT_LOG] write failed: {e}")

BOOT_SYSTEM_PROMPT = """你是 colombo（cwsoft 的工程師）跟同事的 LINE OA「阿全經理(測式)」背後的 AI。
你的角色是 cwsoft 內部的輔助 AI，最常做的事是：
- 幫使用者查 cwsoft-aquan-manager 專案內的檔案內容
- 之後會有查資料庫的 MCP tool 接上、用來回答 POS 系統相關問題
- 純粹回答問題、不主動下指令動任何東西

【對話原則】
- 你看得到所有先前訊息（單一 session、跨 turn 有記憶）。
- 回覆要簡潔、適合 LINE 對話框（每段不超過 200 字）。
- 純文字、不要表格 / markdown 大量縮排。
- 使用繁體中文、台灣用語。

【你能用的工具】
- 只能用 mcp__cwsoft_ai_tools__* 系列的 MCP tool（目前有 list_project_files / read_doc）。
- **絕對不要呼叫 web.run** ── 不要上網查任何東西。
- **絕對不要呼叫 apply_patch** ── 不要嘗試修改任何檔案。
- 寫入 / 修改類動作（加扣點、改設定、改檔案等）── 之後 colombo 會單獨做成各自的 MCP tool；
  在那之前你沒有對應的 tool 可用，**不要嘗試**走其他路徑去做。

【self-report 標記字串約定】
有以下情境時、在你回覆**結尾**加上對應的單行標記，colombo 上班會 grep 對話 log 處理：

- 你發現對 POS 系統 / cwsoft 知識的理解錯了（使用者糾正你）：
    [KB_FIX] 一句話說明哪裡之前理解錯了

- 使用者要你做一件事、但你沒有對應的 MCP tool 可用：
    [WANT_NEW_TOOL] 一句話描述需要的功能

- 使用者問的問題你完全不知道答案、也沒有相關工具能查：
    [NEED_INFO] 一句話描述缺什麼資訊

- 你發現任何疑似 bug / 異常的東西（檔案內容矛盾、tool 回奇怪結果等）：
    [BUG_SIGHTED] 一句話描述

標記只在當 turn 出現一次、放在回覆最後、自成一行。沒有上述情境就不要加。
你不需要把這些標記讀回來給使用者解釋，他們不關心；標記是給 colombo 後台 grep 用的。

接下來會有一連串使用者訊息進來。"""


def _parse_session_id(stderr: str) -> str | None:
    m = re.search(r"session id:\s*([0-9a-f-]+)", stderr or "", re.IGNORECASE)
    return m.group(1) if m else None


def _parse_codex_reply(stdout: str) -> str:
    """
    codex exec 用 capture_output=True 跑時，**stdout 純粹是最終 reply 文字**——
    所有 metadata（header、`session id`、`codex` 標籤、`tokens used`）都在 **stderr**。
    所以 stdout 直接 strip 拿來用就好，不需要 regex。

    （之前 v1 在這裡寫了 `\\ncodex\\n(.*?)\\ntokens used` 的 regex，是把 `2>&1` 合併過的
    格式錯當 stdout 解，每次都 match fail 然後 fallback 取最後一行——多段 reply 就被截斷。）
    """
    return (stdout or "").strip()


def _load_persisted_sid() -> str | None:
    """從 SESSION_FILE 讀上次保留的 session_id；不存在 / 空檔回 None。"""
    try:
        sid = SESSION_FILE.read_text(encoding="utf-8").strip()
        return sid or None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[BOOT] WARN: read {SESSION_FILE} failed: {e}")
        return None


def _save_sid(sid: str) -> None:
    try:
        SESSION_FILE.write_text(sid, encoding="utf-8")
        print(f"[BOOT] persisted session_id to {SESSION_FILE}")
    except Exception as e:
        print(f"[BOOT] WARN: write {SESSION_FILE} failed: {e}")


def _mint_new_session() -> str | None:
    """跑一次 codex exec + system prompt、回傳新的 session_id。"""
    cmd = [args.codex_cli, "exec", "--skip-git-repo-check",
           "-C", str(PROJECT_DIR)]   # 顯式設 codex workdir → 自動載入 PROJECT_DIR/AGENTS.md
    if args.codex_sandbox == "danger-full-access":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.extend(["-s", args.codex_sandbox])
    for feat in args.disable_builtin:
        cmd.extend(["--disable", feat])
    cmd.extend(["-m", args.model, BOOT_SYSTEM_PROMPT])
    print(f"[BOOT] minting new codex session (model={args.model}, sandbox={args.codex_sandbox}, "
          f"disabled={args.disable_builtin})...")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print(f"[BOOT] FAILED: codex boot timeout 120s")
        return None
    sid = _parse_session_id(proc.stderr or "")
    if not sid:
        print(f"[BOOT] FAILED: cannot parse session id")
        print(f"[BOOT] stderr first 800: {(proc.stderr or '')[:800]}")
        return None
    boot_reply = _parse_codex_reply(proc.stdout)
    print(f"[BOOT] mint OK in {time.time() - t0:.1f}s, reply preview: {boot_reply[:120]!r}")
    return sid


def boot_codex_session():
    """
    Boot 順序:
      1. --session-id 顯式傳入 → 直接用（不寫 SESSION_FILE）
      2. --fresh → mint 新的、寫 SESSION_FILE
      3. SESSION_FILE 有 → resume 同一條（log 印「reusing」）
      4. 都沒有 → mint 新的、寫 SESSION_FILE
    """
    global SESSION_ID

    if args.session_id:
        SESSION_ID = args.session_id.strip()
        print(f"[BOOT] using explicit --session-id = {SESSION_ID}")
        return

    if not args.fresh:
        persisted = _load_persisted_sid()
        if persisted:
            SESSION_ID = persisted
            print(f"[BOOT] reusing persisted session_id = {SESSION_ID}")
            print(f"[BOOT] (從 {SESSION_FILE} 讀的；要 mint 新的請加 --fresh 或 hit /reset)")
            return

    sid = _mint_new_session()
    if not sid:
        sys.exit(1)
    SESSION_ID = sid
    _save_sid(SESSION_ID)
    print(f"[BOOT] codex session_id = {SESSION_ID}")


def reset_codex_session():
    """清掉 SESSION_FILE + mint 新 session。/reset endpoint 用。"""
    global SESSION_ID
    SESSION_ID = None
    try:
        SESSION_FILE.unlink()
        print(f"[RESET] removed {SESSION_FILE}")
    except FileNotFoundError:
        pass
    sid = _mint_new_session()
    if sid:
        SESSION_ID = sid
        _save_sid(SESSION_ID)


def ask_codex(user_text: str) -> tuple[str, dict]:
    """Resume the global session with user_text. Returns (reply_text, meta)."""
    if not SESSION_ID:
        return "（codex 尚未初始化）", {"error": "no session"}
    # exec resume 不收 -s/-a/-C 旗標、要用 --full-auto / --dangerously-bypass-... 切沙箱。
    # workdir 在 mint 時已綁定、resume 自動繼承（同樣自動繼承載入過的 AGENTS.md）。
    cmd = [args.codex_cli, "exec", "resume", "--skip-git-repo-check"]
    if args.codex_sandbox == "workspace-write":
        cmd.append("--full-auto")
    elif args.codex_sandbox == "danger-full-access":
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    # read-only：什麼都不加，繼承原 session 的 sandbox（mint 時設的）
    for feat in args.disable_builtin:
        cmd.extend(["--disable", feat])
    cmd.extend(["-m", args.model, SESSION_ID, user_text])
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.reply_timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return f"（codex 逾時 {args.reply_timeout}s）", {"error": "timeout", "elapsed": elapsed}

    elapsed = time.time() - t0
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:400]
        return f"（codex 錯誤 rc={proc.returncode}）{err}", {"error": "returncode", "elapsed": elapsed}

    reply = _parse_codex_reply(proc.stdout) or "（無回應）"
    return reply, {"elapsed": elapsed, "session_id": SESSION_ID}


# ===== Flask =====
app = Flask(__name__)


@app.route("/callback/<oaid>", methods=["POST"])
def callback(oaid):
    if oaid not in OA_REG:
        abort(404, f"unknown oaid: {oaid}")
    oa = OA_REG[oaid]
    secret = oa["channel_secret"]
    token = oa["channel_access_token"]

    body = request.get_data(as_text=True)
    sig = request.headers.get("X-Line-Signature", "")

    parser_wb = WebhookParser(secret)
    try:
        events = parser_wb.parse(body, sig)
    except InvalidSignatureError:
        abort(400, "invalid signature")

    cfg = Configuration(access_token=token)
    with ApiClient(cfg) as api_client:
        line_bot_api = MessagingApi(api_client)
        for event in events:
            etype = getattr(event, "type", "")
            mtype = getattr(getattr(event, "message", None), "type", "")
            if etype != "message" or mtype != "text":
                continue
            user_text = (event.message.text or "").strip()
            user_id = getattr(event.source, "user_id", "?")
            uid_short = user_id[-6:] if user_id else "?"
            print(f"[WEBHOOK] oa={oaid} user={uid_short} msg={user_text}")
            log_chat_turn("user", user_text, source=f"line:{uid_short}")

            # 在等 codex 回覆期間、給 LINE 對話框「跳動的點點」(loading animation)、避免使用者乾等。
            # API 限制：只支援 1-on-1 user 聊天（group/room 沒效）；loading_seconds 5-60；codex 通常 5-15s。
            src_type = getattr(event.source, "type", "")
            if src_type == "user" and user_id and user_id != "?":
                try:
                    line_bot_api.show_loading_animation(
                        ShowLoadingAnimationRequest(chat_id=user_id, loading_seconds=30)
                    )
                except Exception as e:
                    print(f"[WEBHOOK] show_loading_animation failed (non-fatal): {e}")

            reply_text, meta = ask_codex(user_text)
            elapsed = meta.get("elapsed", 0)
            print(f"[CODEX] reply in {elapsed:.1f}s ({len(reply_text)} chars)")
            # chat log 保留**完整**內容（含標記），給 trace 用
            log_chat_turn("bot", reply_text, source=f"line:{uid_short}", elapsed=elapsed)
            # 攔截 [TAG] 行、寫進 notebooks/notebook_YYYYMMDD.md
            clean_reply, tags = split_tags_from_reply(reply_text)
            if tags:
                print(f"[NOTEBOOK] {len(tags)} tag(s): {[t for t,_ in tags]}")
                append_to_notebook(tags, source=f"line:{uid_short}")
            # 攔截 [QUICK_REPLY: ...] 行、組 LINE QuickReply 按鈕附在訊息底部
            clean_reply, qr_options = extract_quickreply_from_reply(clean_reply)
            if qr_options:
                print(f"[QUICKREPLY] {len(qr_options)} option(s): {qr_options}")

            safe_reply = clean_reply[:4900] if clean_reply else "（codex 沒回應）"
            try:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[build_text_message(safe_reply, qr_options)],
                    )
                )
            except Exception as e:
                print(f"[WEBHOOK] reply_message failed: {e}")

    return "OK"


@app.route("/sim/<oaid>", methods=["POST"])
def sim(oaid):
    """Dry-run backdoor — 不需要 LINE 簽章、純測 codex 對話。"""
    payload = request.get_json(force=True, silent=True) or {}
    user_text = (payload.get("user_text") or "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "user_text empty"}), 400
    log_chat_turn("user", user_text, source="sim")
    reply, meta = ask_codex(user_text)
    log_chat_turn("bot", reply, source="sim", elapsed=meta.get("elapsed"))
    # 同 /callback：攔截 tag + 寫 notebook + 攔截 quickreply、回傳乾淨 reply
    clean_reply, tags = split_tags_from_reply(reply)
    if tags:
        append_to_notebook(tags, source="sim")
    clean_reply, qr_options = extract_quickreply_from_reply(clean_reply)
    return jsonify({
        "ok": True,
        "oaid": oaid,
        "session_id": SESSION_ID,
        "user_text": user_text,
        "reply_clean": clean_reply,             # 跟 LINE 看到的一樣（已剝掉 tag + quickreply）
        "reply_raw": reply,                     # codex 原始輸出（含 tag + quickreply、debug 用）
        "tags": [{"tag": t, "payload": p} for t, p in tags],
        "quick_reply_options": qr_options,      # LINE QuickReply 按鈕選項（沒有就 null）
        "meta": meta,
    })


@app.route("/reset", methods=["POST"])
def reset():
    """重啟 codex session（dev 工具）。"""
    old_sid = SESSION_ID
    reset_codex_session()
    return jsonify({"ok": True, "old_session_id": old_sid, "new_session_id": SESSION_ID})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "session_id": SESSION_ID,
        "model": args.model,
        "oa_count": len(OA_REG),
    })


if __name__ == "__main__":
    boot_codex_session()
    print(f"[BOOT] listening on 0.0.0.0:{args.port}")
    print(f"[BOOT] webhook: POST /callback/<oaid>  (LINE 真打)")
    print(f"[BOOT] sim:     POST /sim/<oaid>       (dry-run)")
    print(f"[BOOT] reset:   POST /reset            (re-boot codex session)")
    print(f"[BOOT] health:  GET /health")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
