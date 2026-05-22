"""
GTB 共用的 LLM / agent backend 呼叫層。

這個檔案的角色：
- 把各家供應商或 CLI 的呼叫細節，從 GTB 主流程抽離。
- 對外統一提供 `call_provider(...)` 介面。
- 不論底層是 API、HTTP worker、或 agent CLI，最後都回傳同一個 `LLMResult`。
- 讓 benchmark / 偵錯腳本 / 未來 GTB 主流程都能用相同方式切換後端。

------------------------------------------------------------------------------
目前支援的後端類型
------------------------------------------------------------------------------

1. API 型（直接打雲端模型 API）
- `groq`
- `openrouter`
- `hf`

2. SSH tunnel 型（透過 SSH 進私網機器跑本地推理服務）
- `home_ollama`

3. agent CLI 型（透過命令列工具呼叫 agent）
- `codex_cli`
- `gemini_cli`

------------------------------------------------------------------------------
目前支援的 provider 與預設 benchmark model
------------------------------------------------------------------------------

- `groq`
  - 類型：API
  - benchmark 預設模型：`openai/gpt-oss-20b`

- `openrouter`
  - 類型：API
  - benchmark 預設模型：`openai/gpt-4o-mini`

- `hf`
  - 類型：API
  - benchmark 預設模型：`meta-llama/Llama-3.1-8B-Instruct`

- `home_ollama`
  - 類型：SSH tunnel + Ollama
  - benchmark 預設模型：`gemma3:4b`（home 機器 RTX 2060 上跑，~70 tok/s）

- `codex_cli`
  - 類型：agent CLI
  - benchmark 預設模型：`gpt-5.4`

- `gemini_cli`
  - 類型：agent CLI
  - benchmark 預設模型：`gemini-2.5-pro`

------------------------------------------------------------------------------
設計提醒
------------------------------------------------------------------------------

- 這層只做「呼叫」與「結果標準化」，不處理 GTB 任務邏輯。
- `model` 參數是通用欄位：
  - 對 API 型是模型名稱
  - 對 HTTP worker 型是遠端 worker 收到的模型名稱
  - 對 CLI 型則通常是 CLI 的 `--model`
- `codex_cli` 這類 agent backend 雖然也走相同介面，但輸出風格、延遲、
  穩定性不一定與傳統 completion API 相同，後續仍需靠 benchmark 驗證。
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


@dataclass
class LLMResult:
    # provider/model 用來標示這次實際呼叫的端點與模型。
    provider: str
    model: str
    # ok=True 代表請求成功且通過 provider 專屬成功條件。
    ok: bool
    text: str
    latency_sec: float
    # status_code:
    # - API / HTTP worker 通常放 HTTP status
    # - CLI provider 則放 process return code
    status_code: Optional[int] = None
    error: str = ""
    # raw 保留 provider 專屬資訊，方便 benchmark 或 debug 時追查。
    raw: Optional[Dict[str, Any]] = None


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _resolve_cli_path(env_key: str, default_name: str, windows_fallbacks: Optional[list[str]] = None) -> str:
    """
    解析 CLI 可執行檔路徑。

    優先順序：
    1. 明確指定的環境變數（例如 CODEX_CLI_PATH / GEMINI_CLI_PATH）
    2. PATH 中可找到的指令
    3. Windows 常見 npm global 安裝位置 fallback
    """
    configured = os.getenv(env_key, "").strip()
    if configured:
        return configured

    found = shutil.which(default_name)
    if found:
        return found

    for candidate in windows_fallbacks or []:
        if os.path.exists(candidate):
            return candidate

    return default_name


def _success(provider: str, model: str, text: str, t0: float, status_code: Optional[int] = None, raw: Optional[Dict[str, Any]] = None) -> LLMResult:
    """建立統一格式的成功結果。"""
    return LLMResult(
        provider=provider,
        model=model,
        ok=True,
        text=(text or "").strip(),
        latency_sec=time.time() - t0,
        status_code=status_code,
        raw=raw,
    )


def _failure(provider: str, model: str, t0: float, error: str, status_code: Optional[int] = None, raw: Optional[Dict[str, Any]] = None) -> LLMResult:
    """建立統一格式的失敗結果。"""
    return LLMResult(
        provider=provider,
        model=model,
        ok=False,
        text="",
        latency_sec=time.time() - t0,
        status_code=status_code,
        error=error,
        raw=raw,
    )


def call_groq(prompt: str, model: str, timeout: int = 60, max_retries: int = 4) -> LLMResult:
    """呼叫 Groq（OpenAI 相容 chat API）。

    遇 429 / 500 系列短暫錯誤會做指數退避重試（預設最多 4 次，最長等 60s）。
    Retry-After header 優先採用（Groq 會主動回提示）。

    max_retries=0 → fail-fast，只跑 1 次嘗試。適用於外層已有 provider fallback 的情境
    （例如 kbcs 的 groq → hf → home_ollama 鏈），避免在單一 provider 內等過長。
    """
    provider = "groq"
    api_key = os.getenv("GROQ_API_KEY")
    t0 = time.time()
    if not api_key:
        return _failure(provider, model, t0, "GROQ_API_KEY missing")
    last_status, last_body = None, ""
    base_delays = [1.0, 2.0, 5.0, 12.0]
    delays = base_delays[:max(0, max_retries)]   # 累積最長 ~20s 等待
    for attempt in range(len(delays) + 1):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                timeout=timeout,
            )
            status = r.status_code
            if status == 200:
                data = r.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                return _success(provider, model, text, t0, status_code=status, raw=data)
            last_status, last_body = status, (r.text or "")[:800]
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt >= len(delays):
                return _failure(provider, model, t0, f"http {status}", status_code=status, raw={"body": last_body, "attempts": attempt + 1})
            # Retry-After 優先；否則用預設退避
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_for = max(0.5, min(60.0, float(retry_after)))
                except ValueError:
                    sleep_for = delays[attempt]
            else:
                sleep_for = delays[attempt]
            time.sleep(sleep_for)
        except Exception as e:
            if attempt >= len(delays):
                return _failure(provider, model, t0, str(e), status_code=last_status, raw={"body": last_body, "attempts": attempt + 1})
            time.sleep(delays[attempt])
    return _failure(provider, model, t0, "exhausted retries", status_code=last_status, raw={"body": last_body})


def call_openrouter(prompt: str, model: str, timeout: int = 60) -> LLMResult:
    """呼叫 OpenRouter（OpenAI 相容 chat API）。"""
    provider = "openrouter"
    api_key = os.getenv("OPENROUTER_API_KEY")
    t0 = time.time()
    if not api_key:
        return _failure(provider, model, t0, "OPENROUTER_API_KEY missing")
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=timeout,
        )
        status = r.status_code
        if status != 200:
            return _failure(provider, model, t0, f"http {status}", status_code=status, raw={"body": (r.text or "")[:800]})
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return _success(provider, model, text, t0, status_code=status, raw=data)
    except Exception as e:
        return _failure(provider, model, t0, str(e))


def call_home_ollama(prompt: str, model: str, timeout: int = 120, think: bool = False) -> LLMResult:
    """
    呼叫 home 機器（DESKTOP-J17AJFD, RTX 2060）上的 Ollama。

    走法：先在 work 端開好持久 SSH tunnel（local 11434 → home 11434），
    本函式只打 `http://localhost:11434/api/generate`，不再每次都 spawn ssh subprocess。

    為什麼要這樣：原本「每次 LLM 呼叫都 ssh + curl」在 Flask 多執行緒下，
    5 個 cascade call = 5 個 ssh handshake，會互相塞死 Tailscale handshake / SSH auth。
    持久 tunnel 一條連線承擔所有呼叫，沒 spawn 開銷、沒並發死結。

    起 tunnel 的方式（work 端開機跑一次即可）：
        ssh -N -L 11434:127.0.0.1:11434 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes home

    注意 remote 那一端要寫 `127.0.0.1` 不是 `localhost`——home 那邊 localhost 名稱解析
    在 Windows OpenSSH 上會走到別的 socket，導致連線通但 reply 為空。

    home 不是 24/7（colombo 偶爾關機）— tunnel 沒開 / 連不上會回 ConnectionError fallback。
    Tunnel 端點可用 HOME_OLLAMA_URL 環境變數覆蓋（預設 http://localhost:11434）。

    預設模型：gemma3:4b（4B、~70 tok/s）；中文細膩用 qwen2.5:7b；超短任務用 qwen3.5:2b。
    """
    provider = "home_ollama"
    t0 = time.time()
    if not model:
        model = "gemma3:4b"

    base_url = os.getenv("HOME_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "think": bool(think)}
    # reasoning 模型（qwen3.5、gpt-oss）預設會做 thinking、嚴重拖慢嚴格格式 extractor。
    # 對非 reasoning 模型 think 旗標是 no-op；對 reasoning 模型 think=False 關推理（cascade 用）、
    # think=True 開推理（agent mega-prompt 用、需要思考）。

    try:
        r = requests.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            return _failure(provider, model, t0, f"http {r.status_code}",
                            status_code=r.status_code, raw={"body": (r.text or "")[:800]})
        data = r.json()
        text = (data.get("response", "") or "").strip()
        if not text:
            return _failure(provider, model, t0, "ollama returned empty response",
                            status_code=r.status_code, raw=data)
        return _success(provider, model, text, t0, status_code=r.status_code, raw=data)
    except requests.exceptions.ConnectionError as e:
        return _failure(provider, model, t0, f"tunnel down: {e}".replace("\n", " ")[:600])
    except requests.exceptions.Timeout:
        return _failure(provider, model, t0, f"timeout after {timeout}s")
    except Exception as e:
        return _failure(provider, model, t0, str(e))


def call_hf(prompt: str, model: str, timeout: int = 60, max_retries: int = 4) -> LLMResult:
    """呼叫 HuggingFace InferenceClient chat completion。

    遇 rate limit / 5xx 系列短暫錯誤會做指數退避重試（預設最多 4 次重試，累積最長 ~20s）。
    HF Inference 把 HTTP 細節包在 HfHubHTTPError 裡，所以這裡用 message 字串嗅探判定可重試。

    max_retries=0 → fail-fast，只跑 1 次嘗試。同 call_groq 設計，給外層 fallback chain 用。
    """
    provider = "hf"
    token = os.getenv("HF_TOKEN")
    t0 = time.time()
    if not token:
        return _failure(provider, model, t0, "HF_TOKEN missing")
    if InferenceClient is None:
        return _failure(provider, model, t0, "huggingface_hub not installed")
    client = InferenceClient(api_key=token)
    base_delays = [1.0, 2.0, 5.0, 12.0]
    delays = base_delays[:max(0, max_retries)]
    last_err = ""
    for attempt in range(len(delays) + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = completion.choices[0].message
            text = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
            return _success(provider, model, text, t0, raw={"attempts": attempt + 1})
        except Exception as e:
            last_err = str(e)
            # 判定 retryable：rate limit / quota / overloaded / 5xx 字眼 / 429 / 503
            lower = last_err.lower()
            retryable = any(kw in lower for kw in (
                "429", "rate limit", "rate-limit", "too many", "quota",
                "503", "502", "504", "overload", "temporarily", "timeout",
            ))
            if not retryable or attempt >= len(delays):
                return _failure(provider, model, t0, last_err[:1200], raw={"attempts": attempt + 1})
            time.sleep(delays[attempt])
    return _failure(provider, model, t0, "exhausted retries: " + last_err[:600])


def call_gemini(prompt: str, model: str, timeout: int = 60) -> LLMResult:
    """呼叫 Google Gemini REST API（AI Studio 免費額度）。"""
    provider = "gemini"
    api_key = os.getenv("GEMINI_API_KEY")
    t0 = time.time()
    if not api_key:
        return _failure(provider, model, t0, "GEMINI_API_KEY missing")
    url = GEMINI_API_URL.format(model=model)
    try:
        r = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3},
            },
            timeout=timeout,
        )
        status = r.status_code
        if status != 200:
            return _failure(provider, model, t0, f"http {status}", status_code=status, raw={"body": (r.text or "")[:800]})
        data = r.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or ""
        return _success(provider, model, text, t0, status_code=status, raw=data)
    except Exception as e:
        return _failure(provider, model, t0, str(e))


def call_codex_cli(prompt: str, model: str, timeout: int = 120) -> LLMResult:
    """
    呼叫 Codex CLI，並以與其他 provider 相同的 LLMResult 介面回傳。

    做法：
    - 使用 `codex exec -` 從 stdin 讀 prompt，避免命令列跳脫問題。
    - 使用 `--output-last-message` 把最後訊息寫到暫存檔，盡量避開 CLI 額外日誌污染 stdout。
    - 使用 `--ephemeral` 降低 session/state 檔案鎖定對 GTB 的干擾。

    注意：
    - 這仍是 agent CLI，不是純 completion API。
    - 雖然介面相同，但輸出未必像傳統 API 一樣穩定守格式。
    - 因此很適合先納入 benchmark，比較速度與格式穩定性，再決定是否上主路徑。
    """
    provider = "codex_cli"
    cli_path = _resolve_cli_path(
        "CODEX_CLI_PATH",
        "codex",
        windows_fallbacks=[
            r"D:\npm-global\codex.cmd",
            os.path.expandvars(r"%APPDATA%\npm\codex.cmd"),
        ],
    )
    t0 = time.time()
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix="gtb_codex_", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            output_path = tmp.name

        cmd = [
            cli_path,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "-o",
            output_path,
        ]
        if model:
            cmd.extend(["-m", model])
        cmd.append("-")

        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        text = ""
        if output_path and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()

        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or "codex exec failed").strip()
            return _failure(
                provider,
                model,
                t0,
                error[:1200],
                status_code=proc.returncode,
                raw={
                    "command": cmd,
                    "stdout": (proc.stdout or "")[:1200],
                    "stderr": (proc.stderr or "")[:1200],
                },
            )

        if not text:
            text = (proc.stdout or "").strip()

        if not text:
            return _failure(
                provider,
                model,
                t0,
                "codex exec returned empty output",
                status_code=proc.returncode,
                raw={
                    "command": cmd,
                    "stdout": (proc.stdout or "")[:1200],
                    "stderr": (proc.stderr or "")[:1200],
                },
            )

        return _success(
            provider,
            model,
            text,
            t0,
            status_code=proc.returncode,
            raw={
                "command": cmd,
                "stdout": (proc.stdout or "")[:1200],
                "stderr": (proc.stderr or "")[:1200],
            },
        )
    except subprocess.TimeoutExpired:
        return _failure(provider, model, t0, f"timeout after {timeout}s")
    except Exception as e:
        return _failure(provider, model, t0, str(e))
    finally:
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def call_gemini_cli(prompt: str, model: str, timeout: int = 120) -> LLMResult:
    """
    呼叫 Gemini CLI，並以統一的 LLMResult 介面回傳。

    做法：
    - 使用 `gemini -p` 走非互動模式。
    - 使用 `--output-format text` 盡量取得乾淨文字輸出。
    - PATH 抓不到時，會嘗試常見 npm global 安裝位置。

    注意：
    - Gemini CLI 需要先完成登入，或提供 `GEMINI_API_KEY` 等認證環境變數。
    - 若尚未登入，這裡會回傳 provider 失敗結果，不會讓上層程式直接炸掉。
    """
    provider = "gemini_cli"
    cli_path = _resolve_cli_path(
        "GEMINI_CLI_PATH",
        "gemini",
        windows_fallbacks=[
            r"D:\npm-global\gemini.cmd",
            os.path.expandvars(r"%APPDATA%\npm\gemini.cmd"),
        ],
    )
    t0 = time.time()

    cmd = [
        cli_path,
        "-p",
        prompt,
        "--output-format",
        "text",
    ]
    if model:
        cmd.extend(["-m", model])

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            return _failure(
                provider,
                model,
                t0,
                (stderr or stdout or "gemini CLI failed")[:1200],
                status_code=proc.returncode,
                raw={
                    "command": cmd,
                    "stdout": stdout[:1200],
                    "stderr": stderr[:1200],
                },
            )

        if not stdout:
            return _failure(
                provider,
                model,
                t0,
                "gemini CLI returned empty output",
                status_code=proc.returncode,
                raw={
                    "command": cmd,
                    "stdout": stdout[:1200],
                    "stderr": stderr[:1200],
                },
            )

        return _success(
            provider,
            model,
            stdout,
            t0,
            status_code=proc.returncode,
            raw={
                "command": cmd,
                "stdout": stdout[:1200],
                "stderr": stderr[:1200],
            },
        )
    except subprocess.TimeoutExpired:
        return _failure(provider, model, t0, f"timeout after {timeout}s")
    except Exception as e:
        return _failure(provider, model, t0, str(e))


def call_provider(provider: str, prompt: str, model: str, timeout: int = 60, **kwargs) -> LLMResult:
    """
    Provider 路由器。
    保持簡單，讓 benchmark 腳本或上層程式只用一個 provider 字串就能切換後端。

    目前 provider 一覽：
    - groq
    - openrouter
    - home_ollama
    - hf
    - codex_cli
    - gemini_cli

    kwargs（provider 專屬 / 共用）：
    - groq / hf: `max_retries=N` 控制 retry 次數（預設 4，外層有 fallback 鏈時可傳 0 跑 fail-fast）
    - home_ollama: `think=True/False` 切 reasoning 開關
    """
    p = (provider or "").strip().lower()
    if p == "groq":
        return call_groq(prompt, model, timeout=timeout, max_retries=kwargs.get("max_retries", 4))
    if p == "openrouter":
        return call_openrouter(prompt, model, timeout=timeout)
    if p == "home_ollama":
        return call_home_ollama(prompt, model, timeout=timeout, think=bool(kwargs.get("think", False)))
    if p == "hf":
        return call_hf(prompt, model, timeout=timeout, max_retries=kwargs.get("max_retries", 4))
    if p == "gemini":
        return call_gemini(prompt, model, timeout=timeout)
    if p == "codex_cli":
        return call_codex_cli(prompt, model, timeout=timeout)
    if p == "gemini_cli":
        return call_gemini_cli(prompt, model, timeout=timeout)
    t0 = time.time()
    return _failure(p, model, t0, f"unknown provider: {provider}")
