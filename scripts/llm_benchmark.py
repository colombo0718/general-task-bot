"""
GTB 的 LLM 批量測試腳本。

衡量指標：
- success：請求成功，且 provider 有回可用內容。
- correct：在 success 基礎上，再通過該題的格式/內容檢查。

此腳本使用固定的小題組，方便快速比較不同 provider/model 的速度與輸出穩定度。
"""

import argparse
import json
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_clients import call_provider


def _safe_stdout():
    # 避免 Windows cp950 主控台在輸出 Unicode 字元時炸掉。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


PROMPTS = [
    "用一句話解釋強化學習。",
    '嚴格只輸出 JSON：{"intent":"...","confidence":0-100}。問題：幫我把千訊加6點',
    "請判斷這句話比較像哪種任務：查點數、加點數、查帳密。句子：千訊還有幾點？",
    "如果使用者說「這個判斷錯了」，系統下一步該怎麼做？請用三點列出。",
]


def check_correctness(prompt_idx: int, text: str) -> Tuple[bool, str]:
    """
    每題的輕量正確性檢查。
    規則保持簡單、可重現，方便快速做回歸比較。
    """
    t = (text or "").strip()
    low = t.lower()
    if prompt_idx == 1:
        ok = ("強化學習" in t) or ("reinforcement learning" in low) or ("獎勵" in t)
        return ok, "contains RL concept keyword"
    if prompt_idx == 2:
        try:
            obj = json.loads(t)
            intent = obj.get("intent")
            conf = obj.get("confidence")
            ok = isinstance(intent, str) and isinstance(conf, (int, float)) and 0 <= float(conf) <= 100
            return ok, "valid json with intent + confidence[0..100]"
        except Exception:
            return False, "not valid strict JSON"
    if prompt_idx == 3:
        ok = ("查點數" in t) or ("query" in low and "point" in low) or ("點數" in t)
        return ok, "classifies as query points"
    if prompt_idx == 4:
        bullets = len(re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-*])\s+", t))
        if bullets >= 3:
            return True, "has >=3 listed steps"
        return False, "missing 3-step list"
    return False, "unknown prompt index"


def summarize_latencies(values: List[float]) -> Dict[str, float]:
    """回傳平均/最小/最大延遲（秒，四捨五入）。"""
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(statistics.mean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def parse_models_arg(raw: str) -> Dict[str, str]:
    """解析 --models，例如：groq=openai/gpt-oss-20b,remote_worker=qwen2.5:7b。"""
    mapping = {}
    if not raw:
        return mapping
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        mapping[k.strip().lower()] = v.strip()
    return mapping


def default_model_for(provider: str) -> str:
    """各 provider 的預設模型，可由對應環境變數覆蓋。"""
    env_key_map = {
        "groq": "GROQ_BENCH_MODEL",
        "remote_worker": "REMOTE_WORKER_BENCH_MODEL",
        "openrouter": "OPENROUTER_BENCH_MODEL",
        "hf": "HF_BENCH_MODEL",
        "codex_cli": "CODEX_BENCH_MODEL",
        "gemini_cli": "GEMINI_BENCH_MODEL",
    }
    fallback = {
        "groq": "openai/gpt-oss-20b",
        "remote_worker": "qwen2.5:7b",
        "openrouter": "openai/gpt-4o-mini",
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "codex_cli": "gpt-5.4",
        "gemini_cli": "gemini-2.5-pro",
    }
    return os.getenv(env_key_map.get(provider, ""), "").strip() or fallback.get(provider, "")


def main():
    _safe_stdout()
    parser = argparse.ArgumentParser(description="Batch benchmark for GTB LLM providers (speed + basic correctness).")
    parser.add_argument("--providers", default="groq,remote_worker", help="Comma-separated providers. e.g. groq,remote_worker")
    parser.add_argument("--models", default="", help="Override model per provider. e.g. groq=openai/gpt-oss-20b,remote_worker=qwen2.5:7b")
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt for each provider.")
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout (seconds).")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel request workers.")
    parser.add_argument("--json-out", default="", help="Optional output file path for detailed JSON report.")
    args = parser.parse_args()

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    model_overrides = parse_models_arg(args.models)

    # 扁平化結果列：每列代表一次（provider, run, prompt）的實測結果。
    results = []
    jobs = []
    print("== LLM Benchmark Start ==")
    for provider in providers:
        model = model_overrides.get(provider) or default_model_for(provider)
        print(f"-- provider={provider} model={model} --")
        for run_idx in range(1, args.runs + 1):
            for i, prompt in enumerate(PROMPTS, 1):
                jobs.append(
                    {
                        "provider": provider,
                        "model": model,
                        "run": run_idx,
                        "prompt_index": i,
                        "prompt": prompt,
                    }
                )

    max_workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for job in jobs:
            future = executor.submit(
                call_provider,
                job["provider"],
                job["prompt"],
                job["model"],
                args.timeout,
            )
            future_map[future] = job

        for future in as_completed(future_map):
            job = future_map[future]
            r = future.result()
            passed, reason = check_correctness(job["prompt_index"], r.text if r.ok else "")
            row = {
                "provider": job["provider"],
                "model": job["model"],
                "run": job["run"],
                "prompt_index": job["prompt_index"],
                "prompt": job["prompt"],
                "ok": r.ok,
                "status_code": r.status_code,
                "latency_sec": round(r.latency_sec, 3),
                "response": r.text,
                "error": r.error,
                "correct": passed if r.ok else False,
                "correct_reason": reason if r.ok else "request_failed",
            }
            results.append(row)
            brief = (r.text or r.error or "").replace("\n", " ")
            print(
                f"[{row['provider']}][run {row['run']}][p{row['prompt_index']}] "
                f"ok={row['ok']} correct={row['correct']} latency={row['latency_sec']}s"
            )
            print(f"  -> {brief[:180]}")

    # 讓輸出檔案與後續分析保持穩定順序（與提交順序一致）。
    results.sort(key=lambda x: (x["provider"], x["run"], x["prompt_index"]))

    print("\n== Summary ==")
    for provider in providers:
        pr = [x for x in results if x["provider"] == provider]
        lat = [x["latency_sec"] for x in pr if x["ok"]]
        total = len(pr)
        success = sum(1 for x in pr if x["ok"])
        correct = sum(1 for x in pr if x["ok"] and x["correct"])
        s = summarize_latencies(lat)
        print(
            f"{provider}: success={success}/{total}, correct={correct}/{total}, "
            f"avg={s['avg']}s min={s['min']}s max={s['max']}s"
        )

    if args.json_out:
        payload = {"providers": providers, "runs": args.runs, "results": results}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved: {args.json_out}")


if __name__ == "__main__":
    main()
