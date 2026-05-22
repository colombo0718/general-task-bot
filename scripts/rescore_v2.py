"""Rescore v2 baselines against the patched corpus (without re-running):
- Match each result item by user_text (id renumbering proof)
- Skip results whose user_text was removed from corpus
- Recompute intent_match + values_match against current expected
- Print updated comparison matrix
"""
import json
import re
from pathlib import Path

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")
REPORTS_DIR = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\reports")

V2_BASELINES = [
    ("2026-05-13_1108_v2_home_gemma3_4b",  "gemma3:4b",   "LLM cascade"),
    ("2026-05-13_1101_v2_home_qwen25_7b",  "qwen2.5:7b",  "LLM cascade"),
    ("2026-05-13_1127_v2_home_qwen35_2b",  "qwen3.5:2b",  "LLM cascade (think=false)"),
    ("2026-05-13_1159_v2_agent_codex_cli", "codex_cli",   "Agent"),
    ("2026-05-13_1152_v2_agent_qwen25_7b", "qwen2.5:7b",  "Agent"),
    ("2026-05-13_1144_v2_agent_gemma3_4b", "gemma3:4b",   "Agent"),
    ("2026-05-13_1139_v2_agent_qwen35_2b", "qwen3.5:2b",  "Agent"),
]

def camel_to_snake(s):
    if not s: return s
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

# Build expected map keyed by user_text
exam = json.loads(EXAM.read_text(encoding="utf-8"))
expected_by_text = {it["user_text"]: it for it in exam["items"]}
print(f"current corpus: {len(expected_by_text)} items\n")

print(f"{'Model':14s} {'Mode':28s}  Intent  Values  Both    Time   N")
print("-" * 80)

rows = []
for folder, model, mode in V2_BASELINES:
    rpath = REPORTS_DIR / folder / "results.json"
    if not rpath.exists():
        print(f"  [missing] {folder}")
        continue
    data = json.loads(rpath.read_text(encoding="utf-8"))
    results = data["items"]
    is_agent = mode.startswith("Agent")

    n = 0
    intent_pass = 0
    values_pass = 0
    both_pass = 0
    for r in results:
        ut = r.get("user_text", "")
        if ut not in expected_by_text:
            continue   # 題目被移出了
        exp_item = expected_by_text[ut]
        exp_intent = exp_item.get("expected_intent")
        exp_values = exp_item.get("expected_values") or {}

        actual_intent = r.get("actual_intent")
        actual_values = r.get("actual_values") or {}

        if is_agent and exp_intent:
            intent_ok = actual_intent == camel_to_snake(exp_intent)
        else:
            intent_ok = actual_intent == exp_intent

        values_ok = all(
            str(actual_values.get(k, "")).strip() == str(v).strip()
            for k, v in exp_values.items()
        )

        n += 1
        if intent_ok: intent_pass += 1
        if values_ok: values_pass += 1
        if intent_ok and values_ok: both_pass += 1

    total_sec = data.get("total_sec", 0)
    minutes = total_sec / 60
    intent_pct = intent_pass / max(n, 1) * 100
    values_pct = values_pass / max(n, 1) * 100
    both_pct   = both_pass   / max(n, 1) * 100

    print(f"{model:14s} {mode:28s}  {intent_pct:5.1f}%  {values_pct:5.1f}%  {both_pct:5.1f}%  {minutes:4.1f}m  {n}")
    rows.append({
        "model": model, "mode": mode,
        "intent": intent_pct, "values": values_pct, "both": both_pct,
        "minutes": minutes, "n": n,
    })

print()
print("=== Ranked by Both ===")
for r in sorted(rows, key=lambda x: -x["both"]):
    print(f"  {r['both']:5.1f}%  {r['model']:14s} {r['mode']}")
