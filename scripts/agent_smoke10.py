"""Pick 10 representative items from boss_exam_v1.json, hit /sim with mode=agent, tally."""
import json
import time
from pathlib import Path

import requests

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")
TARGET = "http://localhost:6010/sim/@708juxdz_dev"

exam = json.loads(EXAM.read_text(encoding="utf-8"))
items = exam["items"]

# 一個 expected_intent 抓一題，最多 10 題；不足 10 用題本順序補
seen = {}
for it in items:
    ei = it.get("expected_intent")
    if ei and ei not in seen:
        seen[ei] = it
    if len(seen) >= 10:
        break
picked = list(seen.values())[:10]
if len(picked) < 10:
    for it in items:
        if it not in picked:
            picked.append(it)
        if len(picked) >= 10:
            break

print(f"\n=== Agent smoke 10 (picked {len(picked)} items) ===\n", flush=True)
results = []
for idx, it in enumerate(picked, 1):
    print(f"[{idx}/{len(picked)}] id={it['id']:3d} expect={it['expected_intent']:20s} text={it['user_text'][:30]}", flush=True)
    t0 = time.time()
    try:
        r = requests.post(TARGET, json={
            "user_id": it.get("user_id", "Usmoke"),
            "user_text": it["user_text"],
            "mode": "agent",
        }, timeout=180)
        resp = r.json()
        wall = round(time.time() - t0, 2)
        actual_intent = resp.get("intent") or "NONE"
        actual_values = resp.get("values") or {}
        expected_values = it.get("expected_values") or {}
        intent_ok = (actual_intent == it["expected_intent"])
        values_ok = all(
            str(actual_values.get(k, "")).strip() == str(v).strip()
            for k, v in expected_values.items()
        )
        verdict = "OK" if (intent_ok and values_ok) else ("PART" if intent_ok else "FAIL")
        debug = resp.get("agent_debug", {})
        q1_lat = debug.get("q1", {}).get("latency_sec")
        q2_lat = debug.get("q2", {}).get("latency_sec")
        print(f"      -> {verdict:4s} actual_intent={actual_intent:20s} wall={wall}s Q1={q1_lat}s Q2={q2_lat}s", flush=True)
        print(f"        expected_values = {expected_values}", flush=True)
        print(f"        actual_values   = {actual_values}", flush=True)
        results.append({
            "id": it["id"], "text": it["user_text"],
            "expected_intent": it["expected_intent"], "actual_intent": actual_intent,
            "intent_ok": intent_ok, "values_ok": values_ok,
            "expected_values": expected_values, "actual_values": actual_values,
            "wall_sec": wall, "q1_lat": q1_lat, "q2_lat": q2_lat,
            "q1_raw": debug.get("q1", {}).get("raw"),
            "q2_raw": debug.get("q2", {}).get("raw"),
        })
    except Exception as e:
        wall = round(time.time() - t0, 2)
        print(f"      -> EXC after {wall}s: {type(e).__name__}: {str(e)[:200]}", flush=True)
        results.append({"id": it["id"], "text": it["user_text"], "error": str(e)[:300]})
    print(flush=True)

intent_pass = sum(1 for r in results if r.get("intent_ok"))
both_pass = sum(1 for r in results if r.get("intent_ok") and r.get("values_ok"))
avg_wall = sum(r.get("wall_sec", 0) for r in results) / max(len(results), 1)
print(f"=== summary ===", flush=True)
print(f"  intent correct: {intent_pass}/{len(results)}", flush=True)
print(f"  intent + values correct: {both_pass}/{len(results)}", flush=True)
print(f"  avg wall per item: {avg_wall:.1f}s", flush=True)

out = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\agent_smoke10.json")
out.write_text(json.dumps({"summary": {"intent_pass": intent_pass, "both_pass": both_pass, "n": len(results), "avg_wall_sec": avg_wall}, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  saved: {out}", flush=True)
