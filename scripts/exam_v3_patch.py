"""Patch boss_exam_v1.json round 2:
- 「有支手機加2點」(ids 46/57/72/74) → name 從 '首麗' 改為 '有支手機'（id 35 上輪已改）
- 「法博加6點」(id 42) → expected_intent 從 'queryPoints' 改為 'adjustPoints'
"""
import json
from pathlib import Path

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")
exam = json.loads(EXAM.read_text(encoding="utf-8"))
items = exam["items"]

changes = []
for it in items:
    cid = it["id"]
    ev = it.get("expected_values") or {}

    # rule 1: 「有支手機加2點」name → 有支手機
    if it["user_text"] == "有支手機加2點":
        old = ev.get("name")
        if old != "有支手機":
            ev["name"] = "有支手機"
            it["expected_values"] = ev
            changes.append((cid, it["user_text"], f"name: {old!r} -> '有支手機'"))

    # rule 2: 「法博加6點」intent → adjustPoints
    if it["user_text"] == "法博加6點":
        old = it.get("expected_intent")
        if old != "adjustPoints":
            it["expected_intent"] = "adjustPoints"
            changes.append((cid, it["user_text"], f"intent: {old!r} -> 'adjustPoints'"))

EXAM.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"patched {len(changes)} items in {EXAM}")
for cid, ut, msg in changes:
    print(f"  id={cid:3d} '{ut}' {msg}")
