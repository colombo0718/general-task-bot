"""Patch boss_exam_v1.json: align expected_values.name to what gemma3:4b LLM actually extracted (verified-correct cases).

Source of correction: 2026-05-13 v2 baseline gemma3:4b LLM cascade results.
colombo manually verified these 8 ids — the LLM's answer was correct, the corpus's expected was wrong.
"""
import json
from pathlib import Path

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")

# id -> new name
PATCHES = {
    2:  "禾名",
    7:  "水里遠傳",
    10: "華騏",
    11: "華騏",
    35: "有支手機",
    37: "正欣",
    45: "有支手機",
    70: "正欣",
}

exam = json.loads(EXAM.read_text(encoding="utf-8"))
items = exam["items"]

changes = []
for it in items:
    if it["id"] in PATCHES:
        ev = it.get("expected_values") or {}
        old = ev.get("name", "<missing>")
        new = PATCHES[it["id"]]
        if old != new:
            ev["name"] = new
            it["expected_values"] = ev
            changes.append((it["id"], it["user_text"][:30], old, new))

EXAM.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"patched {len(changes)} items in {EXAM}")
for cid, ut, o, n in changes:
    print(f"  id={cid:3d} '{ut:30s}' name: {o!r} -> {n!r}")
