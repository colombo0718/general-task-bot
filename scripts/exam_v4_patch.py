"""Patch corpus round 3 (v4):
- id 2/10/11: remove start_month, set due_month per "X月開始 → due_month = (X-1)月" rule.
  「12月開始」 → due_month = 2025/11
- 用 due_month 統一描述「下個月開始」的概念（colombo 的 schema 設計）
"""
import json
from pathlib import Path

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")
exam = json.loads(EXAM.read_text(encoding="utf-8"))
items = exam["items"]

# id -> (operations)
# operations: list of (action, key, [value])
# action: "set" / "del"
TARGET_PATCHES = {
    2:  [("del", "start_month"), ("set", "due_month", "2025/11")],
    10: [("del", "start_month"), ("set", "due_month", "2025/11")],
    11: [("del", "start_month"), ("set", "due_month", "2025/11")],
}

changes = []
for it in items:
    cid = it["id"]
    if cid not in TARGET_PATCHES:
        continue
    ev = it.get("expected_values") or {}
    before = dict(ev)
    for op in TARGET_PATCHES[cid]:
        if op[0] == "del":
            ev.pop(op[1], None)
        elif op[0] == "set":
            ev[op[1]] = op[2]
    it["expected_values"] = ev
    changes.append((cid, it["user_text"], before, dict(ev)))

EXAM.write_text(json.dumps(exam, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"patched {len(changes)} items in {EXAM}")
for cid, ut, before, after in changes:
    print(f"  id={cid:3d} '{ut[:35]}'")
    print(f"    before: {json.dumps(before, ensure_ascii=False)}")
    print(f"    after:  {json.dumps(after, ensure_ascii=False)}")
