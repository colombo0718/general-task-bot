"""Dump boss_exam_v1.json to CSV for human review of expected answers."""
import csv
import json
import sys
from pathlib import Path

EXAM = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.json")
OUT = Path(r"C:\Users\pos\Desktop\cwsoft-aquan-manager\benchmark\boss_exam_v1.csv")

exam = json.loads(EXAM.read_text(encoding="utf-8"))
items = exam["items"]

# 收集所有出現過的 field key
all_field_keys = set()
for it in items:
    ev = it.get("expected_values") or {}
    all_field_keys.update(ev.keys())

# 排序：常見欄位優先
preferred_order = ["name", "delta", "amount", "include_tax", "start_month", "due_month",
                   "charge_months", "shop_name", "branch_code", "region_code", "new_due",
                   "restore_time", "value", "employee_id", "paper", "carbon", "machine",
                   "db_name", "account", "password", "start_date", "end_date"]
ordered_fields = [k for k in preferred_order if k in all_field_keys]
ordered_fields += sorted(k for k in all_field_keys if k not in preferred_order)

header = ["id", "user_text", "expected_intent"] + [f"v.{k}" for k in ordered_fields]

# UTF-8 BOM 讓 Excel 直接認中文
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    for it in items:
        ev = it.get("expected_values") or {}
        row = [
            it.get("id", ""),
            it.get("user_text", ""),
            it.get("expected_intent", ""),
        ]
        for k in ordered_fields:
            row.append(ev.get(k, ""))
        w.writerow(row)

print(f"wrote {len(items)} rows -> {OUT}")
print(f"fields ({len(ordered_fields)}): {ordered_fields}")
