#!/usr/bin/env python3
"""Summarize NMR matching rollout accuracy.

Expected rollout rows look like standalone_infer_matching.py output:
  - answer: JSON string or dict with final_answer = Yes/No
  - _label or label: 1 for Yes, 0 for No
  - _negative_type or negative_type: class name

Example:
  python /fs_mol/cxq/SpecMaster_jp/scripts/summarize_matching_eval.py \
    --in /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v6_eval_test2000.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def norm_answer(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return "Yes" if x else "No"
    if isinstance(x, (int, float)):
        return "Yes" if int(x) == 1 else "No"
    s = str(x).strip()
    if s.lower() in {"yes", "true", "1"}:
        return "Yes"
    if s.lower() in {"no", "false", "0"}:
        return "No"
    return None


def pred_answer(row: dict[str, Any]) -> str | None:
    # Most standalone eval files store answer as a JSON string.
    ans = row.get("answer")
    if isinstance(ans, dict):
        return norm_answer(ans.get("final_answer") or ans.get("answer") or ans.get("prediction"))
    if isinstance(ans, str):
        try:
            obj = json.loads(ans)
            if isinstance(obj, dict):
                got = norm_answer(obj.get("final_answer") or obj.get("answer") or obj.get("prediction"))
                if got:
                    return got
        except Exception:
            pass
        m = re.search(r'"final_answer"\s*:\s*"?(Yes|No)"?', ans, flags=re.I)
        if m:
            return norm_answer(m.group(1))

    # Fallbacks for alternate output formats.
    for key in ("pred", "prediction", "final_answer", "pred_answer", "output"):
        got = norm_answer(row.get(key))
        if got:
            return got

    # Last-resort transcript parse.
    for turn in reversed(row.get("transcript") or []):
        action = turn.get("action") or {}
        if action.get("type") == "answer":
            got = pred_answer({"answer": action.get("answer")})
            if got:
                return got
        assistant = turn.get("assistant") or ""
        m = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', assistant, flags=re.S)
        if m:
            got = pred_answer({"answer": m.group(1)})
            if got:
                return got
    return None


def gold_answer(row: dict[str, Any]) -> str | None:
    for key in ("_label", "label", "gt", "gold", "ground_truth", "target"):
        if key in row:
            return norm_answer(row[key])
    return None


def row_type(row: dict[str, Any], gold: str | None) -> str:
    return (
        row.get("_negative_type")
        or row.get("negative_type")
        or row.get("task_type")
        or row.get("type")
        or row.get("category")
        or ("positive" if gold == "Yes" else "unknown")
    )


def short_candidate(row: dict[str, Any], limit: int = 120) -> str:
    s = str(row.get("case_candidate") or row.get("candidate_smiles") or row.get("smiles") or "")
    return s[:limit]


def pct(correct: int, total: int) -> str:
    return "nan" if total == 0 else f"{correct / total * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="rollout JSON file")
    ap.add_argument("--errors-out", default=None, help="optional CSV path for all errors")
    ap.add_argument("--show-errors", type=int, default=30, help="number of error rows to print")
    args = ap.parse_args()

    path = Path(args.input_path)
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise SystemExit(f"Expected JSON list, got {type(rows).__name__}")

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: Counter[tuple[str | None, str | None]] = Counter()
    errors: list[dict[str, Any]] = []
    unparsed: list[int] = []

    for i, row in enumerate(rows):
        pred = pred_answer(row)
        gold = gold_answer(row)
        typ = row_type(row, gold)

        if pred not in {"Yes", "No"} or gold not in {"Yes", "No"}:
            unparsed.append(i)

        ok = pred == gold
        by_type[typ][0] += int(ok)
        by_type[typ][1] += 1
        confusion[(gold, pred)] += 1

        if not ok:
            errors.append(
                {
                    "idx": i,
                    "type": typ,
                    "gold": gold,
                    "pred": pred,
                    "candidate": short_candidate(row),
                    "fail_reason": row.get("fail_reason"),
                }
            )

    total_correct = sum(correct for correct, total in by_type.values())
    total_n = sum(total for correct, total in by_type.values())

    print(f"file: {path}")
    print(f"n: {total_n}")
    print(f"overall: {total_correct}/{total_n} = {pct(total_correct, total_n)}")
    print(f"errors: {len(errors)}")
    print(f"unparsed: {len(unparsed)}" + (f" first={unparsed[:20]}" if unparsed else ""))

    print("\nconfusion:")
    for (gold, pred), count in sorted(confusion.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  gold={gold} pred={pred}: {count}")

    print("\nby_type:")
    for typ, (correct, total) in sorted(by_type.items(), key=lambda kv: (kv[0] != "positive", kv[0])):
        print(f"  {typ}: {correct}/{total} = {pct(correct, total)}  errors={total - correct}")

    print("\nerrors_by_type:")
    for typ, count in sorted(Counter(e["type"] for e in errors).items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {typ}: {count}")

    if args.show_errors > 0:
        print(f"\nfirst_{args.show_errors}_errors:")
        for e in errors[: args.show_errors]:
            print(
                f"  idx={e['idx']} type={e['type']} gold={e['gold']} pred={e['pred']} "
                f"candidate={e['candidate']}"
            )

    if args.errors_out:
        out = Path(args.errors_out)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "type", "gold", "pred", "candidate", "fail_reason"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"\nwrote errors: {out}")


if __name__ == "__main__":
    main()
