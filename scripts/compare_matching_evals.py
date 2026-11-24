#!/usr/bin/env python3
"""Pairwise comparison for two NMR matching rollout files.

This is diagnostic: it uses labels to report where model B fixes or breaks model A.
It can also scan simple confidence-gated overrides, but those are for validation only
unless re-tested on a held-out set.

Example:
  python /fs_mol/cxq/SpecMaster_jp/scripts/compare_matching_evals.py \
    --base /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json \
    --new  /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v6_eval_test2000.json
"""

from __future__ import annotations

import argparse
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


def parse_answer_obj(row: dict[str, Any]) -> dict[str, Any]:
    ans = row.get("answer")
    if isinstance(ans, dict):
        return ans
    if isinstance(ans, str):
        try:
            obj = json.loads(ans)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', ans, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def pred_answer(row: dict[str, Any]) -> str | None:
    obj = parse_answer_obj(row)
    got = norm_answer(obj.get("final_answer") or obj.get("answer") or obj.get("prediction"))
    if got:
        return got
    ans = row.get("answer")
    if isinstance(ans, str):
        m = re.search(r'"final_answer"\s*:\s*"?(Yes|No)"?', ans, flags=re.I)
        if m:
            return norm_answer(m.group(1))
    for key in ("pred", "prediction", "final_answer", "pred_answer", "output"):
        got = norm_answer(row.get(key))
        if got:
            return got
    return None


def confidence(row: dict[str, Any]) -> float:
    obj = parse_answer_obj(row)
    val = obj.get("confidence")
    try:
        return float(val)
    except Exception:
        return 0.0


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


def acc(correct: int, total: int) -> str:
    return "nan" if total == 0 else f"{correct / total * 100:.2f}%"


def summarize_one(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    by_type = defaultdict(lambda: [0, 0])
    conf = Counter()
    preds = []
    for r in rows:
        g = gold_answer(r)
        p = pred_answer(r)
        t = row_type(r, g)
        ok = p == g
        by_type[t][0] += int(ok)
        by_type[t][1] += 1
        conf[(g, p)] += 1
        preds.append(p)
    c = sum(v[0] for v in by_type.values())
    n = sum(v[1] for v in by_type.values())
    print(f"\n[{name}] overall: {c}/{n} = {acc(c, n)}")
    print(f"[{name}] confusion:")
    for (g, p), v in sorted(conf.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  gold={g} pred={p}: {v}")
    return {"correct": c, "total": n, "preds": preds, "by_type": by_type}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline rollout, e.g. v5")
    ap.add_argument("--new", required=True, help="new rollout, e.g. v6")
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--new-name", default="new")
    ap.add_argument("--show-flips", type=int, default=40)
    args = ap.parse_args()

    base_rows = json.loads(Path(args.base).read_text())
    new_rows = json.loads(Path(args.new).read_text())
    if len(base_rows) != len(new_rows):
        raise SystemExit(f"Length mismatch: base={len(base_rows)} new={len(new_rows)}")

    base_sum = summarize_one(base_rows, args.base_name)
    new_sum = summarize_one(new_rows, args.new_name)

    n = len(base_rows)
    counts = Counter()
    by_type = defaultdict(lambda: Counter())
    flips = []
    oracle = 0

    for i, (br, nr) in enumerate(zip(base_rows, new_rows)):
        gold = gold_answer(br)
        if gold != gold_answer(nr):
            raise SystemExit(f"Gold mismatch at idx={i}: {gold} vs {gold_answer(nr)}")
        typ = row_type(br, gold)
        bp = pred_answer(br)
        np = pred_answer(nr)
        bok = bp == gold
        nok = np == gold
        if bok and nok:
            key = "both_correct"
        elif bok and not nok:
            key = "base_only"
        elif (not bok) and nok:
            key = "new_only"
        else:
            key = "both_wrong"
        counts[key] += 1
        by_type[typ][key] += 1
        oracle += int(bok or nok)
        if bok != nok:
            flips.append(
                {
                    "idx": i,
                    "type": typ,
                    "gold": gold,
                    "base_pred": bp,
                    "new_pred": np,
                    "base_conf": confidence(br),
                    "new_conf": confidence(nr),
                    "direction": "fixed_by_new" if nok else "broken_by_new",
                    "candidate": str(br.get("case_candidate") or br.get("candidate_smiles") or "")[:120],
                }
            )

    print("\n[pairwise]")
    for key in ("both_correct", "base_only", "new_only", "both_wrong"):
        print(f"  {key}: {counts[key]}")
    print(f"  oracle_if_choose_correct_model: {oracle}/{n} = {acc(oracle, n)}")
    print(f"  net_delta_new_minus_base: {counts['new_only'] - counts['base_only']} cases")

    print("\n[pairwise_by_type]")
    for typ, c in sorted(by_type.items(), key=lambda kv: (kv[0] != "positive", kv[0])):
        total = sum(c.values())
        delta = c["new_only"] - c["base_only"]
        print(
            f"  {typ}: total={total} both_correct={c['both_correct']} "
            f"base_only={c['base_only']} new_only={c['new_only']} both_wrong={c['both_wrong']} delta={delta:+d}"
        )

    # Diagnostic threshold scan: default to base, override with new only when they disagree
    # and new confidence is at least T. This is intentionally simple and label-tuned.
    print("\n[diagnostic_confidence_override_scan]")
    best = None
    thresholds = [x / 100 for x in range(50, 101, 5)]
    for direction in ("new_no_only", "new_yes_only", "both_directions"):
        for t in thresholds:
            correct = 0
            overrides = 0
            for br, nr in zip(base_rows, new_rows):
                gold = gold_answer(br)
                bp = pred_answer(br)
                np = pred_answer(nr)
                use_new = bp != np and confidence(nr) >= t
                if direction == "new_no_only":
                    use_new = use_new and np == "No"
                elif direction == "new_yes_only":
                    use_new = use_new and np == "Yes"
                pred = np if use_new else bp
                overrides += int(use_new)
                correct += int(pred == gold)
            item = (correct, overrides, direction, t)
            if best is None or item > best:
                best = item
    if best:
        correct, overrides, direction, t = best
        print(
            f"  best_label_tuned_rule: default={args.base_name}, override={args.new_name}, "
            f"direction={direction}, new_conf>={t:.2f}, overrides={overrides}, "
            f"acc={correct}/{n} = {acc(correct, n)}"
        )
        print("  NOTE: this is tuned on this labeled file; use only as a clue unless validated elsewhere.")

    if args.show_flips > 0:
        print(f"\n[first_{args.show_flips}_flips]")
        for f in flips[: args.show_flips]:
            print(
                f"  idx={f['idx']} {f['direction']} type={f['type']} gold={f['gold']} "
                f"base={f['base_pred']}({f['base_conf']:.2f}) new={f['new_pred']}({f['new_conf']:.2f}) "
                f"candidate={f['candidate']}"
            )


if __name__ == "__main__":
    main()
