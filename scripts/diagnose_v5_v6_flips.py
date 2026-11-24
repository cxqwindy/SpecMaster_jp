#!/usr/bin/env python3
"""Diagnose v5/v6 disagreement cases with deterministic tool evidence.

The goal is to understand whether v6 fixes vs breaks are separable by tool-derived
features. This script does NOT tune labels into a deployable rule; it only prints
feature summaries for the pairwise flip groups.

Example:
  python /fs_mol/cxq/SpecMaster_jp/scripts/diagnose_v5_v6_flips.py \
    --base /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json \
    --new  /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v6_eval_test2000.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
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
    s = str(x).strip().lower()
    if s in {"yes", "true", "1"}:
        return "Yes"
    if s in {"no", "false", "0"}:
        return "No"
    return None


def pred_answer(row: dict[str, Any]) -> str | None:
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


def parse_observation(turn: dict[str, Any]) -> dict[str, Any]:
    obs = turn.get("tool_observation")
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, str):
        try:
            obj = json.loads(obs)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def extract_tools(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for turn in row.get("transcript") or []:
        action = turn.get("action") or {}
        name = action.get("name") or action.get("tool")
        obs = parse_observation(turn)
        if not name or not obs:
            continue
        if name == "count_atoms":
            element = obs.get("element") or (action.get("arguments") or {}).get("element")
            if element:
                out[f"count_{element}"] = obs
        elif name == "spectral_match":
            out["spectral_match"] = obs
        elif name == "functional_group_scan":
            out["functional_group_scan"] = obs
    return out


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def features(row: dict[str, Any]) -> dict[str, Any]:
    tools = extract_tools(row)
    c = tools.get("count_C") or {}
    h = tools.get("count_H") or {}
    s = tools.get("spectral_match") or {}

    c_cand = fnum(c.get("candidate_count"))
    c_exp = fnum(c.get("experimental_distinct_signals"))
    h_cand = fnum(h.get("candidate_count"))
    h_exp = fnum(h.get("experimental_total_nH"))

    return {
        "c_match": c.get("match"),
        "h_match": h.get("match"),
        "c_gap": None if c_cand is None or c_exp is None else c_cand - c_exp,
        "h_abs_gap": None if h_cand is None or h_exp is None else abs(h_cand - h_exp),
        "similarity": fnum(s.get("similarity")),
        "c_sim": fnum(s.get("c_sim")),
        "h_sim": fnum(s.get("h_sim")),
        "n_pred_c": fnum(s.get("n_pred_c")),
        "n_exp_c": fnum(s.get("n_exp_c")),
        "n_pred_h": fnum(s.get("n_pred_h")),
        "n_exp_h": fnum(s.get("n_exp_h")),
    }


def summarize_numeric(values: list[float]) -> str:
    if not values:
        return "n=0"
    values = sorted(values)
    q = lambda p: values[min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))]
    return (
        f"n={len(values)} mean={stats.fmean(values):.3f} "
        f"p10={q(0.10):.3f} p50={q(0.50):.3f} p90={q(0.90):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline rollout, e.g. v5")
    ap.add_argument("--new", required=True, help="new rollout, e.g. v6")
    ap.add_argument("--show", type=int, default=25, help="examples per flip direction")
    args = ap.parse_args()

    base_rows = json.loads(Path(args.base).read_text())
    new_rows = json.loads(Path(args.new).read_text())
    if len(base_rows) != len(new_rows):
        raise SystemExit(f"Length mismatch: {len(base_rows)} vs {len(new_rows)}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, (br, nr) in enumerate(zip(base_rows, new_rows)):
        gold = gold_answer(br)
        bp = pred_answer(br)
        np = pred_answer(nr)
        bok = bp == gold
        nok = np == gold
        if bok == nok:
            continue
        direction = "fixed_by_v6" if nok else "broken_by_v6"
        # Use the v6 transcript for feature summaries because we diagnose why v6 said yes/no.
        feat = features(nr)
        feat.update(
            {
                "idx": i,
                "type": row_type(br, gold),
                "gold": gold,
                "base_pred": bp,
                "new_pred": np,
                "candidate": str(br.get("case_candidate") or br.get("candidate_smiles") or "")[:120],
            }
        )
        groups[direction].append(feat)

    for direction in ("fixed_by_v6", "broken_by_v6"):
        rows = groups[direction]
        print(f"\n[{direction}] n={len(rows)}")
        print("by_type:")
        for typ, n in sorted(Counter(r["type"] for r in rows).items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {typ}: {n}")
        print("pred/gold:")
        for k, n in sorted(Counter((r["gold"], r["new_pred"]) for r in rows).items(), key=lambda kv: str(kv[0])):
            print(f"  gold={k[0]} v6={k[1]}: {n}")

        print("feature distributions from v6 tool logs:")
        for key in ("similarity", "c_sim", "h_sim", "c_gap", "h_abs_gap"):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            print(f"  {key}: {summarize_numeric(vals)}")
        for key in ("c_match", "h_match"):
            print(f"  {key}: {dict(Counter(r.get(key) for r in rows))}")

        print(f"examples first {args.show}:")
        for r in rows[: args.show]:
            print(
                f"  idx={r['idx']} type={r['type']} gold={r['gold']} "
                f"v5={r['base_pred']} v6={r['new_pred']} "
                f"sim={r.get('similarity')} c_sim={r.get('c_sim')} h_sim={r.get('h_sim')} "
                f"c_match={r.get('c_match')} h_match={r.get('h_match')} "
                f"c_gap={r.get('c_gap')} h_abs_gap={r.get('h_abs_gap')} "
                f"candidate={r['candidate']}"
            )


if __name__ == "__main__":
    main()
