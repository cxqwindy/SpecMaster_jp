#!/usr/bin/env python3
"""Score think-tool-consistency on a rollout dump.

For every trajectory:
  * extract every <think>...</think> block from the assistant's turns
  * verify that all integer atom counts and similarity floats mentioned in the think appear in
    some PRIOR <tool_response> of the same trajectory
  * (delta-ppm floats whitelisted; they may come from the input spectrum)

A trajectory is "consistent" if it has zero unsourced numbers. Overall metric:
    think-tool-consistency = # consistent trajectories / total.

Input formats supported:
  1) M3-style rollout dump — a JSON array of rows each having a `transcript` field (list of
     {role, content, ...} messages).
  2) Hermes multiturn dump — jsonl with `messages` list per row (our v5 SFT data format).

Usage:
    python eval_think_consistency.py --in <rollout.json|rollout.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact validator our v5 rewriter uses so eval and data-cleaning use the same rule.
from recipe.nmr_matching_agent.cot_to_trajectory_v5 import (  # noqa: E402
    _validate_think_number_provenance,
)


def iter_rows(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text()
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        # JSON array
        data = json.loads(text)
        for r in data:
            yield r
    else:
        # jsonl
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a row into the messages list our validator expects."""
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    if "transcript" in row and isinstance(row["transcript"], list):
        # M3-style rollout dump: each turn = {turn, assistant, action, tool_observation, ...}.
        # Emit an interleaved (assistant, tool) message stream so <think> block ordering vs
        # tool_response ordering is preserved.
        out: List[Dict[str, str]] = []
        for m in row["transcript"]:
            asst = m.get("assistant")
            if asst:
                out.append({"role": "assistant", "content": str(asst)})
            tobs = m.get("tool_observation")
            if tobs is not None:
                if isinstance(tobs, (dict, list)):
                    obs_text = json.dumps(tobs, ensure_ascii=False)
                else:
                    obs_text = str(tobs)
                tool_name = None
                act = m.get("action") or {}
                if isinstance(act, dict) and act.get("type") == "tool":
                    tool_name = act.get("name")
                entry: Dict[str, str] = {"role": "tool", "content": obs_text}
                if tool_name:
                    entry["name"] = tool_name
                out.append(entry)
        return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--rejects_out", default=None)
    args = ap.parse_args()

    n_total = n_ok = 0
    by_class: Counter = Counter()
    by_class_ok: Counter = Counter()
    violations_kind: Counter = Counter()
    rejects = []
    for row in iter_rows(args.inp):
        msgs = to_messages(row)
        if not msgs:
            continue
        n_total += 1
        cls = (row.get("_negative_type") or row.get("negative_type")
               or row.get("meta", {}).get("negative_type") or "unknown")
        if row.get("_label") == 1:
            cls = "positive"
        elif row.get("label") == 1:
            cls = "positive"
        by_class[cls] += 1
        vios = _validate_think_number_provenance(msgs)
        if not vios:
            n_ok += 1
            by_class_ok[cls] += 1
        else:
            for v in vios:
                kind = v.split(":")[-1].strip()
                # collapse to broad kind
                if "integer" in v:
                    violations_kind["integer_count"] += 1
                elif "kv-int" in v:
                    violations_kind["kv_int"] += 1
                elif "similarity" in v:
                    violations_kind["similarity_float"] += 1
                elif "kv-float" in v:
                    violations_kind["kv_float"] += 1
                else:
                    violations_kind["other"] += 1
            rejects.append({"row_key": row.get("case_candidate") or row.get("case_id"),
                            "class": cls, "violations": vios})

    print(f"[think-tool-consistency] {n_ok}/{n_total} = {100.0*n_ok/max(n_total,1):.2f}%")
    print("\nBy class:")
    for cls in sorted(by_class):
        tot = by_class[cls]
        ok = by_class_ok[cls]
        print(f"  {cls:60s} {ok}/{tot} ({100.0*ok/max(tot,1):.1f}%)")
    print("\nViolation kinds:")
    for k, v in violations_kind.most_common():
        print(f"  {k:20s} {v}")

    if args.rejects_out and rejects:
        Path(args.rejects_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.rejects_out, "w") as f:
            for r in rejects:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[write] {len(rejects)} rejects -> {args.rejects_out}")


if __name__ == "__main__":
    main()
