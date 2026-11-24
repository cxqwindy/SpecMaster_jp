#!/usr/bin/env python3
"""Rewrite <think> blocks in nmr_matching_tool_sft_v4 that contradict the next
deterministic tool observation.

For each finding in the audit report:
  - Replace the entire <think>...</think> content in the offending function_call
    turn with a neutral verification statement, e.g.:
        <think>
        Let me verify the candidate's carbon count via count_atoms.
        </think>
  - Preserve the tool call JSON that follows.
  - Do NOT touch later turns (the tool observation is already correct, and the
    subsequent thinks may consume the correct value from the observation).

Writes a new SFT JSON with `_v4_1` suffix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

NEUTRAL_TEMPLATES = {
    "count_atoms": (
        "<think>\nLet me verify the candidate's {element_word} count "
        "against the experimental NMR signal count.\n</think>"
    ),
    "spectral_match": (
        "<think>\nLet me score the predicted-vs-experimental spectral similarity "
        "for the candidate.\n</think>"
    ),
    "check_substructure": (
        "<think>\nLet me check whether the candidate contains the substructure I "
        "have in mind.\n</think>"
    ),
}


def infer_element_word(function_call_text: str) -> str:
    """Extract element='C'|'H' from the tool call JSON, return 'carbon'|'hydrogen'."""
    m = re.search(r'"element"\s*:\s*"?([CHch])"?', function_call_text)
    if not m:
        return "atom"
    return "carbon" if m.group(1).upper() == "C" else "hydrogen"


def rewrite_think(function_call_text: str, tool_name: str) -> str:
    """Replace the <think>...</think> block with a neutral template."""
    tpl = NEUTRAL_TEMPLATES.get(tool_name)
    if tpl is None:
        return function_call_text
    if tool_name == "count_atoms":
        new_think = tpl.format(element_word=infer_element_word(function_call_text))
    else:
        new_think = tpl
    return _THINK_RE.sub(new_think, function_call_text, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4.json")
    ap.add_argument("--report", default="/fs_mol/cxq/SpecMaster_jp/sft_think_tool_mismatch_report.json")
    ap.add_argument("--out", default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4_1.json")
    ap.add_argument("--val", default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4_val.json")
    ap.add_argument("--val_out", default="/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4_1_val.json")
    args = ap.parse_args()

    data = json.load(open(args.sft))
    report: Dict[str, list] = json.load(open(args.report))
    print(f"[load] SFT: {len(data)} trajectories, report flags: {len(report)}")

    # Turn (int_key -> findings). JSON keys are strings.
    findings_by_traj: Dict[int, list] = {int(k): v for k, v in report.items()}

    n_traj_fixed = 0
    n_turns_fixed = 0
    for idx, findings in findings_by_traj.items():
        rec = data[idx]
        convs = rec["conversations"]
        # Collect the turn indices that need rewriting.
        turns_to_fix = {}
        for f in findings:
            t = f["turn"]
            tool = f["tool"]
            turns_to_fix[t] = tool
        touched = False
        for t, tool in turns_to_fix.items():
            if t >= len(convs):
                continue
            if convs[t].get("from") != "function_call":
                continue
            new_val = rewrite_think(convs[t]["value"], tool)
            if new_val != convs[t]["value"]:
                convs[t]["value"] = new_val
                n_turns_fixed += 1
                touched = True
        if touched:
            n_traj_fixed += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[write] {args.out}   trajectories touched={n_traj_fixed}, turns rewritten={n_turns_fixed}")

    # Also emit a val split file for convenience (unchanged; val was not audited).
    if Path(args.val).exists():
        val = json.load(open(args.val))
        Path(args.val_out).write_text(json.dumps(val, ensure_ascii=False, indent=2))
        print(f"[write] {args.val_out}   (copied unchanged, {len(val)} rows)")


if __name__ == "__main__":
    main()
