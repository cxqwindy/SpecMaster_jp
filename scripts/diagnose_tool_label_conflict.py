#!/usr/bin/env python3
"""Diagnose label=0 trajectories where all tool signals are strongly positive.

Flags cases where:
  - label = 0  (ground truth = No)
  - spectral_match.similarity > SIM_THRESH
  - both count_atoms calls (C and H) return match=true

For each flagged case, checks whether the final <think> block contains a
peak_assignment-sourced residual (a float followed by 'ppm' near 'unmatched'
or 'residual' or 'mismatch') — indicating the teacher found a specific reason
to say No — or just asserts No without tool-grounded evidence.

Outputs a summary + a JSONL of problematic rows (weak-justification cases).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRAJ_FILE = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_2.jsonl")
OUT_JSONL  = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/tool_label_conflicts.jsonl")
SIM_THRESH = 0.70

# Matches residual-style evidence: e.g. "residual 8.1 ppm", "43.2 ppm unmatched"
_RESIDUAL_RE = re.compile(
    r"(?:"
    r"\b(?:unmatched|residual|mismatch)\b[^.]{0,40}?\d{1,3}\.\d{1,2}\s*ppm"
    r"|"
    r"\d{1,3}\.\d{1,2}\s*ppm[^.]{0,40}?\b(?:unmatched|residual|mismatch)\b"
    r")",
    re.IGNORECASE,
)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def get_label(row: dict) -> int | None:
    v = row.get("meta", {}).get("label")
    if v is None:
        v = row.get("label")
    return int(v) if v is not None else None


def parse_tool_log(row: dict) -> dict:
    tlog = row.get("tool_log")
    if tlog:
        return tlog
    # Extract from messages: find role=tool responses and identify by content
    messages = row.get("messages") or row.get("conversations") or []
    result = {}
    for m in messages:
        if m.get("role") != "tool" and m.get("from") != "observation":
            continue
        content = m.get("content") or m.get("value") or ""
        try:
            obj = json.loads(content)
        except Exception:
            continue
        if "element" in obj and obj["element"] == "C":
            result["count_atoms_C"] = obj
        elif "element" in obj and obj["element"] == "H":
            result["count_atoms_H"] = obj
        elif "similarity" in obj and "c_sim" in obj:
            result["spectral_match"] = obj
    return result


def parse_conversations(row: dict) -> list[dict]:
    convs = row.get("conversations") or []
    if convs:
        return convs
    return row.get("messages") or []


def extract_final_think(messages: list[dict]) -> str:
    """Return the last <think> block found in the last assistant message."""
    for m in reversed(messages):
        if m.get("role") == "assistant" or m.get("from") == "gpt":
            content = m.get("content") or m.get("value") or ""
            thinks = _THINK_RE.findall(content)
            if thinks:
                return thinks[-1]
    return ""


def has_peak_assignment_evidence(think: str) -> bool:
    return bool(_RESIDUAL_RE.search(think))


def main():
    rows = [json.loads(l) for l in TRAJ_FILE.open() if l.strip()]
    print(f"total trajectories: {len(rows)}")

    flagged_all_tool_pos   = []  # label=0, all tools positive
    weak_justification     = []  # subset: no peak_assignment residual in final think
    strong_justification   = []  # subset: has residual evidence

    for i, row in enumerate(rows):
        label = get_label(row)
        if label != 0:
            continue

        tlog = parse_tool_log(row)
        spec = tlog.get("spectral_match") or {}
        sim = spec.get("similarity")
        if sim is None or float(sim) <= SIM_THRESH:
            continue

        # Check both count_atoms calls
        c_atoms = tlog.get("count_atoms_C") or tlog.get("count_atoms", {})
        h_atoms = tlog.get("count_atoms_H") or {}
        # handle flat tool_log where count_atoms is keyed by element
        if isinstance(c_atoms, dict) and "C" in c_atoms:
            c_match = c_atoms["C"].get("match")
            h_match = c_atoms.get("H", {}).get("match")
        else:
            c_match = c_atoms.get("match") if isinstance(c_atoms, dict) else None
            h_match = h_atoms.get("match") if isinstance(h_atoms, dict) else None

        # If we can't read count_atoms match, check the messages for tool responses
        if c_match is None or h_match is None:
            messages = parse_conversations(row)
            c_match_from_msg = None
            h_match_from_msg = None
            for m in messages:
                if m.get("role") == "tool" or m.get("from") == "observation":
                    content = m.get("content") or m.get("value") or ""
                    try:
                        obj = json.loads(content)
                        if obj.get("element") == "C":
                            c_match_from_msg = obj.get("match")
                        elif obj.get("element") == "H":
                            h_match_from_msg = obj.get("match")
                    except Exception:
                        pass
            if c_match is None:
                c_match = c_match_from_msg
            if h_match is None:
                h_match = h_match_from_msg

        if not (c_match is True and h_match is True):
            continue

        # All tool signals positive but label=0 — flagged
        messages = parse_conversations(row)
        final_think = extract_final_think(messages)
        has_evidence = has_peak_assignment_evidence(final_think)

        entry = {
            "row_index": i,
            "smiles": row.get("candidate_smiles") or row.get("meta", {}).get("candidate_smiles"),
            "negative_type": row.get("meta", {}).get("negative_type") or row.get("negative_type"),
            "similarity": float(sim),
            "c_match": c_match,
            "h_match": h_match,
            "has_residual_evidence": has_evidence,
            "final_think_snippet": final_think[:300].replace("\n", " "),
        }
        flagged_all_tool_pos.append(entry)
        if has_evidence:
            strong_justification.append(entry)
        else:
            weak_justification.append(entry)

    print(f"\nlabel=0 with sim>{SIM_THRESH} AND both count_atoms match=true: {len(flagged_all_tool_pos)}")
    print(f"  of those, has peak_assignment residual evidence (keep): {len(strong_justification)}")
    print(f"  of those, NO residual evidence (drop candidates):       {len(weak_justification)}")

    # Breakdown by negative_type
    from collections import Counter
    neg_type_all  = Counter(e["negative_type"] for e in flagged_all_tool_pos)
    neg_type_weak = Counter(e["negative_type"] for e in weak_justification)
    print("\nnegative_type breakdown (all flagged):")
    for nt, cnt in neg_type_all.most_common():
        print(f"  {nt}: {cnt}")
    print("\nnegative_type breakdown (weak/no-evidence):")
    for nt, cnt in neg_type_weak.most_common():
        print(f"  {nt}: {cnt}")

    # Similarity distribution of weak cases
    if weak_justification:
        sims = sorted(e["similarity"] for e in weak_justification)
        print(f"\nweak-justification similarity: min={sims[0]:.3f}  "
              f"median={sims[len(sims)//2]:.3f}  max={sims[-1]:.3f}")

    OUT_JSONL.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in weak_justification))
    print(f"\n[write] {OUT_JSONL}  ({len(weak_justification)} rows)")
    print("These are the drop candidates — review before deciding.")


if __name__ == "__main__":
    main()
