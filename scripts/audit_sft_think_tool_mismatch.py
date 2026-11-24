#!/usr/bin/env python3
"""Audit nmr_matching_tool_sft_v4 for cases where the assistant's <think> narration
contradicts the deterministic tool observation that comes next.

Categories detected:
  A. count_atoms:  <think> claims "N carbons/hydrogens" that contradicts tool's
                   candidate_count (allowing ±1 for a plausible symmetry claim).
  B. spectral_match: <think> asserts a qualitative "high/low match" that
                     contradicts the returned numeric similarity.
  C. check_substructure: <think> asserts presence (or absence) of a group that
                         the tool says otherwise.

Outputs:
  --report_json : {trajectory_index: [{turn, category, evidence}], ...}
  --sample_md   : Markdown with a few examples per category for eyeball review.
  --summary     : Aggregate counts printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SFT_PATH = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4.json"

_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r'(\{\s*"name"\s*:.*?"arguments"\s*:\s*\{[^{}]*\}\s*\})', re.DOTALL)

# ---- count_atoms extractor -------------------------------------------------
# We ONLY flag claims about the CANDIDATE's total atom count. Environment / signal
# counts, sub-region proton claims ("8 aromatic protons"), and nucleus labels
# ("13C NMR") are excluded. Requires an anchor phrase like "candidate has N ..."
# or "molecule has N ...".
_ANCHOR = (
    r"(?:the\s+candidate(?:\s+(?:structure|molecule|smiles))?|"
    r"candidate|molecule|structure|smiles)\s+"
    r"(?:\w+\s+){0,3}?"  # allow up to 3 filler words, e.g. "structure, X, has"
    r"(?:has|contains?|possesses?|features)\s+"
    r"(?:a\s+total\s+of\s+)?"
)
_NUM_C = re.compile(
    _ANCHOR + r"(\d{1,3})\s+"
    r"(?!(?:different|distinct|non-?equivalent|unique|chemically|inequivalent))"
    r"(?:carbon|carbons|C\s*atoms?)"
    r"(?!\s*(?:environments?|signals?|shifts?|peaks?))",
    re.IGNORECASE,
)
_NUM_H = re.compile(
    _ANCHOR + r"(\d{1,3})\s+"
    r"(?!(?:different|distinct|non-?equivalent|unique|chemically|inequivalent|aromatic|aliphatic|methyl|methylene|vinyl|allyl))"
    r"(?:hydrogen|hydrogens|H\s*atoms?|protons?)"
    r"(?!\s*(?:signals?|peaks?))",
    re.IGNORECASE,
)

# ---- spectral_match qualitative extractor ---------------------------------
_POS_MATCH = re.compile(
    r"\b(?:high|good|strong|excellent|perfect|near[- ]?perfect|matches?\s+(?:well|closely)|consistent)\b",
    re.IGNORECASE,
)
_NEG_MATCH = re.compile(
    r"\b(?:low|poor|weak|bad|clearly\s+different|significant(?:ly)?\s+different|mismatch(?:es|ed)?|"
    r"does\s+not\s+match|do\s+not\s+match|do(?:es)?n['’]t\s+match|inconsistent)\b",
    re.IGNORECASE,
)

# ---- check_substructure extractor -----------------------------------------
_PRESENCE_POS = re.compile(
    r"\b(?:contains?|has|possess(?:es)?|feature[sd]?|bear[sd]?|"
    r"is\s+present|are\s+present|there\s+(?:is|are)|include[sd]?)\b",
    re.IGNORECASE,
)
_PRESENCE_NEG = re.compile(
    r"\b(?:lacks?|does\s+not\s+contain|do\s+not\s+contain|do(?:es)?n['’]t\s+"
    r"(?:contain|have)|absent|not\s+present|has\s+no)\b",
    re.IGNORECASE,
)


def parse_tool_call(function_call_text: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Return (tool_name, arguments, think_content) from a function_call turn."""
    think_m = _THINK_RE.search(function_call_text or "")
    think = think_m.group(1).strip() if think_m else None
    body = function_call_text
    if think_m:
        body = body.replace(think_m.group(0), "")
    tc = _TOOL_CALL_RE.search(body or "")
    if not tc:
        return None, {}, think
    try:
        obj = json.loads(tc.group(1))
    except Exception:
        return None, {}, think
    return obj.get("name"), obj.get("arguments") or {}, think


def parse_obs(observation_text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(observation_text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def check_count_atoms(think: str, args: Dict, obs: Dict) -> List[Tuple[str, str]]:
    """Return list of (evidence_summary, category_tag) for count_atoms mismatches."""
    if not think or not obs:
        return []
    element = str(args.get("element", "")).upper()
    tool_n = obs.get("candidate_count")
    if not isinstance(tool_n, int):
        return []
    finds = []
    if element == "C":
        pattern = _NUM_C
        exp_ref = obs.get("experimental_distinct_signals")
    elif element == "H":
        pattern = _NUM_H
        exp_ref = obs.get("experimental_total_nH")
    else:
        return []
    claims = [int(m.group(1)) for m in pattern.finditer(think)]
    if not claims:
        return []
    for c in claims:
        if c == tool_n:
            continue
        if isinstance(exp_ref, int) and c == exp_ref:
            continue
        # Hard contradiction: model asserted candidate has c atoms, tool says tool_n.
        finds.append(
            (f"think says candidate has {c} {element}; tool candidate_count={tool_n} (exp_ref={exp_ref})",
             f"count_{element}"))
    return finds


def check_spectral_match(think: str, args: Dict, obs: Dict) -> List[Tuple[str, str]]:
    if not think or not obs or not obs.get("ok"):
        return []
    sim = obs.get("similarity")
    if not isinstance(sim, (int, float)):
        return []
    pos = bool(_POS_MATCH.search(think))
    neg = bool(_NEG_MATCH.search(think))
    finds = []
    # NOTE: <think> comes BEFORE the tool result, so a claim of "high match"
    # is a hypothesis to be verified. It's only a bug if the CoT-time think
    # was later contradicted AND the same trajectory concluded consistently
    # with the wrong hypothesis. To keep the audit simple, we flag the raw
    # signal here and let the reviewer decide.
    if pos and sim < 0.35:
        finds.append((f"think asserts strong/high match; tool similarity={sim:.3f}",
                      "spectral_pos_false"))
    if neg and sim > 0.7:
        finds.append((f"think asserts poor/low match; tool similarity={sim:.3f}",
                      "spectral_neg_false"))
    return finds


def check_check_substructure(think: str, args: Dict, obs: Dict) -> List[Tuple[str, str]]:
    if not think or not obs:
        return []
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return []
    is_present = obs.get("is_present")
    if is_present is None and "match" in obs:
        is_present = obs.get("match")
    if not isinstance(is_present, bool):
        return []
    # Find sentences in <think> that mention the pattern name.
    lower = pattern.lower()
    sentences = [s.strip() for s in re.split(r"[.!?\n]", think) if s.strip()]
    finds = []
    for s in sentences:
        if lower not in s.lower():
            continue
        pos = bool(_PRESENCE_POS.search(s))
        neg = bool(_PRESENCE_NEG.search(s))
        if pos and not is_present:
            finds.append((f"think asserts presence of '{pattern}'; tool is_present=False | {s[:120]}",
                          "substr_present_false"))
        if neg and is_present:
            finds.append((f"think asserts absence of '{pattern}'; tool is_present=True | {s[:120]}",
                          "substr_absent_false"))
    return finds


CHECKERS = {
    "count_atoms": check_count_atoms,
    "spectral_match": check_spectral_match,
    "check_substructure": check_check_substructure,
}


def audit_record(rec: Dict) -> List[Dict]:
    """Walk conversation turns; for each function_call → observation pair, check."""
    convs = rec.get("conversations", [])
    findings = []
    for i, turn in enumerate(convs):
        if turn.get("from") != "function_call":
            continue
        if i + 1 >= len(convs) or convs[i + 1].get("from") != "observation":
            continue
        name, args, think = parse_tool_call(turn.get("value", ""))
        if name not in CHECKERS:
            continue
        obs = parse_obs(convs[i + 1].get("value", ""))
        if obs is None:
            continue
        for evidence, tag in CHECKERS[name](think, args, obs):
            findings.append({
                "turn": i,
                "tool": name,
                "category": tag,
                "evidence": evidence,
                "think_snippet": (think or "")[:280],
                "obs_snippet": json.dumps(obs, ensure_ascii=False)[:280],
            })
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=SFT_PATH)
    ap.add_argument("--report_json", default="/fs_mol/cxq/SpecMaster_jp/sft_think_tool_mismatch_report.json")
    ap.add_argument("--sample_md", default="/fs_mol/cxq/SpecMaster_jp/sft_think_tool_mismatch_samples.md")
    ap.add_argument("--samples_per_cat", type=int, default=4)
    args = ap.parse_args()

    data = json.load(open(args.sft))
    print(f"loaded {len(data)} SFT trajectories from {args.sft}")

    all_findings = {}
    cat_counter = Counter()
    tool_counter = Counter()
    traj_with_any_issue = 0
    per_traj_counts = Counter()

    for idx, rec in enumerate(data):
        fs = audit_record(rec)
        if fs:
            all_findings[idx] = fs
            traj_with_any_issue += 1
            per_traj_counts[len(fs)] += 1
            for f in fs:
                cat_counter[f["category"]] += 1
                tool_counter[f["tool"]] += 1

    total_trajectories = len(data)
    print()
    print(f"====== Summary ======")
    print(f"Total trajectories                     : {total_trajectories}")
    print(f"Trajectories with any mismatch flagged : {traj_with_any_issue}  "
          f"({100*traj_with_any_issue/total_trajectories:.1f}%)")
    print(f"Total findings across dataset          : {sum(cat_counter.values())}")
    print()
    print("Findings by category:")
    for cat, n in cat_counter.most_common():
        print(f"  {cat:32s} {n}")
    print()
    print("Findings by tool:")
    for t, n in tool_counter.most_common():
        print(f"  {t:32s} {n}")
    print()
    print("Findings-per-trajectory distribution:")
    for k in sorted(per_traj_counts):
        print(f"  {k} findings : {per_traj_counts[k]} trajectories")

    Path(args.report_json).write_text(json.dumps(all_findings, ensure_ascii=False, indent=2))
    print(f"\n[write] full report → {args.report_json}")

    # Sample doc — a few examples per category.
    per_cat_samples: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
    for idx, fs in all_findings.items():
        for f in fs:
            per_cat_samples[f["category"]].append((idx, f))
    md = ["# SFT think ↔ tool observation mismatch samples", ""]
    md.append(f"Audited {total_trajectories} SFT trajectories. "
              f"{traj_with_any_issue} ({100*traj_with_any_issue/total_trajectories:.1f}%) have "
              f"at least one automated mismatch flag; {sum(cat_counter.values())} findings total.")
    md.append("")
    md.append("Category counts: " + ", ".join(f"`{c}`={n}" for c, n in cat_counter.most_common()))
    md.append("")
    for cat, samples in per_cat_samples.items():
        md.append(f"## `{cat}` — {len(samples)} findings, showing {min(len(samples), args.samples_per_cat)}")
        md.append("")
        for idx, f in samples[: args.samples_per_cat]:
            md.append(f"### traj {idx}, turn {f['turn']}, tool `{f['tool']}`")
            md.append(f"- **Evidence**: {f['evidence']}")
            md.append("- **Think snippet**:")
            md.append("  > " + f["think_snippet"].replace("\n", "\n  > "))
            md.append("- **Obs snippet**:")
            md.append(f"  ```json\n  {f['obs_snippet']}\n  ```")
            md.append("")
    Path(args.sample_md).write_text("\n".join(md))
    print(f"[write] sample markdown → {args.sample_md}")


if __name__ == "__main__":
    main()
