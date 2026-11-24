#!/usr/bin/env python3
"""Extract successful M3 tool-agent cases for expert review of reasoning chains.

Reads:
  - m3_eval_full_ckpt200.json    (per-case rollout: transcript + tool_sequence + answer)
  - nmr_matching_test_1000.json  (original alpaca input with NMR peaks)

Pairs them by candidate SMILES, samples a diverse set (positive + each negative type),
writes a Markdown review doc with:
  - candidate + gt + label + type
  - full experimental NMR (from the alpaca input)
  - per-turn: <think> narration, tool call JSON, deterministic tool observation
  - final <answer>

Output is intended for a chemistry expert to spot hallucinations (peaks mis-quoted,
chemical shifts mis-assigned, tool outputs mis-read, etc.).
"""

from __future__ import annotations

import json
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

RES_PATH = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/m3_eval_full_ckpt200.json"
TEST_PATH = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json"
OUT_PATH = "/fs_mol/cxq/SpecMaster_jp/expert_review_m3_cases.md"

# how many per type to sample (correct-answer only)
SAMPLES_PER_TYPE = {
    "positive": 2,
    "neg/same_formula": 2,
    "neg/similar_structure": 2,
    "neg/random": 1,
    "neg/hard_nmr/alkyl_chain_shift/insert_ch2": 1,
    "neg/perturbation/alkyl_chain_shift/insert_ch2": 1,
    "neg/perturbation/heteroatom_replacement/O_to_S": 1,
    "neg/perturbation/functional_group_replacement/alcohol_to_thiol": 1,
    "neg/perturbation/functional_group_replacement/ester_to_amide": 1,
    "neg/perturbation/alkyl_chain_shift/add_methyl_branch": 1,
}
SEED = 20260906


def extract_final(a: str) -> Dict:
    if not a:
        return {"final_answer": "", "confidence": None, "key_evidence": ""}
    try:
        d = json.loads(a)
    except Exception:
        d = {}
    return {
        "final_answer": str(d.get("final_answer", "")).strip(),
        "confidence": d.get("confidence"),
        "key_evidence": d.get("key_evidence", ""),
    }


def pretty_tool_call(assistant_text: str) -> str:
    """Extract <think>...</think> and <tool_call>{...}</tool_call> content for display."""
    think_m = re.search(r"<think>([\s\S]*?)</think>", assistant_text, re.IGNORECASE)
    tool_m = re.search(r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>", assistant_text, re.IGNORECASE)
    ans_m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", assistant_text, re.IGNORECASE)
    parts = []
    if think_m:
        parts.append(("think", think_m.group(1).strip()))
    if tool_m:
        parts.append(("tool_call", tool_m.group(1).strip()))
    if ans_m:
        parts.append(("answer", ans_m.group(1).strip()))
    return parts


def format_tool_obs(obs_json: str) -> str:
    try:
        obj = json.loads(obs_json)
    except Exception:
        return obs_json[:600]
    return json.dumps(obj, ensure_ascii=False, indent=2)


def parse_input_block(input_text: str) -> Dict[str, str]:
    """Split alpaca input into candidate line + C NMR line + H NMR line."""
    lines = input_text.strip().splitlines()
    d = {"candidate_line": "", "c_nmr_line": "", "h_nmr_line": ""}
    for line in lines:
        s = line.strip()
        if s.startswith("Candidate SMILES"):
            d["candidate_line"] = s
        elif s.startswith("13C NMR"):
            d["c_nmr_line"] = s
        elif s.startswith("1H NMR"):
            d["h_nmr_line"] = s
    return d


def main():
    res = json.load(open(RES_PATH))
    test = json.load(open(TEST_PATH))

    # Index test by candidate SMILES (fallback tuple).
    test_by_cand: Dict[str, Dict] = {}
    for rec in test:
        cand = rec.get("_candidate_smiles") or ""
        if cand and cand not in test_by_cand:
            test_by_cand[cand] = rec

    # Group correct rollouts by type.
    correct_by_type: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(res):
        f = extract_final(r.get("answer", ""))
        pred = f["final_answer"].lower()
        gt = "yes" if r.get("_label") == 1 else "no"
        if pred != gt:
            continue
        typ = r.get("_negative_type") or ("positive" if r.get("_label") == 1 else "unknown")
        correct_by_type[typ].append(i)

    rng = random.Random(SEED)
    picks: List[int] = []
    for typ, n in SAMPLES_PER_TYPE.items():
        pool = correct_by_type.get(typ, [])
        if not pool:
            continue
        # Prefer cases with multi-tool sequences (more interesting for review).
        pool_scored = sorted(
            pool,
            key=lambda i: -len(set(res[i].get("tool_sequence", []))),
        )
        # Pick first `n * 2` most-diverse, then randomly select `n`.
        candidates = pool_scored[: max(n * 3, n)]
        rng.shuffle(candidates)
        picks.extend(candidates[:n])

    # Build markdown.
    out = []
    out.append("# M3 Tool-Agent SFT — Expert Review of Reasoning Chains")
    out.append("")
    out.append(
        "Sampled successful cases from `m3_eval_full_ckpt200.json` (test_1000, "
        f"overall acc 81.4%). Each case shows the model's `<think>` narration, "
        f"the deterministic tool call, and the tool's raw observation. Please flag any "
        f"reasoning that misquotes peaks, mis-assigns shifts, mis-reads tool output, "
        f"or otherwise appears hallucinatory."
    )
    out.append("")
    out.append(f"Sampled: {len(picks)} cases across {len(SAMPLES_PER_TYPE)} types.")
    out.append("")

    for k, idx in enumerate(picks, 1):
        r = res[idx]
        f = extract_final(r.get("answer", ""))
        gt_label = r.get("_label")
        gt_str = "Yes" if gt_label == 1 else "No"
        typ = r.get("_negative_type") or ("positive" if gt_label == 1 else "unknown")
        cand = r.get("case_candidate", "")
        test_rec = test_by_cand.get(cand, {})
        input_parts = parse_input_block(test_rec.get("input", ""))
        gt_smi = test_rec.get("_gt_smiles", "(unknown)")

        out.append(f"---")
        out.append("")
        out.append(f"## Case {k} — `{typ}` (ground truth: **{gt_str}**)")
        out.append("")
        out.append(f"**Candidate SMILES**: `{cand}`  ")
        out.append(f"**GT SMILES**: `{gt_smi}`  ")
        out.append(f"**Model answer**: **{f['final_answer']}**  (confidence {f['confidence']})  ")
        out.append(f"**Model key_evidence**: {f['key_evidence']}")
        out.append("")
        out.append("### Experimental NMR (from input)")
        out.append("```")
        if input_parts["c_nmr_line"]:
            out.append(input_parts["c_nmr_line"])
        if input_parts["h_nmr_line"]:
            out.append(input_parts["h_nmr_line"])
        out.append("```")
        out.append("")
        out.append(f"### Tool sequence: `{r.get('tool_sequence')}`")
        out.append("")

        for t, turn in enumerate(r.get("transcript", []), 1):
            parts = pretty_tool_call(turn.get("assistant", ""))
            action = turn.get("action", {})
            action_type = action.get("type")
            action_name = action.get("name", "")
            out.append(f"#### Turn {t}  (action: `{action_type}` `{action_name}`)")
            out.append("")
            for kind, content in parts:
                if kind == "think":
                    out.append("**Model `<think>`:**")
                    out.append("")
                    out.append("> " + content.replace("\n", "\n> "))
                    out.append("")
                elif kind == "tool_call":
                    out.append("**Tool call:**")
                    out.append("```json")
                    out.append(content)
                    out.append("```")
                    out.append("")
                elif kind == "answer":
                    out.append("**Model `<answer>`:**")
                    out.append("```json")
                    out.append(content)
                    out.append("```")
                    out.append("")
            if action_type == "tool" and turn.get("tool_observation"):
                out.append(f"**Deterministic tool observation** (`{action_name}`):")
                out.append("```json")
                out.append(format_tool_obs(turn["tool_observation"]))
                out.append("```")
                out.append("")

        out.append("### Reviewer notes")
        out.append("- Does the `<think>` accurately describe the experimental peaks?  [ ] yes / [ ] hallucinated peaks / [ ] mis-assigned shifts")
        out.append("- Does the model correctly interpret the tool observation?  [ ] yes / [ ] mis-read / [ ] ignored")
        out.append("- Overall reasoning judgment: [ ] sound / [ ] shaky / [ ] hallucinated")
        out.append("- Free-form comment:")
        out.append("")

    Path(OUT_PATH).write_text("\n".join(out), encoding="utf-8")
    print(f"[write] {OUT_PATH}   ({len(picks)} cases, {sum(len(x) for x in out)} chars)")


if __name__ == "__main__":
    main()
