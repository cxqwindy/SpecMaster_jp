"""Generate Chain-of-Thought reasoning traces using Gemini-2.5-Pro.

Reads matching benchmark JSONL files, samples records, calls Gemini to
generate structured reasoning, and appends each result to output JSONL
immediately (crash-safe).

Usage::

    python -m perturbation_generator.generate_cot \
        --output_dir /fs_mol/cxq/SpecMaster_jp/cot_output/ \
        --n_total 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion

litellm.drop_params = True

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

MODEL = "openai/specxmaster/gemini-3.1-pro-preview"
API_BASE = "https://ai-gateway-global.dp.tech"

# ---------------------------------------------------------------------------
# Prompt templates (3 reasoning levels, all English)
# ---------------------------------------------------------------------------

# Full CoT: detailed multi-step reasoning (10-20% of data)
SYSTEM_PROMPT_FULL = """You are an NMR spectrum-structure matching expert. Given NMR experimental data and a candidate molecular structure (SMILES), perform structured reasoning to determine whether the candidate matches the NMR spectrum.

Note: You do NOT have the target molecular formula. You must judge solely by analyzing the consistency between the candidate's chemical environment and the NMR spectral data.

Output ONLY a JSON object in the following format (no other text):

{
  "steps": [
    {"action": "<action_type>", "observation": "<what you observe>", "conclusion": "<conclusion for this step>"},
    ...
  ],
  "final_answer": "Yes" or "No",
  "confidence": <float 0.0-1.0>,
  "key_evidence": "<one sentence summarizing the key evidence>"
}

Available action types:
- count_carbons: Count unique carbon environments in the candidate vs 13C NMR peak count
- count_hydrogens: Count total hydrogens in the candidate vs 1H NMR integral sum
- analyze_c_nmr: Analyze whether 13C chemical shift regions match the candidate's carbon environments
- analyze_h_nmr: Analyze whether 1H chemical shifts, splitting patterns, and coupling constants match the candidate
- check_substructure: Verify specific functional groups/substructures against NMR features
- conclude: Synthesize all evidence for final judgment

Guidelines:
1. Reason step-by-step from simple to complex
2. First check atom counts (C, H) against the spectrum
3. Then analyze chemical shift region matching (aromatic, aliphatic, carbonyl)
4. Finally verify fine-grained functional group / substructure correspondence
5. If a step clearly shows mismatch, conclude early without completing all steps"""

# Medium CoT: concise key-point reasoning (60-70% of data)
SYSTEM_PROMPT_MEDIUM = """You are an NMR spectrum-structure matching expert. Given NMR data and a candidate SMILES, determine whether they match.

You do NOT have the target molecular formula. Judge only from NMR data consistency.

Output ONLY a JSON object:

{
  "steps": [
    {"action": "<action_type>", "observation": "<brief observation>", "conclusion": "<brief conclusion>"}
  ],
  "final_answer": "Yes" or "No",
  "confidence": <float 0.0-1.0>,
  "key_evidence": "<one sentence>"
}

Actions: count_carbons, count_hydrogens, analyze_c_nmr, analyze_h_nmr, check_substructure, conclude

Keep each step to 1-2 sentences. Use at most 3-4 steps total. Focus on the most decisive evidence. No verbose explanations."""

# Short evidence: minimal reasoning (20-30% of data)
SYSTEM_PROMPT_SHORT = """You are an NMR spectrum-structure matching expert. Given NMR data and a candidate SMILES, determine whether they match.

You do NOT have the target molecular formula.

Output ONLY a JSON object:

{
  "final_answer": "Yes" or "No",
  "confidence": <float 0.0-1.0>,
  "key_evidence": "<one sentence with the decisive evidence for your judgment>"
}

Be concise. Only output the JSON above, nothing else."""

# Map reasoning level to system prompt
SYSTEM_PROMPTS = {
    "full": SYSTEM_PROMPT_FULL,
    "medium": SYSTEM_PROMPT_MEDIUM,
    "short": SYSTEM_PROMPT_SHORT,
}

# ---------------------------------------------------------------------------
# Tolerant prompt (for positive samples — teach model to accept minor diffs)
# ---------------------------------------------------------------------------

TOLERANCE_ADDENDUM = """

IMPORTANT - Experimental Tolerance Rules:
When judging a MATCH (positive case), keep in mind these real-world tolerances:
- 13C NMR peak count may differ by ±2 from unique carbon count due to overlapping peaks, equivalent carbons, or quaternary carbons with weak signals
- 1H NMR integral sum may differ by ±1-2H due to experimental integration errors or exchangeable protons (OH, NH, COOH)
- Chemical shift predictions have typical errors of ±3-5 ppm for 13C and ±0.3 ppm for 1H
- Symmetric molecules may show fewer 13C peaks than total carbon count
- If the overall spectral PATTERN matches well (correct regions, correct multiplicities), minor count discrepancies should NOT invalidate a match
- Focus on PATTERN consistency rather than exact numerical matching"""

# ---------------------------------------------------------------------------
# Targeted prompts (for specific hard negative types)
# ---------------------------------------------------------------------------

TARGETED_HINTS = {
    "alcohol_to_thiol": """
ANALYSIS HINT - This may involve an O/S substitution (alcohol vs thiol):
- C-OH carbons typically appear at 60-70 ppm (aliphatic) or 150-160 ppm (phenolic)
- C-SH carbons typically appear at 25-35 ppm (aliphatic) or 125-135 ppm (thiophenolic)
- This ~30 ppm difference is a key diagnostic feature in 13C NMR
- OH protons are broad/variable (1-5 ppm), SH protons appear around 1-2 ppm
- Check adjacent carbons: O vs S causes different alpha-carbon shifts
- A heteroatom identity difference is a REAL structural mismatch. The ~30 ppm shift of the attached carbon is far outside any experimental tolerance — if the C-X carbon sits in the wrong region, answer No.""",

    "insert_ch2": """
ANALYSIS HINT - This may involve a CH2 insertion/deletion in a carbon chain:
- An extra CH2 adds exactly 1 carbon to 13C NMR and 2H to 1H NMR integrals
- Count the 13C peaks carefully: if the candidate has N carbons, expect ~N peaks
- Check if the 1H integral sum matches the candidate's total proton count
- Pay attention to the aliphatic region (0-50 ppm in 13C, 0.5-2.5 ppm in 1H)
- Chain length changes affect multiplet patterns of neighboring protons
- CRITICAL: an inserted/deleted CH2 is a REAL structural difference, NOT experimental error. It produces its own distinct 13C peak and 2 extra proton integrals. Do NOT dismiss a one-carbon count mismatch as peak overlap or integration tolerance — if the aliphatic carbon count or total 1H integral is off by exactly the amount of one CH2, that is decisive evidence for No.""",

    "add_methyl_branch": """
ANALYSIS HINT - This may involve an added/removed methyl branch:
- An extra methyl (CH3) branch adds exactly 1 carbon (typically 10-25 ppm) and 3H (a ~0.8-1.0 ppm doublet/singlet) to the spectrum
- Branching also SHIFTS the adjacent CH/CH2 carbons (branch point moves downfield, ~5-10 ppm)
- Count aliphatic carbons carefully and look for an unexplained upfield methyl peak
- CRITICAL: an added methyl branch is a REAL structural difference, NOT experimental error. The extra CH3 carbon and its 3H integral, plus the branch-point shift, are decisive. Do NOT excuse a methyl-sized count mismatch as tolerance — if the candidate has an extra or missing methyl relative to what the peaks support, answer No.""",

    "remove_terminal_methyl": """
ANALYSIS HINT - This may involve a removed/added terminal methyl:
- A terminal CH3 shows a characteristic ~0.8-0.9 ppm triplet (3H) in 1H NMR and a ~14 ppm carbon in 13C
- Removing it deletes exactly 1 carbon and 3H, and turns the neighboring CH2 triplet into a different pattern
- CRITICAL: a terminal methyl difference is a REAL structural mismatch. If the diagnostic upfield triplet / 14 ppm carbon is present in the spectrum but absent in the candidate (or vice versa), answer No — this is not within experimental tolerance.""",

    "func_group_repl": """
ANALYSIS HINT - This may involve a functional group swap (ester/amide/ketone/aldehyde interconversion):
- Ester C=O: carbonyl carbon at ~165-172 ppm; -OCH2/OCH3 gives a distinct 3.6-4.2 ppm / 50-65 ppm signal
- Amide C=O: carbonyl carbon at ~165-175 ppm; N-H broad proton (5-8 ppm), N-CH shifts differ from O-CH
- Ketone C=O: carbonyl carbon at ~195-215 ppm, NO attached O/N-CH, NO aldehyde proton
- Aldehyde C=O: carbonyl carbon at ~190-205 ppm PLUS a highly diagnostic ~9-10 ppm CHO proton (1H)
- CRITICAL: these groups differ sharply in carbonyl carbon position AND in diagnostic protons (ester OCH vs amide NH vs aldehyde CHO). An ester↔amide or ketone↔aldehyde swap changes the chemical environment decisively — this is NOT experimental tolerance. If the carbonyl region or the diagnostic proton (e.g. the ~9.7 ppm CHO for aldehyde, the OCH3 for ester) does not match the candidate's functional group, answer No.""",

    "O_to_S": """
ANALYSIS HINT - This may involve an O→S or S→O heteroatom swap:
- Ether oxygen (C-O-C): adjacent carbons at 60-80 ppm
- Thioether sulfur (C-S-C): adjacent carbons at 25-40 ppm
- Ester C=O with -O-: carbonyl at ~165-170 ppm
- Thioester C=O with -S-: carbonyl at ~190-200 ppm
- This shift difference is highly diagnostic
- A heteroatom identity difference is a REAL mismatch, not tolerance. The 20-40 ppm shift of the attached carbon is decisive — if it sits in the wrong region, answer No.""",

    "similar_structure": """
ANALYSIS HINT - The candidate is structurally similar but NOT identical to the correct molecule:
- Pay close attention to the EXACT number of peaks and their positions
- Look for missing or extra functional groups (e.g., a methyl, hydroxyl, halogen)
- Check if aromatic substitution pattern matches (ortho/meta/para)
- Verify that ALL peaks in the spectrum can be accounted for by the candidate""",

    "same_formula": """
ANALYSIS HINT - The candidate has the same molecular formula as the correct structure but different connectivity:
- Same formula means same atom counts — you cannot use simple counting to distinguish
- Focus on chemical shift POSITIONS and PATTERNS
- Different connectivity means different carbon environments and different coupling patterns
- Look for diagnostic functional groups: carbonyl region (160-220 ppm), aromatic pattern, etc.""",

    # --- Deep-discrimination prompts (re-distillation v6) ---
    # These fix the student's failure to distinguish structurally-similar
    # candidates: it learned surface actions (count atoms, list shifts) but not
    # connectivity verification, so it says Yes to isomers/near-structures.

    "deep_same_formula": """
CRITICAL REASONING REQUIREMENT — this candidate is a CONSTITUTIONAL ISOMER of the true structure (identical molecular formula, DIFFERENT connectivity).
- Atom counts are IDENTICAL by definition, so counting carbons/hydrogens or matching peak COUNT proves NOTHING. You are FORBIDDEN from using count agreement as evidence of a match.
- The ONLY way to decide is connectivity: for each distinct carbon/proton environment the candidate predicts, check whether its EXACT chemical shift and coupling pattern is actually present in the spectrum.
- Work through the candidate's connectivity: which group is bonded to which. Predict the shift each connectivity implies (e.g. a CH2 between two carbonyls vs a CH2 next to an ether O sit ~40 ppm apart), then test it against the observed peaks.
- Find the SPECIFIC carbon or proton environment where the candidate's connectivity predicts a shift/multiplicity that is NOT observed. That mismatch is the decisive evidence for No.
- Only answer Yes if EVERY environment the candidate's connectivity predicts is present at the right shift AND no observed peak is left unexplained.""",

    "deep_similar": """
CRITICAL REASONING REQUIREMENT — this candidate is structurally SIMILAR but NOT identical to the true structure.
- Do NOT stop at "the main functional groups are all present" — a similar structure shares most groups, so a checklist of groups will always look satisfied. That is exactly the trap.
- You MUST verify EACH substructure of the candidate against the spectrum, and hunt specifically for: a group that is present/absent, a substituent in a different position, a different aromatic substitution pattern (ortho/meta/para → different H count & coupling), a ring size or chain length difference.
- For the region that distinguishes this candidate from close analogs, predict the exact expected shifts/multiplicities and confirm each against observed peaks.
- Identify the SPECIFIC peak that is either missing (candidate predicts it but spectrum lacks it) or unexplained (spectrum shows it but candidate cannot account for it). That is the decisive evidence for No.
- Only answer Yes if every substructure is positively confirmed and every observed peak is explained.""",

    "deep_positive": """
CONFIDENCE CALIBRATION — this candidate IS the correct structure; your job is to build the POSITIVE confirmation that justifies Yes.
- Go environment by environment: for each distinct carbon and proton the structure predicts, point to the observed peak that confirms it (right region, right multiplicity, right integral).
- Distinguish REAL structural evidence from experimental noise: small shift deviations (±3-5 ppm for 13C, ±0.3 ppm for 1H), overlapping equivalent carbons, or ±1-2H integral error from exchangeable protons (OH/NH) are NORMAL and must NOT be used to reject a genuine match.
- Do NOT nitpick a well-matching spectrum into a No over a minor count or shift discrepancy that experimental tolerance fully explains. A structure whose every key environment is accounted for IS a match.
- Answer Yes with high confidence when the full set of environments is consistently explained; reserve doubt only for a genuine, unexplainable structural contradiction.""",
}


def build_user_prompt(record: Dict[str, Any], provide_label: bool = False) -> str:
    """Build the user prompt from a matching record.

    If provide_label=True, tells the model the correct answer and asks
    it to generate a reasoning chain that arrives at that answer.
    """
    candidate = record["candidate_smiles"]

    # Format C NMR peaks
    c_peaks = record.get("c_nmr_peaks") or []
    c_values = []
    for p in c_peaks:
        v = p.get("delta (ppm)")
        if v is not None:
            c_values.append(f"{float(v):.1f}")
    c_nmr_str = ", ".join(c_values) if c_values else "N/A"

    # Format H NMR peaks
    h_peaks = record.get("h_nmr_peaks") or []
    h_parts = []
    for p in h_peaks:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        cat = p.get("category", "")
        nH = p.get("nH", 1)
        j_vals = p.get("j_values", "")
        # Clean j_values
        if j_vals and j_vals != "_":
            j_str = j_vals.replace("_", ", ").strip(", ")
            h_parts.append(f"{float(delta):.2f} ({cat}, J={j_str}, {nH}H)")
        else:
            h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
    h_nmr_str = ", ".join(h_parts) if h_parts else "N/A"

    freq = record.get("nmr_frequency", "")
    solvent = record.get("nmr_solvent", "")
    header = f" ({freq}, {solvent})" if freq else ""

    prompt = f"""Determine whether the following candidate structure matches the NMR data:

Candidate SMILES: {candidate}

13C NMR{header}: {c_nmr_str}
1H NMR{header}: {h_nmr_str}"""

    if provide_label:
        label = record.get("label", 0)
        answer = "Yes" if label == 1 else "No"
        prompt += f"""

The correct answer is: {answer}
Generate a detailed reasoning chain that arrives at this conclusion. Explain WHY the structure does {"" if label == 1 else "NOT "}match the NMR data."""
    else:
        prompt += "\n\nOutput your structured reasoning as JSON."

    return prompt


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict]:
    """Robustly extract JSON from LLM response text."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown code block wrappers
    import re
    # Match ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find the first { ... } block (greedy from first { to last })
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def call_gemini(record: Dict[str, Any], reasoning_level: str = "full",
                max_retries: int = 3,
                tolerant: bool = False,
                targeted_hint: str = "",
                provide_label: bool = False) -> Optional[Dict]:
    """Call Gemini to generate CoT for one record.

    reasoning_level: "full", "medium", or "short"
    tolerant: if True, append tolerance rules (for positive samples)
    targeted_hint: key into TARGETED_HINTS dict (for hard negatives)
    provide_label: if True, tell the model the correct answer
    """
    user_prompt = build_user_prompt(record, provide_label=provide_label)
    system_prompt = SYSTEM_PROMPTS[reasoning_level]

    # Append tolerance rules for positive samples
    if tolerant:
        system_prompt += TOLERANCE_ADDENDUM

    # Append targeted hint for hard negatives
    if targeted_hint and targeted_hint in TARGETED_HINTS:
        system_prompt += TARGETED_HINTS[targeted_hint]

    api_key = require_gateway_api_key()
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=MODEL,
                api_key=api_key,
                api_base=API_BASE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=16384,
                temperature=0.3,
            )
            text = resp.choices[0].message.content or ""

            # Also check for thinking content (Gemini may return it separately)
            if hasattr(resp.choices[0].message, 'reasoning_content'):
                thinking = resp.choices[0].message.reasoning_content or ""
                if thinking and not text:
                    text = thinking

            reasoning = _extract_json(text)

            if reasoning is None:
                logger.warning("JSON extraction failed (attempt %d), raw[:%d]: %s",
                               attempt + 1, len(text), text[:300])
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

            # Validate structure — short mode doesn't require "steps"
            if "final_answer" not in reasoning:
                logger.warning("Missing final_answer, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

            return reasoning

        except json.JSONDecodeError as e:
            logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(5)

    return None


def make_output_record(record: Dict[str, Any], reasoning: Dict) -> Dict:
    """Assemble the final output record."""
    # Simplify c_nmr_peaks for output
    c_peaks = record.get("c_nmr_peaks") or []
    c_values = [p.get("delta (ppm)") for p in c_peaks if p.get("delta (ppm)") is not None]

    # Simplify h_nmr_peaks for output
    h_peaks = record.get("h_nmr_peaks") or []
    h_simplified = []
    for p in h_peaks:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        h_simplified.append({
            "delta": float(delta),
            "category": p.get("category", ""),
            "nH": p.get("nH", 1),
            "j_values": p.get("j_values", ""),
        })

    return {
        "input": {
            "candidate_smiles": record["candidate_smiles"],
            "c_nmr_peaks": c_values,
            "h_nmr_peaks": h_simplified,
            "nmr_frequency": record.get("nmr_frequency", ""),
            "nmr_solvent": record.get("nmr_solvent", ""),
        },
        "label": record["label"],
        "gt_smiles": record["gt_smiles"],
        "negative_type": record.get("negative_type", "positive"),
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Data loading and sampling
# ---------------------------------------------------------------------------

def load_all_data() -> Dict[str, List[Dict]]:
    """Load all matching data from the three output directories."""
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    data = {
        "perturbation": [],
        "random": [],
        "same_formula": [],
        "similar_structure": [],
        "hard_nmr": [],
        "positive": [],
    }

    # 1. Perturbation negatives
    path = base / "perturbation_output" / "perturbed_100_matching_benchmark.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] == 0:
                    data["perturbation"].append(rec)
                else:
                    data["positive"].append(rec)

    # 2-4. Database negatives
    path = base / "negative_output" / "matching_100_benchmark.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] == 1:
                    if not data["positive"]:  # avoid duplicates
                        data["positive"].append(rec)
                    continue
                ntype = rec.get("negative_type", "")
                if "random" in ntype:
                    data["random"].append(rec)
                elif "same_formula" in ntype:
                    data["same_formula"].append(rec)
                elif "similar_structure" in ntype:
                    data["similar_structure"].append(rec)

    # 5. Hard NMR negatives
    path = base / "hard_negative_output" / "hard_neg_100_benchmark.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] == 0:
                    data["hard_nmr"].append(rec)
                else:
                    # Deduplicate positives
                    gt = rec.get("gt_smiles", "")
                    if not any(p.get("gt_smiles") == gt for p in data["positive"]):
                        data["positive"].append(rec)

    return data


def sample_records(
    data: Dict[str, List[Dict]],
    n_total: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """Sample records proportionally from all categories.

    Each record gets a ``_reasoning_level`` field assigned:
      - "full"   (15% of samples): detailed multi-step CoT
      - "medium" (60% of samples): concise key-point reasoning
      - "short"  (25% of samples): one-sentence evidence only
    """
    rng = random.Random(seed)

    positives = list(data["positive"])
    neg_types = ["perturbation", "random", "same_formula",
                 "similar_structure", "hard_nmr"]

    # Decide how many of each
    if n_total <= len(positives):
        n_pos = max(1, n_total // 6)
        n_neg = n_total - n_pos
    else:
        n_pos = len(positives)
        n_neg = n_total - n_pos

    # Sample positives
    sampled = rng.sample(positives, min(n_pos, len(positives)))

    # Distribute negatives evenly across types
    n_per_type = max(1, n_neg // len(neg_types))
    for ntype in neg_types:
        pool = data[ntype]
        if not pool:
            continue
        k = min(n_per_type, len(pool))
        sampled.extend(rng.sample(pool, k))

    # Trim to n_total if overshot
    if len(sampled) > n_total:
        sampled = sampled[:n_total]

    rng.shuffle(sampled)

    # Assign reasoning levels: 15% full, 60% medium, 25% short
    n = len(sampled)
    n_full = max(1, int(n * 0.15))
    n_short = max(1, int(n * 0.25))
    n_medium = n - n_full - n_short

    levels = (["full"] * n_full + ["medium"] * n_medium + ["short"] * n_short)
    rng.shuffle(levels)
    for rec, level in zip(sampled, levels):
        rec["_reasoning_level"] = level

    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate CoT reasoning traces using Gemini-2.5-Pro")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_total", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--mode", type=str, default="standard",
                   choices=["standard", "tolerant", "targeted", "new_mols",
                            "negatives_only", "regen_weak", "regen_deep"],
                   help="Generation mode: standard (balanced sampling), "
                        "tolerant (positive samples with tolerance prompt), "
                        "targeted (fresh hard negatives with hints), "
                        "regen_weak (re-generate v4 weak-type negatives with sharpened hints), "
                        "negatives_only (all negative types for consistent regeneration), "
                        "new_mols (new molecules from LMDB)")
    p.add_argument("--lmdb_path", type=str, default=None,
                   help="LMDB path for new_mols mode")
    p.add_argument("--provide_label", action="store_true",
                   help="Tell the model the correct answer, only generate reasoning chain")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _get_targeted_hint_key(negative_type: str) -> str:
    """Map negative_type to a TARGETED_HINTS key."""
    if "alcohol_to_thiol" in negative_type or "thiol_to_alcohol" in negative_type:
        return "alcohol_to_thiol"
    elif "insert_ch2" in negative_type:
        return "insert_ch2"
    elif "add_methyl_branch" in negative_type:
        return "add_methyl_branch"
    elif "remove_terminal_methyl" in negative_type:
        return "remove_terminal_methyl"
    elif "functional_group_replacement" in negative_type:
        return "func_group_repl"
    elif "O_to_S" in negative_type or "S_to_O" in negative_type:
        return "O_to_S"
    elif "similar_structure" in negative_type:
        return "similar_structure"
    elif "same_formula" in negative_type:
        return "same_formula"
    return ""


def _load_targeted_negatives(n_total: int, seed: int) -> List[Dict]:
    """Load unused hard negative records for targeted mode.

    Focuses on the negative types the SFT model is weakest at (functional
    group swaps, methyl-branch and CH2 chain edits, heteroatom swaps).
    Excludes any (candidate, gt) pair already present in the v4 training
    set AND any molecule in the 1000-sample test set (no leakage).
    """
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    rng = random.Random(seed)

    # Load existing CoT pairs to exclude (v4 = current training set).
    used_pairs = set()
    for cot_name in ["cot_traces_v4.jsonl", "cot_traces_enhanced_v2.jsonl",
                     "cot_traces_enhanced.jsonl", "cot_traces_1500.jsonl"]:
        cot_file = base / "cot_output" / cot_name
        if not cot_file.exists():
            continue
        with open(cot_file) as f:
            for line in f:
                rec = json.loads(line)
                inp = rec.get("input", {})
                used_pairs.add((inp.get("candidate_smiles", ""), rec.get("gt_smiles", "")))

    # Exclude the exact (candidate, gt) PAIRS used in the test set. Pair-level
    # isolation (gt molecules may overlap; the specific pair judged at test
    # time must not appear in training) — matches how v4 was built.
    test_pairs = set()
    test_file = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json")
    if test_file.exists():
        with open(test_file) as f:
            for t in json.load(f):
                test_pairs.add((t.get("_candidate_smiles", ""), t.get("_gt_smiles", "")))

    # Target distribution: prioritize the weakest eval types.
    targets = {
        "func_group_repl": 60,      # eval 0-33% — worst, no prior hint
        "add_methyl_branch": 60,    # eval 16.7% — no prior hint
        "insert_ch2": 80,           # eval 33-37% — despite 315 existing
        "O_to_S": 40,               # eval 50%
        "alcohol_to_thiol": 20,     # eval 50%
        "remove_terminal_methyl": 15,
        "similar_structure": 40,    # eval 55.5%
    }

    pools = {k: [] for k in targets}

    # Load from all matching files
    for jsonl_path in [
        base / "perturbation_output" / "perturbed_100_matching_benchmark.jsonl",
        base / "negative_output" / "matching_100_benchmark.jsonl",
        base / "hard_negative_output" / "hard_neg_100_benchmark.jsonl",
    ]:
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] != 0:
                    continue
                cand = rec["candidate_smiles"]
                gt = rec["gt_smiles"]
                if (cand, gt) in used_pairs:
                    continue
                if (cand, gt) in test_pairs:
                    continue
                hint_key = _get_targeted_hint_key(rec.get("negative_type", ""))
                if hint_key in pools:
                    pools[hint_key].append(rec)

    sampled = []
    for hint_key, target_n in targets.items():
        pool = pools[hint_key]
        k = min(target_n, len(pool))
        logger.info("  targeted %s: %d available, taking %d", hint_key, len(pool), k)
        if k > 0:
            chosen = rng.sample(pool, k)
            for r in chosen:
                r["_targeted_hint"] = hint_key
                r["_reasoning_level"] = "medium"
            sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled[:n_total] if n_total else sampled


def _v4_to_raw(rec: Dict) -> Dict:
    """Convert a simplified v4 CoT record back to the raw benchmark format
    that build_user_prompt / make_output_record expect."""
    inp = rec.get("input", {})
    # c_nmr_peaks in v4 is a flat list of floats -> wrap as [{"delta (ppm)": v}]
    c_raw = [{"delta (ppm)": v} for v in (inp.get("c_nmr_peaks") or [])]
    # h_nmr_peaks in v4 is a list of dicts with "delta" -> pass through (build
    # prompt reads p.get("centroid") or p.get("delta"))
    h_raw = inp.get("h_nmr_peaks") or []
    return {
        "candidate_smiles": inp.get("candidate_smiles", ""),
        "gt_smiles": rec.get("gt_smiles", ""),
        "label": rec.get("label", 0),
        "c_nmr_peaks": c_raw,
        "h_nmr_peaks": h_raw,
        "nmr_frequency": inp.get("nmr_frequency", ""),
        "nmr_solvent": inp.get("nmr_solvent", ""),
        "negative_type": rec.get("negative_type", "positive"),
    }


def _load_regen_weak(n_total: int, seed: int) -> List[Dict]:
    """Load the weak-type negatives from the v4 training set for
    re-generation with sharpened, tolerance-countering hints.

    These are the negative types the SFT model scored worst on
    (functional-group swaps, methyl/CH2 chain edits, heteroatom swaps).
    We re-use the exact same (candidate, gt) pairs already in v4 — no new
    molecules, no leakage — but regenerate their reasoning chains with the
    targeted hints so the model learns these are REAL differences.
    """
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    rng = random.Random(seed)

    # Which coarse hint keys count as "weak" and should be regenerated.
    weak_keys = {
        "insert_ch2", "add_methyl_branch", "remove_terminal_methyl",
        "func_group_repl", "O_to_S", "alcohol_to_thiol",
    }

    samples = []
    with open(base / "cot_output" / "cot_traces_v4.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("label") != 0:
                continue
            hint_key = _get_targeted_hint_key(rec.get("negative_type", ""))
            if hint_key not in weak_keys:
                continue
            raw = _v4_to_raw(rec)
            raw["_targeted_hint"] = hint_key
            raw["_reasoning_level"] = "medium"
            samples.append(raw)

    rng.shuffle(samples)
    from collections import Counter
    dist = Counter(s["_targeted_hint"] for s in samples)
    logger.info("regen_weak pool by hint: %s", dict(dist))
    return samples[:n_total] if n_total else samples


def _load_regen_deep(n_total: int, seed: int) -> List[Dict]:
    """Load same_formula / similar_structure / positive samples from v4 for
    re-generation with DEEP-DISCRIMINATION prompts.

    Diagnosis (teacher 94% vs student 72.9%): the student learned surface
    reasoning (count atoms, list shifts) but not connectivity verification, so
    it says Yes to isomers/near-structures (same_formula/similar FP) and,
    lacking positive confirmation, says No to true matches (positive FN). These
    three types hold 75% of the recoverable gap. We regenerate their reasoning
    with prompts that force connectivity-level discrimination; each type gets a
    prompt calibrated to its error direction.
    """
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    rng = random.Random(seed)

    def deep_key(rec):
        nt = rec.get("negative_type", "")
        if rec.get("label") == 1:
            return "deep_positive"
        if "same_formula" in nt:
            return "deep_same_formula"
        if "similar_structure" in nt:
            return "deep_similar"
        return None

    samples = []
    with open(base / "cot_output" / "cot_traces_v4.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            k = deep_key(rec)
            if k is None:
                continue
            raw = _v4_to_raw(rec)
            raw["_targeted_hint"] = k
            # positive uses tolerance mindset (confirm), negatives use full CoT
            # so the deeper connectivity reasoning has room to unfold.
            raw["_reasoning_level"] = "full"
            samples.append(raw)

    rng.shuffle(samples)
    from collections import Counter
    dist = Counter(s["_targeted_hint"] for s in samples)
    logger.info("regen_deep pool by prompt: %s", dict(dist))
    return samples[:n_total] if n_total else samples


def _load_tolerant_positives(n_total: int, seed: int) -> List[Dict]:
    """Load all positive samples for tolerant mode re-generation."""
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    rng = random.Random(seed)

    positives = []
    for jsonl_path in [
        base / "perturbation_output" / "perturbed_100_matching_benchmark.jsonl",
        base / "negative_output" / "matching_100_benchmark.jsonl",
        base / "hard_negative_output" / "hard_neg_100_benchmark.jsonl",
    ]:
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["label"] == 1:
                    positives.append(rec)

    # Deduplicate by gt_smiles
    seen = set()
    unique = []
    for r in positives:
        gt = r.get("gt_smiles", r.get("candidate_smiles", ""))
        if gt not in seen:
            seen.add(gt)
            unique.append(r)

    # Assign levels: 50% full, 50% medium
    sampled = unique[:n_total]
    for i, r in enumerate(sampled):
        r["_reasoning_level"] = "full" if i % 2 == 0 else "medium"
        r["_tolerant"] = True

    return sampled


def _load_new_mols(lmdb_path: str, n_total: int, seed: int) -> List[Dict]:
    """Load new molecules from LMDB (excluding existing 100)."""
    from .lmdb_reader import read_lmdb
    rng = random.Random(seed)

    # Get existing gt_smiles to exclude (from all CoT files + test set)
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    existing_gt = set()
    for cot_name in ["cot_traces_1500.jsonl", "cot_traces_enhanced.jsonl"]:
        cot_file = base / "cot_output" / cot_name
        if cot_file.exists():
            with open(cot_file) as f:
                for line in f:
                    rec = json.loads(line)
                    existing_gt.add(rec.get("gt_smiles", ""))

    # Also exclude test set molecules
    test_file = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json")
    if test_file.exists():
        import json as _json
        with open(test_file) as f:
            test_data = _json.load(f)
        for t in test_data:
            inp = t.get("input", "")
            if "Candidate SMILES:" in inp:
                smi = inp.split("Candidate SMILES:")[1].split("\n")[0].strip()
                existing_gt.add(smi)

    # Read LMDB
    all_records = read_lmdb(lmdb_path, require_nmr=True)
    new_records = [r for r in all_records
                   if r["smiles_canonical"] not in existing_gt]

    logger.info("LMDB: %d total, %d new (excluding %d existing)",
                len(all_records), len(new_records), len(existing_gt))

    # Convert to matching format (positive: candidate = gt)
    sampled = []
    for r in new_records[:n_total]:
        sampled.append({
            "candidate_smiles": r["smiles_canonical"],
            "gt_smiles": r["smiles_canonical"],
            "gt_formula": r["molecular_formula"],
            "label": 1,
            "c_nmr_peaks": r.get("c_nmr_peaks", []),
            "h_nmr_peaks": r.get("h_nmr_peaks", []),
            "nmr_frequency": r.get("nmr_frequency", ""),
            "nmr_solvent": r.get("nmr_solvent", ""),
            "negative_type": "positive",
            "_reasoning_level": "medium",
            "_tolerant": True,
        })

    return sampled


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cot_traces.jsonl"

    # --- Load data based on mode ---
    if args.mode == "standard":
        logger.info("Mode: standard (balanced sampling)")
        data = load_all_data()
        for k, v in data.items():
            logger.info("  %s: %d records", k, len(v))
        samples = sample_records(data, n_total=args.n_total, seed=args.seed)
    elif args.mode == "tolerant":
        logger.info("Mode: tolerant (positive samples with tolerance prompt)")
        samples = _load_tolerant_positives(args.n_total, args.seed)
    elif args.mode == "targeted":
        logger.info("Mode: targeted (hard negatives with hints)")
        samples = _load_targeted_negatives(args.n_total, args.seed)
    elif args.mode == "regen_weak":
        logger.info("Mode: regen_weak (re-generate v4 weak-type negatives with sharpened hints)")
        samples = _load_regen_weak(args.n_total, args.seed)
    elif args.mode == "regen_deep":
        logger.info("Mode: regen_deep (re-generate same_formula/similar/positive with deep-discrimination prompts)")
        samples = _load_regen_deep(args.n_total, args.seed)
    elif args.mode == "new_mols":
        logger.info("Mode: new_mols (new molecules from LMDB)")
        if not args.lmdb_path:
            args.lmdb_path = "/fs_mol/fjxu/proj_data/Spectra/NMR/si_0820/lmdb_dataset/test_1000sample.lmdb"
        samples = _load_new_mols(args.lmdb_path, args.n_total, args.seed)
    elif args.mode == "negatives_only":
        logger.info("Mode: negatives_only (all negative types)")
        data = load_all_data()
        neg_types = ["perturbation", "random", "same_formula", "similar_structure", "hard_nmr"]
        all_negs = []
        for ntype in neg_types:
            all_negs.extend(data[ntype])
            logger.info("  %s: %d records", ntype, len(data[ntype]))
        rng = random.Random(args.seed)
        rng.shuffle(all_negs)
        samples = all_negs[:args.n_total]
        # Assign reasoning levels: 15% full, 60% medium, 25% short
        n = len(samples)
        n_full = max(1, int(n * 0.15))
        n_short = max(1, int(n * 0.25))
        n_medium = n - n_full - n_short
        levels = ["full"] * n_full + ["medium"] * n_medium + ["short"] * n_short
        rng.shuffle(levels)
        for rec, level in zip(samples, levels):
            rec["_reasoning_level"] = level
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info("Sampled %d records (pos=%d, neg=%d)",
                len(samples),
                sum(1 for s in samples if s.get("label") == 1),
                sum(1 for s in samples if s.get("label") == 0))

    # Log reasoning level distribution
    from collections import Counter
    level_dist = Counter(s.get("_reasoning_level", "full") for s in samples)
    logger.info("Reasoning levels: %s", dict(level_dist))

    # --- Resume support ---
    done_count = 0
    if output_file.exists():
        with open(output_file) as f:
            done_count = sum(1 for _ in f)
        logger.info("Resuming from record %d (already done)", done_count)

    # --- Generate CoT ---
    n_success = 0
    n_fail = 0

    with open(output_file, "a", encoding="utf-8") as fout:
        for i, record in enumerate(samples):
            if i < done_count:
                continue  # Skip already processed

            label_str = "POS" if record["label"] == 1 else "NEG"
            ntype = record.get("negative_type", "positive")
            level = record.get("_reasoning_level", "full")
            logger.info("[%d/%d] %s (%s) level=%s - calling Gemini...",
                        i + 1, len(samples), label_str, ntype, level)

            reasoning = call_gemini(
                record,
                reasoning_level=level,
                max_retries=args.max_retries,
                tolerant=record.get("_tolerant", False),
                targeted_hint=record.get("_targeted_hint", ""),
                provide_label=args.provide_label,
            )

            if reasoning is None:
                logger.error("[%d] Failed after retries, skipping", i + 1)
                n_fail += 1
                # Write a placeholder so resume index stays correct
                placeholder = {
                    "input": {"candidate_smiles": record["candidate_smiles"]},
                    "label": record["label"],
                    "gt_smiles": record.get("gt_smiles", ""),
                    "negative_type": ntype,
                    "reasoning_level": level,
                    "reasoning": None,
                    "error": "generation_failed",
                }
                fout.write(json.dumps(placeholder, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            out_record = make_output_record(record, reasoning)
            out_record["reasoning_level"] = level
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()
            n_success += 1

            # Brief pause to be polite to the API
            time.sleep(0.5)

    # --- Summary ---
    total_done = done_count + n_success + n_fail
    logger.info("=" * 50)
    logger.info("Done! Total processed: %d", total_done)
    logger.info("  Success: %d, Failed: %d", done_count + n_success, n_fail)
    logger.info("  Output: %s", output_file)


if __name__ == "__main__":
    main()
