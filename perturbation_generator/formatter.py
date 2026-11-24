"""Output formatting: Alpaca JSON for SFT training + benchmark JSONL.

Converts :class:`PerturbedSample` objects into the two output formats
described in the plan:

1. **Alpaca JSON** — compatible with LLaMA-Factory, with NMR spectral data
   in the instruction and repair reasoning in the output.
2. **Benchmark JSONL** — one JSON object per line, matching the format of
   ``repair_benchmark_merged.jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import PerturbedSample


# ===================================================================
# NMR peak formatting helpers
# ===================================================================

def format_c_nmr_peaks(peaks: Optional[List[Dict]]) -> str:
    """Format ¹³C NMR peaks into a readable string.

    Example output: ``"167.0, 150.1, 149.2, 133.8, 128.5"``
    """
    if not peaks:
        return "N/A"
    values = []
    for p in peaks:
        # Each peak dict may have "delta (ppm)" or "delta" as the key
        delta = p.get("delta (ppm)") or p.get("delta")
        if delta is not None:
            try:
                values.append(f"{float(delta):.1f}")
            except (ValueError, TypeError):
                continue
    if not values:
        return "N/A"
    return ", ".join(values)


def format_h_nmr_peaks(peaks: Optional[List[Dict]]) -> str:
    r"""Format ¹H NMR peaks into a readable string.

    Example output: ``"8.37 (s, 1H), 7.72 (d, J=15.7, 1H), 3.90 (s, 3H)"``
    """
    if not peaks:
        return "N/A"
    parts = []
    for p in peaks:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        try:
            delta_str = f"{float(delta):.2f}"
        except (ValueError, TypeError):
            continue

        category = p.get("category", "")
        nH = p.get("nH")
        j_values = p.get("j_values", "")

        inner_parts = []
        if category:
            inner_parts.append(str(category))
        if j_values:
            inner_parts.append(f"J={j_values}")
        if nH is not None:
            inner_parts.append(f"{nH}H")

        if inner_parts:
            parts.append(f"{delta_str} ({', '.join(inner_parts)})")
        else:
            parts.append(delta_str)

    if not parts:
        return "N/A"
    return ", ".join(parts)


# ===================================================================
# Alpaca format
# ===================================================================

def sample_to_alpaca(
    sample: PerturbedSample,
    *,
    include_nmr: bool = True,
) -> Dict[str, str]:
    """Convert a PerturbedSample to Alpaca training format.

    Parameters
    ----------
    sample : PerturbedSample
    include_nmr : bool
        If True, include NMR spectral data in the instruction.

    Returns
    -------
    dict with keys ``instruction``, ``input``, ``output``.
    """
    # Build instruction
    lines = [
        "Given an incorrect molecular SMILES representation, your task is to:",
        "1. Analyze the structural differences needed to match the target "
        "molecular formula",
        "2. Generate the corrected SMILES representation",
        "3. Provide detailed reasoning for each correction step",
        "",
        f"Input SMILES: {sample.input_smiles}",
        f"Target Formula: {sample.target_formula}",
        f"Required Changes: {sample.formula_delta}",
    ]

    if include_nmr:
        freq = sample.nmr_frequency or ""
        solvent = sample.nmr_solvent or ""

        c_nmr_str = format_c_nmr_peaks(sample.c_nmr_peaks)
        h_nmr_str = format_h_nmr_peaks(sample.h_nmr_peaks)

        header_parts = []
        if freq:
            header_parts.append(freq)
        if solvent:
            header_parts.append(solvent)
        header = ", ".join(header_parts) if header_parts else ""

        if c_nmr_str != "N/A":
            c_label = f"13C NMR ({header})" if header else "13C NMR"
            lines.append(f"\n{c_label}: {c_nmr_str}")
        if h_nmr_str != "N/A":
            h_label = f"1H NMR ({header})" if header else "1H NMR"
            lines.append(f"{h_label}: {h_nmr_str}")

    lines.extend([
        "",
        "Please provide:",
        "1. Reasoning: Step-by-step analysis of required corrections",
        "2. Corrected SMILES: The final corrected molecular structure",
    ])

    instruction = "\n".join(lines)

    # Build output — reasoning is auto-generated from the perturbation type
    # (populated during generation or verification)
    reasoning = _build_auto_reasoning(sample)
    output = f"Reasoning: {reasoning}\n\nCorrected SMILES: {sample.gt_smiles}"

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
    }


def _build_auto_reasoning(sample: PerturbedSample) -> str:
    """Auto-generate repair reasoning from perturbation metadata."""
    ptype = sample.perturbation_type
    delta = sample.formula_delta

    parts = []
    parts.append(f"The input SMILES {sample.input_smiles} has a structural error.")
    parts.append(f"The molecular formula of the input is {sample.input_formula}, "
                 f"but the target formula is {sample.target_formula}.")
    parts.append(f"The required changes are: {delta}.")

    # Add perturbation-specific reasoning
    if "heteroatom" in ptype:
        if "O_to_S" in ptype:
            parts.append("A sulfur atom in the input should be oxygen in the "
                         "correct structure.")
        elif "S_to_O" in ptype:
            parts.append("An oxygen atom in the input should be sulfur in the "
                         "correct structure.")
        elif "ring_O_to_NH" in ptype:
            parts.append("The aromatic ring contains a nitrogen (pyrrole-type) "
                         "that should be oxygen (furan-type) in the correct "
                         "structure.")
        elif "ring_NH_to_O" in ptype:
            parts.append("The aromatic ring contains an oxygen (furan-type) "
                         "that should be nitrogen (pyrrole-type) in the correct "
                         "structure.")
    elif "functional_group" in ptype:
        if "ester_to_amide" in ptype:
            parts.append("The ester linkage (C(=O)-O) has been changed to an "
                         "amide (C(=O)-NH). Replace N with O to restore the ester.")
        elif "amide_to_ester" in ptype:
            parts.append("The amide bond (C(=O)-NH) has been changed to an "
                         "ester (C(=O)-O). Replace O with N to restore the amide.")
        elif "aldehyde_to_ketone" in ptype:
            parts.append("An aldehyde (CHO) has been converted to a methyl "
                         "ketone (COCH3). Remove the extra methyl group.")
        elif "ketone_to_aldehyde" in ptype:
            parts.append("A methyl ketone (COCH3) has been converted to an "
                         "aldehyde (CHO). Add a methyl group to restore the ketone.")
        elif "alcohol_to_thiol" in ptype:
            parts.append("The hydroxyl group (-OH) has been replaced with a "
                         "thiol (-SH). Replace S with O.")
        elif "thiol_to_alcohol" in ptype:
            parts.append("The thiol group (-SH) has been replaced with a "
                         "hydroxyl (-OH). Replace O with S.")
    elif "alkyl_chain" in ptype:
        if "add_methyl" in ptype:
            parts.append("An extra methyl branch has been added. Remove the "
                         "CH3 group to match the target formula.")
        elif "remove_terminal_methyl" in ptype:
            parts.append("A terminal methyl group has been removed. Add a "
                         "CH3 group to match the target formula.")
        elif "insert_ch2" in ptype:
            parts.append("An extra CH2 group has been inserted into the chain. "
                         "Remove one CH2 to match the target formula.")
    elif "protecting_group" in ptype:
        if "TBS_to_TMS" in ptype:
            parts.append("The TBS protecting group has been changed to TMS. "
                         "Add the tert-butyl group back to the silicon atom.")
        elif "TMS_to_TBS" in ptype:
            parts.append("The TMS protecting group has been changed to TBS. "
                         "Remove the tert-butyl group from the silicon atom.")
        elif "remove_acetyl" in ptype:
            parts.append("The acetyl protecting group has been removed. "
                         "Restore the -OC(=O)CH3 group.")

    parts.append(f"The corrected SMILES is {sample.gt_smiles}.")
    return " ".join(parts)


# ===================================================================
# JSONL benchmark format
# ===================================================================

def sample_to_jsonl_record(sample: PerturbedSample) -> Dict[str, Any]:
    """Convert a PerturbedSample to a benchmark JSONL record."""
    # Determine difficulty from edit_lb
    if sample.edit_lb <= 1:
        difficulty = "easy"
    elif sample.edit_lb == 2:
        difficulty = "medium"
    else:
        difficulty = "hard"

    record = {
        "input_smiles": sample.input_smiles,
        "input_formula": sample.input_formula,
        "target_formula": sample.target_formula,
        "formula_delta": sample.formula_delta,
        "edit_lb": sample.edit_lb,
        "gt_smiles": sample.gt_smiles,
        "difficulty": difficulty,
        "mode": "ch",  # Default mode
        "case_id": sample.case_id,
        "source_tool": sample.source_tool,
        "source_rank": 1,
        "perturbation_type": sample.perturbation_type,
    }

    # Add NMR data if present
    if sample.c_nmr_peaks is not None:
        record["c_nmr_peaks"] = sample.c_nmr_peaks
    if sample.h_nmr_peaks is not None:
        record["h_nmr_peaks"] = sample.h_nmr_peaks
    if sample.nmr_frequency:
        record["nmr_frequency"] = sample.nmr_frequency
    if sample.nmr_solvent:
        record["nmr_solvent"] = sample.nmr_solvent

    # Add verification results if available
    if sample.repair_verified is not None:
        record["repair_verified"] = sample.repair_verified
        record["repair_pred_smiles"] = sample.repair_pred_smiles
        record["structurally_correct"] = sample.structurally_correct
        record["fallback_used"] = sample.fallback_used

    # Add Tanimoto similarity for similar-structure negatives
    if sample.tanimoto_similarity is not None:
        record["tanimoto_similarity"] = round(sample.tanimoto_similarity, 4)

    return record


# ===================================================================
# File writers
# ===================================================================

def write_alpaca_json(
    samples: Sequence[PerturbedSample],
    output_path: str,
    *,
    include_nmr: bool = True,
    indent: int = 2,
) -> int:
    """Write samples as Alpaca-format JSON array.

    Returns the number of samples written.
    """
    alpaca_records = [
        sample_to_alpaca(s, include_nmr=include_nmr) for s in samples
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alpaca_records, f, indent=indent, ensure_ascii=False)
    return len(alpaca_records)


def write_benchmark_jsonl(
    samples: Sequence[PerturbedSample],
    output_path: str,
) -> int:
    """Write samples as JSONL (one JSON object per line).

    Returns the number of samples written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            record = sample_to_jsonl_record(s)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_both(
    samples: Sequence[PerturbedSample],
    output_dir: str,
    *,
    prefix: str = "perturbed",
    include_nmr: bool = True,
) -> Dict[str, int]:
    """Write both Alpaca JSON and benchmark JSONL.

    Returns a dict with counts: ``{"alpaca": N, "jsonl": M}``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_alpaca = write_alpaca_json(
        samples,
        str(out_dir / f"{prefix}_alpaca.json"),
        include_nmr=include_nmr,
    )
    n_jsonl = write_benchmark_jsonl(
        samples,
        str(out_dir / f"{prefix}_benchmark.jsonl"),
    )

    return {"alpaca": n_alpaca, "jsonl": n_jsonl}


# ===================================================================
# Spectrum-matching task format (binary classification)
# ===================================================================

def sample_to_matching_alpaca(
    sample: PerturbedSample,
    *,
    is_positive: bool = False,
) -> Dict[str, str]:
    """Convert a sample to spectrum-matching Alpaca format.

    The task is binary classification: given NMR spectral data and a
    candidate SMILES, determine whether they match (Yes/No).

    Parameters
    ----------
    sample : PerturbedSample
        For negatives: ``input_smiles`` is the wrong candidate,
        ``gt_smiles`` is the correct one (used only for positive pairs).
    is_positive : bool
        If True, use ``gt_smiles`` as the candidate (positive pair → "Yes").
        If False, use ``input_smiles`` as the candidate (negative pair → "No").
    """
    candidate_smi = sample.gt_smiles if is_positive else sample.input_smiles

    freq = sample.nmr_frequency or ""
    solvent = sample.nmr_solvent or ""

    c_nmr_str = format_c_nmr_peaks(sample.c_nmr_peaks)
    h_nmr_str = format_h_nmr_peaks(sample.h_nmr_peaks)

    header_parts = []
    if freq:
        header_parts.append(freq)
    if solvent:
        header_parts.append(solvent)
    header = ", ".join(header_parts) if header_parts else ""

    lines = [
        "Given the following NMR spectral data and a candidate molecular "
        "structure (SMILES), determine whether the candidate structure "
        "matches the NMR spectrum.",
        "",
    ]

    if c_nmr_str != "N/A":
        c_label = f"13C NMR ({header})" if header else "13C NMR"
        lines.append(f"{c_label}: {c_nmr_str}")
    if h_nmr_str != "N/A":
        h_label = f"1H NMR ({header})" if header else "1H NMR"
        lines.append(f"{h_label}: {h_nmr_str}")

    lines.extend([
        "",
        f"Candidate SMILES: {candidate_smi}",
        "",
        "Does this molecular structure match the NMR spectrum? "
        "Answer Yes or No.",
    ])

    instruction = "\n".join(lines)
    answer = "Yes" if is_positive else "No"

    return {
        "instruction": instruction,
        "input": "",
        "output": answer,
    }


def sample_to_matching_jsonl(
    sample: PerturbedSample,
    *,
    is_positive: bool = False,
) -> Dict[str, Any]:
    """Convert a sample to spectrum-matching JSONL record."""
    candidate_smi = sample.gt_smiles if is_positive else sample.input_smiles
    candidate_formula = sample.target_formula if is_positive else sample.input_formula

    record = {
        "candidate_smiles": candidate_smi,
        "candidate_formula": candidate_formula,
        "gt_smiles": sample.gt_smiles,
        "gt_formula": sample.target_formula,
        "label": 1 if is_positive else 0,
        "c_nmr_peaks": sample.c_nmr_peaks,
        "h_nmr_peaks": sample.h_nmr_peaks,
        "nmr_frequency": sample.nmr_frequency,
        "nmr_solvent": sample.nmr_solvent,
        "case_id": sample.case_id,
        "source_tool": sample.source_tool,
    }

    # Negative-specific fields
    if not is_positive:
        record["negative_type"] = sample.perturbation_type
        record["formula_delta"] = sample.formula_delta
        record["edit_lb"] = sample.edit_lb
        if sample.tanimoto_similarity is not None:
            record["tanimoto_similarity"] = round(sample.tanimoto_similarity, 4)
        if sample.spectral_similarity is not None:
            record["spectral_similarity"] = round(sample.spectral_similarity, 4)

    return record


def write_matching_alpaca_json(
    negative_samples: Sequence[PerturbedSample],
    output_path: str,
    *,
    include_positives: bool = True,
    indent: int = 2,
) -> Dict[str, int]:
    """Write spectrum-matching Alpaca JSON with positive + negative pairs.

    For each negative sample, optionally also generates a positive pair
    using the same NMR spectrum + correct SMILES.

    Returns dict with counts: ``{"positive": N, "negative": M, "total": T}``.
    """
    records = []
    seen_positive_gt = set()

    for s in negative_samples:
        # Negative pair
        records.append(sample_to_matching_alpaca(s, is_positive=False))

        # Positive pair (deduplicate by gt_smiles)
        if include_positives and s.gt_smiles not in seen_positive_gt:
            seen_positive_gt.add(s.gt_smiles)
            records.append(sample_to_matching_alpaca(s, is_positive=True))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=indent, ensure_ascii=False)

    n_pos = len(seen_positive_gt) if include_positives else 0
    n_neg = len(negative_samples)
    return {"positive": n_pos, "negative": n_neg, "total": len(records)}


def write_matching_jsonl(
    negative_samples: Sequence[PerturbedSample],
    output_path: str,
    *,
    include_positives: bool = True,
) -> Dict[str, int]:
    """Write spectrum-matching JSONL with positive + negative pairs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen_positive_gt = set()
    count_pos = 0
    count_neg = 0

    with open(path, "w", encoding="utf-8") as f:
        for s in negative_samples:
            # Negative
            rec = sample_to_matching_jsonl(s, is_positive=False)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count_neg += 1

            # Positive (dedup by gt_smiles)
            if include_positives and s.gt_smiles not in seen_positive_gt:
                seen_positive_gt.add(s.gt_smiles)
                rec = sample_to_matching_jsonl(s, is_positive=True)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_pos += 1

    return {"positive": count_pos, "negative": count_neg,
            "total": count_pos + count_neg}


def write_matching_both(
    negative_samples: Sequence[PerturbedSample],
    output_dir: str,
    *,
    prefix: str = "matching",
    include_positives: bool = True,
) -> Dict[str, Any]:
    """Write both spectrum-matching Alpaca JSON and JSONL."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_counts = write_matching_alpaca_json(
        negative_samples,
        str(out_dir / f"{prefix}_alpaca.json"),
        include_positives=include_positives,
    )
    jsonl_counts = write_matching_jsonl(
        negative_samples,
        str(out_dir / f"{prefix}_benchmark.jsonl"),
        include_positives=include_positives,
    )

    return {"alpaca": alpaca_counts, "jsonl": jsonl_counts}

