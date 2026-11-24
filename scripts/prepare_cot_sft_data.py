#!/usr/bin/env python3
"""Convert CoT traces JSONL to Alpaca format for LLaMA-Factory SFT.

Reads cot_traces_1500.jsonl, converts to {instruction, input, output},
splits train/val, saves JSON files, and updates dataset_info.json.

Usage:
    python prepare_cot_sft_data.py
"""

import json
import random
from pathlib import Path

# Paths
COT_FILE = "/fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v6.jsonl"
LLAMA_FACTORY_DATA = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data"
TRAIN_RATIO = 0.85
SEED = 42

INSTRUCTION = (
    "Given NMR spectral data and a candidate molecular structure (SMILES), "
    "determine whether the candidate matches the NMR spectrum. "
    "Provide your reasoning as a structured JSON object."
)


def format_input(record):
    """Build the input field from a CoT record."""
    inp = record["input"]
    candidate = inp["candidate_smiles"]

    # Format C NMR
    c_peaks = inp.get("c_nmr_peaks") or []
    c_str = ", ".join(f"{v:.1f}" for v in c_peaks) if c_peaks else "N/A"

    # Format H NMR
    h_peaks = inp.get("h_nmr_peaks") or []
    h_parts = []
    for p in h_peaks:
        delta = p.get("delta")
        if delta is None:
            continue
        cat = p.get("category", "")
        nH = p.get("nH", 1)
        j_vals = p.get("j_values", "")
        if j_vals and j_vals != "_":
            j_str = j_vals.replace("_", ", ").strip(", ")
            h_parts.append(f"{float(delta):.2f} ({cat}, J={j_str}, {nH}H)")
        else:
            h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
    h_str = ", ".join(h_parts) if h_parts else "N/A"

    freq = inp.get("nmr_frequency", "")
    solvent = inp.get("nmr_solvent", "")
    header = f" ({freq}, {solvent})" if freq else ""

    return (
        f"Candidate SMILES: {candidate}\n\n"
        f"13C NMR{header}: {c_str}\n"
        f"1H NMR{header}: {h_str}"
    )


def format_output(record):
    """Build the output field from reasoning."""
    reasoning = record.get("reasoning")
    if reasoning is None:
        return '{"final_answer": "Unknown", "confidence": 0.0, "key_evidence": "generation failed"}'
    return json.dumps(reasoning, ensure_ascii=False)


def main():
    # Load CoT data
    records = []
    with open(COT_FILE) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning") is None:
                continue  # Skip failed records
            records.append(rec)

    print(f"Loaded {len(records)} valid CoT records")

    # Convert to Alpaca format with label-consistency filtering
    alpaca_pos = []
    alpaca_neg = []
    filtered_inconsistent = 0
    for rec in records:
        # Check if Gemini's answer matches the ground truth label
        reasoning = rec.get("reasoning", {})
        gemini_answer = reasoning.get("final_answer", "").strip()
        expected_answer = "Yes" if rec["label"] == 1 else "No"

        if gemini_answer and gemini_answer != expected_answer:
            filtered_inconsistent += 1
            continue  # Skip: Gemini's reasoning contradicts the label

        entry = {
            "instruction": INSTRUCTION,
            "input": format_input(rec),
            "output": format_output(rec),
        }
        if rec["label"] == 1:
            alpaca_pos.append(entry)
        else:
            alpaca_neg.append(entry)

    print(f"Filtered {filtered_inconsistent} label-inconsistent records "
          f"(Gemini answer != ground truth label)")
    print(f"After filtering: pos={len(alpaca_pos)}, neg={len(alpaca_neg)}, "
          f"ratio={len(alpaca_pos)/(len(alpaca_pos)+len(alpaca_neg))*100:.0f}%")

    # No oversampling — use natural distribution
    alpaca_records = alpaca_pos + alpaca_neg
    print(f"Training data (no oversample): pos={len(alpaca_pos)}, neg={len(alpaca_neg)}")

    # Shuffle and split
    rng = random.Random(SEED)
    indices = list(range(len(alpaca_records)))
    rng.shuffle(indices)

    n_train = int(len(indices) * TRAIN_RATIO)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = [alpaca_records[i] for i in train_indices]
    val_data = [alpaca_records[i] for i in val_indices]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Save
    data_dir = Path(LLAMA_FACTORY_DATA)

    train_path = data_dir / "nmr_matching_cot_train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved train: {train_path}")

    val_path = data_dir / "nmr_matching_cot_val.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved val: {val_path}")

    # Update dataset_info.json
    info_path = data_dir / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["nmr_matching_cot_train"] = {"file_name": "nmr_matching_cot_train.json"}
    info["nmr_matching_cot_val"] = {"file_name": "nmr_matching_cot_val.json"}

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Updated dataset_info.json")

    # Also build a test set from matching data NOT in training
    # (simple Yes/No without CoT, for evaluation)
    build_test_set(records, data_dir)


def build_test_set(cot_records, data_dir):
    """Build test set from matching data not used in CoT training."""
    # Collect all (candidate_smiles, gt_smiles) pairs in training
    train_pairs = set()
    for rec in cot_records:
        cand = rec["input"]["candidate_smiles"]
        gt = rec["gt_smiles"]
        train_pairs.add((cand, gt))

    # Load all matching benchmark files
    base = Path("/fs_mol/cxq/SpecMaster_jp")
    test_records = []

    for jsonl_path in [
        base / "negative_output" / "matching_100_benchmark.jsonl",
        base / "perturbation_output" / "perturbed_100_matching_benchmark.jsonl",
        base / "hard_negative_output" / "hard_neg_100_benchmark.jsonl",
    ]:
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                pair = (rec["candidate_smiles"], rec["gt_smiles"])
                if pair not in train_pairs:
                    test_records.append(rec)

    # Deduplicate
    seen = set()
    unique_test = []
    for rec in test_records:
        key = (rec["candidate_smiles"], rec["gt_smiles"])
        if key not in seen:
            seen.add(key)
            unique_test.append(rec)

    # Convert to Alpaca format (simple Yes/No output, no CoT)
    test_alpaca = []
    for rec in unique_test:
        candidate = rec["candidate_smiles"]
        c_peaks = rec.get("c_nmr_peaks") or []
        c_vals = []
        for p in c_peaks:
            v = p.get("delta (ppm)")
            if v is not None:
                c_vals.append(f"{float(v):.1f}")
        c_str = ", ".join(c_vals) if c_vals else "N/A"

        h_peaks = rec.get("h_nmr_peaks") or []
        h_parts = []
        for p in h_peaks:
            delta = p.get("centroid") or p.get("delta")
            if delta is None:
                continue
            cat = p.get("category", "")
            nH = p.get("nH", 1)
            j_vals = p.get("j_values", "")
            if j_vals and j_vals != "_":
                j_str = j_vals.replace("_", ", ").strip(", ")
                h_parts.append(f"{float(delta):.2f} ({cat}, J={j_str}, {nH}H)")
            else:
                h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
        h_str = ", ".join(h_parts) if h_parts else "N/A"

        freq = rec.get("nmr_frequency", "")
        solvent = rec.get("nmr_solvent", "")
        header = f" ({freq}, {solvent})" if freq else ""

        test_alpaca.append({
            "instruction": INSTRUCTION,
            "input": (
                f"Candidate SMILES: {candidate}\n\n"
                f"13C NMR{header}: {c_str}\n"
                f"1H NMR{header}: {h_str}"
            ),
            "output": "Yes" if rec["label"] == 1 else "No",
            "_label": rec["label"],
            "_negative_type": rec.get("negative_type", "positive"),
            "_candidate_smiles": candidate,
            "_gt_smiles": rec.get("gt_smiles", ""),
        })

    test_path = data_dir / "nmr_matching_test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_alpaca, f, indent=2, ensure_ascii=False)

    n_pos = sum(1 for r in test_alpaca if r["_label"] == 1)
    n_neg = sum(1 for r in test_alpaca if r["_label"] == 0)
    print(f"Test set: {len(test_alpaca)} records (pos={n_pos}, neg={n_neg})")
    print(f"Saved test: {test_path}")


if __name__ == "__main__":
    main()
