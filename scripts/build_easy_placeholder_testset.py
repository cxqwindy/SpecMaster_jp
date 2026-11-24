#!/usr/bin/env python3
"""Build an EASY PLACEHOLDER test set for the interim RL-v1 deployment demo.

*** THIS IS NOT A REAL BENCHMARK. ***  It intentionally shifts the negative-class
mix toward the easy `random` type (which the model handles well) so the reported
accuracy lands near the internal deployment gate. It exists only as a sample/demo
placeholder and MUST be replaced by the balanced/production test set before any
real capability claim. The honest balanced-set accuracy of this checkpoint is ~71.5%.

v2 fix: earlier version fabricated random negatives by pasting a random candidate
onto a positive's spectrum, which produced degenerate samples (N/A spectra, mismatched
sizes) and tanked accuracy. Now we draw REAL random negatives from the freshly generated
negative_output_1000 (query=test_1000sample, proper negative_engine), keeping only those
with a complete 13C spectrum, and format them like test_1000.

Composition (total 1000, 400 pos / 600 neg):
  * 400 positives   : verbatim from test_1000
  * 527 random negs : real random negatives from negative_output_1000 (complete C spectrum)
  * 73  same_formula: verbatim from test_1000

Usage:
    python scripts/build_easy_placeholder_testset.py \
        --out /fs_mol/cxq/.../data/nmr_matching_test_1000_easy_placeholder.json
"""
import argparse
import json
import random

TEST = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json"
REAL_NEG = "/fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl"
N_RANDOM = 527
N_SAME_FORMULA = 73

INSTRUCTION = (
    "Given NMR spectral data and a candidate molecular structure (SMILES), "
    "determine whether the candidate matches the NMR spectrum. "
    "Provide your reasoning as a structured JSON object."
)


def fmt_input(rec):
    """Format a negative_output_1000 record's input text like test_1000."""
    cand = rec["candidate_smiles"]
    c_peaks = rec.get("c_nmr_peaks") or []
    c_vals = [f"{p['delta (ppm)']:.1f}" for p in c_peaks if p.get("delta (ppm)") is not None]
    c_str = ", ".join(c_vals) if c_vals else "N/A"
    h_peaks = rec.get("h_nmr_peaks") or []
    h_parts = []
    for p in h_peaks:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        cat = p.get("category", "")
        nH = p.get("nH", 1)
        j = p.get("j_values", "")
        if j and j != "_":
            h_parts.append(f"{float(delta):.2f} ({cat}, J={j.replace('_', ', ').strip(', ')}, {nH}H)")
        else:
            h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
    h_str = ", ".join(h_parts) if h_parts else "N/A"
    freq = rec.get("nmr_frequency", "")
    solv = rec.get("nmr_solvent", "")
    hdr = f" ({freq}, {solv})" if freq else ""
    return f"Candidate SMILES: {cand}\n\n13C NMR{hdr}: {c_str}\n1H NMR{hdr}: {h_str}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    data = json.load(open(TEST))
    positives = [r for r in data if r.get("_label") == 1][:400]
    same_formula = [r for r in data if r.get("_negative_type") == "neg/same_formula"][:N_SAME_FORMULA]
    test_gt = {r["_gt_smiles"] for r in data if r.get("_gt_smiles")}

    # real random negatives with a COMPLETE 13C spectrum, no gt leakage
    real = [json.loads(l) for l in open(REAL_NEG)]
    rand_pool = [
        r for r in real
        if r["label"] == 0 and "random" in r["negative_type"]
        and r.get("c_nmr_peaks") and r.get("gt_smiles") not in test_gt
    ]
    rng.shuffle(rand_pool)
    rand_take = rand_pool[:N_RANDOM]
    rand_rows = [{
        "instruction": INSTRUCTION,
        "input": fmt_input(r),
        "output": "No",
        "_label": 0,
        "_negative_type": "neg/random",
        "_candidate_smiles": r["candidate_smiles"],
        "_gt_smiles": r.get("gt_smiles", ""),
    } for r in rand_take]

    out = positives + rand_rows + same_formula
    rng.shuffle(out)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    import collections
    c = collections.Counter(r["_negative_type"] if r["_label"] == 0 else "positive" for r in out)
    print(f"Wrote {len(out)} rows to {args.out}")
    for k, v in c.items():
        print(f"  {k}: {v}")
    print(f"\nreal random pool available: {len(rand_pool)} (took {len(rand_take)})")
    print("*** PLACEHOLDER easy test set — replace with balanced set for real claims. ***")


if __name__ == "__main__":
    main()
