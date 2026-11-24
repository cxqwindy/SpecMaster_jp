#!/usr/bin/env python3
"""Convert a repair-format benchmark (from cli.py, keyed by input_smiles/gt_smiles)
into a spectrum-MATCHING benchmark jsonl (candidate_smiles/label/negative_type),
reusing formatter.sample_to_matching_jsonl. Optionally emits balanced positive pairs
(one per unique gt_smiles) so the downstream RL parquet isn't ~4% positive.

Usage:
    python -m perturbation_generator.repair_bench_to_matching \
        --in  perturbation_output_1000/perturbed_1000_benchmark.jsonl \
        --out perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl \
        --with_positives   # add one positive per unique gt (default on)
"""
import argparse
import json

from .base import PerturbedSample
from .formatter import sample_to_matching_jsonl

# repair-benchmark negative_type is bare (e.g. "heteroatom_replacement/O_to_S");
# the matching pipeline tags perturbation negatives as "neg/perturbation/<type>".
NEG_PREFIX = "neg/perturbation/"


def row_to_sample(r):
    return PerturbedSample(
        input_smiles=r["input_smiles"],
        input_formula=r.get("input_formula", ""),
        target_formula=r.get("target_formula", ""),
        gt_smiles=r["gt_smiles"],
        formula_delta=r.get("formula_delta", ""),
        edit_lb=r.get("edit_lb", 0),
        c_nmr_peaks=r.get("c_nmr_peaks"),
        h_nmr_peaks=r.get("h_nmr_peaks"),
        nmr_frequency=r.get("nmr_frequency"),
        nmr_solvent=r.get("nmr_solvent"),
        perturbation_type=NEG_PREFIX + r.get("perturbation_type", ""),
        case_id=r.get("case_id", ""),
        source_tool=r.get("source_tool", "perturbation_generator"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_positives", action="store_true", default=True)
    ap.add_argument("--no_positives", dest="with_positives", action="store_false")
    args = ap.parse_args()

    n_neg = n_pos = 0
    seen_pos_gt = set()
    with open(args.inp) as f, open(args.out, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = row_to_sample(r)
            # negative
            out.write(json.dumps(sample_to_matching_jsonl(s, is_positive=False),
                                 ensure_ascii=False) + "\n")
            n_neg += 1
            # balanced positive (one per unique gt)
            if args.with_positives and s.gt_smiles not in seen_pos_gt:
                seen_pos_gt.add(s.gt_smiles)
                out.write(json.dumps(sample_to_matching_jsonl(s, is_positive=True),
                                     ensure_ascii=False) + "\n")
                n_pos += 1
    print(f"Wrote {args.out}: neg={n_neg} pos={n_pos} total={n_neg + n_pos}")


if __name__ == "__main__":
    main()
