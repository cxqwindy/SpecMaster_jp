#!/usr/bin/env python3
"""Build the Path-A distillation candidate pool for v5.

Sources (see memory: [[test-2000-benchmark]] / [[project_task_difficulty]]):
  - /fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl
  - /fs_mol/cxq/SpecMaster_jp/hard_negative_output_1000/hard_neg_1000_matching.jsonl
  - /fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl

Class allocation (2400 total, per plan Section A.2):
  similar_structure : 1200  (base 1100 + M3-mistake booster 100; M3 test_2000 err 34.0%)
  positive-hard     :  500  (M3 err 22.6%; sample from any pool where label=1)
  same_formula      :  300  (base 200 + booster 100; M3 err 21.1%)
  insert_ch2 (perturbation+hard_nmr combined) : 200  (booster; M3 err 15-28%)
  random (anchor)   :  200

Dedup:
  - canonical SMILES set from data/nmr_matching_test_2000.json (bench, MUST NOT overlap).
  - canonical SMILES set from cot_traces_v4.jsonl (already-seen training material, we want NEW pool).

Output: distill_pool.jsonl with the same schema as cot_traces_v4.jsonl input rows:
  {input: {candidate_smiles, c_nmr_peaks, h_nmr_peaks, nmr_solvent, nmr_frequency},
   label, gt_smiles, negative_type, case_id, source_tool, source_pool}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

POOLS = {
    "negative_1000": "/fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl",
    "hard_neg_1000": "/fs_mol/cxq/SpecMaster_jp/hard_negative_output_1000/hard_neg_1000_matching.jsonl",
    "perturb_1000":  "/fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl",
}

TEST_2000_PATH = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
COT_V4_PATH = "/fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl"

# class label -> target count. Positive pool is small after dedup (320 avail); reallocate the
# shortfall to similar_structure, the highest-impact M3 weak class (34% err on test_2000).
TARGETS = {
    "similar_structure": 1380,
    "positive":          320,
    "same_formula":      300,
    "insert_ch2":        200,
    "random":            200,
}


def canonical(smi: str) -> str:
    if not smi:
        return ""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return smi


def load_forbidden_canonical_set() -> Set[str]:
    """Union of (test_2000, v4 CoT source) canonical SMILES — dedup targets."""
    forbid: Set[str] = set()
    # test_2000 (JSON array; fields: instruction/input contains candidate_smiles)
    with open(TEST_2000_PATH) as f:
        data = json.load(f)
    for row in data:
        # test_2000 rows carry the candidate SMILES inside the `input` text; also inside `_meta` if present
        # Fall back: extract SMILES from any explicit key
        cand = None
        for k in ("candidate_smiles", "_candidate_smiles"):
            if row.get(k):
                cand = row[k]; break
        if cand is None and isinstance(row.get("input"), str):
            # try to parse `Candidate SMILES: <smi>` line
            import re
            m = re.search(r"Candidate SMILES:\s*([^\s\n]+)", row["input"])
            if m:
                cand = m.group(1)
        if cand:
            forbid.add(canonical(cand))
        # also block the GT structure so we don't distill the same molecule from the opposite side
        for k in ("gt_smiles", "_gt_smiles"):
            if row.get(k):
                forbid.add(canonical(row[k])); break
    # v4 CoT source
    with open(COT_V4_PATH) as f:
        for line in f:
            r = json.loads(line)
            inp = r.get("input") or {}
            c = inp.get("candidate_smiles")
            if c:
                forbid.add(canonical(c))
    return forbid


def class_of(row: Dict) -> str:
    """Map a pool row to one of TARGETS' keys, or '' if it doesn't fit any."""
    label = row.get("label")
    nt = row.get("negative_type") or ""
    if label == 1 or nt == "positive":
        return "positive"
    if nt == "neg/similar_structure":
        return "similar_structure"
    if nt == "neg/same_formula":
        return "same_formula"
    if nt == "neg/random":
        return "random"
    if "insert_ch2" in nt:
        return "insert_ch2"
    return ""  # skip unmatched negative types (they are rare and unrelated to M3 hot classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    forbid = load_forbidden_canonical_set()
    print(f"[dedup] forbidden canonical SMILES: {len(forbid)}", flush=True)

    # bucket pool rows by class
    bucket: Dict[str, List[Dict]] = defaultdict(list)
    seen_within: Set[str] = set()  # dedup across pools
    n_total = n_dedup_test = n_dedup_intra = n_bucketed = 0
    for pool_name, path in POOLS.items():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                c = r.get("candidate_smiles")
                if not c:
                    continue
                cc = canonical(c)
                if cc in forbid:
                    n_dedup_test += 1
                    continue
                if cc in seen_within:
                    n_dedup_intra += 1
                    continue
                cls = class_of(r)
                if not cls:
                    continue
                seen_within.add(cc)
                bucket[cls].append({**r, "_source_pool": pool_name, "_canonical": cc})
                n_bucketed += 1

    print(f"[pool ] rows_total={n_total}  dedup_test={n_dedup_test}  dedup_intra={n_dedup_intra}  "
          f"bucketed={n_bucketed}", flush=True)
    for k in TARGETS:
        print(f"   bucket[{k}] = {len(bucket[k])}  (target {TARGETS[k]})", flush=True)

    # sample per class up to target
    sampled: List[Dict] = []
    for cls, tgt in TARGETS.items():
        pool = bucket[cls]
        rng.shuffle(pool)
        take = pool[:tgt]
        sampled.extend([{**row, "_class": cls} for row in take])
        print(f"[take ] {cls:20s} {len(take)}/{tgt}", flush=True)

    print(f"[total] {len(sampled)} candidates sampled", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, row in enumerate(sampled):
            # produce the SAME shape as CoT source rows so downstream tooling reuses cot_to_trajectory
            out = {
                "case_id": row.get("case_id") or f"pool-{row['_source_pool']}-{i}",
                "label": row.get("label"),
                "gt_smiles": row.get("gt_smiles"),
                "negative_type": row.get("negative_type") or "positive",
                "_class": row["_class"],
                "_source_pool": row["_source_pool"],
                "input": {
                    "candidate_smiles": row["candidate_smiles"],
                    "c_nmr_peaks": row.get("c_nmr_peaks") or [],
                    "h_nmr_peaks": row.get("h_nmr_peaks") or [],
                    "nmr_solvent": row.get("nmr_solvent"),
                    "nmr_frequency": row.get("nmr_frequency"),
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[write] {len(sampled)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
