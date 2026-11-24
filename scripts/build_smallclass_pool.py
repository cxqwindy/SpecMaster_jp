#!/usr/bin/env python3
"""Build a v5.1 distillation pool targeting the 6 functional_group_replacement sub-classes
that regressed in v5 (alcohol_to_thiol, amide_to_ester, ester_to_amide × {perturbation, hard_nmr}).

Sources: same three pools as build_distill_pool.py.
Dedup: canonical-SMILES against test_2000 + cot_traces_v4 + existing distill_pool.jsonl (v5).

Target: ~40 rows per sub-class where source is deep enough, otherwise take all available.
Over-sample vs the 8-row test-set target because teacher-wrong drops will remove some.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

POOLS = {
    "hard_neg_1000": "/fs_mol/cxq/SpecMaster_jp/hard_negative_output_1000/hard_neg_1000_matching.jsonl",
    "perturb_1000":  "/fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl",
    # negative_output_1000 does NOT contain functional_group_replacement variants (only
    # random / same_formula / similar_structure), so we skip it here.
}

# Target class -> its full `negative_type` string keys we accept
TARGET_TYPES = {
    "alcohol_to_thiol": [
        "neg/perturbation/functional_group_replacement/alcohol_to_thiol",
        "neg/hard_nmr/functional_group_replacement/alcohol_to_thiol",
    ],
    "amide_to_ester": [
        "neg/perturbation/functional_group_replacement/amide_to_ester",
        "neg/hard_nmr/functional_group_replacement/amide_to_ester",
    ],
    "ester_to_amide": [
        "neg/perturbation/functional_group_replacement/ester_to_amide",
        "neg/hard_nmr/functional_group_replacement/ester_to_amide",
    ],
}
TARGET_TAKE_PER_CLASS = 40  # per class; teacher-wrong drops will leave ~30 usable

TEST_2000_PATH = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
COT_V4_PATH = "/fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl"
V5_POOL_PATH = "/fs_mol/cxq/SpecMaster_jp/distill_out/distill_pool.jsonl"


def canonical(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


def load_forbidden() -> Set[str]:
    forbid: Set[str] = set()
    # test_2000
    with open(TEST_2000_PATH) as f:
        data = json.load(f)
    import re
    for row in data:
        for k in ("candidate_smiles", "_candidate_smiles"):
            if row.get(k):
                forbid.add(canonical(row[k])); break
        else:
            if isinstance(row.get("input"), str):
                m = re.search(r"Candidate SMILES:\s*([^\s\n]+)", row["input"])
                if m:
                    forbid.add(canonical(m.group(1)))
        for k in ("gt_smiles", "_gt_smiles"):
            if row.get(k):
                forbid.add(canonical(row[k])); break
    # v4 CoT source
    with open(COT_V4_PATH) as f:
        for line in f:
            r = json.loads(line)
            c = (r.get("input") or {}).get("candidate_smiles")
            if c:
                forbid.add(canonical(c))
    # v5 distill pool (existing Path A teacher rows)
    with open(V5_POOL_PATH) as f:
        for line in f:
            r = json.loads(line)
            c = (r.get("input") or {}).get("candidate_smiles")
            if c:
                forbid.add(canonical(c))
    return forbid


def class_key_of(neg_type: str) -> str:
    for cls, types in TARGET_TYPES.items():
        if neg_type in types:
            return cls
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs_mol/cxq/SpecMaster_jp/distill_out/smallclass_pool.jsonl")
    ap.add_argument("--seed", type=int, default=46)
    args = ap.parse_args()

    forbid = load_forbidden()
    print(f"[dedup] forbidden canonical SMILES (test+v4+v5-pool): {len(forbid)}", flush=True)

    bucket: Dict[str, List[Dict]] = defaultdict(list)
    seen_within: Set[str] = set()
    n_total = n_dedup = n_bucketed = 0
    for pool_name, path in POOLS.items():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                cand = r.get("candidate_smiles")
                if not cand:
                    continue
                cc = canonical(cand)
                if cc in forbid:
                    n_dedup += 1; continue
                if cc in seen_within:
                    continue
                cls = class_key_of(r.get("negative_type") or "")
                if not cls:
                    continue
                seen_within.add(cc)
                bucket[cls].append({**r, "_source_pool": pool_name, "_canonical": cc})
                n_bucketed += 1

    print(f"[pool ] rows_total={n_total}  dedup={n_dedup}  bucketed={n_bucketed}", flush=True)
    for cls in TARGET_TYPES:
        print(f"   bucket[{cls}] available: {len(bucket[cls])}   target: {TARGET_TAKE_PER_CLASS}", flush=True)

    rng = random.Random(args.seed)
    sampled: List[Dict] = []
    for cls in TARGET_TYPES:
        pool = bucket[cls]
        rng.shuffle(pool)
        take = pool[:TARGET_TAKE_PER_CLASS]
        sampled.extend([{**row, "_class": cls} for row in take])
        print(f"[take ] {cls:20s} {len(take)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, row in enumerate(sampled):
            out = {
                "case_id": row.get("case_id") or f"pool-{row['_source_pool']}-sc-{i}",
                "label": row.get("label"),
                "gt_smiles": row.get("gt_smiles"),
                "negative_type": row.get("negative_type"),
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
