#!/usr/bin/env python3
"""Sample the v6 distillation pool: 2200 NEW candidates on top of v5's 2400.

Focus classes (per wrong-analysis on v5's test_2000 rollout):
  * similar_structure : 1200  -- biggest v5 gap (73.4% vs teacher's ~96%). Pool has 2835 avail
                                  after v5's 1380, so ~1450 fresh candidates remain.
  * same_formula     :  700  -- v5 80.7%, teacher ~97%. Pool 3897 avail after v5's 300.
  * random           :  300  -- anchor to keep class balance (v5 use 200; +300 -> 500 total).

Dedup targets: test_2000 + v4 CoT source + v5's distill_pool.jsonl (SMILES canonical).
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
    "negative_1000": "/fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl",
    "hard_neg_1000": "/fs_mol/cxq/SpecMaster_jp/hard_negative_output_1000/hard_neg_1000_matching.jsonl",
    "perturb_1000":  "/fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl",
}
TEST_2000 = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
COT_V4 = "/fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl"
V5_POOL = "/fs_mol/cxq/SpecMaster_jp/distill_out/distill_pool.jsonl"

TARGETS = {
    "similar_structure": 1200,
    "same_formula":       700,
    "random":             300,
}


def canonical(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m, canonical=True) if m else smi
    except Exception:
        return smi


def load_forbidden() -> Set[str]:
    forbid: Set[str] = set()
    import re
    with open(TEST_2000) as f:
        for row in json.load(f):
            for k in ("_candidate_smiles", "candidate_smiles"):
                if row.get(k):
                    forbid.add(canonical(row[k])); break
            else:
                if isinstance(row.get("input"), str):
                    m = re.search(r"Candidate SMILES:\s*(\S+)", row["input"])
                    if m: forbid.add(canonical(m.group(1)))
            for k in ("_gt_smiles", "gt_smiles"):
                if row.get(k):
                    forbid.add(canonical(row[k])); break
    with open(COT_V4) as f:
        for line in f:
            r = json.loads(line)
            c = (r.get("input") or {}).get("candidate_smiles")
            if c: forbid.add(canonical(c))
    with open(V5_POOL) as f:
        for line in f:
            r = json.loads(line)
            c = (r.get("input") or {}).get("candidate_smiles")
            if c: forbid.add(canonical(c))
    return forbid


def class_key(row: Dict) -> str:
    label = row.get("label")
    nt = row.get("negative_type") or ""
    if label == 1 or nt == "positive": return ""  # skip positive (source exhausted)
    if nt == "neg/similar_structure": return "similar_structure"
    if nt == "neg/same_formula": return "same_formula"
    if nt == "neg/random": return "random"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fs_mol/cxq/SpecMaster_jp/distill_out/v6_pool.jsonl")
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    forbid = load_forbidden()
    print(f"[dedup] forbidden set: {len(forbid)}", flush=True)

    bucket: Dict[str, List[Dict]] = defaultdict(list)
    seen_within: Set[str] = set()
    n_total = n_dedup = n_bucketed = 0
    for pool_name, path in POOLS.items():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n_total += 1
                cand = r.get("candidate_smiles")
                if not cand: continue
                cc = canonical(cand)
                if cc in forbid: n_dedup += 1; continue
                if cc in seen_within: continue
                cls = class_key(r)
                if not cls: continue
                seen_within.add(cc)
                bucket[cls].append({**r, "_source_pool": pool_name, "_canonical": cc})
                n_bucketed += 1

    print(f"[pool ] total={n_total}  dedup={n_dedup}  bucketed={n_bucketed}", flush=True)
    for cls, tgt in TARGETS.items():
        print(f"   bucket[{cls}] = {len(bucket[cls])}  (target {tgt})", flush=True)

    sampled = []
    for cls, tgt in TARGETS.items():
        pool = bucket[cls]
        rng.shuffle(pool)
        take = pool[:tgt]
        sampled.extend([{**row, "_class": cls} for row in take])
        print(f"[take ] {cls:20s} {len(take)}/{tgt}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, row in enumerate(sampled):
            out = {
                "case_id": row.get("case_id") or f"v6-{row['_source_pool']}-{i}",
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
