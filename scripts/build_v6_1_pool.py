#!/usr/bin/env python3
"""Build the v6.1 combined distillation pool.

Strategy (fixes v6's over-rejection regression):
  Base   : full v5 distill_pool (2400 rows: 320 pos + 2080 neg)
  New neg: 1500 hard negatives from v6_pool (dedup against v5)
             similar_structure 800 | same_formula 500 | random 200
  Upsample pos: duplicate all 320 v5 positives once → +320 rows
  ──────────────────────────────────────────────────────────────
  Total  : ~4220 rows, 640/4220 = 15.2% positive ratio

Dedup for NEW negatives only: must not overlap test_2000 or v5_pool.
Positives are deliberately re-included (upsampled) so no dedup applied.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

V5_POOL    = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/distill_pool.jsonl")
V6_POOL    = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/v6_pool.jsonl")
TEST_2000  = Path("/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json")
COT_V4     = Path("/fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl")
OUT_DEFAULT = Path("/fs_mol/cxq/SpecMaster_jp/distill_out/v6_1_pool.jsonl")

NEW_NEG_TARGETS = {
    "similar_structure": 800,
    "same_formula":      500,
    "random":            200,
}


def canon(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else smi
    except Exception:
        return smi


def load_forbidden() -> Set[str]:
    """Canonical SMILES that must not appear in the new-negative additions."""
    forbid: Set[str] = set()
    for row in json.loads(TEST_2000.read_text()):
        for k in ("_candidate_smiles", "candidate_smiles"):
            if row.get(k):
                forbid.add(canon(row[k])); break
        else:
            if isinstance(row.get("input"), str):
                m = re.search(r"Candidate SMILES:\s*(\S+)", row["input"])
                if m: forbid.add(canon(m.group(1)))
        for k in ("_gt_smiles", "gt_smiles"):
            if row.get(k):
                forbid.add(canon(row[k])); break
    for line in COT_V4.open():
        r = json.loads(line)
        c = (r.get("input") or {}).get("candidate_smiles")
        if c: forbid.add(canon(c))
    for line in V5_POOL.open():
        r = json.loads(line)
        c = (r.get("input") or {}).get("candidate_smiles")
        if c: forbid.add(canon(c))
    return forbid


def class_key(row: Dict) -> str:
    label = row.get("label")
    if label == 1:
        return "positive"
    nt = (row.get("negative_type") or row.get("_class") or "").lower()
    for cls in ("similar_structure", "same_formula", "insert_ch2", "random"):
        if cls in nt:
            return cls
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--no-upsample", action="store_true",
                    help="skip positive upsampling (for ablation)")
    ap.add_argument("--neg-targets", type=str, default="",
                    help="override new-neg targets as JSON, e.g. '{\"similar_structure\":600}'")
    args = ap.parse_args()
    random.seed(args.seed)

    neg_targets = dict(NEW_NEG_TARGETS)
    if args.neg_targets:
        neg_targets.update(json.loads(args.neg_targets))

    print("=== loading v5 pool (full replay) ===")
    v5_rows: List[Dict] = [json.loads(l) for l in V5_POOL.open()]
    v5_positives = [r for r in v5_rows if r.get("label") == 1]
    v5_negatives = [r for r in v5_rows if r.get("label") == 0]
    print(f"  v5 pool: {len(v5_rows)} total ({len(v5_positives)} pos, {len(v5_negatives)} neg)")

    print("\n=== loading forbidden SMILES (test_2000 + v4_cot + v5_pool) ===")
    forbidden = load_forbidden()
    print(f"  forbidden: {len(forbidden)} canonical SMILES")

    print("\n=== sampling new negatives from v6_pool ===")
    by_class: Dict[str, List[Dict]] = {}
    for line in V6_POOL.open():
        r = json.loads(line)
        if r.get("label") == 1:
            continue  # v6_pool has no positives, but guard anyway
        c_smi = canon((r.get("input") or {}).get("candidate_smiles") or "")
        if c_smi in forbidden:
            continue
        cls = class_key(r)
        by_class.setdefault(cls, []).append(r)

    new_negatives: List[Dict] = []
    for cls, target in neg_targets.items():
        pool = by_class.get(cls, [])
        random.shuffle(pool)
        taken = pool[:target]
        new_negatives.extend(taken)
        print(f"  {cls}: {len(taken)}/{target} (pool had {len(pool)})")

    print(f"  new negatives total: {len(new_negatives)}")

    # Upsample positives: duplicate all v5 positives once
    upsampled_pos: List[Dict] = []
    if not args.no_upsample:
        upsampled_pos = list(v5_positives)  # one full duplicate
        print(f"\n=== upsampling positives: +{len(upsampled_pos)} (1× duplicate of {len(v5_positives)} v5 pos) ===")
    else:
        print("\n=== positive upsampling disabled ===")

    # Combine in stable order so teacher distillation indices align:
    # v5 rows first (positions 0..2399), then new negatives, then upsampled positives.
    # Do NOT shuffle — trajectory builder matches distillation by pool row index.
    combined = v5_rows + new_negatives + upsampled_pos

    # Stats
    total_pos = sum(1 for r in combined if r.get("label") == 1)
    total_neg = sum(1 for r in combined if r.get("label") == 0)
    by_cls_final: Dict[str, int] = {}
    for r in combined:
        k = class_key(r)
        by_cls_final[k] = by_cls_final.get(k, 0) + 1

    print(f"\n=== v6.1 pool summary ===")
    print(f"  total: {len(combined)}  pos={total_pos} ({100*total_pos/len(combined):.1f}%)  neg={total_neg}")
    for k, v in sorted(by_cls_final.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[write] {out} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
