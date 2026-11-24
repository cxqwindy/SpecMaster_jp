#!/usr/bin/env python3
"""Step 1 of v6.1 rebuild: regenerate prediction_cache.json with fixed _extract_peaks_from_result.

Old cache used _merge_equivalent_peaks which spuriously merged distinct atoms.
New logic uses C_groups/H_groups from the API response directly.

Collects all unique canonical SMILES from:
  - existing prediction_cache.json  (11940 entries)
  - v5/v6 distill pool jsonl files
  - raw pool jsonl files (matching_1000, hard_neg_1000, perturbed_1000)
  - optionally test_2000 (for completeness)

Calls predict_nmr_peaks() (which now uses the fixed extractor) for every SMILES,
writes the new cache, and reports how many entries changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem

# ---- paths ---------------------------------------------------------------
_BASE = Path("/fs_mol/cxq/SpecMaster_jp")
_CACHE_IN  = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json")
_CACHE_OUT = _CACHE_IN  # overwrite in-place (old is backed up first)

_POOL_FILES = [
    _BASE / "negative_output_1000/matching_1000_matching.jsonl",
    _BASE / "hard_negative_output_1000/hard_neg_1000_matching.jsonl",
    _BASE / "perturbation_output_1000/perturbed_1000_matching_benchmark.jsonl",
]
_DISTILL_POOLS = [
    _BASE / "distill_out/distill_pool.jsonl",
    _BASE / "distill_out/v6_pool.jsonl",
]
_TEST_2000 = _BASE / "data/nmr_matching_test_2000.json"

sys.path.insert(0, str(_BASE / "perturbation_generator"))


def _canon(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def collect_smiles(include_test: bool = True) -> set[str]:
    """Collect all unique canonical SMILES that need a cache entry."""
    smiles: set[str] = set()

    # 1. existing cache keys
    if _CACHE_IN.exists():
        old = json.loads(_CACHE_IN.read_text())
        for k in old:
            c = _canon(k)
            if c:
                smiles.add(c)
        print(f"  existing cache: {len(old)} keys → {len(smiles)} canonical")

    # 2. distill pool files
    for p in _DISTILL_POOLS:
        if not p.exists():
            continue
        n_before = len(smiles)
        for line in p.open():
            r = json.loads(line)
            cand = (r.get("input") or {}).get("candidate_smiles") or r.get("candidate_smiles")
            if cand:
                c = _canon(cand)
                if c:
                    smiles.add(c)
        print(f"  {p.name}: +{len(smiles)-n_before} new")

    # 3. raw pool files
    for p in _POOL_FILES:
        if not p.exists():
            continue
        n_before = len(smiles)
        for line in p.open():
            r = json.loads(line)
            cand = r.get("candidate_smiles") or r.get("gt_smiles")
            if cand:
                c = _canon(cand)
                if c:
                    smiles.add(c)
        print(f"  {p.name}: +{len(smiles)-n_before} new")

    # 4. test_2000 (so cache is complete for eval)
    if include_test and _TEST_2000.exists():
        n_before = len(smiles)
        for r in json.loads(_TEST_2000.read_text()):
            for k in ("candidate_smiles", "_candidate_smiles"):
                if r.get(k):
                    c = _canon(r[k])
                    if c:
                        smiles.add(c)
                    break
        print(f"  test_2000: +{len(smiles)-n_before} new")

    return smiles


def rebuild_cache(
    max_workers: int = 16,
    backup: bool = True,
    dry_run: bool = False,
    include_test: bool = True,
) -> None:
    from nmr_predictor import predict_nmr_peaks  # type: ignore

    print("=== collect SMILES ===")
    all_smi = sorted(collect_smiles(include_test=include_test))
    print(f"Total unique SMILES to predict: {len(all_smi)}")

    # load old cache for comparison
    old_cache: dict = {}
    if _CACHE_IN.exists():
        old_cache = json.loads(_CACHE_IN.read_text())

    if dry_run:
        print("[dry-run] would predict and save — exiting")
        return

    # predict in parallel
    print(f"\n=== predict ({max_workers} workers) ===")
    new_cache: dict[str, dict] = {}
    failed = 0
    changed_c = 0
    changed_h = 0

    def _predict_one(smi: str):
        return smi, predict_nmr_peaks(smi)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_predict_one, s): s for s in all_smi}
        for fut in as_completed(futures):
            smi, result = fut.result()
            new_cache[smi] = result
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(all_smi)} done …", flush=True)

    print(f"Predicted {len(new_cache)} entries ({failed} failed)")

    # compare old vs new
    for smi, new in new_cache.items():
        old = old_cache.get(smi) or {}
        old_nc = len(old.get("c_peaks", []))
        new_nc = len(new.get("c_peaks", []))
        old_nh = len(old.get("h_peaks", []))
        new_nh = len(new.get("h_peaks", []))
        if old_nc != new_nc:
            changed_c += 1
        if old_nh != new_nh:
            changed_h += 1

    print(f"\n=== delta vs old cache ===")
    print(f"  n_pred_c changed: {changed_c}/{len(new_cache)}")
    print(f"  n_pred_h changed: {changed_h}/{len(new_cache)}")

    if backup and _CACHE_IN.exists():
        bak = _CACHE_IN.with_suffix(".json.bak")
        bak.write_bytes(_CACHE_IN.read_bytes())
        print(f"\nBacked up old cache → {bak}")

    _CACHE_OUT.write_text(json.dumps(new_cache, ensure_ascii=False))
    print(f"Saved new cache ({len(new_cache)} entries) → {_CACHE_OUT}")


def main():
    ap = argparse.ArgumentParser(description="Rebuild prediction_cache.json with fixed peak extractor")
    ap.add_argument("--workers", type=int, default=16, help="parallel predict workers")
    ap.add_argument("--no-backup", action="store_true", help="skip .bak copy of old cache")
    ap.add_argument("--dry-run", action="store_true", help="count SMILES without predicting")
    ap.add_argument("--no-test", action="store_true", help="skip test_2000 SMILES")
    args = ap.parse_args()

    rebuild_cache(
        max_workers=args.workers,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        include_test=not args.no_test,
    )


if __name__ == "__main__":
    main()
