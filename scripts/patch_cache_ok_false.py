#!/usr/bin/env python3
"""Retry prediction for cache entries where ok=False.

Reads prediction_cache.json, identifies all entries with ok=False,
re-runs predict_nmr_peaks() on them, and updates the cache in-place
for entries that now succeed. Failed entries are left unchanged.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/fs_mol/cxq/SpecMaster_jp/perturbation_generator")
from nmr_predictor import predict_nmr_peaks  # noqa: E402

CACHE_PATH = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json")
MAX_WORKERS = 8


def main() -> None:
    cache = json.loads(CACHE_PATH.read_text())
    print(f"Cache total: {len(cache)} entries")

    failed_keys = [k for k, v in cache.items() if not v.get("ok")]
    print(f"ok=False entries: {len(failed_keys)}")

    if not failed_keys:
        print("Nothing to patch.")
        return

    # backup
    bak = CACHE_PATH.with_suffix(".json.patch_bak")
    bak.write_bytes(CACHE_PATH.read_bytes())
    print(f"Backed up → {bak}")

    def _predict_one(smi: str):
        return smi, predict_nmr_peaks(smi)

    fixed = 0
    still_failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_predict_one, s): s for s in failed_keys}
        done = 0
        for fut in as_completed(futures):
            smi, result = fut.result()
            done += 1
            if result.get("ok"):
                cache[smi] = result
                fixed += 1
            else:
                still_failed += 1
            if done % 20 == 0:
                print(f"  {done}/{len(failed_keys)}  fixed={fixed}  still_failed={still_failed}", flush=True)

    print(f"\nDone: fixed={fixed}, still_failed={still_failed}")

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"Saved updated cache → {CACHE_PATH}")


if __name__ == "__main__":
    main()
