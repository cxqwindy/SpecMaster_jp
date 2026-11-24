#!/usr/bin/env python3
"""Merge regenerated weak-type negative CoT into v4 to produce v5.

Strategy: for every weak-type negative in v4, replace its reasoning chain
with the freshly generated one (sharpened, tolerance-countering hints).
All other records (positives, strong negatives) pass through unchanged.

Matching is by (candidate_smiles, gt_smiles) pair — the regen set was built
from the exact same v4 pairs, so this is a clean 1:1 swap.

Usage:
    python merge_regen_weak.py
"""

import json
from pathlib import Path

BASE = Path("/fs_mol/cxq/SpecMaster_jp")
V4 = BASE / "cot_output" / "cot_traces_v4.jsonl"
REGEN = BASE / "cot_output_regen_weak" / "cot_traces.jsonl"
OUT = BASE / "cot_output" / "cot_traces_v5.jsonl"


def key(rec):
    inp = rec.get("input", {})
    return (inp.get("candidate_smiles", ""), rec.get("gt_smiles", ""))


def main():
    # Load regenerated reasoning, keyed by pair. Skip failed generations.
    regen = {}
    n_regen_fail = 0
    with open(REGEN) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning") is None:
                n_regen_fail += 1
                continue
            regen[key(rec)] = rec

    print(f"Loaded {len(regen)} regenerated reasoning chains "
          f"({n_regen_fail} failed, skipped)")

    # Walk v4, swap in regenerated reasoning where available.
    n_total = 0
    n_swapped = 0
    out_records = []
    with open(V4) as f:
        for line in f:
            rec = json.loads(line)
            n_total += 1
            k = key(rec)
            if k in regen:
                # Replace reasoning + reasoning_level, keep everything else.
                new = regen[k]
                rec["reasoning"] = new["reasoning"]
                rec["reasoning_level"] = new.get("reasoning_level",
                                                 rec.get("reasoning_level"))
                n_swapped += 1
            out_records.append(rec)

    print(f"v4 records: {n_total}, swapped: {n_swapped}, "
          f"unchanged: {n_total - n_swapped}")

    # Sanity: any regen chains that didn't match a v4 pair?
    v4_keys = {key(r) for r in out_records}
    unmatched = [k for k in regen if k not in v4_keys]
    if unmatched:
        print(f"WARNING: {len(unmatched)} regen chains had no matching v4 pair")

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_records)} records to {OUT}")


if __name__ == "__main__":
    main()
