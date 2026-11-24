#!/usr/bin/env python3
"""Identify which v6.2 pool rows need fresh teacher distillation.

v6.2 only changes spectral_scorer.py (no double-penalty, peak-count weights).
The pool is identical to v6.1.  We only re-distill rows where the spectral_match
similarity changed by > SIM_THRESH versus the v6.1 tool_log.

Outputs:
  distill_out/v6_2_redistill_indices.json   -- pool row indices for --only_indices flag
  distill_out/v6_2_redistill_summary.json   -- counts by reason
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DISTILL_DIR = Path("/fs_mol/cxq/SpecMaster_jp/distill_out")
SIM_THRESH  = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_tool_log", default=str(DISTILL_DIR / "v6_1_tool_log.jsonl"))
    ap.add_argument("--new_tool_log", default=str(DISTILL_DIR / "v6_2_tool_log.jsonl"))
    ap.add_argument("--sim_thresh", type=float, default=SIM_THRESH)
    ap.add_argument("--out_dir", default=str(DISTILL_DIR))
    args = ap.parse_args()

    old_rows = [json.loads(l) for l in Path(args.old_tool_log).open()]
    new_rows = [json.loads(l) for l in Path(args.new_tool_log).open()]
    assert len(old_rows) == len(new_rows), \
        f"pool size mismatch: old={len(old_rows)} new={len(new_rows)}"
    n_pool = len(new_rows)
    print(f"pool size: {n_pool}")

    indices = []
    n_no_spec = 0
    deltas = []
    for i, (old, new) in enumerate(zip(old_rows, new_rows)):
        old_s = (old.get("tool_log") or {}).get("spectral_match", {}).get("similarity")
        new_s = (new.get("tool_log") or {}).get("spectral_match", {}).get("similarity")
        if old_s is None or new_s is None:
            n_no_spec += 1
            continue
        delta = abs(float(new_s) - float(old_s))
        deltas.append(delta)
        if delta > args.sim_thresh:
            indices.append(i)

    if deltas:
        import statistics
        print(f"similarity delta stats: mean={statistics.mean(deltas):.4f}  "
              f"max={max(deltas):.4f}  p90={sorted(deltas)[int(0.9*len(deltas))]:.4f}")
    print(f"rows with spectral_match unavailable: {n_no_spec}")
    print(f"\n=== redistillation indices ===")
    print(f"  total to re-distill: {len(indices)} / {n_pool}  (sim_thresh={args.sim_thresh})")
    print(f"  reuse existing CoT: {n_pool - len(indices) - n_no_spec}")

    out_dir = Path(args.out_dir)
    idx_path = out_dir / "v6_2_redistill_indices.json"
    idx_path.write_text(json.dumps(indices))
    print(f"\n[write] {idx_path} ({len(indices)} indices)")

    summary = {
        "total_pool": n_pool,
        "redistill_count": len(indices),
        "reuse_count": n_pool - len(indices),
        "no_spectral_match": n_no_spec,
        "sim_thresh": args.sim_thresh,
    }
    summary_path = out_dir / "v6_2_redistill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
