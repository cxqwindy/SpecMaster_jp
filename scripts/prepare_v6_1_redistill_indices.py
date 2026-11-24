#!/usr/bin/env python3
"""Identify which v6.1 pool rows need fresh teacher distillation.

Logic:
  - Positions 0..(N_v5-1)  : v5 replay rows. Re-distill only if spectral_match
                              similarity changed by > SIM_THRESH vs old tool_log.
  - Positions N_v5..N_v5+N_neg-1 : new v6 negatives. Always re-distill (no existing CoT).
  - Positions N_v5+N_neg..end    : upsampled v5 positives. Re-distill (fresh tool_log,
                                   ~320 rows, low cost, avoids stale CoT risk).

Outputs:
  distill_out/v6_1_redistill_indices.json   -- pool row indices for --only_indices flag
  distill_out/v6_1_redistill_summary.json   -- counts by reason
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DISTILL_DIR = Path("/fs_mol/cxq/SpecMaster_jp/distill_out")
V5_POOL     = DISTILL_DIR / "distill_pool.jsonl"
V6_1_POOL   = DISTILL_DIR / "v6_1_pool.jsonl"
OLD_TL      = DISTILL_DIR / "tool_log_pool.jsonl"
NEW_TL      = DISTILL_DIR / "v6_1_tool_log.jsonl"
SIM_THRESH  = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6_1_pool", default=str(V6_1_POOL))
    ap.add_argument("--old_tool_log", default=str(OLD_TL))
    ap.add_argument("--new_tool_log", default=str(NEW_TL))
    ap.add_argument("--v5_pool", default=str(V5_POOL))
    ap.add_argument("--sim_thresh", type=float, default=SIM_THRESH)
    ap.add_argument("--out_dir", default=str(DISTILL_DIR))
    args = ap.parse_args()

    # Load v5 pool to know N_v5 and identify v5 case_ids
    v5_rows = [json.loads(l) for l in Path(args.v5_pool).open()]
    n_v5 = len(v5_rows)
    v5_case_ids = {r["case_id"] for r in v5_rows}
    print(f"v5 pool: {n_v5} rows")

    # Load old tool_log similarities (indexed by case_id)
    old_sim: dict[str, float] = {}
    if Path(args.old_tool_log).exists():
        for line in Path(args.old_tool_log).open():
            r = json.loads(line)
            cid = r.get("case_id")
            sim = (r.get("tool_log") or {}).get("spectral_match", {}).get("similarity")
            if cid and sim is not None:
                old_sim[cid] = float(sim)
    print(f"old tool_log: {len(old_sim)} similarity values")

    # Load new tool_log similarities (indexed by position for fast access)
    new_tl_rows = [json.loads(l) for l in Path(args.new_tool_log).open()]
    n_pool = len(new_tl_rows)
    print(f"v6.1 pool size: {n_pool} rows")

    indices_to_distill = []
    reasons: dict[str, int] = {"sim_changed": 0, "new_neg": 0, "upsampled_pos": 0}

    for i, row in enumerate(new_tl_rows):
        cid = row.get("case_id")
        label = row.get("label")

        if i < n_v5:
            # v5 row — re-distill only if similarity changed
            new_s = (row.get("tool_log") or {}).get("spectral_match", {}).get("similarity")
            old_s = old_sim.get(cid)
            if old_s is not None and new_s is not None:
                delta = abs(float(new_s) - old_s)
                if delta > args.sim_thresh:
                    indices_to_distill.append(i)
                    reasons["sim_changed"] += 1
            elif old_s is None:
                # No old similarity record (edge case) — re-distill to be safe
                indices_to_distill.append(i)
                reasons["sim_changed"] += 1
        else:
            # Position i >= n_v5: either new v6 negative or upsampled v5 positive
            # Always re-distill: new negatives have no CoT; upsampled positives need
            # fresh trajectories with correct tool_log numbers.
            if label == 1:
                indices_to_distill.append(i)
                reasons["upsampled_pos"] += 1
            else:
                indices_to_distill.append(i)
                reasons["new_neg"] += 1

    print(f"\n=== redistillation indices ===")
    print(f"  total to re-distill: {len(indices_to_distill)} / {n_pool}")
    print(f"  v5 sim-changed: {reasons['sim_changed']}")
    print(f"  new negatives: {reasons['new_neg']}")
    print(f"  upsampled positives: {reasons['upsampled_pos']}")
    print(f"  reuse existing CoT: {n_pool - len(indices_to_distill)}")

    out_dir = Path(args.out_dir)
    idx_path = out_dir / "v6_1_redistill_indices.json"
    idx_path.write_text(json.dumps(indices_to_distill))
    print(f"\n[write] {idx_path} ({len(indices_to_distill)} indices)")

    summary = {
        "total_pool": n_pool,
        "redistill_count": len(indices_to_distill),
        "reuse_count": n_pool - len(indices_to_distill),
        "reasons": reasons,
        "sim_thresh": args.sim_thresh,
        "n_v5": n_v5,
    }
    summary_path = out_dir / "v6_1_redistill_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
