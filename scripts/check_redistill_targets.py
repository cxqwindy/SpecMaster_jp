#!/usr/bin/env python3
"""Identify v6.1 pool rows that need fresh teacher distillation.

A row needs re-distillation if:
  (a) No existing teacher CoT exists for that case_id (new negatives from v6_pool), OR
  (b) The spectral_match similarity in the NEW tool_log differs by > SIM_DELTA_THRESH from
      the old tool_log (teacher cited wrong numbers; old CoT is stale).

Outputs:
  distill_out/v6_1_redistill_needed.jsonl  -- rows to feed back to distill_teacher_cot.py
  distill_out/v6_1_redistill_reuse.jsonl   -- rows where old CoT is still valid
  distill_out/v6_1_redistill_report.json   -- summary statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

DISTILL_DIR = Path("/fs_mol/cxq/SpecMaster_jp/distill_out")
OLD_TOOL_LOG = DISTILL_DIR / "tool_log_pool.jsonl"
NEW_TOOL_LOG = DISTILL_DIR / "v6_1_tool_log.jsonl"
GPT55_DISTILL = DISTILL_DIR / "distill_openai__specmaster__gpt-5.5.jsonl"
SIM_DELTA_THRESH = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_tool_log", default=str(NEW_TOOL_LOG))
    ap.add_argument("--old_tool_log", default=str(OLD_TOOL_LOG))
    ap.add_argument("--teacher_cot", default=str(GPT55_DISTILL))
    ap.add_argument("--sim_thresh", type=float, default=SIM_DELTA_THRESH)
    ap.add_argument("--out_dir", default=str(DISTILL_DIR))
    args = ap.parse_args()

    # Load old tool_log by case_id
    old_sim: Dict[str, float] = {}
    if Path(args.old_tool_log).exists():
        for line in Path(args.old_tool_log).open():
            r = json.loads(line)
            cid = r.get("case_id")
            sim = (r.get("tool_log") or {}).get("spectral_match", {}).get("similarity")
            if cid and sim is not None:
                old_sim[cid] = float(sim)
    print(f"Old tool_log: {len(old_sim)} similarity values loaded")

    # Load existing teacher CoT by case_id (GPT-5.5 is primary teacher)
    existing_cot: Dict[str, Dict] = {}
    if Path(args.teacher_cot).exists():
        for line in Path(args.teacher_cot).open():
            r = json.loads(line)
            cid = r.get("case_id")
            if cid:
                existing_cot[cid] = r
    print(f"Existing teacher CoT: {len(existing_cot)} entries")

    # Process new tool log
    new_rows: List[Dict] = [json.loads(l) for l in Path(args.new_tool_log).open()]
    print(f"New tool_log rows: {len(new_rows)}")

    redistill: List[Dict] = []
    reuse: List[Dict] = []
    reasons: Dict[str, int] = {"no_cot": 0, "sim_changed": 0, "reuse": 0}

    for row in new_rows:
        cid = row.get("case_id")
        new_sim_val = (row.get("tool_log") or {}).get("spectral_match", {}).get("similarity")

        cot = existing_cot.get(cid) if cid else None

        if cot is None:
            # No existing CoT → always re-distill
            row["_redistill_reason"] = "no_cot"
            redistill.append(row)
            reasons["no_cot"] += 1
            continue

        # Has existing CoT — check if similarity changed significantly
        old_s = old_sim.get(cid)
        if old_s is not None and new_sim_val is not None:
            delta = abs(float(new_sim_val) - old_s)
            if delta > args.sim_thresh:
                row["_redistill_reason"] = f"sim_delta={delta:.3f}"
                row["_old_similarity"] = old_s
                row["_new_similarity"] = float(new_sim_val)
                redistill.append(row)
                reasons["sim_changed"] += 1
                continue

        # Reuse: attach existing CoT to row for trajectory building
        row["_existing_cot"] = cot
        reuse.append(row)
        reasons["reuse"] += 1

    print(f"\n=== redistillation analysis ===")
    print(f"  need re-distill: {len(redistill)}")
    print(f"    - no CoT exists (new rows): {reasons['no_cot']}")
    print(f"    - similarity changed > {args.sim_thresh}: {reasons['sim_changed']}")
    print(f"  can reuse existing CoT: {len(reuse)}")

    # Separate redistill by class for cost estimation
    by_class: Dict[str, int] = {}
    for r in redistill:
        cls = r.get("_class") or r.get("negative_type") or ("positive" if r.get("label")==1 else "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1
    print(f"\n  redistill by class:")
    for k, v in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")

    out_dir = Path(args.out_dir)
    needed_path = out_dir / "v6_1_redistill_needed.jsonl"
    reuse_path  = out_dir / "v6_1_redistill_reuse.jsonl"
    report_path = out_dir / "v6_1_redistill_report.json"

    with needed_path.open("w") as f:
        for r in redistill:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[write] {needed_path} ({len(redistill)} rows)")

    with reuse_path.open("w") as f:
        for r in reuse:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {reuse_path} ({len(reuse)} rows)")

    report = {
        "total": len(new_rows),
        "redistill_needed": len(redistill),
        "redistill_reuse": len(reuse),
        "reasons": reasons,
        "redistill_by_class": by_class,
        "sim_delta_threshold": args.sim_thresh,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
