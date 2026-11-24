#!/usr/bin/env python3
"""Post-process GPT-6-Astra redistillation results.

Steps:
  1. Load Astra distill output (352 rows).
  2. Keep only rows where pred matches GT label ("rescued").
  3. For rescued rows, build Hermes trajectories via teacher_cot_to_trajectory.build_trajectory().
  4. Load trajectories_v6_1.jsonl; remove the 352 original inconsistent rows (by candidate_smiles).
  5. Append rescued trajectories and write the merged output.

Output:
  distill_out/trajectories_v6_1.jsonl  (overwritten in-place)
  distill_out/trajectories_v6_1_astra_rescued.jsonl  (rescued-only, for inspection)
  distill_out/merge_astra_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.teacher_cot_to_trajectory import build_trajectory  # noqa

DISTILL_OUT = Path("/fs_mol/cxq/SpecMaster_jp/distill_out")

ASTRA_DISTILL = DISTILL_OUT / "v6_1/distill_openai__specxmaster__gpt-6-astra.jsonl"
ASTRA_INDICES = DISTILL_OUT / "v6_1_astra_indices.json"
POOL_PATH     = DISTILL_OUT / "v6_1_pool.jsonl"
TOOL_LOG_PATH = DISTILL_OUT / "v6_1_tool_log_patched.jsonl"
TRAJ_PATH     = DISTILL_OUT / "trajectories_v6_1.jsonl"

OUT_RESCUED   = DISTILL_OUT / "trajectories_v6_1_astra_rescued.jsonl"
SUMMARY_PATH  = DISTILL_OUT / "merge_astra_summary.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    print("[load] pool...", flush=True)
    pool = load_jsonl(POOL_PATH)
    pool_by_i: Dict[int, Dict[str, Any]] = {i: row for i, row in enumerate(pool)}

    print("[load] patched tool_log...", flush=True)
    tool_log_rows = load_jsonl(TOOL_LOG_PATH)
    tool_log_by_i: Dict[int, Dict[str, Any]] = {i: row for i, row in enumerate(tool_log_rows)}

    print("[load] Astra distill results...", flush=True)
    astra_rows = load_jsonl(ASTRA_DISTILL)
    print(f"  {len(astra_rows)} rows", flush=True)

    # Filter: keep only rows where pred matches GT
    rescued: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    for r in astra_rows:
        label = r.get("label")
        pred  = r.get("pred", "Unknown")
        gt_ans = "Yes" if label == 1 else "No"
        if pred == gt_ans:
            rescued.append(r)
        else:
            discarded.append(r)

    print(f"  rescued: {len(rescued)}, discarded: {len(discarded)}", flush=True)

    # Build set of candidate_smiles that were in the inconsistent pool
    inconsistent_indices = set(json.loads(ASTRA_INDICES.read_text()))
    inconsistent_smiles: set = {
        pool_by_i[i]["input"]["candidate_smiles"]
        for i in inconsistent_indices
        if i in pool_by_i
    }
    print(f"  inconsistent pool indices: {len(inconsistent_indices)}", flush=True)

    # Build trajectories for rescued rows
    print("[build] trajectories for rescued rows...", flush=True)
    new_trajs: List[Dict[str, Any]] = []
    n_prov_fail = 0
    for r in rescued:
        pi = r.get("i")
        if pi is None or pi not in tool_log_by_i:
            print(f"  WARN: pool index {pi} not in tool_log, skip", flush=True)
            continue
        pool_row = tool_log_by_i[pi]
        traj = build_trajectory(pool_row, r)
        if traj is None:
            continue
        if traj["meta"].get("provenance_violations"):
            n_prov_fail += 1
        new_trajs.append(traj)

    print(f"  built {len(new_trajs)} trajectories, provenance_fail={n_prov_fail}", flush=True)

    # Write rescued-only file for inspection
    with open(OUT_RESCUED, "w") as f:
        for t in new_trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[write] {OUT_RESCUED}", flush=True)

    # Load existing trajectories, remove the 352 inconsistent ones
    print("[load] existing trajectories_v6_1.jsonl...", flush=True)
    existing_trajs = load_jsonl(TRAJ_PATH)
    n_before = len(existing_trajs)
    kept_trajs = [
        t for t in existing_trajs
        if t.get("meta", {}).get("candidate_smiles") not in inconsistent_smiles
    ]
    n_removed = n_before - len(kept_trajs)
    print(f"  removed {n_removed} inconsistent rows from {n_before} existing", flush=True)

    # Merge: kept originals + rescued Astra trajectories
    merged = kept_trajs + new_trajs
    print(f"  merged total: {len(merged)} ({len(kept_trajs)} kept + {len(new_trajs)} rescued)", flush=True)

    # Overwrite trajectories_v6_1.jsonl
    with open(TRAJ_PATH, "w") as f:
        for t in merged:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[write] {TRAJ_PATH}  ({len(merged)} rows)", flush=True)

    # Label balance check
    from collections import Counter
    label_dist = Counter(t["meta"].get("label") for t in merged)
    ans_dist   = Counter(t["meta"].get("final_answer") for t in merged)
    print(f"  label dist: {dict(label_dist)}", flush=True)
    print(f"  final_answer dist: {dict(ans_dist)}", flush=True)

    summary = {
        "astra_total": len(astra_rows),
        "rescued": len(rescued),
        "discarded": len(discarded),
        "new_trajectories_built": len(new_trajs),
        "provenance_fail": n_prov_fail,
        "removed_from_existing": n_removed,
        "merged_total": len(merged),
        "label_dist": {str(k): v for k, v in label_dist.items()},
        "final_answer_dist": {str(k): v for k, v in ans_dist.items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[write] {SUMMARY_PATH}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
