#!/usr/bin/env python3
"""Extract pool indices where GPT-5.5 teacher disagrees with GT label, rebuild their
tool_logs with the fixed runtime (c_data_absent field), and write the redistill index
list for GPT-6-Astra to re-run.

Input:
  distill_out/trajectories_v6_1.jsonl  — existing trajectories (meta.label, meta.final_answer, meta.teacher_pred)
  distill_out/v6_1_pool.jsonl           — original pool (stable-ordered, same as trajectory i)
  distill_out/v6_1_tool_log.jsonl       — old tool logs (to be patched with c_data_absent)

Output:
  distill_out/v6_1_astra_indices.json       — list of pool indices to re-distill with Astra
  distill_out/v6_1_tool_log_patched.jsonl   — all 4220 rows with c_data_absent in spectral_match
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa

CACHE_PATH = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json"
TRAJ_PATH  = "/fs_mol/cxq/SpecMaster_jp/distill_out/trajectories_v6_1.jsonl"
POOL_PATH  = "/fs_mol/cxq/SpecMaster_jp/distill_out/v6_1_pool.jsonl"
OLD_TLOG   = "/fs_mol/cxq/SpecMaster_jp/distill_out/v6_1_tool_log.jsonl"
OUT_IDX    = "/fs_mol/cxq/SpecMaster_jp/distill_out/v6_1_astra_indices.json"
OUT_TLOG   = "/fs_mol/cxq/SpecMaster_jp/distill_out/v6_1_tool_log_patched.jsonl"


async def run_spectral_match(record: Dict[str, Any], pred_cache: Dict[str, Any]) -> Dict[str, Any]:
    rt = NmrMatchingRuntime(record, prediction_cache=pred_cache, allow_remote_predict=False)
    res = await rt.run_tool_call_async({"function": {"name": "spectral_match", "arguments": {}}})
    try:
        return json.loads(res.get("content", "{}"))
    except Exception:
        return {}


def main():
    print("[load] prediction cache...", flush=True)
    with open(CACHE_PATH) as f:
        pred_cache = json.load(f)
    print(f"  {len(pred_cache)} entries", flush=True)

    print("[load] trajectories...", flush=True)
    trajs: List[Dict[str, Any]] = []
    with open(TRAJ_PATH) as f:
        for line in f:
            trajs.append(json.loads(line))
    print(f"  {len(trajs)} trajectories", flush=True)

    print("[load] pool...", flush=True)
    with open(POOL_PATH) as f:
        pool = [json.loads(l) for l in f]
    print(f"  {len(pool)} pool rows", flush=True)

    print("[load] old tool_log...", flush=True)
    with open(OLD_TLOG) as f:
        old_tlog = [json.loads(l) for l in f]
    print(f"  {len(old_tlog)} tool_log rows", flush=True)

    # --- identify inconsistent trajectory indices ---
    # trajectories are indexed 0..N-1 and correspond to pool rows by stable order.
    # (pool has 4220 rows; one row had no teacher → only 4219 trajectories were kept.
    #  We need to recover the pool index for each trajectory via candidate_smiles match.)
    smiles_to_pool_i: Dict[str, int] = {}
    for i, row in enumerate(pool):
        smi = row["input"]["candidate_smiles"]
        smiles_to_pool_i[smi] = i

    inconsistent_pool_indices: List[int] = []
    for traj in trajs:
        meta = traj.get("meta", {})
        label = meta.get("label")
        final_answer = meta.get("final_answer")
        gt = "Yes" if label == 1 else "No"
        if final_answer != gt:
            smi = meta.get("candidate_smiles", "")
            pi = smiles_to_pool_i.get(smi)
            if pi is not None:
                inconsistent_pool_indices.append(pi)

    inconsistent_pool_indices = sorted(set(inconsistent_pool_indices))
    print(f"\n[found] {len(inconsistent_pool_indices)} inconsistent pool indices to redistribute", flush=True)
    Path(OUT_IDX).write_text(json.dumps(inconsistent_pool_indices))
    print(f"[write] {OUT_IDX}", flush=True)

    # --- rebuild tool_log with fixed runtime (adds c_data_absent field) ---
    # We rebuild spectral_match for ALL rows so the patched tool_log is complete.
    # Other tools (count_atoms, fg_scan) are deterministic and unchanged — copy from old log.
    loop = asyncio.new_event_loop()
    n_patched = 0
    with open(OUT_TLOG, "w") as fo:
        for i, row in enumerate(old_tlog):
            inp = row["input"]
            record = {
                "candidate_smiles": inp["candidate_smiles"],
                "c_nmr_peaks": inp.get("c_nmr_peaks") or [],
                "h_nmr_peaks": inp.get("h_nmr_peaks") or [],
            }
            new_spec = loop.run_until_complete(run_spectral_match(record, pred_cache))
            old_spec = row.get("tool_log", {}).get("spectral_match", {})
            if new_spec.get("c_data_absent") != old_spec.get("c_data_absent"):
                n_patched += 1
            row.setdefault("tool_log", {})["spectral_match"] = new_spec
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  patched {i+1}/{len(old_tlog)}", flush=True)

    print(f"[done] rebuilt tool_log: {len(old_tlog)} rows, {n_patched} spectral_match entries changed", flush=True)
    print(f"[write] {OUT_TLOG}", flush=True)


if __name__ == "__main__":
    main()
