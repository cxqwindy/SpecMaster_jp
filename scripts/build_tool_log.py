#!/usr/bin/env python3
"""Pre-run the four deterministic matching tools on every candidate in the distill pool.

Purpose: the Path-A teacher distillation prompt injects a TOOL_LOG block containing the exact
JSON that count_atoms(C/H), spectral_match, and functional_group_scan return. The teacher then
"reproduces" the reasoning grounded in those actual numbers — the root fix for the "Gemini
guesses 23 carbons" hallucination that seeded v4's noisy think blocks.

Input : distill_out/distill_pool.jsonl  (from build_distill_pool.py)
Output: distill_out/tool_log_pool.jsonl (same rows + tool_log field)

Tools invoked per row:
  * count_atoms(element=C)
  * count_atoms(element=H)
  * spectral_match()
  * functional_group_scan()
  (check_substructure is pattern-conditional; teacher can name specific patterns in its CoT.)

Requires: prediction_cache.json to cover the candidates (run precompute_matching_cache.py first).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402


async def run_tools(record: Dict[str, Any], pred_cache: Dict[str, Any]) -> Dict[str, Any]:
    runtime = NmrMatchingRuntime(record, prediction_cache=pred_cache, allow_remote_predict=False)
    tools_to_run = [
        ("count_atoms_C", {"name": "count_atoms", "arguments": {"element": "C"}}),
        ("count_atoms_H", {"name": "count_atoms", "arguments": {"element": "H"}}),
        ("spectral_match", {"name": "spectral_match", "arguments": {}}),
        ("functional_group_scan", {"name": "functional_group_scan", "arguments": {}}),
    ]
    log: Dict[str, Any] = {}
    for key, call in tools_to_run:
        res = await runtime.run_tool_call_async({"function": call})
        try:
            log[key] = json.loads(res.get("content", "{}"))
        except Exception:
            log[key] = {"error": "unparseable", "raw": res.get("content", "")}
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/fs_mol/cxq/SpecMaster_jp/distill_out/distill_pool.jsonl")
    ap.add_argument("--pred_cache", default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json")
    ap.add_argument("--out", default="/fs_mol/cxq/SpecMaster_jp/distill_out/tool_log_pool.jsonl")
    args = ap.parse_args()

    with open(args.pred_cache) as f:
        pred_cache = json.load(f)
    print(f"[cache] loaded {len(pred_cache)} predictions", flush=True)

    loop = asyncio.new_event_loop()
    n_ok = n_spec_ok = n_total = 0
    with open(args.inp) as fi, open(args.out, "w") as fo:
        for line in fi:
            row = json.loads(line)
            n_total += 1
            inp = row["input"]
            record = {
                "candidate_smiles": inp["candidate_smiles"],
                "c_nmr_peaks": inp.get("c_nmr_peaks") or [],
                "h_nmr_peaks": inp.get("h_nmr_peaks") or [],
                "nmr_solvent": inp.get("nmr_solvent"),
                "nmr_frequency": inp.get("nmr_frequency"),
            }
            log = loop.run_until_complete(run_tools(record, pred_cache))
            row["tool_log"] = log
            if log.get("spectral_match", {}).get("ok"):
                n_spec_ok += 1
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"[progress] {n_ok}/{n_total}  spec_ok={n_spec_ok}", flush=True)
    print(f"[done] {n_ok}/{n_total} candidates processed; spectral_match ok = {n_spec_ok} "
          f"({100*n_spec_ok/max(n_total,1):.1f}%)", flush=True)
    print(f"[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
