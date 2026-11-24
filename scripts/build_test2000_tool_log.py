#!/usr/bin/env python3
"""Build tool_log for every row in nmr_matching_test_2000.json.

Input schema is alpaca-style (from LLaMA-Factory data prep): each row has an `input` string
containing 'Candidate SMILES:' + '13C NMR:' + '1H NMR:' lines, plus `_label` / `_negative_type`
side-channel. We parse the SMILES + peak lists (reuse the parser from standalone_infer_matching)
and pre-run all four deterministic tools against them.

Output: /tmp/test2000_tool_log.jsonl (one row per test case, tool_log field attached).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402
from recipe.nmr_matching_agent.standalone_infer_matching import parse_alpaca_input  # noqa: E402


async def run_tools(record, pred_cache):
    runtime = NmrMatchingRuntime(record, prediction_cache=pred_cache, allow_remote_predict=False)
    tools = [
        ("count_atoms_C", {"name": "count_atoms", "arguments": {"element": "C"}}),
        ("count_atoms_H", {"name": "count_atoms", "arguments": {"element": "H"}}),
        ("spectral_match", {"name": "spectral_match", "arguments": {}}),
        ("functional_group_scan", {"name": "functional_group_scan", "arguments": {}}),
    ]
    out = {}
    for key, call in tools:
        r = await runtime.run_tool_call_async({"function": call})
        try:
            out[key] = json.loads(r.get("content", "{}"))
        except Exception:
            out[key] = {"error": "unparseable"}
    return out


def main():
    src = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
    cache_p = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json"
    out = "/fs_mol/cxq/SpecMaster_jp/eval_results_testset_api_2000/test2000_tool_log.jsonl"

    with open(cache_p) as f:
        pred_cache = json.load(f)
    print(f"[cache] {len(pred_cache)} entries", flush=True)

    data = json.load(open(src))
    loop = asyncio.new_event_loop()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_spec_ok = 0
    with open(out, "w") as fo:
        for i, row in enumerate(data):
            parsed = parse_alpaca_input(row["input"])
            record = {
                "candidate_smiles": parsed["candidate_smiles"],
                "c_nmr_peaks": parsed["c_nmr_peaks"],
                "h_nmr_peaks": parsed["h_nmr_peaks"],
            }
            log = loop.run_until_complete(run_tools(record, pred_cache))
            if log.get("spectral_match", {}).get("ok"):
                n_spec_ok += 1
            fo.write(json.dumps({
                "i": i,
                "_label": row.get("_label"),
                "_negative_type": row.get("_negative_type"),
                "candidate_smiles": parsed["candidate_smiles"],
                "tool_log": log,
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(data)}  spec_ok={n_spec_ok}", flush=True)
    print(f"[done] {len(data)} rows; spectral_match ok = {n_spec_ok} ({100*n_spec_ok/len(data):.1f}%)", flush=True)
    print(f"[write] {out}", flush=True)


if __name__ == "__main__":
    main()
