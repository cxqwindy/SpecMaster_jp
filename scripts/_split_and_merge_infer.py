#!/usr/bin/env python3
"""Helper for run_infer_parallel.sh: split a test JSON into N shards, or merge shard results.

split writes copies. The source file is not modified. Each copied row receives a
global ``_harness_pair_id`` and a run id before the file is cut into shards.
``shard_manifest.json`` records the expected ids. Those keys are harness
bookkeeping; inference must not copy them into the model task.

merge restores global order from the ids stamped onto prediction rows. It
rejects duplicate ids, missing ids, rows from another run, and a manifest
whose results have no harness ids. Prediction rows that have no harness ids
can be concatenated only when no manifest is passed. That read keeps the given
input order, reports the reorder as unverified, and does not invent global ids.

split:  python _split_and_merge_infer.py split --input test.json --n 8 --outdir /tmp/dir
merge:  python _split_and_merge_infer.py merge --inputs r0.json r1.json --manifest /tmp/dir/shard_manifest.json --output merged.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

_OA = os.environ.get("OPEN_AGENTRL_ROOT", "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
if _OA not in sys.path:
    sys.path.insert(0, _OA)

from recipe.nmr_matching_agent.standalone_infer_matching import merge_eval_results  # noqa: E402
from recipe.nmr_matching_agent.task_input import (  # noqa: E402
    HARNESS_PAIR_ID_KEY,
    HARNESS_RUN_ID_KEY,
)

MANIFEST_NAME = "shard_manifest.json"


def cmd_split(args):
    with open(args.input) as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit("split input must be a JSON list")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    stamped = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise SystemExit(f"row {index} is not an object")
        item = dict(row)
        item[HARNESS_PAIR_ID_KEY] = index
        item[HARNESS_RUN_ID_KEY] = run_id
        stamped.append(item)
    total = len(stamped)
    chunk = (total + args.n - 1) // args.n if total else 0
    shards = []
    for shard_index in range(args.n):
        shard = stamped[shard_index * chunk:(shard_index + 1) * chunk] if chunk else []
        name = f"shard_{shard_index}.json"
        (outdir / name).write_text(json.dumps(shard, ensure_ascii=False))
        pair_ids = [row[HARNESS_PAIR_ID_KEY] for row in shard]
        shards.append({"name": name, "pair_ids": pair_ids})
        print(f"  {name}: {len(shard)} rows")
    manifest = {
        "harness_run_id": run_id,
        "source_rows": total,
        "pair_ids": list(range(total)),
        "shards": shards,
    }
    (outdir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"Split {total} rows into {args.n} shards → {outdir}")
    return manifest


def cmd_merge(args):
    parts = []
    for path in args.inputs:
        with open(path) as handle:
            parts.append(json.load(handle))
    manifest = None
    if args.manifest:
        with open(args.manifest) as handle:
            manifest = json.load(handle)
    outcome = merge_eval_results(parts, manifest)
    Path(args.output).write_text(json.dumps(outcome["rows"], ensure_ascii=False, indent=2))
    if outcome["reorder_verified"]:
        print(
            f"Merged {len(outcome['rows'])} rows → {args.output} "
            f"(run {outcome['harness_run_id']}; reordered by global harness pair id)"
        )
    else:
        print(
            f"Merged {len(outcome['rows'])} rows → {args.output} "
            "(legacy input order; these rows have no harness pair ids, so cross-shard "
            "reorder was not verified and no global ids were invented)",
            file=sys.stderr,
        )
        print(f"Merged {len(outcome['rows'])} rows → {args.output}")
    return outcome


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split")
    sp.add_argument("--input", required=True)
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--outdir", required=True)

    mp = sub.add_parser("merge")
    mp.add_argument("--inputs", nargs="+", required=True)
    mp.add_argument("--output", required=True)
    mp.add_argument("--manifest", default=None, help="shard_manifest.json written by split")

    args = ap.parse_args()
    if args.cmd == "split":
        cmd_split(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
