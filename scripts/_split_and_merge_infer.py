#!/usr/bin/env python3
"""Helper for run_infer_parallel.sh: split a test JSON into N shards, or merge shard results.

split:  python _split_and_merge_infer.py split --input test.json --n 8 --outdir /tmp/dir
merge:  python _split_and_merge_infer.py merge --inputs r0.json r1.json ... --output merged.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def cmd_split(args):
    data = json.load(open(args.input))
    n = args.n
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    total = len(data)
    chunk = (total + n - 1) // n
    for i in range(n):
        shard = data[i * chunk: (i + 1) * chunk]
        out = outdir / f"shard_{i}.json"
        out.write_text(json.dumps(shard, ensure_ascii=False))
        print(f"  shard_{i}.json: {len(shard)} rows")
    print(f"Split {total} rows into {n} shards → {outdir}")


def cmd_merge(args):
    merged = []
    for path in args.inputs:
        part = json.load(open(path))
        merged.extend(part)
    Path(args.output).write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Merged {len(merged)} rows → {args.output}")


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

    args = ap.parse_args()
    if args.cmd == "split":
        cmd_split(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
