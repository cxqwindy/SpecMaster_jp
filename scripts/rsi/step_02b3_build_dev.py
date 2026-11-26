#!/usr/bin/env python3
"""CLI for step 02B-3 development-set construction.

The command does not start a model, a predictor, a router, or a training job.
"""

from __future__ import annotations

import argparse
import json
import sys

from step_02b3_dev import execute_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the 02B-3 development sets.")
    parser.add_argument("command", choices=["plan", "build", "validate", "rebuild"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["real", "fixture"], default="real")
    parser.add_argument("--index")
    parser.add_argument("--records")
    parser.add_argument("--groups", type=int)
    parser.add_argument("--reuse-cache")
    parser.add_argument("--from-run")
    args = parser.parse_args(argv)
    if args.command == "rebuild" and not args.from_run:
        print(json.dumps({"status": "refused", "reason": "rebuild_requires_from_run", "exit": 3}))
        return 3
    result = execute_cli({
        "command": args.command,
        "out": args.out,
        "mode": args.mode,
        "index": args.index,
        "records": args.records,
        "groups": args.groups,
        "reuse_cache": args.reuse_cache,
        "from_run": args.from_run,
    })
    print(json.dumps({
        "exit": result.get("exit"),
        "logical_data_sha256": result.get("logical_data_sha256"),
        "reason": result.get("reason"),
        "status": result.get("status"),
    }, ensure_ascii=False, sort_keys=True))
    return int(result["exit"])


if __name__ == "__main__":
    sys.exit(main())
