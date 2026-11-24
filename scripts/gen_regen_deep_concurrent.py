#!/usr/bin/env python3
"""Concurrent runner for regen_deep CoT generation.

The generate_cot.py main loop is sequential (1 Gemini call at a time + sleep),
which is too slow for 1617 samples. This reuses its call_gemini /
make_output_record / _load_regen_deep with a thread pool and index-keyed cache
so it resumes safely and runs many calls in parallel.

Output: cot_output_regen_deep/cot_traces.jsonl (one line per sample, keyed by
_idx for resume; failed samples get reasoning=null).

Usage:
    python gen_regen_deep_concurrent.py --workers 12
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from perturbation_generator.generate_cot import (
    _load_regen_deep, call_gemini, make_output_record,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--output_dir", default="/fs_mol/cxq/SpecMaster_jp/cot_output_regen_deep/")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "cot_traces.jsonl"

    samples = _load_regen_deep(0, args.seed)
    for idx, s in enumerate(samples):
        s["_idx"] = idx
    print(f"regen_deep total samples: {len(samples)}")

    # Resume: skip indices already done with non-null reasoning.
    done = set()
    if cache.exists():
        with open(cache) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_idx" in r and r.get("reasoning") is not None:
                    done.add(r["_idx"])
    todo = [s for s in samples if s["_idx"] not in done]
    print(f"cached_ok={len(done)} todo={len(todo)}")

    def work(rec):
        reasoning = call_gemini(
            rec, reasoning_level=rec.get("_reasoning_level", "full"),
            max_retries=args.max_retries,
            targeted_hint=rec.get("_targeted_hint", ""),
            provide_label=True,
        )
        if reasoning is None:
            return rec["_idx"], None
        out = make_output_record(rec, reasoning)
        out["reasoning_level"] = rec.get("_reasoning_level", "full")
        out["_idx"] = rec["_idx"]
        out["_targeted_hint"] = rec.get("_targeted_hint", "")
        return rec["_idx"], out

    fout = open(cache, "a", encoding="utf-8")
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            idx, out = fut.result()
            if out is None:
                n_fail += 1
                fout.write(json.dumps({"_idx": idx, "reasoning": None,
                                       "error": "generation_failed"}) + "\n")
            else:
                n_ok += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            if (n_ok + n_fail) % 100 == 0:
                print(f"  {n_ok + n_fail}/{len(todo)}  (ok={n_ok} fail={n_fail})")
    fout.close()
    print(f"Done. ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
