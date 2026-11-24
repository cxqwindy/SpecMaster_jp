"""批跑 verify_cases_v5.json 到已起服务的 v5 vLLM。
逐条调用 call_v5_vllm.run_case, 打印一览表 + 汇总。

用法:
  python run_v5_verify.py --vllm-url http://localhost:8000/v1 --model nmr-matching
  python run_v5_verify.py --cases /path/to/other_cases.json
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
from recipe.nmr_matching_agent.call_v5_vllm import run_case  # noqa: E402

DEFAULT_CASES = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_v5.json"
DEFAULT_CACHE = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="nmr-matching")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--cache-path", default=DEFAULT_CACHE)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cases = json.load(open(args.cases))
    cache = json.load(open(args.cache_path))
    print(f"[cache] {len(cache)} entries  |  [cases] {len(cases)}\n")

    print(f"{'idx':<5}{'neg_type':<58}{'gt':<5}{'pred':<8}{'turns':<7}{'lat/s':<8}tools")
    print("-"*160)

    results = []
    n_correct = 0
    for c in cases:
        t0 = time.time()
        res = asyncio.run(run_case(args.vllm_url, args.model, c, cache,
                                    max_turns=args.max_turns, allow_remote_predict=True))
        lat = time.time() - t0
        final = res.get("final") or {}
        pred = final.get("final_answer") or ("FAIL:" + (res.get("fail_reason") or "?"))
        tools = ",".join([r.get("tool") for r in res["transcript"] if r.get("tool")])[:80]
        ok = "✓" if pred == c["gt_answer"] else "✗"
        if pred == c["gt_answer"]:
            n_correct += 1
        nt = c["negative_type"][:56]
        print(f"{c['index']:<5}{nt:<58}{c['gt_answer']:<5}{pred:<8}{res['turns']:<7}{lat:<8.1f}{tools}  {ok}")
        results.append({**res, "index": c["index"], "gt_answer": c["gt_answer"],
                        "negative_type": c["negative_type"], "latency_s": round(lat, 2)})

    print("-"*160)
    print(f"\n通过 {n_correct}/{len(cases)}  (== {len(cases)} 部署工作正常)")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
