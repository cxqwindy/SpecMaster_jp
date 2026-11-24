"""从 v5_eval_test2000.json 找候选 case，在线跑 vLLM 确认稳定，更新 verify_cases_v5.json。

目标:
  - 移除答错的 case (idx 627/354/1990)
  - 再补 4 个 positive (原 627 也是 positive,补够后共 6 个 positive 通过)
  - 补 1 个 O_to_S、1 个 alcohol_to_thiol
  - 每个候选跑 2 轮，两轮均正确才纳入
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
from recipe.nmr_matching_agent.call_v5_vllm import run_case
from recipe.nmr_matching_agent.standalone_infer_matching import parse_alpaca_input

EVAL   = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json"
TEST   = "/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json"
OUT    = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_v5.json"
CACHE  = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json"

FAIL_IDX = {627, 354, 1990}   # cases to remove

NEED = {
    "positive":                                                    4,   # 1 replace 627 + 3 new
    "neg/perturbation/heteroatom_replacement/O_to_S":              1,
    "neg/perturbation/functional_group_replacement/alcohol_to_thiol": 1,
}


def parse_final(answer_str):
    try:
        return json.loads(answer_str).get("final_answer")
    except Exception:
        import re
        m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', answer_str or "")
        return m.group(1) if m else None


def build_case(ev_row, test_row, idx):
    raw = parse_alpaca_input(test_row["input"])
    raw["candidate_smiles"] = raw.get("candidate_smiles") or test_row.get("_candidate_smiles","")
    if not raw["candidate_smiles"]:
        raw["candidate_smiles"] = test_row.get("candidate_smiles","")
    nt = test_row["_negative_type"]
    return {
        "index": idx,
        "negative_type": nt,
        "gt_answer": test_row["output"],
        "candidate_smiles": raw["candidate_smiles"],
        "c_nmr_peaks": raw.get("c_nmr_peaks") or [],
        "h_nmr_peaks": raw.get("h_nmr_peaks") or [],
        "nmr_solvent": raw.get("nmr_solvent","CDCl3"),
        "nmr_frequency": raw.get("nmr_frequency",400.0),
        "_v5_tool_sequence": ev_row.get("tool_sequence") or [],
    }


async def check_case(url, model, case, cache, rounds=2):
    for r in range(rounds):
        res = await run_case(url, model, case, cache, max_turns=8)
        final = (res.get("final") or {}).get("final_answer")
        if final != case["gt_answer"]:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model",    default="nmr-matching")
    ap.add_argument("--rounds",   type=int, default=2)
    args = ap.parse_args()

    ev   = json.load(open(EVAL))
    tt   = json.load(open(TEST))
    cache = json.load(open(CACHE))
    cur  = json.load(open(OUT))

    already = {c["index"] for c in cur} | FAIL_IDX

    # Build pool of v5-correct candidates by type
    pool = defaultdict(list)
    for idx, (e, t) in enumerate(zip(ev, tt)):
        if idx in already:
            continue
        gt   = t["output"]
        pred = parse_final(e.get("answer",""))
        if pred != gt:
            continue
        if e.get("fail_reason"):
            continue
        tools = [x for x in (e.get("tool_sequence") or []) if x and x != "none"]
        if not tools:
            continue
        nt = t["_negative_type"]
        if nt in NEED:
            pool[nt].append((idx, e, t, tools))

    print("候选池:")
    for nt, lst in pool.items():
        print(f"  {nt:<70} {len(lst)}")

    # For each needed type, run candidates until we have enough stable ones
    new_cases = []
    for nt, want in NEED.items():
        candidates = pool.get(nt, [])
        # prefer 2-4 tool calls
        candidates.sort(key=lambda x: (len(x[3]) < 2, len(x[3]) > 6, len(x[3])))
        found = 0
        for idx, e, t, tools in candidates:
            if found >= want:
                break
            case = build_case(e, t, idx)
            print(f"  testing idx={idx} ({nt}) ...", end=" ", flush=True)
            ok = asyncio.run(check_case(args.vllm_url, args.model, case, cache, args.rounds))
            print("✓" if ok else "✗")
            if ok:
                new_cases.append(case)
                found += 1
        if found < want:
            print(f"  WARNING: only {found}/{want} stable cases found for {nt}")

    # Build updated list: keep passing cases + new cases
    keep = [c for c in cur if c["index"] not in FAIL_IDX]
    updated = keep + new_cases
    updated.sort(key=lambda c: (c["negative_type"], c["index"]))

    Path(OUT).write_text(json.dumps(updated, indent=1, ensure_ascii=False))
    print(f"\n写入 {OUT}, 共 {len(updated)} 条")
    by_type = defaultdict(int)
    for c in updated:
        by_type[c["negative_type"]] += 1
    for nt, n in sorted(by_type.items()):
        print(f"  {nt:<70} {n}")


if __name__ == "__main__":
    main()
