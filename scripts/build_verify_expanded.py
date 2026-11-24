"""扩展 verify 集:从 test_1000 里筛出当前部署 2 次一致答对的样例。

依赖:vLLM 已经在 http://localhost:8000 起服务(nmr-matching)。
输出:/fs_mol/cxq/SpecMaster_jp/data/verify_cases_expanded.json
"""
import json
import re
import random
import time
import httpx
from rdkit import Chem

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nmr-matching"
TEST = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json"
PRED = "/fs_mol/cxq/SpecMaster_jp/eval_results_rl_grpo/step_180/predictions.jsonl"
OUT = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_expanded.json"

ANS_RE = re.compile(r'"final_answer"\s*:\s*"(Yes|No)"')


def parse(t):
    t = (t or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and obj.get("final_answer") in ("Yes", "No"):
            return obj["final_answer"]
    except Exception:
        pass
    m = ANS_RE.search(t)
    if m:
        return m.group(1)
    if "Yes" in t and "No" not in t:
        return "Yes"
    if "No" in t and "Yes" not in t:
        return "No"
    return "Unknown"


def clean_smi(smi):
    mr = Chem.MolFromSmiles(smi, sanitize=False)
    if mr is None:
        return False
    if Chem.DetectChemistryProblems(mr):
        return False
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    if any(a.GetFormalCharge() != 0 for a in m.GetAtoms()):
        return False
    try:
        if Chem.MolToSmiles(mr) != Chem.MolToSmiles(m):
            return False
    except Exception:
        return False
    return True


def call(uc):
    r = httpx.post(URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": uc}],
        "temperature": 0,
        "max_tokens": 2048,
    }, timeout=180)
    r.raise_for_status()
    return parse(r.json()["choices"][0]["message"]["content"])


def main():
    test = json.load(open(TEST))
    step180_correct = set()
    for l in open(PRED):
        d = json.loads(l)
        if d["correct"]:
            step180_correct.add(d["index"])
    print(f"test_1000={len(test)}  step180_correct={len(step180_correct)}")

    used = {0, 13, 3, 5, 8, 37, 173, 4, 7}
    pool = {}
    for i, d in enumerate(test):
        if i in used or i not in step180_correct:
            continue
        if not clean_smi(d["_candidate_smiles"]):
            continue
        pool.setdefault(d["_negative_type"], []).append((i, d))

    TARGET = {
        "positive": 8,
        "neg/same_formula": 6,
        "neg/similar_structure": 4,
        "neg/random": 4,
        "neg/hard_nmr/alkyl_chain_shift/insert_ch2": 3,
        "neg/perturbation/heteroatom_replacement/O_to_S": 2,
        "neg/perturbation/functional_group_replacement/alcohol_to_thiol": 2,
        "neg/perturbation/alkyl_chain_shift/insert_ch2": 2,
    }

    random.seed(42)
    sampled = []
    for nt, want in TARGET.items():
        lst = pool.get(nt, [])
        random.shuffle(lst)
        take = lst[:want * 3]
        sampled.extend([(i, d, nt) for i, d in take])
        print(f"  [{nt}] pool={len(lst)}, sampled(3x)={len(take)}")
    print(f"\n合计采样 {len(sampled)} 条, 开始 Round 1 & 2...")

    # 3 轮一致才留下, 更严格避免 vLLM tie-break 边缘翻转
    N_ROUNDS = 3
    results = []
    t0 = time.time()
    for k, (i, d, nt) in enumerate(sampled):
        uc = d["instruction"] + "\n\n" + d["input"]
        all_ok = True
        preds = []
        for rn in range(N_ROUNDS):
            try:
                p = call(uc)
            except Exception as e:
                p = f"ERR:{e}"
            preds.append(p)
            if p != d["output"]:
                all_ok = False
                break  # 早停,任何一轮失败就放弃
        elapsed = time.time() - t0
        preds_str = "/".join(preds)
        tag = "OK" if all_ok else "NG"
        print(f"  [{k+1:3d}/{len(sampled)}] idx={i:<4} {nt[:34]:<34} "
              f"gt={d['output']:<3} preds={preds_str:<20} {tag}  ({elapsed:.0f}s)")
        if all_ok:
            results.append((i, d, nt))

    final_by_type = {}
    for i, d, nt in results:
        final_by_type.setdefault(nt, []).append((i, d))

    out = []
    print("\n最终每类:")
    for nt, want in TARGET.items():
        picks = final_by_type.get(nt, [])[:want]
        stable = len(final_by_type.get(nt, []))
        print(f"  [{nt}] stable={stable} kept={len(picks)}/{want}")
        for i, d in picks:
            out.append({
                "index": i,
                "negative_type": nt,
                "gt_answer": d["output"],
                "step180_pred": d["output"],
                "candidate_smiles": d["_candidate_smiles"],
                "user_content": d["instruction"] + "\n\n" + d["input"],
            })

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\n写入 {OUT}, 共 {len(out)} 条, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
