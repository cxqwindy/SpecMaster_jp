#!/usr/bin/env python3
"""Evaluate closed-source LLMs as binary NMR matchers on FairNMR-500 rerank,
using the SAME protocol as our SFT model (evaluate_rerank_fairnmr500.py):
score every candidate Yes/No + confidence, then rank.

This gives an apples-to-apples capability ceiling: how well do Gemini/Opus/GPT
do the exact task our model does, so we can tell whether our 41.8% reflects a
model-training gap or intrinsic task difficulty.

Calls models through the LiteLLM gateway with a thread pool, caches each
(model, idx, cand_pos) verdict to a JSONL so re-runs resume.

Usage:
    python evaluate_rerank_fairnmr500_api.py --model openai/specxmaster/gemini-3.1-pro-preview
    python evaluate_rerank_fairnmr500_api.py --model openai/specmaster/gpt-5.5
    python evaluate_rerank_fairnmr500_api.py --model openai/specxmaster/global.anthropic.claude-opus-4-8
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem, RDLogger
import litellm
from litellm import completion

litellm.drop_params = True

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key

RDLogger.DisableLog("rdApp.*")

# Reuse the exact input formatting + parsing from the vLLM script.
sys.path.insert(0, "/fs_mol/cxq/SpecMaster_jp/scripts")
from evaluate_rerank_fairnmr500 import (
    DATA, INSTRUCTION, canon, format_input, parse_yes_conf,
)

API_BASE = "https://ai-gateway-global.dp.tech"

# Same system prompt style as our SFT training (short structured JSON).
SYSTEM = (
    "You are an NMR spectrum-structure matching expert. Given NMR data and a "
    "candidate SMILES, determine whether they match. You do NOT have the target "
    "molecular formula. Output ONLY a JSON object: "
    '{"final_answer": "Yes" or "No", "confidence": <float 0.0-1.0>, '
    '"key_evidence": "<one sentence>"}'
)


def score_one(model, user_prompt, max_retries=8, max_tokens=16384):
    api_key = require_gateway_api_key()
    import time
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model, api_key=api_key, api_base=API_BASE,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens, temperature=0.0, timeout=180,
            )
            text = resp.choices[0].message.content or ""
            if not text and hasattr(resp.choices[0].message, "reasoning_content"):
                text = resp.choices[0].message.reasoning_content or ""
            ans, conf = parse_yes_conf(text)
            # Empty/garbage parse -> treat as transient, retry with backoff.
            if ans == "Unknown" and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return ans, conf
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                return "Unknown", 0.5
            # Rate-limit (Vertex AI for Gemini) needs long backoff.
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg or "resource" in msg:
                time.sleep(min(60, 5 * (2 ** attempt)))  # 5,10,20,40,60...
            else:
                time.sleep(3 * (attempt + 1))
    return "Unknown", 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate_source", default="rerank_topk",
                    choices=["rerank_topk", "pred_topk"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/eval_results_rerank_fairnmr500_api/")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "__")
    cache_path = out_dir / f"verdicts_{safe}.jsonl"

    # Load dataset
    records = []
    with open(DATA) as f:
        for line in f:
            records.append(json.loads(line)["original_data"])
    n = len(records)

    # Build jobs
    jobs = []  # (ri, pos, smi, user_prompt)
    for ri, od in enumerate(records):
        c_peaks, h_peaks = od.get("c_nmr_peaks"), od.get("h_nmr_peaks")
        solvent = od.get("nmr_solvent", "") or ""
        for pos, c in enumerate(od[args.candidate_source]):
            smi = c.get("smiles_std") or c.get("smiles")
            if not smi:
                continue
            user = INSTRUCTION + "\n\n" + format_input(smi, c_peaks, h_peaks, solvent)
            jobs.append((ri, pos, smi, user))

    # Resume from cache. A cached "Unknown" means the API call failed last
    # time (retries exhausted) — treat it as NOT done so it gets retried.
    done = {}
    n_unknown_cached = 0
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                if r["ans"] == "Unknown":
                    n_unknown_cached += 1
                    continue  # will retry
                done[(r["ri"], r["pos"])] = (r["ans"], r["conf"])
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"[{args.model}] total={len(jobs)} good_cached={len(done)} "
          f"unknown_to_retry={n_unknown_cached} todo={len(todo)}")

    # Run remaining with thread pool, append to cache
    fout = open(cache_path, "a", encoding="utf-8")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2job = {ex.submit(score_one, args.model, j[3]): j for j in todo}
        for fut in as_completed(fut2job):
            ri, pos, smi, _ = fut2job[fut]
            ans, conf = fut.result()
            done[(ri, pos)] = (ans, conf)
            fout.write(json.dumps({"ri": ri, "pos": pos, "ans": ans,
                                   "conf": conf}, ensure_ascii=False) + "\n")
            fout.flush()
            completed += 1
            if completed % 500 == 0:
                print(f"  {completed}/{len(todo)} done")
    fout.close()

    # --- Build per-record detail and score (same logic as vLLM script) ---
    detail = {}
    for (ri, pos, smi, _) in jobs:
        ans, conf = done.get((ri, pos), ("Unknown", 0.5))
        detail.setdefault(ri, []).append({"pos": pos, "smiles": smi,
                                          "ans": ans, "conf": conf})
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od[args.candidate_source]
        for d in detail.get(ri, []):
            ts = cands[d["pos"]].get("total_score")
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                ts = float("inf")
            d["cost"] = ts
            d["is_gt"] = (canon(d["smiles"]) == gt)

    def model_signed(d):
        return d["conf"] if d["ans"] == "Yes" else -d["conf"]

    baseline = opt1 = opt2 = 0
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od[args.candidate_source]
        base_pick = canon(cands[0].get("smiles_std") or cands[0].get("smiles")) if cands else None
        if base_pick == gt:
            baseline += 1
        cs = detail.get(ri, [])
        if cs and max(cs, key=model_signed)["is_gt"]:
            opt1 += 1
        cs_sorted = sorted(cs, key=lambda d: d["pos"])
        pick = next((d for d in cs_sorted if d["ans"] == "Yes"), None)
        opt2_pick = canon(pick["smiles"]) if pick else base_pick
        if opt2_pick == gt:
            opt2 += 1

    # Fusion sweep
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    fusion = {}
    for a in alphas:
        c = 0
        for ri in range(n):
            cs = detail.get(ri, [])
            if not cs:
                continue
            m = len(cs)
            crs = {d["pos"]: 1.0 - d["pos"] / m for d in cs}
            mns = {d["pos"]: (model_signed(d) + 1.0) / 2.0 for d in cs}
            if max(cs, key=lambda d: a * crs[d["pos"]] + (1 - a) * mns[d["pos"]])["is_gt"]:
                c += 1
        fusion[a] = c

    ceiling = sum(1 for ri in range(n) if any(d["is_gt"] for d in detail.get(ri, [])))
    n_unknown = sum(1 for v in done.values() if v[0] == "Unknown")

    print("\n" + "=" * 60)
    print(f"RERANK — {args.model}")
    print("=" * 60)
    print(f"Ceiling:                       {ceiling}/{n} ({100*ceiling/n:.1f}%)")
    print(f"Baseline (cost top-1):         {baseline}/{n} ({100*baseline/n:.1f}%)")
    print(f"Option 1 (confidence):         {opt1}/{n} ({100*opt1/n:.1f}%)")
    print(f"Option 2 (first Yes):          {opt2}/{n} ({100*opt2/n:.1f}%)")
    print(f"Unknown/failed verdicts:       {n_unknown}")
    print("--- fusion (alpha=1 -> pure cost) ---")
    for a in alphas:
        print(f"  alpha={a:.1f}: {fusion[a]}/{n} ({100*fusion[a]/n:.1f}%)")

    summary = {
        "model": args.model, "n": n,
        "ceiling_acc": round(ceiling / n, 4),
        "baseline_acc": round(baseline / n, 4),
        "option1_acc": round(opt1 / n, 4),
        "option2_acc": round(opt2 / n, 4),
        "fusion_alpha_acc": {f"{a:.1f}": round(fusion[a] / n, 4) for a in alphas},
        "n_unknown": n_unknown,
    }
    with open(out_dir / f"summary_{safe}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/summary_{safe}.json")


if __name__ == "__main__":
    main()
