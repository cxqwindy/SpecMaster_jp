#!/usr/bin/env python3
"""Evaluate our NMR matching (Yes/No) model as a reranker on FairNMR-500.

The dataset gives, per NMR spectrum, a list of candidate SMILES already
reranked by a cost function (rerank_topk). Our binary matcher scores each
candidate; we then compare two ways of turning it into a top-1 pick against
the existing cost-rerank baseline (strict_acc = 0.676):

  Option 1 (standalone rerank):
      Run the matcher on every rerank_topk candidate, sort by
      P(Yes) * confidence descending, take top-1.
      -> "can our model rerank better than the cost function alone?"

  Option 2 (combined / filter):
      Walk candidates in the existing cost-rerank order, take the FIRST
      one the model judges "Yes".  Falls back to rank-1 if none is Yes.
      -> "can our model, layered on top of cost rerank, push past 0.676?"

Hit is judged by RDKit canonical SMILES == canonical(gt_smiles_std),
matching the official scoring script.

Inference is done with vLLM for speed (~19k candidate scorings).

Usage:
    python evaluate_rerank_fairnmr500.py \
        --model_path <full-ft checkpoint dir> \
        --template qwen3
"""

import argparse
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA = "/fs_mol/geyutang/proteinlm_dataset/Open-AgentRL/results/nmr/standalone_fairnmr500_cached/fairnmr500_with_formula_generate_rerank_reference.jsonl"

INSTRUCTION = (
    "Given NMR spectral data and a candidate molecular structure (SMILES), "
    "determine whether the candidate matches the NMR spectrum. "
    "Provide your reasoning as a structured JSON object."
)


def canon(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def format_input(candidate, c_peaks, h_peaks, solvent=""):
    """Build the input field — mirrors prepare_cot_sft_data.format_input."""
    c_vals = []
    for p in c_peaks or []:
        v = p.get("delta (ppm)")
        if v is not None:
            c_vals.append(f"{float(v):.1f}")
    c_str = ", ".join(c_vals) if c_vals else "N/A"

    h_parts = []
    for p in h_peaks or []:
        delta = p.get("centroid") or p.get("delta")
        if delta is None:
            continue
        cat = p.get("category", "")
        nH = p.get("nH", 1)
        j_vals = p.get("j_values", "")
        if j_vals and j_vals != "_":
            j_str = j_vals.replace("_", ", ").strip(", ")
            h_parts.append(f"{float(delta):.2f} ({cat}, J={j_str}, {nH}H)")
        else:
            h_parts.append(f"{float(delta):.2f} ({cat}, {nH}H)")
    h_str = ", ".join(h_parts) if h_parts else "N/A"

    header = f" (, {solvent})" if solvent else ""
    return (
        f"Candidate SMILES: {candidate}\n\n"
        f"13C NMR{header}: {c_str}\n"
        f"1H NMR{header}: {h_str}"
    )


def parse_yes_conf(text):
    """Extract (answer, confidence) from model output JSON."""
    text = text.strip()
    ans, conf = "Unknown", 0.5
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # grab first {...}
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            try:
                obj = json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                obj = None
    if isinstance(obj, dict):
        a = str(obj.get("final_answer", "")).strip()
        if a in ("Yes", "No"):
            ans = a
        c = obj.get("confidence")
        if isinstance(c, (int, float)):
            conf = float(c)
    else:
        m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', text)
        if m:
            ans = m.group(1)
        m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if m:
            try:
                conf = float(m.group(1))
            except ValueError:
                pass
    return ans, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path",
                    default="/fs_mol/cxq/SpecMaster_jp/models/Qwen3-32B",
                    help="Load tokenizer from here (base model) to avoid the "
                         "checkpoint's newer-transformers tokenizer format that "
                         "breaks under the vLLM env's transformers version.")
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--output_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/eval_results_rerank_fairnmr500/")
    ap.add_argument("--candidate_source", default="rerank_topk",
                    choices=["rerank_topk", "pred_topk"],
                    help="Which candidate list to score.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    records = []
    with open(DATA) as f:
        for line in f:
            od = json.loads(line)["original_data"]
            records.append(od)
    print(f"Loaded {len(records)} records")

    # --- Build all (record_idx, cand_pos, prompt) scoring jobs ---
    # We score the candidates in each record's chosen source list, in order.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    jobs = []          # (rec_i, cand_pos, cand_smiles)
    prompts = []
    for ri, od in enumerate(records):
        cands = od[args.candidate_source]
        c_peaks = od.get("c_nmr_peaks")
        h_peaks = od.get("h_nmr_peaks")
        solvent = od.get("nmr_solvent", "") or ""
        for pos, c in enumerate(cands):
            smi = c.get("smiles_std") or c.get("smiles")
            if not smi:
                continue
            user = INSTRUCTION + "\n\n" + format_input(smi, c_peaks, h_peaks, solvent)
            msg = [{"role": "user", "content": user}]
            prompt = tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            jobs.append((ri, pos, smi))
            prompts.append(prompt)

    print(f"Total candidate scorings: {len(prompts)}")

    # --- vLLM batch inference ---
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sp)

    # --- Collect scores per record ---
    # detail[rec_i] = list of dicts {pos, smiles, ans, conf, cost, is_gt}
    detail = {}
    for (ri, pos, smi), out in zip(jobs, outputs):
        text = out.outputs[0].text
        ans, conf = parse_yes_conf(text)
        detail.setdefault(ri, []).append({"pos": pos, "smiles": smi,
                                          "ans": ans, "conf": conf})

    n = len(records)

    # Attach cost (total_score from source list) + gt flag to each candidate.
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od[args.candidate_source]
        for d in detail.get(ri, []):
            c = cands[d["pos"]]
            ts = c.get("total_score")
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                ts = float("inf")
            d["cost"] = ts
            d["is_gt"] = (canon(d["smiles"]) == gt)

    def model_signed(d):
        """Model score: +conf if Yes, -conf if No. Higher = more likely match."""
        return d["conf"] if d["ans"] == "Yes" else -d["conf"]

    def pick_hit(ri, key):
        """Pick argmax(key) among a record's candidates; return 1 if it's gt."""
        cs = detail.get(ri, [])
        if not cs:
            return 0
        best = max(cs, key=key)
        return 1 if best["is_gt"] else 0

    # ---- Baseline + standalone options (recomputed) ----
    baseline_correct = 0
    opt1_correct = 0
    opt2_correct = 0
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od[args.candidate_source]
        base_pick = canon(cands[0].get("smiles_std") or cands[0].get("smiles")) if cands else None
        if base_pick is not None and base_pick == gt:
            baseline_correct += 1
        # opt1: pure model confidence
        opt1_correct += pick_hit(ri, model_signed)
        # opt2: first Yes in cost order, else fall back to rank1
        cs = sorted(detail.get(ri, []), key=lambda d: d["pos"])
        pick = next((d for d in cs if d["ans"] == "Yes"), None)
        opt2_pick = canon(pick["smiles"]) if pick else base_pick
        if opt2_pick is not None and opt2_pick == gt:
            opt2_correct += 1

    # ---- Fusion sweep (Direction B) ----
    # Rank-based fusion: convert cost order -> normalized rank score in [0,1]
    # (rank1 -> 1.0, last -> ~0), fuse with model_signed normalized to [0,1].
    # final = alpha * cost_rank_score + (1-alpha) * model_score
    # alpha=1.0 recovers pure cost rerank (baseline). alpha=0.0 = pure model.
    def cost_rank_score(cs):
        """Map each candidate to a [0,1] score by its position in cost order
        (position already reflects cost sorting). rank1 -> 1.0."""
        m = len(cs)
        return {d["pos"]: (1.0 - d["pos"] / m) for d in cs}

    def model_norm(cs):
        """Map model_signed (-1..1) to (0..1)."""
        return {d["pos"]: (model_signed(d) + 1.0) / 2.0 for d in cs}

    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    fusion_acc = {}
    for alpha in alphas:
        correct = 0
        for ri, od in enumerate(records):
            cs = detail.get(ri, [])
            if not cs:
                continue
            crs = cost_rank_score(cs)
            mns = model_norm(cs)
            best = max(cs, key=lambda d: alpha * crs[d["pos"]] + (1 - alpha) * mns[d["pos"]])
            if best["is_gt"]:
                correct += 1
        fusion_acc[alpha] = correct

    # ---- Top-K reorder fusion: only let the model reorder cost top-K ----
    topk_acc = {}
    for K in [2, 3, 5]:
        correct = 0
        for ri, od in enumerate(records):
            cs = sorted(detail.get(ri, []), key=lambda d: d["pos"])
            if not cs:
                continue
            head = cs[:K]
            best = max(head, key=model_signed)  # model reorders only the top-K
            if best["is_gt"]:
                correct += 1
        topk_acc[K] = correct

    ceiling = sum(1 for ri, od in enumerate(records)
                  if any(d["is_gt"] for d in detail.get(ri, [])))

    summary = {
        "model": args.model_path,
        "candidate_source": args.candidate_source,
        "n": n,
        "ceiling_gt_in_candidates": ceiling,
        "ceiling_acc": round(ceiling / n, 4),
        "baseline_cost_rerank_acc": round(baseline_correct / n, 4),
        "option1_standalone_confidence_acc": round(opt1_correct / n, 4),
        "option2_combined_first_yes_acc": round(opt2_correct / n, 4),
        "fusion_alpha_acc": {f"{a:.1f}": round(c / n, 4) for a, c in fusion_acc.items()},
        "topk_reorder_acc": {f"top{k}": round(c / n, 4) for k, c in topk_acc.items()},
    }

    print("\n" + "=" * 60)
    print("RERANK EVALUATION — FairNMR-500")
    print("=" * 60)
    print(f"Ceiling (gt in candidates):        {ceiling}/{n} ({100*ceiling/n:.1f}%)")
    print(f"Baseline (cost rerank top-1):      {baseline_correct}/{n} ({100*baseline_correct/n:.1f}%)")
    print(f"Option 1 (our model, confidence):  {opt1_correct}/{n} ({100*opt1_correct/n:.1f}%)")
    print(f"Option 2 (cost rerank + our Yes):  {opt2_correct}/{n} ({100*opt2_correct/n:.1f}%)")
    print("\n--- Direction B: cost+confidence fusion (alpha=1 -> pure cost) ---")
    for a in alphas:
        c = fusion_acc[a]
        mark = "  <-- baseline" if a == 1.0 else ""
        print(f"  alpha={a:.1f}: {c}/{n} ({100*c/n:.1f}%){mark}")
    print("\n--- Direction B: model reorders cost top-K only ---")
    for k in [2, 3, 5]:
        c = topk_acc[k]
        print(f"  top-{k}: {c}/{n} ({100*c/n:.1f}%)")

    with open(out_dir / "rerank_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "rerank_detail.jsonl", "w") as f:
        for ri, od in enumerate(records):
            f.write(json.dumps({
                "idx": od.get("idx", ri),
                "gt": canon(od["gt_smiles_std"]),
                "candidates": detail.get(ri, []),
            }, ensure_ascii=False) + "\n")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
