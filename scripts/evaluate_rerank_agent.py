#!/usr/bin/env python3
"""Rerank FairNMR-500 via the multi-turn tool agent (M3).

Modes:
  --stage extract  : dump unique top-K candidate SMILES to a JSONL that
                     precompute_matching_cache.py can consume, then exit.
  --stage rollout  : score (rec_i, cand_pos) pairs using the multi-turn
                     agent (HF transformers + NmrMatchingRuntime). Supports
                     --shard i/N for fan-out across GPUs. Appends jsonl.
  --stage aggregate: read all shard jsonl files under --output_dir and
                     compute the same rerank metrics as
                     evaluate_rerank_fairnmr500.py.

Score signal:  from parsed <answer>{"final_answer": Yes|No, "confidence": f}</answer>.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# --- import the tool agent scaffold from Open-AgentRL ------------------------
AGENT_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
sys.path.insert(0, str(AGENT_ROOT))

_infer_path = AGENT_ROOT / "recipe" / "nmr_matching_agent" / "standalone_infer_matching.py"
_spec = importlib.util.spec_from_file_location("nmr_matching_infer", _infer_path)
_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_infer)  # exposes TOOL_SCHEMAS, parse_action, build_system_prompt, generate_once, load_tools

from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime  # noqa: E402

FAIRNMR_DATA = (
    "/fs_mol/geyutang/proteinlm_dataset/Open-AgentRL/results/nmr/"
    "standalone_fairnmr500_cached/fairnmr500_with_formula_generate_rerank_reference.jsonl"
)
DEFAULT_CACHE = str(
    AGENT_ROOT / "recipe" / "nmr_matching_agent" / "prediction_cache.json"
)


# ---------------------------------------------------------------------------
def canon(smi: str) -> Optional[str]:
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def load_fair_records() -> List[Dict[str, Any]]:
    records = []
    with open(FAIRNMR_DATA) as f:
        for line in f:
            od = json.loads(line)["original_data"]
            records.append(od)
    return records


def build_case(od: Dict[str, Any], cand_smi: str) -> Dict[str, Any]:
    """Build the user JSON case that NmrMatchingRuntime expects (candidate + peaks)."""
    return {
        "idx": od.get("idx"),
        "candidate_smiles": cand_smi,
        "c_nmr_peaks": od.get("c_nmr_peaks") or [],
        "h_nmr_peaks": od.get("h_nmr_peaks") or [],
        "nmr_solvent": od.get("nmr_solvent"),
    }


# ---------------------------------------------------------------------------
# stage: extract
# ---------------------------------------------------------------------------
def stage_extract(args) -> None:
    records = load_fair_records()
    K = args.topk
    src = args.candidate_source
    unique = {}
    for od in records:
        cands = od.get(src, [])[:K]
        for c in cands:
            smi = c.get("smiles_std") or c.get("smiles")
            if not smi:
                continue
            unique.setdefault(smi, True)
    print(f"[extract] {len(records)} records, top-{K} from {src}, {len(unique)} unique candidates")
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for smi in unique:
            f.write(json.dumps({"candidate_smiles": smi}) + "\n")
    print(f"[extract] wrote {out}")
    print()
    print("Next: precompute NMR predictions for these candidates:")
    print(
        f"  cd {AGENT_ROOT}\n"
        f"  python recipe/nmr_matching_agent/precompute_matching_cache.py \\\n"
        f"    --jsonl {out} \\\n"
        f"    --cache {DEFAULT_CACHE} \\\n"
        f"    --out {args.cache_out or DEFAULT_CACHE}"
    )


# ---------------------------------------------------------------------------
# stage: rollout
# ---------------------------------------------------------------------------
def run_agent_sync(
    tok,
    model,
    system_prompt: str,
    case: Dict[str, Any],
    tools_list: List[Dict[str, Any]],
    runtime: NmrMatchingRuntime,
    max_turns: int = 6,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Multi-turn rollout: model -> parse -> runtime -> extend messages. All sync."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
    ]
    tool_sequence: List[str] = []
    final_answer = ""
    fail_reason: Optional[str] = None

    for _ in range(max_turns):
        # Render prompt via HF chat_template WITH tools= (matches training).
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools_list
        )
        # Use the same generate_once helper but bypass its chat_template call by
        # inlining. Simpler: replicate generate here to avoid re-rendering.
        assistant_text = _infer.generate_once(tok, model, messages, max_new_tokens=max_new_tokens)
        action = _infer.parse_action(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        if action["type"] == "answer":
            final_answer = action.get("answer", "")
            tool_sequence.append("answer")
            break
        if action["type"] != "tool":
            fail_reason = action["type"]
            tool_sequence.append(action["type"])
            break
        name = action.get("name")
        args_json = action.get("arguments") or {}
        tool_sequence.append(name or "unknown_tool")
        handler = runtime.handlers.get(name) if name else None
        if handler is None:
            obs_json = json.dumps({"error": f"unknown tool: {name}"})
        else:
            try:
                obs, _extra = handler(args_json)
                obs_json = json.dumps(obs, ensure_ascii=False)
            except Exception as e:
                obs_json = json.dumps({"error": f"{type(e).__name__}: {e}"})
        messages.append({"role": "tool", "name": name, "content": obs_json})

    return {
        "final_answer": final_answer,
        "tool_sequence": tool_sequence,
        "fail_reason": fail_reason,
    }


def parse_yes_conf(answer_text: str) -> Tuple[str, float]:
    if not answer_text:
        return "Unknown", 0.5
    try:
        obj = json.loads(answer_text)
    except Exception:
        a, b = answer_text.find("{"), answer_text.rfind("}")
        obj = None
        if a >= 0 and b > a:
            try:
                obj = json.loads(answer_text[a:b + 1])
            except Exception:
                obj = None
    ans, conf = "Unknown", 0.5
    if isinstance(obj, dict):
        a = str(obj.get("final_answer", "")).strip()
        if a in ("Yes", "No"):
            ans = a
        c = obj.get("confidence")
        if isinstance(c, (int, float)):
            conf = float(c)
    return ans, conf


def stage_rollout(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Build all jobs first, then shard.
    records = load_fair_records()
    K = args.topk
    src = args.candidate_source
    jobs = []  # (rec_i, cand_pos, cand_smi)
    for ri, od in enumerate(records):
        for pos, c in enumerate(od.get(src, [])[:K]):
            smi = c.get("smiles_std") or c.get("smiles")
            if smi:
                jobs.append((ri, pos, smi))

    shard_i, shard_n = args.shard
    shard_jobs = [j for k, j in enumerate(jobs) if k % shard_n == shard_i]
    print(f"[rollout] total jobs={len(jobs)}  shard {shard_i}/{shard_n} -> {len(shard_jobs)}")

    # Load model + tokenizer once.
    print(f"[rollout] loading model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Load prediction cache once.
    print(f"[rollout] loading prediction cache from {args.prediction_cache}")
    with open(args.prediction_cache) as f:
        pred_cache = json.load(f)
    print(f"[rollout] cache size={len(pred_cache)}")

    tools_list = list(_infer.TOOL_SCHEMAS.values())
    system_prompt = _infer.build_system_prompt()

    out_path = Path(args.output_dir) / f"shard_{shard_i:02d}_of_{shard_n:02d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip jobs already recorded.
    done_keys = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r["rec_i"], r["pos"]))
                except Exception:
                    continue
        print(f"[rollout] resuming — {len(done_keys)} already recorded in {out_path}")

    t0 = time.time()
    n_done = 0
    with out_path.open("a") as fout:
        for k, (ri, pos, smi) in enumerate(shard_jobs):
            if (ri, pos) in done_keys:
                continue
            case = build_case(records[ri], smi)
            runtime = NmrMatchingRuntime(
                record=case,
                prediction_cache=pred_cache,
                allow_remote_predict=args.allow_remote_predict,
            )
            res = run_agent_sync(tok, model, system_prompt, case, tools_list, runtime,
                                 max_turns=args.max_turns, max_new_tokens=args.max_new_tokens)
            ans, conf = parse_yes_conf(res["final_answer"])
            row = {
                "rec_i": ri, "pos": pos, "smiles": smi,
                "ans": ans, "conf": conf,
                "tool_sequence": res["tool_sequence"],
                "fail_reason": res["fail_reason"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(shard_jobs) - k - 1) / max(rate, 1e-6)
                print(f"[rollout] shard {shard_i}: {n_done}/{len(shard_jobs)}  "
                      f"({rate:.2f}/s, eta {eta/60:.1f} min)  last=(rec_i={ri}, pos={pos}, ans={ans})")

    print(f"[rollout] shard {shard_i} done: {n_done} new rollouts, total time {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# stage: aggregate + rerank metrics (copied from evaluate_rerank_fairnmr500.py)
# ---------------------------------------------------------------------------
def stage_aggregate(args) -> None:
    records = load_fair_records()
    src = args.candidate_source
    K = args.topk
    n_records = len(records)

    # Collect all shard rows into detail[rec_i].
    detail: Dict[int, List[Dict[str, Any]]] = {}
    total = 0
    for p in sorted(Path(args.output_dir).glob("shard_*.jsonl")):
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                detail.setdefault(r["rec_i"], []).append(r)
                total += 1
    print(f"[aggregate] loaded {total} rollout rows from {args.output_dir}")
    covered = sum(1 for ri in range(n_records) if ri in detail)
    print(f"[aggregate] covered records: {covered}/{n_records}")

    # Attach cost + is_gt to each cand.
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od.get(src, [])[:K]
        pos_to_c = {i: c for i, c in enumerate(cands)}
        for d in detail.get(ri, []):
            c = pos_to_c.get(d["pos"])
            if c is None:
                d["cost"] = float("inf"); d["is_gt"] = False; continue
            ts = c.get("total_score")
            try: ts = float(ts)
            except (TypeError, ValueError): ts = float("inf")
            d["cost"] = ts
            d["is_gt"] = (canon(d["smiles"]) == gt)

    def model_signed(d):
        return d["conf"] if d["ans"] == "Yes" else -d["conf"]

    baseline_correct = 0
    opt1_correct = 0
    opt2_correct = 0
    for ri, od in enumerate(records):
        gt = canon(od["gt_smiles_std"])
        cands = od.get(src, [])[:K]
        base = canon(cands[0].get("smiles_std") or cands[0].get("smiles")) if cands else None
        if base is not None and base == gt:
            baseline_correct += 1
        cs = detail.get(ri, [])
        if cs:
            best = max(cs, key=model_signed)
            if best["is_gt"]:
                opt1_correct += 1
            cs_sorted = sorted(cs, key=lambda d: d["pos"])
            pick = next((d for d in cs_sorted if d["ans"] == "Yes"), None)
            opt2_pick = canon(pick["smiles"]) if pick else base
            if opt2_pick is not None and opt2_pick == gt:
                opt2_correct += 1

    def cost_rank_score(cs):
        m = len(cs)
        return {d["pos"]: (1.0 - d["pos"] / m) for d in cs}
    def model_norm(cs):
        return {d["pos"]: (model_signed(d) + 1.0) / 2.0 for d in cs}

    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    fusion_acc = {}
    for a in alphas:
        c = 0
        for ri, od in enumerate(records):
            cs = detail.get(ri, [])
            if not cs: continue
            crs = cost_rank_score(cs); mns = model_norm(cs)
            best = max(cs, key=lambda d: a * crs[d["pos"]] + (1 - a) * mns[d["pos"]])
            if best["is_gt"]: c += 1
        fusion_acc[a] = c

    topk_acc = {}
    for k in [2, 3, 5]:
        c = 0
        for ri, od in enumerate(records):
            cs = sorted(detail.get(ri, []), key=lambda d: d["pos"])
            if not cs: continue
            head = cs[:k]
            best = max(head, key=model_signed)
            if best["is_gt"]: c += 1
        topk_acc[k] = c

    ceiling = sum(1 for ri in range(n_records) if any(d["is_gt"] for d in detail.get(ri, [])))

    print("\n" + "=" * 62)
    print(f"RERANK EVAL — FairNMR-500  (agent multi-turn, top-{K} from {src})")
    print("=" * 62)
    print(f"Ceiling (gt in top-{K}):           {ceiling}/{n_records} ({100*ceiling/n_records:.1f}%)")
    print(f"Baseline (cost rerank top-1):      {baseline_correct}/{n_records} ({100*baseline_correct/n_records:.1f}%)")
    print(f"Option 1 (agent conf argmax):      {opt1_correct}/{n_records} ({100*opt1_correct/n_records:.1f}%)")
    print(f"Option 2 (cost order + first Yes): {opt2_correct}/{n_records} ({100*opt2_correct/n_records:.1f}%)")
    print(f"\nDirection B: rank fusion (alpha=1 -> pure cost baseline):")
    for a in alphas:
        mark = "  <-- baseline" if a == 1.0 else ""
        print(f"  alpha={a:.1f}: {fusion_acc[a]}/{n_records} ({100*fusion_acc[a]/n_records:.1f}%){mark}")
    print(f"\nDirection B: model reorders cost top-K only:")
    for k in [2, 3, 5]:
        print(f"  top-{k}: {topk_acc[k]}/{n_records} ({100*topk_acc[k]/n_records:.1f}%)")

    summary = {
        "n": n_records, "topk": K, "candidate_source": src,
        "ceiling": ceiling,
        "baseline_cost_rerank_acc": round(baseline_correct / n_records, 4),
        "option1_agent_confidence_acc": round(opt1_correct / n_records, 4),
        "option2_combined_first_yes_acc": round(opt2_correct / n_records, 4),
        "fusion_alpha_acc": {f"{a:.1f}": round(fusion_acc[a] / n_records, 4) for a in alphas},
        "topk_reorder_acc": {f"top{k}": round(topk_acc[k] / n_records, 4) for k in [2, 3, 5]},
    }
    with open(Path(args.output_dir) / "rerank_agent_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {Path(args.output_dir) / 'rerank_agent_summary.json'}")


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "rollout", "aggregate"])

    # dataset / candidates
    ap.add_argument("--topk", type=int, default=5, help="rerank top-K per record")
    ap.add_argument("--candidate_source", default="rerank_topk",
                    choices=["rerank_topk", "pred_topk"])

    # extract-stage output
    ap.add_argument("--out_jsonl", default="/tmp/fairnmr_topk_cands.jsonl")
    ap.add_argument("--cache_out", default="")

    # rollout stage
    ap.add_argument("--model_path")
    ap.add_argument("--prediction_cache", default=DEFAULT_CACHE)
    ap.add_argument("--output_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/eval_results_rerank_agent")
    ap.add_argument("--shard", nargs=2, type=int, default=[0, 1], metavar=("I", "N"),
                    help="this worker handles jobs where idx %% N == I")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--allow_remote_predict", action="store_true",
                    help="fall back to remote NMR predictor for cache misses (slow)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.stage == "extract":
        stage_extract(args)
    elif args.stage == "rollout":
        if not args.model_path:
            raise SystemExit("--model_path required for rollout stage")
        stage_rollout(args)
    elif args.stage == "aggregate":
        stage_aggregate(args)


if __name__ == "__main__":
    main()
