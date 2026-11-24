#!/usr/bin/env python3
"""Baseline evaluation of multiple LLMs on NMR matching task.

Tests Qwen3.5-35B (base, no SFT), Gemini-2.5-Pro, Gemini-3-Pro,
GPT-5.5, Claude-4.6-Opus on the balanced test set.

Usage:
    python evaluate_baselines.py \
        --test_file data/nmr_matching_test_balanced.json \
        --output_dir /fs_mol/cxq/SpecMaster_jp/eval_baselines/
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import litellm
from litellm import completion

litellm.drop_params = True

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key


API_BASE = "https://ai-gateway-global.dp.tech"

MODELS = {
    "gemini-2.5-pro": "openai/gemini-2.5-pro",
    "gemini-2.5-flash": "openai/gemini-2.5-flash",
    "gemini-3-flash": "openai/gemini-3-flash",
    "gemini-3.1-pro": "openai/specxmaster/gemini-3.1-pro-preview",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-5.5": "openai/specmaster/gpt-5.5",
    "claude-sonnet-4.6": "openai/claude-sonnet-4-6",
    "claude-opus-4.6": "openai/cds/Claude-4.6-opus",
}

SYSTEM_PROMPT = """You are an NMR spectrum-structure matching expert. Given NMR spectral data and a candidate molecular structure (SMILES), determine whether the candidate matches the NMR spectrum.

You do NOT have the target molecular formula. Judge only from NMR data consistency.

Output ONLY a JSON object:

{
  "final_answer": "Yes" or "No",
  "confidence": <float 0.0-1.0>,
  "key_evidence": "<one sentence>"
}

Be concise. Only output the JSON above, nothing else."""


def parse_answer(text):
    """Extract Yes/No from model output."""
    text = text.strip()
    # Try JSON parse
    try:
        obj = json.loads(text)
        ans = obj.get("final_answer", "")
        if ans in ("Yes", "No"):
            return ans
    except json.JSONDecodeError:
        pass
    # Try regex
    m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', text)
    if m:
        return m.group(1)
    # Extract from code block
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            ans = obj.get("final_answer", "")
            if ans in ("Yes", "No"):
                return ans
        except:
            pass
    # Find first { ... }
    first = text.find('{')
    last = text.rfind('}')
    if first >= 0 and last > first:
        try:
            obj = json.loads(text[first:last+1])
            ans = obj.get("final_answer", "")
            if ans in ("Yes", "No"):
                return ans
        except:
            pass
    # Direct
    if text.startswith("Yes"):
        return "Yes"
    if text.startswith("No"):
        return "No"
    if "Yes" in text and "No" not in text:
        return "Yes"
    if "No" in text and "Yes" not in text:
        return "No"
    return "Unknown"


def call_model(model_id, instruction, input_text, max_retries=3):
    """Call a model via LiteLLM API."""
    api_key = require_gateway_api_key()
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model_id,
                api_key=api_key,
                api_base=API_BASE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction + "\n\n" + input_text},
                ],
                max_tokens=16384,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return ""


def compute_metrics(results):
    """Compute accuracy, P/R/F1, per-type accuracy."""
    n = len(results)
    n_correct = sum(r["correct"] for r in results)

    tp = sum(1 for r in results if r["gt"] == "Yes" and r["pred"] == "Yes")
    fp = sum(1 for r in results if r["gt"] == "No" and r["pred"] == "Yes")
    tn = sum(1 for r in results if r["gt"] == "No" and r["pred"] == "No")
    fn = sum(1 for r in results if r["gt"] == "Yes" and r["pred"] == "No")

    p_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_yes = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_yes = 2 * p_yes * r_yes / (p_yes + r_yes) if (p_yes + r_yes) > 0 else 0

    p_no = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_no = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_no = 2 * p_no * r_no / (p_no + r_no) if (p_no + r_no) > 0 else 0

    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        t = r["neg_type"]
        by_type[t]["total"] += 1
        if r["correct"]:
            by_type[t]["correct"] += 1

    return {
        "total": n,
        "accuracy": round(n_correct / n, 4) if n else 0,
        "yes_precision": round(p_yes, 4),
        "yes_recall": round(r_yes, 4),
        "yes_f1": round(f1_yes, 4),
        "no_precision": round(p_no, 4),
        "no_recall": round(r_no, 4),
        "no_f1": round(f1_no, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_type": {
            k: {**v, "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
            for k, v in sorted(by_type.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+",
                        default=list(MODELS.keys()),
                        help="Models to test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per model (for quick test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    with open(args.test_file) as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    n_yes = sum(1 for r in test_data if r["output"] == "Yes")
    n_no = sum(1 for r in test_data if r["output"] == "No")
    print(f"Test set: {len(test_data)} records (Yes={n_yes}, No={n_no})")

    all_metrics = {}

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        model_id = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_id})")
        print(f"{'='*60}")

        results = []
        pred_file = output_dir / f"{model_name}_predictions.jsonl"

        # Resume support
        done = 0
        if pred_file.exists():
            with open(pred_file) as f:
                done = sum(1 for _ in f)
            print(f"  Resuming from record {done}")

        with open(pred_file, "a", encoding="utf-8") as fout:
            for i, rec in enumerate(test_data):
                if i < done:
                    continue

                gt = rec["output"]
                neg_type = rec.get("_negative_type", "positive")

                raw = call_model(model_id, rec["instruction"], rec["input"])
                pred = parse_answer(raw)

                entry = {
                    "index": i,
                    "gt": gt,
                    "pred": pred,
                    "correct": pred == gt,
                    "neg_type": neg_type,
                    "raw": raw[:500],
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(test_data)}]")

                time.sleep(0.3)

        # Load all predictions
        results = []
        with open(pred_file) as f:
            for line in f:
                results.append(json.loads(line))

        metrics = compute_metrics(results)
        all_metrics[model_name] = metrics

        print(f"\n  Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"  Yes F1: {metrics['yes_f1']:.3f}  No F1: {metrics['no_f1']:.3f}")
        print(f"  Confusion: {metrics['confusion']}")

    # Save summary
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Acc':>6} {'Yes-F1':>7} {'No-F1':>7} {'Yes-R':>6} {'No-R':>6}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<20} {m['accuracy']*100:>5.1f}% {m['yes_f1']:>7.3f} {m['no_f1']:>7.3f} "
              f"{m['yes_recall']:>5.1f}% {m['no_recall']:>5.1f}%")


if __name__ == "__main__":
    main()
