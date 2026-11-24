#!/usr/bin/env python3
"""Evaluate fine-tuned NMR matching model on test set.

Usage:
    python evaluate_nmr_matching.py \
        --model_path /fs_mol/cxq/SpecMaster_jp/models/Qwen3.5-35B-A3B \
        --adapter_path saves/qwen3_5-nmr-matching/lora-exp1 \
        --test_file data/nmr_matching_test.json \
        --output_dir /fs_mol/cxq/SpecMaster_jp/eval_results/
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from llamafactory.chat import ChatModel


def parse_answer(text: str) -> str:
    """Extract Yes/No from model output."""
    text = text.strip()

    # Try to parse as JSON
    try:
        obj = json.loads(text)
        ans = obj.get("final_answer", "")
        if ans in ("Yes", "No"):
            return ans
    except json.JSONDecodeError:
        pass

    # Try regex for JSON fragment
    m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', text)
    if m:
        return m.group(1)

    # Direct Yes/No
    if text.startswith("Yes"):
        return "Yes"
    if text.startswith("No"):
        return "No"

    # Search in text
    if "Yes" in text and "No" not in text:
        return "Yes"
    if "No" in text and "Yes" not in text:
        return "No"

    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Path to LoRA adapter. Empty string = base model only.")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--template", type=str, default="qwen3_5_textonly")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--quantization_bit", type=int, default=0,
                        help="Quantization bits (4 for QLoRA, 0 for none)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    with open(args.test_file) as f:
        test_data = json.load(f)
    print(f"Test set: {len(test_data)} records")

    # Load model
    print("Loading model...")
    model_kwargs = dict(
        model_name_or_path=args.model_path,
        template=args.template,
    )
    if args.adapter_path:
        model_kwargs["finetuning_type"] = "lora"
        model_kwargs["adapter_name_or_path"] = args.adapter_path
    else:
        model_kwargs["finetuning_type"] = "full"
    if args.quantization_bit > 0:
        model_kwargs["quantization_bit"] = args.quantization_bit
        model_kwargs["quantization_method"] = "bitsandbytes"
    chat_model = ChatModel(model_kwargs)
    print("Model loaded.")

    # Run inference
    results = []
    for i, rec in enumerate(test_data):
        messages = [{"role": "user", "content": rec["instruction"] + "\n\n" + rec["input"]}]

        response = ""
        for chunk in chat_model.stream_chat(messages):
            response += chunk

        pred_answer = parse_answer(response)
        gt_answer = rec["output"]  # "Yes" or "No"
        label = rec.get("_label", 1 if gt_answer == "Yes" else 0)
        neg_type = rec.get("_negative_type", "positive")

        results.append({
            "index": i,
            "candidate_smiles": rec.get("_candidate_smiles", ""),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": pred_answer == gt_answer,
            "negative_type": neg_type,
            "raw_output": response[:500],
        })

        if (i + 1) % 20 == 0:
            n_correct = sum(r["correct"] for r in results)
            print(f"[{i+1}/{len(test_data)}] Accuracy so far: "
                  f"{n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")

    # Compute metrics
    n_total = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / n_total if n_total else 0

    # Per-class metrics
    tp = sum(1 for r in results if r["gt_answer"] == "Yes" and r["pred_answer"] == "Yes")
    fp = sum(1 for r in results if r["gt_answer"] == "No" and r["pred_answer"] == "Yes")
    tn = sum(1 for r in results if r["gt_answer"] == "No" and r["pred_answer"] == "No")
    fn = sum(1 for r in results if r["gt_answer"] == "Yes" and r["pred_answer"] == "No")

    precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes) if (precision_yes + recall_yes) > 0 else 0

    precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_no = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_no = 2 * precision_no * recall_no / (precision_no + recall_no) if (precision_no + recall_no) > 0 else 0

    # Per-type accuracy
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        nt = r["negative_type"]
        by_type[nt]["total"] += 1
        if r["correct"]:
            by_type[nt]["correct"] += 1

    # Summary
    metrics = {
        "total": n_total,
        "accuracy": round(accuracy, 4),
        "yes_precision": round(precision_yes, 4),
        "yes_recall": round(recall_yes, 4),
        "yes_f1": round(f1_yes, 4),
        "no_precision": round(precision_no, 4),
        "no_recall": round(recall_no, 4),
        "no_f1": round(f1_no, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_type": {
            k: {"total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0}
            for k, v in sorted(by_type.items())
        },
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {n_correct}/{n_total} ({accuracy*100:.1f}%)")
    print(f"Yes - P: {precision_yes:.3f}  R: {recall_yes:.3f}  F1: {f1_yes:.3f}")
    print(f"No  - P: {precision_no:.3f}  R: {recall_no:.3f}  F1: {f1_no:.3f}")
    print(f"\nPer-type accuracy:")
    for k, v in sorted(by_type.items()):
        acc = v["correct"] / v["total"] if v["total"] > 0 else 0
        print(f"  {k}: {v['correct']}/{v['total']} ({acc*100:.1f}%)")

    # Save results
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    predictions_path = output_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
