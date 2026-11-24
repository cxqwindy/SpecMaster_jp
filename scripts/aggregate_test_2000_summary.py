#!/usr/bin/env python3
"""Aggregate closed-source LLM results on nmr_matching_test_2000.json.

Reads each model's `preds_{safe_model}.jsonl` in --results_dir, computes
overall accuracy, Yes/No recall, per-negative-type accuracy, and old-1000 vs
new-1000 subset accuracies. Writes comparison.md and prints tables.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MODELS = [
    ("Gemini-3.1-Pro", "openai/specxmaster/gemini-3.1-pro-preview"),
    ("GPT-5.5",         "openai/specmaster/gpt-5.5"),
    ("GPT-5.2",         "openai/gpt-5.2"),
    ("Claude-Opus-4-8", "openai/specxmaster/global.anthropic.claude-opus-4-8"),
]


def load_preds(path):
    """Return {i: pred} keeping only non-Unknown preds; latest row wins on duplicates."""
    d = {}
    if not path.exists():
        return d
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["pred"] == "Unknown":
                continue
            d[r["i"]] = r["pred"]
    return d


def compute(data, preds):
    n = len(data)
    tp = fp = tn = fn = unk = 0
    by_type_total = Counter()
    by_type_correct = Counter()
    old_tot = old_correct = 0
    new_tot = new_correct = 0
    for i, rec in enumerate(data):
        gt = rec["output"]
        pred = preds.get(i, "Unknown")
        nt = rec["_negative_type"]
        by_type_total[nt] += 1
        if pred == "Unknown":
            unk += 1
            continue
        correct = (pred == gt)
        if correct:
            by_type_correct[nt] += 1
        if i < 1000:
            old_tot += 1
            old_correct += int(correct)
        else:
            new_tot += 1
            new_correct += int(correct)
        if gt == "Yes" and pred == "Yes": tp += 1
        elif gt == "No"  and pred == "Yes": fp += 1
        elif gt == "No"  and pred == "No":  tn += 1
        elif gt == "Yes" and pred == "No":  fn += 1
    acc = (tp + tn) / n if n else 0
    yes_r = tp / (tp + fn) if (tp + fn) else 0
    no_r  = tn / (tn + fp) if (tn + fp) else 0
    return {
        "n": n, "unknown": unk, "acc": acc,
        "yes_r": yes_r, "no_r": no_r,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "old_acc": old_correct / old_tot if old_tot else 0,
        "new_acc": new_correct / new_tot if new_tot else 0,
        "old_n": old_tot, "new_n": new_tot,
        "by_type": {
            k: (by_type_correct[k], by_type_total[k],
                by_type_correct[k] / by_type_total[k] if by_type_total[k] else 0)
            for k in sorted(by_type_total)
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file",
                    default="/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json")
    ap.add_argument("--results_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/eval_results_testset_api_2000/")
    ap.add_argument("--out", default=None,
                    help="Path to write comparison.md (default: <results_dir>/comparison.md)")
    args = ap.parse_args()

    data = json.load(open(args.test_file))
    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "comparison.md"

    results = {}
    for label, model_id in MODELS:
        safe = model_id.replace("/", "__")
        preds_path = results_dir / f"preds_{safe}.jsonl"
        preds = load_preds(preds_path)
        results[label] = {
            "model_id": model_id,
            "preds_path": str(preds_path),
            "cached_rows": len(preds),
            **compute(data, preds),
        }

    lines = []
    lines.append("# Closed-source LLM comparison on nmr_matching_test_2000\n")
    lines.append(f"Test set: `{args.test_file}` ({len(data)} rows)\n")
    lines.append("## Overall\n")
    lines.append("| Model | N | Cached | Unknown | Accuracy | Yes-recall | No-recall | Old-1000 acc | New-1000 acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in results:
        r = results[label]
        lines.append(
            f"| {label} | {r['n']} | {r['cached_rows']} | {r['unknown']} | "
            f"{r['acc']*100:.1f}% | {r['yes_r']*100:.1f}% | {r['no_r']*100:.1f}% | "
            f"{r['old_acc']*100:.1f}% ({r['old_n']}) | {r['new_acc']*100:.1f}% ({r['new_n']}) |"
        )
    lines.append("")

    lines.append("## Confusion matrix\n")
    lines.append("| Model | TP | FP | TN | FN |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in results:
        r = results[label]
        lines.append(f"| {label} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |")
    lines.append("")

    all_types = sorted({t for r in results.values() for t in r["by_type"]})
    lines.append("## Per negative-type accuracy\n")
    header = "| _negative_type | Total | " + " | ".join(m for m, _ in MODELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(MODELS)))
    for t in all_types:
        total = next(iter(results.values()))["by_type"].get(t, (0, 0, 0))[1]
        row = f"| `{t}` | {total} |"
        for label, _ in MODELS:
            c, n, a = results[label]["by_type"].get(t, (0, 0, 0))
            row += f" {a*100:.1f}% ({c}/{n}) |"
        lines.append(row)
    lines.append("")

    lines.append("## Sanity checks\n")
    for label in results:
        r = results[label]
        drift = (r["new_acc"] - r["old_acc"]) * 100
        flag = " ⚠" if abs(drift) > 3 and r["new_n"] > 0 and r["old_n"] > 0 else ""
        unk_pct = 100 * r["unknown"] / r["n"] if r["n"] else 0
        unk_flag = " ⚠" if unk_pct > 1 else ""
        lines.append(f"- **{label}**: new−old = {drift:+.1f}pt{flag} · unknown = {r['unknown']}/{r['n']} ({unk_pct:.2f}%){unk_flag}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
