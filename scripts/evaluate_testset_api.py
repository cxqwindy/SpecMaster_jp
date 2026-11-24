#!/usr/bin/env python3
"""Evaluate a closed-source LLM (teacher) on the 1000-sample NMR matching test
set, single-candidate Yes/No — SAME test set our SFT student (exp1) was scored
on. Produces per-type accuracy + confusion matrix so teacher vs student is
directly comparable.

Purpose: measure the distillation ceiling. If teacher >> student (72.9%),
re-distillation has room; if teacher ≈ student, SFT already transferred and we
should switch to RL.

Usage:
    python evaluate_testset_api.py --model openai/specxmaster/gemini-3.1-pro-preview
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
from litellm import completion

litellm.drop_params = True


import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key


TEST = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json"
API_BASE = "https://ai-gateway-global.dp.tech"

SYSTEM = (
    "You are an NMR spectrum-structure matching expert. Given NMR data and a "
    "candidate SMILES, determine whether they match. You do NOT have the target "
    "molecular formula. Output ONLY a JSON object: "
    '{"final_answer": "Yes" or "No", "confidence": <float 0.0-1.0>, '
    '"key_evidence": "<one sentence>"}'
)


def parse_answer(text):
    import re
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        a = str(obj.get("final_answer", "")).strip()
        if a in ("Yes", "No"):
            return a
    except json.JSONDecodeError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b + 1])
            a2 = str(obj.get("final_answer", "")).strip()
            if a2 in ("Yes", "No"):
                return a2
        except json.JSONDecodeError:
            pass
    m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', text)
    if m:
        return m.group(1)
    if text.startswith("Yes"):
        return "Yes"
    if text.startswith("No"):
        return "No"
    return "Unknown"


def score_one(model, user, max_retries=8, max_tokens=16384):
    api_key = require_gateway_api_key()
    import time
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model, api_key=api_key, api_base=API_BASE,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0, timeout=180,
            )
            m = resp.choices[0].message
            text = m.content or ""
            if not text and hasattr(m, "reasoning_content"):
                text = m.reasoning_content or ""
            ans = parse_answer(text)
            if ans == "Unknown" and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return ans
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                return "Unknown"
            msg = str(e).lower()
            if any(k in msg for k in ("rate", "429", "quota", "resource")):
                time.sleep(min(60, 5 * (2 ** attempt)))
            else:
                time.sleep(3 * (attempt + 1))
    return "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--test_file", default=TEST,
                    help="Alpaca JSON test set; defaults to the 1000-row set.")
    ap.add_argument("--output_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/eval_results_testset_api/")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Optional: only score the first N rows (for smoke tests).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "__")
    cache_path = out_dir / f"preds_{safe}.jsonl"

    data = json.load(open(args.test_file))
    if args.max_rows is not None:
        data = data[:args.max_rows]
    n = len(data)

    # Resume, retrying Unknown
    done = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                if r["pred"] == "Unknown":
                    continue
                done[r["i"]] = r["pred"]
    todo = [i for i in range(n) if i not in done]
    print(f"[{args.model}] n={n} cached={len(done)} todo={len(todo)}")

    fout = open(cache_path, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, args.model,
                          data[i]["instruction"] + "\n\n" + data[i]["input"]): i
                for i in todo}
        c = 0
        for fut in as_completed(futs):
            i = futs[fut]
            pred = fut.result()
            done[i] = pred
            fout.write(json.dumps({"i": i, "pred": pred}) + "\n")
            fout.flush()
            c += 1
            if c % 100 == 0:
                print(f"  {c}/{len(todo)}")
    fout.close()

    # Metrics (same shape as evaluate_nmr_matching.py)
    tp = fp = tn = fn = 0
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    n_unknown = 0
    for i, rec in enumerate(data):
        gt = rec["output"]
        pred = done.get(i, "Unknown")
        if pred == "Unknown":
            n_unknown += 1
        correct = (pred == gt)
        nt = rec["_negative_type"]
        by_type[nt]["total"] += 1
        if correct:
            by_type[nt]["correct"] += 1
        if gt == "Yes" and pred == "Yes":
            tp += 1
        elif gt == "No" and pred == "Yes":
            fp += 1
        elif gt == "No" and pred == "No":
            tn += 1
        elif gt == "Yes" and pred == "No":
            fn += 1

    n_correct = tp + tn
    acc = n_correct / n
    yes_r = tp / (tp + fn) if (tp + fn) else 0
    no_r = tn / (tn + fp) if (tn + fp) else 0

    print("\n" + "=" * 60)
    print(f"TEST-SET EVAL — {args.model}")
    print("=" * 60)
    print(f"Accuracy: {n_correct}/{n} ({100*acc:.1f}%)   [unknown={n_unknown}]")
    print(f"Yes recall: {100*yes_r:.1f}%   No recall: {100*no_r:.1f}%")
    print(f"Confusion: tp={tp} fp={fp} tn={tn} fn={fn}")
    print("\nPer-type accuracy:")
    for k, v in sorted(by_type.items()):
        a = v["correct"] / v["total"] if v["total"] else 0
        print(f"  {k}: {v['correct']}/{v['total']} ({100*a:.1f}%)")

    summary = {
        "model": args.model, "n": n, "accuracy": round(acc, 4),
        "yes_recall": round(yes_r, 4), "no_recall": round(no_r, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_unknown": n_unknown,
        "by_type": {k: {"total": v["total"], "correct": v["correct"],
                        "acc": round(v["correct"] / v["total"], 4) if v["total"] else 0}
                    for k, v in sorted(by_type.items())},
    }
    with open(out_dir / f"summary_{safe}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/summary_{safe}.json")


if __name__ == "__main__":
    main()
