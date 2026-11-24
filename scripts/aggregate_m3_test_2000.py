#!/usr/bin/env python3
"""Merge M3 tool-agent predictions on test_1000 with new-1000 predictions and
report per-type / old-vs-new accuracy on nmr_matching_test_2000.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ANS_RE = re.compile(r'"final_answer"\s*:\s*"(Yes|No)"')


def parse_answer(rec):
    a = rec.get("answer") or ""
    m = ANS_RE.search(a)
    if m:
        return m.group(1)
    try:
        obj = json.loads(a) if isinstance(a, str) else a
        v = obj.get("final_answer", "")
        if v in ("Yes", "No"):
            return v
    except Exception:
        pass
    a = a.strip()
    if a.startswith("Yes"): return "Yes"
    if a.startswith("No"):  return "No"
    return "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/m3_eval_full_ckpt200.json",
                    help="M3 predictions on test_1000 (rows 0..999)")
    ap.add_argument("--new",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/m3_eval_full_ckpt200_new1000.json",
                    help="M3 predictions on the new 1000 rows (indices 1000..1999)")
    ap.add_argument("--merged",
                    default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/m3_eval_full_ckpt200_test2000.json")
    ap.add_argument("--test_file",
                    default="/fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json")
    args = ap.parse_args()

    old = json.load(open(args.old))
    new = json.load(open(args.new))
    assert len(old) == 1000, f"old len {len(old)}"
    assert len(new) == 1000, f"new len {len(new)}"
    merged = old + new
    json.dump(merged, open(args.merged, "w"), ensure_ascii=False, indent=2)

    data = json.load(open(args.test_file))
    assert len(data) == 2000

    tp = fp = tn = fn = unk = 0
    by_total = Counter(); by_correct = Counter()
    old_c = old_n = new_c = new_n = 0
    for i, (rec, gt_rec) in enumerate(zip(merged, data)):
        gt = gt_rec["output"]
        pred = parse_answer(rec)
        nt = gt_rec["_negative_type"]
        by_total[nt] += 1
        if pred == "Unknown":
            unk += 1; continue
        correct = pred == gt
        if correct: by_correct[nt] += 1
        if i < 1000:
            old_n += 1; old_c += int(correct)
        else:
            new_n += 1; new_c += int(correct)
        if   gt == "Yes" and pred == "Yes": tp += 1
        elif gt == "No"  and pred == "Yes": fp += 1
        elif gt == "No"  and pred == "No":  tn += 1
        elif gt == "Yes" and pred == "No":  fn += 1

    n = len(data)
    acc = (tp + tn) / n
    yes_r = tp / (tp + fn) if (tp + fn) else 0
    no_r  = tn / (tn + fp) if (tn + fp) else 0

    print(f"Merged {len(merged)} rows -> {args.merged}")
    print(f"Accuracy: {(tp+tn)}/{n} ({acc*100:.1f}%)  unknown={unk}")
    print(f"Yes-R: {yes_r*100:.1f}%   No-R: {no_r*100:.1f}%")
    print(f"Confusion: tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"Old-1000 acc: {old_c}/{old_n} ({old_c/old_n*100:.1f}%)   "
          f"New-1000 acc: {new_c}/{new_n} ({new_c/new_n*100:.1f}%)")
    print("Per-type:")
    for t in sorted(by_total):
        tot = by_total[t]; cor = by_correct[t]
        print(f"  {t}: {cor}/{tot} ({cor/tot*100:.1f}%)")


if __name__ == "__main__":
    main()
