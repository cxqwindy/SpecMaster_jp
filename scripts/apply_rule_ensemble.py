#!/usr/bin/env python3
"""Apply the rule ensemble on top of the v5 rollout for test_2000, and report:
  * baseline v5 metrics
  * rule-only metrics
  * v5 + rule voting metrics

Inputs:
  - v5 rollout : /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json
  - tool_log   : /fs_mol/cxq/SpecMaster_jp/eval_results_testset_api_2000/test2000_tool_log.jsonl

Also writes an audit file with per-row (rule_verdict, rule_conf, model_verdict, final, source).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rule_predict import rule_predict, combine_with_model  # noqa: E402


def parse_final(ans_str: str) -> str:
    m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', ans_str or "")
    return m.group(1) if m else "Unknown"


def report(name: str, per_class: dict):
    total = sum(v["n"] for v in per_class.values())
    correct = sum(v["cor"] for v in per_class.values())
    print(f"\n=== {name} ===")
    print(f"overall = {correct}/{total} = {100*correct/max(total,1):.2f}%")
    # yes_recall / no_recall
    yc = per_class.get("positive", {"cor": 0, "n": 0})
    no_total = sum(v["n"] for k, v in per_class.items() if k != "positive")
    no_cor = sum(v["cor"] for k, v in per_class.items() if k != "positive")
    print(f"yes_recall = {yc['cor']}/{yc['n']} = {100*yc['cor']/max(yc['n'],1):.1f}%   "
          f"no_recall = {no_cor}/{no_total} = {100*no_cor/max(no_total,1):.1f}%")
    for k in sorted(per_class):
        v = per_class[k]
        print(f"  {k:60s} {v['cor']}/{v['n']} ({100*v['cor']/max(v['n'],1):.1f}%)")


def extract_tool_log_from_transcript(transcript):
    """Reconstruct a partial tool_log dict from an M3-style rollout transcript. Only tools the
    model actually called are populated; missing tools get {} (rule handles that by only checking
    fields that exist)."""
    log = {"count_atoms_C": {}, "count_atoms_H": {}, "spectral_match": {},
           "functional_group_scan": {}}
    for turn in transcript or []:
        act = turn.get("action") or {}
        if act.get("type") != "tool":
            continue
        name = act.get("name")
        args = act.get("arguments") or {}
        tobs = turn.get("tool_observation")
        if not tobs:
            continue
        try:
            obs = json.loads(tobs) if isinstance(tobs, str) else tobs
        except Exception:
            continue
        if name == "count_atoms":
            el = (args.get("element") or obs.get("element") or "").upper()
            log[f"count_atoms_{el}"] = obs
        elif name == "spectral_match":
            log["spectral_match"] = obs
        elif name == "functional_group_scan":
            log["functional_group_scan"] = obs
    return log


def main():
    v5_dump = "/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/v5_eval_test2000.json"
    audit_out = "/fs_mol/cxq/SpecMaster_jp/eval_results_testset_api_2000/rule_ensemble_audit.jsonl"

    v5 = json.load(open(v5_dump))
    if not isinstance(v5, list):
        raise SystemExit("v5 dump must be a list")

    baseline = defaultdict(lambda: {"n": 0, "cor": 0})
    ruleonly = defaultdict(lambda: {"n": 0, "cor": 0})
    ensemble = defaultdict(lambda: {"n": 0, "cor": 0})
    src_counter = Counter()
    override_flips = {"correct_to_wrong": 0, "wrong_to_correct": 0, "wrong_to_wrong": 0,
                      "correct_to_correct": 0}
    unsure_count = 0

    with open(audit_out, "w") as fo:
        for i, row in enumerate(v5):
            label = row.get("_label")
            gt = "Yes" if label == 1 else "No"
            neg_type = row.get("_negative_type") or "positive"
            model_pred = parse_final(row.get("answer") or "")

            tool_log = extract_tool_log_from_transcript(row.get("transcript") or [])
            rule_v, rule_c = rule_predict(tool_log)
            if rule_v == "UNSURE":
                unsure_count += 1
            final, src = combine_with_model(rule_v, rule_c, model_pred)
            src_counter[src] += 1

            # bookkeeping
            baseline[neg_type]["n"] += 1
            if model_pred == gt:
                baseline[neg_type]["cor"] += 1
            if rule_v != "UNSURE":
                # rule-only counts only cases the rule spoke on
                ruleonly[neg_type]["n"] += 1
                if rule_v == gt:
                    ruleonly[neg_type]["cor"] += 1
            ensemble[neg_type]["n"] += 1
            if final == gt:
                ensemble[neg_type]["cor"] += 1

            # flip analytics
            if src == "rule_override":
                mc = (model_pred == gt); rc = (rule_v == gt)
                if mc and not rc:
                    override_flips["correct_to_wrong"] += 1
                elif not mc and rc:
                    override_flips["wrong_to_correct"] += 1
                elif not mc and not rc:
                    override_flips["wrong_to_wrong"] += 1
                else:
                    override_flips["correct_to_correct"] += 1

            fo.write(json.dumps({
                "i": i, "label": label, "gt": gt, "neg_type": neg_type,
                "model_pred": model_pred, "rule_verdict": rule_v, "rule_conf": rule_c,
                "final": final, "source": src,
            }) + "\n")

    print(f"rule abstain (UNSURE) rate: {unsure_count}/{len(v5)} = {100*unsure_count/len(v5):.1f}%")
    print(f"decision source counts: {dict(src_counter)}")
    print(f"rule_override outcome breakdown: {override_flips}")
    report("baseline v5", baseline)
    report("rule-only (subset where rule spoke)", ruleonly)
    report("v5 + rule ensemble", ensemble)
    print(f"\n[audit] wrote {audit_out}")


if __name__ == "__main__":
    main()
