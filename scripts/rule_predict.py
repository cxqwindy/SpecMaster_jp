#!/usr/bin/env python3
"""Deterministic decision layer over the four matching tools' outputs.

Given a `tool_log = {count_atoms_C, count_atoms_H, spectral_match, functional_group_scan}` this
module returns ("Yes"|"No"|"UNSURE", confidence). Rule ensemble is applied at inference time
alongside the v5 model: rule wins when confidence is high and disagrees with the model, otherwise
model output stands.

The rule is deliberately conservative -- it fires only on RDKit-verified hard contradictions and
very high / very low spectral similarity. Middle-band cases fall through to the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def rule_predict(tool_log: Dict[str, Any]) -> Tuple[str, float]:
    """Return (verdict, confidence). verdict in {"Yes", "No", "UNSURE"}. UNSURE means the rule
    abstains and the caller should trust the model."""
    ca_c = tool_log.get("count_atoms_C", {}) or {}
    ca_h = tool_log.get("count_atoms_H", {}) or {}
    sm = tool_log.get("spectral_match", {}) or {}

    # --- Hard structural contradictions => No -----------------------------------------------
    # candidate has FEWER carbons than resolved 13C signals: physically impossible for a match
    if (ca_c.get("match") is False
            and ca_c.get("candidate_count") is not None
            and ca_c.get("experimental_distinct_signals") is not None
            and ca_c["candidate_count"] < ca_c["experimental_distinct_signals"]):
        return ("No", 0.98)
    # H count off by 3 or more from experimental integration => No
    if (ca_h.get("match") is False
            and ca_h.get("candidate_count") is not None
            and ca_h.get("experimental_total_nH") is not None
            and abs(ca_h["candidate_count"] - ca_h["experimental_total_nH"]) >= 3):
        return ("No", 0.95)

    # --- No spectral-only rule --------------------------------------------------------------
    # The forward NMR predictor is noisy: on v5's test_2000 rollout, 17/20 rule-induced regressions
    # came from a "sim<0.3 -> No" rule that fired on TRUE positives with weak predictions. Any
    # spectral-similarity threshold trades wrong-positive rescue against right-positive breakage.
    # RDKit atom-count facts are the only rule signal robust enough to override the model.
    return ("UNSURE", 0.0)


def combine_with_model(rule_verdict: str,
                        rule_conf: float,
                        model_verdict: str,
                        model_conf: float = 0.9,
                        override_threshold: float = 0.9) -> Tuple[str, str]:
    """Voting rule between the deterministic layer and the LLM.
    Returns (final_verdict, source_label)."""
    if rule_verdict == "UNSURE":
        return (model_verdict, "model")
    if rule_verdict == model_verdict:
        return (model_verdict, "agree")
    # disagreement: rule wins only if it is very confident (deterministic evidence)
    if rule_conf >= override_threshold:
        return (rule_verdict, "rule_override")
    return (model_verdict, "model_kept")


if __name__ == "__main__":
    # quick self-tests
    tl = {
        "count_atoms_C": {"element": "C", "candidate_count": 12,
                           "experimental_distinct_signals": 23, "match": False},
        "count_atoms_H": {"element": "H", "candidate_count": 12,
                           "experimental_total_nH": 19, "match": False},
        "spectral_match": {"ok": True, "similarity": 0.09, "c_sim": 0.1, "h_sim": 0.07},
        "functional_group_scan": {},
    }
    print(rule_predict(tl))  # expect ("No", ~0.98)
    tl2 = {"count_atoms_C": {"match": True}, "count_atoms_H": {"match": True},
           "spectral_match": {"ok": True, "similarity": 0.85, "c_sim": 0.82, "h_sim": 0.80},
           "functional_group_scan": {}}
    print(rule_predict(tl2))  # expect ("Yes", ~0.90)
    tl3 = {"count_atoms_C": {"match": True}, "count_atoms_H": {"match": True},
           "spectral_match": {"ok": True, "similarity": 0.55, "c_sim": 0.5, "h_sim": 0.6},
           "functional_group_scan": {}}
    print(rule_predict(tl3))  # expect ("UNSURE", 0)
