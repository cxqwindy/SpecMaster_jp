#!/usr/bin/env python3
"""Smoke-test the M4 GRPO setup end-to-end WITHOUT actually training.

Runs on any machine (no A100 needed) — verifies static integrity of every
piece that GRPO will touch:

  1. Parquet: train + val files exist, load, have {prompt, reward_model, ...}
     schema; prompt is [system, user] JSON; ground_truth has required keys.
  2. Reward: import compute_score from reward_matching_agent.py, run on
     synthetic transcripts (correct / wrong / no-tool / diverse / malformed).
  3. Tool config yaml: loads, each tool entry has required fields.
  4. Tool runtime: NmrMatchingRuntime instantiates from a case dict, each
     handler runs without exception and returns the expected observation shape.
  5. Prediction cache: covers all train candidates (sanity — spectral_match
     needs this).
  6. Tokenizer + chat template: renders a training-shape prompt WITH the tool
     schemas appended (via tools=), matches the training-time prompt shape.
  7. GRPO launcher script: parseable, no obvious typos.

Prints PASS/FAIL per subtest and exits non-zero if anything fails.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
LF_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory")
RECIPE = AGENT_ROOT / "recipe" / "nmr_matching_agent"

sys.path.insert(0, str(AGENT_ROOT))

FAILURES = []


def _pass(name, _detail=""): print(f"  ✓ {name}")
def _fail(name, err):
    print(f"  ✗ {name}\n    {err}")
    FAILURES.append((name, str(err)))


def section(title):
    print(f"\n===== {title} =====")


# ---------------------------------------------------------------------------
def test_parquet():
    section("1. Parquet integrity")
    import pandas as pd
    for label, path in [
        ("train", RECIPE / "data/rl_train_expanded_agent.parquet"),
        ("val", RECIPE / "data/rl_val_1000_agent.parquet"),
    ]:
        try:
            assert path.exists(), f"missing: {path}"
            df = pd.read_parquet(path)
            assert len(df) > 0, "empty parquet"
            required_cols = {"data_source", "prompt", "reward_model", "extra_info"}
            missing = required_cols - set(df.columns)
            assert not missing, f"missing columns: {missing}"
            # Row-shape checks on the first row.
            r = df.iloc[0]
            prompt = list(r["prompt"])
            assert len(prompt) == 2, f"prompt not length 2: {len(prompt)}"
            assert prompt[0].get("role") == "system", "first prompt not system"
            assert prompt[1].get("role") == "user", "second prompt not user"
            case = json.loads(prompt[1]["content"])
            assert "candidate_smiles" in case, "user content lacks candidate_smiles"
            rm = r["reward_model"] if isinstance(r["reward_model"], dict) else json.loads(r["reward_model"])
            gt = rm["ground_truth"]
            for k in ("label", "candidate_smiles", "gt_smiles"):
                assert k in gt, f"ground_truth missing {k}"
            _pass(f"{label}: {len(df)} rows, schema OK, prompt[0]=system prompt[1]=user JSON case")
        except Exception as e:
            _fail(f"parquet {label}", e)


# ---------------------------------------------------------------------------
def test_reward():
    section("2. Reward function")
    spec = importlib.util.spec_from_file_location(
        "reward_matching_agent",
        RECIPE / "reward_matching_agent.py",
    )
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    def transcript(*calls, answer_yes_no="Yes", conf=0.9):
        parts = ["<think>hypothesizing</think>"]
        for name, args in calls:
            parts.append(
                f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args)}}}\n</tool_call>'
            )
            parts.append("<tool_response>{\"ok\": true}</tool_response>")
        parts.append(
            f'<answer>{{"final_answer": "{answer_yes_no}", "confidence": {conf}, '
            f'"key_evidence": "test"}}</answer>'
        )
        return "\n".join(parts)

    # Correct Yes with 2 tools
    r = m.compute_score(
        solution_str=transcript(("count_atoms", {"element": "C"}),
                                ("spectral_match", {}),
                                answer_yes_no="Yes"),
        ground_truth={"label": 1, "spectral_sim": 0.7},
    )
    ok = (r["r_correct"] == 1.0 and r["r_format"] > 0 and r["r_tool_use"] > 0
          and r["n_tool_calls"] == 2 and r["pred_answer"] == "Yes")
    (_pass if ok else _fail)("correct Yes, 2 tools", f"score={r}" if not ok else "")

    # Wrong answer
    r = m.compute_score(
        solution_str=transcript(("count_atoms", {"element": "C"}), answer_yes_no="No"),
        ground_truth={"label": 1},
    )
    ok = (r["r_correct"] == m.WRONG_ANSWER_REWARD)
    (_pass if ok else _fail)("wrong answer (label=Yes, pred=No)", f"score={r}" if not ok else "")

    # No tool call → penalty
    r = m.compute_score(
        solution_str='<think>skipping tools</think><answer>{"final_answer": "Yes", "confidence": 0.5, "key_evidence": ""}</answer>',
        ground_truth={"label": 1},
    )
    ok = (r["r_tool_use"] == m.NO_TOOL_PENALTY and r["r_tool_diversity"] == 0.0)
    (_pass if ok else _fail)("no tool → penalty", f"score={r}" if not ok else "")

    # Tool diversity: 3 unique tool types
    r = m.compute_score(
        solution_str=transcript(
            ("count_atoms", {"element": "C"}),
            ("spectral_match", {}),
            ("check_substructure", {"pattern": "methoxy"}),
            answer_yes_no="No",
        ),
        ground_truth={"label": 0},
    )
    ok = (r["r_correct"] == 1.0 and r["n_tool_types"] == 3
          and r["r_tool_diversity"] == 2 * m.R_TOOL_DIVERSITY_REWARD)
    (_pass if ok else _fail)("3-tool diversity bonus", f"score={r}" if not ok else "")

    # Unparseable answer
    r = m.compute_score(
        solution_str=transcript(("count_atoms", {"element": "C"}), answer_yes_no="?"),
        ground_truth={"label": 1},
    )
    # answer is "?" → not Yes/No → treated as Unknown
    ok = (r["r_correct"] == m.UNKNOWN_ANSWER_REWARD and r["pred_answer"] == "Unknown")
    (_pass if ok else _fail)("unparseable answer → UNKNOWN penalty",
                             f"score={r}" if not ok else "")

    # Empty solution
    r = m.compute_score(solution_str="", ground_truth={"label": 0})
    ok = (r["r_correct"] == m.UNKNOWN_ANSWER_REWARD and r["r_format"] == 0.0)
    (_pass if ok else _fail)("empty solution → UNKNOWN + zero format",
                             f"score={r}" if not ok else "")


# ---------------------------------------------------------------------------
def test_tool_config_yaml():
    section("3. Tool config yaml")
    import yaml
    p = RECIPE / "tool_config_matching.yaml"
    try:
        cfg = yaml.safe_load(open(p))
        assert "tools" in cfg, "missing tools:"
        names = []
        for entry in cfg["tools"]:
            assert "class_name" in entry
            assert "tool_schema" in entry
            fn = entry["tool_schema"].get("function", {})
            names.append(fn.get("name"))
        expected = {"count_atoms", "spectral_match", "check_substructure", "functional_group_scan"}
        assert set(names) >= expected, f"missing tools: {expected - set(names)}"
        _pass(f"4 tool schemas present: {names}")
    except Exception as e:
        _fail("tool_config_matching.yaml", e)


# ---------------------------------------------------------------------------
def test_runtime():
    section("4. Tool runtime handlers")
    from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
    case = {
        "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "c_nmr_peaks": [{"delta (ppm)": v} for v in [170.5, 169.5, 150.5, 134.0, 131.5, 126.0, 123.5, 121.5, 21.0]],
        "h_nmr_peaks": [{"delta": 8.0, "centroid": 8.0, "nH": 1},
                        {"delta": 7.55, "centroid": 7.55, "nH": 1},
                        {"delta": 7.3, "centroid": 7.3, "nH": 1},
                        {"delta": 7.1, "centroid": 7.1, "nH": 1},
                        {"delta": 2.3, "centroid": 2.3, "nH": 3}],
    }
    rt = NmrMatchingRuntime(record=case, allow_remote_predict=False)
    checks = [
        ("count_atoms(C)", rt.handlers["count_atoms"], {"element": "C"},
         lambda o: "candidate_count" in o and "match" in o),
        ("count_atoms(H)", rt.handlers["count_atoms"], {"element": "H"},
         lambda o: "candidate_count" in o),
        ("spectral_match(no cache)", rt.handlers["spectral_match"], {},
         lambda o: "ok" in o),
        ("check_substructure(methoxy)", rt.handlers["check_substructure"], {"pattern": "methoxy"},
         lambda o: isinstance(o, dict)),
        ("functional_group_scan", rt.handlers["functional_group_scan"], {},
         lambda o: "groups_present" in o),
    ]
    for name, fn, args, validator in checks:
        try:
            obs, extra = fn(args)
            assert validator(obs), f"validator failed on obs: {obs}"
            _pass(f"{name} → keys={list(obs)[:6]}")
        except Exception as e:
            _fail(name, e)


# ---------------------------------------------------------------------------
def test_prediction_cache_coverage():
    section("5. Prediction cache coverage (spectral_match hot path)")
    import pandas as pd
    try:
        cache = json.load(open(RECIPE / "prediction_cache.json"))
        train = pd.read_parquet(RECIPE / "data/rl_train_expanded_agent.parquet")
        cache_keys = set(cache.keys())
        need = set()
        for _, r in train.iterrows():
            rm = r["reward_model"] if isinstance(r["reward_model"], dict) else json.loads(r["reward_model"])
            gt = rm["ground_truth"]
            smi = gt.get("candidate_smiles")
            if smi:
                need.add(smi)
        hits = need & cache_keys
        miss = need - cache_keys
        pct = 100.0 * len(hits) / max(len(need), 1)
        # canonical fallback isn't checked here — this is a sanity floor
        if pct >= 80:
            _pass(f"cache hit rate on train candidates: {pct:.1f}% ({len(hits)}/{len(need)})")
            if miss:
                print(f"    (canonical fallback expected to cover the rest; {len(miss)} raw misses)")
        else:
            _fail("prediction cache", f"coverage too low ({pct:.1f}%); rollout will thrash cache misses")
    except Exception as e:
        _fail("prediction cache", e)


# ---------------------------------------------------------------------------
def test_tokenizer_and_chat_template():
    section("6. Tokenizer + chat template (WITH tools=)")
    try:
        from transformers import AutoTokenizer
        ckpt = LF_ROOT / "saves/qwen3-32b-nmr-matching-tool/full-exp1/checkpoint-200"
        tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        # Import TOOL_SCHEMAS + build_system_prompt
        spec = importlib.util.spec_from_file_location(
            "standalone_infer_matching",
            RECIPE / "standalone_infer_matching.py",
        )
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

        case = {"candidate_smiles": "CCO", "c_nmr_peaks": [], "h_nmr_peaks": []}
        msgs = [
            {"role": "system", "content": m.build_system_prompt()},
            {"role": "user", "content": json.dumps(case)},
        ]
        # WITHOUT tools=
        p_no_tools = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # WITH tools=
        p_with = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          tools=list(m.TOOL_SCHEMAS.values()))
        gain = len(p_with) - len(p_no_tools)
        # Sanity: with tools= should add the <tools>...</tools> schema block (~1500 chars).
        if "<tools>" in p_with and gain > 800:
            _pass(f"chat template renders with tools=: +{gain} chars (contains <tools>)")
        else:
            _fail("chat template", f"tools= did not inject schema block (delta {gain}, has <tools>: {'<tools>' in p_with})")
    except Exception as e:
        _fail("tokenizer / chat template", e)


# ---------------------------------------------------------------------------
def test_launcher_parse():
    section("7. GRPO launcher script static check")
    p = RECIPE / "grpo_qwen3_32b_agent_8gpu.sh"
    try:
        # bash -n parses without executing
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if r.returncode == 0:
            _pass("bash -n parses cleanly")
        else:
            _fail("launcher parse", r.stderr)
        text = p.read_text()
        # A couple of key sanity checks
        checks = [
            ("n_gpus_per_node=8", "n_gpus_per_node=8"),
            ("save_freq env var", "SAVE_FREQ:-"),
            ("max_num_seqs", "max_num_seqs="),
            ("save_contents=[extra]", "save_contents=[extra]"),
            ("NCCL_IB_DISABLE", "NCCL_IB_DISABLE"),
            ("reward path", "recipe/nmr_matching_agent/reward_matching_agent.py"),
            ("model path", "qwen3-32b-nmr-matching-tool/full-exp1/checkpoint-200"),
            ("tool_config path", "tool_config_matching.yaml"),
            ("multi_turn.enable=True", "multi_turn.enable=True"),
        ]
        for name, needle in checks:
            if needle in text:
                _pass(f"contains {name}")
            else:
                _fail(name, f"expected '{needle}' in launcher script")
    except Exception as e:
        _fail("launcher parse", e)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in [test_parquet, test_reward, test_tool_config_yaml, test_runtime,
               test_prediction_cache_coverage, test_tokenizer_and_chat_template,
               test_launcher_parse]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e!r}")
            FAILURES.append((fn.__name__, repr(e)))

    print()
    if not FAILURES:
        print(f"===== ALL PASS =====")
        sys.exit(0)
    print(f"===== {len(FAILURES)} FAILURE(S) =====")
    for name, msg in FAILURES:
        print(f"  - {name}: {msg}")
    sys.exit(1)
