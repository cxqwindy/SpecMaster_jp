#!/usr/bin/env python3
"""Distill teacher CoT (Gemini-3.1-Pro / GPT-5.5 / DeepSeek-V4-Pro) on the v5 candidate pool.

Difference from evaluate_testset_api.py:
  1) System prompt no longer forces "Output ONLY JSON" — asks for a 5–8 step reasoning trace
     that CITES tool_log numbers (atom counts, similarity floats), and ends with a `FINAL:` line
     containing the JSON answer.
  2) User prompt embeds a TOOL_LOG block with real deterministic tool outputs so the teacher
     "reproduces" reasoning grounded in numbers, killing v4's Gemini "at-least-23-carbons"
     hallucination at the source.
  3) Persists BOTH `content` AND `reasoning_content` — the whole point of this run is to keep
     the CoT, unlike the eval harness which discarded it.
  4) Passes reasoning/thinking budgets appropriate to each model family.

Input : distill_out/tool_log_pool.jsonl   (from build_tool_log.py, ~2400 rows)
Output: distill_out/{safe_model}.jsonl     (one row per input row, teacher CoT preserved)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import litellm
from litellm import completion

litellm.drop_params = True


import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from gateway_credentials import require_gateway_api_key


API_BASE = "https://ai-gateway-global.dp.tech"

SYSTEM_PROMPT = (
    "You are an NMR spectrum-structure matching expert. You will be given a candidate SMILES, "
    "its experimental 13C and 1H NMR peak list, and a TOOL_LOG block containing the exact JSON "
    "output of four deterministic verification tools (count_atoms(C), count_atoms(H), "
    "spectral_match, functional_group_scan). Your job: decide whether the candidate matches the "
    "spectrum.\n\n"
    "Reasoning requirements (ALL mandatory):\n"
    "  * Produce 5 to 8 numbered reasoning steps.\n"
    "  * Every step MUST cite at least one concrete number — an atom count, an experimental "
    "signal count, a similarity value (0..1), or a specific δ ppm shift.\n"
    "  * Where a number comes from the TOOL_LOG, quote the tool name inline "
    "(e.g. 'count_atoms(C) returned candidate_count=22, experimental_signals=23').\n"
    "  * Do NOT guess an atom count in your reasoning — always cite the tool value.\n"
    "  * IMPORTANT — handling missing 13C data: if spectral_match shows c_data_absent=true "
    "(n_exp_c=0), the 13C spectrum is unavailable for this sample. In that case c_sim=0.0 is "
    "UNINFORMATIVE and must NOT be used as evidence against a match. Base your judgment on "
    "h_sim, count_atoms(H), and functional_group_scan only; do not penalise the candidate for "
    "the absent 13C data.\n"
    "  * A single decisive contradiction is enough to answer 'No'. A 'Yes' requires ALL "
    "available (non-absent) verified fields to be consistent.\n\n"
    "End your response with EXACTLY one line of the form:\n"
    "FINAL: {\"final_answer\": \"Yes\" or \"No\", \"confidence\": <float 0.0-1.0>, "
    "\"key_evidence\": \"<one sentence>\"}"
)


def _peaks_pretty(peaks_c, peaks_h) -> str:
    def fmt_c(p):
        d = p.get("delta (ppm)") or p.get("delta")
        return f"{float(d):.2f}" if d is not None else "?"

    def fmt_h(p):
        d = p.get("delta")
        nh = p.get("nH")
        s = f"{float(d):.2f}" if d is not None else "?"
        if nh is not None:
            s += f" (nH={nh})"
        return s

    c = ", ".join(fmt_c(p) for p in (peaks_c or []))
    h = ", ".join(fmt_h(p) for p in (peaks_h or []))
    return f"  13C peaks (ppm): [{c}]\n  1H peaks: [{h}]"


def build_user(row: Dict[str, Any]) -> str:
    inp = row["input"]
    tl = row.get("tool_log") or {}
    return (
        f"Candidate SMILES: {inp['candidate_smiles']}\n"
        f"Experimental NMR ({inp.get('nmr_solvent') or '?'} @ "
        f"{inp.get('nmr_frequency') or '?'} MHz):\n"
        f"{_peaks_pretty(inp.get('c_nmr_peaks'), inp.get('h_nmr_peaks'))}\n\n"
        "TOOL_LOG (deterministic verification tools already ran on this candidate; quote these "
        "numbers verbatim in your reasoning):\n"
        f"  count_atoms(element=C): {json.dumps(tl.get('count_atoms_C', {}), ensure_ascii=False)}\n"
        f"  count_atoms(element=H): {json.dumps(tl.get('count_atoms_H', {}), ensure_ascii=False)}\n"
        f"  spectral_match(): {json.dumps(tl.get('spectral_match', {}), ensure_ascii=False)}\n"
        f"  functional_group_scan(): {json.dumps(tl.get('functional_group_scan', {}), ensure_ascii=False)}\n\n"
        "Now produce your 5–8 step reasoning trace citing these numbers, then end with the "
        "FINAL: {...} JSON line."
    )


def _reasoning_kwargs(model: str) -> Dict[str, Any]:
    """Model-family-specific reasoning budget switches. LiteLLM drops unsupported params, so a
    kwarg that a given gateway silently ignores is harmless — but including them for the models
    that DO support them gets us the full CoT."""
    m = model.lower()
    if "gemini" in m:
        return {"thinking": {"type": "enabled", "budget_tokens": 8192}}
    if "gpt-5" in m or "gpt5" in m:
        return {"reasoning_effort": "high"}
    if "claude" in m or "anthropic" in m:
        # Anthropic native reasoning param (LiteLLM passes through for the openai/anthropic bridge).
        return {"thinking": {"type": "enabled", "budget_tokens": 8192}}
    if "deepseek" in m:
        # DS-V4-Pro tends to burn max_tokens on unbounded reasoning; keep default effort so it
        # finishes and emits a FINAL: line inside the token budget.
        return {}
    return {}


def call_teacher(model: str, user: str, max_retries: int = 6, max_tokens: int = 8192
                 ) -> Dict[str, Any]:
    kw = _reasoning_kwargs(model)
    if "deepseek" in model.lower():
        max_tokens = 24000
    # Some models (e.g. gpt-6-astra) only accept default temperature; skip the param for them.
    m_lower = model.lower()
    api_key = require_gateway_api_key()
    temp_kw: Dict[str, Any] = {} if ("gpt-6" in m_lower or "astra" in m_lower) else {"temperature": 0.0}
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = completion(
                model=model, api_key=api_key, api_base=API_BASE,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                max_completion_tokens=max_tokens, timeout=300, **temp_kw, **kw,
            )
            m = resp.choices[0].message
            content = m.content or ""
            reasoning_content = getattr(m, "reasoning_content", None) or ""
            usage = getattr(resp, "usage", None)
            usage_d = {}
            if usage is not None:
                usage_d = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            return {"content": content, "reasoning_content": reasoning_content, "usage": usage_d}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            msg = last_err.lower()
            if any(k in msg for k in ("rate", "429", "quota", "resource", "overloaded")):
                time.sleep(min(90, 6 * (2 ** attempt)))
            else:
                time.sleep(3 * (attempt + 1))
    return {"content": "", "reasoning_content": "", "error": last_err}


def parse_final_answer(content: str) -> str:
    """Extract 'Yes'/'No' from a FINAL: {...} tail; fall back to substring search."""
    import re
    if not content:
        return "Unknown"
    # find the LAST occurrence of `FINAL:` and parse the JSON that follows
    idx = content.rfind("FINAL:")
    tail = content[idx + len("FINAL:"):] if idx >= 0 else content
    a, b = tail.find("{"), tail.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(tail[a:b + 1])
            v = str(obj.get("final_answer", "")).strip()
            if v in ("Yes", "No"):
                return v
        except json.JSONDecodeError:
            pass
    m = re.search(r'"final_answer"\s*:\s*"(Yes|No)"', content)
    if m:
        return m.group(1)
    # DeepSeek reasoning-mode fallback: if the visible content ends abruptly (no FINAL: line),
    # sometimes the last verdict phrase is buried in prose like "answer is No" or "final: Yes"
    m = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*(Yes|No)\b", content, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="LiteLLM model id, e.g. openai/specxmaster/gemini-3.1-pro-preview")
    ap.add_argument("--pool",
                    default="/fs_mol/cxq/SpecMaster_jp/distill_out/tool_log_pool.jsonl")
    ap.add_argument("--out_dir",
                    default="/fs_mol/cxq/SpecMaster_jp/distill_out/")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--only_indices", default="",
                    help="optional path to a JSON list of pool row indices to (re)run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "__")
    cache_path = out_dir / f"distill_{safe}.jsonl"

    pool = []
    with open(args.pool) as f:
        for line in f:
            pool.append(json.loads(line))
    if args.max_rows is not None:
        pool = pool[:args.max_rows]
    n = len(pool)

    keep_i = None
    if args.only_indices:
        keep_i = set(json.load(open(args.only_indices)))

    done: Dict[int, Dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("content") and r.get("pred") != "Unknown":
                    done[r["i"]] = r
    todo = [i for i in range(n)
            if i not in done and (keep_i is None or i in keep_i)]
    print(f"[{args.model}] pool_n={n} cached={len(done)} todo={len(todo)}", flush=True)

    fout = open(cache_path, "a", encoding="utf-8")

    def _one(i: int):
        row = pool[i]
        user = build_user(row)
        resp = call_teacher(args.model, user)
        pred = parse_final_answer(resp.get("content", ""))
        return i, {
            "i": i,
            "case_id": row.get("case_id"),
            "candidate_smiles": row["input"]["candidate_smiles"],
            "gt_smiles": row.get("gt_smiles"),
            "label": row.get("label"),
            "negative_type": row.get("negative_type"),
            "_class": row.get("_class"),
            "teacher": args.model,
            "pred": pred,
            "content": resp.get("content", ""),
            "reasoning_content": resp.get("reasoning_content", ""),
            "usage": resp.get("usage", {}),
            "error": resp.get("error"),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, i): i for i in todo}
        n_done = 0
        n_ok = 0
        for fut in as_completed(futs):
            i, rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if rec["pred"] in ("Yes", "No"):
                n_ok += 1
            if n_done % 50 == 0:
                print(f"  {n_done}/{len(todo)}  parsed_ok={n_ok}", flush=True)
    fout.close()
    print(f"[done] wrote {cache_path}   parsed_ok≈{n_ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
