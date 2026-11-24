#!/usr/bin/env python3
"""Evaluate verify_cases_v5.json against the deployed NMR Matching /match API."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CASES = "/fs_mol/cxq/SpecMaster_jp/data/verify_cases_v5.json"


def payload_from_case(case: dict[str, Any], include_reasoning: bool) -> dict[str, Any]:
    frequency = case.get("nmr_frequency")
    if frequency is not None and not isinstance(frequency, str):
        frequency = str(frequency)
    return {
        "candidate_smiles": case["candidate_smiles"],
        "c_nmr_peaks": case.get("c_nmr_peaks") or [],
        "h_nmr_peaks": case.get("h_nmr_peaks") or [],
        "nmr_solvent": case.get("nmr_solvent"),
        "nmr_frequency": frequency,
        "experimental_total_nH": case.get("experimental_total_nH"),
        "include_model_reasoning_in_key_evidence": include_reasoning,
    }


def tool_sequence(response: dict[str, Any]) -> list[str]:
    return [
        turn["tool_call"]["name"]
        for turn in response.get("trajectory") or []
        if isinstance(turn.get("tool_call"), dict)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output", default="/tmp/v6_4_api_verify15.json")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--include-model-reasoning", action="store_true")
    parser.add_argument("--fail-on-incorrect", action="store_true")
    args = parser.parse_args()

    base = args.api_url.rstrip("/")
    cases = json.loads(Path(args.cases).read_text())
    if not isinstance(cases, list) or len(cases) != 15:
        raise SystemExit(f"Expected 15 cases, found {len(cases) if isinstance(cases, list) else type(cases).__name__}")

    results: list[dict[str, Any]] = []
    correct = failures = 0
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(base + "/health")
        health.raise_for_status()
        health_body = health.json()
        if health_body.get("ready") is not True:
            raise SystemExit(f"API is not ready: {health_body}")
        print(f"[health] {base} ready=true  [cases] {len(cases)}")
        print(f"{'idx':<6}{'negative_type':<56}{'gt':<5}{'pred':<7}{'lat/s':<9}tools")
        print("-" * 145)

        for case in cases:
            started = time.perf_counter()
            error = None
            response: dict[str, Any] = {}
            try:
                http_response = client.post(
                    base + "/match",
                    json=payload_from_case(case, args.include_model_reasoning),
                )
                http_response.raise_for_status()
                response = http_response.json()
                pred = response.get("final_answer")
                if pred not in {"Yes", "No"}:
                    raise ValueError(f"invalid final_answer: {pred!r}")
            except Exception as exc:
                pred = "FAIL"
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
            latency = time.perf_counter() - started
            is_correct = pred == case["gt_answer"]
            correct += int(is_correct)
            tools = tool_sequence(response)
            print(
                f"{case['index']:<6}{case['negative_type'][:54]:<56}"
                f"{case['gt_answer']:<5}{pred:<7}{latency:<9.1f}"
                f"{','.join(tools)[:60]}  {'✓' if is_correct else '✗'}"
            )
            results.append({
                "index": case["index"],
                "negative_type": case["negative_type"],
                "gt_answer": case["gt_answer"],
                "pred_answer": pred,
                "correct": is_correct,
                "latency_s": round(latency, 3),
                "tools": tools,
                "error": error,
                "response": response,
            })

    print("-" * 145)
    print(f"correct={correct}/{len(cases)}  failures={failures}  output={args.output}")
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if failures or (args.fail_on_incorrect and correct != len(cases)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
