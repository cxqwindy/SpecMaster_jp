#!/usr/bin/env python3
"""Rewrite final NMR-matching think/key_evidence from current tool observations.

The v6.3 converter appended teacher key_evidence after a deterministic summary.
When the prediction cache or scorer changed, that teacher prose could contain stale
similarity/count values. This script replaces the complete final assistant turn with
an observation-grounded version while preserving final_answer and confidence.

By default only rows whose meta.provenance_violations is non-empty are rewritten.
The input is never modified in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def infer_tool_name(message: dict[str, Any], obs: dict[str, Any]) -> str:
    name = message.get("name")
    if name:
        return str(name)
    if obs.get("element") in {"C", "H"} and "candidate_count" in obs:
        return "count_atoms"
    if "similarity" in obs:
        return "spectral_match"
    if "groups_present" in obs:
        return "functional_group_scan"
    if any(key in obs for key in ("is_present", "present_in_candidate", "n_matches")):
        return "check_substructure"
    return "unknown_tool"


def collect_evidence(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    evidence: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            obs = json.loads(message.get("content") or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(obs, dict):
            evidence.append((infer_tool_name(message, obs), obs))
    return evidence


def summarize_observation(tool_name: str, obs: dict[str, Any]) -> str:
    if tool_name == "count_atoms":
        element = str(obs.get("element", "")).upper()
        candidate = obs.get("candidate_count")
        if element == "C":
            experimental = obs.get("experimental_distinct_signals")
            if obs.get("match") is True:
                verdict = "consistent (candidate C >= signals)"
            elif obs.get("match") is False:
                verdict = "CONTRADICTION (fewer candidate carbons than signals)"
            else:
                verdict = "comparison unavailable"
            return (
                f"count_atoms(C) returned candidate_count={candidate}, "
                f"experimental_signals={experimental} — {verdict}"
            )
        if element == "H":
            experimental = obs.get("experimental_total_nH")
            if experimental is None or obs.get("match") is None:
                return (
                    f"count_atoms(H) returned candidate_count={candidate}, "
                    "experimental_total_nH=unavailable"
                )
            verdict = "counts match" if obs.get("match") else "MISMATCH in total proton count"
            return (
                f"count_atoms(H) returned candidate_count={candidate}, "
                f"experimental_total_nH={experimental} — {verdict}"
            )

    if tool_name == "spectral_match":
        if not obs.get("ok"):
            return "spectral_match returned ok=false (prediction unavailable)"
        summary = (
            f"spectral_match returned similarity={obs.get('similarity')}, "
            f"c_sim={obs.get('c_sim')}, h_sim={obs.get('h_sim')}, "
            f"n_pred_c={obs.get('n_pred_c')} vs n_exp_c={obs.get('n_exp_c')}, "
            f"n_pred_h={obs.get('n_pred_h')} vs n_exp_h={obs.get('n_exp_h')}"
        )
        if obs.get("c_data_absent"):
            summary += (
                "; c_data_absent=true — n_exp_c=0 so c_sim=0.0 is "
                "uninformative, rely on h_sim only"
            )
        return summary

    if tool_name == "functional_group_scan":
        groups = obs.get("groups_present") or []
        return f"functional_group_scan returned {obs.get('n_groups', len(groups))} groups: {groups}"

    if tool_name == "check_substructure":
        present = obs.get("present_in_candidate")
        if present is None:
            present = obs.get("is_present")
        name = obs.get("resolved_name") or obs.get("pattern") or "requested pattern"
        verdict = "present" if present else "NOT present"
        return (
            f"check_substructure({name}) returned {verdict} in candidate "
            f"(n_matches={obs.get('n_matches')})"
        )

    return f"{tool_name} returned {json.dumps(obs, ensure_ascii=False, sort_keys=True)}"


def evidence_by_kind(evidence: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool_name, obs in evidence:
        if tool_name == "count_atoms" and obs.get("element") in {"C", "H"}:
            result[f"count_{obs['element']}"] = obs
        else:
            result[tool_name] = obs
    return result


def build_key_evidence(final_answer: str, evidence: list[tuple[str, dict[str, Any]]]) -> str:
    """Build one factual sentence without importing stale teacher interpretation."""
    tools = evidence_by_kind(evidence)
    carbon = tools.get("count_C") or {}
    hydrogen = tools.get("count_H") or {}
    spectral = tools.get("spectral_match") or {}

    if final_answer == "No":
        if carbon.get("match") is False:
            return (
                f"count_atoms(C) returned candidate_count={carbon.get('candidate_count')} "
                f"versus experimental_signals={carbon.get('experimental_distinct_signals')}, "
                "a hard carbon-count contradiction."
            )
        if hydrogen.get("match") is False:
            return (
                f"count_atoms(H) returned candidate_count={hydrogen.get('candidate_count')} "
                f"versus experimental_total_nH={hydrogen.get('experimental_total_nH')}, "
                "a total-proton-count mismatch."
            )
        if spectral.get("ok"):
            prefix = "The C/H count checks did not distinguish this candidate"
            if hydrogen.get("match") is None:
                prefix = "The carbon-count check passed but proton-count comparison was unavailable"
            return (
                f"{prefix}; spectral_match returned similarity={spectral.get('similarity')} "
                f"(c_sim={spectral.get('c_sim')}, h_sim={spectral.get('h_sim')})."
            )
        return "The current spectral prediction was unavailable, so the remaining tools did not establish a match."

    if carbon.get("match") is True and hydrogen.get("match") is True:
        count_status = "count_atoms found no carbon- or hydrogen-count contradiction"
    elif carbon.get("match") is True and hydrogen.get("match") is None:
        count_status = "count_atoms(C) found no hard contradiction and proton-count comparison was unavailable"
    else:
        count_status = "the available atom-count checks found no hard contradiction"

    if spectral.get("ok"):
        return (
            f"{count_status}; spectral_match returned similarity={spectral.get('similarity')} "
            f"(c_sim={spectral.get('c_sim')}, h_sim={spectral.get('h_sim')})."
        )
    return f"{count_status}; spectral prediction was unavailable."


def parse_final_turn(messages: list[dict[str, Any]], row_index: int) -> tuple[int, dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        match = ANSWER_RE.search(message.get("content") or "")
        if not match:
            continue
        try:
            answer = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row_index}: invalid final <answer> JSON: {exc}") from exc
        if answer.get("final_answer") not in {"Yes", "No"}:
            raise ValueError(f"row {row_index}: unsupported final_answer={answer.get('final_answer')!r}")
        return index, answer
    raise ValueError(f"row {row_index}: final assistant <answer> not found")


def has_stale_spectral_context(messages: list[dict[str, Any]], violations: list[str]) -> bool:
    final_text = next(
        (str(message.get("content") or "") for message in reversed(messages)
         if message.get("role") == "assistant"),
        "",
    )
    think_match = THINK_RE.search(final_text)
    think = think_match.group(1) if think_match else ""
    for violation in violations:
        number_match = re.search(r"similarity/float '([^']+)'", violation)
        if not number_match:
            continue
        number = number_match.group(1)
        for occurrence in re.finditer(re.escape(number), think):
            context = think[max(0, occurrence.start() - 100):occurrence.end() + 100].lower()
            if re.search(
                r"similarity|spectral_match|c_sim|h_sim|match score|spectral score",
                context,
            ):
                return True
    return False


def load_provenance_validator(root: Path) -> Callable[[list[dict[str, str]]], list[str]]:
    if not root.exists():
        raise FileNotFoundError(f"validator root does not exist: {root}")
    sys.path.insert(0, str(root))
    try:
        from recipe.nmr_matching_agent.cot_to_trajectory_v5 import (  # type: ignore
            _validate_think_number_provenance,
        )
    except ImportError as exc:
        raise ImportError(f"could not import authoritative provenance validator from {root}") from exc
    return _validate_think_number_provenance


def rewrite_row(
    row: dict[str, Any],
    row_index: int,
    validator: Callable[[list[dict[str, str]]], list[str]],
) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"row {row_index}: messages is not a list")
    evidence = collect_evidence(messages)
    if not evidence:
        raise ValueError(f"row {row_index}: no parseable tool observations")
    final_index, answer = parse_final_turn(messages, row_index)
    original_answer = dict(answer)
    final_answer = answer["final_answer"]
    key_evidence = build_key_evidence(final_answer, evidence)
    answer["key_evidence"] = key_evidence
    summaries = " ; ".join(summarize_observation(name, obs) for name, obs in evidence)
    final_think = f"{summaries}. {key_evidence} Therefore final_answer = {final_answer}."
    messages[final_index]["content"] = (
        f"<think>{final_think}</think>\n"
        f"<answer>{json.dumps(answer, ensure_ascii=False)}</answer>"
    )

    remaining = validator(messages)
    if remaining:
        raise ValueError(f"row {row_index}: rewritten final turn still violates provenance: {remaining}")
    reparsed_index, reparsed_answer = parse_final_turn(messages, row_index)
    if reparsed_index != final_index:
        raise ValueError(f"row {row_index}: final assistant turn moved unexpectedly")
    for key, value in original_answer.items():
        if key != "key_evidence" and reparsed_answer.get(key) != value:
            raise ValueError(f"row {row_index}: answer field {key!r} changed")
    row.setdefault("meta", {})["provenance_violations"] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--scope",
        choices=("provenance", "all"),
        default="provenance",
        help="rewrite rows with meta.provenance_violations (default), or every row",
    )
    parser.add_argument(
        "--validator-root",
        default="/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL",
        help="repository containing the authoritative provenance validator",
    )
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-repaired", type=int)
    parser.add_argument("--expected-stale-spectral", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.out)
    report_path = Path(args.report) if args.report else None
    resolved = [input_path.resolve(), output_path.resolve()]
    if report_path:
        resolved.append(report_path.resolve())
    if len(set(resolved)) != len(resolved):
        raise ValueError("--in, --out, and --report must resolve to distinct paths")
    if not args.dry_run and not args.force:
        existing = [str(path) for path in (output_path, report_path) if path and path.exists()]
        if existing:
            raise FileExistsError(f"refusing to replace existing output without --force: {existing}")

    validator = load_provenance_validator(Path(args.validator_root))
    rows = jsonl_rows(input_path)
    repaired_indices: list[int] = []
    class_counts: Counter[str] = Counter()
    stale_spectral_rows = 0
    violations_before = 0

    for index, row in enumerate(rows):
        meta = row.get("meta") or {}
        violations = list(meta.get("provenance_violations") or [])
        selected = args.scope == "all" or bool(violations)
        if not selected:
            continue
        violations_before += len(violations)
        stale_spectral_rows += int(has_stale_spectral_context(row.get("messages") or [], violations))
        label = int(meta.get("label"))
        class_name = "positive" if label == 1 else str(meta.get("negative_type") or "negative")
        class_counts[class_name] += 1
        rewrite_row(row, index, validator)
        repaired_indices.append(index)

    expected = {
        "total rows": (args.expected_total, len(rows)),
        "repaired rows": (args.expected_repaired, len(repaired_indices)),
        "stale spectral rows": (args.expected_stale_spectral, stale_spectral_rows),
    }
    for name, (wanted, actual) in expected.items():
        if wanted is not None and wanted != actual:
            raise ValueError(f"expected {wanted} {name}, found {actual}")

    answer_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        violations = validator(messages)
        if violations:
            raise ValueError(f"row {index}: dataset validation found provenance violations: {violations}")
        _, answer = parse_final_turn(messages, index)
        final_answer = answer["final_answer"]
        meta = row.get("meta") or {}
        expected_answer = "Yes" if int(meta.get("label")) == 1 else "No"
        if final_answer != expected_answer or meta.get("final_answer") != final_answer:
            raise ValueError(
                f"row {index}: label/meta/answer disagreement: "
                f"label={meta.get('label')}, meta.final_answer={meta.get('final_answer')}, "
                f"answer={final_answer}"
            )
        answer_counts[final_answer] += 1

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "scope": args.scope,
        "total_rows": len(rows),
        "repaired_rows": len(repaired_indices),
        "stale_spectral_rows_among_repaired": stale_spectral_rows,
        "prior_violation_messages": violations_before,
        "repaired_by_class": dict(class_counts),
        "answer_counts_after_repair": dict(answer_counts),
        "provenance_validation": f"{len(rows)}/{len(rows)} passed",
        "repaired_indices": repaired_indices,
        "input_sha256": sha256(input_path),
    }

    print(json.dumps({key: value for key, value in report.items() if key != "repaired_indices"}, indent=2))
    if args.dry_run:
        return

    atomic_write_jsonl(output_path, rows)
    report["output_sha256"] = sha256(output_path)
    if report_path:
        atomic_write_json(report_path, report)
    print(f"[write] {len(rows)} rows -> {output_path}")
    if report_path:
        print(f"[write] report -> {report_path}")


if __name__ == "__main__":
    main()
