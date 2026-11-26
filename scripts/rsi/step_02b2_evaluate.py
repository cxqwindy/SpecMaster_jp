#!/usr/bin/env python3
"""CLI for the step 02B-2 offline evaluator.

The command reads dataset, gold, run, and prediction files that were already
written. It does not start a model, a predictor, or a training job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matching_evaluator import (
    DuplicateJsonKey,
    EvaluatorError,
    PromotionRefused,
    _failed_report,
    compare_documents,
    dump_json,
    evaluate_documents,
    load_config,
    loads_strict,
)


def _split_draw(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [item for item in value.split(",") if item != ""]


def _status_line(report: dict) -> str:
    integrity = (report.get("integrity") or {}).get("status")
    primary = report.get("primary_value") or {}
    fraction = primary.get("fraction") if isinstance(primary, dict) else None
    return f"integrity={integrity} formal_primary={report.get('formal_primary_published')} primary={fraction} promotion={report.get('promotion', {}).get('status')}"


def _exit_code(report: dict) -> int:
    status = (report.get("integrity") or {}).get("status")
    explicit = report.get("explicit_group_draw") or {}
    if explicit.get("errors"):
        return 2
    if status in {"passed", "legacy_diagnostic"}:
        return 0
    return 2


def _load_documents(pairs: list[tuple[str, str]]) -> tuple[dict, list[dict[str, str]]]:
    loaded = {}
    duplicates = []
    for role, path in pairs:
        try:
            loaded[role] = loads_strict(Path(path).read_text(encoding="utf-8"))
        except DuplicateJsonKey as exc:
            duplicates.append({"code": "duplicate_json_key", "detail": f"{role}: duplicate key {exc.key!r}"})
        except json.JSONDecodeError as exc:
            raise EvaluatorError("json_parse_error", f"{role}: {exc.msg}") from exc
    return loaded, duplicates


def _reject_duplicates(duplicates: list[dict[str, str]], loaded: dict, out: Path) -> int | None:
    if not duplicates:
        return None
    roles = {item["detail"].split(":", 1)[0] for item in duplicates}
    if roles == {"config"}:
        print(duplicates[0]["detail"], file=sys.stderr)
        return 4
    config = None
    if "config" in loaded:
        try:
            config = load_config(loaded["config"])
        except EvaluatorError:
            config = None
    report = _failed_report(duplicates, config)
    dump_json(out, report)
    print(_status_line(report))
    return 2


def _evaluate(args: argparse.Namespace) -> int:
    loaded, duplicates = _load_documents([
        ("dataset", args.dataset),
        ("gold", args.gold),
        ("run", args.run),
        ("predictions", args.predictions),
        ("config", args.config),
    ])
    rejected = _reject_duplicates(duplicates, loaded, Path(args.out))
    if rejected is not None:
        return rejected
    report = evaluate_documents(
        loaded["dataset"],
        Path(args.dataset),
        loaded["gold"],
        loaded["run"],
        loaded["predictions"],
        loaded["config"],
        explicit_group_draw=_split_draw(args.explicit_group_draw),
        legacy=args.legacy_diagnostic,
    )
    dump_json(Path(args.out), report)
    print(_status_line(report))
    return _exit_code(report)


def _compare(args: argparse.Namespace) -> int:
    loaded, duplicates = _load_documents([
        ("dataset", args.dataset),
        ("gold", args.gold),
        ("run_a", args.run_a),
        ("predictions_a", args.predictions_a),
        ("run_b", args.run_b),
        ("predictions_b", args.predictions_b),
        ("config", args.config),
    ])
    rejected = _reject_duplicates(duplicates, loaded, Path(args.out))
    if rejected is not None:
        return rejected
    report = compare_documents(
        loaded["dataset"],
        Path(args.dataset),
        loaded["gold"],
        loaded["run_a"],
        loaded["predictions_a"],
        loaded["run_b"],
        loaded["predictions_b"],
        loaded["config"],
    )
    dump_json(Path(args.out), report)
    integrity = (report.get("integrity") or {}).get("status")
    print(f"integrity={integrity} conclusion={report.get('conclusion')} promotion={report.get('promotion', {}).get('status')}")
    return 0 if integrity == "passed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="step_02b2_evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--explicit-group-draw", default=None)
    evaluate.add_argument("--legacy-diagnostic", action="store_true")
    evaluate.set_defaults(func=_evaluate)
    compare = sub.add_parser("compare")
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--gold", required=True)
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--predictions-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.add_argument("--predictions-b", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PromotionRefused as exc:
        print(f"promotion_refused: {exc.detail}", file=sys.stderr)
        return 3
    except EvaluatorError as exc:
        print(f"{exc.code}: {exc.detail}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
