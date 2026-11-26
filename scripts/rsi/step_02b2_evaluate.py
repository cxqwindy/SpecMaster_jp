#!/usr/bin/env python3
"""CLI for the step 02B-2 offline evaluator.

The command reads dataset, gold, run, and prediction files that were already
written. It does not start a model, a predictor, or a training job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matching_evaluator import (
    EvaluatorError,
    PromotionRefused,
    compare_documents,
    dump_json,
    evaluate_documents,
    load_json,
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


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate_documents(
        load_json(Path(args.dataset)),
        Path(args.dataset),
        load_json(Path(args.gold)),
        load_json(Path(args.run)),
        load_json(Path(args.predictions)),
        load_json(Path(args.config)),
        explicit_group_draw=_split_draw(args.explicit_group_draw),
        legacy=args.legacy_diagnostic,
    )
    dump_json(Path(args.out), report)
    print(_status_line(report))
    return _exit_code(report)


def _compare(args: argparse.Namespace) -> int:
    report = compare_documents(
        load_json(Path(args.dataset)),
        Path(args.dataset),
        load_json(Path(args.gold)),
        load_json(Path(args.run_a)),
        load_json(Path(args.predictions_a)),
        load_json(Path(args.run_b)),
        load_json(Path(args.predictions_b)),
        load_json(Path(args.config)),
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
