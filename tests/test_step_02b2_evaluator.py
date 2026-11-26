"""Independent checks for the offline evaluator.

Expected fractions are read from the hand table or written as literals. They
are not produced by calling the metric function a second time.
"""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import matching_evaluator as ev  # noqa: E402
from step_02b2_fixture import scoring_config, write_bundle, write_hand_fixture  # noqa: E402

EXPECTED = json.loads((REPO / "docs" / "rsi" / "step_02b2" / "expected_hand.json").read_text(encoding="utf-8"))
PYTHON = sys.executable


def _ratio(pack, expected):
    assert pack["numerator"] == expected["numerator"]
    assert pack["denominator"] == expected["denominator"]
    assert pack["reason"] is None


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [PYTHON, "-B", "scripts/rsi/step_02b2_evaluate.py", *argv],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _cli_evaluate(bundle: dict, run_index: int, out: Path, *, draw: str | None = None, legacy: bool = False):
    run = bundle["runs"][run_index]
    argv = [
        "evaluate",
        "--dataset", bundle["dataset"],
        "--gold", bundle["gold"],
        "--run", run["run"],
        "--predictions", run["predictions"],
        "--config", bundle["config"],
        "--out", str(out),
    ]
    if draw is not None:
        argv.extend(["--explicit-group-draw", draw])
    if legacy:
        argv.append("--legacy-diagnostic")
    return _run(argv)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_no_nonfinite(path: Path):
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)


def test_import_does_not_load_a_model_or_open_a_socket(monkeypatch):
    for name in sys.modules:
        assert name.split(".")[0] not in {"torch", "transformers", "verl"}
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network")))
    report = ev.classification_metrics([])
    assert report["status"] == "no_valid_predictions"


def test_hand_oracle_and_group_draw_go_through_the_cli(tmp_path: Path):
    bundle = write_hand_fixture(tmp_path / "hand")
    aliases = bundle["aliases"]
    assert aliases["gA"] != aliases["gB"]
    out = tmp_path / "report_a.json"
    draw = ",".join([aliases["gA"], aliases["gA"], aliases["gD"], aliases["gD"]])
    result = _cli_evaluate(bundle, 0, out, draw=draw)
    assert result.returncode == 0, result.stderr
    report = _read(out)
    _assert_no_nonfinite(out)
    expected = EXPECTED["model_a"]
    metrics = report["metrics"]
    assert report["primary_metric"] == "balanced_accuracy_valid"
    assert report["invalid_policy"] == "exclude_from_primary"
    assert report["formal_primary_published"] is True
    assert report["promotion"]["status"] == "disabled_pending_protocol"
    assert report["promotion"]["model_pointer_changed"] is False
    assert [metrics["N"], metrics["V"], metrics["I"]] == [expected["N"], expected["V"], expected["I"]]
    _ratio(metrics["coverage"], expected["coverage"])
    _ratio(metrics["invalid_rate"], expected["invalid_rate"])
    _ratio(metrics["by_label"]["0"]["coverage"], expected["label_0"]["coverage"])
    _ratio(metrics["by_label"]["1"]["coverage"], expected["label_1"]["coverage"])
    assert metrics["by_label"]["0"]["V"] == 5
    assert metrics["by_label"]["1"]["V"] == 4
    assert metrics["confusion_matrix_valid"] == expected["confusion_matrix_valid"]
    assert metrics["confusion_matrix_valid_sum"] == 9
    assert metrics["gold_by_outcome"] == expected["gold_by_outcome"]
    assert metrics["gold_by_outcome_sum"] == 12
    _ratio(metrics["recall_no_valid"], expected["recall_no_valid"])
    _ratio(metrics["recall_yes_valid"], expected["recall_yes_valid"])
    _ratio(metrics["balanced_accuracy_valid"], expected["balanced_accuracy_valid"])
    _ratio(metrics["accuracy_valid"], expected["accuracy_valid"])
    _ratio(metrics["macro_f1_valid"], expected["macro_f1_valid"])
    _ratio(report["primary_value"], expected["balanced_accuracy_valid"])
    drawn = report["explicit_group_draw"]["metrics"]
    group_expected = EXPECTED["group_draw_gA_gA_gD_gD"]
    assert [drawn["N"], drawn["V"]] == [group_expected["N"], group_expected["V"]]
    assert drawn["confusion_matrix_valid"] == group_expected["confusion_matrix_valid"]
    _ratio(drawn["balanced_accuracy_valid"], group_expected["balanced_accuracy_valid"])
    _ratio(drawn["accuracy_valid"], group_expected["accuracy_valid"])
    _ratio(drawn["macro_f1_valid"], group_expected["macro_f1_valid"])
    _ratio(drawn["coverage"], group_expected["coverage"])
    assert report["explicit_group_draw"]["weights"][aliases["gA"]] == 2
    assert report["explicit_group_draw"]["weights"][aliases["gD"]] == 2
    row3 = next(row for row in report["rows"] if row["record_id"] == "3")
    assert row3["source_group_id"] == aliases["gA"]
    assert row3["source_group_id"] != aliases["gB"]
    assert row3["model_observed_task_sha256"] is None
    assert row3["model_observation_compared"] is False
    assert report["groups"]["difficulty"]["unknown"]["N"] == 12
    assert report["groups"]["ambiguity"]["unannotated"]["N"] == 12
    cost = report["cost"]["tool_calls"]
    assert cost["known"] == 2
    assert cost["total_attempts"] == 12
    assert cost["known_sum"] == 3
    assert cost["complete"] is False
    assert cost["total_cost"] is None
    assert report["cost"]["tokens"]["known_sum"] is None
    assert report["cost"]["tokens"]["availability"] == "not_recorded"
    invalid = {row["record_id"]: row["invalid_reason"] for row in report["rows"] if row["prediction_status"] == "invalid"}
    assert invalid == {"6": "empty_answer", "10": "timeout", "11": "parse_error"}


def test_compare_reports_both_denominators_and_stays_unpromoted(tmp_path: Path):
    bundle = write_hand_fixture(tmp_path / "hand")
    out = tmp_path / "compare.json"
    result = _run([
        "compare",
        "--dataset", bundle["dataset"],
        "--gold", bundle["gold"],
        "--run-a", bundle["runs"][0]["run"],
        "--predictions-a", bundle["runs"][0]["predictions"],
        "--run-b", bundle["runs"][1]["run"],
        "--predictions-b", bundle["runs"][1]["predictions"],
        "--config", bundle["config"],
        "--out", str(out),
    ])
    assert result.returncode == 0, result.stderr
    report = _read(out)
    _assert_no_nonfinite(out)
    expected = EXPECTED["compare"]
    _ratio(report["own_valid"]["B"]["balanced_accuracy_valid"], EXPECTED["model_b"]["balanced_accuracy_valid"])
    _ratio(report["own_valid"]["B"]["coverage"], EXPECTED["model_b"]["coverage"])
    _ratio(report["own_valid"]["delta_ba_own_valid"], expected["delta_ba_own_valid"])
    _ratio(report["own_valid"]["delta_coverage"], expected["delta_coverage"])
    _ratio(report["own_valid"]["delta_coverage_1"], expected["delta_coverage_1"])
    assert report["own_valid"]["populations_may_differ"] is True
    assert report["common_valid"]["N"] == expected["common_n"]
    _ratio(report["common_valid"]["A_balanced_accuracy_valid"], expected["common_ba"])
    _ratio(report["common_valid"]["B_balanced_accuracy_valid"], expected["common_ba"])
    _ratio(report["common_valid"]["delta_ba_common_valid"], expected["delta_ba_common_valid"])
    assert report["coverage_table"] == {**expected["coverage_table"], "sum": 12}
    assert report["promotion"]["status"] == "disabled_pending_protocol"
    assert report["conclusion"] == "comparison_only_promotion_disabled"
    assert "promoted" not in json.dumps(report)
    assert report["run_id_a"] != report["run_id_b"]


def test_shuffled_files_keep_the_same_score(tmp_path: Path):
    bundle = write_hand_fixture(tmp_path / "hand")
    first = tmp_path / "first.json"
    assert _cli_evaluate(bundle, 0, first).returncode == 0
    dataset = _read(Path(bundle["dataset"]))
    predictions = _read(Path(bundle["runs"][0]["predictions"]))
    dataset["records"].reverse()
    predictions["predictions"].reverse()
    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    for name in ("dataset_manifest.json", "gold_sidecar.json", "config.json", "model_a_run_manifest.json"):
        source = tmp_path / "hand" / name
        (shuffled / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (shuffled / "dataset_manifest.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (shuffled / "model_a_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    second = tmp_path / "second.json"
    result = _run([
        "evaluate",
        "--dataset", str(shuffled / "dataset_manifest.json"),
        "--gold", str(shuffled / "gold_sidecar.json"),
        "--run", str(shuffled / "model_a_run_manifest.json"),
        "--predictions", str(shuffled / "model_a_predictions.json"),
        "--config", str(shuffled / "config.json"),
        "--out", str(second),
    ])
    assert result.returncode == 0, result.stderr
    assert _read(second)["metrics"]["balanced_accuracy_valid"] == _read(first)["metrics"]["balanced_accuracy_valid"]
    assert _read(second)["dataset_content_sha256"] == _read(first)["dataset_content_sha256"]


def _tiny(tmp_path: Path, records, predictions, **kwargs):
    return write_bundle(
        tmp_path,
        dataset_id=kwargs.get("dataset_id", "synthetic-tiny"),
        records=records,
        runs=[{
            "stem": "only",
            "run_id": kwargs.get("run_id", "synthetic-tiny-run"),
            "evaluated_version": "synthetic-tiny-model",
            "profile": kwargs.get("profile", "canonical"),
            "predictions": predictions,
        }],
        config=kwargs.get("config"),
    )


def _record(record_id, parent, gold, candidate=None):
    return {
        "record_id": record_id,
        "parent_gt_smiles": parent,
        "gold_label": gold,
        "candidate_smiles": candidate or parent,
    }


def _answer(label):
    return {"terminal": {"state": "answer", "final_answer": "Yes" if label == 1 else "No"}}


def test_class_edges_and_empty_manifest(tmp_path: Path):
    records = [
        _record("1", "CCO", 0, "c1ccccc1"),
        _record("2", "CCCO", 1),
        _record("3", "CCCCO", 1),
        _record("4", "CC(C)O", 1),
    ]
    predictions = [_answer(0), _answer(1), _answer(1), _answer(0)]
    bundle = _tiny(tmp_path / "imbalance", records, predictions)
    out = tmp_path / "imbalance.json"
    assert _cli_evaluate(bundle, 0, out).returncode == 0
    metrics = _read(out)["metrics"]
    _ratio(metrics["balanced_accuracy_valid"], {"numerator": 5, "denominator": 6})
    _ratio(metrics["recall_yes_valid"], {"numerator": 2, "denominator": 3})
    _ratio(metrics["recall_no_valid"], {"numerator": 1, "denominator": 1})

    missing_class = _tiny(
        tmp_path / "one-class",
        [_record("1", "CCO", 0, "c1ccccc1"), _record("2", "CCCO", 1)],
        [_answer(0), {"terminal": {"state": "timeout", "final_answer": None}}],
    )
    out = tmp_path / "one-class.json"
    assert _cli_evaluate(missing_class, 0, out).returncode == 0
    metrics = _read(out)["metrics"]
    assert metrics["status"] == "missing_valid_class"
    assert metrics["balanced_accuracy_valid"]["reason"] == "missing_valid_class"
    assert metrics["macro_f1_valid"]["reason"] == "missing_valid_class"
    assert metrics["accuracy_valid"]["fraction"] == "1/1"
    assert metrics["recall_no_valid"]["fraction"] == "1/1"
    assert metrics["recall_yes_valid"]["reason"] == "zero_valid_class"

    all_invalid = _tiny(
        tmp_path / "none",
        [_record("1", "CCO", 1), _record("2", "CCCO", 0, "c1ccccc1")],
        [{"terminal": {"state": "timeout", "final_answer": None}}, {"terminal": {"state": "timeout", "final_answer": None}}],
    )
    out = tmp_path / "none.json"
    assert _cli_evaluate(all_invalid, 0, out).returncode == 0
    metrics = _read(out)["metrics"]
    assert metrics["status"] == "no_valid_predictions"
    assert metrics["V"] == 0
    assert metrics["coverage"]["fraction"] == "0/1"
    assert metrics["invalid_rate"]["fraction"] == "1/1"
    assert metrics["balanced_accuracy_valid"]["numerator"] is None
    assert metrics["confusion_matrix_valid_sum"] == 0
    assert metrics["gold_by_outcome_sum"] == 2

    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = {
        "schema_version": ev.DATASET_SCHEMA,
        "dataset_id": "empty",
        "source_group_definition": {"id": ev.SOURCE_GROUP_ID, "version": ev.SOURCE_GROUP_VERSION},
        "records": [],
    }
    ev.dump_json(empty / "dataset_manifest.json", manifest)
    prepared, errors = ev.prepare_dataset(manifest, empty / "dataset_manifest.json", verify_hash=False)
    assert prepared is None
    assert errors[0]["code"] == "empty_evaluation_set"


def test_yes_no_parser_rejects_loose_and_partial_answers():
    assert ev.parse_answer_container({"final_answer": " Yes "})[1] == 1
    assert ev.parse_answer_container('{"final_answer":"no"}')[1] == 0
    think = '<think>The tool said No</think><answer>{"final_answer":"Yes","key_evidence":"No"}</answer>'
    assert ev.parse_answer_container(think)[1] == 1
    for value in (1, 0, True, False, 2, -1, 0.7, "0", "1"):
        status, label, reason = ev.parse_answer_container({"final_answer": value})
        assert status == "invalid"
        assert label is None
        assert reason == "non_binary_answer"
    truncated = ev.parse_answer_container('<answer>{"final_answer": "Yes"</answer>')
    assert truncated[2] == "parse_error"
    multiple = ev.parse_answer_container(
        '<answer>{"final_answer":"Yes"}</answer><answer>{"final_answer":"Yes"}</answer>'
    )
    assert multiple[2] == "multiple_answers"
    prose = ev.parse_answer_container("I conclude Yes because the tool said No")
    assert prose[0] == "invalid"
    assert prose[1] is None
    evidence_only = ev.parse_answer_container({"key_evidence": "Yes", "confidence": 0.9})
    assert evidence_only[2] == "empty_answer"


def test_production_adapters_and_terminal_conflicts(tmp_path: Path):
    standalone = ev.parse_standalone_row({
        "answer": '<answer>{"final_answer":"No","confidence":0.2,"key_evidence":"Yes"}</answer>',
        "fail_reason": None,
        "trace": [{"type": "tool", "name": "count_atoms", "error": "x"}, {"type": "answer"}],
    })
    assert standalone["valid"] is True
    assert standalone["pred_label"] == 0
    assert standalone["diagnostic"]["tool_error_before_answer"] is True
    conflict = ev.parse_standalone_row({
        "answer": {"final_answer": "Yes"},
        "fail_reason": "timeout",
    })
    assert conflict["invalid_reason"] == "terminal_conflict"
    assert conflict["valid"] is False
    api = ev.parse_api_response({
        "final_answer": "Yes",
        "final_think": "maybe No",
        "key_evidence": "No",
        "trajectory": [{"turn": 1, "think": "Yes?", "final_answer": "Yes"}],
    })
    assert api["pred_label"] == 1
    disagree = ev.parse_api_response({
        "final_answer": "Yes",
        "trajectory": [{"final_answer": "No"}],
    })
    assert disagree["invalid_reason"] == "terminal_conflict"
    records = [_record("1", "CCO", 1)]
    bundle = _tiny(
        tmp_path / "api",
        records,
        [{"response": {"final_answer": "Yes", "key_evidence": "No", "trajectory": []}}],
        profile="api",
    )
    out = tmp_path / "api.json"
    assert _cli_evaluate(bundle, 0, out).returncode == 0
    assert _read(out)["rows"][0]["pred_label"] == 1
    bundle = _tiny(
        tmp_path / "standalone",
        records,
        [{"answer": '{"final_answer": "Yes"}', "fail_reason": None}],
        profile="standalone",
    )
    out = tmp_path / "standalone.json"
    assert _cli_evaluate(bundle, 0, out).returncode == 0
    assert _read(out)["rows"][0]["prediction_status"] == "valid"


def _break(bundle_dir: Path, mutate):
    dataset = bundle_dir / "dataset_manifest.json"
    gold = bundle_dir / "gold_sidecar.json"
    run = bundle_dir / "only_run_manifest.json"
    predictions = bundle_dir / "only_predictions.json"
    mutate(_read(dataset), _read(gold), _read(run), _read(predictions), dataset, gold, run, predictions)


def test_integrity_failures_do_not_publish_a_primary_score(tmp_path: Path):
    records = [_record("1", "CCO", 1), _record("2", "CCCO", 0, "c1ccccc1")]
    predictions = [_answer(1), _answer(0)]
    bundle = _tiny(tmp_path / "base", records, predictions)
    cases = {}

    def missing(dataset, gold, run, preds, *paths):
        preds["predictions"] = preds["predictions"][:1]
        ev.dump_json(paths[3], preds)
    cases["missing"] = missing

    def extra(dataset, gold, run, preds, *paths):
        preds["predictions"].append({"pair_id": 99, "run_id": run["run_id"], "terminal": {"state": "timeout", "final_answer": None}})
        ev.dump_json(paths[3], preds)
    cases["extra"] = extra

    def duplicate(dataset, gold, run, preds, *paths):
        preds["predictions"].append(dict(preds["predictions"][0]))
        ev.dump_json(paths[3], preds)
    cases["duplicate"] = duplicate

    def mixed(dataset, gold, run, preds, *paths):
        preds["predictions"][1]["run_id"] = "other-run"
        ev.dump_json(paths[3], preds)
    cases["mixed"] = mixed

    def bad_gold(dataset, gold, run, preds, *paths):
        gold["records"][0]["gold_label"] = 0
        ev.dump_json(paths[1], gold)
    cases["bad_gold"] = bad_gold

    def bad_hash(dataset, gold, run, preds, *paths):
        dataset["content_sha256"] = "0" * 64
        ev.dump_json(paths[0], dataset)
    cases["bad_hash"] = bad_hash

    def bad_task(dataset, gold, run, preds, *paths):
        dataset["records"][0]["task"]["c_nmr_peaks"] = {}
        ev.dump_json(paths[0], dataset)
    cases["bad_task"] = bad_task

    def prediction_gold(dataset, gold, run, preds, *paths):
        preds["predictions"][0]["_label"] = 0
        ev.dump_json(paths[3], preds)
    cases["prediction_gold"] = prediction_gold

    for name, mutate in cases.items():
        case_dir = tmp_path / name
        copied = _tiny(case_dir, records, predictions, dataset_id=f"synthetic-{name}")
        _break(case_dir, mutate)
        out = tmp_path / f"{name}.json"
        result = _cli_evaluate(copied, 0, out)
        assert result.returncode == 2, name
        report = _read(out)
        assert report["metrics"] is None
        assert report["primary_value"] is None
        assert report["formal_primary_published"] is False
        assert "timeout" not in json.dumps(report["integrity"]["errors"]) or name == "extra"


def test_explicit_timeout_is_coverage_not_a_missing_row(tmp_path: Path):
    bundle = _tiny(
        tmp_path / "timeout",
        [_record("1", "CCO", 1), _record("2", "CCCO", 0, "c1ccccc1")],
        [
            {"terminal": {"state": "timeout", "final_answer": None}},
            _answer(0),
        ],
    )
    out = tmp_path / "timeout.json"
    assert _cli_evaluate(bundle, 0, out).returncode == 0
    report = _read(out)
    assert report["metrics"]["I"] == 1
    assert report["metrics"]["coverage"]["fraction"] == "1/2"
    assert report["rows"][0]["invalid_reason"] == "timeout"


def test_common_valid_does_not_borrow_the_own_valid_score(tmp_path: Path):
    records = [
        _record("1", "CCO", 1),
        _record("2", "CCCO", 1),
        _record("3", "CCCCO", 0, "c1ccccc1"),
        _record("4", "CC(C)O", 0, "c1ccccc1"),
    ]
    invalid = {"terminal": {"state": "timeout", "final_answer": None}}
    bundle = write_bundle(
        tmp_path / "disjoint",
        dataset_id="synthetic-disjoint",
        records=records,
        runs=[
            {
                "stem": "a",
                "run_id": "run-a",
                "evaluated_version": "synthetic-a",
                "profile": "canonical",
                "predictions": [_answer(1), invalid, _answer(1), invalid],
            },
            {
                "stem": "b",
                "run_id": "run-b",
                "evaluated_version": "synthetic-b",
                "profile": "canonical",
                "predictions": [invalid, _answer(1), invalid, _answer(0)],
            },
        ],
    )
    out = tmp_path / "disjoint.json"
    result = _run([
        "compare", "--dataset", bundle["dataset"], "--gold", bundle["gold"],
        "--run-a", bundle["runs"][0]["run"], "--predictions-a", bundle["runs"][0]["predictions"],
        "--run-b", bundle["runs"][1]["run"], "--predictions-b", bundle["runs"][1]["predictions"],
        "--config", bundle["config"], "--out", str(out),
    ])
    assert result.returncode == 0, result.stderr
    report = _read(out)
    assert report["common_valid"]["delta_ba_common_valid"]["reason"] == "empty_common_valid"
    assert report["common_valid"]["A_balanced_accuracy_valid"]["numerator"] is None
    _ratio(report["own_valid"]["delta_ba_own_valid"], {"numerator": 1, "denominator": 2})
    assert report["coverage_table"] == {"both_valid": 0, "A_only_valid": 2, "B_only_valid": 2, "neither_valid": 0, "sum": 4}


def test_bootstrap_is_reproducible_shared_and_does_not_invent_intervals(tmp_path: Path):
    records = [
        _record("1", "CCO", 1),
        _record("2", "CCO", 0, "c1ccccc1"),
        _record("3", "CCCO", 1),
        _record("4", "CCCO", 0, "c1ccccc1"),
    ]
    predictions = [_answer(1), _answer(0), _answer(1), _answer(0)]
    config = scoring_config({
        "enabled": True,
        "method": "paired_cluster_percentile",
        "bootstrap_replicates": 6,
        "confidence_level": 0.5,
        "seed": 11,
    })
    bundle = write_bundle(
        tmp_path / "boot",
        dataset_id="synthetic-boot",
        records=records,
        runs=[
            {"stem": "a", "run_id": "boot-a", "evaluated_version": "synthetic-a", "profile": "canonical", "predictions": predictions},
            {"stem": "b", "run_id": "boot-b", "evaluated_version": "synthetic-b", "profile": "canonical", "predictions": predictions},
        ],
        config=config,
    )
    out = tmp_path / "boot.json"
    result = _run([
        "compare", "--dataset", bundle["dataset"], "--gold", bundle["gold"],
        "--run-a", bundle["runs"][0]["run"], "--predictions-a", bundle["runs"][0]["predictions"],
        "--run-b", bundle["runs"][1]["run"], "--predictions-b", bundle["runs"][1]["predictions"],
        "--config", bundle["config"], "--out", str(out),
    ])
    assert result.returncode == 0, result.stderr
    report = _read(out)
    draws = report["statistics"]["shared_group_index_draws"]
    expected_draws = ev.group_draw_indices(2, 6, 11)
    assert draws == expected_draws
    own = report["statistics"]["delta_ba_own_valid"]
    common = report["statistics"]["delta_ba_common_valid"]
    assert own["interval"]["constant_zero"] is True
    assert own["interval"]["low"]["numerator"] == 0
    assert own["interval"]["high"]["numerator"] == 0
    assert common["interval"]["constant_zero"] is True
    assert "p_value" not in json.dumps(report["statistics"])
    assert own["formal_threshold"] is False

    shuffled = copy.deepcopy(_read(Path(bundle["runs"][0]["predictions"])))
    shuffled["predictions"].reverse()
    shuffled_path = tmp_path / "boot" / "shuffled_predictions.json"
    ev.dump_json(shuffled_path, shuffled)
    out2 = tmp_path / "boot2.json"
    result = _run([
        "compare", "--dataset", bundle["dataset"], "--gold", bundle["gold"],
        "--run-a", bundle["runs"][0]["run"], "--predictions-a", str(shuffled_path),
        "--run-b", bundle["runs"][1]["run"], "--predictions-b", bundle["runs"][1]["predictions"],
        "--config", bundle["config"], "--out", str(out2),
    ])
    assert result.returncode == 0, result.stderr
    assert _read(out2)["statistics"]["shared_group_index_draws"] == draws

    pure = write_bundle(
        tmp_path / "pure",
        dataset_id="synthetic-pure",
        records=[_record("1", "CCO", 0, "c1ccccc1"), _record("2", "CCCO", 1)],
        runs=[{
            "stem": "only",
            "run_id": "pure-run",
            "evaluated_version": "synthetic-pure",
            "profile": "canonical",
            "predictions": [_answer(0), _answer(1)],
        }],
        config=scoring_config({
            "enabled": True,
            "method": "paired_cluster_percentile",
            "bootstrap_replicates": 8,
            "confidence_level": 0.5,
            "seed": 0,
        }),
    )
    out = tmp_path / "pure.json"
    assert _cli_evaluate(pure, 0, out).returncode == 0
    stat = _read(out)["statistics"]["balanced_accuracy_valid"]
    mono = sum(len(set(draw)) == 1 for draw in ev.group_draw_indices(2, 8, 0))
    assert mono > 0
    assert stat["replicates_undefined"] == mono
    assert stat["interval"] is None
    assert stat["reason"] == "undefined_replicate"

    one = _tiny(
        tmp_path / "one-group",
        [_record("1", "CCO", 1), _record("2", "CCO", 0, "c1ccccc1")],
        [_answer(1), _answer(0)],
        config=scoring_config({
            "enabled": True,
            "method": "paired_cluster_percentile",
            "bootstrap_replicates": 4,
            "confidence_level": 0.5,
            "seed": 3,
        }),
    )
    out = tmp_path / "one-group.json"
    assert _cli_evaluate(one, 0, out).returncode == 0
    stat = _read(out)["statistics"]["balanced_accuracy_valid"]
    assert stat["reason"] == "fewer_than_two_groups"
    assert stat["interval"] is None
    assert stat["group_index_draws"] == []


def test_trace_cost_counts_calls_without_filling_unknowns():
    trace = [
        {"type": "tool", "name": "count_atoms", "cache": "hit"},
        {"type": "tool", "name": "count_atoms", "cache": "network"},
        {"type": "answer"},
    ]
    counted = ev.cost_from_trace(trace)
    assert counted["tool_calls"] == 2
    assert counted["tool_cache_hits"] == 1
    assert counted["tool_network_calls"] == 1
    unknown = ev.cost_from_trace([
        {"type": "tool", "name": "count_atoms"},
        {"type": "tool", "name": "count_atoms", "cache": "hit"},
    ])
    assert unknown["tool_calls"] == 2
    assert unknown["tool_cache_hits"] is None
    assert unknown["tool_network_calls"] is None


def test_promotion_cannot_be_enabled(tmp_path: Path):
    bundle = write_hand_fixture(tmp_path / "hand")
    config = _read(Path(bundle["config"]))
    config["promotion"]["enabled"] = True
    path = tmp_path / "promote.json"
    ev.dump_json(path, config)
    out = tmp_path / "out.json"
    result = _run([
        "evaluate", "--dataset", bundle["dataset"], "--gold", bundle["gold"],
        "--run", bundle["runs"][0]["run"], "--predictions", bundle["runs"][0]["predictions"],
        "--config", str(path), "--out", str(out),
    ])
    assert result.returncode == 3
    assert "promotion_refused" in result.stderr
    assert not out.exists()


def test_legacy_diagnostic_is_not_a_formal_comparison(tmp_path: Path):
    records = [_record("1", "CCO", 1), _record("2", "CCCO", 0, "c1ccccc1")]
    bundle = _tiny(tmp_path / "legacy", records, [_answer(1), _answer(0)])
    predictions = _read(Path(bundle["runs"][0]["predictions"]))
    predictions["profile"] = "legacy_diagnostic"
    for row in predictions["predictions"]:
        row.pop("pair_id", None)
        row.pop("run_id", None)
    ev.dump_json(Path(bundle["runs"][0]["predictions"]), predictions)
    out = tmp_path / "legacy.json"
    result = _cli_evaluate(bundle, 0, out, legacy=True)
    assert result.returncode == 0, result.stderr
    report = _read(out)
    assert report["pairing_verified"] is False
    assert report["formal_primary_published"] is False
    assert report["primary_value"] is None
    assert report["integrity"]["status"] == "legacy_diagnostic"
    assert report["diagnostic_metrics"]["N"] == 2
    strict = tmp_path / "strict.json"
    strict_result = _cli_evaluate(bundle, 0, strict, legacy=False)
    assert strict_result.returncode == 2
    assert _read(strict)["metrics"] is None


def test_task_contract_keeps_gold_out_and_rejects_bad_containers(tmp_path: Path):
    normalized, digest, errors = ev.validate_task({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [],
        "h_nmr_peaks": [{"delta": 1.2, "nH": None}],
        "_label": 1,
    })
    assert normalized is None
    assert errors[0]["code"] == "gold_in_task"
    normalized, digest, errors = ev.validate_task({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": [],
        "h_nmr_peaks": [{"delta": 1.2, "nH": None}],
    })
    assert errors == []
    assert normalized["c_nmr_peaks"] == []
    assert normalized["h_nmr_peaks"][0]["nH"] is None
    assert set(normalized) <= {
        "candidate_smiles", "c_nmr_peaks", "h_nmr_peaks", "nmr_solvent",
        "nmr_frequency", "nmr_frequency_nucleus", "experimental_total_nH",
    }
    bad, _digest, bad_errors = ev.validate_task({
        "candidate_smiles": "CCO",
        "c_nmr_peaks": {},
        "h_nmr_peaks": [{"delta": 1.2, "nH": 3}],
    })
    assert bad is None
    assert bad_errors[0]["code"] == "illegal_task"
    assert digest
