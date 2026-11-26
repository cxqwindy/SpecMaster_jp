"""E1-E3 regressions for the offline evaluator.

Expected counts are handwritten. They are not produced by a second metric function.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import matching_evaluator as ev  # noqa: E402
from step_02b2_fixture import rebind_run_manifest, scoring_config, write_bundle, write_hand_fixture  # noqa: E402

PYTHON = sys.executable


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


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(record_id, parent, gold, candidate=None):
    return {
        "record_id": record_id,
        "parent_gt_smiles": parent,
        "gold_label": gold,
        "candidate_smiles": candidate or parent,
    }


def _answer(label):
    return {"terminal": {"state": "answer", "final_answer": "Yes" if label == 1 else "No"}}


def _bundle(directory: Path, records, predictions, **kwargs):
    runs = kwargs.get("runs")
    if runs is None:
        runs = [{
            "stem": "only",
            "run_id": kwargs.get("run_id", "synthetic-fix-run"),
            "evaluated_version": kwargs.get("evaluated_version", "synthetic-fix-model"),
            "profile": kwargs.get("profile", "canonical"),
            "predictions": predictions,
        }]
    return write_bundle(
        directory,
        dataset_id=kwargs.get("dataset_id", "synthetic-fix"),
        records=records,
        runs=runs,
        config=kwargs.get("config"),
    )


def _evaluate(bundle, out: Path, *, config=None, run=None, predictions=None, legacy=False):
    argv = [
        "evaluate",
        "--dataset", bundle["dataset"],
        "--gold", bundle["gold"],
        "--run", run or bundle["runs"][0]["run"],
        "--predictions", predictions or bundle["runs"][0]["predictions"],
        "--config", config or bundle["config"],
        "--out", str(out),
    ]
    if legacy:
        argv.append("--legacy-diagnostic")
    return _run(argv)


def _compare(bundle, out: Path, *, config=None, run_a=None, predictions_a=None, run_b=None, predictions_b=None):
    return _run([
        "compare",
        "--dataset", bundle["dataset"],
        "--gold", bundle["gold"],
        "--run-a", run_a or bundle["runs"][0]["run"],
        "--predictions-a", predictions_a or bundle["runs"][0]["predictions"],
        "--run-b", run_b or bundle["runs"][-1]["run"],
        "--predictions-b", predictions_b or bundle["runs"][-1]["predictions"],
        "--config", config or bundle["config"],
        "--out", str(out),
    ])


def _codes(report: dict) -> list[str]:
    return [item["code"] for item in report["integrity"]["errors"]]


def _assert_unpublished(result, report: dict):
    assert result.returncode == 2, result.stderr
    assert report["metrics"] is None
    assert report["primary_value"] is None
    assert report["formal_primary_published"] is False
    assert report["promotion"]["status"] == "disabled_pending_protocol"


def _two_records():
    return [
        _record("1", "CCO", 1),
        _record("2", "CCCO", 0, "c1ccccc1"),
    ]


def test_duplicate_keys_are_rejected_before_dict_collapse():
    with pytest.raises(ev.DuplicateJsonKey) as caught:
        ev.loads_strict('{"final_answer":"Yes","final_answer":"Yes"}')
    assert caught.value.key == "final_answer"
    status, label, reason = ev.parse_answer_container('{"final_answer":"No","final_answer":"Yes"}')
    assert status == "invalid"
    assert label is None
    assert reason == "duplicate_json_key"
    unclosed = ev.parse_answer_container('<answer>{"final_answer":"Yes"}</answer><answer>{"final_answer":"No"')
    assert unclosed[1] is None
    assert unclosed[2] == "multiple_answers"
    state = ev.parse_canonical_terminal('{"state":"timeout","state":"answer","final_answer":"Yes"}')
    assert state["valid"] is False
    assert state["pred_label"] is None
    assert state["invalid_reason"] == "duplicate_json_key"
    for value in (0, 1, True, False, "0", "1"):
        assert ev.parse_answer_container({"final_answer": value})[2] == "non_binary_answer"
    bare = ev.parse_answer_container({"final_answer": "Yes"})
    assert bare[0] == "valid"
    assert bare[1] == 1


def test_e1_production_identity_scores_and_aliases_must_agree(tmp_path: Path):
    records = _two_records()
    bundle = _bundle(
        tmp_path / "prod",
        records,
        [
            {"answer": '{"final_answer":"Yes"}', "fail_reason": None},
            {"answer": '{"final_answer":"No"}', "fail_reason": None},
        ],
        profile="standalone",
        dataset_id="synthetic-e1-prod",
    )
    path = Path(bundle["runs"][0]["predictions"])
    document = _read(path)
    document.pop("run_id")
    for index, row in enumerate(document["predictions"], start=1):
        row.pop("pair_id")
        row.pop("run_id")
        row["_pair_id"] = index
        row["_harness_run_id"] = "synthetic-fix-run"
    ev.dump_json(path, document)
    out = tmp_path / "prod.json"
    result = _evaluate(bundle, out)
    assert result.returncode == 0, result.stderr
    report = _read(out)
    assert report["formal_primary_published"] is True
    assert report["metrics"]["balanced_accuracy_valid"]["fraction"] == "1/1"
    assert report["run_binding"]["identity_verified"] is True
    assert report["rows"][0]["pred_label"] == 1
    assert report["rows"][1]["pred_label"] == 0

    document["predictions"].reverse()
    ev.dump_json(path, document)
    shuffled = tmp_path / "shuffled.json"
    assert _evaluate(bundle, shuffled).returncode == 0
    assert _read(shuffled)["metrics"]["balanced_accuracy_valid"] == report["metrics"]["balanced_accuracy_valid"]

    agreed = _read(path)
    agreed["run_id"] = "synthetic-fix-run"
    for row in agreed["predictions"]:
        row["pair_id"] = row["_pair_id"]
        row["run_id"] = row["_harness_run_id"]
    ev.dump_json(path, agreed)
    agreed_out = tmp_path / "agreed.json"
    assert _evaluate(bundle, agreed_out).returncode == 0
    assert _read(agreed_out)["metrics"]["balanced_accuracy_valid"]["fraction"] == "1/1"

    zero = _bundle(
        tmp_path / "zero",
        records,
        [{"pair_id": 0, **_answer(1)}, {"pair_id": 5, **_answer(0)}],
        dataset_id="synthetic-e1-zero",
    )
    zero_out = tmp_path / "zero.json"
    assert _evaluate(zero, zero_out).returncode == 0
    assert {row["pair_id"] for row in _read(zero_out)["rows"]} == {"0", "5"}


def test_e1_conflicting_or_illegal_identity_is_not_a_model_invalid(tmp_path: Path):
    records = _two_records()
    cases = {}

    def save(name, mutate):
        directory = tmp_path / name
        bundle = _bundle(directory, records, [_answer(1), _answer(0)], dataset_id=f"synthetic-e1-{name}")
        document = _read(Path(bundle["runs"][0]["predictions"]))
        mutate(document)
        ev.dump_json(Path(bundle["runs"][0]["predictions"]), document)
        cases[name] = bundle

    def conflict(document):
        document["predictions"][0]["_pair_id"] = 2

    def illegal_alias(document):
        document["predictions"][0]["_pair_id"] = True

    def negative(document):
        document["predictions"][0]["pair_id"] = -1

    def harness_other(document):
        document["predictions"][0]["_harness_run_id"] = "other-run"

    def harness_null(document):
        document["predictions"][0]["_harness_run_id"] = None

    def bad_envelope(document):
        document["run_id"] = "other-run"

    def null_envelope(document):
        document["run_id"] = None

    def guessed_column(document):
        row = document["predictions"][0]
        row.pop("pair_id")
        row["_harness_pair_id"] = 1

    save("conflict", conflict)
    save("illegal", illegal_alias)
    save("negative", negative)
    save("harness_other", harness_other)
    save("harness_null", harness_null)
    save("envelope", bad_envelope)
    save("null_envelope", null_envelope)
    save("guessed", guessed_column)
    expected = {
        "conflict": "pair_id_conflict",
        "illegal": "illegal_pair_id",
        "negative": "illegal_pair_id",
        "harness_other": "run_id_conflict",
        "harness_null": "illegal_run_id",
        "envelope": "run_id_conflict",
        "null_envelope": "illegal_run_id",
        "guessed": "unbound_pair_id",
    }
    for name, code in expected.items():
        out = tmp_path / f"{name}.json"
        result = _evaluate(cases[name], out)
        report = _read(out)
        _assert_unpublished(result, report)
        assert code in _codes(report), name
        assert "prediction_status" not in json.dumps(report["integrity"])


def test_e1_distinct_pair_numbers_compare_the_same_records(tmp_path: Path):
    records = _two_records()
    bundle = _bundle(
        tmp_path / "pairs",
        records,
        None,
        dataset_id="synthetic-e1-pairs",
        runs=[
            {
                "stem": "a",
                "run_id": "run-a",
                "evaluated_version": "synthetic-a",
                "profile": "canonical",
                "predictions": [{"pair_id": 1, **_answer(1)}, {"pair_id": 2, **_answer(0)}],
            },
            {
                "stem": "b",
                "run_id": "run-b",
                "evaluated_version": "synthetic-b",
                "profile": "canonical",
                "predictions": [{"pair_id": 10, **_answer(1)}, {"pair_id": 20, **_answer(0)}],
            },
        ],
    )
    out = tmp_path / "pairs.json"
    result = _compare(bundle, out)
    assert result.returncode == 0, result.stderr
    report = _read(out)
    assert report["run_id_a"] != report["run_id_b"]
    assert report["own_valid"]["delta_ba_own_valid"]["fraction"] == "0/1"
    assert report["run_binding"]["A"]["identity_verified"] is True
    assert report["run_binding"]["B"]["identity_verified"] is True
    assert report["run_binding"]["A"]["evaluated_version"] == "synthetic-a"
    assert report["run_binding"]["B"]["evaluated_version"] == "synthetic-b"
    assert report["config_sha256"] == report["run_binding"]["A"]["config_sha256"]
    assert report["gold_sidecar_sha256"] == report["run_binding"]["A"]["gold_sidecar_sha256"]
    assert report["task_schema_version"] == "nmr-task-v2"
    assert report["promotion"]["status"] == "disabled_pending_protocol"
    assert "p_value" not in json.dumps(report)

    api = _bundle(
        tmp_path / "api",
        [_record("1", "CCO", 1)],
        [{"response": {"final_answer": "Yes", "pair_id": 99, "key_evidence": "No"}}],
        profile="api",
        dataset_id="synthetic-e1-api",
    )
    api_out = tmp_path / "api.json"
    assert _evaluate(api, api_out).returncode == 0
    assert _read(api_out)["rows"][0]["pred_label"] == 1


def test_e2_answer_strings_are_invalid_and_stay_in_coverage(tmp_path: Path):
    cases = {
        "duplicate_answer": (
            "standalone",
            {"answer": '{"final_answer":"No","final_answer":"Yes"}', "fail_reason": None},
            "duplicate_json_key",
        ),
        "unclosed_answer": (
            "standalone",
            {"answer": '<answer>{"final_answer":"Yes"}</answer><answer>{"final_answer":"No"', "fail_reason": None},
            "multiple_answers",
        ),
        "duplicate_state": (
            "canonical",
            {"terminal": '{"state":"timeout","state":"answer","final_answer":"Yes"}'},
            "duplicate_json_key",
        ),
    }
    for name, (profile, prediction, reason) in cases.items():
        bundle = _bundle(
            tmp_path / name,
            [_record("1", "CCO", 1)],
            [prediction],
            profile=profile,
            dataset_id=f"synthetic-e2-{name}",
        )
        out = tmp_path / f"{name}.json"
        result = _evaluate(bundle, out)
        assert result.returncode == 0, result.stderr
        report = _read(out)
        assert report["integrity"]["status"] == "passed"
        assert report["metrics"]["N"] == 1
        assert report["metrics"]["V"] == 0
        assert report["metrics"]["I"] == 1
        assert report["metrics"]["coverage"]["fraction"] == "0/1"
        assert report["rows"][0]["pred_label"] is None
        assert report["rows"][0]["invalid_reason"] == reason
        assert report["rows"][0]["prediction_status"] == "invalid"


def test_e2_hand_replacement_keeps_valid_only_balanced_accuracy(tmp_path: Path):
    bundle = write_hand_fixture(tmp_path / "hand")
    path = Path(bundle["runs"][0]["predictions"])
    document = _read(path)
    for row in document["predictions"]:
        if row["pair_id"] == 2:
            row["terminal"] = '{"state":"answer","final_answer":"No","final_answer":"Yes"}'
        if row["pair_id"] == 10:
            row["terminal"] = '{"state":"timeout","state":"answer","final_answer":"Yes"}'
    ev.dump_json(path, document)
    out = tmp_path / "hand.json"
    result = _evaluate(bundle, out)
    assert result.returncode == 0, result.stderr
    report = _read(out)
    metrics = report["metrics"]
    assert metrics["N"] == 12
    assert metrics["V"] == 8
    assert metrics["I"] == 4
    assert metrics["confusion_matrix_valid"] == [[3, 2], [0, 3]]
    assert metrics["confusion_matrix_valid_sum"] == 8
    assert metrics["gold_by_outcome"] == [[3, 2, 1], [0, 3, 3]]
    assert metrics["gold_by_outcome_sum"] == 12
    assert metrics["balanced_accuracy_valid"]["fraction"] == "4/5"
    assert metrics["balanced_accuracy_valid"]["numerator"] == 4
    assert metrics["balanced_accuracy_valid"]["denominator"] == 5
    assert metrics["accuracy_valid"]["fraction"] == "3/4"
    assert metrics["coverage"]["fraction"] == "2/3"
    assert metrics["by_label"]["0"]["coverage"]["fraction"] == "5/6"
    assert metrics["by_label"]["1"]["coverage"]["fraction"] == "1/2"
    by_id = {row["record_id"]: row for row in report["rows"]}
    assert by_id["2"]["pred_label"] is None
    assert by_id["2"]["invalid_reason"] == "duplicate_json_key"
    assert by_id["10"]["pred_label"] is None
    assert by_id["10"]["invalid_reason"] == "duplicate_json_key"
    assert by_id["6"]["invalid_reason"] == "empty_answer"
    assert by_id["11"]["invalid_reason"] == "parse_error"
    assert report["primary_metric"] == "balanced_accuracy_valid"
    assert report["invalid_policy"] == "exclude_from_primary"
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


def test_e2_outer_duplicate_pair_id_fails_both_commands(tmp_path: Path):
    bundle = _bundle(
        tmp_path / "outer",
        [_record("1", "CCO", 1)],
        [_answer(1)],
        dataset_id="synthetic-e2-outer",
    )
    path = Path(bundle["runs"][0]["predictions"])
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"pair_id": 1', '"pair_id": 1, "pair_id": 1', 1), encoding="utf-8")
    evaluate_out = tmp_path / "evaluate.json"
    evaluate = _evaluate(bundle, evaluate_out)
    evaluate_report = _read(evaluate_out)
    _assert_unpublished(evaluate, evaluate_report)
    assert "duplicate_json_key" in _codes(evaluate_report)
    compare_out = tmp_path / "compare.json"
    compare = _compare(bundle, compare_out)
    compare_report = _read(compare_out)
    _assert_unpublished(compare, compare_report)
    assert "duplicate_json_key" in _codes(compare_report)


def test_e3_evaluate_and_compare_reject_the_same_binding_errors(tmp_path: Path):
    records = _two_records()
    bundle = _bundle(tmp_path / "bind", records, [_answer(1), _answer(0)], dataset_id="synthetic-e3")
    run_path = Path(bundle["runs"][0]["predictions"]).with_name("only_run_manifest.json")
    prediction_path = Path(bundle["runs"][0]["predictions"])

    def mutate_run(change):
        document = _read(run_path)
        change(document)
        ev.dump_json(run_path, document)

    def mutate_predictions(change):
        document = _read(prediction_path)
        change(document)
        ev.dump_json(prediction_path, document)

    mutations = [
        ("zero_config", lambda: mutate_run(lambda run: run.__setitem__("config_sha256", "0" * 64)), "config_hash_mismatch"),
        ("missing_config", lambda: mutate_run(lambda run: run.pop("config_sha256")), "config_hash_missing"),
        ("false_config", lambda: mutate_run(lambda run: run.__setitem__("config_sha256", False)), "config_hash_invalid"),
        ("empty_version", lambda: mutate_run(lambda run: run.__setitem__("evaluated_version", "")), "evaluated_version_missing"),
        ("missing_schema", lambda: mutate_predictions(lambda preds: preds.pop("schema_version")), "prediction_schema_missing"),
        ("unknown_schema", lambda: mutate_predictions(lambda preds: preds.__setitem__("schema_version", "nmr-eval-prediction-v0")), "prediction_schema_rejected"),
    ]
    original_run = run_path.read_bytes()
    original_predictions = prediction_path.read_bytes()
    for name, change, code in mutations:
        run_path.write_bytes(original_run)
        prediction_path.write_bytes(original_predictions)
        change()
        evaluate = _evaluate(bundle, tmp_path / f"{name}-evaluate.json")
        evaluate_report = _read(tmp_path / f"{name}-evaluate.json")
        compare = _compare(bundle, tmp_path / f"{name}-compare.json")
        compare_report = _read(tmp_path / f"{name}-compare.json")
        _assert_unpublished(evaluate, evaluate_report)
        _assert_unpublished(compare, compare_report)
        assert code in _codes(evaluate_report), name
        assert code in _codes(compare_report), name

    run_path.write_bytes(original_run)
    prediction_path.write_bytes(original_predictions)
    legacy = _read(prediction_path)
    legacy["profile"] = "legacy_diagnostic"
    legacy.pop("schema_version")
    for row in legacy["predictions"]:
        row.pop("pair_id")
        row.pop("run_id")
    ev.dump_json(prediction_path, legacy)
    legacy_out = tmp_path / "legacy.json"
    legacy_result = _evaluate(bundle, legacy_out, legacy=True)
    assert legacy_result.returncode == 0, legacy_result.stderr
    assert _read(legacy_out)["integrity"]["status"] == "legacy_diagnostic"
    assert _read(legacy_out)["formal_primary_published"] is False
    legacy["schema_version"] = "nmr-eval-prediction-v0"
    ev.dump_json(prediction_path, legacy)
    unknown_legacy = _evaluate(bundle, tmp_path / "legacy-unknown.json", legacy=True)
    unknown_report = _read(tmp_path / "legacy-unknown.json")
    _assert_unpublished(unknown_legacy, unknown_report)
    assert "prediction_schema_rejected" in _codes(unknown_report)


def test_e3_observed_hash_null_is_unknown_and_false_is_an_error(tmp_path: Path):
    records = _two_records()
    bundle = _bundle(tmp_path / "obs", records, [_answer(1), _answer(0)], dataset_id="synthetic-e3-obs")
    path = Path(bundle["runs"][0]["predictions"])
    first = tmp_path / "missing.json"
    assert _evaluate(bundle, first).returncode == 0
    missing = _read(first)["rows"][0]
    assert missing["observed_task_availability"] == "not_recorded"
    assert missing["model_observed_task_sha256"] is None
    assert missing["model_observation_compared"] is False
    task_hash = missing["task_sha256"]

    document = _read(path)
    document["predictions"][0]["observed_task_sha256"] = None
    ev.dump_json(path, document)
    null_out = tmp_path / "null.json"
    assert _evaluate(bundle, null_out).returncode == 0
    null_row = _read(null_out)["rows"][0]
    assert null_row["observed_task_availability"] == "unknown"
    assert null_row["model_observed_task_sha256"] is None
    assert null_row["model_observed_task_sha256"] != task_hash

    document["predictions"][0]["observed_task_sha256"] = task_hash
    ev.dump_json(path, document)
    match_out = tmp_path / "match.json"
    assert _evaluate(bundle, match_out).returncode == 0
    assert _read(match_out)["rows"][0]["model_observation_compared"] is True
    assert _read(match_out)["rows"][0]["model_observed_task_sha256"] == task_hash

    document["predictions"][0]["observed_task_sha256"] = "0" * 64
    ev.dump_json(path, document)
    mismatch = _evaluate(bundle, tmp_path / "mismatch.json")
    mismatch_report = _read(tmp_path / "mismatch.json")
    _assert_unpublished(mismatch, mismatch_report)
    assert "observed_task_hash_mismatch" in _codes(mismatch_report)
    compare = _compare(bundle, tmp_path / "mismatch-compare.json")
    compare_report = _read(tmp_path / "mismatch-compare.json")
    _assert_unpublished(compare, compare_report)
    assert "observed_task_hash_mismatch" in _codes(compare_report)

    document["predictions"][0]["observed_task_sha256"] = False
    ev.dump_json(path, document)
    illegal = _evaluate(bundle, tmp_path / "false.json")
    illegal_report = _read(tmp_path / "false.json")
    _assert_unpublished(illegal, illegal_report)
    assert "illegal_observed_task_hash" in _codes(illegal_report)
    assert task_hash not in json.dumps(illegal_report.get("rows"))
    compare = _compare(bundle, tmp_path / "false-compare.json")
    compare_report = _read(tmp_path / "false-compare.json")
    _assert_unpublished(compare, compare_report)
    assert "illegal_observed_task_hash" in _codes(compare_report)


def test_e3_statistics_need_a_new_manifest_bound_to_that_config(tmp_path: Path):
    records = [
        _record("1", "CCO", 1),
        _record("2", "CCO", 0, "c1ccccc1"),
        _record("3", "CCCO", 1),
        _record("4", "CCCO", 0, "c1ccccc1"),
    ]
    predictions = [_answer(1), _answer(0), _answer(1), _answer(0)]
    stats = scoring_config({
        "enabled": True,
        "method": "paired_cluster_percentile",
        "bootstrap_replicates": 4,
        "confidence_level": 0.5,
        "seed": 20261126,
    })
    bundle = _bundle(
        tmp_path / "stats",
        records,
        None,
        dataset_id="synthetic-e3-stats",
        runs=[
            {"stem": "a", "run_id": "stats-a", "evaluated_version": "synthetic-a", "profile": "canonical", "predictions": predictions},
            {"stem": "b", "run_id": "stats-b", "evaluated_version": "synthetic-b", "profile": "canonical", "predictions": predictions},
        ],
    )
    stats_path = tmp_path / "stats" / "statistics_config.json"
    ev.dump_json(stats_path, stats)
    before_a = Path(bundle["runs"][0]["predictions"]).read_bytes()
    before_b = Path(bundle["runs"][1]["predictions"]).read_bytes()
    rejected = _evaluate(bundle, tmp_path / "rejected.json", config=str(stats_path))
    rejected_report = _read(tmp_path / "rejected.json")
    _assert_unpublished(rejected, rejected_report)
    assert "config_hash_mismatch" in _codes(rejected_report)
    rejected_compare = _compare(bundle, tmp_path / "rejected-compare.json", config=str(stats_path))
    rejected_compare_report = _read(tmp_path / "rejected-compare.json")
    _assert_unpublished(rejected_compare, rejected_compare_report)
    assert "config_hash_mismatch" in _codes(rejected_compare_report)

    new_runs = []
    for spec in bundle["runs"]:
        rebound = rebind_run_manifest(_read(Path(spec["run"])), stats)
        new_path = Path(spec["run"]).with_name(Path(spec["run"]).name.replace("_run_manifest", "_stats_run_manifest"))
        ev.dump_json(new_path, rebound)
        new_runs.append(new_path)
    assert Path(bundle["runs"][0]["predictions"]).read_bytes() == before_a
    assert Path(bundle["runs"][1]["predictions"]).read_bytes() == before_b
    accepted = _evaluate(
        bundle,
        tmp_path / "accepted.json",
        config=str(stats_path),
        run=str(new_runs[0]),
        predictions=bundle["runs"][0]["predictions"],
    )
    assert accepted.returncode == 0, accepted.stderr
    accepted_report = _read(tmp_path / "accepted.json")
    assert accepted_report["formal_primary_published"] is True
    assert accepted_report["config_sha256"] == ev.load_config(stats)["content_sha256"]
    assert accepted_report["run_binding"]["manifest_config_sha256"] == accepted_report["config_sha256"]
    assert "balanced_accuracy_valid" in accepted_report["statistics"]
    assert "p_value" not in json.dumps(accepted_report["statistics"])
    compared = _compare(
        bundle,
        tmp_path / "accepted-compare.json",
        config=str(stats_path),
        run_a=str(new_runs[0]),
        predictions_a=bundle["runs"][0]["predictions"],
        run_b=str(new_runs[1]),
        predictions_b=bundle["runs"][1]["predictions"],
    )
    assert compared.returncode == 0, compared.stderr
    compared_report = _read(tmp_path / "accepted-compare.json")
    assert compared_report["integrity"]["status"] == "passed"
    assert compared_report["config_sha256"] == accepted_report["config_sha256"]
    assert compared_report["run_binding"]["A"]["identity_verified"] is True
    assert compared_report["promotion"]["status"] == "disabled_pending_protocol"
    assert Path(bundle["runs"][0]["run"]).read_bytes() != new_runs[0].read_bytes()
