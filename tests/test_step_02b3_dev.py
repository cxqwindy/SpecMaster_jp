"""CLI checks for the step 02B-3 development-set builder.

Selection expectations are computed here with hashlib and RDKit. They are not
produced by calling the builder's chooser a second time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import step_02b3_dev as dev  # noqa: E402
from step_02a_fingerprints import shift_sequence_hash, spectrum_views, structure_keys, identity_token  # noqa: E402

SPEC = json.loads((REPO / "docs" / "rsi" / "step_02b3" / "build_config.json").read_text(encoding="utf-8"))
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
CATALOG = """
C CC CCC CCCC CC(C)C CCCCC CC(C)CC CC(C)(C)C CCCCCC CC(C)CCC CCC(C)CC
CC(C)(C)CC CC(C)CC(C)C CCCCCCC CC(C)CCCC CCC(C)CCC CC(C)(C)CCC CC(C)C(C)CC
CCCCCCCC CC(C)CCCCC CCC(C)CCCC CC(C)(C)CCCC CC(C)CC(C)CC CC(C)C(C)CCC CCC(C)(C)CCC
CO CCO CCCO CC(C)O CCCCO CC(C)CO CC(C)(C)O CCOCC COCCC CC(C)OC
C[C@H](O)CC C[C@@H](O)CC CCCCCO CC(C)CCO CCOCCC C[C@H](O)CCC C[C@@H](O)CCC
COC OCCO COCO CC(=O)C CCC=O C=CCO CCC(=O)C CC(=O)CC CCCC=O C1CCOC1
c1ccccc1 Cc1ccccc1 Cc1ccc(O)cc1 Cc1cccc(O)c1 Cc1ccccc1O COc1ccccc1
CCc1ccccc1 Cc1ccccc1C Cc1cccc(C)c1 Cc1ccc(C)cc1 CCN CCCN c1ccccc1Cl CC(=O)O CCC(=O)O CCCCCCO CCCCCCCO CCCCCCCCO
c1ccc(O)cc1 c1ccc(O)c(O)c1 CC(C)N CCNC C#N CCC#N
C1CCCCC1 C1CCCCCC1 CC(C)C(C)C CCC(C)C CCCC(C)C
COCCOC CCOCCO CC(C)OCC c1ccc(N)cc1 c1ccc(Cl)cc1
""".split()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-B", "scripts/rsi/step_02b3_build_dev.py", *args],
        cwd=REPO, env=env, text=True, capture_output=True,
    )


def _assert_code(result: subprocess.CompletedProcess[str], code: int) -> None:
    assert result.returncode == code, result.stdout + "\n" + result.stderr[-4000:]


def _rank(domain: str, *parts: object) -> str:
    payload = json.dumps([domain, *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return "canonical:" + Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _formula(token: str) -> str:
    smiles = token.split("canonical:", 1)[1]
    return rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))


def _morgan_int(token: str) -> int:
    smiles = token.split("canonical:", 1)[1]
    value = 0
    for bit in GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits():
        value |= 1 << bit
    return value


def _bits_int(text: str) -> int:
    value = 0
    for index, bit in enumerate(text):
        if bit == "1":
            value |= 1 << index
    return value


def _tanimoto(left: int, right: int) -> float:
    inter = (left & right).bit_count()
    union = (left | right).bit_count()
    return 1.0 if union == 0 else inter / union


def _ownership(tokens: list[str]) -> dict[str, str]:
    ranked = sorted(tokens, key=lambda token: (_rank(SPEC["ownership_domain"], SPEC["seed"], token), token))
    return {token: "search-dev" if index % 2 == 0 else "gate-dev" for index, token in enumerate(ranked)}


def _hash_choice(domain: str, source: str, tokens: list[str]) -> str:
    best = None
    best_rank = None
    for token in tokens:
        rank = _rank(domain, SPEC["seed"], source, token)
        if best_rank is None or rank < best_rank or (rank == best_rank and token < best):
            best = token
            best_rank = rank
    assert best is not None
    return best


def _neighbor(source: str, tokens: list[str], fps: dict[str, int]) -> str:
    best = None
    best_sim = None
    for token in tokens:
        sim = _tanimoto(fps[source], fps[token])
        if best_sim is None or sim > best_sim or (sim == best_sim and token < best):
            best = token
            best_sim = sim
    assert best is not None
    return best


def _constructable(tokens: list[str], formulas: dict[str, str]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        grouped[formulas[token]].append(token)
    chosen = []
    for token in tokens:
        same = len(grouped[formulas[token]]) - 1
        different = len(tokens) - len(grouped[formulas[token]])
        if same >= 1 and different >= 2:
            chosen.append(token)
    return chosen


def _selected(tokens: list[str], formulas: dict[str, str], groups: int) -> list[str]:
    constructable = _constructable(tokens, formulas)
    return sorted(constructable, key=lambda token: (_rank(SPEC["source_domain"], SPEC["seed"], token), token))[:groups]


def _spectrum(raw: dict) -> dict:
    return {
        field: raw[field]
        for field in ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency")
        if field in raw
    }


def _default_raw(shift: float) -> dict:
    return {
        "c_nmr_peaks": [{"delta (ppm)": shift}],
        "h_nmr_peaks": [{"centroid": shift / 10, "delta": shift / 10, "nH": 1, "j_values": "7.5", "category": "s"}],
        "nmr_frequency": "400 MHz",
        "nmr_solvent": "CDCl3",
    }


def _patterns(shift: float) -> list[dict]:
    return [
        {
            "c_nmr_peaks": [{"delta (ppm)": shift}, shift + 0.5, {"delta (ppm)": shift}],
            "h_nmr_peaks": [
                {"centroid": 1.25, "delta": 1.25, "nH": 0, "j_values": "7.5", "category": "d"},
                {"centroid": 3.5, "delta": 3.5, "nH": None, "rangeMin": 3.25, "rangeMax": 3.75},
                {"centroid": 2.25, "delta": 2.25},
            ],
            "nmr_frequency": "400 MHz",
            "nmr_solvent": "CDCl3",
            "molecular_formula": "C999",
            "atoms": ["C", "O"],
            "coordinates": [[0.0, 0.0, 0.0]],
        },
        {
            "c_nmr_peaks": [{"delta (ppm)": shift}],
            "nmr_frequency": "100 MHz",
            "experimental_total_nH": 0,
        },
        {
            "c_nmr_peaks": [{"delta (ppm)": shift}],
            "h_nmr_peaks": None,
            "nmr_frequency": "400 MHz",
        },
        {
            "c_nmr_peaks": [{"delta (ppm)": shift}],
            "h_nmr_peaks": [],
            "nmr_solvent": "DMSO",
        },
    ]


def _molecules(fp_bits: str | None = None) -> list[dict]:
    rows = []
    seen = set()
    for smiles in CATALOG:
        token = _token(smiles)
        if token in seen:
            continue
        seen.add(token)
        canonical = token.split("canonical:", 1)[1]
        raw = "C(O)C" if canonical == "CCO" else canonical
        item = {
            "key": f"k{len(rows)}",
            "smiles": raw,
            "declared": canonical,
            "token": token,
            "raw": _default_raw(20.0 + len(rows)),
        }
        if fp_bits is not None:
            item["raw"] = dict(item["raw"])
            item["raw"]["_fixture_fp_bits"] = fp_bits
        rows.append(item)
    return rows


def _write_case(directory: Path, molecules: list[dict], extra_index: list[dict] | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.jsonl"
    records_path = directory / "records.jsonl"
    with index_path.open("w", encoding="utf-8") as index, records_path.open("w", encoding="utf-8") as records:
        for molecule in molecules:
            raw = dict(molecule["raw"])
            bits = raw.pop("_fixture_fp_bits", None)
            viewed = spectrum_views(_spectrum(raw))
            row = {
                "candidate_token": molecule["token"],
                "exposure_reasons": [],
                "gt_token": molecule["token"],
                "in_auto_positive_pool": True,
                "quality_reasons": [],
                "shift_sequence_hash": shift_sequence_hash(_spectrum(raw)),
                "source_key": molecule["key"],
                "source_lmdb": "valid.lmdb",
                "spectrum_norm_hash": viewed["normalized_hash"],
            }
            index.write(dev.canonical_json(row))
            index.write("\n")
            stored = dict(raw)
            stored["smiles"] = molecule["smiles"]
            stored["smiles_canonical"] = molecule["declared"]
            if bits is not None:
                stored["_fixture_fp_bits"] = bits
            records.write(dev.canonical_json({
                "raw": stored,
                "source_key": molecule["key"],
                "source_lmdb": "valid.lmdb",
            }))
            records.write("\n")
        for row in extra_index or []:
            index.write(dev.canonical_json(row))
            index.write("\n")
    return index_path, records_path


def _build(out: Path, index: Path, records: Path, groups: int, reuse: Path | None = None) -> subprocess.CompletedProcess[str]:
    args = ["build", "--out", str(out), "--mode", "fixture", "--index", str(index), "--records", str(records), "--groups", str(groups)]
    if reuse is not None:
        args.extend(["--reuse-cache", str(reuse)])
    return _run(args)


def test_spec_literals_and_hand_written_hash():
    assert SPEC["seed"] == 20261126
    assert SPEC["groups_per_split"] == 500
    assert SPEC["ownership_domain"] == "step_02b3.ownership.v1"
    assert SPEC["source_domain"] == "step_02b3.source_sample.v1"
    assert SPEC["same_formula_domain"] == "step_02b3.neg_same_formula.v1"
    assert SPEC["random_domain"] == "step_02b3.neg_random.v1"
    assert SPEC["auto_positive_pool"]["valid.lmdb"] == 48234
    assert SPEC["auto_positive_pool"]["train.lmdb"] == 916419
    assert SPEC["auto_positive_pool"]["test.lmdb"] == 104695
    assert SPEC["eligibility_index_sha256"] == "b1f756d55a99cc3dcf872504b66c2f04738df9c96d9b549d2ae62bab80288665"
    assert SPEC["eligibility_index_bytes"] == 817296028
    assert SPEC["fingerprint"]["radius"] == 2
    assert SPEC["fingerprint"]["n_bits"] == 2048
    assert SPEC["fingerprint"]["include_chirality"] is False
    assert SPEC["promotion_enabled"] is False
    assert SPEC["statistics_enabled"] is False
    assert SPEC["example_status"] is None
    assert SPEC["purpose_real"] == "real_development_dataset"
    assert dev.SEED == 20261126
    assert dev.domain_rank("step_02b3.ownership.v1", 20261126, "canonical:C") == (
        "f8f73200f7ea23a436a57fc6b83e1a9f4a8f0c8da491897fde0243502c513341"
    )


def test_similarity_tie_breaks_on_the_structure_token():
    chosen, similarity = dev.pick_max_similarity([
        ("canonical:M", 0.9),
        ("canonical:B", 0.4),
        ("canonical:A", 0.4),
    ])
    assert chosen == "canonical:M"
    assert similarity == 0.9
    chosen, similarity = dev.pick_max_similarity([
        ("canonical:B", 0.4),
        ("canonical:A", 0.4),
        ("canonical:M", 0.2),
    ])
    assert (chosen, similarity) == ("canonical:A", 0.4)


def test_identity_keeps_salt_and_stereo():
    salt = identity_token(structure_keys("[Na+].[Cl-]"))
    right = identity_token(structure_keys("C[C@H](O)CC"))
    left = identity_token(structure_keys("C[C@@H](O)CC"))
    assert salt.startswith("canonical:")
    assert "Na" in salt and "Cl" in salt
    assert right != left
    assert structure_keys("C[C@H](O)CC")["connectivity"] == structure_keys("C[C@@H](O)CC")["connectivity"]


def test_formula_failure_is_explicit_and_not_replaced():
    result = dev.formula_result(None, {"molecular_formula": "C2H6O"})
    assert result["status"] == "failed"
    assert result["calculated"] is None
    assert result["raw"] == "C2H6O"
    assert result["relation"] == "failed"


def test_odd_ownership_and_quota_shortfall(tmp_path: Path):
    molecules = []
    for index, smiles in enumerate(("C", "CC", "CCC")):
        token = _token(smiles)
        molecules.append({
            "key": f"s{index}",
            "smiles": smiles,
            "declared": token.split("canonical:", 1)[1],
            "token": token,
            "raw": _default_raw(30.0 + index),
        })
    index, records = _write_case(tmp_path / "in", molecules)
    out = tmp_path / "out"
    result = _build(out, index, records, 1)
    _assert_code(result, 2)
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "incomplete"
    assert "quota_shortfall" in status["blocking_reasons"]
    assert not (out / "search-dev" / "model_inputs.jsonl").exists()
    owned = {}
    for line in (out / "ownership.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["register"] == "valid_pool":
            owned[row["structure_token"]] = row["assignment"]
    expected = _ownership([item["token"] for item in molecules])
    assert owned == expected
    assert list(owned.values()).count("search-dev") == 2
    assert list(owned.values()).count("gate-dev") == 1


def test_duplicate_parent_stops(tmp_path: Path):
    token = _token("CCO")
    molecules = []
    for index, smiles in enumerate(("CCO", "C(O)C")):
        molecules.append({
            "key": f"d{index}",
            "smiles": smiles,
            "declared": "CCO",
            "token": token,
            "raw": _default_raw(40.0 + index),
        })
    index, records = _write_case(tmp_path / "in", molecules)
    out = tmp_path / "out"
    result = _build(out, index, records, 1)
    _assert_code(result, 2)
    audit = json.loads((out / "duplicate_source_audit.json").read_text(encoding="utf-8"))
    assert audit["count"] == 1
    assert not (out / "search-dev").exists()


def test_future_train_overlap_stops(tmp_path: Path):
    token = _token("CCO")
    molecule = {
        "key": "v1",
        "smiles": "CCO",
        "declared": "CCO",
        "token": token,
        "raw": _default_raw(50.0),
    }
    extra = [{
        "candidate_token": token,
        "exposure_reasons": [],
        "gt_token": token,
        "in_auto_positive_pool": True,
        "quality_reasons": [],
        "shift_sequence_hash": None,
        "source_key": "t1",
        "source_lmdb": "train.lmdb",
        "spectrum_norm_hash": "0" * 64,
    }]
    index, records = _write_case(tmp_path / "in", [molecule], extra)
    out = tmp_path / "out"
    result = _build(out, index, records, 1)
    _assert_code(result, 2)
    report = json.loads((out / "source_pool_report.json").read_text(encoding="utf-8"))
    assert report["valid_pool_intersections"]["valid_vs_future_train"] == 1
    assert not (out / "model_inputs.jsonl").exists()
    assert not (out / "search-dev").exists()


def _expected_groups(molecules: list[dict], fps: dict[str, int], groups: int) -> dict[str, list[dict[str, str]]]:
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    by_key = {item["token"]: item["key"] for item in molecules}
    assigned = _ownership([item["token"] for item in molecules])
    expected = {}
    for split in ("search-dev", "gate-dev"):
        pool = [token for token, side in assigned.items() if side == split]
        selected = _selected(pool, formulas, groups)
        groups_out = []
        for source in selected:
            same = [token for token in pool if token != source and formulas[token] == formulas[source]]
            different = [token for token in pool if formulas[token] != formulas[source]]
            same_choice = _hash_choice(SPEC["same_formula_domain"], source, same)
            neighbor = _neighbor(source, different, fps)
            random_choice = _hash_choice(
                SPEC["random_domain"], source, [token for token in different if token != neighbor],
            )
            groups_out.append({
                "source": source,
                "source_key": by_key[source],
                "pos": source,
                "neg/same_formula": same_choice,
                "neg/similar_structure": neighbor,
                "neg/random": random_choice,
            })
        expected[split] = groups_out
    return expected


def _fps_for(molecules: list[dict], fp_bits: str | None) -> dict[str, int]:
    if fp_bits is None:
        return {item["token"]: _morgan_int(item["token"]) for item in molecules}
    value = _bits_int(fp_bits)
    return {item["token"]: value for item in molecules}


def _apply_patterns(molecules: list[dict], expected: dict[str, list[dict[str, str]]]) -> dict[str, dict]:
    assigned = {}
    for split, split_groups in expected.items():
        base = 1000.0 if split == "search-dev" else 3000.0
        for index, group in enumerate(split_groups):
            pattern = _patterns(base + index)[index % 4]
            for molecule in molecules:
                if molecule["token"] == group["source"]:
                    bits = molecule["raw"].get("_fixture_fp_bits")
                    molecule["raw"] = dict(pattern)
                    if bits is not None:
                        molecule["raw"]["_fixture_fp_bits"] = bits
                    assigned[molecule["token"]] = pattern
    return assigned


def _assert_dataset(out: Path, molecules: list[dict], expected: dict[str, list[dict[str, str]]], patterns: dict[str, dict]):
    status = json.loads((out / "build_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "fixture_checked"
    assert status["logical_data_sha256"]
    config = dev.load_config(json.loads((out / "config.json").read_text(encoding="utf-8")))
    assert config["example_status"] is None
    assert config["promotion"]["enabled"] is False
    assert config["statistics"]["enabled"] is False
    protocol = json.loads((out / "dataset_protocol.json").read_text(encoding="utf-8"))
    assert protocol["purpose"] == "synthetic_cli_fixture"
    assert protocol["promotion"]["status"] == "disabled_pending_protocol"
    owned = {}
    for line in (out / "ownership.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["register"] == "valid_pool":
            owned[row["structure_token"]] = row["assignment"]
    assert owned == _ownership([item["token"] for item in molecules])
    by_token = {item["token"]: item for item in molecules}
    for split, groups in expected.items():
        split_dir = out / split
        manifest = json.loads((split_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        prepared, errors = dev.prepare_dataset(manifest, split_dir / "dataset_manifest.json", verify_hash=True)
        assert errors == [] and prepared is not None
        gold = json.loads((split_dir / "gold_sidecar.json").read_text(encoding="utf-8"))
        gold_hash, gold_errors = dev.check_gold_sidecar(gold, prepared, verify_hash=True)
        assert gold_errors == [] and gold_hash
        model_rows = [json.loads(line) for line in (split_dir / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        sidecars = [json.loads(line) for line in (split_dir / "input_sidecars.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(model_rows) == len(manifest["records"]) == len(sidecars) == len(groups) * 4
        pairs = json.loads((split_dir / "pair_to_record.json").read_text(encoding="utf-8"))
        assert pairs["ordered"][0]["pair_id"] == 0
        record_ids = []
        by_source = defaultdict(list)
        for sidecar in sidecars:
            by_source[sidecar["structure_token"]].append(sidecar)
            record_ids.append(sidecar["record_id"])
        assert set(by_source) == {group["source"] for group in groups}
        for group in groups:
            rows = by_source[group["source"]]
            assert len(rows) == 4
            by_role = {row["role"]: row for row in rows}
            assert set(by_role) == {"pos", "neg/same_formula", "neg/similar_structure", "neg/random"}
            for role, token in group.items():
                if role in {"source", "source_key"}:
                    continue
                assert by_role[role]["candidate_token"] == token
                assert by_role[role]["donor_locator"]["structure_token"] == token
                assert owned[token] == split
            assert by_role["pos"]["gold_label"] == 1
            assert sum(row["gold_label"] for row in rows) == 1
            assert len({row["candidate_token"] for row in rows}) == 4
            source_key = by_token[group["source"]]["key"]
            assert all(row["source_key"] == source_key for row in rows)
            for role, row in by_role.items():
                expected_id = _rank(
                    SPEC["record_domain"], SPEC["protocol"], "valid.lmdb", source_key, role, row["candidate_token"],
                )
                assert row["record_id"] == expected_id
        ordered_ids = [row["record_id"] for row in sorted(
            (row for rows in by_source.values() for row in rows),
            key=lambda row: (_rank(SPEC["output_domain"], SPEC["seed"], row["record_id"]), row["record_id"]),
        )]
        assert [item["record_id"] for item in pairs["ordered"]] == ordered_ids
        for row in model_rows:
            task = row["task"]
            assert set(task) == set(dev.TASK_FIELDS)
            assert json.loads(row["user_json"]) == task
            assert "gold_label" not in row["user_json"]
            assert "negative_type" not in task
            assert task["nmr_frequency_nucleus"] == "unknown"
        for token, pattern in patterns.items():
            if token not in by_source or by_source[token][0]["split"] != split:
                continue
            sidecar = by_source[token][0]
            task = next(row["task"] for row in model_rows if row["record_id"] == sidecar["record_id"])
            assert sidecar["presence"]["c_nmr_peaks"] == ("present" if "c_nmr_peaks" in pattern else "missing")
            if "h_nmr_peaks" not in pattern:
                assert sidecar["presence"]["h_nmr_peaks"] == "missing"
                assert task["h_nmr_peaks"] == []
            elif pattern["h_nmr_peaks"] is None:
                assert sidecar["presence"]["h_nmr_peaks"] == "null"
                assert task["h_nmr_peaks"] == []
            elif pattern["h_nmr_peaks"] == []:
                assert sidecar["presence"]["h_nmr_peaks"] == "empty_list"
                assert task["h_nmr_peaks"] == []
            else:
                assert task["h_nmr_peaks"][0]["nH"] == 0
                assert task["h_nmr_peaks"][1]["nH"] is None
                assert "nH" not in task["h_nmr_peaks"][2]
                assert task["h_nmr_peaks"][0]["j_values"] == "7.5"
            if "experimental_total_nH" in pattern:
                assert task["experimental_total_nH"] == 0
                assert sidecar["presence"]["experimental_total_nH"] == "present"
            else:
                assert task["experimental_total_nH"] is None
                assert sidecar["presence"]["experimental_total_nH"] == "missing"
            if pattern.get("molecular_formula") == "C999":
                assert sidecar["formula"]["relation"] == "differ"
                assert sidecar["formula"]["used_for_retrieval"] == "calculated"
                assert sidecar["formula"]["calculated"] == _formula(token)
            if "atoms" in pattern:
                assert "atoms" not in task and "coordinates" not in task
                assert "atoms" in sidecar["theoretical_fields_not_copied"]
            viewed = spectrum_views(_spectrum(pattern))
            assert sidecar["spectrum_norm_hash"] == viewed["normalized_hash"]
            deltas = []
            for peak in pattern["c_nmr_peaks"]:
                deltas.append(peak.get("delta (ppm)") if isinstance(peak, dict) else peak)
            got = [peak.get("delta (ppm)") for peak in task["c_nmr_peaks"]]
            assert got == deltas
    ethanol = next(item for item in molecules if item["smiles"] == "C(O)C")
    assert ethanol["token"] == "canonical:CCO"
    assert owned[ethanol["token"]] in {"search-dev", "gate-dev"}
    assert not (out / "run_manifest.json").exists()
    assert not (out / "predictions.json").exists()


def test_cli_real_fingerprints_match_the_hand_oracle(tmp_path: Path):
    molecules = _molecules()
    assert len(molecules) >= 40
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    assigned = _ownership([item["token"] for item in molecules])
    counts = {
        split: len(_constructable([token for token, side in assigned.items() if side == split], formulas))
        for split in ("search-dev", "gate-dev")
    }
    groups = min(4, counts["search-dev"], counts["gate-dev"])
    assert groups >= 4, counts
    fps = _fps_for(molecules, None)
    expected = _expected_groups(molecules, fps, groups)
    patterns = _apply_patterns(molecules, expected)
    index, records = _write_case(tmp_path / "in", molecules)
    out = tmp_path / "out"
    result = _build(out, index, records, groups)
    _assert_code(result, 0)
    _assert_dataset(out, molecules, expected, patterns)
    validated = _run(["validate", "--out", str(out)])
    _assert_code(validated, 0)
    audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert audit["token_intersections"]["search_union_vs_gate_union"] == 0
    assert audit["raw_spectrum_cross_split"] == 0
    assert audit["scaffold_intersection_required_to_be_zero"] is False


def test_cli_equal_similarity_uses_the_token_order(tmp_path: Path):
    bits = "1" * 100 + "0" * (2048 - 100)
    molecules = _molecules(bits)
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    assigned = _ownership([item["token"] for item in molecules])
    counts = {
        split: len(_constructable([token for token, side in assigned.items() if side == split], formulas))
        for split in ("search-dev", "gate-dev")
    }
    groups = min(2, counts["search-dev"], counts["gate-dev"])
    assert groups >= 1, counts
    fps = _fps_for(molecules, bits)
    expected = _expected_groups(molecules, fps, groups)
    for split, groups_out in expected.items():
        pool = [token for token, side in assigned.items() if side == split]
        for group in groups_out:
            different = [token for token in pool if formulas[token] != formulas[group["source"]]]
            assert group["neg/similar_structure"] == min(different)
    index, records = _write_case(tmp_path / "in", molecules)
    out = tmp_path / "out"
    result = _build(out, index, records, groups)
    _assert_code(result, 0)
    _assert_dataset(out, molecules, expected, {})


def test_cross_split_partner_is_not_a_donor(tmp_path: Path):
    molecules = _molecules()
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    found = None
    by_formula = defaultdict(list)
    for item in molecules:
        by_formula[formulas[item["token"]]].append(item)
    for formula, members in by_formula.items():
        if len(members) < 2 or found is not None:
            continue
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                keep = {members[left]["token"], members[right]["token"]}
                trial = [item for item in molecules if formulas[item["token"]] != formula or item["token"] in keep]
                assigned = _ownership([item["token"] for item in trial])
                if assigned[members[left]["token"]] == assigned[members[right]["token"]]:
                    continue
                trial_formulas = {item["token"]: formulas[item["token"]] for item in trial}
                counts = [
                    len(_constructable(
                        [item["token"] for item in trial if assigned[item["token"]] == split],
                        trial_formulas,
                    ))
                    for split in ("search-dev", "gate-dev")
                ]
                if min(counts) >= 1:
                    found = (trial, members[left]["token"], members[right]["token"])
                    break
            if found is not None:
                break
    assert found is not None
    trial, left, right = found
    trial_formulas = {item["token"]: formulas[item["token"]] for item in trial}
    assigned = _ownership([item["token"] for item in trial])
    oracle_counts = {
        split: len(_constructable(
            [item["token"] for item in trial if assigned[item["token"]] == split],
            trial_formulas,
        ))
        for split in ("search-dev", "gate-dev")
    }
    index, records = _write_case(tmp_path / "in", trial)
    out = tmp_path / "out"
    result = _build(out, index, records, 1)
    _assert_code(result, 0)
    report = json.loads((out / "construction_eligibility_report.json").read_text(encoding="utf-8"))
    assert report["splits"]["search-dev"]["constructable"] == oracle_counts["search-dev"]
    assert report["splits"]["gate-dev"]["constructable"] == oracle_counts["gate-dev"]
    selected = set()
    for split in ("search-dev", "gate-dev"):
        for line in (out / split / "source_groups.jsonl").read_text(encoding="utf-8").splitlines():
            selected.add(json.loads(line)["source_group_id"])
        for line in (out / split / "input_sidecars.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert assigned[row["structure_token"]] == split
            assert assigned[row["candidate_token"]] == split
    assert left not in selected
    assert right not in selected


def test_rebuild_ignores_input_order_and_reuses_a_bound_cache(tmp_path: Path):
    molecules = _molecules()
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    assigned = _ownership([item["token"] for item in molecules])
    groups = min(
        1,
        len(_constructable([token for token, side in assigned.items() if side == "search-dev"], formulas)),
        len(_constructable([token for token, side in assigned.items() if side == "gate-dev"], formulas)),
    )
    assert groups == 1
    first_in = tmp_path / "first-in"
    second_in = tmp_path / "second-in"
    index, records = _write_case(first_in, molecules)
    reversed_molecules = list(reversed(molecules))
    index_b, records_b = _write_case(second_in, reversed_molecules)
    assert index.read_bytes() != index_b.read_bytes()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _build(first, index, records, groups)
    second_result = _build(second, index_b, records_b, groups)
    _assert_code(first_result, 0)
    _assert_code(second_result, 0)
    first_status = json.loads((first / "build_status.json").read_text(encoding="utf-8"))
    second_status = json.loads((second / "build_status.json").read_text(encoding="utf-8"))
    assert first_status["logical_data_sha256"] == second_status["logical_data_sha256"]
    rebuild_out = tmp_path / "rebuild"
    rebuilt = _run(["rebuild", "--from-run", str(first), "--out", str(rebuild_out)])
    _assert_code(rebuilt, 0)
    report = json.loads((first / "reproducibility_report.json").read_text(encoding="utf-8"))
    assert report["equal"] is True
    assert report["fingerprint_cache_reused"] is True
    assert report["sampling_reexecuted"] is True
    assert report["first_logical_data_sha256"] == report["second_logical_data_sha256"]


def test_cache_binding_mismatch_is_refused(tmp_path: Path):
    molecules = _molecules()[:12]
    # The slice may be too small to construct both sides; use the full catalog.
    molecules = _molecules()
    formulas = {item["token"]: _formula(item["token"]) for item in molecules}
    assigned = _ownership([item["token"] for item in molecules])
    groups = min(
        1,
        len(_constructable([token for token, side in assigned.items() if side == "search-dev"], formulas)),
        len(_constructable([token for token, side in assigned.items() if side == "gate-dev"], formulas)),
    )
    index, records = _write_case(tmp_path / "in", molecules)
    first = tmp_path / "first"
    _assert_code(_build(first, index, records, groups), 0)
    manifest_path = first / "cache" / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["eligibility_file_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refused = tmp_path / "refused"
    result = _build(refused, index, records, groups, reuse=first / "cache")
    _assert_code(result, 3)
    status = json.loads((refused / "build_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "refused"
    assert status["blocking_reasons"] == ["cache_binding_mismatch"]
    assert not (refused / "search-dev" / "model_inputs.jsonl").exists()
    assert json.loads((first / "build_status.json").read_text(encoding="utf-8"))["status"] == "fixture_checked"


def test_frozen_output_is_not_overwritten(tmp_path: Path):
    out = tmp_path / "frozen"
    out.mkdir()
    marker = out / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    (out / "build_status.json").write_text('{"status":"prepared_pending_independent_review"}\n', encoding="utf-8")
    molecules = _molecules()[:1]
    index, records = _write_case(tmp_path / "in", molecules)
    result = _build(out, index, records, 1)
    _assert_code(result, 3)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert json.loads((out / "build_status.json").read_text(encoding="utf-8"))["status"] == "prepared_pending_independent_review"


def test_plan_does_not_write_model_inputs(tmp_path: Path):
    molecules = _molecules()[:3]
    index, records = _write_case(tmp_path / "in", molecules)
    out = tmp_path / "out"
    result = _run([
        "plan", "--out", str(out), "--mode", "fixture", "--index", str(index), "--records", str(records),
    ])
    _assert_code(result, 0)
    status = json.loads((out / "plan_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "planned"
    assert not (out / "build_status.json").exists()
    assert not (out / "search-dev").exists()
