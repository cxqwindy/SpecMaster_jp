"""Build the step 02B-3 search-dev and gate-dev datasets.

This module reads the accepted eligibility index and original valid LMDB.
It does not import a model, a predictor, torch, or the old negative CLIs,
and it does not write a run manifest or a prediction file.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

from matching_evaluator import (
    CONFIG_SCHEMA,
    DATASET_SCHEMA,
    EVALUATOR_VERSION,
    GOLD_SCHEMA,
    INVALID_POLICY,
    PRIMARY_METRIC,
    SOURCE_GROUP_ID,
    SOURCE_GROUP_VERSION,
    canonical_json,
    check_gold_sidecar,
    dump_json,
    load_config,
    load_json,
    prepare_dataset,
    sha256_text,
    source_group_for_parent,
    validate_task,
)
from nmr_lossless_task import SCHEMA_VERSION as TASK_SCHEMA
from recipe.nmr_matching_agent.task_input import TASK_FIELDS
from step_02a_fingerprints import (
    RDKIT_VERSION,
    identity_token,
    shift_sequence_hash,
    spectrum_views,
    structure_keys,
)

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "docs" / "rsi" / "step_02b3" / "build_config.json"
SPEC = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

PROTOCOL = str(SPEC["protocol"])
SEED = int(SPEC["seed"])
GROUPS = int(SPEC["groups_per_split"])
OWNERSHIP_DOMAIN = str(SPEC["ownership_domain"])
SOURCE_DOMAIN = str(SPEC["source_domain"])
SAME_FORMULA_DOMAIN = str(SPEC["same_formula_domain"])
RANDOM_DOMAIN = str(SPEC["random_domain"])
OUTPUT_DOMAIN = str(SPEC["output_domain"])
RECORD_DOMAIN = str(SPEC["record_domain"])
FP_RADIUS = int(SPEC["fingerprint"]["radius"])
FP_BITS = int(SPEC["fingerprint"]["n_bits"])
FP_CHIRALITY = bool(SPEC["fingerprint"]["include_chirality"])
RECHECK_LIMIT = int(SPEC["full_fingerprint_recheck_limit"])
SPLITS = ("search-dev", "gate-dev")
SUCCESS_REAL = "prepared_pending_independent_review"
SUCCESS_FIXTURE = "fixture_checked"
SHANGHAI = timezone(timedelta(hours=8))
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_BITS,
    includeChirality=FP_CHIRALITY,
)
SPECTRUM_FIELDS = ("c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency")
THEORETICAL_FIELDS = ("atoms", "coordinates")
TASK_INPUT_KEYS = (
    "c_nmr_peaks",
    "experimental_13c_peaks",
    "h_nmr_peaks",
    "experimental_1h_peaks",
    "nmr_solvent",
    "solvent",
    "nmr_frequency",
    "nmr_freq",
    "experimental_total_nH",
    "nmr_frequency_nucleus",
)
ROLES = (
    ("pos", 1, None),
    ("neg/same_formula", 0, "neg/same_formula"),
    ("neg/similar_structure", 0, "neg/similar_structure"),
    ("neg/random", 0, "neg/random"),
)
OA_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL")
LF_ROOT = Path("/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory")
PRESERVED = {
    "open_agentrl": {
        "root": OA_ROOT,
        "expected_head": "8285ddfdabd210815086c577600b7fe038dcb6bd",
        "files": {
            "recipe/demystify/grpo_tcr_qwen2_7b.sh": "e7024f5a398fdc66407e7511c4d30532962a7bdc2b3b86381e7728e26cbabd88",
            "recipe/demystify/qwen2_7b_sft.sh": "6c63edb112f17c947ee3bc4a0d182875e3811b268c7c7df6f5f8cebf1c49b904",
            "verl/experimental/agent_loop/agent_loop.py": "ea2b23aa91469501766e96593e47a266ba8dacbb1a02b6163eecb9008c00c19a",
            "verl/experimental/agent_loop/tool_agent_loop.py": "06362e9c54c2eb96887eae3aa3c84ea9c673e0cb5dbf9827fae219b380b90a1a",
            "verl/model_merger/base_model_merger.py": "0929e92761dbe9ab242b0a0ca145abb6f2c1b447bf80973c16f7a2a0d8e1e7d9",
            "verl/model_merger/fsdp_model_merger.py": "3dd04b312e01e1a6bd353042034a309ff815ab9c23240cc1e41263ba162ff6a9",
            "verl/protocol.py": "3e5947ebdfca69ac0803d752daaf0c3198c541f5489f43ff79673252ad212e84",
            "verl/trainer/config/_generated_ppo_trainer.yaml": "69c4ca285176bdc424be6cf7380856d04b62b189bf7c0aa685457ce1b9d474ca",
            "verl/trainer/config/rollout/rollout.yaml": "8c93596ee3e1e3b679e5355519e797e20e7aecb11cb6602a1372ca5d41924753",
            "verl/trainer/main_ppo.py": "94df335617b5db7f2b33018d2a488c27163cf68605542c4b0557bae9b145587e",
            "verl/trainer/ppo/core_algos.py": "a6ec0ea3ee1621edc137461399c3ab60ea9fc5686a32a75494ea499085f7f314",
            "verl/trainer/ppo/metric_utils.py": "9b82fc214c20595e46ce6166bee1d1262aef489c8639112011c4fa7a016c43b9",
            "verl/trainer/ppo/ray_trainer.py": "9b474d6b04dca8130c75ce72cf32410a2f6f5c351fb935849635aba30dcb43c9",
            "verl/utils/dataset/multiturn_sft_dataset.py": "39dd4c0e362cdea2b5f4cbcadbae1b2197c7266e221d6e26f6931c1c59a16759",
            "verl/workers/config/rollout.py": "66957a09425f75fe32441ee11d6aabca05b4ed80e9130912894153b961aa9317",
            "verl/workers/fsdp_workers.py": "0fb3db49143838d909c86c3d077bb03aa1ae1fa922a53247e0d4018b0edd07e7",
            "verl/workers/reward_manager/__init__.py": "03ae3fd97de19c3cf7860e34d35210ccc8272e7cdc16752aee9b54c7bc4dee26",
            "verl/workers/rollout/vllm_rollout/vllm_async_server.py": "347355a9a924cce8aa8b4d62d5d1e5f2fe8094577f2828d6cad657474b67f5cd",
        },
    },
    "llama_factory": {
        "root": LF_ROOT,
        "expected_head": "cabe59a343c89e343dd731798ea21f6fdf9989ca",
        "files": {
            "data/dataset_info.json": "a9e7da2c3a373b905dbfebfc3d92b898d3f07be80102a8dc062529883f8eeb38",
            "src/llamafactory/data/template.py": "7bcc160a2f84408143c3658c0f06a1b304a8bc61578f2d3682e0c808bcbef1f7",
        },
    },
}


class DirectoryFrozen(Exception):
    pass


class StopBuild(Exception):
    def __init__(self, exit_code: int, status: str, reason: str, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.exit_code = exit_code
        self.status = status
        self.reason = reason
        self.details = details or {}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def now_iso() -> str:
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def domain_rank(domain: str, *parts: Any) -> str:
    """SHA-256 of canonical JSON. This is not Python hash()."""
    return sha256_text(canonical_json([domain, *parts]))


def better_similarity(token: str, sim: float, best_token: str | None, best_sim: float | None):
    if best_sim is None or sim > best_sim or (sim == best_sim and token < best_token):
        return token, sim
    return best_token, best_sim


def tanimoto_int(left: int, right: int) -> float:
    inter = (left & right).bit_count()
    union = (left | right).bit_count()
    if union == 0:
        return 1.0
    return inter / union


def fp_to_int(fp: Any) -> int:
    value = 0
    for bit in fp.GetOnBits():
        value |= 1 << bit
    return value


def int_to_hex(value: int) -> str:
    return f"{value:0{FP_BITS // 4}x}"


def hex_to_int(text: str) -> int:
    width = FP_BITS // 4
    if len(text) != width or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("fingerprint hex does not match the bound width")
    return int(text, 16)


def fingerprint_self_check() -> dict[str, Any]:
    """Lock the generator to the legacy Morgan call and to an exact popcount."""
    from rdkit.Chem import rdMolDescriptors as legacy_descriptors

    smiles = ("CCO", "C[C@H](O)CC", "c1ccccc1")
    legacy = []
    current = []
    for item in smiles:
        mol = Chem.MolFromSmiles(item)
        old = legacy_descriptors.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_BITS, useChirality=FP_CHIRALITY,
        )
        new = GENERATOR.GetFingerprint(mol)
        if old.ToBitString() != new.ToBitString():
            raise StopBuild(3, "refused", "fingerprint_self_check", {"smiles_check": item})
        legacy.append(old)
        current.append(new)
    stereo_a = GENERATOR.GetFingerprint(Chem.MolFromSmiles("C[C@H](O)CC"))
    stereo_b = GENERATOR.GetFingerprint(Chem.MolFromSmiles("C[C@@H](O)CC"))
    if DataStructs.TanimotoSimilarity(stereo_a, stereo_b) != 1.0:
        raise StopBuild(3, "refused", "fingerprint_self_check", {"stereo_tanimoto": "not_1"})
    pairs = ((0, 1), (0, 2), (1, 2))
    for left, right in pairs:
        rdkit_sim = DataStructs.TanimotoSimilarity(current[left], current[right])
        exact_sim = tanimoto_int(fp_to_int(current[left]), fp_to_int(current[right]))
        if rdkit_sim != exact_sim:
            raise StopBuild(3, "refused", "fingerprint_self_check", {"pair": [left, right]})
    return {"legacy_bit_identity": True, "exact_popcount_identity": True, "stereo_tanimoto": 1.0}


def assign_ownership(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Even ranks are search-dev. An odd count leaves the extra token there."""
    if len(tokens) != len(set(tokens)):
        raise StopBuild(2, "incomplete", "duplicate_parent_token")
    ranked = sorted(tokens, key=lambda token: (domain_rank(OWNERSHIP_DOMAIN, SEED, token), token))
    assigned = []
    for index, token in enumerate(ranked):
        split = "search-dev" if index % 2 == 0 else "gate-dev"
        assigned.append((token, split))
    return assigned


def pick_min_hash(domain: str, source: str, tokens: Sequence[str]) -> str | None:
    best = None
    best_rank = None
    for token in tokens:
        rank = domain_rank(domain, SEED, source, token)
        if best_rank is None or rank < best_rank or (rank == best_rank and token < best):
            best = token
            best_rank = rank
    return best


def pick_max_similarity(scored: Sequence[tuple[str, float]]) -> tuple[str | None, float | None]:
    best_token = None
    best_sim = None
    for token, sim in scored:
        best_token, best_sim = better_similarity(token, sim, best_token, best_sim)
    return best_token, best_sim


def formula_result(mol: Any, raw: Mapping[str, Any]) -> dict[str, Any]:
    present = "molecular_formula" in raw
    raw_value = raw.get("molecular_formula") if present else None
    stored = raw_value if isinstance(raw_value, str) else None
    try:
        calculated = rdMolDescriptors.CalcMolFormula(mol)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": type(exc).__name__,
            "calculated": None,
            "raw": stored,
            "relation": "failed",
        }
    if not present:
        relation = "raw_missing"
    elif not isinstance(raw_value, str):
        relation = "raw_non_string"
    elif raw_value == calculated:
        relation = "match"
    else:
        relation = "differ"
    return {
        "status": "ok",
        "reason": None,
        "calculated": calculated,
        "raw": stored,
        "relation": relation,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value
    name = type(value).__name__
    if name.startswith("float"):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return number
    if name.startswith("int") or name.startswith("uint"):
        return int(value)
    raise TypeError(name)


def presence_state(raw: Mapping[str, Any], key: str) -> str:
    if key not in raw:
        return "missing"
    value = raw.get(key)
    if value is None:
        return "null"
    if value == []:
        return "empty_list"
    return "present"


def signal(peaks: Any) -> bool:
    if not isinstance(peaks, list):
        return False
    for peak in peaks:
        if isinstance(peak, bool):
            continue
        if isinstance(peak, (int, float)):
            return math.isfinite(float(peak))
        if isinstance(peak, dict):
            for key in ("delta (ppm)", "delta", "centroid", "rangeMin", "rangeMax"):
                number = peak.get(key)
                if isinstance(number, bool):
                    continue
                if isinstance(number, (int, float)) and math.isfinite(float(number)):
                    return True
    return False


def nucleus_of(raw: Mapping[str, Any]) -> str:
    carbon = signal(raw.get("c_nmr_peaks") if "c_nmr_peaks" in raw else None)
    proton = signal(raw.get("h_nmr_peaks") if "h_nmr_peaks" in raw else None)
    if carbon and proton:
        return "both"
    if carbon:
        return "c_only"
    if proton:
        return "h_only"
    return "neither"


def nh_counts(raw: Mapping[str, Any]) -> dict[str, int]:
    counts = {"missing": 0, "null": 0, "zero": 0, "known": 0}
    peaks = raw.get("h_nmr_peaks") if "h_nmr_peaks" in raw else None
    if not isinstance(peaks, list):
        return counts
    for peak in peaks:
        if not isinstance(peak, dict) or "nH" not in peak:
            counts["missing"] += 1
        elif peak.get("nH") is None:
            counts["null"] += 1
        elif peak.get("nH") == 0:
            counts["zero"] += 1
        else:
            counts["known"] += 1
    return counts


def split_theoretical(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    chemical: dict[str, Any] = {}
    theoretical: dict[str, str] = {}
    for key, value in raw.items():
        if key in THEORETICAL_FIELDS or type(value).__name__ == "ndarray":
            theoretical[str(key)] = type(value).__name__
            continue
        if str(key).startswith("_fixture_"):
            continue
        chemical[str(key)] = value
    return chemical, theoretical


def spectrum_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {field: raw.get(field) for field in SPECTRUM_FIELDS if field in raw}


def projection_issues(chemical: Mapping[str, Any], normalized: Mapping[str, Any]) -> list[str]:
    issues = []
    if set(normalized) != set(TASK_FIELDS):
        issues.append("task_fields")
    for key in THEORETICAL_FIELDS:
        if key in normalized:
            issues.append("theoretical_field_in_task")
    original_c = chemical.get("c_nmr_peaks") if "c_nmr_peaks" in chemical else None
    if not isinstance(original_c, list) or original_c == []:
        if normalized.get("c_nmr_peaks") != []:
            issues.append("c_runtime_not_empty")
    else:
        expected = []
        for peak in original_c:
            expected.append(peak.get("delta (ppm)") if isinstance(peak, dict) else peak)
        got = []
        for peak in normalized.get("c_nmr_peaks") or []:
            got.append(peak.get("delta (ppm)") if isinstance(peak, dict) else peak)
        if got != expected:
            issues.append("c_delta_mismatch")
    original_h = chemical.get("h_nmr_peaks") if "h_nmr_peaks" in chemical else None
    normalized_h = normalized.get("h_nmr_peaks") or []
    if not isinstance(original_h, list) or original_h == []:
        if normalized_h != []:
            issues.append("h_runtime_not_empty")
    elif len(original_h) != len(normalized_h):
        issues.append("h_length_mismatch")
    else:
        for src, dst in zip(original_h, normalized_h):
            if not isinstance(dst, dict):
                issues.append("h_projection_type")
                break
            if isinstance(src, dict):
                for key in ("centroid", "delta", "nH", "j_values", "category", "rangeMin", "rangeMax"):
                    if key in src and dst.get(key) != src.get(key):
                        issues.append("h_value_mismatch")
                        break
                    if key == "nH" and key not in src and key in dst:
                        issues.append("h_nh_invented")
                        break
            else:
                if dst.get("delta") != src or dst.get("centroid") != src or dst.get("nH") is not None:
                    issues.append("h_scalar_mismatch")
                    break
    if "nmr_frequency" in chemical and isinstance(chemical.get("nmr_frequency"), str) and chemical.get("nmr_frequency"):
        if normalized.get("nmr_frequency") != chemical.get("nmr_frequency"):
            issues.append("frequency_changed")
    if "nmr_frequency_nucleus" not in chemical and normalized.get("nmr_frequency_nucleus") != "unknown":
        issues.append("frequency_nucleus_inferred")
    if "experimental_total_nH" in chemical:
        if normalized.get("experimental_total_nH") != chemical.get("experimental_total_nH"):
            issues.append("total_nh_changed")
    elif normalized.get("experimental_total_nH") is not None:
        issues.append("total_nh_invented")
    if "nmr_solvent" in chemical and isinstance(chemical.get("nmr_solvent"), str) and chemical.get("nmr_solvent"):
        if normalized.get("nmr_solvent") != chemical.get("nmr_solvent"):
            issues.append("solvent_changed")
    return issues


def task_payload(chemical: Mapping[str, Any], candidate_smiles: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"candidate_smiles": candidate_smiles}
    for key in TASK_INPUT_KEYS:
        if key in chemical:
            payload[key] = chemical[key]
    return payload


def heavy_bin(value: int) -> str:
    if value <= 10:
        return "0-10"
    if value <= 20:
        return "11-20"
    if value <= 30:
        return "21-30"
    if value <= 40:
        return "31-40"
    return "41+"


def coverage_of(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nuclei: Counter[str] = Counter()
    bins: Counter[str] = Counter()
    formulas = set()
    scaffolds = set()
    connectivity = set()
    missing_scaffold = 0
    for record in records:
        nuclei[str(record["nucleus"])] += 1
        formulas.add(record["formula"])
        scaffolds.add(record["scaffold"])
        connectivity.add(record["connectivity"])
        if record["scaffold"] is None:
            missing_scaffold += 1
        bins[heavy_bin(int(record["heavy_atoms"]))] += 1
    return {
        "structures": len(records),
        "both_nuclei": nuclei["both"],
        "c_only": nuclei["c_only"],
        "h_only": nuclei["h_only"],
        "neither": nuclei["neither"],
        "unique_formulas": len(formulas),
        "unique_scaffolds": len(scaffolds),
        "unique_connectivity": len(connectivity),
        "scaffold_missing": missing_scaffold,
        "heavy_atom_bins": {key: bins[key] for key in ("0-10", "11-20", "21-30", "31-40", "41+") if bins[key]},
    }


def constructability(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    usable = donor_pool(records)
    by_formula: dict[str, list[str]] = defaultdict(list)
    for token, record in usable.items():
        by_formula[str(record["formula"])].append(token)
    total = len(usable)
    reasons = {}
    for token, record in records.items():
        if token not in usable:
            if record.get("formula_status") != "ok":
                reasons[token] = "formula_failed"
            elif record.get("nucleus") == "neither":
                reasons[token] = "no_valid_spectrum_on_reread"
            else:
                reasons[token] = "fingerprint_missing"
            continue
        formula = str(record["formula"])
        same = len(by_formula[formula]) - 1
        different = total - len(by_formula[formula])
        if same < 1:
            reasons[token] = "no_same_formula_donor"
        elif different < 2:
            reasons[token] = "fewer_than_two_different_formula_donors"
        else:
            reasons[token] = "constructable"
    return reasons


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rows.append(json.loads(raw.decode("utf-8")))
    return rows


def git_state(path: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], text=True,
        ).strip()
        return {"head": head, "branch": branch, "error": None}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "branch": None, "error": type(exc).__name__}


def preservation_report() -> dict[str, Any]:
    report: dict[str, Any] = {"specmaster": git_state(REPO), "trees": {}, "mismatch_count": 0}
    for name, spec in PRESERVED.items():
        root = spec["root"]
        state = git_state(root)
        files = []
        for relative, expected in spec["files"].items():
            file_path = root / relative
            if not file_path.is_file():
                observed = None
                match = False
            else:
                observed = file_sha256(file_path)
                match = observed == expected
            if not match or state.get("head") != spec["expected_head"]:
                report["mismatch_count"] += 0 if match else 1
            files.append({"path": relative, "sha256": observed, "expected": expected, "match": match})
        head_match = state.get("head") == spec["expected_head"]
        if not head_match:
            report["mismatch_count"] += 1
        report["trees"][name] = {
            "head": state.get("head"),
            "expected_head": spec["expected_head"],
            "head_match": head_match,
            "branch": state.get("branch"),
            "files": files,
        }
    return report


def version_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("step_02b3_build_dev.py"), SPEC_PATH):
        data = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def rows_identity(digests: list[bytes]) -> str:
    ordered = sorted(digests)
    digest = hashlib.sha256()
    digest.update(str(len(ordered)).encode("ascii"))
    digest.update(b"\n")
    for item in ordered:
        digest.update(item)
    return digest.hexdigest()


def token_set_hash(tokens: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(tokens)) + "\n")


def ownership_set_hash(rows: Sequence[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for split, token, register in rows:
        digest.update(split.encode("utf-8"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(register.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def scan_index(path: Path, mode: str) -> dict[str, Any]:
    if mode == "real":
        size = path.stat().st_size
        if size != int(SPEC["eligibility_index_bytes"]):
            raise StopBuild(2, "incomplete", "eligibility_index_size_mismatch", {"bytes": size})
    valid_rows = []
    pool_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    train_tokens: set[str] = set()
    test_tokens: set[str] = set()
    exposure_tokens: set[str] = set()
    test_gt_tokens: set[str] = set()
    test_candidate_tokens: set[str] = set()
    seen_valid: dict[str, list[str]] = {}
    duplicates = []
    violations = Counter()
    digests = []
    file_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            file_digest.update(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                continue
            digests.append(hashlib.sha256(stripped).digest())
            row = json.loads(stripped.decode("utf-8"))
            source = str(row.get("source_lmdb"))
            row_counts[source] += 1
            reasons = row.get("exposure_reasons") or []
            quality = row.get("quality_reasons") or []
            if reasons:
                for token in (row.get("candidate_token"), row.get("gt_token")):
                    if isinstance(token, str) and token:
                        exposure_tokens.add(token)
                if "test_gt_structure" in reasons:
                    for token in (row.get("candidate_token"), row.get("gt_token")):
                        if isinstance(token, str) and token:
                            test_gt_tokens.add(token)
                if "test_candidate_structure" in reasons:
                    for token in (row.get("candidate_token"), row.get("gt_token")):
                        if isinstance(token, str) and token:
                            test_candidate_tokens.add(token)
            if row.get("in_auto_positive_pool") is True:
                pool_counts[source] += 1
                if mode == "real":
                    if row.get("protocol_version") != SPEC["eligibility_protocol"]:
                        violations["protocol"] += 1
                    if row.get("evidence_bucket") != "remaining" or row.get("quality_status") != "not_rejected":
                        violations["bucket"] += 1
                    if reasons or quality:
                        violations["reasons_on_pool_row"] += 1
                    expected_index = SPEC["index_sha256"].get(f"lmdb_{source}.index.jsonl")
                    if expected_index and row.get("source_index_sha256") != expected_index:
                        violations["source_index_sha256"] += 1
                candidate = row.get("candidate_token")
                gt = row.get("gt_token")
                if source == "train.lmdb":
                    for token in (candidate, gt):
                        if isinstance(token, str) and token.startswith("canonical:"):
                            train_tokens.add(token)
                elif source == "test.lmdb":
                    for token in (candidate, gt):
                        if isinstance(token, str) and token.startswith("canonical:"):
                            test_tokens.add(token)
                elif source == "valid.lmdb":
                    if not isinstance(candidate, str) or not candidate.startswith("canonical:"):
                        violations["noncanonical_valid"] += 1
                    if candidate != gt:
                        violations["parent_token_disagrees"] += 1
                    locator = f"{source}:{row.get('source_key')}"
                    if isinstance(candidate, str):
                        if candidate in seen_valid:
                            duplicates.append({"locator": locator, "also": seen_valid[candidate][0]})
                        else:
                            seen_valid[candidate] = [locator]
                    valid_rows.append({
                        "source_lmdb": source,
                        "source_key": str(row.get("source_key")),
                        "candidate_token": candidate,
                        "gt_token": gt,
                        "spectrum_norm_hash": row.get("spectrum_norm_hash"),
                        "shift_sequence_hash": row.get("shift_sequence_hash"),
                    })
            if line_number % 200000 == 0:
                log(f"index rows read {line_number}")
    if mode == "real":
        file_hash = file_digest.hexdigest()
        if file_hash != SPEC["eligibility_index_sha256"]:
            raise StopBuild(2, "incomplete", "eligibility_index_hash_mismatch", {"sha256": file_hash})
        for name, expected in SPEC["auto_positive_pool"].items():
            if pool_counts[name] != int(expected):
                violations[f"pool_count_{name}"] += 1
    identity = rows_identity(digests)
    return {
        "file_sha256": file_digest.hexdigest(),
        "rows_identity_sha256": identity,
        "valid_rows": valid_rows,
        "pool_counts": dict(pool_counts),
        "row_counts": dict(row_counts),
        "train_tokens": train_tokens,
        "test_tokens": test_tokens,
        "exposure_tokens": exposure_tokens,
        "test_gt_tokens": test_gt_tokens,
        "test_candidate_tokens": test_candidate_tokens,
        "duplicates": duplicates,
        "violations": dict(violations),
    }


def asset_manifest(mode: str) -> dict[str, Any]:
    if mode != "real":
        return {"mode": "fixture", "binding": "fixture records are not the original LMDB"}
    root = Path(SPEC["lmdb_root"])
    audit = Path(SPEC["audit_02a"])
    files = {}
    mismatches = []
    for name, expected in SPEC["lmdb_metadata"].items():
        path = root / name
        stat = path.stat()
        observed = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        match = observed["bytes"] == expected["bytes"] and observed["mtime_ns"] == expected["mtime_ns"]
        files[name] = {"path": str(path), "observed": observed, "expected": expected, "match": match}
        if not match:
            mismatches.append(name)
    indexes = {}
    for name, expected in SPEC["index_sha256"].items():
        path = audit / name
        observed = file_sha256(path)
        indexes[name] = {"path": str(path), "sha256": observed, "expected": expected, "match": observed == expected}
        if observed != expected:
            mismatches.append(name)
    if mismatches:
        raise StopBuild(2, "incomplete", "asset_metadata_mismatch", {"files": mismatches})
    return {
        "mode": "real",
        "lmdb": files,
        "compact_indexes": indexes,
        "eligibility_index": SPEC["eligibility_index"],
        "eligibility_index_sha256": SPEC["eligibility_index_sha256"],
        "note": (
            "Train, valid, and test are the 02A partition of all.lmdb. "
            "all.lmdb is not opened and is not a donor pool. "
            "These pool counts are not a claim that a base model never saw the structures."
        ),
    }


def load_fixture_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    found = {}
    for row in read_jsonl(path):
        found[(str(row["source_lmdb"]), str(row["source_key"]))] = row["raw"]
    return found


def bits_to_int(text: str) -> int:
    if len(text) != FP_BITS or any(bit not in "01" for bit in text):
        raise ValueError("fixture_fp_bits")
    value = 0
    for bit_index, bit in enumerate(text):
        if bit == "1":
            value |= 1 << bit_index
    return value


def int_to_bitvect(value: int) -> Any:
    fp = DataStructs.ExplicitBitVect(FP_BITS)
    remaining = value
    bit = 0
    while remaining and bit < FP_BITS:
        if remaining & 1:
            fp.SetBit(bit)
        remaining >>= 1
        bit += 1
    return fp


def expected_fp_int(record: Mapping[str, Any]) -> int:
    if record.get("fixture_bits"):
        return bits_to_int(str(record["fixture_bits"]))
    return fresh_fp_int(record)


def decode_pool(scan: Mapping[str, Any], mode: str, records_path: Path | None, compute_fp: bool) -> dict[str, Any]:
    fixtures = load_fixture_records(records_path) if records_path is not None else {}
    decoded: dict[str, dict[str, Any]] = {}
    failures = []
    env = None
    if mode == "real":
        import lmdb
        import pickle
        env = lmdb.open(
            str(Path(SPEC["lmdb_root"]) / "valid.lmdb"),
            subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
        )
        cursor_raw = {}
        with env.begin(buffers=True) as txn:
            for key, value in txn.cursor():
                try:
                    text = bytes(key).decode("utf-8")
                except UnicodeError:
                    text = bytes(key).decode("utf-8", "replace")
                cursor_raw[text] = bytes(value)
    else:
        pickle = None
        cursor_raw = {}
    try:
        for index, row in enumerate(scan["valid_rows"], 1):
            key = (row["source_lmdb"], row["source_key"])
            try:
                fixture_bits = None
                if mode == "fixture":
                    if key not in fixtures:
                        raise KeyError("missing fixture record")
                    raw_input = dict(fixtures[key])
                    if "_fixture_fp_bits" in raw_input:
                        fixture_bits = str(raw_input.pop("_fixture_fp_bits"))
                    raw_bytes = canonical_json(json_safe(raw_input)).encode("utf-8")
                    codec = "canonical_json"
                    chemical, theoretical = split_theoretical(raw_input)
                else:
                    if row["source_key"] not in cursor_raw:
                        raise KeyError("missing lmdb record")
                    raw_bytes = cursor_raw[row["source_key"]]
                    codec = "pickle"
                    loaded = pickle.loads(raw_bytes)
                    if not isinstance(loaded, dict):
                        raise TypeError("lmdb record is not a dict")
                    if any(str(name).startswith("_fixture_") for name in loaded):
                        raise StopBuild(3, "refused", "fixture_control_in_real_record")
                    chemical, theoretical = split_theoretical(loaded)
                smiles = chemical.get("smiles") if isinstance(chemical.get("smiles"), str) else None
                declared = chemical.get("smiles_canonical") if isinstance(chemical.get("smiles_canonical"), str) else None
                identity = smiles or declared
                keys = structure_keys(identity)
                token = identity_token(keys)
                gt_token = identity_token(structure_keys(declared))
                spectrum = spectrum_identity(chemical)
                viewed = spectrum_views(spectrum)
                shift = shift_sequence_hash(spectrum)
                if token != row["candidate_token"] or gt_token != row["gt_token"]:
                    raise ValueError("token_mismatch")
                if viewed.get("normalized_hash") != row["spectrum_norm_hash"]:
                    raise ValueError("spectrum_hash_mismatch")
                if shift != row["shift_sequence_hash"]:
                    raise ValueError("shift_hash_mismatch")
                if keys.get("status") != "parsed" or not keys.get("canonical_isomeric"):
                    raise ValueError("unparsed_on_reread")
                mol = Chem.MolFromSmiles(identity)
                if mol is None:
                    raise ValueError("mol_parse_failed")
                formula = formula_result(mol, chemical)
                if fixture_bits is not None:
                    fp_int = bits_to_int(fixture_bits) if compute_fp else None
                elif compute_fp:
                    fp_int = fp_to_int(GENERATOR.GetFingerprint(mol))
                else:
                    fp_int = None
                decoded[token] = {
                    "token": token,
                    "source_lmdb": row["source_lmdb"],
                    "source_key": row["source_key"],
                    "formula": formula["calculated"],
                    "formula_status": formula["status"],
                    "formula_relation": formula["relation"],
                    "formula_reason": formula["reason"],
                    "raw_formula": formula["raw"],
                    "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                    "scaffold": keys.get("scaffold"),
                    "connectivity": keys.get("connectivity"),
                    "canonical_smiles": keys.get("canonical_isomeric"),
                    "raw_smiles": smiles,
                    "declared_smiles": declared,
                    "nucleus": nucleus_of(chemical),
                    "nh": nh_counts(chemical),
                    "presence": {field: presence_state(chemical, field) for field in (
                        "c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency", "experimental_total_nH",
                    )},
                    "spectrum_norm_hash": viewed.get("normalized_hash"),
                    "spectrum_raw_hash": viewed.get("raw_content_hash"),
                    "shift_sequence_hash": shift,
                    "raw_record_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "raw_record_codec": codec,
                    "theoretical_fields": theoretical,
                    "chemical": chemical,
                    "fixture_bits": fixture_bits,
                    "fp_int": fp_int,
                }
            except StopBuild:
                raise
            except Exception as exc:
                failures.append({
                    "source_lmdb": row["source_lmdb"],
                    "source_key": row["source_key"],
                    "reason": type(exc).__name__,
                })
            if index % 5000 == 0:
                log(f"decoded {index}")
    finally:
        if env is not None:
            env.close()
    return {"decoded": decoded, "failures": failures}


def cache_manifest_body(scan: Mapping[str, Any], own_hash: str, tokens: Sequence[str], structures_sha: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "eligibility_file_sha256": scan["file_sha256"],
        "ownership_set_sha256": own_hash,
        "rows_identity_sha256": scan["rows_identity_sha256"],
        "rdkit_version": RDKIT_VERSION,
        "fingerprint": {
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
            "include_chirality": FP_CHIRALITY,
            "generator": SPEC["fingerprint"]["generator"],
        },
        "mapping_order": "sorted_structure_token",
        "count": len(tokens),
        "token_set_sha256": token_set_hash(tokens),
        "structures_sha256": structures_sha,
    }


def write_cache(cache_dir: Path, scan: Mapping[str, Any], own_hash: str, decoded: Mapping[str, Mapping[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens = sorted(decoded)
    structures = cache_dir / "structures.jsonl"
    with structures.open("w", encoding="utf-8") as handle:
        for token in tokens:
            handle.write(canonical_json({"fp_int_hex": int_to_hex(int(decoded[token]["fp_int"])), "token": token}))
            handle.write("\n")
    body = cache_manifest_body(scan, own_hash, tokens, file_sha256(structures))
    dump_json(cache_dir / "cache_manifest.json", body)


def load_cache(cache_dir: Path, scan: Mapping[str, Any], own_hash: str, decoded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    manifest_path = cache_dir / "cache_manifest.json"
    structures_path = cache_dir / "structures.jsonl"
    if not manifest_path.is_file() or not structures_path.is_file():
        raise StopBuild(3, "refused", "cache_missing")
    manifest = load_json(manifest_path)
    tokens = sorted(decoded)
    structures_sha = file_sha256(structures_path)
    expected = cache_manifest_body(scan, own_hash, tokens, structures_sha)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise StopBuild(3, "refused", "cache_binding_mismatch", {"field": key})
    loaded = {}
    previous = ""
    with structures_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            token = row["token"]
            if token < previous:
                raise StopBuild(3, "refused", "cache_mapping_order")
            previous = token
            loaded[token] = hex_to_int(row["fp_int_hex"])
    if list(loaded) != tokens:
        raise StopBuild(3, "refused", "cache_token_set_mismatch")
    positions = sorted({0, 1, len(tokens) // 2, max(0, len(tokens) - 2), len(tokens) - 1})
    if len(tokens) <= RECHECK_LIMIT:
        positions = list(range(len(tokens)))
    mismatches = 0
    for position in positions:
        token = tokens[position]
        if expected_fp_int(decoded[token]) != loaded[token]:
            mismatches += 1
    if mismatches:
        raise StopBuild(3, "refused", "cache_fingerprint_mismatch", {"mismatches": mismatches})
    for token, value in loaded.items():
        decoded[token]["fp_int"] = value
    return {"reused": True, "spot_check_count": len(positions), "spot_check_mismatches": 0, "mapping_order": "sorted_structure_token"}


def fresh_fp_int(record: Mapping[str, Any]) -> int:
    mol = Chem.MolFromSmiles(str(record["canonical_smiles"]))
    if mol is None:
        raise StopBuild(2, "incomplete", "canonical_reparse_failed")
    return fp_to_int(GENERATOR.GetFingerprint(mol))


def recheck_selected(decoded: Mapping[str, Mapping[str, Any]], selected: Sequence[str], reused: bool) -> dict[str, Any]:
    mismatches = 0
    for token in selected:
        if expected_fp_int(decoded[token]) != int(decoded[token]["fp_int"]):
            mismatches += 1
    if mismatches:
        reason = "cache_fingerprint_mismatch" if reused else "canonical_fp_mismatch"
        code = 3 if reused else 2
        status = "refused" if reused else "incomplete"
        raise StopBuild(code, status, reason, {"mismatches": mismatches})
    return {"selected_rechecked": len(selected), "mismatches": 0}


def donor_pool(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        token: record
        for token, record in records.items()
        if record.get("formula_status") == "ok" and record.get("nucleus") != "neither" and record.get("fp_int") is not None
    }


def choose_negatives(source: str, pool: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    formula = pool[source]["formula"]
    same = sorted(
        token for token, record in pool.items()
        if token != source and record["formula"] == formula
    )
    different = sorted(token for token, record in pool.items() if record["formula"] != formula)
    same_choice = pick_min_hash(SAME_FORMULA_DOMAIN, source, same)
    query = int(pool[source]["fp_int"])
    best_token = None
    best_sim = None
    for token in different:
        sim = tanimoto_int(query, int(pool[token]["fp_int"]))
        best_token, best_sim = better_similarity(token, sim, best_token, best_sim)
    random_pool = [token for token in different if token != best_token]
    random_choice = pick_min_hash(RANDOM_DOMAIN, source, random_pool)
    if same_choice is None or best_token is None or random_choice is None:
        return {"ok": False, "reason": "negative_selection_failed"}
    chosen = [same_choice, best_token, random_choice]
    if len(set(chosen)) != 3 or source in chosen:
        return {"ok": False, "reason": "negative_tokens_not_distinct"}
    return {
        "ok": True,
        "same_formula": same_choice,
        "neighbor": best_token,
        "neighbor_tanimoto": best_sim,
        "random": random_choice,
    }


def relation(source: Mapping[str, Any], donor: Mapping[str, Any]) -> str:
    if source["connectivity"] and source["connectivity"] == donor["connectivity"]:
        return "same_connectivity_ignoring_stereo"
    return "different_connectivity"


def make_record_id(record: Mapping[str, Any], role: str, candidate_token: str) -> str:
    return domain_rank(RECORD_DOMAIN, PROTOCOL, record["source_lmdb"], record["source_key"], role, candidate_token)


def build_group(source_token: str, choices: Mapping[str, Any], pool: Mapping[str, Mapping[str, Any]], split: str) -> dict[str, Any]:
    source = pool[source_token]
    parent = str(source["canonical_smiles"])
    group_id = source_group_for_parent(parent)
    if group_id != source_token:
        return {"ok": False, "reason": "source_group_mismatch", "source_key": source["source_key"]}
    roles = {
        "pos": source_token,
        "neg/same_formula": choices["same_formula"],
        "neg/similar_structure": choices["neighbor"],
        "neg/random": choices["random"],
    }
    rows = []
    issues = []
    for role, label, negative_type in ROLES:
        donor = pool[roles[role]]
        candidate = str(donor["canonical_smiles"])
        if source_group_for_parent(candidate) != donor["token"]:
            issues.append("donor_token_mismatch")
        payload = task_payload(source["chemical"], candidate)
        normalized, task_sha, task_errors = validate_task(payload)
        if task_errors or normalized is None or task_sha is None:
            issues.append("normalize_rejected")
            break
        issues.extend(projection_issues(source["chemical"], normalized))
        tanimoto = None
        if role != "pos":
            tanimoto = tanimoto_int(int(source["fp_int"]), int(donor["fp_int"]))
        if role == "neg/similar_structure" and tanimoto != choices["neighbor_tanimoto"]:
            issues.append("stored_tanimoto_mismatch")
        record_id = make_record_id(source, role, donor["token"])
        rows.append({
            "record_id": record_id,
            "role": role,
            "label": label,
            "negative_type": negative_type,
            "candidate_token": donor["token"],
            "candidate_smiles": candidate,
            "task": normalized,
            "task_sha256": task_sha,
            "tanimoto": tanimoto,
            "same_formula": donor["formula"] == source["formula"],
            "connectivity_equal": donor["connectivity"] == source["connectivity"],
            "structure_relation": "identical_token" if role == "pos" else relation(source, donor),
            "donor_source_lmdb": donor["source_lmdb"],
            "donor_source_key": donor["source_key"],
        })
    if issues:
        return {"ok": False, "reason": issues[0], "issues": sorted(set(issues)), "source_key": source["source_key"]}
    bodies = [{key: row["task"][key] for key in row["task"] if key != "candidate_smiles"} for row in rows]
    if any(body != bodies[0] for body in bodies[1:]):
        return {"ok": False, "reason": "spectrum_changed_inside_group", "source_key": source["source_key"]}
    tokens = [row["candidate_token"] for row in rows]
    if tokens[0] != source_token or len(set(tokens)) != 4:
        return {"ok": False, "reason": "candidate_identity", "source_key": source["source_key"]}
    labels = [row["label"] for row in rows]
    if labels.count(1) != 1 or labels.count(0) != 3:
        return {"ok": False, "reason": "label_count", "source_key": source["source_key"]}
    return {"ok": True, "split": split, "source": source, "rows": rows, "group_id": group_id}


def dataset_id_for(split: str) -> str:
    return f"step_02b3.{split}.v1"


def evaluator_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "evaluator_version": EVALUATOR_VERSION,
        "invalid_policy": INVALID_POLICY,
        "primary_metric": PRIMARY_METRIC,
        "promotion": {"enabled": False},
        "statistics": {"enabled": False},
        "example_status": None,
    }


def dataset_protocol(mode: str, groups: int) -> dict[str, Any]:
    purpose = SPEC["purpose_real"] if mode == "real" else "synthetic_cli_fixture"
    return {
        "protocol": PROTOCOL,
        "purpose": purpose,
        "example_status": None,
        "seed": SEED,
        "groups_requested_per_split": groups,
        "positives_per_group": 1,
        "negatives_per_group": 3,
        "negative_construction_order": ["neg/same_formula", "neg/similar_structure", "neg/random"],
        "output_order": "hash rank of record_id; pair_id follows that order and is not a gold rule",
        "conditional_sampling": True,
        "engineering_size_note": "500 groups is an engineering starting size, not a statistical power guarantee.",
        "difficulty": "unknown",
        "ambiguity": "unannotated",
        "promotion": {"enabled": False, "status": "disabled_pending_protocol"},
        "statistics": {"enabled": False},
        "primary_metric": PRIMARY_METRIC,
        "invalid_policy": INVALID_POLICY,
        "task_schema": TASK_SCHEMA,
        "gate_dev_use": "not_for_model_tuning",
        "not_scaffold_split": True,
        "gold": "real structure identity; tool thresholds are not gold",
    }


def effective_config(mode: str, groups: int) -> dict[str, Any]:
    config = evaluator_config()
    loaded = load_config(config)
    return {
        "protocol": PROTOCOL,
        "mode": mode,
        "purpose": SPEC["purpose_real"] if mode == "real" else "synthetic_cli_fixture",
        "seed": SEED,
        "groups_per_split": groups,
        "domains": {
            "ownership": OWNERSHIP_DOMAIN,
            "source": SOURCE_DOMAIN,
            "same_formula": SAME_FORMULA_DOMAIN,
            "random": RANDOM_DOMAIN,
            "output": OUTPUT_DOMAIN,
            "record": RECORD_DOMAIN,
        },
        "ownership_index_zero_split": SPEC["ownership_index_zero_split"],
        "ownership_odd_count": SPEC["ownership_odd_count"],
        "fingerprint": {
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
            "include_chirality": FP_CHIRALITY,
            "generator": SPEC["fingerprint"]["generator"],
        },
        "rdkit_version": RDKIT_VERSION,
        "version_hash": version_hash(),
        "evaluator_config_sha256": loaded["content_sha256"],
        "primary_metric": PRIMARY_METRIC,
        "invalid_policy": INVALID_POLICY,
        "promotion_enabled": False,
        "statistics_enabled": False,
        "example_status": None,
        "conditional_sampling": True,
    }


def stamp_dataset(split: str, groups: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    dataset_id = dataset_id_for(split)
    ordered_rows = []
    for group in groups:
        ordered_rows.extend(group["rows"])
    ordered_rows.sort(key=lambda row: (domain_rank(OUTPUT_DOMAIN, SEED, row["record_id"]), row["record_id"]))
    manifest_records = []
    gold_records = []
    model_rows = []
    sidecars = []
    ordered_pairs = []
    for pair_id, row in enumerate(ordered_rows):
        source = next(group["source"] for group in groups if any(item["record_id"] == row["record_id"] for item in group["rows"]))
        manifest_records.append({
            "record_id": row["record_id"],
            "task": row["task"],
            "task_sha256": row["task_sha256"],
            "gold_label": row["label"],
            "source_group_id": source["token"],
            "parent_gt_smiles": source["canonical_smiles"],
            "difficulty": None,
            "ambiguity": None,
        })
        gold_records.append({
            "record_id": row["record_id"],
            "gold_label": row["label"],
            "source_group_id": source["token"],
            "parent_gt_smiles": source["canonical_smiles"],
        })
        model_rows.append({
            "pair_id": pair_id,
            "record_id": row["record_id"],
            "task": row["task"],
            "user_json": canonical_json(row["task"]),
        })
        ordered_pairs.append({"pair_id": pair_id, "record_id": row["record_id"]})
        sidecars.append({
            "record_id": row["record_id"],
            "split": split,
            "source_lmdb": source["source_lmdb"],
            "source_key": source["source_key"],
            "raw_record_sha256": source["raw_record_sha256"],
            "raw_record_codec": source["raw_record_codec"],
            "raw_smiles": source["raw_smiles"],
            "declared_smiles": source["declared_smiles"],
            "structure_token": source["token"],
            "candidate_token": row["candidate_token"],
            "role": row["role"],
            "negative_type": row["negative_type"],
            "gold_label": row["label"],
            "tanimoto": row["tanimoto"],
            "same_formula": row["same_formula"],
            "connectivity_equal": row["connectivity_equal"],
            "structure_relation": row["structure_relation"],
            "difficulty": "unknown",
            "ambiguity": "unannotated",
            "human_basis": None,
            "donor_locator": {
                "source_lmdb": row["donor_source_lmdb"],
                "source_key": row["donor_source_key"],
                "structure_token": row["candidate_token"],
            },
            "construction_rule": row["role"],
            "presence": source["presence"],
            "spectrum_norm_hash": source["spectrum_norm_hash"],
            "spectrum_raw_hash": source["spectrum_raw_hash"],
            "shift_sequence_hash": source["shift_sequence_hash"],
            "formula": {
                "calculated": source["formula"],
                "raw": source["raw_formula"],
                "relation": source["formula_relation"],
                "used_for_retrieval": "calculated",
            },
            "nmr_frequency_nucleus": row["task"]["nmr_frequency_nucleus"],
            "theoretical_fields_not_copied": sorted(source["theoretical_fields"]),
        })
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": dataset_id,
        "source_group_definition": {"id": SOURCE_GROUP_ID, "version": SOURCE_GROUP_VERSION},
        "records": manifest_records,
    }
    prepared, errors = prepare_dataset(manifest, out / "dataset_manifest.json", verify_hash=False)
    if errors or prepared is None:
        return {"ok": False, "errors": errors}
    manifest["content_sha256"] = prepared["content_sha256"]
    prepared, errors = prepare_dataset(manifest, out / "dataset_manifest.json", verify_hash=True)
    if errors or prepared is None:
        return {"ok": False, "errors": errors}
    gold = {
        "schema_version": GOLD_SCHEMA,
        "dataset_id": dataset_id,
        "records": gold_records,
    }
    gold_hash, gold_errors = check_gold_sidecar(gold, prepared, verify_hash=False)
    if gold_errors or gold_hash is None:
        return {"ok": False, "errors": gold_errors}
    gold["content_sha256"] = gold_hash
    gold_hash, gold_errors = check_gold_sidecar(gold, prepared, verify_hash=True)
    if gold_errors:
        return {"ok": False, "errors": gold_errors}
    return {
        "ok": True,
        "manifest": manifest,
        "gold": gold,
        "model_rows": model_rows,
        "sidecars": sidecars,
        "pairs": {
            "pair_to_record": {str(item["pair_id"]): item["record_id"] for item in ordered_pairs},
            "ordered": ordered_pairs,
        },
        "dataset_content_sha256": prepared["content_sha256"],
        "gold_content_sha256": gold_hash,
    }


def source_files(split: str, groups: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    group_rows = []
    for group in groups:
        source = group["source"]
        chemical = source["chemical"]
        kept = {}
        for key in (
            "smiles", "smiles_canonical", "molecular_formula",
            "c_nmr_peaks", "h_nmr_peaks", "nmr_solvent", "nmr_frequency", "experimental_total_nH",
            "experimental_13c_peaks", "experimental_1h_peaks", "solvent", "nmr_freq",
        ):
            if key in chemical:
                kept[key] = json_safe(chemical[key])
        other_keys = sorted(
            key for key in chemical
            if key not in kept and key not in {"smiles", "smiles_canonical"}
        )
        records.append({
            "source_group_id": source["token"],
            "split": split,
            "source_lmdb": source["source_lmdb"],
            "source_key": source["source_key"],
            "raw_record_sha256": source["raw_record_sha256"],
            "raw_record_codec": source["raw_record_codec"],
            "presence": source["presence"],
            "spectrum_norm_hash": source["spectrum_norm_hash"],
            "spectrum_raw_hash": source["spectrum_raw_hash"],
            "shift_sequence_hash": source["shift_sequence_hash"],
            "theoretical_fields": source["theoretical_fields"],
            "other_field_names": [key for key in other_keys if key not in THEORETICAL_FIELDS],
            "raw_fields": kept,
            "donor_locators": [
                {
                    "role": row["role"],
                    "source_lmdb": row["donor_source_lmdb"],
                    "source_key": row["donor_source_key"],
                    "structure_token": row["candidate_token"],
                }
                for row in group["rows"]
            ],
            "record_ids": [row["record_id"] for row in group["rows"]],
        })
        group_rows.append({
            "source_group_id": source["token"],
            "split": split,
            "source_lmdb": source["source_lmdb"],
            "source_key": source["source_key"],
            "record_ids": {row["role"]: row["record_id"] for row in group["rows"]},
        })
    return records, group_rows


def write_split(out: Path, split: str, stamped: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], ownership_sha: str, own_hash: str) -> dict[str, str]:
    split_dir = out / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dump_json(split_dir / "dataset_manifest.json", stamped["manifest"])
    dump_json(split_dir / "gold_sidecar.json", stamped["gold"])
    write_jsonl(split_dir / "model_inputs.jsonl", stamped["model_rows"])
    write_jsonl(split_dir / "input_sidecars.jsonl", stamped["sidecars"])
    source_rows, group_rows = source_files(split, groups)
    write_jsonl(split_dir / "source_records.jsonl", source_rows)
    write_jsonl(split_dir / "source_groups.jsonl", group_rows)
    dump_json(split_dir / "pair_to_record.json", stamped["pairs"])
    dump_json(split_dir / "ownership_ref.json", {
        "path": "../ownership.jsonl",
        "sha256": ownership_sha,
        "ownership_set_sha256": own_hash,
    })
    hashes = {
        "dataset_manifest.json": file_sha256(split_dir / "dataset_manifest.json"),
        "gold_sidecar.json": file_sha256(split_dir / "gold_sidecar.json"),
        "model_inputs.jsonl": file_sha256(split_dir / "model_inputs.jsonl"),
        "input_sidecars.jsonl": file_sha256(split_dir / "input_sidecars.jsonl"),
        "source_records.jsonl": file_sha256(split_dir / "source_records.jsonl"),
        "source_groups.jsonl": file_sha256(split_dir / "source_groups.jsonl"),
        "pair_to_record.json": file_sha256(split_dir / "pair_to_record.json"),
    }
    return hashes


def tanimoto_bins(values: Sequence[float]) -> dict[str, int]:
    bins = {"[0.0,0.2)": 0, "[0.2,0.4)": 0, "[0.4,0.6)": 0, "[0.6,0.8)": 0, "[0.8,1.0)": 0, "1.0": 0}
    for value in values:
        if value == 1.0:
            bins["1.0"] += 1
        elif value < 0.2:
            bins["[0.0,0.2)"] += 1
        elif value < 0.4:
            bins["[0.2,0.4)"] += 1
        elif value < 0.6:
            bins["[0.4,0.6)"] += 1
        elif value < 0.8:
            bins["[0.6,0.8)"] += 1
        else:
            bins["[0.8,1.0)"] += 1
    return bins


def split_statistics(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    relations = Counter()
    reuse = Counter()
    pair_keys = set()
    spectra = []
    shifts = []
    projections = []
    neighbor_sims = []
    for group in groups:
        source = group["source"]
        spectra.append(source["spectrum_norm_hash"])
        shifts.append(source["shift_sequence_hash"])
        body = {key: group["rows"][0]["task"][key] for key in group["rows"][0]["task"] if key != "candidate_smiles"}
        projections.append(sha256_text(canonical_json(body)))
        for row in group["rows"]:
            labels[row["label"]] += 1
            if row["role"] != "pos":
                relations[(row["role"], row["structure_relation"])] += 1
                reuse[row["candidate_token"]] += 1
            if row["role"] == "neg/similar_structure" and row["tanimoto"] is not None:
                neighbor_sims.append(float(row["tanimoto"]))
            pair_keys.add((source["token"], source["spectrum_norm_hash"], row["candidate_token"], row["label"]))
    histogram = Counter(reuse.values())
    return {
        "groups": len(groups),
        "rows": sum(labels.values()),
        "label_1": labels[1],
        "label_0": labels[0],
        "unique_negative_tokens": len(reuse),
        "max_donor_reuse": max(reuse.values()) if reuse else 0,
        "donor_reuse_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "duplicate_pair_keys": sum(labels.values()) - len(pair_keys),
        "structure_relations": {f"{role}|{relation_name}": count for (role, relation_name), count in sorted(relations.items())},
        "neighbor_tanimoto_bins": tanimoto_bins(neighbor_sims),
        "neighbor_tanimoto_min": min(neighbor_sims) if neighbor_sims else None,
        "neighbor_tanimoto_max": max(neighbor_sims) if neighbor_sims else None,
        "raw_spectrum_hashes": spectra,
        "shift_hashes": [item for item in shifts if item],
        "projection_hashes": projections,
        "scaffolds": [group["source"]["scaffold"] for group in groups],
        "connectivity": [group["source"]["connectivity"] for group in groups],
    }


def intersection_count(left: Sequence[Any], right: Sequence[Any]) -> int:
    return len(set(left) & set(right))


def audit_splits(built: Mapping[str, Any], reservations: Mapping[str, set[str]]) -> dict[str, Any]:
    search = split_statistics(built["search-dev"])
    gate = split_statistics(built["gate-dev"])
    search_tokens = [row["candidate_token"] for group in built["search-dev"] for row in group["rows"]]
    gate_tokens = [row["candidate_token"] for group in built["gate-dev"] for row in group["rows"]]
    search_gt = [group["source"]["token"] for group in built["search-dev"]]
    gate_gt = [group["source"]["token"] for group in built["gate-dev"]]
    search_union = set(search_tokens) | set(search_gt)
    gate_union = set(gate_tokens) | set(gate_gt)
    blocking = []
    spectrum_cross = intersection_count(search["raw_spectrum_hashes"], gate["raw_spectrum_hashes"])
    if spectrum_cross:
        blocking.append("raw_spectrum_cross_split")
    for label, stats in (("search-dev", search), ("gate-dev", gate)):
        if stats["duplicate_pair_keys"]:
            blocking.append(f"{label}_duplicate_pair")
        if stats["label_1"] * 3 != stats["label_0"]:
            blocking.append(f"{label}_label_ratio")
    token_checks = {
        "search_gt_vs_gate_gt": intersection_count(search_gt, gate_gt),
        "search_candidate_vs_gate_candidate": intersection_count(search_tokens, gate_tokens),
        "search_union_vs_gate_union": len(search_union & gate_union),
        "search_gt_vs_gate_candidate": intersection_count(search_gt, gate_tokens),
        "gate_gt_vs_search_candidate": intersection_count(gate_gt, search_tokens),
    }
    for name, count in token_checks.items():
        if count:
            blocking.append(name)
    for label, tokens in (("search", search_union), ("gate", gate_union)):
        for reserve_name, reserve in reservations.items():
            count = len(tokens & reserve)
            token_checks[f"{label}_vs_{reserve_name}"] = count
            if count:
                blocking.append(f"{label}_vs_{reserve_name}")
    within = {
        "search-dev": len(search["raw_spectrum_hashes"]) - len(set(search["raw_spectrum_hashes"])),
        "gate-dev": len(gate["raw_spectrum_hashes"]) - len(set(gate["raw_spectrum_hashes"])),
    }
    report = {
        "token_intersections": token_checks,
        "raw_spectrum_cross_split": spectrum_cross,
        "raw_spectrum_within_split_extra": within,
        "shift_sequence_cross_split": intersection_count(search["shift_hashes"], gate["shift_hashes"]),
        "projected_spectrum_cross_split": intersection_count(search["projection_hashes"], gate["projection_hashes"]),
        "scaffold_intersection": intersection_count(search["scaffolds"], gate["scaffolds"]),
        "connectivity_intersection": intersection_count(search["connectivity"], gate["connectivity"]),
        "scaffold_intersection_required_to_be_zero": False,
        "connectivity_intersection_required_to_be_zero": False,
        "not_a_scaffold_split": True,
        "per_split": {
            "search-dev": {key: value for key, value in search.items() if not key.endswith("hashes") and key not in {"scaffolds", "connectivity", "candidate_scaffolds"}},
            "gate-dev": {key: value for key, value in gate.items() if not key.endswith("hashes") and key not in {"scaffolds", "connectivity", "candidate_scaffolds"}},
        },
        "blocking_reasons": blocking,
        "note": (
            "Structure-token isolation is not scaffold isolation. "
            "Shift-sequence overlap is not a full-spectrum match and is not a merge key. "
            "Neighbor Tanimoto bins describe the construction draw, not verified difficulty."
        ),
    }
    return report


def projection_report(decoded: Mapping[str, Mapping[str, Any]], built: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    presence = Counter()
    nucleus = Counter()
    formula_relations = Counter()
    theoretical = Counter()
    for record in decoded.values():
        nucleus[record["nucleus"]] += 1
        formula_relations[record["formula_relation"]] += 1
        for key, state in record["presence"].items():
            presence[f"{key}:{state}"] += 1
        for key in record["theoretical_fields"]:
            theoretical[key] += 1
    issues = Counter()
    for groups in built.values():
        for group in groups:
            for row in group["rows"]:
                for issue in projection_issues(group["source"]["chemical"], row["task"]):
                    issues[issue] += 1
    return {
        "decoded_presence": dict(presence),
        "decoded_nucleus": dict(nucleus),
        "formula_relation": dict(formula_relations),
        "theoretical_fields_not_copied": dict(theoretical),
        "projection_issues": dict(issues),
        "runtime_empty_list_adapts_missing_null_and_empty": True,
        "note": "A runtime empty list is not the raw fingerprint. Sidecars keep missing, null, and empty_list.",
    }


def logical_parts(scan: Mapping[str, Any], own_hash: str, ownership_sha: str, split_hashes: Mapping[str, Mapping[str, str]], stamped: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    splits = {}
    for split in SPLITS:
        record_ids = sorted(item["record_id"] for item in stamped[split]["pairs"]["ordered"])
        splits[split] = {
            "dataset_content_sha256": stamped[split]["dataset_content_sha256"],
            "gold_content_sha256": stamped[split]["gold_content_sha256"],
            "files": split_hashes[split],
            "record_ids_sha256": sha256_text("\n".join(record_ids) + "\n"),
            "pair_to_record_sha256": sha256_text(canonical_json(stamped[split]["pairs"])),
        }
    return {
        "protocol": PROTOCOL,
        "seed": SEED,
        "version_hash": version_hash(),
        "rdkit_version": RDKIT_VERSION,
        "rows_identity_sha256": scan["rows_identity_sha256"],
        "ownership_set_sha256": own_hash,
        "ownership_file_sha256": ownership_sha,
        "fingerprint": {
            "radius": FP_RADIUS,
            "n_bits": FP_BITS,
            "include_chirality": FP_CHIRALITY,
        },
        "splits": splits,
    }


def logical_hash(parts: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(parts))


def write_ownership(path: Path, assigned: Sequence[tuple[str, str]], train_tokens: set[str], test_tokens: set[str]) -> tuple[str, str]:
    rows = [(split, token, "valid_pool") for token, split in assigned]
    rows.extend(("future-train", token, "future_train_reservation") for token in train_tokens)
    rows.extend(("test", token, "test_reservation") for token in test_tokens)
    rows.sort()
    with path.open("w", encoding="utf-8") as handle:
        for split, token, register in rows:
            handle.write(canonical_json({
                "assignment": split,
                "register": register,
                "structure_token": token,
            }))
            handle.write("\n")
    return file_sha256(path), ownership_set_hash(rows)


def pool_report(scan: Mapping[str, Any], assigned: Sequence[tuple[str, str]] | None) -> dict[str, Any]:
    valid_tokens = {row["candidate_token"] for row in scan["valid_rows"] if isinstance(row["candidate_token"], str)}
    overlaps = {
        "valid_vs_future_train": len(valid_tokens & scan["train_tokens"]),
        "valid_vs_test_reservation": len(valid_tokens & scan["test_tokens"]),
        "valid_vs_exposure": len(valid_tokens & scan["exposure_tokens"]),
        "valid_vs_test_2000_gt_reason": len(valid_tokens & scan["test_gt_tokens"]),
        "valid_vs_test_2000_candidate_reason": len(valid_tokens & scan["test_candidate_tokens"]),
    }
    counts = Counter(split for _token, split in assigned) if assigned else {}
    return {
        "pool_counts": scan["pool_counts"],
        "row_counts": scan["row_counts"],
        "reference_auto_positive_pool": SPEC["auto_positive_pool"],
        "violations": scan["violations"],
        "duplicate_valid_parent_tokens": len(scan["duplicates"]),
        "ownership_counts": dict(counts),
        "ownership_odd_count_rule": SPEC["ownership_odd_count"],
        "reservation_tokens": {
            "future_train": len(scan["train_tokens"]),
            "test": len(scan["test_tokens"]),
            "exposure": len(scan["exposure_tokens"]),
            "test_2000_gt_reason": len(scan["test_gt_tokens"]),
            "test_2000_candidate_reason": len(scan["test_candidate_tokens"]),
        },
        "valid_pool_intersections": overlaps,
        "note": (
            "Absence from the listed exposure and reservation sets is not evidence that a base model never saw a structure. "
            "test.lmdb and test_2000 are different. The test_2000 check uses the eligibility-index reason tokens."
        ),
    }


def assert_writable(out: Path) -> None:
    if not out.exists():
        return
    if (out / "build_status.json").exists() or (out / "search-dev" / "dataset_manifest.json").exists():
        raise DirectoryFrozen(str(out))


def finish(out: Path, *, command: str, exit_code: int, status: str, reason: str | None, details: dict[str, Any], logical: str | None, started: str) -> dict[str, Any]:
    payload = {
        "command": command,
        "status": status,
        "blocking_reasons": [] if reason in (None, "ok") else [reason],
        "logical_data_sha256": logical,
        "protocol": PROTOCOL,
        "details": details,
    }
    if command == "plan":
        dump_json(out / "plan_status.json", payload)
    else:
        dump_json(out / "build_status.json", payload)
    dump_json(out / "handoff.json", {
        "protocol": PROTOCOL,
        "command": command,
        "status": status,
        "blocking_reasons": payload["blocking_reasons"],
        "logical_data_sha256": logical,
        "details": details,
        "gate_dev_use": "not_for_model_tuning",
        "promotion": "disabled_pending_protocol",
    })
    runtime = {
        "started_at": started,
        "finished_at": now_iso(),
        "hostname": socket.gethostname(),
        "output_dir": str(out),
        "command": command,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    dump_json(out / "runtime_info.json", runtime)
    result = {"exit": exit_code, "status": status, "reason": reason, "logical_data_sha256": logical}
    result.update(details)
    return result


def cross_check_bulk(groups: Sequence[Mapping[str, Any]], pool: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the exact popcount neighbor with RDKit's bulk query on the same bits."""
    if not groups:
        return {"checked": 0, "mismatches": 0}
    group = groups[0]
    source = group["source"]
    different = [token for token, record in pool.items() if record["formula"] != source["formula"]]
    if not different:
        return {"checked": 0, "mismatches": 0}
    query = int_to_bitvect(int(source["fp_int"]))
    fps = [int_to_bitvect(int(pool[token]["fp_int"])) for token in different]
    sims = DataStructs.BulkTanimotoSimilarity(query, fps)
    best_token, best_sim = pick_max_similarity(list(zip(different, sims)))
    chosen = next(row for row in group["rows"] if row["role"] == "neg/similar_structure")
    if best_token != chosen["candidate_token"] or float(best_sim) != float(chosen["tanimoto"]):
        raise StopBuild(2, "incomplete", "neighbor_search_mismatch", {"checked": 1})
    return {"checked": 1, "mismatches": 0}


def execute(options: Mapping[str, Any]) -> dict[str, Any]:
    command = str(options["command"])
    out = Path(options["out"]).resolve()
    started = now_iso()
    if command == "validate":
        return validate(out)
    if command == "rebuild":
        return rebuild(options, started)
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    mode = str(options["mode"])
    groups = int(options["groups"] if options.get("groups") is not None else GROUPS)
    if mode == "real" and groups != GROUPS:
        raise StopBuild(3, "refused", "real_group_quota_is_fixed")
    if groups < 1:
        raise StopBuild(3, "refused", "groups_must_be_positive")
    fingerprint_self_check()
    dump_json(out / "worktree_preservation.json", preservation_report())
    dump_json(out / "effective_config.json", effective_config(mode, groups))
    index_path = Path(options["index"]) if options.get("index") else Path(SPEC["eligibility_index"])
    records_path = Path(options["records"]) if options.get("records") else None
    if mode == "real" and records_path is not None:
        raise StopBuild(3, "refused", "real_mode_does_not_read_fixture_records")
    if mode == "fixture" and (records_path is None or options.get("index") is None):
        raise StopBuild(3, "refused", "fixture_requires_index_and_records")
    log("scanning eligibility index")
    scan = scan_index(index_path, mode)
    manifest = asset_manifest(mode)
    dump_json(out / "source_manifest.json", {
        "assets": manifest,
        "eligibility_file_sha256": scan["file_sha256"],
        "rows_identity_sha256": scan["rows_identity_sha256"],
        "eligibility_index": str(index_path),
    })
    blocking = []
    if scan["duplicates"]:
        blocking.append("duplicate_parent_token")
        dump_json(out / "duplicate_source_audit.json", {
            "count": len(scan["duplicates"]),
            "locators": scan["duplicates"][:200],
            "note": "Multiple source rows share one parent token. No spectrum was chosen.",
        })
    if scan["violations"]:
        blocking.append("eligibility_rule_mismatch")
    report = pool_report(scan, None)
    if any(report["valid_pool_intersections"].values()):
        blocking.append("isolation_overlap")
    dump_json(out / "source_pool_report.json", report)
    if command == "plan":
        status = "planned" if not blocking else "incomplete"
        return finish(
            out, command="plan", exit_code=0 if not blocking else 2, status=status,
            reason=None if not blocking else blocking[0], details={"blocking_reasons": blocking, "pool_counts": scan["pool_counts"]},
            logical=None, started=started,
        )
    if blocking:
        return finish(
            out, command="build", exit_code=2, status="incomplete", reason=blocking[0],
            details={"blocking_reasons": blocking}, logical=None, started=started,
        )
    log("assigning ownership")
    assigned = assign_ownership([row["candidate_token"] for row in scan["valid_rows"]])
    ownership_sha, own_hash = write_ownership(out / "ownership.jsonl", assigned, scan["train_tokens"], scan["test_tokens"])
    dump_json(out / "source_pool_report.json", pool_report(scan, assigned))
    by_split = {"search-dev": {}, "gate-dev": {}}
    split_of = dict(assigned)
    log("decoding source records")
    decoded_result = decode_pool(scan, mode, records_path, compute_fp=not bool(options.get("reuse_cache")))
    if decoded_result["failures"]:
        dump_json(out / "decode_failures.json", {
            "count": len(decoded_result["failures"]),
            "failures": decoded_result["failures"][:50],
        })
        return finish(
            out, command="build", exit_code=2, status="incomplete", reason="source_reread_mismatch",
            details={"failures": len(decoded_result["failures"])}, logical=None, started=started,
        )
    decoded = decoded_result["decoded"]
    cache_info = {"reused": False, "spot_check_count": 0, "spot_check_mismatches": 0}
    reuse_cache = options.get("reuse_cache")
    if reuse_cache:
        log("checking fingerprint cache")
        cache_info = load_cache(Path(reuse_cache), scan, own_hash, decoded)
    else:
        log("writing fingerprint cache")
        write_cache(out / "cache", scan, own_hash, decoded)
    for token, record in decoded.items():
        by_split[split_of[token]][token] = record
    funnel = {}
    selected_tokens = {}
    for split in SPLITS:
        reasons = constructability(by_split[split])
        reason_counts = Counter(reasons.values())
        constructable = [token for token, reason in reasons.items() if reason == "constructable"]
        chosen = sorted(constructable, key=lambda token: (domain_rank(SOURCE_DOMAIN, SEED, token), token))[:groups]
        selected_tokens[split] = chosen
        funnel[split] = {
            "ownership_subpool": sum(1 for item in split_of.values() if item == split),
            "decoded": len(by_split[split]),
            "exclusion_reasons": dict(reason_counts),
            "constructable": len(constructable),
            "requested": groups,
            "selected": len(chosen) if len(constructable) >= groups else 0,
            "shortfall": max(0, groups - len(constructable)),
            "coverage": {
                "decoded": coverage_of(list(by_split[split].values())),
                "constructable": coverage_of([by_split[split][token] for token in constructable]),
                "selected": coverage_of([by_split[split][token] for token in chosen]) if len(constructable) >= groups else {"structures": 0},
            },
        }
    dump_json(out / "construction_eligibility_report.json", {
        "eligible_pool": scan["pool_counts"].get("valid.lmdb", len(scan["valid_rows"])),
        "conditional_sampling": True,
        "engineering_size_note": "500 groups is an engineering starting size, not a statistical power guarantee.",
        "splits": funnel,
        "formula_note": "Retrieval uses the calculated RDKit formula. A raw-field disagreement is not a fallback.",
    })
    if any(funnel[split]["shortfall"] for split in SPLITS):
        return finish(
            out, command="build", exit_code=2, status="incomplete", reason="quota_shortfall",
            details={"funnel": {split: {"constructable": funnel[split]["constructable"], "requested": groups} for split in SPLITS}},
            logical=None, started=started,
        )
    selected_flat = [token for split in SPLITS for token in selected_tokens[split]]
    recheck_selected(decoded, selected_flat, bool(reuse_cache))
    built = {}
    projection_failures = []
    for split in SPLITS:
        groups_out = []
        pool = donor_pool(by_split[split])
        for token in selected_tokens[split]:
            choices = choose_negatives(token, pool)
            if not choices["ok"]:
                projection_failures.append({"source_key": pool[token]["source_key"], "reason": choices["reason"]})
                continue
            group = build_group(token, choices, pool, split)
            if not group["ok"]:
                projection_failures.append({"source_key": group.get("source_key"), "reason": group["reason"]})
                continue
            for row in group["rows"]:
                if row["role"] != "pos" and split_of[row["candidate_token"]] != split:
                    projection_failures.append({"source_key": pool[token]["source_key"], "reason": "cross_split_donor"})
            groups_out.append(group)
        built[split] = groups_out
        cross_check_bulk(groups_out, pool)
    if projection_failures or any(len(built[split]) != groups for split in SPLITS):
        dump_json(out / "projection_failures.json", {"count": len(projection_failures), "failures": projection_failures[:50]})
        return finish(
            out, command="build", exit_code=2, status="incomplete", reason="group_integrity",
            details={"failures": len(projection_failures)}, logical=None, started=started,
        )
    reservations = {
        "future_train": scan["train_tokens"],
        "test_reservation": scan["test_tokens"],
        "exposure": scan["exposure_tokens"],
        "test_2000_gt": scan["test_gt_tokens"],
        "test_2000_candidate": scan["test_candidate_tokens"],
    }
    split_audit = audit_splits(built, reservations)
    dump_json(out / "split_audit.json", split_audit)
    dump_json(out / "input_projection_report.json", projection_report(decoded, built))
    if split_audit["blocking_reasons"]:
        # Keep the source groups on disk, but do not mark the freeze prepared.
        pass
    stamped = {}
    split_hashes = {}
    for split in SPLITS:
        stamped[split] = stamp_dataset(split, built[split], out / split)
        if not stamped[split]["ok"]:
            dump_json(out / "schema_errors.json", {"split": split, "errors": stamped[split]["errors"][:20]})
            return finish(
                out, command="build", exit_code=2, status="incomplete", reason="schema_rejection",
                details={"split": split}, logical=None, started=started,
            )
    for split in SPLITS:
        split_hashes[split] = write_split(out, split, stamped[split], built[split], ownership_sha, own_hash)
        dump_json(out / split / "config.json", evaluator_config())
        protocol = dataset_protocol(mode, groups)
        protocol["dataset_id"] = dataset_id_for(split)
        protocol["split"] = split
        dump_json(out / split / "dataset_protocol.json", protocol)
    dump_json(out / "config.json", evaluator_config())
    dump_json(out / "dataset_protocol.json", dataset_protocol(mode, groups))
    parts = logical_parts(scan, own_hash, ownership_sha, split_hashes, stamped)
    digest = logical_hash(parts)
    dump_json(out / "logical_parts.json", parts)
    blocking_reasons = list(split_audit["blocking_reasons"])
    status = SUCCESS_REAL if mode == "real" else SUCCESS_FIXTURE
    if blocking_reasons:
        status = "incomplete"
    if mode == "real":
        for split in SPLITS:
            if funnel[split]["selected"] != GROUPS or split_audit["per_split"][split]["rows"] != GROUPS * 4:
                status = "incomplete"
                blocking_reasons.append("quota_not_met_after_write")
    exit_code = 0 if status in {SUCCESS_REAL, SUCCESS_FIXTURE} and not blocking_reasons else 2
    dump_json(out / "reproducibility_report.json", {
        "status": "pending_second_rebuild",
        "logical_data_sha256": digest,
        "fingerprint_cache": cache_info,
    })
    details = {
        "funnel": {split: {"constructable": funnel[split]["constructable"], "selected": funnel[split]["selected"]} for split in SPLITS},
        "negative_rows": {split: split_audit["per_split"][split]["label_0"] for split in SPLITS},
        "blocking_reasons": blocking_reasons,
        "raw_spectrum_cross_split": split_audit["raw_spectrum_cross_split"],
        "scaffold_intersection": split_audit["scaffold_intersection"],
        "connectivity_intersection": split_audit["connectivity_intersection"],
    }
    return finish(
        out, command="build", exit_code=exit_code, status=status,
        reason=None if not blocking_reasons else blocking_reasons[0], details=details, logical=digest, started=started,
    )


def validate(out: Path) -> dict[str, Any]:
    status_path = out / "build_status.json"
    if not status_path.is_file():
        return {"exit": 2, "status": "incomplete", "reason": "build_status_missing", "logical_data_sha256": None}
    status = load_json(status_path)
    errors = []
    if status.get("status") not in {SUCCESS_REAL, SUCCESS_FIXTURE}:
        errors.append("status_not_prepared")
    for name in ("run_manifest.json", "predictions.json", "predictions.jsonl"):
        if (out / name).exists():
            errors.append("forged_run_artifact")
    config = load_config(load_json(out / "config.json"))
    if config["example_status"] is not None or config["promotion"]["enabled"] is not False or config["statistics"]["enabled"] is not False:
        errors.append("config_flags")
    parts = load_json(out / "logical_parts.json") if (out / "logical_parts.json").is_file() else None
    if parts is None or logical_hash(parts) != status.get("logical_data_sha256"):
        errors.append("logical_hash")
    for split in SPLITS:
        split_dir = out / split
        manifest = load_json(split_dir / "dataset_manifest.json")
        prepared, prepare_errors = prepare_dataset(manifest, split_dir / "dataset_manifest.json", verify_hash=True)
        if prepare_errors or prepared is None:
            errors.append(f"{split}_schema")
            continue
        gold = load_json(split_dir / "gold_sidecar.json")
        _gold_hash, gold_errors = check_gold_sidecar(gold, prepared, verify_hash=True)
        if gold_errors:
            errors.append(f"{split}_gold")
        model_rows = read_jsonl(split_dir / "model_inputs.jsonl")
        sidecars = read_jsonl(split_dir / "input_sidecars.jsonl")
        if len(model_rows) != len(manifest["records"]) or len(sidecars) != len(manifest["records"]) or len(gold["records"]) != len(manifest["records"]):
            errors.append(f"{split}_length")
        pairs = load_json(split_dir / "pair_to_record.json")
        if not pairs["ordered"] or pairs["ordered"][0]["pair_id"] != 0:
            errors.append(f"{split}_pair0")
        by_group = defaultdict(list)
        for record in manifest["records"]:
            by_group[record["source_group_id"]].append(record)
        for group_id, records in by_group.items():
            if len(records) != 4:
                errors.append(f"{split}_group_size")
                break
            labels = [record["gold_label"] for record in records]
            if labels.count(1) != 1 or labels.count(0) != 3:
                errors.append(f"{split}_labels")
                break
        seen_ids = set()
        for row in model_rows:
            task = row["task"]
            if set(task) != set(TASK_FIELDS):
                errors.append(f"{split}_task_fields")
                break
            if json.loads(row["user_json"]) != task:
                errors.append(f"{split}_user_json")
                break
            if any(key in task for key in ("gold_label", "label", "gt", "negative_type", "split", "tanimoto")):
                errors.append(f"{split}_label_leak")
                break
            if row["record_id"] in seen_ids:
                errors.append(f"{split}_duplicate_record")
                break
            seen_ids.add(row["record_id"])
        if parts is not None:
            for filename, expected in parts["splits"][split]["files"].items():
                if file_sha256(split_dir / filename) != expected:
                    errors.append(f"{split}_file_hash")
                    break
    report = {"errors": sorted(set(errors)), "status": status.get("status")}
    dump_json(out / "validate_report.json", report)
    if errors:
        return {"exit": 2, "status": status.get("status"), "reason": errors[0], "logical_data_sha256": status.get("logical_data_sha256")}
    return {"exit": 0, "status": status.get("status"), "reason": None, "logical_data_sha256": status.get("logical_data_sha256")}


def rebuild(options: Mapping[str, Any], started: str) -> dict[str, Any]:
    source = Path(options["from_run"]).resolve()
    out = Path(options["out"]).resolve()
    runtime = load_json(source / "runtime_info.json")
    first = load_json(source / "build_status.json")
    if first.get("status") not in {SUCCESS_REAL, SUCCESS_FIXTURE, "incomplete"}:
        raise StopBuild(3, "refused", "rebuild_requires_a_recorded_run")
    mode = load_json(source / "effective_config.json")["mode"]
    groups = load_json(source / "effective_config.json")["groups_per_split"]
    child = execute({
        "command": "build",
        "out": out,
        "mode": mode,
        "index": runtime.get("index"),
        "records": runtime.get("records"),
        "groups": groups,
        "reuse_cache": source / "cache",
    })
    second = load_json(out / "build_status.json")
    equal = child.get("logical_data_sha256") == first.get("logical_data_sha256") and child.get("status") == first.get("status")
    report = {
        "status": "matched" if equal else "mismatched",
        "first_logical_data_sha256": first.get("logical_data_sha256"),
        "second_logical_data_sha256": child.get("logical_data_sha256"),
        "equal": equal,
        "first_output_dir": str(source),
        "second_output_dir": str(out),
        "fingerprint_cache_reused": True,
        "sampling_reexecuted": True,
        "serialization_reexecuted": True,
    }
    dump_json(source / "reproducibility_report.json", report)
    dump_json(out / "reproducibility_report.json", report)
    if not equal:
        child["exit"] = 2
        child["status"] = "incomplete"
        child["reason"] = "rebuild_mismatch"
    return child


def execute_cli(options: Mapping[str, Any]) -> dict[str, Any]:
    started = now_iso()
    try:
        if options["command"] != "rebuild":
            # runtime paths are recorded by execute; stash them for rebuild.
            result = execute(options)
        else:
            result = rebuild(options, started)
    except DirectoryFrozen as exc:
        return {"exit": 3, "status": "refused", "reason": "output_directory_is_frozen", "logical_data_sha256": None, "path": str(exc)}
    except StopBuild as exc:
        out = Path(options["out"]).resolve()
        if options["command"] != "validate" and not (out / "build_status.json").exists() and not (out / "plan_status.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            return finish(
                out, command=str(options["command"]), exit_code=exc.exit_code, status=exc.status,
                reason=exc.reason, details=exc.details, logical=None, started=started,
            )
        return {"exit": exc.exit_code, "status": exc.status, "reason": exc.reason, "logical_data_sha256": None, **exc.details}
    if options["command"] == "build" and (Path(options["out"]) / "runtime_info.json").is_file():
        runtime = load_json(Path(options["out"]) / "runtime_info.json")
        runtime["index"] = str(Path(options["index"]).resolve()) if options.get("index") else SPEC["eligibility_index"]
        runtime["records"] = str(Path(options["records"]).resolve()) if options.get("records") else None
        config_path = Path(options["out"]) / "effective_config.json"
        if config_path.is_file():
            runtime["groups"] = load_json(config_path)["groups_per_split"]
        dump_json(Path(options["out"]) / "runtime_info.json", runtime)
    return result
