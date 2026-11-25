"""Read-only structure and spectrum fingerprints for the step-02A audit.

These helpers do not rewrite SMILES or spectra. An unparsed SMILES stays in the
row denominator. Spectrum identity is exact: peak order, missing keys, nulls,
and numeric values are preserved. No ppm tolerance is applied.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

RDKIT_VERSION = rdBase.rdkitVersion
STRUCTURE_OPTIONS = {
    "mol_from_smiles": "Chem.MolFromSmiles(smiles) with RDKit defaults, including sanitization",
    "canonical_isomeric": "Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)",
    "connectivity": "Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)",
    "scaffold": "MurckoScaffold.MurckoScaffoldSmiles(mol=mol)",
    "not_applied": [
        "salt stripping",
        "charge neutralization",
        "tautomer canonicalization",
        "stereo removal followed by relabeling",
        "ppm rounding or tolerance",
    ],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize with stable dict-key order and the original list order."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite spectrum value")
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def structure_keys(smiles: str | None) -> dict[str, Any]:
    """Return raw, canonical, connectivity, and scaffold views for one SMILES."""
    if smiles is None or smiles == "":
        return {
            "raw": smiles,
            "status": "missing",
            "canonical_isomeric": None,
            "connectivity": None,
            "scaffold": None,
            "error": None,
        }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "raw": smiles,
            "status": "unparsed",
            "canonical_isomeric": None,
            "connectivity": None,
            "scaffold": None,
            "error": "MolFromSmiles returned None",
        }
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception as exc:  # RDKit can raise on unusual valence states
        return {
            "raw": smiles,
            "status": "unparsed",
            "canonical_isomeric": None,
            "connectivity": None,
            "scaffold": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    scaffold = None
    scaffold_error = None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or None
    except Exception as exc:
        scaffold_error = f"{type(exc).__name__}: {exc}"
    return {
        "raw": smiles,
        "status": "parsed",
        "canonical_isomeric": canonical,
        "connectivity": connectivity,
        "scaffold": scaffold,
        "error": scaffold_error,
    }


def _presence(source: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in source:
        return {"present": False, "value": None}
    return {"present": True, "value": _finite(source.get(key))}


def spectrum_views(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Hash the original spectrum object and an exact physical-field view.

    The normalized view keeps every peak key that is present, including J,
    integrals, multiplicity, and ranges. A missing key is not stored as null,
    and a null integral is not stored as zero.
    """
    if source is None:
        return {
            "status": "missing",
            "raw_content_hash": None,
            "normalized_hash": None,
            "normalized": None,
            "field_names": [],
        }
    raw_object = {
        "c_nmr_peaks": _presence(source, "c_nmr_peaks"),
        "experimental_13c_peaks": _presence(source, "experimental_13c_peaks"),
        "h_nmr_peaks": _presence(source, "h_nmr_peaks"),
        "experimental_1h_peaks": _presence(source, "experimental_1h_peaks"),
        "nmr_solvent": _presence(source, "nmr_solvent"),
        "solvent": _presence(source, "solvent"),
        "nmr_frequency": _presence(source, "nmr_frequency"),
        "nmr_freq": _presence(source, "nmr_freq"),
        "experimental_total_nH": _presence(source, "experimental_total_nH"),
        "other_keys": {
            str(key): _finite(source[key])
            for key in source
            if str(key) not in {
                "c_nmr_peaks",
                "experimental_13c_peaks",
                "h_nmr_peaks",
                "experimental_1h_peaks",
                "nmr_solvent",
                "solvent",
                "nmr_frequency",
                "nmr_freq",
                "experimental_total_nH",
                "candidate_smiles",
                "candidate",
            }
        },
    }
    # Preserve absent-versus-null for the fields that participate in identity.
    normalized = {
        "c_nmr_peaks": _finite(source.get("c_nmr_peaks")) if "c_nmr_peaks" in source or "experimental_13c_peaks" in source else None,
        "c_nmr_peaks_present": "c_nmr_peaks" in source or "experimental_13c_peaks" in source,
        "h_nmr_peaks": _finite(source.get("h_nmr_peaks")) if "h_nmr_peaks" in source or "experimental_1h_peaks" in source else None,
        "h_nmr_peaks_present": "h_nmr_peaks" in source or "experimental_1h_peaks" in source,
        "nmr_solvent": _presence(source, "nmr_solvent") if "nmr_solvent" in source else _presence(source, "solvent"),
        "nmr_frequency": _presence(source, "nmr_frequency") if "nmr_frequency" in source else _presence(source, "nmr_freq"),
        "experimental_total_nH": _presence(source, "experimental_total_nH"),
        "extra_keys": {
            str(key): _finite(source[key])
            for key in source
            if key not in {
                "c_nmr_peaks",
                "experimental_13c_peaks",
                "h_nmr_peaks",
                "experimental_1h_peaks",
                "nmr_solvent",
                "solvent",
                "nmr_frequency",
                "nmr_freq",
                "experimental_total_nH",
                "candidate_smiles",
                "candidate",
            }
        },
    }
    if "c_nmr_peaks" not in source and "experimental_13c_peaks" in source:
        normalized["c_nmr_peaks"] = _finite(source.get("experimental_13c_peaks"))
    if "h_nmr_peaks" not in source and "experimental_1h_peaks" in source:
        normalized["h_nmr_peaks"] = _finite(source.get("experimental_1h_peaks"))
    return {
        "status": "present",
        "raw_content_hash": sha256_text(canonical_json(raw_object)),
        "normalized_hash": sha256_text(canonical_json(normalized)),
        "normalized": normalized,
        "field_names": sorted(str(key) for key in source.keys()),
    }


def shift_sequence_hash(source: Mapping[str, Any] | None) -> str | None:
    """Hash ordered chemical-shift numbers only.

    This is a secondary key for finding training rows whose stored peaks dropped
    J values or integrals. It is not a claim that two spectra are identical.
    """
    if source is None:
        return None
    carbons = []
    for peak in source.get("c_nmr_peaks") or source.get("experimental_13c_peaks") or []:
        if isinstance(peak, dict):
            carbons.append(_finite(peak.get("delta (ppm)", peak.get("delta"))))
        else:
            carbons.append(_finite(peak))
    protons = []
    for peak in source.get("h_nmr_peaks") or source.get("experimental_1h_peaks") or []:
        if isinstance(peak, dict):
            protons.append(_finite(peak.get("centroid", peak.get("delta"))))
        else:
            protons.append(_finite(peak))
    if not carbons and not protons:
        return None
    return sha256_text(canonical_json({"c_deltas": carbons, "h_centroids": protons}))


def identity_token(keys: Mapping[str, Any]) -> str:
    """A stable structure token that does not drop unparsed SMILES."""
    status = keys.get("status")
    if status == "parsed":
        return "canonical:" + str(keys.get("canonical_isomeric"))
    if status == "missing":
        return "missing"
    return "unparsed:" + sha256_text("" if keys.get("raw") is None else str(keys.get("raw")))


def pair_token(candidate_keys: Mapping[str, Any], spectrum_hash: str | None) -> str:
    return identity_token(candidate_keys) + "|" + (spectrum_hash or "missing-spectrum")
