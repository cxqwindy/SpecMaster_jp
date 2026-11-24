"""NMR forward prediction API wrapper.

Calls the NMR_predict service to obtain per-atom chemical shifts,
then extracts ¹³C and ¹H peak lists from the atom-mapped output.

Backend selection (auto):
  - NMR_PREDICT_URL env var (e.g. "http://192.168.201.63:7813") — HTTP resident service
    with per-molecule POST /predict. Preferred if set.
  - Otherwise fall back to the legacy fastmcp SSE endpoint via get_shifts_api.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP backend (new resident service).
# ---------------------------------------------------------------------------
# Public: 124.174.48.139:7813 ; VPC: 192.168.201.63:7813 ; container: :8080.
# Set NMR_PREDICT_URL to any of them; default is VPC (lower latency in-cloud).
_HTTP_URL = os.environ.get("NMR_PREDICT_URL", "http://192.168.201.63:7813").rstrip("/")
_HTTP_TIMEOUT_S = float(os.environ.get("NMR_PREDICT_TIMEOUT_S", "60"))
_HTTP_MAX_WORKERS = int(os.environ.get("NMR_PREDICT_MAX_WORKERS", "16"))
_HTTP_TOKEN = os.environ.get("NMR_PREDICT_TOKEN", "")  # optional X-API-Token


def _http_predict_one(smiles: str, solvent: str = "CDCl3", mhz_h: float = 400.0) -> Optional[Dict]:
    """POST /predict for a single SMILES. Returns raw API dict or None on error."""
    body = json.dumps({"smiles": smiles, "solvent": solvent, "mhz_h": mhz_h,
                       "plot": False, "return_image": "none"}).encode()
    headers = {"Content-Type": "application/json"}
    if _HTTP_TOKEN:
        headers["X-API-Token"] = _HTTP_TOKEN
    req = urllib.request.Request(f"{_HTTP_URL}/predict", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
        logger.debug("HTTP predict failed for %r: %s", smiles[:60], e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("HTTP predict failed for %r (unknown): %s", smiles[:60], e)
        return None


# Fallback to legacy fastmcp SSE (used only if NMR_PREDICT_URL_LEGACY=1).
_LEGACY_API_PATH = "/fs_mol/geyutang/proteinlm_dataset/Open-AgentRL/nmr/rank_local"
if _LEGACY_API_PATH not in sys.path:
    sys.path.insert(0, _LEGACY_API_PATH)


def _canonical(smiles: str) -> Optional[str]:
    """Canonical SMILES (no atom maps, implicit H) for order-independent matching.

    The remote batch API does NOT preserve input order in its results, so we must
    align each returned spectrum back to its query by canonical SMILES rather than
    by list position. Accepts both plain and atom-mapped/explicit-H SMILES.
    """
    if not smiles:
        return None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        # fall back to a lenient parse for plain query SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def _extract_peaks_from_result(result: Dict) -> Dict:
    """Extract C13 and H1 peak lists from a single API result.

    Preferred path: use the API-returned C_groups / H_groups, where each group
    contains the map numbers of chemically equivalent atoms (e.g. the 3 H of a
    CH3 group, or two carbons made equivalent by molecular symmetry).  One peak
    is emitted per group (the average shift of its members).  This is correct by
    construction — no heuristic ±ppm merging needed or wanted.

    Fallback path (C_groups / H_groups absent — legacy API or edge case): parse
    atom shifts from the atom-mapped SMILES directly, then apply conservative
    merging (same tolerance as before) so old code paths still work.
    """
    shifts = result.get("atoms_shift", [])
    smi = result.get("smiles_with_atom_order", "")
    c_groups = result.get("C_groups")   # list of lists of 1-based map numbers
    h_groups = result.get("H_groups")

    if not shifts:
        return {"c_peaks": [], "h_peaks": [], "ok": False}

    canon = _canonical(smi) if smi else None

    # --- preferred path: API-provided grouping ----------------------------
    if c_groups is not None and h_groups is not None:
        def _group_peak(group: List[int]) -> Optional[float]:
            vals = [float(shifts[m - 1]) for m in group
                    if 0 < m <= len(shifts) and shifts[m - 1] is not None]
            return sum(vals) / len(vals) if vals else None

        c_peaks = sorted(v for g in c_groups if (v := _group_peak(g)) is not None)
        h_peaks = sorted(v for g in h_groups if (v := _group_peak(g)) is not None)
        return {
            "c_peaks": c_peaks, "h_peaks": h_peaks, "ok": True,
            "_canonical": canon,
            # atom-level data retained for peak_assignment tool
            "atoms_shift": shifts,
            "smiles_with_atom_order": smi,
            "C_groups": c_groups,
            "H_groups": h_groups,
        }

    # --- fallback path: atom-by-atom from SMILES --------------------------
    if not smi:
        return {"c_peaks": [], "h_peaks": [], "ok": False}

    _parser_params = Chem.SmilesParserParams()
    _parser_params.removeHs = False
    mol = Chem.MolFromSmiles(smi, _parser_params)
    if mol is None:
        return {"c_peaks": [], "h_peaks": [], "ok": False}

    c_raw: List[float] = []
    h_raw: List[float] = []
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num <= 0 or map_num > len(shifts):
            continue
        shift_val = shifts[map_num - 1]
        if shift_val is None:
            continue
        z = atom.GetAtomicNum()
        if z == 6:
            c_raw.append(float(shift_val))
        elif z == 1:
            h_raw.append(float(shift_val))

    c_peaks = _merge_equivalent_peaks(sorted(c_raw), tol=0.5)
    h_peaks = _merge_equivalent_peaks(sorted(h_raw), tol=0.1)
    return {"c_peaks": c_peaks, "h_peaks": h_peaks, "ok": True,
            "_canonical": canon}


def _merge_equivalent_peaks(sorted_peaks: List[float], tol: float = 0.5) -> List[float]:
    """Merge peaks that are within ±tol ppm of each other (equivalent atoms)."""
    if not sorted_peaks:
        return []
    merged = []
    group = [sorted_peaks[0]]
    for p in sorted_peaks[1:]:
        if p - group[-1] <= tol:
            group.append(p)
        else:
            merged.append(sum(group) / len(group))
            group = [p]
    merged.append(sum(group) / len(group))
    return merged


def _use_legacy() -> bool:
    """Only fall back to fastmcp/SSE if explicitly requested."""
    return os.environ.get("NMR_PREDICT_URL_LEGACY", "0") == "1"


def predict_nmr_peaks(smiles: str) -> Dict:
    """Predict NMR peaks for a single SMILES.

    Returns:
        {"c_peaks": [float, ...], "h_peaks": [float, ...], "ok": bool}
    """
    try:
        if _use_legacy():
            from get_shifts_api import get_shifts_sync
            results = get_shifts_sync([smiles])
            if not results:
                return {"c_peaks": [], "h_peaks": [], "ok": False}
            return _extract_peaks_from_result(results[0])

        api_result = _http_predict_one(smiles)
        if api_result is None:
            return {"c_peaks": [], "h_peaks": [], "ok": False}
        return _extract_peaks_from_result(api_result)
    except Exception as e:
        logger.debug("NMR prediction failed for %s: %s", smiles, e)
        return {"c_peaks": [], "h_peaks": [], "ok": False}


def predict_nmr_batch(
    smiles_list: List[str],
    batch_size: int = 50,
) -> List[Dict]:
    """Predict NMR peaks for a batch of SMILES, with error tolerance.

    Uses the HTTP resident service (per-molecule POST) with a thread pool for
    concurrency. `batch_size` is repurposed as the concurrency (max workers)
    cap for the HTTP backend so existing callers don't need to change.
    Falls back to fastmcp/SSE batch API when NMR_PREDICT_URL_LEGACY=1.

    Returns:
        list of dict aligned with input; failed predictions have ok=False.
    """
    n = len(smiles_list)
    all_results: List[Optional[Dict]] = [None] * n
    query_canon = [_canonical(s) for s in smiles_list]

    if _use_legacy():
        from get_shifts_api import get_shifts_sync
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch = smiles_list[batch_start:batch_end]
            try:
                api_results = get_shifts_sync(batch)
            except Exception as e:
                logger.warning("Batch API call failed (start=%d): %s", batch_start, e)
                for i in range(batch_start, batch_end):
                    all_results[i] = {"c_peaks": [], "h_peaks": [], "ok": False}
                continue
            canon_to_peaks: Dict[str, Dict] = {}
            for res in api_results:
                peaks = _extract_peaks_from_result(res)
                key = peaks.get("_canonical")
                if key and key not in canon_to_peaks:
                    canon_to_peaks[key] = peaks
            for i in range(batch_start, batch_end):
                peaks = canon_to_peaks.get(query_canon[i])
                all_results[i] = peaks if peaks is not None \
                    else {"c_peaks": [], "h_peaks": [], "ok": False}
    else:
        # HTTP backend: per-molecule POST, fan out with a thread pool.
        # `batch_size` doubles as the max concurrent workers (bounded to sensible range).
        max_workers = max(1, min(_HTTP_MAX_WORKERS, batch_size))
        # Preserve input order and align by canonical SMILES from the returned
        # atom-mapped SMILES (defensive; the HTTP service does preserve order,
        # but we keep the alignment machinery for robustness).
        def _one(i_smi):
            i, smi = i_smi
            api = _http_predict_one(smi)
            if api is None:
                return i, {"c_peaks": [], "h_peaks": [], "ok": False}
            return i, _extract_peaks_from_result(api)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, peaks in ex.map(_one, enumerate(smiles_list)):
                all_results[i] = peaks

    # Final safety check
    for i in range(n):
        if all_results[i] is None:
            all_results[i] = {"c_peaks": [], "h_peaks": [], "ok": False}

    return all_results
