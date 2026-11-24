#!/usr/bin/env python3
"""Parse real_task.json (concatenated JSON objects, mixed schemas, invalid
SMILES backslash escapes) into a clean unified JSONL for evaluation.

Output records: {id, candidate_smiles, formula, c_nmr_peaks, h_nmr_peaks,
                 nmr_frequency, nmr_solvent, source_schema}
  - c_nmr_peaks: list of {"delta (ppm)": float}   (matches training format)
  - h_nmr_peaks: list of {category, centroid, delta, j_values, nH}

Source schemas seen:
  A. {nmr_text, data:[{NMR_type, ..., nmr_processed_dict}], SMILES, formula?}
  B. {h_nmr_peaks: [peak dicts], SMILES, formula?}
  C. {h_nmr_peaks: "1H NMR ... str", c_nmr_peaks: "13C NMR ... str", SMILES, formula?}
"""
import json
import re

SRC = "/fs_mol/cxq/SpecMaster_jp/data/real_task.json"
OUT = "/fs_mol/cxq/SpecMaster_jp/data/real_task_clean.jsonl"


def parse_concatenated(raw):
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    dec = json.JSONDecoder()
    objs, idx, n = [], 0, len(fixed)
    while idx < n:
        while idx < n and fixed[idx] in ' \t\r\n,':
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(fixed, idx)
        objs.append(obj)
        idx = end
    return objs


def parse_cnmr_string(s):
    if not s:
        return []
    body = s
    for sep in ('δ', ')'):
        if sep in body:
            body = body.split(sep)[-1]
    return [{"delta (ppm)": float(v)} for v in re.findall(r'-?\d+\.?\d+', body)]


def extract(obj):
    smiles = obj.get("SMILES", "")
    formula = obj.get("formula", "")
    c_peaks, h_peaks, freq, solvent = [], [], "", ""

    if "data" in obj:  # schema A
        for blk in obj["data"]:
            t = blk.get("NMR_type", "")
            pd = blk.get("nmr_processed_dict", [])
            if "13C" in t:
                c_peaks = pd
            elif "1H" in t:
                h_peaks = pd
                freq = blk.get("NMR_frequency", "")
                solvent = blk.get("NMR_solvent", "")
        schema = "A"
    else:  # schema B / C
        h = obj.get("h_nmr_peaks", [])
        if isinstance(h, list):
            h_peaks = h
            schema = "B"
        else:  # descriptive string
            schema = "C"
        c = obj.get("c_nmr_peaks", "")
        if isinstance(c, str):
            c_peaks = parse_cnmr_string(c)
        elif isinstance(c, list):
            c_peaks = c

    return {
        "candidate_smiles": smiles, "formula": formula,
        "c_nmr_peaks": c_peaks, "h_nmr_peaks": h_peaks,
        "nmr_frequency": freq, "nmr_solvent": solvent,
        "source_schema": schema,
    }


def main():
    objs = parse_concatenated(open(SRC).read())
    print(f"Parsed {len(objs)} objects")
    records = []
    for i, o in enumerate(objs):
        rec = extract(o)
        rec["id"] = i
        records.append(rec)

    from collections import Counter
    sch = Counter(r["source_schema"] for r in records)
    has_c = sum(1 for r in records if r["c_nmr_peaks"])
    has_h = sum(1 for r in records if r["h_nmr_peaks"])
    no_smiles = sum(1 for r in records if not r["candidate_smiles"])
    print(f"  schemas: {dict(sch)}")
    print(f"  with 13C: {has_c}/{len(records)}  with 1H: {has_h}/{len(records)}  missing SMILES: {no_smiles}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} -> {OUT}")


if __name__ == "__main__":
    main()
