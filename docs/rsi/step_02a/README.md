# Step 02A data audit

This step records where the current v6.4 checkpoint-63 data came from, which molecules and spectra are already exposed, and which input fields survive the present formatters. It does not change labels, parsers, prompts, or the test set, and it does not freeze a development split.

The reproducible entry point is `scripts/rsi/step_02a_audit.py`. After the LMDB scan it calls `scripts/rsi/step_02a_refine.py` to recount isolation. A missing spectrum is not treated as one shared spectrum, and a shift-only fingerprint is reported without removing the row. Numeric results belong in `rsi_runs/step_02a_*` because the indexes are large and the source files stay outside git.

## Status labels

- `confirmed`: a training log, trainer state, and the matching yaml agree, or the generator code directly copies the field.
- `inferred`: a yaml and a trainer state point at the same run, but no log names the dataset file.
- `unknown`: the file or checkpoint exists, but its membership in the v6.4 weight chain is not established.

The weight chain used for the confirmed exposure set is checkpoint-63, its v6.4 train rows, the v6.1 train file, and the v5 train file. `full-exp1/checkpoint-200` is the configured initializer of v5, but the dataset that produced it is unknown. `nmr_matching_cot_train.json` is therefore an unresolved possible exposure, not a clean source. Sibling runs v6, v6.2, and v6.3 are reported separately and are not treated as parents of checkpoint-63.

## Fingerprints

RDKit version and the exact `MolToSmiles` options are written into `lineage_report.json`. Unparsed SMILES remain in the row count. The spectrum hash keeps peak order, missing keys, null integrals, and J values. A second shift-sequence hash uses only ordered carbon deltas and proton centroids so rows that dropped J or integrals can still be found; it is not a declaration that those spectra are identical. No ppm tolerance is used.

## What remains pending

The nucleus represented by `nmr_frequency`, the positive/negative ratio of a future development set, ambiguity flags, and any promotion threshold are not chosen here. LMDB spectra are called experimental by the generator help text; this audit does not treat that wording as a data card.
