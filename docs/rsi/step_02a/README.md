# Step 02A data audit

This step records where the current v6.4 checkpoint-63 data came from, which molecules and spectra are already exposed, and which input fields survive the present formatters. It does not change labels, parsers, prompts, or the test set, and it does not freeze a development split.

The reproducible entry point is `scripts/rsi/step_02a_audit.py`. After the LMDB scan it calls `scripts/rsi/step_02a_refine.py` to recount isolation. A missing spectrum is not treated as one shared spectrum, and a shift-only fingerprint is reported without removing the row. Numeric results belong in `rsi_runs/step_02a_*` because the indexes are large and the source files stay outside git.

## Status labels

- `confirmed`: a training log, trainer state, and the matching yaml agree, or the generator code directly copies the field.
- `inferred`: a yaml and a trainer state point at the same run, but no log names the dataset file.
- `unknown`: the file or checkpoint exists, but its membership in the v6.4 weight chain is not established.

Step 02B-1 splits that wording. Only the v6.4 checkpoint-63 registered train rows are `confirmed`. The v6.1 and v5 train files remain `inferred`: a yaml and a trainer state point at the run, but no log names the file. `full-exp1/checkpoint-200` is still `unknown`; `nmr_matching_cot_train.json` and its validation file are only possible sources. The historical field `weight_train`, and any 02A count labeled as the confirmed train set, is the union of confirmed v6.4 plus inferred v6.1 and v5. That union is 11995 rows and 2087 unique ground-truth structures. The 771 test-set ground-truth intersection is the union, not pure confirmed. Do not keep using `weight_train` as if it were confirmed-only. Conservative eligibility still removes the whole union; inferred rows are not put back. Sibling runs v6, v6.2, and v6.3 stay outside the weight chain.

## Fingerprints

RDKit version and the exact `MolToSmiles` options are written into `lineage_report.json`. Unparsed SMILES remain in the row count. The spectrum hash keeps peak order, missing keys, null integrals, and J values. A second shift-sequence hash uses only ordered carbon deltas and proton centroids so rows that dropped J or integrals can still be found; it is not a declaration that those spectra are identical. No ppm tolerance is used.

## What remains pending

The nucleus represented by `nmr_frequency`, the positive/negative ratio of a future development set, ambiguity flags, and any promotion threshold are not chosen here. LMDB spectra are called experimental by the generator help text; this audit does not treat that wording as a data card.
