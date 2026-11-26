# Step 02B-3 development sets

Protocol `step_02b3.v1`. This step builds two real development datasets and checks their isolation. It does not run a model, an NMR predictor, a spectral-similarity tool, a router, or a teacher. It does not start a GPU job, an API call, or training. It does not enter step 03, does not build a predicted-spectrum challenge set, and does not enable promotion.

The published parameter file is `docs/rsi/step_02b3/build_config.json`. `scripts/rsi/step_02b3_dev.py` loads that file. `scripts/rsi/step_02b3_build_dev.py` is the CLI.

## What is being built

`search-dev` and `gate-dev` each contain 500 parent-GT groups when the quota is met. Each group uses one source spectrum, one positive candidate, and three mutually different negative candidates. The first version uses one same-formula negative, one structural neighbor, and one hash-ranked draw. A completed split has 2,000 rows: 500 with gold 1 and 1,500 with gold 0. The two splits together have 1,000 parent groups and 4,000 rows.

Five hundred groups is an engineering starting size. It is not a claim that a statistical power target has been met. The draw is conditional on being able to build all three negative types. It is not an unconditional random sample of `valid.lmdb`.

Gold is the real structure identity. A tool threshold is not gold. `difficulty` stays unknown and `ambiguity` stays unannotated. Stereo isomers are not deleted, relabeled, or turned into an uncertain output. `negative_type`, Tanimoto, same-formula, and same-connectivity-ignoring-stereo are construction metadata. They are not a verified difficulty and they are not evidence that two spectra are indistinguishable.

## Sources

The eligibility index is the accepted 02B-1 file named in the parameter file, checked by byte size and SHA-256, not by name. A row enters the valid pool only when `in_auto_positive_pool` is true, `evidence_bucket` is `remaining`, `quality_status` is `not_rejected`, and both reason lists are empty. The reference pool sizes are 48,234 valid, 916,419 train, and 104,695 test. Those counts are not rewritten as "the base model has never seen these structures."

Only the valid pool is decoded from `valid.lmdb`. Train stays a future-train reservation. Test stays unused. `test.lmdb` and `test_2000` are different objects. The original LMDB filenames do not by themselves prove isolation from historical training. `all.lmdb` is not a donor pool. Atoms and coordinates on a raw record are theoretical fields; they are counted and are not copied into the experimental task.

The parent identity is the 02A RDKit token: `MolFromSmiles` defaults and `MolToSmiles(canonical=True, isomericSmiles=True)`. Stereo, salts, and charge stay. Tautomers are not merged. A candidate row number is not an identity. If one parent token has more than one valid-pool source row, the build records every locator and stops before a dataset is written. It does not pick a spectrum.

## Ownership

Valid-pool tokens are sorted by SHA-256 of the canonical JSON array `[ownership domain, seed, token]`. Python `hash()` is not used. Index 0, 2, 4, ... go to `search-dev`. Index 1, 3, 5, ... go to `gate-dev`. An odd count gives the extra structure to `search-dev`. This assignment is frozen before any negative is chosen. Every source and donor record of a token stays on that side.

`ownership.jsonl` also registers the future-train and test reservation tokens. Isolation is checked on parent GT, on candidates, and on their union, against the other split, the reservations, the exposure tokens in the eligibility index, and the `test_gt_structure` / `test_candidate_structure` tokens. Scaffold overlap and connectivity-ignoring-stereo overlap are reported and are not required to be zero. This is not a scaffold split.

## Negatives

A source is constructable inside its frozen subpool when it has at least one different structure with the same RDKit molecular formula and at least two other structures whose formulas differ from its own. Formula is `CalcMolFormula` on the verified molecule. A disagreement with the stored formula field is counted. A calculation failure is listed and is not replaced with the stored field. The stored formula is not gold and is not added to the experimental input.

Selection takes the first 500 constructable sources in an independent hash order. If either side has fewer than 500, or a selected source later fails, the run stays `incomplete`. The builder does not copy rows, change the 1:3 ratio, change the seed, widen the donor pool, or replace a source to fill the quota.

Construction order, which is not the output order:

1. `neg/same_formula`: minimum hash among other same-formula structures in the subpool. This can be a stereo-only difference. The relation is recorded and is not relabeled.
2. `neg/similar_structure`: maximum Morgan Tanimoto among different-formula structures. Radius 2, 2048 bits, chirality off. Ties break by the structure token ascending. The similarity is the relative nearest neighbor. No absolute difficulty cutoff is applied.
3. `neg/random`: minimum hash in a separate domain among different-formula structures, excluding the parent and the candidates already chosen. "Random" names the draw, not a proven easy item.

The positive candidate is the parent token. All four rows use the parent spectrum. Donor spectra are not substituted. The optimized Tanimoto query is checked against an exact bit-count query, including ties and exclusions. A fingerprint cache is stored only under the new run directory and is refused when its eligibility-file hash, ownership hash, RDKit version, fingerprint parameters, protocol, token order, or structure-file hash does not match. Existing `.pool_cache` files are not read or written.

Different groups may share a donor. Reuse is counted. A repeated `(parent, full spectrum, candidate, label)` tuple is a failure. Shared candidates are not extra parent groups.

## Tasks and files

Each split directory contains `model_inputs.jsonl`, `dataset_manifest.json`, `gold_sidecar.json`, `input_sidecars.jsonl`, `source_records.jsonl`, `source_groups.jsonl`, `pair_to_record.json`, `ownership_ref.json`, `config.json`, and `dataset_protocol.json`.

The model-facing object is the normalized `nmr-task-v2` task. Only `candidate_smiles` changes inside a group. Missing `nH`, unknown frequency, and one-nucleus spectra keep the accepted meanings: no default integral, no frequency-based drop, no peak rounding. Raw missing, null, and `[]` stay distinct in the sidecar. A runtime empty list does not replace the raw spectrum fingerprint. The sidecar is not a model input.

`record_id` is a SHA-256 over the protocol, source file, source key, role, and candidate token. `pair_id` follows one hash order for the whole split, including 0. It is not reset per shard and it is not a gold rule. The run does not write a run manifest or a prediction file.

`config.json` uses `balanced_accuracy_valid`, `exclude_from_primary`, `statistics.enabled=false`, `promotion.enabled=false`, and `example_status=null`. Real data is marked `real_development_dataset`. The synthetic helper `scoring_config()` is not used. Official files are checked with `prepare_dataset` and `check_gold_sidecar`. Evaluate and compare stay on the separate synthetic fixtures from 02B-2.

## Status

`prepared_pending_independent_review` is written only when both real splits have 500 groups and 2,000 rows and every audit passes. Otherwise the status is `incomplete` and the raw counts stay. A frozen output directory is not overwritten. The logical data hash covers the rows, ownership, dataset hashes, and code version. Output paths and timestamps stay in `runtime_info.json`.

Gate rows are not for model tuning. Row-level gate structures stay in `rsi_runs` and are not copied into this document.

## CLI

```text
python -B scripts/rsi/step_02b3_build_dev.py plan --out RUN --mode real
python -B scripts/rsi/step_02b3_build_dev.py build --out RUN --mode real
python -B scripts/rsi/step_02b3_build_dev.py validate --out RUN
python -B scripts/rsi/step_02b3_build_dev.py rebuild --from-run RUN --out RUN/rebuild_check
```

Exit 0 means a real run is `prepared_pending_independent_review`, or a fixture run is `fixture_checked`. Exit 2 means `incomplete`. Exit 3 is a refusal: a frozen output directory, a mismatched fingerprint cache, or a real run asked to change the 500-group quota. A repeated full raw-spectrum fingerprint across the two splits stops the freeze. Shift-sequence overlap does not.
