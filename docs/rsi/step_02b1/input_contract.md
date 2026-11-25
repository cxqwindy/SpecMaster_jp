# Step 02B-1 input contract

Schema version `nmr-task-v2`. Protocol version `step_02b1.v1`.

The shared reader is `recipe/nmr_matching_agent/task_input.py` in Open-AgentRL. SpecMaster imports it through `scripts/rsi/nmr_lossless_task.py`. Standalone inference, the API, the vLLM client, the router, the RL tool, and the parquet converter call that reader. `scripts/build_test_2000.py` `format_alpaca` stays the historical writer and is not used for new data.

## Physical task

The model user message, the router record, and the tool `raw_prompt` may contain only:

- `candidate_smiles`
- `c_nmr_peaks`, each object keeping `delta (ppm)`
- `h_nmr_peaks`, each object keeping `category`, `centroid`, `delta`, `j_values`, `nH`, `rangeMin`, and `rangeMax` when those keys are present
- `nmr_solvent`
- `nmr_frequency`, the raw string
- `nmr_frequency_nucleus`, one of `1H`, `13C`, `spectrometer`, or `unknown`
- `experimental_total_nH` when the input itself provides it, including the value `0`

Precision, duplicate peaks, and peak order are kept. The reader does not deduplicate, sort, round, strip salts, or drop stereo. `normalize_task_input` returns a new object and is idempotent.

Gold, labels, `gt_smiles`, `negative_type`, sample ids, source paths, and review evidence do not enter those three surfaces. `idx` and other peak metadata are dropped. The original-input hash, field source, parse warnings, conflicts, and split id stay in the harness sidecar. The schema version may be recorded beside the physical object; it is not a physical field.

## Lossless text

The canonical text is `json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)`. JSON, the string reader, the API, standalone `make_messages`, the vLLM user turn, the router prefilled user turn, and the parquet converter emit that string. It does not round carbon shifts to 0.1 ppm or proton shifts to 0.01 ppm. A string that starts with `{` or `[` is parsed as a task. `NaN` and `Infinity` raise `TaskInputError`.

## Old Alpaca text

Commas inside parentheses stay inside one peak, so `J=7.3, 2.0` is one coupling string. The `J` pattern does not consume the integer of an `nH` token such as `3H`. Category, the raw `J` string, an explicit `nH`, and an explicit range are kept. A range sets `rangeMin` and `rangeMax` and does not invent a centroid. A single written shift is copied to both `delta` and `centroid` because the text does not distinguish them. `N/A` means no peaks. An unparseable token raises with its location. A missing optional `J` is not a failure. Empty parentheses leave `nH` absent. A bare shift does not become `nH` 1.

Data the old text does not contain stays absent. The reader does not restore a rounded display value to a finer source value.

## Structured input beside text

Alias priority is unchanged: `candidate_smiles` over `candidate`, `c_nmr_peaks` over `experimental_13c_peaks`, `h_nmr_peaks` over `experimental_1h_peaks`, `nmr_solvent` over `solvent`, and `nmr_frequency` over `nmr_freq`. An empty list is still false in that fallback, so an empty `c_nmr_peaks` falls through to `experimental_13c_peaks`. That behavior is locked by the existing step-01 test and is not a claim that `[]` and null are the same spectrum.

One-sided fields are kept. The same value repeated in text and structured input is kept. An explicit disagreement raises. Gold does not choose a side. Null versus a known value is a disagreement when both sides actually carry the field. Peak order is part of the comparison. Alias-priority changes remain pending.

## Empty list, null, and absence

Source records and exact fingerprints keep `[]`, null, and a missing key distinct. The runtime still projects a missing peak list with `or []` when it builds tool state. The sidecar records that projection. The projection does not replace the exact fingerprint. A task with no usable peak on either nucleus is rejected before a model or tool call.

## Frequency

The raw `nmr_frequency` string is unchanged. `nmr_frequency_nucleus` defaults to `unknown` when the key is absent, null, or empty. A magnitude is never used to guess the nucleus. Copying one Alpaca string onto both the 13C and 1H headers keeps one raw string and leaves the nucleus unknown; the sidecar warning is `header_value_copied`. Different header strings are both kept in the sidecar as `header_values_differ`. Neither string is selected, and they are not converted by an approximate factor of four. A header that says `13C NMR (400 MHz)` does not assign the nucleus. A known nucleus is stored only when the input itself has an explicit trusted field whose value is `1H`, `13C`, or `spectrometer`.

Retaining solvent and frequency does not mean the spectrum predictor consumes them. `predict_nmr_peaks` takes a SMILES string. Runtime spectral prediction reads the candidate and the prediction cache. The router variant hash includes solvent and frequency as prompt identity. `h_tolerance` stays 2. The similarity thresholds stay 0.85 and 0.92.

## Integrals

Missing or null `nH` stays unknown on the string reader, CLI, API, vLLM record, RL tool, and `count_atoms` input. `0` is a real value. A complete peak-table sum exists only when every participating peak integral is known and finite. A partial sum is reported as `known_nH_sum` with completeness `partial`. It is not `experimental_total_nH` and it is not a hydrogen-count conflict. An explicit total is an independent observation and is passed through every entry, including `call_v5_vllm`. If an explicit total and a complete peak-table sum disagree, both numbers are retained, `integral_conflict` is true, and neither is chosen as the comparison total. The candidate's theoretical hydrogen count does not fill the experimental total. When `experimental_total_nH` is absent, the router treats the hydrogen count as unavailable, so the result is not a definite No. Bool and non-finite integrals are errors. Numeric strings are not coerced.

## RL rows

The tool projects the physical task out of the user JSON. Reward `ground_truth.candidate_smiles` does not fill or replace the task candidate. A gold candidate that differs from the task candidate does not raise, because a negative may differ. A missing task candidate is an error and is not recovered from gold. A legal outer task schema is used when present. `_gt_smiles` is not a candidate. Historical `_candidate_smiles` is only a consistency check; a conflict is rejected and located. Existing parquet files are not rewritten.

## Source eligibility

Eligibility excludes the confirmed v6.4 train exposure, the inferred v6.1 and v5 train exposure, exact spectrum and pair exposure, known evaluation structures, test_2000 ground truth, test_2000 candidates, and the unknown cot possible structures. The 02A field `weight_train` is the union of confirmed and inferred exposure. It is historical and is not a confirmed-only set. Inferred rows stay excluded. Wording for a row that remains is “未出现在所列已知/推断暴露与评测索引中.” That is not a claim that a base model never saw the molecule.

Structure identity uses RDKit 2026.03.1, `MolFromSmiles` defaults, and `MolToSmiles(canonical=True, isomericSmiles=True)`. Unparsed SMILES stay as `unparsed:<sha256>`. No salt strip, tautomer merge, or stereo drop is applied.

`gold_unresolved` rows stay in the denominator and out of the automatic positive-generation pool. Missing `nH`, a single nucleus, or an unknown frequency is coverage, not a quality rejection. This step does not choose search-dev or gate-dev, a sample size, a class ratio, or a promotion threshold, and it does not call the negative predictor.

## Still pending

The nucleus of each source frequency string, the checkpoint-200 dataset, the Qwen pretraining corpus, the `observation_fixed` command log, v6 sibling ground truth, whether `[]` and null should later be unified, whether a future formatter may render a missing `nH` as 1, alias priority, development-set size, positive/negative ratio, difficulty, the ambiguity-flag rule, the promotion margin, and the statistical criterion.
