# Step 02B-2 offline evaluator

Evaluator version `step_02b2.v1`. Task schema remains the accepted `nmr-task-v2`. This step scores already written predictions. It does not choose a development split, call a model, or enable promotion.

The primary metric is `balanced_accuracy_valid`. `invalid_policy` is `exclude_from_primary`. An invalid terminal stays in the coverage denominator and stays out of the classification denominator. That choice is fixed for this protocol. It is not a default waiting for another confirmation, and it is not implemented as "count invalid as wrong."

## Files

| File | Role |
| --- | --- |
| `scripts/rsi/matching_evaluator.py` | Validation, parsing, metrics, grouped resampling |
| `scripts/rsi/step_02b2_evaluate.py` | CLI |
| `scripts/rsi/step_02b2_fixture.py` | Synthetic input writer only |

```text
python -B scripts/rsi/step_02b2_evaluate.py evaluate \
  --dataset dataset_manifest.json \
  --gold gold_sidecar.json \
  --run run_manifest.json \
  --predictions predictions.json \
  --config config.json \
  --out report.json
```

`compare` takes `--run-a`, `--predictions-a`, `--run-b`, and `--predictions-b` for the same dataset and gold sidecar. Exit 0 means the report was written and its integrity status is `passed` or, for an explicit legacy read, `legacy_diagnostic`. Exit 2 means the report was written and the formal primary score was not published. Exit 3 refuses `promotion.enabled=true`. Exit 4 is a rejected configuration.

## What is scored

Gold is an integer 0 or 1 on the gold sidecar. 0 is No and 1 is Yes. A bool is not a gold label. The dataset manifest carries the same gold, parent GT, and source group; if the two files disagree, scoring stops. A `_label`, `gold_label`, or `gt` left on a prediction may be checked against the sidecar and cannot replace it. Tool output, confidence, and a world-model score are not labels.

`source_group_id` is `step_02a.identity_token` of the parent GT SMILES: RDKit `MolFromSmiles` defaults and `MolToSmiles(canonical=True, isomericSmiles=True)`, or `unparsed:<sha256>` when the SMILES does not parse. Salts, tautomers, and stereo are not removed. The id is not taken from the candidate or from a row number. Records that share a parent GT stay in one group. The primary metric still counts candidates, not the mean of group scores.

The physical task is checked with `normalize_task_input`. Gold keys inside the task are integrity errors. A hash stored for the task is the hash of that normalized manifest task. The evaluator does not copy it into `model_observed_task_sha256`. An observed hash is compared only when the prediction itself records one.

## Profiles

The prediction file names its profile. Columns are not guessed.

- `canonical`: `terminal.state` is `answer`, `timeout`, `max_turns`, `empty_answer`, or `parse_error`. A valid row has state `answer` and `final_answer` Yes or No. A failure state together with a Yes/No answer is `terminal_conflict`.
- `standalone`: the production `answer` and `fail_reason` fields, with `_pair_id` / `pair_id` and the harness run id. Transcript prose is not read.
- `api`: the response object's `final_answer`. Trajectory `final_answer` values must agree with it. `key_evidence` and `final_think` are not answers.
- `legacy_diagnostic`: explicit only. Rows are paired in written file order, `pairing_verified` is false, and no formal primary score or A/B comparison is published.

A Yes/No value may have surrounding whitespace and either case. `true`, `false`, `0`, `1`, `2`, `-1`, `0.7`, and bools are invalid. They are not coerced. Truncated JSON is `parse_error`. Two `<answer>` blocks are `multiple_answers` even when they agree. A Yes/No that appears only in thinking, a tool response, or gold is not a prediction.

A missing prediction is an integrity error. It is not recorded as a timeout. A complete run may contain explicit timeout rows; those rows remain in the coverage counts.

## Denominators

`N` is the number of records that passed the integrity checks. `V` is the number with a legal Yes/No. `I = N - V`. The same letters with a class suffix use gold 0 or gold 1. Coverage is `V/N`. The valid confusion matrix is gold rows `[0, 1]` and prediction columns `[0, 1]`, `[[TN, FP], [FN, TP]]`, and its sum is `V`. The diagnostic table is gold by `{No, Yes, invalid}` and its sum is `N`.

`recall_no_valid = TN/(TN+FP)`, `recall_yes_valid = TP/(TP+FN)`, and balanced accuracy is their mean. Accuracy is `(TP+TN)/V`. The two F1 scores use the usual binary denominators, and macro-F1 is their mean. Every one of those denominators uses valid predictions only. A zero denominator is `null` plus a reason. It is not NaN, infinity, or zero filled in for a missing class.

`V = 0` sets the classification metrics to null and `status` to `no_valid_predictions`. If one class has no valid prediction, balanced accuracy and macro-F1 are null with `missing_valid_class`; accuracy and the defined class recall remain. An empty manifest is an integrity failure, not a score of zero. Difficulty and ambiguity values that are absent are reported as `unknown` and `unannotated`. They are not inferred and the rows are not dropped.

## Comparison and resampling

Both runs must pass integrity on the same dataset hash, gold, source groups, and evaluator version. Their `run_id` values may differ. The report gives each run's own-valid balanced accuracy and `delta_ba_own_valid = B - A`, plus coverage differences. It also gives the intersection `V_A ∩ V_B`. That common-valid difference is auxiliary. If the intersection is empty or misses a class, the common metrics are null and the evaluator does not substitute the own-valid scores. A higher own-valid score is not a promotion.

Statistics are off unless the config sets `statistics.enabled` to true and supplies `bootstrap_replicates`, `confidence_level`, `seed`, and `method`. The only method in this step is `paired_cluster_percentile`. Sampling draws the sorted unique `source_group_id` values with replacement, using `random.Random(seed)` and `randrange`. A group drawn twice has weight two and contributes every spectrum and candidate twice. The same index list is used for both runs. Row order does not change the sorted group index. If fewer than two groups exist, the point estimate is undefined, or any replicate is undefined, the interval is null and the undefined count is kept. A replicate is not dropped, replaced by zero, or redrawn until the interval looks complete. An interval of all zeros is reported as a constant zero. It is not a significance claim. This step does not emit `p < 0.05` or a pass/fail threshold. The synthetic configs in tests are not a formal confidence level.

## Cost and promotion

Cost uses every attempted record, including invalid ones. Known tool calls, cache hits, network calls, tokens, and seconds are summed only over fields that are actually present. A partial sum is not the total cost. An absent field is `not_recorded`, not zero. String length is not a token count. In a trace, each `type: tool` event counts once and an `answer` event does not. Cache class comes from the event's `cache` field, not from the tool name. Unknown cache status leaves the cache counts null.

`promotion.enabled` must be false. Every evaluate and compare report sets `promotion.status` to `disabled_pending_protocol` and does not change a model pointer. Setting enabled to true is rejected.

## Synthetic example

`expected_hand.json` records the 12-row hand calculation: `N=12`, `V=9`, confusion `[[3, 2], [1, 3]]`, balanced accuracy `27/40`. The draw `[gA, gA, gD, gD]` has weighted `N=12`, `V=8`, and balanced accuracy `1/2`. Model B has two valid rows and balanced accuracy `1/1`; the own-valid difference is `13/40`, the common-valid difference is `0`, and the coverage cells are `2 / 7 / 0 / 3`. Those scores are a parser and arithmetic check. They are not a model baseline.

The old `scripts/summarize_matching_eval.py` and `scripts/compare_matching_evals.py` paths are unchanged historical tools. They are not this protocol. Training reward is unchanged.
