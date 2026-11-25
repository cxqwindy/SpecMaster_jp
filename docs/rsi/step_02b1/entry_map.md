# Step 02B-1 entry and field map

Every live entry calls `recipe/nmr_matching_agent/task_input.py`. The user string is `task_to_lossless_text`.

| Entry | Function | What it passes on | What it keeps out |
| --- | --- | --- | --- |
| Standalone inference | `prepare_eval_record`, `make_messages` | `inference_input` only. The user turn is the canonical JSON. | `evaluation_meta` and `input_sidecar`. Gold is not an argument to the model or router. |
| Text plus structured merge | `_compose_inference_input` | One-sided physical fields and exact repeats. | A disagreement raises. `_candidate_smiles` is not a candidate source. |
| vLLM client | `user_record_from_case`, `run_case` | Physical fields, including explicit `experimental_total_nH` of 0. The user turn is the canonical JSON. | Empty spectra raise before HTTP. `--frequency` and `--solvent` are caller defaults, not measured nucleus assignments. |
| API | `MatchRequest.infer` | `normalize_task_input(payload.model_dump())`, then the same router and agent path. | `nmr_frequency_nucleus` omitted by the caller becomes `unknown`. |
| Router | `RuleBasedRouter`, `build_prefilled_messages` | Normalized physical record. Solvent and frequency enter the variant hash. | Labels. An unavailable hydrogen total is not a definite No. |
| Runtime tools | `NmrMatchingRuntime` | Candidate, peak shifts, and the integral description. | Solvent and frequency are not predictor conditions. |
| RL tool | `_build_record_from_user_json` | Physical projection of the user JSON. | `idx`, labels, gold, and arbitrary peak metadata. |
| Parquet converter | `_row_from_parquet_row`, `_row_from_alpaca` | Task candidate and the canonical user JSON. | Reward `ground_truth` stays in the reward sidecar. `_gt_smiles` cannot become the candidate. A conflicting `_candidate_smiles` raises. |
| SpecMaster adapter | `scripts/rsi/nmr_lossless_task.py` | Re-exports the Open-AgentRL reader. | Does not call `format_alpaca` and does not rewrite training files. |
| Historical writer | `scripts/build_test_2000.py` `format_alpaca` | Unchanged reproduction path. | Not the new-data path. |

## Which tool reads which field

| Tool or component | Reads | Does not read |
| --- | --- | --- |
| `count_atoms` for carbon | Candidate SMILES and experimental carbon shifts | Solvent, frequency, gold |
| `count_atoms` for hydrogen | Candidate hydrogen count and `describe_integrals` | A partial peak-table sum used as the experimental total |
| `spectral_match` and peak assignment | Candidate SMILES, prediction cache, experimental shifts | `nmr_solvent`, `nmr_frequency` |
| `predict_nmr_peaks` | SMILES only | Solvent and frequency |
| `functional_group_scan` | Candidate structure | Spectrum metadata |
| Router variant hash | Solvent and frequency as prompt identity | They are not converted into predictor settings |

`h_tolerance` remains 2. Similarity thresholds remain 0.85 and 0.92. The five tool names, the predictor, the weights, and the execution policy are unchanged.
