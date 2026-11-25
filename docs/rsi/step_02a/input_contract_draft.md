# Input contract draft

This draft describes the fields already present in the current task objects. It is not a parser change. Step 02B-1 supersedes it: the shared contract is `docs/rsi/step_02b1/input_contract.md`. The observations below stay historical.

## Fields to keep together

- `candidate_smiles`, with alias `candidate`
- `c_nmr_peaks`, with alias `experimental_13c_peaks`; each carbon peak uses `delta (ppm)` when it is an object
- `h_nmr_peaks`, with alias `experimental_1h_peaks`; peak objects may contain `category`, `centroid`, `delta`, `j_values`, `nH`, `rangeMax`, and `rangeMin`
- `nmr_solvent`, with alias `solvent`
- `nmr_frequency`, with alias `nmr_freq`
- `experimental_total_nH`

## Current priority

Inside one structured object, the existing normalizer keeps `candidate_smiles` over `candidate`, `c_nmr_peaks` over `experimental_13c_peaks`, `h_nmr_peaks` over `experimental_1h_peaks`, `nmr_solvent` over `solvent`, and `nmr_frequency` over `nmr_freq`. A field supplied by only text or only structured input is kept. The same candidate repeated beside text does not drop the text peaks. An explicit disagreement raises, and the label is not used to choose a side. A null `nH` stays null.

## Pending

- `nmr_frequency` strings in the source LMDB include values such as `100 MHz`, `101 MHz`, `125 MHz`, `150 MHz`, and `400 MHz`. This audit does not decide whether each string is the spectrometer's nominal frequency or one nucleus's frequency. `format_alpaca` copies that one string into both the 13C and 1H headers.
- Some Alpaca formatters substitute `1` when `nH` is missing. That is recorded as information loss, not adopted as the desired contract.
- Ambiguity flags, class balance, and the numeric margin for promoting a model are not part of this draft.

Measured field counts and the current mixed-input outcomes are in the step-02A run directory, in `input_preservation_report.json` and `input_contract_draft.json`.
