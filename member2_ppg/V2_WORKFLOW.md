# Final v2

The final v2 is the selected PPG acceptance rule. V1 is unchanged.

- Rule: `final_v2.py`
- Settings: [results/batch_v2/config.json](results/batch_v2/config.json)
- Batch manifest: [results/batch_v2/batch_manifest.json](results/batch_v2/batch_manifest.json)
- Files to commit: [V2_FILES.txt](V2_FILES.txt)

## Results layout

```text
results/batch_v2/
  config.json
  batch_manifest.json
  participant_summary.csv
  overall_summary.json
  <participant>/
    windows.csv
    gaps.csv
```

All 16 participants use this layout, matching `results/batch_v1`. Earlier
experiments are retained separately under `results/experiments`.
The shared files include each participant's `windows.csv` and `gaps.csv`, the
participant summary, three overview figures and the validation records listed
in the batch manifest. `V2_FILES.txt` contains the complete commit list.

## Acceptance

Use green PPG with 8-second windows and a 2-second stride. Every accepted
window must have complete finite data, safe filtering, no documented exclusion,
no flat run of 0.5 seconds or longer, and measured spectral HR strictly between
42 and 198 bpm. Half-rate correction is off. ECG is used only for evaluation.

A window passes either route:

| Check | Ordinary route | Surrounding-anchor route |
|---|---:|---:|
| Spectral concentration | >=0.10 | >=0.10 |
| ACF strength | >=0.60 | >=0.30 |
| Spectral/ACF HR difference | <=12 bpm | <=40 bpm |

The anchor route requires two originally accepted anchors before and two after
the window. Each anchor needs ACF >=0.80, concentration >=0.10 and HR-method
difference <=6 bpm. Same-side anchors must be at least 8 seconds apart and all
anchors within 120 seconds. Their HR range and each difference from the
window's measured HR must be <=5 bpm. Invalid data cannot interrupt the span.
Recovered windows cannot serve as anchors. Future anchors make this an offline
rule. Accepted HR remains the window's own measured spectral HR.

## Use

```powershell
python member2_ppg/final_v2.py --verify-rule
python -m unittest discover -s member2_ppg -p "test_final_v2.py"
```

```python
from member2_ppg.final_v2 import apply_final_v2
windows = apply_final_v2(features)
```

Provide one participant on the complete ordered 2-second window grid, including
rejected rows. Required columns: `center_s`, `hr_spectral_bpm`, `hr_acf_bpm`,
`raw_spectral_concentration`, `acf_peak`, `spectral_entropy`, `finite_ratio`,
`documented_exclusion`, `filter_safe`, and `longest_flat_run_s`. Use measurements
from the documented PPG preprocessing. `spectral_entropy` is retained for input
compatibility and has no acceptance threshold. `participant_id` is optional.

Settings were selected on eight development participants and fixed before
evaluation on the other eight. Evaluation acceptance was 16.79%, mean absolute
error 4.38 bpm, and errors above 10 bpm 7.83% (participant means among ECG-valid
windows). Some participants regressed, and the cohort had been inspected in
earlier work; these averages do not guarantee accuracy for every participant.

The v1 batch runner retains its original defaults. Final v2 is applied through
the entry point above. `--verify` also checks the saved v2 outputs. Raw data,
caches and earlier experiments stay local. No imputation training is included.
