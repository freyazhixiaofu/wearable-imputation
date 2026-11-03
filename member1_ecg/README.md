# Sternum ECG reference heart rate

This module turns the WildPPG sternum ECG (`sternum.ecg`, 128 Hz) into a reference HR on exactly the same 8 s / 2 s window grid as `member2_ppg`. It follows the README's ECG handoff and keeps the same `window_id`, `start_sample` and `end_sample_exclusive`. ECG is only ever used as a reference. It does not enter PPG acceptance or imputer inputs.

```powershell
python member1_ecg/process_ecg.py --data-dir WildPPG/WildPPG/data --all
python -m unittest discover -s member1_ecg -p "test_*.py"
```

## Method (`ecg_ref_v1`)

1. **R-peaks.** `neurokit2.ecg_clean` and `ecg_peaks` run on each finite ECG segment, in 5-min chunks with 5 s overlap. Duplicate peaks at chunk seams are removed.
2. **RR cleaning.** An RR interval must lie in 0.27–2.0 s (30–220 bpm) and must be within 0.75–1.33× the previous interval (quotient filter).
3. **Window HR.** The window HR is 60 / mean of the clean RR intervals that lie fully inside the window.
4. **`ecg_valid`.** A window is valid only when all of the following hold:
   - all of its ECG samples are finite,
   - it has at least 4 clean RR intervals,
   - at least 70% of its RR intervals are clean,
   - the RR coefficient of variation is at most 0.25,
   - its HR is within 25 bpm of the ±30 s rolling median of the other valid windows.

   Failure reasons are recorded in `ecg_reject_reasons`.

The tests use NeuroKit2-simulated ECG at 60 and 110 bpm, a 30 s NaN dropout, and a 60 s noise burst. All four pass.

## Results on WildPPG (`results/ecg_ref_v1/coverage_audit.csv`, `ppg_vs_ecg_check.csv`)

- **ECG coverage.** 86–100% of windows have a valid ECG HR for 13 of the 16 participants. The exceptions are fex (55%), p5d (68%) and kjd (86%), which matches the dataset notes that flag noisy ECG for these three. Coverage inside PPG-rejected windows is nearly identical to overall coverage, so the full real-gap evaluation is feasible and the short-span fallback isn't needed.
- **The ECG reference is trustworthy where it disagrees with PPG.** In the largest disagreements (p9p, qm9) the raw ECG is clean, with exactly one detected R-peak per beat.
- **The PPG rule `ppg_only_dev_v1` accepts many wrong HRs.** Pooled over participants, 8.8% of accepted windows read about half the ECG HR. The spectral and autocorrelation estimates both lock onto the sub-harmonic, so their agreement check passes. Wrist-vs-ECG MAE on accepted windows ranges from 1.5 bpm (fex) to 31 bpm (p9p); for 6 of 16 participants more than 25% of accepted windows are over 10 bpm off.
- **It also rejects many correct ones.** In 20–84% of PPG-rejected windows with valid ECG, the candidate wrist HR was already within 10 bpm of ECG.

The next step is to recalibrate the PPG thresholds on development participants against `hr_ecg`. One option is a harmonic check that compares spectral power at 2f against f. Then re-run the harness, because the rule defines the real gaps.

To extract the ECG from the full 1.2 GB `.mat` files into small `<pid>.npz` files, run `extract_ecg.py`. `process_ecg.py --data-dir` accepts either format.

## Outputs

`results/ecg_ref_v1/<pid>/ecg_windows.csv` and `summary.json`, plus `coverage_audit.csv` across participants. The audit includes the ECG-valid fraction inside PPG-rejected windows (the proposal's coverage audit) and the wrist-PPG vs ECG MAE on accepted windows (a sanity check of the HR extraction).

## PPG v1 vs v2 against ECG (`ppg_vs_ecg.py`)

`python member1_ecg/ppg_vs_ecg.py --ppg-batch batch_v2` writes `results/ecg_ref_v1/ppg_vs_ecg_<batch>.csv`. Means over all 16 participants, on ECG-valid windows:

| Rule | Accepted | MAE on accepted (bpm) | Accepted off by >10 bpm | Accepted at about half HR |
|---|---|---|---|---|
| `ppg_only_dev_v1` | 23.6% | 8.9 | 21.0% | 12.4% |
| `ppg_only_dev_v2` | 15.3% | 3.7 | 6.7% | 5.1% |

v2 was tuned on the ECG of an0, e61, k2s, kjd, n31, qm9, u7y and w4p (see `member2_ppg/results/batch_v2/development/selected.json`), so its accuracy on those eight is not an independent estimate. p9p remains poor under v2 (18.8 bpm, 32% at half HR).
