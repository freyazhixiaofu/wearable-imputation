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

The tests use NeuroKit2-simulated ECG at 60 and 110 bpm, a 30 s NaN dropout, and a 60 s noise burst. All four pass. **The thresholds have not yet been checked on real WildPPG ECG.** Before trusting them, plot a few windows from fex, kjd and p5d, the participants whose notes flag noisy ECG.

## Outputs

`results/ecg_ref_v1/<pid>/ecg_windows.csv` and `summary.json`, plus `coverage_audit.csv` across participants. The audit includes the ECG-valid fraction inside PPG-rejected windows (the proposal's coverage audit) and the wrist-PPG vs ECG MAE on accepted windows (a sanity check of the HR extraction).
