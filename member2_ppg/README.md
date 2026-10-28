# Wrist PPG heart-rate and signal-quality baseline

This module converts wrist PPG from WildPPG into window-level heart-rate (HR) estimates and signal-quality indicators (SQI). Accelerometer (ACC) features describe motion. HR estimation and acceptance decisions use PPG alone; ECG is reserved for later accuracy evaluation.

The current rule set, `ppg_only_dev_v1`, is a development baseline. Its acceptance labels and gap summaries remain provisional until evaluated against reliable ECG reference windows.

## Setup and data

Run the following commands from the repository root:

```powershell
python -m pip install -r member2_ppg/requirements.txt
```

Place the extracted participant files, named `WildPPG_Part_<participant>.mat`, in `WildPPG/WildPPG/data`, or supply a different directory with `--data-dir`. Raw data and ZIP archives are excluded by the repository's `.gitignore` and must be obtained separately.

## Run all participants

```powershell
python member2_ppg/process_all.py --resume
```

The default output directory is `member2_ppg/results/batch_v1`. The batch reviews participant notes before processing and stops on unrecognized notes. Explicit wrist exclusions are masked while preserving sample positions. ECG-noise notes for fex, kjd, and p5d are retained without changing PPG decisions. k2s and l38 have explicit wrist interruptions; an0 has an unresolved ankle/wrist attribution warning.

Resume skips a participant only when its completion marker matches the processor source hash, input path, size, modification time, and configuration, and its output tables pass validation. Interrupted participants are recomputed.

To regenerate aggregate outputs from completed participant results:

```powershell
python member2_ppg/process_all.py --summarize-only
```

## Run one participant

Use a separate output directory for exploratory runs to preserve the saved batch baseline:

```powershell
python member2_ppg/process_wrist.py --data-dir WildPPG/WildPPG/data --participant an0 --out-dir member2_ppg/results/single_an0
```

Changing `--participant` also changes the participant notes that must be reviewed. The reader loads the wrist MATLAB structure and records available channels and sampling rates in the channel audit.

## Signal processing and alignment

- Use green wrist PPG (`ppg_g`) with 8-second windows and a 2-second stride.
- Preserve the original sample grid. Sample indices are zero-based, and window intervals are `[start_sample, end_sample_exclusive)`.
- Align channels relative to sample zero using the dataset synchronization assumption. No empirical time-offset correction is applied.
- Filter finite, non-excluded PPG segments separately with a 0.5-4 Hz Butterworth bandpass and zero-phase `sosfiltfilt`. Filtering does not bridge NaN, Inf, or excluded intervals.
- Apply a 2-second guard at each filtered segment boundary. A window must lie entirely within the safe region.
- Keep missing or invalid measurements as NaN and preserve their rejection reasons.

## HR estimation and quality rules

The primary HR estimate is the maximum of the Hann-window Welch spectrum in the 40-200 bpm search range. It is stored in both `hr_spectral_bpm` and `hr_ppg_candidate`. The FFT uses at least 4096 points; zero padding refines the frequency grid but does not increase the information in an 8-second window.

A second estimate, `hr_acf_bpm`, uses the strongest eligible local peak of the normalized autocorrelation. Autocorrelation is divided by zero-lag energy without overlap-count correction. The difference between the two HR methods is used as a consistency check.

All acceptance checks must pass:

| Check | Default requirement |
|---|---|
| Finite raw PPG samples | 100% |
| Documented exclusion | No overlap |
| Filter safety | Entire window within the safe region |
| Longest constant-value run | At most 0.5 seconds |
| Spectral HR search boundary | At least 2 bpm from either boundary |
| Spectral power within +/-0.15 Hz of the dominant peak | At least 0.40 of the power in the HR search band |
| Normalized autocorrelation peak | At least 0.40 |
| Spectral and autocorrelation HR difference | At most 12 bpm |

These checks do not include a calibrated sensor-rail clipping detector. Thresholds are development settings and have not been calibrated against ECG.

The descriptive score is `sqi_score = clip(min(spectral_concentration, acf_peak), 0, 1)`, with unavailable components represented as NaN. Use `ppg_valid` for the final acceptance decision because the score alone does not encode every check.

Configuration overrides can be supplied with `--config config.json`, for example:

```json
{
  "min_spectral_concentration": 0.45,
  "min_acf_peak": 0.45,
  "max_hr_difference_bpm": 10.0,
  "rule_version": "ppg_only_dev_v2"
}
```

Record rule changes explicitly and use a new output directory for each configuration.

## Participant notes: an0

The notes mention an ankle interruption but explicitly identify wrist MATLAB indices `284544-286401`, inclusive. The default `--notes-policy mask` converts these to Python `[284543:286401]`, corresponding to `[2222.9922, 2237.5078)` seconds. Overlapping PPG windows are rejected without removing samples; ACC features are calculated separately.

The channel attribution remains unresolved and is recorded as `attribution_verified=false` in the manifest. A sensitivity run can use `--notes-policy ignore` with a separate output directory; this option changes only the treatment of documented exclusions.

The saved batch result for an0 contains 22,181 windows: 10,159 accepted (45.80%) and 12,022 rejected. It has 1,314 rejected HR-grid runs, with a median duration of 10 seconds and a maximum of 230 seconds. ACC is valid in 22,166 windows, and 11 windows overlap the documented exclusion. The canonical saved result is `results/batch_v1/an0`; the redundant standalone `results/an0` copy has been removed.

## Per-participant outputs

| File | Contents |
|---|---|
| `windows.csv` | Complete window grid, HR estimates, quality checks, decisions, and motion features |
| `gaps.csv` | Consecutive rejected HR-grid runs |
| `channel_audit.csv` | Wrist-channel sampling rates, lengths, finite-value checks, and ranges |
| `rejection_counts.csv` | Overlapping rejection-reason counts |
| `summary.json` | Coverage and gap statistics |
| `manifest.json` | Input provenance, configuration, notes, exclusions, assumptions, and environment |
| `timeline.png` | HR, SQI, and acceptance timeline |
| `quality_summary.png` | Quality and gap summary plots |
| `qa_windows/*.png` | Sampled raw and filtered PPG with HR, SQI, and motion annotations |
| `qa_windows/selection.csv` | Selected QA windows and categories |
| `processing.log`, `completed.json` | Batch processing log and completion metadata |

Important `windows.csv` fields:

- Keys and timing: `participant_id`, `window_id`, `start_sample`, `end_sample_exclusive`, `start_s`, `end_s`, `center_s`. Window IDs start at zero.
- HR: `hr_ppg_candidate`, `hr_spectral_bpm`, `hr_acf_bpm`, and `hr_ppg`. The final `hr_ppg` is NaN for rejected windows; candidate estimates are retained for inspection where available.
- SQI: `spectral_concentration`, `acf_peak`, `hr_method_difference_bpm`, `spectral_entropy`, `sqi_score`.
- Data checks: `finite_ratio`, `longest_flat_run_s`, `documented_exclusion`, `filter_safe`.
- Decision: `ppg_valid`, `reject_reasons`, `rule_version`. Multiple rejection reasons are separated by semicolons; accepted windows have no rejection reason.
- Motion: `acc_valid`, `acc_finite_ratio`, `acc_magnitude_mean`, `acc_magnitude_std`, `acc_x_std`, `acc_y_std`, `acc_z_std`, `acc_dynamic_rms`, `acc_jerk_rms`.

ACC dynamic RMS uses axis-wise motion after a 0.25 Hz gravity-removal filter: `sqrt(mean(dx^2 + dy^2 + dz^2))`. Jerk RMS uses temporal differences scaled by the sampling rate. Features retain native stored units; no conversion to g or m/s^2 is assumed. Invalid ACC features are NaN rather than zero. Motion features do not determine PPG acceptance.

A run of `n` rejected windows has HR-grid duration `n * stride`; its raw signal support spans `window + (n - 1) * stride`. These are different quantities. `grid_start_center_s` is the first rejected window center, and `grid_end_exclusive_s` is one stride after the last rejected center. A rejected HR-grid run does not necessarily indicate missing raw PPG samples.

QA examples sample accepted/rejected windows across lower/higher within-participant motion levels. These relative motion labels and plots support inspection rather than ECG-based accuracy claims.

## Cross-participant outputs

All paths below are relative to `results/batch_v1`:

| File | Contents |
|---|---|
| `report.html` | Participant tables, summary plots, notes, and links to individual results |
| `participant_summary.csv` | Coverage, recording duration, gap statistics, ACC availability, HR median, exclusions, and notes review |
| `overall_summary.json` | Window-weighted and participant-mean coverage, recording duration, pooled gap statistics, and configuration |
| `all_gaps.csv` | Rejected HR-grid runs for all participants |
| `all_channel_audits.csv` | Combined channel audits |
| `rejection_reasons_by_participant.csv`, `rejection_reasons_overall.csv` | Overlapping rejection counts |
| `motion_summary.csv` | Accepted/rejected motion distributions and acceptance by participant motion quartile |
| `notes_review.csv`, `notes_review.json` | Original notes, reviewed actions, index conversions, and ECG warnings |
| `validation_checks.csv` | Output consistency checks for every participant |
| `batch_manifest.json` | Source hash, configuration, participants, and run times |
| `participant_overview.png`, `gap_reason_overview.png`, `motion_overview.png` | Cross-participant figures |

## Targeted PPG and autocorrelation audit

`audit_windows.py` reconstructs selected baseline windows from raw signals and compares their HR/SQI values with the saved results. It records SHA-256 fingerprints to verify that baseline outputs and processor code remain unchanged.

```powershell
python member2_ppg/audit_windows.py
```

The report is saved to `results/targeted_audit_v1/report.html`. It includes raw and filtered PPG, power spectra, actual autocorrelation curves, motion, sampling availability, and metric-reproduction checks. Selection uses seed 2631 and targeted quality/motion strata from an0, e61, k2s, qm9, and p9p. The selected examples support diagnosis rather than population prevalence or HR accuracy estimates.

## ECG evaluation handoff

Use the same 8-second windows, 2-second stride, and timing convention for ECG outputs, including `hr_ecg` and `ecg_valid`. Evaluate PPG HR error where `ppg_valid & ecg_valid` is true, and report reference availability and PPG coverage separately. Preserve rejected windows and missing HR values when constructing gap and matching-feasibility tables.

## Verification

```powershell
python -m unittest discover -s member2_ppg -p "test_*.py" -v
```

Tests cover signal-processing behavior, missing values and exclusions, gap semantics, ACC behavior, batch notes handling, and audit calculations. Batch validation additionally checks the complete sample grid, unique keys, HR/acceptance/NaN consistency, exact gap runs and durations, exclusion overlaps, configuration, and required plots. These checks verify implementation consistency; ECG comparison is needed to establish HR accuracy.
