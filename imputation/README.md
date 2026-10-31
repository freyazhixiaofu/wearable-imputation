# Gap-generation benchmark: masks, imputers, LOPO harness

This module tests whether a synthetic-gap benchmark predicts which heart-rate imputer works best on real quality-rejection gaps (RQ1/RQ2). It consumes the shared 8 s / 2 s window grid. Wrist HR, acceptance decisions and motion come from `member2_ppg/results/batch_v1/<pid>/windows.csv`. The reference HR comes from `member1_ecg/results/ecg_ref_v1/<pid>/ecg_windows.csv` when it exists.

## Run

```powershell
python -m pip install -r imputation/requirements.txt
# 1. ECG reference (needs raw WildPPG_Part_<pid>.mat files)
python member1_ecg/process_ecg.py --data-dir WildPPG/WildPPG/data --all
# 2. LOPO harness (all 16 folds, 10 imputers, 3 mask replicates; ~1-2 h on CPU)
python imputation/harness.py --out imputation/results/ecg_v1
#    quick check without the GRU models (~2 min)
python imputation/harness.py --out imputation/results/ecg_v1_fast --fast
# 3. Tables and figures
python imputation/analyze.py imputation/results/ecg_v1
# Tests
python -m unittest discover -s imputation -p "test_*.py"
python -m unittest discover -s member1_ecg -p "test_*.py"
```

If no ECG results exist, the harness runs in **provisional** mode. Synthetic gaps are then scored against accepted wrist-PPG HR, and the real-gap rows are skipped. `results/provisional_ppg_ref` is such a run. It validates the pipeline, but it cannot answer the research questions.

## Design (what is fixed, and why)

**Inputs and targets.** The imputer input is `hr_obs`, which is wrist PPG HR on accepted windows and NaN elsewhere. Motion features are the only side information. ECG is used only as the target.

**Real gaps.** A real gap is a run of PPG-rejected windows that is bounded, meaning:
- it has an accepted window on both sides,
- it does not touch a record edge,
- it does not overlap a documented exclusion.

Real-gap MAE is computed on windows that also have a valid ECG HR.

**Synthetic gaps** (`masks.py`). Synthetic gaps hide accepted, reference-valid windows only. Each gap keeps at least one observed window on each side and never touches another synthetic gap. Four protocols are compared:

| protocol | length | position |
|---|---|---|
| `random_point` | 1 window | uniform |
| `random_block` | Uniform{1..60} windows (2–120 s), dataset-agnostic | uniform |
| `replay` | resampled from development participants' real-gap lengths | uniform |
| `context_matched` | (length, movement bin) resampled jointly from dev real gaps | mean block motion must fall in that bin; the nearest feasible bin is used as a recorded fallback |

All protocols target 10% of the test participant's accepted windows. `match_budgets` then trims each protocol, removing whole gaps, to the smallest total achieved, so the amount of missing data is matched across protocols.

**Movement bins.** These are quintiles of log10(`acc_dynamic_rms`) over the development participants' windows (still, light, moderate, active, vigorous).

**Candidates** (`imputers.py`). The 10 imputers are chosen so that their ranking can plausibly change with the gap regime:

| name | idea |
|---|---|
| `locf`, `linear`, `pchip` | short-gap interpolators |
| `local_median` | sanity floor: ±10-min rolling median of observed HR |
| `ar1_kalman` | AR(1) deviation from the local level with an RTS smoother; long gaps relax to the local level. φ, q, r come from dev lag-1/lag-2 moments |
| `acc_only` | local level + gradient-boosted function of current and recent motion; ignores the gap endpoints |
| `gbm_ctx`, `gbm_ctx_acc` | gradient-boosted correction to linear interpolation from gap geometry (distance to each edge, gap length, edge values), without and with motion |
| `gru`, `gru_acc` | 2-layer bidirectional GRU (hidden 32) over the window sequence, without and with motion |

**Neutral training mask.** Every learned model is trained with the same mask: on development participants, 15% of accepted windows are hidden, half as single points and half as Uniform{1..60} blocks. The loss is computed on every missing window that has a valid reference, which includes real gaps once ECG exists. No candidate is trained on any specific evaluation protocol.

**LOPO** (`harness.py`). Within each fold, all of the following are fit on the other 15 participants only:
- the bin edges,
- the replayed gap lengths,
- the context-matching pairs,
- the AR(1) parameters,
- the model weights.

**Endpoints** (`analyze.py`), computed per participant with a participant-bootstrap 95% CI:
- Spearman ρ between a protocol's synthetic MAE and the real-gap MAE, across imputers.
- Calibration: mean |synthetic − real| and mean signed difference (a negative value means the benchmark is optimistic).
- Selection regret: the real-gap MAE of the protocol's top pick minus the best real-gap MAE.

The analysis also reports real-gap MAE broken down by gap duration and by movement bin.

## Outputs per run

`results.csv`, `mask_stats.csv` (manipulation check), `real_breakdown.csv`, `feasibility.csv`, `summary.md`, `protocol_imputer_mae.csv`, `figures/`.
