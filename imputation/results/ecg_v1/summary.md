# Run summary: `ecg_v1`

- Reference: **ecg**; participants: 16; reps per protocol: 3; target hidden fraction: 0.1

## Mean MAE (bpm) by protocol and imputer

| imputer      |   random_point |   random_block |   replay |   context_matched |   real |
|:-------------|---------------:|---------------:|---------:|------------------:|-------:|
| gru_acc      |           4.24 |           3.56 |     3.59 |              3.8  |   8.55 |
| gbm_ctx_acc  |           3.06 |           3.61 |     3.48 |              4.43 |   9.51 |
| gru          |           5.09 |           5.09 |     4.88 |              5.31 |  10.12 |
| acc_only     |           6.46 |           3.95 |     4.14 |              4.55 |  11.63 |
| ar1_kalman   |           4.76 |           3.68 |     3.6  |              4.44 |  17.15 |
| local_median |           6.49 |           4.02 |     4.22 |              4.91 |  16.35 |
| linear       |           4.65 |           3.8  |     3.71 |              4.6  |  20.22 |
| pchip        |           4.69 |           3.88 |     3.79 |              4.68 |  20.4  |
| gbm_ctx      |           4.55 |           8.66 |     7.45 |              8.36 |  10.94 |
| locf         |           5.41 |           4.49 |     4.27 |              5.18 |  21.05 |

Kendall tau of each column's imputer ranking against `real`: random_point=0.33, random_block=0.20, replay=0.16, context_matched=0.29

## Masks: manipulation check

| protocol        |   hidden_log_motion |   median_gap_s |   p90_gap_s |   n_hidden |   bin_fallback_rate |
|:----------------|--------------------:|---------------:|------------:|-----------:|--------------------:|
| random_point    |              -1.481 |            2   |         2   |      413   |             nan     |
| random_block    |              -1.561 |           44   |        72.7 |      413   |             nan     |
| replay          |              -1.553 |           13.5 |        47.4 |      413   |             nan     |
| context_matched |              -1.392 |           12.5 |        49.6 |      389.5 |               0.313 |
| real            |              -0.965 |           19.5 |        91   |    20005   |             nan     |

## Matching feasibility (pooled)

| movement_bin   |   n_real_gaps |   real_gap_time_min |   median_real_gap_s |   accepted_time_min |   supply_demand_ratio |   placements_for_typical_gap |   placements_per_real_gap |
|:---------------|--------------:|--------------------:|--------------------:|--------------------:|----------------------:|-----------------------------:|--------------------------:|
| still          |          2845 |              736.63 |                   8 |             1209.4  |                  1.64 |                         5016 |                      1.76 |
| light          |          3969 |             1564.47 |                  12 |              754.53 |                  0.48 |                         2060 |                      0.52 |
| moderate       |          5277 |             3514.43 |                  22 |              320.23 |                  0.09 |                          233 |                      0.04 |
| active         |          4391 |             2990.03 |                  22 |              203.43 |                  0.07 |                           36 |                      0.01 |
| vigorous       |          2372 |             1200.7  |                  16 |              174.43 |                  0.15 |                           78 |                      0.03 |

## Does the synthetic benchmark predict real-gap performance?

Per-participant mean with participant-bootstrap 95% CI.

| protocol        | spearman            | calib_abs           | calib_signed           | regret             |
|:----------------|:--------------------|:--------------------|:-----------------------|:-------------------|
| random_point    | 0.09 [-0.15, 0.33]  | 9.65 [8.16, 11.29]  | -9.65 [-11.29, -8.16]  | 2.31 [0.70, 4.29]  |
| random_block    | -0.18 [-0.43, 0.08] | 10.35 [8.08, 12.98] | -10.12 [-12.85, -7.78] | 7.04 [4.45, 9.58]  |
| replay          | -0.21 [-0.45, 0.05] | 10.39 [8.18, 12.92] | -10.28 [-12.86, -8.03] | 7.40 [4.73, 10.02] |
| context_matched | 0.09 [-0.15, 0.31]  | 9.74 [7.61, 12.20]  | -9.56 [-12.12, -7.40]  | 3.75 [1.76, 5.92]  |

Pooled selection: best on real gaps = `gru_acc`. random_point picks `gbm_ctx_acc` (regret 0.97 bpm); random_block picks `gru_acc` (regret 0.00 bpm); replay picks `gbm_ctx_acc` (regret 0.97 bpm); context_matched picks `gru_acc` (regret 0.00 bpm)

## Real-gap MAE by gap duration

| imputer      |   10-30s |   30-60s |   60-120s |   <=10s |   >120s |
|:-------------|---------:|---------:|----------:|--------:|--------:|
| acc_only     |     9.16 |    11.16 |     12.71 |    7.66 |   14.65 |
| ar1_kalman   |    13.66 |    16.79 |     19.47 |   10.53 |   21.95 |
| gbm_ctx      |     8.5  |    10.92 |     12.26 |    6.43 |   13.44 |
| gbm_ctx_acc  |     7.28 |     9.38 |     10.65 |    5.41 |   11.9  |
| gru          |     7.73 |     9.33 |     11.12 |    6.38 |   12.64 |
| gru_acc      |     6.84 |     8.25 |      9.57 |    5.51 |   10.76 |
| linear       |    15.54 |    20.12 |     23.45 |   11.17 |   25.44 |
| local_median |    12.22 |    15.68 |     18.87 |    9.31 |   21.73 |
| locf         |    16.23 |    20.92 |     24.23 |   11.84 |   26.34 |
| pchip        |    15.66 |    20.3  |     23.62 |   11.24 |   25.75 |

## Real-gap MAE by movement bin

| imputer      |   active |   light |   moderate |   still |   vigorous |
|:-------------|---------:|--------:|-----------:|--------:|-----------:|
| acc_only     |    13.46 |   10.32 |      11.1  |   11.36 |      13.86 |
| ar1_kalman   |    22.24 |   12.63 |      15    |   12.78 |      28.77 |
| gbm_ctx      |    12.69 |    8.75 |      10.59 |    7.67 |      16.21 |
| gbm_ctx_acc  |    11.34 |    7.93 |       9.36 |    7.66 |      11.45 |
| gru          |    12.13 |    8.25 |       8.96 |    6.67 |      15.36 |
| gru_acc      |    10.09 |    7.73 |       7.97 |    7.85 |      11.23 |
| linear       |    25.46 |   15.04 |      18.51 |   14.47 |      32.39 |
| local_median |    21.39 |   11.95 |      14.21 |   12.49 |      27.59 |
| locf         |    26.34 |   15.63 |      19.18 |   15.64 |      33.41 |
| pchip        |    25.65 |   15.2  |      18.74 |   14.73 |      32.49 |
