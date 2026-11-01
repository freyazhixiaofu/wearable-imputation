# Run summary: `ecg_v1_synthtrain`

- Reference: **ecg**; participants: 16; reps per protocol: 3; target hidden fraction: 0.1

## Mean MAE (bpm) by protocol and imputer

| imputer      |   random_point |   random_block |   replay |   context_matched |   real |
|:-------------|---------------:|---------------:|---------:|------------------:|-------:|
| gru_acc      |           3.08 |           2.24 |     2.2  |              2.62 |  12.76 |
| gbm_ctx_acc  |           2.9  |           2.32 |     2.24 |              2.82 |  14.3  |
| acc_only     |           5.01 |           2.57 |     2.85 |              3.15 |  13.31 |
| gru          |           4.16 |           3.73 |     3.62 |              4.53 |  14.44 |
| gbm_ctx      |           4.07 |           3.97 |     4.08 |              5.01 |  16.68 |
| ar1_kalman   |           4.76 |           3.74 |     3.64 |              4.68 |  17.15 |
| local_median |           6.5  |           4.06 |     4.33 |              5.17 |  16.35 |
| linear       |           4.65 |           3.89 |     3.74 |              4.82 |  20.22 |
| pchip        |           4.68 |           3.97 |     3.84 |              4.89 |  20.4  |
| locf         |           5.4  |           4.39 |     4.4  |              5.32 |  21.05 |

Kendall tau of each column's imputer ranking against `real`: random_point=0.33, random_block=0.64, replay=0.64, context_matched=0.64

## Masks: manipulation check

| protocol        |   hidden_log_motion |   median_gap_s |   p90_gap_s |   n_hidden |   bin_fallback_rate |
|:----------------|--------------------:|---------------:|------------:|-----------:|--------------------:|
| random_point    |              -1.478 |            2   |         2   |        413 |             nan     |
| random_block    |              -1.561 |           41   |        79.9 |        413 |             nan     |
| replay          |              -1.562 |           12   |        45.4 |        413 |             nan     |
| context_matched |              -1.394 |           14   |        46.8 |        387 |               0.324 |
| real            |              -0.965 |           19.5 |        91   |      20005 |             nan     |

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

| protocol        | spearman          | calib_abs            | calib_signed            | regret            |
|:----------------|:------------------|:---------------------|:------------------------|:------------------|
| random_point    | 0.28 [0.10, 0.44] | 12.15 [10.10, 14.27] | -12.15 [-14.27, -10.10] | 1.48 [0.76, 2.27] |
| random_block    | 0.52 [0.39, 0.63] | 13.18 [10.24, 16.47] | -13.18 [-16.47, -10.24] | 1.57 [0.57, 2.88] |
| replay          | 0.45 [0.32, 0.57] | 13.17 [10.38, 16.30] | -13.17 [-16.30, -10.38] | 2.53 [1.08, 4.11] |
| context_matched | 0.57 [0.37, 0.75] | 12.36 [9.61, 15.52]  | -12.36 [-15.52, -9.61]  | 0.89 [0.23, 1.66] |

Pooled selection: best on real gaps = `gru_acc`. random_point picks `gbm_ctx_acc` (regret 1.54 bpm); random_block picks `gru_acc` (regret 0.00 bpm); replay picks `gru_acc` (regret 0.00 bpm); context_matched picks `gru_acc` (regret 0.00 bpm)

## Real-gap MAE by gap duration

| imputer      |   10-30s |   30-60s |   60-120s |   <=10s |   >120s |
|:-------------|---------:|---------:|----------:|--------:|--------:|
| acc_only     |     9.98 |    12.93 |     15.29 |    7.56 |   17.47 |
| ar1_kalman   |    13.66 |    16.79 |     19.47 |   10.53 |   21.95 |
| gbm_ctx      |    12.07 |    16.43 |     19.75 |    8.1  |   22.04 |
| gbm_ctx_acc  |    10.23 |    13.89 |     16.7  |    7.21 |   18.93 |
| gru          |     9.86 |    13.52 |     17.1  |    6.81 |   20.63 |
| gru_acc      |     8.78 |    12.19 |     15.12 |    6.02 |   17.72 |
| linear       |    15.54 |    20.12 |     23.45 |   11.17 |   25.44 |
| local_median |    12.22 |    15.68 |     18.87 |    9.31 |   21.73 |
| locf         |    16.23 |    20.92 |     24.23 |   11.84 |   26.34 |
| pchip        |    15.66 |    20.3  |     23.62 |   11.24 |   25.75 |

## Real-gap MAE by movement bin

| imputer      |   active |   light |   moderate |   still |   vigorous |
|:-------------|---------:|--------:|-----------:|--------:|-----------:|
| acc_only     |    15.94 |   11.69 |      13.09 |   12.55 |      16.04 |
| ar1_kalman   |    22.24 |   12.63 |      15    |   12.78 |      28.77 |
| gbm_ctx      |    21.35 |   12.7  |      15.67 |   12.3  |      25.14 |
| gbm_ctx_acc  |    17.23 |   12.3  |      14.41 |   12.56 |      16.54 |
| gru          |    19.32 |   10.45 |      12.99 |   10.17 |      24.28 |
| gru_acc      |    15.6  |   10.93 |      12.74 |   10.83 |      16.41 |
| linear       |    25.46 |   15.04 |      18.51 |   14.47 |      32.39 |
| local_median |    21.39 |   11.95 |      14.21 |   12.49 |      27.59 |
| locf         |    26.34 |   15.63 |      19.18 |   15.64 |      33.41 |
| pchip        |    25.65 |   15.2  |      18.74 |   14.73 |      32.49 |
