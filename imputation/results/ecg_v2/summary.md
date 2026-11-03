# Run summary: `ecg_v2`

- Reference: **ecg**; participants: 16; reps per protocol: 3; target hidden fraction: 0.1

## Mean MAE (bpm) by protocol and imputer

| imputer      |   random_point |   random_block |   replay |   context_matched |   real |
|:-------------|---------------:|---------------:|---------:|------------------:|-------:|
| gbm_ctx_acc  |           1.43 |           2.39 |     2.09 |              2.19 |  11.54 |
| gru_acc      |           2.13 |           2.66 |     2.54 |              2.51 |  10.31 |
| acc_only     |           2.97 |           2.29 |     2.32 |              2.42 |  10.22 |
| ar1_kalman   |           2.5  |           3.1  |     2.81 |              2.97 |  11.33 |
| gru          |           2.57 |           3.12 |     2.84 |              2.96 |  11.23 |
| gbm_ctx      |           2.48 |           3.33 |     3.04 |              3.15 |  12.73 |
| linear       |           2.4  |           3.27 |     2.86 |              3.07 |  13.32 |
| local_median |           4.08 |           3.4  |     3.26 |              3.37 |  11.42 |
| pchip        |           2.43 |           3.34 |     2.93 |              3.12 |  13.72 |
| locf         |           3.13 |           3.86 |     3.4  |              3.72 |  14.7  |

Kendall tau of each column's imputer ranking against `real`: random_point=-0.07, random_block=0.60, replay=0.51, context_matched=0.56

## Masks: manipulation check

| protocol        |   hidden_log_motion |   median_gap_s |   p90_gap_s |   n_hidden |   bin_fallback_rate |
|:----------------|--------------------:|---------------:|------------:|-----------:|--------------------:|
| random_point    |              -1.572 |            2   |         2   |      201   |             nan     |
| random_block    |              -1.604 |           44.5 |        84.7 |      201   |             nan     |
| replay          |              -1.609 |           12   |        49   |      201   |             nan     |
| context_matched |              -1.514 |           11   |        45   |      187   |               0.308 |
| real            |              -0.984 |           21   |       252.1 |    22112.5 |             nan     |

## Matching feasibility (pooled)

| movement_bin   |   n_real_gaps |   real_gap_time_min |   median_real_gap_s |   accepted_time_min |   supply_demand_ratio |   placements_for_typical_gap |   placements_per_real_gap |
|:---------------|--------------:|--------------------:|--------------------:|--------------------:|----------------------:|-----------------------------:|--------------------------:|
| still          |          2103 |              587.83 |                   8 |              945.73 |                  1.61 |                         3813 |                      1.81 |
| light          |          2407 |             1651.3  |                  12 |              490.97 |                  0.3  |                         1392 |                      0.58 |
| moderate       |          2533 |             4685.4  |                  28 |              141.57 |                  0.03 |                           85 |                      0.03 |
| active         |          1795 |             3430.8  |                  30 |               68.9  |                  0.02 |                            6 |                      0    |
| vigorous       |           742 |              599.9  |                  14 |               56.77 |                  0.09 |                           49 |                      0.07 |

## Does the synthetic benchmark predict real-gap performance?

Per-participant mean with participant-bootstrap 95% CI.

| protocol        | spearman            | calib_abs          | calib_signed          | regret            |
|:----------------|:--------------------|:-------------------|:----------------------|:------------------|
| random_point    | -0.07 [-0.21, 0.08] | 9.44 [7.60, 11.66] | -9.44 [-11.66, -7.60] | 2.09 [1.32, 3.18] |
| random_block    | 0.29 [0.06, 0.47]   | 8.98 [7.03, 11.33] | -8.98 [-11.33, -7.00] | 1.81 [0.82, 3.26] |
| replay          | 0.25 [0.03, 0.43]   | 9.24 [7.33, 11.53] | -9.24 [-11.53, -7.33] | 1.32 [0.72, 2.03] |
| context_matched | 0.36 [0.17, 0.52]   | 9.12 [7.06, 11.58] | -9.10 [-11.57, -7.03] | 1.51 [0.70, 2.70] |

### Only participants held out from PPG-rule tuning

8 participants: fex, l38, ngh, p5d, p9p, ssx, trh, tz8

| protocol        | spearman            | calib_abs          | calib_signed          | regret            |
|:----------------|:--------------------|:-------------------|:----------------------|:------------------|
| random_point    | -0.01 [-0.22, 0.22] | 8.98 [7.10, 10.65] | -8.98 [-10.65, -7.10] | 1.66 [0.94, 2.31] |
| random_block    | 0.27 [0.01, 0.51]   | 8.66 [6.74, 10.51] | -8.66 [-10.51, -6.74] | 1.22 [0.53, 2.04] |
| replay          | 0.24 [-0.05, 0.50]  | 8.88 [6.81, 10.88] | -8.88 [-10.88, -6.81] | 1.24 [0.43, 2.15] |
| context_matched | 0.47 [0.18, 0.68]   | 9.08 [6.74, 11.46] | -9.08 [-11.46, -6.74] | 1.10 [0.41, 1.79] |

Pooled selection: best on real gaps = `acc_only`. random_point picks `gbm_ctx_acc` (regret 1.33 bpm); random_block picks `acc_only` (regret 0.00 bpm); replay picks `gbm_ctx_acc` (regret 1.33 bpm); context_matched picks `gbm_ctx_acc` (regret 1.33 bpm)

## Real-gap MAE by gap duration

| imputer      |   10-30s |   30-60s |   60-120s |   <=10s |   >120s |
|:-------------|---------:|---------:|----------:|--------:|--------:|
| acc_only     |     4.93 |     6.1  |      7.42 |    4.13 |   12.21 |
| ar1_kalman   |     4.92 |     6.63 |      8.03 |    3.89 |   13.6  |
| gbm_ctx      |     4.96 |     7.08 |      9.31 |    3.97 |   15.52 |
| gbm_ctx_acc  |     4.53 |     6.4  |      8.04 |    3.5  |   14.17 |
| gru          |     4.47 |     6.1  |      7.55 |    3.51 |   13.75 |
| gru_acc      |     4.15 |     5.65 |      7.21 |    3.04 |   12.62 |
| linear       |     5.05 |     7.27 |      9.25 |    4    |   16.39 |
| local_median |     5.53 |     6.73 |      8.07 |    4.62 |   13.57 |
| locf         |     5.81 |     7.91 |     10.24 |    4.79 |   17.94 |
| pchip        |     5.13 |     7.39 |      9.5  |    4.03 |   16.94 |

## Real-gap MAE by movement bin

| imputer      |   active |   light |   moderate |   still |   vigorous |
|:-------------|---------:|--------:|-----------:|--------:|-----------:|
| acc_only     |    13.8  |    6.85 |       9.58 |    4.91 |      17.33 |
| ar1_kalman   |    16.16 |    6.81 |      10.39 |    4.45 |      18.78 |
| gbm_ctx      |    17.05 |    9.02 |      12.17 |    5.93 |      20.57 |
| gbm_ctx_acc  |    15.01 |    8.63 |      11.57 |    5.69 |      15.08 |
| gru          |    16.08 |    6.9  |      10.48 |    4.3  |      17.87 |
| gru_acc      |    14.36 |    6.68 |       9.57 |    4.3  |      18.37 |
| linear       |    17.92 |    9.22 |      12.86 |    6.02 |      21.48 |
| local_median |    16.21 |    6.89 |      10.43 |    4.74 |      18.69 |
| locf         |    19.26 |    9.58 |      14.45 |    7.83 |      23.45 |
| pchip        |    18.43 |    9.48 |      13.37 |    6.58 |      21.31 |
