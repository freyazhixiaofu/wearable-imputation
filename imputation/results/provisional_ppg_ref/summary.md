# Run summary: `provisional_ppg_ref`

- Reference: **ppg_provisional**; participants: 16; reps per protocol: 3; target hidden fraction: 0.1

> Provisional run: no ECG reference yet, so synthetic gaps are scored against accepted wrist-PPG HR and real-gap evaluation is not possible. Use these numbers to check the pipeline, not to answer RQ1/RQ2.

## Mean MAE (bpm) by protocol and imputer

| imputer      |   random_point |   random_block |   replay |   context_matched |
|:-------------|---------------:|---------------:|---------:|------------------:|
| gbm_ctx      |           1.11 |           2.24 |     2.13 |              2.23 |
| gbm_ctx_acc  |           1.15 |           2.26 |     2.2  |              2.25 |
| linear       |           1.1  |           2.37 |     2.23 |              2.35 |
| gru          |           1.41 |           2.3  |     2.11 |              2.26 |
| pchip        |           1.07 |           2.43 |     2.29 |              2.41 |
| ar1_kalman   |           1.43 |           2.38 |     2.18 |              2.3  |
| gru_acc      |           1.62 |           2.43 |     2.25 |              2.4  |
| locf         |           1.6  |           2.83 |     2.71 |              2.89 |
| local_median |           4.08 |           2.66 |     2.68 |              2.95 |
| acc_only     |           4.24 |           2.79 |     2.79 |              3.12 |

Kendall tau of each column's imputer ranking against `random_point`: random_block=0.47, replay=0.33, context_matched=0.51

## Masks: manipulation check

| protocol        |   hidden_log_motion |   median_gap_s |   p90_gap_s |   n_hidden |   bin_fallback_rate |
|:----------------|--------------------:|---------------:|------------:|-----------:|--------------------:|
| random_point    |              -1.442 |            2   |         2   |      443.5 |             nan     |
| random_block    |              -1.515 |           49.5 |        94.5 |      443.5 |             nan     |
| replay          |              -1.521 |           14   |        53.9 |      443.5 |             nan     |
| context_matched |              -1.352 |           14   |        55   |      422   |               0.272 |
| real            |              -0.965 |           19.5 |        91   |    20005   |             nan     |

## Matching feasibility (pooled)

| movement_bin   |   n_real_gaps |   real_gap_time_min |   median_real_gap_s |   accepted_time_min |   supply_demand_ratio |   placements_for_typical_gap |   placements_per_real_gap |
|:---------------|--------------:|--------------------:|--------------------:|--------------------:|----------------------:|-----------------------------:|--------------------------:|
| still          |          2845 |              736.63 |                   8 |             1239.9  |                  1.68 |                         5230 |                      1.84 |
| light          |          3969 |             1564.47 |                  12 |              794.2  |                  0.51 |                         2274 |                      0.57 |
| moderate       |          5277 |             3514.43 |                  22 |              356.63 |                  0.1  |                          286 |                      0.05 |
| active         |          4391 |             2990.03 |                  22 |              251.7  |                  0.08 |                           57 |                      0.01 |
| vigorous       |          2372 |             1200.7  |                  16 |              243.53 |                  0.2  |                          154 |                      0.06 |
