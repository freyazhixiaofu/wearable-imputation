"""Leave-one-participant-out (LOPO) harness.

For each held-out participant, everything learned or tuned (imputer weights, AR(1) parameters,
movement-bin edges, replayed gap lengths, context-matching pairs) is fit on the other participants.
The held-out participant is then scored (a) on every synthetic masking protocol and (b) on its
real quality-rejection gaps when an ECG reference exists.

    python imputation/harness.py --out imputation/results/run_name [--fast] [--reps 3] [--folds an0,e61]

Outputs (one row per test participant x protocol x replicate x imputer):
    results.csv        protocol in {random_point, random_block, replay, context_matched, real}
    mask_stats.csv     achieved missing amount, gap lengths, motion of hidden windows, fallbacks
    real_breakdown.csv real-gap absolute-error sums by gap duration and movement bin
    feasibility.csv    pooled matching-feasibility table (all participants, descriptive)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import PPG_BATCH, PPG_RESULTS, STRIDE_S, ecg_available, load_all
from gaps import BIN_NAMES, assign_bin, feasibility_table, motion_bin_edges, real_gaps
from imputers import candidates
from masks import PROTOCOLS, make_all_masks, neutral_training_mask

DUR_BINS = [0, 10, 30, 60, 120, np.inf]  # seconds
DUR_NAMES = ['<=10s', '10-30s', '30-60s', '60-120s', '>120s']


def real_gap_eval_windows(df, edges):
    """Windows inside bounded real gaps with a valid reference, annotated with duration and motion bin."""
    g = real_gaps(df)
    g = g[g.bounded]
    gap_len = np.full(len(df), np.nan)
    gap_motion = np.full(len(df), np.nan)
    for r in g.itertuples(index=False):
        gap_len[r.start:r.stop] = r.duration_s
        gap_motion[r.start:r.stop] = r.log_motion
    ev = np.isfinite(gap_len) & df.ref_valid.values & (df.ref_source.iloc[0] == 'ecg')
    dur_bin = np.digitize(gap_len, DUR_BINS[1:-1], right=True)
    return ev, dur_bin, assign_bin(gap_motion, edges)


def run_fold(test_id, frames, cfg, rng):
    t0 = time.time()
    dev = [f for p, f in frames.items() if p != test_id]
    test = frames[test_id]
    edges = motion_bin_edges(dev)
    dev_gaps = pd.concat([real_gaps(f) for f in dev], ignore_index=True)
    dev_gaps = dev_gaps[dev_gaps.bounded].reset_index(drop=True)
    dev_gaps['bin'] = assign_bin(dev_gaps.log_motion, edges)
    dev_gaps = dev_gaps[dev_gaps.bin >= 0].reset_index(drop=True)

    train_items = []
    for f in dev:
        obs = f.hr_obs.values.copy()
        hole = neutral_training_mask(f, rng)
        obs[hole] = np.nan
        if cfg.get('train_target', 'synthetic') == 'synthetic':
            # loss only on the synthetic training holes; dev real gaps stay unlabeled so that
            # no candidate is trained on the real-gap regime it is later evaluated on
            f = f.assign(ref_valid=f.ref_valid.values & hole)
        train_items.append((f, obs))
    models = [copy.deepcopy(m).fit(train_items, rng) for m in candidates(cfg['fast'])]

    rows, stats, breakdown = [], [], []
    ref = test.hr_ref.values
    for rep in range(cfg['reps']):
        masks, target = make_all_masks(test, cfg['frac'], rng, dev_gaps, edges)
        for proto, (hidden, g) in masks.items():
            ev = hidden & test.ref_valid.values
            obs = test.hr_obs.values.copy()
            obs[hidden] = np.nan
            hm = test.log_motion.values[hidden]
            stats.append(dict(test_participant=test_id, protocol=proto, rep=rep, n_hidden=int(hidden.sum()),
                              budget_after_matching=target, n_gaps=len(g),
                              median_gap_s=float(g.n_windows.median() * STRIDE_S) if len(g) else np.nan,
                              p90_gap_s=float(g.n_windows.quantile(.9) * STRIDE_S) if len(g) else np.nan,
                              hidden_log_motion_mean=float(np.nanmean(hm)) if np.isfinite(hm).any() else np.nan,
                              bin_fallback_rate=float((g.bin_fallback > 0).mean()) if 'bin_fallback' in g else np.nan))
            for m in models:
                err = np.abs(m.impute(test, obs) - ref)[ev]
                rows.append(dict(test_participant=test_id, protocol=proto, rep=rep, imputer=m.name,
                                 n_eval=int(ev.sum()), mae=float(err.mean()) if len(err) else np.nan))

    ev, dbin, mbin = real_gap_eval_windows(test, edges)
    rg = real_gaps(test)
    rg = rg[rg.bounded]
    stats.append(dict(test_participant=test_id, protocol='real', rep=0,
                      n_hidden=int((~test.ppg_valid.values).sum()), n_gaps=len(rg),
                      median_gap_s=float(rg.duration_s.median()), p90_gap_s=float(rg.duration_s.quantile(.9)),
                      hidden_log_motion_mean=float(np.nanmean(test.log_motion.values[~test.ppg_valid.values])),
                      n_eval_real=int(ev.sum())))
    if ev.any():
        obs = test.hr_obs.values
        for m in models:
            err = np.abs(m.impute(test, obs) - ref)
            rows.append(dict(test_participant=test_id, protocol='real', rep=0, imputer=m.name,
                             n_eval=int(ev.sum()), mae=float(err[ev].mean())))
            d = pd.DataFrame(dict(dur=dbin[ev], mot=mbin[ev], err=err[ev]))
            for (di, mi), grp in d.groupby(['dur', 'mot']):
                breakdown.append(dict(test_participant=test_id, imputer=m.name, duration_bin=DUR_NAMES[di],
                                      movement_bin=BIN_NAMES[mi] if mi >= 0 else 'acc_invalid',
                                      abs_err_sum=float(grp.err.sum()), n=len(grp)))
    print(f'{test_id}: {len(models)} imputers, {time.time() - t0:.0f}s, real-gap eval windows={int(ev.sum())}', flush=True)
    return rows, stats, breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--fast', action='store_true', help='skip GRU models')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--frac', type=float, default=0.10, help='target hidden fraction of accepted windows')
    ap.add_argument('--folds', default='', help='comma-separated test participants (default: all)')
    ap.add_argument('--seed', type=int, default=2631)
    ap.add_argument('--ppg-batch', default=PPG_BATCH, help='member2_ppg/results/<batch> to use (batch_v2 = final rule)')
    ap.add_argument('--train-target', choices=['synthetic', 'all'], default='synthetic',
                    help="learned imputers' loss on synthetic training holes only (default) or on all "
                         "missing reference-valid dev windows, including dev real gaps")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frames = load_all(ppg_dir=PPG_RESULTS / args.ppg_batch)
    cfg = dict(fast=args.fast, reps=args.reps, ppg_batch=args.ppg_batch, train_target=args.train_target, frac=args.frac, seed=args.seed,
               reference='ecg' if ecg_available(frames) else 'ppg_provisional',
               protocols=PROTOCOLS, participants=list(frames))
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    feas, _ = feasibility_table(frames, motion_bin_edges(list(frames.values())))
    feas.to_csv(out / 'feasibility.csv', index=False)
    folds = args.folds.split(',') if args.folds else list(frames)
    allr, alls, allb = [], [], []
    for i, p in enumerate(folds):
        r, s, b = run_fold(p, frames, cfg, np.random.default_rng([args.seed, i]))
        allr += r; alls += s; allb += b
        # checkpoint after every fold
        pd.DataFrame(allr).to_csv(out / 'results.csv', index=False)
        pd.DataFrame(alls).to_csv(out / 'mask_stats.csv', index=False)
        pd.DataFrame(allb).to_csv(out / 'real_breakdown.csv', index=False)


if __name__ == '__main__':
    main()
