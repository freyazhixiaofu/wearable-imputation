"""Real (quality-rejection) gaps, accepted runs, movement bins, and the matching-feasibility table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import STRIDE_S


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def real_gaps(df):
    """Bounded rejection runs: an accepted window on both sides, no record edge, no documented exclusion."""
    s, e = runs(~df.ppg_valid.values)
    n = len(df)
    excl = df.documented_exclusion.values.astype(bool)
    motion = df.log_motion.values
    rows = []
    for a, b in zip(s, e):
        bounded = a > 0 and b < n and not excl[a:b].any()
        m = motion[a:b]
        rows.append(dict(participant_id=df.participant_id.iloc[0], start=a, stop=b, n_windows=b - a,
                         duration_s=(b - a) * STRIDE_S, bounded=bounded,
                         log_motion=np.nanmean(m) if np.isfinite(m).any() else np.nan,
                         ref_valid_fraction=df.ref_valid.values[a:b].mean()))
    return pd.DataFrame(rows)


def accepted_runs(df):
    s, e = runs(df.ppg_valid.values & df.ref_valid.values)
    return pd.DataFrame(dict(start=s, stop=e, n_windows=e - s))


def motion_bin_edges(dev_frames, n_bins=5):
    """Quantile edges of log-motion over all dev windows (never touches the test participant)."""
    m = np.concatenate([f.log_motion.dropna().values for f in dev_frames])
    q = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    return q


BIN_NAMES = ['still', 'light', 'moderate', 'active', 'vigorous']


def assign_bin(values, edges):
    v = np.asarray(values, float)
    b = np.searchsorted(edges, v, side='right') - 1
    return np.where(np.isfinite(v), np.clip(b, 0, len(edges) - 2), -1)


def block_motion(log_motion, L):
    """Mean log-motion of every length-L block starting at each index (NaN-aware)."""
    m = np.asarray(log_motion, float)
    fin = np.isfinite(m)
    c = np.r_[0, np.cumsum(np.where(fin, m, 0))]
    k = np.r_[0, np.cumsum(fin)]
    n = len(m) - L + 1
    if n <= 0:
        return np.array([])
    tot, cnt = c[L:] - c[:-L], k[L:] - k[:-L]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, tot / cnt, np.nan)[:n]


def interior_starts(eligible, L, margin=1):
    """Start indices i where windows [i-margin, i+L+margin) are all eligible."""
    e = np.asarray(eligible, bool)
    need = L + 2 * margin
    if len(e) < need:
        return np.array([], int)
    c = np.r_[0, np.cumsum(e)]
    ok = (c[need:] - c[:-need]) == need
    return np.flatnonzero(ok) + margin


def feasibility_table(frames, edges, length_quantile=0.5):
    """Supply (accepted, ref-valid time) vs demand (real-gap time) per movement bin.

    For each bin, `placements_for_typical_gap` counts non-overlapping interior placements
    whose mean motion falls in the bin, for that bin's median real-gap length.
    """
    gaps = pd.concat([real_gaps(f) for f in frames.values()], ignore_index=True)
    gaps = gaps[gaps.bounded]
    gaps['bin'] = assign_bin(gaps.log_motion, edges)
    rows = []
    for b, name in enumerate(BIN_NAMES[:len(edges) - 1]):
        g = gaps[gaps.bin == b]
        L = int(max(1, np.quantile(g.n_windows, length_quantile))) if len(g) else 1
        acc_s, placements = 0.0, 0
        for f in frames.values():
            elig = f.ppg_valid.values & f.ref_valid.values
            wb = assign_bin(f.log_motion.values, edges)
            acc_s += (elig & (wb == b)).sum() * STRIDE_S
            st = interior_starts(elig, L)
            if len(st):
                bm = assign_bin(block_motion(f.log_motion.values, L)[st], edges)
                st = st[bm == b]
                # greedy non-overlapping count
                last, cnt = -10 ** 9, 0
                for i in st:
                    if i >= last + L + 1:
                        cnt, last = cnt + 1, i
                placements += cnt
        demand = g.duration_s.sum()
        rows.append(dict(movement_bin=name, n_real_gaps=len(g), real_gap_time_min=demand / 60,
                         median_real_gap_s=L * STRIDE_S, accepted_time_min=acc_s / 60,
                         supply_demand_ratio=acc_s / demand if demand else np.nan,
                         placements_for_typical_gap=placements,
                         placements_per_real_gap=placements / len(g) if len(g) else np.nan))
    return pd.DataFrame(rows), gaps
