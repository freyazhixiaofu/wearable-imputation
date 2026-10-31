"""Synthetic gap generators. All gaps are placed inside accepted, reference-valid windows, with at
least one observed window on each side, and never adjacent to another synthetic gap.

Protocols
  random_point     isolated single windows
  random_block     contiguous blocks, lengths ~ Uniform{1..60} windows (2-120 s), dataset-agnostic
  replay           lengths resampled from development participants' bounded real gaps
  context_matched  (length, movement bin) pairs resampled jointly from dev real gaps; the block's
                   mean motion must fall in that bin (nearest feasible bin as a recorded fallback)

All protocols target the same number of hidden windows; `match_budgets` then trims every protocol
to the smallest achieved total so the missing amount is matched exactly (up to one gap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gaps import assign_bin, block_motion, interior_starts

PROTOCOLS = ['random_point', 'random_block', 'replay', 'context_matched']
BLOCK_MAX = 60


class _Placer:
    def __init__(self, df, rng):
        self.elig = (df.ppg_valid.values & df.ref_valid.values).copy()
        self.motion = df.log_motion.values
        self.rng = rng
        self.hidden = np.zeros(len(df), bool)
        self.gaps = []

    def starts(self, L):
        return interior_starts(self.elig, L)

    def place(self, i, L, **meta):
        self.hidden[i:i + L] = True
        self.elig[max(0, i - 1):i + L + 1] = False  # keep a gap-free observed margin
        self.gaps.append(dict(start=i, n_windows=L, **meta))

    def n_hidden(self):
        return int(self.hidden.sum())


def make_mask(protocol, df, budget, rng, dev_gaps=None, edges=None, max_tries=20000):
    p = _Placer(df, rng)
    tries = 0
    lengths = dev_gaps.n_windows.values if dev_gaps is not None else None
    while p.n_hidden() < budget and tries < max_tries:
        tries += 1
        if protocol == 'random_point':
            L, meta = 1, {}
        elif protocol == 'random_block':
            L, meta = int(rng.integers(1, BLOCK_MAX + 1)), {}
        elif protocol == 'replay':
            L, meta = int(rng.choice(lengths)), {}
        elif protocol == 'context_matched':
            j = rng.integers(len(dev_gaps))
            L, want = int(dev_gaps.n_windows.values[j]), int(dev_gaps['bin'].values[j])
            meta = dict(requested_bin=want)
        else:
            raise ValueError(protocol)
        L = min(L, budget - p.n_hidden()) if protocol != 'context_matched' else L
        st = p.starts(L)
        if not len(st):
            continue  # length infeasible here; resample (recorded via placed-length stats)
        if protocol == 'context_matched':
            bins = assign_bin(block_motion(p.motion, L)[st], edges)
            order = sorted(set(bins[bins >= 0]), key=lambda b: (abs(b - want), b))
            if not order:
                continue
            b = order[0]
            st = st[bins == b]
            meta.update(placed_bin=int(b), bin_fallback=int(abs(b - want)))
        i = int(rng.choice(st))
        p.place(i, L, **meta)
    g = pd.DataFrame(p.gaps)
    return p.hidden, g


def match_budgets(masks, rng):
    """Trim each protocol's gaps (whole gaps, random order) to the smallest achieved hidden total."""
    target = min(int(m.sum()) for m, _ in masks.values())
    out = {}
    for name, (m, g) in masks.items():
        m = m.copy()
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31))) if len(g) else g
        keep = []
        for r in g.itertuples(index=False):
            if m.sum() <= target:
                keep.append(r)
                continue
            m[r.start:r.start + r.n_windows] = False
        kept = pd.DataFrame(keep, columns=g.columns) if len(g) else g
        # residual overshoot (< one gap) is accepted; record it
        out[name] = (m, kept)
    return out, target


def make_all_masks(df, frac, rng, dev_gaps, edges, protocols=PROTOCOLS):
    budget = int(frac * (df.ppg_valid.values & df.ref_valid.values).sum())
    masks = {p: make_mask(p, df, budget, rng, dev_gaps, edges) for p in protocols}
    return match_budgets(masks, rng)


def neutral_training_mask(df, rng, frac=0.15):
    """Training mask shared by every learned imputer: half isolated points, half Uniform{1..60} blocks."""
    budget = int(frac * df.ppg_valid.values.sum())
    p = _Placer(df.assign(ref_valid=True), rng)
    tries = 0
    while p.n_hidden() < budget and tries < 20000:
        tries += 1
        L = 1 if rng.random() < 0.5 else int(rng.integers(1, BLOCK_MAX + 1))
        st = p.starts(L)
        if len(st):
            p.place(int(rng.choice(st)), L)
    return p.hidden
