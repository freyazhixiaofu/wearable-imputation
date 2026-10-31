"""Candidate HR imputers. Interface:
    m.fit(train_items, rng)   train_items: list of (frame, obs) where obs is hr_obs with the neutral
                               training mask applied (NaN = missing); targets come from frame.hr_ref
    m.impute(frame, obs)      returns a full HR array; observed entries are left unchanged

Candidates span short-gap interpolators, long-gap mean-reverting models, motion-only regressors,
and learned context models with and without accelerometer input, so that their ranking can
plausibly depend on the gap regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from data import ACC_COLS

BASELINE_HALFWIDTH = 300  # windows (+/-10 min)


def baseline(obs):
    """Local HR level from observed windows only (rolling median, +/-10 min)."""
    s = pd.Series(obs)
    b = s.rolling(2 * BASELINE_HALFWIDTH + 1, center=True, min_periods=5).median()
    b = b.ffill().bfill()
    fallback = np.nanmedian(obs) if np.isfinite(obs).any() else 70.0
    return b.fillna(fallback).values


def _fill_edges(x, obs):
    idx = np.flatnonzero(np.isfinite(obs))
    if not len(idx):
        return np.full(len(obs), 70.0)
    x[:idx[0]] = obs[idx[0]]
    x[idx[-1] + 1:] = obs[idx[-1]]
    return x


class Imputer:
    name = 'base'
    uses_acc = False

    def fit(self, train_items, rng):
        return self

    def impute(self, frame, obs):
        raise NotImplementedError


class LOCF(Imputer):
    name = 'locf'

    def impute(self, frame, obs):
        return pd.Series(obs).ffill().bfill().fillna(70.0).values


class Linear(Imputer):
    name = 'linear'

    def impute(self, frame, obs):
        idx = np.flatnonzero(np.isfinite(obs))
        if len(idx) < 2:
            return LOCF().impute(frame, obs)
        return np.interp(np.arange(len(obs)), idx, obs[idx])


class Pchip(Imputer):
    name = 'pchip'

    def impute(self, frame, obs):
        idx = np.flatnonzero(np.isfinite(obs))
        if len(idx) < 3:
            return Linear().impute(frame, obs)
        x = PchipInterpolator(idx, obs[idx], extrapolate=False)(np.arange(len(obs)))
        return _fill_edges(x, obs)


class LocalMedian(Imputer):
    """Sanity floor: the local rolling-median level, ignoring gap boundaries."""
    name = 'local_median'

    def impute(self, frame, obs):
        out = baseline(obs)
        m = np.isfinite(obs)
        out[m] = obs[m]
        return out


class AR1Smoother(Imputer):
    """Kalman/RTS smoother for an AR(1) deviation from the local baseline: long gaps decay to the
    baseline instead of being bridged linearly. phi, q, r estimated from dev lag-1 moments."""
    name = 'ar1_kalman'

    def fit(self, train_items, rng):
        sq, l1, l2 = [], [], []
        for f, obs in train_items:
            d = obs - baseline(obs)
            sq.append(d[np.isfinite(d)] ** 2)
            ok1 = np.isfinite(d[:-1]) & np.isfinite(d[1:])
            ok2 = np.isfinite(d[:-2]) & np.isfinite(d[2:])
            l1.append(d[:-1][ok1] * d[1:][ok1]); l2.append(d[:-2][ok2] * d[2:][ok2])
        var, cov1, cov2 = (np.mean(np.concatenate(v)) for v in (sq, l1, l2))
        # observations = AR(1) state (variance s2) + white noise r:
        #   cov1 = phi*s2, cov2 = phi^2*s2, var = s2 + r
        self.phi = float(np.clip(cov2 / cov1 if cov1 > 0 else 0.95, 0.8, 0.999))
        s2 = max(cov1 / self.phi, 1e-3)
        self.r = float(max(var - s2, 0.5))
        self.q = float(max(s2 * (1 - self.phi ** 2), 1e-3))
        return self

    def impute(self, frame, obs):
        base = baseline(obs)
        y = obs - base
        n, phi, q, r = len(y), self.phi, self.q, self.r
        xf, pf = np.zeros(n), np.zeros(n)
        xp, pp = np.zeros(n), np.zeros(n)
        x, p = 0.0, q / (1 - phi ** 2)
        for t in range(n):
            x, p = phi * x, phi * phi * p + q
            xp[t], pp[t] = x, p
            if np.isfinite(y[t]):
                k = p / (p + r)
                x, p = x + k * (y[t] - x), (1 - k) * p
            xf[t], pf[t] = x, p
        xs = xf.copy()
        for t in range(n - 2, -1, -1):
            c = pf[t] * phi / pp[t + 1]
            xs[t] = xf[t] + c * (xs[t + 1] - xp[t + 1])
        out = base + xs
        m = np.isfinite(obs)
        out[m] = obs[m]
        return out


def _acc_matrix(frame):
    a = frame[ACC_COLS].where(frame.acc_valid.astype(bool), np.nan)
    a = np.log10(a.clip(lower=1e-4))
    feats = {f'{c}_now': a[c] for c in ACC_COLS}
    for k in (15, 60):  # 30 s and 2 min trailing motion history (cardiac lag)
        feats[f'dyn_mean_{k}'] = a['acc_dynamic_rms'].rolling(k, min_periods=1).mean()
    feats['dyn_mean_fwd15'] = a['acc_dynamic_rms'][::-1].rolling(15, min_periods=1).mean()[::-1]
    return pd.DataFrame(feats).values


def _gap_context(obs):
    """Per-window gap geometry from the observed pattern."""
    n = len(obs)
    m = np.isfinite(obs)
    idx = np.arange(n)
    left = np.where(m, idx, -1); left = np.maximum.accumulate(left)
    right = np.where(m, idx, n); right = np.minimum.accumulate(right[::-1])[::-1]
    lv = np.where(left >= 0, obs[np.clip(left, 0, n - 1)], np.nan)
    rv = np.where(right < n, obs[np.clip(right, 0, n - 1)], np.nan)
    dl = np.where(left >= 0, idx - left, np.nan)
    dr = np.where(right < n, right - idx, np.nan)
    glen = np.maximum(dl + dr - 1, 0)
    return lv, rv, dl, dr, glen


class GBMContext(Imputer):
    """Gradient-boosted regressor on gap geometry (+ optional motion), predicting deviation from linear."""

    def __init__(self, use_acc):
        self.uses_acc = use_acc
        self.name = 'gbm_ctx_acc' if use_acc else 'gbm_ctx'

    def _features(self, frame, obs):
        lin = Linear().impute(frame, obs)
        base = baseline(obs)
        lv, rv, dl, dr, glen = _gap_context(obs)
        X = np.c_[lin - base, lv - base, rv - base, np.log1p(dl), np.log1p(dr), np.log1p(glen),
                  dl / np.maximum(glen, 1)]
        if self.uses_acc:
            X = np.c_[X, _acc_matrix(frame)]
        return X, lin

    def fit(self, train_items, rng):
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xs, ys = [], []
        for f, obs in train_items:
            miss = ~np.isfinite(obs) & f.ref_valid.values
            X, lin = self._features(f, obs)
            Xs.append(X[miss]); ys.append((f.hr_ref.values - lin)[miss])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if len(y) > 400_000:
            k = rng.choice(len(y), 400_000, replace=False); X, y = X[k], y[k]
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   loss='absolute_error', random_state=0).fit(X, y)
        return self

    def impute(self, frame, obs):
        X, lin = self._features(frame, obs)
        out = lin + self.model.predict(X)
        m = np.isfinite(obs)
        out[m] = obs[m]
        return out


class AccOnly(Imputer):
    """Motion-only regressor: local HR level + f(motion). Ignores the gap's boundary values."""
    name = 'acc_only'
    uses_acc = True

    def fit(self, train_items, rng):
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xs, ys = [], []
        for f, obs in train_items:
            miss = ~np.isfinite(obs) & f.ref_valid.values
            Xs.append(_acc_matrix(f)[miss]); ys.append((f.hr_ref.values - baseline(obs))[miss])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if len(y) > 400_000:
            k = rng.choice(len(y), 400_000, replace=False); X, y = X[k], y[k]
        self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                   loss='absolute_error', random_state=0).fit(X, y)
        return self

    def impute(self, frame, obs):
        out = baseline(obs) + self.model.predict(_acc_matrix(frame))
        m = np.isfinite(obs)
        out[m] = obs[m]
        return out


class BiGRU(Imputer):
    """Small bidirectional GRU over the window sequence; inputs: observed HR deviation, mask,
    (optional) motion features. Loss on missing, reference-valid windows only."""

    def __init__(self, use_acc, steps=800, seq_len=120, hidden=32, batch=32, device="cpu"):
        self.uses_acc = use_acc
        self.name = 'gru_acc' if use_acc else 'gru'
        self.steps, self.seq_len, self.hidden, self.batch, self.device = steps, seq_len, hidden, batch, device

    def _inputs(self, frame, obs):
        base = baseline(obs)
        dev = np.where(np.isfinite(obs), (obs - base) / 20.0, 0.0)
        X = [dev, np.isfinite(obs).astype(float)]
        if self.uses_acc:
            a = _acc_matrix(frame)
            a = np.nan_to_num((a - self.acc_mu) / self.acc_sd)
            X += list(a.T)
        return np.c_[tuple(X)].astype(np.float32), base

    def fit(self, train_items, rng):
        import torch
        torch.manual_seed(int(rng.integers(1 << 31)))
        if self.uses_acc:
            A = np.concatenate([_acc_matrix(f) for f, _ in train_items])
            self.acc_mu, self.acc_sd = np.nanmean(A, 0), np.nanstd(A, 0) + 1e-6
        seqs = []
        for f, obs in train_items:
            X, base = self._inputs(f, obs)
            y = ((f.hr_ref.values - base) / 20.0).astype(np.float32)
            w = (~np.isfinite(obs) & f.ref_valid.values).astype(np.float32)
            seqs.append((X, np.nan_to_num(y), w))
        d_in = seqs[0][0].shape[1]
        self.net = _GRUNet(d_in, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        lens = np.array([len(s[0]) for s in seqs]); probs = lens / lens.sum()
        self.net.train()
        for step in range(self.steps):
            xb, yb, wb = [], [], []
            while len(xb) < self.batch:
                k = rng.choice(len(seqs), p=probs)
                X, y, w = seqs[k]
                i = int(rng.integers(0, len(X) - self.seq_len))
                if w[i:i + self.seq_len].sum() == 0:
                    continue
                xb.append(X[i:i + self.seq_len]); yb.append(y[i:i + self.seq_len]); wb.append(w[i:i + self.seq_len])
            xb, yb, wb = (torch.from_numpy(np.stack(v)) for v in (xb, yb, wb))
            pred = self.net(xb)
            loss = (torch.abs(pred - yb) * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval()
        return self

    def impute(self, frame, obs, chunk=1200, ctx=150):
        import torch
        X, base = self._inputs(frame, obs)
        n = len(X)
        pred = np.zeros(n, np.float32)
        with torch.no_grad():
            for s in range(0, n, chunk):
                a, b = max(0, s - ctx), min(n, s + chunk + ctx)
                p = self.net(torch.from_numpy(X[a:b][None]))[0].numpy()
                pred[s:min(n, s + chunk)] = p[s - a:s - a + min(chunk, n - s)]
        out = base + 20.0 * pred
        m = np.isfinite(obs)
        out[m] = obs[m]
        return out


def _GRUNet(d_in, hidden):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(d_in, hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * hidden, 1)

        def forward(self, x):
            return self.head(self.gru(x)[0]).squeeze(-1)
    return Net()


def candidates(fast=False):
    m = [LOCF(), Linear(), Pchip(), LocalMedian(), AR1Smoother(), AccOnly(),
         GBMContext(False), GBMContext(True)]
    if not fast:
        m += [BiGRU(False), BiGRU(True)]
    return m
