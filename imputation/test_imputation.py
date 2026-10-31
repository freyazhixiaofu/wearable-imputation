import unittest

import numpy as np
import pandas as pd

from gaps import assign_bin, block_motion, interior_starts, real_gaps
from imputers import AR1Smoother, GBMContext, Linear, LOCF, Pchip
from masks import PROTOCOLS, make_all_masks, neutral_training_mask


def toy(n=3000, seed=0, pid='toy'):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hr = 70 + 10 * np.sin(t / 200) + rng.normal(0, 1, n)
    valid = rng.random(n) > 0.1
    for s in rng.choice(n - 40, 25, replace=False):  # real gaps
        valid[s:s + rng.integers(2, 30)] = False
    valid[:3] = valid[-3:] = True
    motion = np.log10(np.abs(rng.normal(0.05, 0.05, n)) + 1e-3)
    acc = 10 ** motion
    return pd.DataFrame(dict(participant_id=pid, window_id=t, center_s=4 + 2 * t, ppg_valid=valid,
                             hr_ppg=hr, hr_obs=np.where(valid, hr, np.nan), hr_ref=hr, ref_valid=True,
                             ref_source='ecg', acc_valid=True, documented_exclusion=False,
                             acc_dynamic_rms=acc, acc_jerk_rms=acc * 10, acc_magnitude_std=acc,
                             log_motion=motion))


class GapTests(unittest.TestCase):
    def test_interior_starts_margin(self):
        e = np.array([1, 1, 1, 1, 0, 1, 1, 1], bool)
        self.assertEqual(list(interior_starts(e, 2)), [1])
        self.assertEqual(list(interior_starts(e, 1)), [1, 2, 6])

    def test_block_motion(self):
        m = np.array([0, 1, 2, np.nan, 4.0])
        np.testing.assert_allclose(block_motion(m, 2), [0.5, 1.5, 2, 4])

    def test_real_gaps_bounded(self):
        df = toy()
        g = real_gaps(df)
        self.assertEqual(g.n_windows.sum(), (~df.ppg_valid).sum())
        self.assertTrue(g.bounded.all())  # toy edges are accepted

    def test_assign_bin(self):
        edges = np.array([-np.inf, 0, 1, np.inf])
        self.assertEqual(list(assign_bin([-1, 0.5, 2, np.nan], edges)), [0, 1, 2, -1])


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.df = toy()
        dev = real_gaps(toy(seed=1))
        self.edges = np.quantile(self.df.log_motion, np.linspace(0, 1, 6)); self.edges[[0, -1]] = [-np.inf, np.inf]
        dev['bin'] = assign_bin(dev.log_motion, self.edges)
        self.dev = dev

    def test_masks_valid_and_matched(self):
        masks, target = make_all_masks(self.df, 0.1, np.random.default_rng(0), self.dev, self.edges)
        elig = self.df.ppg_valid.values
        sizes = []
        for p in PROTOCOLS:
            m, g = masks[p]
            self.assertTrue((m <= elig).all(), p)  # only accepted windows hidden
            # every hidden run is bounded by observed windows
            e = np.diff(np.r_[0, m.astype(int), 0]); s, t = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
            self.assertTrue((s > 0).all() and (t < len(m)).all())
            self.assertTrue(elig[s - 1].all() and elig[t].all(), p)
            sizes.append(m.sum())
        self.assertLessEqual(max(sizes) - min(sizes), 60)  # matched up to one gap

    def test_neutral_mask(self):
        m = neutral_training_mask(self.df, np.random.default_rng(0))
        self.assertTrue((m <= self.df.ppg_valid.values).all())
        self.assertGreater(m.sum(), 0)


class ImputerTests(unittest.TestCase):
    def test_observed_unchanged_and_finite(self):
        df = toy()
        obs = df.hr_obs.values.copy()
        train = [(toy(seed=s), toy(seed=s).hr_obs.values) for s in (3, 4)]
        rng = np.random.default_rng(0)
        for m in [LOCF(), Linear(), Pchip(), AR1Smoother().fit(train, rng), GBMContext(True).fit(train, rng)]:
            out = m.impute(df, obs)
            self.assertTrue(np.isfinite(out).all(), m.name)
            ok = np.isfinite(obs)
            np.testing.assert_allclose(out[ok], obs[ok], err_msg=m.name)

    def test_linear_exact_on_line(self):
        obs = np.array([60, np.nan, np.nan, 66, np.nan, 70.0])
        np.testing.assert_allclose(Linear().impute(None, obs), [60, 62, 64, 66, 68, 70])


if __name__ == '__main__':
    unittest.main()
