import unittest

import numpy as np
import pandas as pd

from process_ecg import EcgConfig, detect_rpeaks, window_hr

FS = 128.0


def grid(n_samples, pid='syn'):
    w, s = int(8 * FS), int(2 * FS)
    starts = np.arange(0, n_samples - w + 1, s)
    return pd.DataFrame(dict(participant_id=pid, window_id=np.arange(len(starts)),
                             start_sample=starts, end_sample_exclusive=starts + w))


class EcgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import neurokit2 as nk
        a = nk.ecg_simulate(duration=300, sampling_rate=FS, heart_rate=60, noise=0.02, random_state=1)
        b = nk.ecg_simulate(duration=300, sampling_rate=FS, heart_rate=110, noise=0.02, random_state=2)
        cls.ecg = np.r_[a, b].astype(float)
        cls.cfg = EcgConfig()

    def test_hr_recovered(self):
        peaks = detect_rpeaks(self.ecg, FS, self.cfg)
        t = window_hr(grid(len(self.ecg)), self.ecg, FS, peaks, self.cfg)
        c = (t.start_sample + t.end_sample_exclusive) / 2 / FS
        first, second = t[(c > 20) & (c < 280)], t[(c > 320) & (c < 580)]
        self.assertGreater(first.ecg_valid.mean(), 0.95)
        self.assertGreater(second.ecg_valid.mean(), 0.95)
        self.assertLess(abs(first.hr_ecg.median() - 60), 3)
        self.assertLess(abs(second.hr_ecg.median() - 110), 4)

    def test_missing_samples_invalidate(self):
        ecg = self.ecg.copy()
        ecg[int(100 * FS):int(130 * FS)] = np.nan
        peaks = detect_rpeaks(ecg, FS, self.cfg)
        t = window_hr(grid(len(ecg)), ecg, FS, peaks, self.cfg)
        c = (t.start_sample + t.end_sample_exclusive) / 2 / FS
        self.assertFalse(t[(c > 104) & (c < 126)].ecg_valid.any())
        self.assertTrue(t[(c > 104) & (c < 126)].hr_ecg.isna().all())

    def test_noise_mostly_invalid(self):
        rng = np.random.default_rng(0)
        ecg = self.ecg.copy()
        ecg[int(200 * FS):int(260 * FS)] = rng.normal(0, 1, int(60 * FS))
        peaks = detect_rpeaks(ecg, FS, self.cfg)
        t = window_hr(grid(len(ecg)), ecg, FS, peaks, self.cfg)
        c = (t.start_sample + t.end_sample_exclusive) / 2 / FS
        self.assertLess(t[(c > 206) & (c < 254)].ecg_valid.mean(), 0.3)

    def test_keys_match_grid(self):
        g = grid(len(self.ecg))
        t = window_hr(g, self.ecg, FS, detect_rpeaks(self.ecg, FS, self.cfg), self.cfg)
        self.assertTrue((t.window_id.values == g.window_id.values).all())
        self.assertTrue(t.hr_ecg[~t.ecg_valid].isna().all())


if __name__ == '__main__':
    unittest.main()
