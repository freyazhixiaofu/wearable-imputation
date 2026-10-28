"""Verify ACF diagnostics and nonoverlapping targeted selection."""
import unittest
import numpy as np
import pandas as pd
from audit_windows import acf_details, select_windows
from process_wrist import Config, ppg_metrics


class AuditChecks(unittest.TestCase):
    def test_actual_acf_matches_v1_estimator(self):
        fs = 128
        t = np.arange(1024) / fs
        y = np.sin(2 * np.pi * 1.5 * t)
        ac, peaks, chosen = acf_details(y, fs, Config())
        metrics = ppg_metrics(y, fs, Config())
        self.assertAlmostEqual(ac[0], 1)
        self.assertIn(chosen, peaks)
        self.assertAlmostEqual(ac[chosen], metrics['acf_peak'])
        self.assertAlmostEqual(60 * fs / chosen, metrics['hr_acf_bpm'])

    def test_constant_signal_has_no_acf_candidate(self):
        ac, peaks, chosen = acf_details(np.ones(1024), 128, Config())
        self.assertEqual(len(ac), 0)
        self.assertEqual(len(peaks), 0)
        self.assertIsNone(chosen)

    def test_selection_reproducible_and_separated(self):
        n = 300
        d = pd.DataFrame(dict(window_id=np.arange(n), start_s=np.arange(n)*2,
                             acc_valid=True, acc_dynamic_rms=np.tile(np.arange(50), 6),
                             ppg_valid=np.arange(n)<150, acf_peak=.38,
                             reject_reasons=['']*150 + ['weak_periodicity']*150))
        a, availability = select_windows(d)
        b, _ = select_windows(d)
        self.assertEqual(a, b)
        starts = sorted(d.loc[d.window_id.isin([x['window_id'] for x in a]), 'start_s'])
        self.assertTrue(np.all(np.diff(starts)>=30))
        self.assertEqual(len(a), len({x['window_id'] for x in a}))
        categories = {x['category'] for x in a}
        self.assertIn('accepted_low_motion', categories)
        self.assertIn('accepted_high_motion', categories)
        self.assertIn('acf_only_borderline', categories)


if __name__ == '__main__':
    unittest.main()
