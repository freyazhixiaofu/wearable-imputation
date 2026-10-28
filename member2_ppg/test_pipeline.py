"""Synthetic checks for HR estimation, timeline preservation and gap semantics."""
import unittest

import numpy as np

from process_wrist import Config, build_windows, gap_table, ppg_metrics


class PipelineChecks(unittest.TestCase):
    def setUp(self):
        self.fs = 128
        self.cfg = Config()
        t = np.arange(30 * self.fs) / self.fs
        self.data = dict(ppg_g=1000 + np.sin(2 * np.pi * 1.5 * t),
                         acc_x=np.zeros(len(t)), acc_y=np.zeros(len(t)),
                         acc_z=np.ones(len(t)))
        self.rates = {key: self.fs for key in self.data}

    def build(self, notes=''):
        return build_windows('synthetic', self.data, self.rates, notes, self.cfg, 'mask')[0]

    def test_sine_hr_and_valid_interior(self):
        t = np.arange(1024) / self.fs
        result = ppg_metrics(np.sin(2 * np.pi * 1.5 * t), self.fs, self.cfg)
        self.assertAlmostEqual(result['hr_ppg_candidate'], 90, delta=2)
        self.assertAlmostEqual(result['hr_acf_bpm'], 90, delta=2)
        table = self.build()
        self.assertTrue(table.iloc[2:-2].ppg_valid.all())
        self.assertTrue(np.allclose(table.loc[table.acc_valid, 'acc_dynamic_rms'], 0, atol=1e-10))

    def test_nan_keeps_grid_and_never_yields_accepted_hr(self):
        baseline = self.build()
        self.data['ppg_g'][12 * self.fs:13 * self.fs] = np.nan
        table = self.build()
        self.assertEqual(len(table), len(baseline))
        np.testing.assert_array_equal(table.start_sample, baseline.start_sample)
        overlaps = (table.start_s < 13) & (table.end_s > 12)
        self.assertFalse(table.loc[overlaps, 'ppg_valid'].any())
        self.assertTrue(table.loc[overlaps, 'hr_ppg'].isna().all())
        self.assertTrue(table.loc[overlaps, 'reject_reasons'].str.contains('nonfinite_ppg').all())

    def test_flatline_rejected(self):
        self.data['ppg_g'][:] = 5
        table = self.build()
        self.assertFalse(table.ppg_valid.any())
        self.assertTrue(table.reject_reasons.str.contains('flatline_or_plateau').all())

    def test_notes_matlab_index_and_filter_break(self):
        notes = 'Index 1281 to 1408 of all data from the wrist should be excluded from analysis'
        table = self.build(notes)
        expected = (table.start_sample < 1408) & (table.end_sample_exclusive > 1280)
        np.testing.assert_array_equal(table.documented_exclusion, expected)
        self.assertFalse(table.loc[expected, 'ppg_valid'].any())
        self.assertFalse(table.loc[expected, 'acc_valid'].any())

    def test_gap_duration_and_accepted_nan_invariant(self):
        table = self.build()
        table['ppg_valid'] = True
        table.loc[3:5, 'ppg_valid'] = False
        gaps = gap_table(table, self.cfg)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps.iloc[0].grid_duration_s, 6)
        self.assertEqual(gaps.iloc[0].raw_support_duration_s, 12)
        self.assertEqual(gaps.iloc[0].n_windows, 3)

    def test_short_acc_preserves_ppg_timeline(self):
        expected = len(self.build())
        self.data['acc_x'] = self.data['acc_x'][:10 * self.fs]
        table = self.build()
        self.assertEqual(len(table), expected)
        self.assertFalse(table.loc[table.end_s > 10, 'acc_valid'].any())

    def test_search_boundary_is_not_accepted_as_reliable_hr(self):
        t = np.arange(len(self.data['ppg_g'])) / self.fs
        self.data['ppg_g'] = np.sin(2 * np.pi * (40 / 60) * t)
        table = self.build()
        self.assertFalse(table.ppg_valid.any())
        self.assertTrue(table.iloc[2:-2].reject_reasons.str.contains('hr_search_boundary').all())


if __name__ == '__main__':
    unittest.main()
