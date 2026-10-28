"""Batch note handling and review-before-processing checks."""
from pathlib import Path
import tempfile
import unittest

from scipy.io import savemat

from process_all import review_notes
from process_wrist import exclusions_from_notes


class NotesChecks(unittest.TestCase):
    def test_wrist_interruption_not_falsely_called_ambiguous(self):
        notes = 'The wrist device experienced an interruption. Index 100 to 200 of all data from the wrist should be excluded from analysis.'
        exclusions, warnings = exclusions_from_notes(notes, 128, 1000, 'mask')
        self.assertEqual(exclusions[0]['start_sample'], 99)
        self.assertEqual(exclusions[0]['end_sample_exclusive'], 200)
        self.assertTrue(exclusions[0]['attribution_verified'])
        self.assertFalse(any('unresolved' in warning for warning in warnings))

    def test_an0_conflict_remains_explicit(self):
        notes = 'The ankle device experienced an interruption. Index 100 to 200 of all data from the wrist should be excluded from analysis.'
        exclusions, warnings = exclusions_from_notes(notes, 128, 1000, 'mask')
        self.assertFalse(exclusions[0]['attribution_verified'])
        self.assertTrue(any('unresolved' in warning for warning in warnings))

    def test_ecg_notes_do_not_exclude_ppg(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp).resolve()
            assert root.is_relative_to(Path(__file__).resolve().parent)
            path = root / 'WildPPG_Part_test.mat'
            savemat(path, {'id': 'test', 'notes': 'ECG recordings of this participant are noisy.'})
            review = review_notes([path], root)
            self.assertTrue(review.iloc[0].ecg_noise_warning)
            self.assertFalse(review.iloc[0].wrist_exclusion)

    def test_unrecognized_notes_stop_batch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp).resolve()
            assert root.is_relative_to(Path(__file__).resolve().parent)
            path = root / 'WildPPG_Part_test.mat'
            savemat(path, {'id': 'test', 'notes': 'Sensor shifted; alignment needs review.'})
            with self.assertRaisesRegex(ValueError, 'Unrecognized notes'):
                review_notes([path], root)


if __name__ == '__main__':
    unittest.main()
