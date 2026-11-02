"""Final v2 keeps the evaluated rule and the v1 output layout."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import final_v2

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/final_v2.BATCH_PATH
DEVELOPMENT=['an0','e61','k2s','kjd','n31','qm9','u7y','w4p']
HELD_OUT=['fex','l38','ngh','p5d','p9p','ssx','trh','tz8']


class FinalV2Checks(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame(dict(participant_id=['toy']*41,center_s=4+2*np.arange(41),
            hr_spectral_bpm=[90.]*41,hr_acf_bpm=[90.]*41,
            raw_spectral_concentration=[.5]*41,acf_peak=[.3]*41,spectral_entropy=[.5]*41,
            finite_ratio=[1.]*41,documented_exclusion=[False]*41,filter_safe=[True]*41,
            longest_flat_run_s=[.01]*41))
        self.d.loc[[10,14,26,30],'acf_peak']=.8

    def test_config_preserves_the_evaluated_thresholds(self):
        canonical=json.loads((ROOT/final_v2.CONFIG_PATH).read_text())
        self.assertEqual(final_v2.settings_sha(canonical),final_v2.EXPECTED_SETTINGS_SHA256)
        self.assertEqual(canonical['context_rule']['anchor_min_acf'],.8)
        self.assertEqual(canonical['context_rule']['candidate_min_acf'],.3)
        self.assertEqual(canonical['context_rule']['max_deviation_bpm'],5.)
        self.assertEqual(canonical['context_rule']['pulse_support'],'none')
        self.assertEqual(canonical['base_config']['half_rate_action'],'off')

    def test_final_decisions_keep_the_candidates_measured_hr(self):
        # Only rows bracketed by two separated original anchors on each side
        # recover; the other weak-ACF rows remain rejected.
        expected=np.zeros(len(self.d),dtype=bool)
        expected[[10,14,26,30]]=True
        expected[15:26]=True
        before=self.d.copy()
        actual=final_v2.apply_final_v2(self.d,root=ROOT)
        np.testing.assert_array_equal(actual.ppg_valid,expected)
        np.testing.assert_allclose(actual.hr_ppg,np.where(expected,90.,np.nan),equal_nan=True)
        np.testing.assert_allclose(actual.hr_ppg_candidate,90.)
        self.assertTrue(actual.loc[expected,'reject_reasons'].eq('').all())
        self.assertTrue(actual.loc[~expected,'reject_reasons'].str.len().gt(0).all())
        self.assertTrue(actual.rule_version.eq('ppg_only_dev_v2').all())
        self.assertTrue(actual.context_rescued.iloc[20])
        pd.testing.assert_frame_equal(before,self.d)

    def test_ecg_half_rate_and_motion_cannot_change_final_acceptance(self):
        a=final_v2.apply_final_v2(self.d,root=ROOT)
        d=self.d.assign(hr_ecg=220.,hr_double_bpm=180.,half_rate_power_ratio=1.,acc_dynamic_rms=999.)
        b=final_v2.apply_final_v2(d,root=ROOT)
        np.testing.assert_array_equal(a.ppg_valid,b.ppg_valid)
        np.testing.assert_allclose(a.hr_ppg,b.hr_ppg,equal_nan=True)

    def test_every_participant_uses_the_v1_batch_layout(self):
        for pid in DEVELOPMENT+HELD_OUT:
            paths=final_v2.get_participant_paths(pid,root=ROOT)
            expected=BATCH/pid
            self.assertEqual(Path(paths['windows']).resolve(),(expected/'windows.csv').resolve())
            self.assertEqual(Path(paths['gaps']).resolve(),(expected/'gaps.csv').resolve())
            self.assertEqual(paths['split'],'development' if pid in DEVELOPMENT else 'evaluation')

    def test_unknown_participant_has_no_implicit_initial_v2_fallback(self):
        with self.assertRaises((ValueError,KeyError)):
            final_v2.get_participant_paths('not_a_participant',root=ROOT)


class FinalV2MetadataTamperChecks(unittest.TestCase):
    def setUp(self):
        temporary_parent=ROOT/'tmp'
        temporary_parent.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=temporary_parent)
        self.root=Path(self.temporary.name)
        self.manifest=json.loads((ROOT/final_v2.MANIFEST_PATH).read_text())
        paths=[final_v2.CONFIG_PATH,final_v2.MANIFEST_PATH]+list(self.manifest['source_files'])
        for relative in paths:
            destination=self.root/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/relative).read_bytes())
        final_v2.resolve_batch(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_rule_runs_without_any_saved_participant_outputs(self):
        for entry in self.manifest['participant_files'].values():
            self.assertFalse((self.root/entry['windows']).exists())
            self.assertFalse((self.root/entry['gaps']).exists())
        fixture=FinalV2Checks()
        fixture.setUp()
        result=final_v2.apply_final_v2(fixture.d,root=self.root)
        self.assertTrue(result.context_rescued.iloc[20])
        self.assertTrue(final_v2.verify_rule(self.root)['settings_and_sources_unchanged'])

    def test_full_results_verification_still_requires_local_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            final_v2.verify_batch(self.root)

    def test_updated_metadata_hash_cannot_authorize_a_threshold_change(self):
        config_path=self.root/final_v2.CONFIG_PATH
        config=json.loads(config_path.read_text())
        config['context_rule']['max_deviation_bpm']=8.
        config_path.write_text(json.dumps(config),encoding='utf-8')
        self.manifest['config_sha256']=final_v2.sha(config_path)
        self.manifest['config']=config
        (self.root/final_v2.MANIFEST_PATH).write_text(json.dumps(self.manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'thresholds differ'):
            final_v2.resolve_batch(self.root)

    def test_rule_code_cannot_drift_under_final_v2_name(self):
        source=self.root/'member2_ppg/ppg_context_rule.py'
        source.write_bytes(source.read_bytes()+b'\n# altered snapshot\n')
        with self.assertRaisesRegex(ValueError,'decision source changed'):
            final_v2.resolve_batch(self.root)

    def test_release_paths_cannot_escape_the_repository(self):
        with self.assertRaisesRegex(ValueError,'escapes'):
            final_v2.repo_path(self.root,'../other_project/windows.csv')


if __name__=='__main__':
    unittest.main()
