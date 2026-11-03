"""Load the shared window grid (member2 PPG + optional member1 ECG) into per-participant frames.

Every frame has one row per 8 s / 2 s-stride window with:
  hr_obs      wrist PPG HR where the PPG-only rule accepted the window, else NaN (imputer input)
  hr_ref      evaluation reference: ECG HR when ECG results exist, else PPG HR on accepted windows
  ref_valid   whether hr_ref is usable
  ref_source  'ecg' or 'ppg_provisional'
  acc_*       motion features (never used to define gaps)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PPG_RESULTS = ROOT / 'member2_ppg' / 'results'
PPG_BATCH = 'batch_v2'  # final PPG rule; batch_v1 kept for the sensitivity run
PPG_DIR = PPG_RESULTS / PPG_BATCH
ECG_DIR = ROOT / 'member1_ecg' / 'results' / 'ecg_ref_v1'
ACC_COLS = ['acc_dynamic_rms', 'acc_jerk_rms', 'acc_magnitude_std']
STRIDE_S = 2.0


def participants(ppg_dir=PPG_DIR):
    return sorted(p.name for p in Path(ppg_dir).iterdir() if (p / 'windows.csv').exists())


def load_participant(pid, ppg_dir=PPG_DIR, ecg_dir=ECG_DIR, require_ecg=False):
    cols = ['participant_id', 'window_id', 'start_sample', 'end_sample_exclusive', 'center_s',
            'ppg_valid', 'hr_ppg', 'acc_valid', 'documented_exclusion'] + ACC_COLS
    w = pd.read_csv(Path(ppg_dir) / pid / 'windows.csv', usecols=cols)
    w['ppg_valid'] = w.ppg_valid.astype(bool)
    w['hr_obs'] = w.hr_ppg.where(w.ppg_valid)
    ecg_path = Path(ecg_dir) / pid / 'ecg_windows.csv' if ecg_dir else None
    if ecg_path is not None and ecg_path.exists():
        e = pd.read_csv(ecg_path, usecols=['window_id', 'start_sample', 'hr_ecg', 'ecg_valid'])
        if not (e.window_id.values == w.window_id.values).all() or not (e.start_sample.values == w.start_sample.values).all():
            raise ValueError(f'{pid}: ECG grid does not match PPG grid')
        w['hr_ref'] = e.hr_ecg.values
        w['ref_valid'] = e.ecg_valid.astype(bool).values & np.isfinite(e.hr_ecg.values)
        w['ref_source'] = 'ecg'
    elif require_ecg:
        raise FileNotFoundError(ecg_path)
    else:
        w['hr_ref'] = w.hr_obs
        w['ref_valid'] = w.ppg_valid
        w['ref_source'] = 'ppg_provisional'
    # log-motion used for binning and as model input; NaN where ACC invalid
    w['log_motion'] = np.log10(w.acc_dynamic_rms.where(w.acc_valid.astype(bool)).clip(lower=1e-4))
    return w


def load_all(pids=None, **kw):
    pids = pids or participants(kw.get('ppg_dir', PPG_DIR))
    return {p: load_participant(p, **kw) for p in pids}


def ecg_available(frames):
    return all(f.ref_source.iloc[0] == 'ecg' for f in frames.values())
