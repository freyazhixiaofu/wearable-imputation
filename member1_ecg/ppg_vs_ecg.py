"""Compare a PPG batch's accepted HR against the ECG reference, per participant.

    python member1_ecg/ppg_vs_ecg.py --ppg-batch batch_v2

Writes member1_ecg/results/ecg_ref_v1/ppg_vs_ecg_<batch>.csv with, per participant:
  accepted_with_ecg        accepted PPG windows that also have a valid ECG HR
  coverage_among_ecg_valid share of ECG-valid windows the PPG rule accepted
  mae_bpm, off_gt10        |PPG - ECG| on those windows; share with error > 10 bpm
  half_rate                share of those windows with PPG/ECG in (0.4, 0.6)
  rejected_but_correct     share of rejected ECG-valid windows whose PPG candidate HR was within 10 bpm
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ECG = ROOT / 'member1_ecg' / 'results' / 'ecg_ref_v1'


def check(batch):
    ppg = ROOT / 'member2_ppg' / 'results' / batch
    rows = []
    for p in sorted(d.name for d in ppg.iterdir() if (d / 'windows.csv').exists()):
        w = pd.read_csv(ppg / p / 'windows.csv', usecols=['window_id', 'ppg_valid', 'hr_ppg', 'hr_ppg_candidate'])
        e = pd.read_csv(ECG / p / 'ecg_windows.csv', usecols=['window_id', 'hr_ecg', 'ecg_valid'])
        assert (w.window_id.values == e.window_id.values).all(), p
        acc, ev = w.ppg_valid.astype(bool), e.ecg_valid.astype(bool)
        m = acc & ev
        d = (w.hr_ppg - e.hr_ecg)[m]
        r = (w.hr_ppg / e.hr_ecg)[m]
        rej = ~acc & ev
        cand_ok = (w.hr_ppg_candidate - e.hr_ecg).abs() <= 10
        rows.append(dict(participant=p, accepted_with_ecg=int(m.sum()),
                         coverage_among_ecg_valid=m.sum() / ev.sum(),
                         mae_bpm=d.abs().mean() if m.any() else np.nan,
                         off_gt10=(d.abs() > 10).mean() if m.any() else np.nan,
                         half_rate=((r > .4) & (r < .6)).mean() if m.any() else np.nan,
                         rejected_but_correct=(cand_ok & rej).sum() / rej.sum() if rej.any() else np.nan))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppg-batch', default='batch_v2')
    a = ap.parse_args()
    t = check(a.ppg_batch)
    t.to_csv(ECG / f'ppg_vs_ecg_{a.ppg_batch}.csv', index=False, float_format='%.4f')
    print(t.round(3).to_string(index=False))
    print('participant means:', t[['coverage_among_ecg_valid', 'mae_bpm', 'off_gt10', 'half_rate']].mean().round(4).to_dict())
