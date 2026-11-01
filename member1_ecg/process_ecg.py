"""Sternum ECG -> reference heart rate on the same window grid as the wrist PPG pipeline.

ECG is used only as the evaluation reference (and as a training target for learned imputers);
it never enters PPG acceptance decisions or imputer inputs.

Run (one participant):
    python member1_ecg/process_ecg.py --data-dir WildPPG/WildPPG/data --participant an0
Run (all participants with a member2 windows.csv):
    python member1_ecg/process_ecg.py --data-dir WildPPG/WildPPG/data --all

Output: member1_ecg/results/<rule_version>/<participant>/ecg_windows.csv (+ summary.json),
keyed by participant_id/window_id/start_sample/end_sample_exclusive exactly as member2's windows.csv.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
PPG_RESULTS = ROOT / 'member2_ppg' / 'results' / 'batch_v1'


@dataclass
class EcgConfig:
    chunk_s: float = 300.0          # R-peak detection runs on 5-min chunks with overlap
    overlap_s: float = 5.0
    min_rr_s: float = 60 / 220      # physiologic RR bounds
    max_rr_s: float = 60 / 30
    quotient_tol: float = 0.25      # RR_i / RR_{i-1} must be within [1-tol, 1/(1-tol)]
    min_good_rr: int = 4            # in an 8 s window
    min_good_rr_fraction: float = 0.7
    max_rr_cv: float = 0.25
    neighbor_halfwidth: int = 15    # windows (+/-30 s) for the neighbour-consistency check
    max_neighbor_dev_bpm: float = 25.0
    rule_version: str = 'ecg_ref_v1'


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def find_input(data_dir, participant):
    """Full WildPPG .mat, or an ECG-only extract <pid>.npz made by extract_ecg.py."""
    for name in (f'WildPPG_Part_{participant}.mat', f'{participant}.npz'):
        if (Path(data_dir) / name).exists():
            return Path(data_dir) / name
    return None


def read_ecg(path):
    if str(path).endswith('.npz'):
        z = np.load(path)
        return str(z['id']), str(z['notes']), z['ecg'].astype(float), float(z['fs'])
    d = loadmat(path, variable_names=['id', 'notes', 'sternum'], simplify_cells=True)
    ecg = d['sternum']['ecg']
    notes = d.get('notes', '')
    notes = notes if isinstance(notes, str) else ('' if np.size(notes) == 0 else str(notes))
    return str(d['id']), notes, np.asarray(ecg['v'], float).reshape(-1), float(ecg['fs'])


def detect_rpeaks(ecg, fs, cfg):
    """R-peaks (sample indices) via NeuroKit2, run per finite segment in overlapping chunks."""
    import neurokit2 as nk
    peaks = []
    chunk, ov = int(cfg.chunk_s * fs), int(cfg.overlap_s * fs)
    for a, b in runs(np.isfinite(ecg)):
        if b - a < 10 * fs:
            continue
        s = a
        while s < b:
            e = min(b, s + chunk + ov)
            seg = ecg[s:e]
            try:
                clean = nk.ecg_clean(seg, sampling_rate=fs, method='neurokit')
                _, info = nk.ecg_peaks(clean, sampling_rate=fs, method='neurokit')
                p = np.asarray(info['ECG_R_Peaks'], int) + s
            except Exception:  # flat or pathological chunk
                p = np.array([], int)
            # keep peaks in [s, s+chunk) except the final chunk; overlap avoids edge misses
            lo = s if s == a else s + ov // 2
            hi = e if e == b else s + chunk + ov // 2
            peaks.append(p[(p >= lo) & (p < hi)])
            s += chunk
    peaks = np.unique(np.concatenate(peaks)) if peaks else np.array([], int)
    # drop near-duplicates from chunk seams
    if len(peaks) > 1:
        keep = np.r_[True, np.diff(peaks) > cfg.min_rr_s * fs * 0.5]
        peaks = peaks[keep]
    return peaks


def clean_rr(peaks, fs, cfg):
    """Return RR intervals (s), their end-peak sample, and a good mask (bounds + quotient filter)."""
    rr = np.diff(peaks) / fs
    good = (rr >= cfg.min_rr_s) & (rr <= cfg.max_rr_s)
    q = np.r_[1.0, rr[1:] / rr[:-1]]
    lo, hi = 1 - cfg.quotient_tol, 1 / (1 - cfg.quotient_tol)
    good &= (q >= lo) & (q <= hi)
    # an interval following a rejected one is checked against the previous good one only via bounds
    return rr, peaks[:-1], peaks[1:], good


def window_hr(grid, ecg, fs, peaks, cfg):
    rr, start_pk, end_pk, good = clean_rr(peaks, fs, cfg)
    finite = np.isfinite(ecg)
    cfin = np.r_[0, np.cumsum(finite)]
    rows = []
    for w in grid.itertuples(index=False):
        a, b = int(w.start_sample), int(w.end_sample_exclusive)
        b_c = min(b, len(ecg))
        finite_ratio = (cfin[b_c] - cfin[min(a, b_c)]) / (b - a) if b > a else 0.0
        i0, i1 = np.searchsorted(start_pk, a, 'left'), np.searchsorted(end_pk, b, 'right')
        sel = slice(i0, max(i0, i1))
        r, g = rr[sel], good[sel]
        n_rr, n_good = len(r), int(g.sum())
        hr = 60 / r[g].mean() if n_good else np.nan
        cv = r[g].std() / r[g].mean() if n_good > 1 else np.nan
        reasons = []
        if finite_ratio < 1:
            reasons.append('ecg_missing_samples')
        if n_good < cfg.min_good_rr:
            reasons.append('too_few_beats')
        if n_rr and n_good / n_rr < cfg.min_good_rr_fraction:
            reasons.append('irregular_rr')
        if np.isfinite(cv) and cv > cfg.max_rr_cv:
            reasons.append('high_rr_variability')
        rows.append(dict(participant_id=w.participant_id, window_id=w.window_id,
                         start_sample=a, end_sample_exclusive=b,
                         ecg_finite_ratio=finite_ratio, n_rr=n_rr, n_good_rr=n_good,
                         rr_cv=cv, hr_ecg_raw=hr, reasons=reasons))
    t = pd.DataFrame(rows)
    pre = t.reasons.str.len().eq(0) & t.hr_ecg_raw.notna()
    # neighbour consistency: compare with rolling median of otherwise-valid windows
    med = t.hr_ecg_raw.where(pre).rolling(2 * cfg.neighbor_halfwidth + 1, center=True, min_periods=5).median()
    inconsistent = pre & med.notna() & ((t.hr_ecg_raw - med).abs() > cfg.max_neighbor_dev_bpm)
    for i in np.flatnonzero(inconsistent):
        t.at[i, 'reasons'] = t.at[i, 'reasons'] + ['inconsistent_with_neighbors']
    t['ecg_valid'] = t.reasons.str.len().eq(0) & t.hr_ecg_raw.notna()
    t['hr_ecg'] = t.hr_ecg_raw.where(t.ecg_valid)
    t['ecg_reject_reasons'] = t.reasons.str.join(';')
    t['ecg_rule_version'] = cfg.rule_version
    return t.drop(columns='reasons')


def process(participant, data_dir, out_root, cfg, ppg_results=PPG_RESULTS):
    grid = pd.read_csv(ppg_results / participant / 'windows.csv',
                       usecols=['participant_id', 'window_id', 'start_sample', 'end_sample_exclusive', 'ppg_valid', 'hr_ppg'])
    pid, notes, ecg, fs = read_ecg(find_input(data_dir, participant))
    peaks = detect_rpeaks(ecg, fs, cfg)
    t = window_hr(grid, ecg, fs, peaks, cfg)
    out = Path(out_root) / cfg.rule_version / participant
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'ecg_windows.csv', index=False, float_format='%.9g')
    both = t.ecg_valid.values & grid.ppg_valid.values.astype(bool)
    rej = ~grid.ppg_valid.values.astype(bool)
    err = (grid.hr_ppg.values - t.hr_ecg.values)[both]
    summary = dict(participant=participant, fs_ecg=fs, n_samples_ecg=int(len(ecg)), n_rpeaks=int(len(peaks)),
                   n_windows=int(len(t)), ecg_valid_fraction=float(t.ecg_valid.mean()),
                   ecg_valid_fraction_in_ppg_rejected=float(t.ecg_valid.values[rej].mean()) if rej.any() else None,
                   ecg_valid_fraction_in_ppg_accepted=float(t.ecg_valid.values[~rej].mean()) if (~rej).any() else None,
                   ppg_vs_ecg_mae_accepted=float(np.abs(err).mean()) if len(err) else None,
                   ppg_vs_ecg_within_10bpm=float((np.abs(err) <= 10).mean()) if len(err) else None,
                   notes=notes, config=asdict(cfg))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return t, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(ROOT / 'WildPPG' / 'WildPPG' / 'data'))
    ap.add_argument('--participant')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out-dir', default=str(ROOT / 'member1_ecg' / 'results'))
    args = ap.parse_args()
    cfg = EcgConfig()
    parts = sorted(p.name for p in PPG_RESULTS.iterdir() if (p / 'windows.csv').exists()) if args.all else [args.participant]
    rows = []
    for p in parts:
        if find_input(args.data_dir, p) is None:
            print(f'skip {p}: raw file not found'); continue
        _, s = process(p, args.data_dir, args.out_dir, cfg)
        rows.append({k: v for k, v in s.items() if k != 'config'})
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ('config', 'notes')})
    if rows:
        pd.DataFrame(rows).to_csv(Path(args.out_dir) / cfg.rule_version / 'coverage_audit.csv', index=False)


if __name__ == '__main__':
    main()
