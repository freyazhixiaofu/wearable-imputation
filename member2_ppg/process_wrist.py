"""Offline wrist PPG development baseline. No ECG enters HR/SQI decisions.

Run: python process_wrist.py --data-dir PATH --participant an0 --out-dir results/an0
Dependencies: numpy, scipy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt, welch


@dataclass
class Config:
    window_s: float = 8.0
    stride_s: float = 2.0
    channel: str = 'ppg_g'
    band_low_hz: float = 0.5
    band_high_hz: float = 4.0
    hr_min_bpm: float = 40.0
    hr_max_bpm: float = 200.0
    search_boundary_margin_bpm: float = 2.0
    spectral_halfwidth_hz: float = 0.15
    min_spectral_concentration: float = 0.40
    min_acf_peak: float = 0.40
    max_hr_difference_bpm: float = 12.0
    max_flat_run_s: float = 0.5
    filter_guard_s: float = 2.0
    acc_gravity_cutoff_hz: float = 0.25
    rule_version: str = 'ppg_only_dev_v1'


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def max_flat_run(x):
    if len(x) < 2:
        return len(x)
    equal = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.diff(x) == 0)
    return max((b - a + 1 for a, b in runs(equal)), default=1)


def filter_segments(x, good, sos, fs, guard_s):
    """Never filter across missing/excluded samples; retain an edge guard mask."""
    y = np.full(len(x), np.nan)
    safe = np.zeros(len(x), bool)
    guard = int(round(guard_s * fs))
    for a, b in runs(good):
        if b - a <= max(32, 2 * guard):
            continue
        try:
            y[a:b] = sosfiltfilt(sos, x[a:b])
        except ValueError:
            continue
        safe[a + guard:b - guard] = True
    return y, safe


def ppg_metrics(y, fs, cfg):
    result = dict(hr_spectral_bpm=np.nan, hr_acf_bpm=np.nan,
                  hr_ppg_candidate=np.nan, spectral_concentration=np.nan,
                  acf_peak=np.nan, hr_method_difference_bpm=np.nan,
                  spectral_entropy=np.nan)
    if not np.isfinite(y).all() or np.ptp(y) <= 0:
        return result
    y = y - np.mean(y)
    energy = np.dot(y, y)
    if energy <= 0:
        return result
    f, p = welch(y, fs=fs, window='hann', nperseg=len(y),
                 nfft=max(4096, 2 ** int(np.ceil(np.log2(len(y))))), detrend='constant')
    band = (f >= cfg.hr_min_bpm / 60) & (f <= cfg.hr_max_bpm / 60)
    fb, pb = f[band], p[band]
    if not len(pb) or pb.sum() <= 0:
        return result
    peak_f = fb[np.argmax(pb)]
    concentration = pb[np.abs(fb - peak_f) <= cfg.spectral_halfwidth_hz].sum() / pb.sum()
    distribution = pb / pb.sum()
    entropy = -np.sum(distribution * np.log(distribution + 1e-15)) / np.log(len(pb))
    ac = correlate(y, y, mode='full', method='fft')[len(y) - 1:] / energy
    lo = int(np.ceil(fs * 60 / cfg.hr_max_bpm))
    hi = min(len(ac) - 1, int(np.floor(fs * 60 / cfg.hr_min_bpm)))
    # Only interior local maxima, not an arbitrary value on a monotonic slope.
    peaks, _ = find_peaks(ac)
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    hr_acf, acf_peak = np.nan, np.nan
    if len(peaks):
        lag = peaks[np.argmax(ac[peaks])]
        hr_acf, acf_peak = 60 * fs / lag, ac[lag]
    hr_spectral = 60 * peak_f
    result.update(hr_spectral_bpm=hr_spectral, hr_acf_bpm=hr_acf,
                  hr_ppg_candidate=hr_spectral, spectral_concentration=concentration,
                  acf_peak=acf_peak, spectral_entropy=entropy,
                  hr_method_difference_bpm=abs(hr_spectral - hr_acf))
    return result


def read_participant(path):
    d = loadmat(path, variable_names=['id', 'notes', 'wrist'], simplify_cells=True)
    w = d['wrist']
    data, rates, audit = {}, {}, []
    for name, item in w.items():
        if not isinstance(item, dict) or 'v' not in item:
            continue
        x = np.asarray(item['v'], dtype=float).reshape(-1)
        fs = float(item['fs'])
        data[name], rates[name] = x, fs
        finite = x[np.isfinite(x)]
        audit.append(dict(channel=name, fs_hz=fs, n_samples=len(x),
                          duration_s=len(x) / fs, n_nonfinite=int((~np.isfinite(x)).sum()),
                          minimum=float(finite.min()) if len(finite) else None,
                          maximum=float(finite.max()) if len(finite) else None,
                          description=str(item.get('descr', ''))))
    notes = d.get('notes', '')
    notes = notes if isinstance(notes, str) else ('' if np.size(notes) == 0 else str(notes))
    return str(d['id']), notes, data, rates, audit


def exclusions_from_notes(notes, fs, n, policy):
    """Parse only the explicitly named wrist exclusion sentence, not arbitrary notes."""
    import re
    exclusions = []
    warnings = []
    if notes.strip():
        warnings.append('Participant notes must be reviewed; automatic parsing is limited to explicit wrist index exclusions.')
    match = re.search(r'Index\s+(\d+)\s+to\s+(\d+)\s+of all data from the wrist should be excluded', notes, re.I)
    if match:
        first, last = map(int, match.groups())
        if not 1 <= first <= last <= n:
            raise ValueError('Documented wrist exclusion falls outside the wrist record.')
        ambiguous = 'ankle' in notes.lower()
        if ambiguous:
            warnings.append('Notes name wrist indices but also mention ankle; channel attribution is unresolved. Default conservatively masks the explicit wrist range; compare --notes-policy ignore as sensitivity analysis.')
        exclusions.append(dict(start_sample=first - 1, end_sample_exclusive=last,
                               start_s=(first - 1) / fs, end_s=last / fs,
                               source='participant notes: explicit MATLAB wrist indices',
                               applied=policy == 'mask', attribution_verified=not ambiguous))
    return exclusions, warnings


def build_windows(participant, data, rates, notes, cfg, notes_policy):
    required = [cfg.channel, 'acc_x', 'acc_y', 'acc_z']
    for name in required:
        if name not in data:
            raise ValueError(f'Missing required channel: {name}')
    fs = rates[cfg.channel]
    if fs <= 2 * cfg.band_high_hz:
        raise ValueError('PPG sampling frequency is too low for the configured filter.')
    if any(rates[k] != fs for k in required):
        raise ValueError('PPG/ACC sampling rates differ; explicit time alignment is required.')
    n = len(data[cfg.channel])
    win, step = int(round(cfg.window_s * fs)), int(round(cfg.stride_s * fs))
    if cfg.window_s <= 0 or cfg.stride_s <= 0 or win < 32 or step < 1:
        raise ValueError('Window/stride must be positive and the window must contain at least 32 samples.')
    if not np.isclose(win / fs, cfg.window_s) or not np.isclose(step / fs, cfg.stride_s):
        raise ValueError('Window/stride must be integer numbers of samples.')
    if n < win:
        raise ValueError('Recording is shorter than one window.')
    x = data[cfg.channel]
    excluded = np.zeros(n, bool)
    exclusions, warnings = exclusions_from_notes(notes, fs, n, notes_policy)
    for item in exclusions:
        if item['applied']:
            excluded[item['start_sample']:item['end_sample_exclusive']] = True
    sos = butter(4, [cfg.band_low_hz, cfg.band_high_hz], fs=fs, btype='bandpass', output='sos')
    filtered, filter_safe = filter_segments(x, np.isfinite(x) & ~excluded, sos, fs, cfg.filter_guard_s)
    # Pad shorter ACC channels rather than truncate the PPG timeline or fabricate samples.
    acc = np.full((n, 3), np.nan)
    for j, axis in enumerate('xyz'):
        arr = data[f'acc_{axis}']
        acc[:min(n, len(arr)), j] = arr[:n]
        if len(arr) != n:
            warnings.append(f'acc_{axis} length differs from PPG: {len(arr)} versus {n}; no start-offset correction inferred.')
    acc_good = np.isfinite(acc).all(axis=1) & ~excluded
    gravity_sos = butter(2, cfg.acc_gravity_cutoff_hz, fs=fs, btype='lowpass', output='sos')
    dynamic = np.full_like(acc, np.nan)
    acc_safe = np.ones(n, bool)
    for j in range(3):
        gravity, safe = filter_segments(acc[:, j], acc_good, gravity_sos, fs, cfg.filter_guard_s)
        dynamic[:, j] = acc[:, j] - gravity
        acc_safe &= safe
    rows = []
    for k, a in enumerate(range(0, n - win + 1, step)):
        b = a + win
        raw, y = x[a:b], filtered[a:b]
        finite = np.isfinite(raw)
        finite_ratio = float(finite.mean())
        flat_s = max_flat_run(raw) / fs
        has_exclusion = bool(excluded[a:b].any())
        safe = bool(filter_safe[a:b].all())
        metrics = ppg_metrics(y, fs, cfg)
        reasons = []
        if finite_ratio < 1:
            reasons.append('nonfinite_ppg')
        if has_exclusion:
            reasons.append('documented_exclusion')
        if not safe:
            reasons.append('filter_edge_or_unavailable')
        if flat_s >= cfg.max_flat_run_s:
            reasons.append('flatline_or_plateau')
        if not np.isfinite(metrics['hr_ppg_candidate']):
            reasons.append('hr_unavailable')
        elif (metrics['hr_ppg_candidate'] <= cfg.hr_min_bpm + cfg.search_boundary_margin_bpm
              or metrics['hr_ppg_candidate'] >= cfg.hr_max_bpm - cfg.search_boundary_margin_bpm):
            reasons.append('hr_search_boundary')
        if np.isfinite(metrics['spectral_concentration']) and metrics['spectral_concentration'] < cfg.min_spectral_concentration:
            reasons.append('low_spectral_concentration')
        if not np.isfinite(metrics['acf_peak']):
            reasons.append('acf_unavailable')
        elif metrics['acf_peak'] < cfg.min_acf_peak:
            reasons.append('weak_periodicity')
        if np.isfinite(metrics['hr_method_difference_bpm']) and metrics['hr_method_difference_bpm'] > cfg.max_hr_difference_bpm:
            reasons.append('hr_method_disagreement')
        valid = not reasons
        acc_valid = bool(acc_good[a:b].all() and acc_safe[a:b].all())
        features = dict(acc_magnitude_mean=np.nan, acc_magnitude_std=np.nan,
                        acc_dynamic_rms=np.nan, acc_jerk_rms=np.nan,
                        acc_x_std=np.nan, acc_y_std=np.nan, acc_z_std=np.nan)
        if acc_valid:
            segment = acc[a:b]
            magnitude = np.linalg.norm(segment, axis=1)
            features.update(acc_magnitude_mean=float(magnitude.mean()),
                            acc_magnitude_std=float(magnitude.std()),
                            acc_dynamic_rms=float(np.sqrt(np.mean(np.sum(dynamic[a:b] ** 2, axis=1)))),
                            acc_jerk_rms=float(np.sqrt(np.mean(np.sum((np.diff(segment, axis=0) * fs) ** 2, axis=1)))))
            features.update({f'acc_{axis}_std': float(segment[:, j].std()) for j, axis in enumerate('xyz')})
        # This is a conjunction of interpretable SQI checks, not a calibrated probability.
        components = [metrics['spectral_concentration'], metrics['acf_peak']]
        sqi_score = float(np.clip(min(components), 0, 1)) if np.isfinite(components).all() else np.nan
        rows.append(dict(participant_id=participant, window_id=k,
                         start_sample=a, end_sample_exclusive=b,
                         start_s=a / fs, end_s=b / fs, center_s=(a + b) / (2 * fs),
                         ppg_channel=cfg.channel, finite_ratio=finite_ratio,
                         longest_flat_run_s=flat_s, documented_exclusion=has_exclusion,
                         filter_safe=safe, sqi_score=sqi_score,
                         ppg_valid=valid, reject_reasons=';'.join(reasons),
                         hr_ppg=metrics['hr_ppg_candidate'] if valid else np.nan,
                         acc_valid=acc_valid, acc_finite_ratio=float(np.isfinite(acc[a:b]).all(axis=1).mean()),
                         rule_version=cfg.rule_version, **metrics, **features))
        if (k + 1) % 5000 == 0:
            print(f'Processed {k + 1} windows', flush=True)
    table = pd.DataFrame(rows)
    return table, filtered, exclusions, warnings


def gap_table(table, cfg):
    rows = []
    for gap_id, (a, b) in enumerate(runs(~table.ppg_valid.to_numpy())):
        group = table.iloc[a:b]
        reasons = sorted({r for value in group.reject_reasons for r in value.split(';') if r})
        rows.append(dict(participant_id=group.participant_id.iloc[0], gap_id=gap_id,
                         first_window_id=int(group.window_id.iloc[0]),
                         last_window_id=int(group.window_id.iloc[-1]), n_windows=b - a,
                         grid_start_center_s=float(group.center_s.iloc[0]),
                         grid_end_exclusive_s=float(group.center_s.iloc[-1] + cfg.stride_s),
                         grid_duration_s=(b - a) * cfg.stride_s,
                         raw_support_start_s=float(group.start_s.iloc[0]),
                         raw_support_end_s=float(group.end_s.iloc[-1]),
                         raw_support_duration_s=float(group.end_s.iloc[-1] - group.start_s.iloc[0]),
                         touches_record_start=a == 0, touches_record_end=b == len(table),
                         reasons=';'.join(reasons)))
    columns = ['participant_id', 'gap_id', 'first_window_id', 'last_window_id', 'n_windows',
               'grid_start_center_s', 'grid_end_exclusive_s', 'grid_duration_s',
               'raw_support_start_s', 'raw_support_end_s', 'raw_support_duration_s',
               'touches_record_start', 'touches_record_end', 'reasons']
    return pd.DataFrame(rows, columns=columns)


def make_plots(table, gaps, raw, filtered, fs, cfg, out):
    # This directory contains generated QA windows; remove prior generated PNGs
    # so a repeated run cannot leave stale accepted/rejected examples behind.
    for previous in (out / 'qa_windows').glob('window_*.png'):
        previous.unlink()
    hours = table.center_s / 3600
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].scatter(hours, table.hr_ppg_candidate, s=1, color='0.7', label='Candidate (all available)')
    axes[0].scatter(hours, table.hr_ppg, s=1, color='tab:green', label='Accepted')
    axes[0].set_ylabel('PPG HR (bpm)'); axes[0].legend(loc='upper right', markerscale=4)
    axes[1].plot(hours, table.spectral_concentration, lw=.4, label='Spectral concentration')
    axes[1].plot(hours, table.acf_peak, lw=.4, alpha=.6, label='ACF peak')
    axes[1].axhline(cfg.min_spectral_concentration, color='tab:blue', ls='--', lw=.8)
    axes[1].axhline(cfg.min_acf_peak, color='tab:orange', ls=':', lw=.8)
    axes[1].set_ylabel('SQI components'); axes[1].legend(loc='upper right')
    axes[2].plot(hours, table.acc_dynamic_rms, lw=.5)
    axes[2].set_ylabel('Dynamic ACC RMS\n(stored units)')
    axes[3].step(hours, table.ppg_valid.astype(int), where='mid', lw=.6)
    axes[3].set_yticks([0, 1], ['Rejected', 'Accepted']); axes[3].set_xlabel('Hours since record start')
    fig.suptitle(f'{table.participant_id.iloc[0]} | {cfg.rule_version} | development baseline; no ECG validation')
    fig.tight_layout(); fig.savefig(out / 'timeline.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    reasons = table.loc[~table.ppg_valid, 'reject_reasons'].str.split(';').explode().value_counts()
    axes[0, 0].barh(reasons.index[::-1], reasons.values[::-1])
    axes[0, 0].set_title('Rejection reasons (multiple per window)'); axes[0, 0].set_xlabel('Windows')
    if len(gaps):
        axes[0, 1].hist(gaps.grid_duration_s, bins=[0, 2.01, 6, 10, 30, 60, 120, 300, max(301, gaps.grid_duration_s.max() + 1)])
        axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Rejected HR-grid runs'); axes[0, 1].set_xlabel('Grid gap duration (s)')
    values = [table.loc[table.ppg_valid & table.acc_valid, 'acc_dynamic_rms'].to_numpy(),
              table.loc[~table.ppg_valid & table.acc_valid, 'acc_dynamic_rms'].to_numpy()]
    nonempty = [(label, v) for label, v in zip(['Accepted', 'Rejected'], values) if len(v)]
    if nonempty:
        axes[1, 0].boxplot([v for _, v in nonempty], tick_labels=[label for label, _ in nonempty], showfliers=False)
    axes[1, 0].set_title('Movement by PPG decision'); axes[1, 0].set_ylabel('Dynamic ACC RMS (stored units)')
    available = table.dropna(subset=['spectral_concentration', 'acf_peak'])
    if len(available) > 5000:
        available = available.sample(5000, random_state=0)
    axes[1, 1].scatter(available.spectral_concentration, available.acf_peak,
                       c=np.where(available.ppg_valid, 'tab:green', 'tab:red'), s=3, alpha=.3)
    axes[1, 1].axvline(cfg.min_spectral_concentration, ls='--', color='black')
    axes[1, 1].axhline(cfg.min_acf_peak, ls='--', color='black')
    axes[1, 1].set_xlabel('Spectral concentration'); axes[1, 1].set_ylabel('ACF peak')
    axes[1, 1].set_title('PPG SQI (green accepted / red rejected)')
    fig.tight_layout(); fig.savefig(out / 'quality_summary.png', dpi=150); plt.close(fig)

    # Deterministic sampling for reproducible QA, not cherry-picked waveforms.
    selected = []
    usable = table[table.acc_valid]
    if len(usable):
        q1, q2 = usable.acc_dynamic_rms.quantile([1/3, 2/3])
        for decision, accepted in [('accepted', True), ('rejected', False)]:
            for movement, motion in [('low_motion', table.acc_dynamic_rms <= q1), ('high_motion', table.acc_dynamic_rms >= q2)]:
                candidates = table[table.ppg_valid.eq(accepted) & table.acc_valid & motion]
                if len(candidates):
                    for idx in candidates.sample(min(2, len(candidates)), random_state=0).index:
                        selected.append((int(idx), f'{decision}_{movement}'))
    for reason in ['documented_exclusion', 'flatline_or_plateau', 'hr_method_disagreement']:
        candidates = table[table.reject_reasons.str.contains(reason)]
        if len(candidates):
            selected.append((int(candidates.index[len(candidates)//2]), reason))
    chosen = []
    seen = set()
    for idx, stratum in selected:
        if idx in seen:
            continue
        seen.add(idx)
        row = table.loc[idx]
        a, b = int(row.start_sample), int(row.end_sample_exclusive)
        t = np.arange(b - a) / fs
        y = filtered[a:b]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8))
        axes[0].plot(t, raw[a:b], lw=.8); axes[0].set_ylabel('Raw PPG'); axes[0].set_xlabel('Seconds in window')
        axes[1].plot(t, y, lw=.8); axes[1].set_ylabel('Filtered PPG'); axes[1].set_xlabel('Seconds in window')
        axes[1].set_xlim(0, cfg.window_s)
        if not np.isfinite(y).any():
            axes[1].text(.1, .5, 'Filtered signal unavailable: missing / excluded interval', transform=axes[1].transAxes)
        if np.isfinite(y).all() and np.ptp(y) > 0:
            f, p = welch(y, fs=fs, nperseg=len(y), nfft=4096)
            axes[2].plot(f * 60, p, lw=1)
            axes[2].axvline(row.hr_spectral_bpm, color='tab:red', ls='--', label='Spectral HR')
            if np.isfinite(row.hr_acf_bpm):
                axes[2].axvline(row.hr_acf_bpm, color='tab:orange', ls=':', label='ACF HR')
            axes[2].legend()
        else:
            axes[2].text(.1, .5, 'Spectrum unavailable: missing / excluded / constant signal', transform=axes[2].transAxes)
        axes[2].set_xlim(0, 260); axes[2].set_xlabel('Frequency expressed as bpm'); axes[2].set_ylabel('PSD')
        title = (f'Window {idx} | {stratum} | start={row.start_s:.1f}s\n'
                 f'candidate={row.hr_ppg_candidate:.1f}, ACF={row.hr_acf_bpm:.1f} bpm; '
                 f'SC={row.spectral_concentration:.2f}, ACF peak={row.acf_peak:.2f}\n'
                 f'reasons: {row.reject_reasons or "accepted"}')
        fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, .90])
        filename = f'window_{idx:05d}_{stratum}.png'
        fig.savefig(out / 'qa_windows' / filename, dpi=140); plt.close(fig)
        chosen.append(dict(window_id=idx, stratum=stratum, file=filename))
    pd.DataFrame(chosen).to_csv(out / 'qa_windows' / 'selection.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--participant', default='an0')
    parser.add_argument('--out-dir', type=Path, default=Path('results/an0'))
    parser.add_argument('--config', type=Path, help='JSON object overriding Config fields')
    parser.add_argument('--notes-policy', choices=['mask', 'ignore'], default='mask')
    args = parser.parse_args()
    cfg = Config(**(json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True); (out / 'qa_windows').mkdir(exist_ok=True)
    path = args.data_dir / f'WildPPG_Part_{args.participant}.mat'
    participant, notes, data, rates, audit = read_participant(path)
    if participant != args.participant:
        raise ValueError('Filename participant differs from participant ID stored in file.')
    print(f'Loaded {participant}: {len(data[cfg.channel])} PPG samples', flush=True)
    table, filtered, exclusions, warnings = build_windows(participant, data, rates, notes, cfg, args.notes_policy)
    gaps = gap_table(table, cfg)
    table.to_csv(out / 'windows.csv', index=False, float_format='%.9g')
    gaps.to_csv(out / 'gaps.csv', index=False, float_format='%.9g')
    pd.DataFrame(audit).to_csv(out / 'channel_audit.csv', index=False)
    reasons = table.loc[~table.ppg_valid, 'reject_reasons'].str.split(';').explode().value_counts()
    reasons.rename_axis('reason').reset_index(name='n_windows').to_csv(out / 'rejection_counts.csv', index=False)
    summary = dict(participant_id=participant, n_windows=len(table),
                   n_accepted=int(table.ppg_valid.sum()), n_rejected=int((~table.ppg_valid).sum()),
                   coverage=float(table.ppg_valid.mean()), n_gaps=len(gaps),
                   gap_grid_total_s=float(gaps.grid_duration_s.sum()),
                   gap_grid_median_s=float(gaps.grid_duration_s.median()) if len(gaps) else None,
                   gap_grid_max_s=float(gaps.grid_duration_s.max()) if len(gaps) else None,
                   n_acc_valid=int(table.acc_valid.sum()),
                   n_documented_exclusion_windows=int(table.documented_exclusion.sum()),
                   trailing_samples_without_complete_window=int(len(data[cfg.channel]) - table.end_sample_exclusive.iloc[-1]))
    manifest = dict(input_file=str(path.resolve()), input_size_bytes=path.stat().st_size,
                    participant_notes=notes, notes_policy=args.notes_policy, exclusions=exclusions,
                    warnings=warnings, config=asdict(cfg), summary=summary,
                    alignment='Relative sample-zero alignment assumed from dataset synchronization; no empirical offset correction.',
                    acc_units='Native stored units; no conversion to g or m/s^2 inferred.',
                    gap_definition='Consecutive rejected HR-grid windows; grid duration=n_windows*stride. Raw support reported separately.',
                    status='Development baseline; SQI thresholds unvalidated against ECG.',
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    make_plots(table, gaps, data[cfg.channel], filtered, rates[cfg.channel], cfg, out)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Results: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
