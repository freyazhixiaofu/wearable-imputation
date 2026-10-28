"""Read-only audit of batch_v1: reconstruct filtering, show actual v1 ACF, preserve all baseline files."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, welch

from process_wrist import Config, filter_segments, ppg_metrics, read_participant


SEED = 2631


def fingerprints(baseline, code_root):
    paths = sorted(p for p in baseline.rglob('*') if p.is_file())
    paths += [code_root / 'process_wrist.py', code_root / 'process_all.py']
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def acf_details(y, fs, cfg):
    if not np.isfinite(y).all() or np.ptp(y) <= 0:
        return np.array([]), np.array([], dtype=int), None
    z = y - y.mean()
    ac = correlate(z, z, mode='full', method='fft')[len(z)-1:] / np.dot(z, z)
    lo = int(np.ceil(fs * 60 / cfg.hr_max_bpm))
    hi = min(len(ac)-1, int(np.floor(fs * 60 / cfg.hr_min_bpm)))
    peaks, _ = find_peaks(ac)
    peaks = peaks[(peaks >= lo) & (peaks <= hi)]
    chosen = int(peaks[np.argmax(ac[peaks])]) if len(peaks) else None
    return ac, peaks, chosen


def select_windows(d):
    reasons = d.reject_reasons.fillna('')
    good_acc = d.acc_valid & d.acc_dynamic_rms.notna()
    q1, q3 = d.loc[good_acc, 'acc_dynamic_rms'].quantile([.25, .75])
    acf_only = reasons.eq('weak_periodicity')
    categories = [
        ('accepted_low_motion', d.ppg_valid & good_acc & d.acc_dynamic_rms.le(q1), 1),
        ('accepted_high_motion', d.ppg_valid & good_acc & d.acc_dynamic_rms.ge(q3), 1),
        ('acf_only_borderline', acf_only & d.acf_peak.ge(.35), 2),
        ('acf_only_weaker', acf_only & d.acf_peak.lt(.35), 1),
        ('boundary_only', reasons.eq('hr_search_boundary'), 1),
        ('boundary_multiple_failures', reasons.str.contains('hr_search_boundary') & reasons.str.contains(';'), 1),
        ('disagreement_only', reasons.eq('hr_method_disagreement'), 1),
        ('acf_unavailable_only', reasons.eq('acf_unavailable'), 1),
        ('mixed_periodicity_rejection', reasons.eq('low_spectral_concentration;weak_periodicity'), 1),
    ]
    chosen, availability, starts = [], [], []
    for category, mask, target in categories:
        pool = d[mask]
        candidates = pool.sample(frac=1, random_state=SEED)
        count = 0
        for idx, row in candidates.iterrows():
            if any(abs(row.start_s - t) < 30 for t in starts):
                continue
            chosen.append(dict(window_id=int(row.window_id), category=category))
            starts.append(float(row.start_s)); count += 1
            if count == target:
                break
        availability.append(dict(category=category, candidate_windows=len(pool), target=target, selected=count))
    return chosen, availability


def window_plot(row, raw, filtered, acc, fs, cfg, category, path):
    a, b = int(row.start_sample), int(row.end_sample_exclusive)
    y = filtered[a:b]
    ac, peaks, lag = acf_details(y, fs, cfg)
    metrics = ppg_metrics(y, fs, cfg)
    for field, value in metrics.items():
        stored = row[field]
        if not np.isclose(stored, value, rtol=1e-7, atol=1e-8, equal_nan=True):
            raise AssertionError(f'{row.participant_id} window {row.window_id}: {field} differs: {stored} vs {value}')
    fig, ax = plt.subplots(5, 1, figsize=(12, 14), gridspec_kw={'height_ratios':[1, 1, 1.2, 1.3, .8]})
    t = np.arange(b-a) / fs
    ax[0].plot(t, raw[a:b], lw=.8); ax[0].set_ylabel('Raw green PPG\n(stored amplitude)')
    ax[1].plot(t, y, lw=.9); ax[1].set_ylabel('Filtered PPG\n(stored amplitude)')
    ax[0].set_xlim(0, cfg.window_s); ax[1].set_xlim(0, cfg.window_s)
    ax[0].set_xlabel('Seconds in window'); ax[1].set_xlabel('Seconds in window')
    f, power = welch(y, fs=fs, window='hann', nperseg=len(y), nfft=max(4096, 2 ** int(np.ceil(np.log2(len(y))))), detrend='constant')
    ax[2].plot(f * 60, power, color='tab:blue', lw=1)
    ax[2].axvspan(cfg.hr_min_bpm, cfg.hr_max_bpm, color='tab:green', alpha=.07, label='HR search range')
    ax[2].axvspan(row.hr_spectral_bpm - cfg.spectral_halfwidth_hz * 60,
                 row.hr_spectral_bpm + cfg.spectral_halfwidth_hz * 60, color='tab:red', alpha=.12)
    ax[2].axvline(row.hr_spectral_bpm, color='tab:red', ls='--', label=f'Spectral HR {row.hr_spectral_bpm:.1f}')
    if np.isfinite(row.hr_acf_bpm):
        ax[2].axvline(row.hr_acf_bpm, color='tab:orange', ls=':', label=f'ACF HR {row.hr_acf_bpm:.1f}')
    ax[2].set_xlim(0, 260); ax[2].set_ylabel('PSD (amplitude²/Hz)'); ax[2].set_xlabel('Frequency expressed as bpm (Hz × 60)'); ax[2].legend(fontsize=8, loc='upper right')
    lags = np.arange(len(ac)) / fs
    lo, hi = np.ceil(fs * 60 / cfg.hr_max_bpm) / fs, np.floor(fs * 60 / cfg.hr_min_bpm) / fs
    ax[3].plot(lags, ac, lw=1, color='tab:blue', label='v1 normalized ACF')
    ax[3].axvspan(lo, hi, alpha=.08, color='tab:green', label='Eligible lag range')
    ax[3].axhline(cfg.min_acf_peak, ls='--', color='black', lw=.8, label='v1 threshold 0.40')
    ax[3].axhline(0, color='0.7', lw=.7)
    if len(peaks):
        ax[3].scatter(peaks / fs, ac[peaks], facecolors='none', edgecolors='tab:orange', s=45, label='Eligible local maxima')
    if lag is not None:
        ax[3].scatter([lag/fs], [ac[lag]], color='tab:red', s=60, marker='*', zorder=5, label=f'Selected peak {ac[lag]:.3f} at {lag/fs:.3f}s')
    else:
        ax[3].text(.43, .82, 'No eligible local maximum', transform=ax[3].transAxes, color='tab:red')
    spectral_lag = 60 / row.hr_spectral_bpm
    ax[3].axvline(spectral_lag, color='tab:purple', ls=':', label='Period implied by spectral HR')
    ax[3].set_xlim(0, 2.1); ax[3].set_ylim(-1.05, 1.05)
    ax[3].set_ylabel('Normalized ACF\n(unitless)'); ax[3].set_xlabel('Lag (seconds)'); ax[3].legend(fontsize=7, loc='upper right', ncol=2)
    magnitude = np.linalg.norm(acc[a:b], axis=1)
    ax[4].plot(t, magnitude, lw=.7, color='tab:gray'); ax[4].set_xlim(0, cfg.window_s)
    ax[4].set_ylabel('ACC magnitude\n(native units)'); ax[4].set_xlabel('Seconds in window')
    reasons = row.reject_reasons if isinstance(row.reject_reasons, str) else ''
    title = (f'{row.participant_id} | window {int(row.window_id)} | {category} | {row.start_s:.0f}-{row.end_s:.0f}s\n'
             f'{"ACCEPTED" if row.ppg_valid else "REJECTED"}: {reasons or "all v1 checks passed"}\n'
             f'SC={row.spectral_concentration:.3f}; ACF peak={row.acf_peak:.3f}; HR difference={row.hr_method_difference_bpm:.2f} bpm; '
             f'ACC dynamic RMS={row.acc_dynamic_rms:.3f}')
    fig.suptitle(title, fontsize=10); fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(path, dpi=125); plt.close(fig)
    ratio = row.hr_spectral_bpm / row.hr_acf_bpm if np.isfinite(row.hr_acf_bpm) else np.nan
    return dict(selected_acf_lag_s=lag/fs if lag is not None else np.nan,
                eligible_acf_peak_count=len(peaks),
                acf_at_spectral_period=float(np.interp(spectral_lag, lags, ac)),
                spectral_to_acf_hr_ratio=ratio,
                possible_factor_two_disagreement=bool(np.isfinite(ratio) and row.hr_method_difference_bpm > cfg.max_hr_difference_bpm and (1.8 <= ratio <= 2.2 or .45 <= ratio <= .55)),
                metrics_reproduced=True, plot_file=str(path.name))


def comparison_figure(selections, output):
    ids = selections.participant_id.drop_duplicates().tolist()
    for kind in ['low_motion', 'high_motion']:
        fig, ax = plt.subplots(len(ids), 2, figsize=(12, 3.0 * len(ids)), squeeze=False)
        for i, pid in enumerate(ids):
            for j, category in enumerate([f'accepted_{kind}', 'acf_only_borderline']):
                sub = selections[selections.participant_id.eq(pid) & selections.category.eq(category)]
                axis = ax[i, j]
                if not len(sub):
                    axis.text(.1, .5, 'Stratum unavailable', transform=axis.transAxes); continue
                row = sub.iloc[0]
                curve = pd.read_csv(output / 'acf_curves' / f'{pid}_{int(row.window_id):05d}.csv')
                axis.plot(curve.lag_s, curve.acf_v1, lw=1)
                axis.axhline(.4, ls='--', color='black', lw=.8)
                axis.axvspan(.3046875, 1.5, color='tab:green', alpha=.07)
                if np.isfinite(row.selected_acf_lag_s):
                    axis.scatter([row.selected_acf_lag_s], [row.acf_peak], color='tab:red', s=25)
                axis.set_xlim(0, 2.1); axis.set_ylim(-.8, 1.05)
                axis.set_title(f'{pid} ({row.acceptance_group}) | {category}\nwindow {int(row.window_id)}, ACF={row.acf_peak:.3f}, SC={row.spectral_concentration:.3f}', fontsize=9)
                axis.set_xlabel('Lag (s)'); axis.set_ylabel('v1 ACF')
        fig.suptitle('Selected accepted vs ACF-only rejected windows | threshold unchanged | no ECG ground truth', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(output / f'acf_comparison_{kind}.png', dpi=135); plt.close(fig)


def main():
    code_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=code_root / 'results' / 'batch_v1')
    parser.add_argument('--out-dir', type=Path, default=code_root / 'results' / 'targeted_audit_v1')
    args = parser.parse_args()
    baseline, out = args.baseline.resolve(), args.out_dir.resolve()
    if out == baseline or out.is_relative_to(baseline):
        raise ValueError('Audit output must be outside batch_v1')
    out.mkdir(parents=True, exist_ok=True)
    for folder in ['plots', 'acf_curves']:
        (out / folder).mkdir(exist_ok=True)
    before = fingerprints(baseline, code_root)
    summary = pd.read_csv(baseline / 'participant_summary.csv').sort_values(['coverage', 'participant_id'], ascending=[False, True])
    high = summary.head(2).participant_id.tolist()
    low = summary.tail(2).sort_values('coverage').participant_id.tolist()
    groups = [(p, 'high acceptance') for p in high] + [(p, 'low acceptance') for p in low]
    # Supplemental motion outlier, not a substitute for high/low participant selection.
    if 'p9p' in set(summary.participant_id) and 'p9p' not in high + low:
        groups.append(('p9p', 'supplemental motion flag'))
    selected_rows, availability_rows, group_rows, category_rows = [], [], [], []
    for pid, group in groups:
        print(f'Auditing {pid}: {group}', flush=True)
        manifest = json.loads((baseline / pid / 'manifest.json').read_text(encoding='utf-8'))
        cfg = Config(**manifest['config'])
        d = pd.read_csv(baseline / pid / 'windows.csv')
        d['reject_reasons'] = d.reject_reasons.fillna('')
        choices, availability = select_windows(d)
        for item in availability:
            availability_rows.append(dict(participant_id=pid, acceptance_group=group, **item))
        _, notes, data, rates, audit = read_participant(manifest['input_file'])
        fs = rates[cfg.channel]; raw = data[cfg.channel]
        good = np.isfinite(raw)
        for exclusion in manifest['exclusions']:
            if exclusion['applied']:
                good[exclusion['start_sample']:exclusion['end_sample_exclusive']] = False
        sos = butter(4, [cfg.band_low_hz, cfg.band_high_hz], fs=fs, btype='bandpass', output='sos')
        filtered, safe = filter_segments(raw, good, sos, fs, cfg.filter_guard_s)
        acc = np.column_stack([data[f'acc_{a}'] for a in 'xyz'])
        for choice in choices:
            row = d.loc[d.window_id.eq(choice['window_id'])].iloc[0]
            path = out / 'plots' / f'{pid}_{int(row.window_id):05d}_{choice["category"]}.png'
            extra = window_plot(row, raw, filtered, acc, fs, cfg, choice['category'], path)
            a, b = int(row.start_sample), int(row.end_sample_exclusive)
            ac, _, _ = acf_details(filtered[a:b], fs, cfg)
            keep = np.arange(len(ac)) / fs <= 2.1
            pd.DataFrame(dict(lag_s=np.arange(len(ac))[keep] / fs, acf_v1=ac[keep])).to_csv(out / 'acf_curves' / f'{pid}_{int(row.window_id):05d}.csv', index=False)
            selected_rows.append(dict(**row.to_dict(), category=choice['category'], acceptance_group=group, **extra))
        reasons = d.reject_reasons
        acf_only = reasons.eq('weak_periodicity')
        disagree = reasons.str.contains('hr_method_disagreement')
        ratio = d.hr_spectral_bpm / d.hr_acf_bpm
        factor_two = disagree & (ratio.between(1.8, 2.2) | ratio.between(.45, .55))
        group_rows.append(dict(participant_id=pid, acceptance_group=group, n_windows=len(d), coverage=d.ppg_valid.mean(),
                               weak_periodicity_n=int(reasons.str.contains('weak_periodicity').sum()),
                               acf_only_rejections=int(acf_only.sum()), acf_only_borderline_035_040=int((acf_only & d.acf_peak.ge(.35)).sum()),
                               acf_only_weaker_below035=int((acf_only & d.acf_peak.lt(.35)).sum()),
                               acf_unavailable_n=int(reasons.str.contains('acf_unavailable').sum()),
                               boundary_n=int(reasons.str.contains('hr_search_boundary').sum()),
                               disagreement_n=int(disagree.sum()), possible_factor_two_disagreement_n=int(factor_two.sum())))
        for accepted in [True, False]:
            sub = d[d.ppg_valid.eq(accepted)]
            category_rows.append(dict(participant_id=pid, acceptance_group=group, decision='accepted' if accepted else 'rejected',
                                      n=len(sub), median_acf=sub.acf_peak.median(), median_concentration=sub.spectral_concentration.median(),
                                      median_hr_difference=sub.hr_method_difference_bpm.median(), median_motion=sub.acc_dynamic_rms.median()))
        print(f'  {len(choices)} windows: all saved metrics reproduced', flush=True)
        del data, raw, filtered, acc, good, safe
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / 'selected_windows.csv', index=False)
    pd.DataFrame(availability_rows).to_csv(out / 'stratum_availability.csv', index=False)
    participant_audit = pd.DataFrame(group_rows)
    participant_audit.to_csv(out / 'participant_audit_summary.csv', index=False)
    pd.DataFrame(category_rows).to_csv(out / 'accepted_rejected_summary.csv', index=False)
    comparison_figure(selected, out)
    after = fingerprints(baseline, code_root)
    assert before == after, 'Baseline fingerprint changed during the audit'
    (out / 'audit_manifest.json').write_text(json.dumps(dict(seed=SEED, selection='Top two and bottom two participants by v1 acceptance, plus p9p motion flag; fixed-seed stratified samples separated by at least 30 seconds within each participant.',
        participants=groups, n_selected=len(selected), all_selected_metrics_reproduced=bool(selected.metrics_reproduced.all()),
        baseline_unchanged=True, baseline_file_sha256=before, no_ecg_used=True,
        warning='Targeted examples are not a random prevalence sample. No manual quality ground truth or HR accuracy claims.'), indent=2), encoding='utf-8')
    notes_path = code_root / 'audit_notes.html'
    review_notes = notes_path.read_text(encoding='utf-8-sig') if notes_path.exists() else ''
    cards = []
    for row in selected.itertuples():
        cards.append(f'<article><h3>{html.escape(row.participant_id)} / window {row.window_id}: {html.escape(row.category)}</h3><p>{html.escape(row.reject_reasons or "accepted: all v1 checks passed")}</p><img loading="lazy" src="plots/{html.escape(row.plot_file)}"></article>')
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Targeted PPG and ACF audit</title>'
      '<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#172331}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ccc;text-align:right}article{border-top:2px solid #ccc;margin-top:30px}nav a{display:inline-block;margin:8px}</style>'
      '<h1>Targeted PPG / ACF audit: v1 unchanged</h1>'
      '<p>High acceptance: an0 and e61. Low acceptance: k2s and qm9. Supplemental: p9p, flagged for higher median movement in accepted windows. Participant ranks use full v1 tables.</p>'
      '<p>Each example shows raw PPG, the exact reconstructed v1 filtered PPG, PSD, actual normalized ACF and raw ACC magnitude. No ECG is used.</p>'
      '<p>The ACF is the demeaned finite-window autocorrelation divided by zero-lag energy. The red star is the maximum eligible local peak, which may be negative. The green lag band matches the discrete v1 search. The purple line is the period implied by spectral HR. ACF unavailable means no local maximum in the search band, not missing raw PPG.</p>'
      '<p>The fixed zero-lag normalization makes longer-lag values depend on fewer overlapping sample pairs; there is no overlap-count correction. This is the existing v1 definition, not a new score. Selected examples do not establish whether an HR is correct.</p>'
      '<p>Sampling: fixed seed 2631, targeted strata, at least 30 seconds between selected starts per participant. Borderline means ACF 0.35–0.40 and only the ACF rule fails. Candidate counts are overlapping-window counts, not independent events. Availability and exact selections are exported.</p>'
      '<p>All selected HR/SQI values reproduced from raw signals; SHA-256 fingerprints confirm the processor, batch runner and all baseline result files were unchanged.</p>'
      '<nav><a href="selected_windows.csv">Selected windows</a><a href="stratum_availability.csv">Sampling availability</a><a href="participant_audit_summary.csv">Full-table participant audit</a><a href="accepted_rejected_summary.csv">Accepted/rejected metrics</a><a href="audit_manifest.json">Provenance</a></nav>'
      + review_notes + '<h2>Full-table patterns for audited participants</h2>' + participant_audit.round(4).to_html(index=False)
      + '<p>Possible factor-two disagreements use spectral/ACF HR ratios 1.8–2.2 or 0.45–0.55 among method-disagreement windows. This is a diagnostic pattern, not proof of a harmonic error.</p>'
      '<h2>Actual ACF: accepted low-motion vs borderline rejected</h2><img src="acf_comparison_low_motion.png">'
      '<h2>Actual ACF: accepted high-motion vs borderline rejected</h2><img src="acf_comparison_high_motion.png">'
      '<h2>Individual windows</h2>' + ''.join(cards) + '</html>')
    (out / 'report.html').write_text(report, encoding='utf-8')
    print(participant_audit.to_string(index=False), flush=True)
    print(f'Completed {len(selected)} examples; baseline fingerprints unchanged. Report: {out / "report.html"}', flush=True)


if __name__ == '__main__':
    main()
