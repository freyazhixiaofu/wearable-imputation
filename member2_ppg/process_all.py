"""Batch the unchanged PPG-only v1 pipeline, audit outputs, and summarize participants."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

from process_wrist import Config, runs


def review_notes(paths, output):
    rows = []
    for path in paths:
        d = loadmat(path, variable_names=['id', 'notes'], simplify_cells=True)
        participant = str(d['id'])
        raw = d.get('notes', '')
        notes = raw if isinstance(raw, str) else ('' if np.size(raw) == 0 else str(raw))
        match = re.search(r'Index\s+(\d+)\s+to\s+(\d+)\s+of all data from the wrist should be excluded', notes, re.I)
        first, last = (map(int, match.groups()) if match else (None, None))
        ecg_noisy = 'ecg' in notes.lower() and 'noisy' in notes.lower()
        ambiguous = bool(match and 'ankle' in notes.lower())
        if match:
            action = 'Mask explicit wrist sample range; retain original timeline.'
            if ambiguous:
                action += ' Ankle/wrist attribution conflict remains unresolved; conservative wrist mask.'
        elif ecg_noisy:
            action = 'Record ECG warning for later evaluation; no PPG exclusion.'
        elif not notes.strip():
            action = 'No participant-specific exclusion documented.'
        else:
            raise ValueError(f'Unrecognized notes require review before processing {participant}: {notes}')
        rows.append(dict(participant_id=participant, input_file=str(path.resolve()), notes=notes,
                         review_status='reviewed_with_unresolved_attribution' if ambiguous else 'reviewed',
                         ecg_noise_warning=ecg_noisy, wrist_exclusion=bool(match),
                         matlab_first_inclusive=first, matlab_last_inclusive=last,
                         python_start_sample=first - 1 if match else None,
                         python_end_sample_exclusive=last, action=action))
    df = pd.DataFrame(rows)
    if df.participant_id.duplicated().any():
        raise ValueError('Duplicate participant IDs')
    df.to_csv(output / 'notes_review.csv', index=False)
    (output / 'notes_review.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Notes reviewed before batch processing:', flush=True)
    for row in rows:
        print(f"  {row['participant_id']}: {row['action']}", flush=True)
    return df


def check_result(participant, directory):
    d = pd.read_csv(directory / 'windows.csv')
    g = pd.read_csv(directory / 'gaps.csv')
    m = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    c = m['config']
    audit = pd.read_csv(directory / 'channel_audit.csv')
    sensor = audit[audit.channel.eq(c['channel'])].iloc[0]
    fs, n = float(sensor.fs_hz), int(sensor.n_samples)
    win, step = round(c['window_s'] * fs), round(c['stride_s'] * fs)
    expected = np.arange(0, n - win + 1, step)
    assert np.array_equal(d.window_id, np.arange(len(d))), participant
    assert np.array_equal(d.start_sample, expected), participant
    assert np.all(d.end_sample_exclusive - d.start_sample == win), participant
    assert d.participant_id.eq(participant).all(), participant
    assert not d[['participant_id', 'window_id']].duplicated().any(), participant
    assert np.array_equal(d.ppg_valid, d.hr_ppg.notna()), participant
    assert d.loc[~d.acc_valid, 'acc_dynamic_rms'].isna().all(), participant
    assert (~d.ppg_valid).sum() == g.n_windows.sum(), participant
    expected_runs = runs(~d.ppg_valid.to_numpy())
    assert len(g) == len(expected_runs), participant
    for row, (a, b) in zip(g.itertuples(), expected_runs):
        assert row.first_window_id == a and row.last_window_id == b - 1, participant
    assert np.allclose(g.grid_duration_s, g.n_windows * c['stride_s']), participant
    assert np.allclose(g.raw_support_duration_s, c['window_s'] + (g.n_windows - 1) * c['stride_s']), participant
    assert d.loc[d.ppg_valid, 'hr_ppg'].between(c['hr_min_bpm'] + c['search_boundary_margin_bpm'], c['hr_max_bpm'] - c['search_boundary_margin_bpm'], inclusive='neither').all(), participant
    for e in m['exclusions']:
        overlaps = (d.start_sample < e['end_sample_exclusive']) & (d.end_sample_exclusive > e['start_sample'])
        if e['applied']:
            assert d.loc[overlaps, 'documented_exclusion'].all(), participant
            assert not d.loc[overlaps, 'ppg_valid'].any(), participant
    expected_valid = (d.finite_ratio.eq(1) & ~d.documented_exclusion & d.filter_safe
                      & d.longest_flat_run_s.lt(c['max_flat_run_s'])
                      & d.hr_ppg_candidate.gt(c['hr_min_bpm'] + c['search_boundary_margin_bpm'])
                      & d.hr_ppg_candidate.lt(c['hr_max_bpm'] - c['search_boundary_margin_bpm'])
                      & d.spectral_concentration.ge(c['min_spectral_concentration'])
                      & d.acf_peak.ge(c['min_acf_peak'])
                      & d.hr_method_difference_bpm.le(c['max_hr_difference_bpm']))
    assert np.array_equal(d.ppg_valid, expected_valid), participant
    reasons = d.reject_reasons.fillna('')
    assert reasons[d.ppg_valid].eq('').all() and reasons[~d.ppg_valid].ne('').all(), participant
    assert (directory / 'timeline.png').is_file() and (directory / 'quality_summary.png').is_file(), participant
    selections = pd.read_csv(directory / 'qa_windows' / 'selection.csv')
    assert all((directory / 'qa_windows' / f).is_file() for f in selections.file), participant
    assert m['summary']['n_accepted'] == int(d.ppg_valid.sum()), participant
    return d, g, m, audit


def summarize(review, output, cfg):
    summary, gaps, audits, reason_tables, motion, checks = [], [], [], [], [], []
    for participant in review.participant_id:
        folder = output / participant
        d, g, manifest, audit = check_result(participant, folder)
        if manifest['config'] != asdict(cfg):
            raise ValueError(f'Configuration mismatch for {participant}')
        s = dict(manifest['summary'])
        raw = audit[audit.channel.eq(cfg.channel)].iloc[0]
        s.update(record_duration_h=raw.duration_s / 3600,
                 gap_grid_p90_s=float(g.grid_duration_s.quantile(.9)) if len(g) else np.nan,
                 gap_grid_p95_s=float(g.grid_duration_s.quantile(.95)) if len(g) else np.nan,
                 n_gaps_ge_60s=int(g.grid_duration_s.ge(60).sum()),
                 n_gaps_ge_300s=int(g.grid_duration_s.ge(300).sum()),
                 accepted_hr_median_bpm=float(d.hr_ppg.median()),
                 accepted_acc_dynamic_rms_median=float(d.loc[d.ppg_valid & d.acc_valid, 'acc_dynamic_rms'].median()),
                 rejected_acc_dynamic_rms_median=float(d.loc[~d.ppg_valid & d.acc_valid, 'acc_dynamic_rms'].median()),
                 n_qa_plots=len(pd.read_csv(folder / 'qa_windows' / 'selection.csv')),
                 rule_version=cfg.rule_version)
        summary.append(s)
        gaps.append(g)
        audit.insert(0, 'participant_id', participant); audits.append(audit)
        counts = pd.read_csv(folder / 'rejection_counts.csv')
        counts.insert(0, 'participant_id', participant); reason_tables.append(counts)
        for valid, decision in [(True, 'accepted'), (False, 'rejected')]:
            a = d[d.ppg_valid.eq(valid) & d.acc_valid]
            for feature in ['acc_dynamic_rms', 'acc_magnitude_std', 'acc_jerk_rms']:
                v = a[feature].dropna()
                motion.append(dict(participant_id=participant, decision=decision, feature=feature,
                                   n=len(v), mean=v.mean(), median=v.median(),
                                   p10=v.quantile(.1), p25=v.quantile(.25), p75=v.quantile(.75), p90=v.quantile(.9)))
        # Relative motion quartiles are computed separately for each participant.
        a = d[d.acc_valid].copy()
        if a.acc_dynamic_rms.nunique() >= 4:
            a['quartile'] = pd.qcut(a.acc_dynamic_rms, 4, labels=False, duplicates='drop')
            for q, sub in a.groupby('quartile'):
                motion.append(dict(participant_id=participant, decision='all', feature=f'motion_quartile_{int(q)+1}',
                                   n=len(sub), acceptance=sub.ppg_valid.mean(),
                                   median=sub.acc_dynamic_rms.median()))
        checks.append(dict(participant_id=participant, passed=True, n_windows=len(d),
                           checks='Complete grid; unique keys; HR/decision consistency; ACC NaN semantics; gap runs and durations; notes masks; configuration; plots present.'))
    df = pd.DataFrame(summary).merge(review.drop(columns=['input_file']), on='participant_id', validate='one_to_one')
    df.to_csv(output / 'participant_summary.csv', index=False)
    gg = pd.concat(gaps, ignore_index=True); gg.to_csv(output / 'all_gaps.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / 'all_channel_audits.csv', index=False)
    rr = pd.concat(reason_tables, ignore_index=True); rr.to_csv(output / 'rejection_reasons_by_participant.csv', index=False)
    rr.groupby('reason', as_index=False).n_windows.sum().sort_values('n_windows', ascending=False).to_csv(output / 'rejection_reasons_overall.csv', index=False)
    mm = pd.DataFrame(motion); mm.to_csv(output / 'motion_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'validation_checks.csv', index=False)
    overall = dict(n_participants=len(df), total_recording_hours=float(df.record_duration_h.sum()),
                   n_windows=int(df.n_windows.sum()), n_accepted=int(df.n_accepted.sum()),
                   n_rejected=int(df.n_rejected.sum()),
                   window_weighted_coverage=float(df.n_accepted.sum() / df.n_windows.sum()),
                   participant_mean_coverage=float(df.coverage.mean()),
                   n_gaps=len(gg), gap_grid_median_s=float(gg.grid_duration_s.median()),
                   gap_grid_p90_s=float(gg.grid_duration_s.quantile(.9)), gap_grid_max_s=float(gg.grid_duration_s.max()),
                   n_acc_valid=int(df.n_acc_valid.sum()), n_documented_exclusion_windows=int(df.n_documented_exclusion_windows.sum()),
                   all_participant_validations_passed=True, config=asdict(cfg),
                   status='Provisional PPG-only baseline; no ECG accuracy evaluation or threshold calibration.')
    (output / 'overall_summary.json').write_text(json.dumps(overall, indent=2), encoding='utf-8')
    make_summary_plots(df, gg, rr, mm, output)
    motion_flags = ', '.join(df.loc[df.accepted_acc_dynamic_rms_median > df.rejected_acc_dynamic_rms_median, 'participant_id']) or 'none'
    cols = ['participant_id', 'record_duration_h', 'n_windows', 'coverage', 'n_gaps',
            'gap_grid_median_s', 'gap_grid_p90_s', 'gap_grid_max_s', 'n_documented_exclusion_windows']
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>WildPPG member2 batch v1</title>'
              '<style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#172331}'
              'table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:right}'
              'img{width:100%}a{color:#145ba6}pre{white-space:pre-wrap}</style>'
              '<h1>WildPPG wrist PPG: all participants</h1><p>Provisional ppg_only_dev_v1. '
              '8-second windows, 2-second stride. Acceptance uses PPG only; no ECG accuracy claims.</p>'
              f'<p>{overall["n_participants"]} participants; {overall["total_recording_hours"]:.2f} recording hours; '
              f'{overall["n_windows"]:,} windows; {overall["window_weighted_coverage"]:.2%} window-weighted acceptance. '
              f'{overall["n_gaps"]:,} rejected HR-grid runs. All output consistency checks passed.</p>'
              f'<p>Participant acceptance ranges from {df.coverage.min():.2%} to {df.coverage.max():.2%}. '
              'The same thresholds were retained despite between-participant differences. '
              'These are provisional signal-quality decisions, not verified HR accuracy.</p>'
              f'<p>Motion review flags (accepted median dynamic ACC RMS exceeds rejected median): {html.escape(motion_flags)}. '
              'Inspect flagged high-motion accepted waveforms and compare to ECG when available; '
              'movement is not itself a rejection criterion.</p>'
              '<p>Coverage includes documented exclusions and filter guards. Reasons can overlap. '
              'Grid gap duration = rejected-window count Ã— stride; it is not raw missing-signal duration. '
              'Overlapping windows are not independent observations. Do not sum raw window durations.</p>'
              '<img src="participant_overview.png" alt="Participant coverage and gaps">'
              + df[cols].round(4).to_html(index=False)
              + '<h2>Gap lengths and rejection reasons</h2><img src="gap_reason_overview.png">'
              '<h2>Motion distribution</h2><img src="motion_overview.png">'
              '<p>ACC units are native stored units. Motion quartiles are within-participant descriptive groups; '
              'ACC does not enter rejection rules.</p><h2>Notes review</h2>'
              + review[['participant_id', 'notes', 'action']].to_html(index=False)
              + '<h2>Participant outputs</h2><ul>'
              + ''.join(f'<li>{html.escape(p)}: <a href="{p}/windows.csv">window table</a> | '
                        f'<a href="{p}/gaps.csv">gaps</a> | <a href="{p}/timeline.png">timeline</a> | '
                        f'<a href="{p}/quality_summary.png">quality</a> | <a href="{p}/qa_windows/selection.csv">QA selection</a></li>' for p in df.participant_id)
              + '</ul></html>')
    (output / 'report.html').write_text(report, encoding='utf-8')
    print('OVERALL SUMMARY', json.dumps(overall, indent=2), flush=True)


def make_summary_plots(df, gaps, reasons, motion, output):
    x = np.arange(len(df)); labels = df.participant_id
    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    ax[0].bar(x, 100 * df.coverage, color='#267a69'); ax[0].set_ylabel('Accepted windows (%)'); ax[0].set_ylim(0, 100)
    ax[1].bar(x, df.n_gaps, color='#3c6fa6'); ax[1].set_ylabel('Rejected HR-grid runs')
    ax[2].bar(x, df.record_duration_h, color='#876da8'); ax[2].set_ylabel('Recording hours'); ax[2].set_xticks(x, labels)
    fig.suptitle('PPG-only development baseline | same rules for all participants | no ECG validation')
    fig.tight_layout(); fig.savefig(output / 'participant_overview.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    bins = [0, 2.01, 6, 10, 30, 60, 120, 300, 900, 3600, max(3601, gaps.grid_duration_s.max() + 1)]
    counts, _ = np.histogram(gaps.grid_duration_s, bins=bins)
    labels = ['2', '4', '6-8', '10-28', '30-58', '60-118', '120-298', '300-898', '900-3598', '3600+']
    ax[0].bar(np.arange(len(counts)), counts, color='#3c6fa6'); ax[0].set_xticks(np.arange(len(counts)), labels, rotation=45)
    ax[0].set_yscale('log'); ax[0].set_xlabel('HR-grid gap duration (s)'); ax[0].set_ylabel('Gap count (log scale)')
    total = reasons.groupby('reason').n_windows.sum().sort_values()
    ax[1].barh(total.index, total.values, color='#b76043'); ax[1].set_xlabel('Windows (multiple reasons possible)')
    fig.tight_layout(); fig.savefig(output / 'gap_reason_overview.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(14, 9))
    for decision, color in [('accepted', '#267a69'), ('rejected', '#b76043')]:
        sub = motion[(motion.feature == 'acc_dynamic_rms') & motion.decision.eq(decision)].set_index('participant_id').reindex(df.participant_id)
        ax[0].plot(x, sub['median'], 'o-', color=color, label=decision)
    ax[0].set_xticks(x, df.participant_id); ax[0].set_ylabel('Median dynamic ACC RMS\n(native stored units)'); ax[0].legend()
    for q in range(1, 5):
        sub = motion[motion.feature.eq(f'motion_quartile_{q}')].set_index('participant_id').reindex(df.participant_id)
        ax[1].plot(x, 100 * sub.acceptance, 'o-', label=f'Within-person motion Q{q}')
    ax[1].set_xticks(x, df.participant_id); ax[1].set_ylim(0, 100); ax[1].set_ylabel('Accepted windows (%)'); ax[1].legend(ncol=2)
    fig.tight_layout(); fig.savefig(output / 'motion_overview.png', dpi=150); plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root.parent / 'WildPPG' / 'WildPPG' / 'data')
    parser.add_argument('--out-dir', type=Path, default=root / 'results' / 'batch_v1')
    parser.add_argument('--resume', action='store_true', help='Reuse only results with matching configuration, input identity and source hash, plus a completion marker')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob('WildPPG_Part_*.mat'))
    if not paths:
        raise ValueError('No participant MAT files found')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    review = review_notes(paths, args.out_dir)
    source_hash = hashlib.sha256((root / 'process_wrist.py').read_bytes()).hexdigest()
    run_meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hash,
                    participants=review.participant_id.tolist(), config=asdict(cfg), notes_review_before_processing=True)
    (args.out_dir / 'batch_manifest.json').write_text(json.dumps(run_meta, indent=2), encoding='utf-8')
    if not args.summarize_only:
        for i, row in enumerate(review.itertuples(), 1):
            participant = row.participant_id
            folder = args.out_dir / participant; folder.mkdir(exist_ok=True)
            input_path = Path(row.input_file)
            signature = dict(source_sha256=source_hash, input_path=str(input_path.resolve()),
                             input_size=input_path.stat().st_size, input_mtime_ns=input_path.stat().st_mtime_ns,
                             config=asdict(cfg))
            marker = folder / 'completed.json'
            if args.resume and marker.exists() and json.loads(marker.read_text()) == signature:
                check_result(participant, folder)
                print(f'[{i}/{len(review)}] {participant}: validated existing output', flush=True)
                continue
            marker.unlink(missing_ok=True)
            print(f'[{i}/{len(review)}] {participant}: processing', flush=True)
            started = time.monotonic()
            with (folder / 'processing.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(root / 'process_wrist.py'), '--data-dir', str(args.data_dir),
                                '--participant', participant, '--out-dir', str(folder), '--notes-policy', 'mask'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            check_result(participant, folder)
            marker.write_text(json.dumps(signature, indent=2), encoding='utf-8')
            s = json.loads((folder / 'summary.json').read_text())
            print(f'[{i}/{len(review)}] {participant}: {s["n_windows"]:,} windows, {s["coverage"]:.2%} accepted; checked; {time.monotonic()-started:.1f}s', flush=True)
    summarize(review, args.out_dir, cfg)
    run_meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
    run_meta['status'] = 'completed; all participants validated'
    (args.out_dir / 'batch_manifest.json').write_text(json.dumps(run_meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
