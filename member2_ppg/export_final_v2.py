"""Export existing final v2 decisions with the same layout as batch_v1.

Reads saved PPG window tables only. No ECG, tuning or training is performed.
"""
from __future__ import annotations

import hashlib
import html
import json
import csv
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = 'ppg_only_dev_v2'
GAP_COLUMNS = ['participant_id', 'gap_id', 'first_window_id', 'last_window_id',
               'n_windows', 'grid_start_center_s', 'grid_end_exclusive_s',
               'grid_duration_s', 'raw_support_start_s', 'raw_support_end_s',
               'raw_support_duration_s', 'touches_record_start',
               'touches_record_end', 'reasons']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def rejected_runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return list(zip(edges[::2], edges[1::2]))


def gap_table(table, stride_s):
    rows = []
    for number, (start, end) in enumerate(rejected_runs(~table.ppg_valid.to_numpy())):
        group = table.iloc[start:end]
        reasons = sorted(set(';'.join(group.reject_reasons.fillna('')).split(';')) - {''})
        rows.append(dict(participant_id=str(group.participant_id.iloc[0]), gap_id=number,
                         first_window_id=int(group.window_id.iloc[0]),
                         last_window_id=int(group.window_id.iloc[-1]), n_windows=end - start,
                         grid_start_center_s=float(group.center_s.iloc[0]),
                         grid_end_exclusive_s=float(group.center_s.iloc[-1] + stride_s),
                         grid_duration_s=float((end - start) * stride_s),
                         raw_support_start_s=float(group.start_s.iloc[0]),
                         raw_support_end_s=float(group.end_s.iloc[-1]),
                         raw_support_duration_s=float(group.end_s.iloc[-1] - group.start_s.iloc[0]),
                         touches_record_start=start == 0, touches_record_end=end == len(table),
                         reasons=';'.join(reasons)))
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def validate_windows(table, participant, base):
    if table.empty or not table.participant_id.eq(participant).all():
        raise ValueError(f'{participant}: missing or inconsistent participant rows')
    if not np.array_equal(table.window_id, np.arange(len(table))):
        raise ValueError(f'{participant}: incomplete window grid')
    if not np.allclose(np.diff(table.center_s), base['stride_s']):
        raise ValueError(f'{participant}: inconsistent stride')
    if not np.allclose(table.end_s - table.start_s, base['window_s']):
        raise ValueError(f'{participant}: inconsistent window duration')
    samples = table.end_sample_exclusive.iloc[0] - table.start_sample.iloc[0]
    step = round(samples * base['stride_s'] / base['window_s'])
    if not np.array_equal(table.start_sample, np.arange(len(table)) * step):
        raise ValueError(f'{participant}: original sample grid was not retained')
    if not np.all(table.end_sample_exclusive - table.start_sample == samples):
        raise ValueError(f'{participant}: inconsistent sample window lengths')
    accepted = table.ppg_valid.to_numpy(bool)
    if not np.array_equal(accepted, table.hr_ppg.notna()):
        raise ValueError(f'{participant}: HR missingness and acceptance disagree')
    if not np.isfinite(table.loc[accepted, 'hr_ppg']).all():
        raise ValueError(f'{participant}: accepted HR must be finite')
    if not np.allclose(table.loc[accepted, 'hr_ppg'], table.loc[accepted, 'hr_spectral_bpm'],
                       rtol=0, atol=1e-9):
        raise ValueError(f'{participant}: accepted HR differs from measured spectral HR')
    hard = (~table.finite_ratio.eq(1) | table.documented_exclusion | ~table.filter_safe
            | table.longest_flat_run_s.ge(base['max_flat_run_s']))
    if table.loc[hard, 'ppg_valid'].any():
        raise ValueError(f'{participant}: hard-invalid window accepted')
    reasons = table.reject_reasons.fillna('')
    if not reasons[accepted].eq('').all() or not reasons[~accepted].ne('').all():
        raise ValueError(f'{participant}: rejection reason mismatch')


def validate_gap_spans(gaps, original, participant):
    for column in ['first_window_id', 'last_window_id', 'n_windows',
                   'grid_start_center_s', 'grid_end_exclusive_s', 'grid_duration_s',
                   'touches_record_start', 'touches_record_end']:
        if len(gaps) != len(original) or not np.array_equal(gaps[column], original[column]):
            raise ValueError(f'{participant}: saved rejected-run spans changed: {column}')


def _number(series, method, *args):
    value = getattr(series, method)(*args)
    return float(value) if pd.notna(value) else None


def participant_summary(pid, table, gaps, audit, base):
    sensor = audit[audit.channel.eq(base['channel'])]
    duration = float(sensor.duration_s.iloc[0]) if len(sensor) else float(table.end_s.iloc[-1])
    trailing = (int(sensor.n_samples.iloc[0] - table.end_sample_exclusive.iloc[-1])
                if len(sensor) else None)
    accepted = table.ppg_valid
    return dict(participant_id=pid, n_windows=len(table), n_accepted=int(accepted.sum()),
                n_rejected=int((~accepted).sum()), coverage=float(accepted.mean()), n_gaps=len(gaps),
                gap_grid_total_s=float(gaps.grid_duration_s.sum()),
                gap_grid_median_s=_number(gaps.grid_duration_s, 'median'),
                gap_grid_max_s=_number(gaps.grid_duration_s, 'max'),
                n_acc_valid=int(table.acc_valid.sum()),
                n_documented_exclusion_windows=int(table.documented_exclusion.sum()),
                trailing_samples_without_complete_window=trailing, record_duration_h=duration / 3600,
                gap_grid_p90_s=_number(gaps.grid_duration_s, 'quantile', .9),
                gap_grid_p95_s=_number(gaps.grid_duration_s, 'quantile', .95),
                n_gaps_ge_60s=int(gaps.grid_duration_s.ge(60).sum()),
                n_gaps_ge_300s=int(gaps.grid_duration_s.ge(300).sum()),
                accepted_hr_median_bpm=_number(table.loc[accepted, 'hr_ppg'], 'median'),
                accepted_acc_dynamic_rms_median=_number(table.loc[accepted & table.acc_valid, 'acc_dynamic_rms'], 'median'),
                rejected_acc_dynamic_rms_median=_number(table.loc[~accepted & table.acc_valid, 'acc_dynamic_rms'], 'median'),
                n_qa_plots=0, n_context_rescued=int(table.context_rescued.sum()), rule_version=RULE_VERSION)


def plots(table, gaps, folder):
    hours = table.center_s / 3600
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].scatter(hours, table.hr_ppg_candidate, s=1, color='#aaa', label='measured candidate')
    axes[0].scatter(hours, table.hr_ppg, s=1, color='#147d64', label='accepted')
    axes[0].set_ylabel('HR (bpm)'); axes[0].legend(markerscale=4)
    axes[1].plot(hours, table.spectral_concentration, lw=.4, label='concentration')
    axes[1].plot(hours, table.acf_peak, lw=.4, label='ACF')
    axes[1].set_ylabel('PPG quality'); axes[1].legend()
    axes[2].step(hours, table.ppg_valid.astype(int), where='mid', lw=.4)
    axes[2].set_ylabel('Accepted'); axes[2].set_xlabel('Recording time (hours)')
    figure.suptitle(f'{table.participant_id.iloc[0]} | final v2 | window-table summary')
    figure.tight_layout(); figure.savefig(folder / 'timeline.png', dpi=130); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(['accepted', 'rejected'], [int(table.ppg_valid.sum()), int((~table.ppg_valid).sum())])
    axes[0].set_ylabel('Windows')
    if len(gaps):
        axes[1].hist(gaps.grid_duration_s, bins=35)
    axes[1].set_xlabel('HR-grid gap duration (s)'); axes[1].set_ylabel('Gap count')
    figure.tight_layout(); figure.savefig(folder / 'quality_summary.png', dpi=130); plt.close(figure)


def write_participant(table, participant, folder, config, source_entry=None):
    """Write one already-decided participant using V1 filenames.

    Returns summary plus gaps, rejection_counts and channel_audit DataFrames.
    The input table's acceptance and measured HR are preserved.
    """
    folder = Path(folder).resolve()
    if 'batch_v1' in folder.parts or (folder.exists() and any(folder.iterdir())):
        raise FileExistsError(f'Preserving existing output: {folder}')
    saved = table.copy()
    table = table.copy()
    table['rule_version'] = RULE_VERSION
    base = config['base_config']
    validate_windows(table, participant, base)
    gaps = gap_table(table, base['stride_s'])
    folder.mkdir(parents=True, exist_ok=True)
    if source_entry is not None and 'windows' in source_entry:
        # Keep every original measurement's decimal text exactly; only the
        # version cell changes when consolidating an existing saved table.
        source_windows = (ROOT / source_entry['windows']).resolve()
        if not source_windows.is_relative_to(ROOT):
            raise ValueError('Source table escapes the repository')
        with source_windows.open('r', newline='', encoding='utf-8') as source_stream:
            reader = csv.reader(source_stream)
            header = next(reader)
            version_column = header.index('rule_version')
            with (folder / 'windows.csv').open('w', newline='', encoding='utf-8') as target_stream:
                writer = csv.writer(target_stream, lineterminator='\n')
                writer.writerow(header)
                for row in reader:
                    row[version_column] = RULE_VERSION
                    writer.writerow(row)
    else:
        table.to_csv(folder / 'windows.csv', index=False, float_format='%.17g')
    gaps.to_csv(folder / 'gaps.csv', index=False, float_format='%.12g')
    written = pd.read_csv(folder / 'windows.csv')
    for column in ['window_id', 'start_sample', 'end_sample_exclusive', 'ppg_valid']:
        if not np.array_equal(written[column], saved[column]):
            raise ValueError(f'{participant}: exported grid/decisions changed: {column}')
    for column in ['hr_ppg', 'hr_ppg_candidate', 'hr_spectral_bpm']:
        if not np.allclose(written[column], saved[column], rtol=0, atol=1e-9, equal_nan=True):
            raise ValueError(f'{participant}: exported measured HR changed: {column}')
    audit_source = ROOT / 'member2_ppg/results/feature_cache_v2' / participant / 'channel_audit.csv'
    metadata_source = audit_source.with_name('manifest.json')
    if not audit_source.is_file():
        audit_source = ROOT / 'member2_ppg/results/batch_v1' / participant / 'channel_audit.csv'
    if not audit_source.is_file():
        raise FileNotFoundError(f'{participant}: channel audit not available')
    shutil.copyfile(audit_source, folder / 'channel_audit.csv')
    audit = pd.read_csv(audit_source)
    summary = participant_summary(participant, table, gaps, audit, base)
    write_json(folder / 'summary.json', summary)
    metadata = json.loads(metadata_source.read_text(encoding='utf-8')) if metadata_source.exists() else {}
    for key in ['source_sha256', 'config', 'summary', 'status']:
        metadata.pop(key, None)
    metadata.update(rule_version=RULE_VERSION, status='completed', config=base,
                    context_rule=config['context_rule'], summary=summary,
                    channel_audit_source=relative(audit_source), measured_hr_unchanged=True,
                    qa='Window-table plots; raw waveform QA is not included.', training=False)
    if source_entry is not None:
        metadata['source_outputs'] = source_entry
    write_json(folder / 'manifest.json', metadata)
    counts = (table.loc[~table.ppg_valid, 'reject_reasons'].str.split(';').explode()
              .value_counts().rename_axis('reason').reset_index(name='n_windows'))
    counts.to_csv(folder / 'rejection_counts.csv', index=False)
    plots(table, gaps, folder)
    write_json(folder / 'completed.json', dict(status='completed', rule_version=RULE_VERSION,
               windows_sha256=sha256(folder / 'windows.csv'), gaps_sha256=sha256(folder / 'gaps.csv'),
               measured_hr_unchanged=True, training=False))
    return dict(summary=summary, gaps=gaps, rejection_counts=counts, channel_audit=audit)


def export_saved(source_registry: dict, config: dict, output: Path):
    """Consolidate saved final decisions; never select or recalculate acceptance."""
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'member2_ppg/results') or 'batch_v1' in output.parts:
        raise ValueError('Output must be a separate directory under member2_ppg/results')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Preserving nonempty output: {output}')
    base = config['base_config']
    if base.get('half_rate_action') != 'off':
        raise ValueError('Final v2 requires half-rate handling off')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    summaries, gap_tables, reason_tables, audits, motion, validations, participants = [], [], [], [], [], [], {}
    for pid, entry in sorted(source_registry['participants'].items()):
        source = (ROOT / entry['windows']).resolve()
        gap_source = (ROOT / entry['gaps']).resolve()
        if not source.is_relative_to(ROOT) or not gap_source.is_relative_to(ROOT):
            raise ValueError('Source table escapes the repository')
        if sha256(source) != entry['windows_sha256'] or sha256(gap_source) != entry['gaps_sha256']:
            raise ValueError(f'{pid}: source table hash mismatch')
        saved = pd.read_csv(source)
        table = saved.copy()
        table['rule_version'] = RULE_VERSION
        validate_windows(table, pid, base)
        gaps = gap_table(table, base['stride_s'])
        validate_gap_spans(gaps, pd.read_csv(gap_source), pid)
        folder = output / pid
        written = write_participant(table, pid, folder, config, source_entry=entry)
        summary = written['summary']
        gaps = written['gaps']
        audit = written['channel_audit']
        counts = written['rejection_counts']
        reason_tables.append(counts.assign(participant_id=pid)[['participant_id', 'reason', 'n_windows']])
        for decision, label in [(True, 'accepted'), (False, 'rejected')]:
            for feature in ['acc_dynamic_rms', 'acc_magnitude_std', 'acc_jerk_rms']:
                values = table.loc[table.ppg_valid.eq(decision) & table.acc_valid, feature].dropna()
                motion.append(dict(participant_id=pid, decision=label, feature=feature, n=len(values),
                                   mean=values.mean(), median=values.median(), p10=values.quantile(.1),
                                   p25=values.quantile(.25), p75=values.quantile(.75), p90=values.quantile(.9)))
        summaries.append(summary); gap_tables.append(gaps)
        audits.append(audit.assign(participant_id=pid))
        validations.append(dict(participant_id=pid, passed=True, n_windows=len(table),
                                checks='Original grid, acceptance and measured HR retained; exact rejected runs.'))
        participants[pid] = dict(split=entry['split'], windows=relative(folder / 'windows.csv'),
                                 windows_sha256=sha256(folder / 'windows.csv'), gaps=relative(folder / 'gaps.csv'),
                                 gaps_sha256=sha256(folder / 'gaps.csv'))
        print(f'Exported {pid}: {len(table):,} windows, {summary["coverage"]:.2%} accepted', flush=True)
    s = pd.DataFrame(summaries)
    g = pd.concat(gap_tables, ignore_index=True)
    r = pd.concat(reason_tables, ignore_index=True)
    s.to_csv(output / 'participant_summary.csv', index=False)
    g.to_csv(output / 'all_gaps.csv', index=False)
    r.to_csv(output / 'rejection_reasons_by_participant.csv', index=False)
    r.groupby('reason', as_index=False).n_windows.sum().to_csv(output / 'rejection_reasons_overall.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / 'all_channel_audits.csv', index=False)
    pd.DataFrame(motion).to_csv(output / 'motion_summary.csv', index=False)
    pd.DataFrame(validations).to_csv(output / 'validation_checks.csv', index=False)
    for name in ['notes_review.csv', 'notes_review.json']:
        notes = ROOT / 'member2_ppg/results/batch_v1' / name
        if notes.exists():
            shutil.copyfile(notes, output / name)
    overall = dict(n_participants=len(s), n_windows=int(s.n_windows.sum()),
                   n_accepted=int(s.n_accepted.sum()), n_rejected=int(s.n_rejected.sum()),
                   window_weighted_coverage=float(s.n_accepted.sum() / s.n_windows.sum()),
                   participant_mean_coverage=float(s.coverage.mean()), n_gaps=len(g),
                   gap_grid_median_s=_number(g.grid_duration_s, 'median'),
                   gap_grid_p90_s=_number(g.grid_duration_s, 'quantile', .9),
                   gap_grid_max_s=_number(g.grid_duration_s, 'max'),
                   all_participant_validations_passed=True, rule_version=RULE_VERSION, config=base)
    write_json(output / 'overall_summary.json', overall)
    manifest = dict(rule_version=RULE_VERSION, status='completed; all participants validated',
                    participants=participants, config=base, context_rule=config['context_rule'], training=False)
    write_json(output / 'batch_manifest.json', manifest)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(s.participant_id, 100 * s.coverage); axes[0].set_ylabel('Accepted windows (%)')
    axes[1].bar(s.participant_id, s.n_gaps); axes[1].set_ylabel('Gap count')
    figure.suptitle('Final v2 | PPG window summaries')
    figure.tight_layout(); figure.savefig(output / 'participant_overview.png', dpi=130); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    if len(g):
        axes[0].hist(g.grid_duration_s, bins=35)
        axes[0].set_yscale('log')
    axes[0].set_xlabel('HR-grid gap duration (s)'); axes[0].set_ylabel('Gap count')
    totals = r.groupby('reason').n_windows.sum().sort_values()
    axes[1].barh(totals.index, totals.values)
    axes[1].set_xlabel('Rejected windows (multiple reasons possible)')
    figure.tight_layout(); figure.savefig(output / 'gap_reason_overview.png', dpi=130); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    motion_frame = pd.DataFrame(motion)
    for axis, feature in zip(axes, ['acc_dynamic_rms', 'acc_jerk_rms']):
        for decision in ['accepted', 'rejected']:
            values = motion_frame[(motion_frame.feature == feature) & (motion_frame.decision == decision)]
            values = values.set_index('participant_id').reindex(s.participant_id)
            axis.plot(s.participant_id, values['median'], 'o-', label=decision)
        axis.set_ylabel(f'Median {feature}'); axis.legend()
    figure.tight_layout(); figure.savefig(output / 'motion_overview.png', dpi=130); plt.close(figure)
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Final v2</title>'
              '<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}'
              'table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}</style>'
              '<h1>Final v2</h1><p>PPG acceptance and gap summaries. Existing measured HR and decisions '
              'are unchanged. No threshold selection or training was performed.</p>'
              f'<p>{overall["n_windows"]:,} windows; {overall["window_weighted_coverage"]:.2%} accepted; '
              f'{len(g):,} gaps. Acceptance here includes all PPG windows, not only ECG-valid windows.</p>'
              '<p>Evaluation participant means among ECG-valid windows: acceptance 16.79%, MAE 4.38 bpm, '
              'error above 10 bpm 7.83%. Some participants regressed; the cohort was previously inspected. '
              'These averages do not guarantee accuracy for every participant.</p>'
              + s[['participant_id', 'n_windows', 'n_accepted', 'coverage', 'n_gaps']].round(4).to_html(index=False)
              + '<p>Window-table plots only; raw waveform QA is not included.</p><ul>'
              + ''.join(f'<li>{html.escape(pid)}: <a href="{pid}/timeline.png">timeline</a>, '
                        f'<a href="{pid}/quality_summary.png">quality</a>, '
                        f'<a href="{pid}/windows.csv">windows</a>, <a href="{pid}/gaps.csv">gaps</a></li>'
                        for pid in participants) + '</ul></html>')
    (output / 'report.html').write_text(report, encoding='utf-8')
    return manifest, overall
