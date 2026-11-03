"""Summaries and figures from a harness run.

    python imputation/analyze.py imputation/results/<run>

Writes summary.md, protocol_imputer_mae.csv, and figures/*.png into the run directory.
With an ECG reference ('real' rows present) it also computes, per protocol:
  rank agreement   Spearman rho between synthetic MAE and real-gap MAE across imputers (per participant)
  calibration      mean |synthetic MAE - real MAE| across imputers, and mean signed (optimism < 0)
  regret           real-gap MAE of the protocol's top-ranked imputer minus the best imputer's
each with a participant-level bootstrap 95% CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

PROTO_ORDER = ['random_point', 'random_block', 'replay', 'context_matched']
COLORS = {'random_point': '#2a78d6', 'random_block': '#eb6834', 'replay': '#1baf7a',
          'context_matched': '#6250d6', 'real': '#52514e'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def boot_ci(values, n=2000, seed=0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v))).mean(1)
    return tuple(np.quantile(means, [0.025, 0.975]))


ROOT = Path(__file__).resolve().parents[1]


def endpoint_table(per):
    rows = []
    for proto, g in per.groupby('protocol'):
        row = dict(protocol=proto)
        for m in ('spearman', 'calib_abs', 'calib_signed', 'regret'):
            lo, hi = boot_ci(g[m])
            row[m] = f'{g[m].mean():.2f} [{lo:.2f}, {hi:.2f}]'
        rows.append(row)
    cols = ['spearman', 'calib_abs', 'calib_signed', 'regret']
    return pd.DataFrame(rows, columns=['protocol'] + cols).set_index('protocol').reindex(PROTO_ORDER)


def per_participant(res):
    syn = res[res.protocol != 'real'].groupby(['test_participant', 'protocol', 'imputer']).mae.mean().rename('syn')
    real = res[res.protocol == 'real'].set_index(['test_participant', 'imputer']).mae.rename('real')
    if real.empty:
        return pd.DataFrame(), syn
    d = syn.reset_index().join(real, on=['test_participant', 'imputer'])
    rows = []
    for (p, proto), g in d.groupby(['test_participant', 'protocol']):
        g = g.dropna()
        if len(g) < 3:
            continue
        pick = g.loc[g.syn.idxmin()]
        rows.append(dict(test_participant=p, protocol=proto,
                         spearman=spearmanr(g.syn, g.real).statistic,
                         calib_abs=(g.syn - g.real).abs().mean(),
                         calib_signed=(g.syn - g.real).mean(),
                         regret=pick.real - g.real.min(), pick=pick.imputer))
    return pd.DataFrame(rows), syn


def main(run):
    run = Path(run)
    res = pd.read_csv(run / 'results.csv')
    stats = pd.read_csv(run / 'mask_stats.csv')
    feas = pd.read_csv(run / 'feasibility.csv')
    cfg = json.loads((run / 'config.json').read_text())
    figs = run / 'figures'; figs.mkdir(exist_ok=True)
    has_real = (res.protocol == 'real').any()

    # ---- protocol x imputer MAE (mean over participants of the per-participant rep-mean)
    pp = res.groupby(['test_participant', 'protocol', 'imputer']).mae.mean().reset_index()
    tab = pp.pivot_table(index='imputer', columns='protocol', values='mae', aggfunc='mean')
    tab = tab[[c for c in PROTO_ORDER + ['real'] if c in tab.columns]]
    order = tab.mean(1).sort_values().index
    tab = tab.loc[order]
    tab.to_csv(run / 'protocol_imputer_mae.csv', float_format='%.3f')
    ranks = tab.rank()
    taus = {}
    ref_col = 'real' if has_real else 'random_point'
    for c in tab.columns:
        if c != ref_col:
            taus[c] = kendalltau(tab[c], tab[ref_col]).statistic

    # ---- figure: dot plot of MAE by imputer, one colour per protocol
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(tab) + 1.4))
    style(ax)
    y = np.arange(len(tab))
    cols = list(tab.columns)
    off = np.linspace(-0.27, 0.27, len(cols))
    for k, c in enumerate(cols):
        ax.scatter(tab[c], y + off[k], s=40, color=COLORS[c], edgecolor='white', lw=1.5, zorder=3,
                   label=c.replace('_', ' ') + (' gaps (ECG)' if c == 'real' else ''))
    ax.set_yticks(y, [i.replace('_', ' ') for i in tab.index])
    ax.invert_yaxis()
    ax.set_xlabel(f'MAE (bpm) vs {"ECG" if has_real else "accepted PPG HR (provisional)"}', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(cols))
    ax.set_title('Imputer error under each gap-generation protocol', loc='left', fontsize=11, color=INK, pad=24)
    fig.tight_layout(); fig.savefig(figs / 'mae_by_protocol.png', dpi=200); plt.close(fig)

    # ---- figure: feasibility (supply vs demand per movement bin)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    style(ax); ax.grid(axis='x', visible=False); ax.grid(axis='y', color=GRID, lw=0.8)
    x = np.arange(len(feas)); w = 0.38
    ax.bar(x - w / 2 - 0.01, feas.real_gap_time_min / 60, w, color=COLORS['real'], label='real-gap time (demand)')
    ax.bar(x + w / 2 + 0.01, feas.accepted_time_min / 60, w, color=COLORS['context_matched'], label='accepted time (supply)')
    for i, r in feas.iterrows():
        ax.text(i, max(r.real_gap_time_min, r.accepted_time_min) / 60 + 0.6, f'{r.placements_per_real_gap:.2f}',
                ha='center', fontsize=8, color=MUTED)
    ax.set_xticks(x, feas.movement_bin)
    ax.set_ylabel('hours (all participants)', color=INK)
    ax.set_title('Matching feasibility by movement bin (labels: placements per real gap)', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figs / 'feasibility.png', dpi=200); plt.close(fig)

    # ---- manipulation check: motion of hidden windows per protocol
    mc = stats.groupby('protocol').agg(hidden_log_motion=('hidden_log_motion_mean', 'mean'),
                                       median_gap_s=('median_gap_s', 'median'),
                                       p90_gap_s=('p90_gap_s', 'median'),
                                       n_hidden=('n_hidden', 'median'),
                                       bin_fallback_rate=('bin_fallback_rate', 'mean'))
    mc = mc.reindex([p for p in PROTO_ORDER + ['real'] if p in mc.index])

    lines = [f'# Run summary: `{run.name}`', '',
             f'- Reference: **{cfg["reference"]}**; participants: {len(cfg["participants"])}; '
             f'reps per protocol: {cfg["reps"]}; target hidden fraction: {cfg["frac"]}', '']
    if not has_real:
        lines += ['> Provisional run: no ECG reference yet, so synthetic gaps are scored against accepted wrist-PPG HR '
                  'and real-gap evaluation is not possible. Use these numbers to check the pipeline, not to answer RQ1/RQ2.', '']
    lines += ['## Mean MAE (bpm) by protocol and imputer', '', tab.round(2).to_markdown(), '',
              f'Kendall tau of each column\'s imputer ranking against `{ref_col}`: ' +
              ', '.join(f'{k}={v:.2f}' for k, v in taus.items()), '',
              '## Masks: manipulation check', '', mc.round(3).to_markdown(), '',
              '## Matching feasibility (pooled)', '', feas.round(2).to_markdown(index=False), '']

    if has_real:
        per, _ = per_participant(res)
        per.to_csv(run / 'per_participant_metrics.csv', index=False)
        summ = endpoint_table(per)
        # PPG rules tuned on some participants' ECG: also report the participants held out from that tuning
        sel = ROOT / 'member2_ppg' / 'results' / cfg.get('ppg_batch', 'batch_v1') / 'development' / 'selected.json'
        heldout_lines = []
        if sel.exists() and (~per.test_participant.isin(json.loads(sel.read_text())['selection_participants'])).any():
            tuned = set(json.loads(sel.read_text())['selection_participants'])
            held = per[~per.test_participant.isin(tuned)]
            heldout_lines = ['### Only participants held out from PPG-rule tuning', '',
                             f'{held.test_participant.nunique()} participants: ' + ', '.join(sorted(held.test_participant.unique())), '',
                             endpoint_table(held).to_markdown(), '']
        pooled_best = tab['real'].idxmin()
        pooled = {c: dict(pick=tab[c].idxmin(), regret=tab.loc[tab[c].idxmin(), 'real'] - tab['real'].min())
                  for c in PROTO_ORDER if c in tab}
        lines += ['## Does the synthetic benchmark predict real-gap performance?', '',
                  'Per-participant mean with participant-bootstrap 95% CI.', '', summ.to_markdown(), ''] + heldout_lines + [
                  f'Pooled selection: best on real gaps = `{pooled_best}`. ' +
                  '; '.join(f'{k} picks `{v["pick"]}` (regret {v["regret"]:.2f} bpm)' for k, v in pooled.items()), '']
        # scatter synthetic vs real
        fig, axes = plt.subplots(1, len(PROTO_ORDER), figsize=(10, 2.9), sharex=True, sharey=True)
        lim = np.nanmax(tab.values) * 1.08
        for ax, c in zip(axes, PROTO_ORDER):
            style(ax); ax.grid(axis='y', color=GRID, lw=0.8)
            ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls='--')
            ax.scatter(tab[c], tab['real'], s=36, color=COLORS[c], edgecolor='white', lw=1.5, zorder=3)
            for name in tab.index[:3]:
                ax.annotate(name.replace('_', ' '), (tab.loc[name, c], tab.loc[name, 'real']), fontsize=7,
                            color=MUTED, xytext=(3, 3), textcoords='offset points')
            ax.set_title(c.replace('_', ' '), fontsize=10, color=INK)
            ax.set_xlabel('synthetic-gap MAE', color=INK, fontsize=9)
        axes[0].set_ylabel('real-gap MAE (bpm)', color=INK)
        fig.tight_layout(); fig.savefig(figs / 'synthetic_vs_real.png', dpi=200); plt.close(fig)

        bd = pd.read_csv(run / 'real_breakdown.csv')
        if len(bd):
            b = bd.groupby(['imputer', 'duration_bin']).apply(lambda g: g.abs_err_sum.sum() / g.n.sum(), include_groups=False).unstack()
            lines += ['## Real-gap MAE by gap duration', '', b.round(2).to_markdown(), '']
            b = bd.groupby(['imputer', 'movement_bin']).apply(lambda g: g.abs_err_sum.sum() / g.n.sum(), include_groups=False).unstack()
            lines += ['## Real-gap MAE by movement bin', '', b.round(2).to_markdown(), '']

    (run / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
