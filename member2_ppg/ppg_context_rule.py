"""PPG-only recovery supported by multiple original high-confidence anchors.

No ECG, half-rate features, inferred HR, or recovered anchor enters a decision.
The optional pulse corroboration uses direct timing agreement, never doubling.
"""
from dataclasses import dataclass

import numpy as np

from ppg_spectral_rule import components, apply_spectral_rule


PULSE_SUPPORT = {
    'none': None,
    'strict_one': (.3, .7, 5., 1),
    'loose_one': (.5, .5, 10., 1),
    'loose_two': (.5, .5, 5., 2),
}


@dataclass(frozen=True)
class ContextRule:
    anchor_min_acf: float = .7
    anchor_min_concentration: float = .1
    anchor_max_difference_bpm: float = 6.
    radius_s: float = 120.
    max_deviation_bpm: float = 8.
    anchors_each_side: int = 2
    anchor_separation_s: float = 8.
    candidate_min_acf: float = .1
    candidate_min_concentration: float = .1
    candidate_max_difference_bpm: float = 40.
    pulse_support: str = 'none'


def prepare(table, base_config):
    p = components(table, base_config)
    p['time'] = table.center_s.to_numpy(float)
    if len(table) > 1 and not np.allclose(np.diff(p['time']), base_config['stride_s']):
        raise ValueError('Context requires a complete ordered window grid')
    if 'participant_id' in table and table.participant_id.nunique() != 1:
        raise ValueError('Context cannot cross participant boundaries')
    p['data_hard'] = (table.finite_ratio.eq(1).to_numpy() & ~table.documented_exclusion.to_numpy(bool)
                      & table.filter_safe.to_numpy(bool)
                      & table.longest_flat_run_s.lt(base_config['max_flat_run_s']).to_numpy())
    p['blocked'] = np.r_[0, np.cumsum(~p['data_hard'])]
    p['context_cache'] = {}
    p['pulse_cache'] = {}
    p['pulses'] = None
    names = ['hr', 'interval_cv', 'template_corr', 'prominence_cv', 'n_peaks']
    if all(f'p{j}_{name}' in table for j in range(3) for name in names):
        p['pulses'] = [{name: table[f'p{j}_{name}'].to_numpy(float) for name in names} for j in range(3)]
    return p


def context_mask(p, cfg):
    if cfg.anchors_each_side not in (1, 2) or cfg.radius_s <= 0 or cfg.max_deviation_bpm <= 0 or cfg.anchor_separation_s < 8:
        raise ValueError('Require one or two anchors per side, positive range, and >=8-second separation')
    key = (cfg.anchor_min_acf, cfg.anchor_min_concentration, cfg.anchor_max_difference_bpm,
           cfg.radius_s, cfg.max_deviation_bpm, cfg.anchors_each_side, cfg.anchor_separation_s)
    if key in p['context_cache']:
        return p['context_cache'][key]
    t, h = p['time'], p['hr']
    anchors = np.flatnonzero(p['base'] & (p['acf'] >= cfg.anchor_min_acf)
                             & (p['concentration'] >= cfg.anchor_min_concentration)
                             & (p['difference'] <= cfg.anchor_max_difference_bpm))
    if len(anchors) < 2:
        result = np.zeros(len(h), bool)
        p['context_cache'][key] = result
        return result
    idx = np.arange(len(h)); pos = np.searchsorted(anchors, idx)
    left = anchors[np.clip(pos - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(pos, 0, len(anchors) - 1)]
    result = ((left < idx) & (right > idx) & (t - t[left] <= cfg.radius_s)
              & (t[right] - t <= cfg.radius_s)
              & (np.abs(h - h[left]) <= cfg.max_deviation_bpm)
              & (np.abs(h - h[right]) <= cfg.max_deviation_bpm))
    anchor_values = [h[left], h[right]]
    if cfg.anchors_each_side == 2:
        lpos = np.searchsorted(t[anchors], t[left] - cfg.anchor_separation_s, side='right') - 1
        rpos = np.searchsorted(t[anchors], t[right] + cfg.anchor_separation_s, side='left')
        l2 = anchors[np.clip(lpos, 0, len(anchors) - 1)]
        r2 = anchors[np.clip(rpos, 0, len(anchors) - 1)]
        result &= ((lpos >= 0) & (rpos < len(anchors)) & (t - t[l2] <= cfg.radius_s)
                   & (t[r2] - t <= cfg.radius_s)
                   & (np.abs(h - h[l2]) <= cfg.max_deviation_bpm)
                   & (np.abs(h - h[r2]) <= cfg.max_deviation_bpm))
        anchor_values += [h[l2], h[r2]]
        left, right = l2, r2
    result &= np.maximum.reduce(anchor_values) - np.minimum.reduce(anchor_values) <= cfg.max_deviation_bpm
    result &= p['blocked'][right + 1] == p['blocked'][left]
    p['context_cache'][key] = result
    return result


def pulse_mask(p, name):
    if name not in PULSE_SUPPORT:
        raise ValueError('Unknown pulse corroboration option')
    if name == 'none':
        return np.ones(len(p['hr']), bool)
    if p['pulses'] is None:
        raise ValueError('Pulse corroboration requires measured pulse features')
    if name not in p['pulse_cache']:
        cv, corr, difference, minimum_count = PULSE_SUPPORT[name]
        count = np.zeros(len(p['hr']), int)
        for pulse in p['pulses']:
            count += ((pulse['n_peaks'] >= 5) & (pulse['interval_cv'] <= cv)
                      & (pulse['template_corr'] >= corr) & (pulse['prominence_cv'] <= 1.)
                      & (np.abs(pulse['hr'] - p['hr']) <= difference))
        p['pulse_cache'][name] = count >= minimum_count
    return p['pulse_cache'][name]


def masks(p, cfg):
    if cfg is None:
        return p['base'].copy(), np.zeros(len(p['hr']), bool), p['hr'].copy()
    if not (0 <= cfg.anchor_min_acf <= 1 and 0 <= cfg.anchor_min_concentration <= 1
            and 0 <= cfg.candidate_min_acf <= 1 and 0 <= cfg.candidate_min_concentration <= 1
            and cfg.anchor_max_difference_bpm > 0 and cfg.candidate_max_difference_bpm > 0):
        raise ValueError('Invalid context quality thresholds')
    added = (p['hard'] & ~p['base'] & context_mask(p, cfg) & pulse_mask(p, cfg.pulse_support)
             & (p['acf'] >= cfg.candidate_min_acf)
             & (p['concentration'] >= cfg.candidate_min_concentration)
             & (p['difference'] <= cfg.candidate_max_difference_bpm))
    return p['base'] | added, added, p['hr'].copy()


def apply_context_rule(table, base_config, cfg):
    accepted, added, candidate = masks(prepare(table, base_config), cfg)
    d = apply_spectral_rule(table, base_config, None)
    d['pre_context_recovery_reject_reasons'] = d.reject_reasons.copy()
    d['context_rescued'], d['ppg_valid'] = added, accepted
    d.loc[added, 'reject_reasons'] = ''
    d['hr_ppg_candidate'] = candidate
    d['hr_ppg'] = np.where(accepted, candidate, np.nan)
    d['quality_route'] = np.where(added, 'original_high_confidence_anchors', np.where(d.base_ppg_valid, 'original_v2', 'rejected'))
    d['rule_version'] = 'ppg_only_dev_v2_context'
    return d
