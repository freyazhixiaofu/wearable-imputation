"""Two-route PPG acceptance rule. No ECG, motion, neighbours or harmonic flags.

The existing v2 route is preserved. A second route can accept weaker ACF only
when spectral concentration, entropy and direct spectral/ACF agreement pass.
Every accepted HR is the measured spectral HR; no HR is doubled or filled in.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectralBranch:
    min_concentration: float = .6
    min_acf: float = .4
    max_difference_bpm: float = 6.
    max_entropy: float = .6


def components(table, base_config):
    if base_config.get('half_rate_action', 'off') != 'off' or base_config.get('rescue_min_acf_peak') is not None:
        raise ValueError('Spectral rule requires original v2 with half-rate and neighbour recovery OFF')
    h = table.hr_spectral_bpm.to_numpy(float)
    acf_hr = table.hr_acf_bpm.to_numpy(float)
    concentration = table.raw_spectral_concentration.to_numpy(float)
    acf = table.acf_peak.to_numpy(float)
    entropy = table.spectral_entropy.to_numpy(float)
    difference = np.abs(h - acf_hr)
    hard = (table.finite_ratio.eq(1).to_numpy() & ~table.documented_exclusion.to_numpy(bool)
            & table.filter_safe.to_numpy(bool)
            & table.longest_flat_run_s.lt(base_config['max_flat_run_s']).to_numpy()
            & np.isfinite(h) & np.isfinite(acf_hr) & np.isfinite(concentration) & np.isfinite(acf)
            & (h > base_config['hr_min_bpm'] + base_config['search_boundary_margin_bpm'])
            & (h < base_config['hr_max_bpm'] - base_config['search_boundary_margin_bpm']))
    base = (hard & (concentration >= base_config['min_spectral_concentration'])
            & (acf >= base_config['min_acf_peak']) & (difference <= base_config['max_hr_difference_bpm']))
    return dict(hr=h, concentration=concentration, acf=acf, entropy=entropy,
                difference=difference, hard=hard, base=base)


def masks(parts, branch):
    if branch is None:
        return parts['base'].copy(), np.zeros(len(parts['hr']), bool)
    if (not 0 <= branch.min_concentration <= 1 or not 0 <= branch.min_acf <= 1
            or not 0 <= branch.max_entropy <= 1 or branch.max_difference_bpm <= 0):
        raise ValueError('Invalid spectral branch thresholds')
    secondary = (parts['hard'] & (parts['concentration'] >= branch.min_concentration)
                 & (parts['acf'] >= branch.min_acf) & (parts['difference'] <= branch.max_difference_bpm)
                 & np.isfinite(parts['entropy']) & (parts['entropy'] <= branch.max_entropy))
    added = secondary & ~parts['base']
    return parts['base'] | added, added


def apply_spectral_rule(table, base_config, branch):
    parts = components(table, base_config)
    accepted, added = masks(parts, branch)
    d = table.copy()
    reasons = np.full(len(d), '', dtype=object)
    checks = [('nonfinite_ppg', ~d.finite_ratio.eq(1).to_numpy()),
              ('documented_exclusion', d.documented_exclusion.to_numpy(bool)),
              ('filter_edge_or_unavailable', ~d.filter_safe.to_numpy(bool)),
              ('flatline_or_plateau', d.longest_flat_run_s.ge(base_config['max_flat_run_s']).to_numpy()),
              ('hr_or_quality_unavailable', ~np.isfinite(parts['hr']) | ~np.isfinite(parts['difference'])
               | ~np.isfinite(parts['concentration']) | ~np.isfinite(parts['acf'])),
              ('hr_search_boundary', (parts['hr'] <= base_config['hr_min_bpm'] + base_config['search_boundary_margin_bpm'])
               | (parts['hr'] >= base_config['hr_max_bpm'] - base_config['search_boundary_margin_bpm'])),
              ('low_spectral_concentration', parts['concentration'] < base_config['min_spectral_concentration']),
              ('weak_periodicity', parts['acf'] < base_config['min_acf_peak']),
              ('hr_method_disagreement', parts['difference'] > base_config['max_hr_difference_bpm'])]
    for name, failed in checks:
        reasons[failed] = np.where(reasons[failed] == '', name, reasons[failed] + ';' + name)
    d['pre_spectral_recovery_reject_reasons'] = reasons.copy()
    reasons[accepted] = ''
    if np.any(~accepted & (reasons == '')):
        raise ValueError('Rejected window lacks an auditable reason')
    d['base_ppg_valid'], d['spectral_rescued'], d['ppg_valid'] = parts['base'], added, accepted
    d['reject_reasons'] = reasons
    d['quality_route'] = np.where(added, 'spectral_branch', np.where(parts['base'], 'original_v2', 'rejected'))
    d['hr_ppg_candidate'] = parts['hr']
    d['hr_ppg'] = np.where(accepted, parts['hr'], np.nan)
    d['spectral_concentration'], d['hr_method_difference_bpm'] = parts['concentration'], parts['difference']
    d['sqi_score'] = np.minimum(parts['concentration'], parts['acf']).clip(0, 1)
    d['rule_version'] = 'ppg_only_dev_v2_spectral'
    return d
