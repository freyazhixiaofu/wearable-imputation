"""Apply and verify the final v2 PPG rule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import types

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = 'member2_ppg/results/batch_v2'
CONFIG_PATH = BATCH_PATH + '/config.json'
MANIFEST_PATH = BATCH_PATH + '/batch_manifest.json'
RULE_VERSION = 'ppg_only_dev_v2'
EXPECTED_SETTINGS_SHA256 = 'faf1938d5a146feebca936229c6cf4126d1b14c234e8c1508705d3583b576ad0'
EXPECTED_SOURCES = {
    'ppg_context_rule.py': '37f418f00e229ee212f1dda641117dac80786eaefb4da1f5b21b5304678b61e7',
    'ppg_spectral_rule.py': 'b27ce24bbfaf7dc51bab2d681cac0cfd97f6c4431889a44c0c1bf5cf5f5ec625',
}
DEVELOPMENT = ['an0', 'e61', 'k2s', 'kjd', 'n31', 'qm9', 'u7y', 'w4p']
EVALUATION = ['fex', 'l38', 'ngh', 'p5d', 'p9p', 'ssx', 'trh', 'tz8']
_IMPORT_LOCK = threading.RLock()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def repo_path(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError('Batch paths must be repository-relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Batch path escapes the repository')
    return path


def settings_sha(config):
    settings = {name: config[name] for name in ['base_config', 'context_rule']}
    data = json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def resolve_batch(root=ROOT):
    """Check the final rule and settings without requiring participant tables."""
    root = Path(root).resolve()
    manifest = json.loads(repo_path(root, MANIFEST_PATH).read_text(encoding='utf-8'))
    if (manifest['release_name'] != 'v2' or manifest['rule_version'] != RULE_VERSION
            or manifest['config_file'] != CONFIG_PATH):
        raise ValueError('Unexpected final v2 identity')
    config_path = repo_path(root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if settings_sha(config) != EXPECTED_SETTINGS_SHA256:
        raise ValueError('Final v2 thresholds differ from the evaluated settings')
    if config['rule_version'] != RULE_VERSION or config['release_name'] != 'v2':
        raise ValueError('Unexpected final v2 settings identity')
    if sha(config_path) != manifest['config_sha256']:
        raise ValueError('Final v2 settings file changed')
    if manifest['config'] != config:
        raise ValueError('Batch manifest and settings disagree')
    for name, digest in EXPECTED_SOURCES.items():
        relative = 'member2_ppg/' + name
        if (manifest['source_files'].get(relative) != digest
                or sha(repo_path(root, relative)) != digest):
            raise ValueError('Final v2 decision source changed: ' + name)
    if (manifest['development_participants'] != DEVELOPMENT
            or manifest['evaluation_participants'] != EVALUATION):
        raise ValueError('The final v2 participant split changed')
    participants = manifest['participant_files']
    if sorted(participants) != sorted(DEVELOPMENT + EVALUATION):
        raise ValueError('Final v2 must index the exact 16 participants')
    if sorted(manifest['participants']) != sorted(participants):
        raise ValueError('Batch participant lists disagree')
    for pid, entry in participants.items():
        expected_split = 'development' if pid in DEVELOPMENT else 'evaluation'
        if entry['split'] != expected_split:
            raise ValueError('Incorrect participant split: ' + pid)
        for name in ['windows', 'gaps']:
            expected = BATCH_PATH + '/' + pid + '/' + name + '.csv'
            if entry[name] != expected:
                raise ValueError('Unexpected participant output path: ' + pid)
            repo_path(root, entry[name])
    return manifest


def get_participant_paths(pid, root=ROOT):
    manifest = resolve_batch(root)
    if pid not in manifest['participant_files']:
        raise ValueError('Unknown participant')
    entry = manifest['participant_files'][pid]
    return dict(windows=repo_path(root, entry['windows']),
                gaps=repo_path(root, entry['gaps']), split=entry['split'])


def _rule_module(root):
    source = repo_path(root, 'member2_ppg')
    names = ['ppg_context_rule', 'ppg_spectral_rule']
    with _IMPORT_LOCK:
        previous = {name: sys.modules.pop(name, None) for name in names}
        try:
            for name in ['ppg_spectral_rule', 'ppg_context_rule']:
                path = source / (name + '.py')
                module = types.ModuleType(name)
                module.__file__ = str(path)
                module.__package__ = ''
                sys.modules[name] = module
                exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
            return sys.modules['ppg_context_rule']
        finally:
            for name in names:
                sys.modules.pop(name, None)
                if previous[name] is not None:
                    sys.modules[name] = previous[name]


def apply_final_v2(table, root=ROOT):
    """Apply final v2 to one participant's complete ordered PPG feature grid."""
    resolve_batch(root)
    config = json.loads(repo_path(root, CONFIG_PATH).read_text(encoding='utf-8'))
    module = _rule_module(root)
    result = module.apply_context_rule(
        table, config['base_config'], module.ContextRule(**config['context_rule']))
    result['rule_version'] = RULE_VERSION
    return result


def load_participant(pid, root=ROOT):
    paths = get_participant_paths(pid, root)
    return pd.read_csv(paths['windows'])


def verify_rule(root=ROOT):
    manifest = resolve_batch(root)
    return dict(rule_version=RULE_VERSION, participants=len(manifest['participants']),
                settings_and_sources_unchanged=True, training=False)


def verify_batch(root=ROOT):
    """Also check saved participant outputs and evaluation evidence."""
    manifest = resolve_batch(root)
    checked = 0
    for entry in manifest['participant_files'].values():
        for name in ['windows', 'gaps']:
            if sha(repo_path(root, entry[name])) != entry[name + '_sha256']:
                raise ValueError('A final v2 participant output changed: ' + entry[name])
            checked += 1
    for entry in manifest.get('evidence', {}).values():
        if sha(repo_path(root, entry['path'])) != entry['sha256']:
            raise ValueError('Final v2 evaluation evidence changed: ' + entry['path'])
        checked += 1
    return dict(rule_version=RULE_VERSION, participants=len(manifest['participants']),
                hashed_artifacts_checked=checked,
                settings_and_sources_unchanged=True, training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-rule', action='store_true',
                        help='Check final v2 code and settings')
    parser.add_argument('--verify', action='store_true',
                        help='Also check the saved participant outputs')
    parser.add_argument('--participant', choices=sorted(DEVELOPMENT + EVALUATION))
    parser.add_argument('--out-dir', type=Path, help='New participant output directory')
    args = parser.parse_args()
    if args.verify_rule:
        print(json.dumps(verify_rule(), indent=2))
    if args.verify:
        print(json.dumps(verify_batch(), indent=2))
    if args.participant:
        if args.out_dir is None:
            parser.error('--participant requires --out-dir')
        if args.out_dir.exists():
            parser.error('Output directory already exists')
        try:
            from .export_final_v2 import write_participant
        except ImportError:
            from export_final_v2 import write_participant
        source = ROOT / 'member2_ppg/results/feature_cache_v2' / args.participant / 'windows.csv'
        result = apply_final_v2(pd.read_csv(source))
        config = json.loads((ROOT / CONFIG_PATH).read_text(encoding='utf-8'))
        write_participant(result, args.participant, args.out_dir, config)
        metadata_path = args.out_dir / 'manifest.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(input_features=str(source), input_features_sha256=sha(source),
                        settings_file=CONFIG_PATH, training=False)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('Final v2 participant output: ' + str(args.out_dir))
    elif args.out_dir:
        parser.error('--out-dir requires --participant')
    elif not (args.verify or args.verify_rule):
        parser.print_help()


if __name__ == '__main__':
    main()

