"""Extract only the sternum ECG from each ~1.2 GB WildPPG .mat into a small <pid>.npz (float32).

    python member1_ecg/extract_ecg.py --data-dir ../WildPPG --out-dir ../WildPPG/ecg_extract
process_ecg.py accepts either the full .mat files or these extracts via --data-dir.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', required=True)
ap.add_argument('--out-dir', required=True)
a = ap.parse_args()
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
for f in sorted(Path(a.data_dir).glob('WildPPG_Part_*.mat')):
    d = loadmat(f, variable_names=['id', 'notes', 'sternum'], simplify_cells=True)
    e = d['sternum']['ecg']
    notes = d.get('notes', '')
    notes = notes if isinstance(notes, str) else ''
    pid = f.stem.replace('WildPPG_Part_', '')
    np.savez_compressed(Path(a.out_dir) / f'{pid}.npz', ecg=np.asarray(e['v'], np.float32).reshape(-1),
                        fs=float(e['fs']), id=str(d['id']), notes=notes)
    print(pid, 'done')
