"""Preferred verified saved-sample admission for later parent training.

No simulation or training. Hashes bind both decision arrays and gate report to the
independent verifier's recomputation. Frozen running api.py is left unchanged.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
CASES=('halfspeed_straight','halfspeed_smallleft')

def load_verified_samples(case_name, *, post_only=False):
    if case_name not in CASES: raise ValueError('Unregistered case')
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    if not verification['all_integrity_checks_passed']:
        raise ValueError('Independent integrity verification did not pass')
    if not verification['recomputed'][case_name]['all_halfspeed']:
        raise ValueError('Independent replay reports failed halfspeed gates; not curriculum eligible')
    folder=HERE/case_name
    for name in ('RESULT.json','SAMPLES_5MS.npz'):
        p=folder/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=verification['artifact_sha256'][str(p)]:
            raise ValueError('Saved '+name+' changed after independent verification')
    result=json.loads((folder/'RESULT.json').read_text())
    if not result['all_halfspeed_gates'] or not all(result['halfspeed_gates'].values()):
        raise ValueError('Failed halfspeed case')
    with np.load(folder/'SAMPLES_5MS.npz',allow_pickle=False) as z:
        mask=z['post_prime'] if post_only else np.ones(len(z['actual_time']),dtype=bool)
        return {k:z[k][mask].copy() for k in z.files}

def load_teacher_api():
    """Explicit-path load of frozen teacher API, including baseline pinning."""
    spec=importlib.util.spec_from_file_location('_deceleration_frozen_teacher_api',HERE/'api.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
