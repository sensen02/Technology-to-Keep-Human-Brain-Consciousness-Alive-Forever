"""Explicit-path unmodified moderate phase-slowdown teacher API."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BASELINE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
TEACHER=HERE.parent/'teacher'/'turning_teacher.py'
_spec=importlib.util.spec_from_file_location('_moderate_previous_turning_teacher',TEACHER)
previous=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=previous;_spec.loader.exec_module(previous);previous.pin_baseline()
CONDITIONS=(dict(name='straight_control_100',speed_scale=1.,yaw_rate=0.),dict(name='straight_085',speed_scale=.85,yaw_rate=0.),dict(name='straight_090',speed_scale=.90,yaw_rate=0.))
REFERENCE_REPORT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T093347Z/early_unload/RESULT.json')
REFERENCE_FORWARD=.06881632432222345
# Initial historical-reference threshold superseded BEFORE physics by amendment.
# Functional reduction uses actual matched control passed explicitly below.

def make_teacher(core,case_name):
    cfg=next(c for c in CONDITIONS if c['name']==case_name)
    return previous.make_teacher(core,speed_scale=cfg['speed_scale'],yaw_rate=0.,prime_s=4.8)

def command2(reference,elapsed_s):
    phase,scale,curvature=reference.command(elapsed_s)
    return np.array([scale-1.,0.]),np.array([.075*scale,0.]),phase

def functional_gates(summary,cycles,control_forward=None,control_passed=True):
    gates=dict(**summary['gates'],strict_nonfoot_zero=summary['nonfoot_force_max_native']==0,
        valid_slip=bool(all(summary['slip_valid_per_leg']) and summary['slip_has_progress']),all6cycles=cycles['all_legs_completed'])
    if control_forward is not None:
        gates.update(matched_control_passed=bool(control_passed),deceleration=summary['forward_signed_native_per_s']<=.95*control_forward)
    return gates

def load_verified_samples(case_name,post_only=False):
    if case_name not in [c['name'] for c in CONDITIONS]:raise ValueError('Unregisteredcase')
    v=json.loads((HERE/'VERIFICATION.json').read_text())
    if not v['all_integrity_checks_passed'] or not v['recomputed'][case_name]['all_functional_gates']:
        raise ValueError('Independentreplay didnot establish functionaldeceleration')
    for name in ('RESULT.json','SAMPLES_5MS.npz'):
        p=HERE/case_name/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=v['artifact_sha256'][str(p)]:raise ValueError('Changedartifact '+name)
    with np.load(HERE/case_name/'SAMPLES_5MS.npz',allow_pickle=False) as z:
        mask=z['post_prime'] if post_only else np.ones(len(z['actual_time']),bool)
        return {k:z[k][mask].copy() for k in z.files}
