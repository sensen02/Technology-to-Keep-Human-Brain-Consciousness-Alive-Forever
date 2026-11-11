"""Fixed-cadence sagittal stride teacher only; no simulation/training on import."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BASELINE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
TEACHER=HERE.parent/'teacher'/'turning_teacher.py'
_spec=importlib.util.spec_from_file_location('_stride_previous_turning_teacher',TEACHER)
previous=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=previous;_spec.loader.exec_module(previous)
previous.pin_baseline()
CONDITIONS=(dict(name='stride_018',stride_scale=.5,stride_native=.18,forward_threshold=.025),
            dict(name='stride_027',stride_scale=.75,stride_native=.27,forward_threshold=.04))

class FixedCadenceStrideReference:
    """Preserve measured lateral/vertical profiles, original time and stance."""
    def __init__(self,base,stride_scale,prime_s=4.8,ramp_s=1.):
        if stride_scale not in (.5,.75) or prime_s!=4.8 or ramp_s!=1.:
            raise ValueError('Exactly preregistered scales/timing only')
        self.base=base;self.rest=np.array(base.rest_body_feet,copy=True)
        self.stride_scale=float(stride_scale);self.prime_s=prime_s;self.ramp_s=ramp_s
        self.metadata=dict(label='algorithm teacher fixed cadence sagittal stride shortening',stride_scale=stride_scale,
                           phase_slowing=False,physical_state_writes=False,extra_forces=False)
    def scale(self,t):
        u=float(np.clip((t-self.prime_s)/self.ramp_s,0,1))
        blend=u*u*(3-2*u)
        return 1+(self.stride_scale-1)*blend,(self.stride_scale-1)*6*u*(1-u)/self.ramp_s
    def sample(self,t):
        sample=self.base.sample(t)
        scale,rate=self.scale(t)
        delta=sample['target_xy'][:,0]-self.rest[:,0]
        # Neutral prefix is bitexact, with no subtract/add roundtrip.
        if scale!=1 or rate!=0:
            sample['target_xy']=sample['target_xy'].copy()
            sample['velocity_xy']=sample['velocity_xy'].copy()
            sample['target_xy'][:,0]=self.rest[:,0]+scale*delta
            sample['velocity_xy'][:,0]=scale*sample['velocity_xy'][:,0]+rate*delta
        return sample

def make_teacher(core,case_name):
    from static_teacher import StaticTeacher
    from student_wave import teacher_config
    cfg=next(c for c in CONDITIONS if c['name']==case_name)
    teacher=StaticTeacher(core,teacher_config(100))
    teacher.reference=FixedCadenceStrideReference(teacher.reference,cfg['stride_scale'])
    teacher.metadata['fixed_cadence_stride']=teacher.reference.metadata
    return teacher

def command2(reference,elapsed_s):
    scale,_=reference.scale(elapsed_s)
    return np.array([scale-1,0.]),np.array([.075*scale,0.]),float(elapsed_s)

def stride_gates(summary,cycles,threshold):
    return dict(forward=summary['forward_signed_native_per_s']>=threshold,slip=summary['slip_ratio_max']<=.5,
        support=summary['foot_support_mean']>=.90,com=summary['com_height_min']>=1.20,
        strict_nonfoot_zero=summary['nonfoot_force_max_native']==0,
        valid_slip=bool(all(summary['slip_valid_per_leg']) and summary['slip_has_progress']),all6cycles=cycles['all_legs_completed'])

def load_verified_samples(case_name,post_only=False):
    if case_name not in [c['name'] for c in CONDITIONS]:raise ValueError('Unregistered case')
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    if not verification['all_integrity_checks_passed'] or not verification['recomputed'][case_name]['all_stride_gates']:
        raise ValueError('Case not independently verified passing stride evidence')
    for name in ('RESULT.json','SAMPLES_5MS.npz'):
        p=HERE/case_name/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=verification['artifact_sha256'][str(p)]:
            raise ValueError('Artifact changed after verification: '+name)
    with np.load(HERE/case_name/'SAMPLES_5MS.npz',allow_pickle=False) as z:
        mask=z['post_prime'] if post_only else np.ones(len(z['actual_time']),bool)
        return {k:z[k][mask].copy() for k in z.files}
