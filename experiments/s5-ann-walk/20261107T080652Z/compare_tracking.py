"""Matched finite-actuator gain diagnostic on the independently accepted v4 gait.
Only local kp/kd change. Original model, gravity, contacts and native torque caps
stay unchanged. All control is explicitly algorithm teacher in these screens.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
WRAPPER=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T080221Z/run_stance.py')
spec=importlib.util.spec_from_file_location('v4_stance_runtime',WRAPPER)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
n=b.n
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T080652Z')
n.OUT=OUT
OriginalActuator=n.IncrementalActuator


class DiagnosticActuator(OriginalActuator):
    scale=2.
    def __init__(self,*args,**kwargs):
        kwargs['kp']=np.asarray(kwargs['kp'])*(self.scale/2.)
        kwargs['kd']=np.asarray(kwargs['kd'])*np.sqrt(self.scale/2.)
        super().__init__(*args,**kwargs)


n.IncrementalActuator=DiagnosticActuator

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'PREREGISTRATION.json'
    if protocol.exists():raise FileExistsError('preserve prior matched comparison')
    n.dump(protocol,dict(importance='核心必需',stage='S5 teacher tracking and support diagnostic',
        context='V3 targetfeet lift but physical mid/hind feet mostly remainloaded; distinguish targetgeometry from actuator tracking.',
        gait=str(n.GAIT),gait_sha256=n.sha(n.GAIT),fk='v4 independent2049pose gates pass before thisrun',
        changes=dict(old_gain_scale=2.,tested_gain_scales=[2.,8.],
                     kp='native_limit/.2*scale',kd='native_limit*.005*sqrt(scale)',
                     torque_cap='unchanged original motor limits, no gain in torque authority'),
        protocol=dict(total_seconds=4.,frequency_hz=1.,controller='continuous algorithm teacher; no ANN outputs',
                      rate_cap_rad_s=30.,physics_dt_s=.0000125,actuator_dt_s=.001,policy_dt_s=.005),
        physics='original71mesh/fullfreebody, originalgravity/contact/friction/torquecaps, no rootcontrol/externalforces/statewrites',
        assessment='Unchanged fullrelease Dgates, sustained real unloadings, actual jointtrackingerror and torque saturation, materialpointslip via independent replay. No neural success claim.',
        sources={str(p):n.sha(p) for p in [Path(__file__),WRAPPER,Path(n.__file__),n.GAIT]},
        units='native; absolute SI force NOT_ESTABLISHED'))
    path=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z/MODEL.npz')
    for scale in [2.,8.]:
        DiagnosticActuator.scale=scale
        n.simulate(path,'teacher_gain'+str(int(scale))+'_4s',4.,'teacher',False)
