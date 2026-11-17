"""Zero-integration model/source qualification before bounded diagnostic physics."""
import platform
import sys
import importlib.metadata
import torch
from context import *

assert sha(CURRENT['checkpoint']) == CURRENT['sha256']
assert CURRENT['sha256']=='44e0ca425de238a95e0aafe8fa865ace39b5c3bbe5503039578910ea5f79af23'
body=body_run.Body()
body.at_start(119)
legs=body.fly.get_legs_order()
foot_ids=[body.sim._internal_bodyids_by_fly[body.name][body.fly.get_bodysegs_order().index(BodySegment(f'{leg}_tarsus5'))] for leg in legs]
actor=phase.read_policy(Path(CURRENT['checkpoint']))
files=[Path(__file__).resolve(),HERE/'context.py',HERE/'drive.sh',FIRST/'body_run.py',FIRST/'FROZEN_INIT.npz',
       PHASE/'phase.py',Path(CURRENT['checkpoint']),BASE/'BODY_CURRENT.json',Path(CURRENT['frozen_neck_parameters']),
       Path(CURRENT['fixed_readout']),BASE/'20261117T170820Z-body-readaptation/TEACHER/tr_cruise_0/TRACE.npz',
       BASE/'20261117T170820Z-body-readaptation/TEACHER/tr_cruise_0/RESULT.json']
files += [Path(mod.__file__).resolve() for mod in list(sys.modules.values()) if getattr(mod,'__file__',None)
          and str(ROOT) in str(mod.__file__) and str(mod.__file__).endswith('.py')]
files=list(dict.fromkeys(files))
joint_ids=body.m.actuator_trnid[body.actuators,0]
preflight=dict(status='MODEL_IDENTITY_QUALIFIED', physical_steps=0,initial_forward_calls=1,
 model=model_identity(body),legs=legs,feet_world=body.d.xpos[foot_ids].tolist(),
 thorax_world=body.d.xpos[body.thorax].tolist(),body_x_extent=[float(body.d.geom_xpos[:,0].min()),float(body.d.geom_xpos[:,0].max())],
 residual_limit_rad=phase.RESIDUAL_LIMIT, versions=dict(python=platform.python_version(),torch=torch.__version__,
 mujoco=mujoco.__version__,flygym=importlib.metadata.version('flygym')),exe=sys.executable,
 source_sha256={str(p):sha(p) for p in files})
write(HERE/'PREFLIGHT.json',preflight)
print(json.dumps({k:preflight[k] for k in ('status','versions','feet_world','thorax_world','body_x_extent','residual_limit_rad')},ensure_ascii=False))
print('model',body.m.nq,body.m.nv,body.m.nu,'force range',body.m.actuator_forcerange[body.actuators].min(),body.m.actuator_forcerange[body.actuators].max())
