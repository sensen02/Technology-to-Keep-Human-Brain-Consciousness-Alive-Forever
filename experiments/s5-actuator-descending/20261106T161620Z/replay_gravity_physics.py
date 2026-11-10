import json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from tendon_muscle import build,JOINT
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/gravity_body/20261106T201513Z');checks=[]
for g in [.01,.05,.1]:
 folder=O/f'g{g}_v0.0';p=json.loads((folder/'PREREGISTRATION.json').read_text());source=folder/'muscle_tendon_zero_output_p250_dt0.0078125.npz';a=np.load(source)['rows'];m=build();m.opt.gravity[:]=p['gravity'];d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);mj.mj_forward(m,d);j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];actual=[]
 for k in range(250):
  mj.mj_forward(m,d)
  for step in range(80):
   idx=k*80+step;mj.mj_forward(m,d);d.ctrl[:]=0;d.ctrl[-2:]=-a[idx,9:11];mj.mj_step(m,d);actual.append([d.time,d.qpos[qa],d.qvel[va]])
 actual=np.array(actual);err=float(abs(actual-a[:,:3]).max());assert err<1e-12;checks.append(dict(gravity_fraction=g,max_state_error=err,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),nmesh=m.nmesh,nq=m.nq,nv=m.nv,neq=m.neq,gravity=m.opt.gravity.tolist(),body_mass_sha256=hashlib.sha256(m.body_mass.tobytes()).hexdigest()))
assert len({c['body_mass_sha256'] for c in checks})==1
report=dict(importance='核心必需',verified=True,physics_validation_replays=3,new_neural_runs=0,checks=checks,scope='independent forward simulation of recorded zero-command tendon forces, confirms gravity schedule and unchanged mass; not autonomous motion')
(O/'PHYSICS_REPLAY.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
