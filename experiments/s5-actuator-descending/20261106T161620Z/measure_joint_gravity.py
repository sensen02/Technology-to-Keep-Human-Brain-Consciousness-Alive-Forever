import json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from tendon_muscle import build,P,JOINT,NAMES,_load_mjcf
R=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=R/'holdout/20261106T200439Z';assert json.loads((O/'VERIFICATION.json').read_text())['holdout_gate_passed']
original=_load_mjcf(P/'MODEL_JOINT_TORQUE.xml').compile();gravity=original.opt.gravity.copy();m=build();d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];q0=d.qpos[qa];tids=[m.tendon(n).id for n in NAMES];fmax=np.array(json.loads((R/'tendon_muscle/PREREGISTRATION.json').read_text())['fmax']);rows=[]
for offset in [-.025,0,.025]:
 lengths=[]
 for sign in [-1,1]:
  d.qpos[qa]=q0+offset+sign*1e-6;mj.mj_forward(m,d);lengths.append(d.ten_length[tids].copy())
 slopes=(lengths[1]-lengths[0])/2e-6;d.qpos[qa]=q0+offset
 for fraction in [0,.01,.05,.1,1]:
  m.opt.gravity[:]=gravity*fraction;mj.mj_forward(m,d);rows.append(dict(angle_offset_rad=offset,gravity_fraction=fraction,qfrc_bias=float(d.qfrc_bias[va]),qfrc_passive=float(d.qfrc_passive[va]),static_required_active_torque=float(d.qfrc_bias[va]-d.qfrc_passive[va]),signed_tendon_torque_caps=(-slopes*fmax).tolist()))
report=dict(importance='核心必需',physics_forward_only=True,trajectory_runs=0,neural_runs=0,full_gravity=gravity.tolist(),rows=rows,scope='native model units, SI NOT_ESTABLISHED; local static joint requirement with other joints/root held, not standing or dynamic capacity proof',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'GRAVITY_LOAD_MEASUREMENT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
