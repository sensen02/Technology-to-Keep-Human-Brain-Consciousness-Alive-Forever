import json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
from tendon_muscle import build,JOINT
O=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z/holdout/20261106T200439Z');m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];stiffness=float(m.jnt_stiffness[j]);damping=float(m.dof_damping[va]);reference=float(m.qpos_spring[qa]);rows=[]
for offset in [-.025,0,.025]:
 for velocity in [-.1,0,.1]:
  d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]+=offset;d.qvel[va]=velocity;mj.mj_forward(m,d);predicted=stiffness*(reference-d.qpos[qa])-damping*velocity;actual=float(d.qfrc_passive[va]);assert abs(actual-predicted)<1e-12;rows.append(dict(offset=offset,velocity=velocity,actual_passive=actual,predicted_spring_damping=float(predicted)))
report=dict(importance='核心必需',verified=True,trajectory_runs=0,neural_runs=0,joint=JOINT,stiffness_native=stiffness,damping_native=damping,spring_reference_rad=reference,rows=rows,interpretation='inherited plant spring/damping contributes recovery, including zero neural output; normal-vs-cut measures incremental feedback contribution only; not neuron-only restoration',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(O/'PASSIVE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
