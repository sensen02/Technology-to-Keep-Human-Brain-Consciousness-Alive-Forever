"""Read-only compilation of the real plant; no simulation or state mutation."""
import hashlib
import json
from pathlib import Path
from flygym.compose.fly.musculoskeletal import _load_mjcf

SOURCE = Path('/home/sensen/Desktop/engine修复验收/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml')
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
m = _load_mjcf(SOURCE).compile()
assert (m.nq, m.nv, m.nu, m.nmesh, m.neq) == (49, 48, 42, 71, 0)
records = []
for a in range(m.nu):
    j = int(m.actuator_trnid[a, 0])
    records.append(dict(actuator=a, joint=m.joint(j).name,
        range_rad=m.jnt_range[j].tolist(), ctrlrange=m.actuator_ctrlrange[a].tolist(),
        gear=m.actuator_gear[a].tolist(), qpos_address=int(m.jnt_qposadr[j]),
        dof_address=int(m.jnt_dofadr[j]), gain=m.actuator_gainprm[a].tolist(),
        bias=m.actuator_biasprm[a].tolist()))
result = dict(importance='核心必需', stage='plant inspection, zero physics steps',
    source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,neq=m.neq),
    gravity=m.opt.gravity.tolist(), physics_dt_s=m.opt.timestep,
    candidate_joint='joint_LFTibia_pitch', selection_status='pending antagonist annotation audit',
    actuators=records, absolute_force_units='NOT_ESTABLISHED')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'PLANT_INSPECTION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',actuators=len(records),physics_steps=0,output=str(OUT/'PLANT_INSPECTION.json'))))
