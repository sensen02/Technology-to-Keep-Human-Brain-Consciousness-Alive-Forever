"""Recover summary from already saved arrays; no model load or physics calls.
Original run finished all steps, failed serializing np.int64 torque-entry count.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype/tools')
import dynamic_support_metrics as dsm
P=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_physics')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(P/'BODY.npz') as z:body={k:z[k] for k in z.files}
with np.load(P/'NATIVE.npz') as z:n=z['rows']
with np.load(P/'SLIP.npz') as z:s=z['rows']
with np.load(P/'COMMANDS.npz') as z:c=z['rows']
prot=json.loads((P/'PROTOCOL.json').read_text())
assert len(c)==2422 and len(n)==193760 and len(body['time'])==2423
assert all(np.isfinite(v).all() for v in body.values()) and np.isfinite(n).all()
r=n[n[:,0]>=.2];height=float(body['com'][0,2]);gate=dsm.evaluate_native(dict(foot_support_mean=1.,com_z_mean=height),r)
gate['checks']['D1_no_nonfoot_ground_force']=bool(r[:,7].max()==0);gate['passed']=all(gate['checks'].values())
speed=float(np.linalg.norm(np.diff(body['com'][200:,:2],axis=0),axis=1).sum()/prot['source_duration_s']);mask=s[:,0]>=.2
slip=s[mask,7:13].sum(0)/np.maximum(s[mask,1:7].sum(0),1e-15)
result=dict(status='RECOVERED_ARRAY_DIAGNOSTIC_NOT_NEURAL_SUCCESS',native_steps=len(n),commands=len(c),sim_s=float(body['time'][-1]),reference_source=prot['source'],physical_gate=gate,
 nonfoot_force_max=float(n[:,7].max()),root_displacement=(body['qpos'][-1,:3]-body['qpos'][200,:3]).tolist(),posthold_COM_path_speed=speed,material_slip_weighted_speed=slip.tolist(),slip_to_pathspeed_ratio=(slip/max(speed,1e-12)).tolist(),
 initial_COM_height=height,COM_height_min=float(body['com'][:,2].min()),torque_saturated_entries=None,torque_total_entries=len(n)*42,
 warning_status='All per-command original finite/warning assertions passed before array saving; independent replay checks again',wall_s=None,
 recovery='bash177 exit1 AFTER full simulation/arrays: np.int64 saturation count JSON serialization failed. Originalpartial kept RESULT_INCOMPLETE.json. This script recalculates summary with0physics; saturation count can be recovered only by replay, no invented value.',
 files={p.name:sha(p) for p in P.iterdir() if p.is_file()},source_sha256=sha(Path(__file__)))
with (P/'RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
