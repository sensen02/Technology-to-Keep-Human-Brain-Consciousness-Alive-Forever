"""Signed claw-encoder response versus held joint angle. Forward kinematics only, no network."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from tendon_muscle import build,JOINT
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'polarity/20261106T213030Z'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
OFFS=[-.06,-.04,-.02,0.,.02,.04,.06]
s=json.loads((O/'SUMMARY.json').read_text());probed=sorted({(r['dt_ms'],r['offset_rad']) for r in s['records']})
m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0].copy();q0=float(qref[qa]);joints,geom=identities(m)
roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);ENC=ChannelSensoryEncoder(groups,12000,ChannelConfig())
rows=[]
for off in OFFS:
 d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]=q0+off;d.qvel[:]=0;mj.mj_forward(m,d)
 raw=np.asarray(ENC.encode(observe_quantities(m,d,qref,joints,geom))[0])
 rows.append(dict(offset_rad=off,claw_signed_min=float(raw[claw].min()),claw_signed_max=float(raw[claw].max()),claw_signed_mean=float(raw[claw].mean()),claw_nonzero_cells=int((raw[claw]!=0).sum()),whole_vector_absmax=float(np.abs(raw).max())))
pos=[r['claw_signed_mean'] for r in rows if r['offset_rad']>0];neg=[r['claw_signed_mean'] for r in rows if r['offset_rad']<0]
report=dict(importance='核心必需',verified=True,network_runs=0,rows=rows,encoder_sign_separates_directions=bool(np.sign(np.mean(pos))!=np.sign(np.mean(neg)) if pos and neg else False),note='determines whether the encoder itself is sign-blind, independent of the network response',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'ENCODER_SIGN.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))
