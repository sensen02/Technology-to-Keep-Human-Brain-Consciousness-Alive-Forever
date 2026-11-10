"""Stage C1: search sensory-background range for a graded (monotone) restoring map.
Open-loop static polarity probe; joint never integrated. Screening only.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from tendon_muscle import build,JOINT,P
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
from interval_readout import step as readout_step

ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
BIAS=float(sys.argv[1]);GAIN=.1;DN=.1
OFFS=[-.06,-.03,0.,.03,.06];DT=.0078125;DUR=150
O=ROOT/'linearise/20261106T225055Z'/('b'+str(BIAS))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),Path(__file__).with_name('interval_readout.py'),Path(__file__).with_name('tendon_muscle.py'),ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C1 loop linearisation sweep',bias_nA=BIAS,offsets_rad=OFFS,dt_ms=DT,duration_ms=DUR,conditions=['normal','sensory_cut'],runs=len(OFFS)*2+1,dynamic_gain=GAIN,descending_nA=DN,efficacy_scale=.25,gravity=0,note='sensory_cut is offset-independent by construction; one duplicate cut at a second offset is run as a spot check',criterion='GRADED_RESTORING requires, at this bias: (1) baseline ext drive present at delta=0, i.e. net_drive(0) < 0; (2) net_drive strictly decreasing across offsets -.06>-.03>0>+.03>+.06; (3) total span net_drive(-.06)-net_drive(+.06) >= 1e-3; (4) no saturation to identical values at .03 and .06. Selection is by this map property only, never by physical recovery',hashes=hashes))
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0].copy();q0=float(qref[qa]);joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);ENC=ChannelSensoryEncoder(groups,12000,ChannelConfig())
 we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();records=[]
 runs=[('normal',o) for o in OFFS]+[('sensory_cut',0.),('sensory_cut',.06)]
 for cond,off in runs:
  d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]=q0+off;d.qvel[:]=0;mj.mj_forward(m,d)
  raw=np.asarray(ENC.encode(observe_quantities(m,d,qref,joints,geom))[0])
  cur=np.zeros(12000);cur[claw]+=BIAS;cur[cand]+=DN
  if cond=='normal':cur=cur+GAIN*raw
  np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);net.set_connectivity(we.copy(),wi.copy())
  n=round(1/DT);steps=round(DUR/DT);dr=np.zeros(2);avg=np.zeros(2);packed=[];cmds=[]
  for k in range(steps):
   if k%n==0:cmds.append(avg.copy());avg=np.zeros(2)
   sp=net.step(current_nA=cur);packed.append(np.packbits(sp));forcing=np.array([sp[flex].mean(),sp[ext].mean()])/(DT/1000)/100
   dr,area=readout_step(dr,forcing,DT);avg=avg+area
  packed=np.array(packed);cmds=np.array(cmds);tail=cmds[-25:]
  tag=f'{cond}_off{off}';np.savez_compressed(O/(tag+'.npz'),packed=packed,commands=cmds)
  r=dict(tag=tag,condition=cond,offset_rad=off,bias_nA=BIAS,steady_flex=float(tail[:,0].mean()),steady_ext=float(tail[:,1].mean()),net_drive=float((tail[:,0]-tail[:,1]).mean()),spikes=int(np.unpackbits(packed,axis=1)[:,:12000].sum()),finite=bool(np.isfinite(net.v).all()))
  records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,runs=len(records),physics_integrated=False,inputs_unchanged=True))
if __name__=='__main__':main()
