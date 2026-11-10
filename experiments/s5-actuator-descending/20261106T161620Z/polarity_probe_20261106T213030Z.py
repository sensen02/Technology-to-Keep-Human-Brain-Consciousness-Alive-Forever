"""Stage B: static input-output polarity of the frozen loop versus held joint angle.
Diagnostic only: the joint is NOT integrated, sensory current is held constant per pose.
No external controller, no error-driven command generation.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
import mujoco as mj
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from tendon_muscle import build,NAMES,JOINT,P
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelConfig
from run_neural_muscle_closure import channel_index_map,observe_quantities
from run_integrated_repaired_closure import identities
from interval_readout import step as readout_step

ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'polarity/20261106T213030Z'
A=ROOT/'synaptic_scale/20261106T193236Z/0.25';H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
BIAS=.03;GAIN=.1;DN=.1
OFFSETS=[-.06,-.04,-.02,0.,.02,.04,.06];DTS=[.015625,.0078125];DUR=150
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),Path(__file__).with_name('interval_readout.py'),Path(__file__).with_name('tendon_muscle.py'),ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='B loop polarity map',held_joint_offsets_rad=OFFSETS,dt_ms=DTS,duration_ms=DUR,conditions=['normal','sensory_cut'],runs=28,bias_nA=BIAS,dynamic_gain=GAIN,descending_nA=DN,efficacy_scale=.25,gravity=0,cells=dict(claw=claw,descending=cand,flex=flex,ext=ext),protocol='joint held at q0+offset, qvel0, forward kinematics only; sensory current computed once and held for the whole probe; joint is never integrated, so this is an open-loop static polarity measurement, not control',net_drive_definition='D = flex_command - ext_command, positive drives toward larger joint angle (flexion)',sign_rule='negative feedback requires dD/d(offset) < 0, i.e. flexed poses must reduce net flexion drive; loop_gain = -dD/d(offset) > 0 restoring',criterion='report sign and magnitude at all seven poses and both dt without selecting favourable poses; identify any sign reversal and the restoring interval',hashes=hashes))
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0].copy();q0=float(qref[qa]);joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);ENC=ChannelSensoryEncoder(groups,12000,ChannelConfig())
 we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();records=[];packed_out={}
 for dt in DTS:
  for cond in ['normal','sensory_cut']:
   for off in OFFSETS:
    d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]=q0+off;d.qvel[:]=0;mj.mj_forward(m,d)
    raw=np.asarray(ENC.encode(observe_quantities(m,d,qref,joints,geom))[0])
    cur=np.zeros(12000);cur[claw]+=BIAS;cur[cand]+=DN
    if cond=='normal':cur=cur+GAIN*raw
    np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt);net.set_connectivity(we.copy(),wi.copy())
    n=round(1/dt);steps=round(DUR/dt);dr=np.zeros(2);avg=np.zeros(2);packed=[];cmds=[];sens=[]
    for k in range(steps):
     if k%n==0:cmds.append(avg.copy());sens.append(float(np.abs(raw[claw]).max()));avg=np.zeros(2)
     sp=net.step(current_nA=cur);packed.append(np.packbits(sp));forcing=np.array([sp[flex].mean(),sp[ext].mean()])/(dt/1000)/100
     dr,area=readout_step(dr,forcing,dt);avg=avg+area
    packed=np.array(packed);cmds=np.array(cmds);tail=cmds[-25:]
    tag=f'{cond}_dt{dt}_off{off}';np.savez_compressed(O/(tag+'.npz'),packed=packed,commands=cmds,sensory_absmax=np.array(sens))
    r=dict(tag=tag,condition=cond,dt_ms=dt,offset_rad=off,steady_flex=float(tail[:,0].mean()),steady_ext=float(tail[:,1].mean()),net_drive=float((tail[:,0]-tail[:,1]).mean()),claw_absmax_current_nA=float(np.abs(raw[claw]).max()),network_spikes=int(np.unpackbits(packed,axis=1)[:,:12000].sum()),finite=bool(np.isfinite(net.v).all()))
    records.append(r);packed_out[cond,dt,off]=r;print(json.dumps(r),flush=True)
 lines=[]
 for dt in DTS:
  for off in OFFSETS:
   delta=packed_out['normal',dt,off]['net_drive']-packed_out['sensory_cut',dt,off]['net_drive']
   lines.append(dict(dt_ms=dt,offset_rad=off,sensory_drive_delta=delta))
 slopes=[]
 for dt in DTS:
  xs=np.array(OFFSETS);ys=np.array([l['sensory_drive_delta'] for l in lines if l['dt_ms']==dt]);sl=np.gradient(ys,xs)
  for i,off in enumerate(OFFSETS):slopes.append(dict(dt_ms=dt,offset_rad=off,local_slope=float(sl[i]),loop_gain=float(-sl[i])))
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,paired=lines,slopes=slopes,dt_ms=DTS,offsets=OFFSETS,q0=q0,runs=len(records),physics_integrated=False,inputs_unchanged=True))
if __name__=='__main__':main()
