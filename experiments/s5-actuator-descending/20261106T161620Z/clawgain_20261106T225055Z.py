"""Stage C8: position-feedback map using ONLY the anatomically position-encoding claw channel.
Other sensory modalities (direction, vibration, touch, strain) are excluded from the injected
current because they do not encode joint position. Static open loop; joint never integrated.
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
COND=sys.argv[1];OFF=float(sys.argv[2]);PRE=float(sys.argv[3]);GAIN=float(sys.argv[4]);BIAS=.03;DT=.0078125;DUR=150
OFFS=[-.06,-.03,0.,.03,.06];assert COND in ['normal','sensory_cut'] and OFF in OFFS
O=ROOT/'clawgain/20261106T225055Z'/('g'+str(GAIN)+'_pre'+str(PRE)+'_'+COND+'_off'+str(OFF))
FLEX_PRE=[8179,8164,1496,11176,8422,6341];EXT_PRE=[11051,9574,6642,4476,4183,3003]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),Path(__file__).with_name('interval_readout.py'),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C8 claw-only position feedback',condition=COND,offset_rad=OFF,premotor_nA=PRE,bias_nA=BIAS,dynamic_gain=GAIN,dt_ms=DT,duration_ms=DUR,runs=1,injected_channels=['lf/claw_position'],zeroed_channels=['hook_direction','club_vibration','bristle_touch','campaniform_strain','hairplate_angle','multidendritic_noxious'],rationale='claw is the anatomically position-encoding channel and projects to the extensor; direction/vibration/touch do not encode position and were previously injected as if they did',criterion='GRADED_RESTORING on net_drive across the five offsets',hashes=hashes))
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0].copy();q0=float(qref[qa]);joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);ENC=ChannelSensoryEncoder(groups,12000,ChannelConfig())
 we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr()
 d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]=q0+OFF;d.qvel[:]=0;mj.mj_forward(m,d)
 raw=np.asarray(ENC.encode(observe_quantities(m,d,qref,joints,geom))[0]);claw_raw=np.zeros(12000);claw_raw[claw]=raw[claw]
 cur=np.zeros(12000);cur[claw]+=BIAS;cur[FLEX_PRE]+=PRE;cur[EXT_PRE]+=PRE
 if COND=='normal':cur=cur+GAIN*claw_raw
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);net.set_connectivity(we,wi)
 n=round(1/DT);dr=np.zeros(2);avg=np.zeros(2);packed=[];cmds=[]
 for k in range(round(DUR/DT)):
  if k%n==0:cmds.append(avg.copy());avg=np.zeros(2)
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp));forcing=np.array([sp[flex].mean(),sp[ext].mean()])/(DT/1000)/100
  dr,area=readout_step(dr,forcing,DT);avg=avg+area
 packed=np.array(packed);cmds=np.array(cmds);tail=cmds[-25:];sp=np.unpackbits(packed,axis=1)[:,:12000]
 r=dict(tag='g'+str(GAIN)+'_pre'+str(PRE)+'_'+COND+'_off'+str(OFF),condition=COND,offset_rad=OFF,premotor_nA=PRE,claw_added_nA=float(GAIN*float(raw[claw].max())) if COND=='normal' else 0.0,steady_flex=float(tail[:,0].mean()),steady_ext=float(tail[:,1].mean()),net_drive=float((tail[:,0]-tail[:,1]).mean()),flex_spikes=int(sp[:,flex].sum()),ext_spikes=int(sp[:,ext].sum()),network_spikes=int(sp.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed,commands=cmds);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
