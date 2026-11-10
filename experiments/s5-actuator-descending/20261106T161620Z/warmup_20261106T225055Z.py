"""Stage D3: is the premotor current a warm-up or a permanent crutch?
Inject briefly, withdraw, and measure whether motor pools sustain firing afterwards.
Static probe (joint held at an offset, no physics integration).
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
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z');LOOK=Path('/home/sensen/Desktop/engine修复验收/s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json')
MODE=sys.argv[1];OFF=float(sys.argv[2]);GAIN=1.0;BIAS=.03;PRE=.3;DT=.0078125;DUR=200;WARM=100
OFFS=[-.06,0.,.06];assert MODE in ['sustained','warmup100','none'] and OFF in OFFS
O=ROOT/'warmup/20261106T225055Z'/(MODE+'_off'+str(OFF))
FLEX_PRE=[8179,8164,1496,11176,8422,6341];EXT_PRE=[11051,9574,6642,4476,4183,3003]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',P/'MODEL_JOINT_TORQUE.xml',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',LOOK,R/'engine/neural_cond.py',R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='D3 warm-up versus crutch',mode=MODE,offset_rad=OFF,sensory_gain=GAIN,premotor_nA=PRE,warmup_window_ms=[0,WARM] if MODE=='warmup100' else None,dt_ms=DT,duration_ms=DUR,readout_window_ms=[WARM,DUR],criterion='warm-up is established only if the post-withdrawal window has motor firing comparable to the sustained case and clearly above the no-premotor case; anything less is a crutch',hashes=hashes))
 m=build();j=m.joint(JOINT).id;qa=m.jnt_qposadr[j];qref=m.key_qpos[0].copy();q0=float(qref[qa]);joints,geom=identities(m)
 roots=np.load(H/'IDENTITIES.npz')['root_ids'];groups,_=channel_index_map(12000,roots,lookup_path=LOOK);ENC=ChannelSensoryEncoder(groups,12000,ChannelConfig())
 we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr()
 d=mj.MjData(m);mj.mj_resetDataKeyframe(m,d,0);d.qpos[qa]=q0+OFF;d.qvel[:]=0;mj.mj_forward(m,d)
 _raw=np.asarray(ENC.encode(observe_quantities(m,d,qref,joints,geom))[0]);_m=np.zeros(12000);_m[claw]=_raw[claw]
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);net.set_connectivity(we,wi)
 steps=round(DUR/DT);wsteps=round(WARM/DT);packed=[]
 for k in range(steps):
  cur=GAIN*_m;cur[claw]+=BIAS
  if MODE=='sustained' or (MODE=='warmup100' and k<wsteps):cur[FLEX_PRE]+=PRE;cur[EXT_PRE]+=PRE
  packed.append(np.packbits(net.step(current_nA=cur)))
 packed=np.array(packed);sp=np.unpackbits(packed,axis=1)[:,:12000]
 post=sp[wsteps:];pre_w=sp[:wsteps]
 r=dict(tag=MODE+'_off'+str(OFF),mode=MODE,offset_rad=OFF,flex_total=int(sp[:,flex].sum()),ext_total=int(sp[:,ext].sum()),flex_post=int(post[:,flex].sum()),ext_post=int(post[:,ext].sum()),flex_warm=int(pre_w[:,flex].sum()),ext_warm=int(pre_w[:,ext].sum()),network_post=int(post.sum()),network_warm=int(pre_w.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
