"""Stage C7: local slope of claw-drive -> motor drive at a fixed premotor baseline.
Sweeps the added claw current around the current bias to find whether a usable gradient exists.
Static open loop; joint never integrated; no external controller.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from interval_readout import step as readout_step
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
ADD=float(sys.argv[1]);PRE=float(sys.argv[2]);BIAS=.03;DT=.0078125;DUR=150
O=ROOT/'slope/20261106T225055Z'/('pre'+str(PRE)+'_add'+str(ADD))
FLEX_PRE=[8179,8164,1496,11176,8422,6341];EXT_PRE=[11051,9574,6642,4476,4183,3003]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),Path(__file__).with_name('interval_readout.py'),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C7 local claw-to-motor slope',added_claw_nA=ADD,premotor_nA=PRE,bias_nA=BIAS,dt_ms=DT,duration_ms=DUR,runs=1,claw_cells=claw,criterion='report steady flex/ext drive; a usable proportional pathway needs net_drive to be a monotone function of the added claw current over the range the joint actually produces',hashes=hashes))
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();net.set_connectivity(we,wi)
 cur=np.zeros(12000);cur[claw]+=BIAS+ADD;cur[FLEX_PRE]+=PRE;cur[EXT_PRE]+=PRE
 n=round(1/DT);dr=np.zeros(2);avg=np.zeros(2);packed=[];cmds=[]
 for k in range(round(DUR/DT)):
  if k%n==0:cmds.append(avg.copy());avg=np.zeros(2)
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp));forcing=np.array([sp[flex].mean(),sp[ext].mean()])/(DT/1000)/100
  dr,area=readout_step(dr,forcing,DT);avg=avg+area
 packed=np.array(packed);cmds=np.array(cmds);tail=cmds[-25:]
 sp=np.unpackbits(packed,axis=1)[:,:12000]
 r=dict(tag='pre'+str(PRE)+'_add'+str(ADD),added_claw_nA=ADD,premotor_nA=PRE,steady_flex=float(tail[:,0].mean()),steady_ext=float(tail[:,1].mean()),net_drive=float((tail[:,0]-tail[:,1]).mean()),flex_spikes=int(sp[:,flex].sum()),ext_spikes=int(sp[:,ext].sum()),network_spikes=int(sp.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed,commands=cmds);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
