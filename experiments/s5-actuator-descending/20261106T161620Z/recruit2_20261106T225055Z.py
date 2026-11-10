"""Stage C3: can real descending cells recruit the two motor pools at higher current?
mode = 'pool' (all 19 audited candidates) or 'cell:<index>'. Static open loop, no physics.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
MODE=sys.argv[1];DN=float(sys.argv[2]);BIAS=.03;DT=.0078125;DUR=150
O=ROOT/'recruit2/20261106T225055Z'/(MODE.replace(':','')+'_dn'+str(DN))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 driven=cand if MODE=='pool' else [int(MODE.split(':')[1])];assert all(c in cand for c in driven)
 files=[Path(__file__),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C3 descending recruitment at higher current',mode=MODE,driven=driven,current_nA=DN,bias_nA=BIAS,offset_rad=0.,dt_ms=DT,duration_ms=DUR,runs=1,criterion='report per-pool spike counts only; higher current is a declared engineering input, not a physiological claim',hashes=hashes))
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();net.set_connectivity(we,wi)
 cur=np.zeros(12000);cur[claw]+=BIAS;cur[driven]+=DN
 packed=[]
 for k in range(round(DUR/DT)):
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp))
 packed=np.array(packed);sp=np.unpackbits(packed,axis=1)[:,:12000];tail=sp[round(100/DT):]
 r=dict(tag=MODE+'_dn'+str(DN),mode=MODE,driven=driven,current_nA=DN,flex_counts=sp[:,flex].sum(0).tolist(),ext_counts=sp[:,ext].sum(0).tolist(),flex_spikes=int(sp[:,flex].sum()),ext_spikes=int(sp[:,ext].sum()),flex_tail=int(tail[:,flex].sum()),ext_tail=int(tail[:,ext].sum()),network_spikes=int(sp.sum()),network_tail=int(tail.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
