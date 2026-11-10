"""Stage C1 fallback: find the descending tonic level at which the flexor pool is recruited.
Static probe at delta=0; joint never integrated. Reports per-pool rates, not a control claim.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
DN=float(sys.argv[1]);BIAS=.03;DT=.0078125;DUR=150
O=ROOT/'recruit/20261106T225055Z'/('dn'+str(DN))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1']
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C1 recruitment threshold scan',descending_nA=DN,bias_nA=BIAS,offset_rad=0.,dt_ms=DT,duration_ms=DUR,conditions=['normal'],runs=1,purpose='measure whether and how the flexor pool is recruited as the declared descending tonic rises; static open loop, no physics, no control claim',criterion='report flex and ext rates and spike counts without selecting on physical performance',hashes=hashes))
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();net.set_connectivity(we,wi)
 cur=np.zeros(12000);cur[claw]+=BIAS;cur[cand]+=DN
 packed=[]
 for k in range(round(DUR/DT)):
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp))
 packed=np.array(packed);sp=np.unpackbits(packed,axis=1)[:,:12000]
 flex_counts=sp[:,flex].sum(0);ext_counts=sp[:,ext].sum(0);tail=sp[round(100/DT):]
 r=dict(tag='normal_off0.0',descending_nA=DN,bias_nA=BIAS,flex_counts=flex_counts.tolist(),ext_counts=ext_counts.tolist(),flex_spikes=int(flex_counts.sum()),ext_spikes=int(ext_counts.sum()),flex_tail_spikes=int(tail[:,flex].sum()),ext_tail_spikes=int(tail[:,ext].sum()),network_spikes=int(sp.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'normal_off0.0.npz',packed=packed);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
