"""Stage C2: per-cell descending routing profile at delta=0.
Each of the 19 audited descending candidates is driven ALONE at the same per-cell current
already used by the joint tonic interface. No connection is invented; static open loop.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
K=int(sys.argv[1]);BIAS=.03;DN=.1;DT=.0078125;DUR=150
O=ROOT/'routing/20261106T225055Z'/('cell'+str(K))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1'];assert K in cand
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 files=[Path(__file__),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C2 per-cell descending routing',driven_cell=K,bias_nA=BIAS,descending_nA_on_single_cell=DN,offset_rad=0.,dt_ms=DT,duration_ms=DUR,runs=1,criterion='classify by tail-window (100-150ms) spike counts only; no physical performance, no invented edges',hashes=hashes))
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr();net.set_connectivity(we,wi)
 cur=np.zeros(12000);cur[claw]+=BIAS;cur[K]+=DN
 packed=[]
 for k in range(round(DUR/DT)):
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp))
 packed=np.array(packed);sp=np.unpackbits(packed,axis=1)[:,:12000];tail=sp[round(100/DT):]
 r=dict(tag='cell'+str(K),driven_cell=K,flex_spikes=int(sp[:,flex].sum()),ext_spikes=int(sp[:,ext].sum()),flex_tail=int(tail[:,flex].sum()),ext_tail=int(tail[:,ext].sum()),flex_counts=sp[:,flex].sum(0).tolist(),ext_counts=sp[:,ext].sum(0).tolist(),network_spikes=int(sp.sum()),network_tail=int(tail.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
