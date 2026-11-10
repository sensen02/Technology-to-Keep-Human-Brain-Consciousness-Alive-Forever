"""Stage C5: drive real ventral-nerve-cord premotor partners of each motor pool.
Partners are the strongest excitatory-dominant VNC-intrinsic inputs from the frozen matrix.
Static open loop; no invented edges; no external controller.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');A=ROOT/'synaptic_scale/20261106T193236Z/0.25'
H=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
POOL=sys.argv[1];CUR=float(sys.argv[2]);BIAS=.03;DT=.0078125;DUR=150;N=6
O=ROOT/'premotor/20261106T225055Z'/(POOL+'_a'+str(CUR))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def select(we,wi,pool_cells,exclude):
    net={}
    for c in pool_cells:
        for p,w in zip(we[c].tocoo().col.tolist(),we[c].tocoo().data.tolist()):net[p]=net.get(p,0.)+float(w)
        for p,w in zip(wi[c].tocoo().col.tolist(),wi[c].tocoo().data.tolist()):net[p]=net.get(p,0.)-float(w)
    ok=[(p,v) for p,v in net.items() if v>0 and p not in exclude and p not in pool_cells]
    ok.sort(key=lambda x:-x[1]);return [p for p,_ in ok[:N]],[round(v,6) for _,v in ok[:N]]
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());flex=audit['targets']['1'];ext=audit['targets']['-1'];cand=set(audit['ranked_candidate_indices'])
 ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());claw=[r['local_index'] for r in ann['rows']]
 we=sparse.load_npz(A/'EFFECTIVE_We.npz').tocsr();wi=sparse.load_npz(A/'EFFECTIVE_Wi.npz').tocsr()
 exclude=set(claw)|cand
 driven,weights=select(we,wi,flex if POOL=='flex' else ext,exclude)
 assert not set(driven)&set(flex+ext)
 files=[Path(__file__),A/'EFFECTIVE_We.npz',A/'EFFECTIVE_Wi.npz',ROOT/'descending/PATH_AUDIT.json',ROOT/'claw_direction/ANNOTATION_AUDIT.json',H/'IDENTITIES.npz',H/'RUN/MAPPING.json',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',stage='C5 premotor recruitment',pool=POOL,driven=driven,net_exc_weight=weights,top_n=N,current_nA=CUR,bias_nA=BIAS,offset_rad=0.,dt_ms=DT,duration_ms=DUR,runs=1,excluded='claw sensory cells and the 19 descending candidates are excluded, so these are VNC-intrinsic premotor partners',criterion='report per-pool counts only; a usable interface needs the target pool to spike',hashes=hashes))
 np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=DT);net.set_connectivity(we,wi)
 cur=np.zeros(12000);cur[claw]+=BIAS;cur[driven]+=CUR
 packed=[]
 for k in range(round(DUR/DT)):
  sp=net.step(current_nA=cur);packed.append(np.packbits(sp))
 packed=np.array(packed);sp=np.unpackbits(packed,axis=1)[:,:12000];tail=sp[round(100/DT):]
 r=dict(tag=POOL+'_a'+str(CUR),pool=POOL,driven=driven,current_nA=CUR,flex_counts=sp[:,flex].sum(0).tolist(),ext_counts=sp[:,ext].sum(0).tolist(),flex_spikes=int(sp[:,flex].sum()),ext_spikes=int(sp[:,ext].sum()),flex_tail=int(tail[:,flex].sum()),ext_tail=int(tail[:,ext].sum()),driven_spikes=int(sp[:,driven].sum()),network_spikes=int(sp.sum()),finite=bool(np.isfinite(net.v).all()))
 np.savez_compressed(O/'trail.npz',packed=packed);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=[r],runs=1,physics_integrated=False))
if __name__=='__main__':main()
