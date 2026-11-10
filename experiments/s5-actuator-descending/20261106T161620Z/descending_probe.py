"""Declared tonic input probe; topology-defined candidates, no physical controller."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z')
O=ROOT/'descending_probe';A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A')
M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices']
 other=sorted(set(r['local_index'] for r in audit['records'])-set(cand));rng=np.random.default_rng(20261106);control=sorted(rng.choice(other,len(cand),replace=False).tolist())
 flex=audit['targets']['1'];ext=audit['targets']['-1'];assert len(cand)==len(control)==19
 arms=[('none',0,[])] + [(name,g,ids) for name,ids in [('candidate',cand),('matched',control)] for g in [.01,.1,.3]] + [('candidate_cut',.3,cand)]
 files=[Path(__file__),ROOT/'descending/PATH_AUDIT.json',M,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py']
 hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',candidate=cand,matched=control,matched_rule='seed20261106 random same-size descending set excluding both-pool <=4hop candidates; equal per-cell and total input, not degree-matched',amps_nA=[.01,.1,.3],dt_ms=[.5,.0625],duration_ms=200,onset_ms=50,analysis_window_ms=[50,200],sensory='none: pure neural input-transfer diagnostic',RH='incoming excitation x.375 once',cut='zero all outgoing We/Wi candidate columns, ablation only',criterion='report target firing increases and abolition under output cut at .3nA; no support or biological command claim; do not select based on physics',hashes=hashes))
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr()
 rh=sorted({i for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']});scale=np.ones(12000);scale[rh]=.375;we=(sparse.diags(scale)@we).tocsr()
 records=[]
 for dt in [.5,.0625]:
  for name,g,ids in arms:
   np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt)
   e=we.copy();i=wi.copy()
   if name=='candidate_cut':
    mask=np.ones(12000);mask[cand]=0;e=(e@sparse.diags(mask)).tocsr();i=(i@sparse.diags(mask)).tocsr();e.eliminate_zeros();i.eliminate_zeros()
   net.set_connectivity(e,i);current=np.zeros(12000);current[ids]=g;packed=[]
   for k in range(round(200/dt)):
    sp=net.step(current_nA=current if k*dt>=50 else np.zeros(12000));packed.append(np.packbits(sp))
   bits=np.array(packed);sp=np.unpackbits(bits,axis=1)[:,:12000];win=sp[round(50/dt):];counts=win.sum(0);rates=counts/.15
   tag=name+'_'+str(g)+'_'+str(dt);np.savez_compressed(O/(tag+'.npz'),packed=bits,injected_ids=ids,current_nA=g,dt_ms=dt)
   r=dict(tag=tag,arm=name,current_nA=g,dt_ms=dt,total_spikes=int(counts.sum()),flex_Hz=float(rates[flex].mean()),ext_Hz=float(rates[ext].mean()),input_Hz=float(rates[ids].mean()) if ids else 0,flex_counts=counts[flex].tolist(),ext_counts=counts[ext].tolist(),finite=bool(np.isfinite(net.v).all()))
   records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,physics_runs=0,neural_runs=len(records),inputs_unchanged=True))
if __name__=='__main__':main()
