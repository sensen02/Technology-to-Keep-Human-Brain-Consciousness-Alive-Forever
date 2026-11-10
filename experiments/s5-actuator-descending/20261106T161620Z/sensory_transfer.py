"""Declared sensory transfer hypotheses, fixed existing connectivity and descending input."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'sensory_transfer'
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 aud=json.loads((ROOT/'neural_tendon/SENSORY_AUDIT.json').read_text());path=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=path['ranked_candidate_indices'];flex=path['targets']['1'];ext=path['targets']['-1']
 inputs=[Path(__file__),ROOT/'neural_tendon/SENSORY_AUDIT.json',ROOT/'descending/PATH_AUDIT.json',M,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py'];hashes={str(p):sha(p) for p in inputs}
 cases=[(base,delta,False) for base in [0,.02] for delta in [-.03,0,.03]]+[(0,delta,False) for delta in [.1,.3]]+[(0,delta,True) for delta in [0,.3]]
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',channels=['claw_position','hook_direction'],dt_ms=[.5,.0625],cases=cases,background_nA='engineering hypothesis 0 or .02 on selected sensory group throughout 150ms, not measured spontaneous rate',sensory_delta_window_ms=[50,100],duration_ms=150,descending_nA=.1,descending_ids=cand,flex=flex,ext=ext,channel_ids={ch:aud['channels'][ch]['ids'] for ch in ['claw_position','hook_direction']},cut='all outgoing sensory-group columns removed in both We/Wi; paired cut delta0 control',criterion='report motor changes versus same-background/dt/channel delta0, require causal contrast before selecting physical condition; no fictitious edges or motor-current injection; no physical support claim',hashes=hashes))
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh=sorted({i for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']});scale=np.ones(12000);scale[rh]=.375;we=(sparse.diags(scale)@we).tocsr();records=[]
 for dt in [.5,.0625]:
  for ch in ['claw_position','hook_direction']:
   ids=aud['channels'][ch]['ids'];assert not set(ids)&set(cand+flex+ext)
   for base,delta,cut in cases:
    np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt);e=we.copy();i=wi.copy()
    if cut:
     mask=np.ones(12000);mask[ids]=0;e=(e@sparse.diags(mask)).tocsr();i=(i@sparse.diags(mask)).tocsr()
    net.set_connectivity(e,i);constant=np.zeros(12000);constant[cand]=.1;constant[ids]+=base;packed=[]
    for k in range(round(150/dt)):
     current=constant.copy()
     if 50<=k*dt<100:current[ids]+=delta
     sp=net.step(current_nA=current);packed.append(np.packbits(sp))
    bits=np.array(packed);sp=np.unpackbits(bits,axis=1)[:,:12000];counts=sp[round(50/dt):].sum(0);tag=f'{ch}_dt{dt}_base{base}_delta{delta}_cut{cut}'
    np.savez_compressed(O/(tag+'.npz'),packed=bits)
    r=dict(tag=tag,dt_ms=dt,channel=ch,base_nA=base,delta_nA=delta,cut=cut,flex_counts=counts[flex].tolist(),ext_counts=counts[ext].tolist(),sensory_spikes=int(counts[ids].sum()),network_spikes=int(counts.sum()),finite=bool(np.isfinite(net.v).all()))
    records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in inputs}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,neural_runs=len(records),physics_runs=0,inputs_unchanged=True))
if __name__=='__main__':main()
