import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'claw_direction/fine_exact/20261106T191739Z';A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists();ann=json.loads((ROOT/'claw_direction/ANNOTATION_AUDIT.json').read_text());path=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=path['ranked_candidate_indices'];flex=path['targets']['1'];ext=path['targets']['-1'];allids=[r['local_index'] for r in ann['rows']]
 groups={'all_claw':allids}
 files=[Path(__file__),ROOT/'claw_direction/ANNOTATION_AUDIT.json',ROOT/'descending/PATH_AUDIT.json',M,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py'];hashes={str(p):sha(p) for p in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',voltage_method='frozen_exact; exact only for interval-start frozen conductance, not fully exact spiking solver',channel_ids=groups,all_background_cells=allids,background_nA=.04,deltas_nA=[-.005,0,.005],window_ms=[50,100],duration_ms=150,dt_ms=[.03125,.015625],descending_nA=.1,descending_ids=cand,flex=flex,ext=ext,interpretation='local engineering operating point .04nA approximates .05 baseline plus .1 sensory gain at -.1rad; not measured biology; types from frozen annotation, no assigned preferred direction; unequal group sizes, do not compare per-group potency',criterion='fine pair convergence screen: target spike counts equal for each stimulus; paired ordinal motor event max difference <=1ms; 1ms two-drive mean absolute difference <=.01; signed integrated response changes <=.001s across fine pair; all conditions must pass. Screening tolerances are engineering, not physiological; no physical claim',change_ledger={'old_file':str(Path(__file__).with_name('sensory_fine_20261106T191739Z.py')),'old_sha256':sha(Path(__file__).with_name('sensory_fine_20261106T191739Z.py')),'new_file':str(Path(__file__)),'new_sha256':sha(Path(__file__)),'reason':'compare existing frozen_exact voltage method against backward_euler at same fine dt, fixed current/connectivity; no engine edit'},hashes=hashes))
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh=sorted({i for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']});scale=np.ones(12000);scale[rh]=.375;we=(sparse.diags(scale)@we).tocsr();records=[]
 for dt in [.03125,.015625]:
  for ch,ids in groups.items():
   assert not set(ids)&set(cand+flex+ext)
   for delta in [-.005,0,.005]:
    np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt,voltage_method='frozen_exact');net.set_connectivity(we.copy(),wi.copy());base=np.zeros(12000);base[cand]=.1;base[allids]+=.04;packed=[]
    for k in range(round(150/dt)):
     cur=base.copy()
     if 50<=k*dt<100:cur[ids]+=delta
     sp=net.step(current_nA=cur);packed.append(np.packbits(sp))
    bits=np.array(packed);sp=np.unpackbits(bits,axis=1)[:,:12000];counts=sp[round(50/dt):].sum(0);tag=f'{ch}_dt{dt}_delta{delta}';np.savez_compressed(O/(tag+'.npz'),packed=bits)
    r=dict(tag=tag,channel=ch,dt_ms=dt,base_nA=.04,delta_nA=delta,cut=False,flex_counts=counts[flex].tolist(),ext_counts=counts[ext].tolist(),sensory_spikes=int(counts[ids].sum()),network_spikes=int(counts.sum()),finite=bool(np.isfinite(net.v).all()));records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in files};dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,neural_runs=len(records),physics_runs=0,inputs_unchanged=True))
if __name__=='__main__':main()
