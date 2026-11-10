"""Fixed-readout duration x timestep pure-neural diagnostic; no model selection."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'pulse_timing'
A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists()
 audit=json.loads((ROOT/'descending/PATH_AUDIT.json').read_text());cand=audit['ranked_candidate_indices'];flex=audit['targets']['1'];ext=audit['targets']['-1'];cells=flex+ext
 inputs=[Path(__file__),ROOT/'descending/PATH_AUDIT.json',M,A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',R/'engine/neural_cond.py'];hashes={str(p):sha(p) for p in inputs}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',dt_ms=[.5,.25,.125,.0625],pulse_ms=[30,50,100,150],duration_ms=250,current_nA=.1,ids=cand,flex=flex,ext=ext,sensory='zero; pure descending input; compare with previous embodied trials only as diagnostic',command_dt_ms=1,filter_tau_ms=5,readout='clip((mean flex rate-mean ext rate)/100,-1,1), causal prior-interval rates, integrate 2rad/s *u*1ms',criterion='report signed direction for all 16 conditions, no choosing winning duration/timestep; zero if abs(target shift)<1e-6 rad',hashes=hashes))
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh=sorted({i for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']});scale=np.ones(12000);scale[rh]=.375;we=(sparse.diags(scale)@we).tocsr();records=[]
 for dt in [.5,.25,.125,.0625]:
  n=round(1/dt)
  for pulse in [30,50,100,150]:
   np.random.seed(0);net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=dt);net.set_connectivity(we.copy(),wi.copy());current=np.zeros(12000);current[cand]=.1
   packed=[];state=[];rates=np.zeros(12000);commands=[];decay=np.exp(-dt/5);shift=0.
   for k in range(round(250/dt)):
    if k%n==0:
     u=float(np.clip((rates[flex].mean()-rates[ext].mean())/100,-1,1));shift+=.002*u;commands.append([k*dt,u,shift])
    state.append(np.stack([net.v[cells],net.ge[cells],net.gi[cells]],axis=1))
    sp=net.step(current_nA=current if k*dt<pulse else np.zeros(12000));packed.append(np.packbits(sp));rates=decay*rates+(1-decay)*sp/(dt/1000)
   bits=np.array(packed);sp=np.unpackbits(bits,axis=1)[:,:12000];counts=sp.sum(0);tag=f'dt{dt}_pulse{pulse}'
   np.savez_compressed(O/(tag+'.npz'),packed=bits,motor_state_before_step=state,cells=cells,commands=commands)
   r=dict(tag=tag,dt_ms=dt,pulse_ms=pulse,flex_counts=counts[flex].tolist(),ext_counts=counts[ext].tolist(),weighted_count_difference=float(counts[flex].mean()-counts[ext].mean()),target_shift_rad=shift,motor_spike_times_ms={str(v):((np.flatnonzero(sp[:,v])+1)*dt).tolist() for v in cells},finite=bool(np.isfinite(net.v).all()))
   records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(p):sha(p) for p in inputs}
 dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,neural_runs=16,physics_runs=0,inputs_unchanged=True))
if __name__=='__main__':main()
