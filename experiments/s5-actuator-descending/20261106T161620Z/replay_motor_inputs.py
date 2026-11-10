"""Diagnostic replay of real presynaptic events; never autonomous neural control."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
R=Path('/run/media/sensen/Data2/cell_wound_prototype');sys.path.insert(0,str(R))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-actuator-descending/20261106T161620Z');O=ROOT/'motor_replay/20261106T192549Z';A=Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS/A');M=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=True);assert not (O/'SUMMARY.json').exists();sources={'backward_euler':ROOT/'claw_direction/fine/20261106T191739Z','frozen_exact':ROOT/'claw_direction/fine_exact/20261106T191739Z'};sdt=.015625;p=json.loads((sources['backward_euler']/'PREREGISTRATION.json').read_text());cells=p['flex']+p['ext'];assert not set(cells)&set(p['all_background_cells']+p['descending_ids'])
 files=[Path(__file__),R/'engine/neural_cond.py',A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',M]+[folder/f'all_claw_dt0.015625_delta{delta}.npz' for folder in sources.values() for delta in [-.005,0,.005]];hashes={str(f):sha(f) for f in files}
 dump(O/'PREREGISTRATION.json',dict(importance='核心必需',source_dt_ms=sdt,replay_dt_ms=[.015625,.03125,.0625],methods=list(sources),deltas_nA=[-.005,0,.005],cells=cells,source_dirs={k:str(v) for k,v in sources.items()},arrival_rule='source step-end spike +2ms delay; ceil to next replay interval START, never early',current='zero: no direct current ever injected in target7',connectivity='zero local recurrent matrix; ALL true incoming recorded source events including target-source events are delivered through actual original We/Wi row weights; not closed-loop autonomy',criterion='same dt must reproduce target spike array bitwise; coarser same-input event timing<=1ms/count equality/drive MAE<=.01 diagnostic. Aggregation can add <one replay step lateness, explicitly included',hashes=hashes))
 we=sparse.load_npz(A/'ACTUAL_We.npz').tocsr();wi=sparse.load_npz(A/'ACTUAL_Wi.npz').tocsr();rh={i for row in json.loads(M.read_text())['rows'] if row['leg']=='RH' for i in row['local_indices']};e=we[cells].copy();scale=np.array([.375 if c in rh else 1 for c in cells]);e=(sparse.diags(scale)@e).tocsr();inh=wi[cells].copy();records=[]
 for method,folder in sources.items():
  for delta in [-.005,0,.005]:
   source=np.unpackbits(np.load(folder/f'all_claw_dt0.015625_delta{delta}.npz')['packed'],axis=1)[:,:12000];inc_e=np.array([e@sp for sp in source]);inc_i=np.array([inh@sp for sp in source])
   for dt in [.015625,.03125,.0625]:
    steps=round(150/dt);ee=np.zeros((steps,7));ii=ee.copy();arrival=np.ceil(((np.arange(len(source))+1)*sdt+2)/dt).astype(int);valid=arrival<steps;np.add.at(ee,arrival[valid],inc_e[valid]);np.add.at(ii,arrival[valid],inc_i[valid]);net=ConductanceNetwork(7,ConductanceParams(),dt_ms=dt,voltage_method=method);spikes=[];states=[]
    for k in range(steps):
     sp=net.step(exc_events_uS=ee[k],inh_events_uS=ii[k]);spikes.append(sp);states.append(np.r_[net.v,net.ge,net.gi,net.adaptation])
    spikes=np.array(spikes);tag=f'{method}_delta{delta}_dt{dt}';np.savez_compressed(O/(tag+'.npz'),spikes=spikes,states=states,exc_increments=ee,inh_increments=ii)
    exact=bool(np.array_equal(spikes,source[:,cells])) if dt==sdt else None
    r=dict(tag=tag,method=method,delta_nA=delta,dt_ms=dt,counts=spikes.sum(0).tolist(),same_dt_bitwise=exact,finite=bool(np.isfinite(states).all()));records.append(r);print(json.dumps(r),flush=True)
 assert hashes=={str(f):sha(f) for f in files};dump(O/'SUMMARY.json',dict(importance='核心必需',records=records,replay_runs=18,physics_runs=0,full_network_runs=0,source_files_unchanged=True))
if __name__=='__main__':main()
