"""Saved-only matched .25s comparison of recruitment trials, not runtime feedback."""
from pathlib import Path
import numpy as np,json
import run_physical_walk_prewarm as w
ROOT=Path('/home/sensen/Desktop/engine修复验收/s5-neural-recruitment')
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
FAST=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
def load(o):
 with np.load(o/'BODY.npz') as f:b={k:f[k] for k in f.files}
 rec=json.loads((o/'NEURAL_RECORDS.json').read_text());boundary=np.load(o/'HANDOVER_BEFORE.npz');return b,rec,boundary

def metrics(o,base_b,base_boundary):
 b,r,h=load(o);t=b['time'];switch=float(h['time']);neural=(t>=switch-1e-9)&(t<=switch+.25+1e-9);last=neural&(t>=switch+.05-1e-9);first=np.flatnonzero(neural&(b['nonfoot_count']>0));before=t<=switch+1e-9
 for k in h.files:np.testing.assert_array_equal(h[k],base_boundary[k])
 for key in ['qpos','qvel','activation'] :np.testing.assert_array_equal(b[key][before],base_b[key][:int(before.sum())])
 assert np.isfinite(b['qpos']).all() and not b['assist'][neural].any() and not b['external'].any() and not b['warning'].any()
 assert all(x['applied_source_interval']==k-1 for k,x in enumerate(r))
 idx=int(np.flatnonzero(neural)[-1]);weight=json.loads((o/'GAINS.json').read_text())['weight']
 nonzero=next((x['relative_start_s'] for x in r if np.any(np.asarray(x['applied_command'])>0)),None)
 # Air bouts in neural stage only; complete bounded bouts, >=10ms, excluding left/right cutoff.
 durations=[]
 for j in range(6):
  air=(~b['foot_contact'][:,j])&neural;pad=np.r_[False,air,False].astype(int);starts=np.flatnonzero(np.diff(pad)==1);ends=np.flatnonzero(np.diff(pad)==-1);valid=[]
  for st,en in zip(starts,ends):
   if st>0 and en<len(t) and neural[st-1] and neural[en] and t[en]-t[st]>=.01-1e-9:valid.append(float(t[en]-t[st]))
  durations.append(valid)
 return dict(directory=str(o),warmup_and_before_boundary_exact_058=True,command_dt_s=r[0]['relative_end_s']-r[0]['relative_start_s'],first_nonzero_applied_command_relative_s=nonzero,first_nonfoot_relative_s=None if len(first)==0 else float(t[first[0]]-switch),last200ms_nonfoot_fraction=float((b['nonfoot_count'][last]>0).mean()),last200ms_mean_foot_load_ratio=float(b['foot_force'][last,:,2].sum(1).mean()/weight),matched250ms_final_com_height=float(b['com'][idx,2]),motor_spike_events_matched250ms=int(sum(x['motor_spikes'] for x in r if x['relative_end_s']<=.25+1e-9)),max_activation_first15ms=float(b['activation'][neural&(t<=switch+.015+1e-9)].max()),mean_command_first15ms=float(b['command'][neural&(t<=switch+.015+1e-9)].mean()),mean_activation_matched250ms=float(b['activation'][neural].mean()),complete_neural_air_bouts_ge10ms=[len(x) for x in durations],pure_neural_auxiliary_force_zero=True,walking_certified=False)

def main():
 base_b,_,base_h=load(BASE);results=[metrics(o,base_b,base_h) for o in [BASE,FAST,ROOT/'20261028T062000Z',ROOT/'20261028T063000Z']]
 out=ROOT/'RECRUITMENT_COMPARISON.json';assert not out.exists();w.dump(out,dict(results=results,scope='All endpoints matched250ms. Stable/coordination not inferred from parameter changes or spike counts.',new_body_steps=0,new_neural_steps=0));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
