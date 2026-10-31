"""Saved-only prospective phase comparison at identical neural duration and controller."""
from pathlib import Path
import json,numpy as np
B=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z');T=Path('/home/sensen/Desktop/engine修复验收/s5-selected-phase-neural-takeover/20261028T065000Z')
def load(p):
 with np.load(p/'BODY.npz') as f:b={k:f[k] for k in f.files}
 before=np.load(p/'HANDOVER_BEFORE.npz');after=np.load(p/'HANDOVER_AFTER.npz');protected=set(before.files)-{'ctrl','qfrc_applied','xfrc_applied'};assert all(np.array_equal(before[k],after[k]) for k in protected)
 switch=float(after['time']);t=b['time'];mask=t>=switch-1e-10;assert not b['assist'][mask].any() and not b['external'].any() and not b['warning'].any();first=np.flatnonzero(mask&(b['nonfoot_count']>0));rec=json.loads((p/'NEURAL_RECORDS.json').read_text());assert len(rec)==250
 for k,r in enumerate(rec):assert r['applied_source_interval']==k-1
 last=t>=t[-1]-.2;w=json.loads((p/'GAINS.json').read_text())['weight'];return b,dict(handover_time_s=switch,neural_duration_s=float(t[-1]-switch),first_nonfoot_relative_s=None if not len(first) else float(t[first[0]]-switch),final_com_height=float(b['com'][-1,2]),last200ms_nonfoot_fraction=float((b['nonfoot_count'][last]>0).mean()),last200ms_mean_foot_load_ratio=float(b['foot_force'][last,:,2].sum(1).mean()/w),motor_spikes=sum(r['motor_spikes'] for r in rec),body_velocity_at_transition=b['qvel'][np.flatnonzero(mask)[0]].tolist(),state_continuity_exact=True,neural_auxiliary_force_zero=True,numerical_warning_zero=True)
def main():
 b,x=load(B);t,y=load(T);n=int(np.flatnonzero(t['time']>=y['handover_time_s']-1e-10)[0])+1
 for k in ['time','qpos','qvel','activation']:np.testing.assert_array_equal(b[k][:n],t[k][:n])
 out=dict(baseline=x,alternate=y,warmup_prefix_exact=True,scope='Prospective second-ranked distinct phase test, same neural controller. Does not establish global phase optimum, and a different physical state changes feedback-driven neural recruitment legitimately.')
 target=T/'PHASE_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
