"""Saved-only first250ms clock comparison, never advances physics or neurons."""
from pathlib import Path
import json,numpy as np,hashlib
BASE=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T058000Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
def summarize(p,end=1.25):
 with np.load(p/'BODY.npz') as f:b={k:f[k] for k in f.files}
 records=json.loads((p/'NEURAL_RECORDS.json').read_text());switch=float(np.load(p/'HANDOVER_AFTER.npz')['time']);t=b['time'];mask=(t>=switch-1e-9)&(t<=end+1e-9);ix=np.flatnonzero(mask&(b['nonfoot_count']>0));last=mask&(t>=end-.2-1e-9);weight=json.loads((p/'GAINS.json').read_text())['weight']
 motor=[r for r in records if r['motor_spikes']>0 and r['relative_end_s']<=.25+1e-9];cmd=[r for r in records if np.asarray(r['applied_command']).any() and r['relative_start_s']<.25-1e-9]
 assert not b['assist'][mask].any() and not b['external'].any() and not b['assist'][:,:6].any() and not b['warning'].any()
 assert all(r['applied_source_interval']==k-1 for k,r in enumerate(records))
 a=np.load(p/'HANDOVER_BEFORE.npz');z=np.load(p/'HANDOVER_AFTER.npz');assert all(np.array_equal(a[k],z[k]) for k in a.files if k not in ('ctrl','qfrc_applied','xfrc_applied'))
 return dict(body=b,metrics=dict(first_motor_window_start_s=None if not motor else motor[0]['relative_start_s'],first_motor_window_end_s=None if not motor else motor[0]['relative_end_s'],first_nonzero_applied_command_relative_s=None if not cmd else cmd[0]['relative_start_s'],first_nonfoot_relative_s=None if not len(ix) else float(t[ix[0]]-switch),matched_neural_end_s=.25,matched_final_com_height=float(b['com'][np.flatnonzero(mask)[-1],2]),last200ms_foot_load_ratio=float(b['foot_force'][last,:,2].sum(1).mean()/weight),last200ms_nonfoot_fraction=float((b['nonfoot_count'][last]>0).mean()),motor_spikes_matched=sum(r['motor_spikes'] for r in records if r['relative_end_s']<=.25+1e-9),command_windows_matched=sum(r['relative_end_s']<=.25+1e-9 for r in records),auxiliary_and_external_zero=True,warnings=0))
def main():
 old=summarize(BASE);new=summarize(OUT);a=old['body'];b=new['body'];warm=a['time']<1-1e-9;warm2=b['time']<1-1e-9;assert np.array_equal(a['qpos'][warm],b['qpos'][warm2]);assert np.array_equal(a['qvel'][warm],b['qvel'][warm2]);aa=np.load(BASE/'HANDOVER_BEFORE.npz');bb=np.load(OUT/'HANDOVER_BEFORE.npz');assert all(np.array_equal(aa[k],bb[k]) for k in aa.files)
 result=dict(passed=True,warmup_and_handover_state_exact_against_baseline=True,baseline_5ms=old['metrics'],fast_1ms=new['metrics'],scope='Single clock change, matched first250ms after same1sphysicalwarmup. No claim of standing from timer alone.',new_physics_steps=0,new_neural_steps=0,body_sha256=hashlib.sha256((OUT/'BODY.npz').read_bytes()).hexdigest());target=OUT/'TIMING_COMPARISON.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
