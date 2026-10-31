"""Saved-only actual augmented no-command versus DNp09 pulse comparison."""
from pathlib import Path
import json,numpy as np
R=Path('/home/sensen/Desktop/engine修复验收/s5-descending-command')
def main():
 result=[];arrays=[]
 for label in ['20261028T071000Z','20261028T072000Z']:
  p=R/label
  with np.load(p/'BODY.npz') as z:b={k:z[k] for k in z.files}
  arrays.append(b);rec=json.loads((p/'NEURAL_RECORDS.json').read_text());t=b['time'];sw=float(np.load(p/'HANDOVER_AFTER.npz')['time']);mask=t>=sw-1e-9;late=t>=t[-1]-.2;first=np.flatnonzero(mask&(b['nonfoot_count']>0));w=json.loads((p/'GAINS.json').read_text())['weight'];assert not b['assist'][mask].any() and not b['external'].any() and not b['warning'].any()
  result.append(dict(directory=str(p),counts=json.loads((p/'EXECUTION.json').read_text())['counts'],first_nonfoot_relative_s=None if not len(first) else float(t[first[0]]-sw),last200ms_foot_load_ratio=float(b['foot_force'][late,:,2].sum(1).mean()/w),last200ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][late]>0)),final_com_height=float(b['com'][-1,2]),motor_events_first15ms=sum(x['motor_spikes'] for x in rec if x['relative_start_s']<.015-1e-12),motor_events_250ms=sum(x['motor_spikes'] for x in rec),DNp09_spikes_first50ms=sum(x['descending_target_spikes'] for x in rec if x['relative_start_s']<.05-1e-12),DNp09_spikes_after_withdrawal=sum(x['descending_target_spikes'] for x in rec if x['relative_start_s']>=.05-1e-12),declared_pulse_amplitude_nA=rec[0]['descending_command_steps'][0]['amplitude_nA'],no_auxiliary_external_force=True,walking_certified=False))
 a,b=arrays
 for k in ['time','qpos','qvel','activation']:np.testing.assert_array_equal(a[k][:1001],b[k][:1001])
 for key in ['HANDOVER_BEFORE.npz']:
  x=np.load(R/'20261028T071000Z'/key);y=np.load(R/'20261028T072000Z'/key)
  assert all(np.array_equal(x[k],y[k]) for k in x.files)
 diff=np.max(abs(a['qpos']-b['qpos']),axis=1);idx=np.flatnonzero(diff>1e-12)
 out=dict(trials=result,prewarm_and_entire_boundary_exact=True,first_qpos_divergence_relative_s=None if not len(idx) else float(a['time'][idx[0]]-a['time'][1000]),max_final_qpos_difference=float(diff[-1]),scope='Explicit external neuronal start pulse only; post-pulse support and coordination still require behavior, never count target spiking as walking.',new_body_steps=0,new_neural_steps=0)
 (R/'DESCENDING_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
