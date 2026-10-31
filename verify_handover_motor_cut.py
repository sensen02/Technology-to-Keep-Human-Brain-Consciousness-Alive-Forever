"""Independent saved-only verification of motor-output disconnection causal control."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def load(p):
 with np.load(p/'BODY.npz') as z:return {k:z[k] for k in z.files}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('--baseline',default='/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z');args=ap.parse_args();p=Path(args.directory);base=Path(args.baseline);b=load(p);a=load(base)
 before=np.load(p/'HANDOVER_BEFORE.npz');after=np.load(p/'HANDOVER_AFTER.npz');switch=float(after['time']);t=b['time'];neural=t>=switch-1e-9
 assert all(np.array_equal(before[k],after[k]) for k in before.files if k not in ['ctrl','qfrc_applied','xfrc_applied'])
 assert np.all(np.diff(t)>0) and not b['command'][neural].any() and not b['assist'][neural].any() and not b['assist'][:,:6].any() and not b['external'].any() and not b['warning'].any()
 for x in b.values():
  if np.issubdtype(x.dtype,np.number):assert np.isfinite(x).all()
 j=int(np.flatnonzero(neural)[0]);assert j==int(np.flatnonzero(a['time']>=switch-1e-9)[0])
 for key in ['qpos','qvel','activation','time']:np.testing.assert_array_equal(a[key][:j+1],b[key][:j+1])
 rec=json.loads((p/'NEURAL_RECORDS.json').read_text());assert len(rec)==250 and sum(x['motor_spikes'] for x in rec)>0
 assert any(np.any(x['decoded_next_command']) for x in rec)
 assert all(not np.any(x['applied_command']) for x in rec)
 on=np.flatnonzero(neural&(b['nonfoot_count']>0));late=t>=t[-1]-.1;w=json.loads((p/'GAINS.json').read_text())['weight']
 # Compare identical relative times to detect actual physical causal influence, not just spike changes.
 assert len(t)==len(a['time']);np.testing.assert_array_equal(t,a['time'])
 delta=np.max(np.abs(b['qpos']-a['qpos']),axis=1);div=np.flatnonzero(delta>1e-12)
 result=dict(passed=True,prewarm_and_boundary_exact_to_connected_baseline=True,zero_applied_motor_command=True,decoded_neural_commands_nonzero=True,motor_spikes=sum(x['motor_spikes'] for x in rec),samples=len(t),neural_duration_s=float(t[-1]-switch),first_nonfoot_relative_s=None if not len(on) else float(t[on[0]]-switch),first_physical_divergence_relative_s=None if not len(div) else float(t[div[0]]-switch),final_max_abs_qpos_difference=float(delta[-1]),last100ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][late]>0)),last100ms_foot_load_ratio=float(np.mean(b['foot_force'][late,:,2].sum(1))/w),final_com_height=float(b['com'][-1,2]),auxiliary_external_force_zero=True,numerical_warning_zero=True,new_physics_steps=0,new_neural_steps=0,body_sha256=hashlib.sha256((p/'BODY.npz').read_bytes()).hexdigest(),scope='Motor disconnection causal control only; unchanged collapse time does not imply the neural output has no physical effect at later times.')
 out=p/'PARENT_MOTOR_CUT_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
