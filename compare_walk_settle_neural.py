"""Saved-only physical settling versus original continuous handover comparison."""
from pathlib import Path
import json,numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z');B=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
def load(p):
 with np.load(p/'BODY.npz') as z:return {k:z[k] for k in z.files}
def main():
 b=load(P);a=load(B);before=np.load(P/'HANDOVER_BEFORE.npz');old=np.load(B/'HANDOVER_BEFORE.npz');s=float(before['time']);warm=b['time']<s-1e-9;neu=~warm;settle=(b['time']>=1.-1e-9)&warm
 for k in ['qpos','qvel','activation','time']:np.testing.assert_array_equal(b[k][:1001],a[k][:1001])
 assert not b['assist'][:,:6].any() and not b['assist'][neu].any() and not b['external'].any() and not b['warning'].any()
 def metrics(x,sw,p):
  mask=x['time']>=sw-1e-9;late=x['time']>=x['time'][-1]-.2;on=np.flatnonzero(mask&(x['nonfoot_count']>0));weight=json.loads((p/'GAINS.json').read_text())['weight'];return dict(first_neural_nonfoot_s=None if not len(on) else float(x['time'][on[0]]-sw),last200_nonfoot_fraction=float(np.mean(x['nonfoot_count'][late]>0)),last200_foot_load_ratio=float(np.mean(x['foot_force'][late,:,2].sum(1))/weight),final_com_height=float(x['com'][-1,2]))
 out=dict(passed=True,entire_first1s_time_qpos_qvel_activation_exact=True,settle_and_hold_nonfoot_absent=bool(not b['nonfoot_count'][settle].any()),warmup_nonfoot_absent=bool(not b['nonfoot_count'][warm].any()),before_joint_velocity_norm_old=float(np.linalg.norm(old['qvel'][6:])),before_joint_velocity_norm_settled=float(np.linalg.norm(before['qvel'][6:])),before_root_velocity_norm_old=float(np.linalg.norm(old['qvel'][:6])),before_root_velocity_norm_settled=float(np.linalg.norm(before['qvel'][:6])),before_assist_norm_old=float(np.linalg.norm(old['qfrc_applied'][6:])),before_assist_norm_settled=float(np.linalg.norm(before['qfrc_applied'][6:])),settle_last_root_pose=b['qpos'][np.flatnonzero(warm)[-1],:7].tolist(),all_neural_auxiliary_external_zero=True,warnings_zero=True,baseline=metrics(a,float(old['time']),B),settled=metrics(b,s,P),scope='Settling is algorithm-assisted physics, not autonomous behavior. Compare actual neural support only after all1.2s assistance withdrawn.')
 (P/'SETTLE_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
