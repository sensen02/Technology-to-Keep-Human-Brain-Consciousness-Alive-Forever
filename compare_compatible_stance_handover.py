"""Saved-only actual compatible-target vs original settled-target comparison."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/s5-compatible-stance-handover/20261028T080000Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
def load(p):
 with np.load(p/'BODY.npz') as z:return {k:z[k] for k in z.files}
def main():
 a=load(B);b=load(P);assert (B/'MODEL.xml').read_bytes()==(P/'MODEL.xml').read_bytes()
 for k in ['time','qpos','qvel','activation']:np.testing.assert_array_equal(a[k][:1001],b[k][:1001])
 rows=[]
 for p,z in [(B,a),(P,b)]:
  before=np.load(p/'HANDOVER_BEFORE.npz');t=float(before['time']);near=z['time']>=t-1e-9;late=z['time']>=z['time'][-1]-.2;warm=~near;non=np.flatnonzero(near&(z['nonfoot_count']>0));w=json.loads((p/'GAINS.json').read_text())['weight'];rec=json.loads((p/'NEURAL_RECORDS.json').read_text())
  assert not z['assist'][near].any() and not z['assist'][:,:6].any() and not z['external'].any() and not z['warning'].any()
  rows.append(dict(directory=str(p),handover_time_s=t,prewarm_nonfoot_fraction=float(np.mean(z['nonfoot_count'][warm]>0)),last50ms_prewarm_contact_count=float(np.mean(z['foot_contact'][(z['time']>=t-.05)&warm].sum(1))),handover_qvel_norm=float(np.linalg.norm(before['qvel'])),handover_assist_norm=float(np.linalg.norm(before['qfrc_applied'])),handover_hinge_qpos=before['qpos'][7:],first_neural_nonfoot_relative_s=None if not len(non) else float(z['time'][non[0]]-t),last200ms_nonfoot_fraction=float(np.mean(z['nonfoot_count'][late]>0)),last200ms_foot_load_ratio=float(np.mean(z['foot_force'][late,:,2].sum(1))/w),final_com_height=float(z['com'][-1,2]),motor_spikes=sum(r['motor_spikes'] for r in rec)))
 target=json.loads((P/'SETTLE_TARGET.json').read_text());qa=np.array(target['qpos_addresses']);before=np.load(P/'HANDOVER_BEFORE.npz');result=dict(original_first_second_qpos_qvel_activation_time_exact=True,entire_model_xml_byte_exact=True,all_auxiliary_and_external_checks_passed=True,candidate_hinge_tracking_l2=float(np.linalg.norm(before['qpos'][qa]-np.array(target['target']))),runs=rows,scope='Only assisted final hinge target changes. Offline LP proxy remains imperfect; no success from reduced assistance or nonzero neural output alone.')
 out=P/'COMPATIBLE_STANCE_COMPARISON.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist())+'\n');print(json.dumps(result,indent=2,default=lambda x:x.tolist()))
if __name__=='__main__':main()
