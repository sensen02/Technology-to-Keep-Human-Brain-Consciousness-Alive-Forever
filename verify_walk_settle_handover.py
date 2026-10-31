"""Independent saved-only verification of actual walk -> settle -> neural execution."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();p=Path(args.directory);base=Path('/home/sensen/Desktop/engine修复验收/s5-fast-neural-takeover/20261028T061000Z')
 with np.load(p/'BODY.npz') as z:b={k:z[k] for k in z.files}
 with np.load(base/'BODY.npz') as z:a={k:z[k] for k in z.files}
 before=np.load(p/'HANDOVER_BEFORE.npz');switch=float(before['time']);assert switch>1.1
 pre=b['time']<=1.+1e-9;assert int(pre.sum())==1001
 for key in ['qpos','qvel','activation','time']:np.testing.assert_array_equal(b[key][pre],a[key][:1001])
 settle=(b['time']>1.+1e-9)&(b['time']<switch-1e-9);assert settle.any() and b['assist'][settle].any();assert not b['assist'][:,:6].any() and not b['external'].any()
 late=settle&(b['time']>switch-.05);baseline_before=np.load(base/'HANDOVER_BEFORE.npz');neural=b['time']>=switch-1e-9
 def n(x):return float(np.linalg.norm(x))
 nonfoot=np.flatnonzero(neural&(b['nonfoot_count']>0))
 result=dict(passed=True,entire_original_one_second_walk_qpos_qvel_activation_time_exact=True,settling_is_real_assisted_physics_not_neural=True,root_and_external_assistance_zero=True,neural_start_time_s=switch,settle_saved_states=int(settle.sum()),settle_nonfoot_fraction=float(np.mean(b['nonfoot_count'][settle]>0)),late_settle_mean_foot_contact_count=float(np.mean(b['foot_contact'][late].sum(1))),before_handover_qvel_norm=n(before['qvel']),original_handover_qvel_norm=n(baseline_before['qvel']),before_handover_root_angular_velocity_norm=n(before['qvel'][3:6]),original_handover_root_angular_velocity_norm=n(baseline_before['qvel'][3:6]),before_handover_assistance_norm=n(before['qfrc_applied']),original_handover_assistance_norm=n(baseline_before['qfrc_applied']),first_neural_nonfoot_relative_s=None if not len(nonfoot) else float(b['time'][nonfoot[0]]-switch),new_physics_steps=0,new_neural_steps=0,scope='Validates the new assisted settling stage and original walking prefix; lower velocity alone is not a certificate of native muscle support or autonomous walking.')
 out=p/'PARENT_SETTLE_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
