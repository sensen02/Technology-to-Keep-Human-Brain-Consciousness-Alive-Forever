"""Independent saved-only checks of physical muscle preactivation and continuous takeover."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();p=Path(args.directory)
 with np.load(p/'BODY.npz') as z:b={k:z[k] for k in z.files}
 before=np.load(p/'HANDOVER_BEFORE.npz');after=np.load(p/'HANDOVER_AFTER.npz');switch=float(before['time']);assert abs(switch-1.2)<1e-8
 candidate=Path('/home/sensen/Desktop/engine修复验收/s5-settled-muscle-support/20261028T076000Z/FIXED_POSE_ACTIVATION_CANDIDATE.npz');c=np.load(candidate);target=c['actuator_control'];G=c['G'];assert G.shape==(90,66)
 pre=b['time']<1.15-1e-8;assert not b['command'][pre].any()
 warm=(b['time']>1.151)&(b['time']<switch-1e-8);assert warm.any();np.testing.assert_allclose(b['command'][warm],np.repeat(target[None,:],int(warm.sum()),axis=0),atol=1e-14,rtol=0)
 unresolved=G.sum(1)==0;assert not b['command'][:,unresolved].any()
 for ids in [np.flatnonzero(G[:,j]) for j in range(66)]:np.testing.assert_allclose(target[ids],np.repeat(target[ids[0]],len(ids)),atol=0,rtol=0)
 np.testing.assert_array_equal(before['act'],after['act']);assert np.max(before['act'])>.01 and not after['ctrl'].any()
 records=json.loads((p/'NEURAL_RECORDS.json').read_text());assert not np.array(records[0]['applied_command']).any();assert all(not r['auxiliary_force_max'] and not r['external_force_max'] for r in records)
 neural=b['time']>=switch-1e-9;foot=b['time']>=b['time'][-1]-.1;non=np.flatnonzero(neural&(b['nonfoot_count']>0));result=dict(passed=True,algorithmic_muscle_preactivation_only_in_declared_warmup=True,admissible_66_ties_16_zero=True,nonzero_activation_physically_retained_at_neural_boundary=True,retained_activation_max=float(np.max(before['act'])),retained_activation_norm=float(np.linalg.norm(before['act'])),first_neural_command_zero=True,prewarm_nonfoot_fraction=float(np.mean(b['nonfoot_count'][~neural]>0)),first_neural_nonfoot_relative_s=None if not len(non) else float(b['time'][non[0]]-switch),last100ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][foot]>0)),new_physics_steps=0,new_neural_steps=0,scope='Verifies actual warmup commands and retained native activation. Slow activation memory is a declared plant change, not evidence of neural autonomy or physiological calibration.')
 out=p/'PARENT_MUSCLE_WARMUP_VERIFICATION.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
