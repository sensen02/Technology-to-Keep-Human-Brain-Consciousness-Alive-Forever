"""Saved-only independent integrity/behavior checks of alternate end-stance trial."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();p=Path(args.directory);base=Path('/home/sensen/Desktop/engine修复验收/s5-walk-settle-neural/20261028T074000Z')
 with np.load(p/'BODY.npz') as z:b={k:z[k] for k in z.files}
 with np.load(base/'BODY.npz') as z:a={k:z[k] for k in z.files}
 before=np.load(p/'HANDOVER_BEFORE.npz');switch=float(before['time']);assert abs(switch-1.2)<1e-8
 for key in ['qpos','qvel','activation','time']:np.testing.assert_array_equal(b[key][:1001],a[key][:1001])
 assert (p/'MODEL.xml').read_bytes()==(base/'MODEL.xml').read_bytes(),'Pose-only experiment unexpectedly changed physical model'
 warm=b['time']<switch-1e-9;neural=~warm;late=b['time']>b['time'][-1]-.1;idx=np.flatnonzero(neural&(b['nonfoot_count']>0));w=json.loads((p/'GAINS.json').read_text())['weight']
 result=dict(passed=True,original_walking_prefix_exact=True,entire_model_xml_identical_to_074=True,physical_parameters_unchanged=True,handover_s=switch,prewarm_nonfoot_fraction=float(np.mean(b['nonfoot_count'][warm]>0)),prewarm_dx_native=float(b['qpos'][np.flatnonzero(neural)[0],0]-b['qpos'][0,0]),root_assistance_zero=bool(not b['assist'][:,:6].any()),first_neural_nonfoot_s=None if not len(idx) else float(b['time'][idx[0]]-switch),last100ms_nonfoot_fraction=float(np.mean(b['nonfoot_count'][late]>0)),last100ms_foot_load_ratio=float(np.mean(b['foot_force'][late,:,2].sum(1))/w),before_assistance_norm=float(np.linalg.norm(before['qfrc_applied'])),before_velocity_norm=float(np.linalg.norm(before['qvel'])),new_models=0,new_physics_steps=0,new_neural_steps=0,scope='Pose-only physical transition integrity and actual behavior, not autonomous-walking certification.')
 assert result['root_assistance_zero']
 out=p/'PARENT_COMPATIBLE_STANCE_CHECK.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
