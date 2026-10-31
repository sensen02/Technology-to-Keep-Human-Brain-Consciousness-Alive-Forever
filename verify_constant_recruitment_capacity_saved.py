"""Saved-only direct-muscle diagnostic verification; no MuJoCo/optimizer import."""
from pathlib import Path
import json,hashlib,numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s2-constant-recruitment-capacity/20261028T001000Z')
def main():
 p=json.loads((O/'PREREGISTRATION.json').read_text());ex=json.loads((O/'EXECUTION.json').read_text());authority=json.loads((O/'AUTHORITY.json').read_text());summary=json.loads((O/'SUMMARY.json').read_text());checks={};initial=np.load('/home/sensen/Desktop/engine修复验收/_历史归档/actual-neural-matched-20261018T031000Z/initial_runtime_state.npz')['qpos'];checks['inputs_unchanged']=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in p['source_hashes'].items());checks['two_rollout_budget']=ex['counts']['rollouts']<=2 and ex['counts']['mj_step']<=40000;checks['forward_budget_each']=all(n<=1000 for n in ex['counts']['optimization_forward'].values());checks['neural_zero']=ex['counts']['neural_steps']==0
 for mode in summary:
  b=np.load(O/mode/'BODY.npz');u=np.load(O/mode/'CONSTANT_COMMAND.npy');np.testing.assert_array_equal(b['qpos'][0],initial);assert not b['qvel'][0].any() and not b['activation'][0].any();np.testing.assert_array_equal(b['command'],np.repeat(u[None,:],20001,axis=0));assert not b['external_force'].any();assert np.all((u>=0)&(u<=1));assert all(np.isfinite(b[k]).all() for k in b.files);assert not b['warnings'].any();np.testing.assert_allclose(b['ground_force_foot_world'].sum(1)+b['ground_force_nonfoot_world'],b['ground_force_world'],rtol=1e-12,atol=1e-12);checks[mode+'/original_initial_constant_bound_noexternal_finite_worldpartition']=True
  if mode.startswith('mapped'):
   assert not u[authority['unmapped_indices']].any()
   for group in authority['mapped_groups']:assert np.unique(u[group['actuator_indices']]).size==1
   checks['mapped_exact_ties_and_unresolved_zero']=True
 result=dict(passed=all(checks.values()),checks=checks,command='venv_body/bin/python '+str(Path(__file__)),exit_code=0,model=0,forward=0,step=0,neural=0,optimization=0)
 (O/'SAVED_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(result)
if __name__=='__main__':main()
