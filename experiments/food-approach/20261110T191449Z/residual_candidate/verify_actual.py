"""Independent saved actual-run verification; zero physical steps or optimization."""
import json
import numpy as np
import torch
from residual_motor import HERE,BASE,sha256
from evaluate_candidate import load_checked


def main():
    torch.set_num_threads(1)
    ck=HERE/'TRAIN_RESIDUAL_500/checkpoint_000500.pt';digest=json.loads((ck.parent/'RESULT.json').read_text())['checkpoint_sha256']
    model,payload=load_checked(ck,digest);reports={};neutral=None
    reference=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500'
    for condition in ('neutral','left','right'):
        run=HERE/('EVAL_RESIDUAL500_'+condition);result=json.loads((run/'RESULT.json').read_text());checks={}
        with np.load(run/'neural_decisions.npz',allow_pickle=False) as z:
            x=torch.tensor(z['X238'],dtype=torch.float64)[:,None];u=torch.tensor(z['command'],dtype=torch.float64)[:,None]
            with torch.no_grad():mean,states=model.sequence(x,u)
            both=torch.cat(states,dim=-1)[:,0].numpy()
            checks['2880_checkpoint_means_exact']=np.array_equal(mean[:,0].numpy(),z['mean'])
            checks['2880_checkpoint_actions_exact']=np.array_equal(mean[:,0].tanh().numpy(),z['policy_action'])
            checks['2880_both_hidden_exact']=np.array_equal(both,z['hidden_after'])
            checks['2880_hidden_chain_exact']=np.array_equal(z['hidden_before'][0],np.zeros(192)) and np.array_equal(z['hidden_before'][1:],z['hidden_after'][:-1])
            checks['actual_action_equals_policy_afterwithdrawal']=np.array_equal(z['actually_applied_action'][960:],z['policy_action'][960:])
            checks['control_source_one_handover']=np.all(z['control_source'][:960]==z['control_source'][0]) and np.all(z['control_source'][960:]==z['control_source'][960]) and z['control_source'][0]!=z['control_source'][960]
            checks['prefix_command_zero']=bool(np.all(z['command'][:960]==0))
            prefix={k:z[k][:960].copy() for k in ('X238','mean','hidden_before','hidden_after','actually_applied_action')}
            if neutral is None:neutral=prefix
            checks['matched_neutral_prefix_allconditions']=all(np.array_equal(v,neutral[k]) for k,v in prefix.items())
        with np.load(run/'trace_1ms.npz',allow_pickle=False) as z:
            checks['all15400_physical_endpoints']=len(z['time'])==15400
            checks['no_extra_body_or_generalized_force']=bool(np.all(z['external_body_force_max']==0) and np.all(z['external_generalized_force_max']==0))
            if condition=='neutral':
                with np.load(reference/'trace_1ms.npz',allow_pickle=False) as ref:
                    shared=set(z.files)&set(ref.files)
                    checks['neutral_all_shared_physical_arrays_bitexact']=all(np.array_equal(z[k],ref[k]) for k in shared)
                    checks['neutral_qpos_qvel_present']= {'qpos','qvel'}<=shared
        checks['handover_exact_noreset']=bool(result['transition']['no_reset'] and all(result['transition']['exact'].values()))
        reports[condition]=dict(integrity_pass=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},engineering_pass=result['candidate_engineering_pass'],gates=result['autonomous']['gates'],cycles=result['autonomous_cycles']['complete_per_leg'],yaw=result['yaw_change_rad'],slip=result['autonomous']['slip_ratio_max'],result_sha256=sha256(run/'RESULT.json'))
    out=dict(integrity_all_pass=all(r['integrity_pass'] for r in reports.values()),engineering_all3_pass=all(r['engineering_pass'] for r in reports.values()),reports=reports,checkpoint_sha256=digest,physical_steps=0,optimizer_steps=0,scope='saved-run integrity verification, no new physical/native certification')
    (HERE/'ACTUAL_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not out['integrity_all_pass']:raise ValueError('actual saved evidence integrity failure')

if __name__=='__main__':main()
