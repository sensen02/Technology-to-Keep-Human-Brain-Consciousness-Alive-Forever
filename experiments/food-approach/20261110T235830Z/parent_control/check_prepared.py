"""Offline parent500 binding and neutral parity checks, no physical construction."""
import ast
import json
import numpy as np
import torch
import evaluate_parent as ep
import verify_actual


def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}


def main():
    torch.set_num_threads(1);checks={}
    raw=(ep.EVALUATION/'evaluate_motor.py').read_text();derived=ep.derived_source()
    f,g=functions(raw),functions(derived)
    for name in ('jsonable','command_at','score','flatten','parser'):checks['unchanged_AST_'+name]=f[name]==g[name]
    checks['evaluate_only_declared_five_bindings']=sum(a!=b for a,b in zip(raw.splitlines(),derived.splitlines()))>0 and "prereg['checkpoint_step']!=500" in derived
    mod=ep.derived_module();checks['explicit_output_and_checkpoint_binding']=mod.HERE==ep.HERE and mod.CHECKPOINT==ep.CHECKPOINT and mod.COMMANDS==ep.COMMANDS
    sw,identity,io=ep.pin();model,p=ep.checked_load(io,ep.CHECKPOINT,ep.SHA)
    checks['actual500_curve']=len(p['config']['curve'])==501 and p['config']['curve'][-1]['step']==500
    rejected=False
    try:ep.checked_load(io,ep.CHECKPOINT,'0'*64)
    except ValueError:rejected=True
    checks['wrong_sha_rejected']=rejected
    assoc=ep.neutral_association(identity);checks['matched_neutral500_passed_identity']=assoc['checkpoint_sha256']==ep.SHA
    with np.load(ep.NEUTRAL/'neural_decisions.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    state=model.initial_state(1);means=[];hidden=[];actions=[]
    with torch.no_grad():
        for x,u in zip(data['X238'],data['command']):
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
            means.append(mean[0].numpy());hidden.append(ep.flatten(state)[0].numpy());actions.append(mean[0].tanh().numpy())
    checks['parent500_all2880_neutral_means_exact']=np.array_equal(means,data['mean'])
    checks['parent500_all2880_neutral_dual192_exact']=np.array_equal(hidden,data['hidden_after'])
    checks['parent500_all2880_neutral_actions_exact']=np.array_equal(actions,data['policy_action'])
    for condition in ep.COMMANDS:
        u=np.array([ep.command_at(condition,k) for k in range(2880)])
        checks[condition+'_neutral_prime_zero']=np.array_equal(u[:961],np.zeros((961,2)))
        checks[condition+'_settled_after1s']=np.array_equal(u[1160:],np.tile(ep.COMMANDS[condition],(1720,1)))
        checks[condition+'_same_new1500_command']=np.array_equal(u,np.array([ep.old.command_at(condition,k) for k in range(2880)]))
    checks['original_gates_and_turn_score']=ep.score('left',dict(all_gates=True,nonfoot_force_max_native=0.),dict(all_legs_completed=True),.01)['eligible'] and not ep.score('left',dict(all_gates=True,nonfoot_force_max_native=0.),dict(all_legs_completed=True),-.02)['eligible']
    checks['verifier_derived_compiles']=callable(verify_actual.module().verify)
    paths=[p for p in ep.source_paths() if p.name!='READINESS.json']
    report=dict(all_passed=all(checks.values()),checks=checks,checkpoint=str(ep.CHECKPOINT),checkpoint_sha256=ep.SHA,
                neutral_association=assoc,source_hashes={str(p):ep.sha256(p) for p in paths},physics_integration_steps=0,Core_constructions=0,optimizer_updates=0,
                explicit_bindings=['parent500 strict legacy loader/ancestry and fixed paths','new prereg500 metadata','same physics and scoring','old neutral hash/runtime association, separate old/new source identity','new owned output paths and audit source names'])
    (ep.HERE/'READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),neutral_result=assoc['result'])))
    if not report['all_passed']:raise ValueError('parent controls not ready')

if __name__=='__main__':main()
