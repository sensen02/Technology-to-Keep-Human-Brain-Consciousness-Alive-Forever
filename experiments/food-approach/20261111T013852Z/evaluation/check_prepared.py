"""Offline final10000 contract and frozen1500-right preservation checks; no Core."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
import evaluate_motor as ev
HERE=Path(__file__).resolve().parent
OLD_HEAD=ev.PREDECESSOR.parent/'train/TRAIN/checkpoint_002000.pt'
OLD_HEAD_SHA='eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df'


def functions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


def main():
    torch.set_num_threads(1);checks={}
    files=[HERE/n for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py')]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
        else:json.loads(p.read_text())
        checks['syntax:'+p.name]=True
    original=functions((ev.PRIMARY/'evaluate_motor.py').read_text());derived=functions(ev.derived_source())
    for name in ('jsonable','command_at','score','pin','flatten','parser'):
        checks['original_function_unchanged:'+name]=original[name]==derived[name]
    checks['exact_final_path']=ev.CHECKPOINT==HERE.parent/'train/TRAIN/checkpoint_010000.pt'
    checks['fixed10000_actual_loader_contract']="payload['curve'][-1]['step']!=10000" in ev.derived_source() and "physical_eval_only_step']!=10000" in ev.derived_source()
    checks['right_reference_is_c6b1500_not_continuation2000']=ev.PARENT_SHA=='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb' and ev.PARENT_CHECKPOINT.name=='checkpoint_001500.pt'
    prereg=json.loads((HERE/'EVAL_PREREG.json').read_text())
    checks['step_and_conditions']=prereg['checkpoint_step']==10000 and list(prereg['conditions'])==['neutral','left','right']
    checks['full9p6_window']=prereg['timing']['autonomous_ms']==9600
    checks['native12p5us']=prereg['timing']['native_dt_s']==.0000125
    checks['first_command_zero']=all(np.array_equal(ev.command_at(c,960),np.zeros(2)) for c in ev.COMMANDS)
    checks['one_second_full_command']=all(np.array_equal(ev.command_at(c,1160),np.asarray(u)) for c,u in ev.COMMANDS.items())
    good=dict(all_gates=True,nonfoot_force_max_native=0.)
    checks['score_left_pass']=ev.score('left',good,dict(all_legs_completed=True),.01)['eligible']
    checks['score_direction_reject']=not ev.score('right',good,dict(all_legs_completed=True),.01)['eligible']
    checks['score_cycles_reject']=not ev.score('left',good,dict(all_legs_completed=False),.02)['eligible']
    checks['score_nonfoot_reject']=not ev.score('left',dict(all_gates=True,nonfoot_force_max_native=.001),dict(all_legs_completed=True),.02)['eligible']
    checks['numpy_scalar_json']=json.loads(json.dumps(ev.jsonable(dict(flag=np.bool_(True),n=np.int64(2)))))==dict(flag=True,n=2)
    class ForbiddenLoader:
        @staticmethod
        def load_checked(*args,**kwargs):raise AssertionError('unexpected weight load')
    rejected=False
    try:ev.checked_load(ForbiddenLoader,str(ev.CHECKPOINT.with_name('checkpoint_002000.pt')),OLD_HEAD_SHA)
    except ValueError:rejected=True
    checks['wrong_path_rejected_before_model_load']=rejected
    spec=importlib.util.spec_from_file_location('_offline_final10000_io',ev.TRAIN/'model_io.py')
    io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
    final_sha='3bac5af7aa5058a617383eb9deef4a848778d04ae47f66600ddd53788703344e'
    model,payload=ev.checked_load(io,str(ev.CHECKPOINT),final_sha)
    checks['actual_final10000_architecture']=payload['curve'][-1]['step']==10000 and payload['config']['physical_eval_only_step']==10000 and payload['identity']['parent_sha256']==ev.PARENT_SHA
    checks['parent194878_frozen']=sum(p.numel() for p in model.parent.parameters())==194878 and all(not p.requires_grad for p in model.parent.parameters())
    checks['head2730']=sum(p.numel() for p in model.readout.parameters())==2730
    d=ev.arrays(ev.PARENT_RIGHT/'neural_decisions.npz');t=ev.arrays(ev.PARENT_RIGHT/'trace_1ms.npz');g=ev.arrays(ev.PARENT_RIGHT/'geometry_1ms.npz')
    state=model.initial_state(1);means=[];states=[];actions=[]
    with torch.no_grad():
        for x,u in zip(d['X238'],d['command']):
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
            means.append(mean[0].numpy());states.append(torch.cat(state,dim=-1)[0].numpy());actions.append(mean[0].tanh().numpy())
    checks['actual10000_all2880_right_mean_c6b1500_exact']=np.array_equal(np.asarray(means),d['mean'])
    checks['actual10000_all2880_right_action_c6b1500_exact']=np.array_equal(np.asarray(actions),d['policy_action'])
    checks['actual10000_all2880_right_full192state_c6b1500_exact']=np.array_equal(np.asarray(states),d['hidden_after'])
    sys.path.insert(0,str(ev.BASE));import identity
    ident=json.loads((ev.PARENT_RIGHT/'IDENTITY.json').read_text())
    comparison=ev.verify_parent_right(t,d,g,ident,identity)
    checks['full_right_array_comparator_pass']=comparison['all_exact'] and all(comparison['checks'].values())
    bad=dict(d);bad['hidden_after']=d['hidden_after'].copy();bad['hidden_after'][1000,191]+=.01
    rejected=False
    try:ev.verify_parent_right(t,bad,g,ident,identity)
    except ValueError:rejected=True
    checks['right_residual_state_corruption_rejected']=rejected
    import verify_actual as audit
    fa=functions(audit.derived_source());fo=functions((ev.PRIMARY/'verify_actual.py').read_text())
    for name in ('arrays','ROOT_HEAD_AUDIT','main'):checks['original_audit_function_unchanged:'+name]=fa[name]==fo[name]
    # All inherited source maps are checked, and the saved full right reference
    # remains anchored to the qualified1500 source/checkpoint, not continuation cfg.
    inherited={p:h for p,h in prereg['frozen_primary_sources'].items()}
    for path,h in inherited.items():
        if ev.sha256(path)!=h:raise ValueError('source drift '+path)
    checks['no_mujoco_or_physics_backend_import']=all(n not in sys.modules for n in ('mujoco','physics','student_wave'))
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes={str(p):ev.sha256(p) for p in files},
                frozen_predecessor_sources=inherited,actual_offline_model=str(ev.CHECKPOINT),actual_offline_model_sha256=final_sha,
                parent_right_reference_comparison=comparison,parent_right_audit_sha256=ev.sha256(ev.PARENT_AUDIT),
                final10000_checkpoint_loaded=True,new_loader_status='actual new strictmodelio checked final10000 and ancestryc6b1500',
                physical_steps=0,optimizer_updates=0,Core_constructions=0,
                limitation='Actual final10000 head inference on qualified1500 saved right observations/arrays only; no new physical execution or final motor qualification.')
    (HERE/'READINESS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,optimizer_updates=0)))
    if not report['all_passed']:raise ValueError('preparation checks failed '+json.dumps(checks))

if __name__=='__main__':main()
