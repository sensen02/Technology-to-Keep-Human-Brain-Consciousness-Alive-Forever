"""Offline evaluation contracts and tuple-state identity. No Core or physics imports."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np
from evaluate_motor import HERE,ROOT,COMMANDS,command_at,score,jsonable,parser,flatten,sha256


def main():
    checks={}
    for name in ('evaluate_motor.py','verify_actual.py','pipeline.py','check_prepared.py'):
        tree=ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
        if name=='verify_actual.py':
            bad=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ('Core','mj_step','mj_step1','mj_step2','mj_forward','tick')) or (isinstance(n.func,ast.Name) and n.func.id=='Core'))]
            checks['auditor_no_physical_construction_or_step']=not bad
    checks['strict_json_numpy_scalars']=json.loads(json.dumps(jsonable(dict(flag=np.bool_(True),n=np.int64(3),a=np.array([1.,2.]))),allow_nan=False))==dict(flag=True,n=3,a=[1.,2.])
    auto=dict(all_gates=True,nonfoot_force_max_native=0.)
    cycles=dict(all_legs_completed=True)
    checks['neutral_no_direction_gate']=score('neutral',auto,cycles,-1.)['eligible']
    checks['left_correct_yaw']=score('left',auto,cycles,.01)['eligible']
    checks['left_wrong_yaw_reject']=not score('left',auto,cycles,-.02)['eligible']
    checks['right_correct_yaw']=score('right',auto,cycles,-.01)['eligible']
    checks['strict_nonfoot_reject']=not score('left',dict(all_gates=True,nonfoot_force_max_native=1e-9),cycles,.1)['eligible']
    checks['cycles_reject']=not score('left',auto,dict(all_legs_completed=False),.1)['eligible']
    checks['original_gates_reject']=not score('left',dict(all_gates=False,nonfoot_force_max_native=0.),cycles,.1)['eligible']
    for c in COMMANDS:
        u=np.asarray([command_at(c,k) for k in range(2880)])
        checks[c+'_zero_prime_and_firstANN']=np.array_equal(u[:961],np.zeros((961,2)))
        checks[c+'_fixed_target_after1s']=np.array_equal(u[1160:],np.tile(COMMANDS[c],(1720,1)))
        checks[c+'_forward_zero']=np.max(np.abs(u[:,0]))==0
        if c!='neutral':
            p=ROOT.parent/f'20261110T191449Z/teacher/{c}_full/SAMPLES_5MS.npz'
            with np.load(p,allow_pickle=False) as z:expected=z['command2'].copy()
            checks[c+'_actual2880teacher_commands_match']=np.max(np.abs(u-expected))<=1e-14
    a=parser().parse_args(['--checkpoint','/abs/model.pt','--sha256','a'*64,'--condition','neutral','--out','/abs/out'])
    checks['parser_default_no_authorization']=not a.training_complete and not a.native
    from evaluate_motor import evaluate
    try:evaluate(a);guard=False
    except ValueError as exc:guard='parent' in str(exc)
    checks['launch_guard_before_loading_backend']=guard
    checks['pre_backend_no_mujoco']='mujoco' not in sys.modules and 'physics' not in sys.modules
    # Real immutable old tuple actor confirms logging flatten/split never changes dynamics.
    import torch
    torch.set_num_threads(1)
    old=ROOT.parent/'20261110T191449Z/residual_candidate'
    spec=importlib.util.spec_from_file_location('_round2_offline_old_residual',old/'residual_motor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ck=old/'TRAIN_RESIDUAL_500/checkpoint_000500.pt'
    if sha256(ck)!='040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3':raise ValueError('old checkpoint drift')
    model,_=module.load_candidate(ck);model.eval()
    for name in ('left','right'):
        path=ROOT.parent/f'20261110T191449Z/teacher/{name}_full/SAMPLES_5MS.npz'
        with np.load(path,allow_pickle=False) as z:x=z['X238'].copy();u=z['command2'].copy()
        state=model.initial_state(1);flatstate=flatten(state).clone();same_mean=same_state=True
        with torch.no_grad():
            for k,(xx,uu) in enumerate(zip(x,u)):
                obs=torch.tensor(xx[None],dtype=torch.float64);cmd=torch.tensor(uu[None],dtype=torch.float64)
                mean,state=model.step(obs,cmd,state)
                reconstructed=(flatstate[:,:128],flatstate[:,128:])
                mean2,reconstructed=model.step(obs,cmd,reconstructed);flatstate=flatten(reconstructed)
                same_mean=same_mean and torch.equal(mean,mean2);same_state=same_state and torch.equal(flatten(state),flatstate)
                if k==959:checks[name+'_handover192_shape']=tuple(flatstate.shape)==(1,192)
        checks[name+'_all2880_flatten_split_means_exact']=same_mean
        checks[name+'_all2880_both_states_exact']=same_state
    reference=ROOT.parent.parent/'banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
    with np.load(reference,allow_pickle=False) as z:
        neutral_x=z['X238'].copy();neutral_mean=z['mean'].copy();neutral_h=z['hidden_after'].copy()
    torch.manual_seed(917)
    initial=model.initial_state(1);state=(initial[0],torch.randn((1,64),dtype=torch.float64))
    neutral_means=[];neutral_hidden=[]
    with torch.no_grad():
        for row in neutral_x:
            mean,state=model.step(torch.tensor(row[None],dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64),state)
            neutral_means.append(mean[0].numpy());neutral_hidden.append(state[0][0].numpy())
    checks['all2880_neutral_mean_exact_with_nonzero_residual_memory']=np.array_equal(np.asarray(neutral_means),neutral_mean)
    checks['all2880_neutral_basehidden_exact_with_nonzero_residual_memory']=np.array_equal(np.asarray(neutral_hidden),neutral_h)
    checks['tuple_check_no_physics_backend']='mujoco' not in sys.modules and 'physics' not in sys.modules
    # Saved quaternion comparison verifies the root-yaw audit formula against actual old records.
    oldrun=ROOT.parent/'20261110T203537Z/EVAL_FAST_neutral'
    with np.load(oldrun/'trace_1ms.npz',allow_pickle=False) as z:q=z['qpos'][:,3:7].copy()
    with np.load(oldrun/'neural_decisions.npz',allow_pickle=False) as z:y=z['yaw_after_each_1ms'].copy()
    qyaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    checks['rootquat_yaw_actual15400_matches']=np.max(np.abs(np.angle(np.exp(1j*(qyaw-y)))))<1e-10
    sources={str(HERE/n):sha256(HERE/n) for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','check_prepared.py','EVAL_PREREG.json')}
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=sources,physical_integration_steps=0,optimizer_updates=0,
                old_reference_checkpoint=str(ck),old_reference_sha256=sha256(ck),
                scope='CODE_ONLY contracts and real saved-input tuple-state equivalence; new1500 loader not yet physically executed; no behavioral qualification')
    (HERE/'READINESS.json').write_text(json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(jsonable(report),indent=2))
    if not report['all_passed']:raise ValueError('prepared evaluation checks failed')

if __name__=='__main__':main()
