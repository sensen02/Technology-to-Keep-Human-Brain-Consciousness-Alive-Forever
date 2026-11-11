"""Offline real saved-data parity; no physical model construction/integration."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
from flat_motor import FlatResidualMotor
from evaluate_residual_profile import ROOT,HERE,prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'residual_candidate'))
    from evaluate_candidate import load_checked
    from residual_motor import sha256
    model,payload=load_checked(args.checkpoint,args.sha256)
    wrapper=FlatResidualMotor(model).eval();checks={};datasets={}
    with torch.no_grad():
        init=model.initial_state(1);flat=wrapper.initial_state(1)
        checks['initial_cat_exact']=torch.equal(flat,torch.cat(init,dim=-1))
        for name in ('left_full','right_full'):
            path=ROOT/'teacher'/name/'SAMPLES_5MS.npz'
            with np.load(path,allow_pickle=False) as z:
                x=torch.tensor(z['X238'],dtype=torch.float64)[:,None];u=torch.tensor(z['command2'],dtype=torch.float64)[:,None]
            if x.shape!=(2880,1,238) or u.shape!=(2880,1,2):raise ValueError('actual saved dataset count mismatch')
            means,states=model.sequence(x,u)
            flat_means,flat_states=wrapper.sequence(x,u)
            checks[name+'_all2880_means_exact']=torch.equal(means,flat_means)
            checks[name+'_all192_states_exact']=torch.equal(torch.cat(states,dim=-1),flat_states)
            tuple_state=model.initial_state(1);flat_state=wrapper.initial_state(1);step_exact=True
            for k in range(2880):
                m,tuple_state=model.step(x[k],u[k],tuple_state)
                fm,flat_state=wrapper.step(x[k],u[k],flat_state)
                step_exact=step_exact and torch.equal(m,fm) and torch.equal(torch.cat(tuple_state,dim=-1),flat_state)
            checks[name+'_each_actual_step_exact']=step_exact
            # Resume across actual neutralprime boundary without resetting either128/64 slice.
            _,prefix=wrapper.sequence(x[:960],u[:960])
            resumed,resumed_states=wrapper.sequence(x[960:],u[960:],prefix[-1])
            checks[name+'_handover_resume_exact']=torch.equal(resumed,flat_means[960:]) and torch.equal(resumed_states,flat_states[960:])
            datasets[str(path)]=sha256(path)
        try:wrapper.step(x[0],u[0],torch.zeros((1,191),dtype=torch.float64))
        except ValueError:checks['reject_wrong_flat_state']=True
        else:checks['reject_wrong_flat_state']=False
    original=prepare() # Creates explicit hook but does not call it or initialize physics.
    draft=json.loads((HERE/'PREREG.json').read_text())
    checks['fiveprofiles_unchanged']=draft['profiles']==list(original.PROFILES) and draft['turns_first_second_4p8s']=={k:list(v) for k,v in original.PROFILES.items()}
    checks['no_physical_imports']=not any(name in sys.modules for name in ('mujoco','physics','student_wave'))
    for path in HERE.glob('*.py'):
        compile(path.read_text(),str(path),'exec');checks['syntax:'+path.name]=True
    report=dict(all_passed=all(checks.values()),checks=checks,checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256(args.checkpoint),
        saved_data_sha256=datasets,physical_steps=0,training_updates=0,sources={str(path):sha256(path) for path in HERE.iterdir() if path.suffix in ('.py','.json') and path.name!='OFFLINE_CHECKS.json'},
        limitations='Saved-observation parity only; no physical command profile execution or motor qualification assertion.')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,out=str(out))))
    if not report['all_passed']:raise ValueError('flat adapter parity failed')

if __name__=='__main__':main()
