"""CODEONLY preparation or exactly one authorized selected native LEFT recording."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
from contract import HERE, ROOT, PYTHON, admitted, require, load, sha, save_new, pinned


def selected_model_preflight(checkpoint,digest,proof):
    import execute_native
    import numpy as np
    import torch
    torch.set_num_threads(1)
    parent=execute_native.parent_module();sw,identity,io=parent.pin()
    model,payload=io.load_checked(checkpoint,digest,require_final=True)
    left=Path(proof['cases']['left']['result']).parent
    with np.load(left/'neural_decisions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in ('X238','command','hidden_before','hidden_after','mean','policy_action')}
    require(d['X238'].shape==(2880,238) and d['hidden_before'].shape==(2880,192),'full matched selection neural evidence required')
    state=model.initial_state(1);means=[];states=[];actions=[]
    with torch.no_grad():
        for x,u in zip(d['X238'],d['command']):
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
            means.append(mean[0].numpy());states.append(parent.flatten(state)[0].numpy());actions.append(mean[0].tanh().numpy())
    checks=dict(means_all2880_bitexact=np.array_equal(means,d['mean']),hidden_all2880_192_bitexact=np.array_equal(states,d['hidden_after']),actions_all2880_bitexact=np.array_equal(actions,d['policy_action']),hidden_chain_exact=np.array_equal(d['hidden_before'][1:],d['hidden_after'][:-1]),initial_hidden_zero=bool(np.all(d['hidden_before'][0]==0)))
    require(all(checks.values()),'selected checkpoint replay did not match engineering LEFT')
    return {k:bool(v) for k,v in checks.items()}


def command(argv,log):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with Path(log).open('x') as stream:
        result=subprocess.run([str(PYTHON),'-B',*map(str,argv)],env=env,stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0,'native stage failed '+str(argv[0])+': exit '+str(result.returncode)+'; preserve log/partial output')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--launch-authorized',action='store_true')
    parser.add_argument('--engineering-summary');parser.add_argument('--checkpoint');parser.add_argument('--sha256');parser.add_argument('--run-plan',default=str(HERE/'RUN_PLAN.json'))
    args=parser.parse_args()
    if args.prepare_only:
        import check_prepared
        report=check_prepared.check()
        print(json.dumps(dict(code_checks_pass=report['all_passed'],actual_native_ready=False,selected_candidate_required=True,Core_constructions=0,physics_steps=0)))
        return
    require(args.launch_authorized,'CODEONLY: explicit parent --launch-authorized required')
    require(args.checkpoint and args.sha256 and args.engineering_summary,'selected checkpoint/SHA/engineering summary required')
    plan,proof=admitted(args.run_plan,args.checkpoint,args.sha256)
    require(Path(args.engineering_summary).resolve()==Path(plan['engineering_summary']).resolve(),'CLI engineering proof differs from registered plan')
    require(shutil.disk_usage(HERE).free>=12*1024**3,'reserve at least12GiB free before selected native run')
    require(not (HERE/'selected_native').exists(),'single selected native attempt already has output; never overwrite or silently repeat')
    checks=selected_model_preflight(args.checkpoint,args.sha256,proof)
    case=HERE/'selected_native';case.mkdir()
    source_case=Path(proof['cases']['left']['result']).parent.parent
    save_new(case/'EVAL_PREREG.json',load(source_case/'EVAL_PREREG.json'))
    save_new(case/'SOURCE_BARRIER.json',dict(selection_hashes=proof['hashes'],native_source_hashes=plan['preparation_source_hashes'],run_plan_sha256=sha(args.run_plan)))
    save_new(case/'READINESS.json',dict(all_passed=True,checks=checks,source_hashes=plan['preparation_source_hashes'],actual_selected_checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=args.sha256,scope='actual selected model/three matched engineering cases verified; authorizes one recording, not a native PASS'))
    state=dict(phase='recording selected LEFT once',checkpoint=args.checkpoint,checkpoint_sha256=args.sha256,physics_launched=True,native_passed=False)
    save_new(HERE/'LAUNCH_IDENTITY.json',state)
    run=case/'EVAL_left'
    command([HERE/'execute_native.py','--run-plan',args.run_plan,'--checkpoint',args.checkpoint,'--sha256',args.sha256,'--out',run,'--launch-authorized'],HERE/'native_recording.log')
    command([HERE/'verify_native.py','--run',run,'--run-plan',args.run_plan,'--checkpoint',args.checkpoint,'--sha256',args.sha256,'--out',HERE/'NATIVE_AUDIT.json'],HERE/'native_audit.log')
    audit=load(HERE/'NATIVE_AUDIT.json')
    save_new(HERE/'PIPELINE_RESULT.json',dict(status=audit['status'],native_audit=str(HERE/'NATIVE_AUDIT.json'),native_audit_sha256=sha(HERE/'NATIVE_AUDIT.json'),run=str(run),checkpoint_sha256=args.sha256,video='NOT_RUN: separate parent render authorization and actual image review required',full_food_goal_status='NOT_ESTABLISHED'))
    print(json.dumps(dict(status=audit['status'],run=str(run),full_goal_status='NOT_ESTABLISHED')))

if __name__=='__main__':main()
