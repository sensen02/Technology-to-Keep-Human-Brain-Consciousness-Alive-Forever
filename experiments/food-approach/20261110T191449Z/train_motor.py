"""Fixed-budget conditional BC with neutral retention; no physics or teacher calls."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from conditional_motor import HERE,ConditionalMotor,load_baseline_actor,load_motor,save_motor,sha256


def load_data():
    design=json.loads((HERE/'MOTOR_TRAINING_DESIGN.json').read_text())
    identity=json.loads((HERE/'teacher/IDENTITY.json').read_text())
    for p,h in identity['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('teacher source drift '+p)
    reports=[];episodes=[];datahashes={}
    baseline=load_baseline_actor().eval()
    for name in design['candidate_teacher_conditions']:
        result_path=HERE/'teacher'/name/'RESULT.json'
        r=json.loads(result_path.read_text())
        cfg=r['condition'];full=cfg['speed_scale']==1
        eligible=(r['early_stop'] is None and r['samples']['decisions']==2880 and r['strict_nonfoot_zero'] and r['foot_support_mean']>=.9 and r['com_height_min']>=1.2 and r['slip_ratio_max']<=.5 and r['cycles']['all_legs_completed'] and r['requested_sign_agreement'] and r['forward_signed_native_per_s']>=(.05 if full else .025))
        reports.append(dict(name=name,eligible=bool(eligible),original_all5_gates=r['original_all5_gates']))
        if not eligible:continue
        path=Path(r['samples']['path'])
        if sha256(path)!=r['samples']['sha256']:raise ValueError('teacher samples changed')
        datahashes[str(path)]=sha256(path);datahashes[str(result_path)]=sha256(result_path)
        with np.load(path,allow_pickle=False) as z:
            x=torch.tensor(z['X238'],dtype=torch.float64)
            y=torch.tensor(z['Y42'],dtype=torch.float64)
            u=torch.tensor(z['command2'],dtype=torch.float64)
        if x.shape!=(2880,238) or y.shape!=(2880,42) or u.shape!=(2880,2) or not torch.equal(u[:960],torch.zeros_like(u[:960])):
            raise ValueError('teacher source dimensions or neutral prime incorrect')
        # Warmup primes the actor but does not ask it to refit imperfect teacher predictions.
        state=baseline.initial_state(1)
        with torch.no_grad():
            for k in range(960):
                mean,state=baseline.step(x[k:k+1],state);y[k]=mean[0].tanh()
        episodes.append((name,x,u,y))
    admitted={e[0] for e in episodes}
    if not {'left_full','right_full'}<=admitted:
        raise ValueError('paired valid full-speed teacher demonstrations required: '+json.dumps(reports))
    # Original exact deployed behavior at neutral conditions provides retention, not validation.
    reference=HERE.parent.parent/'banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
    with np.load(reference,allow_pickle=False) as z:
        x=torch.tensor(z['X238'],dtype=torch.float64);y=torch.tensor(z['policy_action'],dtype=torch.float64)
    episodes.append(('neutral_retention',x,torch.zeros((2880,2),dtype=torch.float64),y))
    datahashes[str(reference)]=sha256(reference)
    return episodes,reports,datahashes,design


def metrics(model,episodes):
    scores={}
    with torch.no_grad():
        for name,x,u,y in episodes:
            mean,_=model.sequence(x[:,None],u[:,None]);error=(mean[:,0].tanh()-y).square()
            scores[name]=dict(MSE=float(error.mean()),post_prime_MSE=float(error[960:].mean()))
    return scores


def train(args):
    torch.set_num_threads(1);torch.manual_seed(20261110);np.random.seed(20261110)
    checks=json.loads((HERE/'CONDITIONAL_CHECKS.json').read_text())
    if not checks['all_passed']:raise ValueError('interface checks required')
    for p,h in checks['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('conditional source drift')
    episodes,eligibility,datahashes,design=load_data()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    sourcepaths=[HERE/n for n in ('train_motor.py','conditional_motor.py','MOTOR_TRAINING_DESIGN.json','PREREG.json','CONDITIONAL_CHECKS.json')]
    hashes={str(p):sha256(p) for p in sourcepaths}
    if args.stage=='adapter':
        model=ConditionalMotor(load_baseline_actor());model.set_training_stage('adapter')
        optimizer=torch.optim.Adam([model.command_encoder.weight],lr=.003)
        start,end=0,100;curve=[]
    else:
        if not args.resume:raise ValueError('finetune requires exact adapter100 checkpoint')
        model,parent=load_motor(args.resume)
        if len(parent['curve'])!=101 or parent['curve'][-1]['step']!=100 or parent['config']['stage']!='adapter':raise ValueError('requires adapter100 parent')
        for p,h in parent['config']['data_hashes'].items():
            if sha256(p)!=h:raise ValueError('parent dataset drift')
        model.set_training_stage('finetune')
        optimizer=torch.optim.Adam([model.command_encoder.weight],lr=.003)
        optimizer.load_state_dict(parent['optimizer']);optimizer.param_groups[0]['lr']=.001
        optimizer.add_param_group(dict(params=list(model.base.parameters()),lr=.0001))
        torch.set_rng_state(parent['rng']['torch']);np.random.set_state(parent['rng']['numpy'])
        start,end=100,500;curve=parent['curve'].copy()
    config=dict(stage=args.stage,seed=20261110,design=design,eligibility=eligibility,data_hashes=datahashes,source_hashes=hashes,
                loss='equal episode bounded action BC, exact full contiguous2880 BPTT; priming rows distill parent; neutral retained',
                resume=None if not args.resume else str(Path(args.resume).resolve()),resume_sha256=None if not args.resume else sha256(args.resume))
    (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
    started=time.monotonic()
    if start==0:
        initial=metrics(model,episodes);curve.append(dict(step=0,metrics=initial,wall_s=0.))
        print(json.dumps(curve[-1]),flush=True)
    for step in range(start+1,end+1):
        optimizer.zero_grad(set_to_none=True)
        # Sequential graph/backward per episode bounds peak memory while accumulating exact equal weights.
        losses={}
        for name,x,u,y in episodes:
            mean,_=model.sequence(x[:,None],u[:,None])
            loss=(mean[:,0].tanh()-y).square().mean()/len(episodes)
            loss.backward();losses[name]=float(loss.detach())*len(episodes)
            del mean,loss
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
        optimizer.step()
        row=dict(step=step,train_MSE=losses,gradient_norm=float(grad),wall_s=time.monotonic()-started)
        curve.append(row)
        if step%25==0:print(json.dumps(row),flush=True)
        if step%100==0:
            row['metrics']=metrics(model,episodes)
            for p,h in {**hashes,**datahashes}.items():
                if sha256(p)!=h:raise ValueError('training source/data drift '+p)
            ck=out/f'checkpoint_{step:06d}.pt'
            save_motor(ck,model,optimizer,config,curve)
            restored,payload=load_motor(ck)
            if not all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()):raise ValueError('saved weights mismatch')
            if payload['optimizer']['param_groups']!=optimizer.state_dict()['param_groups']:raise ValueError('saved optimizer groups mismatch')
            print(json.dumps(dict(checkpoint=str(ck),sha256=sha256(ck),step=step,saved_reload_exact=True,metrics=row['metrics'])),flush=True)
    result=dict(training_complete=True,global_steps=end,stage=args.stage,last_checkpoint=str(ck),checkpoint_sha256=sha256(ck),curve=curve,
                physical_evaluation='NOT_RUN_BY_TRAINING',elapsed_wall_s=time.monotonic()-started,source_hashes=hashes,data_hashes=datahashes)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--stage',choices=['adapter','finetune'],required=True);p.add_argument('--resume')
    train(p.parse_args())
