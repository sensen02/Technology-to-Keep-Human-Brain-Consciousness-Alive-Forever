"""Prepared standalone fixed500 physical evaluator. NOT run during candidate preparation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from residual_motor import HERE,BASE,CHECKPOINT,load_candidate,sha256
from cached_training import verify_hashes


def load_checked(path,digest):
    if sha256(path)!=digest:raise ValueError('selected checkpoint digest mismatch')
    model,payload=load_candidate(path);cfg=payload['config']
    verify_hashes(cfg['source_hashes']);verify_hashes(cfg['data_hashes'])
    if cfg['steps']!=500 or cfg['eval_only_step']!=500 or len(cfg['curve'])!=501 or cfg['curve'][-1]['step']!=500:raise ValueError('only fixed500 may be physically evaluated')
    return model.eval(),payload


def evaluate(args):
    torch.set_num_threads(1);torch.manual_seed(20261110)
    model,payload=load_checked(args.checkpoint,args.expected_sha256)
    sys.path.insert(0,str(BASE))
    import student_wave as sw
    from identity import capture,verify,atomic_json
    sw.teacher_preflight();sw.student_preflight()
    c=sw.Core(adhesion=False);original,_=sw.load_student(CHECKPOINT,c)
    if not all(torch.equal(v,model.base.state_dict()[k]) for k,v in original.actor.state_dict().items()):raise ValueError('runtime baseline differs')
    out=Path(args.out).resolve()
    if not out.is_relative_to(HERE):raise ValueError('output must be within candidate')
    out.mkdir(exist_ok=False)
    paths=[HERE/'evaluate_candidate.py',Path(args.checkpoint).resolve()]+[Path(p) for p in payload['config']['source_hashes']]
    ident=capture(c,dict(kind='alternate frozen base residual fixed500 evaluation',cli=vars(args),settle_s=1.,warmup_s=4.8,autonomous_s=9.6,actor_target_queries=False),extra_paths=paths)
    atomic_json(out/'IDENTITY.json',ident)
    tr=sw.Trace(full=True);sw.settle(c,tr);teacher=sw.StaticTeacher(c,sw.teacher_config(3))
    previous=np.zeros(42);state=model.initial_state(1);obsrv=sw.tracker(c);events=[];rows=[];yaws=[];transition=None
    cmd=np.array([0.,{'neutral':0.,'left':1.,'right':-1.}[args.condition]])
    for k in range(2880):
        warm=k<960
        if k==960:
            before=sw.continuity_snapshot(c,previous,torch.cat(state,dim=-1))
            obsrv=sw.tracker(c);events=[]
            after=sw.continuity_snapshot(c,previous,torch.cat(state,dim=-1))
            exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
            if not all(exact.values()):raise ValueError('handover discontinuity')
            transition=dict(time_s=float(c.d.time),exact=exact,no_reset=True,both_hidden_states_included=True,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after))
        x,_=sw.observations(c,previous,(2880-k)*.005);u=np.zeros(2) if warm else cmd
        hb=torch.cat(state,dim=-1).clone()
        with torch.no_grad():mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(u[None],dtype=torch.float64),state)
        policy=mean[0].tanh().numpy()
        if warm:action,_=teacher.action(c,k*.005);source=sw.SOURCE['teacher']
        else:action=policy;source=sw.SOURCE['student']
        rows.append(dict(time=float(c.d.time),X238=x,command=u.copy(),hidden_before=hb[0].numpy(),hidden_after=torch.cat(state,dim=-1)[0].numpy(),mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=source))
        for _ in range(5):
            tr.tick(c,action,source);rot=c.d.xmat[c.th].reshape(3,3);yaws.append(float(np.arctan2(rot[1,0],rot[0,0])))
            if not warm:events.extend(obsrv.measure(c)['events'])
        previous=action.copy()
    auto=tr.summary(c,sw.SOURCE['student']);cycles=sw.cycle_report(obsrv,events);yaw=np.unwrap(np.asarray(yaws));delta=float(yaw[-1]-yaw[4799])
    trace=tr.save(out/'trace_1ms.npz')
    np.savez_compressed(out/'neural_decisions.npz',**{key:np.asarray([r[key] for r in rows]) for key in rows[0]},yaw_after_each_control_ms=np.asarray(yaws))
    verify(ident);verify_hashes(payload['config']['source_hashes']);verify_hashes(payload['config']['data_hashes'])
    neutral_parity=None
    if args.condition=='neutral':
        reference=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
        with np.load(reference,allow_pickle=False) as ref:
            neutral_parity={key:bool(np.array_equal(np.asarray([r[key] for r in rows]),ref[key])) for key in ('X238','mean','policy_action','actually_applied_action','control_source')}
            neutral_parity['base_hidden']=bool(np.array_equal(np.asarray([r['hidden_after'][:128] for r in rows]),ref['hidden_after']))
        if not all(neutral_parity.values()):raise ValueError('actual neutral baseline parity failed')
    sign_ok=args.condition=='neutral' or (abs(delta)>=.01 and delta*cmd[1]>0)
    acceptance=bool(all(auto['gates'].values()) and cycles['all_legs_completed'] and sign_ok)
    result=dict(candidate_engineering_pass=acceptance,yaw_sign_and_magnitude_pass=sign_ok,neutral_parity=neutral_parity,status='MEASURED_NOT_FULL_NATIVE_CERTIFICATION',autonomous=auto,autonomous_cycles=cycles,transition=transition,yaw_change_rad=delta,command=cmd.tolist(),trace=trace,identity_sha256=ident['identity_sha256'],checkpoint_sha256=args.expected_sha256,physical_steps=1232000,teacher_calls_after_handover=0,extra_forces=False,odor_or_food_success=False,state_layout='base128,residual64',source_frame='yaw world positive CCW; original worldX gates unchanged')
    atomic_json(out/'RESULT.json',result);print(json.dumps(dict(out=str(out),gates=auto['gates'],slip=auto['slip_ratio_max'],yaw_change_rad=delta)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--condition',choices=['neutral','left','right'],required=True);evaluate(p.parse_args())
