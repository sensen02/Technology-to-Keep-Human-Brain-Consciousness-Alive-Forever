"""Independent physical evaluation of new conditional neural motor (no odor yet)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from conditional_motor import HERE,BASE,CHECKPOINT,ConditionalMotor,load_baseline_actor,load_motor,sha256
import student_wave as sw
from identity import capture,verify,atomic_json


def evaluate(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    checks=json.loads((HERE/'CONDITIONAL_CHECKS.json').read_text())
    if not checks['all_passed']:
        raise ValueError('neural conditional checks required')
    for p,digest in checks['source_hashes'].items():
        if sha256(p)!=digest:raise ValueError('conditional check source drift')
    sw.teacher_preflight();sw.student_preflight()
    c=sw.Core(adhesion=False)
    original,_=sw.load_student(CHECKPOINT,c)  # Actual source/asset/runtime provenance, not bare state_dict.
    if args.checkpoint:
        model,payload=load_motor(args.checkpoint)
    else:
        model=ConditionalMotor(original.actor)
    model.eval()
    paths=[HERE/'evaluate_motor.py',HERE/'conditional_motor.py',HERE/'PREREG.json',HERE/'CONDITIONAL_CHECKS.json',CHECKPOINT]
    if args.checkpoint:paths.append(Path(args.checkpoint).resolve())
    ident=capture(c,dict(kind='conditional neural motor evaluation',cli=vars(args),settle_s=1.,warmup_s=4.8,autonomous_s=9.6,command_order=['forward_modulation','signed_turn_modulation'],actor_target_queries=False),extra_paths=paths)
    atomic_json(out/'IDENTITY.json',ident)
    tr=sw.Trace(full=True);sw.settle(c,tr)
    teacher=sw.StaticTeacher(c,sw.teacher_config(3))
    previous=np.zeros(42);state=model.initial_state(1)
    decisions=[];yaws=[]
    command=np.array([args.forward,args.turn],dtype=np.float64)
    obsrv,events=sw.tracker(c),[]
    transition=None
    for k in range(2880):
        warm=k<960
        if k==960:
            before=sw.continuity_snapshot(c,previous,state)
            obsrv,events=sw.tracker(c),[]
            after=sw.continuity_snapshot(c,previous,state)
            exact={key:bool(np.array_equal(before[key],after[key])) for key in before}
            if not all(exact.values()):raise ValueError('handover discontinuity')
            transition=dict(time_s=float(c.d.time),all_exact=True,exact=exact,before_sha256=sw.snapshot_hashes(before),after_sha256=sw.snapshot_hashes(after),no_reset=True)
        x,_=sw.observations(c,previous,(2880-k)*.005)
        cmd=np.zeros(2) if warm else command
        hb=state.clone()
        with torch.no_grad():
            mean,state=model.step(torch.tensor(x[None],dtype=torch.float64),torch.tensor(cmd[None],dtype=torch.float64),state)
        policy=mean[0].tanh().numpy()
        if warm:
            action,_=teacher.action(c,k*.005);source=sw.SOURCE['teacher']
        else:
            action=policy;source=sw.SOURCE['student']
        decisions.append(dict(time=float(c.d.time),X238=x,command=cmd.copy(),hidden_before=hb[0].numpy(),hidden_after=state[0].numpy(),mean=mean[0].numpy(),policy_action=policy,actually_applied_action=action.copy(),control_source=source))
        for _ in range(5):
            tr.tick(c,action,source)
            rot=c.d.xmat[c.th].reshape(3,3)
            yaws.append(float(np.arctan2(rot[1,0],rot[0,0])))
            if not warm:events.extend(obsrv.measure(c)['events'])
        previous=action.copy()
    auto=tr.summary(c,sw.SOURCE['student'])
    cycles=sw.cycle_report(obsrv,events)
    # Source window is 1000 settle +4800 teacher; yaws exclude settle.
    yaw=np.unwrap(np.asarray(yaws))
    yaw_delta=float(yaw[-1]-yaw[4799])
    traceinfo=tr.save(out/'trace_1ms.npz')
    np.savez_compressed(out/'neural_decisions.npz',**{key:np.asarray([row[key] for row in decisions]) for key in decisions[0]},yaw_after_each_control_ms=np.asarray(yaws))
    verify(ident)
    result=dict(status='MEASURED_NOT_FULL_NATIVE_CERTIFICATION',autonomous=auto,autonomous_cycles=cycles,transition=transition,
                yaw_change_rad=yaw_delta,command=command.tolist(),trace=traceinfo,identity_sha256=ident['identity_sha256'],
                baseline_checkpoint_sha256=sha256(CHECKPOINT),neural_decisions_sha256=sha256(out/'neural_decisions.npz'),
                source_frame='yaw world positive counterclockwise; original world-X gates reported unchanged',
                physical_steps=1232000,teacher_calls_after_handover=0,extra_forces=False,odor_or_food_success=False)
    if not args.checkpoint and not np.any(command):
        reference=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
        with np.load(reference,allow_pickle=False) as ref:
            exact={name:bool(np.array_equal(np.asarray([r[name] for r in decisions]),ref[name])) for name in ('X238','hidden_before','hidden_after','mean','policy_action','actually_applied_action','control_source')}
        result['baseline_actual_physics_neural_parity']=exact
        if not all(exact.values()):raise ValueError('neutral physical replay differs from verified baseline')
    atomic_json(out/'RESULT.json',result)
    print(json.dumps(dict(out=str(out),gates=auto['gates'],slip=auto['slip_ratio_max'],yaw_change_rad=yaw_delta,cycles=cycles['complete_per_leg'],neutral_parity=result.get('baseline_actual_physics_neural_parity')),ensure_ascii=False),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--checkpoint');p.add_argument('--forward',type=float,default=0.);p.add_argument('--turn',type=float,default=0.);p.add_argument('--seed',type=int,default=20261110)
    a=p.parse_args()
    if not np.isfinite([a.forward,a.turn]).all() or max(abs(a.forward),abs(a.turn))>1:raise ValueError('commands must be in [-1,1]')
    evaluate(a)
