"""Finite physical validation for arc-to-straight handoff and learned replay."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from controller import HERE,TURN_SOURCE,Controller,turn
from verify_native import Record,SO

W=turn.W
stan=turn.stand
read=turn.read
BASE_PACE=.071713
DT=.001
# Command features in engineering units, signed yaw positive for left.
UNITS=['native length/s','radians/s']


def projected_metrics(k,ms,a,b,*,yaw_index):
    """Retain legacy world displacement denominator for load-weighted slip."""
    pose=k.poses[a*80:b*80]
    contact=k.forces[a*80:b*80]
    use=b-a
    if len(contact)!=use*80 or use<=0:
        return dict(passed=False)
    yaw=np.unwrap(ms['yaw'])
    h=np.array([np.cos(yaw[yaw_index]),np.sin(yaw[yaw_index])])
    delta=ms['com'][b,:2]-ms['com'][a,:2]
    vel=np.linalg.norm(delta)/(use*DT)
    slipping=contact[:,6:12].sum(0)/np.maximum(1e-12,contact[:,:6].sum(0))
    count=turn.PivotCycles(W)
    for i in range(max(0,a-10),b+1):
        count.update(ms['time'][i],ms['loads'][i],ms['padz'][i])
    out=dict(forward_native_s=float(delta@h/(use*DT)),net_yaw_deg=float(np.rad2deg(yaw[b]-yaw[a])),
         minimum_com=float(min(pose[:,2].min(),ms['com'][b,2])),nonfoot_peak=float(contact[:,12].max()),
         foot_W=float(contact[:,:6].sum(1).mean()/W),native_steps=len(contact),
         max_normalized_slip=float(slipping.max()/vel) if vel>1e-10 else None,
         absolute_slide_native_s=dict(zip(turn.LEGS,slipping.tolist())),
         cross_track_native=float(delta@np.array([-h[1],h[0]])),cycles=dict(zip(turn.LEGS,map(int,count.counts))),
         heading_anchor_deg=float(np.rad2deg(yaw[yaw_index])),bounds=[a,b],reference_heading_ms=yaw_index)
    out['passed']=bool(out['forward_native_s']>=.05 and out['minimum_com']>=1.2 and out['nonfoot_peak']==0.
                         and out['foot_W']>=.9 and out['max_normalized_slip'] is not None and out['max_normalized_slip']<=.5)
    return out


def input_paths(plan_path,direction,subject_path):
    bank=turn.PARENT/'PROTOTYPES.npz'
    paths=[HERE/'rollout.py',HERE/'controller.py',plan_path,turn.TEACHER,TURN_SOURCE/'turning.py',
           TURN_SOURCE/'native_contact.cpp',SO,TURN_SOURCE/'verify_native.py',
           TURN_SOURCE/'tarsus_clearance.py',turn.PARENT/'walking_bench.py',turn.PARENT/'prototype.py',
           turn.PARENT/'FIT_INITIAL/step_1200.pt',bank,TURN_SOURCE/'BUILD.json']
    if subject_path is not None:
        paths.append(subject_path)
    return paths


def run(output,direction,config,label='teacher',subject=None,*,start='nominal',subject_path=None):
    """Run one named initial condition and declared command profile."""
    torch.set_num_threads(1)
    output=Path(output).resolve()
    output.mkdir()
    config=Path(config).resolve()
    p=read(config)
    w=turn.WARM
    entry=w+p['entry_ms']
    arc=entry+p['arc_ms']
    straight=arc+p['taper_ms']
    coast=arc+p['coast_ms']
    finish=coast+p['stop_ms']
    assert (p['arc_ms']+p['coast_ms'])%800==0 and p['coast_ms']>=4800
    bench=stan.Bench(output)
    c=bench.body
    c.physics_order=stan.physics.rc.SENS_IDX
    snapshots=turn.read(turn.STARTS/'BANK.json')
    snapshot=turn.STARTS/f'{start}.npz'
    turn.load_initial(bench,snapshot,snapshots[start]['sha256'])
    actor=turn.straight_policy()
    ident=stan.identity.capture(c,dict(scope='continuous arc exit into straight with frozen stand benchmark',
          direction=direction,label=label,requested_command_units=UNITS,plan=p,start=start,
          initial_sha256=snapshots[start]['sha256']),extra_paths=[snapshot,*input_paths(config,direction,subject_path)])
    stan.put(output/'RUN_IDENTITY.json',ident)
    record=Record(c,finish)
    c.recorder=record
    measure=c.measure()
    trace=[turn.sample(c,measure)]
    decisions=[]
    teacher=None
    unsafe=0
    began=time.monotonic()
    for tick in range(finish):
        if tick%5==0:
            obs=turn.features(c,measure)
            if tick<entry:
                command=np.array([BASE_PACE if tick>=w else 0.,0.])
                action=actor(obs,int(tick>=w))
            else:
                if teacher is None:
                    teacher=Controller(c,direction,p)
                t=(tick-entry)*DT
                if subject is None:
                    action,command=teacher.action(c,t)
                else:
                    teacher.reference.handle_plant_read(c,t)
                    command=teacher.reference.command(t)
                    action=subject(obs,command)
            if action.shape!=(42,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.:
                raise ValueError('Actuator path requires bounded finite 42-rates')
            decisions.append((tick,obs.copy(),command.copy(),action.copy()))
        measure=c.tick(action)
        if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
            raise RuntimeError('Unexpected applied force')
        trace.append(turn.sample(c,measure))
        unsafe=unsafe+1 if measure['com']<.80 or measure['nonfoot']>.5*W else 0
        if unsafe>=100:
            break
    ms={k:np.asarray([value[k] for value in trace]) for k in trace[0]}
    d={k:np.asarray(v) for k,v in zip(('ms','x','command','action'),zip(*decisions,strict=True),strict=True)}
    assert record.count==80*(len(ms['time'])-1)
    np.testing.assert_array_equal(record.endpoints[:len(trace)],ms['qpos'])
    assert not record.poses[:record.count,7].any()
    np.savez_compressed(output/'TRACE.npz',**ms)
    np.savez_compressed(output/'DECISIONS.npz',**d)
    np.savez_compressed(output/'NATIVE.npz',forces=record.forces[:record.count],pose=record.poses[:record.count])
    last=len(trace)-1
    result=dict(direction=direction,mode='arc_'+direction,start=start,bounds=[entry,arc,coast,finish],
                straight_begin_ms=straight,kind=label,command_units=UNITS,actual_ms=last,
                first_yaw_deg=float(np.rad2deg(ms['yaw'][0])),
                phase={'stand_start_ms':w,'arc_begin_ms':entry,'arc_end_ms':arc,
                       'taper_complete_ms':straight,'stop_start_ms':coast},
                trace_sha256=stan.sha(output/'TRACE.npz'),decisions_sha256=stan.sha(output/'DECISIONS.npz'),
                native_sha256=stan.sha(output/'NATIVE.npz'),native_integrations=record.count,
                native_force_cross_checks=record.cross_checks,elapsed=time.monotonic()-began,full_D1_D8=False)
    if last==finish:
        original=turn.score(ms,'arc_'+direction,(entry,arc,coast,finish))
        exits=projected_metrics(record,ms,arc,arc+4800,yaw_index=arc)
        continued=projected_metrics(record,ms,straight,coast,yaw_index=straight)
        full_exit=projected_metrics(record,ms,arc,coast,yaw_index=straight)
        legacy_actual=read(TURN_SOURCE/'FINAL_ACCEPTANCE.json')
        after=np.unwrap(ms['yaw'])
        terminal_yaw=float(np.rad2deg(after[-1]-after[-201]))
        result.update(original_arc=original,legacy_exit_window=exits,straight_window=continued,full_exit_window=full_exit,
                      original_direct_switch_reference=legacy_actual['additional_low_slip_exit']['straight_slip_ratio']['NATIVE_arc_'+direction])
        gates=dict(arc_pass=bool(original['low_slip_pass'] and original['function_pass']),
                   exit_pass=bool(exits['passed'] and abs(exits['net_yaw_deg'])<=10.),
                   full_exit=bool(full_exit['passed']),
                   straight_pass=bool(continued['passed'] and abs(continued['net_yaw_deg'])<=4.),
                   all_legs_resume=bool(all(full_exit['cycles'].values())),
                   initial_and_terminal_stand=bool(original['initial_stand']['passed'] and original['tail']['passed']),
                   terminal_residual_yaw_abs_le_0_5_deg=bool(abs(terminal_yaw)<=.5))
        result.update(checks=gates,terminal_residual_yaw_deg=terminal_yaw,passed=all(gates.values()),
                      heading_feedback_delta_deg=np.rad2deg(teacher.reference.heading_error or 0.).item())
    else:
        result.update(passed=False,reason='stopped before whole native episode')
    stan.identity.verify_runtime(ident,c)
    stan.put(output/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_arc',)},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('direction',choices=['left','right'])
    p.add_argument('output',type=Path)
    p.add_argument('config',type=Path)
    p.add_argument('--start',default='nominal')
    a=p.parse_args()
    run(a.output,a.direction,a.config,start=a.start)
