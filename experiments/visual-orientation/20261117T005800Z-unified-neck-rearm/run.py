"""Continuous frozen-body replay using one visual/afferent navigation brain."""
import argparse
import json
import inspect
from pathlib import Path
import time

import numpy as np
import torch

from unified import NavigationBrain, Neck, Motion, CHECKPOINT, BINDING, HERE, sha, turn, stand_window
from run_brain import FRONT, SCALE, LEGS
from collect_touch import physical_onsets
from walking_bench import cycles as count_cycles
from touch_runtime import build, BODY, Eyes, Fixture, Recording, START, W, TOUCH_W

SCENES=HERE/"SCENES.json"
WIDTH=.15
COLOR=(1.,.045,.035,1.)


def assess(case, scene, trace, decisions, recorder, goal):
    on, off, end=(scene[k] for k in ("on","off","end"))
    yaw=np.degrees(np.unwrap(trace["yaw"]))
    travel=trace["com"][:,:2]
    target=goal[:2]-trace["head"][:,:2]
    bearing=(np.degrees(np.arctan2(target[:,1],target[:,0]))-yaw+180.)%360.-180.
    distance=np.linalg.norm(target,axis=1)
    active=(decisions[:,3]>0.) | (decisions[:,4]!=0)
    walk=decisions[:,3]>0.
    requested=(decisions[:,1]>0.) | (decisions[:,2]!=0)
    stopped=int(decisions[active,0][-1]+5) if active.any() else None
    start=int(decisions[active,0][0]) if active.any() else None
    final=stand_window(trace,end)
    initial=stand_window(trace,on)
    settled=next((i-stopped for i in range(stopped+200,min(stopped+1300,end)+1,5)
                  if stand_window(trace,i)["passed"]), None) if stopped else None
    checks=dict(initial_stand=initial["passed"],final_stand=final["passed"],
                nonfoot_zero=not trace["nonfoot"].any(),com_safe=float(trace["com"][:,2].min())>=1.2,
                actual_clock=recorder.count==end*80)
    heading=np.radians(yaw[on])
    axis=np.array([np.cos(heading),np.sin(heading)])
    def pace(a,b):
        return float((travel[b]-travel[a])@axis/((b-a)*.001))

    def longest_speed(select):
        ticks=decisions[select,0].astype(int)
        if not len(ticks):
            return None
        parts=np.split(ticks,np.flatnonzero(np.diff(ticks)>5)+1)
        part=max(parts,key=len)
        a,b=int(part[0]),int(part[-1])+5
        if b-a<900:
            return None
        return dict(begin=a,end=b,v=pace(a+250,b))

    speed_fast=longest_speed(decisions[:,3]>.080)
    speed_slow=longest_speed((decisions[:,3]>.069)&(decisions[:,3]<.076))
    onsets=physical_onsets(recorder.raw) if "probe" in scene else [None,None]
    delivered_yaw=np.flatnonzero(decisions[:,4])
    finish_turn=int(decisions[delivered_yaw[-1],0]+5) if len(delivered_yaw) else None
    yaw_at_turn=float(yaw[finish_turn]-yaw[on]) if finish_turn else None
    cycles=None
    values=dict(speed_fast=speed_fast,speed_slow=speed_slow)
    if case=="straight":
        windows=[pace(2000,10000),pace(12000,20000),pace(22000,30000)]
        cycles,_=count_cycles(trace,2000,30000)
        moving_fraction=float(decisions[(decisions[:,0]>=2000)&(decisions[:,0]<30000),3].astype(bool).mean())
        checks.update(sustained=all(v>=.055 for v in windows) and moving_fraction>=.98,
                      six_cycles=all(cycles.values()), no_turn=not delivered_yaw.size,
                      source_stop=stopped is not None and abs(stopped-off)<=55,
                      physically_settles=settled is not None)
        values.update(speed_windows=windows,moving_fraction=moving_fraction,cycles=cycles)
    if case in ("left","right"):
        sign=1 if case=="left" else -1
        checks.update(turn=yaw_at_turn is not None and sign*yaw_at_turn>=10.,
                      alignment=abs(bearing[-1])<=17. and abs(bearing[-1])<abs(scene["angle"])-4.,
                      forward_after_turn=bool(finish_turn is not None and walk[decisions[:,0]>finish_turn].sum()>=480),
                      closes_range=float(distance[on]-distance[-1])>=.35,
                      physically_settles=settled is not None)
    if case=="pace":
        checks.update(two_speeds=speed_fast is not None and speed_slow is not None
                      and speed_fast["v"]-speed_slow["v"]>=.004,
                      physically_settles=settled is not None, closes_range=float(distance[on]-distance[-1])>=.45)
    if case=="dn_cut":
        checks.update(requested=bool((decisions[:,2]!=0).any()),
                      no_delivered_movement=not active.any(),
                      position_stays=float(np.linalg.norm(travel[-1]-travel[0]))<.015)
    if "probe" in scene:
        on_contact=onsets[0]
        checks.update(physical_single_side=on_contact is not None and onsets[1] is None
                      and float(recorder.raw[:,4].max())<W*TOUCH_W,
                      was_walking=start is not None and on_contact is not None and start<on_contact-100,
                      physically_settles=settled is not None)
        if case=="an_cut":
            checks["continues_until_cue_removed"]=stopped is not None and abs(stopped-off)<=55
            checks["an_unknown"]=not decisions[:,12].any()
        else:
            limit=scene.get("an_restore",on_contact)
            checks["stops_after_actual_feedback"]=limit is not None and stopped is not None and limit<=stopped<=limit+95
    checks={k:bool(v) for k,v in checks.items()}
    return dict(passed=all(checks.values()),checks=checks, first_motion_ms=start, stop_ms=stopped,
                settle_ms=settled, contact_ms=onsets, native_steps=recorder.count,
                initial_range=float(distance[on]),final_range=float(distance[-1]),
                final_bearing_deg=float(bearing[-1]),signed_yaw_deg=float(yaw[-1]-yaw[on]),
                signed_yaw_at_withdrawal_deg=yaw_at_turn, terminal=final, initial=initial,
                lowest_com=float(trace["com"][:,2].min()), **values)


def run(case, out):
    torch.set_num_threads(1)
    scene=json.loads(SCENES.read_text())[case]
    out.mkdir()
    body=build(out)
    neck=Neck()
    if scene.get("dn_cut"):
        neck.switch(True)
    if scene.get("an_cut"):
        neck.cut_an(True)
    checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    brain=NavigationBrain().eval()
    brain.load_state_dict(checkpoint["brain"])
    assert all(sha(Path(p))==d for p,d in checkpoint["parents"].items())
    motor=Motion(body)
    eyes=Eyes(body)
    goal=eyes.place(scene["angle"],scene["distance"])
    background=(eyes.pose()[0].copy(),scene["shift"],.263)
    end=scene["end"]
    fixture=Fixture(body,*scene["probe"],55) if "probe" in scene else None
    rec=Recording(body,end)
    body.recorder=rec
    measured=body.measure()
    angles=[int(body.plant.joint(f"joint_{leg}Tibia_pitch").qposadr[0]) for leg in LEGS]
    velocities=[int(body.plant.joint(f"joint_{leg}Tibia_pitch").dofadr[0]) for leg in LEGS]
    pins=[Path(__file__),HERE/"unified.py",BINDING,CHECKPOINT,START,SCENES,
          Path(Motion.action.__code__.co_filename),Path(type(motor.pace).__call__.__code__.co_filename),
          Path(inspect.getfile(motor.stock if inspect.isfunction(motor.stock) else type(motor.stock)))]
    identity=turn.stand.identity.capture(body,dict(case=case,scene=scene,scope="unified BANC-source neck"),extra_paths=pins)
    turn.stand.put(out/"RUN_IDENTITY.json",identity)
    observations, state, poses, video, stamps, decisions, predicted = [],[],[],[],[],[],[]
    sensed=np.zeros(4)
    action=np.zeros(42)
    t0=time.monotonic()
    try:
        for ms in range(end+1):
            sample=turn.sample(body,measured)
            state.append({k:v for k,v in sample.items() if k not in ("qpos","qvel","ctrl","target")}|
                         {"head":eyes.pose()[0].copy()})
            if ms%20==0:
                poses.append(body.d.qpos.copy())
            if ms==end:
                break
            if scene.get("an_restore")==ms:
                neck.cut_an(False)
            if ms%20==0:
                frame=eyes.capture(ms,goal if scene["on"]<=ms<scene["off"] else None,WIDTH,COLOR,background)
                if frame is not None:
                    captured,rgb=frame
                    brain.new_frame(rgb)
                    video.append(rgb)
                    stamps.append(captured)
            if ms%5==0:
                arrived,valid=neck.observe()
                command,gate,pred=brain.decide(arrived,valid)
                neck.send(*command)
                delivered=neck.current()
                action=motor.action(turn.features(body,measured),*delivered)
                decisions.append((ms,*command,*delivered,*brain.proposal,*arrived,gate,float(valid),
                                  *brain.cue, -1 if neck.an_step is None else neck.an_step,
                                  -1 if neck.time is None else neck.time))
                observed_speed=float(body.com_velocity()@body.d.xmat[body.th].reshape(3,3)[:,0])
                observations.append(np.r_[body.d.qpos[angles],body.d.qvel[velocities],
                                          measured["loads"]/W,observed_speed,body.d.xmat[body.th,8]])
                predicted.append(np.concatenate([p.numpy().reshape(-1) for p in pred]))
            sensed[:]=np.r_[rec.activity(),np.clip(measured["loads"][FRONT]/(SCALE*W),0.,1.)]
            neck.sense(sensed)
            if fixture:
                fixture.move(ms)
            measured=body.tick(action)
            neck.advance()
            if body.d.qfrc_applied.any() or body.d.xfrc_applied.any():
                raise RuntimeError("Unexpected force")
            if ms%5000==4999:
                print(case,ms+1,round(time.monotonic()-t0,2),flush=True)
    finally:
        eyes.close()
    trace={k:np.asarray([row[k] for row in state]) for k in state[0]}
    dec=np.asarray(decisions)
    result=assess(case,scene,trace,dec,rec,goal)
    raw_touch = rec.raw[:,2:5].reshape(end,80,3).max(1)
    np.savez_compressed(out/"TRACE.npz",**trace,decisions=dec,eye_stamps=stamps,eyes=video,
                        poses=poses,physical=observations,forecasts=predicted,touch_max=raw_touch,goal=goal)
    turn.stand.put(out/"TRANSPORT.json",neck.link.journal())
    turn.stand.identity.verify_runtime(identity,body)
    result.update(case=case,scene=scene,wall_s=time.monotonic()-t0,
                  hashes={k:sha(out/k) for k in ("TRACE.npz","TRANSPORT.json","RUN_IDENTITY.json")},
                  source=neck.link.verify_final())
    turn.stand.put(out/"RESULT.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.case,args.out.resolve())
