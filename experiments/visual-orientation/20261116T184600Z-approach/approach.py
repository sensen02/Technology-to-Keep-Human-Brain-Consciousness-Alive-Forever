"""Navigate toward a landmark using rendered eye frames and four named DN axons."""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from approach_runtime import (HERE, TURN, SPEED, BINDING, ApproachLink, Motion, Eyes,
                               prepare, Record, stand_window, turn, sha)
from approach_vision import network, features, CHECKPOINT, WIDTH, COLOR

START_MS=500
NEAR=2.41
STOP=2.10
FAST=.08318
SLOW=.071713


class Policy:
    def __init__(self):
        self.model=network().eval()
        self.model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True))
        self.history=deque(maxlen=5)
        self.phase="align"
        self.wait=0
        self.near=False
        self.direction=0

    @torch.inference_mode()
    def act(self,image):
        angle,distance,confidence=self.model(features(torch.as_tensor(image[None])))[0].tolist()
        angle, distance = angle*40., distance+2.5
        if confidence<0.:
            self.history.clear()
            return (0.,0), (angle,distance,confidence)
        self.history.append((angle,distance))
        angle,distance=np.median(self.history,axis=0)
        if self.phase=="done":
            return (0.,0),(angle,distance,confidence)
        if self.phase=="align":
            if self.direction==0 and abs(angle)>7.5:
                self.direction=int(np.sign(angle))
            if self.direction and angle*self.direction>5.:
                return (0.,self.direction),(angle,distance,confidence)
            self.phase="settle"
            self.wait=40
            self.direction=0
        if self.phase=="settle":
            if self.wait:
                self.wait-=1
                return (0.,0),(angle,distance,confidence)
            self.phase="walk"
        if distance<STOP:
            self.phase="done"
            return (0.,0),(angle,distance,confidence)
        if abs(angle)>25. and distance>NEAR:
            self.phase="align"
            return (0.,0),(angle,distance,confidence)
        self.near = self.near or distance < NEAR
        return (SLOW if self.near else FAST,0),(angle,distance,confidence)


def score(trace,decisions,goal,actual,angle,mode,rec):
    xyz=trace["head"]
    attended=goal[:2]-xyz[:,:2]
    range_=np.linalg.norm(attended,axis=1)
    initial=float(range_[START_MS])
    ending=float(range_[-1])
    true_bearing=np.rad2deg(np.unwrap(np.arctan2(attended[:,1],attended[:,0])-trace["yaw"]))
    true_bearing=(true_bearing+180.)%360.-180.
    y=np.rad2deg(np.unwrap(trace["yaw"]))
    forward=decisions[:,3]
    yaw_cmd=decisions[:,4]
    fast=decisions[forward>.080,0].astype(int)
    slow=decisions[(forward>.069)&(forward<.076),0].astype(int)
    displacement=np.linalg.norm(trace["com"][-1,:2]-trace["com"][0,:2])
    end_stand=stand_window(trace,actual)
    max_nonfoot=float(rec.forces[:rec.count,12].max())
    min_com=float(trace["com"][:,2].min())
    def speed(stamps):
        if len(stamps)*5<900:
            return None
        start,end=int(stamps[0]),int(stamps[-1])+5
        if (end-start)/5!=len(stamps):
            return None
        a=max(start+250,start)
        delta=trace["com"][end,:2]-trace["com"][a,:2]
        return float(delta @ np.array([np.cos(trace["yaw"][a]),np.sin(trace["yaw"][a])])/((end-a)*.001))
    fast_v,slow_v=speed(fast),speed(slow)
    checks=dict(stand=end_stand["passed"],no_floor_nonfoot=max_nonfoot==0.,min_com=min_com>=1.2)
    active=bool(forward.any() or yaw_cmd.any())
    if mode in ("cut","blank"):
        requested=decisions[:,1:3].any()
        checks.update(no_body_motion=not active and float(displacement)<.03,
                      brain_requested_move=bool(requested)==(mode=="cut"))
    else:
        turn_sign=float((y[-1]-y[START_MS])*np.sign(angle))
        checks.update(correct_turn=turn_sign>=abs(angle)*.45,look_at=abs(true_bearing[-1])<15.,
                      approach=initial-ending>=.45,stop_by_visual_range=bool(ending<2.34 and 1.75<ending),
                      measured_speeds=fast_v is not None and slow_v is not None and fast_v-slow_v>=.004)
    return dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                initial_range=initial,final_range=ending,head_bearing_tail=float(true_bearing[-1]),
                actual_body_turn_deg=float(y[-1]-y[START_MS]),body_displacement=float(displacement),
                speed_fast=fast_v,speed_slow=slow_v,fast_ms=len(fast)*5,slow_ms=len(slow)*5,
                standing=end_stand,nonfoot_native=max_nonfoot,actual_ms=actual,native_steps=rec.count)


def trial(out,angle,distance,mode,shift,duration):
    torch.set_num_threads(1)
    body,_ = prepare(out,"nominal")
    eyes=Eyes(body)
    brain=Policy()
    motion=Motion(body)
    neck=ApproachLink()
    if mode=="cut":
        neck.switch(True)
    background=(eyes.pose()[0].copy(),shift,.255)
    goal=eyes.place(angle,distance)
    m=body.measure()
    def snap():
        return {**turn.sample(body,m),"head":eyes.pose()[0].copy()}
    rows=[snap()]
    decision, stamps, images=[],[],[]
    intended=(0.,0)
    estimate=(0.,0.,0.)
    action=np.zeros(42)
    record=Record(body,duration)
    body.recorder=record
    last_activity=0
    try:
        for ms in range(duration):
            if ms%20==0:
                frame=eyes.capture(ms,goal if mode!="blank" and ms>=START_MS else None,WIDTH,COLOR,background)
                if frame:
                    timed,image=frame
                    intended,estimate=brain.act(image)
                    stamps.append(timed)
                    images.append(image.copy())
                neck.send(float(intended[0]),int(intended[1]))
            if ms%5==0:
                v,w=neck.current()
                action=motion.action(turn.features(body,m),v,w)
                if w or v>1.e-7:
                    last_activity=ms
                decision.append((ms,*intended,v,w,*estimate))
            m=body.tick(action)
            if body.d.qfrc_applied.any() or body.d.xfrc_applied.any():
                raise RuntimeError("Unexpected external force")
            rows.append(snap())
            neck.advance()
            if brain.phase=="done" and ms-last_activity>=1600:
                break
    finally:
        eyes.close()
    trace={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    actual=len(rows)-1
    np.testing.assert_array_equal(record.endpoints[:actual+1],trace["qpos"])
    assert record.count==80*actual and not record.poses[:record.count,7].any()
    decisions=np.asarray(decision)
    result=score(trace,decisions,goal,actual,angle,mode,record)
    np.savez_compressed(out/"TRACE.npz",**trace)
    np.savez_compressed(out/"OBS.npz",decisions=decisions,eyes=images,stamps=stamps,goal=goal)
    np.savez_compressed(out/"NATIVE.npz",forces=record.forces[:record.count],pose=record.poses[:record.count])
    turn.stand.put(out/"TRANSPORT.json",neck.link.journal())
    result.update(angle=angle,distance=distance,shift=shift,mode=mode,source=neck.link.verify_final(),
                  files={name:sha(out/name) for name in ("TRACE.npz","OBS.npz","NATIVE.npz","TRANSPORT.json")},
                  inputs={str(p):sha(p) for p in (BINDING,CHECKPOINT,Path(__file__),HERE/"approach_runtime.py")})
    turn.stand.put(out/"RESULT.json",result)
    print(result,flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--angle",type=float,default=20.)
    p.add_argument("--distance",type=float,default=2.68)
    p.add_argument("--shift",type=float,default=.12)
    p.add_argument("--mode",choices=("intact","cut","blank"),default="intact")
    p.add_argument("--duration",type=int,default=18200)
    a=p.parse_args()
    result=trial(a.out.resolve(),a.angle,a.distance,a.mode,a.shift,a.duration)
    raise SystemExit(0 if result["passed"] else 1)
