"""Collect physical behavior demonstrations or evaluate the shared trained brain."""
import argparse
from pathlib import Path

import numpy as np
import torch

from collect_touch import classify, physical_onsets
from separation import SeparatedLink, SCALE
from shared_brain import SharedBrain
from touch_runtime import (BINDING as OLD_BINDING, GRAPH as OLD_GRAPH, HERE as TOUCH_DIR, START,
                           BODY, W, TOUCH_W, build, Eyes, Fixture, Recording, turn,
                           stand_window, load_body_policy, sha)
from walk_pilot import VISION_WEIGHT
from bind import BINDING, GRAPH

VISUAL_WEIGHTS = TOUCH_DIR.parent/"20261116T170000Z/WALK_HEAD.pt"
TEACH_CASES = (
    (0, "intact", 680, 3870, 1765, -1.5, 3.52, .0038),
    (1, "right", 810, 3960, 2715, 1.5, 3.52, .0042),
    (2, "intact", 920, 4230, 2835, -.5, 3.46, .0045),
    (3, "right", 1040, 4160, 1990, 2.5, 3.55, .0036),
    (4, "sham", 750, 3780, 2185, -1.0, 3.50, .0038),
    (5, "sham", 955, 4130, 2645, 1.0, 3.58, .0038),
    (6, "cut", 880, 4020, 1910, 0.0, 3.50, .0040),
    (7, "restore", 695, 4010, 2620, -.8, 3.53, .0042),
    (8, "empty", 780, 3910, 2380, 0.0, 3.52, .0038),
)
TEST = (None,None,840,4130,2295,-1.2,3.50,.0041)
LEGS = tuple(BODY.LEG_ORDER)
FRONT = [LEGS.index(leg) for leg in ("LF","RF")]


def vision_init(brain):
    brain.vision.load_state_dict(torch.load(VISION_WEIGHT,map_location="cpu",weights_only=True))
    brain.visual_head.load_state_dict(torch.load(VISUAL_WEIGHTS,map_location="cpu",weights_only=True))


def qualification(trace, rows, native, mode, plan):
    begins,ends,contact = plan[2:5]
    side = int(mode=="right")
    time = physical_onsets(native)
    expected = mode in ("intact","right","restore")
    if mode in ("empty","dn_cut","sham"):
        truthful = all(v is None for v in time)
    else:
        truthful = time[side] is not None and time[1-side] is None
    moves = np.flatnonzero(rows[:,4])
    first = int(rows[moves[0],0]) if len(moves) else None
    stop = int(rows[moves[-1],0]+5) if len(moves) else None
    stop_delay = stop-time[side] if stop is not None and time[side] is not None else None
    # In restored runs the clock is allowed the actual post-restore delivery latency.
    if expected:
        policy = first is not None and first < contact-100 and stop_delay is not None and 0.<=stop_delay<=115.
    elif mode in ("empty","dn_cut"):
        policy = first is None and bool(rows[:,3].any())==(mode=="dn_cut")
    else:
        policy = first is not None and first < contact-100 and abs(stop-ends)<=105.
    if mode=="restore":
        policy = policy and stop >= contact+8
    settled_at = next((t-stop for t in range(stop+200,len(trace["time"]),5)
                       if stand_window(trace,t)["passed"]),None) if stop is not None else None
    stable = stand_window(trace,len(trace["time"])-1)
    checks = dict(truthful_fixture=truthful and native[:,4].max()<W*TOUCH_W,
                  behavior=policy,stop_settles=settled_at is None if mode in ("empty","dn_cut")
                       else settled_at is not None and settled_at<=1200,
                  no_nonfoot_ground=not np.any(trace["nonfoot"]), min_com=float(trace["com"][:,2].min())>=1.2,
                  stand=stable["passed"])
    return dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},first=first,stop=stop,
                side_onsets_ms=time,stop_delay_ms=stop_delay,settle_ms=settled_at,stand=stable,
                native_steps=len(native),end_range_ms=ends,visual_start_ms=begins)


def run(out,group,mode,checkpoint):
    torch.set_num_threads(1)
    if group is not None:
        plan=TEACH_CASES[group]
        mode=plan[1]
    else:
        plan=TEST
    _,_,begin,end,contact,angle,distance,depth=plan
    body=build(out) if out.is_dir() else build_after_mkdir(out)
    motor=load_body_policy()
    brain=SharedBrain().eval()
    if checkpoint:
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        brain.load_state_dict(state["brain"])
    else:
        vision_init(brain)
    link=SeparatedLink()
    if mode in ("cut","restore"):
        link.cut_an(True)
    if mode == "dn_cut":
        link.switch(True)
    stop_readout = turn.read(TOUCH_DIR/"DATA/TOUCH_READOUT.json")
    assert (stop_readout["binding_sha256"],stop_readout["graph_sha256"])==(sha(OLD_BINDING),sha(OLD_GRAPH))
    side=int(mode=="right" or (mode=="sham" and group==5))
    no_probe=mode in ("empty","dn_cut")
    fixture=Fixture(body,side,contact,depth if mode!="sham" else -.010,55)
    finish=(end+1280)//5*5
    recorder=Recording(body,finish)
    body.recorder=recorder
    sensors = [body.plant.joint(f"joint_{leg}Tibia_pitch") for leg in LEGS]
    angles=[int(joint.qposadr[0]) for joint in sensors]
    velocities=[int(joint.dofadr[0]) for joint in sensors]
    capture=turn.stand.identity.capture(body,dict(stage="joint brain teacher" if not checkpoint else "joint brain physical",
                                case=mode),extra_paths=[Path(__file__),BINDING,GRAPH,START,VISION_WEIGHT,
                                VISUAL_WEIGHTS,Path(__file__).with_name("shared_brain.py")]+([checkpoint] if checkpoint else []))
    turn.stand.put(out/"IDENTITY.json",capture)
    eyes=Eyes(body)
    background=(eyes.pose()[0].copy(),.133,.27)
    target=None
    measured=body.measure()
    copy=0
    visual=-4.
    memory=None
    paused=False
    action=np.zeros(42)
    trace=[turn.sample(body,measured)]
    features,conditions,physical,commands,probe,views,stamps = [],[],[],[],[],[],[]
    with torch.inference_mode():
        try:
            for ms in range(finish):
                if ms==begin:
                    target=eyes.place(angle,distance)
                if ms==end:
                    target=None
                if mode=="restore" and ms==contact+8:
                    link.cut_an(False)
                if ms%20==0:
                    sample=eyes.capture(ms,target if mode!="empty" else None,.161,(1.,.045,.035,1.),background)
                    if sample is not None:
                        _, image=sample
                        visual=float(brain.visual_feature(torch.as_tensor(image[None])).item())
                        views.append(image.copy())
                        stamps.append(ms)
                if ms%5==0:
                    signals,available=link.observe_packet()
                    x=np.r_[visual,signals,float(available),copy].astype(np.float32)
                    decoded=classify(signals[:2] if available else None,stop_readout)
                    if checkpoint:
                        logits,_,memory=brain.advance(torch.from_numpy(x)[None,None],memory)
                        copy=int(logits.item()>0.)
                    else:
                        paused=paused or decoded>=0
                        copy=int(visual>0. and not paused)
                    link.send(copy)
                    effective=link.action()
                    action=motor(turn.features(body,measured),effective,0)
                    load=measured["loads"]/W
                    conditions.append((ms,visual>0,decoded,copy,effective,-1 if not available else link.an_time))
                    features.append(x)
                    commands.append(copy)
                    physical.append(np.r_[body.d.qpos[angles],body.d.qvel[velocities],load,
                          body.com_velocity()@body.d.xmat[body.th].reshape(3,3)[:,0],body.d.xmat[body.th,8]])
                if not no_probe:
                    fixture.move(ms)
                probe.append(body.d.mocap_pos[0].copy())
                sensed=np.r_[recorder.activity(),np.clip(measured["loads"][FRONT]/(SCALE*W),0.,1.)]
                if not link.an_cut:
                    link.sense(sensed.tolist())
                measured=body.tick(action)
                if body.d.qfrc_applied.any() or body.d.xfrc_applied.any():
                    raise RuntimeError("Unexpected external wrench")
                trace.append(turn.sample(body,measured))
                link.advance()
        finally:
            eyes.close()
    assert recorder.count==finish*80
    state={key:np.asarray([r[key] for r in trace]) for key in trace[0]}
    decisions=np.asarray(conditions)
    result=qualification(state,decisions,recorder.raw,mode,plan)
    native=(recorder.raw[:,2:4].reshape(-1,400,2)>TOUCH_W*W).sum(1)>=8
    np.savez_compressed(out/"RUN.npz",features=features,intended=commands,physical=physical,raw=recorder.raw,
                       contact=native,commands=decisions,probe=probe,eyes=views,eye_stamps=stamps,**state)
    turn.stand.put(out/"TRANSPORT.json",link.transport.journal())
    turn.stand.identity.verify_runtime(capture,body)
    result.update(mode=mode,group=group,contract="teacher" if checkpoint is None else "learned shared brain",
                  checkpoint_sha256=None if checkpoint is None else sha(checkpoint),
                  hashes={name:sha(out/name) for name in ("RUN.npz","TRANSPORT.json")},binding=link.transport.verify_final())
    turn.stand.put(out/"RESULT.json",result)
    print(result,flush=True)
    return result["passed"]


def build_after_mkdir(out):
    out.mkdir()
    return build(out)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out",type=Path)
    modes=cli.add_mutually_exclusive_group(required=True)
    modes.add_argument("--teacher",type=int,choices=range(len(TEACH_CASES)))
    modes.add_argument("--mode",choices=("intact","right","sham","cut","restore","empty","dn_cut"))
    cli.add_argument("--checkpoint",type=Path)
    args=cli.parse_args()
    if (args.teacher is None) != bool(args.checkpoint):
        cli.error("Teacher and learned runtime have different entry conditions")
    raise SystemExit(0 if run(args.out.resolve(),args.teacher,args.mode,args.checkpoint) else 1)
