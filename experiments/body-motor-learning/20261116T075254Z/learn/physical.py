"""Apply one memoryless body policy to a frozen physical command course."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from course import Course

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "20261116T000536Z"
TURN = HERE.parents[1] / "20261116T004235Z"
sys.path[:0] = [str(BASE), str(TURN)]
import walking_bench as bench
from prototype import BANK, CHECKPOINT, policy as stock_policy
from verify_native import Record, SO

W = bench.W
DT = bench.stand.physics.NATIVE_DT
NATIVE_PER_MS = round(.001/DT)


def write(path, value):
    bench.stand.put(path, value)


def sha(path):
    return bench.stand.sha(Path(path))


def pin_base():
    return [BASE / "prototype.py", BANK, CHECKPOINT, BASE / "PROTOTYPES.json",
            BASE / "walking_bench.py", TURN / "turning.py", TURN / "verify_native.py", SO]


def stock():
    frozen = stock_policy()
    return lambda x, command: frozen(x, int(command[0] > 0.))


def load_provider(path, checkpoint):
    spec = importlib.util.spec_from_file_location("pace_policy_provider", path.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_policy(checkpoint)


def verify_record(raw, trace, actual_ms):
    if raw.count != actual_ms*NATIVE_PER_MS:
        raise ValueError("Native steps and saved milliseconds disagree")
    np.testing.assert_array_equal(raw.endpoints[:actual_ms+1], trace["qpos"])
    np.testing.assert_allclose(raw.poses[:raw.count, 6],
                               trace["time"][0]+np.arange(raw.count)*DT,
                               atol=5e-8, rtol=0.)
    if raw.poses[:raw.count, 7].any():
        raise ValueError("Free body has unexpected applied forces")


def projected_distance(raw, trace, a, b):
    start, end = a*NATIVE_PER_MS, b*NATIVE_PER_MS
    states = raw.poses[start:end]
    xy = np.r_[states[:, :2], trace["com"][None,b,:2]]
    forward = states[:,3:5]
    forward = forward / np.linalg.norm(forward,axis=1,keepdims=True)
    return float(np.sum(np.diff(xy,axis=0)*forward))


def window(raw, trace, a, b, *, fixed_heading=False):
    dt = (b-a)*.001
    native = raw.forces[a*NATIVE_PER_MS:b*NATIVE_PER_MS]
    heading = np.array([np.cos(trace["yaw"][a]), np.sin(trace["yaw"][a])])
    delta = trace["com"][b,:2]-trace["com"][a,:2]
    forward = (float(delta@heading) if fixed_heading else projected_distance(raw,trace,a,b))/dt
    normalization = float(np.linalg.norm(delta)/dt if fixed_heading else forward)
    normal = native[:,:6]
    individual = np.divide(native[:,6:12].sum(0), normal.sum(0),
                           out=np.zeros(6), where=normal.sum(0)>0)
    if normalization>0. and (normal.sum(0)>0).all():
        slip = float(individual.max()/normalization)
    else:
        slip = None
    yaw = np.rad2deg(np.unwrap(trace["yaw"][a:b+1])[-1]-trace["yaw"][a])
    support = float(normal.sum(1).mean()/W)
    minimum_com = float(min(raw.poses[a*NATIVE_PER_MS:b*NATIVE_PER_MS,2].min(),
                            trace["com"][b,2]))
    nonfoot = float(native[:,12].max())
    return dict(a_ms=a,b_ms=b,steps=len(native),projection="start_body_heading" if fixed_heading
                else "instantaneous_body_heading",forward=forward,normalization_speed=normalization,
                yaw_deg=float(yaw),support_w=support,com_min=minimum_com,nonfoot_native=nonfoot,
                per_foot_absolute_slip=dict(zip(bench.LEGS,individual.tolist())), slip=slip,
                gates=dict(forward=forward>=.05,support=support>=.90,com=minimum_com>=1.20,
                           nonfoot=nonfoot==0.,slip=slip is not None and slip<=.50))


def standing(raw, trace, a, b):
    sampled = bench.standing(trace,a,b)
    if not sampled["passed"]:
        return sampled
    low=(b-200)*NATIVE_PER_MS
    contacts=raw.forces[low:b*NATIVE_PER_MS]
    pose=raw.poses[low:b*NATIVE_PER_MS]
    yaw_drift = float(np.rad2deg(np.unwrap(trace["yaw"][b-200:b+1])[-1]-trace["yaw"][b-200]))
    gates = dict(sampled=True,support=bool(contacts[:,:6].sum(1).mean()/W>=.90),
                 nonfoot=bool(contacts[:,12].max()<.01*W),
                 three=bool((np.count_nonzero(contacts[:,:6]>=.02*W,axis=1)>=3).mean()>=.95),
                 posture=bool(pose[:,5].min()>=np.cos(np.deg2rad(20.))),
                 com=bool(pose[:,2].min()>=1.2),yaw=abs(yaw_drift)<=.5)
    return dict(passed=all(gates.values()),gates=gates,sampled=sampled,yaw_drift_deg=yaw_drift)


def audit(course, trace, raw):
    actual=len(trace["time"])-1
    verify_record(raw,trace,actual)
    out = dict(actual_ms=actual,complete=actual==course.finish,native_integrations=raw.count,
               zero_extra_force=True,native_force_cross_checks=raw.cross_checks,full_D1_D8=False)
    if actual!=course.finish:
        return dict(**out,native_joint_pass=False)
    start,stop=course.segments[0].end,course.segments[-1].begin
    out["initial_stand"]=standing(raw,trace,0,start)
    out["terminal_stand"]=standing(raw,trace,stop,actual)
    counts,events=bench.cycles(trace,start,stop,arm_before=20)
    out["first_step"]=next((e for e in events if e["type"]=="complete"),None)
    out["cycles"]=counts
    gates = dict(initial_stand=out["initial_stand"]["passed"],
                 terminal_stand=out["terminal_stand"]["passed"],six_cycles=all(counts.values()))
    walking=window(raw,trace,start,stop)
    out["walking"]=walking
    out["segments"]={seg.name:window(raw,trace,seg.begin,seg.end,
                       fixed_heading=bool(course.turn and seg.name==course.turn["exit"]))
                     for seg in course.segments[1:-1]}
    gates["walking"] = all(walking["gates"].values())
    if course.speed_pair:
        pair=course.speed_pair
        speed = {k:out["segments"][pair[k]]["forward"] for k in ("low","high","slowed")}
        floor=float(pair["delta_min"])
        gate_low_high=speed["high"]-speed["low"]>=floor
        gate_high_slowed=speed["high"]-speed["slowed"]>=floor
        out["speed_response"]=dict(**speed,min_delta=floor,accelerates=gate_low_high,
                                     decelerates=gate_high_slowed)
        # Use the frozen slip bound on the whole gait and full exit windows.
        # Short speed estimates can touch only part of a single loaded cycle.
        gates["speed"] = bool(gate_low_high and gate_high_slowed and
                             all(all(flag for name,flag in out["segments"][pair[k]]["gates"].items()
                                     if name != "slip") for k in ("low","high","slowed")))
    if course.turn:
        turn=course.turn
        arc=out["segments"][turn["arc"]]
        exit_window=out["segments"][turn["exit"]]
        gates["arc"] = bool(arc["yaw_deg"]*turn["direction"]>=turn["minimum_yaw_deg"] and
                            all(arc["gates"].values()))
        gates["exit"] = bool(abs(exit_window["yaw_deg"])<=10. and all(exit_window["gates"].values()))
    out["gates"]=gates
    out["native_joint_pass"]=all(gates.values())
    return out


def run(output, start, profile, policy, provenance, control, *, recording=None):
    course=Course.load(profile)
    output.mkdir()
    rig=bench.stand.Bench(output)
    body=rig.body
    body.physics_order=bench.stand.physics.rc.SENS_IDX
    np.testing.assert_array_equal(body.physics_order,bench.SENSOR_BY_LEG)
    snapshot=bench.STARTS/f"{start}.npz"
    snapshot_sha=bench.read(bench.STARTS/"BANK.json")[start]["sha256"]
    bench.load_initial(rig,snapshot,snapshot_sha)
    pins=[HERE/"course.py",Path(__file__),profile,snapshot,bench.STARTS/"BANK.json",*pin_base(),*provenance]
    identity=bench.stand.identity.capture(body,dict(kind=control,start=start,snapshot_sha256=snapshot_sha,
                     horizon_ms=course.finish,command_units=course.default["units"]),extra_paths=pins)
    write(output/"RUN_IDENTITY.json",identity)
    if body.adhesion_on:
        raise ValueError("Continuous body course must use frozen friction-only physical plant")
    raw=Record(body,course.finish)
    body.recorder=raw
    trace=[]
    decisions=[]
    measured=body.measure()
    action=rig.zero.copy()
    start_time=time.monotonic()

    def sample(applied):
        trace.append(dict(time=float(body.d.time),qpos=body.d.qpos.copy(),qvel=body.d.qvel.copy(),
            com=body.d.subtree_com[body.th].copy(),loads=measured["loads"].copy(),
            nonfoot=float(measured["nonfoot"]),padz=measured["padz"].copy(),
            speed=measured["speed"].copy(),foot=measured["foot"].copy(),
            target=body.actuator.target_rad.copy(),ctrl=body.d.ctrl.copy(),applied=applied.copy(),
            upright=float(body.d.xmat[body.th,8]),
            yaw=float(np.arctan2(body.d.xmat[body.th,3],body.d.xmat[body.th,0]))))

    sample(action)
    bad=0
    for tick in range(course.finish):
        if tick%5==0:
            observation=bench.features(body,measured)
            command=course.command(tick)
            if recording is None:
                action=policy(observation,command)
            else:
                idx=tick//5
                np.testing.assert_array_equal(observation,recording["x"][idx])
                np.testing.assert_array_equal(command,recording["command"][idx])
                action=recording["action"][idx]
            action=bench.legacy.bounded_action(action)
            decisions.append((tick,observation.copy(),command.copy(),action.copy()))
        measured=body.tick(action)
        if np.any(body.d.xfrc_applied) or np.any(body.d.qfrc_applied):
            raise RuntimeError("Unexpected applied wrench")
        sample(action)
        bad=bad+1 if measured["com"]<.8 or measured["nonfoot"]>.5*W else 0
        if bad>=100:
            break
    trace={k:np.asarray([row[k] for row in trace]) for k in trace[0]}
    decisions={k:np.asarray(v) for k,v in zip(("ms","x","command","action"),zip(*decisions),strict=True)}
    np.savez_compressed(output/"TRACE.npz",**trace)
    np.savez_compressed(output/"DECISIONS.npz",**decisions)
    np.savez_compressed(output/"NATIVE.npz",forces=raw.forces[:raw.count],pose=raw.poses[:raw.count])
    result=audit(course,trace,raw)
    result["native_sha256"]=sha(output/"NATIVE.npz")
    result["native_library_sha256"]=sha(SO)
    bench.stand.identity.verify_runtime(identity,body)
    write(output/"NATIVE_AUDIT.json",result)
    summary=dict(control=control,start=start,initial_snapshot_sha256=snapshot_sha,actual_ms=result["actual_ms"],
                 profile=str(profile),profile_sha256=sha(profile),course=course.default,native_joint_pass=result["native_joint_pass"],
                 trace_sha256=sha(output/"TRACE.npz"),decision_sha256=sha(output/"DECISIONS.npz"),
                 native_audit_sha256=sha(output/"NATIVE_AUDIT.json"),elapsed_wall_s=time.monotonic()-start_time,
                 full_D1_D8=False)
    write(output/"RESULT.json",summary)
    print(json.dumps({"result":summary,"audit":result},indent=2),flush=True)


def check_repeat(reference, output):
    old=bench.read(reference/"RESULT.json")
    new=bench.read(output/"RESULT.json")
    assert old["profile_sha256"]==new["profile_sha256"]
    assert old["initial_snapshot_sha256"]==new["initial_snapshot_sha256"]
    hashes={}
    for file in ("TRACE.npz","DECISIONS.npz","NATIVE.npz"):
        first,second=sha(reference/file),sha(output/file)
        if first!=second:
            raise ValueError(f"Independent physical replay differs for {file}")
        hashes[file]=first
    write(output/"REPLAY_AUDIT.json",dict(reference=str(reference),repeat=str(output),
                                        bitwise_raw_artifacts=hashes,passed=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("profile",type=Path)
    parser.add_argument("--start",default="nominal")
    parser.add_argument("--provider",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--repeat",type=Path)
    parser.add_argument("--action-replay",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint and not args.provider:
        parser.error("--checkpoint needs --provider")
    if args.action_replay and not args.repeat:
        parser.error("--action-replay needs a source to replay")
    policy=stock() if not args.provider else load_provider(args.provider,args.checkpoint)
    evidence=[x.resolve() for x in (args.provider,args.checkpoint) if x]
    if args.repeat:
        ref=args.repeat.resolve()
        evidence += [ref/name for name in ("TRACE.npz","DECISIONS.npz","NATIVE.npz","RESULT.json","NATIVE_AUDIT.json")]
    recording=None
    if args.action_replay:
        with np.load(ref/"DECISIONS.npz",allow_pickle=False) as z:
            recording={k:np.asarray(z[k]) for k in z.files}
    run(args.out.resolve(),args.start,args.profile.resolve(),policy,evidence,
        "saved_action_replay" if recording is not None else ("stock" if not args.provider else "provider"),
        recording=recording)
    if args.repeat:
        check_repeat(ref,args.out.resolve())


if __name__=="__main__":
    main()
