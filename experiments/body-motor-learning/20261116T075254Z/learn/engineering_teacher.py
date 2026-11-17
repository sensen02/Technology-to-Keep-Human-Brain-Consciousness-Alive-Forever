"""Test the two qualified physical teachers continuously on one unmodified body."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from physical import HERE, NATIVE_PER_MS, Record, SO, bench, pin_base, sha, verify_record, write

SPEED=HERE.parent/"speed"
TURN=HERE.parent/"turn"
sys.path[:0]=[str(SPEED),str(TURN)]
from controller import Controller, turn
from pace_bench import profile_value, raw_window
from pace_extended import ContinuousLoop
from rollout import projected_metrics


def run(output, direction, profile, plan, start):
    torch.set_num_threads(1)
    speed_plan=bench.read(profile)
    turn_plan=bench.read(plan)
    warm=500
    entry=warm+int(speed_plan["walk_ms"])
    arc=entry+turn_plan["arc_ms"]
    straight=arc+turn_plan["taper_ms"]
    coast=arc+turn_plan["coast_ms"]
    finish=coast+turn_plan["stop_ms"]
    output.mkdir()
    session=bench.stand.Bench(output)
    c=session.body
    c.physics_order=bench.stand.physics.rc.SENS_IDX
    np.testing.assert_array_equal(c.physics_order,bench.SENSOR_BY_LEG)
    snapshot=bench.STARTS/f"{start}.npz"
    snapshot_hash=bench.read(bench.STARTS/"BANK.json")[start]["sha256"]
    bench.load_initial(session,snapshot,snapshot_hash)
    pace=ContinuousLoop(c)
    ident=bench.stand.identity.capture(c,dict(scope="continuous physical teacher concatenation",
                 direction=direction,kind="engineering_teacher",start=start,start_sha256=snapshot_hash,
                 units=["native_length_per_s","rad_per_s"],speed_plan=speed_plan,turn_plan=turn_plan),
           extra_paths=[Path(__file__),profile,plan,snapshot,*pin_base(),
              SPEED/"pace_bench.py",SPEED/"pace_extended.py",SPEED/"speed_teacher.py",
              bench.STARTS/"BANK.json",SPEED/"PACE_RAMP_RUN/RESULT.json",
              TURN/"controller.py",TURN/"rollout.py",turn.TEACHER])
    write(output/"RUN_IDENTITY.json",ident)
    raw=Record(c,finish)
    c.recorder=raw
    measured=c.measure()
    rows=[turn.sample(c,measured)]
    decisions=[]
    teacher=None
    failures=0
    wall=time.monotonic()
    for tick in range(finish):
        if tick%5==0:
            x=bench.features(c,measured)
            if tick<entry:
                command=np.array([profile_value(speed_plan,tick-warm),0.])
                action=pace(x,command)
            else:
                if teacher is None:
                    teacher=Controller(c,direction,turn_plan)
                action,command=teacher.action(c,(tick-entry)*.001)
            action=bench.legacy.bounded_action(action)
            decisions.append((tick,x.copy(),command.copy(),action.copy()))
        measured=c.tick(action)
        if np.any(c.d.qfrc_applied) or np.any(c.d.xfrc_applied):
            raise RuntimeError("Unexpected added force")
        rows.append(turn.sample(c,measured))
        failures=failures+1 if measured["com"]<.8 or measured["nonfoot"]>.5*bench.W else 0
        if failures>=100:
            break
    ms={k:np.asarray([mark[k] for mark in rows]) for k in rows[0]}
    dc={k:np.asarray(v) for k,v in zip(("ms","x","command","action"),zip(*decisions),strict=True)}
    actual=len(rows)-1
    verify_record(raw,ms,actual)
    np.savez_compressed(output/"TRACE.npz",**ms)
    np.savez_compressed(output/"DECISIONS.npz",**dc)
    np.savez_compressed(output/"NATIVE.npz",forces=raw.forces[:raw.count],pose=raw.poses[:raw.count])
    body_solution=dict(scope="paced gait + moving arc + zero-yaw exit + stand, same native episode",
         kind="engineering_teacher",direction=direction,start=start,actual_ms=actual,native_integrations=raw.count,
         bounds=[entry,arc,coast,finish],phase=dict(stand_start_ms=warm,speed_start_ms=warm,arc_begin_ms=entry,
                arc_end_ms=arc,taper_complete_ms=straight,stop_start_ms=coast),
         command_units=["native_length_per_s","rad_per_s"],trace_sha256=sha(output/"TRACE.npz"),
         decision_sha256=sha(output/"DECISIONS.npz"),native_sha256=sha(output/"NATIVE.npz"),
         zero_extra_force=True,native_force_cross_checks=raw.cross_checks,full_D1_D8=False,passed=False)
    if actual==finish:
        main=turn.score(ms,"arc_"+direction,(entry,arc,coast,finish))
        exit4=projected_metrics(raw,ms,arc,arc+4800,yaw_index=arc)
        exit_all=projected_metrics(raw,ms,arc,coast,yaw_index=straight)
        straight_metrics=projected_metrics(raw,ms,straight,coast,yaw_index=straight)
        speed_full=raw_window(raw.forces,raw.poses,ms,warm,entry)
        slots=[raw_window(raw.forces,raw.poses,ms,warm+i,warm+j)
               for i,j in speed_plan["observe_ms"]]
        speeds=[s["signed_forward_native_s"] for s in slots]
        yaw=np.unwrap(ms["yaw"])
        tail_yaw=np.rad2deg(yaw[-1]-yaw[-201])
        checks=dict(arc=bool(main["function_pass"] and main["low_slip_pass"]),
            exit_legacy=bool(exit4["passed"] and abs(exit4["net_yaw_deg"])<=10.),
            exit_all=bool(exit_all["passed"] and all(exit_all["cycles"].values())),
            straight=bool(straight_metrics["passed"] and abs(straight_metrics["net_yaw_deg"])<=4.),
            stand=bool(main["initial_stand"]["passed"] and main["tail"]["passed"] and abs(tail_yaw)<=.5),
            accelerates_and_slows=bool(speeds[2]-speeds[0]>.005 and speeds[2]-speeds[-1]>.005),
            speed_safety_full=bool(speed_full["signed_forward_native_s"]>=.05 and
                     speed_full["ground_foot_support_mean_W"]>=.9 and speed_full["minimum_com"]>=1.2 and
                     speed_full["max_ground_normalized_slip"]<=.5 and speed_full["ground_nonfoot_peak_W"]==0.))
        body_solution.update(passed=all(checks.values()),checks=checks,speed_whole=speed_full,speed_segments=slots,
            arc_metrics=main,legacy_exit_window=exit4,full_exit_window=exit_all,straight_window=straight_metrics,
            terminal_residual_yaw_deg=float(tail_yaw))
    bench.stand.identity.verify_runtime(ident,c)
    body_solution["wall_s"]=time.monotonic()-wall
    write(output/"RESULT.json",body_solution)
    print(json.dumps({k:v for k,v in body_solution.items() if k not in {"speed_segments","arc_metrics"}},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction",choices=("left","right"))
    parser.add_argument("out",type=Path)
    parser.add_argument("--start",default="nominal")
    parser.add_argument("--profile",type=Path,default=SPEED/"PACE_RAMP.json")
    parser.add_argument("--plan",type=Path)
    a=parser.parse_args()
    run(a.out.resolve(),a.direction,a.profile.resolve(),
        a.plan.resolve() if a.plan else TURN/f"{a.direction.upper()}_PLAN.json",a.start)
