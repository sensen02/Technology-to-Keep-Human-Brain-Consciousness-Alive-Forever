"""Reapply the established turning and exit windows to a fitted composite replay."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from physical import HERE, NATIVE_PER_MS, bench, sha, window, write

sys.path.insert(0,str(HERE.parent/"turn"))
from rollout import projected_metrics,turn


def audit(source,run,output):
    reference=bench.read(source/"RESULT.json")
    actual=bench.read(run/"RESULT.json")
    moves=np.load(run/"TRACE.npz",allow_pickle=False)
    with moves as z:
        trace={k:np.asarray(z[k]) for k in z.files}
    actual_ms=len(trace["time"])-1
    if actual_ms!=reference["actual_ms"]:
        raise ValueError("Composite replay terminated before its original endpoints")
    assert sha(run/"TRACE.npz")==actual["trace_sha256"]
    files=bench.read(run/"NATIVE_AUDIT.json")
    assert sha(run/"NATIVE.npz")==files["native_sha256"]
    with np.load(run/"NATIVE.npz",allow_pickle=False) as archive:
        raw=SimpleNamespace(poses=archive["pose"],forces=archive["forces"])
    if len(raw.poses)!=actual_ms*NATIVE_PER_MS or raw.poses[:,7].any():
        raise ValueError("Unexpected native record")
    mode="arc_"+reference["direction"]
    entry,arc,coast,end=reference["bounds"]
    straight=reference["phase"]["taper_complete_ms"]
    old=turn.score(trace,mode,(entry,arc,coast,end))
    native_arc=window(raw,trace,entry,arc)
    late=projected_metrics(raw,trace,arc,arc+4800,yaw_index=arc)
    complete=projected_metrics(raw,trace,arc,coast,yaw_index=straight)
    directed=projected_metrics(raw,trace,straight,coast,yaw_index=straight)
    yaw=float(np.rad2deg(np.unwrap(trace["yaw"][-201:])[-1]-trace["yaw"][-201]))
    gates=dict(arc=bool(old["function_pass"] and old["low_slip_pass"] and all(native_arc["gates"].values())),
               old_exit_raw=bool(late["passed"] and abs(late["net_yaw_deg"])<=10.),
               full_exit=bool(complete["passed"] and all(complete["cycles"].values())),
               post_taper_straight=bool(directed["passed"] and abs(directed["net_yaw_deg"])<=4.),
               zero_yaw_continued_replay=bool(files["gates"]["exit"]),
               acceleration_and_slowing=bool(files["gates"]["speed"]),
               initial_and_terminal_stands=bool(files["initial_stand"]["passed"] and files["terminal_stand"]["passed"] and abs(yaw)<=.5))
    value=dict(scope="Held original 4.8 s and kept the full exit including taper and brake",
               sampled_arc=old,native_arc=native_arc,old_exit_raw=late,full_exit=complete,
               post_taper_straight=directed,native_steps=actual_ms*NATIVE_PER_MS,
               zero_extra_force=True,terminal_yaw_deg=yaw,
               source=str(source),native_sha256=sha(run/"NATIVE.npz"),
               gates=gates,passed=all(gates.values()))
    write(output,value)
    print(value["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path)
    p.add_argument("run",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    audit(a.reference.resolve(),a.run.resolve(),a.output.resolve())
