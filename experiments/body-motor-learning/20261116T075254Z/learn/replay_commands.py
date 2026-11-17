"""Retest an actor under fixed saved engineering commands from a qualified episode."""
import argparse
from pathlib import Path

import numpy as np
import torch

import physical
from course import Course,UNITS
from physical import HERE,bench,load_provider,sha,write


class SavedCourse(Course):
    """Sample exogenous intent; the body policy has no access to the table or time."""
    @classmethod
    def load(cls,path):
        d=bench.read(path)
        c=cls(d)
        source=Path(d["command_source"])
        if sha(source)!=d["command_source_sha256"]:
            raise ValueError("Declared upstream command source changed")
        with np.load(source,allow_pickle=False) as z:
            ms,u=z["ms"],z["command"]
        np.testing.assert_array_equal(ms,np.arange(c.finish//5)*5)
        if u.shape!=(c.finish//5,2) or not np.isfinite(u).all():
            raise ValueError("Invalid engineering intent archive")
        c.table=u
        return c

    def command(self,ms):
        return self.table[min(ms//5,len(self.table)-1)].copy()


def build(source,path):
    report=bench.read(source/"RESULT.json")
    if report.get("kind")!="engineering_teacher" or report.get("passed") is not True:
        raise ValueError("Only a fully relevant qualified engineering composite can set these benchmarks")
    D=source/"DECISIONS.npz"
    if sha(D)!=report["decision_sha256"]:
        raise ValueError("Decision identity changed")
    with np.load(D,allow_pickle=False) as z:
        u=z["command"]
    cfg=bench.read(source/"RUN_IDENTITY.json")["config"]
    pace=cfg["speed_plan"]
    ph=report["phase"]
    warm=ph["stand_start_ms"]
    arc=ph["arc_begin_ms"]
    coast=ph["stop_start_ms"]
    clock=report["actual_ms"]
    trim_end=ph["taper_complete_ms"]+int(cfg["turn_plan"].get("heading_blend_ms",450))
    spans=[(0,warm,"stand")]
    for (a,b),name in zip(pace["observe_ms"],("low","accel","high","decel","slowed"),strict=True):
        spans.append((a+warm,b+warm,name))
    if spans[-1][1]<arc:
        spans.append((spans[-1][1],arc,"aligned_walk"))
    # Record the motion with zero commanded yaw separately from handoff trim.
    brake=coast-int(cfg["turn_plan"]["brake_ms"])
    spans.extend([(arc,ph["arc_end_ms"],"arc"),(ph["arc_end_ms"],trim_end,"turn_out"),
                  (trim_end,brake,"exit"),(brake,coast,"brake"),(coast,clock,"stop")])
    seg=[]
    for a,b,name in spans:
        start=u[min(a//5,len(u)-1)].copy()
        end=u[min(b//5,len(u)-1)].copy()
        start[np.abs(start)<1e-12]=0.
        end[np.abs(end)<1e-12]=0.
        if name=="stand":
            end=np.zeros(2)
        seg.append(dict(name=name,duration_ms=b-a,first=start.tolist(),last=end.tolist()))
    d=dict(units=list(UNITS),segments=seg,
           speed_pair=dict(low="low",high="high",slowed="slowed",delta_min=.005),
           turn=dict(direction=1 if report["direction"]=="left" else -1,arc="arc",exit="exit",minimum_yaw_deg=20.),
           command_source=str(D),command_source_sha256=sha(D),
           verified_source_result=str(source/"RESULT.json"),source_result_sha256=sha(source/"RESULT.json"),
           scope="exogenous command matched replay; recorded physical actions and state are excluded from the body controller")
    write(path,d)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("provider",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--profile",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    source=a.source.resolve()
    out=a.output.resolve()
    checkpoint=a.checkpoint.resolve()
    provider=a.provider.resolve()
    profile=a.profile.resolve() if a.profile else HERE/f"{out.name}_COMMANDS.json"
    if not profile.is_file():
        build(source,profile)
    physical.Course=SavedCourse
    actor=load_provider(provider,checkpoint)
    library=checkpoint.parent/"LIBRARY.npz" if checkpoint.suffix==".pt" else checkpoint
    run_provenance=[Path(__file__),provider,checkpoint,library,library.parent/"LIBRARY.json",
                    source/"RESULT.json",source/"DECISIONS.npz"]
    physical.run(out,"nominal",profile,actor,run_provenance,"saved_engineering_command_learner")


if __name__=="__main__":
    main()
