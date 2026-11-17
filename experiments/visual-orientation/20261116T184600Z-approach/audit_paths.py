"""Score the physical walking interval without changing the original behavior report."""
import argparse
from pathlib import Path
import numpy as np
from approach_runtime import turn, sha
from walking_bench import cycles


def case(path):
    report=turn.read(path/"RESULT.json")
    assert all(sha(path/name)==fingerprint for name,fingerprint in report["files"].items())
    with np.load(path/"OBS.npz") as data:
        commands=data["decisions"]
    enabled=commands[commands[:,3]>.06,0].astype(int)
    start,end=int(enabled[0]),int(enabled[-1])+5
    assert len(enabled)==(end-start)//5
    with np.load(path/"TRACE.npz") as t:
        packed={k:t[k] for k in ("time","loads","padz","foot")}
        com=t["com"]
    with np.load(path/"NATIVE.npz") as n:
        forces=n["forces"][start*80:end*80]
    support=float(forces[:,:6].sum(1).mean()/turn.W)
    forward=float(np.linalg.norm(com[end,:2]-com[start,:2])/((end-start)*.001))
    slip=float((forces[:,6:12].sum(0)/forces[:,:6].sum(0)).max()/forward)
    gait,_=cycles(packed,start,end,arm_before=20)
    return dict(behavior_pass=report["passed"],native_ground_max_slip=slip,support_W=support,
                forward_native_s=forward,cycles=gait,native_body_pass=bool(report["passed"] and
                  slip<=.50 and support>=.90 and forward>=.05 and all(gait.values())),
                interval_ms=[start,end],record_sha256=sha(path/"RESULT.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("folders",nargs="+",type=Path)
    a=parser.parse_args()
    result={path.name:case(path) for path in a.folders}
    turn.stand.put(a.output,result)
    for k,v in result.items():
        print(k,v["native_body_pass"],v["native_ground_max_slip"],flush=True)
