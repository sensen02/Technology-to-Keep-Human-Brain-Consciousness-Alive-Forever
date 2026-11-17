"""Gate the crossed physical test before fitting a forecast."""
import argparse
from pathlib import Path

import numpy as np

from bind import BINDING, GRAPH
from replay import replay_archive
from separation import GROUPS, SCALE, brain_projection
from touch_runtime import sha, turn


def cases(root):
    for id_ in range(GROUPS):
        for condition in ("L0T0","L0T1","L1T0","L1T1"):
            folder=root/f"group_{id_:02d}"/condition
            result=turn.read(folder/"RESULT.json")
            path=folder/"SAMPLES.npz"
            assert sha(path)==result["sha256"]
            yield result,path


def verify(root, out):
    matrix,pools=brain_projection()
    split_count={}
    all_rows=[]
    independent=None
    minimum=None
    for trial,path in cases(root):
        opts=trial["case"]
        side=opts["side"]
        begin,end=opts["onset"],opts["onset"]+opts["length"]
        with np.load(path,allow_pickle=False) as z:
            rx=z["received"]
            # Measurement at time k reflects activity consumed from k-2.
            window=rx[begin+2:end+2]
            decoded_load=float(np.median(window[:,side+2])*SCALE)
            metrics=(rx[:,side]>.01)&(rx[:,4]>0.)
            any_touch=bool(metrics.any())
            first_detection=int(np.flatnonzero(metrics)[0]) if any_touch else None
            lag=None if first_detection is None or trial["physical_probe_ms"] is None else (
                first_detection - trial["physical_probe_ms"])
            checks={"phase_truth":trial["passed"],
                    "lateral_detection":any_touch == trial["lateral"],
                    "separate_support":(decoded_load>=.035) == trial["support"],
                    "timely_tactile":not trial["lateral"] or lag is not None and -2<=lag<=15,
                    "opposite_side_no_tactile":not bool((rx[:,1-side]>.01).any())}
            row=dict(group=opts["group"],split=opts["split"],side=side,support=trial["support"],lateral=trial["lateral"],
                     output_normal=decoded_load, tactile_first_ms=first_detection,tactile_delay_ms=lag,
                     checks=checks,passed=all(checks.values()))
            all_rows.append(row)
            k=f"{opts['split']}_L{int(trial['support'])}T{int(trial['lateral'])}"
            count=split_count.setdefault(k,dict(total=0,passed=0))
            count["total"]+=1
            count["passed"]+=int(row["passed"])
            if opts["split"]=="held" and trial["support"] and trial["lateral"] and trial["passed"] and independent is None:
                independent=(z["input_activity"],z["commands"],rx,opts["onset"],path)
    assert independent is not None
    physical, cmd, reference, start,path=independent
    arrived,identity=replay_archive(physical,cmd)
    np.testing.assert_array_equal(arrived,reference)
    no_touch,_=replay_archive(physical,cmd,isolate=slice(0,2))
    no_load,_=replay_archive(physical,cmd,isolate=slice(2,4))
    np.testing.assert_array_equal(no_touch[:,2:],arrived[:,2:])
    np.testing.assert_array_equal(no_load[:,:2],arrived[:,:2])
    assert not no_touch[:,:2].any() and not no_load[:,2:4].any()
    interval=(start,start+55)
    disconnected,_=replay_archive(physical,cmd,cut=interval)
    assert not disconnected[slice(*interval),4].any()
    first_back=next(i for i in range(interval[1],len(disconnected)) if disconnected[i,4])
    np.testing.assert_array_equal(disconnected[first_back:],arrived[first_back:])
    held=[r for r in all_rows if r["split"]=="held"]
    max_delay=max(r["tactile_delay_ms"] for r in held if r["tactile_delay_ms"] is not None)
    result=dict(passed=all(row["passed"] for row in held) and len(held)==16,
                counts=split_count, held_maximum_tactile_delay_ms=max_delay,
                native_steps=sum(r["native_steps"] for r,_ in cases(root)),
                rank=int(np.linalg.matrix_rank(matrix)), projection_brain_targets=len(matrix),
                source_exclusive_target_counts=[len(pool) for pool in pools],
                independent=dict(path=str(path.resolve()), exact_replay=True, projection_route=identity,
                                 touch_silenced_load_bitexact=True,load_silenced_touch_bitexact=True,
                                 whole_an_cut=interval,first_valid_after_cut_ms=first_back),
                trials=all_rows)
    turn.stand.put(out,result)
    print({k:v for k,v in result.items() if k!="trials"},flush=True)
    return result["passed"]


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    raise SystemExit(0 if verify(a.root,a.out) else 1)
