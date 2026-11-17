"""Distance audit of every moving mesh and pad over recorded poses."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import turn, sha
from far_fixture import prepare
from geometry import Envelope


def clearance(trial, output, stride):
    torch.set_num_threads(1)
    archive = turn.read(trial/"RESULT.json")
    assert sha(trial/"TRACE.npz") == archive["files"]["TRACE.npz"]
    body,_ = prepare(output.with_suffix(""), "nominal", archive["spec"]["box_shift"])
    env = Envelope(body)
    ids = np.asarray([i for i,_ in env.meshes] + env.pads)
    center = env.data.geom_xpos[env.obstacle.id].copy()
    size = env.obstacle.size.copy()
    r = body.plant.geom_rbound[ids]
    close_max = .25
    closest = close_max
    when = None
    pair = None
    with np.load(trial/"TRACE.npz") as data:
        qpos=data["qpos"]
    stamps = list(range(0, len(qpos), stride))
    if stamps[-1] != len(qpos)-1:
        stamps.append(len(qpos)-1)
    for tick in stamps:
        env.set_pose(qpos[tick])
        bound = np.linalg.norm(np.maximum(np.abs(env.data.geom_xpos[ids]-center)-size,0.),axis=1) - r
        for g in ids[bound < closest]:
            gap = mj.mj_geomDistance(body.plant, env.data, int(g), env.obstacle.id, closest, None)
            if gap < closest:
                closest = gap
                when = tick
                pair = body.plant.geom(int(g)).name
    return dict(stride_ms=stride, sample_count=len(stamps), geometry_count=len(ids), trace_sha=sha(trial/"TRACE.npz"),
                maximum_search_distance=close_max, minimum_sampled_distance=closest,
                at_least_search_distance_all_samples=when is None,
                minimum_sample_time_ms=when, minimum_sample_geometry=pair,
                caveat="Sampled geometric distance; native contact forces are separately archived every 12.5 us")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("trial",type=Path)
    parser.add_argument("--stride",type=int,default=5)
    args=parser.parse_args()
    results=clearance(args.trial,args.out,args.stride)
    turn.stand.put(args.out,results)
    print(results,flush=True)
