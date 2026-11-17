"""Conservative box separation measured on archived configurations."""
import argparse
from pathlib import Path

import numpy as np
import torch

from approach_runtime import turn, sha
from far_fixture import prepare
from geometry import Envelope


def clearance(trial, output, stride):
    torch.set_num_threads(1)
    archive = turn.read(trial / "RESULT.json")
    assert sha(trial / "TRACE.npz") == archive["files"]["TRACE.npz"]
    body,_ = prepare(output.with_suffix(""), "nominal", archive["spec"]["box_shift"])
    env = Envelope(body)
    ids = np.asarray([i for i,_ in env.meshes] + env.pads)
    center = env.data.geom_xpos[env.obstacle.id].copy()
    size = env.obstacle.size.copy()
    local = body.plant.geom_aabb[ids]
    with np.load(trial / "TRACE.npz") as data:
        qpos = data["qpos"]
    box_lo, box_hi = center-size, center+size
    stamps = sorted(set(range(0, len(qpos), stride)) | {len(qpos)-1})
    lowest = 1.e6
    when = None
    pair = None
    overlapping = 0
    for tick in stamps:
        env.set_pose(qpos[tick])
        rot = env.data.geom_xmat[ids].reshape(-1,3,3)
        pos = env.data.geom_xpos[ids]
        mid = np.einsum("ijk,ik->ij",rot,local[:,:3]) + pos
        radius = np.einsum("ijk,ik->ij",abs(rot),local[:,3:])
        radius[len(env.meshes):] = body.plant.geom_size[env.pads,:1]
        lower = np.maximum(mid-radius-box_hi,box_lo-mid-radius).max(1)
        for i in np.flatnonzero(lower[:len(env.meshes)]<max(lowest,.005)):
            verts = env.meshes[i][1] @ rot[i].T + pos[i]
            lower[i] = np.maximum(verts.min(0)-box_hi,box_lo-verts.max(0)).max()
        if (lower <=0).any():
            overlapping += 1
        i = int(lower.argmin())
        if lower[i] < lowest:
            lowest, pair, when = float(lower[i]), body.plant.geom(int(ids[i])).name, tick
    return dict(stride_ms=stride, sample_count=len(stamps), geometry_count=len(ids), trace_sha=sha(trial/"TRACE.npz"),
                smallest_axis_separation=lowest, samples_with_any_AABB_overlap=overlapping,
                minimum_sample_time_ms=when, minimum_sample_geometry=pair,
                scope="Positive conservative axis gap proves no geometry intersection at sampled pose; overlap alone does not establish contact")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("trial",type=Path)
    parser.add_argument("--stride",type=int,default=5)
    args=parser.parse_args()
    results=clearance(args.trial,args.out,args.stride)
    turn.stand.put(args.out,results)
    print(results,flush=True)
