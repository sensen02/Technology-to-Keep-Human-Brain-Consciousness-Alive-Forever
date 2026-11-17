"""Zero-integration view sweep around the recorded terminal heading."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import Eyes, turn
from approach_vision import COLOR, WIDTH, features
from far_fixture import prepare
from visual_rejoin import VisualRejoin, marker_bearing

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("trial",type=Path)
    args=parser.parse_args()
    result=turn.read(args.trial/"RESULT.json")
    torch.set_num_threads(1)
    fixture,_=prepare(args.out.with_suffix(""),"nominal", result["spec"]["box_shift"])
    with np.load(args.trial/"TRACE.npz") as trace, np.load(args.trial/"OBS.npz") as data:
        start=trace["head"][0].copy()
        src_qpos=trace["qpos"][-1]
        src_qvel=trace["qvel"][-1]
        target=data["goal"]
    rows=[]
    frames=[]
    vision=VisualRejoin().model
    eyes=Eyes(fixture)
    try:
        for delta in (-28.,-14.,-7.,0.,7.,14.,28.):
            d = np.deg2rad(delta) / 2.
            qpos = src_qpos.copy()
            mj.mju_mulQuat(qpos[3:7],np.array([np.cos(d),0.,0.,np.sin(d)]),src_qpos[3:7])
            fixture.d.qpos[:], fixture.d.qvel[:] = qpos, src_qvel
            mj.mj_forward(fixture.plant, fixture.d)
            image=eyes.image(target,WIDTH,COLOR,(start,result["shift"],.255))
            with torch.inference_mode():
                angle,dist,conf=vision(features(torch.from_numpy(image[None])))[0].tolist()
            ray=marker_bearing(image)
            rgb=image.astype(float)
            strength=float((rgb[...,0]-.5*(rgb[...,1]+rgb[...,2])).max())
            rows.append(dict(yaw_offset_deg=delta,true_bearing_deg=eyes.bearing(target),
                             estimate_deg=angle*40., range_estimate=dist+2.5, logit=conf,
                             red_difference_max=strength, fallback_bearing_deg=ray))
            frames.append(image)
    finally:
        eyes.close()
    np.savez_compressed(args.out.with_suffix(".npz"),images=frames)
    turn.stand.put(args.out,dict(scope="terminal snapshot rendered in seven heading counterfactuals, zero physics integration",
                               true_trajectory_changed=False, draws=rows))
