import argparse
from pathlib import Path

import numpy as np
import torch

from approach_runtime import Eyes, turn
from approach_vision import COLOR, WIDTH
from far_fixture import prepare
from geometry import Envelope


def read_green(image):
    rgb = image.astype(float)
    return (rgb[...,1] > 75) & (rgb[...,1] > 2 * rgb[...,0]) & (rgb[...,1] > 2 * rgb[...,2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--extra", type=float, default=.43)
    parser.add_argument("--target", type=float, default=6.00)
    args = parser.parse_args()
    torch.set_num_threads(1)
    body,_ = prepare(args.out, "nominal", args.extra)
    geom = Envelope(body)
    eyes = Eyes(body)
    try:
        goal = eyes.place(0., args.target)
        image = eyes.image(goal, WIDTH, COLOR, (eyes.pose()[0].copy(), .12, .255))
    finally:
        eyes.close()
    red = (image[...,0] > 45) & (image[...,1] < .7 * image[...,0]) & (image[...,2] < .7 * image[...,0])
    green = read_green(image)
    bounds = geom.used_bounds()
    result = dict(target=goal.tolist(), box_near=geom.near, box_far=geom.far,
                  initial_whole_body_x=list(bounds[:2]), initial_pad_x=list(bounds[2:]),
                  initial_body_clearance_x=geom.near - bounds[1],
                  initial_pad_clearance_x=geom.near - bounds[3],
                  red_pixels_per_eye=red.sum((1,2)).tolist(), green_pixels_per_eye=green.sum((1,2)).tolist())
    np.savez_compressed(args.out / "ACTUAL_EYES.npz", image=image)
    turn.stand.put(args.out / "RESULT.json", result)
    print(result, flush=True)
