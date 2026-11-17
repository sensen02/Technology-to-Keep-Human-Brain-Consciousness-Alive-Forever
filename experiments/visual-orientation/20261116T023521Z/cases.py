"""Freeze training and independent physical visual cases after short-turn calibration."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def scene(angle, onset, width, distance, background, gray, start="nominal", test="target"):
    return dict(angle=angle, onset_ms=onset, width=width, distance=distance, background=background,
                gray=gray, start=start, test=test, color=[1., .045, .035, 1.])


def main():
    train = {}
    test = {}
    for name, sign in (("L", 1), ("R", -1)):
        for index, angle in enumerate((20., 24., 28., 32.)):
            train[f"{name}{index:02d}"] = scene(sign*angle, 600 + 20*index, .10+.012*index,
                                              2.5+.11*index, -.2+.12*index, .22+.04*index)
        rng = np.random.default_rng(8371 + sign)
        angles = rng.uniform(20., 32., 12)
        starts = ("lean_left", "lean_right", "moving_forward", "moving_backward")
        for i, a in enumerate(angles):
            test[f"{name}{i:02d}"] = scene(sign*float(a), 700+int(rng.integers(0,12))*20,
                                          float(rng.uniform(.085,.155)), float(rng.uniform(2.42,2.96)),
                                          float(rng.uniform(-.35,.35)), float(rng.uniform(.18,.38)), starts[i%4])
        matched = scene(25.*sign, 760, .119, 2.57, .16, .25, starts[0] if sign>0 else starts[1])
        for mode in ("target", "cut", "restore"):
            test[f"{name}_{mode}"] = dict(matched, test=mode)
        test[f"{name}_removed"] = dict(matched, test="remove", event_ms=2160)
        test[f"{name}_switch"] = dict(matched, angle=20.*sign, test="switch", event_ms=5560, new_angle=-20.*sign)
    train["off"] = dict(scene(0., 620, .12, 2.7, .10, .3, test="off"), duration_ms=1600)
    train["center"] = dict(scene(0., 620, .12, 2.7, .10, .3, test="center"), duration_ms=1600)
    train["occluded"] = dict(scene(22., 620, .12, 2.7, .10, .3, test="occluded"), duration_ms=1600)
    for mode, a in (("off", 0.), ("center", 0.), ("occluded", 25.)):
        test[mode] = dict(scene(a, 780, .132, 2.88, -.31, .27, "lean_right", mode), duration_ms=3600)
    for name, data in (("TRAIN_CASES.json", train), ("EVAL_CASES.json", test)):
        with (HERE / name).open("x") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
