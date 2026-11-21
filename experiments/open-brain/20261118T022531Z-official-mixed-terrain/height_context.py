"""Context for the inherited world-Z gate on an uneven course."""
from pathlib import Path
import json

import numpy as np

TRIAL = Path(__file__).resolve().parent
MODES = ("native_full", "reference_boundary", "translated", "repaired")


def floor_below(xy, tiles):
    floor = np.full(len(xy), -np.inf)
    for tile in tiles:
        point = np.asarray(tile["pos"])
        extent = np.asarray(tile["half_size"])
        if tile["type"] == 0:  # MuJoCo infinite plane
            floor[:] = np.maximum(floor, point[2])
        elif tile["type"] == 6:  # MuJoCo box
            inside = (np.abs(xy - point[:2]) <= extent[:2] + 1e-12).all(1)
            floor[inside] = np.maximum(floor[inside], point[2] + extent[2])
    assert np.isfinite(floor).all()
    return floor


def summarize(trace, tiles):
    position = trace["position"]
    surface = floor_below(position[:, :2], tiles)
    threshold = float(.7 * trace["initial_position"][2])
    interval = (position[:, 0] >= 4.935) & (position[:, 0] <= 11.)
    result = {key: None for key in ("min_world_z_in_course", "min_clearance_in_course")}
    if interval.any():
        result.update(min_world_z_in_course=float(position[interval, 2].min()),
                      min_clearance_in_course=float((position[:, 2] - surface)[interval].min()))
    result["world_z_below_inherited_floor_ticks"] = int((position[:,2] < threshold).sum())
    result["projected_clearance_below_inherited_floor_ticks"] = int(((position[:,2] - surface) < threshold).sum())
    result["gap_bottom_feet_contact_ticks"] = int((trace["terrain_normal"][:,4].max(axis=1)>1e-6).sum())
    return result


def main():
    rows = {}
    for seed in (10901, 11311):
        parent = TRIAL/"FORMAL"/f"seed{seed}"
        tiles = json.loads((parent/"ARENA.json").read_text())["tiles"]
        rows[seed] = {}
        for mode in MODES:
            with np.load(parent/mode/"TRACE.npz") as trace:
                rows[seed][mode] = summarize(trace, tiles)
    with (TRIAL/"HEIGHT_CONTEXT.json").open("x") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
