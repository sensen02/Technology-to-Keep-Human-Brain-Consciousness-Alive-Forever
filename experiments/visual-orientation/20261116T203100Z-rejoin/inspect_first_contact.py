"""Find the touching anatomy around an archived first native obstacle-force sample."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import turn
from far_fixture import prepare


def analyze(trial, out):
    result = turn.read(trial / "RESULT.json")
    impact = result["first_obstacle_force_ms"]
    assert impact is not None
    torch.set_num_threads(1)
    body, _ = prepare(out.with_suffix(""), "nominal", .43)
    obstacle = body.plant.geom("navigation_obstacle").id
    tick = int(impact)
    rows = []
    with np.load(trial / "TRACE.npz", allow_pickle=False) as archive:
        qpos, qvel = archive["qpos"], archive["qvel"]
    for ms in range(max(0, tick-2), min(tick+6, len(qpos))):
        body.d.qpos[:], body.d.qvel[:] = qpos[ms], qvel[ms]
        mj.mj_forward(body.plant, body.d)
        contacts = [{"geometry": body.plant.geom(int(pair.geom[0] if pair.geom[1] == obstacle else pair.geom[1])).name,
                     "distance": float(pair.dist)} for pair in body.d.contact[:body.d.ncon] if obstacle in pair.geom]
        rows.append({"ms": ms, "contacts": contacts})
    turn.stand.put(out, {"first_native_ms": impact, "geometry_samples": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analyze(args.trial, args.out)
