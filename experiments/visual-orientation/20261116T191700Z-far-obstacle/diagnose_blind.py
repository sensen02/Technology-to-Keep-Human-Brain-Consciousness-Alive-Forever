"""Locate the contact near the native first-impact time from archived poses."""
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import turn
from far_fixture import prepare

root = Path(__file__).resolve().parent
trial = root / "FAR_BLIND"
result = turn.read(trial/"RESULT.json")
torch.set_num_threads(1)
body,_ = prepare(root/"BLIND_CONTACT_FIXTURE","nominal",result["spec"]["box_shift"])
box = body.plant.geom("navigation_obstacle").id
around = int(result["first_obstacle_force_ms"])
rows = []
with np.load(trial/"TRACE.npz") as data:
    qpos, qvel = data["qpos"], data["qvel"]
    for t in range(around-2, around+5):
        body.d.qpos[:], body.d.qvel[:] = qpos[t], qvel[t]
        mj.mj_forward(body.plant, body.d)
        rows.append(dict(ms=t, candidates=[
            dict(geometry=body.plant.geom(int(pair.geom[0] if pair.geom[1] == box else pair.geom[1])).name,
                 distance=float(pair.dist))
            for pair in body.d.contact[:body.d.ncon] if box in pair.geom]))
turn.stand.put(root/"BLIND_FIRST_CONTACT.json",dict(
    exact_first_native_ms=result["first_obstacle_force_ms"],
    recorder="read-only 1 ms geometry witness; source timestamp from native step", physical_samples=rows))
