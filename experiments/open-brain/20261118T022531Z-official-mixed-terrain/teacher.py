"""Official FlyGym HybridTurningController on the same body and official world."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from arena_trial import TRIAL, MixedBody, live, measure
from neural_runtime import read, sha, write


def commands_for(out, seed, source):
    if source == "forward_probe":
        commands = np.zeros((20000, 2))
        commands[2200:15000] = .8
        return commands, "equal .8/.8 during the shared activation window"
    parent = TRIAL / "FORMAL" / f"seed{seed}"
    mode = {"native_command": "native_full", "repair_command": "repaired"}[source]
    data = parent / mode / "TRACE.npz"
    assert sha(data) == read(data.parent / "RESULT.json")["trace_sha256"]
    with np.load(data) as trace:
        commands = trace["command"].copy()
    assert len(commands) == 20000
    return commands, dict(path=str(data), sha256=sha(data))


def run(seed, source):
    out = TRIAL / "TEACHER" / f"seed{seed}" / source
    out.mkdir(parents=True)
    commands, identity = commands_for(out, seed, source)
    start = time.monotonic()
    body = MixedBody(seed)
    # No project terminal-mask override. Body.apply still uses the same MuJoCo
    # limits and foot-pad mechanism as the four live brain lanes.
    mujoco.mj_saveModel(body.m, str(out / "BODY.mjb"), None)
    initial = body.d.xpos[body.thorax].copy()
    arrays = {key: [] for key in ("qpos", "qvel", "position", "upright", "yaw", "nonfoot",
                                  "control", "pads", "normal_load", "terrain_normal",
                                  "feet_xyz", "command")}
    warning = None
    for tick, command in enumerate(commands):
        control, pads = body.teacher(command)
        body.apply(control, pads)
        load, nonfoot, terrain = body.detail()
        rotation = body.d.xmat[body.thorax].reshape(3, 3)
        record = dict(qpos=body.d.qpos, qvel=body.d.qvel, position=body.d.xpos[body.thorax],
            upright=rotation[2, 2], yaw=np.arctan2(rotation[1, 0], rotation[0, 0]),
            nonfoot=nonfoot, control=control, pads=pads, normal_load=load, terrain_normal=terrain,
            feet_xyz=body.d.xpos[body.feet], command=command)
        for key, value in record.items():
            arrays[key].append(np.array(value, copy=True))
        if body.d.warning.number.any() or not np.isfinite(body.d.qpos).all():
            warning = dict(tick=tick, model=body.d.warning.number.tolist())
            break
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    arrays["initial_position"] = initial
    np.savez_compressed(out / "TRACE.npz", **arrays)
    result = measure(arrays, body)
    result.update(seed=seed, controller="official FlyGym 2.1.0 HybridTurningController",
                  source=source, command_identity=identity, trace_sha256=sha(out / "TRACE.npz"),
                  model_sha256=sha(out / "BODY.mjb"), steps=len(arrays["qpos"]),
                  physics_warning=warning, wall_seconds=time.monotonic()-start,
                  source_sha256={str(p): sha(p) for p in (
                      Path(__file__), TRIAL/"arena_trial.py", TRIAL/"PROTOCOL.zh-CN.md",
                      Path(__import__("inspect").getfile(type(body.motor))), Path(__import__("inspect").getfile(type(body.sim.world))))})
    write(out / "RESULT.json", result)
    print(seed, source, round(result["max_x_mm"], 3), result["course_pass"], result["safety_gates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=(10901, 11311))
    parser.add_argument("source", nargs="+", choices=("forward_probe", "native_command", "repair_command"))
    args = parser.parse_args()
    for source in args.source:
        run(args.seed, source)
