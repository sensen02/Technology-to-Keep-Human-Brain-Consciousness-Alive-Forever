"""Paired right-turn release probe using the archived physical actuator targets."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from approach_runtime import ApproachLink, BINDING, sha, stand_window, turn
from far_fixture import CollisionRecord, prepare
from far_optics import WEIGHTS


def test(out):
    source = WEIGHTS.parent / "DETOUR_A"
    replay = turn.read(source / "RESULT.json")
    assert sha(source / "TRACE.npz") == replay["files"]["TRACE.npz"]
    with np.load(source / "TRACE.npz", allow_pickle=False) as saved:
        pose, velocity, ctrl, target = (saved[key][-1].copy() for key in ("qpos", "qvel", "ctrl", "target"))
        padz = saved["padz"][-2:].copy()
    torch.set_num_threads(1)
    out.mkdir()
    runs = {}
    traces = {}
    for release in (False, True):
        name = "release" if release else "hold"
        folder = out / name
        body, motion = prepare(folder, "nominal", .43)
        body.d.qpos[:], body.d.qvel[:], body.d.ctrl[:] = pose, velocity, ctrl
        body.actuator.target_rad[:] = target
        body._padz_prev = padz[-1].copy()
        body._pad_rate = (padz[-1]-padz[-2])*1000.
        mj.mj_forward(body.plant, body.d)
        neck = ApproachLink()
        # Bring the matched source queue to the last confirmed right turn
        # before integrating the newly constructed physical initial condition.
        for _ in range(5):
            neck.send(0., -1)
            neck.advance()
        assert neck.current() == (0., -1)
        duration = 4500
        native = CollisionRecord(body, duration)
        body.recorder = native
        measured = body.measure()
        samples = [turn.sample(body, measured)]
        commands = []
        for time in range(duration):
            if time % 20 == 0:
                neck.send(0., 0 if release and 500 <= time < 1400 else -1)
            if time % 5 == 0:
                command = neck.current()
                action = motion.action(turn.features(body, measured), *command)
                commands.append([time, *command])
            measured = body.tick(action)
            if body.d.xfrc_applied.any() or body.d.qfrc_applied.any():
                raise RuntimeError("Unexpected externally applied wrench")
            samples.append(turn.sample(body, measured))
            neck.advance()
        trace = {key: np.asarray([sample[key] for sample in samples]) for key in samples[0]}
        commands = np.asarray(commands)
        assert native.count == duration*80
        assert np.array_equal(native.endpoints, trace["qpos"])
        all_yaw = np.rad2deg(np.unwrap(trace["yaw"]))
        runs[name] = dict(yaw_start_to_500_deg=float(all_yaw[500]-all_yaw[0]),
                          yaw_after_1500_deg=float(all_yaw[-1]-all_yaw[1500]),
                          contacts=int(np.count_nonzero(native.block)), native_steps=native.count,
                          minimum_com=float(trace["com"][:, 2].min()),
                          native_nonfoot=float(native.forces[:, 12].max()))
        traces[name] = (trace, commands, native.block.copy())
        np.savez_compressed(folder / "TRACE.npz", decisions=commands, **trace)
        np.savez_compressed(folder / "NATIVE.npz", forces=native.forces, obstacle=native.block)
        turn.stand.put(folder / "TRANSPORT.json", neck.link.journal())
        turn.stand.put(folder / "RESULT.json", runs[name])
    old, old_commands, old_force = traces["hold"]
    new, new_commands, new_force = traces["release"]
    different = np.flatnonzero(np.any(old_commands[:,1:] != new_commands[:,1:], axis=1))
    switch = int(new_commands[different[0],0])
    for key in ("qpos", "qvel", "com", "target", "ctrl"):
        np.testing.assert_array_equal(old[key][:switch+1], new[key][:switch+1])
    np.testing.assert_array_equal(old_force[:switch*80], new_force[:switch*80])
    result = dict(paired_prefix_ms=switch, runs=runs, initiation="archived pose/velocity/targets/ctrl, rebuilt runtime and delivered initial right turn",
                  source_trace=sha(source / "TRACE.npz"), script_sha256=sha(Path(__file__)), binding_sha256=sha(BINDING),
                  gate=dict(hold_stalls=abs(runs["hold"]["yaw_after_1500_deg"])<.3,
                            release_recovers=runs["release"]["yaw_after_1500_deg"]<=-3,
                            safe=all(case["contacts"]==case["native_nonfoot"]==0 and case["minimum_com"]>=1.2
                                     for case in runs.values())))
    turn.stand.put(out / "RESULT.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    test(parser.parse_args().out)
