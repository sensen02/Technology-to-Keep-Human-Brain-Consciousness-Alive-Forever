"""Capture and restore complete standing states from physical mother trajectories."""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "20261115T191709Z"
sys.path.insert(0, str(PREVIOUS))
import bounded
from bench import Neural, stand

CHECKPOINT = PREVIOUS / "RUN/TRAIN/step_0800.pt"
PINNED_SHA = "b53bafa8c9b843c00c1162b1c55f46faf80354afb970ae1d13f6551c764793d3"


@dataclass(frozen=True)
class Preparation:
    amplitudes: tuple = ()
    force_w: tuple = (0., 0.)
    moment_w: tuple = (0., 0., 0.)
    duration_ms: int = 24


PREPARATIONS = {
    "lean_left": Preparation(force_w=(.01, .26), moment_w=(.09, 0., 0.)),
    "lean_right": Preparation(force_w=(-.01, -.23), moment_w=(-.07, .025, 0.), duration_ms=27),
    "moving_forward": Preparation(force_w=(.30, .025), moment_w=(0., .11, 0.), duration_ms=29),
    "moving_backward": Preparation(force_w=(-.24, -.03), moment_w=(0., -.10, 0.), duration_ms=20),
    "uneven_feet": Preparation(amplitudes=((0, .36), (2, -.28), (4, -.22)),
                                force_w=(.06, -.14), moment_w=(.10, .03, -.08), duration_ms=26),
}


def model():
    if stand.sha(CHECKPOINT) != PINNED_SHA:
        raise RuntimeError("Frozen weights changed")
    return bounded.load(CHECKPOINT)


def capture(bench):
    c = bench.body
    state = np.empty_like(bench.state)
    bench.mj.mj_getState(c.plant, c.d, state, bench.spec)
    return dict(state=state, t=np.array(c.t), ticks=np.array(c.actuator.command_ticks),
                observation=stand.observe(c), qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
                **{f"core_{k}": getattr(c, k).copy() for k in bench.fields},
                **{f"act_{k}": getattr(c.actuator, k).copy() for k in bench.actuator_fields})


def load(bench, path, sha):
    if stand.sha(path) != sha:
        raise RuntimeError("Initial snapshot changed")
    with np.load(path, allow_pickle=False) as z:
        bench.state = z["state"]
        bench.t = int(z["t"])
        bench.command_ticks = int(z["ticks"])
        bench.center = z["observation"]
        bench.fields = {k: z[f"core_{k}"] for k in bench.fields}
        bench.actuator_fields = {k: z[f"act_{k}"] for k in bench.actuator_fields}
        bench.reset()
        np.testing.assert_array_equal(bench.body.d.qpos, z["qpos"])
        np.testing.assert_array_equal(bench.body.d.qvel, z["qvel"])


@torch.no_grad()
def generate(output):
    output.mkdir()
    bench = stand.Bench(output)
    parent = bench.body
    control = Neural(model().restoring)
    cases = {}
    for name, prep in (("nominal", None), *PREPARATIONS.items()):
        bench.reset()
        parent_history = None
        if prep is not None:
            original = parent.apply_adhesion
            rate = bench.zero.copy()
            for leg, amplitude in prep.amplitudes:
                rate[7*leg+1:7*leg+4] = amplitude * np.array([1., -.5, .75])
            wrench = np.array([*prep.force_w, 0., *prep.moment_w]) * stand.physics.W
            expected = np.repeat(wrench[None], prep.duration_ms*2, axis=0)
            expected = np.vstack((expected, np.zeros((10, 6))))
            states, applied, actual = [], [], []

            def impose(loads):
                original(loads)
                parent.d.xfrc_applied[parent.th] = expected[len(actual)]
                actual.append(parent.d.xfrc_applied[parent.th].copy())

            parent.apply_adhesion = impose
            try:
                for tick in range(prep.duration_ms+5):
                    if tick % 5 == 0:
                        action = control(stand.observe(parent))
                    net = action + rate if tick < prep.duration_ms else action
                    if np.abs(net).max() > 1.:
                        raise RuntimeError("Preparation actuation clipped")
                    states.append(np.r_[parent.d.qpos, parent.d.qvel, parent.actuator.target_rad])
                    applied.append(net.copy())
                    parent.tick(net)
            finally:
                parent.apply_adhesion = original
            np.testing.assert_array_equal(actual, expected)
            parent_history = output / f"{name}_mother.npz"
            np.savez_compressed(parent_history, inputs=np.asarray(applied), wrench=actual, states=states)
        snap = capture(bench)
        measurement = parent.measure()
        load_w = measurement["loads"] / stand.physics.W
        tilt = np.rad2deg(np.arccos(np.clip(parent.d.xmat[parent.th, 8], -1., 1.)))
        if (not all(np.isfinite(value).all() for value in snap.values())
            or float(measurement["com"]) < 1.2 or tilt > 30.
            or np.sum(load_w >= .02) < 3 or measurement["nonfoot"] != 0.
            or np.any(parent.d.xfrc_applied) or np.any(parent.d.qfrc_applied)):
            raise RuntimeError(f"Preparation did not produce a valid supported state: {name}, tilt={tilt}, load={load_w}")
        path = output / f"{name}.npz"
        np.savez_compressed(path, **snap)
        cases[name] = dict(sha256=stand.sha(path), preparation=asdict(prep) if prep else None,
            native_steps=(prep.duration_ms + 5)*80 if prep else 0,
            mother_sha256=stand.sha(parent_history) if parent_history else None,
            foot_load_w=load_w.tolist(), com_height=measurement["com"], tilt_deg=float(tilt),
            root_linear_native_s=snap["qvel"][:3].tolist(), angular_rad_s=snap["qvel"][3:6].tolist())
        print(name, "tilt", round(tilt, 3), "load", np.round(load_w, 3),
              "root linear / angular", np.round(snap["qvel"][:6], 3), flush=True)
    vectors = [np.load(output / f"{name}.npz", allow_pickle=False) for name in cases]
    try:
        if len({v["observation"].tobytes() for v in vectors}) != len(vectors):
            raise ValueError("Repeated observation in physical snapshot bank")
    finally:
        for vector in vectors:
            vector.close()
    stand.identity.verify_runtime(bench.manifest, parent)
    stand.put(output / "BANK.json", cases)


if __name__ == "__main__":
    torch.set_num_threads(1)
    HERE.joinpath("RUN").mkdir(exist_ok=True)
    generate(HERE / "RUN" / "INITIAL")
