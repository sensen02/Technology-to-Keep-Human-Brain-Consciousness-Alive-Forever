"""Generate supervised optical samples by rendering around saved physical poses."""
from pathlib import Path

import mujoco as mj
import numpy as np

from bench_runtime import HERE, turn
from optics import Eyes


def datasets():
    body = turn.stand.physics.Core(adhesion=False)
    cam = Eyes(body)
    histories = []
    for label in ("0_0", "1_2400", "-1_2400", "1_4000", "-1_4000"):
        with np.load(HERE / "PULSE" / label / "TRACE.npz") as z:
            indices = np.linspace(500, len(z["qpos"])-1, 9, dtype=int)
            histories.extend(zip(z["qpos"][indices], z["qvel"][indices], strict=True))
    try:
        for name, count, seed in (("OPTICAL_TRAIN", 1800, 63210), ("OPTICAL_DEV", 360, 812389)):
            rng = np.random.default_rng(seed)
            frames = []
            labels = []
            for i in range(count):
                pick = int(rng.integers(0, len(histories)))
                qpos, qvel = histories[pick]
                body.d.qpos[:] = qpos
                body.d.qvel[:] = qvel
                mj.mj_forward(body.plant, body.d)
                visible = bool(rng.random() >= .24)
                angle = float(rng.uniform(-45., 45.)) if i % 3 else float(rng.uniform(-13., 13.))
                range_ = float(rng.uniform(2.4, 3.02))
                width = float(rng.uniform(.082, .163))
                origin = cam.pose()[0].copy()
                background = (origin, float(rng.uniform(-.4, .4)), float(rng.uniform(.15, .40)))
                target = cam.place(angle, range_) if visible else None
                rgb = cam.image(target, width, [1.,.045,.035,1.], background)
                frames.append(rgb)
                labels.append([int(visible), angle if visible else 0.])
            path = HERE / (name + ".npz")
            with path.open("xb") as output:
                np.savez_compressed(output, rgb=np.asarray(frames), labels=np.asarray(labels))
            print(name, count, turn.stand.sha(path), flush=True)
    finally:
        cam.close()


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise RuntimeError("Optical data replay disallows physics integration")
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    datasets()
