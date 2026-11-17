"""Collect standing corrections under live, bounded thorax disturbances."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20261115T184423Z"))
import live_push
from live_push import stand

SEED = 2026111519
GAINS = {"v1": (.025, .012, .010), "v2": (.050, .022, .018), "v3": (.090, .035, .030)}


@dataclass(frozen=True)
class Push:
    name: str
    force_w: tuple[float, float]
    duration_ms: int
    offset: float = 0.
    onset_ms: int = 10
    motors: tuple = ()

    @property
    def end(self):
        return max(self.onset_ms + self.duration_ms, 30 if self.motors else 0)


def cases():
    groups = {}
    for index, (group, count) in enumerate((("dev", 6), ("train", 36), ("on", 12), ("test", 12))):
        gen = np.random.default_rng(SEED + index)
        pushes = []
        angles = np.arange(count) * (2 * np.pi / count)
        angles += gen.uniform(-.26, .26, count)
        for i, angle in enumerate(angles):
            amplitude = gen.uniform(.30, .88)
            vector = tuple(amplitude * np.array([np.cos(angle), np.sin(angle)]))
            pushes.append(Push(f"{group}_{i:02}", vector, int(gen.integers(17, 45)),
                               float(gen.uniform(-.30, .30)), int(gen.integers(5, 31))))
        groups[group] = pushes
    groups["test"].extend((Push("quiet", (0., 0.), 0, onset_ms=0),
                           Push("leg_pair", (0., 0.), 0, onset_ms=0,
                                motors=((1, .50), (4, -.35))),
                           Push("mixed", (.20, -.35), 25, onset_ms=30,
                                motors=((2, -.45), (5, .45)))))
    return groups


class Teacher:
    def __init__(self, bench, gain):
        self.bench = bench
        self.translation, self.rotation, self.tilt = GAINS[gain]
        core = bench.body
        self.dofs = core.a_dof.reshape(6, 7)
        self.pads = [core.pads[leg] for leg in stand.physics.LEG_ORDER]
        self.bodies = []
        for ids in self.pads:
            bodies = np.unique(core.plant.geom_bodyid[ids])
            if len(bodies) != 1:
                raise ValueError("Expected all pads on each distal foot body")
            self.bodies.append(int(bodies[0]))

    def __call__(self, x):
        core = self.bench.body
        data = core.d
        rot = data.xmat[core.th].reshape(3, 3)
        linear = rot @ x[177:180]
        spin = rot @ x[180:183]
        # Body z tilted from vertical: restoring root rotation is up_body × up_world.
        tilt = np.cross(np.array([0., 0., 1.]), rot[:, 2])
        angle = self.rotation * spin + self.tilt * tilt
        shift = self.translation * np.r_[linear[:2], 0.]
        normal = core.measure()["loads"] / stand.physics.W
        jac = np.zeros((3, core.plant.nv))
        target_offset = np.zeros((6, 7))
        for leg, cols in enumerate(self.dofs):
            p = data.geom_xpos[self.pads[leg]].mean(axis=0)
            self.bench.mj.mj_jac(core.plant, data, jac, None, p, self.bodies[leg])
            j = jac[:, cols]
            delta = shift + np.cross(angle, p - data.xipos[core.th])
            # Retain support under unloading. Near-zero load does not produce a large horizontal command.
            gain = np.clip(normal[leg] / .10, .12, 1.)
            delta[:2] *= gain
            target_offset[leg] = j.T @ np.linalg.solve(j @ j.T + .07 * np.eye(3), delta)
        rate = 12. * (self.bench.target + target_offset.ravel() - core.actuator.target_rad)
        return (rate / stand.physics.MAX_RATE).clip(-.60, .60)


class Neural:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(torch.from_numpy(x)).numpy()


@torch.no_grad()
def run(bench, push, controller, path, *, prefix=None, switch_ms=0):
    bench.reset()
    core = bench.body
    initial = np.r_[stand.observe(core), core.d.qpos, core.d.qvel, core.actuator.target_rad]
    duration = push.end + 1200
    motor = bench.zero.copy()
    for leg, amplitude in push.motors:
        motor[7 * leg + 1:7 * leg + 4] = amplitude * np.array([1., -.5, .75])
    force = np.array([*push.force_w, 0.]) * stand.physics.W
    arm = core.d.xmat[core.th].reshape(3, 3)[:, 0].copy() * push.offset
    wrench = np.r_[force, np.cross(arm, force)]
    expected = np.zeros((duration * 2, 6))
    expected[2 * push.onset_ms:2 * (push.onset_ms + push.duration_ms)] = wrench
    actual, samples, observations, actions, applied = [], [], [], [], []
    original = core.apply_adhesion

    def apply_wrench(loads):
        original(loads)
        core.d.xfrc_applied[core.th] = expected[len(actual)]
        actual.append(core.d.xfrc_applied[core.th].copy())
        if np.any(core.d.qfrc_applied) or np.any(core.d.xfrc_applied[:core.th]) or np.any(core.d.xfrc_applied[core.th+1:]):
            raise RuntimeError("Unexpected external force")

    def snapshot(m):
        return (float(core.d.time), core.d.subtree_com[core.th].copy(),
                m["loads"].sum() / stand.physics.W, m["nonfoot"],
                core.actuator.target_rad.copy(), core.d.qpos[core.a_qpos].copy(), m["loads"].copy())

    samples.append(snapshot(core.measure()))
    core.apply_adhesion = apply_wrench
    try:
        for tick in range(duration):
            if tick % 5 == 0:
                x = stand.observe(core)
                control = prefix if prefix is not None and tick < switch_ms else controller
                action = np.asarray(control(x))
                if action.shape != (42,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
                    raise ValueError("Invalid action")
                observations.append(x.copy())
                actions.append(action.copy())
            net = action + motor if tick < 30 else action
            if np.any(np.abs(net) > 1.):
                raise ValueError("Motor disturbance clipped")
            m = core.tick(net)
            applied.append(net.copy())
            samples.append(snapshot(m))
    finally:
        core.apply_adhesion = original
        core.d.xfrc_applied[:] = 0.
    np.testing.assert_array_equal(np.asarray(actual), expected)
    data = dict(zip(("time_s", "com", "support", "nonfoot", "target", "joints", "foot_load_native"),
                    map(np.asarray, zip(*samples))))
    data.update(x=np.asarray(observations), action=np.asarray(actions), applied=np.asarray(applied),
                wrench=np.asarray(actual), initial_state=initial, ref_target=bench.target, ref_joints=bench.q,
                motor_injection=motor, teacher_start_row=np.array(switch_ms // 5))
    if not all(np.isfinite(array).all() for array in data.values()):
        raise ValueError("Nonfinite trace")
    np.savez_compressed(path, **data)
    return dict(**live_push.measure(data, push.end, bench.target, bench.q),
                native_steps=80 * duration, action_calls=len(actions), trace_sha256=stand.sha(path))


def improvement(candidate, reference):
    if not candidate["recovered"]:
        return False
    if not reference["recovered"]:
        return True
    timing = candidate["recovery_ms"] - reference["recovery_ms"]
    drift = candidate["end_drift_body_lengths"]
    other = reference["end_drift_body_lengths"]
    return bool((timing <= 60 and drift <= .9 * other)
                or (timing <= -20 and drift <= 1.1 * other))


def eligible(candidate, reference):
    if not candidate["recovered"]:
        return False
    if not reference["recovered"]:
        return True
    return (candidate["recovery_ms"] <= reference["recovery_ms"] + 100
            and candidate["end_drift_body_lengths"] <= 1.25 * reference["end_drift_body_lengths"])
