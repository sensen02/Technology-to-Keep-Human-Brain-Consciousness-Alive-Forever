"""Retarget saved reference poses within the muscle model's operating envelope."""

from pathlib import Path
import hashlib
import json

import mujoco as mj
import numpy as np
from flygym.compose.fly.musculoskeletal import _load_mjcf

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
MODEL = PROJECT / "outputs/muscles_six_legs/fruitfly_six_leg_standing.xml"
TRACE = PROJECT / "experiments/banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/trace_1ms.npz"
JOINTS = ("joint_LHCoxa_roll", "joint_RMCoxa_roll")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ReferenceRetargeter:
    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, self.data, 0)
        self.anchor = self.data.qpos.copy()
        joints = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)
        self.indices = model.jnt_qposadr[joints]
        self.limits = model.jnt_range[joints]
        self.columns = [int(model.joint(name).dofadr[0]) for name in JOINTS]
        self.forwards = 0
        self.reference_margin = self.measure(self.anchor)[1]
        if np.any(self.reference_margin <= 1e-8):
            raise ValueError("Reference lacks bidirectional active muscle authority")

    def measure(self, pose):
        data, model = self.data, self.model
        data.qpos[:] = pose
        data.qvel[:] = 0
        data.act[:] = 0
        mj.mj_forward(model, data)
        self.forwards += 1
        lengths = data.actuator_length.copy()
        passive = data.actuator_force.copy()
        arms = np.zeros((model.nu, len(self.columns)))
        for i in range(model.nu):
            start = data.moment_rowadr[i]
            end = start + data.moment_rownnz[i]
            for j, column in enumerate(self.columns):
                local = data.moment_colind[start:end] == column
                arms[i, j] = data.actuator_moment[start:end][local].sum()
        data.act[:] = 1
        mj.mj_fwdActuation(model, data)
        per_muscle = arms * (data.actuator_force - passive)[:, None]
        margin = np.stack((-np.minimum(per_muscle, 0).sum(axis=0),
                            np.maximum(per_muscle, 0).sum(axis=0)), axis=1)
        return lengths, margin

    def feasible(self, pose):
        length, capacity = self.measure(pose)
        bounds = self.model.actuator_lengthrange
        return (np.all((length >= bounds[:, 0]) & (length <= bounds[:, 1]))
                and np.all(capacity >= .15 * self.reference_margin))

    def retarget(self, reference):
        if reference.shape != self.anchor.shape or not np.isfinite(reference).all():
            raise ValueError("Expected a finite position vector in canonical joint order")
        # Preserve the saved floating root and adjust only joint coordinates.
        pose = reference.copy()
        clipped = np.clip(reference[self.indices], *self.limits.T)
        pose[self.indices] = self.anchor[self.indices]
        assert self.feasible(pose)
        # Step toward the target and keep only verified candidates. Local coordinate
        # passes recover motion that the first blocked joint would otherwise limit.
        for group in [self.indices] + [np.array([i]) for i in self.indices] * 2:
            target = clipped[np.searchsorted(self.indices, group)]
            start = pose[group].copy()
            if np.max(abs(target - start)) < 1e-10:
                continue
            candidate = pose.copy()
            for fraction in (1., .5, .25, .125, .0625):
                candidate[group] = start + fraction * (target - start)
                if self.feasible(candidate):
                    pose = candidate
                    break
        return pose


def main():
    source = {str(path): digest(path) for path in (MODEL, TRACE)}
    assert source[str(MODEL)] == "65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6"
    model = _load_mjcf(MODEL).compile()
    retargeter = ReferenceRetargeter(model)
    with np.load(TRACE, allow_pickle=False) as arrays:
        source_qpos, time = arrays["qpos"], arrays["time"]
    frames = np.rint(np.linspace(5800, 15399, 17)).astype(int)
    output = np.stack([retargeter.retarget(source_qpos[i]) for i in frames])
    indices = retargeter.indices
    assert np.array_equal(source_qpos[frames, :7], output[:, :7])
    assert np.all(output[:, indices] >= retargeter.limits[:, 0])
    assert np.all(output[:, indices] <= retargeter.limits[:, 1])
    margins = []
    for row in output:
        assert retargeter.feasible(row)
        margins.append(retargeter.measure(row)[1])
    error = np.linalg.norm(output[:, indices] - source_qpos[frames][:, indices], axis=1)
    anchor_error = np.linalg.norm(retargeter.anchor[indices] - source_qpos[frames][:, indices], axis=1)
    assert np.all(error < anchor_error)
    assert np.all(model.actuator_trntype == mj.mjtTrn.mjTRN_TENDON)
    assert retargeter.data.time == 0
    assert source == {str(path): digest(path) for path in (MODEL, TRACE)}
    report = dict(scope="offline zero-velocity reference samples, not a gait or motor command",
                  input_sha256=source, frame_count=len(frames), time=time[frames].tolist(),
                  joint_angle_l2_error_rad=error.tolist(),
                  anchor_l2_error_rad=anchor_error.tolist(), test_joints=JOINTS,
                  min_active_capacity_ratio=np.min(np.stack(margins) / retargeter.reference_margin, axis=0).tolist(),
                  model_forward_evaluations=retargeter.forwards, physical_steps=0)
    with (HERE / "MUSCLE_REFERENCE.npz").open("xb") as stream:
        np.savez_compressed(stream, qpos=output, source_frame=frames, time=time[frames])
    with (HERE / "MUSCLE_REFERENCE_RESULT.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
