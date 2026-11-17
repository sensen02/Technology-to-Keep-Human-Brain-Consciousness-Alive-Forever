"""Test the frozen memoryless standing policy while perturbations are active."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "20261115T182419Z"
sys.path.insert(0, str(PARENT))
import stand

TASKS = {k: v for k, v in stand.HOLDOUT.items() if k != "quiet"}
REFERENCE_SHA = "7ad2a4ae0cb26def024b61cc008a094581b653a9ce3394205d6c0435000efc0a"
CHECKPOINT = PARENT / "RUN/TRAIN/step_0180.pt"


def load():
    if stand.sha(CHECKPOINT) != REFERENCE_SHA:
        raise ValueError("Standing checkpoint identity changed")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint["format"] != stand.FORMAT or checkpoint["step"] != 180:
        raise ValueError("Unexpected policy format")
    state = checkpoint["model"]
    model = stand.StandPolicy(state["center"], state["scale"]).double()
    model.load_state_dict(state)
    return model.eval()


def measure(data, after_ms, target, joints):
    com = data["com"]
    support = data["support"]
    nonfoot = data["nonfoot"]
    speed_squared = ((np.diff(com[:, :2], axis=0) / .001 / stand.LENGTH) ** 2).sum(1)
    drift = np.linalg.norm(com[:, :2] - com[0, :2], axis=-1) / stand.LENGTH

    def stable(start, width):
        span = slice(start + 1, start + width + 1)
        return bool(speed_squared[start:start + width].mean() <= .01**2
                    and support[span].mean() >= .9 and com[span, 2].min() >= 1.2
                    and nonfoot[span].max() == 0.)

    recovery = next((k - after_ms for k in range(after_ms, len(com) - 100)
                     if stable(k, 100)), None)
    end_stable = stable(len(com) - 201, 200)
    safe = bool(com[:, 2].min() >= 1.2 and support[1:].mean() >= .9 and nonfoot[1:].max() == 0.)
    target_error = float(np.sqrt(np.square(data["target"][-200:] - target).mean()))
    joint_error = float(np.sqrt(np.square(data["joints"][-200:] - joints).mean()))
    return dict(safe=safe, recovered=safe and recovery is not None and end_stable,
                recovery_ms=recovery, end_stable=end_stable, min_com=float(com[:, 2].min()),
                mean_support=float(support[1:].mean()), nonfoot_peak=float(nonfoot.max()),
                peak_drift_body_lengths=float(drift.max()), end_drift_body_lengths=float(drift[-1]),
                old_peak_drift_gate=bool(drift.max() <= .1),
                end_speed_rms_body_lengths_s=float(np.sqrt(speed_squared[-200:].mean())),
                end_target_error_rms_rad=target_error, end_joint_error_rms_rad=joint_error)


@torch.no_grad()
def run_trial(bench, model, name, output):
    bench.reset()
    core = bench.body
    motors, force_w = TASKS[name]
    motor_ms = 30 if motors else 0
    force_ms = 25 if np.any(force_w) else 0
    after_ms = motor_ms + force_ms
    total_ms = after_ms + 1200
    motor = np.zeros(42)
    for leg, amplitude in motors:
        motor[7 * leg + 1:7 * leg + 4] = amplitude * np.array([1., -.5, .75])
    force = np.array([*force_w, 0.]) * stand.physics.W
    start = np.r_[stand.observe(core), core.d.qpos, core.d.qvel, core.actuator.target_rad]
    originals = core.apply_adhesion
    actual_forces = []
    xs, actions, applied, injected = [], [], [], []
    snapshots = []

    def sample():
        measurement = core.measure()
        return (float(core.d.time), core.d.subtree_com[core.th].copy(),
                measurement["loads"].sum() / stand.physics.W, measurement["nonfoot"],
                core.actuator.target_rad.copy(), core.d.qpos[core.a_qpos].copy())

    def apply_wrench(loads):
        originals(loads)
        if motor_ms <= tick < after_ms and force_ms:
            core.d.xfrc_applied[core.th, :3] = force
        actual_forces.append(core.d.xfrc_applied[core.th, :3].copy())
        if np.any(core.d.qfrc_applied):
            raise RuntimeError("Unexpected external generalized force")

    snapshots.append(sample())
    core.apply_adhesion = apply_wrench
    try:
        for tick in range(total_ms):
            if tick % 5 == 0:
                x = stand.observe(core)
                action = np.zeros(42) if model is None else model(torch.from_numpy(x)).numpy()
                if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
                    raise RuntimeError("Invalid policy output")
                xs.append(x)
                actions.append(action.copy())
            injection = motor if tick < motor_ms else bench.zero
            net = action + injection
            if np.any(np.abs(net) > 1.):
                raise RuntimeError("Motor disturbance changed due to rate clipping")
            core.tick(net)
            applied.append(net.copy())
            injected.append(injection.copy())
            snapshots.append(sample())
    finally:
        core.apply_adhesion = originals
        core.d.xfrc_applied[:] = 0.
    actual_forces = np.asarray(actual_forces)
    expected = np.zeros((total_ms * 2, 3))
    if force_ms:
        expected[motor_ms * 2:after_ms * 2] = force
    np.testing.assert_array_equal(actual_forces, expected)
    data = {key: np.asarray(val) for key, val in zip(
        ("time_s", "com", "support", "nonfoot", "target", "joints"), zip(*snapshots))}
    data.update(x=np.asarray(xs), action=np.asarray(actions), applied=np.asarray(applied),
                motor_injection=np.asarray(injected), wrench_at_refresh=actual_forces,
                initial_state=start, reference_target=bench.target, reference_joints=bench.q)
    np.savez_compressed(output, **data)
    return dict(**measure(data, after_ms, bench.target, bench.q),
                force_ms=force_ms, motor_ms=motor_ms, action_calls=len(xs),
                native_integrations=total_ms * 80, trace_sha256=stand.sha(output)), start


@torch.no_grad()
def replay(output, model):
    result = json.loads((output / "RESULT.json").read_text())
    verification = {}
    for task, pair in result["evaluation"].items():
        trace = {}
        for arm, report in pair.items():
            path = output / f"{task}_{arm}.npz"
            if stand.sha(path) != report["trace_sha256"]:
                raise RuntimeError("Trace changed")
            with np.load(path, allow_pickle=False) as z:
                trace[arm] = {key: z[key] for key in z.files}
            item = trace[arm]
            expected = (model(torch.from_numpy(item["x"])).numpy() if arm == "model"
                        else np.zeros_like(item["action"]))
            np.testing.assert_allclose(expected, item["action"], atol=1e-12, rtol=0.)
            for tick, applied in enumerate(item["applied"]):
                np.testing.assert_array_equal(applied, item["action"][tick // 5] + item["motor_injection"][tick])
            a = measure(item, report["force_ms"] + report["motor_ms"],
                        item["reference_target"], item["reference_joints"])
            if any(a[k] != report[k] for k in a):
                raise RuntimeError("Saved metric mismatch")
        for field in ("initial_state", "wrench_at_refresh", "motor_injection"):
            np.testing.assert_array_equal(trace["model"][field], trace["zero"][field])
        verification[task] = True
    stand.put(output / "REPLAY.json", dict(all_verified=all(verification.values()),
                                           tests=verification, physical_updates=0))


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("output", type=Path)
    arg.add_argument("--replay", action="store_true")
    args = arg.parse_args()
    torch.set_num_threads(1)
    model = load()
    if args.replay:
        replay(args.output, model)
        return
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    bench = stand.Bench(args.output)
    result = {}
    sources = {str(p): stand.sha(p) for p in (Path(__file__), HERE / "PREREG.zh-CN.md", CHECKPOINT)}
    for name in TASKS:
        pair = {}
        for arm, policy in (("model", model), ("zero", None)):
            pair[arm], body_state = run_trial(bench, policy, name, args.output / f"{name}_{arm}.npz")
            if arm == "model":
                initial = body_state
            else:
                np.testing.assert_array_equal(body_state, initial)
            r = pair[arm]
            print(name, arm, r["recovered"], r["recovery_ms"], r["peak_drift_body_lengths"], flush=True)
        result[name] = pair
    stand.identity.verify_runtime(bench.manifest, bench.body)
    if any(stand.sha(Path(p)) != digest for p, digest in sources.items()):
        raise RuntimeError("Run inputs drift")
    stand.put(args.output / "RESULT.json", dict(evaluation=result, checkpoint_sha=REFERENCE_SHA,
                        sources=sources, settle_native_steps=80000, optimizer_updates=0,
                        wall_s=time.monotonic()-start))


if __name__ == "__main__":
    main()
