"""Replay a saved residual on the frozen stand, straight walk and stop body."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from residual import BASE, HERE, TURNS, baseline_policy, base_sources, load, read, stand
from push_walk import sample
from verify_native import Record, SO
from walking_bench import SENSOR_BY_LEG, STARTS, STOP_SCRIPT, W, features, legacy, load_initial, metrics


WARM = 500
TAIL = 1200
KEYS = ("time", "qpos", "qvel", "com", "loads", "nonfoot", "padz", "speed", "foot", "upright", "target", "ctrl")
LIGHT_ONSET = 2500
PUSH_ONSET = 3700
PULSE_MS = 20
PUSH_W = .2


def native_metrics(rec, trace, result, expected):
    n = rec.count
    pose = rec.poses[:n]
    forces = rec.forces[:n]
    np.testing.assert_array_equal(rec.endpoints[:len(trace["qpos"])], trace["qpos"])
    np.testing.assert_array_equal(pose[:, 7], np.repeat(np.abs(expected[:, 1]), 40)[:n])
    np.testing.assert_allclose(pose[:, 6], trace["time"][0] + np.arange(n) * 12.5e-6, atol=5e-8, rtol=0.)
    windows = {}
    partial = None
    end = WARM + result["walk_ms"]
    final = end + TAIL
    for name, a, b in (("initial_stand", 300, WARM), ("walk", WARM, end), ("terminal_stand", final-200, final)):
        if a == b:
            continue
        if n < b * 80:
            windows[name] = dict(passed=False, complete=False)
            continue
        axis = slice(a * 80, b * 80)
        native = forces[axis]
        kinematic = pose[axis]
        foot = native[:, :6]
        support = float(foot.sum() / len(native) / W)
        nonfoot_native = float(native[:, 12].max())
        nonfoot = nonfoot_native / W
        com = float(min(kinematic[:, 2].min(), trace["com"][b, 2]))
        data = dict(complete=True, samples=len(native), foot_support_mean_w=support, nonfoot_peak_w=nonfoot,
                    nonfoot_peak_native=nonfoot_native, nonfoot_exactly_zero=nonfoot_native == 0., com_min=com)
        gates = dict(support=support >= .9, com=com >= 1.2)
        if name == "walk":
            displacement = trace["com"][b, :2] - trace["com"][a, :2]
            forward = float(displacement[0] / ((b - a) * .001))
            speed = float(np.linalg.norm(displacement) / ((b - a) * .001))
            per_leg_load = foot.sum(axis=0)
            vt = native[:, 6:12].sum(axis=0) / np.maximum(1e-15, per_leg_load)
            slip = float(vt.max() / speed) if speed > 0. and np.all(per_leg_load > 0.) else None
            data.update(forward_signed=forward, slip_ratio_max=slip, absolute_contact_speed=vt.tolist(),
                        cycles=result["cycles"])
            gates.update(nonfoot=nonfoot <= .2, forward=forward >= .05,
                         slip=slip is not None and slip <= .5, cycles=all(result["cycles"].values()))
            p10 = float(np.quantile(foot.sum(axis=1)/W, .1))
            legs = float(np.median((foot >= .1 * W).sum(axis=1)))
            partial = {
                "D1": dict(passed=nonfoot_native == 0., peak_nonfoot_w=nonfoot),
                "D2": dict(status="UNKNOWN", reason="Compact native contacts lack world-force and acceleration series"),
                "D3": dict(status="UNKNOWN", reason="No same-trial auxiliary walking"),
                "D4": dict(passed=p10 >= .5, foot_w_p10=p10),
                "D4b": dict(passed=True, nonfoot_fraction=0.) if nonfoot_native == 0. else dict(status="UNKNOWN"),
                "D5": dict(status="UNKNOWN", reason="No same-trial auxiliary walking"),
                "D6": dict(status="UNKNOWN", reason="Compact native contacts lack world-force balance series"),
                "D7": dict(passed=legs >= 3, median_legs_at_point_one_w=legs),
                "D8": dict(passed=com >= .8 * 1.2072528224791905, minimum_com=com),
            }
        else:
            displacement = np.diff(trace["com"][a:b+1, :2], axis=0) / (.001 * stand.LENGTH)
            rms = float(np.sqrt(np.square(displacement).sum(axis=1).mean()))
            three = float(((foot >= .02 * W).sum(axis=1) >= 3).mean())
            upright = float(min(kinematic[:, 5].min(), trace["upright"][b]))
            data.update(rms_vel=rms, three_foot_fraction=three, min_upright_dot=upright)
            gates.update(nonfoot=nonfoot < .01, rms_vel=rms <= .01, three_legs=three >= .95,
                         posture=bool(upright >= np.cos(np.deg2rad(20.))))
        data.update(gates=gates, passed=all(gates.values()))
        windows[name] = data
    return dict(windows=windows, native_samples=n, native_force_cross_checks=rec.cross_checks,
                expected_wrench_exact=True, all_endpoints_bitwise=True,
                physical_pass=bool(n == final * 80 and all(v["passed"] for v in windows.values())),
                partial_D1_D8=partial, full_D1_D8=False)


def run(output, start, walk_ms, checkpoint, pulse, native):
    output.mkdir()
    bench = stand.Bench(output)
    body = bench.body
    body.physics_order = stand.physics.rc.SENS_IDX
    np.testing.assert_array_equal(body.physics_order, SENSOR_BY_LEG)
    initial = STARTS / f"{start}.npz"
    initial_sha = read(STARTS / "BANK.json")[start]["sha256"]
    load_initial(bench, initial, initial_sha)
    extra = [HERE / "physical.py", HERE / "residual.py", HERE / "PREREG.zh-CN.md", initial, STOP_SCRIPT]
    extra.extend(Path(name) for name in base_sources())
    if checkpoint is not None:
        extra.append(checkpoint)
    if native:
        extra.extend([SO, TURNS / "verify_native.py", TURNS / "native_contact.cpp"])
    ident = stand.identity.capture(body, dict(start=start, start_sha256=initial_sha, walk_ms=walk_ms,
                                             pulse=pulse, scope="straight body residual replay"), extra_paths=extra)
    stand.put(output / "RUN_IDENTITY.json", ident)

    begin = time.monotonic()
    end = WARM + walk_ms
    duration = end + TAIL
    policy = baseline_policy() if checkpoint is None else load(checkpoint)
    compact = Record(body, duration) if native else None
    if compact is not None:
        body.recorder = compact
    expected = np.zeros((duration * 2, 6))
    if pulse == "push":
        expected[2*PUSH_ONSET:2*(PUSH_ONSET+PULSE_MS), 1] = PUSH_W * W
    wrench, decisions = [], []
    original = body.apply_adhesion
    light = np.random.default_rng(3011).normal(0., .035, 42)
    measure = body.measure()
    trace = [(*sample(body, measure), bench.zero)]
    unsafe = 0

    def force(loads):
        original(loads)
        if np.any(body.d.xfrc_applied) or np.any(body.d.qfrc_applied):
            raise RuntimeError("Unscheduled external wrench")
        body.d.xfrc_applied[body.th] = expected[len(wrench)]
        wrench.append(body.d.xfrc_applied[body.th].copy())

    body.apply_adhesion = force
    try:
        for tick in range(duration):
            if tick % 5 == 0:
                intent = int(WARM <= tick < end)
                state = features(body, measure)
                action = legacy.bounded_action(policy(state, intent))
                decisions.append((state.copy(), intent, action.copy(), tick))
            applied = np.clip(action + light, -1., 1.) if pulse == "light" and LIGHT_ONSET <= tick < LIGHT_ONSET+PULSE_MS else action
            measure = body.tick(applied)
            np.testing.assert_array_equal(body.d.xfrc_applied[body.th], expected[len(wrench) - 1])
            if np.any(body.d.qfrc_applied) or np.any(body.d.xfrc_applied[:body.th]) or np.any(body.d.xfrc_applied[body.th+1:]):
                raise RuntimeError("Wrench applied outside the programmed thorax site")
            trace.append((*sample(body, measure), applied.copy()))
            unsafe = unsafe + 1 if measure["com"] < .8 or measure["nonfoot"] > .5 * W else 0
            if unsafe >= 100:
                break
    finally:
        body.apply_adhesion = original
        body.d.xfrc_applied[:] = 0.
    np.testing.assert_array_equal(wrench, expected[:len(wrench)])
    assert len(wrench) == 2 * (len(trace) - 1)
    trace = {name: np.asarray(v) for name, v in zip((*KEYS, "applied"), zip(*trace, strict=True), strict=True)}
    actions = {name: np.asarray(v) for name, v in zip(("x", "intent", "action", "ms"), zip(*decisions, strict=True), strict=True)}
    np.savez_compressed(output / "TRACE.npz", **trace, actual_wrench=wrench)
    np.savez_compressed(output / "DECISIONS.npz", **actions)
    result = metrics(trace, WARM, walk_ms, duration)
    result.update(start=start, walk_ms=walk_ms, pulse=pulse, unsafe_stop=bool(unsafe >= 100),
                  control="baseline" if checkpoint is None else "residual", checkpoint=str(checkpoint) if checkpoint else None,
                  checkpoint_sha256=stand.sha(checkpoint) if checkpoint else None, elapsed=time.monotonic()-begin,
                  trace_sha256=stand.sha(output / "TRACE.npz"), decision_sha256=stand.sha(output / "DECISIONS.npz"),
                  native_integrations=(len(trace["time"])-1)*80, raw_native_contacts=bool(native), full_D1_D8=False)
    result["standing_only_pass"] = None if walk_ms else (
        result["initial_stand"]["passed"] and result["terminal_stand"]["passed"] and result["actual_ms"] == duration)
    if compact is not None:
        audit = native_metrics(compact, trace, result, expected)
        np.savez_compressed(output / "NATIVE.npz", forces=compact.forces[:compact.count], pose=compact.poses[:compact.count])
        audit.update(native_arrays_sha256=stand.sha(output / "NATIVE.npz"), native_library_sha256=stand.sha(SO))
        result["native_joint_pass"] = audit["physical_pass"]
        stand.put(output / "NATIVE.json", audit)
    stand.identity.verify_runtime(ident, body)
    stand.put(output / "RESULT.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start")
    p.add_argument("output", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--walk-ms", type=int, default=9600)
    p.add_argument("--pulse", choices=("none", "light", "push"), default="none")
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    if a.walk_ms < 0 or a.walk_ms % 5:
        p.error("Walk duration must be a nonnegative multiple of 5 ms")
    if a.pulse != "none" and WARM + a.walk_ms < ((PUSH_ONSET if a.pulse == "push" else LIGHT_ONSET) + PULSE_MS):
        p.error("Pulse must be fully contained in the walking interval")
    if a.checkpoint and torch.load(a.checkpoint, weights_only=True, map_location="cpu")["archive_preflight"]:
        p.error("An archive preflight checkpoint cannot be used as a candidate")
    torch.set_num_threads(1)
    run(a.output.resolve(), a.start, a.walk_ms, a.checkpoint.resolve() if a.checkpoint else None, a.pulse, a.native)
