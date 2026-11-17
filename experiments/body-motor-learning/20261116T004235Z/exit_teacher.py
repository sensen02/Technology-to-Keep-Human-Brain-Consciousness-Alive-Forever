"""Record a walking continuation from states actually visited by the turn policy."""
import argparse
from pathlib import Path

import numpy as np
import torch

import turning as turn
from tarsus_clearance import add_clearance
from verify_native import Record


class StraightFromArc(turn.ArcReference):
    def __init__(self, body, teacher, condition, order, speed):
        super().__init__(body, teacher, condition, order, speed, .115)
        self.omega = 0.

    def center(self, t):
        distance = self.speed * turn.fixed.old.ramp_integral(t)
        return self.origin + np.r_[distance * self.heading, 0.]


def teacher_at_exit(body, mode, speed):
    teacher = turn.make_teacher(body, mode, .13, .115)
    old = teacher.reference
    teacher.reference = StraightFromArc(body, teacher, old.condition, old.order, speed)
    add_clearance(teacher, body, .028)
    return teacher


def continuation_metrics(record, trace, duration_ms):
    index = duration_ms
    contacts = record.forces[:index * 80]
    delta = trace["com"][index, :2] - trace["com"][0, :2]
    yaw = np.unwrap(trace["yaw"])
    ahead = np.array([np.cos(yaw[0]), np.sin(yaw[0])])
    speed = np.linalg.norm(delta) / (duration_ms * .001)
    slip = (contacts[:, 6:12].sum(0) / np.maximum(contacts[:, :6].sum(0), 1e-12)).max()
    summary = dict(
        forward=float(delta @ ahead / (duration_ms * .001)),
        yaw_change_deg=float(np.rad2deg(yaw[index] - yaw[0])),
        com_min=float(min(record.poses[:index*80, 2].min(), trace["com"][index, 2])),
        support=float(contacts[:, :6].sum() / (len(contacts) * turn.W)),
        nonfoot=float(contacts[:, 12].max()),
        slip=float(slip / speed) if speed > 0. else None,
    )
    summary["pass"] = bool(summary["forward"] >= .05 and abs(summary["yaw_change_deg"]) <= 10
                          and summary["com_min"] >= 1.2 and summary["support"] >= .9
                          and summary["nonfoot"] == 0. and summary["slip"] is not None
                          and summary["slip"] <= .5)
    return summary


def run(path, mode, start, speed):
    path.mkdir()
    bench = turn.stand.Bench(path)
    body = bench.body
    body.physics_order = turn.stand.physics.rc.SENS_IDX
    snapshots = turn.read(turn.STARTS / "BANK.json")
    turn.load_initial(bench, turn.STARTS / f"{start}.npz", snapshots[start]["sha256"])
    straight = turn.straight_policy()
    arc = turn.load_proto(mode)
    measure = body.measure()
    begin = turn.WARM + turn.COAST
    handoff = begin + turn.TURN
    for tick in range(handoff):
        if tick % 5 == 0:
            x = turn.features(body, measure)
            action = straight(x, int(tick >= turn.WARM)) if tick < begin else arc(x)
        measure = body.tick(action)
        assert not np.any(body.d.xfrc_applied) and not np.any(body.d.qfrc_applied)
    sample0 = turn.sample(body, measure)
    prior = turn.HERE / f"NATIVE_{mode}" / "TRACE.npz"
    if start == "nominal":
        with np.load(prior, allow_pickle=False) as trace:
            for key in ("qpos", "qvel", "target", "com", "yaw"):
                np.testing.assert_array_equal(trace[key][handoff], sample0[key])
    record = Record(body, turn.COAST + turn.STOP)
    body.recorder = record
    teacher = teacher_at_exit(body, mode, speed)
    samples = [sample0]
    observations, actions = [], []
    for tick in range(turn.COAST + turn.STOP):
        if tick % 5 == 0:
            x = turn.features(body, measure)
            if tick < turn.COAST:
                action = teacher.action(body, tick * .001)[0]
                observations.append(x)
                actions.append(action.copy())
            else:
                action = straight(x, 0)
        measure = body.tick(action)
        samples.append(turn.sample(body, measure))
    trace = {key: np.asarray([x[key] for x in samples]) for key in sample0}
    assert record.count == (turn.COAST + turn.STOP) * 80
    np.testing.assert_array_equal(record.endpoints, trace["qpos"])
    assert not record.poses[:, 7].any()
    phase = continuation_metrics(record, trace, turn.COAST)
    tail = turn.standing(trace, turn.COAST, turn.COAST + turn.STOP)
    np.savez_compressed(path / "CONTINUATION.npz", **trace)
    np.savez_compressed(path / "IMITATION.npz", x=observations, actions=actions)
    result = dict(mode=mode, start=start, teacher_speed=speed, continuation=phase,
                  tail=tail, joint_pass=phase["pass"] and tail["passed"],
                  prefix_bitwise=start == "nominal", native_checks=record.cross_checks,
                  native_steps=(handoff + turn.COAST + turn.STOP) * 80, setup_steps=80000)
    turn.stand.put(path / "RESULT.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("arc_left", "arc_right"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", default="nominal")
    parser.add_argument("--speed", type=float, default=.09)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output.resolve(), args.mode, args.start, args.speed)
