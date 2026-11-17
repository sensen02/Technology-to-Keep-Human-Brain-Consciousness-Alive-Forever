"""Fresh-process physical or saved-state replay for the body-only bench."""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from policy import BodyPolicy, FORMAT, LEGS
from train import HERE, OLD_MOTOR, MOTION, put, read, sha

TASKS = {
    "walk": ((9.6, (1., 0.)),),
    "stop": ((9.6, (0., 0.)),),
    "left_pivot": ((9.6, (0., 1.)),),
    "right_pivot": ((9.6, (0., -1.)),),
    "transition": ((3.2, (0., 0.)), (4.8, (1., 0.)), (3.2, (0., 0.)), (4.8, (1., 0.))),
}


def load(checkpoint):
    fit = read(checkpoint.parent / "FIT.json")
    if checkpoint.name != fit["checkpoint"] or sha(checkpoint) != fit["sha256"]:
        raise ValueError("Expected exact final checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT or payload["step"] != fit["updates"]:
        raise ValueError("Unexpected checkpoint format/step")
    if not all(sha(Path(p)) == digest for p, digest in payload["source_hashes"].items()):
        raise ValueError("Checkpoint source/data identity drift")
    state = payload["model"]
    policy = BodyPolicy(state["mean"], state["scale"]).double()
    policy.load_state_dict(state)
    return policy.eval(), fit["sha256"]


def yaw(body):
    rot = body.d.xmat[body.th].reshape(3, 3)
    return math.atan2(rot[1, 0], rot[0, 0])


def score(rows, start, length, task, physics, walking, pivot, headings):
    p0, yaw0 = start
    sequence = [(r["time"], r["com"], r["loads"], r["nonfoot"], r["speed"]) for r in rows]
    original = physics.summarise(sequence, len(rows) * 80)
    pos = np.vstack([p0] + [r["com"] for r in rows])
    delta = (pos[-1] - pos[0])[:2]
    dist = np.linalg.norm(pos[1:, :2] - p0[:2], axis=1)
    relative = pos[1:, :2] - p0[:2]
    initial_axis = np.array([math.cos(yaw0), math.sin(yaw0)])
    dyaw = math.degrees(np.unwrap(np.r_[yaw0, headings])[-1] - yaw0)
    speed = np.diff(pos[:, :2], axis=0) / .001 / length
    stop_rms = float(np.sqrt(np.square(speed[min(500, len(speed) - 1):]).sum(axis=1).mean()))
    cycles = walking.CycleTracker(walking.CycleConfig(timeout=4.)).reset(start_time=rows[0]["time"] - .001)
    pivot_cycles = pivot.PivotCycles(physics.W)
    events = []
    for r in rows:
        events.extend(cycles.update(dict(time=r["time"], loads=r["loads"], padz=r["padz"],
                                      relativefoot=r["relativefoot"]))["events"])
        pivot_cycles.update(r["time"], r["loads"], r["padz"])
    counts = {leg: sum(e["type"] == "complete" and e["leg"] == leg for e in events) for leg in LEGS}
    gates = dict(support=original["gates"]["support"], com=original["gates"]["com"],
                 nonfoot_zero=original["nonfoot_force_max_native"] == 0.)
    drift = float(dist.max() / length)
    if task == (1., 0.):
        gates.update(forward=original["gates"]["forward"], slip=original["gates"]["slip"],
                     six_cycles=all(counts.values()))
    elif task == (0., 0.):
        gates.update(drift=drift <= .1, yaw=abs(dyaw) <= 5., steady_speed=stop_rms <= .01)
    else:
        gates.update(drift=drift <= .5, signed_yaw=dyaw * task[1] >= 45.,
                     six_cycles=all(pivot_cycles.counts))
    return dict(command=task, actual_ms=len(rows), original_five=original, yaw_change_deg=dyaw,
                drift_body_lengths=drift, velocity_rms_after_500ms_body_lengths_s=stop_rms,
                signed_heading_progress_native=float(delta @ initial_axis),
                final_xy_native=relative[-1].tolist(), walk_cycles=counts,
                pivot_cycles=list(pivot_cycles.counts), sampled_gates=gates)


@torch.no_grad()
def run(policy, digest, task, output):
    sys.path.insert(0, str(OLD_MOTOR))
    sys.path.insert(0, str(MOTION / "pivot_teacher_design"))
    import common as pivot
    import teacher as pivot_teacher
    import physics
    import identity
    import student_wave as walking

    c = physics.Core(adhesion=False)
    if tuple(physics.LEGS) != LEGS or tuple(physics.rc.SENSORY_ORDER) != ("LF", "LM", "LH", "RF", "RM", "RH"):
        raise RuntimeError("Sensor/actuator leg order changed")
    names = c.actuator.state_dict()["joint_names"]
    if any(not name.startswith("joint_" + leg) for leg, group in zip(LEGS, np.reshape(names, (6, 7))) for name in group):
        raise RuntimeError("Compiled actuator ordering differs")
    manifest = identity.capture(c, dict(scope="body bench, no brain/neck deployment", task=task,
                                        policy_sha256=digest, leg_order=LEGS,
                                        proprio_load_order=physics.rc.SENSORY_ORDER))
    put(output / "IDENTITY.json", manifest)
    trace = walking.Trace(full=True)
    walking.settle(c, trace)
    length = pivot_teacher.body_length_from_compiled_mesh(c)["length_native"]
    previous = np.zeros(42)
    state = None
    arrays = []
    segments = []
    starts = []
    started = time.monotonic()
    teacher = walking.StaticTeacher(c, walking.teacher_config(3)) if task != "transition" else None
    spans = (((4.8, (1., 0.)),) if teacher is not None else ()) + TASKS[task]
    unsafe = 0
    stopped = False
    teacher_updates = 0

    for segment, (duration, command) in enumerate(spans):
        prime = teacher is not None and segment == 0
        n = round(duration / .005)
        begin = len(trace.rows)
        starts.append(begin)
        start = (c.d.subtree_com[c.th].copy(), yaw(c))
        headings = []
        for k in range(n):
            x = c.obs238(previous, target_speed=.075)
            action, state = policy.step(torch.from_numpy(x), torch.tensor(command, dtype=torch.float64), state)
            value = action.numpy().copy()
            if prime:
                previous = walking.bounded_action(teacher.action(c, k * .005)[0])
                teacher_updates += 1
            else:
                previous = walking.bounded_action(value)
            arrays.append((x.copy(), command, value, previous.copy(), prime))
            for _ in range(5):
                m = trace.tick(c, previous, 2 if prime else 3)
                headings.append(yaw(c))
                unsafe = unsafe + 1 if m["com"] < .8 or m["nonfoot"] > .5 * physics.W else 0
                if unsafe == 100:
                    stopped = True
                    break
            if stopped:
                break
        rows = trace.rows[begin:]
        result = score(rows, start, length, command, physics, walking, pivot, headings) if len(rows) > 1 else {}
        result.update(expected_ms=n * 5, full_window=len(rows) == n * 5, teacher=prime,
                      end_time_s=float(c.d.time))
        segments.append(result)
        print(task, segment, len(rows), result.get("sampled_gates"), flush=True)
        if stopped:
            break
    identity.verify_runtime(manifest, c)
    trace_proof = trace.save(output / "trace_1ms.npz")
    x, commands, actions, applied, priming = (np.asarray(a) for a in zip(*arrays))
    decisions = output / "decisions.npz"
    np.savez_compressed(decisions, x=x, commands=commands, actions=actions, applied=applied, priming=priming)
    put(output / "RESULT.json", dict(task=task, policy_sha256=digest, segments=segments,
        segment_trace_starts=starts, unsafe_stop=stopped, teacher_updates=teacher_updates,
        teacher_after_handover=0, actual_native_integrations=len(trace.rows) * 80,
        strict_native_contact_audit=False, original_D1_D8_qualification=False,
        decision_sha256=sha(decisions), trace=trace_proof, wall_s=time.monotonic() - started))


@torch.no_grad()
def replay(policy, digest, output):
    result = read(output / "RESULT.json")
    assert result["policy_sha256"] == digest
    for p, h in ((output / "decisions.npz", result["decision_sha256"]),
                 (output / "trace_1ms.npz", result["trace"]["sha256"])):
        if sha(p) != h:
            raise ValueError(f"Artifact drift {p}")
    with np.load(output / "decisions.npz", allow_pickle=False) as z:
        x, commands, expected, applied, prime = (z[key] for key in ("x", "commands", "actions", "applied", "priming"))
    batched, _ = policy.sequence(torch.from_numpy(x[:, None]), torch.from_numpy(commands[:, None]))
    error = float(np.max(np.abs(expected - batched[:, 0].numpy())))
    if error > 1e-10 or not np.array_equal(expected[~prime], applied[~prime]):
        raise RuntimeError("Saved neural/action replay failed")
    replayed_steps = 0
    with np.load(output / "trace_1ms.npz", allow_pickle=False) as z:
        tick_actions, source = z["actual_requested_action"], z["control_source"]
        if not np.array_equal(tick_actions[1000:], np.repeat(applied, 5, axis=0)[:len(source)-1000]):
            raise RuntimeError("Actual physical actions differ from decisions")
        if np.any(z["external_body_force_max"]) or np.any(z["external_generalized_force_max"]):
            raise RuntimeError("Nonzero external force")
        replayed_steps = len(source)
    put(output / "REPLAY.json", dict(verified=True, max_action_error=error, actual_1ms_rows=replayed_steps,
                                    policy_sha256=digest, physical_integrations=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("task", choices=TASKS)
    p.add_argument("output", type=Path)
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    policy, digest = load(a.checkpoint)
    if a.replay:
        replay(policy, digest, a.output)
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        run(policy, digest, a.task, a.output)


if __name__ == "__main__":
    main()
