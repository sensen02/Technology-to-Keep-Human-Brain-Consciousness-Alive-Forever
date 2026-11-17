"""Collect teacher demonstrations and evaluate the same physical visual loop."""
import argparse
from pathlib import Path

import numpy as np

from bench_runtime import HERE, arrays, check_forces, prepare, sample, stand_window, turn
from brain import VisualPolicy
from neck_runtime import BINDING, Neck
from optics import Eyes

RESTORE_DELAY = 760

def score(case, trace, commands, bearings, first, finish):
    mode = case["test"]
    yaw_trace = np.rad2deg(np.unwrap(trace["yaw"]))
    time = commands[:, 0].astype(int)
    selected = commands[:, 2].astype(int)
    delivered = commands[:, 3].astype(int)
    mover = abs(selected)
    actual = abs(delivered)
    before = stand_window(trace, first)
    tail = stand_window(trace, finish)
    no_nonfoot = not trace["nonfoot"].any()
    nonzero = np.flatnonzero((time >= first) & (mover > 0))
    driven = np.flatnonzero((time >= first) & (actual > 0))
    first_direction = int(selected[nonzero[0]]) if len(nonzero) else 0
    first_x = yaw_trace[first]
    final = float(yaw_trace[-1] - first_x)
    relevant = first
    direction = int(np.sign(case["angle"]))
    if mode == "switch":
        relevant = case["event_ms"]
        direction = int(np.sign(case["new_angle"]))
    if mode == "restore":
        relevant = first + RESTORE_DELAY
    delta = float(yaw_trace[-1] - yaw_trace[relevant])
    error = float(np.max(np.abs(bearings[-201:])))
    required = max(0., abs(bearings[relevant]) - 10.)
    last_drive = int(time[driven[-1]]) + 5 if len(driven) else first
    end_stable = next((i for i in range(max(last_drive+200, relevant+200), finish+1, 5)
                       if stand_window(trace, i)["passed"]), None)
    dwell = None if end_stable is None else end_stable - last_drive
    base_gates = bool(before["passed"] and tail["passed"] and no_nonfoot)
    if mode in {"off", "center", "occluded"}:
        passed = base_gates and not len(nonzero) and abs(final) <= .5
    elif mode == "cut":
        passed = base_gates and first_direction == direction and not len(driven) and abs(final) <= .5
    elif mode == "remove":
        gone = case["event_ms"]
        passed = base_gates and first_direction == direction and not actual[time >= gone+100].any()
        passed = bool(passed and dwell is not None and dwell <= 1200)
    else:
        passed = bool(base_gates and first_direction == int(np.sign(case["angle"]))
                      and error <= 10. and direction*delta >= required
                      and dwell is not None and dwell <= 1200
                      and finish <= first + 9600 + (RESTORE_DELAY if mode == "restore" else 0))
    return dict(passed=bool(passed), first_direction=first_direction, final_body_delta_deg=final,
                goal_window_body_delta_deg=delta, required_body_delta_deg=float(required),
                final_error_bound_deg=error, initial_stand=before, terminal_stand=tail,
                first_intent_ms=int(time[nonzero[0]]) if len(nonzero) else None,
                first_delivery_action_ms=int(time[driven[0]]) if len(driven) else None,
                last_delivery_action_end_ms=last_drive, first_qualified_stand_ms=end_stable,
                settle_after_withdraw_ms=dwell, nonfoot_peak=float(trace["nonfoot"].max()),
                drift_bl=float(np.max(np.linalg.norm(trace["com"][first:,:2]-trace["com"][first,:2],axis=1))/turn.stand.LENGTH))


def trial(out, case, checkpoint=None, include_native=False, codec=None, profile="clean", seed=0, teacher_v2=False):
    body, motor = prepare(out, case["start"])
    eyes = Eyes(body)
    if codec is None:
        neck = Neck()
    else:
        from trained_link import TrainedLink
        neck = TrainedLink(codec, profile, seed)
    brain = VisualPolicy(checkpoint) if checkpoint is not None else None
    onset = case["onset_ms"]
    finish = onset + 9600 + (RESTORE_DELAY if case["test"] == "restore" else 0)
    if "duration_ms" in case:
        finish = onset + case["duration_ms"]
    mode = case["test"]
    event = case.get("event_ms", -1)
    if mode in ("cut", "restore"):
        assert onset >= 500
    origin = eyes.pose()[0].copy()
    background = (origin, case["background"], case["gray"])
    recorder = None
    if include_native:
        from verify_native import Record
        recorder = Record(body, finish)
        body.recorder = recorder
    measured = body.measure()
    frames, labels, commands, bearing_trace = [], [], [], []
    body_observations, body_actions = [], []
    goals = []
    rows = [sample(body, measured)]
    goal = None
    cover = None
    previous, yaw, est, presence, capture = 0, 0, 0., 0., -1
    stop_at = None
    last_active = 0
    released_labels = None
    pending_labels = None
    try:
        for tick in range(finish):
            if tick == onset and mode != "off":
                goal = eyes.place(case["angle"], case["distance"])
                if mode == "occluded":
                    eye = eyes.pose()[0].copy()
                    ray = goal[:2] - eye[:2]
                    ray /= np.linalg.norm(ray)
                    center = (goal + eye) / 2.
                    center[2] = 1.4
                    cover = (center, np.array([[ray[0], -ray[1], 0.], [ray[1], ray[0], 0.], [0., 0., 1.]]))
            if tick == event:
                if mode == "switch":
                    goal = eyes.place(case["new_angle"], case["distance"])
                if mode == "remove":
                    goal = None
            if tick == 0 and mode in ("cut", "restore"):
                neck.switch(True)
            if tick == onset + RESTORE_DELAY and mode == "restore":
                neck.switch(False)
            if tick % 20 == 0:
                exposed = eyes.capture(tick, goal, case["width"], case["color"], background, cover)
                released_labels = pending_labels
                visible = goal is not None and cover is None
                pending_labels = (visible, eyes.bearing(goal) if visible else 0.)
                if exposed is not None:
                    capture, frame = exposed
                    seen, label = released_labels
                    current = eyes.bearing(goal) if goal is not None and seen else 0.
                    intended = teacher(current, seen, previous, corrected=teacher_v2)
                    yaw, est, presence = (intended, current, float(seen)) if brain is None else brain.action(frame)
                    frames.append(frame)
                    labels.append((capture, int(seen), label, previous, intended, yaw))
                    previous = yaw
                neck.send(yaw)
            if tick % 5 == 0:
                body_yaw = neck.action()
                if body_yaw:
                    last_active = tick+5
                observation = turn.features(body, measured)
                action = motor(observation, 0, body_yaw)
                if codec is not None:
                    body_observations.append(observation)
                    body_actions.append(action.copy())
                commands.append((tick, capture, yaw, body_yaw, neck.delivered_step if neck.delivered_step is not None else -1,
                                 est, presence, *neck.latest))
            bearing_trace.append(eyes.bearing(goal) if goal is not None else 0.)
            goals.append(goal.copy() if goal is not None else [np.nan]*3)
            measured = body.tick(action)
            check_forces(body)
            rows.append(sample(body, measured))
            neck.advance()
            elapsed = tick+1
            minimum = onset+1800 if mode not in ("off", "center", "occluded", "cut") else finish
            has_event = event >= 0 and elapsed < event+1500
            # Freeze successful outcomes only after a full release and stable observation window.
            if elapsed % 20 == 0 and elapsed >= minimum and not has_event and last_active > onset and elapsed >= last_active+1200:
                current_error = abs(eyes.bearing(goal)) if goal is not None else 0.
                if current_error <= 10. and yaw == 0:
                    stop_at = elapsed
                    break
    finally:
        eyes.close()
    actual = stop_at if stop_at else finish
    assert len(rows) == actual+1
    bearing_trace.append(eyes.bearing(goal) if goal is not None else 0.)
    goals.append(goal.copy() if goal is not None else [np.nan]*3)
    trace = arrays(rows)
    commands = np.asarray(commands)
    result = score(case, trace, commands, np.array(bearing_trace), onset, actual)
    optical = out / "EYES.npz"
    motion = out / "TRACE.npz"
    np.savez_compressed(optical, rgb=np.asarray(frames), labels=np.asarray(labels), commands=commands,
                        bearing=bearing_trace, goal_world=goals)
    np.savez_compressed(motion, **trace)
    qualify = neck.transport.verify_final()
    assert len(commands) == (actual+4)//5 and qualify["actual_steps"] == actual
    for row in commands:
        if row[1] >= 0:
            assert row[1]+20 <= row[0]
        if row[4] >= 0:
            assert row[4] <= row[0]
    turn.stand.put(out / "TRANSPORT.json", neck.transport.journal())
    if codec is not None:
        steps = np.asarray([entry[0] for entry in neck.measurements], dtype=np.int64)
        classes = np.asarray([entry[1] if entry[1] is not None else -2 for entry in neck.measurements], dtype=np.int8)
        samples = np.asarray([entry[2] for entry in neck.measurements if entry[2] is not None], dtype=np.float32)
        receiver = out / "RECEIVER.npz"
        np.savez_compressed(receiver, release_steps=steps, decoded=classes, measurements=samples,
                            observations=np.asarray(body_observations), actions=np.asarray(body_actions))
        result.update(graph_sha256=neck.graph_sha, codec_sha256=turn.stand.sha(codec), profile=profile,
                      receiver_seed=seed, receiver_sha256=turn.stand.sha(receiver), receiver_dropped=int((classes==-2).sum()))
    if recorder is not None:
        assert recorder.count == actual*80
        np.testing.assert_array_equal(recorder.endpoints[:actual+1], trace["qpos"])
        result["native_nonfoot_peak"] = float(recorder.forces[:actual*80,12].max())
        result["native_checks"] = recorder.cross_checks
        result["passed"] = bool(result["passed"] and result["native_nonfoot_peak"] == 0.)
    result.update(case=case, checkpoint_sha256=turn.stand.sha(checkpoint) if checkpoint else None,
                  body_native_steps=actual*80, setup_native_steps=80000,
                  binding_sha256=turn.stand.sha(BINDING), actual_ms=actual,
                  eyes_sha256=turn.stand.sha(optical), trace_sha256=turn.stand.sha(motion),
                  transport_sha256=turn.stand.sha(out / "TRANSPORT.json"))
    turn.stand.put(out / "RESULT.json", result)
    return result


def teacher(error, visible, previous, corrected=False):
    if not visible:
        return 0
    if corrected and previous != 0:
        return previous if previous * error > -4. else 0
    threshold = 10.5 if previous == 0 else 3.0
    if abs(error) <= threshold:
        return 0
    return 1 if error > 0 else -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--profile", default="clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-v2", action="store_true")
    args = parser.parse_args()
    print(trial(args.output.resolve(), turn.read(args.cases)[args.name], args.checkpoint, args.native,
                args.codec, args.profile, args.seed, args.teacher_v2), flush=True)
