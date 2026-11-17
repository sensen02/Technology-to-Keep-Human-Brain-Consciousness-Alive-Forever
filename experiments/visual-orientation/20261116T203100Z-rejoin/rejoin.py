"""Physical paired continuation and the continuous detour with visual reacquisition."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

import approach
from approach_runtime import BINDING, Eyes, sha, turn
from far_fixture import prepare
from far_navigation import Detour, ProgressRecord, Pursuit
from far_optics import WEIGHTS
from geometry import Envelope
from obstacle import score as obstacle_score
from visual_rejoin import VisualRejoin

HERE = Path(__file__).resolve().parent
FAR = WEIGHTS.parent


class Rejoin:
    def __init__(self):
        self.vision = VisualRejoin()
        self.turn_frames = 0
        self.rest_frames = 0

    @property
    def phase(self):
        return self.vision.phase

    def act(self, image):
        if self.rest_frames:
            self.rest_frames -= 1
            return (0., 0), (0., 0., 0.)
        command, prediction = self.vision.act(image)
        self.turn_frames = self.turn_frames + 1 if command[1] else 0
        if self.turn_frames > 160:
            self.turn_frames = 0
            self.rest_frames = 44
            self.vision.estimate.clear()
            return (0., 0), prediction
        return command, prediction


class RecoveredDetour(Detour):
    def __init__(self):
        super().__init__(310, 1100)
        self.rejoin = Rejoin()

    def act(self, image):
        if not self.finished_detour:
            return super().act(image)
        command, prediction = self.rejoin.act(image)
        self.phase = self.rejoin.phase
        self.resume_samples += int(command[0] > 0.)
        return command, prediction


def run(out, mode):
    baseline = FAR / ("DETOUR_A" if mode in ("turn_hold", "turn_reset") else "DETOUR_B")
    full = mode == "full"
    body_box = None
    brain = None
    body_start = None
    extra = .43
    duration = 90000 if full else 6000
    files = [Path(__file__), Path(VisualRejoin.act.__code__.co_filename),
             Path(Detour.act.__code__.co_filename), Path(prepare.__code__.co_filename),
             Path(approach.__file__), Path(Envelope.path.__code__.co_filename), WEIGHTS, BINDING]
    if not full:
        receipt = turn.read(baseline / "RESULT.json")
        for name in ("TRACE.npz", "OBS.npz"):
            path = baseline / name
            assert sha(path) == receipt["files"][name]
            files.append(path)
        with np.load(baseline / "TRACE.npz", allow_pickle=False) as trace:
            pose, velocity, ctrl = (trace[key][-1].copy() for key in ("qpos", "qvel", "ctrl"))
        with np.load(baseline / "OBS.npz", allow_pickle=False) as saved:
            goal = saved["goal"].copy()

    def setup(path, start):
        nonlocal body_box, body_start
        body, motor = prepare(path, start, extra)
        if not full:
            body.d.qpos[:] = pose
            body.d.qvel[:] = velocity
            body.d.ctrl[:] = ctrl
            mj.mj_forward(body.plant, body.d)
            np.testing.assert_array_equal(body.d.qpos, pose)
            np.testing.assert_array_equal(body.d.qvel, velocity)
            assert all(body.plant.geom("navigation_obstacle").id not in c.geom
                       for c in body.d.contact[:body.d.ncon])
        body_box = Envelope(body)
        body_start = body
        return body, motor

    class FixedGoalEyes(Eyes):
        def place(self, angle, distance):
            return goal.copy()

    class TurnProbe:
        phase = "align"

        def __init__(self):
            self.frames = 0

        def act(self, image):
            self.frames += 1
            yaw = 0 if mode == "turn_reset" and 31 <= self.frames <= 87 else -1
            return (0., yaw), (0., 0., 0.)

    def policy():
        nonlocal brain
        factory = {
            "full": RecoveredDetour, "old": Pursuit, "new": Rejoin,
            "turn_hold": TurnProbe, "turn_reset": TurnProbe,
        }[mode]
        brain = factory()
        return brain

    def score(trace, decisions, goal, actual, angle, transport_mode, rec):
        result = obstacle_score(trace, decisions, goal, actual, angle, transport_mode, rec)
        assert body_box is not None and body_start is not None
        collision_free = bool(not np.any(rec.block[:rec.count]))
        yaw = np.rad2deg(np.unwrap(trace["yaw"]))
        position = trace["com"][:, :2]
        forward = decisions[decisions[:, 3] > .06, 0].astype(int)
        delta = position[-1] - position[500]
        initial_yaw = trace["yaw"][500]
        signed_forward = float(delta @ [np.cos(initial_yaw), np.sin(initial_yaw)])
        checks = dict(native_obstacle_force_zero=collision_free, no_nonfoot_floor=result["nonfoot_native"] == 0.,
                      min_com=float(trace["com"][:, 2].min()) >= 1.2)
        if full:
            survey = body_box.path(trace["qpos"], 545, 545+6200, stride=5)
            terminal = body_box.path(trace["qpos"], actual-200, actual, stride=5)
            result["initial_pivot_pad_clearance"] = float(body_box.near - survey[:, 4].max())
            result["full_body_beyond_box"] = float(terminal[:, 1].min() - body_box.far)
            result["max_lateral"] = float((position[:, 1]-position[0, 1]).max())
            checks.update(visual_trigger=brain.hazard_frames == 2,
                          initial_turn=yaw[6745]-yaw[545] >= 25.,
                          initial_pad_clearance=result["initial_pivot_pad_clearance"] >= .10,
                          lateral=result["max_lateral"] >= .45,
                          full_body_beyond_box=result["full_body_beyond_box"] >= .03,
                          rejoined=brain.resume_samples >= 50,
                          visual_stop=brain.phase == "done" and 1.18 < result["final_range"] < 1.72,
                          stable=result["standing"]["passed"])
        elif mode in ("new", "old"):
            checks.update({
                "rejoined": len(forward) >= 200 and signed_forward >= .12
            } if mode == "new" else {
                "does_not_resume": len(forward) == 0 and float(np.linalg.norm(delta)) < .01
            })
        else:
            turn_delta = float(yaw[-1] - yaw[2400])
            checks["turn_response"] = turn_delta <= -3 if mode == "turn_reset" else abs(turn_delta) < .3
        np.savez_compressed(out / "BLOCK_FORCE.npz", force=rec.block[:rec.count])
        result.update(mode_label=mode, spec=dict(blind=False), checks_rejoin={k:bool(v) for k,v in checks.items()},
                      signed_forward=signed_forward, yaw_deg=float(yaw[-1]-yaw[500]),
                      yaw_after_2400_deg=float(yaw[-1]-yaw[2400]), forward_ms=len(forward)*5,
                      first_forward_ms=int(forward[0]) if forward.size else None,
                      minimum_com=float(trace["com"][:, 2].min()), passed=all(checks.values()))
        return result

    identity = {str(p): sha(p) for p in files}
    approach.prepare, approach.Policy, approach.Record, approach.score = setup, policy, ProgressRecord, score
    approach.STOP = 1.46
    if not full:
        approach.Eyes = FixedGoalEyes
    answer = approach.trial(out.resolve(), 0., 6., "intact", .12, duration)
    assert identity == {str(p): sha(p) for p in files}
    turn.stand.put(out / "AUDIT_IDENTITY.json", dict(inputs=identity, physics_step_us=12.5, mode=mode,
                   initialization="original nominal" if full else "archived qpos/qvel/ctrl, fresh runtime",
                   native_obstacle_sha256=sha(out / "BLOCK_FORCE.npz")))
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("mode", choices=("old", "new", "full", "turn_hold", "turn_reset"))
    args = parser.parse_args()
    result = run(args.out, args.mode)
    raise SystemExit(0 if result["passed"] else 1)
