"""Visually triggered far-box detour over the existing scoped DN pathways."""
import argparse
from pathlib import Path

import numpy as np
import torch

import approach
from approach_runtime import turn, sha
from approach_vision import network
from far_fixture import BODY, CollisionRecord, LIB, prepare
from far_optics import WEIGHTS
from geometry import Envelope
from obstacle import score as obstacle_score
from far_preflight import read_green


class Pursuit(approach.Policy):
    def __init__(self):
        super().__init__()
        self.model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))


class ProgressRecord(CollisionRecord):
    def solved(self, body, before):
        super().solved(body, before)
        if self.count and self.count % 400_000 == 0:
            root = body.d.qpos[:3]
            latest = self.block[self.count - 399999:self.count + 1]
            print(f"physical {self.count / 80_000:.1f} s root={root} box_peak={latest.max():.5g}", flush=True)


class Detour(Pursuit):
    def __init__(self, turn_frames, walk_frames):
        super().__init__()
        self.turn_frames = turn_frames
        self.walk_frames = walk_frames
        self.hazard_frames = 0
        self.elapsed = None
        self.finished_detour = False
        self.resume_samples = 0

    def act(self, image):
        if self.finished_detour:
            nominal, estimate = super().act(image)
            if estimate[2] < 0. and self.phase == "align":
                return (0., -1), estimate
            if nominal[0] > 0.:
                self.resume_samples += 1
            return nominal, estimate
        if self.elapsed is None:
            nominal, estimate = super().act(image)
            o = read_green(image)
            if estimate[2] > 0. and int(o.sum()) >= 6:
                self.hazard_frames += 1
            else:
                self.hazard_frames = 0
            if self.hazard_frames < 2:
                return nominal, estimate
            self.elapsed = 0
        tick = self.elapsed
        self.elapsed += 1
        if tick < self.turn_frames:
            return (0., 1), (0., 0., 0.)
        if tick < self.turn_frames + 65:
            return (0., 0), (0., 0., 0.)
        if tick < self.turn_frames + 65 + self.walk_frames:
            return (.0955, 0), (0., 0., 0.)
        if tick < self.turn_frames + 125 + self.walk_frames:
            return (0., 0), (0., 0., 0.)
        self.finished_detour = True
        self.phase = "align"
        self.direction = 0
        self.near = False
        self.history.clear()
        return self.act(image)


def execute(a):
    box = None
    agent = None
    out = a.out.resolve()

    def setup(path, start):
        nonlocal box
        prepared = prepare(path, start, a.extra)
        box = Envelope(prepared[0])
        return prepared

    def policy():
        nonlocal agent
        agent = Pursuit() if a.blind else Detour(a.turn_frames, a.walk_frames)
        return agent

    def score(trace, decisions, goal, actual, angle, mode, record):
        assert agent is not None and box is not None
        base = obstacle_score(trace, decisions, goal, actual, angle, mode, record)
        force = record.block[:record.count]
        np.savez_compressed(out / "BLOCK_FORCE.npz", force=force)
        terminal = box.path(trace["qpos"], actual-200, actual)
        arm = decisions[decisions[:,4] > 0, 0].astype(int)
        lead_margin = None
        survey = None
        yaw_gain = None
        room = None
        lateral = float((trace["com"][:,1] - trace["com"][0,1]).max())
        if len(arm):
            start = int(arm[0])
            finish = start + 20*a.turn_frames
            survey = box.path(trace["qpos"], start, min(finish, actual), stride=5)
            room = float(box.near - survey[:, 4].max())
            lead_margin = float(box.near - survey[:, 2].max())
            yaw_gain = float(np.rad2deg(trace["yaw"][min(finish, actual)] - trace["yaw"][start]))
        terminal_room = float(terminal[:,1].min() - box.far)
        if a.blind:
            checks = {"unavoided_box_physically_blocks": base["impact_count_native"] > 0}
        else:
            checks = dict(
                visual_triggers=agent.hazard_frames == 2,
                enough_initial_turn=yaw_gain is not None and yaw_gain >= 25.,
                pads_never_enter_box_x_at_initial_turn=room is not None and room >= .10,
                sidesteps=lateral >= .45,
                clears_box_with_all_geometry=terminal_room >= .03,
                obstacle_no_native_force=base["impact_count_native"] == 0,
                resumes_visual_pursuit=agent.resume_samples >= 50,
                stop_by_visual_range=1.18 < base["final_range"] < 1.72,
                no_floor_nonfoot=base["nonfoot_native"] == 0.,
                com_safe=float(trace["com"][:, 2].min()) >= 1.2,
                stable=bool(base["standing"]["passed"]),
            )
        if survey is not None:
            np.savez_compressed(out / "GEOMETRY.npz", initial_pivot=survey, terminal=terminal)
        base.update(passed=all(checks.values()), detour_checks=checks, full_body_clearance_x=terminal_room,
                    initial_pivot_pad_clearance_x=room, initial_pivot_body_clearance_x=lead_margin,
                    initial_pivot_deg=yaw_gain, max_lateral_native=lateral,
                    resumes_visual_samples=getattr(agent,"resume_samples",None),
                    supplemental_hashes={"BLOCK_FORCE.npz": sha(out / "BLOCK_FORCE.npz")},
                    spec=dict(box_shift=a.extra, turn_frames=a.turn_frames, walk_frames=a.walk_frames,
                              max_duration_ms=a.duration, target_distance=a.target, blind=a.blind))
        return base

    approach.prepare = setup
    approach.Policy = policy
    approach.Record = ProgressRecord
    approach.score = score
    approach.STOP = 1.46
    result = approach.trial(out, 0., a.target, "intact", .12, a.duration)
    result["inputs"].update({str(p):sha(p) for p in (Path(__file__).resolve(),WEIGHTS,LIB,
        WEIGHTS.parent / "far_fixture.py", WEIGHTS.parent / "geometry.py",
        WEIGHTS.parent / "far_preflight.py", WEIGHTS.parent / "far_optics.py")})
    if (out / "GEOMETRY.npz").is_file():
        result["supplemental_hashes"]["GEOMETRY.npz"] = sha(out / "GEOMETRY.npz")
    turn.stand.put(out / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--extra", type=float, default=.43)
    parser.add_argument("--duration", type=int, default=92000)
    parser.add_argument("--target", type=float, default=6.00)
    parser.add_argument("--turn-frames", type=int, default=440)
    parser.add_argument("--walk-frames", type=int, default=1210)
    parser.add_argument("--blind", action="store_true")
    result = execute(parser.parse_args())
    raise SystemExit(0 if result["passed"] else 1)
