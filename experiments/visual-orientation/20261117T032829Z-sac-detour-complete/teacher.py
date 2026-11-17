"""Collect qualified single-box trajectories through the archived navigation plant."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "20261117T022700Z-sac-startup-replay"
sys.path.insert(0, str(SOURCE))
import course


def angle_error(yaw, origin, target):
    dx, dy = target[:2] - origin[:2]
    return (float(np.arctan2(dy, dx)) - yaw + np.pi) % (2.*np.pi) - np.pi


class DetourTeacher:
    """A privileged data-collection policy. Its geometry never enters observations."""

    def __init__(self, scene, rules):
        self.prefix = course.Teacher("detour", scene)
        self.rules = rules
        self.stage = "outbound"
        self.turn_ticks = 0
        self.quiet_ticks = 0
        self.reached_heading = False
        self.target_y = None
        self.env = None
        self.rows = []

    def turn_or_walk(self, error, pace):
        # During pivot, use the tighter threshold. During marching, tolerate
        # a small heading error so foot stance can settle.
        align = self.rules["align_deg"] if not self.reached_heading else self.rules["correct_deg"]
        if self.quiet_ticks:
            self.quiet_ticks -= 1
            return 0, .5
        if abs(error) <= np.radians(align):
            if self.turn_ticks:
                self.turn_ticks = 0
                self.quiet_ticks = self.rules["settle_ticks"] - 1
                self.reached_heading = True
                return 0, .5
            self.reached_heading = True
            return 1, pace
        self.reached_heading = False
        if self.turn_ticks >= self.rules["pivot_limit"]:
            self.turn_ticks = 0
            self.quiet_ticks = self.rules["pivot_break_ticks"] - 1
            return 0, .5
        self.turn_ticks += 1
        return (2 if error > 0. else 3), .5

    def choose(self, env, obs):
        self.env = env
        xyz, yaw = env.pose()
        tail, _ = env.footprint.bounds(env.body.d.qpos)
        margin = tail - env.box_edge
        head = env.eyes.pose()[0]
        self.rows.append((self.stage, *xyz, yaw, *head[:2], margin, float(env.measured["nonfoot"] / course.W)))
        original = self.prefix.detour
        if self.stage == "outbound":
            if original.elapsed is None or original.elapsed < original.turn_frames + original.walk_frames + 125:
                return self.prefix.choose(env, obs)
            self.stage = "alongside"
            self.target_y = xyz[1]
        if self.stage == "alongside":
            if margin < self.rules["turn_clearance"]:
                waypoint = np.array([xyz[0] + self.rules["side_lookahead"], self.target_y])
                return self.turn_or_walk(angle_error(yaw, xyz, waypoint), 1.)
            self.stage = "rejoin"
            self.reached_heading = False
            self.turn_ticks = 0
            self.quiet_ticks = 0
        error = angle_error(yaw, head, env.target)
        ready = env.distance() <= self.rules["brake_range"] and abs(error) <= np.radians(self.rules["brake_bearing_deg"])
        if ready:
            self.stage = "goal_hold"
            return 0, .5
        self.stage = "rejoin"
        result = self.turn_or_walk(error, 0.)
        if result[0] == 1 and env.distance() <= self.rules["brake_range"]:
            return (2 if error > 0. else 3), .5
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("cal_left", "cal_right"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads((HERE / "PLAN.json").read_text())
    course.SCENES = HERE / "SCENES.json"
    scene = json.loads(course.SCENES.read_text())[args.case]
    teacher = DetourTeacher(scene, plan["teacher"])
    out = args.out.resolve()
    result = course.rollout(out, args.case, teacher, course.new_brain(), "detour")
    env = teacher.env
    xyz, yaw = env.pose()
    head = env.eyes.pose()[0]
    tail, _ = env.footprint.bounds(env.body.d.qpos)
    teacher.rows.append((teacher.stage, *xyz, yaw, *head[:2], tail-env.box_edge, float(env.measured["nonfoot"]/course.W)))
    np.savez_compressed(out/"REFERENCE.npz",
                        phase=[row[0] for row in teacher.rows],
                        state=np.asarray([row[1:] for row in teacher.rows]),
                        columns=["com_x","com_y","com_z","yaw_rad","head_x","head_y","tail_margin","nonfoot_W_20ms"])
    forces = env.forces
    sources = (HERE/"PLAN.json",course.SCENES,Path(__file__),Path(inspect.getfile(course.Course)),
               Path(inspect.getfile(course.Teacher("detour", {}).detour.__class__)),course.BINDING,course.CHECKPOINT)
    certificate = dict(native_steps_including_warm=forces.count,box_contact_native_counts=forces.counts.tolist(),
                       box_contact_native_first=forces.first_hit.tolist(),
                       ending=result["reason"], reference_sha256=course.sha(out/"REFERENCE.npz"),
                       sources={str(path):course.sha(path) for path in sources})
    course.turn.stand.put(out/"REFERENCE.json",certificate)
    raise SystemExit(0 if result["reason"] == "success" else 1)


if __name__ == "__main__":
    main()
