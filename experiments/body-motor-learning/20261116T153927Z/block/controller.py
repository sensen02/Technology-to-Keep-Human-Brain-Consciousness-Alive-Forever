"""Privileged stepped-foothold teacher using only the frozen motor interface."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
TURN_SOURCE = ARCHIVE / '20261116T004235Z'
STUMBLE_SOURCE = ARCHIVE / '20261116T060654Z' / 'teacher'
sys.path.insert(0, str(TURN_SOURCE))
import turning as turn

previous = turn.fixed.old


class CourseReference(previous.PivotReference):
    def __init__(self, c, teacher, order, plan, terrain_aware):
        super().__init__(c, teacher, 'left_pivot', order)
        self.teacher = teacher
        self.plan = plan
        self.duration = plan['course_ms'] * .001
        self.brake = plan['brake_ms'] * .001
        self.heading = self.rot0[:2, 0] / np.linalg.norm(self.rot0[:2, 0])
        self.speed = plan['speed']
        self.box_min = np.asarray(plan['center']) - np.asarray(plan['half_size'])
        self.box_max = np.asarray(plan['center']) + np.asarray(plan['half_size'])
        self.aware = terrain_aware
        self.levels = np.zeros(6)
        self.swing_end_level = 0.
        self.swing_lift = self.lift
        self.hold = False
        self.last_target = self.anchors.copy()
        self.last_down = np.ones(6, dtype=bool)

    def distance(self, t):
        t = min(t, self.duration)
        return self.speed * (
            previous.ramp_integral(t)
            - previous.ramp_integral(max(0., t - self.duration + self.brake), self.brake)
        )

    def center(self, t):
        return self.origin + np.r_[self.distance(t) * self.heading, 0.]

    def command(self, t):
        if t >= self.duration:
            return np.array([0., 0.])
        speed = self.speed * previous.blend(t)
        brake = previous.blend((t - self.duration + self.brake) / self.brake)
        return np.array([speed * (1. - brake), 0.])

    def _foothold(self, endpoint):
        xy = endpoint[:2]
        lower, upper = self.box_min[:2], self.box_max[:2]
        margin = self.plan['edge_margin']
        if np.all(xy >= lower) and np.all(xy <= upper):
            xy[:] = np.clip(xy, lower + margin, upper - margin)
            return self.box_max[2]
        if np.all(xy > lower - margin) and np.all(xy < upper + margin):
            candidates = np.array([xy[0]-lower[0], upper[0]-xy[0], xy[1]-lower[1], upper[1]-xy[1]])
            side = int(np.argmin(candidates))
            if side % 2 == 0:
                xy[side//2] = lower[side//2] - margin
            else:
                xy[side//2] = upper[side//2] + margin
        return 0.

    def _near_box(self, start, end):
        low, high = np.minimum(start[:2], end[:2]), np.maximum(start[:2], end[:2])
        margin = self.plan['edge_margin']
        return bool(np.all(high > self.box_min[:2] - margin) and np.all(low < self.box_max[:2] + margin))

    def _commit_touchdown(self):
        leg = self.order[self.slot % 6]
        self.anchors[leg] = self.swing_end.copy()
        self.levels[leg] = self.swing_end_level

    def sample(self, t):
        if self.last_t is not None and not np.isclose(t-self.last_t,.005,atol=1e-9,rtol=0):
            raise ValueError('Expected contiguous 5 ms teacher calls')
        self.last_t = t
        world = self.anchors.copy()
        baseline = self.levels.copy()
        clearance = np.zeros(6)
        stance = np.ones(6, dtype=bool)
        phase = 'course_walk'
        if t < self.duration - 1e-10:
            slot, leg, u = previous.schedule(t, self.order, self.step_s)
            if slot != self.slot:
                if self.slot >= 0:
                    self._commit_touchdown()
                self.slot = slot
                self.swing_start = self.anchors[leg].copy()
                self.swing_end = self.center((slot + 1) * self.step_s + 2.) + self.rot0 @ self.rest[leg]
                self.swing_end_level = self._foothold(self.swing_end) if self.aware else 0.
                self.swing_lift = self.lift
                if self.aware and self._near_box(self.swing_start, self.swing_end):
                    self.swing_lift = self.plan['terrain_lift']
            s = previous.blend(u)
            world = self.anchors.copy()
            baseline = self.levels.copy()
            world[leg] = (1.-s)*self.swing_start + s*self.swing_end
            baseline[leg] = (1.-s)*self.levels[leg] + s*self.swing_end_level
            clearance[leg] = self.swing_lift * 16. * u*u*(1.-u)*(1.-u)
            stance[leg] = False
        else:
            if not self.hold:
                self._commit_touchdown()
                self.hold = True
            world = self.anchors.copy()
            baseline = self.levels.copy()
            phase = 'course_stop'
        # Height allocation is relative to each planned supporting surface.
        # This alters private IK targets, never actual position or load.
        self.teacher.rest_pad_bottom[:] = self.bottom0 + baseline
        self.last_target = np.c_[world[:, :2], self.bottom0 + baseline + clearance]
        self.last_down = stance
        self.last_snapshot = dict(t=t, phase=phase, planned_bottom_native=self.last_target.tolist())
        relative = (world - self.center(t)) @ self.rot0
        return dict(target_xy=relative[:, :2], target_z=clearance, stance=stance,
                    source_time_s=None, phase=phase)


class Controller:
    def __init__(self, c, scenario, terrain_aware):
        self.teacher = turn.make_teacher(c, 'arc_left', scenario['speed'], .115)
        old = self.teacher.reference
        self.reference = CourseReference(c, self.teacher, old.order, scenario, terrain_aware)
        self.teacher.reference = self.reference

    def action(self, c, t):
        a, _ = self.teacher.action(c, t)
        return a, self.reference.command(t)
