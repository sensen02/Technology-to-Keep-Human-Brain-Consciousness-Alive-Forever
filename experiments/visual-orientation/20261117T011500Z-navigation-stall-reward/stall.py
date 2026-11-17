"""Physical, episode-local navigation stall detection."""
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from operator import attrgetter

TIME_EPS = 1.e-9
IDLE_WINDOW_S = 4.
PIT_WINDOW_S = 5.
REVERSAL_WINDOW_S = 8.
ROTATION_WINDOW_S = 16.
RECOVERY_S = 8.

# Per-second cost after detection and terminal score after the recovery window.
COSTS = {
    "idle": (.30, -16.),
    "spin": (.35, -16.),
    "pit": (.40, -16.),
}


@dataclass(frozen=True)
class Motion:
    xy: tuple[float, float]
    yaw_deg: float
    pit_id: str | None = None
    # Distance advanced towards the prescribed exit, while in this pit.
    pit_progress: float | None = None

    def __post_init__(self):
        if len(self.xy) != 2 or not all(isfinite(v) for v in (*self.xy, self.yaw_deg)):
            raise ValueError("Invalid physical motion measurement.")
        if (self.pit_id is None) != (self.pit_progress is None):
            raise ValueError("Pit progress requires a pit identity and vice versa.")
        if self.pit_progress is not None and not isfinite(self.pit_progress):
            raise ValueError("Invalid pit progress.")


@dataclass(frozen=True)
class Sample:
    time: float
    motion: Motion
    heading_change: float
    yaw_travel: float
    pit_best: float | None


class Stall:
    def __init__(self):
        self.clock = 0.
        self.history = deque()
        self.duration = 0.
        self.blocked_since = None

    def _window(self, seconds):
        cutoff = self.clock - seconds
        rows=list(self.history)
        left=bisect_right(rows,cutoff+TIME_EPS,key=attrgetter("time"))
        return rows[left-1:] if left else ()

    @staticmethod
    def _extent(samples):
        x, y = zip(*(s.motion.xy for s in samples))
        return hypot(max(x)-min(x), max(y)-min(y))

    def update(self, motion, dt, *, active):
        self.clock += dt
        if not active or motion is None:
            self.history.clear()
            self.blocked_since = None
            self.duration = 0.
            return None, 0., False

        previous = self.history[-1] if self.history else None
        if previous is None:
            yaw = yaw_travel = 0.
        else:
            dyaw = (motion.yaw_deg-previous.motion.yaw_deg+180.) % 360. - 180.
            yaw = previous.heading_change + dyaw
            yaw_travel = previous.yaw_travel + abs(dyaw)
        pit_best = motion.pit_progress
        if pit_best is not None and previous is not None and previous.motion.pit_id == motion.pit_id:
            pit_best = max(pit_best, previous.pit_best)
        self.history.append(Sample(self.clock, motion, yaw, yaw_travel, pit_best))
        while len(self.history) > 1 and self.history[1].time <= self.clock-ROTATION_WINDOW_S:
            self.history.popleft()

        kind = self._kind(motion)
        # A continuous failure to move can alternate between symptoms. A clear
        # physical window restores the recovery time.
        if kind is None:
            self.blocked_since = None
            self.duration = 0.
            return None, 0., False
        if self.blocked_since is None:
            self.blocked_since = self.clock
        self.duration = self.clock-self.blocked_since
        return kind, COSTS[kind][0]*dt, self.duration >= RECOVERY_S-TIME_EPS

    def _kind(self, motion):
        if motion.pit_id is not None:
            pit = self._window(PIT_WINDOW_S)
            if (pit and all(step.motion.pit_id == motion.pit_id for step in pit)
                    and pit[-1].pit_best-pit[0].pit_best < .075):
                return "pit"
            return None

        reversals = self._window(REVERSAL_WINDOW_S)
        if reversals:
            first, last = reversals[0], reversals[-1]
            yaw_travel = last.yaw_travel-first.yaw_travel
            yaw_reversed = yaw_travel-abs(last.heading_change-first.heading_change)
            if yaw_travel >= 70. and yaw_reversed >= 35. and self._extent(reversals) <= .14:
                return "spin"
        rotation = self._window(ROTATION_WINDOW_S)
        recent = self._window(2.)
        if (rotation and rotation[-1].yaw_travel-rotation[0].yaw_travel >= 60.
                and recent[-1].yaw_travel-recent[0].yaw_travel >= 7.
                and self._extent(rotation) <= .14):
            return "spin"

        idle = self._window(IDLE_WINDOW_S)
        if (idle and idle[-1].yaw_travel-idle[0].yaw_travel <= 15.
                and self._extent(idle) <= .07):
            return "idle"
        return None
