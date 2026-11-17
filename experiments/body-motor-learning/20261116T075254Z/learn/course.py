"""Engineering command timelines, in native length per second and radians per second."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

CENTER_SPEED = 0.07171300063848401
DT_MS = 5
UNITS = ("native_length_per_s", "rad_per_s")


@dataclass(frozen=True)
class Segment:
    name: str
    begin: int
    end: int
    first: tuple[float, float]
    last: tuple[float, float]


class Course:
    def __init__(self, data):
        if tuple(data["units"]) != UNITS:
            raise ValueError("Command units must be native length/s and rad/s")
        self.default = data
        self.segments = []
        cursor = 0
        for item in data["segments"]:
            duration = item["duration_ms"]
            if type(duration) is not int or duration <= 0 or duration % DT_MS:
                raise ValueError(f"Segment {item['name']} needs a positive 5 ms duration")
            first = tuple(item["first"])
            last = tuple(item["last"])
            if len(first) != 2 or len(last) != 2:
                raise ValueError("Expected [advance, yaw]")
            if not np.isfinite(np.asarray((first, last))).all():
                raise ValueError("Expected finite intent")
            if any(speed < 0 for speed in (first[0], last[0])):
                raise ValueError("This bench tests forward locomotion")
            if any(speed == 0 and yaw != 0 for speed, yaw in (first, last)):
                raise ValueError("Moving arcs require forward propulsion")
            segment = Segment(item["name"], cursor, cursor + duration, first, last)
            if any(previous.name == segment.name for previous in self.segments):
                raise ValueError(f"Duplicate segment {segment.name}")
            if self.segments and self.segments[-1].last != first:
                if first != (0., 0.) and self.segments[-1].last != (0., 0.):
                    raise ValueError("Non-stop segments require continuous intent")
            self.segments.append(segment)
            cursor += duration
        if len(self.segments) < 3:
            raise ValueError("Expected stand, locomote and stop segments")
        if any(seg.first != (0., 0.) or seg.last != (0., 0.)
               for seg in (self.segments[0], self.segments[-1])):
            raise ValueError("The initial and terminal standing windows use exact zero")
        self.finish = cursor
        self.boundaries = np.asarray([s.end for s in self.segments])
        self.by_name = {s.name: s for s in self.segments}
        self.speed_pair = data.get("speed_pair")
        self.turn = data.get("turn")
        for window in (self.speed_pair, self.turn):
            if window is None:
                continue
            for value in window.values():
                if isinstance(value, str) and value not in self.by_name:
                    raise ValueError(f"Unknown window {value}")
        if self.turn:
            if self.turn["direction"] not in (-1, 1):
                raise ValueError("Arc direction must be -1 or +1")
            exit_segment = self.by_name[self.turn["exit"]]
            if exit_segment.first[1] != 0. or exit_segment.last[1] != 0.:
                raise ValueError("Exit test strictly requests zero yaw")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text()))

    def segment(self, ms):
        idx = min(np.searchsorted(self.boundaries, ms, side="right"), len(self.segments) - 1)
        return self.segments[idx]

    def command(self, ms):
        phase = self.segment(ms)
        a = float(np.clip((ms-phase.begin)/(phase.end-phase.begin), 0., 1.))
        first, last = np.asarray(phase.first), np.asarray(phase.last)
        return first + a * (last-first)
