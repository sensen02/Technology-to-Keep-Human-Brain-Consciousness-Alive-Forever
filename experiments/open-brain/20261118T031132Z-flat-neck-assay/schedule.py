"""Frozen four-arm virtual test schedule."""
import numpy as np

ARMS = ("intact", "sham", "sustained_cut", "restore")
SEEDS = (10901, 11311, 11701)
CUT_ROOT = "720575941633589179"
CUT_ON = 4500
RESTORE = 9000
YAW_SAMPLE = 17999
YAW_EFFECT_THRESHOLD_DEG = 5.


class AssaySchedule:
    def __init__(self, ports, target):
        self.blocked = np.zeros((len(ARMS), ports), dtype=bool)
        self.target = target

    def step(self, tick):
        self.blocked[2, self.target] = tick >= CUT_ON
        self.blocked[3, self.target] = CUT_ON <= tick < RESTORE
        return self.blocked

    def scheduled_event(self, tick):
        flags = np.zeros(len(ARMS), dtype=np.int8)
        if tick == CUT_ON:
            flags[1:] = 1
        elif tick == RESTORE:
            flags[1] = -1
            flags[3] = -1
        return flags
