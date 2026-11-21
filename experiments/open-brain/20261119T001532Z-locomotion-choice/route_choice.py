"""Translate eye evidence into recurrent Eon inputs for a takeoff decision."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261118T231627Z-open-fruit-loop'))
from learn_reach import Reachability
from cervical import FlightPaths, THRESHOLD

MAX_RATE = 180.
MIN_PROBABILITY = .80


class ChoiceReadout:
    def __init__(self, index, seed, ticks, *, enabled, eye_masked=False, cut=False):
        self.flight = FlightPaths()
        self.flight_indices = np.array([index[root] for root in self.flight.eon_roots])
        self.flight.block[self.flight.columns[:2]] = cut
        self.enabled = enabled
        self.eye_masked = eye_masked
        self.cut = cut
        self.indices = np.empty(0, dtype=int)
        if enabled:
            fit = json.loads((HERE / 'REACH_FIT/FIT.json').read_text())
            self.visual = Reachability(fit)
            scans = json.loads((HERE / 'LAUNCH_SCAN.json').read_text())
            selected = next(s for s in scans if s['name'] == 'mixed8' and s['rate'] == MAX_RATE)
            self.roots = selected['roots']
            self.indices = np.array([index[root] for root in self.roots])
            assert not np.intersect1d(self.indices, self.flight_indices).size
        self.draws = np.random.default_rng(seed + 773).random((ticks, len(self.indices)), dtype=np.float32)
        self.prob = 0.
        self.rate = 0.
        self.seen_frames = 0
        self.selected_s = None
        self.arrived = np.zeros((ticks,4),dtype=bool)
        self.fired = np.zeros_like(self.arrived)
        self.levels = np.zeros((ticks,4))
        self.score = np.zeros(ticks)
        self.drive_count = np.zeros(ticks,dtype=int)

    def observe(self, visual, pitch, present, mouth_contact):
        if not self.enabled:
            return
        self.prob = self.visual.predict(visual, pitch, present and not self.eye_masked)
        self.seen_frames = self.seen_frames + 1 if self.prob >= MIN_PROBABILITY else 0
        self.rate = MAX_RATE * self.prob if self.seen_frames >= 2 and not mouth_contact else 0.

    def stimulus(self, tick):
        active = self.indices[self.draws[tick] < self.rate * .0001]
        self.drive_count[tick] = len(active)
        self.score[tick] = self.prob
        return active

    def record(self, tick, spikes):
        e = spikes[self.flight_indices]
        self.fired[tick] = e
        self.arrived[tick], self.levels[tick] = self.flight.step(e)
        # The body uses this same threshold. Mark the event in the receipt now.
        if self.selected_s is None and np.min(self.levels[tick, :2]) >= THRESHOLD:
            self.selected_s = tick * .0001

    def check(self, ticks):
        for i, delay in enumerate(self.flight.delays):
            expected = np.zeros(ticks, dtype=bool)
            if not self.cut or i>=2:
                expected[delay:] = self.fired[:ticks-delay, i]
            assert np.array_equal(self.arrived[:ticks, i], expected)
