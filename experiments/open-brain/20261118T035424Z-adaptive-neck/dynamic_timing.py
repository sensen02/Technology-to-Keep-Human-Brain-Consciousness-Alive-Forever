"""Compatible front-AN source model and causal post-neck representations."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BEHAVIOR = HERE.parent / "20261118T002600Z-ascending-behavior-comparison"
sys.path.insert(0, str(BEHAVIOR))
from dynamic_transport import (CAPACITANCE_PF, CHARGE_NS_MS, DECAY_MS, RELEASE_MS,
                             REVERSAL_MV, RISE_MS, FixedCircuit)
from circuit import Circuit, eon

DT=.0001


class IncomingConductance:
    """Reuse fixed release, receptor kinetics, and charge for observed DN arrivals."""

    def __init__(self, matrices, dt_ms, enabled):
        self.matrices = matrices
        self.enabled = np.asarray(enabled, dtype=bool)
        self.dt_ms = dt_ms
        self.rise_factor = np.exp(-dt_ms / RISE_MS)[:, None, None]
        self.decay_factor = np.exp(-dt_ms / DECAY_MS)[:, None, None]
        self.rise_average = (RISE_MS / dt_ms)[:, None, None] * (1 - self.rise_factor)
        self.decay_average = (DECAY_MS / dt_ms)[:, None, None] * (1 - self.decay_factor)
        shape = (len(matrices), len(enabled), matrices[0].shape[0])
        self.rise = np.zeros(shape)
        self.decay = np.zeros(shape)
        self.pending = np.zeros((int(np.ceil(RELEASE_MS / dt_ms)), *shape))
        self.slot = 0

    def voltage_step(self, arrivals, voltage):
        release = self.pending[self.slot] / (DECAY_MS - RISE_MS)[:, None, None]
        self.rise += release
        self.decay += release
        g = np.maximum(0., self.decay * self.decay_average - self.rise * self.rise_average)
        self.rise *= self.rise_factor
        self.decay *= self.decay_factor
        arriving = arrivals & self.enabled[:, None]
        for k, matrix in enumerate(self.matrices):
            self.pending[self.slot, k] = (matrix @ arriving.T).T * CHARGE_NS_MS[k]
        self.slot = (self.slot + 1) % len(self.pending)
        total = g.sum(axis=0)
        # Treat the fixed incoming conductance as a separate voltage kick; this
        # preserves Eon's local reference integrator when the arriving drive is zero.
        current = (g * (REVERSAL_MV[:, None, None] - voltage)).sum(axis=0)
        duration = np.full_like(total, self.dt_ms / CAPACITANCE_PF)
        np.divide(-np.expm1(-self.dt_ms * total / CAPACITANCE_PF), total,
                  out=duration, where=total > 0.)
        self.last_g = g
        return (current * duration).astype(np.float32)


class FrontTouchSources:
    """Two body-side somata on the published BANC local receptor graph."""

    def __init__(self, neck, roots, lanes, feedback_lanes):
        assert all(neck.ports[root]["group"] == "front_touch" for root in roots)
        assert set(feedback_lanes) <= set(lanes)
        self.lanes = np.asarray(lanes, dtype=int)
        self.circuit = Circuit({"ports": {root: neck.ports[root] for root in roots}},
                               batch=len(lanes))
        self.neck_inputs = [neck.index[root] for root in self.circuit.roots]
        self.neck_cells = [neck.index[root] for root in roots]
        self.feedback = IncomingConductance([matrix[self.neck_cells] for matrix in neck.neck],
                                             neck.dt_ms,
                                             [lane in feedback_lanes for lane in lanes])
        self.state = self.circuit._cells.state_init()

    def step(self, drive, arrivals):
        local = (self.circuit.local @ self.state[2].numpy().T).T
        stimulus = drive[np.ix_(self.lanes, self.neck_inputs)]
        columns = self.circuit.port_columns
        stimulus[:, columns] += self.feedback.voltage_step(
            arrivals[self.lanes], self.state[3].numpy()[:, columns])
        self.state = self.circuit._cells(torch.from_numpy(local * eon.MODEL_PARAMS["wScale"]),
                                         torch.from_numpy(stimulus), *self.state)
        return self.state[2].numpy()[:, columns].astype(bool), self.state[3].numpy()[:, columns]


class CanonicalEvents:
    """Use the established 20 ms reference representation at the brain boundary."""

    def __init__(self, shape):
        self.rate = np.zeros(shape)
        self.phase = np.zeros(shape)
        self.alpha = -np.expm1(-DT / .020)

    def step(self, arrivals):
        self.rate += self.alpha * (arrivals / DT - self.rate)
        self.phase += self.rate * DT
        events = self.phase >= 1.
        self.phase -= events
        return events


class FrontTouchNeck(FixedCircuit):
    """Keep fixed transport/receivers and select the AN soma equation per lane."""

    def __init__(self, catalog, roots, matched_lanes, feedback_lanes, batch, max_extra_delay=0):
        super().__init__(catalog, batch=batch, max_extra_delay=max_extra_delay)
        self.front_sources = FrontTouchSources(self, roots, matched_lanes, feedback_lanes)

    def update_port_sources(self, drive, arrivals):
        sources = self.front_sources
        indices = np.ix_(sources.lanes, sources.neck_cells)
        events, voltage = sources.step(drive, arrivals)
        self._spikes[indices] = events
        self._voltage[indices] = voltage
