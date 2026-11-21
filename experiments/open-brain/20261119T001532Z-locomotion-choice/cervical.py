"""Read DNp11/DNp10 only after the registered axons and VNC membrane dynamics."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FLIGHT = HERE.parent / '20261118T072349Z-flight-descending'
REFERENCE = HERE.parent / '20261118T083045Z-complete-landing/NECK'
sys.path.insert(0, str(FLIGHT))
from probe_neck import FixedCircuit, full_catalog, REST_MV, make_probes, FILTER_MS

LABELS = ('takeoff/left', 'takeoff/right', 'landing/left', 'landing/right')
THRESHOLD = .22


class FlightPaths:
    def __init__(self):
        ref = json.loads((REFERENCE / 'IDENTITY.json').read_text())
        records = [ref['ports'][label] for label in LABELS]
        roots = [row['banc'] for row in records]
        self.eon_roots = tuple(int(row['eon']) for row in records)
        self.circuit = FixedCircuit(full_catalog())
        probes = make_probes(self.circuit, roots)
        assert [[list(p) for p in group] for group in probes] == [
            row['targets_and_contact_counts'] for row in records]
        self.columns = np.array([self.circuit.port_index[root] for root in roots])
        self.targets = np.array([[self.circuit.index[target] for target, _ in group] for group in probes])
        self.scale = np.array([row['reference_mv'] for row in records])
        self.alpha = -np.expm1(-ref['dt_ms'] / FILTER_MS)
        self.parameters_sha256 = self.circuit.params_sha256
        self.emitted = np.zeros((1, len(self.circuit.ports)), dtype=bool)
        self.block = np.zeros(len(self.circuit.ports), dtype=bool)
        self.zero = np.zeros((1, len(self.circuit.roots)), dtype=np.float32)
        self.local = np.zeros(len(LABELS))
        self.delays = self.circuit.neck_delay[self.columns]
        assert np.array_equal(self.delays, [row['delay_ticks'] for row in records])

    def step(self, spikes):
        self.emitted[0, self.columns] = spikes
        _, arrived, _, _ = self.circuit.step(self.zero, self.block, eon_output=self.emitted)
        voltage = (self.circuit.pre_reset_voltage[0, self.targets] - REST_MV).mean(axis=-1)
        self.local += self.alpha * (voltage - self.local)
        return arrived[0, self.columns], np.clip(self.local / self.scale, 0., 1.25)

    def integrate(self, emitted, cut=False):
        self.block[self.columns[:2]] = cut
        arrived = np.zeros_like(emitted)
        levels = np.zeros_like(emitted, dtype=float)
        for tick, events in enumerate(emitted):
            arrived[tick], levels[tick] = self.step(events)
        for col, delay in enumerate(self.delays):
            expected = np.zeros(len(emitted), dtype=bool)
            if not cut or col >= 2:
                expected[delay:] = emitted[:-delay, col]
            assert np.array_equal(arrived[:, col], expected)
        return arrived, levels


def pulses(source):
    source = np.asarray(source, dtype=bool)
    if source.shape[1] == 2:
        source = np.column_stack((source, np.zeros_like(source)))
    return source
