"""Define lesion-time outcomes without modifying source spikes."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("intact", "tear_an", "dropout_an", "block_an", "dropout_dn", "block_dn")
SEEDS = (10901, 11311, 11701)
FAULT_ON, FAULT_OFF = 6500, 12000
DROPOUT_PROBABILITY = .35
RNG_DOMAIN = 921100
GEOMETRY = HERE / "GEOMETRY_FAULT.json"


def registered_targets():
    rows = json.loads(GEOMETRY.read_text())["trace_intersections"]
    tear, = [r for r in rows if r["cut_by_projected_slit"]]
    dn, = [r for r in rows if r["direction"] == "DN"]
    assert tear["direction"] == "AN" and not dn["cut_by_projected_slit"]
    return tear, dn


def lesion_tables(seed, n):
    """Time advances in lesion space, independent of the two closed loops."""
    tear, dn = registered_targets()
    length = n + max(tear["entry_to_fault_ticks"], dn["entry_to_fault_ticks"])
    tick = np.arange(length)
    during = (FAULT_ON <= tick) & (tick < FAULT_OFF)
    rng = np.random.default_rng(np.random.SeedSequence([seed, RNG_DOMAIN]))
    an_drop, dn_drop = rng.random((2, length)) < DROPOUT_PROBABILITY
    return np.column_stack((np.zeros(length,dtype=bool), tick >= FAULT_ON,
                           during & an_drop, during, during & dn_drop, during))


class LesionSchedule:
    def __init__(self, neck, seed, n):
        tear, dn = registered_targets()
        self.target_roots = [None] + [tear["root"]]*3 + [dn["root"]]*2
        self.offsets = np.asarray([0] + [tear["entry_to_fault_ticks"]]*3 + [dn["entry_to_fault_ticks"]]*2)
        self.target_columns = [neck.port_index[root] if root else 0 for root in self.target_roots]
        for item in (tear, dn):
            assert neck.neck_delay[neck.port_index[item["root"]]] == item["total_delay_ticks"]
        self.lesion_loss = lesion_tables(seed, n)
        self.lanes = np.arange(len(ARMS))
        at_lesion = np.arange(n)[:,None] + self.offsets
        self.departure_loss = self.lesion_loss[at_lesion, self.lanes]
        self.mask = np.zeros((len(ARMS), len(neck.ports)), dtype=bool)

    def step(self, tick):
        self.mask[self.lanes, self.target_columns] = self.departure_loss[tick]
        return self.mask
