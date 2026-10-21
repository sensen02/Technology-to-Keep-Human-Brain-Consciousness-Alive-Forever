"""Episode-only six-leg relative gains; no runtime controller or neural changes."""
import numpy as np
from .motor_translation import FiniteMotorTranslator
from .pool_gain_translation import pool_gain_weights
from .support_duration_search import score_tuple

LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
EXPECTED_COUNTS = (12, 10, 11, 12, 10, 11)


def pool_leg_mapping(manifest):
    """Validate MotorOutput contract through existing translator, then explicit legs.

    Identical mapped pools share exactly one leg/gain. Empty rows are ignored.
    No actuator/muscle-name interpretation and no actual network/model import.
    """
    translator = FiniteMotorTranslator(manifest)
    owners = {}
    for row, pool_index in zip(manifest['rows'], translator.actuator_pool):
        if pool_index < 0:
            continue
        leg = row.get('leg')
        if leg not in LEGS:
            raise ValueError('mapped row requires explicit known leg')
        previous = owners.setdefault(int(pool_index), leg)
        if previous != leg:
            raise ValueError('shared pool crosses explicit legs')
    if len(translator.pools) != 66 or set(owners) != set(range(66)):
        raise ValueError('exactly 66 unique completely assigned pools required')
    mapping = np.asarray([LEGS.index(owners[i]) for i in range(66)], dtype=int)
    if tuple(np.bincount(mapping, minlength=6)) != EXPECTED_COUNTS:
        raise ValueError('unexpected explicit leg pool counts')
    return mapping


def gain_key(gains):
    g = np.asarray(gains, dtype=float)
    if g.shape != (6,) or not np.isfinite(g).all() or np.any((g < 0) | (g > 2)):
        raise ValueError('six finite engineering gains in [0,2] required')
    return tuple(g.tolist())


def expand_leg_gains(gains, mapping):
    g = np.asarray(gain_key(gains))
    m = np.asarray(mapping)
    if m.shape != (66,) or m.dtype.kind not in 'iu' or np.any((m < 0) | (m > 5)):
        raise ValueError('66 integer leg indices required')
    return g[m].copy()


class LegGainSearch:
    def __init__(self, metric, provenance, mapping):
        self.mapping = np.asarray(mapping).copy()
        self.center = np.ones(6)
        self.weights_for(self.center)
        self.metric, self.label, self.coordinate = metric, 'original_cached_baseline', 0
        self.cache = {gain_key(self.center): dict(label=self.label, metric=metric, provenance=provenance)}

    def weights_for(self, gains):
        return pool_gain_weights(expand_leg_gains(gains, self.mapping))

    def probes(self):
        if self.coordinate >= 6:
            raise ValueError('one-pass six-coordinate budget exhausted')
        pair = []
        for sign in (1, -1):
            value = self.center.copy()
            value[self.coordinate] = np.clip(value[self.coordinate] + sign*.25, 0, 2)
            pair.append(value)
        return tuple(pair)

    def remember(self, gains, label, metric, provenance):
        key = gain_key(gains)
        if key not in {gain_key(g) for g in self.probes()}:
            raise ValueError('only current pair may be measured')
        if key in self.cache:
            raise ValueError('duplicate must reuse cache')
        if not label or any(v['label'] == label for v in self.cache.values()):
            raise ValueError('unique measurement label required')
        self.cache[key] = dict(label=label, metric=metric, provenance=provenance)

    def update(self):
        pair = self.probes()
        if any(gain_key(g) not in self.cache for g in pair):
            raise ValueError('both probes required before updating incumbent')
        before = dict(label=self.label, metric=self.metric, gains6=self.center.tolist())
        changed = False
        for g in pair:
            candidate = self.cache[gain_key(g)]
            if score_tuple(candidate['metric']) > score_tuple(self.metric):
                self.center, self.metric, self.label = g.copy(), candidate['metric'], candidate['label']
                changed = True
        self.coordinate += 1
        return dict(coordinate=self.coordinate, leg=LEGS[self.coordinate-1], incumbent_before=before,
                    incumbent_after=dict(label=self.label, metric=self.metric, gains6=self.center.tolist()), changed=changed)
