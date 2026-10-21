"""Fixed one-pass episode-only coordinate search, never a runtime controller."""
import numpy as np
from .pool_gain_translation import POOL_COUNT, pool_gain_weights


def _score(value):
    ticks, foot = value
    if isinstance(ticks, (bool, np.bool_)) or not isinstance(ticks, (int, np.integer)) or not 0 <= ticks <= 88000:
        raise ValueError('integer prefix ticks in [0,88000] required')
    if not np.isfinite(foot) or foot < 0:
        raise ValueError('finite nonnegative good-interval mean foot/W required')
    return int(ticks), float(foot)


class PoolGainCoordinateSearch:
    """66 fixed coordinates +/- .5 in [0,2], original incumbent, exact ties.

    Caller supplies completed-episode scores from support_duration_score. This
    class cannot run a body/network, observe state, generate commands or retry.
    Cache includes original; proposal arrays are copies. No unmeasured updates.
    """
    def __init__(self, baseline_score):
        self._gains = np.ones(POOL_COUNT)
        self._score = _score(baseline_score)
        self._label = 'original_cached_baseline'
        self._coordinate = 0
        self._cache = {tuple(self._gains): (self._score, self._label)}

    @property
    def gains(self):
        return self._gains.copy()

    @property
    def weights(self):
        return pool_gain_weights(self._gains)

    @property
    def score(self):
        return self._score

    @property
    def label(self):
        return self._label

    @property
    def coordinate(self):
        return self._coordinate

    def probes(self):
        if self._coordinate >= POOL_COUNT:
            raise ValueError('fixed one-pass budget exhausted')
        pair = []
        for sign in (1, -1):
            g = self._gains.copy()
            g[self._coordinate] = np.clip(g[self._coordinate] + sign*.5, 0, 2)
            pair.append(g)
        return tuple(pair)

    @staticmethod
    def _key(gains):
        g = np.asarray(gains, dtype=float)
        pool_gain_weights(g)  # common shape/finite contract
        if np.any(g > 2):
            raise ValueError('search gains require [0,2]')
        return tuple(g)

    def cached(self, gains):
        return self._cache.get(self._key(gains))

    def record(self, gains, score, label):
        key = self._key(gains)
        allowed = {self._key(g) for g in self.probes()}
        if key not in allowed:
            raise ValueError('only current coordinate probes may be recorded')
        if key in self._cache:
            raise ValueError('duplicate must reuse cache, not another episode')
        if not isinstance(label, str) or not label or any(label == v[1] for v in self._cache.values()):
            raise ValueError('unique measurement label required')
        self._cache[key] = (_score(score), label)

    def update(self):
        pair = self.probes()
        keys = [self._key(g) for g in pair]
        if any(k not in self._cache for k in keys):
            raise ValueError('both probes must be measured/cached before update')
        # Transactional: missing/invalid data cannot partially mutate incumbent.
        best_score, best_label, best_gains = self._score, self._label, self._gains
        for g, key in zip(pair, keys):
            score, label = self._cache[key]
            if score > best_score:
                best_score, best_label, best_gains = score, label, g
        self._score, self._label = best_score, best_label
        self._gains = best_gains.copy()
        self._coordinate += 1
