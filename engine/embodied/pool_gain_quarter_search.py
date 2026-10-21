"""New frozen .25-step experiment; historical .5 implementation stays untouched."""
import numpy as np
from .pool_gain_search import PoolGainCoordinateSearch
from .pool_gain_translation import POOL_COUNT


class PoolGainQuarterSearch(PoolGainCoordinateSearch):
    """Only change probes to latest actual leg-search step .25."""
    def probes(self):
        if self._coordinate >= POOL_COUNT:
            raise ValueError('fixed one-pass budget exhausted')
        pair = []
        for sign in (1, -1):
            g = self._gains.copy()
            g[self._coordinate] = np.clip(g[self._coordinate] + sign*.25, 0, 2)
            pair.append(g)
        return tuple(pair)
