"""Episode-only five-dimensional shared temporal search; no runtime decoding changes."""
import numpy as np
from .body_reward import AntitheticBodySearch


class SharedTemporalBodySearch(AntitheticBodySearch):
    """Tile five private-RNG signs across 66 pools; keep inherited incumbent rule."""
    @staticmethod
    def _require_shared(weights):
        a = np.asarray(weights)
        if a.shape != (66, 5) or not np.array_equal(a, np.tile(a[0], (66, 1))):
            raise ValueError('66 identical temporal rows required')

    def __init__(self, weights, **kwargs):
        super().__init__(weights, **kwargs)
        self._require_shared(self.center)

    def probes(self):
        self._require_shared(self.center)
        direction = np.tile(self.rng.choice([-1., 1.], size=5), (66, 1))
        return (direction, np.clip(self.center+self.radius*direction, 0, 4),
                np.clip(self.center-self.radius*direction, 0, 4))

    def update(self, candidates):
        self._require_shared(self.center)
        for candidate in candidates:
            self._require_shared(candidate['weights'])
        result = super().update(candidates)
        self._require_shared(self.center)
        return result
