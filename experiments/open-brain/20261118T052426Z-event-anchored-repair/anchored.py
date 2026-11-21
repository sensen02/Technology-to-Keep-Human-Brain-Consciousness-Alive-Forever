"""Separate causal event coding from advisory amplitude reconciliation."""
import numpy as np


class AnchoredReadout:
    """The event encoder has only the actual arrivals as input."""

    def __init__(self, codec, lane_count, step_seconds):
        self.codec = codec
        self.count = np.zeros(lane_count, dtype=np.int32)
        self.settings = dict(relative_threshold=.10, minimum_rate_hz=15.,
                             required_ticks=int(round(.100 / step_seconds)))

    def encode(self, actual_arrivals):
        return self.codec.step(actual_arrivals)

    @property
    def rates(self):
        return self.codec.rate

    def reconcile_left_amplitude(self, observed, independent_events_verified):
        """Reconstruct an advisory amplitude from an independent event stream.

        Trust must be based on distinct acquisition, not the prior or a lack of
        spikes. The recovery flag and value do not enter the event encoder.
        """
        reference = self.rates[:, 0]
        observable = (independent_events_verified & (reference >= self.settings["minimum_rate_hz"])
                      & np.isfinite(observed))
        conflict = (observable &
                    (np.abs(observed - reference) > self.settings["relative_threshold"] * reference))
        self.count = np.where(conflict, self.count + 1, 0)
        use_reference = self.count >= self.settings["required_ticks"]
        return np.where(use_reference, reference, observed), use_reference
