"""Optional body-current intervention, not enabled by any default episode.

Apply AFTER ChannelSensoryEncoder.encode and BEFORE adding external stimuli.
This is a declared numerical intervention, not biological calibration or a
motor controller. No clipping, neural stepping, or body dynamics occurs here.
"""
from __future__ import annotations

import numpy as np

from .sensory_channels import CHANNEL_SPEC, LEGS, ChannelSensoryEncoder

JOINT_PROPRIOCEPTION_CHANNELS = (
    "claw_position", "hook_direction", "hairplate_angle",
)


class BodyCurrentModulator:
    """Snapshot a validated selection and neutral current for repeatable trials.

    Inputs must be native float64 vectors, as emitted by the encoder; leg
    names must be canonical lowercase and selections nonempty and unique.
    Unselected cells are copied unchanged. Gain 0/1 have exact endpoint
    semantics; other gains use neutral + gain * (live - neutral). The caller
    retains ownership of external stimulus composition and recording.
    """

    def __init__(self, channel_groups, n_neurons, neutral_current_nA, *,
                 channels, legs=LEGS, gain=1.0):
        encoder = ChannelSensoryEncoder(channel_groups, n_neurons)
        self._n = encoder.n
        for leg, groups in channel_groups.items():
            if leg not in LEGS or any(ch not in CHANNEL_SPEC for ch in groups):
                raise ValueError("unknown leg/channel in channel groups")
        channels = self._selection(channels, CHANNEL_SPEC, "channels")
        legs = self._selection(legs, LEGS, "legs")
        if (isinstance(gain, (bool, np.bool_, str, bytes)) or not np.isscalar(gain)
                or not np.isrealobj(gain)):
            raise ValueError("gain must be a finite nonnegative real scalar")
        try:
            gain = float(gain)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("gain must be a finite nonnegative real scalar") from None
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("gain must be a finite nonnegative real scalar")
        self._gain = gain
        self._neutral = self._vector(neutral_current_nA, "neutral current").copy()
        self._mask = np.zeros(self._n, dtype=bool)
        for leg in legs:
            for channel in channels:
                self._mask[encoder.groups[leg][channel]] = True
        self._channels = channels
        self._legs = legs

    @staticmethod
    def _selection(values, allowed, name):
        if isinstance(values, str):
            raise ValueError(f"{name} must be an explicit nonempty sequence")
        try:
            values = tuple(values)
        except TypeError:
            raise ValueError(f"{name} must be an explicit nonempty sequence") from None
        if (not values or any(not isinstance(v, str) or v not in allowed for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain unique known names")
        return values

    def _vector(self, value, name):
        try:
            vector = np.asarray(value)
            # Encoder output is float64. Reject coercions so endpoint and
            # unselected-byte identity are meaningful, including signed zero.
            if vector.dtype != np.dtype(np.float64):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite real vector") from None
        if vector.shape != (self._n,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must have shape ({self._n},) and be finite")
        return vector

    def apply(self, live_body_current_nA):
        """Return an owned float64 body-current vector; do not pass total input."""
        live = self._vector(live_body_current_nA, "live body current")
        out = live.copy()
        if self._gain == 1:
            return out
        if self._gain == 0:
            out[self._mask] = self._neutral[self._mask]
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                out[self._mask] = self._neutral[self._mask] + self._gain * (
                    live[self._mask] - self._neutral[self._mask])
        if not np.all(np.isfinite(out)):
            raise ValueError("modulated body current is nonfinite; no clipping applied")
        return out

    def describe(self):
        """Return independent JSON-compatible intervention metadata."""
        return {"gain": self._gain, "legs": list(self._legs),
                "channels": list(self._channels),
                "selected_cells": int(self._mask.sum()),
                "n_neurons": self._n,
                "composition": "encode body -> modulate body -> add external stimulus",
                "biological_calibration": "NOT_ESTABLISHED"}
