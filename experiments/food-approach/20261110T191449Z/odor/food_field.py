"""Engineering odor transduction: NumPy only, no action generation.

Coordinates/offsets share the simulation's declared length units (SI calibration
NOT_ESTABLISHED). Head local +x is forward, +y left. Offsets are engineering
choices, not anatomical receptor locations. Keep this module environment-side:
pass only sample()'s two numbers to the actor.
"""
from dataclasses import dataclass
import math
import numpy as np


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"invalid {name}")
    return result


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


@dataclass(frozen=True)
class FoodSource:
    """Immutable Gaussian source; patch_radius is food-contact metadata only."""
    center_xy: tuple
    patch_radius: float
    amplitude: float
    sigma: float
    onset_s: float
    rise_s: float = 0.2

    def __post_init__(self):
        center = _array(self.center_xy, (2,), "center_xy")
        object.__setattr__(self, "center_xy", tuple(float(v) for v in center))
        for name in ("patch_radius", "sigma", "rise_s"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, positive=True))
        for name in ("amplitude", "onset_s"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, nonnegative=True))

    def concentration(self, sensor_positions_xy, time_s):
        """Environment diagnostic: Gaussian concentration at two world XY points."""
        positions = _array(sensor_positions_xy, (2, 2), "sensor_positions_xy")
        time = _scalar(time_s, "time_s", nonnegative=True)
        if time <= self.onset_s or self.amplitude == 0:
            return np.zeros(2, dtype=np.float64)
        # Overflow in a finite coordinate difference denotes an infinitely far
        # point at float64 precision, hence zero Gaussian concentration.
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            fraction = min((time - self.onset_s) / self.rise_s, 1.0)
            onset = fraction * fraction * (3.0 - 2.0 * fraction)
            delta = (positions - np.asarray(self.center_xy)) / self.sigma
            distance = np.hypot(delta[:, 0], delta[:, 1])
            return (self.amplitude * onset) * np.exp(-0.5 * distance * distance)


def bilateral_positions(head_position, head_rotation, forward_offset, lateral_offset):
    """Return (left,right) world XY positions from a head body pose.

    Accept 3D position and proper 3x3 local-to-world rotation (e.g. MuJoCo
    data.xpos[head_id], data.xmat[head_id].reshape(3,3)), or planar position
    and 2x2 rotation. Transform before projecting to XY. Offsets are explicit;
    forward may be signed, lateral must be nonnegative.
    """
    forward = _scalar(forward_offset, "forward_offset")
    lateral = _scalar(lateral_offset, "lateral_offset", nonnegative=True)
    position = np.asarray(head_position, dtype=np.float64)
    if position.shape not in ((2,), (3,)):
        raise ValueError("head_position must have shape (2,) or (3,)")
    n = position.size
    position = _array(position, (n,), "head_position")
    rotation = _array(head_rotation, (n, n), "head_rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(n), atol=1e-8, rtol=0) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8):
        raise ValueError("head_rotation must be a proper orthonormal rotation")
    local = np.zeros((2, n), dtype=np.float64)
    local[:, 0] = forward
    local[:, 1] = (lateral, -lateral)
    with np.errstate(over="ignore", invalid="ignore"):
        world = position + local @ rotation.T
    if not np.isfinite(world).all():
        raise ValueError("transformed sensor positions overflow")
    return world[:, :2].copy()


class BilateralOdorSensory:
    """Causal exact exponential receptor with optional seeded output noise.

    sample(..., time_s, dt_s=.005) holds START-of-interval concentration
    fixed for dt_s and returns END-of-interval [left,right] response. Default
    tau=.05 s. Calls after the first must be contiguous; a new instance starts
    a new episode. Arbitrary positive dt permits numerical composition checks;
    production callers should update every .005 s. Noise is per call, so noisy
    trajectories do not claim timestep-composition invariance.
    """
    def __init__(self, source, *, K=1.0, tau_s=0.05, noise_std=0.0, seed=None):
        if not isinstance(source, FoodSource):
            raise ValueError("source must be FoodSource")
        self._source = source
        self._K = _scalar(K, "K", positive=True)
        self._tau_s = _scalar(tau_s, "tau_s", positive=True)
        self._noise_std = _scalar(noise_std, "noise_std", nonnegative=True)
        if seed is not None and (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        if self._noise_std > 0 and seed is None:
            raise ValueError("noise requires an explicit seed")
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(2, dtype=np.float64)
        self._next_time = None

    def sample(self, sensor_positions_xy, time_s, dt_s=0.005):
        """Return only two bounded receptor outputs; no source/time metadata."""
        time = _scalar(time_s, "time_s", nonnegative=True)
        dt = _scalar(dt_s, "dt_s", positive=True)
        end = time + dt
        if not math.isfinite(end) or end <= time:
            raise ValueError("dt_s must give a finite, advancing endpoint")
        if self._next_time is not None:
            tolerance = 8 * max(math.ulp(time), math.ulp(self._next_time))
            if abs(time - self._next_time) > tolerance:
                raise ValueError("sample intervals must be contiguous and causal")
        concentration = self._source.concentration(sensor_positions_xy, time)
        # Scaled C/(K+C) avoids overflowing the denominator for finite C,K.
        scale = np.maximum(concentration, self._K)
        c = concentration / scale
        target = c / (self._K / scale + c)
        alpha = -math.expm1(-dt / self._tau_s)
        state = self._state + alpha * (target - self._state)
        if self._noise_std:
            with np.errstate(over="ignore"):
                output = np.clip(state + self._noise_std * self._rng.standard_normal(2), 0.0, 1.0)
        else:
            output = state.copy()
        self._state = state
        self._next_time = end
        return output
