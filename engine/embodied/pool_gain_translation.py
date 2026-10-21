"""66 episode-frozen MN-pool gains; unchanged causal FIR runtime decoder.

W[p,b] = gain[p] * ORIGINAL_FIR[b]. This is a 66-dimensional subset
of the existing 330 weights, not a body-feedback controller or action generator.
Pool order and shared-actuator semantics come ONLY from FiniteMotorTranslator.
"""
import numpy as np
from .motor_translation import FiniteMotorTranslator

ORIGINAL_FIR = (.4, .25, .2, .1, .05)
POOL_COUNT = 66
# Preserve the existing per-weight [0,4] contract without clipping/distorting FIR.
MAX_GAIN = FiniteMotorTranslator.MAX_WEIGHT / max(ORIGINAL_FIR)


def pool_gain_weights(gains):
    """Validate exactly 66 finite gains in [0,10], return fresh 66x5 weights.

    Bounds are an engineering representation contract, NOT a recommended search
    range or biological calibration. No independent bin clipping is performed.
    """
    g = np.asarray(gains, dtype=float)
    if g.shape != (POOL_COUNT,) or not np.isfinite(g).all() or np.any((g < 0) | (g > MAX_GAIN)):
        raise ValueError('66 finite pool gains within [0,10] required')
    return g[:, None] * np.asarray(ORIGINAL_FIR)[None, :]


def gains_from_weights(weights):
    """Recover representable gains; reject other 330/shared-temporal kernels.

    Arithmetic reconstruction tolerance is serialization-only, never a search
    score tolerance; it cannot change the exact lexicographic selection rule.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (POOL_COUNT, 5) or not np.isfinite(w).all():
        raise ValueError('finite 66x5 FIR matrix required')
    gains = w[:, 0] / ORIGINAL_FIR[0]
    expected = pool_gain_weights(gains)
    if not np.allclose(w, expected, rtol=1e-14, atol=0):
        raise ValueError('weights do not preserve original FIR shape')
    return gains.copy()


def make_pool_gain_translator(manifest, gains=None):
    """Use existing validated manifest ordering; freeze gains as ordinary weights.

    No new consume/reset/state or body/neural inputs: callers receive the original
    FiniteMotorTranslator. Empty actuator mappings remain zero; shared actuators
    still share one MN pool and therefore one gain. No actual network is executed.
    """
    values = np.ones(POOL_COUNT) if gains is None else gains
    translator = FiniteMotorTranslator(manifest, weights=pool_gain_weights(values))
    if len(translator.pools) != POOL_COUNT:
        raise ValueError('this parameterization requires exactly 66 unique MN pools')
    return translator
