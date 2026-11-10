"""Fixed engineering DNp09 candidate-pair current; not brain-generated output.

Identity audited in sibling PORTS.json. Laterality, homology, optogenetic
 equivalence and biological function are not established. No body/time/phase API.
"""
import numbers
import numpy as np

N_NEURONS = 20000
LOCAL_INDICES = (1414, 13170)
ROOT_IDS = (720575941433155799, 720575941566493282)
MAX_AMPLITUDE_NA = 0.05


def named_current(amplitude_nA: float) -> np.ndarray:
    """Return a new float64 (20000,) vector, equal current at the two ports.

    Reject non-real scalars, booleans, nonfinite values and values outside
    inclusive [0, 0.05] nA. Zero is the numeric no-injection control.
    """
    if isinstance(amplitude_nA, (bool, np.bool_)) or not isinstance(amplitude_nA, numbers.Real):
        raise ValueError('amplitude_nA must be a real scalar in [0, 0.05]')
    amplitude = float(amplitude_nA)
    if not np.isfinite(amplitude) or not 0.0 <= amplitude <= MAX_AMPLITUDE_NA:
        raise ValueError('amplitude_nA must be finite and in [0, 0.05] nA')
    current = np.zeros(N_NEURONS, dtype=np.float64)
    current[list(LOCAL_INDICES)] = amplitude
    return current
