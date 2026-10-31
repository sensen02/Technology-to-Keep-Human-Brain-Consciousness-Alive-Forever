"""Strict JSON evidence adapter; never turns measurement state into acceptance."""
import math
import numpy as np
from phase_metrology import measure_phase


def json_value(value):
    """Undefined numeric samples serialize as null; phase_defined preserves reason."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=json_value(value.real), imag=json_value(value.imag))
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        raise ValueError('infinite evidence value')
    return value


def collect_phase_evidence(time_s, analytic_a, analytic_b, *, endpoint_samples):
    """Called inside candidate functions with their actual same-call ha/hb.

    Exceptions propagate. This schema has no passed/certified boolean: callers
    must not treat successful collection or NOT_ASSESSED as locking success.
    """
    measurement = measure_phase(time_s, analytic_a, analytic_b,
                                endpoint_samples=endpoint_samples)
    return json_value(dict(schema='phase-metrology-evidence-v1',
                           scope='ADDITIONAL_EVIDENCE_NOT_CERTIFICATION',
                           analytic_A=analytic_a, analytic_B=analytic_b,
                           defined_at_samples=measurement['phase_defined'],
                           measurement=measurement))
