"""Offline S4/S5 support-target eligibility, not behavioral optimization.

Samples are interval values on one shared, contiguous edge vector. Eligibility
is only an engineering preflight; even a synthetic eligible fixture establishes
neither physical support nor behavioral success. This is a necessary, not
sufficient, support eligibility check: it proves no target provenance and grants
no behavioral optimization authorization. No model/body is loaded here.
"""
from math import fsum, isfinite


_ALIGNMENT = 'next_interval_after_completed_source_block'

import numpy as np


_WINDOW_START_S = 0.05
_WINDOW_END_S = 0.55
_ENDPOINT_TOLERANCE_S = 1e-12
_FIELDS = frozenset({
    "sample_edges_s", "foot_normal", "effective_nonfoot", "root_height",
    "tilt_rad", "model_weight", "initial_root_height",
})


def _real_array(name, value, ndim):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if array.ndim != ndim or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a {ndim}-dimensional real numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers before conversion")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def qualify_training_dataset(metadata, arrays):
    """Pre-optimizer gate. Behavior optimization stays disabled without provenance.

    5ms datasets require actual per-block source/target timestamps, not merely
    a declaration. Legacy 1-tick INTERFACE_TEST remains explicitly separate.
    BEHAVIOR data must supply support_<field> raw interval statistics. Passing
    support is necessary only and cannot establish target-label provenance.
    """
    if not isinstance(metadata, dict):
        raise ValueError('metadata must be an object with target_kind INTERFACE_TEST or BEHAVIOR')
    kind = metadata.get('target_kind')
    if kind not in ('INTERFACE_TEST', 'BEHAVIOR'):
        raise ValueError('target_kind must be INTERFACE_TEST or BEHAVIOR (behavior optimization disabled)')
    ticks = metadata.get('command_ticks', 1)
    if isinstance(ticks, (bool, np.bool_)) or not isinstance(ticks, (int, np.integer)) or ticks not in (1, 10):
        raise ValueError('command_ticks must be integer 1 or 10')
    if kind == 'BEHAVIOR' and ticks != 10:
        raise ValueError('BEHAVIOR requires 5ms command_ticks=10')
    spikes = np.asarray(arrays['spikes'])
    targets = _real_array('targets', arrays['targets'], 2)
    if spikes.ndim != 2 or len(spikes) == 0 or len(spikes) % ticks:
        raise ValueError('spikes require nonempty complete source blocks')
    blocks = len(spikes)//ticks
    if targets.ndim != 2 or len(targets) != blocks:
        raise ValueError('targets must have one row per completed source block')
    timestamp_names = ('source_start_s', 'source_end_s', 'target_start_s', 'target_end_s')
    if ticks == 10 or any(name in arrays for name in timestamp_names):
        if metadata.get('target_alignment') != _ALIGNMENT:
            raise ValueError('5ms targets require explicit next_interval_after_completed_source_block alignment')
        dt = ticks*.0005
        expected = (np.arange(blocks)*dt, (np.arange(blocks)+1)*dt,
                    (np.arange(blocks)+1)*dt, (np.arange(blocks)+2)*dt)
        for name, wanted in zip(timestamp_names, expected):
            if name not in arrays:
                raise ValueError('missing command timestamp field: '+name)
            actual = _real_array(name, arrays[name], 1)
            if actual.shape != (blocks,) or not np.allclose(actual, wanted, rtol=0, atol=1e-12):
                raise ValueError('misaligned command timestamps: '+name)
    support_arrays = {name[len('support_'):]: arrays[name]
                      for name in arrays if name.startswith('support_')}
    support = None
    if kind == 'BEHAVIOR' or support_arrays:
        support = qualify_behavior_support(**support_arrays)
        if not support['eligible']:
            raise ValueError('support qualification failed: '+', '.join(support['reasons']))
    if kind == 'BEHAVIOR':
        raise ValueError('support passed but target provenance NOT_ESTABLISHED; behavioral optimization disabled; use INTERFACE_TEST only for synthetic labels')
    return dict(target_kind=kind, command_ticks=int(ticks),
                clock='5ms_next_interval' if ticks == 10 else 'legacy_tick_INTERFACE_TEST',
                support_qualification=support,
                target_provenance='SYNTHETIC_INTERFACE_TEST_NOT_BEHAVIOR',
                behavior_success=False, behavioral_optimization_authorized=False)


def qualify_behavior_support(**fields):
    """Return a dict with eligible, reasons, measured, scope, behavior_success.

    Required fields: sample_edges_s[N+1] spanning exactly [0.05, 0.55]
    seconds (endpoint absolute tolerance 1e-12, no relative tolerance),
    foot_normal[N], effective_nonfoot[N] (binary), root_height[N],
    tilt_rad[N] (nonnegative tilt magnitude), and positive scalar model_weight
    and initial_root_height. Each sample applies throughout its edge interval;
    shared strictly increasing edges enforce contiguous coverage. Interval values
    must be mean foot force, any effective nonfoot flag, minimum root height,
    and maximum tilt over each interval; sparse point snapshots cannot establish
    continuous-window eligibility. Caller/collector must establish these semantics.
    No cropping,
    interpolation, extrapolation, or caller-supplied pass flag is accepted.

    Eligibility requires time-weighted foot_normal/model_weight >= 0.8,
    effective nonfoot duration == 0, min root_height/initial_root_height >=
    0.85, and max tilt_rad <= 0.35. Thresholds have no relaxation tolerance.
    Malformed/missing/unknown/nonfinite inputs raise ValueError (fail closed).
    This function does not authorize optimization or establish support success.
    """
    missing, unknown = _FIELDS - fields.keys(), fields.keys() - _FIELDS
    if missing or unknown:
        raise ValueError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")

    edges = _real_array("sample_edges_s", fields["sample_edges_s"], 1)
    if edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("sample_edges_s must have strictly increasing N+1 edges, N >= 1")
    if (abs(float(edges[0]) - _WINDOW_START_S) > _ENDPOINT_TOLERANCE_S
            or abs(float(edges[-1]) - _WINDOW_END_S) > _ENDPOINT_TOLERANCE_S):
        raise ValueError("sample_edges_s must cover exactly the full 0.05–0.55 s window")

    arrays = {}
    for name in ("foot_normal", "effective_nonfoot", "root_height", "tilt_rad"):
        array = _real_array(name, fields[name], 1)
        if array.size != edges.size - 1:
            raise ValueError(f"{name} must have N entries for N+1 sample edges")
        arrays[name] = array
    if not np.isin(arrays["effective_nonfoot"], (0, 1)).all():
        raise ValueError("effective_nonfoot must be binary (0 or 1)")
    if np.any(arrays["foot_normal"] < 0) or np.any(arrays["tilt_rad"] < 0):
        raise ValueError("foot_normal and tilt_rad magnitudes must be nonnegative")

    scalars = {}
    for name in ("model_weight", "initial_root_height"):
        value = float(_real_array(name, fields[name], 0))
        if value <= 0 or isinstance(fields[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive real scalar")
        scalars[name] = value

    durations = np.diff(edges)
    duration = fsum(map(float, durations))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios = arrays["foot_normal"] / scalars["model_weight"]
        height_ratio = float(np.min(arrays["root_height"]) / scalars["initial_root_height"])
    if not np.isfinite(ratios).all() or not isfinite(height_ratio):
        raise ValueError("derived normalized measurements must be finite")
    # Center the weighted mean so a constant threshold trace stays exactly at
    # that threshold even when decimal edge differences round asymmetrically.
    base_ratio = float(np.min(ratios))
    foot_ratio = base_ratio + fsum((float(r) - base_ratio) * (float(dt) / duration)
                                   for r, dt in zip(ratios, durations))
    nonfoot_duration = fsum(float(dt) for dt, flag in
                           zip(durations, arrays["effective_nonfoot"]) if flag == 1)
    max_tilt = float(np.max(arrays["tilt_rad"]))
    if not isfinite(foot_ratio):
        raise ValueError("derived foot support measurement must be finite")

    reasons = []
    if foot_ratio < 0.8:
        reasons.append("time_weighted_foot_normal_ratio_below_0.8")
    if nonfoot_duration != 0:
        reasons.append("effective_nonfoot_duration_nonzero")
    if height_ratio < 0.85:
        reasons.append("min_root_height_ratio_below_0.85")
    if max_tilt > 0.35:
        reasons.append("max_tilt_rad_above_0.35")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "measured": {
            "window_start_s": float(edges[0]),
            "window_end_s": float(edges[-1]),
            "duration_s": duration,
            "sample_count": int(durations.size),
            "time_weighted_foot_normal_ratio": foot_ratio,
            "effective_nonfoot_duration_s": nonfoot_duration,
            "min_root_height": float(np.min(arrays["root_height"])),
            "min_root_height_ratio": height_ratio,
            "max_tilt_rad": max_tilt,
        },
        "scope": "engineering_preflight_only",
        "behavior_success": False,
    }
