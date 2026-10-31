"""S2/S4 physical preparation and translation training, not a runtime controller.

Scores consume actual physics records. Optimizer only changes existing FIR
weights offline; no pose, clock or reference trajectory enters runtime commands.
Native force/length units are not a biological calibration.
"""
import numpy as np


def score_trajectory(qpos, qvel, contacts, foot_force, nonfoot_depth,
                     limit_excess, reference_qpos, weight_native,
                     dt_s=.0001, prepare_s=.2):
    q = np.asarray(qpos, float)
    v = np.asarray(qvel, float)
    c = np.asarray(contacts)
    f = np.asarray(foot_force, float)
    depth = np.asarray(nonfoot_depth, float)
    limit = np.asarray(limit_excess, float)
    ref = np.asarray(reference_qpos, float)
    n = len(q)
    if (q.ndim != 2 or q.shape[1] < 7 or v.ndim != 2 or
            v.shape[0] != n or v.shape[1] < 3 or c.shape != (n, 6) or
            f.shape != (n, 6, 3) or depth.shape != (n,) or limit.shape != (n,) or
            ref.ndim != 1 or ref.shape[0] < 7):
        raise ValueError('aligned full trajectory and six physical foot forces required')
    if (not all(np.isfinite(a).all() for a in (q, v, c, f, depth, limit, ref)) or
            not np.isfinite(weight_native) or weight_native <= 0 or
            not np.isfinite(dt_s) or dt_s <= 0 or
            not np.isfinite(prepare_s) or prepare_s <= 0 or ref[2] <= 0 or
            np.any(c < 0) or np.any(depth < 0) or np.any(limit < 0)):
        raise ValueError('finite physical records and positive scales required')
    split = int(round(prepare_s / dt_s))
    if not np.isclose(split * dt_s, prepare_s, rtol=0, atol=1e-12):
        raise ValueError('preparation duration must align with physics steps')
    if n != 2 * split + 1:
        raise ValueError('complete equally long preparation and evaluation required')
    norms = np.linalg.norm(q[:, 3:7], axis=1)
    rnorm = np.linalg.norm(ref[3:7])
    if rnorm == 0 or np.any(norms == 0):
        raise ValueError('nonzero physical quaternion required')
    height = q[:, 2] / ref[2]
    angle = 2 * np.arccos(np.clip(np.abs(q[:, 3:7] @ ref[3:7]) / (norms*rnorm), 0, 1))
    support_legs = (c > 0).sum(axis=1)
    vertical_force_ratio = f[:, :, 2].sum(axis=1) / weight_native
    # A force floor prevents contact-list-only airwalking from passing.
    # These are engineering gates, NOT calibrated animal behavior criteria.
    good = ((height >= .85) & (angle <= .35) & (support_legs >= 3) &
            (vertical_force_ratio >= .8) & (depth <= 1e-12) & (limit <= 1e-6))
    series = {
        'engineering': 10 * (~good).astype(float),
        'contactloss': np.maximum(3-support_legs, 0) / 3,
        'load_deficit': np.maximum(1-vertical_force_ratio, 0)**2,
        'height': 2*(np.maximum(1-height, 0)/.15)**2,
        'descent': (np.maximum(-v[:, 2], 0)/10)**2,
        'attitude': (angle/.35)**2,
        'nonfoot': 10*(depth/.01)**2,
        'limit': 10*(limit/.01)**2,
    }
    parts = {key: float(np.trapezoid(value, dx=dt_s)/(2*prepare_s))
             for key, value in series.items()}
    evaluation = good[split:]
    bad = np.flatnonzero(~evaluation)
    prefix = max(0, int(bad[0])-1)*dt_s*1000 if len(bad) else prepare_s*1000
    # Terminal 20ms must already support the body before handover is accepted.
    terminal_start = max(0, split-int(round(.02/dt_s)))
    result = dict(loss=sum(parts.values()), parts=parts, prefix_ms=float(prefix),
                  preparation_pass=bool(good[terminal_start:split+1].all()),
                  evaluation_pass=bool(evaluation.all()),
                  min_height_ratio=float(height.min()),
                  max_angle_rad=float(angle.max()),
                  max_joint_limit_excess=float(limit.max()),
                  max_nonfoot_depth=float(depth.max()),
                  min_evaluation_support_legs=int(support_legs[split:].min()),
                  min_evaluation_vertical_force_ratio=float(vertical_force_ratio[split:].min()),
                  initial_weight_native=float(weight_native),
                  physical_units='NOT_ESTABLISHED', behavior_claim='NONE')
    result['eligible'] = result['preparation_pass'] and result['evaluation_pass']
    if not np.isfinite(result['loss']):
        raise ValueError('nonfinite physical loss')
    return result


def propose_weights(weights, lp, lm, wp, wm, k, max_step=.02):
    """Bounded simultaneous secant step; clipped probes are not a gradient proof."""
    w, plus, minus = [np.asarray(a, float) for a in (weights, wp, wm)]
    if (w.ndim != 2 or w.shape[1] != 5 or plus.shape != w.shape or minus.shape != w.shape or
            not all(np.isfinite(a).all() for a in (w, plus, minus)) or
            any(np.any((a < 0) | (a > 4)) for a in (w, plus, minus)) or
            not np.isfinite([lp, lm, max_step]).all() or max_step <= 0 or
            isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0):
        raise ValueError('valid bounded FIR arrays and finite objectives required')
    # Zero entries remain zero: do not introduce previously absent bin weights.
    mask = w > 0
    if np.any(plus[~mask] != 0) or np.any(minus[~mask] != 0):
        raise ValueError('zero mask must be preserved')
    separation = plus-minus
    if np.any(separation[mask] == 0):
        raise ValueError('active probes need nonzero signed separation')
    secant = np.divide(lp-lm, separation, out=np.zeros_like(w), where=mask)
    raw = -.05/(k+1)**.602 * secant
    scale = min(1., max_step/max(float(np.max(np.abs(raw))), np.finfo(float).tiny))
    new = np.clip(w+scale*raw, np.finfo(float).eps, 4.)
    new[~mask] = 0
    return new
