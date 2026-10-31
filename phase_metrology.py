"""Offline dual-path phase evidence. No interpolation, amplitude threshold or lock decision.

Finite samples alone do not identify continuous phase winding. An external sampling
bound would be needed to establish it. Existing low-amplitude flags may be passed
through unchanged; they never mask observations here. Exact zero is undefined.
"""
import numpy as np


def _segments(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def measure_phase(time_s, analytic_a, analytic_b, *, endpoint_samples,
                  existing_low_amplitude=None):
    """Return radians for paths and degrees for original endpoint statistics.

    Path gauges restart independently on each nonzero segment. Statistics spanning
    a sampled zero are None. Legacy numerical outputs are retained separately,
    explicitly nonphysical at zero. No state returned by this API means LOCKED.
    """
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(analytic_a, dtype=complex)
    b = np.asarray(analytic_b, dtype=complex)
    if (t.ndim != 1 or len(t) < 2 or a.shape != t.shape or b.shape != t.shape
            or not np.isfinite(t).all() or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.all(np.diff(t) > 0)):
        raise ValueError('finite equal-length 1D arrays and strictly increasing times required')
    if (not isinstance(endpoint_samples, (int, np.integer))
            or not 1 <= endpoint_samples <= len(t)//2):
        raise ValueError('endpoint_samples must be an integer in [1, n//2]')
    low = None if existing_low_amplitude is None else np.asarray(existing_low_amplitude, dtype=bool)
    if low is not None and low.shape != t.shape:
        raise ValueError('existing_low_amplitude must match time shape')
    # Unit complex products avoid overflow/underflow of amplitude multiplication.
    defined = (np.abs(a) != 0) & (np.abs(b) != 0)
    principal = np.full(len(t), np.nan)
    principal[defined] = np.angle((a[defined]/np.abs(a[defined])) *
                                 np.conj(b[defined]/np.abs(b[defined])))
    product = np.full(len(t), np.nan)
    separate = np.full(len(t), np.nan)
    turns = np.full(len(t), np.nan)
    intervals = []
    for start, stop in _segments(defined):
        sl = slice(start, stop)
        product[sl] = np.unwrap(principal[sl])
        separate[sl] = np.unwrap(np.angle(a[sl])) - np.unwrap(np.angle(b[sl]))
        # Compare transport, not the arbitrary initial principal-angle gauge.
        difference = separate[sl] - product[sl]
        difference -= difference[0]
        turns[sl] = np.rint(difference/(2*np.pi))
        for j in np.flatnonzero(np.diff(turns[sl]) != 0) + start:
            intervals.append(dict(start_index=int(j), end_index=int(j+1),
                                  start_s=float(t[j]), end_s=float(t[j+1]),
                                  change_turns=int(turns[j+1]-turns[j])))
    def stats(u):
        advance = float(np.degrees(np.median(u[-endpoint_samples:]) -
                                   np.median(u[:endpoint_samples])))
        return dict(advance_deg=advance, halfturn_count=int(np.floor(abs(advance)/180)))
    # Reproduce both archived numerical conventions; never silently choose one.
    legacy_product = np.unwrap(np.angle(a*np.conj(b)))
    legacy_separate = np.unwrap(np.unwrap(np.angle(a))-np.unwrap(np.angle(b)))
    valid_all = bool(defined.all())
    disagreement = bool(np.any(np.abs(turns[defined]) > 0))
    status = ('UNDEFINED_AT_SAMPLED_ZERO' if not valid_all else
              'PATH_DISAGREEMENT_CONTINUOUS_BRANCH_UNIDENTIFIED' if disagreement else
              'PATHS_AGREE_CONTINUOUS_BRANCH_UNIDENTIFIED')
    return dict(time_s=t.copy(), principal_rad=principal,
                product_unwrapped_rad=product, separate_unwrapped_rad=separate,
                separate_minus_product_turns=turns, phase_defined=defined,
                existing_low_amplitude=None if low is None else low.copy(),
                branch_change_intervals=intervals, status=status,
                continuous_branch_identified=False, locking_status='NOT_ASSESSED',
                product_statistics=stats(product) if valid_all else None,
                separate_statistics=stats(separate) if valid_all else None,
                legacy_product_rad=legacy_product, legacy_separate_reunwrapped_rad=legacy_separate,
                legacy_product_statistics=stats(legacy_product),
                legacy_separate_reunwrapped_statistics=stats(legacy_separate),
                legacy_statistics_defined_at_samples=valid_all,
                sampled_zero_indices=np.flatnonzero(~defined).tolist())
