"""Pure-array future capture validation. Not integrated with a dynamics runner."""
import numpy as np

STAT_COLUMNS = ('improvement', 'gradient', 'lineslope', 'nactive', 'nchange', 'neval', 'nupdate')


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name)
    return int(value)


def ints(value, size, name, lower=0, upper=None):
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in 'iu' or np.any(a < lower) or (upper is not None and np.any(a > upper)):
        raise ValueError(name)
    return a.copy()


def validate_capture(meta, fields):
    """Validate already copied endpoint arrays; never attests forward ordering."""
    try:
        return _validate(meta, fields)
    except KeyError as exc:
        raise ValueError(f'missing {exc.args[0]}') from exc


def _validate(meta, fields):
    if meta['mujoco_version'] != '3.9.0' or integer(meta['solver'], 'solver') != 2 or integer(meta['noslip_iterations'], 'noslip') != 0:
        raise ValueError('unsupported solver semantics')
    if not isinstance(meta['islands_enabled'], (bool, np.bool_)):
        raise ValueError('islands_enabled')
    n = integer(meta['nefc'], 'nefc')
    ni = integer(meta['nisland'], 'nisland')
    cap = integer(meta['mjNSOLVER'], 'mjNSOLVER', 1)
    slots = integer(meta['mjNISLAND'], 'mjNISLAND', 1)
    if ni > slots:
        raise ValueError('island stats capacity')
    raw = ints(fields['efc_state_global_raw'], n, 'global state', upper=4)
    if not n:
        if ni:
            raise ValueError('islands without constraints')
        return dict(final_state=raw.copy(), global_state_raw=raw, source='empty', mismatch_count=0, statistics=[])
    island_path = bool(meta['islands_enabled']) and ni > 0
    if island_path:
        starts = ints(fields['island_iefcadr'], ni, 'island addresses', upper=n-1)
        counts = ints(fields['island_nefc'], ni, 'island row counts', lower=1, upper=n)
        owner = np.zeros(n, dtype=int)
        for start, count in zip(starts, counts):
            start, count = int(start), int(count)
            if start+count > n:
                raise ValueError('island range')
            owner[start:start+count] += 1
        if np.any(owner != 1):
            raise ValueError('island coverage')
        forward = ints(fields['map_iefc2efc'], n, 'forward map', upper=n-1)
        inverse = ints(fields['map_efc2iefc'], n, 'inverse map', upper=n-1)
        # Cast only after range checks, including uint64 overflow protection.
        forward, inverse = forward.astype(np.int64), inverse.astype(np.int64)
        if not np.array_equal(np.sort(forward), np.arange(n)) or not np.array_equal(forward[inverse], np.arange(n)):
            raise ValueError('non-bijective or non-inverse mapping')
        local = ints(fields['iefc_state'], n, 'island state', upper=4)
        final = local[inverse].copy()
        source = 'island_primal_observed'
        used = ni
    else:
        final = raw.copy()
        source = 'monolithic_primal_observed'
        used = 1
    nit = ints(fields['solver_niter'], slots, 'solver_niter')
    stat = np.asarray(fields['solver_stats_flat'])
    if stat.shape != (slots*cap, len(STAT_COLUMNS)) or stat.dtype.kind not in 'fiu' or not np.isfinite(stat).all():
        raise ValueError('solver stats storage')
    statistics = []
    for island in range(used):
        total = int(nit[island]); saved = min(total, cap)
        active = stat[island*cap:island*cap+saved]
        counts = active[:, 3:]
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError('noninteger or negative active statistics counts')
        statistics.append(dict(slot=island, niter_total=total, niter_saved=saved, truncated=total>cap,
                               values=active.copy()))
    return dict(final_state=final, global_state_raw=raw, source=source,
                mismatch_count=int(np.sum(raw != final)), statistics=statistics)
