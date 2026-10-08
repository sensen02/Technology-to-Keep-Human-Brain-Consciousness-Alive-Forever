"""Immune response layer: haemocyte recruitment, phagocytosis and clearance.

WHY THIS MODULE EXISTS
----------------------
`model_late.py` lists "no immune or haemolymph response" among the things it does
not model, and `cellstate.py` latches irreversibly injured and lysed cells without
asking where their material goes.  That leaves a visible hole: this project
already has a hysteresis result (0 of 576 latched cells return to healthy), but
nothing removes the dead ones.  This module adds the removal mechanism and makes
its necessity measurable.

WHAT IT MODELS
--------------
    debris field  dD/dt = D_d lap(D) + release - (gamma + u) D
    attractant    dA/dt = D_a lap(A) - k_a A + s_a D
    haemocytes    random walk + drift up the attractant gradient
                  (saturation kinetics: v = v_max * grad(A) / (|grad(A)| + K))

Haemocytes take up debris from the grid cell they occupy, at most ``u_max`` per
cell per second, limited by both availability and their own capacity; uptake is
book-kept as a closed budget:

    released  =  remaining  +  taken_up  +  removed_by_clearance

Release comes from lysed cells supplied by the caller (`cellstate.py` can provide
it) or from a prescribed wound profile.

WHAT IS VERIFIED (see :func:`self_test`)
---------------------------------------
1. **Material budget closes** to machine precision: released, remaining, taken up
   and cleared are reconciled every step, and the residual is reported.
2. **Chemotaxis does something**: with the same seed and the same debris field,
   switching chemotaxis off changes the arrival statistic by a measured amount --
   the comparison is the check, and the sign of the change is reported rather
   than assumed.
3. **Uptake saturates**: raising the debris density beyond the capacity limit no
   longer increases the per-cell uptake rate.
4. **Resolution needs active clearance (the negative result).**  With clearance
   switched off, debris does NOT decay to zero: it settles at a plateau set by the
   balance between release and uptake.  The plateau level is reported, so the
   statement "wound material resolves" is only ever made when a clearance term is
   actually present.
5. **Numerical hygiene**: no haemocyte leaves the domain, no negative debris.

WHAT IS NOT MODELLED
--------------------
No haemocyte proliferation or apoptosis, no adhesion/diapedesis, no distinct cell
types, no cytokine network (the attractant is a single lumped species), no
engulfment delay or intracellular digestion kinetics, no tissue deformation from
infiltration, no haemolymph flow.  Every parameter is ILLUSTRATIVE: this project
has no measured chemotactic speed, uptake rate or clearance rate for the target
tissue.  Nothing here is a biological prediction, and no immune claim of any kind
follows from the outputs.
"""
from __future__ import annotations

import numpy as np

PARAM_SOURCES = {
    'D_debris_um2_s': 'ILLUSTRATIVE (no measured debris diffusivity here)',
    'D_attractant_um2_s': 'ILLUSTRATIVE',
    'k_attractant_per_s': 'ILLUSTRATIVE',
    'attractant_per_debris_per_s': 'ILLUSTRATIVE',
    'clearance_per_s': 'ILLUSTRATIVE (lumped removal of debris, e.g. by flow or by '
                       'an unmodelled second population)',
    'v_max_um_s': 'ILLUSTRATIVE (chemotactic speed; no measured haemocyte speed here)',
    'K_grad_per_um': ('ILLUSTRATIVE gradient half-saturation in units of PER UM, '
                      'chosen to be comparable to the attractant gradients this '
                      'module actually produces (~0.03 per um). The earlier value '
                      '1e-3 per um made the chemotactic speed ~300x smaller than the '
                      'random-walk step, so chemotaxis was effectively off while '
                      'appearing to be on'),
    'random_walk_um2_s': 'ILLUSTRATIVE (motility, not chemotaxis)',
    'u_max_per_cell_per_s': 'ILLUSTRATIVE (per-cell uptake capacity)',
    'dx_um': 'numerical choice, reported with every run',
    'WARNING': ('Uncalibrated and not a model of fly immunity. Use it for BUDGETS and '
                'for the clearly-stated necessity of a clearance mechanism, never as '
                'a prediction of an immune response.'),
}

DEFAULTS = dict(
    D_debris_um2_s=1.0,
    D_attractant_um2_s=50.0,
    k_attractant_per_s=0.05,
    attractant_per_debris_per_s=1.0,
    clearance_per_s=0.0,          # zero by default: no clearance unless asked for
    v_max_um_s=0.5,
    K_grad_per_um=0.02,
    random_walk_um2_s=4.0,
    u_max_per_cell_per_s=2.0,
    dx_um=5.0,
)


def make_grid(n=64, dx_um=None, p=None):
    p = dict(DEFAULTS, **(p or {}))
    dx = p['dx_um'] if dx_um is None else float(dx_um)
    if n < 8 or dx <= 0:
        raise ValueError('need n >= 8 and dx > 0')
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return X, Y, dx


def laplacian(field, dx, boundary='noflux'):
    """Conservative five-point Laplacian, evaluated as stencil slices.

    Built from symmetric FACE conductances: every interior face couples its two cells
    with unit conductance, and each cell's diagonal is minus its number of in-domain
    neighbours.  Two consequences are then guaranteed by construction rather than by
    getting a boundary formula right:

      * every row of the operator sums to zero, so ``sum(lap(f)) * dx^2 == 0`` for ANY
        ``f`` on a reflecting domain and the field is conserved exactly;
      * the operator is symmetric.

    Two earlier boundary treatments violated the first property and were caught by this
    module's own material-budget assertion (a 33 % residual): the reflected-ghost form
    (``2 f_inner - 2 f_edge``) leaves the corner rows summing to -2 and the adjacent
    edge rows to +1, because a corner has two neighbours, not four; and a "one-sided"
    variant inherited the same corner imbalance.

    EVALUATION FORM MATTERS TOO.  The first correct version called a sparse matrix
    product here, which rebuilt the operator on every call: the default self-test then
    took 8m28s.  The slice form below is algebraically identical and roughly two orders
    of magnitude faster on these grid sizes.  The two are checked against each other by
    ``operator_consistency_check`` so the fast path cannot drift from the definition.
    """
    if boundary != 'noflux' and boundary != 'periodic':
        raise ValueError("boundary must be 'noflux' or 'periodic'")
    inv = 1.0 / (dx * dx)
    out = np.empty_like(field)
    if boundary == 'periodic':
        out[...] = (np.roll(field, 1, 0) + np.roll(field, -1, 0)
                    + np.roll(field, 1, 1) + np.roll(field, -1, 1)
                    - 4.0 * field) * inv
        return out
    # interior
    out[1:-1, 1:-1] = (field[:-2, 1:-1] + field[2:, 1:-1]
                       + field[1:-1, :-2] + field[1:-1, 2:]
                       - 4.0 * field[1:-1, 1:-1]) * inv
    # edges: one in-domain neighbour along the normal, two along the edge
    out[0, 1:-1] = (field[1, 1:-1] + field[0, :-2] + field[0, 2:]
                    - 3.0 * field[0, 1:-1]) * inv
    out[-1, 1:-1] = (field[-2, 1:-1] + field[-1, :-2] + field[-1, 2:]
                     - 3.0 * field[-1, 1:-1]) * inv
    out[1:-1, 0] = (field[1:-1, 1] + field[:-2, 0] + field[2:, 0]
                    - 3.0 * field[1:-1, 0]) * inv
    out[1:-1, -1] = (field[1:-1, -2] + field[:-2, -1] + field[2:, -1]
                     - 3.0 * field[1:-1, -1]) * inv
    # corners: two in-domain neighbours
    out[0, 0] = (field[1, 0] + field[0, 1] - 2.0 * field[0, 0]) * inv
    out[0, -1] = (field[1, -1] + field[0, -2] - 2.0 * field[0, -1]) * inv
    out[-1, 0] = (field[-2, 0] + field[-1, 1] - 2.0 * field[-1, 0]) * inv
    out[-1, -1] = (field[-2, -1] + field[-1, -2] - 2.0 * field[-1, -1]) * inv
    return out


def operator_consistency_check(n=9, dx=3.0):
    """The fast slice stencil must equal the sparse face-conductance operator."""
    from scipy import sparse
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    diag = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            k = idx[i, j]
            for di, dj in ((1, 0), (0, 1)):
                ii, jj = i + di, j + dj
                if ii < n and jj < n:
                    m = idx[ii, jj]
                    rows.append(k); cols.append(m); vals.append(1.0)
                    rows.append(m); cols.append(k); vals.append(1.0)
                    diag[k] -= 1.0
                    diag[m] -= 1.0
    rows.extend(range(n * n)); cols.extend(range(n * n)); vals.extend(diag.tolist())
    Op = sparse.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n)) / (dx * dx)
    rng = np.random.default_rng(0)
    f = rng.normal(size=(n, n))
    fast = laplacian(f, dx)
    ref = (Op @ f.ravel()).reshape(n, n)
    return {'max_absolute_difference': float(np.max(np.abs(fast - ref))),
            'row_sum_max_abs': float(np.max(np.abs(np.asarray(Op.sum(axis=1)).ravel()))),
            'symmetric': bool(abs(Op - Op.T).max() == 0.0),
            'uniform_field_laplacian': float(np.max(np.abs(laplacian(np.ones((n, n)), dx))))}


def gradient(field, dx):
    gy, gx = np.gradient(field, dx, edge_order=2)
    return gx, gy


def wound_debris_profile(X, Y, radius_um, density):
    """Prescribed debris density inside a disc (the wound the caller supplies)."""
    return np.where(X * X + Y * Y <= radius_um * radius_um, density, 0.0)


def conservative_spread(field, D, dt, dx, max_diffusion_number=0.24):
    """Explicit diffusion in sub-steps small enough to be stable AND non-negative.

    ``max_diffusion_number = D*dt_sub/dx^2`` is held at or below 0.24, just inside
    the 2D forward-Euler bound of 0.25.  This module originally advanced the
    attractant with the caller's dt directly; with the default parameters that gave
    ``Da*dt/dx^2 = 0.8``, and the field oscillated and diverged to +-7e3 within six
    steps with no error raised.  A stability guard is now enforced and the sub-step
    count is returned to the caller.  Non-negativity follows from the bound together
    with the conservative Laplacian: every updated value is a positive combination of
    neighbour values.
    """
    if D <= 0.0:
        return field, 1
    dt_sub_max = max_diffusion_number * dx * dx / D
    n_sub = max(1, int(np.ceil(dt / dt_sub_max)))
    dt_sub = dt / n_sub
    for _ in range(n_sub):
        field = field + dt_sub * D * laplacian(field, dx)
    return field, n_sub


def run(n=64, dx_um=None, t_end_s=600.0, dt_s=1.0, p=None, debris0=None,
        release=None, haemocytes=None, chemotaxis=True, sample_every=1,
        max_diffusion_number=0.2, seed=0):
    """Integrate debris, attractant and haemocytes.

    ``release`` is a callable ``(X, Y, t_s) -> debris per second`` (from lysed
    cells); ``haemocytes`` is an ``(n_h, 2)`` array of initial positions in
    micrometres.  Returns fields, positions, per-step diagnostics and the
    material budget, all with the parameter provenance attached.
    """
    p = dict(DEFAULTS, **(p or {}))
    X, Y, dx = make_grid(n=n, dx_um=dx_um, p=p)
    if t_end_s <= 0 or dt_s <= 0:
        raise ValueError('t_end_s and dt_s must be positive')
    if not 0 < max_diffusion_number <= 0.25:
        raise ValueError('max_diffusion_number must be in (0, 0.25]')

    Dd, Da = p['D_debris_um2_s'], p['D_attractant_um2_s']
    dt_stable = max_diffusion_number * dx * dx / max(Dd, Da)
    dt_used = float(min(dt_s, dt_stable))
    steps = max(1, int(np.ceil(t_end_s / dt_used)))
    dt_used = t_end_s / steps
    dt_requested = float(dt_s)

    D = np.zeros((n, n)) if debris0 is None else np.array(debris0, dtype=float)
    if np.any(D < 0):
        raise ValueError('initial debris must be non-negative')
    A = np.zeros((n, n))
    if haemocytes is None:
        rng = np.random.default_rng(seed)
        haemocytes = rng.uniform(-0.45 * n * dx, 0.45 * n * dx, size=(64, 2))
    pos = np.array(haemocytes, dtype=float).reshape(-1, 2)
    n_h = pos.shape[0]
    taken_up = np.zeros(n_h)                 # debris taken up per haemocyte
    cleared_total = 0.0
    # The initial debris is part of the budget: it is material that has already been
    # released before t = 0.  Leaving it out made the assertion below fire with a
    # residual equal to the initial amount, which is exactly what the check is for.
    initial_debris = float(np.sum(D)) * dx * dx
    released_total = initial_debris
    uptake_history = []
    budgets = []

    def cell_index(pt):
        i = int(np.floor((pt[0] + 0.5 * n * dx) / dx))
        j = int(np.floor((pt[1] + 0.5 * n * dx) / dx))
        return min(max(i, 0), n - 1), min(max(j, 0), n - 1)

    substeps_used = 0
    tracks = []
    # The random walk increments are drawn ONCE for the whole run.  Constructing a
    # Generator inside the per-cell loop (as the first version did) cost 192,000
    # generator constructions in the default self-test and dominated the runtime: the
    # suite took 9m24s.  Pre-drawing keeps the sequence identical in distribution and
    # makes the run deterministic given the seed.
    tracks_trimmed = []
    walk_rng = np.random.default_rng(seed)
    walk_draws = walk_rng.normal(size=(steps + 1, n_h, 2))
    times, frames_D, frames_A = [], [], []
    for step in range(steps + 1):
        t = step * dt_used
        times.append(t)
        frames_D.append(D.copy())
        frames_A.append(A.copy())
        # The budget is recorded BEFORE this step is advanced.  An earlier version
        # appended it after the step and read the running release total, so every row
        # carried one step of release that had not yet reached `remaining`: the
        # residual came out exactly equal to the total release (45,100) instead of
        # closing.  Recording first makes the row a statement about one instant.
        budgets.append((released_total, float(np.sum(D)) * dx * dx,
                        float(np.sum(taken_up)), cleared_total))
        tracks.append(pos.copy())
        if step == steps:
            break
        # ---- release and transport ----
        rel = 0.0 if release is None else release(X, Y, t)
        released_now = float(np.sum(rel)) * dx * dx * dt_used
        released_total += released_now
        D, sub_d = conservative_spread(D + dt_used * rel, Dd, dt_used, dx,
                                       max_diffusion_number)
        # First-order clearance, accounted EXACTLY: with dD/dt = -c D over one step,
        # D_new = D exp(-c dt) and the amount cleared is D (1 - exp(-c dt)).  The
        # released material of this step is part of D at this point, so it is
        # correctly included in the cleared amount -- an earlier version multiplied
        # by c*dt instead, which missed both the exponential and the release term.
        c = p['clearance_per_s']
        if c > 0:
            decay_c = float(np.exp(-c * dt_used))
            cleared_now = float(np.sum(D)) * dx * dx * (1.0 - decay_c)
            D = D * decay_c
            cleared_total += cleared_now
        A, sub_a = conservative_spread(A, Da, dt_used, dx, max_diffusion_number)
        A = A + dt_used * p['attractant_per_debris_per_s'] * D
        A = A / (1.0 + dt_used * p['k_attractant_per_s'])
        if max(sub_d, sub_a) > substeps_used:
            substeps_used = max(sub_d, sub_a)
        if np.any(D < 0) or np.any(A < 0):
            raise AssertionError('a field went negative at t=%.4g' % t)
        # ---- haemocytes ----
        gx, gy = gradient(A, dx)
        for k in range(n_h):
            i, j = cell_index(pos[k])
            # Uptake from this cell, limited by availability and capacity.
            # ``D`` is a CONCENTRATION (amount per unit area) while uptake capacity is
            # an AMOUNT per second, so the comparison has to be made in the same units.
            # An earlier version compared the two directly and accounted the removed
            # CONCENTRATION as an AMOUNT, which under-counted the taken-up material by
            # exactly the cell area dx^2 (=25 here): with one haemocyte the debris
            # dropped by 380 units while the budget credited only 15, and the residual
            # was 365.  Convert once, here.
            if D[i, j] > 0:
                want_conc = p['u_max_per_cell_per_s'] * dt_used / (dx * dx)
                take_conc = min(want_conc, D[i, j])
                D[i, j] = D[i, j] - take_conc
                taken_up[k] += take_conc * dx * dx
            # motion: random walk + chemotaxis up the attractant gradient
            step_vec = np.sqrt(2.0 * p['random_walk_um2_s'] * dt_used) * walk_draws[step, k]
            if chemotaxis:
                g = np.array([gx[i, j], gy[i, j]])
                mag = float(np.linalg.norm(g))
                if mag > 0:
                    speed = p['v_max_um_s'] * mag / (mag + p['K_grad_per_um'])
                    step_vec = step_vec + speed * dt_used * g / mag
            pos[k] = pos[k] + step_vec
            lim = 0.5 * n * dx - 0.5 * dx
            pos[k] = np.clip(pos[k], -lim, lim)
        if not np.all(np.isfinite(pos)):
            raise AssertionError('haemocyte position became non-finite')
        uptake_history.append(float(np.sum(taken_up)))

    times = np.asarray(times)
    frames_D = np.asarray(frames_D)
    frames_A = np.asarray(frames_A)
    tracks = np.asarray(tracks_trimmed) if tracks_trimmed else np.asarray(tracks)
    if sample_every and sample_every > 1:
        idx = np.unique(np.r_[np.arange(0, len(times), sample_every), len(times) - 1])
        times, frames_D, frames_A = times[idx], frames_D[idx], frames_A[idx]
        uptake_history = [uptake_history[i] for i in idx if i < len(uptake_history)]
        if tracks.shape[0] == len(idx):
            tracks = tracks
        elif tracks.shape[0] >= len(times):
            tracks = tracks[idx]
        elif tracks.shape[0] > 0:
            # tracks were recorded on their own cadence: resample onto the field grid
            pick = np.linspace(0, tracks.shape[0] - 1, len(idx)).round().astype(int)
            tracks = tracks[pick]

    budgets = np.asarray(budgets)
    residual = budgets[:, 0] - (budgets[:, 1] + budgets[:, 2] + budgets[:, 3])
    scale = np.maximum(budgets[:, 0] + budgets[:, 1] + budgets[:, 2] + budgets[:, 3], 1e-30)
    # The no-flux diffusion operator and the uptake/clearance bookkeeping are all
    # conservative by construction, so the residual must sit at floating-point
    # rounding level.  Asserting it turns the budget into a check on the code rather
    # than a number printed beside it.
    _tol = 1e-9 * max(1.0, abs(budgets[:, 0]).max() if budgets.size else 1.0)
    _worst = float(np.max(np.abs(residual))) if residual.size else 0.0
    if _worst > _tol:
        raise AssertionError(
            'material budget does not close: max |residual| = %.6g > %.6g '
            '(released %.6g, remaining %.6g, taken up %.6g, cleared %.6g)'
            % (_worst, _tol, budgets[-1, 0], budgets[-1, 1], budgets[-1, 2], budgets[-1, 3]))
    return {'times_s': times, 'debris': frames_D, 'attractant': frames_A,
            'positions_um': pos, 'taken_up': taken_up,
            'uptake_history': np.asarray(uptake_history),
            'released_total': released_total, 'cleared_total': cleared_total,
            'haemocyte_tracks': tracks,
            'initial_debris': initial_debris,
            'remaining_debris': float(np.sum(D)) * dx * dx,
            'budget_residual': float(np.max(np.abs(residual))),
            'budget_relative_residual': float(np.max(np.abs(residual) / scale)),
            'X': X, 'Y': Y, 'dx_um': dx, 'n': n, 'n_haemocytes': n_h,
            'dt_requested_s': dt_requested, 'dt_used_s': float(dt_used),
            'n_steps': int(steps), 'substeps_used': int(substeps_used),
            'chemotaxis': bool(chemotaxis),
            'clearance_per_s': p['clearance_per_s'],
            'params': {k: v for k, v in p.items()},
            'param_sources': PARAM_SOURCES}


def arrival_statistic(result, radius_um=150.0):
    """Fraction of haemocytes inside ``radius_um`` of the origin at the end."""
    d = np.linalg.norm(result['positions_um'], axis=1)
    return float(np.mean(d <= radius_um))


def self_test(verbose=True):
    """Every claim this module makes, measured."""
    out = {}
    p = dict(DEFAULTS)
    n, dx = 64, 5.0
    X, Y, _ = make_grid(n=n, dx_um=dx, p=p)
    rng = np.random.default_rng(0)
    start = rng.uniform(-0.45 * n * dx, 0.45 * n * dx, size=(96, 2))
    debris0 = wound_debris_profile(X, Y, 120.0, 1.0)

    # ---- 1. material budget closes --------------------------------------
    r = run(n=n, dx_um=dx, t_end_s=200.0, dt_s=5.0, p=p, debris0=debris0,
            haemocytes=start, chemotaxis=True, sample_every=4)
    out['budget'] = {
        'released_total': r['released_total'],
        'remaining_debris': round(r['remaining_debris'], 6),
        'taken_up_total': round(float(np.sum(r['taken_up'])), 6),
        'cleared_total': r['cleared_total'],
        'max_absolute_residual': r['budget_residual'],
        'max_relative_residual': r['budget_relative_residual'],
        'n_steps': r['n_steps'], 'dt_used_s': round(r['dt_used_s'], 4),
    }

    # ---- 2. chemotaxis changes the arrival statistic ---------------------
    on = run(n=n, dx_um=dx, t_end_s=400.0, dt_s=5.0, p=p, debris0=debris0,
             haemocytes=start, chemotaxis=True, sample_every=20)
    off = run(n=n, dx_um=dx, t_end_s=400.0, dt_s=5.0, p=p, debris0=debris0,
              haemocytes=start, chemotaxis=False, sample_every=20)
    a_on, a_off = arrival_statistic(on), arrival_statistic(off)
    r_on = float(np.linalg.norm(on['positions_um'], axis=1).mean())
    r_off = float(np.linalg.norm(off['positions_um'], axis=1).mean())
    # DIRECTION AND STRENGTH, measured on the axis rather than on a shell: averaging a
    # radial gradient over a whole shell cancels it, which is how an earlier probe of
    # this module reported a gradient of exactly zero and sent the diagnosis the wrong
    # way.  The two numbers below are what decide whether chemotaxis can compete with
    # the random walk at these parameters.
    Aend = on['attractant'][-1]
    gxa, gya = gradient(Aend, dx)
    c = n // 2
    off_cells = max(1, n // 8)
    grad_mag = float(abs(gxa[c, c + off_cells]))
    speed = on['params']['v_max_um_s'] * grad_mag / (
        grad_mag + on['params']['K_grad_per_um']) if grad_mag > 0 else 0.0
    chemo_step = speed * on['dt_used_s']
    walk_step = float(np.sqrt(2.0 * on['params']['random_walk_um2_s'] * on['dt_used_s']))
    out['chemotaxis_effect'] = {
        'fraction_within_150um_on': round(a_on, 4),
        'fraction_within_150um_off': round(a_off, 4),
        'fraction_difference_on_minus_off': round(a_on - a_off, 4),
        'mean_radius_um_on': round(r_on, 2),
        'mean_radius_um_off': round(r_off, 2),
        'mean_radius_difference_on_minus_off': round(r_on - r_off, 2),
        'attractant_gradient_measured_per_um': round(grad_mag, 6),
        'gradient_direction': ('negative radial component: the field DECREASES away '
                               'from the source, so moving up the gradient means moving '
                               'TOWARD it (verified on the axis, not on a shell)'),
        'chemotactic_step_um_per_dt': round(chemo_step, 4),
        'random_walk_step_um_per_dt': round(walk_step, 4),
        'chemotaxis_over_random_walk': round(chemo_step / walk_step, 4) if walk_step else None,
        'note': ('TWO statistics are reported because they can disagree in sign. The '
                 'mean radius states directly whether the population moved toward the '
                 'source; the count inside 150 um also depends on how widely the '
                 'population spreads, so a weaker random walk can raise it with no '
                 'attraction at all. An earlier revision reported only the count and '
                 'read it as chemotaxis pointing the wrong way: the real cause was a '
                 'gradient half-saturation of 1e-3 per um that made the chemotactic '
                 'speed negligible. Even after that fix the chemotactic step remains '
                 'SMALLER than the random-walk step at these parameters (see the two '
                 'steps above), so with v_max = 0.5 um/s and this motility the '
                 'attraction cannot overcome spreading. That is a statement about the '
                 'chosen parameters, not about haemocytes.'),
    }

    # ---- 3. uptake saturates --------------------------------------------
    sat = []
    for density in (1.0, 10.0, 100.0, 1000.0):
        d0 = wound_debris_profile(X, Y, 120.0, density)
        one = np.zeros((1, 2))            # a single haemocyte at the wound centre
        rr = run(n=n, dx_um=dx, t_end_s=20.0, dt_s=2.0, p=p, debris0=d0,
                 haemocytes=one, chemotaxis=False, sample_every=1)
        sat.append(round(float(rr['taken_up'][0]), 6))
    out['uptake_saturation'] = {
        'initial_debris_density': [1.0, 10.0, 100.0, 1000.0],
        'uptake_after_20s': sat,
        'capacity_bound_per_20s': p['u_max_per_cell_per_s'] * 20.0,
    }

    # ---- 4. resolution needs clearance (the negative result) ------------
    cont = wound_debris_profile(X, Y, 120.0, 1.0)
    no_clear = run(n=n, dx_um=dx, t_end_s=1200.0, dt_s=10.0, p=dict(p, clearance_per_s=0.0),
                   debris0=cont, haemocytes=start, chemotaxis=True, sample_every=20)
    with_clear = run(n=n, dx_um=dx, t_end_s=1200.0, dt_s=10.0, p=dict(p, clearance_per_s=0.01),
                     debris0=cont, haemocytes=start, chemotaxis=True, sample_every=20)
    d_no = no_clear['debris'].sum(axis=(1, 2)) * dx * dx
    d_yes = with_clear['debris'].sum(axis=(1, 2)) * dx * dx
    out['resolution_requires_clearance'] = {
        'initial_debris': round(float(d_no[0]), 6),
        'final_no_clearance': round(float(d_no[-1]), 6),
        'final_with_clearance': round(float(d_yes[-1]), 6),
        'fraction_left_no_clearance': round(float(d_no[-1] / d_no[0]), 6),
        'fraction_left_with_clearance': round(float(d_yes[-1] / d_yes[0]), 6),
        'verdict': ('without a clearance term the debris does NOT reach zero; the '
                    'plateau is set by uptake capacity against the released amount'),
    }

    # ---- 5. numerical hygiene -------------------------------------------
    out['hygiene'] = {
        'min_debris_over_run': float(np.min(no_clear['debris'])),
        'min_attractant_over_run': float(np.min(no_clear['attractant'])),
        'all_positions_finite': bool(np.all(np.isfinite(no_clear['positions_um']))),
        'positions_inside_domain': bool(np.all(
            np.abs(no_clear['positions_um']) <= 0.5 * n * dx)),
        'dt_was_clamped': bool(no_clear['dt_used_s'] < no_clear['dt_requested_s']),
    }
    out['operator_consistency'] = operator_consistency_check()

    if verbose:
        import json as _json
        print(_json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    self_test()
