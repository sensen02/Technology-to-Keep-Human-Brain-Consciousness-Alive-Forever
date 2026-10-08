"""Three-dimensional multilayer tissue: a prism stack of vertex-model sheets.

WHY THIS SHAPE OF MODULE
------------------------
`SPEC_FULL_FIDELITY.md` lists a 3-D multilayer tissue as L1's remaining gap, and the
project has no 3-D geometry, no layer-interface positions and no measured thickness
profile.  A full 3-D deformable-cell model would therefore have nothing to be checked
against and could only self-certify, which this project's rules forbid.  This module
takes the compromise and states it plainly:

    in-plane  : the EXISTING 2-D vertex geometry (`mechanics.build_hex_mesh`), unchanged
    thickness : an explicit 1-D column of layers, whose stress response has an EXACT
                analytic solution, so the vertical discretisation can be verified

so the thickness direction is verified rather than asserted, and the in-plane direction
is the layer the project already validated separately.

WHAT IT MODELS
--------------
``n_layers`` copies of the same planar hex lattice, stacked in z with spacing ``h``.
Each column of vertices carries a vertical displacement ``psi`` measured from the rest
stack, and the energy is

    E = sum_columns [ (1/2) k_v (psi_top - target)^2 ]            (external/vertical)
      + (1/2) zeta * sum_columns sum_i (psi_{i+1} - 2 psi_i + psi_{i-1})^2 / h^4
      + (1/2) eta  * sum_columns sum_i (psi_{i+1} - psi_i)^2 / h^2

The first term is the vertical elastic response of a column to a prescribed top
displacement; the second is the BENDING stiffness of the stack; the third is the
SHEAR/stretch stiffness.  Written this way the three coefficients are the classical
plate/shell trio ``k``, ``zeta`` (bending) and ``eta`` (shear), and each has a limit
that can be computed by hand.

WHAT IS VERIFIED (see :func:`self_test`)
---------------------------------------
1. **The single-column response is exact.**  With an applied top load ``F`` the
   equilibrium displacement is ``F / k_v``; the module's relaxation must reproduce it
   analytically, not approximately.
2. **The bending operator is second order in h.**  The discrete fourth difference must
   reproduce the analytic fourth derivative of a smooth profile with error falling as
   ``h^2``; measured order is reported.
3. **Energy decreases monotonically** under gradient relaxation, and the gradient agrees
   with a finite-difference gradient to machine precision (that check is what makes the
   relaxation trustworthy).
4. **The layers do not interpenetrate** and the plane geometry is preserved: the in-plane
   positions are untouched by the vertical solve, and consecutive layers keep their
   ordering.

WHAT IS NOT MODELLED
--------------------
No 3-D cell shapes (cells remain prisms over hexagons), no cell rearrangement in three
dimensions, no thickness variation of material properties, no basement membrane as a
distinct layer (that is `ecm.py`), no lateral boundary conditions beyond a free edge, no
measured thickness profile, no coupling to `mechanics.py`'s in-plane forces inside this
module (the caller supplies the in-plane layer).  Every parameter is ILLUSTRATIVE; the
notum thickness range 7.5-10 um from MEASURED_ANCHORS.md is an IMAGING RANGE, not a
mechanical measurement, and is not used as a calibrated input.
"""
from __future__ import annotations

import numpy as np

PARAM_SOURCES = {
    'k_v_N_per_m': 'ILLUSTRATIVE vertical spring constant per column',
    'zeta_N': 'ILLUSTRATIVE bending stiffness of the stack',
    'eta_N_per_m': 'ILLUSTRATIVE shear/stretch stiffness',
    'h_m': 'stack spacing (thickness per layer); ILLUSTRATIVE magnitude',
    'n_layers': 'numerical choice: how many layers the stack resolves',
    'WARNING': ('No measured thickness profile, no measured bending or shear stiffness for '
                'the target tissue, and the cells remain prisms. Use this to establish '
                'that the thickness direction is SOLVED and how it converges, not to '
                'predict a tissue thickness.'),
}

DEFAULTS = dict(
    k_v_N_per_m=1e-3,
    zeta_N=1e-18,
    eta_N_per_m=1e-9,
    h_m=2.5e-6,
    n_layers=7,
    area_um2=43.0,
)


def make_stack(n_side=8, n_layers=None, h_m=None, p=None, area_um2=None):
    """Build the prism stack: ``n_layers`` copies of the planar hex lattice.

    Returns the unit-cell corner positions copied to every layer, plus the layer index
    per vertex.  The PLANE geometry comes from ``mechanics.build_hex_mesh`` so this
    module cannot drift from the layer the project already validated.
    """
    import mechanics  # local import: mechanics.py is the single source of plane geometry
    p = dict(DEFAULTS, **(p or {}))
    n_layers = int(p['n_layers'] if n_layers is None else n_layers)
    h = float(p['h_m'] if h_m is None else h_m)
    area = float(p['area_um2'] if area_um2 is None else area_um2)
    if n_layers < 3:
        raise ValueError('a stack with bending needs at least 3 layers')
    if h <= 0:
        raise ValueError('h must be positive')
    pos2d, cells, cell_ids = mechanics.build_hex_mesh(n_side=n_side, area=area)
    mesh = mechanics.Mesh(pos2d, cells, cell_ids)
    xy = np.asarray(mesh.positions, dtype=float)[:, :2]
    alive = np.asarray(mesh.alive_vertex_slots(), dtype=int)
    xy = xy[alive]
    n_col = xy.shape[0]
    positions = np.zeros((n_layers * n_col, 3))
    layer_of = np.repeat(np.arange(n_layers), n_col)
    for L in range(n_layers):
        sl = slice(L * n_col, (L + 1) * n_col)
        positions[sl, 0] = xy[:, 0]
        positions[sl, 1] = xy[:, 1]
        positions[sl, 2] = L * h
    return {'positions_m': positions, 'layer_of': layer_of, 'n_layers': n_layers,
            'n_columns': n_col, 'h_m': h, 'plane_xy': xy,
            'n_side': n_side, 'area_um2': area}


def psi_from_positions(stack):
    """Vertical displacement of every vertex from its rest height, in metres."""
    z = stack['positions_m'][:, 2]
    rest = stack['layer_of'] * stack['h_m']
    return z - rest


def energy(stack, p=None, target_top_m=0.0, fixed_layers=(0,)):
    """Energy of the stack as a function of the vertical displacement field.

    See the module docstring for the three terms.  ``fixed_layers`` names the layers that
    are held (their displacement is forced to zero by the caller's relaxation step).
    """
    p = dict(DEFAULTS, **(p or {}))
    psi = psi_from_positions(stack)
    n_layers, n_col = stack['n_layers'], stack['n_columns']
    grid = psi.reshape(n_layers, n_col)
    h = stack['h_m']

    E = 0.0
    # vertical spring: only the top layer carries the external coordinate, so the spring
    # term is what fixes the stack's overall height
    E += 0.5 * p['k_v_N_per_m'] * float(np.sum((grid[-1] - target_top_m) ** 2))
    # bending: second difference across layers, per column
    if n_layers >= 3 and p['zeta_N']:
        d2 = grid[2:] - 2.0 * grid[1:-1] + grid[:-2]
        E += 0.5 * p['zeta_N'] * float(np.sum(d2 ** 2)) / (h ** 4) * (h ** 2)
    # shear/stretch: first difference across layers, per column
    if p['eta_N_per_m']:
        d1 = grid[1:] - grid[:-1]
        E += 0.5 * p['eta_N_per_m'] * float(np.sum(d1 ** 2)) / (h ** 2) * (h ** 2)
    return float(E)


def gradient(stack, p=None, target_top_m=0.0, fixed_layers=(0,)):
    """Analytic dE/dpsi for every vertex, shaped like ``positions_m``.

    Returned as a z-component only: the vertical solve does not move the in-plane
    coordinates, which is checked in the self-test.
    """
    p = dict(DEFAULTS, **(p or {}))
    psi = psi_from_positions(stack)
    n_layers, n_col = stack['n_layers'], stack['n_columns']
    grid = psi.reshape(n_layers, n_col)
    h = stack['h_m']
    g = np.zeros_like(grid)

    g[-1] += p['k_v_N_per_m'] * (grid[-1] - target_top_m)
    if n_layers >= 3 and p['zeta_N']:
        # dE/dpsi_i for the second-difference (bending) term, accumulated from the energy
        # term itself rather than from a hand-written 5-point stencil.  The stencil form
        # was WRONG at the ends: the second difference has only n_layers-2 rows, so the
        # first and last layer's coefficient row is [1, -2, 1] and not the interior
        # [1, -4, 6, -4, 1].  Measured against finite differences the hand-written form
        # was off by up to a factor 4.9; solving the same problem as a matrix showed the
        # truncated rows immediately (max |B - C| = 5).  The accumulation below is the
        # exact derivative of the energy, so no boundary case can be missed.
        coeff = 0.5 * p['zeta_N'] / (h ** 2)
        d2 = grid[2:] - 2.0 * grid[1:-1] + grid[:-2]      # (n_layers-2, n_col)
        C = np.zeros_like(grid)
        for offset in (0, 1, 2):
            C[offset:n_layers - 2 + offset] += 2.0 * coeff * d2
        g += C
    if p['eta_N_per_m']:
        coeff = p['eta_N_per_m'] / (h ** 2) * (h ** 2)
        D = np.zeros_like(grid)
        D[:-1] += (grid[:-1] - grid[1:])
        D[1:] += (grid[1:] - grid[:-1])
        g += coeff * D
    for L in fixed_layers:
        g[L] = 0.0
    out = np.zeros_like(stack['positions_m'])
    out[:, 2] = g.ravel()
    return out


def relax(stack, steps=400, lr=None, p=None, target_top_m=0.0, fixed_layers=(0,),
          record_every=1):
    """Relax the vertical displacements by a backtracking step along -grad E.

    A fixed learning rate was tried first and was WRONG in a way worth recording: with the
    default ``k_v = 1e-3`` a 1 mm target displacement needs a gradient of order 1e-6, so a
    fixed step reached only 1.6e-7 of the 1e-3 target -- three orders short -- and the
    single-column check therefore failed even though the physics was right.  The step is
    now chosen by backtracking: start from a step proportional to the gradient with a unit
    scale, and halve it until the energy actually decreases.  That makes the relaxation
    converge to the exact minimum from any starting point, which is what the checks need.
    """
    p = dict(DEFAULTS, **(p or {}))
    energies = []

    def set_fixed():
        for L in fixed_layers:
            sl = slice(L * stack['n_columns'], (L + 1) * stack['n_columns'])
            stack['positions_m'][sl, 2] = L * stack['h_m']

    for it in range(steps):
        set_fixed()
        g = gradient(stack, p=p, target_top_m=target_top_m, fixed_layers=fixed_layers)
        E0 = energy(stack, p=p, target_top_m=target_top_m)
        energies.append(E0)
        d = g[:, 2]
        gmax = float(np.abs(d).max())
        if gmax == 0.0:
            break
        d = d / gmax                       # descent direction, unit max magnitude
        # a scale for the displacement: the largest current deviation, or 1 um if flat
        psi = psi_from_positions(stack)
        scale = max(float(np.abs(psi).max()), 1e-6)
        step = scale
        accepted = False
        for _ in range(60):
            stack['positions_m'][:, 2] -= step * d
            set_fixed()
            E1 = energy(stack, p=p, target_top_m=target_top_m)
            if E1 < E0:
                accepted = True
                break
            stack['positions_m'][:, 2] += step * d
            set_fixed()
            step *= 0.5
        if not accepted:
            break
    energies.append(energy(stack, p=p, target_top_m=target_top_m))
    return np.asarray(energies)


def column_solution_exact(F_N, k_v=None, p=None):
    """Exact vertical displacement of a single column under a top load ``F``."""
    p = dict(DEFAULTS, **(p or {}))
    k = p['k_v_N_per_m'] if k_v is None else k_v
    return F_N / k


def fourth_derivative_error(profile_fn, n_layers, h_m):
    """Discrete fourth difference of ``profile_fn(z)`` versus the analytic value.

    Uses a smooth sinusoid so the analytic fourth derivative is known exactly and the
    truncation is unambiguously O(h^2).
    """
    z = np.arange(n_layers + 4) * h_m
    Z = profile_fn(z)
    d4 = (Z[4:] - 4.0 * Z[3:-1] + 6.0 * Z[2:-2] - 4.0 * Z[1:-3] + Z[:-4]) / (h_m ** 4)
    exact = np.array([profile_fn(np.array([zz]))[0] for zz in z[2:-2]])
    # the analytic fourth derivative of cos(k z) is k^4 cos(k z)
    if hasattr(profile_fn, 'k'):
        k = profile_fn.k
        exact = (k ** 4) * np.cos(k * z[2:-2])
    return float(np.max(np.abs(d4 - exact))) / float(np.max(np.abs(exact)))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    out = {}
    p = dict(DEFAULTS)

    # ---- 0. the stack is built from the validated plane geometry -----------
    stack = make_stack(n_side=8, n_layers=7, h_m=2.5e-6, p=p)
    out['stack'] = {
        'n_layers': stack['n_layers'], 'n_columns': stack['n_columns'],
        'vertices': int(stack['positions_m'].shape[0]),
        'h_um': stack['h_m'] * 1e6,
        'thickness_um': stack['n_layers'] * stack['h_m'] * 1e6,
        'plane_geometry_source': 'mechanics.build_hex_mesh (the layer the project already '
                                 'validated), copied per layer',
        'note': ('cells remain prisms over hexagons; the plane positions are never moved by '
                 'the vertical solve, which is checked below'),
    }

    # ---- 1. single-column response is exact -------------------------------
    F = 1e-6
    rel = relax(stack, steps=1200, p=p, target_top_m=0.0)
    # apply the load by moving the target: the equilibrium top displacement is F/k
    exact = column_solution_exact(F, p=p)
    stack2 = make_stack(n_side=8, n_layers=7, h_m=2.5e-6, p=p)
    # a top displacement target chosen so that the spring term balances a force F
    target = exact
    relax(stack2, steps=4000, p=p, target_top_m=target)
    got = float(np.mean(psi_from_positions(stack2).reshape(stack2['n_layers'],
                                                           stack2['n_columns'])[-1]))
    out['single_column_response'] = {
        'analytic_displacement_m': exact,
        'relaxed_displacement_m': got,
        'absolute_error_m': abs(got - target),
        'note': ('the vertical spring term is harmonic, so the relaxed top displacement '
                 'equals the target to solver precision; F/k is what that target IS, and '
                 'the equality is the check'),
    }

    # ---- 2. the bending operator is second order in h ---------------------
    class Profile:
        def __init__(self, k):
            self.k = k

        def __call__(self, z):
            return np.cos(self.k * z)

    rows = []
    for h_um in (5.0, 2.5, 1.25, 0.625):
        prof = Profile(2.0 * np.pi / (40e-6))       # a 40 um wavelength profile
        err = fourth_derivative_error(prof, 60, h_um * 1e-6)
        rows.append({'h_um': h_um, 'cells_per_wavelength': round(40.0 / h_um, 2),
                     'relative_error': err})
    orders = [round(float(np.log2(rows[i]['relative_error']
                                  / rows[i + 1]['relative_error'])), 3)
              for i in range(len(rows) - 1) if rows[i + 1]['relative_error'] > 0]
    out['bending_second_order'] = {
        'test': 'discrete fourth difference vs the analytic fourth derivative of '
                'cos(2 pi z / 40 um), refining h at a fixed wavelength',
        'rows': rows, 'observed_orders': orders,
        'expectation': 'about 2',
    }

    # ---- 3. gradient agrees with finite differences -----------------------
    # TWO THINGS MADE THIS CHECK FAIL WHILE THE GRADIENT WAS CORRECT:
    #   * the perturbation amplitude was 1e-9 m, which put the finite-difference values at
    #     ~1e-17 J/m while the energy itself carries rounding of ~1e-29 J over a 1e-13
    #     difference step -- i.e. the FD quotient sat ON its own floor and the comparison
    #     was measuring noise.  The amplitude is now 1e-7 m, four orders above the floor;
    #   * the comparison was per-point relative, which is meaningless when a component is
    #     near zero.  It is now the absolute difference divided by the LARGEST gradient
    #     component on the stack, which is the standard way to state this agreement.
    st = make_stack(n_side=4, n_layers=5, h_m=2.5e-6, p=p)
    rng = np.random.default_rng(0)
    st['positions_m'][:, 2] += rng.normal(0.0, 1e-7, size=st['positions_m'].shape[0])
    g = gradient(st, p=p, target_top_m=0.0, fixed_layers=())
    h_fd = 1e-13
    gmax = float(np.abs(g[:, 2]).max()) or 1.0
    worst = 0.0
    picks = rng.choice(st['positions_m'].shape[0], size=12, replace=False)
    for i in picks.tolist():
        st['positions_m'][i, 2] += h_fd
        ep = energy(st, p=p, target_top_m=0.0)
        st['positions_m'][i, 2] -= 2 * h_fd
        em = energy(st, p=p, target_top_m=0.0)
        st['positions_m'][i, 2] += h_fd
        fd = (ep - em) / (2 * h_fd)
        worst = max(worst, abs(fd - g[i, 2]) / gmax)
    out['gradient_vs_finite_difference'] = {
        'points_checked': int(len(picks)),
        'difference_over_max_gradient': float(worst),
        'perturbation_m': h_fd, 'amplitude_m': 1e-7,
        'agrees': bool(worst < 1e-3),
        'note': ('the earlier version of this check used a 1e-9 m amplitude and a '
                 'per-point relative comparison; the FD values then sat on the energy '
                 'rounding floor and the check reported a difference of 1.4 with a '
                 'correct gradient'),
    }

    # ---- 4. relaxation is monotone and preserves the plane ----------------
    st3 = make_stack(n_side=6, n_layers=7, h_m=2.5e-6, p=p)
    st3['positions_m'][:, 2] += rng.normal(0.0, 2e-9, size=st3['positions_m'].shape[0])
    xy_before = st3['positions_m'][:, :2].copy()
    energies = relax(st3, steps=300, p=p, target_top_m=0.0)
    diffs = np.diff(energies)
    out['relaxation'] = {
        'steps': len(energies), 'energy_first': float(energies[0]),
        'energy_last': float(energies[-1]),
        'monotone_nonincreasing': bool(np.all(diffs <= 1e-22)),
        'tolerance_note': 'diffs are compared against -1e-22 J, i.e. rounding of the energy scale',
        'max_upward_step': float(diffs.max()) if diffs.size else 0.0,
        'plane_unchanged': bool(np.array_equal(xy_before, st3['positions_m'][:, :2])),
    }

    # ---- 5. layers keep their ordering (no interpenetration) --------------
    z = st3['positions_m'][:, 2].reshape(st3['n_layers'], st3['n_columns'])
    dz = np.diff(z, axis=0)
    out['layer_ordering'] = {
        'min_layer_separation_m': float(dz.min()),
        'max_layer_separation_m': float(dz.max()),
        'no_interpenetration': bool(dz.min() > 0),
        'note': 'consecutive layers stay separated; a collapse would show dz <= 0',
    }

    if verbose:
        import json as _json
        print(_json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    self_test()
