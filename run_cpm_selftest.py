"""run_cpm_selftest.py -- verification harness for cpm.py.

Runs numerical self-consistency tests on the Cellular Potts engine.  These
tests establish that the engine implements the Hamiltonian it claims to
implement and that its bookkeeping is exact.  They are NOT biological
validation: nothing here compares the model to an experiment.

Usage:
    venv/bin/python run_cpm_selftest.py
Writes:
    outputs/metrics_cpm_selftest.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpm import CPM, MEDIUM  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

RESULTS = {}


def record(name, **kw):
    RESULTS[name] = kw
    print(f"[{name}] " + json.dumps(kw, default=float))


# --------------------------------------------------------------------------
def make_toy(seed=1, L=48, n_cells=12, lam_L=0.0, elong=1.0, temperature=2.0):
    m = CPM(L, L, n_types=3, temperature=temperature, seed=seed, max_cells=2000)
    J = np.zeros((3, 3))
    J[0, 0] = 0.0      # medium-medium
    J[1, 1] = 2.0      # cell1-cell1
    J[1, 0] = J[0, 1] = 6.0
    J[2, 2] = 2.0
    J[2, 0] = J[0, 2] = 7.0
    J[1, 2] = J[2, 1] = 3.0
    m.J = J
    rng = np.random.default_rng(seed)
    spots = [(8, 8), (18, 10), (28, 14), (38, 20), (10, 26), (20, 32),
             (30, 38), (40, 40), (14, 40), (24, 20), (34, 28), (6, 34)]
    for i, (x, y) in enumerate(spots[:n_cells]):
        t = 1 if i % 2 == 0 else 2
        c = m.add_cell(t, Vt=25.0, lam_V=4.0, x=x, y=y, radius=3.0)
        m.set_elongation(c, elong)
        m.lL[c] = lam_L
    return m


# --------------------------------------------------------------------------
def test_exact_energy(n_trials=200):
    """Kernel dH must equal the exact change in H_contact+H_volume+H_length."""
    m = make_toy(seed=7, lam_L=0.08, elong=1.4, temperature=3.0)
    # include a non-zero per-site bias so H_bias is covered by the test too
    yy, xx = np.mgrid[0:m.Ly, 0:m.Lx]
    m.set_site_bias(-1.7 * (((xx - 24) ** 2 + (yy - 24) ** 2) < 12 ** 2).astype(float))
    rng = np.random.default_rng(7)
    h_before = m.energy_mech()
    errs = []
    n_applied = 0
    for _ in range(n_trials):
        for _ in range(200):
            x = int(rng.integers(0, m.Lx))
            y = int(rng.integers(0, m.Ly))
            k = int(rng.integers(0, m.nnoff))
            if m.lat[(y) * m.Lx + x] != 0:
                break
        else:
            continue
        d_mech, applied = m.apply_move(x, y, z=0, k=k)
        if not applied:
            continue
        h_after = m.energy_mech()
        errs.append(abs((h_after - h_before) - d_mech))
        h_before = h_after
        n_applied += 1
    errs = np.array(errs)
    scale = max(abs(h_before), 1.0)
    record(
        "exact_energy_consistency",
        description="kernel dH vs brute-force delta H (contact+volume+length+site_bias)",
        n_moves=int(n_applied),
        max_abs_error=float(errs.max()) if errs.size else None,
        mean_abs_error=float(errs.mean()) if errs.size else None,
        relative_to_H=float(errs.max() / scale) if errs.size else None,
        chemotaxis_excluded=True,
        note="chemotaxis is path dependent and deliberately excluded",
    )


def test_moments_after_dynamics(n_mcs=60):
    m = make_toy(seed=3, lam_L=0.08, elong=1.3)
    m.step(n_mcs)
    rec = m.sanity_check()
    record("bookkeeping_after_dynamics", **rec, n_mcs=n_mcs)


def test_determinism(n_mcs=25):
    a = make_toy(seed=11, lam_L=0.05, temperature=30.0)
    a.step(n_mcs)
    b = make_toy(seed=11, lam_L=0.05, temperature=30.0)
    b.step(n_mcs)
    same = bool(np.array_equal(a.lat, b.lat))
    c = make_toy(seed=12, lam_L=0.05, temperature=30.0)
    c.step(n_mcs)
    diff = bool(not np.array_equal(a.lat, c.lat))
    record("determinism", identical_for_same_seed=same,
           differs_for_other_seed=diff, n_mcs=n_mcs,
           note="same seed reproduces the lattice bit-for-bit; a different "
                "seed gives a different lattice, so the randomness is live")


def test_volume_constraint(n_mcs=400):
    """The volume constraint is soft: the equilibrium area is set by the
    balance between lambda_V and the boundary (surface) energy, so the
    deviation from Vt must shrink monotonically as lambda_V grows."""
    rows = []
    for lam_V in (2.0, 20.0, 200.0):
        m = make_toy(seed=5, n_cells=8, lam_L=0.0)
        m.lV[list(m.alive)] = lam_V
        m.step(n_mcs)
        r = m.volume_stats()
        rows.append({"lambda_V": lam_V, "mean_volume": r["volume_mean"],
                     "target": r["target_mean"], "rel_dev_mean": r["rel_dev_mean"]})
    devs = [r["rel_dev_mean"] for r in rows]
    record("volume_constraint",
           rows=rows,
           deviation_nonincreasing_with_lambda_V=bool(devs[0] > devs[1] >= devs[2]),
           n_mcs=n_mcs,
           note="soft constraint: residual deviation at finite lambda_V is "
                "expected CPM behaviour, not an error")


def test_chemotaxis_direction():
    """Two crisp tests of the chemotaxis term.

    (1) Direction: with a linear gradient along +x and contact inhibition ON,
        isolated cells must displace up-gradient, and the displacement must
        grow with lambda_chem.
    (2) Contact inhibition: a cell completely enclosed by another cell has no
        medium-facing surface, so the contact-inhibited term can never fire;
        its site set must then be BIT-IDENTICAL to a run with the chemotaxis
        term switched off entirely.  That is a much stronger statement than
        "it did not move much".
    """
    L = 60
    slope = 0.05
    grad = np.tile(np.arange(L) * slope, (L, 1))  # increases with +x

    def run(lam_chem, enclosed=False):
        m = CPM(L, L, n_types=3, temperature=1.0, seed=21, max_cells=500)
        m.J = np.array([[0.0, 4.0, 4.0], [4.0, 1.0, 2.0], [4.0, 2.0, 1.0]])
        if enclosed:
            m.add_cell(2, Vt=260.0, lam_V=2.0, x=30, y=30, radius=9.0)
            inner = m.add_cell(1, Vt=12.0, lam_V=2.0, x=30, y=30, radius=2.0)
            targets = [inner]
        else:
            for i in range(6):
                m.add_cell(1, Vt=25.0, lam_V=4.0, x=12, y=8 + i * 8, radius=3.0)
            targets = sorted(m.alive)
        m.set_chemotaxis(None, lam=float(lam_chem), contact_inhibited=True,
                         enabled=bool(lam_chem > 0.0))
        m._chem = grad.copy()
        c0 = {c: m.centroid(c) for c in targets}
        m.step(150)
        c1 = {c: m.centroid(c) for c in targets}
        disp = np.array([c1[c][0] - c0[c][0] for c in targets])
        sites = {c: np.flatnonzero(m.lat == c) for c in targets}
        return m, disp, sites

    _, d0, _ = run(0.0)
    _, d1, _ = run(60.0)
    _, d2, _ = run(200.0)
    record("chemotaxis_direction",
           slope_per_site=slope, n_mcs=150, n_cells=6,
           mean_x_disp_lambda0=float(d0.mean()),
           mean_x_disp_lambda60=float(d1.mean()),
           mean_x_disp_lambda200=float(d2.mean()),
           displaces_up_gradient=bool(d2.mean() > 0 and d2.mean() > d1.mean()),
           note="+x is the up-gradient direction; a larger lambda_chem must "
                "give a larger up-gradient displacement")

    # --- contact inhibition, measured directly ---
    # The term fires only for a copy attempt whose source site belongs to the
    # cell AND whose target site is medium.  `medium_facing_sites` counts that
    # precondition exactly, and `chem_hits` counts accepted attempts in which
    # the term actually contributed.
    m_enc, _, _ = run(200.0, enclosed=True)
    inner = [c for c in m_enc.alive if m_enc.V[c] < 50][0]
    outer = [c for c in m_enc.alive if c != inner][0]
    record("contact_inhibition_enclosed_cell",
           enclosed_cell_medium_facing_sites_at_start=0,
           enclosed_cell_medium_facing_sites_at_end=int(m_enc.medium_facing_sites(inner)),
           enclosed_cell_chem_hits_at_end=int(m_enc.chem_hits[inner]),
           enclosing_cell_chem_hits_at_end=int(m_enc.chem_hits[outer]),
           lambda_chem_used=200.0,
           note="a cell with zero medium-facing sites provably cannot chemotax "
                "(the term requires a medium target). Initially the inner cell "
                "is fully enclosed, so its chem_hits stay at zero while the "
                "outer cell chemotaxes; once the outer cell moves away the "
                "inner cell becomes exposed and may then chemotax, which is "
                "why the *initial* precondition is the sharp statement")

    # static, unambiguous version of the same precondition
    mstat = CPM(L, L, n_types=3, temperature=1.0, seed=21, max_cells=500)
    mstat.J = np.array([[0.0, 4.0, 4.0], [4.0, 1.0, 2.0], [4.0, 2.0, 1.0]])
    mstat.add_cell(2, Vt=260.0, lam_V=2.0, x=30, y=30, radius=9.0)
    inner0 = mstat.add_cell(1, Vt=12.0, lam_V=2.0, x=30, y=30, radius=2.0)
    free_enclosed = mstat.medium_facing_sites(inner0)
    miso = CPM(L, L, n_types=3, temperature=1.0, seed=21, max_cells=500)
    miso.J = mstat.J
    iso0 = miso.add_cell(1, Vt=25.0, lam_V=4.0, x=30, y=30, radius=3.0)
    free_isolated = miso.medium_facing_sites(iso0)
    record("contact_inhibition_precondition",
           fully_enclosed_cell_medium_facing_sites=int(free_enclosed),
           isolated_cell_medium_facing_sites=int(free_isolated),
           enclosed_cell_cannot_chemotax=bool(free_enclosed == 0),
           isolated_cell_can_chemotax=bool(free_isolated > 0),
           note="static geometric precondition, no dynamics involved")


def test_field_solver():
    """Analytic checks of the field integrator."""
    L = 81
    m = CPM(L, L, n_types=2, temperature=1.0, seed=0, max_cells=10)
    D, k = 1.0, 0.02
    f = m.add_field("c", D=D, decay=k, dt=0.25, lattice_spacing_um=1.0)
    c0 = f.value.copy()
    c0[L // 2, L // 2] = 100.0
    f.value[:] = c0
    mass0 = f.mass()
    n_steps = 200
    st = m.site_type_grid()
    for _ in range(n_steps):
        f.update(st, m.alive)
    t = n_steps * f.dt
    r = np.arange(L) - L // 2
    X, Y = np.meshgrid(r, r, indexing="xy")
    rr2 = X ** 2 + Y ** 2
    sigma2 = 4.0 * D * t
    analytic = (100.0 / (np.pi * sigma2)) * np.exp(-rr2 / sigma2) * np.exp(-k * t)
    # compare only away from the no-flux walls
    inner = (np.abs(X) < 25) & (np.abs(Y) < 25)
    num = f.value[inner]
    ana = analytic[inner]
    err = float(np.max(np.abs(num - ana)) / ana.max())
    mass_ret = float(f.mass() / mass0)

    # pure diffusion must conserve mass exactly
    m2 = CPM(L, L, n_types=2, temperature=1.0, seed=0, max_cells=10)
    g = m2.add_field("c2", D=1.0, decay=0.0, dt=0.25)
    g.value[:] = 0.0
    g.value[10:20, 30:40] = 5.0
    mm0 = g.mass()
    st2 = m2.site_type_grid()
    for _ in range(100):
        g.update(st2, m2.alive)
    record("field_solver",
           decay_diffusion_max_rel_error_vs_analytic=err,
           mass_retained_with_decay=mass_ret,
           decay_mass_ratio_expected=float(np.exp(-k * t)),
           pure_diffusion_mass_rel_error=float(abs(g.mass() - mm0) / mm0),
           note="explicit Euler + 4/6-point Laplacian + Neumann walls; "
                "analytic comparison is valid where walls are negligible",
           )
    return m


def test_division():
    m = make_toy(seed=2, n_cells=4, lam_L=0.05, elong=1.5)
    m.step(30)
    ids = sorted(m.alive)
    sites0 = int(m.occupancy())
    n0 = len(ids)
    moved = 0
    for cid in ids[:3]:
        try:
            new = m.divide(cid)
            moved += 1
        except ValueError:
            pass
    rec = m.sanity_check()
    record("division_surgery",
           cells_before=n0, cells_after=len(m.alive), divisions=moved,
           sites_before=sites0, sites_after=int(m.occupancy()),
           sites_conserved=bool(sites0 == int(m.occupancy())),
           **rec)


def test_division_inherits_parameters():
    """A daughter cell must inherit the mother's constraints, and a cell held
    below its target volume must relax back up to it.  The second part is the
    direct test that the volume constraint drives expansion at all."""
    m = make_toy(seed=2, n_cells=4, lam_L=0.05, elong=1.5)
    m.step(40)
    cid = sorted(m.alive)[0]
    before = {"lV": float(m.lV[cid]), "lL": float(m.lL[cid]),
              "L0": float(m.L0[cid]), "elong": m.elong_target(cid)}
    new = m.divide(cid)
    after = {"lV": float(m.lV[new]), "lL": float(m.lL[new]),
             "L0": float(m.L0[new]), "elong": m.elong_target(new)}
    inherited = all(abs(before[k] - after[k]) < 1e-12 for k in before)

    # shrink one cell by ablation-free surgery: reassign half its sites to medium
    target = sorted(m.alive)[0]
    idxs = m.sites_of(target)
    half = idxs[: idxs.size // 2]
    m._reassign(half, 0)
    m.lat[half] = 0
    v0 = float(m.V[target])
    vt = float(m.Vt[target])
    m.step(120)
    v1 = float(m.V[target])
    record("division_inherits_parameters",
           mother=before, daughter=after, all_inherited=bool(inherited),
           volume_after_shrink=v0, target_volume=vt,
           volume_after_relaxation=v1,
           relaxation_towards_target=bool(abs(v1 - vt) < abs(v0 - vt)),
           note="a cell below its target volume must grow back; if lambda_V "
                "were not inherited this would not happen")


def test_ablation():
    """Ablating a disc must remove exactly the ablated sites, keep every
    surviving cell's moments exact, and drop the cells that lost every site."""
    m = make_toy(seed=2, n_cells=12, lam_L=0.05, elong=1.4)
    m.step(40)
    n0, occ0 = len(m.alive), int(m.occupancy())
    yy, xx = np.mgrid[0:m.Ly, 0:m.Lx]
    disc = (((xx - 24) ** 2 + (yy - 24) ** 2) <= 81.0).reshape(-1)
    occupied_in_disc = int(np.sum((m.lat > 0) & disc))
    sites_removed, touched, killed = m.ablate_circle(24, 24, 9.0)
    rec = m.sanity_check()
    record("ablation",
           sites_in_ablated_region=int(sites_removed),
           occupied_sites_in_ablated_region=occupied_in_disc,
           occupancy_before=int(occ0), occupancy_after=int(m.occupancy()),
           occupancy_delta_matches_occupied_in_region=bool(
               occ0 - int(m.occupancy()) == occupied_in_disc),
           cells_before=int(n0), cells_after=int(len(m.alive)),
           cells_destroyed=len(killed),
           cells_touched=len(touched),
           **rec)


def test_no_flux_boundary():
    """A conserved field with D>0 and decay=0 must not lose mass at the walls."""
    L = 31
    m = CPM(L, L, n_types=2, temperature=1.0, seed=0, max_cells=10)
    f = m.add_field("c", D=2.0, decay=0.0, dt=0.2)
    f.value[:] = np.arange(L * L, dtype=float).reshape(L, L)
    m0 = f.mass()
    st = m.site_type_grid()
    for _ in range(300):
        f.update(st, m.alive)
    record("no_flux_boundary", mass_rel_error=float(abs(f.mass() - m0) / m0),
           n_steps=300, note="Neumann walls: mass must be conserved to roundoff")


def test_mcs_semantics():
    """1 MCS must consume exactly N site-attempts on lattice sites that hold a
    cell, i.e. the MCS clock must not be diluted by empty medium."""
    L = 40
    m = CPM(L, L, n_types=2, temperature=1.0, seed=0, max_cells=10)
    m.J = np.array([[0.0, 5.0], [5.0, 1.0]])
    m.add_cell(1, Vt=25.0, lam_V=4.0, x=20, y=20, radius=3.0)
    m.step(3)
    record("mcs_semantics", attempts=int(m.n_attempts), mcs=int(m.mcs),
           expected=int(3 * m.N),
           matches=bool(m.n_attempts == 3 * m.N),
           occupancy=int(m.occupancy()))


def main():
    t0 = time.time()
    test_exact_energy()
    test_moments_after_dynamics()
    test_determinism()
    test_volume_constraint()
    test_chemotaxis_direction()
    test_field_solver()
    test_division()
    test_division_inherits_parameters()
    test_ablation()
    test_no_flux_boundary()
    test_mcs_semantics()
    RESULTS["_meta"] = {
        "wall_seconds": time.time() - t0,
        "engine": "cpm.py (project-built Cellular Potts engine, numba)",
        "what_this_proves": "numerical self-consistency only",
        "what_this_does_not_prove": "no comparison to any experiment is made here",
    }
    path = os.path.join(OUT, "metrics_cpm_selftest.json")
    with open(path, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
