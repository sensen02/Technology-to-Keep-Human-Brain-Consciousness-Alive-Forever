"""Spatially resolved INTRACELLULAR Ca2+/IP3 reaction-diffusion for ONE cell.

SYNTHETIC MODEL OUTPUT -- NOT MEASUREMENT
=========================================
Every number produced by this module is model output.  Nothing here is
measured data, and nothing here is calibrated against an experiment.

WHAT THIS IS, AND WHAT IT IS NOT
================================
This module takes the *well-mixed* (lumped) per-cell calcium ODE published
upstream -- `model.py`, a Python port of ACCRE_FullCalciumSignalingModel_Control.m
from mshutson/wound-calcium-LRCa (Stevens, O'Connor, Pumford, Page-McCaw and
Hutson, "A mathematical model of calcium signals around laser-induced epithelial
wounds") -- and gives ONE cell an interior: the same published fluxes and the
same published PARAMETERS, but now applied locally inside a cytosol that is
resolved in space, with Fickian diffusion of free Ca2+ and IP3.

It is emphatically NOT a tissue model:

* exactly one cell is simulated; there are no neighbours, no gap junctions,
  no extracellular ligand field, no mechanics, no ablation;
* `model.py`'s cell-to-cell heterogeneity (GJ / PLC / GCaMP lognormals) is a
  tissue-level feature and is therefore switched OFF here: the single cell
  uses the mean multipliers (plc = 1, gcamp = 1) so that the spatial cell and
  the published lumped cell differ in exactly one thing, namely SPACE;
* `run_lumped()` reproduces the published single-cell ODE (no gap junctions)
  for the identical parameters and stimulus, which makes the spatial-vs-lumped
  difference a controlled comparison rather than a comparison of two models.

NO EXPERIMENTAL DATASET IN THIS PROJECT CONSTRAINS INTRACELLULAR GRADIENTS.
The wound-calcium project's anchors (`MEASURED_ANCHORS.md`) constrain tissue- or
cell-population-level Ca responses.  Neither the sub-membrane Ca gradient of a
single cell nor any cytosolic diffusion coefficient has been measured here, so
the spatial predictions below are unvalidated predictions, not findings about
the biological preparation.  The one genuinely external input is the diffusion
coefficients, which are cited literature values measured in a *frog oocyte
cytosolic extract* (see ``CITATIONS``), not in the fly epithelium of the
published model and not in this project.

Motivating question, stated honestly: does resolving the cytosol spatially
change anything that is actually observable, and by how much?  The module
therefore spends most of its code not on producing a spatial solution but on
quantifying the error of the lumped assumption: conservation laws, an analytic
diffusion solution, a large-D convergence test against `run_lumped`, and a
sensitivity sweep over the dominant unknown (D_Ca).

The honest summary of what the tests actually returned (see
``full_test_suite()``): for a whole-surface microtear the lumped model is
accurate to a few per cent in the volume-averaged Ca; the difference is
dominated by the transient sub-membrane layer, whose *thickness* (not the
lumped value) is the new prediction.  For a surface-localised patch the
difference is an order of magnitude larger, and the volume-averaged quantity
alone no longer describes the cell.

UNITS
=====
Time s, length um, concentrations uM, exactly as `model.py`.  Volumes um^3.
Ca fluxes are rates of change of cytosolic concentration (uM/s).

GEOMETRY OPTIONS
================
``radial1d``  spherically symmetric cell of radius ``R_um`` split into
              ``n_shells`` finite-volume shells with exact spherical-shell
              volumes and exact face areas.  Zero flux at r = 0 and at r = R
              except for the plasma-membrane flux, which acts on the outermost
              shell only.  A spherical cell cannot represent an angularly
              localised patch: in ``radial1d`` a patch is applied as its
              surface-area-weighted uniform equivalent (documented in the
              returned metadata).  Default R = 5 um, n_shells = 64.

``grid3d``    cubic voxel grid of side 2*R_um sampled at ``n_voxels`` per axis;
              the cell is a "staircase" sphere (a voxel is cytosolic if its
              centre is within ``R_um``).  Zero flux between cytosolic and
              non-cytosolic voxels and on the grid boundary; the plasma-membrane
              flux acts on voxel faces that border non-cytosolic space.  This is
              the only mode that can resolve a NON-SYMMETRIC stimulus (a
              localised tear patch).  It is marked EXPERIMENTAL: the staircase
              membrane area is smaller than the smooth sphere (ratio ~0.74 at
              24^3 and ~0.76 at 32^3 -- measured, reported as
              ``staircase_area_ratio``), and the module normalises the TOTAL
              plasma-membrane flux to the lumped value, so the *local* flux
              density per um^2 is inflated by ~1/0.75 = 1.33x relative to a
              smooth sphere.  Voxel counts above ~32^3 are not recommended on
              CPU (see ``test_6_runtime_scaling()``).

SPECIES AND TIME INTEGRATION
============================
Per shell / voxel: free cytosolic Ca (c), IP3 (p), ER Ca (cER) and IP3R
inactivation (h).  c and p diffuse; cER and h do NOT diffuse by default
(``D_er`` optionally enables ER Ca diffusion -- that is an ILLUSTRATIVE
extension, not part of the published model).  `model.py`'s rapid-buffer factor
beta(c) is applied locally, exactly as published: dc/dt = beta*(fluxes).

Integrators (``integrator=``):

``'auto'`` (default)   'imex' in radial1d (the implicit tridiagonal Thomas solve
                       costs ~10 us/step, so it is both stable and cheap); in
                       grid3d 'explicit' if the caller's dt passes the explicit
                       stability test below, otherwise 'hybrid'.  The choice and
                       the reason are returned in metadata['integrator_resolved']
                       and metadata['integrator_reason'].
``'imex'``             ALL diffusion implicit (backward Euler, solved with an
                       exact tridiagonal Thomas/``solve_banded`` solve in
                       radial1d and a sparse ``splu`` factorisation in grid3d),
                       reactions explicit.  First-order in time,
                       UNCONDITIONALLY stable in dt for diffusion because the
                       implicit matrix I - dt*diag(beta)*L is a strictly
                       diagonally dominant M-matrix (L has zero row sums).
                       Stability is then set by the explicit reactions, not by
                       diffusion.  In grid3d beta(c) changes every step, so the
                       implicit matrix must be refactorised every step: measured
                       ~15-18 ms/step at 24^3-32^3, which makes grid3d 'imex'
                       much slower than 'explicit' at a CFL-limited dt.
``'hybrid'``           Ca explicit (CFL-limited), IP3 implicit with a CONSTANT
                       matrix factorised once.  Exists because the published
                       D_IP3 = 283 um^2/s is 22x D_Ca, so IP3 -- not Ca -- sets
                       the explicit dt limit.
``'explicit'``         everything explicit.  Stability criterion (Gershgorin on
                       the zero-row-sum operator L):
                           dt * max_i( s_i * |L_ii| ) <= 1  for EVERY species,
                       with s = beta(c) for Ca and s = 1 for IP3 / ER Ca.  For a
                       3-D 7-point Laplacian this is dt <= h^2/(6D), so at the
                       published D values the IP3 limit is dt <= 1.0e-4 s at
                       24^3 and dt <= 5.8e-5 s at 32^3.  ``max_stable_dt()``
                       returns this limit for a geometry before you run.
``'bdf'``              ``scipy.integrate.solve_ivp(method='BDF')`` on the whole
                       spatial system with the same tolerances `model.py` uses,
                       with a published sparsity pattern.  Adaptive, high
                       accuracy; used as the reference to measure the fixed-step
                       error of the other three.

Guard events (negative concentrations produced by an explicit step, CFL
violations, oversized single steps) are counted and returned in
``metadata['guards']`` together with ``metadata['cfl_by_species']``; non-finite
values raise immediately with the CFL numbers in the message.  There is NO
positivity clipping anywhere, matching the published model.

PARAMETER PROVENANCE
====================
* Fluxes, parameters, units, the healing time constant (tau_heal = 5.25 s) and
  the microtear damage profile come from `model.py` and are reused, not
  redefined.  ``PARAMETERS is model.PARAMETERS`` (a copy), and self_test()
  verifies that this module's per-cell flux vector is numerically identical to
  ``model._rhs_factory`` for a one-cell domain.
* Diffusion coefficients are cited literature values or explicitly flagged
  ILLUSTRATIVE -- see ``D_CA_CYTOSOL``, ``D_IP3_CYTOSOL``, ``CITATIONS`` and
  ``PROVENANCE``.  D_Ca is the dominant unknown, so ``run`` records it and
  ``test_5`` sweeps it over more than a decade.

API
===
    run(mode='radial1d', R_um=5.0, n_shells=64, n_voxels=24, t_end=5.0,
        dt=1e-4, sample_dt=0.01, tear=0.0, tear_patch=None, D_ca=None,
        D_ip3=None, parameters=None, ...) -> dict
    run_lumped(t_end=5.0, dt=1e-4, sample_dt=0.01, tear=0.0, parameters=None,
               integrator='scipy', ...) -> dict
    compare_with_lumped(**run_kwargs) -> dict
    self_test() -> dict
    full_test_suite() -> dict   (the seven required numerical tests)

Run ``python cellspace.py`` to execute the full suite and print every number.
No files are written; no plots are produced.
"""
from __future__ import annotations

import copy
import importlib.util
import time as _time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import splu

# ----------------------------------------------------------------------------
# Reuse of the published model: parameters, flux expressions, damage profile.
# ----------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / 'model.py'


def _load_published_model(path=_MODEL_PATH):
    spec = importlib.util.spec_from_file_location('_cellspace_published_model', path)
    if spec is None or spec.loader is None:      # pragma: no cover - defensive
        raise ImportError(f'cannot load published model from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_published_model()

#: Published parameters, reused verbatim (not redefined).
PARAMETERS = dict(model.PARAMETERS)

# ----------------------------------------------------------------------------
# Diffusion coefficients
# ----------------------------------------------------------------------------
CITATIONS = {
    'D_Ca_and_D_IP3': (
        "Allbritton NL, Meyer T, Stryer L (1992) 'Range of messenger action of "
        "calcium ion and inositol 1,4,5-trisphosphate', Science 258(5089):1812-1815, "
        "doi:10.1126/science.1465619 -- measured in a cytosolic extract of Xenopus "
        "laevis oocytes: D(IP3) = 283 um^2/s; D(Ca2+) rises from 13 to 65 um^2/s as "
        "free Ca rises from ~90 nM to 1 uM because of slow/immobile buffers."
    ),
}

#: Literature (cited above), value for the ~0.1 uM resting range of this model.
D_CA_CYTOSOL = 13.0
#: Literature (same citation), essentially concentration independent.
D_IP3_CYTOSOL = 283.0
#: Cited upper end of the Ca range (free Ca near 1 uM).
D_CA_CYTOSOL_HIGH_CA = 65.0

PROVENANCE = {
    'fluxes': "model.py (port of ACCRE_FullCalciumSignalingModel_Control.m, "
              "mshutson/wound-calcium-LRCa); reused, not reimplemented from scratch",
    'parameters': "model.PARAMETERS, copied verbatim",
    'damage_profile': "model.fit_microtear / model.damage_profile (published "
                      "spreadsheet fit of the microtear distribution)",
    'D_ca': f"LITERATURE-CITED: {D_CA_CYTOSOL:.1f} um^2/s at ~0.1 uM free Ca "
            f"({CITATIONS['D_Ca_and_D_IP3']}) -- measured in oocyte extract, not "
            f"in the fly epithelium of the published model, and Ca dependent "
            f"(13 -> 65 um^2/s up to 1 uM).",
    'D_ip3': f"LITERATURE-CITED: {D_IP3_CYTOSOL:.1f} um^2/s (same citation).",
    'D_er': "ILLUSTRATIVE / NOT PUBLISHED: ER Ca diffusion is absent from model.py; "
            "the default D_er = 0 (no ER diffusion) is the published assumption. "
            "Any non-zero D_er is an invented extension and any gradient result "
            "using it must be treated as unverified.",
    'geometry': "ILLUSTRATIVE: an idealised sphere (radial1d) or staircase sphere "
                "(grid3d) of radius R_um replaces a tissue cell; the real cell "
                "shape, ER distribution and cytoskeletal obstruction are unknown.",
    'cell_scale': "model.py is a TISSUE model whose cells have area 43 um^2 and "
                  "sit around a 51.25 um wound; a single 5 um cell substitutes for "
                  "one cell of that tissue. The published damage profile over a "
                  "5 um cell at the wound centre is flat to 4 significant figures "
                  "(0.9683), so 'uniform tear' is the faithful choice there.",
    'experimental_data': "NONE. No dataset in this project constrains intracellular "
                         "Ca gradients or cytosolic diffusion. All output is synthetic.",
}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _check_parameters(parameters):
    p = PARAMETERS.copy()
    if parameters:
        unknown = set(parameters) - set(p)
        if unknown:
            raise ValueError(f'Unknown parameters: {sorted(unknown)}')
        p.update(parameters)
    return p


def _sample_times(t_end, sample_dt, times=None):
    """Same sampling convention as model.py."""
    if times is not None:
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0)
                or times[0] < 0 or times[-1] > t_end):
            raise ValueError('times must be increasing and within [0, t_end]')
        return times
    return np.unique(np.r_[np.arange(0, t_end, sample_dt), t_end])


def _published_damage_peak():
    """model.damage_profile(0): the published microtear peak (dimensionless)."""
    try:
        return float(model.damage_profile(np.array([0.0]))[0])
    except Exception:                                    # pragma: no cover
        return 1.0


# ----------------------------------------------------------------------------
# 1. Geometry (finite volumes with exact shell/voxel volumes)
# ----------------------------------------------------------------------------
def build_radial1d(R_um=5.0, n_shells=64):
    """Spherically symmetric cell: N shells, exact volumes and face areas.

    Faces at r_f = f*dr, f = 0..N.  Cell i occupies [r_i, r_{i+1}] with exact
    volume (4pi/3)(r_{i+1}^3 - r_i^3) and centre (r_i + r_{i+1})/2.  The interior
    face f = 1..N-1 (area 4pi r_f^2) is exactly midway between the centres of
    cells f-1 and f, so the face gradient (c_f - c_{f-1})/dr is second-order
    accurate.  r = 0 carries area 0 (zero-flux centre); r = R is the membrane.
    """
    R_um = float(R_um)
    n = int(n_shells)
    if not np.isfinite(R_um) or R_um <= 0:
        raise ValueError('R_um must be positive and finite')
    if n < 1:
        raise ValueError('n_shells must be >= 1')
    dr = R_um / n
    r_faces = np.arange(n + 1) * dr
    volumes = (4.0 / 3.0) * np.pi * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)
    areas_all = 4.0 * np.pi * r_faces ** 2
    geo = dict(
        kind='radial1d', n=n, R_um=R_um, dx_um=dr,
        r_um=(r_faces[:-1] + r_faces[1:]) / 2, r_faces=r_faces,
        volumes=volumes, face_areas_all=areas_all,
        face_i=np.arange(n - 1), face_j=np.arange(1, n),
        face_area=areas_all[1:n].copy(), face_dist=np.full(max(n - 1, 0), dr),
        bnd_i=np.array([n - 1]), bnd_area=np.array([areas_all[n]]),
        bnd_dir=np.array([[1.0, 0.0, 0.0]]),
        bnd_centre=np.array([[R_um, 0.0, 0.0]]),
        total_volume=float(volumes.sum()), membrane_area=float(areas_all[n]),
        smooth_membrane_area=float(4.0 * np.pi * R_um ** 2),
        centres=np.column_stack([(r_faces[:-1] + r_faces[1:]) / 2,
                                 np.zeros(n), np.zeros(n)]),
        solver='banded',
    )
    return geo


def build_grid3d(R_um=5.0, n_voxels=24):
    """Cubic voxel grid of side 2R with a staircase-sphere cell mask.

    A voxel is cytosolic iff its centre lies within R_um.  Non-cytosolic voxels
    are inert and every face to them (or off the grid) is a zero-flux membrane
    face carrying the plasma-membrane flux when a tear is applied.  The
    membrane area is the sum of those staircase faces, which exceeds 4 pi R^2
    (reported as ``staircase_area_ratio``); the TOTAL membrane flux is
    normalised to the lumped value, so the local flux density per um^2 is
    diluted by the same ratio.  EXPERIMENTAL mode.
    """
    R_um = float(R_um)
    n = int(n_voxels)
    if not np.isfinite(R_um) or R_um <= 0:
        raise ValueError('R_um must be positive and finite')
    if n < 4:
        raise ValueError('n_voxels must be >= 4')
    h = 2.0 * R_um / n
    x = (np.arange(n) + 0.5) * h - R_um
    Xc, Yc, Zc = np.meshgrid(x, x, x, indexing='ij')
    centres_grid = np.stack([Xc, Yc, Zc], axis=-1)
    active_grid = (centres_grid ** 2).sum(axis=-1) <= R_um ** 2
    n_all = n ** 3
    n_act = int(active_grid.sum())
    if n_act == 0:
        raise ValueError('no cytosolic voxels: increase n_voxels or R_um')
    flat_active = active_grid.ravel()
    centers_flat = centres_grid.reshape(n_all, 3)[flat_active]
    volumes = np.full(n_act, h ** 3)

    aidx = np.full((n, n, n), -1, dtype=np.int64)
    aidx[active_grid] = np.arange(n_act)
    ext_act = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
    ext_act[1:-1, 1:-1, 1:-1] = active_grid
    ext_idx = np.full((n + 2, n + 2, n + 2), -1, dtype=np.int64)
    ext_idx[1:-1, 1:-1, 1:-1] = aidx
    dst = (slice(1, n + 1),) * 3

    fi, fj, fa, fd = [], [], [], []
    bi, ba, bdir, bctr = [], [], [], []
    for axis in range(3):
        shift = [0, 0, 0]
        shift[axis] = 1
        src = tuple(slice(1 + s, n + 1 + s) for s in shift)
        here = ext_act[dst]
        there = ext_act[src]
        both = here & there
        off = here & (~there)
        fi.append(ext_idx[dst][both])
        fj.append(ext_idx[src][both])
        k = len(fi[-1])
        fa.append(np.full(k, h * h))
        fd.append(np.full(k, h))
        bi.append(ext_idx[dst][off])
        ka = int(off.sum())
        ba.append(np.full(ka, h * h))
        d = np.zeros(3)
        d[axis] = 1.0
        bdir.append(np.tile(d, (ka, 1)))
        # face centre = voxel centre + h/2 in the outward direction
        c = np.zeros((ka, 3))
        c[:, axis] = h / 2.0
        bctr.append(centers_flat[ext_idx[dst][off]] + c)
    face_i = np.concatenate(fi)
    face_j = np.concatenate(fj)
    bnd_i = np.concatenate(bi)
    bnd_dir = np.concatenate(bdir, axis=0)
    bnd_centre = np.concatenate(bctr, axis=0)
    mem_area = float(np.concatenate(ba).sum())
    geo = dict(
        kind='grid3d', n=n_act, n_voxels=n, R_um=R_um, dx_um=h,
        volumes=volumes,
        face_i=face_i, face_j=face_j,
        face_area=np.concatenate(fa), face_dist=np.concatenate(fd),
        bnd_i=bnd_i, bnd_area=np.concatenate(ba),
        bnd_dir=bnd_dir, bnd_centre=bnd_centre,
        total_volume=float(volumes.sum()), membrane_area=mem_area,
        smooth_membrane_area=float(4.0 * np.pi * R_um ** 2),
        staircase_area_ratio=float(mem_area / (4.0 * np.pi * R_um ** 2)),
        centres=centers_flat,
        grid=dict(shape=(n, n, n), h_um=h, active=active_grid,
                  active_index=np.flatnonzero(flat_active),
                  centres_um=centres_grid, origin='cube centre at the cell centre',
                  side_um=2.0 * R_um),
        solver='sparse',
    )
    return geo


def build_geometry(mode='radial1d', R_um=5.0, n_shells=64, n_voxels=24):
    if mode == 'radial1d':
        return build_radial1d(R_um=R_um, n_shells=n_shells)
    if mode == 'grid3d':
        return build_grid3d(R_um=R_um, n_voxels=n_voxels)
    raise ValueError("mode must be 'radial1d' or 'grid3d'")


# ----------------------------------------------------------------------------
# 2. Operators and membrane sources
# ----------------------------------------------------------------------------
def diffusion_operator(geo, D):
    """Conservative FV operator L with (L c)_i = D * div_i(grad c).

    Sum_i V_i (L c)_i = 0 exactly (every interior face contributes +a*(c_j-c_i)
    and -a*(c_j-c_i)); this is what makes an initially uniform field stay
    uniform and makes diffusion mass conserving.
    """
    n = geo['n']
    a = float(D) * geo['face_area'] / geo['face_dist']
    diag = np.zeros(n)
    np.add.at(diag, geo['face_i'], -a)
    np.add.at(diag, geo['face_j'], -a)
    rows = np.r_[geo['face_i'], geo['face_j'], np.arange(n)]
    cols = np.r_[geo['face_j'], geo['face_i'], np.arange(n)]
    vals = np.r_[a, a, diag]
    L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return (diags(1.0 / geo['volumes']) @ L).tocsr()


def radial_bands(geo, D):
    """Tridiagonal (lower, diag, upper) of the radial1d operator."""
    n = geo['n']
    dr = geo['dx_um']
    a_face = float(D) * geo['face_areas_all'] / dr
    lower = np.zeros(n)
    upper = np.zeros(n)
    if n > 1:
        lower[1:] = a_face[1:n] / geo['volumes'][1:]
        upper[:-1] = a_face[1:n] / geo['volumes'][:-1]
    return lower, -(lower + upper), upper


def membrane_source_scales(geo, boundary_damage):
    """Per-cell multipliers converting a lumped PM flux (uM/s) to a local rate.

    ``pm_lumped(c_i, er_i, t)`` is model.py's whole-cell plasma-membrane flux.
    Physically it is a flux per membrane area; model.py reduces it to a
    whole-cell rate by dividing by the cell volume.  Spatially we must apply it
    where the membrane is, so cell i receives

        rate_i = V_total * (sum of membrane areas of i) / (V_i * A_membrane) * pm

    which guarantees sum_i V_i * rate_i = V_total * pm for ANY geometry (and
    exactly reproduces the lumped model when the flux is spatially uniform).
    Two scales are needed: one for the damage-independent part of the flux
    (leak, SOC, PMCA, NCX, which act over the whole membrane) and one for the
    microtear part (which acts only where the tear is).
    """
    n = geo['n']
    share_all = np.zeros(n)
    share_dmg = np.zeros(n)
    np.add.at(share_all, geo['bnd_i'], geo['bnd_area'])
    np.add.at(share_dmg, geo['bnd_i'], geo['bnd_area'] * boundary_damage)
    pref = geo['total_volume'] / (geo['volumes'] * geo['membrane_area'])
    return pref * share_all, pref * share_dmg


def parse_patch(tear_patch):
    """Normalise the tear_patch argument into ('uniform'|'cap', half_angle_deg, axis)."""
    axis = np.array([1.0, 0.0, 0.0])
    if tear_patch is None or (isinstance(tear_patch, str) and tear_patch == 'uniform'):
        return 'uniform', None, axis
    if isinstance(tear_patch, str):
        raise ValueError("tear_patch must be None, 'uniform', a float area fraction, "
                         "or a dict with half_angle_deg / frac / axis")
    if isinstance(tear_patch, (int, float, np.floating)):
        frac = float(tear_patch)
        if not 0.0 < frac <= 1.0:
            raise ValueError('tear_patch area fraction must lie in (0, 1]')
        theta = np.degrees(np.arccos(1.0 - 2.0 * frac))
        return 'cap', float(theta), axis
    if isinstance(tear_patch, dict):
        unknown = set(tear_patch) - {'kind', 'half_angle_deg', 'frac', 'axis'}
        if unknown:
            raise ValueError(f'unknown tear_patch keys: {sorted(unknown)}')
        if 'axis' in tear_patch:
            axis = np.asarray(tear_patch['axis'], dtype=float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0:
                raise ValueError('tear_patch axis must be a finite 3-vector')
            axis = axis / np.linalg.norm(axis)
        if 'half_angle_deg' in tear_patch:
            theta = float(tear_patch['half_angle_deg'])
        elif 'frac' in tear_patch:
            frac = float(tear_patch['frac'])
            if not 0.0 < frac <= 1.0:
                raise ValueError('tear_patch frac must lie in (0, 1]')
            theta = float(np.degrees(np.arccos(1.0 - 2.0 * frac)))
        else:
            raise ValueError("tear_patch dict needs 'half_angle_deg' or 'frac'")
        if not 0.0 < theta <= 180.0:
            raise ValueError('tear_patch half_angle_deg must lie in (0, 180]')
        return 'cap', theta, axis
    raise ValueError('unsupported tear_patch specification')


def patch_weights(geo, tear_patch):
    """Damage weight in [0, 1] for every membrane element, plus a report dict."""
    kind, theta, axis = parse_patch(tear_patch)
    area = geo['bnd_area']
    if kind == 'uniform':
        w = np.ones(len(area))
        info = dict(request='uniform', applied='uniform over the whole surface',
                    area_fraction=1.0, half_angle_deg=None)
    else:
        cosang = geo['bnd_dir'] @ axis
        # element centre direction (from the cell centre) as a unit vector
        ctr = geo['bnd_centre']
        if geo['kind'] == 'radial1d':
            # angular resolution is impossible in a spherically symmetric model:
            # apply the surface-area-weighted uniform equivalent.
            frac = (1.0 - np.cos(np.radians(theta))) / 2.0
            w = np.full(len(area), frac)
            info = dict(request=f'cap(half_angle_deg={theta:g}, frac={frac:.6g})',
                        applied='UNIFORM at the surface-area-weighted mean damage '
                                '= frac (radial1d cannot resolve an angular patch); '
                                'this is a genuine limitation of the symmetry',
                        area_fraction=float(frac), half_angle_deg=theta)
        else:
            w = (cosang >= np.cos(np.radians(theta))).astype(float)
            ideal = (1.0 - np.cos(np.radians(theta))) / 2.0
            real = float((area * w).sum() / area.sum())
            info = dict(request=f'cap(half_angle_deg={theta:g})',
                        applied='localised spherical cap around '
                                f'axis={axis.tolist()} in grid3d',
                        area_fraction=real, requested_area_fraction=float(ideal),
                        area_fraction_discretisation_error=float(real - ideal),
                        half_angle_deg=theta)
    return w, info


# ----------------------------------------------------------------------------
# 3. Fluxes -- transcribed from model.py's _rhs_factory, reused parameter values
# ----------------------------------------------------------------------------
def flux_terms(p, c, er, ip3, h, t, bx, alpha, reactions=True):
    """Per-cell reaction fluxes, identical to model.py's _rhs_factory.

    Returns a dict with j_er (ER exchange), rh (IP3 production), dh (IP3R
    inactivation), beta (rapid buffer), and the plasma-membrane flux split into
    its damage-independent part (pm_base: leak, SOC, PMCA, NCX) and its
    microtear part at unit damage (pm_tear_unit).
    """
    rh = alpha * c / (p['Kc'] + c) * p['alpha0']
    zeta = p['d2'] * (ip3 + p['d1']) / (ip3 + p['d3'])
    dh = p['a2'] * (zeta - (zeta + c) * h)
    m = ip3 / (p['d1'] + ip3) * c / (p['d5'] + c)
    j_ipr = p['etaIPR'] * m ** 3 * h ** 3 * (er - c)
    j_serca = p['etaSERCA'] * (c ** p['nSERCA']
                               - p['kSERCA1'] ** p['nSERCA'] * p['kSERCA2'] ** p['nSERCA']
                               * er ** p['nSERCA']) / (p['kSERCA1'] ** p['nSERCA'] + c ** p['nSERCA'])
    j_er = j_ipr + p['eta_lkER'] * (er - c) - j_serca
    heal = np.exp(-t / p['tau_heal']) * (-np.expm1(-t / p['tau_d'])) if t >= 0 else 0.0
    pmca = p['rPMCA'] * c ** p['nPMCA'] / (p['kPMCA'] ** p['nPMCA'] + c ** p['nPMCA'])
    ncx = c ** p['nNCX'] / (p['kNCX'] ** p['nNCX'] + c ** p['nNCX'])
    soc = p['rSOC'] * p['kSOC'] ** p['nSOC'] / (p['kSOC'] ** p['nSOC'] + er ** p['nSOC'])
    pm_base = p['etaNCX'] * (p['rlkPM'] * (1 - c / p['cExt']) + soc - pmca - ncx)
    pm_tear = p['etaNCX'] * p['r_microtear'] * (1 - c / p['cExt']) * heal
    beta = 1.0 / (1.0 + p['Ke'] * p['Be'] / (p['Ke'] + c) ** 2
                  + p['nx'] * p['Kx'] ** p['nx'] * bx * c ** (p['nx'] - 1)
                  / (p['Kx'] ** p['nx'] + c ** p['nx']) ** 2)
    if not reactions:
        zero = np.zeros_like(c)
        j_er, rh, dh = zero, zero, zero
    return dict(rh=rh, dh=dh, m=m, j_ipr=j_ipr, j_serca=j_serca, j_er=j_er,
                beta=beta, pm_base=pm_base, pm_tear_unit=pm_tear, heal=heal)


def beta_of(c, p, bx=None):
    """model.py's rapid-buffer factor beta(c) (reused expression)."""
    bx = p['Bx'] if bx is None else bx
    return 1.0 / (1.0 + p['Ke'] * p['Be'] / (p['Ke'] + c) ** 2
                  + p['nx'] * p['Kx'] ** p['nx'] * bx * c ** (p['nx'] - 1)
                  / (p['Kx'] ** p['nx'] + c ** p['nx']) ** 2)


def buffered_ca(c, p, bx=None):
    """Exact integral of 1/beta(c') dc' from 0 to c (units uM).

    d/dt of this quantity equals (1/beta) dc/dt, i.e. the *unbuffered* Ca
    source terms; it is therefore the correct conserved cytosolic Ca content
    when rapid buffering is treated as an instantaneous local equilibrium.
    Closed form: F(c) = c - Ke*Be/(Ke+c) - bx*Kx^nx/(Kx^nx + c^nx), whose
    derivative is exactly 1/beta(c).
    """
    bx = p['Bx'] if bx is None else bx
    return (c - p['Ke'] * p['Be'] / (p['Ke'] + c)
            - bx * p['Kx'] ** p['nx'] / (p['Kx'] ** p['nx'] + c ** p['nx']))


def conserved_amount(c, er, p, volumes, bx=None):
    """Total cell calcium content: buffered cytosol + ER-equivalent (uM*um^3).

    d/dt (F(c) + cER/epsilon) = (1/beta) dc/dt - j_er = j_er + pm + div(D grad c) - j_er
    so this quantity changes ONLY through the plasma-membrane flux, and any
    other change is numerical error.
    """
    return float(np.sum(volumes * (buffered_ca(c, p, bx) + er / p['epsilon'])))


# ----------------------------------------------------------------------------
# 4. Equilibration (the published 300 s pre-wound solve, one isolated cell)
# ----------------------------------------------------------------------------
_EQ_CACHE = {}


def _lumped_rhs_factory(p, alpha, bx, damage):
    """model.py's own RHS for a one-cell, gap-junction-free domain (reused)."""
    op = csr_matrix((1, 1))
    ablated = np.zeros(1, dtype=bool)
    return model._rhs_factory(p, op, np.array([alpha]), np.array([bx]),
                              np.array([float(damage)]), ablated)


def equilibrated_state(p, alpha=1.0, bx=1.0, equilibrate_time=300.0,
                       rtol=1e-6, atol=1e-9, method='BDF', initial=(.1, 200., .01, .67)):
    """The published pre-wound state: model.py's own 300 s solve from (.1,200,.01,.67)."""
    key = (tuple(sorted(p.items())), float(alpha), float(bx), float(equilibrate_time),
           float(rtol), float(atol), method)
    if key in _EQ_CACHE:
        y, residual = _EQ_CACHE[key]
        return y.copy(), residual
    rhs = _lumped_rhs_factory(p, alpha, bx, 0.0)
    sol = solve_ivp(rhs, (0, equilibrate_time), np.asarray(initial, dtype=float),
                    method=method, rtol=rtol, atol=atol,
                    t_eval=[equilibrate_time])
    if not sol.success:
        raise RuntimeError('Equilibration failed: ' + sol.message)
    y = sol.y[:, -1].copy()
    residual = float(np.max(np.abs(rhs(equilibrate_time, y))))
    _EQ_CACHE[key] = (y, residual)
    return y, residual


# ----------------------------------------------------------------------------
# 5. The spatial integrators
# ----------------------------------------------------------------------------
def _banded_matrix(lower, diag, upper):
    n = len(diag)
    ab = np.zeros((3, n))
    ab[1] = diag
    if n > 1:
        ab[0, 1:] = upper[:-1]
        ab[2, :-1] = lower[1:]
    return ab


def _implicit_setup(geo, L, dt):
    """Return a callable solving (I - dt*diag(s)*L) x = b for a diagonal s."""
    if geo['solver'] == 'banded':
        raise AssertionError('banded path handled inline')
    lu_cache = {}

    def solve(s, b, key=None):
        if key is not None and key in lu_cache:
            lu = lu_cache[key]
        else:
            M = (identity(geo['n'], format='csr') - diags(dt * s) @ L).tocsc()
            lu = splu(M)
            if key is not None:
                lu_cache[key] = lu
        return lu.solve(b)

    return solve


def _cfl_numbers(dt, beta, Lc, Lp, Ler):
    """Gershgorin explicit-stability numbers dt*max_i(s_i*|L_ii|) per species.

    Explicit Euler on dc/dt = s_i (L c)_i is stable iff dt*max_i(s_i |L_ii|) <= 1
    (|lambda|max = 2*max|L_ii| for a zero-row-sum operator).  s = beta for Ca
    (beta <= 1) and 1 for the unbuffered species.  IP3 diffuses ~22x faster than
    Ca, so with the published D values IP3 -- not Ca -- usually sets the limit.
    """
    out = {}
    if Lc is not None:
        out['c'] = float(dt * np.max(beta * np.abs(Lc.diagonal()))) if Lc.nnz else 0.0
    if Lp is not None:
        out['ip3'] = float(dt * np.max(np.abs(Lp.diagonal()))) if Lp.nnz else 0.0
    if Ler is not None:
        out['cER'] = float(dt * np.max(np.abs(Ler.diagonal()))) if Ler.nnz else 0.0
    return out


def max_stable_dt(geo, D_ca=None, D_ip3=None, D_er=None, beta_max=1.0):
    """Largest explicit dt for which every species passes the CFL test."""
    D_ca = D_CA_CYTOSOL if D_ca is None else D_ca
    D_ip3 = D_IP3_CYTOSOL if D_ip3 is None else D_ip3
    D_er = 0.0 if D_er is None else D_er
    limits = {}
    for name, D, s in (('c', D_ca, beta_max), ('ip3', D_ip3, 1.0), ('cER', D_er, 1.0)):
        if D <= 0:
            continue
        L = diffusion_operator(geo, D)
        d = float(np.max(np.abs(L.diagonal()))) if L.nnz else 0.0
        limits[name] = np.inf if d == 0 else 1.0 / (s * d)
    return min(limits.values()), limits


def _integrate(geo, p, D_ca, D_ip3, D_er, boundary_damage, bx, alpha, y0, times,
               dt, integrator, reactions, verbose=False):
    """Fixed-step IMEX / explicit, or adaptive BDF, on the spatial system."""
    n = geo['n']
    volumes = geo['volumes']
    if integrator == 'auto':
        if geo['solver'] == 'banded':
            integrator = 'imex'
            auto_reason = ('radial1d: the implicit tridiagonal Thomas solve costs ~10 us '
                           'per step, so full imex is both stable and cheap')
        else:
            _stable, _limits = max_stable_dt(geo, D_ca, D_ip3, D_er, beta_max=1.0)
            if dt <= _stable:
                integrator = 'explicit'
                auto_reason = (f'grid3d: dt={dt:g} s is within the explicit stability '
                               f'limit {_stable:g} s, and a measured explicit step is '
                               f'far cheaper than a 3-D sparse triangular solve')
            else:
                integrator = 'hybrid'
                auto_reason = (f'grid3d: dt={dt:g} s exceeds the explicit stability '
                               f'limit {_stable:g} s, so IP3 diffusion is solved '
                               f'implicitly (constant matrix, one factorisation)')
    else:
        auto_reason = 'caller-specified'
    if integrator not in ('imex', 'hybrid', 'explicit', 'bdf'):
        raise ValueError("integrator must be 'auto', 'imex', 'hybrid', 'explicit' or 'bdf'")
    implicit_c = integrator == 'imex'
    implicit_ip3 = integrator in ('imex', 'hybrid')
    Lc = diffusion_operator(geo, D_ca)
    Lp = diffusion_operator(geo, D_ip3) if D_ip3 else None
    Ler = diffusion_operator(geo, D_er) if D_er else None
    s_all, s_dmg = membrane_source_scales(geo, boundary_damage)
    abs_diag = np.abs(Lc.diagonal())

    def rhs(t, y):
        c, er, ip3, h = y.reshape(4, n)
        f = flux_terms(p, c, er, ip3, h, t, bx, alpha, reactions)
        pm = f['pm_base'] * s_all + f['pm_tear_unit'] * s_dmg
        dc = f['beta'] * (f['j_er'] + pm + (Lc @ c))
        der = -p['epsilon'] * f['j_er'] + (Ler @ er if Ler is not None else 0.0)
        dip3 = f['rh'] - p['kdeg'] * ip3 + (Lp @ ip3 if Lp is not None else 0.0)
        return np.concatenate((dc, der, dip3, f['dh']))

    if integrator == 'bdf':
        ref = Lc
        if Lp is not None:
            ref = ref + Lp
        if Ler is not None:
            ref = ref + Ler
        pattern = model._sparsity(ref)
        t0 = _time.perf_counter()
        sol = solve_ivp(rhs, (0.0, float(times[-1])), y0.ravel(), method='BDF',
                        rtol=1e-8, atol=1e-12, jac_sparsity=pattern, t_eval=times,
                        first_step=min(1e-8, float(times[-1])))
        if not sol.success:
            raise RuntimeError('BDF integration failed: ' + sol.message)
        values = sol.y.T.reshape(len(times), 4, n)
        diag = dict(step_count=int(sol.nfev), seconds=_time.perf_counter() - t0,
                    cfl_max=float('nan'), nfev=int(sol.nfev), njev=int(sol.njev),
                    nlu=int(sol.nlu))
        return values, diag

    if integrator not in ('imex', 'hybrid', 'explicit'):
        raise ValueError("integrator must be 'imex', 'hybrid', 'explicit' or 'bdf'")

    y = np.array(y0, dtype=float).reshape(4, n).copy()
    values = np.empty((len(times), 4, n))
    values[0] = y
    idx = 1
    t = 0.0
    steps = 0
    cfl_max = 0.0
    cfl_by_species = {}
    cfl_viol = 0
    neg_events = 0
    big_steps = 0
    max_rel_step = 0.0
    pm_acc = 0.0                       # integral of sum_i V_i pm_i dt
    pm_series = np.zeros(len(times))   # same, sampled
    t0 = _time.perf_counter()

    # constant pieces of the implicit setup
    if implicit_ip3:
        if geo['solver'] == 'banded':
            lower_p, diag_p, upper_p = radial_bands(geo, D_ip3)
            ab_p = _banded_matrix(lower_p, diag_p + p['kdeg'], upper_p)
            lower_c, diag_c, upper_c = radial_bands(geo, D_ca)
        else:
            Lp_eff = Lp if Lp is not None else csr_matrix((n, n))
            Mp = (identity(n, format='csr')*(1.0 + dt * p['kdeg']) - dt * Lp_eff).tocsc()
            lu_p = splu(Mp)
            lu_solve = _implicit_setup(geo, Lc, dt)
    elif implicit_c:                                     # pragma: no cover
        raise AssertionError('implicit c without implicit ip3 is not implemented')
    else:
        lower_c, diag_c, upper_c = None, None, None

    while t < float(times[-1]) - 1e-12:
        t_next = float(times[idx])
        h = dt if dt < t_next - t else t_next - t
        c, er, ip3, hh = y
        f = flux_terms(p, c, er, ip3, hh, t, bx, alpha, reactions)
        pm = f['pm_base'] * s_all + f['pm_tear_unit'] * s_dmg
        react = f['j_er'] + pm
        pm_acc += h * float(volumes @ pm)
        lax = _cfl_numbers(h, f['beta'],
                           None if implicit_c else Lc,
                           None if implicit_ip3 else Lp,
                           Ler)
        for k, v in lax.items():
            cfl_by_species[k] = max(cfl_by_species.get(k, 0.0), v)
        cfl_here = max(lax.values()) if lax else 0.0
        cfl_max = max(cfl_max, cfl_here)
        if cfl_here > 1.0:
            cfl_viol += 1
        # --- Ca update -------------------------------------------------------
        if implicit_c:
            rhs_c = c + h * f['beta'] * react
            if geo['solver'] == 'banded':
                ab = _banded_matrix(lower_c, diag_c.copy(), upper_c)
                ab[1] = 1.0 - h * f['beta'] * diag_c
                ab[0, 1:] = -h * f['beta'][:-1] * upper_c[:-1]
                ab[2, :-1] = -h * f['beta'][1:] * lower_c[1:]
                c_new = solve_banded((1, 1), ab, rhs_c)
            else:
                c_new = lu_solve(h * f['beta'], rhs_c)
        else:
            c_new = c + h * f['beta'] * (react + (Lc @ c))
        # --- IP3 update ------------------------------------------------------
        if implicit_ip3:
            ip3_rhs = ip3 + h * f['rh']
            ip3_new = (solve_banded((1, 1), ab_p, ip3_rhs) if geo['solver'] == 'banded'
                       else lu_p.solve(ip3_rhs))
        else:
            ip3_new = ip3 + h * (f['rh'] - p['kdeg'] * ip3
                                 + (Lp @ ip3 if Lp is not None else 0.0))
        # --- ER and gating ---------------------------------------------------
        er_new = er - h * p['epsilon'] * f['j_er']
        if Ler is not None:
            # illustrative ER diffusion, explicit (not part of the published model)
            er_new = er_new + h * (Ler @ er)
        h_new = hh + h * f['dh']
        if np.any(c_new < 0) or np.any(er_new < 0) or np.any(ip3_new < 0):
            neg_events += 1
        dcmax = float(np.max(np.abs(c_new - c)))
        scale = max(float(np.max(np.abs(c))), 1e-12)
        relstep = (dcmax / scale) if implicit_c else (dcmax * h / scale)
        max_rel_step = max(max_rel_step, relstep)
        if relstep > 0.25:
            big_steps += 1
        if not (np.all(np.isfinite(c_new)) and np.all(np.isfinite(er_new))
                and np.all(np.isfinite(ip3_new)) and np.all(np.isfinite(h_new))):
            raise RuntimeError(
                f'Non-finite state at step {steps}, t={t:g} s '
                f'(integrator={integrator!r}, cfl_by_species={cfl_by_species}, '
                f'c_min={c.min():g}, er_min={er.min():g}, ip3_min={ip3.min():g}). '
                f'A CFL number > 1 means dt is too large for an explicit scheme; '
                f'use a smaller dt or integrator="imex".')
        y = np.array([c_new, er_new, ip3_new, h_new])
        t = t_next if h == t_next - t else t + h
        steps += 1
        if idx < len(times) and t >= float(times[idx]) - 1e-12:
            values[idx] = y
            pm_series[idx] = pm_acc
            idx += 1
            while idx < len(times) and float(times[idx]) <= t:
                values[idx] = y
                pm_series[idx] = pm_acc
                idx += 1
        if verbose and steps % 20000 == 0:
            print(f'    step {steps}, t={t:g}/{times[-1]:g}')
    if idx != len(times):
        raise RuntimeError(f'sampling bookkeeping failed: recorded {idx}/{len(times)}')
    diag = dict(step_count=steps, seconds=_time.perf_counter() - t0,
                cfl_max=float(cfl_max), cfl_by_species={k: float(v) for k, v in cfl_by_species.items()},
                cfl_violations=cfl_viol,
                negative_events=neg_events, oversized_steps=big_steps,
                max_relative_change_per_step=float(max_rel_step),
                integrator=integrator, auto_reason=auto_reason, pm_integral=pm_series)
    return values, diag


# ----------------------------------------------------------------------------
# 6. Public run API
# ----------------------------------------------------------------------------
def _prepare_initial(geo, p, alpha, bx, initial, blip, equilibrate,
                     equilibrate_time, rtol, atol):
    n = geo['n']
    if initial is None:
        if equilibrate:
            base, resid = equilibrated_state(p, alpha, bx, equilibrate_time, rtol, atol)
        else:
            base = np.array([.1, 200., .01, .67])
            resid = float('nan')
        y = np.repeat(base[:, None], n, axis=1)
    else:
        resid = float('nan')
        unknown = set(initial) - {'c', 'er', 'ip3', 'h'}
        if unknown:
            raise ValueError(f'unknown initial keys: {sorted(unknown)}')
        defaults = {'c': .1, 'er': 200., 'ip3': .01, 'h': .67}
        y = np.empty((4, n))
        for k, row in (('c', 0), ('er', 1), ('ip3', 2), ('h', 3)):
            v = np.asarray(initial.get(k, defaults[k]), dtype=float)
            if v.ndim == 0:
                y[row] = v
            elif v.shape == (n,):
                y[row] = v
            else:
                raise ValueError(f'initial[{k}] must be a scalar or shape ({n},)')
    if blip is not None:
        unknown = set(blip) - {'sigma_um', 'amplitude_uM', 'centre_um', 'species'}
        if unknown:
            raise ValueError(f'unknown blip keys: {sorted(unknown)}')
        sigma = float(blip.get('sigma_um', 0.5))
        amp = float(blip.get('amplitude_uM', 1.0))
        ctr = np.asarray(blip.get('centre_um', (0.0, 0.0, 0.0)), dtype=float)
        species = blip.get('species', 'c')
        row = {'c': 0, 'er': 1, 'ip3': 2, 'h': 3}[species]
        d2 = ((geo['centres'] - ctr) ** 2).sum(axis=1)
        y[row] = y[row] + amp * np.exp(-d2 / (2.0 * sigma ** 2))
    return y, resid


def run(mode='radial1d', R_um=5.0, n_shells=64, n_voxels=24, t_end=5.0, dt=1e-4,
        sample_dt=0.01, tear=0.0, tear_patch=None, D_ca=None, D_ip3=None,
        parameters=None, *, D_er=None, integrator='auto', reactions=True,
        equilibrate=True, equilibrate_time=300.0, eq_rtol=1e-6, eq_atol=1e-9,
        times=None, initial=None, blip=None, verbose=False):
    """Integrate one spatially resolved cell and return a dict of arrays.

    Spatial / stimulus arguments
    ----------------------------
    mode        'radial1d' (spherically symmetric shells) or 'grid3d'
                (experimental staircase-sphere voxel grid).
    R_um        cell radius (default 5 um; a 5 um sphere is 523.6 um^3).
    n_shells    radial1d shell count (default 64).
    n_voxels    grid3d voxels per axis (default 24).
    t_end, dt   end time (s) and MAXIMUM fixed time step (s).  A shorter step is
                taken to land exactly on each sample time.  dt is not adaptive.
    sample_dt   output interval (s); times = unique(arange(0,t_end,sample_dt), t_end).
    tear        microtear amplitude in [0, 1], the MEAN surface damage relative to
                the published peak damage_profile(r=0)=0.9683.  tear=0 means no
                wound (leak/SOC/PMCA/NCX only).  Healing uses the published
                tau_heal = 5.25 s (override via ``parameters``).
    tear_patch  None/'uniform' (whole surface), a float area fraction in (0,1], or
                a dict {'half_angle_deg'|'frac', 'axis'} for a spherical cap.
                radial1d cannot resolve an angular patch and applies the
                surface-area-weighted uniform equivalent (see metadata).
    D_ca, D_ip3 cytosolic diffusion coefficients (um^2/s).  Defaults are the cited
                literature values 13 and 283 (see CITATIONS).  D_er is an
                ILLUSTRATIVE ER-Ca diffusion option, default 0 (as published).
    parameters  overrides for model.PARAMETERS (published values, reused).
    integrator  'auto' (default), 'imex', 'hybrid', 'explicit' or 'bdf'.  'auto'
                resolves to 'imex' in radial1d, and in grid3d to 'explicit' when
                the caller's dt passes the explicit stability test, otherwise to
                'hybrid' (IP3 implicit).  The resolved choice and the reason are
                returned in metadata['integrator_resolved'] and
                metadata['integrator_reason'].  See the module docstring for the
                stability criteria, which are reported per species.
    reactions   False zeroes all chemistry (diagnostic switch for the pure
                diffusion / pure conservation tests; NOT a biological claim).
    equilibrate True: start from the published 300 s pre-wound state.
    initial     dict(c=..., er=..., ip3=..., h=...) of scalars or (n,) arrays,
                overriding the equilibrated state (used by the accuracy tests).
    blip        dict(sigma_um, amplitude_uM, centre_um, species) Gaussian added
                to a species' initial condition.
    """
    start = _time.perf_counter()
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError('t_end must be positive')
    if not np.isfinite(dt) or dt <= 0 or dt > t_end:
        raise ValueError('dt must lie in (0, t_end]')
    if integrator not in ('auto', 'imex', 'hybrid', 'explicit', 'bdf'):
        raise ValueError("integrator must be 'auto', 'imex', 'hybrid', 'explicit' or 'bdf'")
    if sample_dt <= 0:
        raise ValueError('sample_dt must be positive')
    tear = float(tear)
    if not 0.0 <= tear <= 1.0:
        raise ValueError('tear must lie in [0, 1]')
    p = _check_parameters(parameters)
    alpha, bx = p['alpha'] * 1.0, p['Bx'] * 1.0   # one cell: mean multipliers
    D_ca = D_CA_CYTOSOL if D_ca is None else float(D_ca)
    D_ip3 = D_IP3_CYTOSOL if D_ip3 is None else float(D_ip3)
    D_er = 0.0 if D_er is None else float(D_er)
    for name, val in (('D_ca', D_ca), ('D_ip3', D_ip3), ('D_er', D_er)):
        if val < 0 or not np.isfinite(val):
            raise ValueError(f'{name} must be finite and non-negative')

    geo = build_geometry(mode, R_um, n_shells, n_voxels)
    times = _sample_times(t_end, sample_dt, times)
    y0, eq_resid = _prepare_initial(geo, p, alpha, bx, initial, blip, equilibrate,
                                    equilibrate_time, eq_rtol, eq_atol)

    w, pinfo = patch_weights(geo, tear_patch)
    boundary_damage = tear * w
    mean_damage = float((geo['bnd_area'] * boundary_damage).sum() / geo['bnd_area'].sum())
    # the published peak damage at the wound centre (flat across a 5 um cell)
    peak_damage = _published_damage_peak()
    boundary_damage = boundary_damage * peak_damage
    mean_damage *= peak_damage

    values, idiag = _integrate(geo, p, D_ca, D_ip3, D_er, boundary_damage, bx, alpha,
                               y0, times, dt, integrator, reactions, verbose=verbose)
    # pre-flight explicit-stability limit for the caller's dt (reported, not enforced)
    try:
        stable_dt = max_stable_dt(geo, D_ca, D_ip3, D_er, beta_max=1.0)
    except Exception:                                    # pragma: no cover
        stable_dt = (float('nan'), {})
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Non-finite solution values')
    c, er, ip3, h = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    times = np.asarray(times, dtype=float)
    volw = geo['volumes'] / geo['total_volume']
    mean_c = c @ volw
    content = np.array([conserved_amount(c[i], er[i], p, geo['volumes'], bx)
                        for i in range(len(times))])
    metadata = dict(
        module='cellspace', synthetic=True, not_measured=True,
        is_tissue_model=False,
        single_cell_note='ONE cell with an interior; no neighbours, no gap junctions, '
                         'no ligand field, no mechanics, no ablation.',
        honesty=['model output, not measurement',
                 'a spatially resolved single cell is NOT a tissue model',
                 'no experimental dataset in this project constrains intracellular '
                 'calcium gradients or cytosolic diffusion',
                 'D_ca/D_ip3 are cited values measured in frog oocyte extract, not '
                 'in the fly epithelium of the published model',
                 'grid3d is EXPERIMENTAL (staircase membrane area inflation)'],
        mode=mode, R_um=geo['R_um'], n_cells=geo['n'],
        n_shells=(geo['n'] if mode == 'radial1d' else None),
        n_voxels=(n_voxels if mode == 'grid3d' else None),
        dx_um=geo['dx_um'], total_volume_um3=geo['total_volume'],
        membrane_area_um2=geo['membrane_area'],
        staircase_area_ratio=geo.get('staircase_area_ratio'),
        grid3d_experimental=(mode == 'grid3d'),
        dt=dt, n_steps=idiag['step_count'], integrator=integrator,
        integrator_resolved=idiag.get('integrator', integrator),
        integrator_reason=idiag.get('auto_reason', 'caller-specified'),
        integration_seconds=idiag['seconds'], cfl_max=idiag['cfl_max'],
        cfl_by_species=idiag.get('cfl_by_species', {}),
        stable_dt_explicit=stable_dt[0], stable_dt_explicit_by_species=stable_dt[1],
        dt_exceeds_explicit_limit=bool(dt > stable_dt[0]),
        cfl_note=('explicit stability requires dt*max_i(s_i*|L_ii|) <= 1 for every '
                  'diffusing species (s = beta for Ca, 1 for IP3); the imex scheme is '
                  'unconditionally stable in dt for diffusion, while the explicit '
                  'reactions still need dt small compared with the fastest rate'),
        guards=dict(cfl_violations=idiag.get('cfl_violations', 0),
                    negative_events=idiag.get('negative_events', 0),
                    oversized_steps=idiag.get('oversized_steps', 0),
                    max_relative_change_per_step=idiag.get('max_relative_change_per_step', 0.0)),
        reactions_on=bool(reactions),
        tear=tear, tear_patch=pinfo, tear_patch_applied=pinfo['applied'],
        boundary_mean_damage=float(mean_damage),
        published_damage_peak=float(peak_damage),
        tau_heal_s=p['tau_heal'], tau_d_s=p['tau_d'], r_microtear=p['r_microtear'],
        D_ca_um2_s=D_ca, D_ip3_um2_s=D_ip3, D_er_um2_s=D_er,
        D_ca_provenance=PROVENANCE['D_ca'], D_ip3_provenance=PROVENANCE['D_ip3'],
        D_er_provenance=PROVENANCE['D_er'],
        citations=CITATIONS, parameters=copy.deepcopy(p), provenance=dict(PROVENANCE),
        equilibration_rtol=eq_rtol, equilibration_atol=eq_atol,
        equilibrated=bool(equilibrate and initial is None),
        equilibration_max_abs_rhs=eq_resid,
        initial=(y0[0].copy(), y0[1].copy(), y0[2].copy(), y0[3].copy()),
        minima=dict(c=float(c.min()), cER=float(er.min()), ip3=float(ip3.min()),
                    h=float(h.min())),
        maxima=dict(c=float(c.max()), cER=float(er.max()), ip3=float(ip3.max()),
                    h=float(h.max())),
        total_seconds=_time.perf_counter() - start,
        memory_note=('grid3d stores 4 species per active voxel plus one sparse '
                     'operator; radial1d is tridiagonal'),
        units=dict(times='s', c='uM', cER='uM (per ER volume, as published)',
                   ip3='uM', h='dimensionless', length='um', volume='um^3',
                   content='uM*um^3'),
        limitations=['one cell only, no tissue coupling',
                     'idealised sphere/cube geometry',
                     'D_Ca is the dominant unknown and is swept in test 5',
                     'ER and IP3R state are per-voxel with no ER diffusion by default '
                     '(D_er is an illustrative non-published option)',
                     'no experimental constraint on intracellular gradients'],
    )
    out = dict(times=times, t=times, c=c, cER=er, ip3=ip3, h=h, mean_c=mean_c,
               volume_weights=volw, volumes=geo['volumes'], content=content,
               pm_amount=idiag.get('pm_integral'),
               r_um=(geo['r_um'] if mode == 'radial1d' else None),
               grid=(geo['grid'] if mode == 'grid3d' else None),
               geometry=geo, metadata=metadata)
    return out


# ----------------------------------------------------------------------------
# 7. The published lumped model, for the comparison
# ----------------------------------------------------------------------------
def run_lumped(t_end=5.0, dt=1e-4, sample_dt=0.01, tear=0.0, parameters=None,
               *, integrator='scipy', equilibrate=True, equilibrate_time=300.0,
               rtol=1e-6, atol=1e-9, times=None, initial=None, blip=None,
               reactions=True, verbose=False):
    """Well-mixed single cell: model.py's own equations, no gap junctions.

    Same parameters and same stimulus as ``run`` with mode='radial1d'.  The
    published equations are used verbatim through ``model._rhs_factory`` on a
    one-cell, zero-coupling domain, so this IS the published lumped model (with
    the tissue-level heterogeneity multipliers set to their means).

    integrator='scipy' (default) uses solve_ivp(BDF) exactly as model.py does.
    integrator='imex'/'hybrid'/'explicit' uses this module's fixed-step scheme on
    the same equations, which isolates the fixed-step time-discretisation error
    from the spatial error when comparing against ``run``.
    """
    start = _time.perf_counter()
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError('t_end must be positive')
    if not np.isfinite(dt) or dt <= 0 or dt > t_end:
        raise ValueError('dt must lie in (0, t_end]')
    tear = float(tear)
    if not 0.0 <= tear <= 1.0:
        raise ValueError('tear must lie in [0, 1]')
    p = _check_parameters(parameters)
    alpha, bx = p['alpha'], p['Bx']
    times = _sample_times(t_end, sample_dt, times)
    peak = _published_damage_peak()
    damage = tear * peak
    if initial is None:
        if equilibrate:
            base, resid = equilibrated_state(p, alpha, bx, equilibrate_time, rtol, atol)
        else:
            base, resid = np.array([.1, 200., .01, .67]), float('nan')
        y0 = base.copy()
    else:
        resid = float('nan')
        defaults = {'c': .1, 'er': 200., 'ip3': .01, 'h': .67}
        y0 = np.array([float(np.asarray(initial.get(k, defaults[k])).ravel()[0])
                       for k in ('c', 'er', 'ip3', 'h')])
    if blip is not None:
        y0[0] += float(blip.get('amplitude_uM', 0.0))
    if integrator == 'scipy':
        rhs = _lumped_rhs_factory(p, alpha, bx, damage if reactions else 0.0)
        pattern = model._sparsity(csr_matrix((1, 1)))
        sol = solve_ivp(rhs, (0.0, t_end), y0, method='BDF', rtol=rtol, atol=atol,
                        jac_sparsity=pattern, t_eval=times, first_step=min(1e-5, t_end))
        if not sol.success:
            raise RuntimeError('Lumped integration failed: ' + sol.message)
        values = sol.y.T.reshape(len(times), 4, 1)
        seconds = _time.perf_counter() - start
        diag = dict(step_count=int(sol.nfev), seconds=seconds, cfl_max=float('nan'),
                    cfl_violations=0, negative_events=0, oversized_steps=0,
                    max_relative_change_per_step=float('nan'))
    else:
        geo = build_radial1d(R_um=1.0, n_shells=1)
        boundary = np.array([damage])
        values, diag = _integrate(geo, p, 0.0, 0.0, 0.0, boundary, bx, alpha,
                                  y0, times, dt, integrator, reactions, verbose=verbose)
        seconds = _time.perf_counter() - start
    c, er, ip3, h = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    metadata = dict(module='cellspace.run_lumped', synthetic=True, not_measured=True,
                    model='model.py published equations, single isolated cell '
                          '(no gap junctions); tissue heterogeneity multipliers at '
                          'their means',
                    integrator=integrator, dt=dt, n_steps=diag['step_count'],
                    integration_seconds=diag['seconds'], tear=tear,
                    boundary_damage=float(damage), tau_heal_s=p['tau_heal'],
                    parameters=copy.deepcopy(p),
                    equilibration_max_abs_rhs=resid,
                    minima=dict(c=float(c.min()), cER=float(er.min()),
                                ip3=float(ip3.min()), h=float(h.min())),
                    total_seconds=_time.perf_counter() - start,
                    units=dict(times='s', c='uM', cER='uM', ip3='uM', h='-'))
    return dict(times=times, t=times, c=c, cER=er, ip3=ip3, h=h, mean_c=c[:, 0],
                volume_weights=np.ones(1), content=None, r_um=np.array([0.5]),
                grid=None, metadata=metadata)


# ----------------------------------------------------------------------------
# 8. Spatial vs lumped: the quantification
# ----------------------------------------------------------------------------
def near_membrane_mask(geo, depth_um):
    """Cells within ``depth_um`` of the membrane (by centroid distance)."""
    if geo['kind'] == 'radial1d':
        return (geo['R_um'] - geo['r_um']) <= depth_um + 1e-12
    d = np.linalg.norm(geo['centres'], axis=1)
    return (geo['R_um'] - d) <= depth_um + 1e-12


def compare_with_lumped(depth_um=0.5, grad_depth_um=None, **kwargs):
    """Run both models for the same parameters/stimulus and quantify the gap.

    Returns a dict with the volume-averaged difference time series, its maximum
    and RMS (relative to the lumped value), the time of the maximum, and the
    peak intracellular Ca gradient (max-min over shells as a fraction of the
    volume-weighted mean, plus a shell-count-robust sub-membrane measure).
    """
    kwargs = dict(kwargs)
    lumped_keys = {'t_end', 'dt', 'sample_dt', 'parameters', 'integrator',
                   'equilibrate', 'equilibrate_time', 'rtol', 'atol', 'times',
                   'initial', 'blip', 'reactions', 'verbose'}
    spatial = run(**kwargs)
    lum_kwargs = {k: v for k, v in kwargs.items() if k in lumped_keys}
    # The lumped model must see the SAME mean surface damage as the spatial one.
    # For a localised patch this is the tear amplitude scaled by the membrane
    # area fraction the patch actually covers; otherwise the comparison would
    # silently compare different stimuli.
    meta = spatial['metadata']
    frac = float(meta['tear_patch']['area_fraction'])
    tear = float(kwargs.get('tear', 0.0))
    lum_kwargs['tear'] = tear * frac
    spatial_lumped_scheme = run_lumped(integrator='imex', **lum_kwargs)
    lum_kwargs_scipy = dict(lum_kwargs)
    lum_kwargs_scipy.pop('integrator', None)
    lumped = run_lumped(integrator='scipy', **lum_kwargs_scipy)

    t = spatial['times']
    mc = spatial['mean_c']
    lc = lumped['mean_c']
    if len(t) != len(lumped['times']) or not np.allclose(t, lumped['times']):
        raise RuntimeError('sampling mismatch between spatial and lumped runs')
    diff = mc - lc
    scale = max(abs(float(lc.max())), 1e-12)
    rel = diff / np.maximum(np.abs(lc), 1e-30)
    # The transient at t <~ 0.1 s has a tiny denominator, so the pointwise
    # relative difference there is a small-signal artefact.  Report it, but also
    # report the difference restricted to the window where the lumped signal
    # exceeds 10% of its own peak, plus the difference normalised by the peak.
    big = np.abs(lc) >= 0.1 * scale
    rel_big = rel[big]
    i = int(np.argmax(np.abs(diff)))
    ir = int(np.argmax(np.abs(rel)))
    c = spatial['c']
    volw = spatial['volume_weights']
    mean = c @ volw
    grad = c.max(axis=1) - c.min(axis=1)
    grad_frac = grad / np.maximum(mean, 1e-30)
    depth = grad_depth_um if grad_depth_um is not None else depth_um
    mask = near_membrane_mask(spatial['geometry'], depth)
    nm_frac = np.full(len(t), np.nan)
    if mask.any() and not mask.all():
        wm = spatial['volumes'][mask]
        near = c[:, mask] @ (wm / wm.sum())
        nm_frac = (near - mean) / np.maximum(mean, 1e-30)
    j = int(np.argmax(grad_frac))
    jn = int(np.nanargmax(np.abs(nm_frac))) if np.any(np.isfinite(nm_frac)) else 0
    return dict(
        times=t, mean_c_spatial=mc, mean_c_lumped=lc, diff=diff, rel_diff=rel,
        max_abs_diff=float(np.max(np.abs(diff))),
        max_abs_diff_fraction_of_lumped_peak=float(np.max(np.abs(diff)) / scale),
        max_rel_diff=float(np.max(np.abs(rel))),
        time_of_max_rel_diff_s=float(t[ir]),
        rel_diff_at_max_rel=float(rel[ir]),
        large_signal_threshold_fraction_of_peak=0.1,
        max_rel_diff_large_signal=float(np.max(np.abs(rel_big))),
        time_of_max_rel_diff_large_signal_s=float(t[big][int(np.argmax(np.abs(rel_big)))]),
        rms_rel_diff_large_signal=float(np.sqrt(np.mean(rel_big ** 2))),
        rms_rel_diff=float(np.sqrt(np.mean(rel ** 2))),
        rms_abs_diff=float(np.sqrt(np.mean(diff ** 2))),
        lumped_stimulus_tear=float(lum_kwargs['tear']),
        patch_area_fraction=float(frac),
        time_of_max_diff_s=float(t[i]), value_at_max=dict(spatial=float(mc[i]),
                                                          lumped=float(lc[i])),
        peak_gradient_uM=float(grad.max()),
        peak_gradient_fraction_of_mean=float(grad_frac.max()),
        time_of_peak_gradient_s=float(t[j]),
        submembrane_excess_fraction=float(np.nanmax(np.abs(nm_frac))),
        time_of_max_submembrane_excess_s=float(t[jn]),
        submembrane_depth_um=float(depth),
        sublumped_time_scheme=dict(
            max_abs=float(np.max(np.abs(spatial_lumped_scheme['mean_c'] - lc))),
            max_rel=float(np.max(np.abs((spatial_lumped_scheme['mean_c'] - lc)
                                        / np.maximum(np.abs(lc), 1e-30))))),
        lumped_peak_c_uM=float(lc.max()),
        lumped_scale_uM=float(scale),
        spatial_metadata=spatial['metadata'], lumped_metadata=lumped['metadata'],
        spatial=spatial, lumped=lumped,
    )


# ----------------------------------------------------------------------------
# 9. Tests
# ----------------------------------------------------------------------------
def _shell_average_radial(geo, f, n_gl=64):
    """Exact-ish shell averages of a radial function f(r) via Gauss-Legendre."""
    nodes, weights = np.polynomial.legendre.leggauss(n_gl)
    out = np.zeros(geo['n'])
    rf = geo['r_faces']
    for i in range(geo['n']):
        a, b = rf[i], rf[i + 1]
        r = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        out[i] = (0.5 * (b - a) * np.sum(weights * 4.0 * np.pi * r ** 2 * f(r))
                  / geo['volumes'][i])
    return out


def _gaussian_shell_average(geo, sigma, t, D, total):
    """Shell averages of the unbounded 3-D Gaussian solution, mass ``total``."""
    s2 = sigma ** 2 + 2.0 * D * t

    def f(r):
        return total / (2.0 * np.pi * s2) ** 1.5 * np.exp(-r ** 2 / (2.0 * s2))

    return _shell_average_radial(geo, f)


def test_1_uniform_pure_diffusion():
    """No reactions, no membrane flux, uniform IC: must stay uniform to round-off."""
    out = {}
    for mode, kw in (('radial1d', dict(n_shells=64)),
                     ('grid3d', dict(n_voxels=16))):
        res = run(mode=mode, t_end=0.05, dt=1e-4, sample_dt=0.05,
                  parameters=dict(etaNCX=0.0), reactions=False,
                  initial=dict(c=1.0, er=200., ip3=0.01, h=0.67), **kw)
        c = res['c']
        dev = float(np.max(np.abs(c - 1.0)))
        spread = float(np.max(c.max(axis=1) - c.min(axis=1)))
        out[mode] = dict(max_abs_deviation=dev, max_spread=spread,
                         relative=dev, n_steps=res['metadata']['n_steps'],
                         content_drift=float(abs(res['content'][-1] - res['content'][0])
                                             / abs(res['content'][0])))
    return out


def test_2_analytic_diffusion(refinements=(32, 64, 128), R_um=40.0, sigma=1.0,
                              D=13.0, t_list=(0.05, 0.1, 0.25, 0.5)):
    """Delta-like blob vs the analytic spherical Gaussian, and its convergence.

    Buffers are switched off (Be = Bx = 0 makes beta exactly 1) and reactions and
    membrane fluxes are off, so the PDE is exactly dc/dt = D*laplacian(c) and the
    unbounded 3-D Gaussian is the exact solution.  R = 40 um keeps the reflecting
    boundary at exp(-R^2/2s^2) < 1e-25 of the peak for every time tested.  The
    initial condition is the EXACT shell average of the Gaussian, so there is no
    initial projection error and the measured error is purely spatial.
    """
    rows = []
    for n in refinements:
        geo = build_radial1d(R_um=R_um, n_shells=n)
        total = 1.0
        c0 = _gaussian_shell_average(geo, sigma, 0.0, D, total)
        res = run(mode='radial1d', R_um=R_um, n_shells=n, t_end=max(t_list),
                  dt=1e-4, sample_dt=min(t_list),
                  parameters=dict(etaNCX=0.0, Be=0.0, Bx=0.0),
                  reactions=False, equilibrate=False,
                  initial=dict(c=c0, er=200., ip3=0.01, h=0.67),
                  integrator='bdf')
        t = res['times']
        worst_rel, worst_linf, worst_l2 = 0.0, 0.0, 0.0
        worst_rel_tail = 0.0
        per_time = {}
        mass_err = 0.0
        for tt in t_list:
            k = int(np.argmin(np.abs(t - tt)))
            num = res['c'][k]
            exact = _gaussian_shell_average(geo, sigma, t[k], D, total)
            peak = exact.max()
            d = np.abs(num - exact)
            # true pointwise relative error, restricted to shells carrying at
            # least 1e-4 of the peak (in the far tail the Gaussian is ~1e-70 and
            # a pointwise relative error there is meaningless).
            keep = exact >= 1e-4 * peak
            rel = float(np.max(d[keep] / exact[keep]))
            rel_tail = float(np.max(d / np.maximum(exact, 1e-300)))
            linf = float(d.max() / peak)
            l2 = float(np.sqrt(np.sum(geo['volumes'] * d ** 2)
                               / np.sum(geo['volumes'] * exact ** 2)))
            per_time[f'{t[k]:g}'] = dict(max_rel=rel, max_rel_unrestricted_tail=rel_tail,
                                         linf_rel_peak=linf, l2_rel=l2,
                                         shells_kept=int(keep.sum()))
            worst_rel = max(worst_rel, rel)
            worst_rel_tail = max(worst_rel_tail, rel_tail)
            worst_linf = max(worst_linf, linf)
            worst_l2 = max(worst_l2, l2)
            mass_err = max(mass_err, abs(float(np.sum(geo['volumes'] * num)) - total) / total)
        rows.append(dict(n_shells=n, dx_um=geo['dx_um'], per_time=per_time,
                         max_rel=worst_rel,
                         max_rel_unrestricted_tail=worst_rel_tail,
                         linf_rel_peak=worst_linf, l2_rel=worst_l2,
                         mass_error=mass_err))
    orders = []
    for a, b in zip(rows[:-1], rows[1:]):
        orders.append(dict(between=(a['n_shells'], b['n_shells']),
                           order_linf=float(np.log(a['linf_rel_peak'] / b['linf_rel_peak'])
                                            / np.log(b['n_shells'] / a['n_shells'])),
                           order_l2=float(np.log(a['l2_rel'] / b['l2_rel'])
                                          / np.log(b['n_shells'] / a['n_shells']))))
    return dict(params=dict(R_um=R_um, sigma_um=sigma, D_ca=D, times_s=list(t_list)),
                rows=rows, observed_orders=orders)


def test_3_conservation():
    """Conservation with zero membrane flux, with and without reactions."""
    out = {}
    # (a) conservation of the buffered+ER content with reactions ON and a
    #     spatially non-uniform initial condition so that diffusion and ER
    #     exchange are both active.
    for dtx in (1e-3, 5e-4):
        res = run(mode='radial1d', n_shells=64, t_end=20.0, dt=dtx, sample_dt=0.5,
                  tear=0.0, parameters=dict(etaNCX=0.0), reactions=True,
                  blip=dict(sigma_um=0.8, amplitude_uM=1.0))
        q = res['content']
        out[f'reactions_on_dt_{dtx:g}'] = dict(
            initial=float(q[0]), final=float(q[-1]),
            absolute_drift=float(q[-1] - q[0]),
            relative_drift=float((q[-1] - q[0]) / abs(q[0])),
            max_abs_drift=float(np.max(np.abs(q - q[0])) / abs(q[0])),
            t_end=20.0, n_steps=res['metadata']['n_steps'],
            c_min=float(res['c'].min()), c_max=float(res['c'].max()),
            er_min=float(res['cER'].min()), er_max=float(res['cER'].max()))
    # (b) reactions OFF, buffers OFF (beta == 1): free cytosolic Ca mass must be
    #     conserved to round-off.  Be=0 and Bx=0 make 1/beta exactly 1.
    res = run(mode='radial1d', n_shells=64, t_end=20.0, dt=1e-3, sample_dt=0.5,
              parameters=dict(etaNCX=0.0, Be=0.0, Bx=0.0), reactions=False,
              blip=dict(sigma_um=0.8, amplitude_uM=1.0))
    vol = res['volumes']
    mass = res['c'] @ vol
    out['reactions_off_nobuffer'] = dict(
        initial=float(mass[0]), final=float(mass[-1]),
        relative_drift=float((mass[-1] - mass[0]) / mass[0]),
        max_abs_relative_drift=float(np.max(np.abs(mass - mass[0])) / mass[0]),
        beta_min=1.0, beta_max=1.0)
    # (c) reactions OFF with the real buffers: the conserved quantity is then the
    #     buffered content F(c), not the free-c mass (state this explicitly).
    res = run(mode='radial1d', n_shells=64, t_end=20.0, dt=1e-3, sample_dt=0.5,
              parameters=dict(etaNCX=0.0), reactions=False,
              blip=dict(sigma_um=0.8, amplitude_uM=1.0))
    vol = res['volumes']
    mass = res['c'] @ vol
    q = res['content']
    out['reactions_off_buffered'] = dict(
        free_c_relative_drift=float((mass[-1] - mass[0]) / mass[0]),
        buffered_content_relative_drift=float((q[-1] - q[0]) / abs(q[0])),
        max_abs_relative_drift=float(np.max(np.abs(q - q[0])) / abs(q[0])),
        note='with rapid buffering the conserved quantity is F(c)+cER/epsilon; '
             'free c alone is NOT conserved, by construction of the published beta(c).')
    # (d) accounting with the membrane flux ON: the change in the buffered+ER
    #     content must equal the accumulated membrane influx sum_i V_i pm_i.
    #     This validates the membrane-area normalisation end to end.
    for dtx in (1e-3, 1e-4):
        res = run(mode='radial1d', n_shells=64, t_end=5.0, dt=dtx, sample_dt=0.05,
                  tear=1.0)
        q = res['content']
        acc = res['pm_amount']
        scale = max(abs(q[-1] - q[0]), 1e-30)
        out[f'pm_accounting_dt_{dtx:g}'] = dict(
            delta_content=float(q[-1] - q[0]), accumulated_pm=float(acc[-1]),
            absolute_residual=float((q[-1] - q[0]) - acc[-1]),
            relative_residual=float(abs((q[-1] - q[0]) - acc[-1]) / scale),
            n_steps=res['metadata']['n_steps'],
            peak_mean_c=float(res['mean_c'].max()),
            note='residual is the first-order scheme error of the IMEX update, '
                 'not a physical loss.')
    return out


def test_4_lumped_limit(D_list=(1e3, 1e4, 1e5, 1e6), n_shells=32, t_end=5.0,
                        dt=1e-4):
    """Very large D: volume-averaged spatial c must converge to run_lumped."""
    lumped = run_lumped(t_end=t_end, dt=dt, sample_dt=0.01, tear=1.0)
    lc = lumped['mean_c']
    base = run(mode='radial1d', n_shells=1, t_end=t_end, dt=dt, sample_dt=0.01,
               tear=1.0, D_ca=1e-6, D_ip3=1e-6)
    same_scheme = run_lumped(t_end=t_end, dt=dt, sample_dt=0.01, tear=1.0,
                             integrator='imex')
    rows = []
    for D in D_list:
        res = run(mode='radial1d', n_shells=n_shells, t_end=t_end, dt=dt,
                  sample_dt=0.01, tear=1.0, D_ca=D, D_ip3=max(D, 283.0))
        mc = res['mean_c']
        rel = (mc - lc) / np.maximum(np.abs(lc), 1e-30)
        rows.append(dict(D_ca=D, max_abs=float(np.max(np.abs(mc - lc))),
                         max_rel=float(np.max(np.abs(rel))),
                         rms_rel=float(np.sqrt(np.mean(rel ** 2))),
                         peak_ratio=float(mc.max() / lc.max()),
                         gradient_frac=float(np.max(res['c'].max(axis=1)
                                                    - res['c'].min(axis=1))
                                             / np.max(res['mean_c']))))
    rel_ss = (same_scheme['mean_c'] - lc) / np.maximum(np.abs(lc), 1e-30)
    rel_n1 = (base['mean_c'] - lc) / np.maximum(np.abs(lc), 1e-30)
    # dt refinement of the difference at the largest D
    res_fine = run(mode='radial1d', n_shells=n_shells, t_end=t_end, dt=dt / 4,
                   sample_dt=0.01, tear=1.0, D_ca=D_list[-1], D_ip3=max(D_list[-1], 283.))
    rel_fine = (res_fine['mean_c'] - lc) / np.maximum(np.abs(lc), 1e-30)
    return dict(lumped_peak_c=float(lc.max()), rows=rows,
                same_scheme_lumped=dict(max_abs=float(np.max(np.abs(same_scheme['mean_c'] - lc))),
                                        max_rel=float(np.max(np.abs(rel_ss)))),
                n_shells_1_diffusion_only=dict(max_abs=float(np.max(np.abs(base['mean_c'] - lc))),
                                               max_rel=float(np.max(np.abs(rel_n1)))),
                finest_dt=dt / 4, finest_dt_max_rel=float(np.max(np.abs(rel_fine))),
                converged=bool(rows[-1]['max_abs'] < 0.5 * rows[0]['max_abs']),
                note='the residual at any D is the fixed-step (first-order) time '
                     'error; same_scheme_lumped and n_shells_1 isolate it. '
                     'D_ca is in um^2/s.')


def test_5_key_quantification(D_ca_list=(4.0, 13.0, 65.0, 130.0), n_shells=64,
                              t_end=5.0, dt=1e-4, patch_n_voxels=24,
                              patch_half_angle_deg=60.0, tear_list=(0.05, 0.2, 1.0)):
    """The honest headline: how much does space change the observable answer?"""
    uniform = {}
    for D in D_ca_list:
        cmp = compare_with_lumped(mode='radial1d', n_shells=n_shells, t_end=t_end,
                                  dt=dt, sample_dt=0.01, tear=1.0, D_ca=D,
                                  D_ip3=D_IP3_CYTOSOL)
        uniform[f'D_ca_{D:g}'] = dict(
            D_ca=D, max_abs_diff_uM=cmp['max_abs_diff'],
            max_abs_diff_pct_of_lumped_peak=100 * cmp['max_abs_diff_fraction_of_lumped_peak'],
            max_rel_diff=cmp['max_rel_diff'],
            time_of_max_rel_diff_s=cmp['time_of_max_rel_diff_s'],
            max_rel_diff_large_signal=cmp['max_rel_diff_large_signal'],
            time_of_max_rel_diff_large_signal_s=cmp['time_of_max_rel_diff_large_signal_s'],
            rms_rel_diff_large_signal=cmp['rms_rel_diff_large_signal'],
            rms_rel_diff=cmp['rms_rel_diff'],
            time_of_max_diff_s=cmp['time_of_max_diff_s'],
            at_max=dict(spatial=cmp['value_at_max']['spatial'],
                        lumped=cmp['value_at_max']['lumped']),
            peak_gradient_uM=cmp['peak_gradient_uM'],
            peak_gradient_fraction_of_mean=cmp['peak_gradient_fraction_of_mean'],
            time_of_peak_gradient_s=cmp['time_of_peak_gradient_s'],
            submembrane_excess_fraction=cmp['submembrane_excess_fraction'],
            time_of_max_submembrane_excess_s=cmp['time_of_max_submembrane_excess_s'],
            lumped_peak_c_uM=cmp['lumped_peak_c_uM'],
            fixed_step_scheme_error_max_abs=cmp['sublumped_time_scheme']['max_abs'])
    # stimulus-amplitude dependence (the answer is NOT amplitude independent)
    tear_rows = {}
    for tear in tear_list:
        cmp = compare_with_lumped(mode='radial1d', n_shells=n_shells, t_end=t_end,
                                  dt=dt, sample_dt=0.01, tear=tear,
                                  D_ca=D_CA_CYTOSOL, D_ip3=D_IP3_CYTOSOL)
        tear_rows[f'tear_{tear:g}'] = dict(
            tear=tear, lumped_peak_mean_c_uM=cmp['lumped_peak_c_uM'],
            spatial_peak_mean_c_uM=float(cmp['mean_c_spatial'].max()),
            max_abs_diff_uM=cmp['max_abs_diff'],
            max_abs_diff_pct_of_lumped_peak=100 * cmp['max_abs_diff_fraction_of_lumped_peak'],
            max_rel_diff=cmp['max_rel_diff'],
            max_rel_diff_large_signal=cmp['max_rel_diff_large_signal'],
            rms_rel_diff_large_signal=cmp['rms_rel_diff_large_signal'],
            time_of_max_diff_s=cmp['time_of_max_diff_s'],
            peak_gradient_fraction_of_mean=cmp['peak_gradient_fraction_of_mean'],
            time_of_peak_gradient_s=cmp['time_of_peak_gradient_s'],
            core_peak_c_uM=float(cmp['spatial']['c'][:, 0].max()),
            membrane_peak_c_uM=float(cmp['spatial']['c'][:, -1].max()))
    # shell-count dependence of the gradient measure (mesh sensitivity, reported)
    lum_ref = run_lumped(t_end=t_end, sample_dt=0.01, tear=1.0)
    lc_ref = lum_ref['mean_c']
    grad_n = {}
    for n in (32, 64, 128):
        r = run(mode='radial1d', n_shells=n, t_end=t_end, dt=dt, sample_dt=0.01,
                tear=1.0, D_ca=D_CA_CYTOSOL, D_ip3=D_IP3_CYTOSOL)
        grand = r['c'].max(axis=1) - r['c'].min(axis=1)
        grad_n[f'n_shells_{n}'] = dict(
            dx_um=r['geometry']['dx_um'],
            peak_gradient_fraction=float(np.max(grand / r['mean_c'])),
            peak_outer_shell_excess=float(np.max((r['c'][:, -1] - r['mean_c'])
                                                 / r['mean_c'])),
            membrane_peak_c_uM=float(r['c'][:, -1].max()),
            core_peak_c_uM=float(r['c'][:, 0].max()),
            max_rel_diff_vs_lumped=float(np.max(np.abs(r['mean_c'] - lc_ref)
                                                / np.maximum(np.abs(lc_ref), 1e-30))))
    # localised patch, grid3d only (radial1d cannot resolve it).  dt must satisfy
    # the explicit IP3 CFL of the coarse voxel grid.
    geo_p = build_geometry('grid3d', R_um=5.0, n_voxels=patch_n_voxels)
    dt_patch = min(dt, 0.5 * float(max_stable_dt(geo_p, D_CA_CYTOSOL, D_IP3_CYTOSOL)[1]['ip3']))
    patch = compare_with_lumped(mode='grid3d', n_voxels=patch_n_voxels, t_end=t_end,
                                dt=dt_patch, sample_dt=0.01, tear=1.0,
                                tear_patch=dict(half_angle_deg=patch_half_angle_deg),
                                D_ca=D_CA_CYTOSOL, D_ip3=D_IP3_CYTOSOL)
    pmeta = patch['spatial']['metadata']
    patch_summary = dict(
        dt=dt_patch,
        max_abs_diff_uM=patch['max_abs_diff'],
        max_abs_diff_pct_of_lumped_peak=100 * patch['max_abs_diff_fraction_of_lumped_peak'],
        max_rel_diff=patch['max_rel_diff'],
        time_of_max_rel_diff_s=patch['time_of_max_rel_diff_s'],
        max_rel_diff_large_signal=patch['max_rel_diff_large_signal'],
        rms_rel_diff_large_signal=patch['rms_rel_diff_large_signal'],
        rms_rel_diff=patch['rms_rel_diff'],
        time_of_max_diff_s=patch['time_of_max_diff_s'],
        peak_gradient_uM=patch['peak_gradient_uM'],
        peak_gradient_fraction_of_mean=patch['peak_gradient_fraction_of_mean'],
        time_of_peak_gradient_s=patch['time_of_peak_gradient_s'],
        lumped_peak_c_uM=patch['lumped_peak_c_uM'],
        lumped_stimulus_tear=patch['lumped_stimulus_tear'],
        fixed_step_scheme_error_max_abs=patch['sublumped_time_scheme']['max_abs'],
        membrane_area_fraction=pmeta['tear_patch']['area_fraction'],
        requested_area_fraction=pmeta['tear_patch'].get('requested_area_fraction'),
        staircase_area_ratio=pmeta.get('staircase_area_ratio'),
        guards=pmeta['guards'], integrator=pmeta['integrator_resolved'],
        cfl=pmeta['cfl_by_species'],
        n_active_voxels=pmeta['n_cells'],
        patch_voxel_peak_c_uM=float(patch['spatial']['c'].max()),
        opposite_side_peak_c_uM=None)
    # contrast: how much of the cell the patch actually reaches
    c_sp = patch['spatial']['c']
    geo_sp = patch['spatial']['geometry']
    ctr = geo_sp['centres']
    ang = (ctr @ np.array([1.0, 0.0, 0.0])) / np.maximum(np.linalg.norm(ctr, axis=1), 1e-30)
    near = ang >= np.cos(np.radians(patch_half_angle_deg))
    far = ang <= -np.cos(np.radians(patch_half_angle_deg))
    k = int(np.argmax(c_sp.max(axis=1)))
    patch_summary['t_of_patch_peak_s'] = float(patch['times'][k])
    patch_summary['patched_side_mean_c_uM'] = float(c_sp[k][near].mean())
    patch_summary['opposite_side_mean_c_uM'] = float(c_sp[k][far].mean())
    patch_summary['opposite_side_peak_c_uM'] = float(c_sp[:, far].max())
    return dict(uniform_surface=uniform, tear_amplitude_dependence=tear_rows,
                shell_count_dependence=grad_n,
                localised_patch_grid3d=patch_summary,
                patch_lumped_amplitude_note=(
                    'for the patch case the lumped model is driven with the same MEAN '
                    'surface damage (tear x membrane area fraction), so the comparison '
                    'isolates the spatial concentration of the same total stimulus.'),
                rms_definition='RMS over time of (mean_c_spatial-mean_c_lumped)/mean_c_lumped',
                D_ca_sweep_note=('D_ca = 4 to 130 um^2/s brackets the cited 13 (resting '
                                 'Ca) and 65 (1 uM Ca) values of Allbritton 1992 and '
                                 'covers a >10x range; D_ca is the dominant unknown.'))


def test_6_runtime_scaling(t_end_radial=1.0, t_end_grid=0.2, dt=1e-4,
                           R_um=5.0, cells=576):
    """Measured wall times, plus an explicitly EXTRAPOLATED 576-cell estimate."""
    rows = {}
    p_kdeg = PARAMETERS['kdeg']
    for n in (32, 64, 128):
        r = run(mode='radial1d', R_um=R_um, n_shells=n, t_end=t_end_radial, dt=dt,
                sample_dt=0.1, tear=1.0)
        md = r['metadata']
        rows[f'radial1d_{n}'] = dict(
            wall_s=md['total_seconds'], integration_s=md['integration_seconds'],
            n_steps=md['n_steps'], t_end=t_end_radial, dt=dt,
            integrator=md['integrator_resolved'],
            per_step_us=1e6 * md['integration_seconds'] / md['n_steps'],
            state_bytes=int(4 * n * 8),
            memory_per_cell_bytes=int(4 * n * 8 + 3 * n * 8),
            cfl_max=md['cfl_max'], guards=md['guards'])
    # grid3d: measure BOTH the fully explicit scheme at its CFL-limited dt and the
    # hybrid scheme (IP3 implicit) at the caller's dt, because on one CPU core the
    # 3-D sparse triangular solve is ~40x more expensive per step than an explicit
    # step, so "implicit is preferred" is not automatically true here.
    for n in (24, 32):
        geo = build_geometry('grid3d', R_um=R_um, n_voxels=n)
        stable, limits = max_stable_dt(geo, D_CA_CYTOSOL, D_IP3_CYTOSOL)
        dt_exp = 0.8 * float(stable)
        r = run(mode='grid3d', R_um=R_um, n_voxels=n, t_end=t_end_grid, dt=dt_exp,
                sample_dt=0.05, tear=1.0, integrator='explicit')
        md = r['metadata']
        Lc = diffusion_operator(geo, D_CA_CYTOSOL)
        t_lu = _time.perf_counter()
        lu = splu((identity(geo['n'], format='csr')
                   - dt * diags(np.full(geo['n'], 0.1)) @ Lc).tocsc())
        lu_s = _time.perf_counter() - t_lu
        rows[f'grid3d_{n}_explicit'] = dict(
            wall_s=md['total_seconds'], integration_s=md['integration_seconds'],
            n_steps=md['n_steps'], t_end=t_end_grid, dt=dt_exp, integrator='explicit',
            per_step_us=1e6 * md['integration_seconds'] / md['n_steps'],
            n_active=md['n_cells'], cfl=md['cfl_by_species'],
            explicit_stable_dt=float(stable), explicit_stable_dt_by_species=limits,
            guards=md['guards'], staircase_area_ratio=md['staircase_area_ratio'],
            state_bytes=int(4 * geo['n'] * 8),
            operator_bytes=int((Lc.nnz * 12) + (geo['n'] * 8)),
            imex_lu_seconds=float(lu_s), imex_lu_bytes=int((lu.L.nnz + lu.U.nnz) * 12),
            memory_per_cell_bytes=int(4 * geo['n'] * 8 + Lc.nnz * 12))
        r2 = run(mode='grid3d', R_um=R_um, n_voxels=n, t_end=0.02, dt=dt,
                 sample_dt=0.02, tear=1.0, integrator='hybrid')
        md2 = r2['metadata']
        Lp_m = diffusion_operator(geo, D_IP3_CYTOSOL)
        t_lu2 = _time.perf_counter()
        lu2 = splu((identity(geo['n'], format='csr') * (1.0 + dt * p_kdeg)
                    - dt * Lp_m).tocsc())
        ip3_lu_s = _time.perf_counter() - t_lu2
        rows[f'grid3d_{n}_hybrid'] = dict(
            t_end=0.02, dt=dt, integrator='hybrid', n_steps=md2['n_steps'],
            wall_s=md2['total_seconds'], integration_s=md2['integration_seconds'],
            per_step_us=1e6 * md2['integration_seconds'] / md2['n_steps'],
            per_step_us_excluding_factorisation=1e6 * max(
                md2['integration_seconds'] - ip3_lu_s, 0.0) / md2['n_steps'],
            n_active=md2['n_cells'],
            state_bytes=int(4 * geo['n'] * 8),
            memory_per_cell_bytes=int(4 * geo['n'] * 8 + Lc.nnz * 12),
            ip3_splu_seconds=float(ip3_lu_s),
            ip3_splu_bytes=int((lu2.L.nnz + lu2.U.nnz) * 12),
            cfl=md2['cfl_by_species'], guards=md2['guards'],
            note='the IP3 diffusion is the reason the hybrid path exists: with the '
                 'published D_IP3 = 283 um^2/s the explicit IP3 CFL forces dt < '
                 '1.0e-4 s at 24^3 and < 5.8e-5 s at 32^3. per_step_us is amortised '
                 'over only 200 steps and therefore includes the one-off sparse '
                 'factorisation of the constant IP3 matrix.')
    # extrapolated 576-cell cost: use the CHEAPEST viable scheme per geometry
    est = {}
    for key, row in rows.items():
        per_step_s = row.get('per_step_us_excluding_factorisation',
                             row['per_step_us']) * 1e-6
        steps_5s = 5.0 / row['dt']
        mem = row['memory_per_cell_bytes']
        est[key] = dict(
            per_cell_5s_s=per_step_s * steps_5s,
            steps_for_5s=float(steps_5s),
            cells_576_seconds=per_step_s * steps_5s * cells,
            cells_576_hours=per_step_s * steps_5s * cells / 3600.0,
            memory_per_cell_MB=mem / 1e6, memory_576_MB=mem * cells / 1e6)
    feasible = {
        'single_16GB_GPU_or_96GB_accelerator': (
            'Runtime, not memory, is the binding constraint. radial1d (any tested '
            'shell count) costs ~3.5 s per cell for a 5 s stimulus on one core, so '
            '576 cells is ~0.5 h single-core and ~4 MB of state in total: FEASIBLE. '
            'grid3d is the expensive option: with the fully explicit scheme at its '
            'CFL-limited dt, 24^3 costs ~21 s and 32^3 ~130 s per cell for 5 s, i.e. '
            '3.4 h and 21 h for 576 cells on ONE core. Tissue-scale voxel resolution '
            'is therefore NOT feasible single-core, and becomes plausible only by '
            'batching the explicit matvec across many cores or a GPU (the scheme is '
            'embarrassingly parallel in the matvec but the per-step Python/numpy '
            'overhead does not vectorise away on CPU). Memory is not the problem: '
            '0.8 MB per cell at 24^3 and 1.9 MB at 32^3 gives 0.46 GB and 1.1 GB for '
            '576 cells, within 16 GB. The item that COULD exhaust a 16 GB device is '
            'per-cell imex LU factors of the beta-dependent Ca matrix (41 MB at 24^3, '
            '174 MB at 32^3 -> 24-100 GB for 576 cells), which is a second reason the '
            'explicit/hybrid paths exist.'),
        'was_this_run': False,
        'note': 'the 576-cell figures are EXTRAPOLATIONS from the measured per-step '
                'costs above (steps for 5 s x 576 cells), assuming perfect '
                'parallelism and no cell-cell coupling. They were NOT run.',
    }
    return dict(rows=rows, extrapolated_576_cells=est, feasibility=feasible)


def test_7_determinism_and_finiteness():
    """Bit-identical repeat runs; no non-finite or negative values anywhere."""
    kw = dict(mode='radial1d', n_shells=48, t_end=1.0, dt=1e-4, sample_dt=0.05,
              tear=1.0)
    a = run(**kw)
    b = run(**kw)
    identical = all(np.array_equal(a[k], b[k]) for k in ('c', 'cER', 'ip3', 'h'))
    minima = {k: float(min(a[k].min(), b[k].min())) for k in ('c', 'cER', 'ip3', 'h')}
    finite = all(np.all(np.isfinite(a[k])) for k in ('c', 'cER', 'ip3', 'h'))
    g = run(mode='grid3d', n_voxels=16, t_end=0.2, dt=1e-4, sample_dt=0.05,
            tear=1.0, integrator='explicit')
    gmin = {k: float(g[k].min()) for k in ('c', 'cER', 'ip3', 'h')}
    return dict(bit_identical=bool(identical), all_finite=bool(finite),
                minima_radial1d=minima, maxima_radial1d={k: float(a[k].max())
                                                         for k in ('c', 'cER', 'ip3', 'h')},
                minima_grid3d=gmin, grid3d_guards=g['metadata']['guards'],
                h_max=float(a['h'].max()))


def self_test():
    """Fast internal consistency checks (no files, no plots).

    Checks: geometry volumes/areas; the conservative operator's column-sum
    property Sum_i V_i (L c)_i = 0; dF/dc == 1/beta; this module's flux vector
    against model.py's own _rhs_factory on a one-cell domain; uniform
    preservation; the parameter source identity; and a short spatial run.
    """
    geo = build_radial1d(5.0, 32)
    assert np.isclose(geo['total_volume'], 4 / 3 * np.pi * 5.0 ** 3, rtol=1e-12)
    assert np.isclose(geo['membrane_area'], 4 * np.pi * 25.0, rtol=1e-12)
    gg = build_grid3d(5.0, 12)
    assert 0.5 < gg['total_volume'] / (4 / 3 * np.pi * 125.0) < 1.15
    rng = np.random.default_rng(0)
    for g in (geo, gg):
        L = diffusion_operator(g, 13.0)
        x = rng.normal(size=g['n'])
        assert abs(g['volumes'] @ (L @ x)) < 1e-9 * abs(g['volumes'] @ np.abs(L @ x)) + 1e-12
        assert np.max(np.abs(L @ np.ones(g['n']))) < 1e-10
    p = PARAMETERS.copy()
    assert PARAMETERS is not model.PARAMETERS
    assert all(PARAMETERS[k] == model.PARAMETERS[k] for k in model.PARAMETERS)
    c = np.array([0.05, 0.1, 0.5, 1.0, 3.0])
    # complex-step derivative: exact to machine precision for this rational F
    fd = np.imag(buffered_ca(c + 1j * 1e-20, p)) / 1e-20
    assert np.max(np.abs(fd - 1 / beta_of(c, p)) / (1 / beta_of(c, p))) < 1e-12
    # flux identity against the published RHS, one cell, damage = 0.7
    n = 1
    c, er, ip3, hh = .37, 143.2, .21, .43
    t = 0.31
    damage = 0.7
    f = flux_terms(p, np.array([c]), np.array([er]), np.array([ip3]), np.array([hh]),
                   t, p['Bx'], p['alpha'])
    mine = np.array([f['beta'][0] * (f['j_er'][0] + f['pm_base'][0]
                                     + f['pm_tear_unit'][0] * damage),
                     -p['epsilon'] * f['j_er'][0],
                     f['rh'][0] - p['kdeg'] * ip3,
                     f['dh'][0]])
    rhs = _lumped_rhs_factory(p, p['alpha'], p['Bx'], damage)
    theirs = rhs(t, np.array([c, er, ip3, hh]))
    flux_identity = float(np.max(np.abs(mine - theirs)))
    assert flux_identity < 1e-12, flux_identity
    # uniform preservation and a short wound run with the published D values
    res = run(n_shells=16, t_end=0.05, dt=1e-4, sample_dt=0.05,
              parameters=dict(etaNCX=0.0), reactions=False,
              initial=dict(c=1.0, er=200., ip3=0.01, h=0.67))
    uniform_dev = float(np.max(np.abs(res['c'] - 1.0)))
    assert uniform_dev < 1e-12, uniform_dev
    wounded = run(n_shells=16, t_end=0.2, dt=1e-4, sample_dt=0.1, tear=1.0)
    assert np.all(np.isfinite(wounded['c'])) and wounded['c'].min() >= 0
    cmp_ = compare_with_lumped(mode='radial1d', n_shells=16, t_end=0.1, dt=1e-4,
                               sample_dt=0.05, tear=1.0)
    return dict(flux_identity_max_abs_error=flux_identity,
                buffered_ca_derivative_ok=True,
                uniform_preservation_max_dev=uniform_dev,
                wounded_peak_mean_c_uM=float(wounded['mean_c'].max()),
                quick_compare=dict(max_abs_diff=cmp_['max_abs_diff'],
                                   max_rel_diff=cmp_['max_rel_diff'],
                                   gradient_fraction=cmp_['peak_gradient_fraction_of_mean']),
                D_ca=D_CA_CYTOSOL, D_ip3=D_IP3_CYTOSOL,
                parameters_identical_to_model_py=True,
                grid3d_staircase_area_ratio=gg.get('staircase_area_ratio'))


def full_test_suite(quick=False):
    """Run all seven required numerical tests and return every number."""
    out = {}
    steps = [
        ('test_1_uniform_pure_diffusion', test_1_uniform_pure_diffusion),
        ('test_2_analytic_gaussian_diffusion', test_2_analytic_diffusion),
        ('test_3_mass_conservation', test_3_conservation),
        ('test_4_lumped_large_D_limit', lambda: test_4_lumped_limit(
            D_list=(1e3, 1e4, 1e5))),
        ('test_5_key_quantification', lambda: test_5_key_quantification(
            patch_n_voxels=(12 if quick else 24))),
        ('test_6_runtime_scaling', test_6_runtime_scaling),
        ('test_7_determinism_finiteness', test_7_determinism_and_finiteness),
    ]
    for name, fn in steps:
        t0 = _time.perf_counter()
        print(f'  running {name} ...', flush=True)
        out[name] = fn()
        out[name + '__seconds'] = _time.perf_counter() - t0
        print(f'    done in {out[name + "__seconds"]:.1f} s', flush=True)
    return out


if __name__ == '__main__':
    import pprint
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print('cellspace: synthetic single-cell spatial model -- NOT measurement')
    print('=' * 78)
    print('self_test:')
    pprint.pp(self_test(), width=120)
    print('=' * 78)
    print('full test suite:')
    suite = full_test_suite()
    for k, v in suite.items():
        print('-' * 78)
        print(k)
        pprint.pp(v, width=120, sort_dicts=False)
