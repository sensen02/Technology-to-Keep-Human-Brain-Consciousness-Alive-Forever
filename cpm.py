"""cpm.py -- Cellular Potts Model engine (2D/3D), numba accelerated.

PURPOSE
-------
This is the self-organisation engine for the project.  It provides the
*lattice* mechanics needed to let cell populations organise themselves
(vessels, wounds, stratified epidermis) out of purely local rules.  It is
deliberately separate from the project's vertex-mechanics code, because the
vertex model was measured to be unable to do T1 transitions (see
`MEASURED_ANCHORS.md` and `outputs/vertex_division_probe.json`).

HAMILTONIAN (exact, as implemented)
-----------------------------------
    H = H_contact + H_volume + H_length

    H_contact = (1/2) * sum_i sum_{j in N(i)} J(tau_i, tau_j)
                (the 1/2 is the standard convention that makes the per-move
                 delta equal to the sum over the target site's neighbours:
                 delta H_contact = sum_{j in N(t)} [ J(tau_a, tau_j)
                                                    - J(tau_b, tau_j) ]
                 for a copy of cell a into site t currently held by b.
                 J must be symmetric.)
    H_volume  = sum_cells lambda_V(c) * (V(c) - Vt(c))^2
    H_length  = sum_cells lambda_L(c) * (L(c) - L0(c))^2
                L(c) = 4*sqrt(lmax), lmax = largest eigenvalue of the
                position covariance matrix of the cell's lattice sites.
                (For a disk of radius R, lmax = R^2/4, so L = 2R = diameter.)

CHEMOTAXIS is added as the standard local-gradient term used by
CompuCell3D / the Merks et al. (2008) vasculogenesis model:

    dH_chem = -lambda_chem * ( c(x_target) - c(x_source) )

This term is NOT the gradient of a potential: it is path dependent, so it is
deliberately EXCLUDED from the exact-Hamiltonian self-test and this is stated
wherever the test result is reported.  With `contact_inhibited=True` the term
is applied only when the target site is medium, which is the contact
inhibition of chemotaxis (a cell cannot pull itself towards another cell).

BOUNDARIES
----------
Closed (no-flux / wall) boundaries: out-of-range neighbours are skipped and
no lattice site is ever copied out of the box.  This is chosen over periodic
boundaries because periodic wrapping corrupts the position covariance used by
the length constraint for cells that straddle the box edge.

UNITS
-----
The lattice spacing is 1.  A "volume" is a number of lattice sites (2D: an
area in sites).  Time is in Monte Carlo Steps (MCS); 1 MCS = N lattice-site
copy attempts, where N = Lx*Ly*Lz.  Nothing in this file carries physical
units; the mapping to um/minutes is done by the caller and must be declared.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from numba import njit

MEDIUM = 0

# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True, inline="always")
def _len2(n, Sx, Sy, Sxx, Syy, Sxy):
    """Long axis length from 2D position moments.  L = 4*sqrt(lmax)."""
    if n <= 1:
        return 0.0
    inv = 1.0 / n
    cx = Sx * inv
    cy = Sy * inv
    exx = Sxx * inv - cx * cx
    eyy = Syy * inv - cy * cy
    exy = Sxy * inv - cx * cy
    d = 0.5 * (exx - eyy)
    lmax = 0.5 * (exx + eyy) + np.sqrt(d * d + exy * exy)
    if lmax < 0.0:
        lmax = 0.0
    return 4.0 * np.sqrt(lmax)


@njit(cache=True, inline="always")
def _len3(n, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz):
    """Long axis length from 3D position moments, power iteration on 3x3 cov."""
    if n <= 1:
        return 0.0
    inv = 1.0 / n
    cx = Sx * inv
    cy = Sy * inv
    cz = Sz * inv
    axx = Sxx * inv - cx * cx
    ayy = Syy * inv - cy * cy
    azz = Szz * inv - cz * cz
    axy = Sxy * inv - cx * cy
    axz = Sxz * inv - cx * cz
    ayz = Syz * inv - cy * cz
    vx = 1.0
    vy = 0.5
    vz = 0.25
    nrm = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= nrm
    vy /= nrm
    vz /= nrm
    for _ in range(24):
        wx = axx * vx + axy * vy + axz * vz
        wy = axy * vx + ayy * vy + ayz * vz
        wz = axz * vx + ayz * vy + azz * vz
        nrm = np.sqrt(wx * wx + wy * wy + wz * wz)
        if nrm <= 1e-300:
            return 0.0
        vx = wx / nrm
        vy = wy / nrm
        vz = wz / nrm
    lmax = (
        axx * vx * vx
        + ayy * vy * vy
        + azz * vz * vz
        + 2.0 * (axy * vx * vy + axz * vx * vz + ayz * vy * vz)
    )
    if lmax < 0.0:
        lmax = 0.0
    return 4.0 * np.sqrt(lmax)


@njit(cache=True, inline="always")
def _len_of(two_d, n, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz):
    if two_d:
        return _len2(n, Sx, Sy, Sxx, Syy, Sxy)
    return _len3(n, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz)


@njit(cache=True)
def _attempt(
    lat, Lx, Ly, Lz, off, nnoff, J, nt, ctype,
    V, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz,
    Vt, lV, L0, lL, lamL_on,
    chem, lam_chem, chem_on, contact_inhib, chem_cc_ratio,
    frozen, T, x, y, z, k, do_apply, two_d, chem_hits, site_bias,
):
    """One Metropolis copy attempt.  Returns (dH, accepted).

    The chemotaxis contribution is included when `chem_on` is True; pass
    lam_chem=0.0 and chem_on=False to obtain the mechanical part alone, which
    is what the exact-Hamiltonian verification compares against.
    """
    idx = (z * Ly + y) * Lx + x
    a = lat[idx]
    if a != 0 and frozen[a]:
        return 0.0, False
    tx = x + off[k, 0]
    ty = y + off[k, 1]
    tz = z + off[k, 2]
    if tx < 0 or tx >= Lx or ty < 0 or ty >= Ly or tz < 0 or tz >= Lz:
        return 0.0, False
    nidx = (tz * Ly + ty) * Lx + tx
    b = lat[nidx]
    if b == a:
        return 0.0, False
    if b != 0 and frozen[b]:
        return 0.0, False
    va = V[a]
    vb = V[b]
    # never let a copy attempt annihilate a cell by taking its last site;
    # a == 0 (medium) as the source is allowed and is what makes cells shrink
    if b > 0 and vb <= 1:
        return 0.0, False

    ta = ctype[a]
    tb = ctype[b]

    # ---- contact energy (site t changes owner from b to a) ----
    dH = 0.0
    for k2 in range(nnoff):
        mx = tx + off[k2, 0]
        my = ty + off[k2, 1]
        mz = tz + off[k2, 2]
        if mx < 0 or mx >= Lx or my < 0 or my >= Ly or mz < 0 or mz >= Lz:
            continue
        tm = ctype[lat[(mz * Ly + my) * Lx + mx]]
        dH += J[ta * nt + tm] - J[tb * nt + tm]

    # ---- volume ----
    dH += lV[a] * ((va + 1.0 - Vt[a]) ** 2 - (va - Vt[a]) ** 2)
    if b > 0:
        dH += lV[b] * ((vb - 1.0 - Vt[b]) ** 2 - (vb - Vt[b]) ** 2)

    # ---- length ----
    # NOTE: cell `a` GAINS the target site (tx,ty,tz); cell `b` LOSES it.
    if lamL_on:
        if lL[a] > 0.0:
            lold = _len_of(two_d, va, Sx[a], Sy[a], Sz[a], Sxx[a], Syy[a], Szz[a],
                           Sxy[a], Sxz[a], Syz[a])
            lnew = _len_of(two_d, va + 1, Sx[a] + tx, Sy[a] + ty, Sz[a] + tz,
                           Sxx[a] + tx * tx, Syy[a] + ty * ty, Szz[a] + tz * tz,
                           Sxy[a] + tx * ty, Sxz[a] + tx * tz, Syz[a] + ty * tz)
            dH += lL[a] * ((lnew - L0[a]) ** 2 - (lold - L0[a]) ** 2)
        if b > 0 and lL[b] > 0.0:
            lold = _len_of(two_d, vb, Sx[b], Sy[b], Sz[b], Sxx[b], Syy[b], Szz[b],
                           Sxy[b], Sxz[b], Syz[b])
            lnew = _len_of(two_d, vb - 1, Sx[b] - tx, Sy[b] - ty, Sz[b] - tz,
                           Sxx[b] - tx * tx, Syy[b] - ty * ty, Szz[b] - tz * tz,
                           Sxy[b] - tx * ty, Sxz[b] - tx * tz, Syz[b] - ty * tz)
            dH += lL[b] * ((lnew - L0[b]) ** 2 - (lold - L0[b]) ** 2)

    # ---- per-site bias (an exact potential) ----
    # H_bias = sum_i site_bias[i] * [lat_i > 0].  Only the target site's
    # occupancy changes, so dH_bias = site_bias[t] * ([a>0] - [b>0]).
    # Used for the wound protrusion drive and the wound-margin line tension.
    if site_bias[nidx] != 0.0:
        if a > 0 and b == 0:
            dH += site_bias[nidx]
        elif a == 0 and b > 0:
            dH -= site_bias[nidx]

    # ---- chemotaxis (path dependent, not a potential) ----
    # `chem_cc_ratio` is the chemotactic sensitivity at a CELL-CELL interface
    # relative to a cell-ECM interface (chi(c,c)/chi(c,M) in Merks et al. 2008).
    # Ratio 0 = full contact inhibition of chemotaxis (the pseudopod is
    # suppressed wherever the membrane touches another cell); ratio 1 = no
    # contact inhibition.  The paper localises the sprouting/non-sprouting
    # phase transition at a ratio of about 0.5, so this must be a continuous
    # parameter, not a boolean.
    if chem_on:
        if b == 0:
            dH -= lam_chem[a] * (chem[nidx] - chem[idx])
        elif chem_cc_ratio > 0.0:
            dH -= lam_chem[a] * chem_cc_ratio * (chem[nidx] - chem[idx])

    if dH <= 0.0 or np.random.random() < np.exp(-dH / T):
        if do_apply:
            # `chem_hits[c]` counts the accepted copy attempts in which the
            # chemotaxis term actually contributed for cell c.  It is the
            # direct diagnostic for which cells are chemotaxing.
            if chem_on and a > 0:
                if b == 0 or chem_cc_ratio > 0.0:
                    chem_hits[a] += 1
            lat[nidx] = a
            if a > 0:
                V[a] = va + 1
                Sx[a] += tx
                Sy[a] += ty
                Sz[a] += tz
                Sxx[a] += tx * tx
                Syy[a] += ty * ty
                Szz[a] += tz * tz
                Sxy[a] += tx * ty
                Sxz[a] += tx * tz
                Syz[a] += ty * tz
            if b > 0:
                V[b] = vb - 1
                Sx[b] -= tx
                Sy[b] -= ty
                Sz[b] -= tz
                Sxx[b] -= tx * tx
                Syy[b] -= ty * ty
                Szz[b] -= tz * tz
                Sxy[b] -= tx * ty
                Sxz[b] -= tx * tz
                Syz[b] -= ty * tz
        return dH, True
    return dH, False


@njit(cache=True)
def _force_apply(
    lat, Lx, Ly, Lz, off,
    V, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz,
    x, y, z, k,
):
    """Unconditional copy of the source site into the target site.

    Used only by the verification harness (to compare the kernel's reported
    dH with the brute-force change in the total Hamiltonian).
    """
    idx = (z * Ly + y) * Lx + x
    a = lat[idx]
    tx = x + off[k, 0]
    ty = y + off[k, 1]
    tz = z + off[k, 2]
    if tx < 0 or tx >= Lx or ty < 0 or ty >= Ly or tz < 0 or tz >= Lz:
        return False
    nidx = (tz * Ly + ty) * Lx + tx
    b = lat[nidx]
    if b == a:
        return False
    lat[nidx] = a
    if a > 0:
        V[a] += 1
        Sx[a] += tx
        Sy[a] += ty
        Sz[a] += tz
        Sxx[a] += tx * tx
        Syy[a] += ty * ty
        Szz[a] += tz * tz
        Sxy[a] += tx * ty
        Sxz[a] += tx * tz
        Syz[a] += ty * tz
    if b > 0:
        V[b] -= 1
        Sx[b] -= tx
        Sy[b] -= ty
        Sz[b] -= tz
        Sxx[b] -= tx * tx
        Syy[b] -= ty * ty
        Szz[b] -= tz * tz
        Sxy[b] -= tx * ty
        Sxz[b] -= tx * tz
        Syz[b] -= ty * tz
    return True


@njit(cache=True)
def _mcs_kernel(
    lat, Lx, Ly, Lz, off, nnoff, J, nt, ctype,
    V, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz,
    Vt, lV, L0, lL, lamL_on,
    chem, lam_chem, chem_on, contact_inhib, chem_cc_ratio,
    frozen, T, nmcs, two_d, chem_hits, site_bias,
):
    """Run `nmcs` Monte Carlo Steps.  Returns (accepted, attempts)."""
    n_sites = Lx * Ly * Lz
    nacc = 0
    natt = 0
    for _ in range(nmcs):
        # standard CPM clock: 1 MCS = n_sites copy attempts drawn uniformly
        # over ALL lattice sites.  Medium (id 0) is a legitimate source, which
        # is what lets cells shrink and lets the system reach an equilibrium
        # area instead of only ever growing.
        for _ in range(n_sites):
            x = np.random.randint(0, Lx)
            y = np.random.randint(0, Ly)
            z = np.random.randint(0, Lz)
            k = np.random.randint(0, nnoff)
            natt += 1
            _, acc = _attempt(
                lat, Lx, Ly, Lz, off, nnoff, J, nt, ctype,
                V, Sx, Sy, Sz, Sxx, Syy, Szz, Sxy, Sxz, Syz,
                Vt, lV, L0, lL, lamL_on,
                chem, lam_chem, chem_on, contact_inhib, chem_cc_ratio,
                frozen, T, x, y, z, k, True, two_d, chem_hits, site_bias,
            )
            if acc:
                nacc += 1
    return nacc, natt


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def neighbour_offsets(dim):
    if dim == 2:
        triples = [
            (dx, dy, 0)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ]
    elif dim == 3:
        triples = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        raise ValueError("dim must be 2 or 3")
    return np.array(triples, dtype=np.int64)


class Field:
    """A diffusing, decaying, secreted chemical field on the CPM lattice.

    Equation integrated (explicit forward Euler):
        dc/dt = D * laplacian(c) - k*c + sum_types secretion[type] * occupied
                - sum_types uptake[type] * occupied * c
    with no-flux (Neumann) boundaries.  `lattice_spacing_um` converts the
    lattice unit to um for D (um^2 per time unit) and for concentrations.
    """

    def __init__(self, name, shape, D=0.0, decay=0.0, secretion=None,
                 uptake=None, dt=0.05, initial=0.0, source_mask=None,
                 source_value=None, saturation=None, lattice_spacing_um=1.0):
        self.name = name
        self.value = np.full(shape, float(initial), dtype=np.float64)
        self.D = float(D)
        self.decay = float(decay)
        self.dt = float(dt)
        self.secretion = dict(secretion or {})
        self.uptake = dict(uptake or {})
        self.saturation = saturation
        self.source_mask = source_mask
        self.source_value = source_value
        self.lattice_spacing_um = float(lattice_spacing_um)
        self.total_secreted = 0.0
        self.total_decayed = 0.0
        if source_mask is not None and source_value is not None:
            self.value[source_mask] = source_value

    def laplacian(self):
        v = self.value
        h2 = self.lattice_spacing_um ** 2
        if v.ndim == 2:
            p = np.pad(v, 1, mode="edge")
            return (
                p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4.0 * v
            ) / h2
        p = np.pad(v, 1, mode="edge")
        return (
            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
            + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] - 6.0 * v
        ) / h2

    def max_stable_dt(self):
        if self.D <= 0.0:
            return np.inf
        h2 = self.lattice_spacing_um ** 2
        ndim = self.value.ndim
        return 0.25 * h2 / (self.D * ndim)

    def update(self, site_type, alive):
        """Integrate one time step `dt` using the current site->type map."""
        nsub = 1
        if self.D > 0.0:
            nsub = int(np.ceil(self.dt / self.max_stable_dt()))
            nsub = max(1, min(nsub, 2000))
        sub_dt = self.dt / nsub
        for _ in range(nsub):
            if self.D > 0.0:
                self.value += sub_dt * self.D * self.laplacian()
            if self.decay > 0.0:
                dec = sub_dt * self.decay * self.value
                self.total_decayed += float(dec.sum())
                self.value -= dec
            for t, rate in self.secretion.items():
                if rate == 0.0:
                    continue
                m = site_type == t
                if not m.any():
                    continue
                add = rate * sub_dt * m
                self.value += add
                self.total_secreted += float(add.sum())
            for t, rate in self.uptake.items():
                if rate == 0.0:
                    continue
                m = site_type == t
                if not m.any():
                    continue
                cons = rate * sub_dt * self.value * m
                self.value -= cons
            if self.saturation is not None:
                np.minimum(self.value, self.saturation, out=self.value)
            if self.source_mask is not None and self.source_value is not None:
                self.value[self.source_mask] = self.source_value
        return self

    def mass(self):
        return float(self.value.sum())


class CPM:
    """Cellular Potts Model on a closed-boundary lattice."""

    def __init__(self, Lx, Ly, Lz=1, n_types=2, temperature=1.0, seed=0,
                 max_cells=40000, lattice_spacing_um=1.0):
        if Lz != 1 and Lz < 1:
            raise ValueError("Lz must be >= 1")
        self.Lx, self.Ly, self.Lz = int(Lx), int(Ly), int(Lz)
        self.dim = 2 if self.Lz == 1 else 3
        self.N = self.Lx * self.Ly * self.Lz
        self.n_types = int(n_types)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_cells = int(max_cells)
        self.lattice_spacing_um = float(lattice_spacing_um)
        self.two_d = self.dim == 2
        self.off = neighbour_offsets(self.dim)
        self.nnoff = self.off.shape[0]

        self.lat = np.zeros(self.N, dtype=np.int64)
        m = self.max_cells + 1
        self.ctype = np.zeros(m, dtype=np.int64)
        self.V = np.zeros(m, dtype=np.float64)
        self.Sx = np.zeros(m, dtype=np.float64)
        self.Sy = np.zeros(m, dtype=np.float64)
        self.Sz = np.zeros(m, dtype=np.float64)
        self.Sxx = np.zeros(m, dtype=np.float64)
        self.Syy = np.zeros(m, dtype=np.float64)
        self.Szz = np.zeros(m, dtype=np.float64)
        self.Sxy = np.zeros(m, dtype=np.float64)
        self.Sxz = np.zeros(m, dtype=np.float64)
        self.Syz = np.zeros(m, dtype=np.float64)
        self.Vt = np.zeros(m, dtype=np.float64)
        self.lV = np.zeros(m, dtype=np.float64)
        self.L0 = np.zeros(m, dtype=np.float64)
        self.lL = np.zeros(m, dtype=np.float64)
        self.frozen = np.zeros(m, dtype=np.uint8)
        self.chem_hits = np.zeros(m, dtype=np.int64)
        # per-cell chemotaxis strength (index 0 unused)
        self.lam_chem_cell = np.zeros(m, dtype=np.float64)
        self._lam_zero = np.zeros(m, dtype=np.float64)
        # per-site potential coefficient (see H_bias in the module docstring)
        self.site_bias = np.zeros(self.N, dtype=np.float64)
        self.J = np.zeros((self.n_types, self.n_types), dtype=np.float64)

        self.alive = set()
        self.next_id = 1
        self.free_ids = []
        self.fields = {}
        self._elong = {}
        self.mcs = 0
        self.n_accepted = 0
        self.n_attempts = 0
        self.time_unit = "MCS"
        # default (disabled) chemotaxis field so the kernels always receive a
        # typed float64 array, never None
        shape = (self.Ly, self.Lx) if self.two_d else (self.Lz, self.Ly, self.Lx)
        self._chem = np.zeros(shape, dtype=np.float64)
        self._lam_chem = 0.0
        self._chem_on = False
        self._contact_inhib = True
        self.chem_cc_ratio = 0.0
        self._lamL_on = True
        _seed(self.seed)

    # ---------------- cell bookkeeping ----------------

    def _alloc_id(self, type_id):
        if self.free_ids:
            cid = self.free_ids.pop()
        else:
            cid = self.next_id
            if cid > self.max_cells:
                raise RuntimeError("max_cells exceeded")
            self.next_id += 1
        self.ctype[cid] = type_id
        self.V[cid] = 0.0
        self.lam_chem_cell[cid] = self._lam_chem
        self.alive.add(cid)
        return cid

    def add_cell(self, type_id, Vt, lam_V, x, y, z=0, radius=3.0, L0=None,
                 lam_L=0.0, shape="disc"):
        cid = self._alloc_id(type_id)
        self.Vt[cid] = float(Vt)
        self.lV[cid] = float(lam_V)
        if L0 is None:
            L0 = 2.0 * np.sqrt(Vt / np.pi) if self.two_d else 2.0 * (3.0 * Vt / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.L0[cid] = float(L0)
        self.lL[cid] = float(lam_L)
        self.paint(cid, x, y, z, radius, shape=shape)
        return cid

    def paint(self, cid, x, y, z, radius, shape="disc"):
        """Overwrite a region with cell `cid` (used only for seeding)."""
        x0, x1 = int(max(0, x - radius)), int(min(self.Lx - 1, x + radius))
        y0, y1 = int(max(0, y - radius)), int(min(self.Ly - 1, y + radius))
        z0, z1 = int(max(0, z - radius)), int(min(self.Lz - 1, z + radius))
        idxs = []
        for zz in range(z0, z1 + 1):
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    ok = (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2 <= radius * radius
                    if shape == "square":
                        ok = True
                    if ok:
                        idxs.append((zz * self.Ly + yy) * self.Lx + xx)
        if not idxs:
            raise ValueError("paint region empty")
        idxs = np.array(idxs, dtype=np.int64)
        self._reassign(idxs, int(cid))
        return len(idxs)

    def _reassign(self, idxs, cid):
        """Set lattice[idx] = cid for flat indices, keeping moments exact."""
        idxs = np.asarray(idxs, dtype=np.int64)
        old = self.lat[idxs]
        # remove from old owners
        for o in np.unique(old):
            o = int(o)
            if o == cid:
                continue
            m = old == o
            if o > 0:
                self._subtract_sites(o, idxs[m])
            self.lat[idxs[m]] = 0
        # handle sites that already belonged to cid: leave untouched
        m_same = old == cid
        keep = idxs[~m_same]
        if keep.size == 0:
            return
        self.lat[keep] = cid
        self._add_sites(cid, keep)

    def _site_coords(self, idxs):
        z = idxs // (self.Lx * self.Ly)
        r = idxs % (self.Lx * self.Ly)
        y = r // self.Lx
        x = r % self.Lx
        return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

    def _add_sites(self, cid, idxs):
        if cid == 0:
            return
        x, y, z = self._site_coords(idxs)
        self.V[cid] += idxs.size
        self.Sx[cid] += x.sum()
        self.Sy[cid] += y.sum()
        self.Sz[cid] += z.sum()
        self.Sxx[cid] += (x * x).sum()
        self.Syy[cid] += (y * y).sum()
        self.Szz[cid] += (z * z).sum()
        self.Sxy[cid] += (x * y).sum()
        self.Sxz[cid] += (x * z).sum()
        self.Syz[cid] += (y * z).sum()

    def _subtract_sites(self, cid, idxs):
        if cid == 0:
            return
        x, y, z = self._site_coords(idxs)
        self.V[cid] -= idxs.size
        self.Sx[cid] -= x.sum()
        self.Sy[cid] -= y.sum()
        self.Sz[cid] -= z.sum()
        self.Sxx[cid] -= (x * x).sum()
        self.Syy[cid] -= (y * y).sum()
        self.Szz[cid] -= (z * z).sum()
        self.Sxy[cid] -= (x * y).sum()
        self.Sxz[cid] -= (x * z).sum()
        self.Syz[cid] -= (y * z).sum()

    def sites_of(self, cid):
        return np.flatnonzero(self.lat == cid)

    def centroid(self, cid):
        n = self.V[cid]
        if n <= 0:
            return None
        return (self.Sx[cid] / n, self.Sy[cid] / n, self.Sz[cid] / n)

    def length_of(self, cid):
        return _len_of(
            self.two_d, self.V[cid], self.Sx[cid], self.Sy[cid], self.Sz[cid],
            self.Sxx[cid], self.Syy[cid], self.Szz[cid],
            self.Sxy[cid], self.Sxz[cid], self.Syz[cid],
        )

    def kill(self, cid):
        idxs = self.sites_of(cid)
        if idxs.size:
            self._subtract_sites(cid, idxs)
            self.lat[idxs] = 0
        self.alive.discard(cid)
        self.free_ids.append(cid)
        self.ctype[cid] = 0
        self.lV[cid] = 0.0
        self.lL[cid] = 0.0
        self.frozen[cid] = 0
        self.lam_chem_cell[cid] = 0.0
        self.chem_hits[cid] = 0

    def divide(self, cid, type_id=None, new_Vt=None, jitter=0.0):
        """Split a cell in two along its long axis (real lattice surgery).

        Sites are partitioned by the sign of their projection on the long
        axis relative to the centroid.  Returns the new cell id.
        """
        idxs = self.sites_of(cid)
        if idxs.size < 2:
            raise ValueError("cell too small to divide")
        x, y, z = self._site_coords(idxs)
        cx, cy, cz = x.mean(), y.mean(), z.mean()
        dx, dy, dz = x - cx, y - cy, z - cz
        if self.two_d:
            exx, eyy, exy = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
            tr = exx + eyy
            dd = 0.5 * (exx - eyy)
            lmax = 0.5 * tr + np.sqrt(dd * dd + exy * exy)
            # eigenvector for lmax
            if abs(exy) > 1e-12:
                ux, uy = lmax - eyy, exy
            else:
                ux, uy = (1.0, 0.0) if exx >= eyy else (0.0, 1.0)
            nrm = np.hypot(ux, uy)
            ux, uy = ux / nrm, uy / nrm
            proj = dx * ux + dy * uy
        else:
            cov = np.cov(np.vstack([dx, dy, dz]))
            w, v = np.linalg.eigh(cov)
            u = v[:, -1]
            proj = dx * u[0] + dy * u[1] + dz * u[2]
        med = np.median(proj)
        side_a = idxs[proj <= med]
        side_b = idxs[proj > med]
        if side_b.size == 0 or side_a.size == 0:
            order = np.argsort(proj)
            half = idxs.size // 2
            side_a, side_b = idxs[order[:half]], idxs[order[half:]]
        new_id = self._alloc_id(self.ctype[cid] if type_id is None else type_id)
        # Inherit the mother's per-cell parameters.  Without this the daughter
        # would be created with lambda_V = 0, i.e. NO volume constraint, and the
        # tissue would never relax back to its preferred cell area.  (This was a
        # real bug: it silently froze tissue growth in P1 and P4.)
        self.lV[new_id] = self.lV[cid]
        self.lL[new_id] = self.lL[cid]
        self.L0[new_id] = self.L0[cid]
        self._elong[new_id] = self.elong_target(cid)
        self.lam_chem_cell[new_id] = self.lam_chem_cell[cid]
        self.frozen[new_id] = 0
        self.lat[side_b] = new_id
        self._subtract_sites(cid, side_b)
        self._add_sites(new_id, side_b)
        base_Vt = self.Vt[cid] if new_Vt is None else new_Vt
        if jitter > 0.0:
            base_Vt = base_Vt * (1.0 + jitter * np.random.uniform(-1.0, 1.0))
        for c in (cid, new_id):
            self.Vt[c] = float(base_Vt)
            self.l0_rescale(c)
        return new_id

    def l0_rescale(self, cid):
        """Recompute the length target from the current target volume."""
        self.Vt_apply(cid)

    def elong_target(self, cid):
        """Elongation target ratio (1.0 = round)."""
        return self._elong.get(cid, 1.0)

    def set_elongation(self, cid, ratio):
        self._elong[cid] = float(ratio)
        self.Vt_apply(cid)

    def Vt_apply(self, cid):
        """(Re)apply target volume and the matching length target."""
        vt = float(self.Vt[cid])
        round_len = (
            2.0 * np.sqrt(vt / np.pi)
            if self.two_d
            else 2.0 * (3.0 * vt / (4.0 * np.pi)) ** (1.0 / 3.0)
        )
        self.Vt[cid] = vt
        self.L0[cid] = round_len * self.elong_target(cid)

    # ---------------- fields ----------------

    def add_field(self, name, **kw):
        shape = (self.Ly, self.Lx) if self.two_d else (self.Lz, self.Ly, self.Lx)
        self.fields[name] = Field(name, shape, **kw)
        return self.fields[name]

    def site_type_grid(self):
        return self.ctype[self.lat].reshape(
            (self.Ly, self.Lx) if self.two_d else (self.Lz, self.Ly, self.Lx)
        )

    # ---------------- dynamics ----------------

    def step(self, n_mcs=1, field_every=1, callback=None, callback_every=1):
        t0 = time.time()
        for i in range(n_mcs):
            acc, att = _mcs_kernel(
                self.lat, self.Lx, self.Ly, self.Lz, self.off, self.nnoff,
                self.J.reshape(-1), self.n_types, self.ctype,
                self.V, self.Sx, self.Sy, self.Sz,
                self.Sxx, self.Syy, self.Szz, self.Sxy, self.Sxz, self.Syz,
                self.Vt, self.lV, self.L0, self.lL, self._lamL_on,
                self._chem.reshape(-1), self.lam_chem_cell, self._chem_on,
                self._contact_inhib, self.chem_cc_ratio,
                self.frozen, self.temperature, 1, self.two_d, self.chem_hits,
                self.site_bias,
            )
            self.n_accepted += acc
            self.n_attempts += att
            self.mcs += 1
            if self.fields and field_every > 0 and (self.mcs % field_every == 0):
                st = self.site_type_grid()
                for f in self.fields.values():
                    f.update(st, self.alive)
            if callback is not None and (i % callback_every == 0):
                callback(self, i)
        return time.time() - t0

    def set_chemotaxis(self, field_name=None, lam=0.0, contact_inhibited=True,
                       enabled=True, chem_cc_ratio=None):
        """Enable chemotaxis up `field_name`.

        `lam` is the *default* per-cell strength; individual cells can be given
        a different strength afterwards via `set_cell_chemotaxis`, which is how
        tip and stalk cells are distinguished.
        """
        shape = (self.Ly, self.Lx) if self.two_d else (self.Lz, self.Ly, self.Lx)
        if field_name is None:
            self._chem = np.zeros(shape, dtype=np.float64)
        else:
            self._chem = self.fields[field_name].value
        self._lam_chem = float(lam)
        self.lam_chem_cell[:] = float(lam)
        self._chem_on = bool(enabled)
        self._contact_inhib = bool(contact_inhibited)
        # explicit ratio wins; otherwise the boolean maps to 0.0 / 1.0
        self.chem_cc_ratio = (0.0 if contact_inhibited else 1.0) \
            if chem_cc_ratio is None else float(chem_cc_ratio)
        return self

    def set_cell_chemotaxis(self, cid, lam):
        self.lam_chem_cell[cid] = float(lam)

    def refresh_chem_field(self):
        pass

    # set defaults so step() works before set_chemotaxis
    _chem = None
    _lam_chem = 0.0
    _chem_on = False
    _contact_inhib = True
    _lamL_on = True

    # ---------------- diagnostics / verification ----------------

    def energy_mech(self):
        """Exact H_contact + H_volume + H_length (chemotaxis excluded).

        H_contact uses the 1/2 convention,
            H_contact = (1/2) * sum_i sum_{j in N(i)} J(tau_i, tau_j),
        because one copy attempt changes both the "i = t" and the "j = t"
        half-sums.  With a symmetric J the kernel's
            dH = sum_{j in N(t)} [ J(tau_a,tau_j) - J(tau_b,tau_j) ]
        is then exactly the change of H_contact.
        """
        if not np.allclose(self.J, self.J.T):
            raise ValueError("contact matrix J must be symmetric")
        lat = self.lat
        h = 0.0
        # contact: sum over all valid (site, neighbour) ordered pairs
        contact = 0.0
        for off in self.off:
            nbr = self._shift_flat(off)
            if nbr is None:
                continue
            ok = self._last_shift_mask
            st_a = self.ctype[lat[ok]]
            st_b = self.ctype[lat[nbr[ok]]]
            contact += float(self.J[st_a, st_b].sum())
        h += 0.5 * contact
        # per-site bias
        h += float(np.sum(self.site_bias * (lat > 0)))
        # volume + length
        ids = np.fromiter(self.alive, dtype=np.int64) if self.alive else np.zeros(0, np.int64)
        if ids.size:
            v = self.V[ids]
            h += float((self.lV[ids] * (v - self.Vt[ids]) ** 2).sum())
            for cid in ids:
                L = self.length_of(int(cid))
                h += float(self.lL[cid] * (L - self.L0[cid]) ** 2)
        return h

    def _shift_flat(self, off):
        """Flat indices of the neighbour lattice for a given offset, or None
        for sites whose neighbour falls outside the closed box."""
        idx = np.arange(self.N, dtype=np.int64)
        z = idx // (self.Lx * self.Ly)
        r = idx % (self.Lx * self.Ly)
        y = r // self.Lx
        x = r % self.Lx
        nx, ny, nz = x + off[0], y + off[1], z + off[2]
        ok = (nx >= 0) & (nx < self.Lx) & (ny >= 0) & (ny < self.Ly) & (nz >= 0) & (nz < self.Lz)
        if not ok.any():
            return None
        out = np.full(self.N, -1, dtype=np.int64)
        out[ok] = (nz[ok] * self.Ly + ny[ok]) * self.Lx + nx[ok]
        self._last_shift_mask = ok
        return out

    def probe_move(self, x, y, z, k):
        """Return (dH_full, dH_without_chem) for one hypothetical copy attempt.

        Does not modify any state.  dH_full is what the Metropolis test uses.
        dH_without_chem is the mechanical part only, which is what the exact
        Hamiltonian comparison is valid for.
        """
        lat_before = self.lat.copy()
        d_mech, _ = _attempt(
            self.lat, self.Lx, self.Ly, self.Lz, self.off, self.nnoff,
            self.J.reshape(-1), self.n_types, self.ctype,
            self.V, self.Sx, self.Sy, self.Sz,
            self.Sxx, self.Syy, self.Szz, self.Sxy, self.Sxz, self.Syz,
            self.Vt, self.lV, self.L0, self.lL, self._lamL_on,
            self._chem.reshape(-1), self._lam_zero, False, self._contact_inhib,
            self.chem_cc_ratio, self.frozen, self.temperature, x, y, z, k, False,
            self.two_d, self.chem_hits, self.site_bias,
        )
        d_full, _ = _attempt(
            self.lat, self.Lx, self.Ly, self.Lz, self.off, self.nnoff,
            self.J.reshape(-1), self.n_types, self.ctype,
            self.V, self.Sx, self.Sy, self.Sz,
            self.Sxx, self.Syy, self.Szz, self.Sxy, self.Sxz, self.Syz,
            self.Vt, self.lV, self.L0, self.lL, self._lamL_on,
            self._chem.reshape(-1), self.lam_chem_cell, self._chem_on,
            self._contact_inhib, self.chem_cc_ratio,
            self.frozen, self.temperature, x, y, z, k, False, self.two_d,
            self.chem_hits, self.site_bias,
        )
        assert np.array_equal(lat_before, self.lat), "probe modified the lattice"
        return d_full, d_mech

    def apply_move(self, x, y, z, k):
        """Force-apply one copy attempt (bypassing Metropolis).

        Returns (dH_reported_by_kernel, applied).  Used by the verification
        harness only.
        """
        d_full, d_mech = self.probe_move(x, y, z, k)
        ok = _force_apply(
            self.lat, self.Lx, self.Ly, self.Lz, self.off,
            self.V, self.Sx, self.Sy, self.Sz,
            self.Sxx, self.Syy, self.Szz, self.Sxy, self.Sxz, self.Syz,
            x, y, z, k,
        )
        return d_mech, bool(ok)

    def contact_pairs(self):
        """Return (counts dict {(cid_a,cid_b): n_contacts}, adhesion pairs).

        Counts each unordered lattice-neighbour pair once.
        """
        lat = self.lat
        counts = {}
        ids = self.alive
        for off in self.off:
            nbr = self._shift_flat(off)
            if nbr is None:
                continue
            ok = self._last_shift_mask
            a = lat[ok]
            b = lat[nbr[ok]]
            m = (a > 0) & (b > 0)
            if not m.any():
                continue
            aa = a[m]
            bb = b[m]
            key = aa * (self.max_cells + 1) + bb
            uniq, cnt = np.unique(key, return_counts=True)
            for u, c in zip(uniq, cnt):
                u = int(u)
                x1, x2 = u // (self.max_cells + 1), u % (self.max_cells + 1)
                if x1 == x2:
                    continue
                kk = (x1, x2) if x1 < x2 else (x2, x1)
                counts[kk] = counts.get(kk, 0) + int(c)
        return counts

    def neighbour_sets(self):
        """For each cell, the set of distinct neighbouring cell ids."""
        nb = {}
        for (a, b) in self.contact_pairs():
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
        return nb

    def isolated_floating(self):
        """Cell ids with no lattice contact to any other cell."""
        nb = self.neighbour_sets()
        return [c for c in self.alive if not nb.get(c)]

    def set_site_bias(self, coef):
        """Set the per-site potential coefficient (flat array of length N)."""
        coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        if coef.size != self.N:
            raise ValueError(f"site bias must have length {self.N}, got {coef.size}")
        self.site_bias = coef.copy()
        return self

    def medium_facing_count_grid(self):
        """Per-site count of medium neighbours (used to build wound bias)."""
        out = np.zeros(self.N, dtype=np.float64)
        for off in self.off:
            nbr = self._shift_flat(off)
            if nbr is None:
                continue
            ok = self._last_shift_mask
            out[ok] += (self.lat[nbr[ok]] == 0)
        return out

    def ablate_circle(self, cx, cy, r, remove_orphans=True):
        """Set every site within radius r of (cx,cy) to medium.

        Returns the number of lattice sites removed and the number of cells
        that were completely destroyed.
        """
        yy, xx = np.mgrid[0:self.Ly, 0:self.Lx]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        idxs = np.flatnonzero(mask.reshape(-1))
        before = set(int(v) for v in np.unique(self.lat[idxs]) if v > 0)
        self._reassign(idxs, 0)
        self.lat[idxs] = 0
        killed = []
        if remove_orphans:
            killed = self.cleanup_empty_cells()
        return int(idxs.size), sorted(before), killed

    def cleanup_empty_cells(self):
        """Drop cells that no longer own any lattice site."""
        gone = [c for c in list(self.alive) if self.V[c] <= 0]
        for c in gone:
            self.alive.discard(c)
            self.free_ids.append(c)
            self.ctype[c] = 0
            self.lV[c] = 0.0
            self.lL[c] = 0.0
            self.frozen[c] = 0
            self.lam_chem_cell[c] = 0.0
        return gone

    def medium_facing_sites(self, cid):
        """Number of lattice bonds from `cid` to a medium site.

        This is exactly the precondition of contact-inhibited chemotaxis: the
        term requires the target site to be medium, so a cell with zero
        medium-facing sites provably cannot chemotax.
        """
        total = 0
        for off in self.off:
            nbr = self._shift_flat(off)
            if nbr is None:
                continue
            ok = self._last_shift_mask
            a = self.lat[ok]
            b = self.lat[nbr[ok]]
            total += int(np.sum((a == cid) & (b == 0)))
        return total

    def occupancy(self):
        return int((self.lat > 0).sum())

    def volume_stats(self):
        ids = np.fromiter(self.alive, dtype=np.int64) if self.alive else np.zeros(0, np.int64)
        if ids.size == 0:
            return {}
        v = self.V[ids]
        tgt = self.Vt[ids]
        return {
            "n_cells": int(ids.size),
            "volume_mean": float(v.mean()),
            "volume_std": float(v.std()),
            "volume_min": float(v.min()),
            "volume_max": float(v.max()),
            "target_mean": float(tgt.mean()),
            "rel_dev_mean": float(np.mean(np.abs(v - tgt) / np.maximum(tgt, 1e-12))),
            "total_sites": int(v.sum()),
        }

    def sanity_check(self, tol=1e-9):
        """Recompute every per-cell moment from scratch and compare."""
        ids = sorted(self.alive)
        worst_V = 0
        worst_S = 0.0
        worst_M = 0.0
        for cid in ids:
            idxs = self.sites_of(cid)
            x, y, z = self._site_coords(idxs)
            worst_V = max(worst_V, abs(self.V[cid] - idxs.size))
            worst_S = max(worst_S, abs(self.Sx[cid] - x.sum()), abs(self.Sy[cid] - y.sum()),
                          abs(self.Sz[cid] - z.sum()))
            worst_M = max(worst_M,
                          abs(self.Sxx[cid] - (x * x).sum()),
                          abs(self.Syy[cid] - (y * y).sum()),
                          abs(self.Szz[cid] - (z * z).sum()),
                          abs(self.Sxy[cid] - (x * y).sum()),
                          abs(self.Sxz[cid] - (x * z).sum()),
                          abs(self.Syz[cid] - (y * z).sum()))
        # lattice consistency
        lat_ids = set(int(v) for v in np.unique(self.lat) if v > 0)
        missing = lat_ids - set(ids)
        extra = set(ids) - lat_ids
        return {
            "worst_volume_error": float(worst_V),
            "worst_first_moment_error": float(worst_S),
            "worst_second_moment_error": float(worst_M),
            "cells_missing_from_inventory": sorted(missing),
            "cells_in_inventory_without_sites": sorted(extra),
            "ok": bool(worst_V == 0 and worst_S <= tol and worst_M <= tol
                       and not missing and not extra),
        }

    def snapshot(self):
        return {
            "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
            "mcs": self.mcs,
            "temperature": self.temperature,
            "seed": self.seed,
            "n_accepted": self.n_accepted,
            "n_attempts": self.n_attempts,
            "acceptance": (self.n_accepted / self.n_attempts) if self.n_attempts else 0.0,
            "n_cells": len(self.alive),
            "occupancy": self.occupancy(),
            "volume_stats": self.volume_stats(),
            "fields": {k: {"min": float(v.value.min()), "max": float(v.value.max()),
                           "mean": float(v.value.mean()), "mass": v.mass()}
                       for k, v in self.fields.items()},
        }

    def save_state(self, path):
        ids = np.fromiter(sorted(self.alive), dtype=np.int64) if self.alive else np.zeros(0, np.int64)
        np.savez_compressed(
            path,
            lat=self.lat.astype(np.int32),
            ids=ids,
            ctype=self.ctype[ids],
            V=self.V[ids],
            Vt=self.Vt[ids],
            **{f"f_{k}": v.value for k, v in self.fields.items()},
        )
