"""merks.py -- faithful re-implementation of the published vasculogenesis model.

SOURCE (this module follows the published model, it does not invent one)
-----------------------------------------------------------------------
Merks RMH, Perryn ED, Shirinifard A, Glazier JA (2008)
"Contact-Inhibited Chemotaxis in De Novo and Sprouting Blood-Vessel Growth"
PLoS Comput Biol 4(9): e1000163.  doi:10.1371/journal.pcbi.1000163
Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC2528254/

WHAT THE PAPER ACTUALLY REQUIRES (corrected after reading it)
------------------------------------------------------------
My earlier hypothesis -- that the missing ingredient was an explicit
extracellular-matrix field with proteolysis -- was WRONG.  The paper's ECM
appears only through the chemoattractant DECAY term ("which degrades in the ECM
at a rate epsilon (e.g. due to proteolytic enzymes or by binding to ECM
components)").  There is no ECM field in the model.

The two ingredients I had wrong are:

 1. **Zero cluster surface tension.**  The paper states: "In most of our
    simulations we set the EC-EC adhesion equal to the EC-ECM adhesion (i.e.
    J(c,c) = 2J(c,M); the factor of 2 arises because we model the ECM as a
    single large generalized cell), which is equivalent to setting the surface
    tension of the cluster to zero."  In my earlier runs J(c,c) << J(c,M),
    i.e. a large POSITIVE surface tension, which is exactly what collapses the
    cells into a compact blob.

 2. **A short chemoattractant diffusion length.**  L = sqrt(D/epsilon).  The
    paper uses alpha = epsilon = 1e-3 /s and D = 1e-13 m^2/s, giving
    L = 10 um = 5 lattice sites (2 um pixels).  My earlier runs used
    L ~ 23 um, far too smooth.

 3. Contact inhibition of chemotaxis enters as the RATIO of the chemotactic
    sensitivity at cell-cell interfaces to that at cell-ECM interfaces,
    chi(c,c)/chi(c,M).  The paper screens this ratio and reports a phase
    transition at about 0.5.  That is a continuous parameter (engine:
    `chem_cc_ratio`), not a boolean.

UNIT CONVERSION (declared, from the paper's own statements)
----------------------------------------------------------
    lattice pixel            = 2 um            (paper)
    1 MCS                    = 30 s            (paper)
    cell area                = 200 um^2 -> 50 lattice sites (paper)
    chemoattractant D        = 1e-13 m^2/s = 0.1 um^2/s (paper)
    secretion alpha          = 1e-3 /s ; decay epsilon = alpha (paper)
    =>  D_lattice = 0.1 um^2/s * 30 s / (2 um)^2        = 0.75 site^2/MCS
        decay_lat = 1e-3 /s * 30 s                      = 0.03 /MCS
        secretion_lat                                   = 0.03 /MCS
        diffusion length = sqrt(0.75/0.03) = 5 sites = 10 um  (matches paper)

By default cell-autonomous elongation is switched OFF, matching the paper's
statement: "Unless we state otherwise, in this paper we neglect cell-autonomous
elongation."
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from cpm import CPM

MEDIUM, EC = 0, 1

TYPE_COLORS = {EC: (0.42, 0.60, 0.90)}

# --- conversions straight from the paper's stated numbers -------------------
UM_PER_SITE = 2.0
SECONDS_PER_MCS = 30.0
CELL_AREA_UM2 = 200.0
CELL_SITES = CELL_AREA_UM2 / UM_PER_SITE ** 2          # 50 sites
D_UM2_PER_S = 1e-13 * 1e12                             # 1 m^2 = 1e12 um^2 -> 0.1 um^2/s
ALPHA_PER_S = 1e-3
EPSILON_PER_S = ALPHA_PER_S

D_LATTICE = D_UM2_PER_S * SECONDS_PER_MCS / UM_PER_SITE ** 2   # 0.75
DECAY_LATTICE = EPSILON_PER_S * SECONDS_PER_MCS                # 0.03
SECRETION_LATTICE = ALPHA_PER_S * SECONDS_PER_MCS              # 0.03
DIFFUSION_LENGTH_SITES = np.sqrt(D_LATTICE / DECAY_LATTICE)    # 5.0 sites = 10 um

PARAMS = {
    # --- taken from the paper ---
    "J_ec_ec": (6.0, "measured/derived: J(c,c)=2*J(c,M) -> ZERO cluster surface tension (Merks 2008)"),
    "J_ec_medium": (3.0, "derived: with J(c,c)=6 gives zero surface tension (Merks 2008)"),
    "vegf_diffusion": (D_LATTICE, "derived from D=1e-13 m^2/s, 2 um pixels, 1 MCS=30 s (Merks 2008)"),
    "vegf_decay": (DECAY_LATTICE, "derived from epsilon=1e-3 /s (Merks 2008)"),
    "vegf_secretion_ec": (SECRETION_LATTICE, "derived from alpha=1e-3 /s (Merks 2008)"),
    "cell_sites": (CELL_SITES, "measured: EC area ~200 um^2, 2 um pixels (Merks 2008)"),
    # --- not specified numerically in the source we read: illustrative ---
    "lambda_volume": (4.0, "illustrative"),
    "lambda_length": (0.0, "paper neglects cell-autonomous elongation by default"),
    "temperature": (3.0, "illustrative (paper's cell-motility parameter T)"),
    "lambda_chem": (20.0, "illustrative"),
    "chem_cc_ratio": (0.0, "0 = full contact inhibition; paper screens this ratio"),
    "max_cells": (1500, "illustrative (compute cap)"),
}


def p(key):
    return PARAMS[key][0]


class MerksVasculogenesis:
    """De novo and cluster (sprouting) vasculogenesis, published ingredients."""

    def __init__(self, L=200, seed=0, mode="cluster", n_cells=128,
                 chem_cc_ratio=0.0, params=None, elong_target=1.0):
        self.cfg = dict(PARAMS)
        if params:
            for k, v in params.items():
                self.cfg[k] = (v, "override") if not isinstance(v, tuple) else v
        if chem_cc_ratio is not None:
            self.cfg["chem_cc_ratio"] = (float(chem_cc_ratio), "input")
        self.L = int(L)
        self.seed = int(seed)
        self.mode = mode
        self.n_requested = int(n_cells)

        m = CPM(self.L, self.L, n_types=2, temperature=self.cfg["temperature"][0],
                seed=self.seed, max_cells=int(self.cfg["max_cells"][0]) + 50,
                lattice_spacing_um=UM_PER_SITE)
        J = np.zeros((2, 2))
        J[EC, EC] = self.cfg["J_ec_ec"][0]
        J[EC, MEDIUM] = J[MEDIUM, EC] = self.cfg["J_ec_medium"][0]
        m.J = J
        self.m = m
        # dt = 1.0 MCS: the diffusion/decay/secretion constants above are all
        # expressed PER MCS, so the field must be integrated one MCS per step.
        self.vegf = m.add_field("VEGF", D=self.cfg["vegf_diffusion"][0],
                                decay=self.cfg["vegf_decay"][0], dt=1.0,
                                # the ECs themselves are the source, as in the
                                # paper (autocrine chemoattractant).  Omitting
                                # this leaves the field identically zero and the
                                # chemotaxis term silently does nothing.
                                secretion={EC: self.cfg["vegf_secretion_ec"][0]})
        m.set_chemotaxis("VEGF", lam=self.cfg["lambda_chem"][0],
                         enabled=True, chem_cc_ratio=self.cfg["chem_cc_ratio"][0])
        self.elong_target = float(elong_target)

        if mode == "cluster":
            self.n_made = self._seed_cluster()
        elif mode == "de_novo":
            self.n_made = self._seed_scattered()
        else:
            raise ValueError("mode must be 'cluster' or 'de_novo'")
        self.history = []

    # ------------------------------------------------------------------
    def _new_cell(self, x, y, radius):
        cid = self.m.add_cell(EC, self.cfg["cell_sites"][0],
                              self.cfg["lambda_volume"][0], x, y, radius=radius)
        self.m.set_elongation(cid, self.elong_target)
        self.m.lL[cid] = self.cfg["lambda_length"][0]
        self.m.Vt_apply(cid)
        self.m.lL[cid] = self.cfg["lambda_length"][0]
        return cid

    def _seed_cluster(self):
        """One rounded cluster of `n_cells` ECs, as in the paper's Fig. 4."""
        m = self.m
        side = int(np.sqrt(self.cfg["cell_sites"][0]))
        area_sites = self.n_requested * self.cfg["cell_sites"][0]
        R = np.sqrt(area_sites / np.pi)
        cx = cy = self.L / 2.0
        made = 0
        step = side
        ys = np.arange(0, self.L, step)
        for y0 in ys:
            for x0 in ys:
                xc, yc = x0 + side / 2.0, y0 + side / 2.0
                if (xc - cx) ** 2 + (yc - cy) ** 2 > R ** 2:
                    continue
                idxs = np.array([yy * self.L + xx
                                 for yy in range(y0, min(y0 + side, self.L))
                                 for xx in range(x0, min(x0 + side, self.L))],
                                dtype=np.int64)
                cid = self.m._alloc_id(EC)
                self.m.Vt[cid] = self.cfg["cell_sites"][0]
                self.m.lV[cid] = self.cfg["lambda_volume"][0]
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                self.m.lat[idxs] = cid
                self.m._add_sites(cid, idxs)
                self.m.set_elongation(cid, self.elong_target)
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                self.m.Vt_apply(cid)
                made += 1
        return made

    def _seed_scattered(self):
        """Dispersed ECs, as in the paper's Fig. 2 (1000 cells over 700 um)."""
        m = self.m
        rng = np.random.default_rng(self.seed)
        made, tries = 0, 0
        margin = 6
        while made < self.n_requested and tries < 400 * self.n_requested + 2000:
            tries += 1
            x = int(rng.integers(margin, self.L - margin))
            y = int(rng.integers(margin, self.L - margin))
            if m.lat[y * self.L + x] != 0:
                continue
            try:
                self._new_cell(x, y, radius=4.0)
            except ValueError:
                continue
            made += 1
        return made

    # ------------------------------------------------------------------
    def ec_sites(self):
        return np.flatnonzero(self.m.ctype[self.m.lat] == EC)

    def compactness(self, ids=None):
        """C = A_cluster / A_hull, exactly as defined in the paper's Fig. 5.

        C = 1 for a perfectly circular cluster; C -> 0 for a branched or
        dispersed cluster.  In 2D `ConvexHull.volume` is the hull area.
        """
        idx = self.ec_sites()
        if idx.size < 3:
            return None, 0, 0
        x = (idx % self.L).astype(float)
        y = (idx // self.L).astype(float)
        pts = np.column_stack([x, y])
        try:
            hull = ConvexHull(pts)
            a_hull = float(hull.volume)
        except Exception:
            return None, int(idx.size), 0
        a_cluster = float(idx.size)
        return (a_cluster / a_hull if a_hull > 0 else None), int(a_cluster), a_hull

    def metrics(self):
        m = self.m
        lat = m.lat.reshape(self.L, self.L)
        ec = m.ctype[lat] == EC
        lbl, n_comp = ndimage.label(ec, structure=np.ones((3, 3)))
        # lacunae = cell-free regions fully enclosed by EC (the network signature)
        mid, n_med = ndimage.label(~ec, structure=np.ones((3, 3)))
        border = set(np.unique(np.concatenate([mid[0, :], mid[-1, :], mid[:, 0], mid[:, -1]])))
        lacunae = [k for k in range(1, n_med + 1) if k not in border]
        sizes = ndimage.sum(np.ones_like(mid), mid, index=lacunae) if lacunae else []
        C, a_clu, a_hull = self.compactness()
        return {
            "mcs": int(m.mcs),
            "hours": float(m.mcs * SECONDS_PER_MCS / 3600.0),
            "n_ec": int((m.ctype[list(m.alive)] == EC).sum()) if m.alive else 0,
            "ec_area_um2": float(ec.sum() * UM_PER_SITE ** 2),
            "n_components": int(n_comp),
            "n_lacunae": int(len(lacunae)),
            "lacuna_mean_area_um2": float(np.mean(sizes) * UM_PER_SITE ** 2) if lacunae else 0.0,
            "compactness": C,
            "cluster_area_sites": a_clu,
            "hull_area_sites": a_hull,
            "vegf_max": float(self.vegf.value.max()),
            "vegf_mean": float(self.vegf.value.mean()),
            "chem_cc_ratio": self.cfg["chem_cc_ratio"][0],
            "acceptance": (m.n_accepted / m.n_attempts) if m.n_attempts else 0.0,
        }

    def run(self, n_mcs, record_every=200, verbose=False):
        m = self.m
        done = 0
        t0 = time.time()
        while done < n_mcs:
            chunk = min(record_every, n_mcs - done)
            m.step(chunk, field_every=1)
            done += chunk
            h = self.metrics()
            self.history.append(h)
            if verbose:
                print(f"  mcs={h['mcs']:5d} EC={h['n_ec']:4d} comp={h['n_components']:3d} "
                      f"lacunae={h['n_lacunae']:3d} C={h['compactness'] if h['compactness'] is None else round(h['compactness'],3)}",
                      flush=True)
        self.wall_seconds = time.time() - t0
        return self.history

    def type_image(self):
        from viz import type_image
        return type_image(self.m, TYPE_COLORS)
