"""epidermis.py -- P4: self-organising stratified epidermis.

MECHANISM (all local rules; the layers are NOT placed by hand)
--------------------------------------------------------------
The tissue is a 2D vertical cross-section with the basement membrane along the
bottom wall.  The rules are:

  1. **Basal cells proliferate.**  A cell divides only while it still owns a
     site on the basement membrane (y = 0).  Division at the base is what
     pushes older cells upward -- there is no explicit "up" force.
  2. **Loss of basement contact triggers differentiation.**  A cell that no
     longer touches the basement membrane is no longer basal; it enters the
     suprabasal (spinous) compartment and then advances through granular to
     cornified on a timer.
  3. **Terminal cells are dead and are shed.**  Cornified cells do not divide
     and are removed after a fixed residence time, which closes the turnover
     cycle (production at the base = loss at the surface).

So the steady-state thickness is an emergent quantity:

    thickness  ~  (proliferation rate) x (transit time through the layers)

which is a falsifiable prediction: doubling the cell-cycle time must roughly
halve the thickness.

O2 COUPLING
-----------
O2 is supplied from below (dermis / basement membrane) and consumed by every
cell, so pO2 falls with height.  Proliferation requires O2 and differentiation
is accelerated where pO2 is low, so the O2 gradient is a second, independent
driver of the layered structure.  The test is that the layer boundaries move
when the O2 supply is changed.

PROVENANCE
----------
There are **no measured skin parameters in this project's verified anchor file**
(`MEASURED_ANCHORS.md` covers the Drosophila notum, not skin).  Every parameter
below is therefore labelled `illustrative`.  What is being demonstrated is that
the layered architecture and a stable turnover follow from the local rules --
not that the numbers match human skin.  Absolute thicknesses are reported in
lattice units converted with a declared, illustrative 1 um lattice spacing.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy import ndimage

from cpm import CPM

MEDIUM, BASAL, SPINOUS, GRANULAR, CORNIFIED = 0, 1, 2, 3, 4

TYPE_NAMES = {BASAL: "basal", SPINOUS: "spinous", GRANULAR: "granular",
              CORNIFIED: "cornified"}

TYPE_COLORS = {
    BASAL: (0.20, 0.35, 0.70),
    SPINOUS: (0.45, 0.62, 0.85),
    GRANULAR: (0.85, 0.75, 0.40),
    CORNIFIED: (0.80, 0.45, 0.35),
}

PARAMS = {
    "lattice_spacing_um": (1.0, "illustrative; all lengths therefore in um"),
    "cell_sites": (100.0, "illustrative; ~10 um keratinocyte diameter"),
    "lambda_volume": (8.0, "illustrative"),
    "lambda_length": (0.15, "illustrative"),
    "elongation_basal": (1.2, "illustrative"),
    "elongation_terminal": (2.0, "illustrative; flattened corneocytes"),
    "temperature": (3.0, "illustrative"),
    "J_self": (1.0, "illustrative"),
    "J_medium": (5.0, "illustrative"),
    "J_cross": (1.0, "illustrative"),
    "J_cornified_medium": (2.0, "illustrative"),
    "cycle_mcs": (400.0, "illustrative"),
    "basal_division_crowding_max": (0.55, "illustrative"),
    "stage_duration_mcs": (600.0, "illustrative"),
    "cornified_lifetime_mcs": (900.0, "illustrative"),
    "substrate_adhesion": (0.8, "illustrative"),
    "o2_diffusion": (1.2, "illustrative"),
    "o2_consumption": (0.02, "illustrative"),
    "o2_source_value": (1.0, "illustrative"),
    "o2_half_saturation": (0.25, "illustrative"),
    "hypoxia_differentiation_gain": (1.5, "illustrative"),
    "max_cells": (1200, "illustrative (compute cap)"),
    "bio_every_mcs": (10, "illustrative"),
    "minutes_per_mcs": (1.0, "illustrative"),
}


def p(key):
    return PARAMS[key][0]


class Epidermis:
    def __init__(self, Lx=200, Ly=200, seed=0, params=None,
                 hypoxia_coupling=True, initial_layers=2):
        self.cfg = dict(PARAMS)
        if params:
            for k, v in params.items():
                self.cfg[k] = (v, "override") if not isinstance(v, tuple) else v
        self.Lx, self.Ly = int(Lx), int(Ly)
        self.seed = int(seed)
        self.hypoxia_coupling = bool(hypoxia_coupling)
        self.h = self.cfg["lattice_spacing_um"][0]
        m = CPM(self.Lx, self.Ly, n_types=5, temperature=self.cfg["temperature"][0],
                seed=self.seed, max_cells=int(self.cfg["max_cells"][0]) + 50,
                lattice_spacing_um=self.h)
        J = np.zeros((5, 5))
        for a in range(1, 5):
            J[a, a] = self.cfg["J_self"][0]
            J[a, MEDIUM] = J[MEDIUM, a] = self.cfg["J_medium"][0]
            for b in range(a + 1, 5):
                J[a, b] = J[b, a] = self.cfg["J_cross"][0]
        # dead, loosely attached surface cells
        J[CORNIFIED, MEDIUM] = J[MEDIUM, CORNIFIED] = self.cfg["J_cornified_medium"][0]
        m.J = J
        self.m = m

        # O2 supplied at the basement membrane (bottom two rows)
        src = np.zeros((self.Ly, self.Lx), dtype=bool)
        src[:2, :] = True
        self.o2 = m.add_field("O2", D=self.cfg["o2_diffusion"][0], decay=0.0, dt=0.25,
                              initial=self.cfg["o2_source_value"][0], source_mask=src,
                              source_value=self.cfg["o2_source_value"][0],
                              uptake={BASAL: self.cfg["o2_consumption"][0],
                                      SPINOUS: self.cfg["o2_consumption"][0],
                                      GRANULAR: self.cfg["o2_consumption"][0],
                                      CORNIFIED: 0.5 * self.cfg["o2_consumption"][0]})
        self.stage_age = {}
        self.cycle = {}
        self.parent = {}
        self.lineage_id = {}
        self._next_lineage = 1
        self.n_divisions_total = 0
        self.n_shed_total = 0
        self.history = []
        self._fill_base(initial_layers)
        self._update_substrate_bias()

    # ------------------------------------------------------------------
    def _fill_base(self, layers):
        s = int(round(np.sqrt(self.cfg["cell_sites"][0])))
        nx = self.Lx // s
        for gy in range(layers):
            for gx in range(nx):
                idxs = [yy * self.Lx + xx
                        for yy in range(gy * s, (gy + 1) * s)
                        for xx in range(gx * s, (gx + 1) * s)]
                cid = self.m._alloc_id(BASAL)
                self.m.Vt[cid] = self.cfg["cell_sites"][0]
                self.m.lV[cid] = self.cfg["lambda_volume"][0]
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                arr = np.array(idxs, dtype=np.int64)
                self.m.lat[arr] = cid
                self.m._add_sites(cid, arr)
                self.m._elong[cid] = self.cfg["elongation_basal"][0]
                self.m.Vt_apply(cid)
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                self.m.lam_chem_cell[cid] = 0.0
                self.stage_age[cid] = 0.0
                self.cycle[cid] = 0.0
                self.parent[cid] = 0
                self.lineage_id[cid] = self._next_lineage
                self._next_lineage += 1

    def _update_substrate_bias(self):
        """Modest adhesion of any cell to the basement membrane.

        Implemented with the engine's exact per-site potential:
            H = -lambda_ad * [site occupied] for sites in the bottom rows.
        """
        yy = np.arange(self.m.N) // self.Lx
        coef = np.zeros(self.m.N)
        coef[yy < 3] = -self.cfg["substrate_adhesion"][0]
        self.m.set_site_bias(coef)

    # ------------------------------------------------------------------
    def is_basal(self, cid):
        idx = np.flatnonzero(self.m.lat == cid)
        if idx.size == 0:
            return False
        return bool(np.any(idx < self.Lx))  # any site in row y = 0

    def cell_o2(self, ids=None):
        tot = np.bincount(self.m.lat, weights=self.o2.value.reshape(-1),
                          minlength=self.m.max_cells + 1)
        cnt = np.bincount(self.m.lat, minlength=self.m.max_cells + 1).astype(float)
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), 0.0)
        ids = self.m.alive if ids is None else ids
        return {int(c): float(mean[int(c)]) for c in ids}

    # ------------------------------------------------------------------
    def biology_step(self, dt_mcs):
        m = self.m
        o2 = self.cell_o2()
        half = self.cfg["o2_half_saturation"][0]
        gain = self.cfg["hypoxia_differentiation_gain"][0]
        n_div = n_diff = n_shed = 0

        # ---- differentiation: loss of basement contact ----
        for cid in list(m.alive):
            t = int(m.ctype[cid])
            if t == CORNIFIED:
                self.stage_age[cid] = self.stage_age.get(cid, 0.0) + dt_mcs
                continue
            if t == BASAL:
                if not self.is_basal(cid):
                    m.ctype[cid] = SPINOUS
                    self.stage_age[cid] = 0.0
                    m.set_elongation(cid, self.cfg["elongation_basal"][0])
                    m.lL[cid] = self.cfg["lambda_length"][0]
                    m.Vt_apply(cid)
                    m.lL[cid] = self.cfg["lambda_length"][0]
                    n_diff += 1
                else:
                    self.stage_age[cid] = 0.0
                continue
            # suprabasal stages advance on a timer, accelerated by hypoxia
            rate = 1.0
            if self.hypoxia_coupling:
                po2 = o2.get(cid, 1.0)
                rate = 1.0 + gain * max(0.0, (half - po2) / max(half, 1e-9))
            self.stage_age[cid] = self.stage_age.get(cid, 0.0) + dt_mcs * rate
            if self.stage_age[cid] >= self.cfg["stage_duration_mcs"][0]:
                self.stage_age[cid] = 0.0
                nxt = {SPINOUS: GRANULAR, GRANULAR: CORNIFIED}.get(t)
                if nxt is not None:
                    m.ctype[cid] = nxt
                    if nxt == CORNIFIED:
                        m.set_elongation(cid, self.cfg["elongation_terminal"][0])
                        m.lL[cid] = self.cfg["lambda_length"][0]
                        m.Vt_apply(cid)
                        m.lL[cid] = self.cfg["lambda_length"][0]
                    n_diff += 1

        # ---- shedding of terminal cells ----
        shed = []
        top = self.Ly - 1
        for cid in list(m.alive):
            if int(m.ctype[cid]) != CORNIFIED:
                continue
            idx = np.flatnonzero(m.lat == cid)
            at_top = bool(np.any(idx // self.Lx >= top - 1)) if idx.size else False
            too_old = (self.stage_age.get(cid, 0.0)
                       >= self.cfg["cornified_lifetime_mcs"][0])
            if at_top or too_old:
                shed.append(cid)
        for cid in shed:
            m.kill(cid)
            for d in (self.stage_age, self.cycle, self.lineage_id, self.parent):
                d.pop(cid, None)
            n_shed += 1
        self.n_shed_total += n_shed

        # ---- basal proliferation ----
        for cid in list(m.alive):
            if int(m.ctype[cid]) != BASAL:
                continue
            if not self.is_basal(cid):
                continue
            free = m.medium_facing_sites(cid) / max(m.V[cid], 1.0)
            if free > self.cfg["basal_division_crowding_max"][0]:
                self.cycle[cid] = 0.0
                continue
            rate = 1.0
            if self.hypoxia_coupling:
                po2 = o2.get(cid, 0.0)
                # O2 is required for division: saturating dependence
                rate = po2 / (po2 + half) if po2 > 0 else 0.0
            self.cycle[cid] = self.cycle.get(cid, 0.0) + dt_mcs * rate
            if self.cycle[cid] >= self.cfg["cycle_mcs"][0]:
                self.cycle[cid] = 0.0
                if len(m.alive) >= int(self.cfg["max_cells"][0]):
                    continue
                try:
                    new = m.divide(cid)
                except ValueError:
                    continue
                m.ctype[new] = BASAL
                m.set_elongation(new, self.cfg["elongation_basal"][0])
                m.lL[new] = self.cfg["lambda_length"][0]
                m.Vt_apply(new)
                m.lL[new] = self.cfg["lambda_length"][0]
                m.lam_chem_cell[new] = 0.0
                self.stage_age[new] = 0.0
                self.cycle[new] = 0.0
                self.parent[new] = cid
                self.lineage_id[new] = self._next_lineage
                self._next_lineage += 1
                n_div += 1
                self.n_divisions_total += 1
        return {"n_divisions": n_div, "n_differentiated": n_diff, "n_shed": n_shed}

    # ------------------------------------------------------------------
    def run(self, n_mcs, bio_every=None, record_every=None, verbose=False):
        m = self.m
        bio_every = bio_every or int(self.cfg["bio_every_mcs"][0])
        record_every = record_every or bio_every * 10
        t0 = time.time()
        done = 0
        while done < n_mcs:
            chunk = min(bio_every, n_mcs - done)
            m.step(chunk, field_every=1)
            done += chunk
            info = self.biology_step(float(chunk))
            if m.mcs % record_every < bio_every or done >= n_mcs:
                h = self.metrics(**info)
                self.history.append(h)
                if verbose:
                    print(f"  mcs={h['mcs']:5d} thick={h['thickness_um']:6.1f}um "
                          f"layers={h['layer_thickness_um']} cells={h['n_cells']:4d} "
                          f"shed={h['n_shed_total']:4d}")
        self.wall_seconds = time.time() - t0
        return self.history

    # ------------------------------------------------------------------
    def metrics(self, n_divisions=0, n_differentiated=0, n_shed=0):
        m = self.m
        lat = m.lat.reshape(self.Ly, self.Lx)
        counts = {}
        for t, name in TYPE_NAMES.items():
            counts[name] = int(np.sum(m.ctype[list(m.alive)] == t))
        area = {name: counts[name] * self.cfg["cell_sites"][0] for name in counts}
        total_sites = int((lat > 0).sum())
        thickness_um = counts["__dummy__"] if False else float(total_sites / self.Lx * self.h)
        # per-layer thickness from the vertical extent of each type
        layer_um = {}
        for t, name in TYPE_NAMES.items():
            mask = m.ctype[lat] == t
            if mask.any():
                ys = np.flatnonzero(mask.any(axis=1))
                layer_um[name] = float((ys[-1] - ys[0] + 1) * self.h)
            else:
                layer_um[name] = 0.0
        prof = self.o2.value.mean(axis=1)
        # correlation between height and differentiation stage
        stages = np.zeros_like(lat, dtype=float)
        for t in TYPE_NAMES:
            stages[m.ctype[lat] == t] = t
        ys = np.arange(self.Ly)[:, None] * np.ones((1, self.Lx))
        occ = lat > 0
        if occ.sum() > 10:
            corr = float(np.corrcoef(ys[occ], stages[occ])[0, 1])
        else:
            corr = None
        return {
            "mcs": int(m.mcs),
            "minutes_illustrative": float(m.mcs * self.cfg["minutes_per_mcs"][0]),
            "n_cells": len(m.alive),
            "counts": counts,
            "thickness_um": thickness_um,
            "layer_thickness_um": layer_um,
            "o2_at_base": float(prof[0]),
            "o2_at_surface": float(prof[min(self.Ly - 1, int(np.flatnonzero(occ.any(axis=1))[-1]))]) if occ.any() else 0.0,
            "o2_min": float(self.o2.value.min()),
            "height_stage_correlation": corr,
            "n_divisions_recent": int(n_divisions),
            "n_differentiated_recent": int(n_differentiated),
            "n_shed_recent": int(n_shed),
            "n_divisions_total": int(self.n_divisions_total),
            "n_shed_total": int(self.n_shed_total),
            "acceptance": (m.n_accepted / m.n_attempts) if m.n_attempts else 0.0,
        }

    # ------------------------------------------------------------------
    def lineage_check(self):
        problems = []
        for cid, par in self.parent.items():
            if par == 0:
                continue
            if par not in self.parent:
                problems.append(f"cell {cid} has missing parent {par}")
                continue
            if self.lineage_id.get(par, 1 << 60) >= self.lineage_id.get(cid, -1):
                problems.append(f"lineage id not increasing for {cid} <- {par}")
        ids = list(self.lineage_id.values())
        if len(set(ids)) != len(ids):
            problems.append("duplicate lineage ids")
        return {"ok": not problems, "problems": problems,
                "n_cells_tracked": len(self.parent)}

    def type_image(self):
        from viz import type_image
        return type_image(self.m, TYPE_COLORS)
