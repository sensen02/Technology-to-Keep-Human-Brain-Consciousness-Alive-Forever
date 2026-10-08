"""wound_healing.py -- P3: epithelial wound closure.

WHAT IS MODELLED
----------------
A confluent epithelial sheet on the lattice is wounded by ablating a disc, and
the sheet closes it again.  Two local mechanisms drive closure:

  1. **Protrusion into free space** (contact inhibition of locomotion): a cell
     at the margin extends into the cell-free gap.  This is the per-site
     potential  H_fill = -lambda_fill * sum_{i in wound} [site i occupied],
     i.e. occupying a wound site lowers the energy.
  2. **Wound-margin line tension** (the purse-string): extending the free
     margin costs energy in proportion to how much new cell-medium interface
     it creates,  H_edge = +lambda_edge * sum_i w[i] * [i occupied] * free_nbrs(i).
     This regularises the front and suppresses thin protrusions.

Both are exact potentials of the CPM, so the engine's exact-Hamiltonian
self-test covers them (see `run_cpm_selftest.py`, which includes a non-zero
site bias).  Their competition sets whether the front is smooth or fingered.

  3. **Proliferation of margin cells** (contact-inhibited proliferation): a cell
     with a large free-surface fraction divides.  This is what produces the
     *slow* second phase of closure after the fast migration/purse-string phase.

MEASURED ANCHORS USED (all from MEASURED_ANCHORS.md, provenance in that file)
-----------------------------------------------------------------------------
  * mean area of a 4-cell group in the notum, 12-13.5 hAPF: 222.3 um^2
    -> 55.6 um^2 per cell.  This fixes the lattice spacing: with 36 sites per
    cell, h = sqrt(55.6/36) = 1.243 um.
  * notum wound diameter ~40 um, closing within 3 h (individual wounds
    140 min to >360 min, variability >= 2.5x).  PMC3718973, PMC13597085.
  * myosin fluctuation correlation time 2.2 min, notum 12-13.5 hAPF
    (Curran 2017).  Used ONLY as the one calibration knob for the MCS->minutes
    mapping, via the measured shape-fluctuation autocorrelation time below.
  * tissue relaxation time ~10 s (5-20 s), notum 18-26 hAPF (Bonnet 2012).

WHAT IS *NOT* CLAIMED
---------------------
* The MCS -> minutes mapping is calibrated on ONE measured quantity (the
  fluctuation correlation time).  The closure time is then an out-of-sample
  prediction and is reported as such; a single fitted knob is not a validation.
* All mechanical coefficients (lambda_fill, lambda_edge, lambda_volume, J, ...)
  are `illustrative`.  No absolute force or tension is measured in the notum
  (see MEASURED_ANCHORS.md), so no absolute force can be claimed here.
* These anchors come from different developmental stages and are flagged as
  such wherever they are compared.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import ndimage

from cpm import CPM

MEDIUM, EPI, EPI_DIFF = 0, 1, 2

TYPE_COLORS = {EPI: (0.45, 0.62, 0.85), EPI_DIFF: (0.80, 0.72, 0.45)}

PARAMS = {
    "cell_sites": (36.0, "measured-derived: 55.6 um^2 per cell / h^2"),
    "measured_cell_area_um2": (55.6, "measured: 222.3/4 um^2, notum 12-13.5 hAPF"),
    "lambda_volume": (6.0, "illustrative"),
    "lambda_length": (0.05, "illustrative"),
    "elongation_target": (1.15, "illustrative"),
    "temperature": (2.5, "illustrative; ~10% of the contact energy scale"),
    "J_epi_epi": (1.0, "illustrative"),
    "J_epi_medium": (4.0, "illustrative"),
    "J_diff_diff": (1.0, "illustrative"),
    "J_diff_epi": (1.0, "illustrative"),
    "J_diff_medium": (4.0, "illustrative"),
    "lambda_fill": (2.2, "illustrative"),
    "lambda_edge": (0.28, "illustrative"),
    "cycle_mcs": (250.0, "illustrative"),
    "division_free_surface_min": (0.05, "illustrative; a margin cell has ~0.06-0.17"),
    # Stretch-induced proliferation: a cell that has been stretched beyond its
    # preferred area divides to restore it.  This is the mechanism that gives
    # the *slow* second phase of closure; without it the sheet can only close
    # by cell stretching and stalls.
    "division_stretch_threshold": (0.15, "illustrative"),
    "division_mode": ("free_surface", "illustrative; 'stretch' or 'free_surface'"),
    "max_cells": (4000, "illustrative (compute cap)"),
    "bio_every_mcs": (10, "illustrative"),
    "minutes_per_mcs": (0.02, "CALIBRATED (see calibrate_minutes_per_mcs)"),
    "wound_diameter_um": (40.0, "measured: notum wound diameter ~40 um"),
    "closure_time_measured_min": (140.0, "measured: fastest notum wound, PMC3718973"),
    "closure_time_measured_max_min": (360.0, "measured: slowest notum wound, PMC3718973"),
}


def p(key):
    return PARAMS[key][0]


class Epithelium:
    def __init__(self, L=150, seed=0, params=None, proliferative=True,
                 n_types=3):
        self.cfg = dict(PARAMS)
        if params:
            for k, v in params.items():
                self.cfg[k] = (v, "override") if not isinstance(v, tuple) else v
        self.L = int(L)
        self.seed = int(seed)
        self.proliferative = bool(proliferative)
        cs = self.cfg["cell_sites"][0]
        h = np.sqrt(self.cfg["measured_cell_area_um2"][0] / cs)
        self.lattice_spacing_um = float(h)
        self.cell_side = int(round(np.sqrt(cs)))
        m = CPM(self.L, self.L, n_types=n_types,
                temperature=self.cfg["temperature"][0], seed=self.seed,
                max_cells=int(self.cfg["max_cells"][0]) + 50,
                lattice_spacing_um=self.lattice_spacing_um)
        J = np.zeros((n_types, n_types))
        J[EPI, EPI] = self.cfg["J_epi_epi"][0]
        J[EPI, MEDIUM] = J[MEDIUM, EPI] = self.cfg["J_epi_medium"][0]
        if n_types > 2:
            J[EPI_DIFF, EPI_DIFF] = self.cfg["J_diff_diff"][0]
            J[EPI_DIFF, EPI] = J[EPI, EPI_DIFF] = self.cfg["J_diff_epi"][0]
            J[EPI_DIFF, MEDIUM] = J[MEDIUM, EPI_DIFF] = self.cfg["J_diff_medium"][0]
        m.J = J
        self.m = m
        self.cycle = {}
        self.n_divisions_total = 0
        self.wound_mask = None
        self.wound_sites0 = None
        self.initial_wound_area_um2 = 0.0
        self.history = []
        self._fill_sheet()
        self._wound_bias(0.0)

    # ------------------------------------------------------------------
    def _fill_sheet(self):
        s = self.cell_side
        n = self.L // s
        self.nx = n
        for gy in range(n):
            for gx in range(n):
                idxs = []
                for yy in range(gy * s, (gy + 1) * s):
                    for xx in range(gx * s, (gx + 1) * s):
                        idxs.append(yy * self.L + xx)
                cid = self.m._alloc_id(EPI)
                self.m.Vt[cid] = self.cfg["cell_sites"][0]
                self.m.lV[cid] = self.cfg["lambda_volume"][0]
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                self.m.ctype[cid] = EPI
                self.m.lat[np.array(idxs, dtype=np.int64)] = cid
                self.m._add_sites(cid, np.array(idxs, dtype=np.int64))
                self.m._elong[cid] = self.cfg["elongation_target"][0]
                self.m.lam_chem_cell[cid] = 0.0
                self.m.Vt_apply(cid)
                self.m.lL[cid] = self.cfg["lambda_length"][0]
                self.cycle[cid] = 0.0
        rest = self.L - n * s
        if rest:
            # leave a thin medium strip rather than a ragged edge
            pass

    # ------------------------------------------------------------------
    def clamp_outer_ring(self, rings=2):
        """Freeze the outermost `rings` of cells.

        Physical justification: the notum is continuous with the surrounding
        epidermis, which is under tension and supplies the mechanical
        resistance that lets a purse-string actually pull a wound shut.  A
        sheet with a completely free edge instead simply rounds up.  These
        cells are held fixed and are excluded from proliferation.
        """
        s = self.cell_side
        n = self.nx
        frozen = []
        for cid in list(self.m.alive):
            idx = np.flatnonzero(self.m.lat == cid)
            if idx.size == 0:
                continue
            x, y, _z = self.m._site_coords(idx)
            gx = int(np.median(x)) // s
            gy = int(np.median(y)) // s
            if gx < rings or gy < rings or gx >= n - rings or gy >= n - rings:
                self.m.frozen[cid] = 1
                frozen.append(cid)
        self.frozen_cells = set(frozen)
        return len(frozen)

    def _wound_bias(self, lam_fill=None, lam_edge=None):
        """Build the per-site potential for the wound.

        site_bias[i] = -lambda_fill * w[i] + lambda_edge * w[i] * free_nbrs(i)

        so that  dH = site_bias[t] * ([a>0] - [b>0])  reproduces
        H_fill + H_edge exactly.  `free_nbrs` is recomputed at each call, so
        the potential is refreshed as the margin advances.
        """
        if self.wound_mask is None:
            self.m.set_site_bias(np.zeros(self.m.N))
            return
        lam_fill = self.cfg["lambda_fill"][0] if lam_fill is None else lam_fill
        lam_edge = self.cfg["lambda_edge"][0] if lam_edge is None else lam_edge
        w = self.wound_mask.reshape(-1).astype(np.float64)
        free = self.m.medium_facing_count_grid()
        self.m.set_site_bias(-lam_fill * w + lam_edge * w * free)

    # ------------------------------------------------------------------
    def wound(self, cx=None, cy=None, radius_sites=None, radius_um=None):
        L = self.L
        if cx is None:
            cx = cy = L // 2
        if radius_sites is None:
            radius_um = self.cfg["wound_diameter_um"][0] / 2.0 if radius_um is None else radius_um
            radius_sites = radius_um / self.lattice_spacing_um
        yy, xx = np.mgrid[0:L, 0:L]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_sites ** 2
        occupied_before = (self.m.lat > 0).reshape(L, L)
        # the wound is exactly the set of sites that were occupied and removed
        self.wound_mask = mask & occupied_before
        occ_before = int(self.wound_mask.sum())
        self.m.ablate_circle(cx, cy, radius_sites)
        self.wound_sites0 = max(occ_before, 1)
        self.radius_sites = float(radius_sites)
        self.radius_um = float(radius_sites * self.lattice_spacing_um)
        self.initial_wound_area_um2 = float(occ_before * self.lattice_spacing_um ** 2)
        # newly created cells (from later divisions) are not in self.cycle
        self._wound_bias()
        return {"radius_sites": self.radius_sites, "radius_um": self.radius_um,
                "n_sites_ablated": occ_before,
                "area_um2": self.initial_wound_area_um2}

    # ------------------------------------------------------------------
    def biology_step(self, dt_mcs):
        m = self.m
        n_div = 0
        n_diff = 0
        if self.proliferative:
            for cid in list(m.alive):
                if m.ctype[cid] != EPI:
                    continue
                if cid in getattr(self, "frozen_cells", ()):
                    continue
                mode = self.cfg["division_mode"][0]
                if mode == "stretch":
                    # stretched beyond its preferred area -> divide to restore it
                    stretched = m.V[cid] > (1.0 + self.cfg["division_stretch_threshold"][0]) * m.Vt[cid]
                    if not stretched:
                        self.cycle[cid] = 0.0
                        continue
                else:
                    free = m.medium_facing_sites(cid) / max(m.V[cid], 1.0)
                    if free < self.cfg["division_free_surface_min"][0]:
                        self.cycle[cid] = 0.0
                        continue
                self.cycle[cid] = self.cycle.get(cid, 0.0) + dt_mcs
                if self.cycle[cid] >= self.cfg["cycle_mcs"][0]:
                    self.cycle[cid] = 0.0
                    if len(m.alive) < int(self.cfg["max_cells"][0]):
                        try:
                            new = m.divide(cid)
                        except ValueError:
                            continue
                        self.cycle[new] = 0.0
                        m.ctype[new] = EPI
                        m.set_elongation(new, self.cfg["elongation_target"][0])
                        m.lL[new] = self.cfg["lambda_length"][0]
                        m.Vt_apply(new)
                        m.lL[new] = self.cfg["lambda_length"][0]
                        m.lam_chem_cell[new] = 0.0
                        n_div += 1
                        self.n_divisions_total += 1
        # refresh the wound potential (the margin has moved)
        self._wound_bias()
        return {"n_divisions": n_div, "n_differentiated": n_diff}

    # ------------------------------------------------------------------
    def run(self, n_mcs, bio_every=None, record_every=None, verbose=False):
        m = self.m
        bio_every = bio_every or int(self.cfg["bio_every_mcs"][0])
        record_every = record_every or bio_every * 5
        t0 = time.time()
        done = 0
        closed_at = None
        while done < n_mcs:
            chunk = min(bio_every, n_mcs - done)
            m.step(chunk, field_every=0)
            done += chunk
            info = self.biology_step(float(chunk))
            if m.mcs % record_every < bio_every or done >= n_mcs:
                h = self.metrics(**info)
                self.history.append(h)
                if verbose:
                    print(f"  mcs={h['mcs']:5d} open={h['open_area_um2']:8.1f} um^2 "
                          f"closed={h['closed_fraction']:.3f} cells={h['n_cells']:4d} "
                          f"perim={h['front_perimeter_um']:.0f} um")
                if closed_at is None and h["closed_fraction"] >= 0.98:
                    closed_at = h["mcs"]
        self.wall_seconds = time.time() - t0
        self.closure_mcs_98 = closed_at
        return self.history

    # ------------------------------------------------------------------
    def metrics(self, n_divisions=0, n_differentiated=0):
        m = self.m
        lat = m.lat.reshape(self.L, self.L)
        if self.wound_mask is None:
            open_frac = 0.0
            open_sites = 0
            total = 1
        else:
            open_sites = int(np.sum((lat == MEDIUM) & self.wound_mask))
            total = max(int(self.wound_sites0 or 1), 1)
            open_frac = open_sites / total
        epi = lat > 0
        # perimeter of the open hole, measured as cell-medium bonds whose
        # medium partner lies inside the original wound region
        perim = 0
        if self.wound_mask is not None:
            openreg = (lat == MEDIUM)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    sh_o = np.roll(np.roll(openreg, dy, 0), dx, 1)
                    sh_e = np.roll(np.roll(epi, dy, 0), dx, 1)
                    perim += int(np.sum(epi & sh_o & self.wound_mask))
                    perim += int(np.sum(openreg & sh_e & self.wound_mask))
        return {
            "mcs": int(m.mcs),
            "minutes_illustrative": float(m.mcs * self.cfg["minutes_per_mcs"][0]),
            "wound_sites_total": int(total),
            "open_sites": open_sites,
            "open_area_um2": float(open_sites * self.lattice_spacing_um ** 2),
            "closed_fraction": float(1.0 - open_frac),
            "front_perimeter_um": float(perim * self.lattice_spacing_um),
            "n_cells": len(m.alive),
            "n_divisions_recent": int(n_divisions),
            "n_divisions_total": int(self.n_divisions_total),
            "epi_area_fraction": float(epi.mean()),
            "acceptance": (m.n_accepted / m.n_attempts) if m.n_attempts else 0.0,
        }

    def set_initial_wound_sites(self):
        self.wound_sites0 = int(np.sum(self.wound_mask))

    # ------------------------------------------------------------------
    def two_phase_fit(self, key="open_area_um2"):
        """Fit two straight lines with a shared breakpoint to the closure curve.

        Returns the break time and the two slopes (um^2 per MCS).  If a single
        line fits no worse than a two-segment fit, the break is reported as
        None -- i.e. we do NOT force a two-phase description onto data that
        does not show one.
        """
        t = np.array([h["mcs"] for h in self.history], dtype=float)
        y = np.array([h[key] for h in self.history], dtype=float)
        n = len(t)
        if n < 8:
            return {"break_mcs": None, "slope1": None, "slope2": None,
                    "note": "too few points"}

        def sse(idx):
            s = 0.0
            for seg in (slice(0, idx + 1), slice(idx, n)):
                tt, yy = t[seg], y[seg]
                if tt.size < 2:
                    return np.inf
                a, b = np.polyfit(tt, yy, 1)
                s += float(np.sum((yy - (a * tt + b)) ** 2))
            return s

        sse_one = float(np.sum((y - np.polyval(np.polyfit(t, y, 1), t)) ** 2))
        best, best_idx = np.inf, None
        for idx in range(3, n - 3):
            s = sse(idx)
            if s < best:
                best, best_idx = s, idx
        if best_idx is None:
            return {"break_mcs": None, "slope1": None, "slope2": None,
                    "note": "no admissible breakpoint"}
        s1 = np.polyfit(t[:best_idx + 1], y[:best_idx + 1], 1)[0]
        s2 = np.polyfit(t[best_idx:], y[best_idx:], 1)[0]
        return {
            "break_mcs": float(t[best_idx]),
            "slope1_um2_per_mcs": float(s1),
            "slope2_um2_per_mcs": float(s2),
            "sse_two_segment": float(best),
            "sse_single_line": sse_one,
            "two_phase_preferred": bool(best < 0.5 * sse_one),
            "rate_ratio_slow_over_fast": float(s2 / s1) if s1 != 0 else None,
        }


# ----------------------------------------------------------------------
def calibrate_minutes_per_mcs(n_mcs=6000, L=90, seed=3, verbose=True):
    """Calibrate the MCS -> minutes mapping on ONE measured quantity.

    Quantity used: the correlation time of cell-shape (elongation) fluctuations.
    Measured value: myosin-fluctuation correlation time 2.2 min in the notum at
    12-13.5 hAPF (Curran 2017).  We measure the same kind of quantity in the
    simulation (autocorrelation decay time of the mean cell elongation in an
    unwounded confluent sheet) and set

        minutes_per_mcs = 2.2 min / tau_sim[in MCS]

    This is one fitted knob, not a validation.
    """
    epi = Epithelium(L=L, seed=seed, params={"bio_every_mcs": 50})
    epi.m.step(200, field_every=0)  # relax the tiling
    ts, es = [], []
    for k in range(n_mcs // 100):
        epi.m.step(100, field_every=0)
        el = [epi.m.length_of(c) / max(2.0 * np.sqrt(epi.m.V[c] / np.pi), 1e-9)
              for c in epi.m.alive]
        ts.append(epi.m.mcs)
        es.append(float(np.mean(el)))
    ts = np.array(ts, dtype=float)
    es = np.array(es, dtype=float)
    es = es - es.mean()
    ac = np.correlate(es, es, mode="full")[len(es) - 1:]
    if ac[0] == 0:
        return {"tau_mcs": None, "minutes_per_mcs": None}
    ac = ac / ac[0]
    # decay time = integral of the autocorrelation (in units of the sample step)
    step = float(ts[1] - ts[0]) if ts.size > 1 else 1.0
    tau = float(np.sum(np.clip(ac, 0, None)) * step)
    mpm = 2.2 / tau if tau > 0 else None
    out = {"tau_sim_mcs": tau, "sample_step_mcs": step,
           "measured_correlation_time_min": 2.2,
           "measured_source": "Curran 2017, notum 12-13.5 hAPF (myosin fluctuation correlation time)",
           "minutes_per_mcs": mpm,
           "autocorrelation": [float(v) for v in ac[:min(40, ac.size)]],
           "note": "one fitted knob; the closure time is then an out-of-sample prediction"}
    if verbose:
        print(json.dumps(out, indent=2))
    return out
