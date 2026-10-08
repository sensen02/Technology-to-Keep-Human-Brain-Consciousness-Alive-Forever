"""vasculogenesis.py -- P1/P2: spontaneous blood-vessel network formation.

MECHANISM (this is the point of the module)
-------------------------------------------
The network is NOT drawn.  It is produced by three purely local rules, which
together are the published "contact-inhibited chemotaxis" mechanism for de novo
vasculogenesis and sprouting angiogenesis:

    Merks RMH, Perryn ED, Shirinifard A, Glazier JA (2008)
    "Contact-Inhibited Chemotaxis in De Novo and Sprouting Blood-Vessel Growth"
    PLoS Computational Biology 4(9): e1000163
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000163

  1. each endothelial cell (EC) secretes a chemoattractant (VEGF),
  2. each EC migrates up the VEGF gradient,
  3. BUT migration is contact-inhibited: the chemotactic force acts only on
     membrane that faces free space, never on membrane in contact with another
     cell.  A cell that is hemmed in by neighbours therefore cannot pull itself
     towards them.

Rule 3 is what turns a clumping instability into a *branching network*: cells
cannot pile up, so growth is forced along free edges, which produces cords and
loops.  That prediction is falsifiable and we test it here by running the same
simulation with contact inhibition switched OFF and comparing the network
descriptors (see `contact_inhibition_control`).

TIP / STALK SELECTION (P2)
--------------------------
ECs are not all equivalent.  Tip cells lead sprouts (high motility, no
proliferation) and stalk cells follow (low motility, proliferate).  The
selection is by Delta-Notch lateral inhibition, implemented here as an explicit
relaxation on the measured contact graph:

    N_i = mean_{j in EC-neighbours(i)} D_j          (Notch signal received)
    D_i = Dmax * f(VEGF_i) / (1 + (N_i/kN)^hill)    (Delta produced)
    tip  <=>  D_i >= max over EC-neighbours of D_j  and  D_i > D_tip

The lateral inhibition is inherently contact-based (Notch is a membrane
receptor), so it needs the contact graph -- which is exactly what the lattice
gives for free.

HYPOXIA COUPLING (P2b)
----------------------
O2 is a second diffusing field, supplied from the domain boundary and consumed
by every cell.  VEGF secretion is amplified where pO2 is low, so a hypoxic core
recruits vessels.  The test is that tip-cell density is higher where pO2 is
lower.

UNITS AND PROVENANCE -- READ THIS
---------------------------------
The primary units are lattice sites and Monte Carlo Steps (MCS).  Nothing in
this module is calibrated to a measured rate: every kinetic parameter is
`illustrative`, and each is labelled as such in `PARAMS`.  The optional mapping
`minutes_per_mcs` is illustrative and is reported only for readability.  The
lattice spacing of 2 um is chosen so that a cell of 25 sites is ~11 um across,
which is the right order for an endothelial cell; that is an order-of-magnitude
choice, not a measurement.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import ndimage

from cpm import CPM

MEDIUM, EC, PERICYTE, FIBROBLAST = 0, 1, 2, 3

TYPE_NAMES = {EC: "EC", PERICYTE: "pericyte", FIBROBLAST: "fibroblast"}

# Colours used by every figure, so panels are comparable across the report.
TYPE_COLORS = {
    EC: (0.55, 0.70, 0.92),
    PERICYTE: (0.30, 0.70, 0.35),
    FIBROBLAST: (0.55, 0.45, 0.75),
}
STATE_COLORS = {
    "tip": (0.75, 0.08, 0.10),
    "stalk": (0.55, 0.70, 0.92),
}

# Every entry: (value, provenance).  "illustrative" means chosen by us and NOT
# measured; "measured" would require a citation, and there is none here.
PARAMS = {
    # lattice mechanics
    "lattice_spacing_um": (2.0, "illustrative (chosen so 25 sites ~ 11 um)"),
    "target_volume_sites": (25.0, "illustrative"),
    "lambda_volume": (10.0, "illustrative"),
    "lambda_length": (0.30, "illustrative"),
    "elongation_target": (1.8, "illustrative"),
    "temperature": (6.0, "illustrative"),
    # contact energies J(type_a, type_b); lower = more adhesive
    "J_ec_ec": (1.0, "illustrative"),
    "J_ec_medium": (6.0, "illustrative"),
    "J_peri_peri": (1.0, "illustrative"),
    "J_peri_ec": (0.5, "illustrative"),
    "J_peri_medium": (6.0, "illustrative"),
    "J_fib_fib": (1.0, "illustrative"),
    "J_fib_ec": (4.0, "illustrative"),
    "J_fib_medium": (6.0, "illustrative"),
    # fields
    "vegf_diffusion": (0.55, "illustrative; decay length sqrt(D/k)=30 sites"),
    "vegf_secretion_ec": (0.004, "illustrative; scaled so the field is order 1"),
    "vegf_secretion_fib": (0.010, "illustrative; fibroblasts as a VEGF source"),
    "vegf_ec_uptake": (0.0, "illustrative; EC consumption of VEGF sharpens gradients"),
    "vegf_decay": (0.006, "illustrative; decay length ~9.6 sites ~2 cell diameters"),
    "tip_fraction_percentile": (75.0, "illustrative; set RELATIVELY because the absolute Delta threshold is not measurable"),
    "vegf_hypoxic_amplification": (2.0, "illustrative"),
    "o2_diffusion": (0.9, "illustrative"),
    "o2_consumption": (0.012, "illustrative"),
    "o2_boundary_value": (1.0, "illustrative"),
    "o2_hypoxia_threshold": (0.35, "illustrative"),
    # chemotaxis
    "lambda_chem_tip": (30.0, "illustrative; comparable to T over one cell diameter"),
    "lambda_chem_stalk": (2.0, "illustrative"),
    "lambda_chem_pericyte": (2.5, "illustrative"),
    # Delta-Notch
    "delta_max": (1.0, "illustrative"),
    "notch_k": (0.6, "illustrative"),
    "notch_hill": (2.0, "illustrative"),
    "vegf_half_saturation": (0.5, "illustrative"),
    "delta_tip_threshold": (0.55, "illustrative"),
    "notch_relaxation_iters": (8, "illustrative"),
    # proliferation
    "cycle_mcs": (240.0, "illustrative"),
    "division_vegf_threshold": (0.15, "illustrative"),
    "division_free_surface_min": (0.25, "illustrative"),
    "max_cells": (1400, "illustrative (compute cap, not biology)"),
    # timing
    "minutes_per_mcs": (0.05, "illustrative"),
    "bio_every_mcs": (5, "illustrative"),
    "pericyte_maturation_contacts": (4, "illustrative"),
}


def p(key):
    return PARAMS[key][0]


class Vasculogenesis:
    def __init__(self, L=220, seed=0, n_ec=300, n_pericyte=0, n_fibroblast=25,
                 contact_inhibited=True, hypoxia_coupling=True,
                 params=None, jitter=0.0, mode="de_novo"):
        self.cfg = dict(PARAMS)
        if params:
            for k, v in params.items():
                self.cfg[k] = (v, "override") if not isinstance(v, tuple) else v
        self.contact_inhibited = bool(contact_inhibited)
        self.hypoxia_coupling = bool(hypoxia_coupling)
        self.L = int(L)
        self.seed = int(seed)
        self.mode = mode

        m = CPM(self.L, self.L, n_types=4, temperature=self.cfg["temperature"][0],
                seed=self.seed, max_cells=int(self.cfg["max_cells"][0]) + 50,
                lattice_spacing_um=self.cfg["lattice_spacing_um"][0])
        J = np.zeros((4, 4))
        J[EC, EC] = self.cfg["J_ec_ec"][0]
        J[EC, MEDIUM] = J[MEDIUM, EC] = self.cfg["J_ec_medium"][0]
        J[PERICYTE, PERICYTE] = self.cfg["J_peri_peri"][0]
        J[PERICYTE, EC] = J[EC, PERICYTE] = self.cfg["J_peri_ec"][0]
        J[PERICYTE, MEDIUM] = J[MEDIUM, PERICYTE] = self.cfg["J_peri_medium"][0]
        J[FIBROBLAST, FIBROBLAST] = self.cfg["J_fib_fib"][0]
        J[FIBROBLAST, EC] = J[EC, FIBROBLAST] = self.cfg["J_fib_ec"][0]
        J[FIBROBLAST, MEDIUM] = J[MEDIUM, FIBROBLAST] = self.cfg["J_fib_medium"][0]
        J[FIBROBLAST, PERICYTE] = J[PERICYTE, FIBROBLAST] = self.cfg["J_fib_ec"][0]
        m.J = J
        self.m = m

        self.vegf = m.add_field("VEGF", D=self.cfg["vegf_diffusion"][0],
                                decay=self.cfg["vegf_decay"][0], dt=0.25)
        ring = np.zeros((self.L, self.L), dtype=bool)
        w = 2
        ring[:w, :] = ring[-w:, :] = ring[:, :w] = ring[:, -w:] = True
        self.o2 = m.add_field("O2", D=self.cfg["o2_diffusion"][0], decay=0.0,
                              dt=0.25, initial=self.cfg["o2_boundary_value"][0],
                              source_mask=ring,
                              source_value=self.cfg["o2_boundary_value"][0],
                              uptake={EC: self.cfg["o2_consumption"][0],
                                      PERICYTE: self.cfg["o2_consumption"][0],
                                      FIBROBLAST: self.cfg["o2_consumption"][0]})
        self.vegf.uptake = {EC: self.cfg["vegf_ec_uptake"][0]}
        m.set_chemotaxis("VEGF", lam=self.cfg["lambda_chem_stalk"][0],
                         contact_inhibited=self.contact_inhibited, enabled=True)

        # start the chemoattractant near its steady state instead of from zero:
        # the decay time is ~1/decay MCS, so starting from zero would leave the
        # field un-equilibrated for thousands of MCS
        ec_frac = sum(1 for c in m.alive if m.ctype[c] == EC) * self.cfg["target_volume_sites"][0] / (self.L * self.L)
        c_ss = self.cfg["vegf_secretion_ec"][0] * ec_frac / max(self.cfg["vegf_decay"][0], 1e-12)
        self.vegf.value[:] = c_ss
        self.vegf_scale = float(c_ss)

        self.state = {}
        self.cycle = {}
        self.contact_run = {}
        self.birth_mcs = {}
        self.parent = {}
        self.lineage_id = {}
        self._next_lineage = 1
        self.history = []

        if self.mode == "sprouting":
            self._seed_sprouting(n_parenchyma=n_fibroblast)
        else:
            self._seed_cells(n_ec, n_pericyte, n_fibroblast, jitter=jitter)

    # ------------------------------------------------------------------
    def _seed_sprouting(self, n_parenchyma=25, dens=0.55, vessel_frac=0.55):
        """Sprouting angiogenesis.

        A single parent vessel runs vertically near the left edge.  It is the
        ONLY O2 source, so the tissue becomes progressively hypoxic away from
        it.  The static parenchymal cells secrete VEGF in proportion to their
        hypoxia, so the VEGF field points away from the vessel.  Endothelial
        cells then sprout off the vessel, up the VEGF gradient, with
        contact-inhibited chemotaxis and Delta-Notch tip/stalk selection.

        This is the standard sprouting-angiogenesis geometry and it is the
        regime in which the hypoxia -> VEGF -> sprout coupling is meaningful.
        """
        m = self.m
        rng = np.random.default_rng(self.seed)
        margin = 6
        Vt = self.cfg["target_volume_sites"][0]

        # --- parenchyma: static, O2-consuming, VEGF-secreting cells ---
        s = 6
        for gy in range(margin, self.L - margin, s):
            for gx in range(int(self.L * 0.30), self.L - margin, s):
                if rng.random() > dens:
                    continue
                x = gx + int(rng.integers(0, 3))
                y = gy + int(rng.integers(0, 3))
                if m.lat[y * self.L + x] != 0:
                    continue
                try:
                    cid = m.add_cell(FIBROBLAST, Vt, self.cfg["lambda_volume"][0],
                                     x, y, radius=3.0)
                except ValueError:
                    continue
                m.set_elongation(cid, self.cfg["elongation_target"][0])
                m.lL[cid] = self.cfg["lambda_length"][0]
                m.Vt_apply(cid)
                m.frozen[cid] = 1
                m.set_cell_chemotaxis(cid, 0.0)

        # --- parent vessel: a vertical cord of ECs ---
        vx = int(self.L * 0.16)
        y0 = int(self.L * (1 - vessel_frac) / 2)
        y1 = self.L - y0
        vessel_mask = np.zeros((self.L, self.L), dtype=bool)
        for y in range(y0, y1, 4):
            for dx in (-2, -1, 0, 1):
                x = vx + dx
                if m.lat[y * self.L + x] != 0:
                    continue
                cid = m.add_cell(EC, Vt, self.cfg["lambda_volume"][0], x, y, radius=2.6)
                m.set_elongation(cid, 2.2)
                m.lL[cid] = self.cfg["lambda_length"][0]
                m.Vt_apply(cid)
                vessel_mask[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
        # widen the O2 source slightly around the vessel
        self.o2.source_mask = vessel_mask
        self.o2.source_value = self.cfg["o2_boundary_value"][0]
        self.o2.value[vessel_mask] = self.cfg["o2_boundary_value"][0]
        self.vessel_mask = vessel_mask
        self.vessel_x = float(vx)

        for cid in list(m.alive):
            self.state[cid] = "stalk"
            self.cycle[cid] = 0.0
            self.contact_run[cid] = 0
            self.birth_mcs[cid] = 0
            self.parent[cid] = 0
            self.lineage_id[cid] = self._next_lineage
            self._next_lineage += 1
            t = int(m.ctype[cid])
            if t == EC:
                m.set_cell_chemotaxis(cid, self.cfg["lambda_chem_stalk"][0])

    def _seed_cells(self, n_ec, n_pericyte, n_fibroblast, jitter=0.0):
        m = self.m
        rng = np.random.default_rng(self.seed)
        margin = 12
        r = 3.0

        def place(n, type_id, radius=r):
            made = 0
            tries = 0
            while made < n and tries < 200 * n + 1000:
                tries += 1
                x = int(rng.integers(margin, self.L - margin))
                y = int(rng.integers(margin, self.L - margin))
                if m.lat[y * self.L + x] != 0:
                    continue
                try:
                    cid = m.add_cell(type_id, self.cfg["target_volume_sites"][0],
                                     self.cfg["lambda_volume"][0], x, y, radius=radius)
                except ValueError:
                    continue
                m.set_elongation(cid, self.cfg["elongation_target"][0])
                m.lL[cid] = self.cfg["lambda_length"][0]
                m.Vt_apply(cid)
                made += 1
            return made

        # ECs: dispersed, minimum spacing enforced by the occupancy check
        place(n_ec, EC)
        place(n_pericyte, PERICYTE)
        # fibroblasts: static (frozen) sources of VEGF / matrix
        made = place(n_fibroblast, FIBROBLAST)
        for cid in list(m.alive):
            if m.ctype[cid] == FIBROBLAST:
                m.frozen[cid] = 1
                m.set_cell_chemotaxis(cid, 0.0)

        for cid in list(m.alive):
            self.state[cid] = "stalk"
            self.cycle[cid] = 0.0
            self.contact_run[cid] = 0
            self.birth_mcs[cid] = 0
            self.parent[cid] = 0
            self.lineage_id[cid] = self._next_lineage
            self._next_lineage += 1
            t = m.ctype[cid]
            if t == EC:
                m.set_cell_chemotaxis(cid, self.cfg["lambda_chem_stalk"][0])
            elif t == PERICYTE:
                m.set_cell_chemotaxis(cid, self.cfg["lambda_chem_pericyte"][0])

    # ------------------------------------------------------------------
    def ec_ids(self):
        m = self.m
        return [c for c in m.alive if m.ctype[c] == EC]

    def cell_field_mean(self, field, ids=None):
        """Mean of a lattice field over each cell's own sites (bincount).

        Returns {cell_id: mean}.  O(N) for the whole population, not O(N) per
        cell, because the lattice gather is done once with np.bincount.
        """
        m = self.m
        flat_field = field.value.reshape(-1) if hasattr(field, "value") else field.reshape(-1)
        tot = np.bincount(m.lat, weights=flat_field, minlength=m.max_cells + 1)
        cnt = np.bincount(m.lat, minlength=m.max_cells + 1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1.0), 0.0)
        ids = m.alive if ids is None else ids
        return {int(c): float(mean[int(c)]) for c in ids}

    def local_vegf(self, ids=None):
        """Mean VEGF over each cell's own sites."""
        return self.cell_field_mean(self.vegf, ids)

    # ------------------------------------------------------------------
    def biology_step(self, dt_mcs):
        """Differentiation, Delta-Notch lateral inhibition, proliferation."""
        m = self.m
        self._secrete(dt_mcs)

        pairs = m.contact_pairs()
        nbr = {}
        for (a, b) in pairs:
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)

        ecs = self.ec_ids()
        ec_set = set(ecs)
        vegf_local = self.local_vegf()
        # The Delta-Notch drive is made scale-free: the half-saturation point is
        # tied to the current population-level VEGF magnitude rather than to an
        # absolute number, because the absolute field level is set by an
        # illustrative secretion/decay pair and carries no units.
        vals = np.array([v for v in vegf_local.values() if v > 0])
        self.vegf_scale = float(np.percentile(vals, 90)) if vals.size else 1.0
        vhalf = max(self.vegf_scale, 1e-12)

        # ---- Delta-Notch relaxation on the contact graph ----
        D = {}
        for c in ecs:
            v = vegf_local.get(c, 0.0)
            D[c] = self.cfg["delta_max"][0] * (v / (v + vhalf) if v > 0 else 0.0)
        kN = self.cfg["notch_k"][0]
        hill = self.cfg["notch_hill"][0]
        for _ in range(int(self.cfg["notch_relaxation_iters"][0])):
            Dn = {}
            for c in ecs:
                nb = [j for j in nbr.get(c, ()) if j in ec_set]
                N = float(np.mean([D[j] for j in nb])) if nb else 0.0
                v = vegf_local.get(c, 0.0)
                drive = v / (v + vhalf) if v > 0 else 0.0
                Dn[c] = self.cfg["delta_max"][0] * drive / (1.0 + (N / kN) ** hill)
            D = Dn

        # The absolute Delta threshold is not measurable, so the tip cut is set
        # RELATIVE to the population (upper quartile by default).  This fixes the
        # tip *fraction* by construction; what the model then predicts is the
        # spatially patterned *placement* of tips, not their number.
        dvals = np.array([D[c] for c in ecs]) if ecs else np.zeros(0)
        d_rel = float(np.percentile(dvals, self.cfg["tip_fraction_percentile"][0])) if dvals.size else np.inf
        n_tip = 0
        for c in ecs:
            nb = [j for j in nbr.get(c, ()) if j in ec_set]
            best = max([D[j] for j in nb], default=-1.0)
            is_tip = (D[c] >= d_rel) and (D[c] >= best - 1e-12)
            if is_tip:
                self.state[c] = "tip"
                m.set_cell_chemotaxis(c, self.cfg["lambda_chem_tip"][0])
                n_tip += 1
            else:
                self.state[c] = "stalk"
                m.set_cell_chemotaxis(c, self.cfg["lambda_chem_stalk"][0])

        # ---- pericyte maturation: prolonged EC contact switches off motility ----
        for cid in list(m.alive):
            if m.ctype[cid] != PERICYTE:
                continue
            has_ec = any(m.ctype[j] == EC for j in nbr.get(cid, ()))
            if has_ec:
                self.contact_run[cid] = self.contact_run.get(cid, 0) + 1
            else:
                self.contact_run[cid] = 0
            if self.contact_run[cid] >= self.cfg["pericyte_maturation_contacts"][0]:
                self.state[cid] = "mature"
                m.set_cell_chemotaxis(cid, 0.0)
            else:
                self.state[cid] = "precursor"

        # ---- proliferation of stalk ECs ----
        n_divisions = 0
        for c in ecs:
            if self.state.get(c) == "tip":
                continue
            v = vegf_local.get(c, 0.0)
            if v < self.cfg["division_vegf_threshold"][0]:
                continue
            free = m.medium_facing_sites(c) / max(m.V[c], 1.0)
            if free < self.cfg["division_free_surface_min"][0]:
                continue
            self.cycle[c] = self.cycle.get(c, 0.0) + dt_mcs
            if self.cycle[c] >= self.cfg["cycle_mcs"][0]:
                if len(m.alive) < int(self.cfg["max_cells"][0]):
                    try:
                        new = m.divide(c)
                    except ValueError:
                        self.cycle[c] = 0.0
                        continue
                    m.lL[new] = self.cfg["lambda_length"][0]
                    m.Vt_apply(new)
                    m.set_elongation(new, self.cfg["elongation_target"][0])
                    m.lL[new] = self.cfg["lambda_length"][0]
                    m.Vt_apply(new)
                    self.state[new] = "stalk"
                    self.cycle[new] = 0.0
                    self.cycle[c] = 0.0
                    self.contact_run[new] = 0
                    self.birth_mcs[new] = m.mcs
                    self.parent[new] = c
                    self.lineage_id[new] = self._next_lineage
                    self._next_lineage += 1
                    m.set_cell_chemotaxis(new, self.cfg["lambda_chem_stalk"][0])
                    n_divisions += 1
        return {"n_tip": n_tip, "n_divisions": n_divisions}

    # ------------------------------------------------------------------
    def _secrete(self, dt_mcs):
        """VEGF secretion, amplified where pO2 is low (hypoxia coupling)."""
        m = self.m
        st = m.site_type_grid()
        ec_mask = st == EC
        fib_mask = st == FIBROBLAST
        rate = (self.cfg["vegf_secretion_ec"][0] * ec_mask.astype(np.float64)
                + self.cfg["vegf_secretion_fib"][0] * fib_mask.astype(np.float64))
        if self.hypoxia_coupling:
            thr = self.cfg["o2_hypoxia_threshold"][0]
            hypoxia = np.clip((thr - self.o2.value) / thr, 0.0, 1.0)
            rate = rate * (1.0 + self.cfg["vegf_hypoxic_amplification"][0] * hypoxia)
        self.vegf.value += rate * dt_mcs

    # ------------------------------------------------------------------
    def run(self, n_mcs, bio_every=None, record_every=None, callback=None,
            verbose=False):
        m = self.m
        bio_every = bio_every or int(self.cfg["bio_every_mcs"][0])
        record_every = record_every or bio_every * 4
        t0 = time.time()
        done = 0
        while done < n_mcs:
            chunk = min(bio_every, n_mcs - done)
            m.step(chunk, field_every=1)
            done += chunk
            info = self.biology_step(float(chunk))
            if m.mcs % record_every < bio_every or done >= n_mcs:
                self.history.append(self.metrics(**info))
                if verbose:
                    h = self.history[-1]
                    print(f"  mcs={h['mcs']:5d} cells={h['n_cells']:4d} "
                          f"tip={h['n_tip']:3d} comp={h['ec_components']:3d} "
                          f"loops={h['ec_loops']:3d} occ={h['ec_area_fraction']:.3f} "
                          f"o2min={h['o2_min']:.3f} vegf={h['vegf_mean']:.4f}")
            if callback is not None:
                callback(self, m.mcs)
        self.wall_seconds = time.time() - t0
        return self.history

    # ------------------------------------------------------------------
    def metrics(self, n_tip=0, n_divisions=0):
        m = self.m
        lat = m.lat.reshape(self.L, self.L)
        st = m.ctype[lat]
        ec_mask = st == EC
        counts = {TYPE_NAMES[t]: int((m.ctype[list(m.alive)] == t).sum()) for t in TYPE_NAMES}

        # network descriptors from the EC mask
        lbl, n_comp = ndimage.label(ec_mask, structure=np.ones((3, 3)))
        # loops = enclosed medium regions (a mature vascular network has loops)
        mid, n_medium = ndimage.label(~ec_mask, structure=np.ones((3, 3)))
        border = set(np.unique(np.concatenate([mid[0, :], mid[-1, :], mid[:, 0], mid[:, -1]])))
        loops = int(sum(1 for k in range(1, n_medium + 1) if k not in border))

        ecs = self.ec_ids()
        elongs = [m.length_of(c) / max(2.0 * np.sqrt(m.V[c] / np.pi), 1e-9) for c in ecs]
        tips = [c for c in ecs if self.state.get(c) == "tip"]

        # per-cell O2 (a cell reads the O2 at its own sites)
        po2 = self.cell_field_mean(self.o2, ecs)
        tip_po2 = [po2[c] for c in tips]
        stalk_po2 = [po2[c] for c in ecs if self.state.get(c) == "stalk"]

        # --- ramification descriptors: a sprouting tree reaches far from its
        # parent vessel and has a high perimeter/area ratio; a clump does not.
        ec_sites = np.flatnonzero(ec_mask.reshape(-1))
        if ec_sites.size:
            xs = ec_sites % self.L
            ys = ec_sites // self.L
            reach = float(xs.max() - (self.vessel_x if hasattr(self, "vessel_x") else 0))
            gyre = float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))
        else:
            reach, gyre = 0.0, 0.0
        perim = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sh = np.roll(np.roll(ec_mask, dy, 0), dx, 1)
                perim += int(np.sum(ec_mask & ~sh))
        p_over_a = float(perim / max(int(ec_mask.sum()), 1))

        return {
            "mcs": int(m.mcs),
            "ec_max_reach_sites": reach,
            "ec_max_reach_um": float(reach * self.m.lattice_spacing_um),
            "ec_gyration_radius_sites": gyre,
            "ec_perimeter_over_area": p_over_a,
            "minutes_illustrative": float(m.mcs * self.cfg["minutes_per_mcs"][0]),
            "n_cells": len(m.alive),
            "counts": counts,
            "n_ec": len(ecs),
            "n_tip": len(tips),
            "n_divisions_recent": int(n_divisions),
            "ec_area_fraction": float(ec_mask.mean()),
            "ec_components": int(n_comp),
            "ec_loops": loops,
            "ec_mean_elongation": float(np.mean(elongs)) if elongs else 0.0,
            "vegf_mean": float(self.vegf.value.mean()),
            "vegf_max": float(self.vegf.value.max()),
            "o2_min": float(self.o2.value.min()),
            "o2_mean": float(self.o2.value.mean()),
            "hypoxic_area_fraction": float(
                (self.o2.value < 0.35 * self.cfg["o2_hypoxia_threshold"][0]).mean()),
            "mean_po2_tip": float(np.mean(tip_po2)) if tip_po2 else None,
            "mean_po2_stalk": float(np.mean(stalk_po2)) if stalk_po2 else None,
            "acceptance": (m.n_accepted / m.n_attempts) if m.n_attempts else 0.0,
            "lineage_max_id": int(self._next_lineage - 1),
            "lineage_unique": bool(len(set(self.lineage_id.values())) == len(self.lineage_id)),
        }

    # ------------------------------------------------------------------
    def lineage_check(self):
        """Parents must exist, be older, and the parent graph must be acyclic."""
        problems = []
        for cid, par in self.parent.items():
            if par == 0:
                continue
            if par not in self.parent:
                problems.append(f"cell {cid} has missing parent {par}")
            if self.birth_mcs.get(par, 1e18) > self.birth_mcs.get(cid, -1):
                problems.append(f"cell {cid} born before its parent {par}")
        # cycle detection
        colour = {}

        def visit(c):
            st = colour.get(c, 0)
            if st == 1:
                return False
            if st == 2:
                return True
            colour[c] = 1
            par = self.parent.get(c, 0)
            ok = True if par == 0 else visit(par)
            colour[c] = 2
            return ok

        for cid in list(self.parent):
            if not visit(cid):
                problems.append(f"cycle in lineage through {cid}")
                break
        ids = list(self.lineage_id.values())
        if len(set(ids)) != len(ids):
            problems.append("duplicate lineage ids")
        return {"ok": not problems, "problems": problems,
                "n_cells_tracked": len(self.parent),
                "n_founders": sum(1 for v in self.parent.values() if v == 0)}

    # ------------------------------------------------------------------
    def type_image(self):
        from viz import type_image
        state_of = {"tip": [c for c, s in self.state.items()
                            if s == "tip" and self.m.ctype[c] == EC]}
        return type_image(self.m, TYPE_COLORS, state_of=state_of,
                          state_colors=STATE_COLORS)
