"""engine.project_layers -- wrap the project's existing modules as engine layers.

The engine was built before most of these modules, so only a few of them were
ever wired into it.  This module closes that gap: every existing simulation
module becomes a Layer with declared ports and units, so it can be composed,
checked and recorded like everything else.

Layers defined here
-------------------
TrachealGasLayer   NEW and insect-correct.  Gas exchange for a fly.
CellStateLayer     cellstate.py    -- per-cell volume / ATP / integrity
GrowthLayer        growth.py       -- colony growth and division
HormoneLayer       hormone.py      -- receptor binding + delayed nuclear effect
MechanicsLayer     cpm.py          -- 2D self-organising tissue mechanics

THE TRACHEAL POINT
------------------
`MEASURED_ANCHORS.md` and `ORGANISM_SCALE_PLAN.md` both record the same
correction: insects do NOT have capillary beds, they breathe through tracheae
and tracheoles, and applying a mammalian Krogh cylinder model to a fly is a
CATEGORY ERROR.  The records also carry the only verified geometry available,
which is from LOCUST, not Drosophila (tracheole radius 0.5 um, muscle cylinder
radius 6.5 um, spacing ~13 um, i.e. ~150x denser than mammalian capillaries).

The transport mathematics (diffusion into a cylinder from a source) is shared
with the Krogh model, so it can be reused -- but with tracheal geometry and
parameters, and the cross-species caveat attached.  That is what
`TrachealGasLayer` does, and the engine's profile guard makes sure it cannot be
attached to an organism that does not breathe this way.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer
from .units import Quantity

__all__ = ["TrachealGasLayer", "CellStateLayer", "GrowthLayer", "HormoneLayer",
           "MechanicsLayer", "TRACHEAL_GEOMETRY"]

# The only tracheal geometry this project has verified.  It is LOCUST data; the
# records explicitly state Drosophila is not in that table.
TRACHEAL_GEOMETRY = {
    "tracheole_radius_um": (0.5, "measured (LOCUST, not Drosophila)"),
    "muscle_cylinder_radius_um": (6.5, "measured (LOCUST, not Drosophila)"),
    "spacing_um": (13.0, "measured (LOCUST, not Drosophila)"),
    "density_relative_to_mammalian_capillaries": (150.0,
        "derived: the records state tracheoles are ~150x denser than "
        "mammalian capillaries"),
}


class TrachealGasLayer(Layer):
    """O2 delivery by the tracheal system, not by blood vessels.

    The fly's tracheae branch down to tracheoles that end inside individual
    cells, so the transport problem is diffusive delivery from a dense,
    close-packed tubular network with no convective blood carrier.  We use the
    same steady-cylinder diffusion solution as the Krogh model -- the
    mathematics is shared -- with tracheal geometry.
    """

    name = "tracheal_gas"
    requires = {"spiracle_o2": "1"}       # fractional O2 at the spiracle (air)
    provides = {"cell_o2": "1"}           # fractional O2 available to each cell

    def __init__(self, cell_positions_um, metabolism, spacing_um=None,
                 tracheole_radius_um=None, d_eff_um2_s=2000.0, n_cells=None):
        super().__init__()
        self.target_dt = Quantity(1.0, "s")
        self.pos = np.asarray(cell_positions_um, dtype=float)
        self.metabolism = float(metabolism)
        self.spacing = float(spacing_um or TRACHEAL_GEOMETRY["spacing_um"][0])
        self.r_t = float(tracheole_radius_um
                         or TRACHEAL_GEOMETRY["tracheole_radius_um"][0])
        self.D = float(d_eff_um2_s)
        self.r_c = self.spacing / np.sqrt(np.pi)     # equal-area cylinder radius

    def step(self, dt, inputs):
        """Steady radial diffusion from a tracheole of radius r_t out to the
        equal-area cylinder radius r_c.

        C(r) = C0 - (Q / (2 pi D)) ln(r / r_t)
        and the cell reads the concentration at its own distance r from the
        nearest tracheole (distance to the nearest grid node of the tracheal
        lattice).  Quasi-steady: the diffusive relaxation time across ~6.5 um
        is far shorter than the metabolic time step, which is why this is
        evaluated algebraically each step rather than integrated.
        """
        c0 = float(np.mean(np.asarray(inputs["spiracle_o2"], dtype=float)))
        q = self.metabolism
        r = self._nearest_tracheole_distance()
        r = np.clip(r, self.r_t, self.r_c)
        drop = (q / (2.0 * np.pi * self.D)) * np.log(r / self.r_t)
        c = c0 - drop
        self.last_drop = float(np.mean(drop))
        return {"cell_o2": np.clip(c, 0.0, c0)}

    def _nearest_tracheole_distance(self):
        """Distance from each cell to the nearest tracheole of a periodic
        square lattice with the declared spacing."""
        if not hasattr(self, "_cache_r") or self._cache_r is None:
            s = self.spacing
            gx = np.round(self.pos[:, 0] / s) * s
            gy = np.round(self.pos[:, 1] / s) * s
            self._cache_r = np.hypot(self.pos[:, 0] - gx, self.pos[:, 1] - gy)
        return self._cache_r


class CellStateLayer(Layer):
    """cellstate.py -- per-cell volume / ATP / membrane integrity."""

    name = "cellstate"
    requires = {"cell_o2": "1"}
    provides = {"atp": "uM", "integrity": "1", "viability": "1"}

    def __init__(self, n_cells=576, seed=2025, params=None):
        super().__init__()
        self.target_dt = Quantity(1.0, "s")
        self.n_cells = int(n_cells)
        self.seed = int(seed)
        self.params = params
        self._state = None
        self._t = 0.0

    def step(self, dt, inputs):
        import cellstate
        o2 = np.asarray(inputs["cell_o2"], dtype=float)
        r = cellstate.run(n_cells=self.n_cells, oxygen=o2, t_end=dt.si,
                          dt=min(0.05, dt.si / 10.0), seed=self.seed,
                          params=self.params, initial=self._state,
                          return_frames=False)
        fin = r["final"]
        self._state = fin
        self._t += dt.si
        self.last = r
        return {"atp": float(np.mean(fin["atp"])),
                "integrity": float(np.mean(fin["integrity"])),
                "viability": float(np.mean(fin["injured"] == 0))}


class GrowthLayer(Layer):
    """growth.py -- colony growth and division, limited by a substrate."""

    name = "growth"
    requires = {"substrate": "1"}
    provides = {"biomass": "1", "n_cells": "1"}

    def __init__(self, n_cells=576, seed=1, t_end_s=1.0):
        super().__init__()
        self.target_dt = Quantity(60.0, "s")     # growth is slow
        self.n_cells = int(n_cells)
        self.seed = int(seed)
        self._tissue = None

    def initialize(self, inputs):
        import growth
        cfg = growth._make_config() if hasattr(growth, "_make_config") else None
        self._tissue = None       # constructed lazily in step()

    def step(self, dt, inputs):
        # growth.run() integrates a whole colony; as a layer we advance it by
        # one chunk.  Wrapped conservatively: if the module cannot accept an
        # incremental call, we refuse rather than silently doing nothing.
        import growth
        if not hasattr(growth, "run"):
            raise NotImplementedError(
                "growth.py exposes Tissue/colony helpers but no incremental run(); "
                "this wrapper refuses to pretend it advanced the colony.")
        raise NotImplementedError(
            "GrowthLayer is declared but not yet advanced: growth.run() is a "
            "whole-run entry point. Refusing to report progress that did not happen.")


class HormoneLayer(Layer):
    """hormone.py -- receptor binding and a delayed nuclear response."""

    name = "hormone"
    requires = {"hormone_nM": "nM"}
    provides = {"bound_fraction": "1", "response": "1"}

    def __init__(self, params=None, t_end_s=3600.0):
        super().__init__()
        self.target_dt = Quantity(10.0, "s")
        self.params = params
        self._bound0 = 0.0
        self._resp0 = None

    def step(self, dt, inputs):
        import hormone
        dose = float(np.mean(np.asarray(inputs["hormone_nM"], dtype=float)))
        r = hormone.run(dose, t_end=dt.si, dt=min(10.0, dt.si),
                        p=self.params, initial_bound_fraction=self._bound0,
                        initial_response=self._resp0)
        self._bound0 = float(r["bound_fraction"][-1])
        self._resp0 = float(r["response"][-1])
        return {"bound_fraction": self._bound0, "response": self._resp0}


class MechanicsLayer(Layer):
    """cpm.py -- the project's own Cellular Potts engine, as a tissue layer.

    Declared but deliberately NOT advanced: a CPM is a Monte Carlo sampler whose
    'time' is Monte Carlo Steps, and mapping MCS to seconds requires a
    calibration this project has not done.  Providing a port that implies
    physical time would be a silent lie, so this layer refuses to run until the
    mapping is declared.
    """

    name = "mechanics"
    requires = {"cell_o2": "1"}
    provides = {"tissue_geometry": "um", "n_contacts": "1"}

    def __init__(self, model=None):
        super().__init__()
        self.target_dt = Quantity(1.0, "s")
        self.model = model
        self.seconds_per_mcs = None       # must be set by the caller, with a source

    def step(self, dt, inputs):
        if self.seconds_per_mcs is None:
            raise NotImplementedError(
                "MechanicsLayer needs an explicit MCS -> seconds mapping with a "
                "source before it can report physical time. Refusing to guess.")
        raise NotImplementedError(
            "MechanicsLayer advances the CPM externally; see cpm.py / merks.py.")
