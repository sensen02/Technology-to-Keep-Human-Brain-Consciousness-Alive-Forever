"""Cell GROWTH and DIVISION layer for the cell-wound prototype.

=========================================================================
HONESTY STATEMENT -- READ FIRST
=========================================================================
* Growth and division in this module are **NOT validated against any
  measurement in this project.**  They are not calibrated against the
  Drosophila pupal notum, against ``model.py``'s calcium data, against
  ``measurement.py``'s half-height radius, or against anything else.  No
  experiment in this repository reports a cell-cycle duration, a growth rate,
  a substrate uptake rate or a division geometry.  There is therefore **no
  observable in this project that could falsify a single number below.**
* A model that can be *run* to large cell counts is **still not a validated
  organism.** The included scaling routine is not evidence it was executed;
  see outputs/growth_verified.json for the actual bounded verification.  Being able to put 1e6 or
  3.7e13 cells in an array is a statement about numpy and about the growth
  law being cheap, not about biology.  Every cell-count number in this file is
  a *software* measurement; none of them is a biological prediction.
* Only TWO numeric values below are taken from a checked source (see
  ``PARAM_SOURCES``): the mean cell area 43 um^2 (project-wide convention,
  matching ``model.py``) and the notum epithelium thickness 7.5-10 um
  (Pinheiro 2017, PMC6143170, read from ``MEASURED_ANCHORS.md``).  Everything
  else -- doubling time, Monod constant, yield coefficient, substrate
  concentration, diffusion coefficient, division-threshold ratio, cell-cycle
  variability, contact-inhibition Hill exponent, relaxation gain -- is tagged
  **ILLUSTRATIVE**: an arbitrary value chosen to make the mechanism run.  No
  citation is fabricated anywhere in this file.  Where the task brief said the
  published doubling time for Drosophila epithelium is "around tens of hours",
  a web search performed while writing this file did **not** return a primary
  source stating a specific number that I was willing to quote, so the
  doubling time here is ILLUSTRATIVE at 20 h (inside "tens of hours"), and
  **no citation is asserted for it.**  That is the honest state of the
  evidence available to me, not an oversight.
* Where the model does reproduce a published order of magnitude (the
  confluence cell density, ~23 cells per 1000 um^2) the reason is stated
  explicitly in CONTACT INHIBITION below: the packing convention is what
  fixes it.  It is **not** an independent prediction.

=========================================================================
WHAT IS IMPLEMENTED
=========================================================================
A first-class, verifiable growth + division layer with two geometries:

(A) **Centre-based (off-lattice) tissue** -- the mode ``run()`` uses.  Each
    cell is a point (its centre) plus a biomass variable ``A`` (um^2), which
    is exactly the per-cell *preferred area* ``A0_c`` that
    ``mechanics.py``'s area-elasticity term ``k_area*(A_c - A0_c)^2`` pulls
    the polygon area toward; driving growth therefore means driving
    ``mechanics.run``'s ``A0`` array, not inventing a parallel quantity.  A
    cell also carries a division-progress variable, an injury state code and a
    substrate amount.  Centres are relaxed by an overdamped soft-repulsion
    (Gauss-Seidel-free Jacobi) pass so that the exclusion discs do not
    interpenetrate.

(B) **Vertex-mesh surgery** -- ``RingMesh`` plus ``divide_in_mechanics_mesh``
    implement the geometry bookkeeping a *real* vertex-mesh division needs
    (new vertex insertion, daughter ring construction, and the re-splicing of
    every neighbouring cell's ring), and the result is verified **on an actual
    ``mechanics.Mesh`` object built by ``mechanics.build_hex_mesh``** using
    mechanics' own validators (``twins_ok``, ``check_valid``).  ``mechanics.py``
    is imported read-only and is never modified.

    ``mechanics.run()`` integrates its own loop and exposes no hook between
    steps, so growth.py cannot make a running ``mechanics`` simulation divide.
    What is demonstrated is the *division operator itself* plus its validity
    proof.  The remaining work for a full vertex coupling is listed verbatim in
    ``VERTEX_ADAPTER_REQUIREMENTS`` -- it is not hidden.

=========================================================================
EQUATIONS
=========================================================================
Units: length um, area um^2, time s, substrate SU ("substrate units",
arbitrary currency) with concentration c in SU/um^3 and amount n = c*V.

1. GROWTH of the biomass/preferred area ``A`` [um^2/s]::

       dA/dt = A * mu_max * M_i * G_i

       M_i = c_i/(K_s + c_i)                     Monod nutrient factor
       G_i = gate_state * gate_ci * gate_space * gate_substrate

   ``gate_state`` is 0 for a cell whose ``cellstate`` code is >= 2
   (irreversible/lysed) -- a dead cell does not grow.  ``mu_max`` is set from
   the requested doubling time by ``mu_max = ln(2)/(3600*T_d)`` so that in the
   unlimited, ungated case a cell's area doubles in exactly ``T_d``.
   The step is applied as ``A <- A*exp(mu_eff*dt)`` (exact for constant rate,
   positivity preserving), not as forward Euler.

2. SUBSTRATE, four modes (``substrate_mode``):
   ``unlimited`` -- c = infinity, M = 1, growth obeys the pure exponential
       law; consumption is still *accounted* so the yield budget can be
       checked (the reservoir is declared infinite).
   ``reservoir`` -- the caller supplies a per-cell concentration array (or a
       scalar), held fixed for the run.  Growth is Monod-limited and can be
       spatially non-uniform, but the field is never depleted.
   ``well_mixed`` -- one shared pool of ``n_pool`` SU.  Consumption is drawn
       from the pool; if the demand exceeds the pool in a step, **all** cells'
       growth is scaled by the same factor so the pool cannot go negative.
       This is the "instantaneous mixing" limit and it is what makes the
       population saturate on a finite supply.
   ``diffusive`` -- per-cell amounts ``n_i`` with a graph-Laplacian exchange
       between neighbouring cells plus per-cell consumption and an optional
       per-cell supply rate::

           dn_i/dt = sum_j G_ij (c_j - c_i) - yield*dA_i/dt + supply_i
           G_ij    = D * h * L_ij / d_ij  =  D*h/sqrt(3)   (hexagonal packing)

       ``L_ij/d_ij`` is the contact width over the centre distance; for a
       hexagonally packed monolayer ``L_ij = d_ij/sqrt(3)``, so the pair
       conductance is **independent of the distance** and equals ``D*h/sqrt(3)``
       (this identity is why diffusion is O(pairs) and not a matrix solve; it
       assumes a hexagonal packing and a uniform cell height h).  Diffusion is
       symmetric in i,j, so it conserves the total amount exactly; the step is
       sub-cycled with ``dt <= cfl_diff * A_min/(2*sqrt(3)*D)`` and the number
       of sub-steps actually needed is reported (``metadata['diff_substeps']``).

3. DIVISION, triggered when ``division_rule`` is satisfied::

       'area'         A_i >= div_area_ratio * A_ref * noise_i
       'accumulator'  progress_i >= 1
       'both'         either of the above

   ``progress`` integrates the cycle rate ``1/T_cycle`` times the same gates
   (Monod, state, contact inhibition), so a nutrient-starved or crowded cell
   does not advance through the cycle.  On division the parent becomes
   daughter 1 in place and daughter 2 is appended with **exactly** the
   complementary area (``A2 = A - A1``, which is exact in IEEE-754 for any
   split fraction), so total area is conserved to the last bit.  The daughters
   are placed at ``x_centroid +/- (R1+R2)/2 * u``, i.e. **exactly touching and
   not overlapping each other**, along an orientation ``u`` chosen by
   ``div_axis`` ('longest_axis' = Hertwig-like long axis of the neighbour
   cloud, 'random' seeded, 'prescribed' fixed angle, 'sparse' = the candidate
   angle minimising overlap with third cells).  Overlap with *third* cells is
   possible when space is tight; it is counted per event
   (``div_third_party_overlaps``) and resolved by the relaxation pass, and the
   residual is reported as the minimum ``d/(R_i+R_j)`` ratio over all pairs.

4. DEATH / DELETION, explicit interface: ``run(state=...)`` accepts either an
   int8 per-cell array using ``cellstate.STATE_CODES`` {0 healthy,
   1 reversibly_injured, 2 irreversibly_injured, 3 lysed} or the two latched
   boolean arrays, or a callable ``state_fn(t_h, tissue) -> array`` for a
   time-varying injury.  Cells with code >= 2 (or latched irreversible/lysed)
   get ``gate_state = 0``: they stop growing AND cannot divide.  With
   ``remove_dead=True`` they are deleted from the simulation and the counts
   (irreversible, lysed, removed) are reported; deletion removes their biomass
   from the total, which is reported separately from the growth-produced
   biomass so the two are never confused.

=========================================================================
CONTACT INHIBITION (documented density-dependent rule)
=========================================================================
Two ingredients, both switchable, both referenced to the **measured mean cell
area 43 um^2** (project convention, the same value ``model.py`` and
``mechanics.py`` use):

* ``gate_space`` (growth)::

      a_i = (sqrt(3)/2) * d_i^2       d_i = mean centre distance to the 6
                                      nearest neighbours (local space per cell)
      gate_space = clip(1 - (A_i/a_i)^n, 0, 1)

  Rationale: ``a_i`` is the area of the cell's hexagonal Voronoi cell in the
  local packing, i.e. the space actually available to it, and it is exactly
  the abscissa where the exclusion discs of a hexagonally packed monolayer
  touch (``A_i`` is the packing convention, see below).  A cell grows while it
  is smaller than its own space.

* ``gate_ci`` (division = contact inhibition *of proliferation*, which is what
  contact inhibition means biologically)::

      rho_i = 1/a_i                       [cells/um^2]
      gate_ci = clip(1 - (rho_i/rho_c)^n, 0, 1),   rho_c = 1/A_REF

  ``A_REF = 43 um^2`` is the measured mean cell area, so ``rho_c = 23.256``
  cells per 1000 um^2 is the project's published epithelial density.  With
  ``ci_hill_n = 4`` both gates are ~1 far below confluence and drop sharply at
  it; the population therefore densifies until ``a_i -> 43 um^2`` and then
  stops dividing and stops growing.

**Why the density comes out right -- and why that is not a validation.**  The
exclusion radius is defined as ``R_i = sqrt(A_i/C_pack)`` with
``C_pack = 2*sqrt(3)`` (default) so that a hexagonally packed monolayer of
touching discs has exactly ``A_i`` of space per cell; that is a *packing
convention*, and it is the only place the 43 um^2 enters the cell-size scale.
With ``C_pack = pi`` instead (a true disc of area ``A_i``) the touching lattice
gives ``2*sqrt(3)/pi * A_i = 1.103 A_i`` of space per cell, i.e. the predicted
confluence density moves by ~10% (21.1 instead of 23.3 per 1000 um^2).  Both
conventions are run in ``self_test`` and both are the right order of
magnitude; **the agreement with 23.3 is inherited from the packing convention,
not predicted by the biology in this module.**  What the model does add is a
*dynamical* statement: without the gate the population has no equilibrium at
all (it grows without bound at fixed substrate), and with the gate it reaches a
stable confluence density from a range of initial seeding densities.

Second, documented negative result, reported by ``self_test``: the
"mechanically self-consistent" variant ``ci_rule='jamming'`` (gate on
``A_i/a_i`` for division too, i.e. no reference to an absolute confluence
area) **locks the tissue at whatever density it was seeded with**, because a
hexagonally packed, non-overlapping colony has ``a_i = A_i`` identically.  That
is a real, checkable failure mode of a purely mechanical crowding rule in an
unbounded 2D colony, where the free space at the rim means no compressive
stress ever builds up.  It is reported, not hidden.

=========================================================================
SUBSTRATE BUDGET (what is actually conserved)
=========================================================================
``consumed = yield * (cell area produced)`` is the *definition* of the
consumption in this file, so the check
``|consumed - yield*sum(dA)| / consumed`` is an **implementation residual**
(it would be 0 even if the mechanism were wrong), and it is reported as such.
The check that is *not* tautological is the field-level balance, which holds
only if the diffusion operator is genuinely antisymmetric and the deletions
are accounted::

    (sum_i n_i)_final - (sum_i n_i)_initial
        = supply_total - consumed_total - biomass_removed_by_deletion

plus a second identity for the biomass itself,
``sum(A)_final - sum(A)_initial = produced - removed``.  Both residuals are
reported.

=========================================================================
COMPLEXITY
=========================================================================
Per step, with N live cells, P neighbour pairs inside the contact cutoff and D
division/deletion events::

    neighbour search (cKDTree.query_pairs)     O(N log N + P)
    6-nearest query (only if contact inhibition is on)  O(N log N)
    growth / gates / substrate update          O(N)
    substrate diffusion (graph Laplacian)      O(P * n_diff_substeps)
    relaxation substeps                        O(P * n_relax)
    division surgery                           O(D) amortised (geometric
                                               capacity growth, no np.append)
    deletion                                   O(N) when it happens
    ---------------------------------------------------------------
    total                                      O(N log N + P + D)

Memory is O(N) (capacity is at most 2x the live count, so the constant is a
fixed number of float64/int8 arrays per cell).

=========================================================================
API
=========================================================================
``run(n_cells=576, ..., t_end_h=48.0, dt_h=0.05, ...)`` -> dict with
``times_h`` (T,), ``n_cells`` (T,), ``total_area`` (T,), ``total_substrate``
(T,), ``substrate_consumed_cum`` (T,), per-frame summary arrays
(``mean_area``, ``p10_area``, ``p90_area``, ``mean_crowding``, ``n_healthy``,
``n_irreversible``, ``n_lysed``, ``n_removed_cum``), the final per-cell arrays
(``positions_final``, ``areas_final``, ``state_final``, ``gid_final``), the
event log/counts, the ``budget`` dict and ``metadata`` (provenance, warnings,
timings).  ``self_test()`` returns every verification number as a dict.
``scaling_benchmark()`` measures the kernel at increasing cell counts.

Also importable: ``Tissue``, ``RingMesh``, ``plan_ring_split``, ``split_rings``,
``divide_in_mechanics_mesh``, ``vertex_division_demo``, ``exclusion_radius``,
``local_available_area``, ``hex_colony``, ``neighbour_pairs``,
``PARAM_SOURCES``, ``PARAMETER_PROVENANCE``, ``LIMITATIONS``,
``VERTEX_ADAPTER_REQUIREMENTS``.

``python growth.py``            run self_test() and the scaling benchmark
``python growth.py --quick``    fast subset

This module creates no files and no plots.
"""

from __future__ import annotations

import math
import os
import sys
import time as _time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from scipy.spatial import cKDTree
except Exception as _exc:  # pragma: no cover - environment error path
    raise ImportError("growth.py requires scipy.spatial.cKDTree") from _exc

try:  # read-only use: hex geometry + Mesh for the vertex-division adapter
    import mechanics as _mech
except Exception as _exc:  # pragma: no cover
    _mech = None
    _MECH_IMPORT_ERROR = _exc
else:
    _MECH_IMPORT_ERROR = None

__all__ = [
    "run", "self_test", "scaling_benchmark", "Tissue", "RingMesh",
    "plan_ring_split", "split_rings", "divide_in_mechanics_mesh",
    "vertex_division_demo", "exclusion_radius", "local_available_area",
    "hex_colony", "colony_from_points", "neighbour_pairs", "six_nn_spacing",
    "DEFAULTS", "PARAM_SOURCES", "PARAMETER_PROVENANCE", "LIMITATIONS",
    "VERTEX_ADAPTER_REQUIREMENTS", "A_REF_UM2", "RHO_C_PER_1000_UM2",
    "STATE_CODES", "PACKING_FACTORS",
]

# ---------------------------------------------------------------------------
# Cited / project anchors.  ONLY these two numeric anchors are not illustrative.
# ---------------------------------------------------------------------------
#: Mean cell area of the project's tissue, um^2.  Project-wide convention:
#: model.py and mechanics.py both use 43 um^2 mean cell area.  This value is
#: the ONLY cell-size anchor in this file.
A_REF_UM2 = 43.0
#: Confluence density implied by A_REF_UM2: 1000/43 = 23.256 cells per 1000 um^2.
RHO_C_PER_1000_UM2 = 1000.0 / A_REF_UM2
#: Apical-basal thickness of the notum epithelium, um.  Pinheiro 2017
#: (PMC6143170) gives 7.5-10 um as the imaging range for notum 15-18 hAPF; the
#: midpoint 8.0 um is used only to convert an area to a volume for the
#: substrate currency.  Cited via MEASURED_ANCHORS.md (checked source).
EPITHELIUM_THICKNESS_UM = 8.0

STATE_CODES = {0: "healthy", 1: "reversibly_injured", 2: "irreversibly_injured",
               3: "lysed"}

#: Exclusion-radius packing conventions: R = sqrt(A/C_pack).
#: ``hex_footprint`` (2*sqrt(3)) makes a hexagonally packed monolayer of
#: touching discs give exactly A of space per cell; ``disc`` (pi) makes the
#: radius the radius of a disc of area A.  The choice moves the confluence
#: density by 2*sqrt(3)/pi = 1.103.
PACKING_FACTORS = {"hex_footprint": 2.0 * math.sqrt(3.0), "disc": math.pi}

# ---------------------------------------------------------------------------
# Defaults.  EVERY value is 'illustrative' unless PARAM_SOURCES says otherwise.
DEFAULTS = dict(
    doubling_time_h=20.0,      # h    ILLUSTRATIVE "tens of hours", no citation
    ks_su_per_um3=0.5,         # SU/um^3  Monod half-saturation, ILLUSTRATIVE
    yield_su_per_um2=1.0,      # SU per um^2 of new cell area, ILLUSTRATIVE
    c0_su_per_um3=1.0,         # SU/um^3  initial/supplied concentration
    d_um2_per_s=1.0,           # um^2/s   substrate diffusion, ILLUSTRATIVE
    div_area_ratio=2.0,        # -    divide at this multiple of A_ref
    cycle_time_h=20.0,         # h    accumulator cycle time, ILLUSTRATIVE
    cycle_cv=0.0,              # -    cell-cycle variability (0 = synchronous)
    ci_hill_n=4.0,             # -    Hill exponent of the crowding gates
    relax_gain=0.25,           # -    Jacobi relaxation gain per substep
    relax_substeps=2,          # -    relaxation substeps per growth step
    cfl_diff=0.4,              # -    diffusion sub-step CFL factor
    max_diff_substeps=2000,    # -    cap on diffusion sub-cycling per step
    overlap_hard_tol=1e-3,     # -    relative overlap allowed by the report
    max_cells=None,            # -    hard cap on the population (None = none)
    asymmetric_split=None,     # -    daughter area fraction (None -> 0.5)
)

#: Provenance of every parameter group.  'illustrative' = arbitrary value with
#: no empirical support; 'project_anchor' = convention/anchor already used by
#: model.py / mechanics.py; 'cited' = a specific checked source.
PARAM_SOURCES = {
    "A_REF_UM2": ("project_anchor", "mean cell area 43 um^2, the project-wide "
                  "convention used by model.py and mechanics.py"),
    "EPITHELIUM_THICKNESS_UM": ("cited", "notum epithelium thickness 7.5-10 um, "
                                "Pinheiro 2017 PMC6143170 (midpoint 8.0 used); "
                                "cited via MEASURED_ANCHORS.md"),
    "doubling_time_h": ("illustrative", "20 h chosen to be inside the 'tens of "
                        "hours' range quoted for Drosophila epithelial tissue in "
                        "the task brief; NO source was verified, so NO citation "
                        "is asserted"),
    "ks_su_per_um3": ("illustrative", "no measured substrate affinity"),
    "yield_su_per_um2": ("illustrative", "defines the arbitrary substrate "
                         "currency; no measured biomass yield"),
    "c0_su_per_um3": ("illustrative", "no measured nutrient concentration"),
    "d_um2_per_s": ("illustrative", "no measured tissue diffusion coefficient; "
                    "the default 1.0 um^2/s is deliberately far below a small "
                    "metabolite in water (~10^2-10^3 um^2/s) so that the "
                    "diffusion-limited regime is reachable at tissue scale"),
    "div_area_ratio": ("illustrative", "a sizer threshold of 2x the reference "
                       "area; real epithelial cells do not double their area "
                       "before dividing"),
    "cycle_time_h": ("illustrative", "same status as doubling_time_h"),
    "cycle_cv": ("illustrative", "no measured cell-cycle variability; 0 = "
                 "perfectly synchronised by construction"),
    "ci_hill_n": ("illustrative", "sharpness of the crowding gate"),
    "relax_gain": ("illustrative", "numerical relaxation parameter"),
    "relax_substeps": ("illustrative", "numerical parameter"),
    "cfl_diff": ("illustrative", "numerical stability parameter"),
    "overlap_hard_tol": ("illustrative", "numerical tolerance"),
    "STATE_CODES": ("project_anchor", "copied from cellstate.py"),
    "PACKING_FACTORS": ("illustrative", "packing convention; see the module "
                        "docstring for why it is the origin of the density"),
}

PARAMETER_PROVENANCE = (
    "GROWTH/DIVISION LAYER -- NOT VALIDATED, NOT CALIBRATED. Exactly two "
    "numeric anchors in this file come from a checked source: the mean cell "
    "area 43 um^2 (project convention of model.py/mechanics.py) and the notum "
    "epithelium thickness 7.5-10 um (Pinheiro 2017 PMC6143170, via "
    "MEASURED_ANCHORS.md). Every other value (doubling time 20 h, Monod K_s, "
    "yield coefficient, substrate concentration, diffusion coefficient, "
    "division area ratio, cell-cycle time and variability, crowding Hill "
    "exponent, relaxation/diffusion numerical parameters) is ILLUSTRATIVE: an "
    "arbitrary value chosen to make the mechanism run, with no empirical "
    "support and no citation. No experiment in this project measures a growth "
    "rate, a cycle time, a substrate uptake or a division geometry, so none of "
    "these values can be falsified by this project's data. Reaching a large "
    "cell count is a software benchmark, not a biological result."
)

LIMITATIONS = (
    "G1 NOT VALIDATED (see the docstring): every dynamical parameter is "
    "illustrative except the 43 um^2 mean area and the 8 um thickness. "
    "G2 The centre-based mode is NOT a vertex model: cell shape is a disc and "
    "the 'area' is the preferred area A0, so the model says nothing about "
    "perimeter, junction tension or T1/T2 statistics. Coupling it to "
    "mechanics.py therefore means feeding its A0 array into mechanics.run, and "
    "that coupling is NOT demonstrated here (mechanics.run exposes no "
    "per-step hook). "
    "G3 The vertex-level division operator IS implemented and verified, but "
    "only in a purely geometric, single-shot sense: no relaxation, no T1/T2 "
    "after the split, and mechanics.run cannot be driven step-by-step from "
    "outside. See VERTEX_ADAPTER_REQUIREMENTS. "
    "G4 Overlap is resolved by a soft Jacobi relaxation with a finite number "
    "of substeps, so a residual overlap is expected and is reported "
    "(min d/(R_i+R_j)). It is not a hard-core constraint solver. "
    "G5 In an unbounded 2D colony no compressive stress ever builds up, so "
    "contact inhibition here is an explicit density-dependent GATE, not an "
    "emergent mechanical response. The 'jamming' variant demonstrates the "
    "failure mode of the mechanical-only rule. "
    "G6 Diffusion uses the hexagonal-packing conductance identity "
    "G = D*h/sqrt(3); this assumes uniform hexagonal packing and uniform cell "
    "height and is exact only in that geometry. "
    "G7 Substrate consumption is defined as yield*dA, so the "
    "consumed-vs-biomass residual is an implementation check, not a "
    "conservation law; the non-tautological checks are the field-level and "
    "biomass-level balances. "
    "G8 No apoptosis programme, no cell-cycle phases, no size checkpoint "
    "beyond the two rules implemented, no stem cells, no lineage "
    "differentiation, no 3D, no migration. "
    "G9 The scaling benchmark measures THIS kernel on THIS machine; the "
    "extrapolation to an insect tissue and to a mammal is arithmetic on a "
    "measured rate, not a simulation of those systems. A mammal (3.7e13 "
    "cells) is not simulable at any rate measured here and is not claimed to "
    "be."
)

VERTEX_ADAPTER_REQUIREMENTS = (
    "What a FULL vertex-mesh division coupling would still need, given that "
    "growth.py implements the geometry surgery and proves it valid on a real "
    "mechanics.Mesh: "
    "(1) a per-cell A0/P0 bookkeeping layer inside the vertex simulation "
    "(mechanics.run builds A0c and P0c once, before the loop, and never "
    "updates them), so that growth can change A0_c and P0_c = "
    "3.72242*sqrt(A0_c) mid-run; "
    "(2) a step-by-step entry point: mechanics.run() integrates its whole "
    "trajectory internally (a single for-loop over steps with its own state) "
    "and exposes no callback, so either mechanics.py must grow a hook or the "
    "parent must re-implement the stepping loop around Mesh.forces / "
    "Mesh.geometry -- growth.py does not duplicate that integrator; "
    "(3) post-division mechanical relaxation: the two daughter polygons "
    "created here are geometrically exact but not stress-free, so the area/"
    "perimeter energy must be relaxed and T1/T2 re-evaluated in the new "
    "topology (mechanics.Mesh.t1_swap / t2_remove exist and can be called, but "
    "their L2/L3 fragility applies); "
    "(4) an explicit rule for what happens to the *injury* annotation "
    "(he_cut flags) of the edges that were split -- growth.py refuses to split "
    "a cell that owns a cut edge unless allow_cut_split=True, and copies the "
    "flag to both daughter edges when it does; "
    "(5) a decision on vertex-slot growth: mechanics keeps vertex slots for "
    "the whole run and expects a fixed (T,V,2) history array, so a dividing "
    "mesh must either preallocate vertex slots or re-shape the recorder."
)


# ---------------------------------------------------------------------------
# geometry / packing helpers
# ---------------------------------------------------------------------------
def exclusion_radius(area, packing: str = "hex_footprint"):
    """Exclusion radius of a cell with footprint ``area`` (um^2).

    ``R = sqrt(A/C_pack)``.  ``packing='hex_footprint'`` (C = 2*sqrt(3))
    makes a hexagonally packed monolayer of touching discs give exactly ``A``
    of space per cell; ``packing='disc'`` (C = pi) is a true disc of area A.
    The convention is documented in the module docstring and is the origin of
    the predicted confluence density.
    """
    c = PACKING_FACTORS.get(packing)
    if c is None:
        raise ValueError(f"packing must be one of {sorted(PACKING_FACTORS)}")
    return np.sqrt(np.asarray(area, dtype=float) / c)


def local_available_area(mean_spacing):
    """Area of the hexagonal Voronoi cell for a centre spacing ``d``.

    ``a = (sqrt(3)/2) d^2``.  For a hexagonally packed, just-touching
    monolayer with ``d = R_i + R_j`` and packing ``hex_footprint`` this equals
    the cell's own area ``A`` exactly -- which is why ``gate_space`` and the
    no-overlap condition coincide.
    """
    d = np.asarray(mean_spacing, dtype=float)
    return 0.5 * math.sqrt(3.0) * d * d


def hex_colony(n_cells: int, area_per_cell: float = A_REF_UM2,
               rows: int | None = None):
    """Hexagonal patch of ``n_cells`` centres, ``area_per_cell`` um^2 each.

    Returns ``(positions (n,2), spacing)`` with ``spacing = sqrt(2*A/sqrt(3))``
    so that a *perfect* hexagonal packing of these centres gives exactly
    ``area_per_cell`` of space per cell.  The patch is centred on the origin.
    Cells are filled row-major, so ``n_cells`` cells may leave the last row
    partially filled (that is reported, not silently padded).
    """
    n = int(n_cells)
    if n < 1:
        raise ValueError("n_cells must be >= 1")
    d = math.sqrt(2.0 * float(area_per_cell) / math.sqrt(3.0))
    if rows is None:
        rows = max(1, int(round(math.sqrt(n / 0.8660254037844386))))
    cols = int(math.ceil(n / rows))
    pos = []
    r = 0
    while len(pos) < n:
        for c in range(cols):
            if len(pos) >= n:
                break
            pos.append((c * d + 0.5 * d * (r % 2), r * 0.5 * math.sqrt(3.0) * d))
        r += 1
    p = np.asarray(pos, dtype=float)
    p -= p.mean(axis=0)
    return p, d


def colony_from_points(points):
    """Wrap an existing centre array (no resampling)."""
    return np.array(points, dtype=float, copy=True)


def neighbour_pairs(positions, cutoff, tree=None, return_vectors=False):
    """Undirected neighbour pairs closer than ``cutoff``.

    Returns ``(i, j, d)`` with ``i < j``, sorted by ``(i, j)`` (so the result
    is deterministic), optionally also the displacement vectors
    ``dx, dy``.  O(N log N + P).
    """
    pos = np.asarray(positions, dtype=float)
    if tree is None:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(float(cutoff), output_type="ndarray")
    if pairs.size == 0:
        z = np.zeros(0, dtype=np.int64)
        out = (z, z, np.zeros(0))
        return out + (np.zeros(0), np.zeros(0)) if return_vectors else out
    i = pairs[:, 0].astype(np.int64)
    j = pairs[:, 1].astype(np.int64)
    order = np.lexsort((j, i))
    i, j = i[order], j[order]
    dvec = pos[j] - pos[i]
    d = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
    if return_vectors:
        return i, j, d, dvec[:, 0], dvec[:, 1]
    return i, j, d


def six_nn_spacing(positions, k: int = 6, tree=None):
    """Mean distance to the ``k`` nearest neighbours of every cell.

    Returns ``(mean_d (N,), n_valid (N,), d_max (N,))``.  Cells with fewer
    than ``k`` neighbours get the mean over the ones they have, and ``n_valid``
    tells the caller how many that was (an isolated cell has ``n_valid = 0``
    and ``mean_d = inf``, which the crowding gates treat as free space).
    O(N log N).
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if n == 1:
        return np.array([np.inf]), np.zeros(1, dtype=np.int64), np.array([np.inf])
    if tree is None:
        tree = cKDTree(pos)
    kk = min(k + 1, n)
    d, _ = tree.query(pos, k=kk)
    if d.ndim == 1:
        d = d[:, None]
    d = d[:, 1:] if kk > 1 else d
    valid = np.isfinite(d)
    cnt = valid.sum(axis=1)
    s = np.where(valid, d, 0.0).sum(axis=1)
    mean_d = np.where(cnt > 0, s / np.maximum(cnt, 1), np.inf)
    d_last = np.where(valid, d, np.nan)
    with np.errstate(invalid="ignore"):
        d_max = np.nanmax(d_last, axis=1) if d.shape[1] else np.full(n, np.nan)
    return mean_d, cnt, d_max


# ---------------------------------------------------------------------------
# capacity-managed tissue state
# ---------------------------------------------------------------------------
class Tissue:
    """Centre-based cell population with capacity-managed numpy arrays.

    Arrays are (cap, ...) with ``n <= cap`` live entries; ``ensure`` grows the
    capacity geometrically (amortised O(1) per appended cell), which is what
    keeps division O(D) instead of O(N) per event.  Every per-cell array is a
    plain float64/int8/int64 numpy array so the whole kernel is vectorised.

    Attributes
    ----------
    pos    : (cap,2) float  centre positions, um
    area   : (cap,)  float  biomass == preferred area A0, um^2
    acc    : (cap,)  float  cell-cycle progress (1.0 = division threshold)
    thr    : (cap,)  float  this cell's division threshold (cycle variability)
    gid    : (cap,)  int64  lineage id (parents keep theirs, daughters get new)
    state  : (cap,)  int8   cellstate.STATE_CODES
    nsub   : (cap,)  float  substrate amount held by the cell, SU
    born_h : (cap,)  float  time of birth, h
    n      : int            live cell count
    cap    : int            allocated capacity
    """

    __slots__ = ("pos", "area", "acc", "thr", "gid", "state", "nsub", "born_h",
                 "n", "cap")

    def __init__(self, positions, areas, state=None, gid=None, acc=None,
                 thr=None, nsub=None, born_h=None, capacity=None):
        areas = np.asarray(areas, dtype=float)
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions must be (n,2)")
        n = len(pos)
        if areas.shape != (n,):
            raise ValueError("areas must be (n,)")
        if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
            raise ValueError("areas must be finite and positive")
        cap = int(capacity if capacity is not None else max(n, 1))
        self.cap = cap
        self.n = n
        self.pos = np.zeros((cap, 2))
        self.area = np.zeros(cap)
        self.acc = np.zeros(cap)
        self.thr = np.ones(cap)
        self.gid = np.zeros(cap, dtype=np.int64)
        self.state = np.zeros(cap, dtype=np.int8)
        self.nsub = np.zeros(cap)
        self.born_h = np.zeros(cap)
        self.pos[:n] = pos
        self.area[:n] = areas
        self.gid[:n] = np.arange(n) if gid is None else np.asarray(gid, np.int64)
        if state is not None:
            self.state[:n] = np.asarray(state, np.int8)
        if acc is not None:
            self.acc[:n] = np.asarray(acc, float)
        if thr is not None:
            self.thr[:n] = np.asarray(thr, float)
        if nsub is not None:
            self.nsub[:n] = np.asarray(nsub, float)
        if born_h is not None:
            self.born_h[:n] = np.asarray(born_h, float)

    # -- live views ---------------------------------------------------------
    def view(self, name):
        return getattr(self, name)[: self.n]

    @property
    def n_dead(self):
        return int(np.count_nonzero(self.view("state") >= 2))

    def ensure(self, extra: int):
        """Make room for ``extra`` more cells; capacity grows geometrically."""
        need = self.n + int(extra)
        if need <= self.cap:
            return
        cap = self.cap
        while cap < need:
            cap = max(2 * cap, 16)
        for name in ("pos", "area", "acc", "thr", "nsub", "born_h"):
            arr = getattr(self, name)
            new = np.zeros((cap,) + arr.shape[1:], dtype=arr.dtype)
            new[: self.n] = arr[: self.n]
            setattr(self, name, new)
        for name in ("gid", "state"):
            arr = getattr(self, name)
            new = np.zeros(cap, dtype=arr.dtype)
            new[: self.n] = arr[: self.n]
            setattr(self, name, new)
        self.cap = cap

    def compact(self, keep_mask):
        """Keep only the cells with ``keep_mask`` True (O(N)); returns how many
        were dropped.  Compaction renumbers cells, so it is only called at a
        point in the step where no index is held elsewhere."""
        keep = np.asarray(keep_mask, bool)
        dropped = int(self.n - keep.sum())
        if dropped == 0:
            return 0
        idx = np.flatnonzero(keep)
        for name in ("pos", "area", "acc", "thr", "nsub", "born_h", "gid",
                     "state"):
            arr = getattr(self, name)
            arr[: len(idx)] = arr[idx]
        self.n = int(len(idx))
        return dropped

    def stats(self):
        a = self.view("area")
        st = self.view("state")
        return dict(n=self.n,
                    total_area=float(a.sum()),
                    mean_area=float(a.mean()) if a.size else float("nan"),
                    min_area=float(a.min()) if a.size else float("nan"),
                    max_area=float(a.max()) if a.size else float("nan"))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _cfg_double(name, value):
    return float(DEFAULTS[name] if value is None else value)


def _make_config(**kw):
    """Resolve the run configuration; unknown keys raise (no silent typos)."""
    known = set(DEFAULTS) | {
        "growth_model", "substrate_mode", "division_rule", "div_axis",
        "contact_inhibition", "ci_rule", "packing", "remove_dead",
        "daughter_state", "prescribed_axis_deg", "prescribed_exact",
        "allow_cut_split", "isolated_slots", "overlap_hard_tol"}
    unknown = set(kw) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in kw.items() if v is not None})
    for key in ("doubling_time_h", "cycle_time_h", "ks_su_per_um3",
                "yield_su_per_um2", "c0_su_per_um3", "d_um2_per_s",
                "div_area_ratio", "cycle_cv", "ci_hill_n", "relax_gain",
                "overlap_hard_tol"):
        cfg[key] = float(cfg[key])
    cfg["relax_substeps"] = int(cfg["relax_substeps"])
    cfg["max_diff_substeps"] = int(cfg["max_diff_substeps"])
    if cfg["growth_model"] not in ("exponential", "monod"):
        raise ValueError("growth_model must be 'exponential' or 'monod'")
    if cfg["substrate_mode"] not in ("unlimited", "reservoir", "well_mixed",
                                     "diffusive"):
        raise ValueError("substrate_mode must be unlimited/reservoir/"
                         "well_mixed/diffusive")
    if cfg["division_rule"] not in ("area", "accumulator", "both"):
        raise ValueError("division_rule must be area/accumulator/both")
    if cfg["div_axis"] not in ("longest_axis", "random", "prescribed",
                               "sparse"):
        raise ValueError("div_axis must be longest_axis/random/prescribed/"
                         "sparse")
    if cfg["ci_rule"] not in ("density", "jamming"):
        raise ValueError("ci_rule must be 'density' or 'jamming'")
    if cfg["packing"] not in PACKING_FACTORS:
        raise ValueError(f"packing must be one of {sorted(PACKING_FACTORS)}")
    if cfg["doubling_time_h"] <= 0 or cfg["cycle_time_h"] <= 0:
        raise ValueError("doubling_time_h and cycle_time_h must be positive")
    if cfg["cycle_cv"] < 0:
        raise ValueError("cycle_cv must be >= 0")
    if not 0.0 < cfg["relax_gain"] <= 0.5:
        raise ValueError("relax_gain must be in (0, 0.5]")
    if cfg["daughter_state"] not in ("inherit", "healthy"):
        raise ValueError("daughter_state must be 'inherit' or 'healthy'")
    cfg["mu_max_per_s"] = math.log(2.0) / (3600.0 * cfg["doubling_time_h"])
    cfg["cycle_rate_per_s"] = 1.0 / (3600.0 * cfg["cycle_time_h"])
    cfg["c_pack"] = PACKING_FACTORS[cfg["packing"]]
    return cfg


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _crowding(pos, n, area, cfg, tree=None):
    """Per-cell crowding measures.

    Returns a dict with ``mean_d`` (mean 6-NN centre distance, um),
    ``a_avail`` (local available area, um^2), ``rho`` (local density, cells per
    um^2), ``interior`` (bool: the 6th neighbour is not much further than the
    1st, i.e. the cell is genuinely surrounded), ``gate_space`` and
    ``gate_prolif``.
    """
    hist = {}
    if n == 0:
        z = np.zeros(0)
        return dict(mean_d=z, a_avail=z, rho=z, interior=np.zeros(0, bool),
                    gate_space=z, gate_prolif=z)
    if not cfg["contact_inhibition"]:
        one = np.ones(n)
        inf = np.full(n, np.inf)
        return dict(mean_d=inf, a_avail=inf, rho=np.zeros(n),
                    interior=np.zeros(n, bool), gate_space=one,
                    gate_prolif=one)
    mean_d, cnt, d_max = six_nn_spacing(pos, k=6, tree=tree)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = local_available_area(np.where(np.isfinite(mean_d), mean_d, np.nan))
    a = np.where(np.isfinite(a) & (a > 0), a, np.inf)
    rho = 1.0 / a
    hill = cfg["ci_hill_n"]
    if cfg["ci_rule"] == "density":          # reference: measured 43 um^2
        ratio_prolif = rho * A_REF_UM2
    else:                                    # 'jamming': reference = own size
        ratio_prolif = area / a
    ratio_space = area / a
    gate_space = np.clip(1.0 - np.power(np.clip(ratio_space, 0, None), hill),
                         0.0, 1.0)
    gate_p = np.clip(1.0 - np.power(np.clip(ratio_prolif, 0, None), hill),
                     0.0, 1.0)
    # "interior": the 6th neighbour is at most 1.5x the 1st (a rim cell has a
    # 6-NN mean that is biased outward and a density that is not comparable).
    d1, _, _ = six_nn_spacing(pos, k=1, tree=tree)
    with np.errstate(invalid="ignore"):
        interior = np.isfinite(d_max) & np.isfinite(d1) & (d_max <= 1.5 * d1)
    return dict(mean_d=mean_d, a_avail=a, rho=rho, interior=interior,
                gate_space=gate_space, gate_prolif=gate_p)


# ---------------------------------------------------------------------------
# substrate
# ---------------------------------------------------------------------------
def _init_substrate(tissue, cfg, out, c0=None, pool_su=None):
    """Seed per-cell amounts / the shared pool from the configuration."""
    mode = cfg["substrate_mode"]
    h = EPITHELIUM_THICKNESS_UM
    n = tissue.n
    info = dict(mode=mode, c0=None, pool_initial=None, initial_amount=0.0)
    if mode == "diffusive":
        c0 = cfg["c0_su_per_um3"] if c0 is None else float(c0)
        tissue.nsub[:n] = c0 * tissue.area[:n] * h
        info["c0"] = c0
        info["initial_amount"] = float(tissue.nsub[:n].sum())
    elif mode == "well_mixed":
        c0 = cfg["c0_su_per_um3"] if c0 is None else float(c0)
        pool = c0 * float(tissue.area[:n].sum() * h) if pool_su is None \
            else float(pool_su)
        out["pool_su"] = pool
        info["c0"] = c0 if pool_su is None else None
        info["pool_initial"] = pool
        info["initial_amount"] = pool
    elif mode == "reservoir":
        conc = np.asarray(out["supplied_conc"], float)
        info["c0"] = float(conc[0]) if conc.size else None
        info["initial_amount"] = float("inf")
    else:  # unlimited
        info["initial_amount"] = float("inf")
    return info


def _monod_factor(c, ks):
    c = np.asarray(c, float)
    return np.where(np.isfinite(c), c / (ks + c), 1.0)


# ---------------------------------------------------------------------------
# division
# ---------------------------------------------------------------------------
def _division_axes(pos, area, n, pi, pj, dx, dy, d, idx, cfg, rng,
                   six=None, log=None):
    """Orientation of the division axis for the dividing cells ``idx``.

    'longest_axis' is Hertwig-like: the principal axis of the neighbour cloud's
    shape tensor ``M = sum_j (dx,dy)(dx,dy)^T / d^2`` -- the direction in which
    the neighbours are spread furthest, i.e. the cell's long axis.  If the local
    environment is isotropic (|l1-l2| below 2% of l1+l2, which is the generic
    case in a hexagonal packing) the axis is NOT defined by the neighbours, so
    it falls back to a seeded random angle -- documented, not hidden.
    """
    D = len(idx)
    rule = cfg["div_axis"]
    if rule == "prescribed":
        ang = math.radians(float(cfg["prescribed_axis_deg"]))
        return np.full(D, ang), dict(isotropic_fallback=0)
    if rule == "random":
        return rng.uniform(0.0, math.pi, D), dict(isotropic_fallback=D)
    if rule == "sparse":
        # candidate angles; pick the one whose daughter pair overlaps the six
        # nearest neighbours least.
        if six is None:
            _, six_idx, six_d = _six_nn_full(pos, n)
        else:
            six_idx, six_d = six
        six_idx = six_idx[idx]
        six_d = six_d[idx]
        ang = np.linspace(0.0, math.pi, 13)[:-1]
        ca, sa = np.cos(ang), np.sin(ang)
        R2 = exclusion_radius(area[idx] * 0.5, cfg["packing"])
        half = R2
        off_x = half[:, None] * ca[None, :]      # (D,A)
        off_y = half[:, None] * sa[None, :]
        best = np.zeros(D)
        worst = np.full(D, np.inf)
        for cand in range(len(ang)):
            ox = off_x[:, cand][:, None]
            oy = off_y[:, cand][:, None]
            nx = pos[six_idx][:, :, 0] - pos[idx][:, None, 0]
            ny = pos[six_idx][:, :, 1] - pos[idx][:, None, 1]
            dxx = np.hypot(nx - ox, ny - oy)
            dxy = np.hypot(nx + ox, ny + oy)
            Rsum = R2[:, None] + exclusion_radius(area[six_idx], cfg["packing"])
            ov = np.maximum(Rsum - np.minimum(dxx, dxy), 0.0)
            worst_c = np.where(np.isfinite(six_d[idx]), ov, 0.0).max(axis=1)
            better = worst_c < worst
            best = np.where(better, ang[cand], best)
            worst = np.where(better, worst_c, worst)
        return best, dict(isotropic_fallback=0,
                          sparse_best_overlap_um=worst)
    # 'longest_axis'
    w = 1.0 / np.maximum(d * d, 1e-12)
    mxx = np.bincount(pi, weights=w * dx * dx, minlength=n) + \
        np.bincount(pj, weights=w * dx * dx, minlength=n)
    myy = np.bincount(pi, weights=w * dy * dy, minlength=n) + \
        np.bincount(pj, weights=w * dy * dy, minlength=n)
    mxy = np.bincount(pi, weights=w * dx * dy, minlength=n) + \
        np.bincount(pj, weights=w * dx * dy, minlength=n)
    A_ = mxx[idx]
    B_ = myy[idx]
    C_ = mxy[idx]
    ang = 0.5 * np.arctan2(2.0 * C_, A_ - B_)
    mean_trace = 0.5 * (A_ + B_)
    with np.errstate(invalid="ignore", divide="ignore"):
        aniso = np.hypot(A_ - B_, 2.0 * C_) / np.maximum(mean_trace, 1e-300)
    iso = ~np.isfinite(aniso) | (aniso < 0.02)
    k = int(iso.sum())
    if k:
        ang[iso] = rng.uniform(0.0, math.pi, k)
    return ang, dict(isotropic_fallback=k)


def _six_nn_full(pos, n, k=6):
    mean_d, _, _ = six_nn_spacing(pos, k=k)
    tree = cKDTree(pos)
    kk = min(k + 1, n)
    _, idx = tree.query(pos, k=kk)
    idx = idx[:, 1:] if kk > 1 else idx
    d = mean_d
    return None, idx, d


def _apply_divisions(tissue, idx, angles, t_h, cfg, rng, log, tree_before=None):
    """Split the cells ``idx`` into two daughters each.

    Area is split as ``a1 = f*A``, ``a2 = A - a1`` (exact in IEEE-754 for any
    f), and the substrate amount is split the same way, so both are conserved
    to the last bit.  The daughters are placed at ``c +/- ((R1+R2)/2)*u``,
    i.e. exactly touching along ``u`` and not overlapping each other.
    """
    D = int(len(idx))
    if D == 0:
        return dict(events=0, max_area_error_um2=0.0,
                    max_area_error_rel=0.0, max_volume_error_um3=0.0,
                    daughters_overlapping_um=0, third_party_overlaps=0)
    # Snapshot the entire population: tree indices are not parent-subset indices.
    pos_before = tissue.pos[:tissue.n].copy()
    area_before = tissue.area[:tissue.n].copy()
    tree_before = cKDTree(pos_before)
    tissue.ensure(D)
    area_p = tissue.area[idx].copy()
    nsub_p = tissue.nsub[idx].copy()
    pos_p = tissue.pos[idx].copy()
    acc_p = tissue.acc[idx].copy()
    state_p = tissue.state[idx].copy()
    gid_p = tissue.gid[idx].copy()
    thr_p = tissue.thr[idx].copy()
    f = 0.5 if not cfg.get("asymmetric_split") else float(cfg["asymmetric_split"])
    a1 = area_p * f
    a2 = area_p - a1
    R1 = exclusion_radius(a1, cfg["packing"])
    R2 = exclusion_radius(a2, cfg["packing"])
    half = 0.5 * (R1 + R2)
    ux = np.cos(angles)
    uy = np.sin(angles)
    new = np.arange(tissue.n, tissue.n + D, dtype=np.int64)
    tissue.n += D
    for name in ("acc", "thr", "gid", "state", "born_h"):
        arr = getattr(tissue, name)
        arr[new] = arr[idx]
    tissue.born_h[idx] = t_h
    tissue.born_h[new] = t_h
    tissue.pos[idx, 0] = pos_p[:, 0] - half * ux
    tissue.pos[idx, 1] = pos_p[:, 1] - half * uy
    tissue.pos[new, 0] = pos_p[:, 0] + half * ux
    tissue.pos[new, 1] = pos_p[:, 1] + half * uy
    tissue.area[idx] = a1
    tissue.area[new] = a2
    tissue.nsub[idx] = nsub_p * f
    tissue.nsub[new] = nsub_p - nsub_p * f
    # cycle progress: both daughters inherit the excess over the threshold
    # (documented choice: it makes the cycle period exactly thr/rate; halving
    # would halve the period)
    carry = np.maximum(acc_p - 1.0, 0.0)
    tissue.acc[idx] = carry
    tissue.acc[new] = carry
    if cfg["daughter_state"] == "healthy":
        tissue.state[new] = 0
    if cfg["cycle_cv"] > 0:
        sigma = math.sqrt(math.log1p(cfg["cycle_cv"] ** 2))
        tissue.thr[idx] = np.exp(rng.normal(0.0, sigma, D))
        tissue.thr[new] = np.exp(rng.normal(0.0, sigma, D))
    else:
        tissue.thr[idx] = 1.0
        tissue.thr[new] = 1.0
    err = np.abs(area_p - (tissue.area[idx] + tissue.area[new]))
    third = _third_party_overlaps(pos_p, area_p, idx, a1, a2, half, ux, uy,
                                  tree_before, cfg, pos_before, area_before)
    ev = dict(events=D,
              max_area_error_um2=float(err.max()),
              max_area_error_rel=float((err / area_p).max()),
              max_volume_error_um3=float((err * EPITHELIUM_THICKNESS_UM).max()),
              daughters_overlapping_um=0.0,
              third_party_overlaps=third)
    if log is not None:
        for k in range(D):
            log.append(dict(kind="division", t_h=float(t_h),
                            parent_gid=int(gid_p[k]),
                            area_before=float(area_p[k]),
                            area_d1=float(tissue.area[idx[k]]),
                            area_d2=float(tissue.area[new[k]]),
                            area_error_um2=float(err[k]),
                            axis_deg=float(math.degrees(angles[k])),
                            pos=(float(pos_p[k, 0]), float(pos_p[k, 1]))))
    return ev


def _third_party_overlaps(pos_p, area_p, idx, a1, a2, half, ux, uy,
                          tree_before, cfg, pos_before, area_before):
    """How many daughter placements overlap a cell that is neither the parent
    nor the sibling.

    Only the parent's own six nearest neighbours are tested (the daughters sit
    within one radius of the parent, so this catches essentially every
    third-party contact); it is O(D log N) for D divisions rather than a
    per-event tree rebuild.
    """
    D = len(idx)
    if D == 0:
        return 0
    if tree_before is None:
        tree_before = cKDTree(pos_before)
    n = len(pos_before)
    if n < 2:
        return 0
    kk = min(7, n)
    _, nn = tree_before.query(pos_p, k=kk)
    if nn.ndim == 1:
        nn = nn[:, None]
    nn = nn[:, 1:] if kk > 1 else nn
    if nn.size == 0:
        return 0
    R1 = exclusion_radius(a1, cfg["packing"])[:, None]
    R2 = exclusion_radius(a2, cfg["packing"])[:, None]
    Rn = exclusion_radius(area_before[nn], cfg["packing"])
    c = pos_p[:, None, :]
    p1 = np.stack([c[..., 0] - half[:, None] * ux[:, None],
                   c[..., 1] - half[:, None] * uy[:, None]], axis=-1)
    p2 = np.stack([c[..., 0] + half[:, None] * ux[:, None],
                   c[..., 1] + half[:, None] * uy[:, None]], axis=-1)
    nb = pos_before[nn]
    d1 = np.sqrt(((p1 - nb) ** 2).sum(axis=-1))
    d2 = np.sqrt(((p2 - nb) ** 2).sum(axis=-1))
    over = ((d1 < R1 + Rn - 1e-12) | (d2 < R2 + Rn - 1e-12))
    return int(over.sum())


# ---------------------------------------------------------------------------
# one step of the kernel
# ---------------------------------------------------------------------------
def _step(tissue, cfg, out, t_s, dt_s, rng, supplied_conc, diff_pairs,
          log_events, ev_log, counters, supply=None):
    """One growth+division step (see the module docstring for the equations).

    Order (all vectorised, all O(N) or O(P)):
      1. neighbour pairs + crowding measures
      2. growth rate -> exact exponential area update
      3. substrate consumption (clipped so no amount goes negative)
      4. cycle progress
      5. diffusion sub-cycling (diffusive mode only)
      6. relaxation substeps
      7. division
      8. deletion
    """
    n0 = tissue.n
    if n0 == 0:
        return
    pos = tissue.pos[:n0]
    area = tissue.area[:n0]
    state = tissue.state[:n0]
    h = EPITHELIUM_THICKNESS_UM

    # ---- 1. neighbours ----------------------------------------------------
    cutoff = float(2.0 * exclusion_radius(area.max(), cfg["packing"]) * 1.05)
    tree = cKDTree(pos)
    pi, pj, d, dx, dy = neighbour_pairs(pos, cutoff, tree=tree,
                                        return_vectors=True)
    crowd = _crowding(pos, n0, area, cfg, tree=tree)
    a_av = crowd["a_avail"]
    fin = np.isfinite(a_av)
    with np.errstate(invalid="ignore", divide="ignore"):
        dens = np.where(crowd["interior"] & fin, 1000.0 / a_av, np.nan)
    dens_fin = dens[np.isfinite(dens)]
    avail_fin = a_av[fin]
    out["last_crowd"] = dict(
        n=n0, n_interior=int(crowd["interior"].sum()),
        interior_density_per_1000=float(dens_fin.mean()) if dens_fin.size
        else float("nan"),
        gate_space_mean=float(crowd["gate_space"].mean()),
        mean_avail_um2=float(avail_fin.mean()) if avail_fin.size
        else float("nan"))

    # ---- 2. growth gates --------------------------------------------------
    alive_state = (state < 2).astype(float)
    nsub = tissue.nsub[:n0]
    mode = cfg["substrate_mode"]
    if cfg["growth_model"] == "exponential" or mode == "unlimited":
        monod = np.ones(n0)
        conc = np.full(n0, np.inf)
    elif mode == "reservoir":
        conc = np.asarray(supplied_conc, float)
        if conc.shape != (n0,):
            raise ValueError("supplied concentration must have shape (n_cells,)")
        monod = _monod_factor(conc, cfg["ks_su_per_um3"])
    elif mode == "well_mixed":
        pool = out["pool_su"]
        tot_v = float(area.sum() * h)
        c = pool / max(tot_v, 1e-300)
        conc = np.full(n0, c)
        monod = _monod_factor(np.array([c]), cfg["ks_su_per_um3"])[0] * \
            np.ones(n0)
    else:  # diffusive
        conc = nsub / np.maximum(area * h, 1e-300)
        conc = np.maximum(conc, 0.0)
        monod = _monod_factor(conc, cfg["ks_su_per_um3"])

    gate = alive_state * monod * crowd["gate_space"]
    mu = cfg["mu_max_per_s"] * gate
    dA_want = area * np.expm1(mu * dt_s)

    # ---- 3. consumption, clipped so that nothing goes negative ------------
    y = cfg["yield_su_per_um2"]
    need = y * dA_want
    frac_limited = np.ones(n0)
    if supply is not None:
        add = supply[:n0] * dt_s
        tissue.nsub[:n0] = np.maximum(nsub + add, 0.0)
        nsub = tissue.nsub[:n0]
        out["supplied_cum"] += float(add.sum())
        if mode in ("reservoir", "unlimited"):
            out["supplied_cum"] += 0.0  # supply is only wired for diffusive
    if mode == "well_mixed":
        pool = out["pool_su"]
        tot = float(need.sum())
        if tot > pool and tot > 0:
            scale = pool / tot
            frac_limited[:] = scale
            need *= scale
            dA_want *= scale
            counters["pool_limited_steps"] += 1
        out["pool_su"] = max(pool - float(need.sum()), 0.0)
    elif mode == "diffusive":
        lim = need > nsub
        if lim.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(lim, nsub / np.maximum(need, 1e-300), 1.0)
            s = np.clip(s, 0.0, 1.0)
            frac_limited = s
            need = need * s
            dA_want = dA_want * s
            counters["substrate_clipped_cells"] += int(lim.sum())
        tissue.nsub[:n0] = np.maximum(nsub - need, 0.0)
        nsub = tissue.nsub[:n0]
    tissue.area[:n0] = area + dA_want
    area = tissue.area[:n0]
    consumed = float(need.sum())
    produced = float(dA_want.sum())
    out["consumed_cum"] += consumed
    out["produced_cum"] += produced

    # ---- 4. cycle progress ------------------------------------------------
    g_p = alive_state * monod * crowd["gate_prolif"]
    rate = cfg["cycle_rate_per_s"] * g_p
    tissue.acc[:n0] += rate * dt_s

    # ---- 5. diffusion sub-cycling ----------------------------------------
    if mode == "diffusive":
        D = cfg["d_um2_per_s"]
        if D > 0 and n0 > 1:
            a_min = float(area.min())
            dt_max = cfg["cfl_diff"] * a_min / (2.0 * math.sqrt(3.0) * D)
            nsub_steps = int(min(cfg["max_diff_substeps"],
                                 max(1, math.ceil(dt_s / max(dt_max, 1e-300)))))
            if nsub_steps >= cfg["max_diff_substeps"]:
                counters["diff_substeps_capped"] += 1
            out["diff_substeps_max"] = max(out["diff_substeps_max"],
                                           nsub_steps)
            out["diff_substeps_total"] += nsub_steps
            g_pair = D * h / math.sqrt(3.0)
            dts = dt_s / nsub_steps
            for _ in range(nsub_steps):
                a_now = tissue.area[:n0]
                c = np.maximum(tissue.nsub[:n0], 0.0) / np.maximum(
                    a_now * h, 1e-300)
                flux = g_pair * (c[pj] - c[pi])
                dn = np.bincount(pi, weights=flux, minlength=n0) - \
                    np.bincount(pj, weights=flux, minlength=n0)
                np.maximum(tissue.nsub[:n0] + dn * dts, 0.0,
                           out=tissue.nsub[:n0])

    # ---- 6. relaxation ----------------------------------------------------
    if cfg["relax_substeps"] > 0 and n0 > 1 and len(pi):
        gain = cfg["relax_gain"]
        c_pack = cfg["c_pack"]
        for _ in range(cfg["relax_substeps"]):
            p_now = tissue.pos[:n0]
            a_now = tissue.area[:n0]
            R = np.sqrt(a_now / c_pack)
            vx = p_now[pj, 0] - p_now[pi, 0]
            vy = p_now[pj, 1] - p_now[pi, 1]
            dd = np.sqrt(vx * vx + vy * vy)
            dd_safe = np.maximum(dd, 1e-12)
            ov = (R[pi] + R[pj]) - dd
            m = ov > 0
            if not m.any():
                break
            w = gain * ov[m] / dd_safe[m]
            wux = w * vx[m]
            wuy = w * vy[m]
            fx = np.bincount(pi[m], weights=wux, minlength=n0) - \
                np.bincount(pj[m], weights=wux, minlength=n0)
            fy = np.bincount(pi[m], weights=wuy, minlength=n0) - \
                np.bincount(pj[m], weights=wuy, minlength=n0)
            mag = np.sqrt(fx * fx + fy * fy)
            cap = 0.5 * np.sqrt(np.maximum(a_now, 1e-12))
            sc = np.where(mag > cap, cap / np.maximum(mag, 1e-300), 1.0)
            counters["relax_clipped_cells"] += int((mag > cap).sum())
            tissue.pos[:n0, 0] -= fx * sc
            tissue.pos[:n0, 1] -= fy * sc

    # ---- 7. division ------------------------------------------------------
    area = tissue.area[:n0]
    state = tissue.state[:n0]
    acc = tissue.acc[:n0]
    thr = tissue.thr[:n0]
    ok = (state < 2) * (monod > 1e-12) * (crowd["gate_prolif"] > 1e-9)
    if cfg["division_rule"] in ("area", "both"):
        a_thr = cfg["div_area_ratio"] * A_REF_UM2 * thr
    else:
        a_thr = np.full(n0, np.inf)
    if cfg["division_rule"] in ("accumulator", "both"):
        acc_hit = acc >= thr
    else:
        acc_hit = np.zeros(n0, bool)
    hit = (ok > 0) & (acc_hit | (area >= a_thr))
    idx = np.flatnonzero(hit)
    counters["pending_divisions"] = int(idx.size)
    if idx.size:
        if cfg["max_cells"] is not None and tissue.n + idx.size > cfg["max_cells"]:
            idx = idx[: max(0, cfg["max_cells"] - tissue.n)]
            counters["divisions_truncated_by_max_cells"] += 1
        angles, ang_info = _division_axes(pos, area, n0, pi, pj, dx, dy, d,
                                          idx, cfg, rng)
        counters["axis_isotropic_fallback"] += int(
            ang_info.get("isotropic_fallback", 0))
        log = ev_log if (log_events and len(ev_log) < cfg["max_logged_events"]) \
            else None
        ev = _apply_divisions(tissue, idx, angles, t_s / 3600.0, cfg, rng, log,
                              tree_before=tree)
        # Daughter fields inherit the parent's external concentration/supply.
        for field in ('supplied_conc', 'supply'):
            values = out.get(field)
            if values is not None:
                out[field] = np.concatenate((values, values[idx]))
        counters["divisions"] += ev["events"]
        out["max_division_area_error_um2"] = max(
            out["max_division_area_error_um2"], ev["max_area_error_um2"])
        out["max_division_area_error_rel"] = max(
            out["max_division_area_error_rel"], ev["max_area_error_rel"])
        counters["div_third_party_overlaps"] += ev["third_party_overlaps"]

    # ---- 8. deletion ------------------------------------------------------
    n1 = tissue.n
    if n1:
        st = tissue.state[:n1]
        if cfg["remove_dead"]:
            dead = st >= 2
            k = int(dead.sum())
            if k:
                lost = float(tissue.area[:n1][dead].sum())
                out["removed_cum"] += k
                out["removed_biomass_cum"] += lost
                counters["deleted"] += k
                for field in ('supplied_conc', 'supply'):
                    if out.get(field) is not None:
                        out[field] = out[field][~dead]
                tissue.compact(~dead)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def _resolve_state(state, n, t_h):
    """Normalise the injury interface to an int8 array of length ``n``.

    Accepted: None (all healthy); an int8/float array of ``cellstate`` codes;
    a dict with 'state' and/or 'latched_irreversible'/'latched_lysed' (the
    arrays returned by ``cellstate.run``); or a callable
    ``state_fn(t_h, tissue) -> array`` for a time-varying injury.
    """
    if state is None:
        return np.zeros(n, dtype=np.int8)
    if callable(state):
        out = state(t_h, None)
        return np.asarray(out, dtype=np.int8)
    if isinstance(state, dict):
        if "state" in state and state["state"] is not None:
            arr = np.asarray(state["state"], dtype=np.int8).ravel()
        else:
            arr = np.zeros(n, dtype=np.int8)
        if arr.size != n:
            raise ValueError(f"state array has size {arr.size}, expected {n}")
        if state.get("latched_irreversible") is not None:
            irr = np.asarray(state["latched_irreversible"], bool).ravel()
            arr = np.maximum(arr, np.where(irr, 2, 0)).astype(np.int8)
        if state.get("latched_lysed") is not None:
            lys = np.asarray(state["latched_lysed"], bool).ravel()
            arr = np.maximum(arr, np.where(lys, 3, 0)).astype(np.int8)
        return arr
    arr = np.asarray(state, dtype=np.int8).ravel()
    if arr.size != n:
        raise ValueError(f"state array has size {arr.size}, expected {n}")
    return arr


def run(n_cells=576, t_end_h=48.0, dt_h=0.05, seed=2025,
        area0_um2=A_REF_UM2, seed_area_um2=None, spacing_um=None,
        positions=None,
        growth_model="monod", doubling_time_h=None, ks_su_per_um3=None,
        substrate_mode="unlimited", c0_su_per_um3=None, pool_su=None,
        nutrient=None, yield_su_per_um2=None, d_um2_per_s=None,
        supply_su_per_s=None,
        division_rule="area", div_area_ratio=None, cycle_time_h=None,
        cycle_cv=None, div_axis="longest_axis", prescribed_axis_deg=0.0,
        asymmetric_split=None,
        contact_inhibition=False, ci_rule="density", ci_hill_n=None,
        packing="hex_footprint", relax_gain=None, relax_substeps=None,
        state=None, remove_dead=False, daughter_state="inherit",
        max_cells=None, sample_every=1, log_events=True,
        max_logged_events=20000, max_steps=2_000_000, verbose=False):
    """Grow and divide a centre-based cell population (see the module docstring).

    Time is in HOURS at the interface (``t_end_h``, ``dt_h``,
    ``doubling_time_h``, ``cycle_time_h``) and in seconds internally, matching
    the rest of the project for the state layer while keeping the
    cell-biological timescale readable.

    Parameters
    ----------
    n_cells : int. Number of cells seeded (hexagonal colony, see ``hex_colony``).
    seed_area_um2 : float or None. Area per cell in the *seeding* lattice; a
        value larger than ``area0_um2`` seeds a sub-confluent colony.  None =
        confluent seeding at ``area0_um2``.
    spacing_um : float or None. Explicit seeding spacing (overrides
        ``seed_area_um2``).
    positions : (n,2) or None. Explicit seeding centres (overrides both).
    area0_um2 : float. Initial biomass == preferred area A0 of every cell.
    growth_model : 'exponential' (no nutrient factor) or 'monod'.
    substrate_mode : 'unlimited' | 'reservoir' | 'well_mixed' | 'diffusive'.
    nutrient : scalar or (n_cells,) array -- the SUPPLIED per-cell
        concentration field used by 'reservoir'.
    division_rule : 'area' | 'accumulator' | 'both'.
    div_axis : 'longest_axis' | 'random' | 'prescribed' | 'sparse'.
    contact_inhibition : bool. Turns on the documented crowding gates.
    ci_rule : 'density' (reference = the measured 43 um^2 mean cell area) or
        'jamming' (reference = the cell's own current area; documented to lock
        the tissue at its seeding density).
    state : None | int8 array of cellstate codes | dict with 'state' /
        'latched_irreversible' / 'latched_lysed' | callable state_fn(t_h, _).
    remove_dead : bool. Delete cells with state >= 2 (counted and reported).
    max_cells : int or None. Hard cap on the population (safety for a runaway
        exponential run); truncation is counted, never silent.
    sample_every : int. Record a summary frame every this many steps.

    Returns
    -------
    dict -- see the module docstring (API section).  Everything is a plain
    numpy array or python scalar; no files, no plots.
    """
    t_wall = _time.perf_counter()
    cfg = _make_config(
        growth_model=growth_model, doubling_time_h=doubling_time_h,
        ks_su_per_um3=ks_su_per_um3, substrate_mode=substrate_mode,
        c0_su_per_um3=c0_su_per_um3, yield_su_per_um2=yield_su_per_um2,
        d_um2_per_s=d_um2_per_s, division_rule=division_rule,
        div_area_ratio=div_area_ratio, cycle_time_h=cycle_time_h,
        cycle_cv=cycle_cv, div_axis=div_axis,
        prescribed_axis_deg=prescribed_axis_deg,
        asymmetric_split=asymmetric_split, contact_inhibition=contact_inhibition,
        ci_rule=ci_rule, ci_hill_n=ci_hill_n, packing=packing,
        relax_gain=relax_gain, relax_substeps=relax_substeps,
        remove_dead=remove_dead, daughter_state=daughter_state,
        max_cells=max_cells)
    cfg["max_logged_events"] = int(max_logged_events)
    rng = np.random.default_rng(int(seed))

    # ---- seeding ----------------------------------------------------------
    if positions is not None:
        pos = colony_from_points(positions)
        n_cells = len(pos)
        seed_d = float("nan")
    elif spacing_um is not None:
        pos, seed_d = hex_colony(n_cells, A_REF_UM2)
        seed_d = float(spacing_um)
        pos = pos * (seed_d / math.sqrt(2.0 * A_REF_UM2 / math.sqrt(3.0)))
    else:
        sa = float(area0_um2 if seed_area_um2 is None else seed_area_um2)
        pos, seed_d = hex_colony(n_cells, sa)
    n = len(pos)
    area0 = np.full(n, float(area0_um2))
    if np.any(area0 <= 0):
        raise ValueError("area0_um2 must be positive")
    st = _resolve_state(state, n, 0.0)
    tissue = Tissue(pos, area0, state=st)

    # ---- supplied concentration ------------------------------------------
    supplied_conc = None
    if cfg["substrate_mode"] == "reservoir":
        if nutrient is None:
            supplied_conc = np.full(n, cfg["c0_su_per_um3"])
            conc_source = (f"uniform {cfg['c0_su_per_um3']} SU/um^3 "
                           "(ILLUSTRATIVE)")
        else:
            supplied_conc = np.broadcast_to(np.asarray(nutrient, float),
                                            (n,)).copy()
            conc_source = "caller-supplied per-cell field"
    else:
        conc_source = "not used (substrate_mode != 'reservoir')"
    supply = None
    if supply_su_per_s is not None:
        supply = np.broadcast_to(np.asarray(supply_su_per_s, float),
                                 (n,)).copy()

    # ---- output containers -------------------------------------------------
    out = dict(consumed_cum=0.0, produced_cum=0.0, removed_cum=0,
               removed_biomass_cum=0.0, supplied_cum=0.0,
               max_division_area_error_um2=0.0, max_division_area_error_rel=0.0,
               diff_substeps_max=0, diff_substeps_total=0, pool_su=0.0,
               supplied_conc=supplied_conc, supply=supply)
    sub_info = _init_substrate(tissue, cfg, out, c0=c0_su_per_um3,
                               pool_su=pool_su)
    counters = _new_counters()
    ev_log = []
    frames = []
    n_steps = int(round(t_end_h / dt_h))
    if n_steps < 1:
        raise ValueError("t_end_h/dt_h must be >= 1 step")
    if n_steps > max_steps:
        raise ValueError(f"{n_steps} steps exceeds max_steps={max_steps}")
    dt_s = dt_h * 3600.0
    capped_warning = None
    if cfg["substrate_mode"] == "diffusive" and cfg["d_um2_per_s"] > 0:
        a_min = float(tissue.area[: tissue.n].min())
        dt_max = cfg["cfl_diff"] * a_min / (2.0 * math.sqrt(3.0)
                                            * cfg["d_um2_per_s"])
        need = max(1, math.ceil(dt_s / max(dt_max, 1e-300)))
        if need >= cfg["max_diff_substeps"]:
            capped_warning = (
                f"diffusive substrate under-resolved: dt={dt_s:g} s needs "
                f"{need} diffusion sub-steps but max_diff_substeps="
                f"{cfg['max_diff_substeps']}; stability is NOT guaranteed for "
                "this configuration")

    def record(step):
        a = tissue.area[: tissue.n]
        s = tissue.state[: tissue.n]
        lc = out.get("last_crowd", {})
        frames.append((
            step * dt_h,
            tissue.n,
            float(a.sum()),
            float(out["pool_su"]) if cfg["substrate_mode"] == "well_mixed"
            else float(tissue.nsub[: tissue.n].sum()),
            float(out["consumed_cum"]),
            float(a.mean()) if a.size else float("nan"),
            float(np.percentile(a, 10)) if a.size else float("nan"),
            float(np.percentile(a, 90)) if a.size else float("nan"),
            int((s == 0).sum()), int((s == 1).sum()), int((s == 2).sum()),
            int((s == 3).sum()), int(out["removed_cum"]),
            float(lc.get("interior_density_per_1000", float("nan"))),
            float(lc.get("gate_space_mean", float("nan"))),
            float(lc.get("mean_avail_um2", float("nan"))),
            int(counters["divisions"]),
        ))

    record(0)
    t = 0.0
    steps_done = 0
    for step in range(n_steps):
        if tissue.n == 0:
            break
        if callable(state):
            tissue.state[: tissue.n] = _resolve_state(state, tissue.n, t) \
                if state is not None else 0
        t = (step + 1) * dt_h
        _step(tissue, cfg, out, t * 3600.0, dt_s, rng,
              out['supplied_conc'], None, log_events, ev_log, counters,
              supply=out.get('supply'))
        steps_done = step + 1
        if (step + 1) % max(1, int(sample_every)) == 0 or step == n_steps - 1:
            record(step + 1)

    frames = np.asarray(frames, dtype=float)
    n_live = tissue.n
    a_final = tissue.area[:n_live].copy()
    pos_final = tissue.pos[:n_live].copy()
    warnings = []
    if capped_warning:
        warnings.append(capped_warning)
    if counters["divisions_truncated_by_max_cells"]:
        warnings.append("population hit max_cells; divisions were truncated")
    if counters["axis_isotropic_fallback"]:
        warnings.append(
            f"{counters['axis_isotropic_fallback']} division axes were drawn "
            "randomly because the local neighbour cloud was isotropic (a "
            "degenerate 'longest_axis'; documented, not hidden)")
    if counters["relax_clipped_cells"]:
        warnings.append(
            f"{counters['relax_clipped_cells']} relaxation displacement caps "
            "were hit (the soft relaxation is not a hard-core solver; L4)")
    if n_live and float(a_final.min()) <= 0:
        warnings.append("NON-POSITIVE cell area present -- bug, report it")

    # ---- overlap / geometry report ---------------------------------------
    ov = overlap_report(pos_final, a_final, cfg) if n_live > 1 else \
        dict(min_distance_ratio=float("nan"), n_pairs=0,
             n_pairs_overlapping=0, max_overlap_um=0.0, min_area_um2=float("nan"),
             n_nonpositive_area=0)

    budget = _budget(out, sub_info, cfg, tissue, supplies_total=
                     float(out["supplied_cum"]))

    meta = dict(
        seed=int(seed), n_steps=n_steps, steps_done=steps_done, dt_h=dt_h,
        t_end_h=t_end_h, n_cells_seed=int(n), n_cells_final=int(n_live),
        seeding_spacing_um=float(seed_d),
        area0_um2=float(area0_um2),
        mu_max_per_s=cfg["mu_max_per_s"],
        mu_max_per_h=cfg["mu_max_per_s"] * 3600.0,
        doubling_time_h=cfg["doubling_time_h"],
        cycle_time_h=cfg["cycle_time_h"],
        config={k: v for k, v in cfg.items() if k != "max_logged_events"},
        conc_source=conc_source,
        substrate_info=sub_info,
        counters=dict(counters),
        parameter_provenance=PARAMETER_PROVENANCE,
        param_sources=PARAM_SOURCES,
        literature_values={
            "mean_cell_area_um2": ("43.0", "project convention, model.py / "
                                   "mechanics.py"),
            "notum_epithelium_thickness_um": ("7.5-10", "Pinheiro 2017 "
                                              "PMC6143170, via MEASURED_ANCHORS.md"),
            "confluence_density_cells_per_1000_um2": (
                f"{RHO_C_PER_1000_UM2:.3f}",
                "1000/43, the project's published epithelial density, used as "
                "the contact-inhibition reference"),
        },
        warnings=warnings,
        limitations=LIMITATIONS,
        vertex_adapter_requirements=VERTEX_ADAPTER_REQUIREMENTS,
        wall_time_s=float(_time.perf_counter() - t_wall),
        complexity=("per step O(N log N + P + D); memory O(N); N cells, "
                    "P neighbour pairs inside the contact cutoff, D division "
                    "events"),
    )
    return dict(
        times_h=frames[:, 0], n_cells=frames[:, 1].astype(np.int64),
        total_area=frames[:, 2], total_substrate=frames[:, 3],
        substrate_consumed_cum=frames[:, 4], mean_area=frames[:, 5],
        p10_area=frames[:, 6], p90_area=frames[:, 7],
        n_healthy=frames[:, 8].astype(np.int64),
        n_reversible=frames[:, 9].astype(np.int64),
        n_irreversible=frames[:, 10].astype(np.int64),
        n_lysed=frames[:, 11].astype(np.int64),
        n_removed_cum=frames[:, 12].astype(np.int64),
        interior_density_per_1000_um2=frames[:, 13],
        mean_gate_space=frames[:, 14], mean_available_um2=frames[:, 15],
        cumulative_divisions=frames[:, 16].astype(np.int64),
        positions_final=pos_final, areas_final=a_final,
        state_final=tissue.state[:n_live].copy(),
        gid_final=tissue.gid[:n_live].copy(),
        acc_final=tissue.acc[:n_live].copy(),
        nsub_final=tissue.nsub[:n_live].copy(),
        subdivision_final=a_final * EPITHELIUM_THICKNESS_UM,
        events=dict(divisions=ev_log, n_logged=len(ev_log)),
        counts=dict(counters), budget=budget, overlap=ov, metadata=meta)


def _new_counters():
    import collections
    keys = ("divisions", "deleted", "pool_limited_steps",
            "substrate_clipped_cells", "relax_clipped_cells",
            "axis_isotropic_fallback", "div_third_party_overlaps",
            "divisions_truncated_by_max_cells", "diff_substeps_capped",
            "pending_divisions", "removed_irreversible", "removed_lysed")
    c = collections.Counter()
    for k in keys:
        c[k] = 0
    return c


def overlap_report(pos, area, cfg, tol=None):
    """Centre-based geometry validity: min pair distance / sum of radii.

    Returns the minimum ``d/(R_i+R_j)`` over all pairs closer than the contact
    cutoff, how many pairs are actually overlapping, the largest absolute
    overlap in um, and the smallest cell area (must be > 0).
    """
    pos = np.asarray(pos, float)
    area = np.asarray(area, float)
    n = len(pos)
    R = exclusion_radius(area, cfg["packing"])
    cutoff = float(2.0 * R.max() * 1.05) if n else 0.0
    if n < 2:
        return dict(min_distance_ratio=float("nan"), n_pairs=0,
                    n_pairs_overlapping=0, max_overlap_um=0.0,
                    min_area_um2=float(area.min()) if n else float("nan"),
                    n_nonpositive_area=int((area <= 0).sum()),
                    cutoff_um=cutoff)
    i, j, d = neighbour_pairs(pos, cutoff)
    if len(i) == 0:
        ratio = float("nan")
        return dict(min_distance_ratio=ratio, n_pairs=0,
                    n_pairs_overlapping=0, max_overlap_um=0.0,
                    min_area_um2=float(area.min()),
                    n_nonpositive_area=int((area <= 0).sum()),
                    cutoff_um=cutoff)
    rsum = R[i] + R[j]
    ratio = d / np.maximum(rsum, 1e-300)
    ov = rsum - d
    return dict(min_distance_ratio=float(ratio.min()), n_pairs=int(len(i)),
                n_pairs_overlapping=int((ov > 0).sum()),
                max_overlap_um=float(max(ov.max(), 0.0)),
                min_area_um2=float(area.min()),
                n_nonpositive_area=int((area <= 0).sum()),
                cutoff_um=cutoff,
                worst_pair=(int(i[int(np.argmin(ratio))]),
                            int(j[int(np.argmin(ratio))])))


def _budget(out, sub_info, cfg, tissue, supplies_total=0.0):
    """Substrate and biomass balances (see the docstring: only the field-level
    and biomass-level residuals are non-tautological)."""
    consumed = float(out["consumed_cum"])
    produced = float(out["produced_cum"])
    y = cfg["yield_su_per_um2"]
    impl = abs(consumed - y * produced) / max(abs(consumed), 1e-300)
    removed_biomass = float(out["removed_biomass_cum"])
    init_amt = sub_info["initial_amount"]
    final_amt = float(tissue.nsub[: tissue.n].sum()) \
        if cfg["substrate_mode"] == "diffusive" else float(out["pool_su"])
    if math.isfinite(init_amt):
        field_res = (init_amt + supplies_total - final_amt - consumed)
        field_res_rel = field_res / max(abs(init_amt), 1e-300)
    else:
        field_res = float("nan")
        field_res_rel = float("nan")
    return dict(
        yield_su_per_um2=y,
        consumed_su=consumed, biomass_produced_um2=produced,
        biomass_consumed_equivalent_um2=consumed / y if y > 0 else float("nan"),
        implementation_residual_rel=float(impl),
        initial_substrate_su=init_amt, final_substrate_su=final_amt,
        supplied_su=supplies_total,
        field_balance_residual_su=float(field_res),
        field_balance_residual_rel=float(field_res_rel),
        biomass_produced_total_um2=produced,
        biomass_removed_by_deletion_um2=removed_biomass,
    )


# ===========================================================================
# VERTEX-MESH DIVISION: the geometry bookkeeping a real vertex model needs
# ===========================================================================
# ``mechanics.py`` is imported READ-ONLY and never modified.  The operators
# below are written so that they work on (i) growth.RingMesh, a minimal
# polygon-ring mesh built inside this file, and (ii) a duck-typed
# ``mechanics.Mesh`` instance, which the self-test exercises for real.
#
# A vertex-cell division is NOT just "split the polygon": the division chord
# ends on two boundary edges of the parent, and those edges are shared with two
# neighbouring cells, so each neighbour must have the new vertex spliced into
# its own ring in the correct order, otherwise the half-edge twins no longer
# pair and the mesh is corrupt.  That is exactly what ``_insert_into_ring``
# below does, and it is verified with mechanics' own ``twins_ok`` /
# ``check_valid``.
# ===========================================================================
def polygon_centroid(pts):
    """Area centroid of a (possibly clockwise) polygon ring."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a2 = cr.sum()
    if abs(a2) < 1e-300:
        return pts.mean(axis=0)
    return np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (3.0 * a2)


def polygon_second_moment(pts):
    """Second-moment (covariance) matrix of a polygon about its centroid.

    Used by the 'longest_axis' division rule: the principal eigenvector of this
    matrix is the cell's long axis (Hertwig-like alignment of the division
    axis along the parent's long axis).
    """
    pts = np.asarray(pts, float)
    c = polygon_centroid(pts)
    p = pts - c
    q = np.roll(p, -1, axis=0)
    cr = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = 0.5 * cr.sum()
    xx = (p[:, 0] ** 2 + p[:, 0] * q[:, 0] + q[:, 0] ** 2)
    yy = (p[:, 1] ** 2 + p[:, 1] * q[:, 1] + q[:, 1] ** 2)
    xy = (p[:, 0] * q[:, 1] + 2.0 * p[:, 0] * p[:, 1] + 2.0 * q[:, 0] * q[:, 1]
          + p[:, 1] * q[:, 0])
    mxx = float((cr * xx).sum()) / 12.0
    myy = float((cr * yy).sum()) / 12.0
    mxy = float((cr * xy).sum()) / 24.0
    if a < 0:
        mxx, myy, mxy = -mxx, -myy, -mxy
    return np.array([[mxx / a if a else 0.0, mxy / a if a else 0.0],
                     [mxy / a if a else 0.0, myy / a if a else 0.0]]), c


def long_axis_angle(pts):
    """Angle (rad) of the polygon's long axis, or None when it is isotropic.

    Isotropic means the two principal moments differ by less than 2% of their
    mean; the caller then falls back to a random orientation (documented).
    """
    m, _ = polygon_second_moment(pts)
    mxx, myy, mxy = m[0, 0], m[1, 1], m[0, 1]
    tr = 0.5 * (mxx + myy)
    if abs(tr) < 1e-300:
        return None
    if math.hypot(mxx - myy, 2.0 * mxy) / abs(tr) < 0.02:
        return None
    return 0.5 * math.atan2(2.0 * mxy, mxx - myy)


def _seg_intersect(p1, p2, p3, p4, eps=1e-12):
    """True if closed segments p1p2 and p3p4 cross at an interior point."""
    d1 = p2 - p1
    d2 = p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < eps:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    s = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / den
    return (eps < t < 1.0 - eps) and (eps < s < 1.0 - eps)


def point_in_polygon(pt, pts):
    """Ray-casting point-in-polygon test (boundary counted as inside)."""
    x, y = float(pt[0]), float(pt[1])
    xp, yp = np.asarray(pts, float)[:, 0], np.asarray(pts, float)[:, 1]
    xn, yn = np.roll(xp, -1), np.roll(yp, -1)
    inside = False
    for i in range(len(xp)):
        if (yp[i] > y) != (yn[i] > y):
            xint = xp[i] + (y - yp[i]) * (xn[i] - xp[i]) / (yn[i] - yp[i])
            if x < xint:
                inside = not inside
    return inside


def plan_ring_split(ring_xy, angle_rad, centroid=None, tol_frac=1e-9):
    """Compute the division chord of a polygon ring.

    The chord is the straight line through the cell centroid (area centroid,
    NOT the vertex mean) with direction ``angle_rad``, clipped to the polygon.
    Returns a dict with::

        ok, reason
        centroid (2,), direction (2,), p (2,), q (2,)   the two chord ends
        p_edge, q_edge : ring edge indices the chord ends lie on (-1 never)
        p_frac, q_frac : position along that edge (0..1)
        p_vertex, q_vertex : ring position of an existing vertex when the chord
            ends exactly on a vertex (-1 when a genuinely new vertex is needed)
        p_new, q_new : bool, whether a new vertex has to be created
        n_vertex_hits : how many chord ends landed on an existing vertex
    """
    pts = np.asarray(ring_xy, float)
    k = len(pts)
    if k < 3:
        return dict(ok=False, reason="ring has <3 vertices")
    c = polygon_centroid(pts) if centroid is None else np.asarray(centroid, float)
    scale = math.sqrt(abs(float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                                        - pts[:, 1] * np.roll(pts[:, 0], -1))
                                * 0.5)))
    if not math.isfinite(scale) or scale <= 0:
        return dict(ok=False, reason="degenerate polygon (zero area)")
    tol_v = tol_frac * scale
    u = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    nrm = np.array([-u[1], u[0]])
    a = pts
    b = np.roll(pts, -1, axis=0)
    fa = (a - c) @ nrm
    fb = (b - c) @ nrm
    events = []
    for v in range(k):
        if abs(fa[v]) <= tol_v:
            events.append(dict(kind="vertex", vertex=v, edge=-1,
                               point=pts[v].copy(),
                               t=float((pts[v] - c) @ u), frac=None))
    den = fb - fa
    for i in range(k):
        if abs(den[i]) < 1e-300:
            continue
        s = -fa[i] / den[i]
        if not math.isfinite(s):
            continue
        if s <= tol_frac or s >= 1.0 - tol_frac:
            continue
        point = a[i] + s * (b[i] - a[i])
        events.append(dict(kind="edge", vertex=-1, edge=i, point=point,
                           t=float((point - c) @ u), frac=float(s)))
    # a vertex on the line makes its two incident edges also "cross"; drop the
    # spurious edge events whose crossing point is that same vertex
    events = [e for e in events
              if e["kind"] == "vertex"
              or all(np.hypot(*(e["point"] - w["point"])) > tol_v
                     for w in events if w["kind"] == "vertex")]
    if len(events) < 2:
        return dict(ok=False, reason=f"line crosses the ring {len(events)} time(s)")
    events.sort(key=lambda e: e["t"])
    en, ex = events[0], events[-1]
    if len(events) > 2:
        return dict(ok=False,
                    reason=f"line crosses the ring {len(events)} times "
                           "(non-convex ring: ambiguous chord)")
    if np.hypot(*(ex["point"] - en["point"])) <= tol_v:
        return dict(ok=False, reason="degenerate chord (zero length)")
    # the chord must lie inside the polygon: no crossing with another edge and
    # an interior midpoint
    for i in range(k):
        if i in (en["edge"], ex["edge"]):
            continue
        if _seg_intersect(en["point"], ex["point"], a[i], b[i]):
            return dict(ok=False, reason=f"chord crosses edge {i}")
    if not point_in_polygon(0.5 * (en["point"] + ex["point"]), pts):
        return dict(ok=False, reason="chord midpoint outside the polygon")
    return dict(ok=True, reason="ok", centroid=c, direction=u,
                p=en["point"], q=ex["point"], p_edge=en["edge"],
                q_edge=ex["edge"], p_frac=en["frac"], q_frac=ex["frac"],
                p_vertex=en["vertex"], q_vertex=ex["vertex"],
                p_new=(en["kind"] == "edge"), q_new=(ex["kind"] == "edge"),
                n_vertex_hits=int((en["kind"] == "vertex")
                                  + (ex["kind"] == "vertex")))


def split_rings(ring, p_slot, q_slot, p_edge, q_edge):
    """Daughter rings of ``ring`` cut by the chord ``p_slot -> q_slot``.

    ``p_slot``/``q_slot`` are vertex slots that ALREADY exist in the caller's
    vertex array; both are appended to the augmented ring in the position where
    the chord end lies: after ``ring[p_edge]`` for p and after ``ring[q_edge]``
    for q (p_edge/q_edge are -1 when that chord end is an existing ring vertex,
    in which case the slot is already in the ring).

    Returns ``(ringA, ringB, i_p, i_q, aug)`` -- both rings counter-clockwise,
    sharing the chord in opposite directions.  Ring A runs forward along the
    parent ring from p to q; ring B is the complementary walk.
    """
    ring = [int(v) for v in ring]
    k = len(ring)
    if (p_edge is None or p_edge < 0) and p_slot not in ring:
        raise ValueError("p_slot not in ring and no insertion edge given")
    if (q_edge is None or q_edge < 0) and q_slot not in ring:
        raise ValueError("q_slot not in ring and no insertion edge given")
    aug = []
    ip = iq = -1
    for i in range(k):
        v = ring[i]
        if p_edge == i:
            ip = len(aug)
            aug.append(int(p_slot))
        if q_edge == i:
            iq = len(aug)
            aug.append(int(q_slot))
        if v == p_slot and ip < 0:
            ip = len(aug)
        if v == q_slot and iq < 0:
            iq = len(aug)
        aug.append(v)
    if ip < 0 or iq < 0:
        raise ValueError("chord end not placed in the ring")
    if ip == iq:
        raise ValueError("both chord ends at the same ring position")
    m = len(aug)
    if ip < iq:
        ringA = [aug[ip]] + aug[ip + 1: iq + 1]
        ringB = [aug[iq]] + aug[iq + 1:] + aug[: ip + 1]
    else:
        ringB = [aug[iq]] + aug[iq + 1: ip + 1]
        ringA = [aug[ip]] + aug[ip + 1:] + aug[: iq + 1]
    return ringA, ringB, ip, iq, aug


def _insert_into_ring(ring, a, b, new_slot):
    """Splice ``new_slot`` between the consecutive pair {a,b} of ``ring``.

    ``ring`` is the neighbour's ring; the parent's edge is directed (a -> b),
    and the neighbour traverses it in the opposite direction, so the insertion
    is (b, new, a) in the neighbour's order -- but this function does not need
    to know that: it finds the adjacent pair in whatever order the neighbour
    stores it and inserts the new vertex between them.  That is what keeps the
    half-edge twins paired.
    """
    ring = [int(v) for v in ring]
    k = len(ring)
    for i in range(k):
        u, v = ring[i], ring[(i + 1) % k]
        if {u, v} == {int(a), int(b)}:
            out = ring[: i + 1] + [int(new_slot)] + ring[i + 1:]
            return out, i + 1
    return None, -1


class MeshDivisionError(RuntimeError):
    """Raised when the division surgery would corrupt the mesh."""


class RingMesh:
    """Minimal polygon-ring mesh used to demonstrate vertex division.

    Own construction (no dependence on mechanics.py): a vertex array, a list of
    counter-clockwise rings, and the two invariants that matter -- every
    directed edge has its reverse present exactly once (twin pairing), and no
    ring self-intersects.
    """

    def __init__(self, positions, rings):
        self.positions = np.array(positions, dtype=float, copy=True)
        self.rings = [list(map(int, r)) for r in rings]
        self.alive = np.ones(len(self.rings), dtype=bool)
        self.parents = [int(i) for i in range(len(self.rings))]
        self.divisions = []

    @classmethod
    def from_mechanics_hex(cls, n_side=5, area=A_REF_UM2):
        """Build the ring mesh from ``mechanics.build_hex_mesh`` (read-only)."""
        if _mech is None:
            raise ImportError(f"mechanics.py not importable: {_MECH_IMPORT_ERROR}")
        pos, cells, _ids = _mech.build_hex_mesh(n_side, area)
        return cls(pos, cells)

    # -- geometry -----------------------------------------------------------
    def ring_xy(self, c):
        return self.positions[self.rings[c]]

    def area(self, c):
        return abs(_mech.polygon_area(self.ring_xy(c))) if _mech is not None \
            else abs(0.5 * float(np.sum(
                self.ring_xy(c)[:, 0] * np.roll(self.ring_xy(c)[:, 1], -1)
                - self.ring_xy(c)[:, 1] * np.roll(self.ring_xy(c)[:, 0], -1))))

    def total_area(self):
        return float(sum(self.area(c) for c in range(len(self.rings))
                         if self.alive[c]))

    def add_vertex(self, xy):
        self.positions = np.vstack([self.positions, np.asarray(xy, float)[None, :]])
        return len(self.positions) - 1

    # -- invariants ---------------------------------------------------------
    def edge_map(self):
        """(a,b) -> list of cells owning the directed edge a->b."""
        m = {}
        for c, r in enumerate(self.rings):
            if not self.alive[c]:
                continue
            k = len(r)
            for i in range(k):
                m.setdefault((r[i], r[(i + 1) % k]), []).append(c)
        return m

    def check(self):
        """Structure/geometry validity of the whole mesh."""
        m = self.edge_map()
        unpaired = [e for e, cs in m.items() if len(cs) != 1
                    or (e[1], e[0]) not in m]
        dup = [e for e, cs in m.items() if len(cs) > 1]
        self_int = []
        neg = []
        for c, r in enumerate(self.rings):
            if not self.alive[c]:
                continue
            pts = self.ring_xy(c)
            if len(set(r)) != len(r):
                self_int.append(("repeated vertex", c))
                continue
            ar = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                                    - pts[:, 1] * np.roll(pts[:, 0], -1)))
            if ar <= 0:
                neg.append(c)
            if len(r) >= 3:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        if j == i + 1 or (i == 0 and j == len(r) - 1):
                            continue
                        if _seg_intersect(pts[i], pts[(i + 1) % len(r)],
                                          pts[j], pts[(j + 1) % len(r)]):
                            self_int.append(("self-intersection", c))
        return dict(ok=(not unpaired and not dup and not self_int and not neg),
                    n_unpaired_edges=len(unpaired),
                    n_duplicate_edges=len(dup), n_self_intersections=len(self_int),
                    n_nonpositive_area=len(neg),
                    examples=(unpaired[:3], dup[:3], self_int[:3]),
                    n_cells=int(self.alive.sum()),
                    n_vertices=int(len(self.positions)))

    # -- the operator -------------------------------------------------------
    def divide_cell(self, c, angle=None, rule="longest_axis", rng=None,
                    tol_frac=1e-9):
        """Split cell ``c`` into two daughters along a chord.

        Returns a dict with ``ok``, the daughter ring indices, the new vertex
        slots, the area before/after and the relative area error.  The parent
        keeps index ``c`` (daughter 1) and daughter 2 is appended.
        """
        if not self.alive[c]:
            return dict(ok=False, reason="cell is not alive")
        pts = self.ring_xy(c)
        if angle is None:
            if rule == "longest_axis":
                angle = long_axis_angle(pts)
                if angle is None:
                    if rng is None:
                        return dict(ok=False, reason="isotropic cell and no rng")
                    angle = float(rng.uniform(0.0, math.pi))
            elif rule == "random":
                if rng is None:
                    return dict(ok=False, reason="rule='random' needs an rng")
                angle = float(rng.uniform(0.0, math.pi))
            elif rule == "prescribed":
                angle = 0.0
            else:
                return dict(ok=False, reason=f"unknown rule {rule!r}")
        plan = plan_ring_split(pts, angle, tol_frac=tol_frac)
        if not plan["ok"]:
            return dict(ok=False, reason=plan["reason"], plan=plan)
        area_before = self.area(c)
        new_slots = {}
        for tag in ("p", "q"):
            if plan[f"{tag}_new"]:
                new_slots[tag] = self.add_vertex(plan[tag])
            else:
                new_slots[tag] = self.rings[c][plan[f"{tag}_vertex"]]
        ring = self.rings[c]
        ringA, ringB, _ip, _iq, _aug = split_rings(
            ring, new_slots["p"], new_slots["q"], plan["p_edge"],
            plan["q_edge"])
        # every OTHER cell that shares a split parent edge must have the new
        # vertex spliced into its own ring, or the twins stop pairing
        emap = self.edge_map()
        for tag in ("p", "q"):
            edge = plan[f"{tag}_edge"]
            if edge is None or edge < 0 or not plan[f"{tag}_new"]:
                continue
            a_, b_ = ring[edge], ring[(edge + 1) % len(ring)]
            owners = emap.get((a_, b_), []) + emap.get((b_, a_), [])
            for n in owners:
                if n == c:
                    continue
                newring, pos = _insert_into_ring(self.rings[n], a_, b_,
                                                 new_slots[tag])
                if newring is None:
                    return dict(ok=False,
                                reason=f"neighbour {n} does not own edge "
                                       f"({a_},{b_}) -- mesh was inconsistent")
                self.rings[n] = newring
        self.rings[c] = ringA
        self.rings.append(ringB)
        self.alive = np.append(self.alive, True)
        self.parents.append(c)
        a1, a2 = self.area(c), self.area(len(self.rings) - 1)
        rec = dict(ok=True, parent=c, d1=c, d2=len(self.rings) - 1,
                   area_before=area_before, area_d1=a1, area_d2=a2,
                   area_error=abs(area_before - (a1 + a2)),
                   area_error_rel=abs(area_before - (a1 + a2)) / area_before,
                   angle_deg=math.degrees(float(angle)),
                   new_vertices={k: int(v) for k, v in new_slots.items()},
                   n_vertex_hits=plan["n_vertex_hits"])
        self.divisions.append(rec)
        return rec


def divide_in_mechanics_mesh(mesh, c, angle=None, rule="longest_axis", rng=None,
                             tol_frac=1e-9, allow_cut_split=False,
                             new_cell_id=-1):
    """Apply the SAME division surgery to a real ``mechanics.Mesh`` instance.

    ``mesh`` is duck-typed on the attributes mechanics uses
    (``positions``, ``cells``, ``cell_alive``, ``cell_ids``, ``n_slots``,
    ``rebuild``, ``he_of``, ``mark_cut``, ``twins_ok``, ``check_valid``).  The
    mesh object is mutated in place -- ``mechanics.py`` itself is not touched.

    Bookkeeping performed:
      * new vertices are APPENDED to ``mesh.positions`` and ``mesh.n_slots`` is
        updated (mechanics never reuses or removes vertex slots);
      * the parent ring is replaced by daughter 1's ring and daughter 2's ring
        is appended, with ``cell_alive`` and ``cell_ids`` extended;
      * every neighbouring cell that shares a split edge gets the new vertex
        spliced into its ring;
      * ``mesh.rebuild()`` is called, which reconstructs the whole half-edge
        table, and then ``twins_ok()`` / ``check_valid()`` are used to VERIFY
        the result.  If the split would corrupt the mesh it is rolled back and
        ``ok=False`` is returned.
    """
    if _mech is None:
        raise ImportError(f"mechanics.py not importable: {_MECH_IMPORT_ERROR}")
    ring = list(map(int, mesh.cells[c]))
    pts = mesh.positions[ring]
    if not allow_cut_split and len(mesh.he_from):
        cut = set()
        for h in np.flatnonzero(mesh.he_cut):
            cut.add((int(mesh.he_from[h]), int(mesh.he_to[h])))
        own = {(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))}
        if (own & cut) or ({(b, a) for a, b in own} & cut):
            return dict(ok=False, reason="cell owns a cut (wound) edge; "
                                         "pass allow_cut_split=True to force")
    if angle is None:
        angle = long_axis_angle(pts)
        if angle is None:
            if rng is None:
                return dict(ok=False, reason="isotropic cell and no rng")
            angle = float(rng.uniform(0.0, math.pi))
    plan = plan_ring_split(pts, angle, tol_frac=tol_frac)
    if not plan["ok"]:
        return dict(ok=False, reason=plan["reason"])
    snap_pos = mesh.positions.copy()
    snap_cells = [list(r) for r in mesh.cells]
    snap_alive = mesh.cell_alive.copy()
    snap_ids = mesh.cell_ids.copy()
    snap_nslots = mesh.n_slots
    snap_cut = mesh.he_cut.copy() if len(mesh.he_from) else np.zeros(0, bool)
    snap_from = mesh.he_from.copy()
    snap_to = mesh.he_to.copy()
    try:
        area_before = abs(_mech.polygon_area(pts))
        slots = {}
        for tag in ("p", "q"):
            if plan[f"{tag}_new"]:
                mesh.positions = np.vstack([mesh.positions,
                                            plan[tag][None, :]])
                slots[tag] = len(mesh.positions) - 1
            else:
                slots[tag] = ring[plan[f"{tag}_vertex"]]
        mesh.n_slots = int(len(mesh.positions))
        ringA, ringB, _ip, _iq, _aug = split_rings(
            ring, slots["p"], slots["q"], plan["p_edge"], plan["q_edge"])
        for tag in ("p", "q"):
            edge = plan[f"{tag}_edge"]
            if edge is None or edge < 0 or not plan[f"{tag}_new"]:
                continue
            a_, b_ = ring[edge], ring[(edge + 1) % len(ring)]
            owners = []
            for cc, rr in enumerate(mesh.cells):
                if not mesh.cell_alive[cc] or cc == c:
                    continue
                k = len(rr)
                for i in range(k):
                    if {rr[i], rr[(i + 1) % k]} == {a_, b_}:
                        owners.append(cc)
            for n in owners:
                newring, _ = _insert_into_ring(mesh.cells[n], a_, b_,
                                               slots[tag])
                if newring is None:
                    raise MeshDivisionError(
                        f"neighbour {n} does not own edge ({a_},{b_})")
                mesh.cells[n] = newring
        mesh.cells[c] = ringA
        mesh.cells.append(ringB)
        mesh.cell_alive = np.append(mesh.cell_alive, True)
        mesh.cell_ids = np.append(mesh.cell_ids, int(new_cell_id))
        mesh.rebuild()
        if allow_cut_split and len(snap_cut):
            for tag in ("p", "q"):
                if not plan[f"{tag}_new"]:
                    continue
                edge = plan[f"{tag}_edge"]
                a_, b_ = ring[edge], ring[(edge + 1) % len(ring)]
                for pair in ((a_, slots[tag]), (slots[tag], b_)):
                    h = mesh.he_of(*pair)
                    if h >= 0:
                        mesh.mark_cut(h, True)
        twin_ok, twin_msg = mesh.twins_ok()
        valid = mesh.check_valid(l_min_edge=0.0, a_min=0.0,
                                 crossing_tol=1e-6 * math.sqrt(A_REF_UM2))
        a1 = abs(_mech.polygon_area(mesh.positions[mesh.cells[c]]))
        d2 = len(mesh.cells) - 1
        a2 = abs(_mech.polygon_area(mesh.positions[mesh.cells[d2]]))
        if not twin_ok or not valid["ok"]:
            raise MeshDivisionError(f"twin_ok={twin_ok} ({twin_msg}) "
                                    f"valid={valid}")
    except Exception as exc:                       # roll back, never corrupt
        mesh.positions = snap_pos
        mesh.cells = snap_cells
        mesh.cell_alive = snap_alive
        mesh.cell_ids = snap_ids
        mesh.n_slots = snap_nslots
        mesh.rebuild()
        if len(snap_cut):
            for h, (a_, b_) in enumerate(zip(snap_from, snap_to)):
                if snap_cut[h]:
                    hh = mesh.he_of(int(a_), int(b_))
                    if hh >= 0:
                        mesh.mark_cut(hh, True)
        return dict(ok=False, reason=f"{type(exc).__name__}: {exc}", rolled_back=True)
    return dict(ok=True, parent=c, d1=c, d2=d2, area_before=area_before,
                area_d1=a1, area_d2=a2,
                area_error=abs(area_before - (a1 + a2)),
                area_error_rel=abs(area_before - (a1 + a2)) / area_before,
                angle_deg=math.degrees(float(angle)),
                new_vertices={k: int(v) for k, v in slots.items()},
                n_vertex_hits=plan["n_vertex_hits"],
                twin_ok=bool(twin_ok), twin_message=twin_msg,
                valid=valid, n_cells_after=int(mesh.cell_alive.sum()),
                n_vertices_after=int(mesh.n_slots))


def vertex_division_demo(n_side=5, n_division_rounds=3, seed=2025,
                         verbose=False):
    """Divide cells on a synthetic vertex mesh of this file's own construction.

    Builds the hexagonal patch with ``mechanics.build_hex_mesh`` (read-only,
    same 43 um^2 cells as ``mechanics.run``), wraps it in a real
    ``mechanics.Mesh``, and divides cells with the surgery above.  Every
    division is checked for area conservation, ring validity, twin pairing and
    mesh validity (mechanics' own ``check_valid``).  This is the evidence that
    the vertex-level operator in this file is not just a plan.
    """
    if _mech is None:
        return dict(ok=False, reason=f"mechanics.py unavailable: {_MECH_IMPORT_ERROR}")
    pos, cells, ids = _mech.build_hex_mesh(n_side, A_REF_UM2)
    mesh = _mech.Mesh(pos, cells, ids)
    rng = np.random.default_rng(int(seed))
    area0 = mesh.geometry()["areas"][mesh.cell_alive].sum()
    results = []
    cent = mesh.positions[mesh.alive_vertex_slots()].mean(axis=0)
    order = [int(c) for c in np.argsort(
        np.linalg.norm(mesh.geometry()["centroids"] - cent, axis=1))]
    ok_count = 0
    for c in order[: int(n_division_rounds)]:
        if not mesh.cell_alive[c]:
            continue
        r = divide_in_mechanics_mesh(mesh, c, rule="longest_axis", rng=rng)
        results.append(r)
        ok_count += int(r["ok"])
        if verbose:
            print(f"  [vertex] divide cell {c}: {r.get('reason', 'ok')}")
    g = mesh.geometry()
    area1 = g["areas"][mesh.cell_alive].sum()
    a_alive = g["areas"][mesh.cell_alive]
    errs = [r["area_error_rel"] for r in results if r.get("ok")]
    valid = mesh.check_valid(l_min_edge=0.0, a_min=0.0,
                             crossing_tol=1e-6 * math.sqrt(A_REF_UM2))
    twin_ok, twin_msg = mesh.twins_ok()
    return dict(ok=bool(ok_count == len(results) and twin_ok and valid["ok"]),
                n_attempted=len(results), n_succeeded=int(ok_count),
                reasons=[r.get("reason") for r in results if not r["ok"]],
                area_before_um2=float(area0), area_after_um2=float(area1),
                total_area_error_um2=float(abs(area0 - area1)),
                total_area_error_rel=float(abs(area0 - area1) / area0),
                max_division_area_error_rel=float(max(errs)) if errs else 0.0,
                min_cell_area_um2=float(a_alive.min()),
                max_cell_area_um2=float(a_alive.max()),
                twin_ok=bool(twin_ok), twin_message=twin_msg,
                mesh_valid=bool(valid["ok"]), check_valid=valid,
                n_cells=int(mesh.cell_alive.sum()),
                n_vertices=int(valid["n_vertices"]),
                n_degree1=int(valid["n_degree1"]),
                divisions=results)


# __APPEND_4__
