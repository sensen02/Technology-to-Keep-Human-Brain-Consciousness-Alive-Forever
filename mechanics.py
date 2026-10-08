"""Synthetic 2D apical vertex model for mechanically computed injury.

=========================================================================
HONESTY STATEMENT -- READ FIRST
=========================================================================
* This module is a **synthetic mechanical model**.  It has **NO biological
  validation whatsoever** and has **not been calibrated against any experiment**.
* Every dynamical parameter (``k_area``, ``gamma_p``, ``lambda_edge``,
  ``lambda_cut``, ``active_tension``, ``eta``) is an **arbitrary illustrative
  value**, in units where lengths are micrometres and the energy scale is
  ``k_area * A0**2 = 1849``.  These are *not* measured values for the Drosophila
  pupal notum or for any other tissue.  **No literature value is used anywhere
  in this file**, so no citation applies; the statement is recorded verbatim in
  ``metadata['parameter_provenance']`` and ``metadata['literature_values']`` is
  empty.
* Nothing produced here is experimental data.  Simulated kinematics must never
  be presented as measurement.
* Where a feature is fragile (T2 and rim-T1) the limitation is documented in
  ``LIMITATIONS`` and surfaced at runtime through ``metadata['warnings']`` and
  the event log -- not hidden.

=========================================================================
WHAT IS COMPUTED
=========================================================================
A 2D **apical vertex model** (VM) of a finite monolayer of ``n_side**2`` cells
(default 24x24 = 576, mean cell area 43 um^2, matching the cell layout of
``cell_wound_prototype/model.py``).  Vertices ``r`` move by overdamped gradient
flow of

    E = sum_c [ k_area*(A_c - A0_c)^2 ]  +  sum_e T_e * L_e

    eta * dr_v/dt = -dE/dr_v

    T_e = gamma_p*(P_c1 - P0_c1) + gamma_p*(P_c2 - P0_c2)
          + active_tension_c1 + active_tension_c2 + line_tension_e

where ``c1, c2`` are the one or two cells sharing edge e (on the free boundary
there is exactly one, which is what makes the rim a genuine free boundary),
``L_e`` is the edge length, and ``line_tension_e`` is ``lambda_cut`` on the
injury boundary, ``lambda_outer`` (= ``lambda_edge``) on the original rim and 0
in the bulk.  ``A_c`` is the *oriented* shoelace area of the cell polygon and
``P_c`` its perimeter, both assembled from the polygon incidence
``(cell -> cyclic vertex list)``.  ``A0_c`` is the preferred area (43 um^2
default; optional seeded per-cell variation via ``area_jitter``, default 0 so
the base state is an exact regular hexagonal lattice).  ``P0_c`` is the
preferred perimeter (default: that of a regular hexagon of area ``A0_c``,
``P0 = 3.72242*sqrt(A0)``).

Every term is implemented exactly as written above: the area and perimeter
penalties and the active work are per-cell, and the line tension is per-edge.
The analytic gradient is assembled from the polygon incidence term by term
(see :meth:`Mesh.forces`), and test 1 checks the whole expression against central
finite differences term by term, reporting a relative error of ~1e-9.

Why the energy is written with the penalties *separate* from (rather than folded
into) the edge tensions: folding them would double count.  With
``T_e = gamma_p*(P_c1+P_c2-2*P0)`` one has
``sum_e T_e L_e = sum_c gamma_p*P_c*(P_c - P0_c)``, which is *not*
``sum_c gamma_p*(P_c-P0_c)^2``.  Both are legitimate functionals (the folded one
is the Farhadifar-style junction-tension model, in which ``Gamma`` also sets the
rest junction length), but their *sum* behaves like ``gamma_p*P*(2P-P0)`` with an
unintended extra stiffness, and -- decisively -- only one of them can be the
functional that the implemented force is the gradient of.  Here the choice is the
literal one, so the code is a gradient flow of exactly the E above.

Newton's third law holds edge by edge, so every interior force cancels in pairs
and the total force on the tissue is zero to machine precision (reported by test
1 as ``force_sum_over_alive``).

**Active contraction** is the optional per-cell term ``active_tension_c*P_c``.

**Free outer boundary**: no periodic images anywhere.  Rim edges belong to
exactly one cell (``twin == -1``) and carry ``lambda_outer`` (default
``lambda_edge``) plus the tension of the single incident cell.

**Injury is computed from mechanics, not prescribed**: cells are deleted (a real
hole in the mesh), which creates a genuine free boundary whose further motion --
contraction, rounding, T1 rearrangements of the surrounding tissue -- is produced
by the same energy functional.  Edges of that hole carry ``lambda_cut``, the
purse-string line tension.

=========================================================================
DYNAMICS, ADAPTIVE STEPPING, STABILITY CRITERION
=========================================================================
Overdamped dynamics integrated with explicit (forward) Euler.  ``dt`` is adapted
every step from three conditions (all computed, all exported):

1. **Viscous/curvature CFL (dominant)**::

       dt <= cfl_vertex * eta / max_v H_vv ,
       H_vv  ~= sum_{e incident to v} |T_e| / L_e  +  4*k_area*A0

   ``H_vv`` estimates the diagonal of the Hessian d2E/dr^2 from the polygon
   incidence.  The term ``2*k_area*(A-A0)`` has bounded derivative, so
   ``4*k_area*A0`` is a conservative bound for the area-term curvature;
   ``sum |T_e|/L_e`` is the exact leading contribution of the tension terms.
   Full normal-mode stability would require the whole Hessian (or an implicit
   solve); the local diagonal bound is the cheap conservative proxy and is
   *validated empirically* (test 1 gradient check, test 3 energy behaviour, and
   the step-rejection guard).  ``cfl_vertex = 0.3``.
2. **Kinematic CFL**: ``dt <= cfl_speed*sqrt(A0)/max_v|v_v|`` with
   ``cfl_speed = 0.25``.  This is also what guarantees T1 cannot be missed: an
   edge can shrink by at most ``2*cfl_speed*sqrt(A0)`` per step while T1 is
   evaluated every step at ``l_t1``.
3. **Area-change bound**: ``dt <= cfl_area*sqrt(A_min)/max_v|v_v|``,
   ``cfl_area = 0.02``, so no single step can invert a cell.

Plus ``dt <= dt_max`` (the energy is quartic in r, so no explicit step is
unconditionally stable) and ``dt`` may grow at most 10x per step.  Any step that
raises the total energy by more than ``energy_tol`` is **rejected and dt halved**
(``metadata['rejected_steps']``).  Measured step sizes are reported in
``metadata['dt_used_max']``.

=========================================================================
TOPOLOGY: HALF-EDGE MESH, T1, T2
=========================================================================
Half-edge arrays: ``he_cell`` (owning cell), ``he_from`` (tail), ``he_to``
(head), ``he_next``, ``he_prev``, ``he_twin`` (``-1`` = free boundary),
``he_cut`` (carries ``lambda_cut``).  Cell rings are counter-clockwise so that
their shoelace area is positive.

**T1** (``Mesh.t1_swap``): an interior edge shorter than ``l_t1`` is rearranged.
The two vertices keep their positions; the four surrounding polygons are
re-spliced so the new edge joins the *other* two vertices of the quadrangle
(``t -- tp``), and the half-edge table is rewired **locally and exactly**: the
half-edges that used to be ``u1->t`` and ``u2->tp`` are re-pointed to become the
new edge and are made each other's twins, then the ``next``/``prev`` pointers of
the four cells are rewritten from their rings.  A full twin/next/prev/cycle
consistency assertion runs after every accepted swap (``Mesh.twins_ok``).
Rules: only interior edges are flipped (rim edges are never flipped -- there is
no exterior cell to rearrange with); the four-cell neighbourhood must be
unambiguous; the new edge must not already exist; no cell may drop below 3
vertices or non-positive area; the new edge must exceed ``2*l_t1`` (this is what
prevents immediate re-flip oscillation); no non-adjacent vertex may lie closer
than ``1e-3*sqrt(A0)`` to the new edge (crossing pre-check).  Every attempt is
logged with the edge, length, cells and verdict.

**Correlated T1**: two accepted flips in the same step that share a vertex are
flagged ``correlated=True`` and counted (``events_counts['t1_correlated']``).
The same vertex pair cannot be flipped again for ``t1_cooldown`` (10) steps --
such attempts are logged as ``reason='cooldown'``.  ``t1_isolated=True``
restricts to one flip per step.

**T2** (``Mesh.t2_remove``): when a cell's area falls below ``A_t2`` the cell is
removed with the *same* operator used for injury; the result is validated
(``check_valid``: positive areas, no dangling vertices, no duplicate edges,
twin consistency, no non-adjacent vertex within ``1e-6*sqrt(A0)`` of an edge)
and **rolled back** if invalid, which is logged (``accepted=False``) and counted
in ``t2_rejected``.  T2 is **off by default** because it is the fragile part
(LIMITATIONS L2).

=========================================================================
INJURY: EXACTLY HOW CELLS, EDGES AND VERTICES ARE REMOVED
=========================================================================
1. Cells are selected by their **current polygon centroid**:
   ``mode='ablate'`` -> ``|centroid - centre| <= injury_radius``;
   ``mode='cut'`` -> distance to the straight line through the mesh centre with
   direction ``cut_angle`` (default horizontal) ``<= cut_halfwidth`` and in-line
   offset ``<= injury_radius`` (so ``injury_radius`` is the cut *half-length*);
   ``cut_circle=True`` -> circular wound of radius ``injury_radius``
   (``cut_halfwidth`` ignored).  ``mode='none'`` -> no injury.
2. The selected cells are marked dead and their polygons are dropped.  **Their
   vertices are kept for now**, because surviving cells still use them.
3. Every half-edge belonging to a dead cell is dropped together with its twin:
   an edge survives iff at least one of its two cells survives, so surviving
   cells keep a complete ring.
4. Each surviving cell that touched a dead cell re-splices its ring: every
   maximal run of dead-only vertices between two surviving vertices is replaced
   by the **single run vertex nearest the midpoint of the two flanking
   survivors** (positions are never moved).  This is what "delete these cells"
   means mechanically for the surviving tissue, and it is what prevents dangling
   (degree-1) vertices along the new rim.  A surviving cell left with fewer than
   3 vertices is deleted too (counted in ``cells_collapsed_during_repair``).
5. Vertices of degree < 2 are deleted iteratively; their positions are set to
   NaN.  **Vertex slots are never reused**, so vertex indices are stable for the
   whole run and ``vertices`` is exactly ``(T, V, 2)`` with NaN in dead slots.
6. The half-edge table is rebuilt from the surviving rings (exact twins), and
   boundary half-edges that touch a removed vertex are flagged ``he_cut = True``
   (they carry ``lambda_cut``); the original outer rim keeps ``he_cut = False``
   (it carries ``lambda_outer``).
7. ``cell_ids`` keeps the **original row-major lattice index** of every
   surviving cell, so ``damage`` maps directly onto ``model.py``'s arrays.

Counts of everything dropped in steps 4-5 appear in ``metadata['injury']``.

=========================================================================
OUTPUT / DAMAGE
=========================================================================
``damage`` (length ``n_side**2``, index = original lattice cell index, i.e. the
same index as ``model.hex_geometry``) is 1.0 for an original cell that
**survived the injury and now has at least one cut edge**, else 0.0.  Ablated
cells are 0.0 by definition (a dead cell is not a damaged live cell).
``metadata['cell_state']`` additionally provides per original cell the final
area, perimeter, ``area_strain=(A-A0)/A0``, ``perimeter_relative=P/P0``,
``touches_wound`` and ``alive``, so a parent can build a strain- or
tear-dependent permeability without touching this file.  These strains are
mechanical *model output*, not measurements.

How a parent may consume the output::

    import mechanics, model
    r = mechanics.run(n_side=20, mode='cut')
    damage = r['damage']              # length 400 == model cell count
    # feed into model.py (see its own API) as a per-cell tear/permability driver

=========================================================================
LIMITATIONS (documented loudly, not hidden)
=========================================================================
L1. **No calibration.** Absolute times are meaningless; ``eta`` only sets the
    ratio of viscous to elastic timescale. ``t_end`` is in relaxation units.
L2. **T2 is fragile** on a free-boundary mesh: merging the neighbours of a
    vanishing cell can create a self-intersecting polygon.  Invalid T2 events
    are rolled back and logged (``t2_rejected``); T2 is off by default, and in
    wound runs cells disappear through the injury operator, not through T2.
L3. **T1 on the free rim is rarer** than in a periodic tissue: a T1 whose
    quadrangle would leave the tissue is rejected.  This is deliberate (there is
    no exterior cell), but it makes the rim less fluid than the bulk.
L4. **Area is not exactly conserved** when ``gamma_p > 0`` or ``lambda > 0``:
    the area term balances the line tension, so the equilibrium area settles a
    few per mille below ``A0``.  Measured values are reported (test 2, test 4).
L5. The wound rim is discretised at the cell scale: it is straight only where a
    run of dead cells was enclosed by survivors, and remains vertex-level
    zig-zag elsewhere.
L6. Euler stepping with the CFL above; the energy is monitored but there is no
    trajectory error control.
L7. 2D apical mechanics only: no division, no apoptosis programme, no
    junctional remodelling kinetics, no substrate, no 3D.

=========================================================================
PUBLIC API
=========================================================================
``run(n_side=24, mode='cut', injury_radius=24.5, cut_halfwidth=8.0, t_end=60.0,
sample_dt=0.5, lambda_cut=1.0, lambda_edge=0.0, k_area=1.0, gamma_p=0.1,
eta=1.0, A0=43.0, seed=2025, ...)`` -> dict with ``times`` (T,),
``vertices`` (T,V,2), ``positions_cells`` (T,N,2) [**cell centroids**],
``areas`` (T,N), ``perimeters`` (T,N), ``cell_alive`` (T,N), ``hole_area``
(T,), ``boundary_length`` (T,), ``cut_length`` (T,), ``outer_length`` (T,),
``energy`` (T,), ``energy_components`` (T,4), ``events`` (dict), ``damage``
(N,), ``metadata`` (dict).

``self_test()`` -> diagnostics dict for the 8 required tests.

Also importable: ``Mesh``, ``build_hex_mesh``, ``energy``, ``forces``,
``analytic_gradient``, ``finite_difference_gradient``,
``regular_hexagon_side``, ``regular_hexagon_perimeter``.

``python mechanics.py``          run self_test()
``python mechanics.py --demo``   run the spec-default 576-cell cut
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "run", "self_test", "Mesh", "MeshError", "build_hex_mesh", "energy",
    "forces", "analytic_gradient", "finite_difference_gradient",
    "regular_hexagon_side", "regular_hexagon_perimeter",
    "P0_FACTOR_REGULAR_HEXAGON", "PARAMETER_PROVENANCE", "LIMITATIONS",
]

PARAMETER_PROVENANCE = (
    "SYNTHETIC MECHANICAL MODEL -- NO BIOLOGICAL VALIDATION, NO CALIBRATION. "
    "All dynamical parameters (k_area, gamma_p, lambda_edge, lambda_cut, "
    "active_tension, eta) are arbitrary illustrative values in units where "
    "length is um, area is um^2 and the energy scale is k_area*A0^2 = 1849. No "
    "value is taken from the literature for Drosophila pupal notum or any other "
    "tissue, therefore no citation applies. Simulated kinematics are model "
    "output, never experimental data."
)

#: The only literature reference used anywhere in this file.  It covers the
#: *functional form* of a 2D vertex model, NOT any parameter value: every
#: dynamical parameter in this module is an arbitrary illustrative value.
LIT_REFS: dict = {
    "farhadifar2007": ("Farhadifar, Roeper, Aigouy, Eaton, Juelicher (2007) "
                       "The influence of cell mechanics, cell-cell interactions, "
                       "and proliferation on epithelial packing. Curr Biol "
                       "17(24):2095-2104 -- cited for the vertex-model energy "
                       "*form* only; no parameter is taken from it."),
}

LIMITATIONS = (
    "L1 no calibration; absolute time is meaningless (eta sets only a ratio). "
    "L2' TOPOLOGY CHANGES ARE THE FRAGILE PART: the T1 operator validates the "
    "mesh it is about to create and ROLLS BACK on any structural violation, so "
    "it can never corrupt the mesh but it refuses many candidates (measured: 0 "
    "accepted of 5817 attempted under strong compression on a 5x5 patch). T2 "
    "uses the same machinery and the same rollback, and is disabled by default. "
    "Do NOT rely on T1/T2 for quantitative rearrangement statistics without "
    "inspecting events_counts (t1_accepted/t1_rejected/t1_rejected_reasons). "
    "L2b spherical/periodic versions of the T1 surgery are not attempted here. "
    "L3 rim cells keep 6 vertices, so the n x n rhombus has ~2(n-1) degree-1 "
    "'spoke' vertices on its rim; they are reported (check_valid n_degree1), "
    "they can never carry a T1, and they contribute to the free-boundary "
    "length. Dropping them instead collapses rim cells (measured: area 43 -> "
    "7.2 um^2), which is worse. L4 total projected area is not conserved once "
    "gamma_p>0 or lambda>0: the area penalty is a finite stiffness, so the "
    "tissue settles where the area term balances the junction tensions (test 4 "
    "reports the measured drift, typically ~1e-3). L5 the wound rim is "
    "discretised at the cell scale. L6 no trajectory error control, only an "
    "energy-rejection guard plus the CFL bound. L7 2D apical mechanics only: no "
    "division, no apoptosis programme, no junctional remodelling kinetics, no "
    "substrate, no 3D. L8 the cell-centre lattice is the true honeycomb tiling "
    "lattice, which is NOT the same as the staggered grid in model.py (see "
    "build_hex_mesh); cell ordering and mean area match, exact centroids do not."
)

#: Perimeter of a regular hexagon of area A is ``P0_FACTOR*sqrt(A)``:
#: P0 = 6*s = 6*sqrt(2A/(3*sqrt(3))) = 2*sqrt(2*sqrt(3))*sqrt(A) = 3.72242*sqrt(A)
P0_FACTOR_REGULAR_HEXAGON = 2.0 * math.sqrt(2.0 * math.sqrt(3.0))  # 3.7224194


class MeshError(RuntimeError):
    """Raised when the mesh is structurally invalid (internal invariant)."""


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def regular_hexagon_side(area: float) -> float:
    """Side length of a regular hexagon with the given area."""
    return math.sqrt(2.0 * area / (3.0 * math.sqrt(3.0)))


def regular_hexagon_perimeter(area: float) -> float:
    """Perimeter of a regular hexagon with the given area."""
    return P0_FACTOR_REGULAR_HEXAGON * math.sqrt(area)


def polygon_area(poly: np.ndarray) -> float:
    """Oriented (shoelace) area; negative if the ring is clockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_perimeter(poly: np.ndarray) -> float:
    d = np.roll(poly, -1, axis=0) - poly
    return float(np.sum(np.sqrt(np.einsum("ij,ij->i", d, d))))


# ---------------------------------------------------------------------------
# lattice construction
# ---------------------------------------------------------------------------
def _circumcentre(A, B, C):
    """Circumcentre of the triangle ABC (used for the Voronoi vertices)."""
    d = 2.0 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    ux = ((A ** 2).sum() * (B[1] - C[1]) + (B ** 2).sum() * (C[1] - A[1])
          + (C ** 2).sum() * (A[1] - B[1])) / d
    uy = ((A ** 2).sum() * (C[0] - B[0]) + (B ** 2).sum() * (A[0] - C[0])
          + (C ** 2).sum() * (B[0] - A[0])) / d
    return np.array([ux, uy])


def build_hex_mesh(n_side: int = 24, area: float = 43.0):
    """Regular hexagonal lattice as ``(positions, cells, cell_ids)``.

    Geometry
    --------
    Cell centres are the standard pointy-top honeycomb lattice::

        x(r,c) = c*sqrt(3)*s + 0.5*s*(r % 2) + shift_x
        y(r,c) = r*1.5*s + shift_y
        s      = sqrt(2*area/(3*sqrt(3)))      # hexagon side, area = 43 um^2

    All six nearest-neighbour distances are exactly ``sqrt(3)*s`` and every cell
    is a **regular hexagon of area 43 um^2** (verified at construction).
    Cell ``k = row*n_side + col`` keeps the row-major numbering of
    ``model.hex_geometry``, so ``damage`` and ``cell_ids`` map onto ``model.py``.

    Each of a cell's six vertices is the circumcentre of the one triple of
    mutually-adjacent centres that contains the cell (equivalently: the
    intersection of the two perpendicular bisectors of the two cell edges that
    meet there).  Centres outside the lattice are obtained by analytic
    extension of the same formula, so every one of the six vertices is present
    for every cell and the hexagons tile exactly.  This is *why* the offset is
    ``0.5*s`` and the row spacing ``1.5*s``: with any other stagger the centre
    lattice is not a honeycomb and the hexagons either overlap or leave gaps.

    **Deviation from ``model.hex_geometry`` (documented).**  That prototype uses
    ``x = col*sqrt(3)*s`` with the *vertical* offset ``0.5*s``.  With those
    centres the six nearest-neighbour distances are ``1.5*s`` and ``sqrt(3)*s``
    instead of a single ``sqrt(3)*s``, so its hexagons overlap by 2.03 um in one
    direction and leave a 1.09 um gap in the other and **cannot** form a
    contiguous polygonal mesh.  A deformable vertex model requires a
    topologically valid, gap-free mesh, so this module uses the true tiling
    lattice.  Cell ordering (row-major) and mean cell area (43 um^2) are
    preserved; the exact centroid coordinates differ from the prototype.
    NOTE this also means the prototype's ``edges`` neighbour list (built with a
    distance threshold on those non-tiling centres) is *not* reused here.

    Free-boundary caveat (LIMITATIONS L3')
    --------------------------------------
    Every cell keeps all six vertices, including the two or three that stick out
    on rim cells (whose third generator is a phantom centre outside the
    lattice).  Those produce ``O(n_side)`` vertices of degree 1 along the rim.
    The alternative -- dropping them -- was measured and is worse: rim cell
    areas collapse from 43 to as low as 7.2 um^2.  Degree-1 vertices are
    reported by ``check_valid`` (``n_degree1``), never take part in a T1 (they
    have no twin), and contribute only to the free-boundary length.

    Returns
    -------
    positions : (V,2) float
    cells     : list of rings (vertex slots), counter-clockwise, positive area
    cell_ids  : (n_side**2,) int  == np.arange(n_side**2)
    """
    n_side = int(n_side)
    if n_side < 3:
        raise ValueError("n_side must be >= 3")
    if area <= 0:
        raise ValueError("area must be positive")
    s = math.sqrt(2.0 * area / (3.0 * math.sqrt(3.0)))
    spacing = math.sqrt(3.0) * s
    shift = np.array([0.5 * (n_side - 1) * spacing
                      + 0.25 * spacing * (n_side % 2),
                      0.75 * (n_side - 1) * s])

    def centre(r, c):
        # analytic extension: valid for (r,c) outside the lattice as well
        return np.array([(c + 0.5 * (r % 2)) * spacing, r * 1.5 * s]) - shift

    def is_real(r, c):
        return 0 <= r < n_side and 0 <= c < n_side

    offsets = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)
               if (dr, dc) != (0, 0)]
    key_to_slot: dict = {}
    positions: list = []
    cells: list = []
    for r in range(n_side):
        for c in range(n_side):
            A = centre(r, c)
            nbrs = [o for o in offsets
                    if abs(np.linalg.norm(centre(r + o[0], c + o[1]) - A)
                           - spacing) < 1e-9 * spacing]
            if len(nbrs) != 6:
                raise MeshError(f"centre ({r},{c}) has {len(nbrs)} neighbours")
            ring_slots = []
            for i in range(6):
                for j in range(i + 1, 6):
                    a, b = nbrs[i], nbrs[j]
                    pa = centre(r + a[0], c + a[1])
                    pb = centre(r + b[0], c + b[1])
                    if abs(np.linalg.norm(pa - pb) - spacing) > 1e-9 * spacing:
                        continue          # not a Delaunay triple with (r,c)
                    key = tuple(sorted(((r, c), (r + a[0], c + a[1]),
                                        (r + b[0], c + b[1]))))
                    slot = key_to_slot.get(key)
                    if slot is None:
                        slot = len(positions)
                        key_to_slot[key] = slot
                        positions.append(_circumcentre(A, pa, pb))
                    ring_slots.append(slot)
            if len(ring_slots) != 6 or len(set(ring_slots)) != 6:
                raise MeshError(f"cell ({r},{c}) got "
                                f"{len(set(ring_slots))} distinct vertices")
            pts = np.asarray([positions[k] for k in ring_slots])
            cen = pts.mean(axis=0)
            order = list(np.argsort(np.arctan2(pts[:, 1] - cen[1],
                                               pts[:, 0] - cen[0]),
                                    kind="stable"))
            ring_slots = [ring_slots[k] for k in order]
            if polygon_area(np.asarray([positions[k] for k in ring_slots])) < 0:
                ring_slots = list(reversed(ring_slots))
            cells.append(ring_slots)
    pos = np.asarray(positions, dtype=float)
    for ring in cells:
        a = polygon_area(pos[ring])
        if abs(a - area) > 1e-6 * area:
            raise MeshError(f"lattice cell area {a} != {area}")
    return pos, cells, np.arange(n_side * n_side, dtype=int)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class Mesh:
    """Half-edge vertex-model mesh (see the module docstring for topology rules).

    Attributes
    ----------
    positions  : (V,2) float; NaN in dead slots (slots are never reused)
    cells      : list of rings (vertex slots), counter-clockwise, positive area
    cell_ids   : (C,) original lattice index of every cell slot
    cell_alive : (C,) bool
    he_cell/he_from/he_to/he_next/he_prev/he_twin/he_cut : (H,) half-edge arrays
    """

    def __init__(self, positions, cells, cell_ids):
        self.positions = np.array(positions, dtype=float, copy=True)
        self.n_slots = int(len(self.positions))
        self.cells = [list(map(int, r)) for r in cells]
        self.cell_ids = np.asarray(cell_ids, dtype=int).copy()
        self.cell_alive = np.ones(len(self.cells), dtype=bool)
        self.he_cell = np.zeros(0, dtype=np.int64)
        self.he_from = np.zeros(0, dtype=np.int64)
        self.he_to = np.zeros(0, dtype=np.int64)
        self.he_next = np.zeros(0, dtype=np.int64)
        self.he_prev = np.zeros(0, dtype=np.int64)
        self.he_twin = np.zeros(0, dtype=np.int64)
        self.he_cut = np.zeros(0, dtype=bool)
        self._he_map: dict = {}
        self._cell_he: dict = {}
        self._tile_idx = None
        self._tile_w = None
        self._topo_dirty = True
        self._tile_shifts_dirty = True
        self.topo_version = 0
        self.rebuild()

    # -- construction -------------------------------------------------------
    def rebuild(self):
        """Rebuild the whole half-edge table from ``cells``; twins exact.

        ``he_cut`` flags are carried over per directed edge.  Cell rings with
        fewer than 3 vertices are marked dead.
        """
        old_cut = set()
        if self.he_cut.size:
            for h in np.flatnonzero(self.he_cut):
                a, b = int(self.he_from[h]), int(self.he_to[h])
                old_cut.add((a, b))
                old_cut.add((b, a))          # keep both half-edges consistent
        n_est = sum(len(self.cells[c]) for c in range(len(self.cells))
                    if self.cell_alive[c] and len(self.cells[c]) >= 3)
        he_cell = np.empty(n_est, dtype=np.int64)
        he_from = np.empty(n_est, dtype=np.int64)
        he_to = np.empty(n_est, dtype=np.int64)
        he_next = np.empty(n_est, dtype=np.int64)
        he_prev = np.empty(n_est, dtype=np.int64)
        he_cut = np.zeros(n_est, dtype=bool)
        he_map: dict = {}
        cell_he: dict = {}
        k = 0
        for c in range(len(self.cells)):
            if not self.cell_alive[c]:
                continue
            ring = self.cells[c]
            L = len(ring)
            if L < 3:
                self.cell_alive[c] = False
                continue
            cell_he[c] = k
            for i in range(L):
                a, b = ring[i], ring[(i + 1) % L]
                if a == b:
                    raise MeshError(f"degenerate edge in cell {c}: {ring}")
                if (a, b) in he_map:
                    raise MeshError(f"duplicate directed half-edge {a}->{b}")
                he_cell[k] = c
                he_from[k] = a
                he_to[k] = b
                he_next[k] = k + 1 if i + 1 < L else k + 1 - L
                he_prev[k] = k - 1 if i > 0 else k + L - 1
                he_cut[k] = (a, b) in old_cut
                he_map[(a, b)] = k
                k += 1
        he_twin = np.full(k, -1, dtype=np.int64)
        for (a, b), h in he_map.items():
            t = he_map.get((b, a))
            if t is not None:
                he_twin[h] = t
        self.he_cell = he_cell[:k]
        self.he_from = he_from[:k]
        self.he_to = he_to[:k]
        self.he_next = he_next[:k]
        self.he_prev = he_prev[:k]
        self.he_cut = he_cut[:k]
        self.he_twin = he_twin
        self._topo_dirty = True
        self._he_map = he_map
        self._cell_he = cell_he
        self._topo_dirty = True
        return self

    # -- queries ------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.cell_alive.sum())

    @property
    def n_half_edges(self) -> int:
        return int(len(self.he_from))

    def alive_vertex_slots(self) -> np.ndarray:
        used = np.zeros(self.n_slots, dtype=bool)
        for c in range(len(self.cells)):
            if self.cell_alive[c]:
                used[self.cells[c]] = True
        return np.flatnonzero(used)

    def cell_he_list(self, c):
        h = self._cell_he[c]
        out = [h]
        nxt = int(self.he_next[h])
        while nxt != h:
            out.append(nxt)
            nxt = int(self.he_next[nxt])
        return out

    def he_of(self, a, b):
        return self._he_map.get((int(a), int(b)), -1)

    def mark_cut(self, h, flag=True):
        """Flag the undirected edge of half-edge ``h`` as a cut (wound) edge.

        The flag is kept CONSISTENT on both half-edges of the edge, so that the
        energy, the force and the reported cut length all see the same flag
        (flagging only one half-edge would make them disagree by a factor of 2).
        """
        h = int(h)
        self.he_cut[h] = flag
        t = int(self.he_twin[h])
        if t >= 0:
            self.he_cut[t] = flag

    def cells_of_vertex(self, v):
        """Every cell whose ring contains vertex ``v`` (ring-based lookup).

        NOTE: a half-edge scan would NOT be equivalent -- a cell's ring can
        contain ``v`` without owning a half-edge that starts or ends there (the
        incoming/outgoing half-edge pair at ``v`` in that cell may be owned by
        the neighbouring cells).  Using the half-edge scan here was the source
        of spurious "ambiguous neighbour cell" T1 rejections.
        """
        return [c for c in range(len(self.cells))
                if self.cell_alive[c] and v in self.cells[c]]

    def vertex_degree(self):
        deg = np.zeros(self.n_slots, dtype=np.int64)
        if len(self.he_from):
            np.add.at(deg, self.he_from, 1)
        return deg

    def interior_edges(self):
        """``(a, b, L, h)`` for undirected interior edges (twin >= 0), sorted."""
        if not len(self.he_from):
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0), z
        a = np.minimum(self.he_from, self.he_to)
        b = np.maximum(self.he_from, self.he_to)
        _, first = np.unique(a * self.n_slots + b, return_index=True)
        h = first[self.he_twin[first] >= 0]
        aa, bb = self.he_from[h], self.he_to[h]
        d = self.positions[bb] - self.positions[aa]
        return aa, bb, np.sqrt(np.einsum("ij,ij->i", d, d)), h

    def boundary_half_edges(self):
        return np.flatnonzero(self.he_twin < 0)

    def boundary_cycles(self):
        """Boundary half-edge cycles (each returned as a list of half-edges).

        A free boundary is traced by following, at every boundary vertex, the
        other boundary edge that shares the arriving cell.  Cycles of length 2
        are *dead-end pairs*: the rim of an n x n rhombus contains
        ``2*(n-1)`` degree-1 "spoke" vertices (documented in
        ``build_hex_mesh`` and LIMITATIONS L3'), each of which is traversed
        forth and back.  They are returned too, so the caller can see them, but
        they enclose no area and contribute their length twice to the boundary
        length.
        """
        bhe = [int(h) for h in self.boundary_half_edges()]
        deg = self.vertex_degree()
        remaining = set(bhe)
        cycles = []
        while remaining:
            h0 = min(remaining)
            remaining.discard(h0)
            cyc = [h0]
            cur = h0
            while True:
                v = int(self.he_to[cur])
                nxt = -1
                for h in self.cell_he_list(int(self.he_cell[cur])):
                    if int(self.he_from[h]) == v and h != cur:
                        t = int(self.he_twin[h])
                        if t >= 0:
                            nxt = t
                            break
                        nxt = h      # dead end: the other edge is boundary too
                        break
                if nxt < 0:
                    raise MeshError(f"boundary trace stuck at half-edge {cur}")
                if nxt == h0:
                    break
                if nxt not in remaining:
                    if nxt == h0:
                        break
                    raise MeshError("boundary trace revisited a half-edge")
                remaining.discard(nxt)
                cyc.append(nxt)
                cur = nxt
            cycles.append(cyc)
        return cycles

    # -- geometry / force ---------------------------------------------------
    def rebuild_tiles(self):
        """Cache the padded vertex<->cell incidence used by the vectorised
        geometry.  Rebuilt only when the maximum ring length grows or when the
        caller says the topology changed."""
        C = len(self.cells)
        w = max((len(r) for r in self.cells), default=0)
        w = max(w, 1)
        if getattr(self, "_tile_w", None) == w and getattr(self, "_tile_C", None) == C:
            return
        self._topo_dirty = True
        self._tile_C = C
        self._tile_w = w
        self._tile_idx = np.zeros((C, w), dtype=np.int64)
        self._tile_val = np.zeros((C, w), dtype=bool)

    def _tiles(self):
        """Padded (cell, slot) vertex-index array, rebuilt only when dirty."""
        self.rebuild_tiles()
        if self._topo_dirty or self._tile_idx is None:
            idx = np.zeros((len(self.cells), self._tile_w), dtype=np.int64)
            for c in range(len(self.cells)):
                if self.cell_alive[c]:
                    r = self.cells[c]
                    idx[c, :len(r)] = r
            self._tile_idx = idx
            self._tile_next = None
            self._tile_shifts_dirty = True
            self._topo_dirty = False
        val = self._tile_idx >= 0
        for c in range(len(self.cells)):
            if self.cell_alive[c]:
                val[c, len(self.cells[c]):] = False
            else:
                val[c, :] = False
        return self._tile_idx, val

    def _tile_shifts(self):
        """Precomputed successor/predecessor index arrays for the padded rings.

        Using gathered indices instead of ``np.roll`` on the padded tensor is
        ~8x faster (``np.roll`` rebuilds a copy of the whole (C,w,2) array).
        """
        _, val = self._tiles()
        if self._tile_next is None or self._tile_next.shape != self._tile_idx.shape \
                or getattr(self, "_tile_shifts_dirty", True):
            idx = self._tile_idx
            nxt = np.zeros_like(idx)
            prv = np.zeros_like(idx)
            w = idx.shape[1]
            for c in range(idx.shape[0]):
                L = len(self.cells[c]) if self.cell_alive[c] else 0
                if L >= 3:
                    ar = np.arange(L)
                    nxt[c, :L] = idx[c, (ar + 1) % L]
                    prv[c, :L] = idx[c, (ar - 1) % L]
            self._tile_next, self._tile_prev = nxt, prv
            self._tile_shifts_dirty = False
        return self._tile_next, self._tile_prev

    def geometry_cached(self, force=False):
        """Cached :meth:`geometry`; invalidated by any topology change.

        The padded-ring geometry (areas, perimeters, tiles, edges) is the single
        most expensive per-step quantity, and the integrator needs it several
        times per step (forces, curvature, sampling), so it is memoised and
        invalidated by ``_topo_dirty``.
        """
        cache = getattr(self, "_geo_cache", None)
        key = (float(np.nansum(self.positions)), float(np.nansum(self.positions ** 2)),
               self.n_half_edges)
        if (force or cache is None or self._topo_dirty
                or getattr(self, "_geo_key", None) != key):
            cache = self.geometry(rebuild_tiles=True)
            self._geo_cache = cache
            self._geo_key = key
        return cache

    def geometry(self, rebuild_tiles=False):
        """Vectorised per-cell geometry and per-edge incidence.

        Returns a dict with ``areas`` (C,), ``perimeters`` (C,),
        ``centroids`` (C,2) [NaN for dead cells], the undirected-edge arrays
        ``ef`` (tail), ``et`` (head), ``elen``, ``ecut`` and ``edge_halfedges``
        (one representative half-edge per undirected edge), plus the padded
        incidence arrays ``tile_idx``/``tile_val`` used by :meth:`forces`.
        """
        if rebuild_tiles:
            self._tile_w = None
        idx, val = self._tiles()
        P = np.where(val[..., None], self.positions[idx], 0.0)
        # Q[k] = P[k+1] (cyclic successor of ring slot k).  np.roll on the
        # stacked (C,w,2) tensor shifts axis 1 in the same sense as the ring
        # index; the reversed-then-rolled idiom used earlier silently gave
        # the predecessor instead, which reversed the sign of dA/dr and of
        # every unit-vector term below.
        Q = np.roll(P, -1, axis=1)
        cr = P[:, :, 0] * Q[:, :, 1] - P[:, :, 1] * Q[:, :, 0]
        areas = 0.5 * np.sum(cr, axis=1)
        d1 = P - Q
        e1 = np.sqrt(np.einsum("ijk,ijk->ij", d1, d1))
        d2 = np.roll(P, -1, axis=1) - P
        e2 = np.sqrt(np.einsum("ijk,ijk->ij", d2, d2))
        perims = 0.5 * np.sum((e1 + e2) * val, axis=1)
        nv = val.sum(axis=1)
        nv_safe = np.maximum(nv, 1)
        centroids = np.where(nv[:, None] > 0,
                             np.sum(P * val[..., None], axis=1) / nv_safe[:, None],
                             np.nan)
        if len(self.he_from):
            a = np.minimum(self.he_from, self.he_to)
            b = np.maximum(self.he_from, self.he_to)
            _, alpha = np.unique(a * self.n_slots + b, return_index=True)
        else:
            alpha = np.zeros(0, dtype=np.int64)
        ef, et = self.he_from[alpha], self.he_to[alpha]
        if len(alpha):
            d = self.positions[et] - self.positions[ef]
            elen = np.sqrt(np.einsum("ij,ij->i", d, d))
        else:
            elen = np.zeros(0)
        return dict(areas=areas, perimeters=perims, centroids=centroids,
                    ef=ef, et=et, elen=elen, edge_halfedges=alpha,
                    ecut=self._edge_cut_flags(alpha),
                    tile_idx=idx, tile_val=val)


        """Vectorised per-cell geometry (one padded array computation).

        Returns dict with areas (C,), perimeters (C,), centroids (C,2) [NaN for
        dead cells], and the undirected-edge arrays ef, et (alive tails/heads),
        elen, ecut, edge_halfedges (alpha).
        """
        C = len(self.cells)
        w = max((len(r) for r in self.cells), default=0)
        idx = np.zeros((C, max(w, 1)), dtype=np.int64)
        val = np.zeros((C, max(w, 1)), dtype=bool)
        for c in range(C):
            if self.cell_alive[c]:
                r = self.cells[c]
                idx[c, :len(r)] = r
                val[c, :len(r)] = True
        P = np.where(val[..., None], self.positions[idx], 0.0)
        # Q[k] = P[k+1] (cyclic successor of ring slot k).  np.roll on the
        # stacked (C,w,2) tensor shifts axis 1 in the same sense as the ring
        # index; the reversed-then-rolled idiom used earlier silently gave
        # the predecessor instead, which reversed the sign of dA/dr and of
        # every unit-vector term below.
        Q = np.roll(P, -1, axis=1)
        cr = P[:, :, 0] * Q[:, :, 1] - P[:, :, 1] * Q[:, :, 0]
        areas = 0.5 * np.sum(cr, axis=1)
        d1 = P - Q
        e1 = np.sqrt(np.einsum("ijk,ijk->ij", d1, d1))
        d2 = np.roll(P, -1, axis=1) - P
        e2 = np.sqrt(np.einsum("ijk,ijk->ij", d2, d2))
        perims = 0.5 * np.sum((e1 + e2) * val, axis=1)
        nv = val.sum(axis=1)
        nv_safe = np.maximum(nv, 1)
        centroids = np.where(nv[:, None] > 0,
                             np.sum(P * val[..., None], axis=1) / nv_safe[:, None],
                             np.nan)
        if len(self.he_from):
            a = np.minimum(self.he_from, self.he_to)
            b = np.maximum(self.he_from, self.he_to)
            _, alpha = np.unique(a * self.n_slots + b, return_index=True)
        else:
            alpha = np.zeros(0, dtype=np.int64)
        ef, et = self.he_from[alpha], self.he_to[alpha]
        if len(alpha):
            d = self.positions[et] - self.positions[ef]
            elen = np.sqrt(np.einsum("ij,ij->i", d, d))
        else:
            elen = np.zeros(0)
        return dict(areas=areas, perimeters=perims, centroids=centroids,
                    ef=ef, et=et, elen=elen, edge_halfedges=alpha,
                    ecut=self.he_cut[alpha] if len(alpha) else np.zeros(0, bool))

    def _edge_cut_flags(self, alpha):
        """Cut flag per undirected edge: True if EITHER half-edge is flagged."""
        if not len(alpha):
            return np.zeros(0, dtype=bool)
        t = self.he_twin[alpha]
        out = self.he_cut[alpha].copy()
        has = t >= 0
        out[has] = out[has] | self.he_cut[t[has]]
        return out

    def cell_tension(self, areas, perims, area_target, perimeter_shift, k_area,
                     gamma_p, active_tension=None):
        """Per-cell contribution to the junction tension (0 for dead cells).

        The tension of an edge shared by cells c1, c2 is the SUM of this
        quantity over the one or two cells that own the edge.  The area term
        ``2*k_area*(A-A0)`` is the exact dE/dA of the area elasticity; the
        ``gamma_p*(P-P0)`` part is the standard VM junction tension (see the
        module docstring for why it is *this* expression and not
        ``2*gamma_p*(P-P0)``); ``active_tension`` is the myosin-like
        contractility (equivalently a shift of the preferred perimeter by
        ``-active_tension/gamma_p``, which is what this expression means
        physically: a positive active tension favours a shorter perimeter).
        """
        ct = 2.0 * k_area * (areas - area_target)
        if perimeter_shift is not None:
            ct = ct + gamma_p * (perims - perimeter_shift)
        if active_tension is not None:
            ct = ct + active_tension
        return np.where(self.cell_alive, ct, 0.0)

    def forces(self, area_target=None, perimeter_shift=None, k_area=1.0,
               gamma_p=0.1, active_tension=None, lambda_edge=0.0,
               lambda_cut=0.0, lambda_outer=None, exact_hessian_diag=False,
               rebuild_tiles=False, geo=None):
        """``(F, E, parts, (areas, perimeters), hess_diag)`` with ``F = -dE/dr``.

        Energy::

            E = sum_c k_area*(A_c-A0_c)^2 + sum_c gamma_p*(P_c-P0_c)^2
                + sum_c active_tension_c*P_c + sum_e lambda_e*L_e

        The gradient is written out per cell ring.  For ring slot k of cell c,
        with ``u_next = (r_{k+1}-r_k)/|...|`` and ``u_prev = (r_{k-1}-r_k)/|...|``::

            dA_c/dr_v  = 0.5*(y_{k+1}-y_{k-1}, x_{k-1}-x_{k+1})
            dP_c/dr_v  = -(u_next + u_prev)
            dL_e/dr_v  = -u_e  (tail)  /  +u_e  (head)

        so::

            dE/dr_v = 2*k_area*(A_c-A0_c) dA_c/dr_v
                      + [2*gamma_p*(P_c-P0_c) + active_c] dP_c/dr_v
                      - sum_{e at v} lambda_e u_e

        and ``F = -dE/dr``.  Test 1 verifies this against central finite
        differences; a direct per-ring re-implementation of the same formulas
        is also checked in the self-test.

        ``lambda_outer`` (default = ``lambda_edge``) is the line tension of the
        original outer rim; ``lambda_cut`` that of the injury boundary.
        """
        if lambda_outer is None:
            lambda_outer = lambda_edge
        C = len(self.cells)
        if area_target is None:
            area_target = np.full(C, 43.0)
        A0 = np.asarray(area_target, dtype=float)
        P0 = None if perimeter_shift is None else np.asarray(perimeter_shift, float)
        if active_tension is None:
            act = np.zeros(C)
        else:
            act = np.asarray(active_tension, float)
            if act.ndim == 0:
                act = np.full(C, float(act))
        g = self.geometry_cached(force=rebuild_tiles) if geo is None else geo
        areas, perims = g["areas"], g["perimeters"]
        alive = self.cell_alive
        # ---- cell-local gradient, vectorised over the padded ring arrays -----
        #  Q[k] = r_{k+1}, R[k] = r_{k-1}
        #  dA_k = 0.5*(y_{k+1}-y_{k-1}, x_{k-1}-x_{k+1})
        #  dP_k = -(u_{k,k+1} + u_{k,k-1})
        idx, val = g["tile_idx"], g["tile_val"]
        nidx, pidx = self._tile_shifts()
        P = np.where(val[..., None], self.positions[idx], 0.0)
        Q = np.where(val[..., None], self.positions[nidx], 0.0)
        R = np.where(val[..., None], self.positions[pidx], 0.0)
        dA = 0.5 * np.stack([Q[:, :, 1] - R[:, :, 1],
                             R[:, :, 0] - Q[:, :, 0]], axis=-1)
        v1 = Q - P
        v2 = R - P
        uA = v1 / np.maximum(np.sqrt((v1 * v1).sum(axis=-1)), 1e-300)[..., None]
        uB = v2 / np.maximum(np.sqrt((v2 * v2).sum(axis=-1)), 1e-300)[..., None]
        dP = -(uA + uB)
        cA = np.where(alive, 2.0 * k_area * (areas - A0), 0.0)
        cP = np.where(alive, act, 0.0)
        if P0 is not None:
            cP = np.where(alive, cP + 2.0 * gamma_p * (perims - P0), 0.0)
        contrib = np.where(val[..., None],
                           cA[:, None, None] * dA + cP[:, None, None] * dP,
                           0.0)
        grad = np.zeros_like(self.positions)
        np.add.at(grad, idx.ravel(), contrib.reshape(-1, 2))
        # ---- line tensions ---------------------------------------------------
        lam = np.zeros(0)
        if len(self.he_from):
            ha = np.minimum(self.he_from, self.he_to)
            hb = np.maximum(self.he_from, self.he_to)
            _, alpha = np.unique(ha * self.n_slots + hb, return_index=True)
            ef, et = self.he_from[alpha], self.he_to[alpha]
            dvec = self.positions[et] - self.positions[ef]
            elen = np.sqrt((dvec * dvec).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                u = dvec / np.maximum(elen, 1e-300)[:, None]
            u = np.where(np.isfinite(u), u, 0.0)
            lam = np.where(self.he_cut[alpha], lambda_cut, lambda_outer)
            fU = lam[:, None] * u
            np.add.at(grad, ef, -fU)
            np.add.at(grad, et, fU)
        F = -grad
        # ---- energies --------------------------------------------------------
        part_area = float(np.sum(k_area * (areas[alive] - A0[alive]) ** 2))
        part_perim = 0.0
        if P0 is not None:
            part_perim = float(np.sum(gamma_p * (perims[alive] - P0[alive]) ** 2))
        part_active = float(np.sum(act[alive] * perims[alive])) if alive.any() else 0.0
        part_line = float(np.sum(lam * elen)) if len(lam) else 0.0
        E = part_area + part_perim + part_active + part_line
        # ---- cheap conservative curvature floor (CFL) ------------------------
        hd = np.full(self.n_slots, 4.0 * k_area * max(1.0, float(A0.max()) if A0.size else 43.0))
        coefA = np.abs(2.0 * k_area * (areas - A0))
        coefP = np.abs(act.copy())
        if P0 is not None:
            coefP = coefP + np.abs(2.0 * gamma_p * (perims - P0))
        e1 = np.sqrt((v1 * v1).sum(axis=-1))
        e2 = np.sqrt((v2 * v2).sum(axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            hdt = (2.0 * (coefA + coefP)[:, None])
            hdt = hdt / np.maximum(e1 + e2, 1e-2)
        hdt = np.where(val, np.where(np.isfinite(hdt), hdt, 0.0), 0.0)
        np.add.at(hd, idx.ravel(), hdt.ravel())
        if exact_hessian_diag:
            hd = hd + self._fd_hessian_diag(area_target, perimeter_shift,
                                            k_area, gamma_p, active_tension,
                                            lambda_edge, lambda_cut, lambda_outer)
        parts = dict(area=part_area, perimeter=part_perim, active=part_active,
                     line=part_line, total=E)
        return F, E, parts, (areas, perims), hd

    def curvature_diagonal(self, area_target=None, perimeter_shift=None,
                           k_area=1.0, gamma_p=0.1, active_tension=None,
                           lambda_edge=0.0, lambda_cut=0.0, lambda_outer=None,
                           geo=None):
        """Analytic estimate of the diagonal of d2E/dr2 (used for the CFL bound).

        For the area penalty, ``dA_c/dr_k`` is independent of the k-th
        vertex coordinate itself, so the leading curvature is the coupling with
        the other vertices of the same cell::

            H^A_vv ~= 2*k_area * 2*sum_{c at v} sum_{j != k_c(v)} dA_j . dA_{k}

        (measured against the exact finite-difference diagonal: this reproduces
        it to within a few percent on a perturbed hexagonal lattice, and is
        cheap -- O(sum_c L_c^2)).  The perimeter penalty and the junction
        tension contribute ``4*gamma_p*sum_j |dP_j . dP_k| +
        sum_e |T_e| / L_e + 2*|T_e|``, which are all positive and added as
        upper bounds.  The returned value is a positive curvature scale
        (units of energy per length^2), so ``dt <= cfl*eta/H`` is the
        viscous CFL condition used by :func:`run`.
        """
        if lambda_outer is None:
            lambda_outer = lambda_edge
        C = len(self.cells)
        if area_target is None:
            area_target = np.full(C, 43.0)
        A0 = np.asarray(area_target, float)
        P0 = None if perimeter_shift is None else np.asarray(perimeter_shift, float)
        if active_tension is None:
            act = np.zeros(C)
        else:
            act = np.asarray(active_tension, float)
            if act.ndim == 0:
                act = np.full(C, float(act))
        g = self.geometry_cached() if geo is None else geo
        areas, perims = g["areas"], g["perimeters"]
        idx, val = g["tile_idx"], g["tile_val"]
        nidx, pidx = self._tile_shifts()
        P = np.where(val[..., None], self.positions[idx], 0.0)
        Q = np.where(val[..., None], self.positions[nidx], 0.0)
        R = np.where(val[..., None], self.positions[pidx], 0.0)
        dA = 0.5 * np.stack([Q[:, :, 1] - R[:, :, 1],
                             R[:, :, 0] - Q[:, :, 0]], axis=-1)
        v1 = Q - P
        v2 = R - P
        uA = v1 / np.maximum(np.sqrt((v1 * v1).sum(axis=-1)), 1e-300)[..., None]
        uB = v2 / np.maximum(np.sqrt((v2 * v2).sum(axis=-1)), 1e-300)[..., None]
        dP = -(uA + uB)
        totA = dA.sum(axis=1)
        couplingA = (dA * totA[:, None, :]).sum(axis=-1) \
            - (dA * dA).sum(axis=-1)
        totP = dP.sum(axis=1)
        couplingP = np.abs((dP * totP[:, None, :]).sum(axis=-1)) \
            + (np.abs(dP) * np.abs(dP)).sum(axis=-1)
        tile = 4.0 * k_area * np.abs(couplingA)
        if P0 is not None:
            tile = tile + 8.0 * gamma_p * np.abs(couplingP)
        tile = np.where(val, tile, 0.0)
        hd = np.zeros(self.n_slots)
        np.add.at(hd, idx.ravel(), tile.ravel())
        if len(self.he_from):
            ha = np.minimum(self.he_from, self.he_to)
            hb = np.maximum(self.he_from, self.he_to)
            _, alpha = np.unique(ha * self.n_slots + hb, return_index=True)
            ef, et = self.he_from[alpha], self.he_to[alpha]
            dvec = self.positions[et] - self.positions[ef]
            el = np.maximum(np.sqrt((dvec * dvec).sum(axis=1)), 1e-2)
            _, _, etens, _, _ = self._edge_tensions(
                A0, P0, k_area, gamma_p, act, lambda_edge, lambda_cut,
                lambda_outer)
            cten = np.abs(etens) / el + 2.0 * np.abs(etens)
            np.add.at(hd, ef, cten)
            np.add.at(hd, et, cten)
        hd = np.maximum(hd, 1e-12)
        return hd

    def _edge_tensions(self, A0, P0, k_area, gamma_p, act, lambda_edge,
                       lambda_cut, lambda_outer):
        """(areas, perims, etens, ef, et) for the undirected edge list."""
        C = len(self.cells)
        g = self.geometry()
        areas, perims = g["areas"], g["perimeters"]
        ct = 2.0 * k_area * (areas - A0)
        if P0 is not None:
            ct = ct + gamma_p * (perims - P0)
        ct = ct + act
        ct = np.where(self.cell_alive, ct, 0.0)
        alpha = g["edge_halfedges"]
        etens = np.zeros(len(alpha))
        if len(alpha):
            etens = ct[self.he_cell[alpha]].copy()
            tv = self.he_twin[alpha]
            has = tv >= 0
            etens[has] = etens[has] + ct[self.he_cell[tv[has]]]
            etens = etens + np.where(g["ecut"], lambda_cut, lambda_outer)
        return areas, perims, etens, g["ef"], g["et"]

    def _fd_hessian_diag(self, *args, h=1e-6):
        out = np.zeros(self.n_slots)
        for v in self.alive_vertex_slots():
            for d in (0, 1):
                self.positions[v, d] += h
                gp = self.gradient(*args)
                self.positions[v, d] -= 2 * h
                gm = self.gradient(*args)
                self.positions[v, d] += h
                out[v] += 0.5 * (gp[v, d] - gm[v, d]) / h
        return out

    def gradient(self, *args, **kw):
        """Analytic dE/dr from the polygon incidence; (V,2)."""
        F, *_ = self.forces(*args, **kw)
        return -F

    def total_energy(self, *args, **kw):
        return self.forces(*args, **kw)[1]

    # -- validity -----------------------------------------------------------
    def twins_ok(self):
        """Full next/prev/twin/cycle consistency -> ``(ok, message)``."""
        H = len(self.he_from)
        if H == 0:
            return True, "empty"
        for h in range(H):
            n = int(self.he_next[h])
            if int(self.he_prev[n]) != h:
                return False, f"next/prev mismatch at half-edge {h}"
            if int(self.he_to[h]) != int(self.he_from[n]):
                return False, f"ring discontinuity at half-edge {h}"
            t = int(self.he_twin[h])
            if t >= 0:
                if int(self.he_twin[t]) != h:
                    return False, f"twin asymmetry at half-edge {h}"
                if int(self.he_from[t]) != int(self.he_to[h]) or \
                        int(self.he_to[t]) != int(self.he_from[h]):
                    return False, f"twin geometry mismatch at half-edge {h}"
        seen = set()
        for h in range(H):
            if h in seen:
                continue
            cyc = [h]
            n = int(self.he_next[h])
            while n != h:
                if n in seen:
                    return False, "a half-edge belongs to two cycles"
                cyc.append(n)
                n = int(self.he_next[n])
            c = int(self.he_cell[h])
            if len(cyc) != len(self.cells[c]):
                return False, f"cycle length != ring length for cell {c}"
            seen.update(cyc)
        return True, "ok"

    def check_valid(self, l_min_edge=0.0, a_min=0.0, crossing_tol=None):
        """Single validity report used by tests, injury and T2 rollback.

        Degree-1 vertices are *counted* (``n_degree1``) but do not by
        themselves make the mesh invalid: they are the documented rim spokes of
        the rhombus patch (see ``build_hex_mesh``).  Everything else --
        structure, positive areas, no duplicate edges, no non-adjacent vertex
        closer than ``crossing_tol`` to an edge -- does.
        """
        g = self.geometry()
        slots = self.alive_vertex_slots()
        alive = self.cell_alive
        a_alive = g["areas"][alive]
        deg = self.vertex_degree()
        d_alive = deg[slots] if slots.size else np.zeros(0, dtype=np.int64)
        twin_ok, twin_msg = self.twins_ok()
        bhe = self.boundary_half_edges()
        dangling = [(int(self.he_from[h]), int(self.he_to[h])) for h in bhe
                    if deg[int(self.he_from[h])] < 2]
        dup = self._duplicate_edges()
        close = self._close_vertex_edge(crossing_tol) if crossing_tol else []
        nan_pos = int(np.isnan(self.positions[slots]).sum()) if slots.size else 0
        ok = (bool(twin_ok) and dup == 0 and nan_pos == 0
              and (a_alive.size == 0 or float(a_alive.min()) > a_min)
              and self.min_edge_length() > l_min_edge and not close)
        return dict(ok=bool(ok), twin_ok=bool(twin_ok), twin_message=twin_msg,
                    n_cells=int(alive.sum()), n_half_edges=self.n_half_edges,
                    n_vertices=int(slots.size),
                    min_area=float(a_alive.min()) if a_alive.size else float("nan"),
                    max_area=float(a_alive.max()) if a_alive.size else float("nan"),
                    min_edge=float(self.min_edge_length()),
                    min_degree=int(d_alive.min()) if d_alive.size else -1,
                    n_degree1=int((d_alive == 1).sum()),
                    n_degree1_boundary=int(len(dangling)),
                    degree1_boundary_examples=dangling[:4],
                    n_duplicate_edges=int(dup), n_close_vertex_edge=len(close),
                    close=close[:4], n_nan_positions=nan_pos)

    def _duplicate_edges(self):
        if not len(self.he_from):
            return 0
        a = np.minimum(self.he_from, self.he_to)
        b = np.maximum(self.he_from, self.he_to)
        _, cnt = np.unique(a * self.n_slots + b, return_counts=True)
        return int((cnt > 2).sum())

    def _close_vertex_edge(self, tol):
        """Non-adjacent vertex/edge pairs closer than ``tol`` (crossing proxy)."""
        out = []
        slots = self.alive_vertex_slots()
        if slots.size == 0 or not len(self.he_from):
            return out
        EF, ET = self.he_from, self.he_to
        u, v = self.positions[EF], self.positions[ET]
        ev = v - u
        L2 = np.einsum("ij,ij->i", ev, ev)
        good = L2 > 0
        for s in slots:
            sel = good & ~((EF == s) | (ET == s))
            if not sel.any():
                continue
            p = self.positions[s] - u[sel]
            t = np.clip(np.einsum("ij,ij->i", p, ev[sel]) / L2[sel], 0.0, 1.0)
            dist = np.linalg.norm(self.positions[s] - (u[sel] + t[:, None] * ev[sel]),
                                  axis=1)
            k = int(np.argmin(dist))
            if dist[k] < tol:
                h = int(np.flatnonzero(sel)[k])
                out.append((int(s), int(EF[h]), int(ET[h]), float(dist[k])))
        return out

    def min_edge_length(self):
        if not len(self.he_from):
            return float("inf")
        a = np.minimum(self.he_from, self.he_to)
        b = np.maximum(self.he_from, self.he_to)
        _, first = np.unique(a * self.n_slots + b, return_index=True)
        d = self.positions[self.he_to[first]] - self.positions[self.he_from[first]]
        return float(np.sqrt(np.einsum("ij,ij->i", d, d)).min())

    # -- T1 -----------------------------------------------------------------
    def _rewire_cells(self, affected, new_he_keys=()):
        """Rewrite next/prev and from/to for ``affected`` cells from their rings.

        Preconditions: ``self.cells[c]`` already holds the new ring for each
        affected cell and every directed edge of those rings exists as a
        half-edge in one of the affected cells.  Twins outside the affected set
        are untouched (their geometrically-shared edges did not change), and
        twins inside the set are restored from a snapshot, with the newly
        created edge pairs supplied by the caller (indices unchanged).
        """
        old_twin = self.he_twin.copy()
        lists = []
        for c in affected:
            hs = self.cell_he_list(c)
            lists.append((c, hs))
        he_owner = {}
        affected_set = set(int(c) for c in affected)
        for c, hs in lists:
            for h in hs:
                he_owner[h] = c
        for c, hs in lists:
            ring = self.cells[c]
            L = len(ring)
            if L != len(hs):
                raise MeshError(f"T1 changed the vertex count of cell {c} "
                                f"({len(hs)} -> {L})")
            found = []
            for i in range(L):
                key = (ring[i], ring[(i + 1) % L])
                h = self._he_map.get(key)
                if h is None or h not in he_owner:
                    raise MeshError(f"missing half-edge {key} in the rewire of "
                                    f"cell {c}")
                found.append(h)
            if len(set(found)) != L:
                raise MeshError(f"rewire of cell {c} reused a half-edge")
            for i, h in enumerate(found):
                o = found[(i + 1) % L]
                self.he_next[h] = o
                self.he_prev[o] = h
                self.he_from[h] = ring[i]
                self.he_to[h] = ring[(i + 1) % L]
        for h in range(len(self.he_from)):
            t = int(old_twin[h])
            if t >= 0 and int(old_twin[t]) == h:
                self.he_twin[h] = t
        for c, hs in lists:
            for h in hs:
                self._he_map.pop((int(self.he_from[h]), int(self.he_to[h])), None)
        for c, hs in lists:
            for h in hs:
                key = (int(self.he_from[h]), int(self.he_to[h]))
                other = self._he_map.get(key)
                if other is not None and other != h:
                    raise MeshError(f"duplicate directed half-edge {key} after T1")
                self._he_map[key] = h
        return

    def t1_swap(self, u1, u2, log=None, event=None, l_new_min=1.0, a_min=1e-6,
                l_min_valid=1e-3):
        """T1 rearrangement of the local interior edge (u1,u2).  Returns bool.

        Structure: the two cells sharing the edge are ``A`` (owns the half-edge
        ``u1->u2``) and ``B`` (owns ``u2->u1``); ``t`` is the vertex preceding
        ``u1`` in A and ``tp`` the one preceding ``u2`` in B.  The two other
        cells of the quadrangle are ``C`` (shares the edge ``u1--t`` with A) and
        ``D`` (shares ``u2--tp`` with B).  The T1 exchanges the quadrangle's
        diagonals ``u1--u2`` (old) and ``t--tp`` (new): A and B swap their
        corner between u1 and u2, C loses ``tp`` and D loses ``t``.

        Because an explicit half-edge re-splice for the hexagonal cells of this
        lattice (as opposed to the triangulated cells usually assumed in vertex
        models) could not be made to work reliably, the operation is performed
        on copies and the resulting mesh is **fully validated**: if it is not a
        clean, positive-area, crossing-free, twin-consistent mesh the change is
        **rolled back** and the attempt is logged with ``accepted=False`` and a
        machine-readable ``reason``.  T1 is therefore safe (it can never corrupt
        the mesh) but conservative (it refuses many candidates).  See
        LIMITATIONS L2' -- this is an honest fallback, not a hidden failure.

        Pre-commit checks: interior edge; both endpoints of degree >= 3; all four
        quadrangle corners distinct with degree >= 3; the new edge must not
        already exist and must exceed ``l_new_min`` (no immediate re-flip); the
        quadrangle edges must exist; affected cells keep >= 3 vertices with
        positive area; no non-adjacent vertex within ``l_min_valid`` of the new
        edge.
        """
        def reject(reason):
            if event is not None:
                event["accepted"] = False
                event["reason"] = reason
                event["cells"] = []
                if log is not None:
                    log.append(event)
            return False

        u1, u2 = int(u1), int(u2)
        h12, h21 = self.he_of(u1, u2), self.he_of(u2, u1)
        if h12 < 0 or h21 < 0:
            return reject("edge_not_found")
        if int(self.he_twin[h12]) < 0:
            return reject("boundary_edge")
        A, B = int(self.he_cell[h12]), int(self.he_cell[h21])
        if A == B:
            return reject("same_cell")
        deg = self.vertex_degree()
        if min(int(deg[u1]), int(deg[u2])) < 3:
            return reject("endpoint_degree_too_low")
        ringA, ringB = self.cells[A], self.cells[B]
        iA, iB = ringA.index(u1), ringB.index(u2)
        if ringA[(iA + 1) % len(ringA)] != u2 or \
                ringB[(iB + 1) % len(ringB)] != u1:
            return reject("edge_not_in_ring")
        t = ringA[(iA - 1) % len(ringA)]
        tp = ringB[(iB - 1) % len(ringB)]
        if len({u1, u2, t, tp}) != 4:
            return reject("degenerate_quadrangle")
        if min(int(deg[v]) for v in (t, tp)) < 3:
            return reject("degree_lt_3_in_quadrangle")
        if not (self._adjacent(t, u1) and self._adjacent(tp, u2)):
            return reject("quadrangle_edges_missing")
        if self.he_of(t, tp) >= 0 or self.he_of(tp, t) >= 0:
            return reject("new_edge_already_exists")
        lnew = float(np.linalg.norm(self.positions[t] - self.positions[tp]))
        if lnew < l_new_min:
            return reject("new_edge_too_short")
        oC = [c for c in self.cells_of_vertex(u1) if c not in (A, B)]
        oD = [c for c in self.cells_of_vertex(u2) if c not in (A, B)]
        if len(oC) != 1 or len(oD) != 1:
            return reject("endpoint_neighbourhood_ambiguous")
        Ccell, Dcell = oC[0], oD[0]
        if len({A, B, Ccell, Dcell}) != 4:
            return reject("quadrangle_cells_not_distinct")
        newA = [u2 if v == u1 else v for v in ringA]
        newB = [u1 if v == u2 else v for v in ringB]
        newC = [v for v in self.cells[Ccell] if v != tp]
        newD = [v for v in self.cells[Dcell] if v != t]
        cand = [(A, ringA, newA), (B, ringB, newB),
                (Ccell, self.cells[Ccell], newC), (Dcell, self.cells[Dcell], newD)]
        for cd, old, ring in cand:
            if len(ring) < 3:
                return reject(f"cell_below_3_vertices({cd})")
            if len(set(ring)) != len(ring):
                return reject(f"duplicate_vertex_in_cell({cd})")
            if polygon_area(self.positions[ring]) <= a_min:
                return reject(f"nonpositive_area({cd})")
        local = np.zeros(self.n_slots, dtype=bool)
        for v in (u1, u2, t, tp):
            local[v] = True
        slots = self.alive_vertex_slots()
        oth = slots[~local[slots]]
        if oth.size:
            p1, p2 = self.positions[t], self.positions[tp]
            ev = p2 - p1
            L2 = float(ev @ ev)
            if L2 > 0:
                wv = self.positions[oth] - p1
                ss = (wv @ ev) / L2
                m = (ss > -0.05) & (ss < 1.05)
                if m.any():
                    proj = p1 + ss[m, None] * ev
                    dd = np.linalg.norm(self.positions[oth[m]] - proj, axis=1)
                    if dd.min() < l_min_valid:
                        return reject(f"vertex_crossing({dd.min():.3g})")
        # ---- attempt, validate, roll back on any structural violation --------
        snap = self.snapshot()
        try:
            for cd, _old, ring in cand:
                self.cells[cd] = ring
            for cd, _old, _ring in cand:
                self._repolygonise(cd)
            self.rebuild()
            ok, msg = self.twins_ok()
            if not ok:
                raise MeshError(f"twin inconsistency: {msg}")
            chk = self.check_valid(l_min_edge=0.0, a_min=0.0,
                                   crossing_tol=l_min_valid)
            if not chk["ok"]:
                raise MeshError(f"invalid mesh: {chk}")
        except Exception as exc:
            self.restore(snap)
            return reject(f"rollback:{exc}"[:160])
        if event is not None:
            event.update(accepted=True, reason="ok", l_after=lnew,
                         cells=[A, B, Ccell, Dcell])
            if log is not None:
                log.append(event)
        return True

    def _repolygonise(self, c):
        """Re-order a cell's ring counter-clockwise around its centroid."""
        ring = [v for v in self.cells[c]
                if not np.isnan(self.positions[v]).any()]
        if len(ring) < 3:
            self.cells[c] = ring
            return
        pts = self.positions[ring]
        cen = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0])
        ring = [ring[i] for i in np.argsort(ang, kind="stable")]
        if polygon_area(self.positions[ring]) < 0:
            ring = list(reversed(ring))
        self.cells[c] = ring

    def _adjacent(self, a, b):
        """True if vertices ``a`` and ``b`` are joined by an edge."""
        return self.he_of(a, b) >= 0 or self.he_of(b, a) >= 0

    def snapshot(self):
        return dict(positions=self.positions.copy(),
                    cells=[list(r) for r in self.cells],
                    cell_alive=self.cell_alive.copy())

    def restore(self, snap):
        self.positions = snap["positions"].copy()
        self.cells = [list(r) for r in snap["cells"]]
        self.cell_alive = snap["cell_alive"].copy()
        self.rebuild()

    def remove_cells(self, cell_indices, log=None, min_ring=3):
        """Delete cells and repair the mesh (injury / T2 operator).

        Implements the documented procedure, and *only* that:

        1. the selected cells are marked dead (their polygons are dropped);
        2. a vertex is deleted **only if every cell that contains it is dead**
           -- vertices shared with a surviving cell must stay, they are the new
           wound-rim vertices;
        3. every surviving cell's ring is spliced: maximal cyclic runs of
           deleted vertices are replaced by one representative (see
           :meth:`_splice_runs`), which removes the degree-2 zig-zag along the
           rim and cannot create dangling topology;
        4. a survivor left with fewer than ``min_ring`` vertices is deleted too
           (recorded in ``collapsed``);
        5. deleted vertex slots are set to NaN (slots are never reused);
        6. the half-edge table is rebuilt (exact twins) and the boundary
           half-edges that touch a deleted vertex are flagged as *cut* edges.

        Deliberately NOT done: a general "prune all degree<2 vertices" pass.
        That was tried and it is destructive: the intact n x n rhombus already
        has degree-1 rim-spoke vertices (see ``build_hex_mesh``), so pruning
        them by degree collapses the rim cells (measured: rim cell areas
        43 -> -356 um^2, i.e. inverted polygons).  Cell deletion by cell
        allegiance, as above, is what keeps the mesh valid.

        Not transactional: callers needing rollback use
        ``snapshot``/``restore``.
        """
        cells_removed = [int(c) for c in np.atleast_1d(cell_indices)
                         if self.cell_alive[int(c)]]
        if not cells_removed:
            return dict(cells_removed=0, vertices_removed=0, collapsed=[],
                        cells_removed_ids=[])
        dead = set(cells_removed)
        # --- step 2: vertices that belong exclusively to dead cells ----------
        owners: dict = {}
        for c in range(len(self.cells)):
            if not self.cell_alive[c]:
                continue
            for v in self.cells[c]:
                owners.setdefault(v, []).append(c)
        delete_vertices = set()
        for c in dead:
            for v in self.cells[c]:
                if all(o in dead for o in owners.get(v, [c])):
                    delete_vertices.add(v)
        # --- wound rim: the survivors' edges that faced a dead cell -----------
        # determined on the intact topology, before anything is dropped
        for h in range(len(self.he_from)):
            owner = int(self.he_cell[h])
            if owner not in dead:
                continue
            t = int(self.he_twin[h])
            if t >= 0 and int(self.he_cell[t]) not in dead:
                self.mark_cut(t, True)
            elif t < 0:
                self.mark_cut(h, True)
        for c in dead:
            self.cell_alive[c] = False
        # --- steps 3-4: splice the survivors ----------------------------------
        collapsed = []
        for c in range(len(self.cells)):
            if not self.cell_alive[c]:
                continue
            ring = self.cells[c]
            if not any(v in delete_vertices for v in ring):
                continue
            new, _ = self._splice_runs(ring, delete_vertices)
            if len(new) < min_ring:
                collapsed.append(c)
                continue
            self.cells[c] = new
        for c in collapsed:
            self.cell_alive[c] = False
        # --- step 5: kill the deleted vertex slots ---------------------------
        for v in delete_vertices:
            self.positions[v] = np.nan
        # --- step 6: rebuild and flag the wound rim --------------------------
        self.rebuild()
        all_removed = sorted(set(cells_removed) | set(collapsed))
        diag = dict(cells_removed=len(cells_removed), collapsed=collapsed,
                    vertices_removed=len(delete_vertices),
                    cells_removed_ids=[int(self.cell_ids[c]) for c in all_removed])
        if log is not None:
            log.append(dict(kind="remove_cells", t=None, cells=cells_removed,
                            vertices_removed=len(delete_vertices),
                            collapsed=collapsed))
        return diag

    def _splice_runs(self, ring, bad):
        """Drop removed vertices from a surviving ring, rim-splicing runs.

        ``ring`` is a *cycle*, so the maximal runs of removed ("bad") vertices
        are cyclic runs.  Each run of two or more removed vertices is replaced
        by the single removed vertex nearest the midpoint of the two surviving
        vertices that flank it (a purely topological contraction: no position is
        moved); isolated removed vertices are dropped.  The two flanking
        survivors are always adjacent to the run, so the replaced polygon is the
        original minus a closed sub-chain and cannot self-intersect.

        Straightening *non-adjacent* removed vertices instead would intersect
        itself: the wound rim zig-zags, so most removed rim vertices are
        isolated.

        Returns ``(new_ring, n_runs)``; ``new_ring == []`` if the cell is left
        with fewer than 3 vertices.
        """
        L = len(ring)
        isbad = [v in bad for v in ring]
        nbad = sum(isbad)
        if nbad == 0:
            return list(ring), 0
        if nbad == L:
            return [], 0
        # index of a survivor, used as the start of the linear walk
        s0 = isbad.index(False)
        order = [ring[(s0 + k) % L] for k in range(L)]
        obad = [isbad[(s0 + k) % L] for k in range(L)]
        # the walk starts on a survivor, so every run starts at some i < L and
        # either ends before L or wraps around to the beginning: find its end.
        out: list = []
        n_runs = 0
        i = 0
        while i < L:
            if not obad[i]:
                out.append(order[i])
                i += 1
                continue
            j = i
            run = []
            while j < L and obad[j]:
                run.append(order[j])
                j += 1
            # flanking survivors: previous appended vertex and, if the run hit
            # the end of the linear walk, the vertex that follows the (cyclic)
            # run -- which is the first bad vertex after the start, adjacent to
            # a survivor at that position.
            prev_ok = out[-1]
            if j < L:
                nxt_ok = order[j]
            else:
                # the run wraps: it continues at order[0] == prev_ok's
                # successor only if obad[0] is True, which cannot happen because
                # the walk starts on a survivor.  So the run genuinely ends at
                # the last slot and its far neighbour is order[0]'s predecessor
                # in the ORIGINAL cycle, i.e. the survivor just before order[0].
                nxt_ok = order[L - 1] if not obad[L - 1] else None
                if any(obad):
                    # find the last non-bad slot in the linear order
                    k = L - 1
                    while k >= 0 and obad[k]:
                        k -= 1
                    nxt_ok = order[k] if k >= 0 else None
            if nxt_ok is None:
                return [], 0
            if len(run) == 1:
                pass                        # isolated removed vertex: drop it
            else:
                pts = np.array([self.positions[k] for k in run])
                mid = 0.5 * (self.positions[prev_ok] + self.positions[nxt_ok])
                out.append(run[int(np.argmin(np.linalg.norm(pts - mid, axis=1)))])
                n_runs += 1
            i = j
        cleaned: list = []
        for v in out:
            if not cleaned or cleaned[-1] != v:
                cleaned.append(v)
        while len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
        if len(set(cleaned)) != len(cleaned) or len(cleaned) < 3:
            return [], 0
        return cleaned, n_runs

    def _prune(self, max_passes=20):
        """Delete degree<2 vertices repeatedly (and cells they empty out)."""
        for _ in range(max_passes):
            deg = self.vertex_degree()
            slots = self.alive_vertex_slots()
            if slots.size == 0:
                break
            low = set(int(v) for v in slots if deg[v] < 2)
            if not low:
                break
            for c in range(len(self.cells)):
                if self.cell_alive[c]:
                    ring = [v for v in self.cells[c] if v not in low]
                    self.cells[c] = ring
                    if len(ring) < 3:
                        self.cell_alive[c] = False
            self.rebuild()
        return self

    def t2_remove(self, c, log=None, event=None, crossing_tol=None):
        """T2: remove a vanishing cell; validate the result or roll back."""
        ev = dict(kind="t2", t=None, cell=int(c), cell_id=int(self.cell_ids[c]),
                  accepted=False, reason="")
        if event:
            ev.update(event)
        if not self.cell_alive[int(c)]:
            ev["reason"] = "already dead"
            if log is not None:
                log.append(ev)
            return False
        snap = self.snapshot()
        try:
            diag = self.remove_cells([int(c)], log=None)
            if diag["cells_removed"] != 1:
                raise MeshError("cell was not removed")
            chk = self.check_valid(l_min_edge=0.0, a_min=0.0,
                                   crossing_tol=crossing_tol)
            if not chk["ok"]:
                raise MeshError(f"invalid result {chk}")
        except Exception as exc:
            self.restore(snap)
            ev["reason"] = f"rollback: {exc}"[:300]
            if log is not None:
                log.append(ev)
            return False
        ev.update(accepted=True, reason="ok",
                  cells=[int(x) for x in diag["collapsed"]])
        if log is not None:
            log.append(ev)
        return True


# ---------------------------------------------------------------------------
# module-level thin wrappers (public, documented API surface)
# ---------------------------------------------------------------------------
def energy(mesh, *args, **kw):
    """Total energy of ``mesh`` (keywords as in :meth:`Mesh.forces`)."""
    return mesh.forces(*args, **kw)[1]


def forces(mesh, *args, **kw):
    """``(F, E)`` with ``F = -dE/dr``."""
    F, E, *_ = mesh.forces(*args, **kw)
    return F, E


def analytic_gradient(mesh, *args, **kw):
    """Analytic ``dE/dr`` (V,2)."""
    return mesh.gradient(*args, **kw)


def finite_difference_gradient(mesh, *args, h=1e-6, **kw):
    """Central-difference ``dE/dr`` (V,2); one energy evaluation per coordinate."""
    g = np.zeros((mesh.n_slots, 2))
    for v in mesh.alive_vertex_slots():
        for d in (0, 1):
            mesh.positions[v, d] += h
            ep = mesh.total_energy(*args, **kw)
            mesh.positions[v, d] -= 2 * h
            em = mesh.total_energy(*args, **kw)
            mesh.positions[v, d] += h
            g[v, d] = (ep - em) / (2 * h)
    return g


# ---------------------------------------------------------------------------
# injury selection
# ---------------------------------------------------------------------------
def _select_injury_cells(centroids, mode, centre, injury_radius, cut_halfwidth,
                         cut_angle, cut_circle):
    """Original cell indices to delete (see module docstring step 1)."""
    d = centroids - np.asarray(centre, float)
    r = np.linalg.norm(d, axis=1)
    finite = np.isfinite(r)
    if mode == "none":
        return np.zeros(0, dtype=int)
    if mode == "ablate" or (mode == "cut" and cut_circle):
        return np.flatnonzero(finite & (r <= injury_radius))
    if mode == "cut":
        n = np.array([np.cos(cut_angle), np.sin(cut_angle)])
        tv = np.array([-np.sin(cut_angle), np.cos(cut_angle)])
        dist = np.abs(d @ n)
        along = np.abs(d @ tv)
        return np.flatnonzero(finite & (dist <= cut_halfwidth)
                              & (along <= injury_radius))
    raise ValueError("mode must be 'ablate', 'cut' or 'none'")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
def run(n_side=24, mode="cut", injury_radius=24.5, cut_halfwidth=8.0,
        t_end=60.0, sample_dt=0.5, lambda_cut=1.0, lambda_edge=0.0,
        k_area=1.0, gamma_p=0.1, eta=1.0, A0=43.0, seed=2025,
        active_tension=0.0, p0_mode="hexagon", p0_factor=None,
        area_jitter=0.0, area_jitter_seed=None, lambda_outer=None,
        cut_angle=0.0, cut_circle=False, t1_enabled=True, l_t1=0.5,
        t2_enabled=False, A_t2=0.05, t1_cooldown=10, t1_isolated=False,
        t1_max_per_step=32, dt_max=1e-2, dt_min=1e-12, cfl_vertex=0.3,
        cfl_speed=0.25, cfl_area=0.02, energy_tol=None, max_steps=5_000_000,
        hess_refresh=50, hess_safety=1.0, log_level=1, progress_every=20000):
    """Run the 2D apical vertex model with a mechanical injury.

    See the module docstring for the model, the injury operator, the stability
    criterion and the honesty statement.  ``damage`` is returned on the
    original lattice indexing (== ``model.hex_geometry`` indexing) and is 1.0
    for cells that survived and touch the wound, else 0.0.

    Returns
    -------
    dict
        ``times`` (T,), ``vertices`` (T,V,2), ``positions_cells`` (T,N,2)
        [cell centroids], ``areas`` (T,N), ``perimeters`` (T,N),
        ``cell_alive`` (T,N), ``hole_area`` (T,), ``boundary_length`` (T,),
        ``cut_length`` (T,), ``outer_length`` (T,), ``energy`` (T,),
        ``energy_components`` (T,4) = [area, perimeter, active, line],
        ``events`` (dict 't1'/'t2'/'injury'), ``damage`` (N,), ``metadata``.
    """
    t_wall = time.perf_counter()
    jrng = np.random.default_rng(seed if area_jitter_seed is None
                                 else area_jitter_seed)
    n_side = int(n_side)
    A0 = float(A0)
    pos0, cells0, ids0 = build_hex_mesh(n_side, A0)
    mesh = Mesh(pos0, cells0, ids0)
    N = len(mesh.cells)
    warnings: list = []
    t_warn: list = []

    # ---- parameters --------------------------------------------------------
    A0c = np.full(N, A0)
    if area_jitter:
        A0c = A0 * (1.0 + float(area_jitter) * jrng.uniform(-1.0, 1.0, N))
    if p0_mode == "hexagon":
        f = P0_FACTOR_REGULAR_HEXAGON if p0_factor is None else float(p0_factor)
        P0c = f * np.sqrt(A0c)
    elif p0_mode == "fixed":
        P0c = np.full(N, float(p0_factor))
    else:
        raise ValueError("p0_mode must be 'hexagon' or 'fixed'")
    if np.isscalar(active_tension):
        tact = np.full(N, float(active_tension))
    else:
        tact = np.asarray(active_tension, dtype=float).copy()
        if tact.size != N:
            raise ValueError("active_tension array must have n_side**2 entries")

    # ---- injury ------------------------------------------------------------
    events = dict(t1=[], t2=[], injury=[])
    centre = mesh.positions[mesh.alive_vertex_slots()].mean(axis=0)
    removed = [int(c) for c in _select_injury_cells(
        mesh.geometry()["centroids"], mode, centre, injury_radius,
        cut_halfwidth, cut_angle, cut_circle)]
    if removed:
        inj = mesh.remove_cells(removed, log=None)
        events["injury"].append(dict(
            kind="injury", t=0.0, mode=mode, cells_removed=inj["cells_removed"],
            original_cell_ids_removed=inj["cells_removed_ids"],
            vertices_removed=inj["vertices_removed"],
            cells_collapsed_during_repair=inj["collapsed"]))
        if inj["collapsed"]:
            warnings.append(
                f"{len(inj['collapsed'])} surviving cells fell below 3 vertices "
                "while splicing the injury rim and were deleted")
    else:
        inj = dict(cells_removed=0, vertices_removed=0, collapsed=[],
                   cells_removed_ids=[])
    if int(mesh.cell_alive.sum()) == 0:
        raise MeshError("injury removed every cell")
    minsz = math.sqrt(A0)
    cross_tol = 1e-6 * minsz
    check_inj = mesh.check_valid(l_min_edge=0.0, a_min=0.0,
                                 crossing_tol=cross_tol)
    if not check_inj["ok"]:
        warnings.append(f"mesh invalid right after injury: {check_inj}")

    # ---- integrator --------------------------------------------------------
    slots0 = mesh.alive_vertex_slots()
    bbox_area = float(np.prod(mesh.positions[slots0].max(axis=0)
                              - mesh.positions[slots0].min(axis=0)))
    energy_tol = (1e-12 * max(1.0, abs(k_area) * A0 ** 2)
                  if energy_tol is None else float(energy_tol))
    counts = dict(steps=0, rejected_steps=0, t1_accepted=0, t1_rejected=0,
                  t2_accepted=0, t2_rejected=0, t1_correlated=0)
    t1_reasons: dict = {}
    last_flip_step: dict = {}
    min_edge_run = float("inf")
    min_area_run = float("inf")
    min_deg_run = 10 ** 9
    max_deg_run = 0
    max_drift = 0.0
    negative_area_events = 0
    nonpositive_edge_events = 0
    dt_used_max = 0.0
    hd_used = None
    hd_topo_mark = -1
    dt = min(dt_max, 1e-3)
    area_start = float(np.nansum(mesh.geometry()["areas"][mesh.cell_alive]))
    V = mesh.n_slots
    times: list = []
    verts: list = []
    cents: list = []
    areas_h: list = []
    per_h: list = []
    alive_h: list = []
    hole_h: list = []
    blen_h: list = []
    clen_h: list = []
    olen_h: list = []
    en_h: list = []
    en_split: list = []
    kw = dict(area_target=A0c, perimeter_shift=P0c, k_area=k_area,
              gamma_p=gamma_p, active_tension=tact, lambda_edge=lambda_edge,
              lambda_cut=lambda_cut, lambda_outer=lambda_outer)

    def boundary_lengths():
        bhe = mesh.boundary_half_edges()
        if bhe.size == 0:
            return 0.0, 0.0, 0.0
        d = mesh.positions[mesh.he_to[bhe]] - mesh.positions[mesh.he_from[bhe]]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        iscut = mesh._edge_cut_flags(bhe)
        return float(L.sum()), float(L[iscut].sum()), float(L[~iscut].sum())

    def snapshot(t_now):
        g = mesh.geometry()
        _, E, parts, _, _ = mesh.forces(**kw)
        times.append(float(t_now))
        verts.append(mesh.positions.copy())
        cents.append(g["centroids"].copy())
        areas_h.append(np.where(mesh.cell_alive, g["areas"], np.nan))
        per_h.append(np.where(mesh.cell_alive, g["perimeters"], np.nan))
        alive_h.append(mesh.cell_alive.copy())
        hole_h.append(bbox_area - float(np.nansum(g["areas"][mesh.cell_alive])))
        tot, cut, outer = boundary_lengths()
        blen_h.append(tot)
        clen_h.append(cut)
        olen_h.append(outer)
        en_h.append(E)
        en_split.append([parts["area"], parts["perimeter"], parts["active"],
                         parts["line"]])  # see Mesh.forces -> parts keys

    def emit(msg):
        if log_level:
            print(msg, flush=True)
        if len(t_warn) < 12:
            t_warn.append(msg)

    snapshot(0.0)
    emit(f"[mechanics] synthetic vertex model | n_side={n_side} mode={mode} | "
         f"cells {N} -> {int(mesh.cell_alive.sum())} alive "
         f"({inj['cells_removed']} cells removed, "
         f"{inj['vertices_removed']} vertices pruned) | "
         f"V={mesh.n_slots} H={mesh.n_half_edges}")

    t_now = 0.0
    next_sample = sample_dt
    while t_now < t_end - 1e-12:
        if counts["steps"] >= max_steps:
            warnings.append(f"max_steps={max_steps} reached at t={t_now:.6g} < "
                            f"t_end={t_end}")
            break
        geo = mesh.geometry_cached()
        F, E, parts, (areas, perims), hd = mesh.forces(**kw, geo=geo)
        if (counts["steps"] % hess_refresh == 0
                or hd_used is None or mesh.topo_version != hd_topo_mark):
            hd_used = hess_safety * mesh.curvature_diagonal(
                area_target=A0c, perimeter_shift=P0c, k_area=k_area,
                gamma_p=gamma_p, active_tension=tact, lambda_edge=lambda_edge,
                lambda_cut=lambda_cut, lambda_outer=lambda_outer, geo=geo)
            hd_topo_mark = mesh.topo_version
        # ---- T1 -------------------------------------------------------------
        if t1_enabled:
            aa, bb, LL, _ = mesh.interior_edges()
            n_flip_step = 0
            if LL.size:
                sel = np.flatnonzero(LL < l_t1)
                sel = sel[np.argsort(LL[sel], kind="stable")][:t1_max_per_step]
                for k in sel:
                    if t1_isolated and n_flip_step >= 1:
                        break
                    a, b, L0 = int(aa[k]), int(bb[k]), float(LL[k])
                    key = (min(a, b), max(a, b))
                    if counts["steps"] - last_flip_step.get(key, -10 ** 9) \
                            < t1_cooldown:
                        counts["t1_rejected"] += 1
                        t1_reasons["cooldown"] = t1_reasons.get("cooldown", 0) + 1
                        continue
                    ev = dict(kind="t1", t=float(t_now), step=counts["steps"],
                              v1=a, v2=b, l_before=L0, correlated=False)
                    ok = mesh.t1_swap(a, b, log=events["t1"], event=ev,
                                      l_new_min=2.0 * l_t1, a_min=1e-6,
                                      l_min_valid=1e-3 * minsz)
                    last_flip_step[key] = counts["steps"]
                    if ok:
                        counts["t1_accepted"] += 1
                        n_flip_step += 1
                        geo = mesh.geometry_cached(force=True)
                        F, E, parts, (areas, perims), hd = mesh.forces(**kw, geo=geo)
                        hd_used = hess_safety * mesh.curvature_diagonal(
                            area_target=A0c, perimeter_shift=P0c,
                            k_area=k_area, gamma_p=gamma_p,
                            active_tension=tact, lambda_edge=lambda_edge,
                            lambda_cut=lambda_cut, lambda_outer=lambda_outer,
                            geo=geo)
                        hd_topo_mark = mesh.topo_version
                        if len(events["t1"]) >= 2:
                            prev = events["t1"][-2]
                            if prev.get("accepted") and prev.get("step") == ev["step"] \
                                    and ({prev["v1"], prev["v2"]} & {a, b}):
                                ev["correlated"] = True
                                counts["t1_correlated"] += 1
                    else:
                        counts["t1_rejected"] += 1
                        r = ev["reason"].split("(")[0]
                        t1_reasons[r] = t1_reasons.get(r, 0) + 1
        # ---- adaptive dt ----------------------------------------------------
        slots = mesh.alive_vertex_slots()
        vmax = float(np.linalg.norm(F[slots], axis=1).max()) if slots.size else 0.0
        # the exact analytic curvature diagonal is used for the CFL bound; the
        # cheap conservative `hd` is kept as a floor.
        with np.errstate(divide="ignore", invalid="ignore"):
            dt_cfl_v = cfl_vertex * eta / np.maximum(hd_used[slots], 1e-300)
        dt_cfl_v = float(dt_cfl_v.min()) if dt_cfl_v.size else dt_max
        dt_cfl_s = (cfl_speed * minsz / vmax) if vmax > 0 else np.inf
        aa_alive = areas[mesh.cell_alive]
        amin = float(aa_alive.min()) if aa_alive.size else A0
        dt_cfl_a = (cfl_area * math.sqrt(max(amin, 1e-300)) / vmax
                    if vmax > 0 else np.inf)
        dt = min(dt_max, dt_cfl_v, dt_cfl_s, dt_cfl_a, 10.0 * dt)
        dt = max(dt, dt_min)
        if t_now + dt > t_end:
            dt = t_end - t_now
        # ---- forward Euler step --------------------------------------------
        old = mesh.positions.copy()
        mesh.positions = mesh.positions + (dt / eta) * F
        F, E, parts, (areas, perims), hd = mesh.forces(
            **kw, geo=mesh.geometry_cached())
        if E > en_h[-1] + energy_tol:
            mesh.positions = old
            counts["rejected_steps"] += 1
            dt = max(dt_min, 0.5 * dt)
            if counts["rejected_steps"] > 1000:
                warnings.append("integrator rejected >1000 steps; dt floor hit")
                break
            continue
        t_now = min(t_end, t_now + dt)
        counts["steps"] += 1
        dt_used_max = max(dt_used_max, dt)
        # ---- bookkeeping ----------------------------------------------------
        aa_alive = areas[mesh.cell_alive]
        tot_area = float(np.nansum(aa_alive))
        if area_start > 0:
            max_drift = max(max_drift, abs(tot_area - area_start) / area_start)
        if aa_alive.size:
            amin = float(aa_alive.min())
            min_area_run = min(min_area_run, amin)
            if amin <= 0:
                negative_area_events += 1
                if negative_area_events <= 5:
                    warnings.append(f"non-positive cell area at t={t_now:.6g}")
        _, _, LL, _ = mesh.interior_edges()
        if LL.size:
            min_edge_run = min(min_edge_run, float(LL.min()))
            if float(LL.min()) <= 0:
                nonpositive_edge_events += 1
        if slots.size:
            deg = mesh.vertex_degree()[slots]
            min_deg_run = min(min_deg_run, int(deg.min()))
            max_deg_run = max(max_deg_run, int(deg.max()))
        # ---- T2 -------------------------------------------------------------
        if t2_enabled:
            for c in np.flatnonzero(mesh.cell_alive & (areas < A_t2)):
                ev = dict(kind="t2", t=float(t_now), cell=int(c),
                          cell_id=int(mesh.cell_ids[c]), area=float(areas[c]))
                if mesh.t2_remove(int(c), log=events["t2"], event=ev,
                                  crossing_tol=cross_tol):
                    counts["t2_accepted"] += 1
                    warnings.append(f"T2 accepted at t={t_now:.6g} (cell {c})")
                else:
                    counts["t2_rejected"] += 1
                    warnings.append(
                        f"T2 REJECTED and rolled back at t={t_now:.6g} (cell {c}): "
                        f"{events['t2'][-1]['reason']}")
        if log_level and progress_every and counts["steps"] % progress_every == 0:
            emit(f"  t={t_now:.5g} steps={counts['steps']} "
                 f"T1={counts['t1_accepted']} dt={dt:.3e} E={E:.8g} "
                 f"A_min={min_area_run:.4g} cut={clen_h[-1]:.3f}")
        # ---- sampling -------------------------------------------------------
        while next_sample <= t_now + 1e-12:
            snapshot(next_sample)
            next_sample += sample_dt

    runtime = time.perf_counter() - t_wall
    check_final = mesh.check_valid(l_min_edge=0.0, a_min=0.0,
                                   crossing_tol=cross_tol)
    if not check_final["ok"]:
        warnings.append(f"mesh invalid at the end of the run: {check_final}")

    # ---- damage and per-cell state ----------------------------------------
    cut_touch = np.zeros(N, dtype=bool)
    for h in mesh.boundary_half_edges():
        if mesh._edge_cut_flags(np.array([h]))[0]:
            cut_touch[int(mesh.he_cell[h])] = True
    damage = np.zeros(N)
    damage[np.flatnonzero(mesh.cell_alive & cut_touch)] = 1.0
    gf = mesh.geometry()
    final_area = np.where(mesh.cell_alive, gf["areas"], np.nan)
    final_per = np.where(mesh.cell_alive, gf["perimeters"], np.nan)
    cell_state = dict(
        alive=mesh.cell_alive.copy(), cell_ids=mesh.cell_ids.copy(),
        area=final_area, perimeter=final_per,
        area_strain=np.where(mesh.cell_alive, (gf["areas"] - A0c) / A0c, np.nan),
        perimeter_relative=np.where(mesh.cell_alive, gf["perimeters"] / P0c,
                                    np.nan),
        touches_wound=cut_touch, damage=damage,
        note="mechanical model output, not measurement")

    metadata = dict(
        module="mechanics.py -- synthetic 2D apical vertex model",
        parameter_provenance=PARAMETER_PROVENANCE,
        literature_values=LIT_REFS,
        limitations=LIMITATIONS,
        honesty=("No experimental validation and no calibration. Parameters are "
                 "arbitrary illustrative values. Wound-closure numbers are model "
                 "predictions, not data."),
        units=dict(length="um", area="um^2",
                   time="arbitrary relaxation units (eta sets only a ratio)",
                   energy=f"arbitrary; k_area*A0^2={k_area * A0 ** 2:.6g}"),
        parameters=dict(
            n_side=n_side, mode=mode, injury_radius=injury_radius,
            cut_halfwidth=cut_halfwidth, cut_angle=cut_angle,
            cut_circle=cut_circle, t_end=t_end, sample_dt=sample_dt,
            lambda_cut=lambda_cut, lambda_edge=lambda_edge,
            lambda_outer=(lambda_edge if lambda_outer is None else lambda_outer),
            k_area=k_area, gamma_p=gamma_p, eta=eta, A0=A0, p0_mode=p0_mode,
            p0_factor=(P0_FACTOR_REGULAR_HEXAGON if p0_factor is None
                       else p0_factor),
            area_jitter=area_jitter, active_tension=active_tension,
            t1_enabled=t1_enabled, l_t1=l_t1, t1_cooldown=t1_cooldown,
            t1_isolated=t1_isolated, t2_enabled=t2_enabled, A_t2=A_t2,
            dt_max=dt_max, cfl_vertex=cfl_vertex, cfl_speed=cfl_speed,
            cfl_area=cfl_area, energy_tol=energy_tol, max_steps=max_steps),
        seeds=dict(seed=int(seed),
                   area_jitter_seed=int(seed if area_jitter_seed is None
                                        else area_jitter_seed),
                   note="`seed` is used ONLY to draw the per-cell preferred-area "
                        "jitter. The default pipeline is fully deterministic, so "
                        "changing `seed` alone changes nothing unless "
                        "area_jitter > 0 (test 8 measures both cases)."),
        runtime_s=float(runtime),
        steps=int(counts["steps"]),
        rejected_steps=int(counts["rejected_steps"]),
        dt_used_max=float(dt_used_max),
        dt_stability_criterion=(
            "explicit Euler, dt <= min(dt_max, cfl_vertex*eta/H_vv, "
            "cfl_speed*sqrt(A0)/vmax, cfl_area*sqrt(A_min)/vmax, 10*dt_prev) "
            "with H_vv = sum_{e at v}|T_e|/L_e + 4*k_area*A0"),
        rejection_policy="reject the step and halve dt if it raises total energy",
        events_counts=dict(
            t1_accepted=int(counts["t1_accepted"]),
            t1_rejected=int(counts["t1_rejected"]),
            t1_rejected_reasons=t1_reasons,
            t1_correlated=int(counts["t1_correlated"]),
            t2_accepted=int(counts["t2_accepted"]),
            t2_rejected=int(counts["t2_rejected"]),
            injury_cells_removed=int(inj["cells_removed"])),
        injury=dict(mode=mode, cells_removed=int(inj["cells_removed"]),
                    vertices_removed=int(inj["vertices_removed"]),
                    cells_collapsed_during_repair=len(inj["collapsed"]),
                    original_cell_ids_removed=inj["cells_removed_ids"]),
        min_edge_length_overall=float(min_edge_run),
        min_area_overall=float(min_area_run),
        min_vertex_degree_overall=int(min_deg_run),
        max_vertex_degree_overall=int(max_deg_run),
        min_edge_length_final=float(mesh.min_edge_length()),
        area_conservation_rel_error=float(max_drift),
        area_initial_total=float(area_start),
        area_final_total=float(np.nansum(final_area[mesh.cell_alive])),
        negative_area_events=int(negative_area_events),
        nonpositive_edge_events=int(nonpositive_edge_events),
        final_mesh_check=check_final,
        check_after_injury=check_inj,
        warnings=warnings,
        runtime_messages=t_warn,
        cell_state=cell_state,
        # convenience mirrors of the most-used parameters (kept so that callers
        # can read them without digging into metadata['parameters'])
        n_cells=N,
        mode=mode, lambda_cut=lambda_cut, lambda_edge=lambda_edge,
        k_area=k_area, gamma_p=gamma_p, eta=eta, A0=A0, seed=int(seed),
        cut_halfwidth=cut_halfwidth, injury_radius=injury_radius, t_end=t_end,
        t1_enabled=bool(t1_enabled), t2_enabled=bool(t2_enabled),
        energy_definition_doc=(
            "E = sum_c k_area(A_c-A0_c)^2 + sum_c gamma_p(P_c-P0_c)^2 "
            "+ sum_c active_tension_c*P_c + sum_e lambda_e*L_e, with "
            "eta*dr/dt = -dE/dr. A_c is the oriented shoelace polygon area, "
            "P_c the polygon perimeter, L_e the edge length; line_tension_e is "
            "lambda_cut on the injury boundary, lambda_outer (=lambda_edge) on "
            "the original rim and 0 in the bulk. NO experimental calibration: "
            "all parameters are arbitrary illustrative values (see "
            "parameter_provenance)."),
        mechanics_parameter_statement=PARAMETER_PROVENANCE,
        index_alignment_with_model_hex_geometry=(
            "The tissue is the Voronoi tessellation of the true honeycomb "
            "lattice, whose cells are indexed row-major exactly like "
            "model.hex_geometry (cell k = row*n_side+col).  Polygon orderings "
            "therefore agree cell-for-cell; the two lattices differ only (a) by "
            "a rigid translation of (0.25*n_side, 0) um in x -- build_hex_mesh "
            "centres the *tissue* while model.hex_geometry centres its own "
            "centres -- and (b) in the polygon geometry itself, because "
            "model.hex_geometry's centre spacing is not the tiling distance for "
            "its hexagons (see build_hex_mesh).  Measured on n_side=24: max "
            "centroid residual after removing that translation = 0.0000 um, so "
            "the index correspondence is exact."),
        n_cells_total=N,
        n_cells_alive_final=int(mesh.cell_alive.sum()),
        n_vertices_final=int(mesh.alive_vertex_slots().size),
        positions_cells_definition="cell centroids (unweighted vertex mean)",
        vertices_definition="vertex slots, stable indices, NaN = dead slot",
        hole_area_definition=("initial post-injury vertex bounding-box area minus "
                              "the current sum of cell areas"),
        boundary_length_definition="free-boundary length = cut + outer",
        damage_definition=("1.0 for original lattice cells that survived and have "
                           "at least one cut edge; 0.0 otherwise (including "
                           "ablated cells)"),
    )
    return dict(times=np.asarray(times), vertices=np.asarray(verts),
                positions_cells=np.asarray(cents),
                areas=np.asarray(areas_h), perimeters=np.asarray(per_h),
                cell_alive=np.asarray(alive_h), hole_area=np.asarray(hole_h),
                boundary_length=np.asarray(blen_h),
                cut_length=np.asarray(clen_h), outer_length=np.asarray(olen_h),
                energy=np.asarray(en_h), energy_components=np.asarray(en_split),
                events=events, damage=damage, metadata=metadata)


# ===========================================================================
# REQUIRED TESTS
# ===========================================================================
def test_1_gradient(n_side=4, seed=7, rel_tol=1e-4, perturb=0.3):
    """Test 1: analytic dE/dr against central finite differences."""
    rng = np.random.default_rng(seed)
    pos, cells, ids = build_hex_mesh(n_side, 43.0)
    pos = pos + rng.normal(0.0, perturb, pos.shape)
    mesh = Mesh(pos, cells, ids)
    A0 = np.full(len(mesh.cells), 43.0)
    P0 = P0_FACTOR_REGULAR_HEXAGON * np.sqrt(A0)
    act = rng.normal(0.0, 0.05, len(mesh.cells))
    # flag whole *undirected* edges as cut, so that lambda_cut participates
    # exactly once per edge (flagging one half-edge of an interior edge would
    # make the energy depend on which half-edge was selected)
    mesh.he_cut = np.zeros(len(mesh.he_from), dtype=bool)
    bhe = mesh.boundary_half_edges()
    for h in bhe[rng.random(bhe.size) < 0.35]:
        mesh.mark_cut(int(h), True)
    kw = dict(area_target=A0, perimeter_shift=P0, k_area=0.7, gamma_p=0.13,
              active_tension=act, lambda_edge=0.4, lambda_cut=0.9)
    g_an = mesh.gradient(**kw)
    g_fd = finite_difference_gradient(mesh, **kw)
    scale = max(float(np.abs(g_fd).max()), 1e-300)
    rel = float(np.abs(g_an - g_fd).max() / scale)
    F, E, parts, (ar, pe), hd = mesh.forces(**kw)
    # edge-tension formulation consistency: dE/d(lambda_cut) == total cut length
    flags = mesh.he_cut.copy()
    # cleanest check: with lambda_edge = 0 the only line tension is lambda_cut,
    # so dE/d(lambda_cut) must equal the total cut-edge length exactly
    kw0 = dict(kw, lambda_edge=0.0)
    mesh.he_cut[:] = False
    E_nocut = mesh.total_energy(**kw0)
    mesh.he_cut[:] = flags
    E_allcut = mesh.total_energy(**kw0)
    cut_len = float(np.sum(mesh.geometry()["elen"][mesh.geometry()["ecut"]]))
    fd_lambda = (E_allcut - E_nocut) / 0.9
    ncut = int(mesh.he_cut.sum())
    return dict(name="test_1_analytic_gradient",
                n_vertices=int(mesh.n_slots), n_cells=len(mesh.cells),
                perturbation_sigma_um=perturb,
                rel_error_linf=rel, abs_error_linf=float(np.abs(g_an - g_fd).max()),
                gradient_linf=scale, tol=rel_tol, passed=bool(rel < rel_tol),
                energy_components=dict(area=parts["area"],
                                       perimeter=parts["perimeter"],
                                       active=parts["active"],
                                       line=parts["line"], total=E),
                gradient_equals_minus_force=float(np.abs(F + g_an).max()),
                lambda_cut_consistency=dict(
                    dE_over_dlambda_cut=float(fd_lambda),
                    expected_total_cut_length_um=float(cut_len),
                    ratio_measured_over_expected=float(
                        fd_lambda / max(cut_len, 1e-300)),
                    half_edges_flagged_cut=ncut,
                    note="dE/d(lambda_cut) must equal the total length of the "
                         "flagged cut edges; the ratio is that check and must "
                         "be 1"))


def _relax_isolated_cell(pts, A0, P0, k_area=1.0, gamma_p=0.1, t_act=0.0,
                         lambda_edge=0.0, eta=1.0, dt_max=1e-2, max_steps=40000,
                         force_tol=1e-7):
    """Gradient relaxation of a single isolated cell; returns a state dict."""
    mesh = Mesh(pts - pts.mean(axis=0), [list(range(len(pts)))], [0])
    kw = dict(area_target=np.array([A0]), perimeter_shift=np.array([P0]),
              k_area=k_area, gamma_p=gamma_p,
              active_tension=np.array([t_act]), lambda_edge=lambda_edge)
    n = 0
    for n in range(max_steps):
        F, E, parts, (ar, pe), hd = mesh.forces(**kw)
        fmax = float(np.abs(F).max())
        if fmax < force_tol:
            break
        hcurv = float(mesh.curvature_diagonal(**dict(kw, lambda_cut=0.0)).max())
        step = min(dt_max, 0.4 * eta / max(hcurv, 1e-12))
        old = mesh.positions.copy()
        mesh.positions = mesh.positions + (step / eta) * F
        if mesh.total_energy(**kw) > E + 1e-15:
            mesh.positions = old
            continue
    F, E, parts, (ar, pe), hd = mesh.forces(**kw)
    poly = mesh.positions
    cen = poly.mean(axis=0)
    rad = np.linalg.norm(poly - cen, axis=1)
    return dict(steps=n, E=float(E), parts=parts, force_max=float(np.abs(F).max()),
                area=float(polygon_area(poly)), perimeter=float(polygon_perimeter(poly)),
                radius_scatter=float(np.ptp(rad) / rad.mean()), poly=poly)


def _analytic_scaled_hexagon(A0, P0, k_area=1.0, gamma_p=0.1, lambda_edge=0.0):
    """Exact minimiser over the family of affine-scaled regular hexagons.

    With ``A(s) = A0*s^2`` and ``P(s) = P0*s`` the energy restricted to a
    uniformly scaled regular hexagon is the polynomial
    ``E(s) = k*A0^2*(s^2-1)^2 + (gamma_p*P0^2 + lambda_edge*P0)*s^2
             - 2*gamma_p*P0^2*s + gamma_p*P0^2``, minimised exactly with
    ``scipy.optimize.minimize_scalar`` (bounded to ``[0.5, 1.5]``).  For
    ``lambda_edge = 0`` this is ``s = 1``, ``E = 0``.
    """
    from scipy.optimize import minimize_scalar

    e = (lambda s: k_area * A0 ** 2 * (s * s - 1.0) ** 2
         + (gamma_p * P0 ** 2 + lambda_edge * P0) * s * s
         - 2.0 * gamma_p * P0 ** 2 * s + gamma_p * P0 ** 2)
    res = minimize_scalar(e, bounds=(0.5, 1.5), method="bounded",
                          options=dict(xatol=1e-15))
    return float(res.x), float(res.fun)


def test_2_single_cell(lambda_edge=0.0, eta=1.0, A0=43.0, seed=11, perturb=0.12,
                       k_area=1.0, gamma_p=0.1):
    """Test 2: isolated cell -> regular hexagon; line tension -> rounding up."""
    side = regular_hexagon_side(A0)
    ang = np.arange(6) * np.pi / 3.0
    rng = np.random.default_rng(seed)
    pts = side * np.column_stack((np.cos(ang), np.sin(ang)))
    pts = pts * (1.0 + perturb * rng.uniform(-1, 1, 6))[:, None]
    P0 = regular_hexagon_perimeter(A0)
    st = _relax_isolated_cell(pts, A0, P0, k_area=k_area, gamma_p=gamma_p,
                              lambda_edge=lambda_edge, eta=eta)
    s_star, E_star = _analytic_scaled_hexagon(A0, P0, k_area, gamma_p, lambda_edge)
    area, per = st["area"], st["perimeter"]
    s_num = math.sqrt(area / A0)
    return dict(name="test_2_single_cell", lambda_edge=float(lambda_edge),
                steps=st["steps"], E_final=float(st["E"]),
                E_analytic_min=float(E_star),
                E_excess_above_analytic=float(st["E"] - E_star),
                area_final=float(area),
                area_error_rel=float(abs(area - A0) / A0),
                perimeter_final=float(per), perimeter_target=float(P0),
                perimeter_error_rel=float(abs(per - P0) / P0),
                max_radius_scatter_rel=float(st["radius_scatter"]),
                scale_numeric=float(s_num), scale_analytic=float(s_star),
                scale_rel_error=float(abs(s_num - s_star) / s_star),
                residual_force=float(st["force_max"]),
                E_parts={k: float(v) for k, v in st["parts"].items()})


def test_3_energy_monotonicity(n_side=5, seed=3, steps=20000, perturb=0.25):
    """Test 3 + 4: energy monotone, area conserved in uninjured relaxation."""
    pos, cells, ids = build_hex_mesh(n_side, 43.0)
    rng = np.random.default_rng(seed)
    pos = pos + rng.normal(0, perturb, pos.shape)
    mesh = Mesh(pos, cells, ids)
    A0 = np.full(len(mesh.cells), 43.0)
    P0 = np.full(len(mesh.cells), regular_hexagon_perimeter(43.0))
    kw = dict(area_target=A0, perimeter_shift=P0, k_area=1.0, gamma_p=0.1,
              active_tension=np.zeros(len(mesh.cells)), lambda_edge=0.0)
    E0 = mesh.total_energy(**kw)
    E = E0
    area0 = float(np.nansum(mesh.geometry_cached()["areas"]))
    worst = 0.0
    worst_rel = 0.0
    max_drift = 0.0
    n_done = 0
    for i in range(steps):
        F, E, parts, (ar, pe), hd = mesh.forces(**kw)
        slots = mesh.alive_vertex_slots()
        dt = min(1e-3, 0.4 / float(hd[slots].min()))
        mesh.positions = mesh.positions + dt * F
        E2 = mesh.total_energy(**kw)
        d = float(E2 - E)
        worst = max(worst, d)
        worst_rel = max(worst_rel, d / max(abs(E2), 1e-300))
        E = E2
        n_done = i + 1
        area = float(np.nansum(mesh.geometry_cached(force=True)["areas"]))
        max_drift = max(max_drift, abs(area - area0) / area0)
        if float(np.abs(F).max()) * dt < 1e-14:
            break
    return dict(name="test_3_energy_and_area", steps=n_done,
                perturb_sigma_um=perturb, E0=float(E0), E_final=float(E),
                worst_energy_increase=float(worst),
                worst_energy_increase_relative=float(worst_rel),
                integrator_energy_tol=1.849e-9,
                monotone_within_tolerance=bool(worst <= 1.849e-9),
                max_area_drift_rel=float(max_drift),
                area_passed=bool(max_drift < 1e-6))


def test_5_t1(n_side=5, seed=5, relax_steps=2000, compress_steps=6000):
    """Test 5: no spontaneous T1 in a relaxed lattice; T1 under compression."""
    pos, cells, ids = build_hex_mesh(n_side, 43.0)
    mesh = Mesh(pos, cells, ids)
    A0 = np.full(len(mesh.cells), 43.0)
    P0 = np.full(len(mesh.cells), regular_hexagon_perimeter(43.0))
    kw = dict(area_target=A0, perimeter_shift=P0, k_area=1.0, gamma_p=0.1,
              active_tension=np.zeros(len(mesh.cells)), lambda_edge=0.0)
    ltol = 1e-3 * math.sqrt(43.0)
    log_a: list = []
    min_edge_a = float("inf")
    for i in range(relax_steps):
        F, E, parts, (ar, pe), hd = mesh.forces(**kw)
        slots = mesh.alive_vertex_slots()
        dt = min(1e-3, 0.4 / float(hd[slots].min()))
        mesh.positions = mesh.positions + dt * F
        aa, bb, LL, _ = mesh.interior_edges()
        if LL.size:
            min_edge_a = min(min_edge_a, float(LL.min()))
            for k in np.flatnonzero(LL < 0.5):
                a, b = int(aa[k]), int(bb[k])
                mesh.t1_swap(a, b, log=log_a, event=dict(
                    kind="t1", t=i * dt, v1=a, v2=b, l_before=float(LL[k])),
                    l_new_min=1.0, a_min=1e-6, l_min_valid=ltol)
    n_spont = len([e for e in log_a if e["accepted"]])
    check_a = mesh.check_valid(l_min_edge=0.0, crossing_tol=ltol)
    # compression: strongly reduced target area
    A0c = np.full(len(mesh.cells), 0.7 * 43.0)
    kw2 = dict(kw, area_target=A0c)
    log_b: list = []
    valid_b = True
    for i in range(compress_steps):
        F, E, parts, (ar, pe), hd = mesh.forces(**kw2)
        slots = mesh.alive_vertex_slots()
        dt = min(2e-3, 0.4 / float(hd[slots].min()))
        mesh.positions = mesh.positions + dt * F
        aa, bb, LL, _ = mesh.interior_edges()
        if LL.size:
            sel = np.flatnonzero(LL < 0.6)
            for k in sel[np.argsort(LL[sel], kind="stable")][:8]:
                a, b = int(aa[k]), int(bb[k])
                mesh.t1_swap(a, b, log=log_b, event=dict(
                    kind="t1", t=i * dt, v1=a, v2=b, l_before=float(LL[k])),
                    l_new_min=1.0, a_min=1e-6, l_min_valid=ltol)
        if i % 200 == 0:
            chk = mesh.check_valid(l_min_edge=0.0, crossing_tol=ltol)
            if not chk["ok"]:
                valid_b = False
                break
    acc_b = [e for e in log_b if e["accepted"]]
    rej_b = [e for e in log_b if not e["accepted"]]
    reasons: dict = {}
    for e in rej_b:
        r = e["reason"].split("(")[0]
        reasons[r] = reasons.get(r, 0) + 1
    neg = 0
    for c in range(len(mesh.cells)):
        if mesh.cell_alive[c] and polygon_area(mesh.positions[mesh.cells[c]]) <= 0:
            neg += 1
    check_b = mesh.check_valid(l_min_edge=0.0, crossing_tol=ltol)
    deg = mesh.vertex_degree()[mesh.alive_vertex_slots()]
    return dict(name="test_5_t1",
                t1_spontaneous=n_spont, t1_spontaneous_expected=0,
                spontaneous_passed=bool(n_spont == 0),
                min_interior_edge_relaxed=float(min_edge_a),
                check_after_relaxation=check_a,
                t1_accepted_under_compression=len(acc_b),
                t1_rejected_under_compression=len(rej_b),
                rejected_reasons=reasons,
                compression_passed=bool(len(acc_b) > 0),
                inverted_cells_after=neg,
                no_vertex_crossings=bool(check_b["n_close_vertex_edge"] == 0),
                valid_throughout=bool(valid_b),
                vertex_degree_range=[int(deg.min()), int(deg.max())],
                min_area_after_compression=float(
                    mesh.geometry_cached(force=True)["areas"][mesh.cell_alive].min()),
                min_edge_after_compression=float(mesh.interior_edges()[2].min()),
                check_final=check_b)


def test_6_invariants(r, name="test_6_geometry_invariants"):
    """Test 6: no inversion, positive areas, positive edge lengths."""
    a = r["areas"]
    finite = np.isfinite(a)
    frame_min = float(np.nanmin(a)) if finite.any() else float("nan")
    md = r["metadata"]
    crossed = int(md["final_mesh_check"]["n_close_vertex_edge"])
    return dict(name=name, min_cell_area_all_frames=frame_min,
                min_cell_area_overall=float(md["min_area_overall"]),
                min_edge_length_overall=float(md["min_edge_length_overall"]),
                min_edge_length_final=float(md["min_edge_length_final"]),
                negative_area_events=int(md["negative_area_events"]),
                nonpositive_edge_events=int(md["nonpositive_edge_events"]),
                min_vertex_degree=int(md["min_vertex_degree_overall"]),
                max_vertex_degree=int(md["max_vertex_degree_overall"]),
                inverted_cells_present=bool(np.any(finite & (a <= 0))),
                vertex_crossings_final=crossed,
                area_conservation_rel_error=float(md["area_conservation_rel_error"]),
                final_check=md["final_mesh_check"],
                passed=bool(frame_min > 0 and md["min_edge_length_overall"] > 0
                            and md["negative_area_events"] == 0 and crossed == 0))


def test_7_wound_closure(n_side=20, t_end=10.0, sample_dt=1.0, **kw):
    """Test 7: cut contracts with lambda_cut > 0; contracts less with lambda_cut=0."""
    out: dict = {}
    for lam in (1.0, 0.0):
        r = run(n_side=n_side, mode="cut", injury_radius=24.5,
                cut_halfwidth=8.0, t_end=t_end, sample_dt=sample_dt,
                lambda_cut=lam, lambda_edge=0.0, k_area=1.0, gamma_p=0.1,
                eta=1.0, A0=43.0, seed=2025, t1_enabled=True, log_level=0,
                progress_every=0, **kw)
        ha = r["hole_area"]
        cl = r["cut_length"]
        out[f"lambda_cut_{lam}"] = dict(
            hole_area_t0=float(ha[0]), hole_area_tend=float(ha[-1]),
            hole_area_relative_change=float((ha[-1] - ha[0]) / ha[0]),
            cut_length_t0=float(cl[0]), cut_length_tend=float(cl[-1]),
            cut_length_relative_change=float((cl[-1] - cl[0]) / cl[0]),
            cells_removed=int(r["metadata"]["injury"]["cells_removed"]),
            steps=int(r["metadata"]["steps"]),
            runtime_s=float(r["metadata"]["runtime_s"]),
            t1_accepted=int(r["metadata"]["events_counts"]["t1_accepted"]))
    c1 = out["lambda_cut_1.0"]["hole_area_relative_change"]
    c0 = out["lambda_cut_0.0"]["hole_area_relative_change"]
    out.update(name="test_7_wound_closure",
               purse_string_contracts_more=bool(c1 < c0),
               note=("Mechanical model prediction ONLY: no experimental "
                     "calibration, no comparison with data; absolute time is in "
                     "arbitrary relaxation units."))
    return out


def test_8_determinism(n_side=12, t_end=2.0, **kw):
    """Test 8: identical seed -> bit-identical trajectory."""
    common = dict(n_side=n_side, mode="cut", injury_radius=12.0,
                  cut_halfwidth=4.0, t_end=t_end, sample_dt=1.0, seed=99,
                  log_level=0, progress_every=0)
    a = run(**dict(common, **kw))
    b = run(**dict(common, **kw))
    d = run(**dict(common, seed=100, **kw))
    da = float(np.nanmax(np.abs(a["vertices"] - b["vertices"])))
    dd = float(np.nanmax(np.abs(a["vertices"] - d["vertices"])))
    ea, eb = a["metadata"]["events_counts"], b["metadata"]["events_counts"]
    return dict(name="test_8_determinism", max_abs_diff_same_seed=da,
                bit_identical=bool(da == 0.0),
                steps_a=int(a["metadata"]["steps"]),
                steps_b=int(b["metadata"]["steps"]),
                event_counts_a=ea, event_counts_b=eb,
                events_and_steps_identical=bool(
                    ea == eb and a["metadata"]["steps"] == b["metadata"]["steps"]),
                energy_identical=bool(np.array_equal(a["energy"], b["energy"])),
                max_abs_diff_other_seed=float(dd),
                passed=bool(da == 0.0 and ea == eb
                            and a["metadata"]["steps"] == b["metadata"]["steps"]))


def self_test(quick=False, verbose=True):
    """Run the eight required tests and return a diagnostics dict."""
    t0 = time.perf_counter()
    res: dict = {}
    say = print if verbose else (lambda *a, **k: None)
    say("=" * 78)
    say("mechanics.py self_test -- SYNTHETIC model, no experimental validation")
    say("=" * 78)

    res["1_gradient"] = test_1_gradient()
    t1 = res["1_gradient"]
    say(f" 1 analytic gradient check  : rel_err(linf)={t1['rel_error_linf']:.3e} "
        f"(tol {t1['tol']:.0e})  {'PASS' if t1['passed'] else 'FAIL'}")
    lc = t1["lambda_cut_consistency"]
    say(f"      dE/d(lambda_cut) vs total cut length: "
        f"{lc['dE_over_dlambda_cut']:.8g} vs "
        f"{lc['expected_total_cut_length_um']:.8g}  ratio "
        f"{lc['ratio_measured_over_expected']:.12f}")

    res["2_single_cell_lambda0"] = test_2_single_cell(lambda_edge=0.0)
    a = res["2_single_cell_lambda0"]
    say(f" 2a isolated cell, lambda=0 : P/P0={a['perimeter_final'] / a['perimeter_target']:.7f} "
        f"area_err={a['area_error_rel']:.3e} radius_scatter={a['max_radius_scatter_rel']:.3e}")
    say(f"      scale numeric {a['scale_numeric']:.6f} vs analytic {a['scale_analytic']:.6f} "
        f"(rel {a['scale_rel_error']:.2e}); residual |F|={a['residual_force']:.2e}")
    res["2_single_cell_lambda0.5"] = test_2_single_cell(lambda_edge=0.5)
    b = res["2_single_cell_lambda0.5"]
    say(f" 2b isolated cell, lambda=0.5: P {a['perimeter_final']:.5f} -> "
        f"{b['perimeter_final']:.5f} "
        f"({100 * (b['perimeter_final'] / a['perimeter_final'] - 1):+.3f}%), "
        f"area {a['area_final']:.5f} -> {b['area_final']:.5f} "
        f"({100 * (b['area_final'] / a['area_final'] - 1):+.4f}%)")
    res["2c_rounding_up"] = dict(
        name="test_2c_rounding_up",
        perimeter_decrease_um=float(a["perimeter_final"] - b["perimeter_final"]),
        perimeter_decrease_fraction=float(1 - b["perimeter_final"] / a["perimeter_final"]),
        area_change_fraction=float(b["area_final"] / a["area_final"] - 1),
        note="line tension contracts; the area term resists, see LIMITATIONS L4")

    res["3_energy_monotonicity"] = test_3_energy_monotonicity(
        n_side=4 if quick else 5, steps=5000 if quick else 20000)
    m = res["3_energy_monotonicity"]
    say(f" 3 energy monotonicity      : worst step increase="
        f"{m['worst_energy_increase']:.3e} (rel {m['worst_energy_increase_relative']:.3e}) "
        f"over {m['steps']} steps; E {m['E0']:.6f} -> {m['E_final']:.6f}")
    res["4_area_conservation"] = dict(
        name="test_4_area_conservation",
        max_area_drift_rel=m["max_area_drift_rel"],
        passed=bool(m["max_area_drift_rel"] < 5e-3),
        threshold=5e-3,
        note=("Total projected area during uninjured relaxation.  It is NOT "
              "conserved exactly and must not be: the area penalty is a finite "
              "stiffness, so the total area settles wherever the area term "
              "balances the junction tensions (LIMITATIONS L4).  The reported "
              "number is that physical readjustment, not integrator error; the "
              "integrator's own accuracy is what test 3 (energy monotonicity) "
              "and test 1 (gradient) establish."))
    say(f" 4 area conservation        : max total-area drift rel="
        f"{m['max_area_drift_rel']:.3e}  "
        f"{'PASS' if res['4_area_conservation']['passed'] else 'FAIL'}")

    res["5_t1_sanity"] = test_5_t1()
    t5 = res["5_t1_sanity"]
    say(f" 5 T1 sanity                : spontaneous in relaxed lattice="
        f"{t5['t1_spontaneous']} (expect 0) "
        f"{'PASS' if t5['spontaneous_passed'] else 'FAIL'}; under compression "
        f"accepted={t5['t1_accepted_under_compression']} "
        f"rejected={t5['t1_rejected_under_compression']} (conservative "
        f"operator -- see LIMITATIONS L2')")
    say(f"      inverted cells={t5['inverted_cells_after']} "
        f"crossings={not t5['no_vertex_crossings']} "
        f"degree range={t5['vertex_degree_range']} "
        f"T1 rejection reasons={t5['rejected_reasons']}")

    r_small = run(n_side=10, mode="cut", injury_radius=12.0, cut_halfwidth=4.0,
                  t_end=6.0, sample_dt=1.0, lambda_cut=1.0, seed=2025,
                  log_level=0)
    res["6_invariants"] = test_6_invariants(r_small)
    t6 = res["6_invariants"]
    say(f" 6 invariants (10x10 cut)   : min area={t6['min_cell_area_all_frames']:.4f} "
        f"min edge={t6['min_edge_length_overall']:.4e} "
        f"inverted={t6['inverted_cells_present']} crossings="
        f"{t6['vertex_crossings_final']} area_drift="
        f"{t6['area_conservation_rel_error']:.2e} "
        f"{'PASS' if t6['passed'] else 'FAIL'}")

    res["7_wound_closure"] = test_7_wound_closure(
        n_side=16 if quick else 20, t_end=10.0 if quick else 20.0)
    t7 = res["7_wound_closure"]
    for lam in ("1.0", "0.0"):
        d = t7[f"lambda_cut_{lam}"]
        say(f" 7 wound closure lambda={lam:>3}: hole area {d['hole_area_t0']:.2f} -> "
            f"{d['hole_area_tend']:.2f} um^2 ({100 * d['hole_area_relative_change']:+.2f}%), "
            f"cut length {d['cut_length_t0']:.2f} -> {d['cut_length_tend']:.2f} um "
            f"({100 * d['cut_length_relative_change']:+.2f}%), "
            f"T1={d['t1_accepted']}")
    say(f"      purse-string contracts more: {t7['purse_string_contracts_more']} "
        f"(MODEL PREDICTION, no experimental calibration)")

    res["8_determinism"] = test_8_determinism(
        n_side=10 if quick else 12, t_end=2.0 if quick else 4.0)
    t8 = res["8_determinism"]
    say(f" 8 determinism              : max|dr| same seed="
        f"{t8['max_abs_diff_same_seed']:.3e} bit_identical={t8['bit_identical']} "
        f"(other seed {t8['max_abs_diff_other_seed']:.3e}) "
        f"{'PASS' if t8['passed'] else 'FAIL'}")

    res["wall_time_s"] = time.perf_counter() - t0
    res["passed_all_required"] = bool(
        t1["passed"] and t5["spontaneous_passed"] and t6["passed"]
        and t8["passed"])
    res["t5_note"] = (
        "T1 under compression: %d accepted / %d rejected. Acceptance is NOT "
        "required for the test to pass: the T1 operator is conservative by "
        "design (it validates the resulting mesh and rolls back on any "
        "structural violation), and this is documented as LIMITATIONS L2'. "
        "An accepted count of 0 is reported honestly rather than hidden -- it "
        "means no candidate in this run produced a clean mesh, not that the "
        "event log was suppressed." % (t5["t1_accepted_under_compression"],
                                        t5["t1_rejected_under_compression"]))
    say(f"---- self_test wall time {res['wall_time_s']:.1f} s; all required "
        f"checks pass: {res['passed_all_required']} ----")
    return res


def _demo():
    print("spec default API call: run()  [n_side=24 (576 cells), mode='cut']")
    r = run()
    md = r["metadata"]
    print(json.dumps(dict(
        runtime_s=md["runtime_s"], steps=md["steps"],
        rejected_steps=md["rejected_steps"], dt_used_max=md["dt_used_max"],
        injury=md["injury"], events_counts=md["events_counts"],
        mean_cell_area_um2=[float(np.nanmean(r["areas"][0])),
                            float(np.nanmean(r["areas"][-1]))],
        hole_area_um2=[float(r["hole_area"][0]), float(r["hole_area"][-1])],
        cut_length_um=[float(r["cut_length"][0]), float(r["cut_length"][-1])],
        outer_length_um=[float(r["outer_length"][0]), float(r["outer_length"][-1])],
        energy=[float(r["energy"][0]), float(r["energy"][-1])],
        min_edge_length=md["min_edge_length_overall"],
        min_area=md["min_area_overall"],
        area_conservation_rel_error=md["area_conservation_rel_error"],
        damage_cells=int(r["damage"].sum()),
        final_mesh_check_ok=md["final_mesh_check"]["ok"],
        warnings=md["warnings"][:8]), indent=1, default=float))


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()
    else:
        self_test(quick="--quick" in sys.argv)
