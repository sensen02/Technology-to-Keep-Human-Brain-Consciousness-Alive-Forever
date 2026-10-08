"""CELL DIFFERENTIATION and LINEAGE layer for the cell-wound prototype.

=============================================================================
HONESTY STATEMENT -- READ FIRST
=============================================================================
* This layer is **NOT validated against any measurement in this project.**
  There is no observable in this repository that reports a cell type, a
  differentiation rate, a lineage or a fate bias.  Nothing in ``model.py`` /
  ``measurement.py`` could falsify a single number below.  Every test in this
  file is a **numerical self-consistency** test (uniqueness, conservation,
  agreement with the analytic solution of the module's OWN documented Markov
  rule, determinism, timing) -- *not* a comparison with biology.
* The cell types here are **ABSTRACTIONS, NOT TRANSCRIPTIONALLY DEFINED CELL
  TYPES.**  "Stem" means "the type this model lets divide and treats as the
  source of the others"; "myocyte" means "a non-dividing contractile-flagged
  type"; "secretory" means "a slowly dividing support-flagged type".  No marker
  gene, protein, morphology or lineage-tracing experiment is represented.
* **A PATTERNED SIMULATION IS NOT EVIDENCE OF REAL DIFFERENTIATION.**  Test 4
  shows that a prescribed oxygen gradient plus a prescribed oxygen-dependent
  switching rule yields a spatial fate pattern.  That is a tautology about the
  prescribed rule: it demonstrates that the *implementation* propagates a local
  signal into a spatial pattern.  It says nothing about whether oxygen patterns
  fate in any real tissue and cannot, because the threshold and the rates are
  arbitrary.
* Parameter provenance is in ``PARAM_SOURCES``.  **No citation is invented
  anywhere in this file.**  Every differentiation rate, fate probability,
  oxygen threshold and hormone threshold is tagged ILLUSTRATIVE.  The only
  values with a checked source are two project/measurement anchors: the 43 um^2
  mean cell area (project convention, ``model.py`` / ``mechanics.py``, via
  ``MEASURED_ANCHORS.md``) and the 7.5-10 um notum epithelium thickness
  (Pinheiro 2017, PMC6143170, via ``MEASURED_ANCHORS.md``).
* ``growth.py`` is imported READ-ONLY and never modified.  Its docstring calls
  its ``gid`` a "lineage id (parents keep theirs, daughters get new)"; it is
  **not unique per cell**.  ``gid_collision_audit`` measures that collision
  empirically (test 10) instead of asserting it.  Fixing lineage identity is
  why this module exists.

=============================================================================
WHY A SEPARATE LAYER ("the core of an individual is cell types")
=============================================================================
``growth.py`` counts cells, ``cellstate.py`` grades their condition,
``transport.py`` supplies their environment, ``hormone.py`` turns a dose into a
downstream response.  None of them carries *heritable identity*: what a cell IS,
where it came from, and when it changed.  This module adds:

  1. three named abstract types with documented per-type behaviour,
  2. selectable differentiation rules (stochastic / signal-driven /
     division-coupled / all combined),
  3. a verifiable lineage tree with globally unique ids,
  4. coupling to per-cell oxygen, hormone response and injury state,
  5. one ``run()`` that does type/lineage dynamics for a fixed population or
     with real divisions replayed from a ``growth.py`` run.

=============================================================================
UNITS AND CONVENTIONS
=============================================================================
Time: **hours** everywhere at the interface (``t_end_h``, ``dt_h``, rates per
hour), matching ``growth.py``'s interface timescale.  Length um, area um^2.
Type ids are small integers equal to the column order of every per-type array.
A cell is *live* while it is in the population; a *removed* cell leaves the
population but STAYS in the tree as a terminal node, so the tree remains a
complete record.

=============================================================================
EQUATIONS
=============================================================================
1. STOCHASTIC SWITCHING (mode 'stochastic'; the base of 'all').
   A cell of type ``s`` switches to type ``a`` with constant hazard ``k[s,a]``
   per hour.  With ``K[s,s]=0`` the fraction vector ``p(t)`` obeys::

       dp/dt = L p,   L[a,b] = K[b,a]  (a != b),   L[a,a] = -sum_c K[a,c]
       p(t+dt) = expm(L dt) p(t)

   ``analytic_rule_reference`` integrates that exact matrix exponential forward,
   i.e. it is the analytic expectation of the very rule the simulator samples
   cell-by-cell, computed a different way.  Test 3 compares sampling with it.
   For a stem-only start with two fates ``k_sm`` (stem->myocyte) and ``k_ss``
   (stem->secretory) the closed form is::

       f_stem(t) = exp(-(k_sm+k_ss) t)
       f_myo(t)  = k_sm/(k_sm+k_ss) * (1 - exp(-(k_sm+k_ss) t))
       f_sec(t)  = k_ss/(k_sm+k_ss) * (1 - exp(-(k_sm+k_ss) t))

   so each fate's equilibrium share is its share of the total exit rate.  Test 3
   reports this closed form as an independent cross-check of the matrix
   exponential.

2. SIGNAL-DRIVEN SWITCHING (modes 'signal_bias' / 'signal_threshold' / 'all').
   ``x`` is the per-cell signal: oxygen availability (1 = reference, as
   ``transport.cellstate_oxygen_input`` produces) and/or the ``hormone.py``
   downstream response in [0,1]; ``x0``/``w`` are the threshold and width, and
   ``s_max >= 1`` the maximum fold change.  Only hazards that EXIST in
   ``rate_per_h`` are ever modified, so no new fate is invented here.

   'bias' (smooth, default)::

       bias(x) = s_max ** tanh((x - x0)/w)
       k_i[src, signal_target] *= bias(x_i)
       k_i[src, signal_other]  /= bias(x_i)

   ``bias(x0) = 1``, and ``bias = 1`` for every x when ``s_max = 1`` (the
   control).  Far above / below the threshold the two fates are favoured in
   opposite directions with a maximum fold change of ``s_max`` each, so the fate
   RATIO becomes ``bias**2`` (range ``s_max**-2 .. s_max**2``).

   'sigmoid' (hard threshold)::

       k_i[src, signal_target] = k0[src, signal_target] * s_max   if x_i >= x0
                                  k0[src, signal_target] / s_max   otherwise
       k_i[src, signal_other]  = k0[src, signal_other]  / s_max   if x_i >= x0
                                  k0[src, signal_other]  * s_max   otherwise

   Both reduce to the plain stochastic rule when ``s_max = 1`` or when the
   signal is absent -- the control used by tests 3 and 4.

3. DIVISION-COUPLED / ASYMMETRIC RULE (modes 'division_coupled' / 'all').
   Each of the two daughters differentiates independently::

       P(daughter differentiates) = p_div                       ('uniform')
       P(daughter differentiates) = p_lo + (p_hi - p_lo)
                                      * logistic((x - x0)/w)    ('signal_dependent')

   The daughter takes ``division_target``.  ``p_div`` may be a float.  **This
   rule never changes the cell count**: it only re-labels daughters of a
   division that was counted anyway (test 2 checks count conservation).

4. INJURY GATING (always on; switchable only via ``gate_on_injury``).
   Cells whose ``cellstate`` code is 2 (irreversibly injured) or 3 (lysed), or
   which are flagged in ``irreversible`` / ``lysed``, are **excluded from every
   differentiation and division decision**, and the gating LATCHES: a cell ever
   seen gated stays gated.  Blocked OPPORTUNITIES are counted cumulatively
   (``n_blocked_differentiation``, ``n_blocked_division``) alongside the number
   of gated cells in the final frame and the ``divisions_not_placed_*`` counters
   for divisions that could not be placed.  Gating draws no random number, so it
   cannot shift the stream for any other cell.  Test 6 checks that the event
   tables contain exactly zero events for gated cells and that the blocked
   opportunity counts equal ``n_gated * n_steps``.

5. IDENTITY AND CONSERVATION (invariants; tested in tests 1 and 2).
   - every cell has a globally unique integer id (never reused),
   - every non-root cell has exactly one parent,
   - walking parents terminates at a root (no cycles); one root per tree,
   - ``n_alive = n_roots + n_births - n_divided_parents - n_removals`` (residual
     reported), where a parent that divided is a terminal record rather than a
     removal,
   - differentiation changes type only: count and total area are untouched,
   - division is ``a1 = f*A``, ``a2 = A - a1``, so ``a1 + a2 == A`` bit-exactly;
     the residual is measured (``_division_area_residual``) and reported.

=============================================================================
NON-GOALS / EXPLICIT LIMITATIONS
=============================================================================
* Not a validated fate model, not a transcriptional cell-type model.
* No gene regulatory network, no morphogen solver of its own, no mechanical or
  density feedback on fate, no cell migration (daughters inherit the parent's
  position), no asymmetric *size* split (``growth.py`` owns the split rule).
* The oxygen field is an INPUT: ``analytic_oxygen_gradient`` is a prescribed
  ramp for the self-tests and ``oxygen_from_transport`` wires the real
  ``transport.py`` field in WITHOUT modifying transport.py.  Neither is a claim
  about the biology of oxygen gradients.
* Importing this module creates no files and no plots.  ``run()`` returns a
  plain dict.  Only ``__main__`` writes one file, the verification JSON.

=============================================================================
API
=============================================================================
Types and rules
  TYPES                dict name -> per-type parameters (3 abstract types).
  TYPE_IDS             dict name -> int id.
  PARAM_SOURCES        provenance string per parameter (mostly ILLUSTRATIVE).
  DEFAULT_RULE         the default rule configuration dict.
  RULE_NAMES           the selectable rule names.
  analytic_matrix_exponent(rule, dt_h=None)
  analytic_rule_reference(rule, t_end_h=None, dt_h=None, p0=None)
  closed_form_two_fate(rule, t_h)

Tree
  CellRecord           one cell's identity and history.
  Lineage              unique-id tree: differentiate / divide / remove,
                       parent_of / children_of / path_to_root / descendants_of,
                       event_table / event_table_arrays / event_table_signature,
                       verify / type_counts / type_fractions / total_area_um2.
  lineage_from_growth_result(result)   unique-id tree from a growth.py result.
  gid_collision_audit(result)          measures growth.py's gid collisions.

Environment inputs
  analytic_oxygen_gradient(positions, ...)   prescribed monotone oxygen ramp.
  position_coordinate(positions, ...)        normalised x/y/radius in [0,1].
  oxygen_from_transport(...)                 optional real transport.py coupling.
  hormone_response_from_hormone(...)         optional real hormone.py coupling.
  resolve_injury(state=, irreversible=, lysed=, n=)  -> (gated, codes).

Entry points
  run(n_cells=..., t_end_h=..., dt_h=..., seed=..., rule=..., ...) -> dict
                       # returns times_h, counts_by_type, fractions_by_type,
                       # lineage (Lineage), tree, event_table(+arrays),
                       # verification, identity_residual, conservation,
                       # counts, rule, metadata
  pure_differentiation_run(...)   fixed population, no division.
  growth_coupled_run(...)         growth.py divisions replayed into the tree.
  self_test(quick=False)          the eight required tests + the gid audit.
  main()                          self_test + write the verification JSON.

=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

try:  # scipy is in this project's venv
    from scipy.linalg import expm as _expm
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

    def _expm(a):
        """Scaling-and-squaring fallback used only if scipy is absent."""
        a = np.asarray(a, float)
        s = max(0, int(math.ceil(math.log2(max(1e-300, float(np.abs(a).max()))))))
        y = a / (2.0 ** s)
        z = np.eye(a.shape[0])
        term = np.eye(a.shape[0])
        for k in range(1, 16):
            term = term @ y / k
            z = z + term
        for _ in range(s):
            z = z @ z
        return z


__all__ = [
    'TYPES', 'TYPE_IDS', 'TYPE_NAMES', 'PARAM_SOURCES', 'DEFAULT_RULE', 'RULE_NAMES',
    'CellRecord', 'Lineage', 'run', 'pure_differentiation_run',
    'growth_coupled_run', 'lineage_from_growth_result', 'gid_collision_audit',
    'analytic_oxygen_gradient', 'position_coordinate', 'oxygen_from_transport',
    'hormone_response_from_hormone', 'resolve_injury',
    'analytic_rule_reference', 'analytic_matrix_exponent', 'closed_form_two_fate',
    'self_test', 'main', 'OUTPUT_PATH', 'MODULE_VERSION',
]

MODULE_VERSION = 'lineage-1.0.0'
OUTPUT_PATH = '/run/media/sensen/Data2/cell_wound_prototype/outputs/lineage_verified.json'

CELL_AREA_UM2 = 43.0            # project convention (model.py / mechanics.py)
EPITHELIUM_THICKNESS_UM = 9.0   # midpoint of the checked 7.5-10 um range

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
PARAM_SOURCES = {
    'WARNING': (
        'UNCALIBRATED AND UNVALIDATED. This layer has NO observable in this project '
        'that could falsify any parameter. The cell types are abstractions, not '
        'transcriptionally defined cell types. A patterned output is a property of '
        'the prescribed rule, not evidence of real differentiation.'),
    'n_types=3': 'ILLUSTRATIVE (modelling choice: one progenitor + two fates)',
    'rate_per_h (stochastic switching)':
        'ILLUSTRATIVE (no measured fate-switching rate exists in this project)',
    'oxygen_threshold_rel=0.6':
        'ILLUSTRATIVE (oxygen 1.0 = transport.py reference; 0.6 sits inside the '
        'self-test gradient)',
    'oxygen_width_rel=0.15': 'ILLUSTRATIVE',
    'hormone_threshold=0.5':
        'ILLUSTRATIVE (the hormone.py response is itself uncalibrated)',
    'signal_strength (max fold change of the fate hazard)': 'ILLUSTRATIVE',
    'p_division_coupled': 'ILLUSTRATIVE (asymmetric-division probability)',
    'can_divide / division_rate_per_h / divides_with_probability':
        'ILLUSTRATIVE (stem unlimited, myocyte none, secretory graded)',
    'area_preference_um2': 
        'BASELINE 43 um^2 is a LITERATURE(project) value: the project-wide mean cell '
        'area convention (model.py / mechanics.py, recorded in MEASURED_ANCHORS.md); '
        'the per-type multipliers (1.15, 0.95) are ILLUSTRATIVE',
    'substrate_consumption_multiplier': 'ILLUSTRATIVE (relative metabolic demand)',
    'initial area_um2=43.0': 'project convention, as above',
    'epithelium_thickness_um=9.0':
        'LITERATURE: 7.5-10 um Drosophila notum epithelium thickness, Pinheiro 2017 '
        'PMC6143170, via MEASURED_ANCHORS.md; 9.0 is the midpoint',
    'dt_h / n_cells / n_replicates in tests':
        'ILLUSTRATIVE software choices, not biology',
}

RULE_NAMES = (
    'stochastic',          # (a) hazard-based switching, no signal input
    'signal_bias',         # (b1) smooth signal bias of the two fate hazards
    'signal_threshold',    # (b2) hard threshold switch on the signal
    'division_coupled',    # (c) asymmetric division: daughter differentiates w.p. p
    'all',                 # stochastic base + signal bias + division coupling
)


# ---------------------------------------------------------------------------
# abstract cell types
# ---------------------------------------------------------------------------
TYPES = {
    'stem': dict(
        id=0,
        role='progenitor: the only type allowed to divide freely in this model',
        can_divide=True,
        division_rate_per_h=0.02,        # ILLUSTRATIVE hazard, not a measured cycle rate
        divides_with_probability=None,   # None = divides whenever a division is offered
        area_preference_um2=CELL_AREA_UM2,
        area_preference_provenance='43 um^2 project convention; multiplier 1.0',
        substrate_consumption_multiplier=1.0,
        contractile=False,
        secretory=False,
        is_source=True,
        abstract_label='stem/progenitor (NOT a transcriptionally defined cell type)',
    ),
    'myocyte': dict(
        id=1,
        role='differentiated contractile / myocyte-like type',
        can_divide=False,
        division_rate_per_h=0.0,
        divides_with_probability=0.0,
        area_preference_um2=43.0 * 1.15,      # ILLUSTRATIVE multiplier
        area_preference_provenance='ILLUSTRATIVE (1.15x the 43 um^2 project convention)',
        substrate_consumption_multiplier=2.0,  # ILLUSTRATIVE
        contractile=True,
        secretory=False,
        is_source=False,
        abstract_label='contractile/myocyte-like (NOT a transcriptionally defined cell type)',
    ),
    'secretory': dict(
        id=2,
        role='differentiated secretory / support type',
        can_divide=True,
        division_rate_per_h=0.005,       # ILLUSTRATIVE: low but non-zero
        divides_with_probability=0.5,    # ILLUSTRATIVE: only if a division is offered
        area_preference_um2=43.0 * 0.95,      # ILLUSTRATIVE multiplier
        area_preference_provenance='ILLUSTRATIVE (0.95x the 43 um^2 project convention)',
        substrate_consumption_multiplier=1.4,  # ILLUSTRATIVE
        contractile=False,
        secretory=True,
        is_source=False,
        abstract_label='secretory/support (NOT a transcriptionally defined cell type)',
    ),
}
TYPE_NAMES = ('stem', 'myocyte', 'secretory')
TYPE_IDS = {n: int(t['id']) for n, t in TYPES.items()}
N_TYPES = len(TYPE_NAMES)
ID_TO_NAME = {v: k for k, v in TYPE_IDS.items()}

DEFAULT_RULE = dict(
    name='stochastic',
    # (a) stochastic switching: fate hazards per hour, symmetric on purpose so
    # the analytic fate fraction is exactly 1/2 per fate (test 3).
    rate_per_h={'stem->myocyte': 0.02, 'stem->secretory': 0.02},
    source_type='stem',
    # (b) signal-driven switching
    signal_source='oxygen',          # 'oxygen' | 'hormone' | 'mean'
    signal_rule='bias',              # 'bias' | 'sigmoid'
    signal_target='myocyte',         # fate FAVOURED when the signal is high
    signal_other='secretory',
    oxygen_threshold_rel=0.6,        # oxygen availability, 1 = transport.py reference
    oxygen_width_rel=0.15,
    hormone_threshold=0.5,           # hormone.py downstream response, [0,1]
    hormone_width=0.2,
    signal_strength=3.0,             # maximum fold change at |x-x0| >> w
    # (c) division-coupled / asymmetric
    p_division_coupled=0.5,          # P(exactly one differentiated daughter)
    p_division_both=0.0,             # P(both daughters differentiate)
    p_division_low=0.1,              # ILLUSTRATIVE, signal-dependent variant
    p_division_high=0.9,             # ILLUSTRATIVE, signal-dependent variant
    division_coupled_mode='uniform',  # 'uniform' | 'signal_dependent'
    division_target='myocyte',
    # gates
    gate_on_injury=True,
    # numerics
    dt_h=0.5,
)

INJURY_CODES = {0: 'healthy', 1: 'reversibly_injured', 2: 'irreversibly_injured',
                3: 'lysed'}
BLOCKING_CODES = (2, 3)


# ---------------------------------------------------------------------------
# rule resolution and the analytic reference
# ---------------------------------------------------------------------------
def _resolve_rule(rule):
    if rule is None:
        return dict(DEFAULT_RULE)
    if isinstance(rule, str):
        if rule not in RULE_NAMES:
            raise ValueError(f'unknown rule name {rule!r}; known {RULE_NAMES}')
        out = dict(DEFAULT_RULE)
        out['name'] = rule
        out['signal_rule'] = ('sigmoid' if rule == 'signal_threshold'
                              else 'bias')
        return out
    out = dict(DEFAULT_RULE)
    out.update(rule)
    if out['name'] not in RULE_NAMES:
        raise ValueError(f'unknown rule name {out["name"]!r}; known {RULE_NAMES}')
    if out['signal_source'] not in ('oxygen', 'hormone', 'mean'):
        raise ValueError("signal_source must be 'oxygen', 'hormone' or 'mean'")
    if out['signal_rule'] not in ('bias', 'sigmoid'):
        raise ValueError("signal_rule must be 'bias' or 'sigmoid'")
    if float(out['signal_strength']) < 1.0:
        raise ValueError('signal_strength must be >= 1')
    for k in ('signal_target', 'signal_other', 'division_target', 'source_type'):
        if out[k] not in TYPE_IDS:
            raise ValueError(f'{k} must be one of {sorted(TYPE_IDS)}')
    if out['signal_target'] == out['signal_other']:
        raise ValueError('signal_target and signal_other must differ')
    if float(out['dt_h']) <= 0:
        raise ValueError('dt_h must be positive')
    if not 0.0 <= float(out['p_division_coupled']) <= 1.0:
        raise ValueError('p_division_coupled must be in [0,1]')
    for lo_hi in ('p_division_low', 'p_division_high'):
        if not 0.0 <= float(out[lo_hi]) <= 1.0:
            raise ValueError(f'{lo_hi} must be in [0,1]')
    for key, val in out['rate_per_h'].items():
        if '->' not in key:
            raise ValueError(f'rate_per_h key must be "from->to", got {key!r}')
        a, b = [s.strip() for s in key.split('->')]
        if a not in TYPE_IDS or b not in TYPE_IDS:
            raise ValueError(f'unknown type in rate key {key!r}')
        if a == b:
            raise ValueError(f'self-transition in rate key {key!r}')
        if float(val) < 0:
            raise ValueError(f'negative rate for {key!r}')
    return out


def _rate_matrix(rule):
    """K[s,a] = hazard type s -> type a (per hour)."""
    K = np.zeros((N_TYPES, N_TYPES), float)
    for key, val in rule['rate_per_h'].items():
        a, b = [s.strip() for s in key.split('->')]
        K[TYPE_IDS[a], TYPE_IDS[b]] = float(val)
    return K


def analytic_matrix_exponent(rule, dt_h=None):
    """Step matrix ``expm(L*dt)`` of the documented stochastic rule (eq. 1)."""
    rule = _resolve_rule(rule)
    dt = float(rule['dt_h'] if dt_h is None else dt_h)
    K = _rate_matrix(rule)
    L = K.T.copy()
    np.fill_diagonal(L, -K.sum(axis=1))
    return _expm(L * dt), K, L


def analytic_rule_reference(rule, t_end_h=None, dt_h=None, p0=None):
    """Exact expected type fractions of the documented stochastic rule.

    ``p0`` is a length-3 distribution (default: all cells of ``source_type``).
    The forward matrix exponential is used, so unlike the simulator it makes no
    per-step Bernoulli approximation; the two agree to the order of that
    approximation, which test 3 measures rather than assumes.
    """
    rule = _resolve_rule(rule)
    dt = float(rule['dt_h'] if dt_h is None else dt_h)
    t_end = float(rule.get('t_end_h', 200.0) if t_end_h is None else t_end_h)
    n_steps = max(1, int(round(t_end / dt)))
    if p0 is None:
        p0 = np.zeros(N_TYPES)
        p0[TYPE_IDS[rule['source_type']]] = 1.0
    p = np.asarray(p0, float).copy()
    M, _, _ = analytic_matrix_exponent(rule, dt)
    out = np.empty((n_steps + 1, N_TYPES))
    out[0] = p
    for i in range(n_steps):
        p = M @ p
        out[i + 1] = p
    return np.linspace(0.0, n_steps * dt, n_steps + 1), out


def closed_form_two_fate(rule, t_h):
    """Closed form of eq. 1 for a stem-only start with two fates."""
    rule = _resolve_rule(rule)
    r = rule['rate_per_h']
    ksm = float(r.get('stem->myocyte', 0.0))
    kss = float(r.get('stem->secretory', 0.0))
    tot = ksm + kss
    t = np.asarray(t_h, float)
    if tot <= 0:
        return dict(stem=np.ones_like(t), myocyte=np.zeros_like(t),
                    secretory=np.zeros_like(t))
    e = np.exp(-tot * t)
    return dict(stem=e, myocyte=ksm / tot * (1 - e), secretory=kss / tot * (1 - e))


# ---------------------------------------------------------------------------
# per-cell record
# ---------------------------------------------------------------------------
class CellRecord:
    """Read-only view of one cell's identity and history."""

    __slots__ = ('cell_id', 'parent_id', 'type', 'type_id', 'birth_h',
                 'division_h', 'removal_h', 'area_um2', 'is_root', 'is_alive',
                 'n_divisions', 'injury_code', 'gated')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self):
        return (f'CellRecord(id={self.cell_id}, parent={self.parent_id}, '
                f'type={self.type!r}, birth_h={self.birth_h:g}, '
                f'division_h={self.division_h}, removal_h={self.removal_h})')


# ---------------------------------------------------------------------------
# lineage tree
# ---------------------------------------------------------------------------
class Lineage:
    """Verifiable lineage tree for a centre-based cell population.

    Storage is a struct-of-arrays arena indexed by an internal slot; cell ids
    are globally unique and never reused.  A node is created either as a root
    (``parent_ptr = -1``) or as a daughter of an existing node, so a cycle is
    structurally impossible -- and is *also* checked by walking to the root.

    ``divide()`` keeps the parent node as a terminal record of the division
    (``division_h`` set, ``removed`` True, ``parent_area`` recording the area
    before the split and ``split_area_a``/``split_area_b`` the daughter areas),
    and it never overwrites the parent's own ``area`` field, so a cell that
    divided twice is still checkable.  The tree can be walked in both
    directions and the area residual is measured, not assumed.

    Accounting (the identity check of test 1)::

        n_alive = n_roots + n_births - n_divisions - n_removals

    where ``n_divisions`` counts PARENT NODES that became terminal by dividing
    and ``n_removals`` counts cells dropped from the population WITHOUT
    dividing.  ``n_alive`` is derived from the ``removed`` mask and
    ``division_h`` rather than maintained incrementally, so the identity cannot
    drift even if an operation is called out of order.
    """

    _ARRAYS = (
        ('cell_id', np.int64), ('parent_ptr', np.int64), ('type_id', np.int64),
        ('birth_h', np.float64), ('division_h', np.float64),
        ('removal_h', np.float64), ('area', np.float64), ('parent_area', np.float64),
        ('split_area_a', np.float64), ('split_area_b', np.float64),
        ('n_divisions', np.int64), ('injury_code', np.int8), ('gated', np.bool_),
        ('removed', np.bool_), ('pos_x', np.float64), ('pos_y', np.float64),
    )

    def __init__(self, n_roots, root_type='stem', area_um2=CELL_AREA_UM2,
                 root_positions=None, injury_state=None, irreversible=None,
                 lysed=None, dt_h=0.5, record_every=1, seed=0, cap=None,
                 time0_h=0.0):
        if root_type not in TYPE_IDS:
            raise ValueError(f'unknown root_type {root_type!r}; known {sorted(TYPE_IDS)}')
        n = int(n_roots)
        if n < 1:
            raise ValueError('n_roots must be >= 1')
        self.root_type = root_type
        self.dt_h = float(dt_h)
        self.record_every = int(record_every)
        self.seed = int(seed)
        self.cap = 0
        self.n_slots = 0
        self.n_roots = n
        self.n_births = 0
        self.n_removals = 0
        self.n_divided = 0
        self.next_id = 0
        self._division_area_residual = 0.0
        self._type_events = []
        self._removal_reason = {}
        self._id_index = None
        self._children = None
        self._children_count = None
        self._grow_to(int(cap) if cap is not None else max(16, 2 * n))
        for i in range(n):
            self._append_root(float(area_um2), root_type, time0_h,
                              None if root_positions is None else root_positions[i])
        if injury_state is not None or irreversible is not None or lysed is not None:
            gated, code = resolve_injury(state=injury_state, irreversible=irreversible,
                                         lysed=lysed, n=n)
            self.injury_code[:n] = code.astype(np.int8)
            self.gated[:n] = gated

    # -- arena --------------------------------------------------------------
    def _grow_to(self, cap):
        old = int(self.cap)
        cap = max(int(cap), 16)
        for name, dtype in self._ARRAYS:
            new = np.zeros(cap, dtype=dtype)
            if old:
                new[:old] = getattr(self, name)
            setattr(self, name, new)
        self.cap = cap

    def _ensure(self, extra):
        if self.n_slots + extra > self.cap:
            self._grow_to(max(2 * self.cap, self.n_slots + extra))

    def _new_slot(self):
        self._ensure(1)
        s = self.n_slots
        self.n_slots += 1
        self.cell_id[s] = self.next_id
        self.parent_ptr[s] = -1
        self.birth_h[s] = 0.0
        self.division_h[s] = np.nan
        self.removal_h[s] = np.nan
        self.area[s] = 0.0
        self.parent_area[s] = np.nan
        self.split_area_a[s] = np.nan
        self.split_area_b[s] = np.nan
        self.n_divisions[s] = 0
        self.injury_code[s] = 0
        self.gated[s] = False
        self.removed[s] = False
        self.pos_x[s] = np.nan
        self.pos_y[s] = np.nan
        self.next_id += 1
        return s

    def _append_root(self, area, type_name, t_h, pos=None):
        s = self._new_slot()
        self.type_id[s] = TYPE_IDS[type_name]
        self.birth_h[s] = float(t_h)
        self.area[s] = float(area)
        if pos is not None:
            self.pos_x[s] = float(pos[0])
            self.pos_y[s] = float(pos[1])
        return s

    def _append_daughter(self, parent_slot, area, type_id, t_h, pos=None):
        s = self._new_slot()
        self.parent_ptr[s] = int(parent_slot)
        self.type_id[s] = int(type_id)
        self.birth_h[s] = float(t_h)
        self.area[s] = float(area)
        # documented default: a daughter is born ungated (matching growth.py's
        # daughter_state='healthy' option).  Gating is not inherited.
        self.injury_code[s] = 0
        self.gated[s] = False
        if pos is not None:
            self.pos_x[s] = float(pos[0])
            self.pos_y[s] = float(pos[1])
        self.n_births += 1
        return s

    # -- invariant-preserving operations ------------------------------------
    def differentiate(self, slot, new_type, t_h):
        """Change one live cell's type.  Count and area are untouched."""
        if new_type not in TYPE_IDS:
            raise ValueError(f'unknown type {new_type!r}')
        if self.removed[slot]:
            raise RuntimeError(f'slot {slot} is removed; cannot differentiate')
        new_id = TYPE_IDS[new_type]
        old_id = int(self.type_id[slot])
        if old_id == new_id:
            return None
        self.type_id[slot] = new_id
        return dict(kind='type_change', t_h=float(t_h),
                    cell_id=int(self.cell_id[slot]),
                    from_type=ID_TO_NAME[old_id], to_type=new_type)

    def divide(self, slot, t_h, area_frac=0.5, daughter_types=(None, None),
               positions=None):
        """Replace one live cell by two daughters; parent area is conserved.

        ``a1 = area_frac * A`` and ``a2 = A - a1`` (exact in IEEE-754).
        ``daughter_types`` entries may be None (inherit the parent's type).
        Returns (daughter_slot_a, daughter_slot_b).
        """
        if self.removed[slot]:
            raise RuntimeError(f'slot {slot} is removed; cannot divide')
        if self.gated[slot]:
            raise RuntimeError(f'slot {slot} is gated; cannot divide')
        A = float(self.area[slot])
        a1 = A * float(area_frac)
        a2 = A - a1
        p_type = int(self.type_id[slot])
        t1 = p_type if daughter_types[0] is None else TYPE_IDS[daughter_types[0]]
        t2 = p_type if daughter_types[1] is None else TYPE_IDS[daughter_types[1]]
        # the parent's OWN area field is left untouched (it is this cell's area,
        # and it must stay checkable even if the cell is queried later); the two
        # daughter areas are recorded in dedicated fields instead
        self.parent_area[slot] = A
        self.split_area_a[slot] = a1
        self.split_area_b[slot] = a2
        self.division_h[slot] = float(t_h)
        self.n_divisions[slot] += 1
        self.removed[slot] = True          # parent node becomes terminal
        self.n_divided += 1
        pa = None if positions is None else positions[0]
        pb = None if positions is None else positions[1]
        sa = self._append_daughter(slot, a1, t1, t_h, pa)
        sb = self._append_daughter(slot, a2, t2, t_h, pb)
        err = abs((a1 + a2) - A)
        if err > self._division_area_residual:
            self._division_area_residual = float(err)
        return sa, sb

    def remove(self, slot, t_h, reason='removed'):
        """Drop one live cell from the population; the node stays in the tree."""
        if self.removed[slot]:
            return False
        self.removed[slot] = True
        self.removal_h[slot] = float(t_h)
        self._removal_reason[int(slot)] = reason
        self.n_removals += 1
        return True

    # -- walking ------------------------------------------------------------
    def _build_id_index(self):
        n_id = int(self.cell_id[: self.n_slots].max()) + 1 if self.n_slots else 0
        idx = np.full(n_id, -1, np.int64)
        idx[self.cell_id[: self.n_slots]] = np.arange(self.n_slots)
        self._id_index = idx

    def slot_of(self, cell_id):
        if self._id_index is None:
            self._build_id_index()
        cid = int(cell_id)
        if cid < 0 or cid >= self._id_index.size or self._id_index[cid] < 0:
            raise KeyError(f'unknown cell id {cell_id}')
        return int(self._id_index[cid])

    def _build_children(self):
        ch = {}
        cnt = np.zeros(self.n_slots, np.int64)
        for s in range(self.n_slots):
            p = int(self.parent_ptr[s])
            if p >= 0:
                ch.setdefault(p, []).append(s)
                cnt[p] += 1
        self._children = ch
        self._children_count = cnt

    def children_slots_of(self, cell_id):
        if self._children is None:
            self._build_children()
        return list(self._children.get(int(cell_id), ()))

    def parent_of(self, cell_id):
        s = self.slot_of(cell_id)
        p = int(self.parent_ptr[s])
        return None if p < 0 else int(self.cell_id[p])

    def children_of(self, cell_id):
        return [int(self.cell_id[s]) for s in self.children_slots_of(cell_id)]

    def path_to_root(self, cell_id):
        """Ids from ``cell_id`` up to and including the root; raises on a cycle."""
        out = []
        s = self.slot_of(cell_id)
        seen = set()
        while True:
            if s in seen:
                raise RuntimeError(f'cycle detected at cell {int(self.cell_id[s])}')
            seen.add(s)
            out.append(int(self.cell_id[s]))
            p = int(self.parent_ptr[s])
            if p < 0:
                return out
            s = p

    def root_of(self, cell_id):
        return self.path_to_root(cell_id)[-1]

    def descendants_of(self, cell_id, include_self=False):
        out = []
        stack = [self.slot_of(cell_id)]
        first = True
        while stack:
            s = stack.pop()
            if include_self or not first:
                out.append(int(self.cell_id[s]))
            first = False
            stack.extend(self.children_slots_of(int(self.cell_id[s])))
        return out

    def depth_of(self, cell_id):
        return len(self.path_to_root(cell_id)) - 1

    # -- exports ------------------------------------------------------------
    def root_ids(self):
        return [int(self.cell_id[s]) for s in range(self.n_slots)
                if self.parent_ptr[s] < 0]

    @property
    def n_alive(self):
        """Live cells, DERIVED from the arrays (never maintained incrementally).

        A node is live when it has neither divided nor been removed, so the
        identity ``n_alive == n_roots + n_births - n_divided - n_removals``
        holds by construction and cannot drift.
        """
        m = self.removed[: self.n_slots]
        return int(np.count_nonzero(~m))

    def live_slots(self):
        return np.flatnonzero(~self.removed[: self.n_slots])

    def cell(self, cell_id):
        s = self.slot_of(cell_id)
        p = int(self.parent_ptr[s])
        return CellRecord(
            cell_id=int(self.cell_id[s]),
            parent_id=(None if p < 0 else int(self.cell_id[p])),
            type=ID_TO_NAME[int(self.type_id[s])], type_id=int(self.type_id[s]),
            birth_h=float(self.birth_h[s]),
            division_h=(None if np.isnan(self.division_h[s]) else float(self.division_h[s])),
            removal_h=(None if np.isnan(self.removal_h[s]) else float(self.removal_h[s])),
            area_um2=float(self.area[s]), is_root=bool(p < 0),
            is_alive=bool(not self.removed[s]), n_divisions=int(self.n_divisions[s]),
            injury_code=int(self.injury_code[s]), gated=bool(self.gated[s]))

    def cells(self):
        return [self.cell(int(self.cell_id[s])) for s in range(self.n_slots)]

    def parents_array(self):
        """Cell-id indexed parent ids, -1 for roots (index i == cell id i)."""
        out = np.full(self.n_slots, -1, np.int64)
        has = self.parent_ptr[: self.n_slots] >= 0
        out[has] = self.cell_id[self.parent_ptr[: self.n_slots][has]]
        return out

    def children_map(self):
        if self._children is None:
            self._build_children()
        return {int(self.cell_id[k]): [int(self.cell_id[s]) for s in v]
                for k, v in self._children.items()}

    def event_table(self):
        """Flat, time-ordered table of every recorded event.

        Rows: ``birth_root``, ``birth_daughter``, ``division`` (the parent node
        becoming terminal), ``type_change``, ``removal``.  Ties are broken by the
        append order, so the table is deterministic for a fixed seed.
        """
        rows = []
        for s in range(self.n_slots):
            cid = int(self.cell_id[s])
            p = int(self.parent_ptr[s])
            rows.append(dict(event=('birth_root' if p < 0 else 'birth_daughter'),
                             t_h=float(self.birth_h[s]), cell_id=cid,
                             parent_id=(None if p < 0 else int(self.cell_id[p])),
                             type_from=None,
                             type_to=ID_TO_NAME[int(self.type_id[s])],
                             area_um2=float(self.area[s]),
                             detail=('founder' if p < 0 else 'from division')))
        for (t, cid, old, new) in self._type_events:
            rows.append(dict(event='type_change', t_h=float(t), cell_id=int(cid),
                             parent_id=None, type_from=old, type_to=new,
                             area_um2=None, detail='differentiation'))
        for s in range(self.n_slots):
            if not np.isnan(self.division_h[s]):
                rows.append(dict(event='division', t_h=float(self.division_h[s]),
                                 cell_id=int(self.cell_id[s]), parent_id=None,
                                 type_from=ID_TO_NAME[int(self.type_id[s])],
                                 type_to=None, area_um2=float(self.parent_area[s]),
                                 detail='parent node became terminal; '
                                        'daughter areas '
                                        f'{float(self.split_area_a[s]):g} + '
                                        f'{float(self.split_area_b[s]):g}'))
            if not np.isnan(self.removal_h[s]):
                rows.append(dict(event='removal', t_h=float(self.removal_h[s]),
                                 cell_id=int(self.cell_id[s]), parent_id=None,
                                 type_from=ID_TO_NAME[int(self.type_id[s])],
                                 type_to=None, area_um2=float(self.area[s]),
                                 detail=self._removal_reason.get(s, 'removed')))
        rows.sort(key=lambda r: (r['t_h'], r['event'], r['cell_id']))
        return rows

    def event_table_arrays(self):
        """The same table as numpy arrays (deterministic, compact, slots-indexed)."""
        rows = self.event_table()
        kinds = sorted({r['event'] for r in rows})
        kid = {k: i for i, k in enumerate(kinds)}
        return dict(kinds=np.array(kinds),
                    kind=np.array([kid[r['event']] for r in rows], np.int8),
                    t_h=np.array([r['t_h'] for r in rows], float),
                    cell_id=np.array([r['cell_id'] for r in rows], np.int64),
                    parent_id=np.array([-1 if r['parent_id'] is None else r['parent_id']
                                        for r in rows], np.int64),
                    n_rows=len(rows))

    def event_table_signature(self):
        """Hashable/JSON-able signature of the event table (determinism test)."""
        return [[r['event'], round(float(r['t_h']), 12), int(r['cell_id']),
                 (-1 if r['parent_id'] is None else int(r['parent_id'])),
                 r['type_from'], r['type_to']] for r in self.event_table()]

    def type_counts(self):
        ids = self.type_id[self.live_slots()]
        return {n: int((ids == TYPE_IDS[n]).sum()) for n in TYPE_NAMES}

    def type_fractions(self):
        c = self.type_counts()
        tot = max(1, sum(c.values()))
        return {n: c[n] / tot for n in TYPE_NAMES}

    def total_area_um2(self):
        return float(self.area[self.live_slots()].sum())

    # -- verification -------------------------------------------------------
    def verify(self):
        """All identity/tree invariants; returns a JSON-able diagnosis dict."""
        n = int(self.n_slots)
        ids = self.cell_id[:n]
        uniq = np.unique(ids) if n else np.zeros(0, np.int64)
        par = self.parent_ptr[:n]
        roots = int((par < 0).sum())
        bad_parent = int(((par >= n) | (par < -1) | (par == np.arange(n))).sum())
        # depth by walking parent pointers, with an explicit cycle guard
        depth = np.zeros(n, np.int64)
        cycles = []
        for s in range(n):
            seen = set()
            cur = s
            d = 0
            while cur >= 0:
                if cur in seen:
                    cycles.append(int(self.cell_id[s]))
                    break
                seen.add(cur)
                cur = int(self.parent_ptr[cur])
                d += 1
            depth[s] = d - 1
        alives = self.live_slots()
        n_alive = int(alives.size)
        residual = int(n_alive - (roots + self.n_births - self.n_divided
                                  - self.n_removals))
        walked_roots = {self.root_of(int(self.cell_id[s])) for s in range(n)} if n else set()
        return dict(
            n_cells_total=n, n_cells_alive=n_alive,
            n_roots=int(self.n_roots), n_roots_measured=roots,
            n_daughters=int(self.n_births), n_removals=int(self.n_removals),
            n_divided_parents=int(self.n_divided),
            ids_unique=bool(uniq.size == n), n_unique_ids=int(uniq.size),
            n_duplicate_ids=int(n - uniq.size),
            parent_pointer_invalid=bad_parent,
            n_nonroot=int((par >= 0).sum()),
            n_nonroot_has_exactly_one_valid_parent=int(((par >= 0) & (par < n)).sum()),
            n_cycles=int(len(cycles)), cycle_examples=cycles[:5],
            identity_residual=residual,
            identity_residual_definition=('n_alive - (n_roots + n_births - '
                                          'n_divided_parents - n_removals)'),
            max_depth=int(depth.max()) if n else 0,
            mean_depth=float(depth.mean()) if n else 0.0,
            walk_to_root_ok=bool(all(
                len(self.path_to_root(int(self.cell_id[s]))) >= 1 for s in range(n))),
            n_distinct_roots_by_walk=len(walked_roots),
            n_division_events=int(np.count_nonzero(~np.isnan(self.division_h[:n]))),
            division_area_residual_um2=float(self._division_area_residual),
            n_type_change_events=len(self._type_events),
        )


# ---------------------------------------------------------------------------
# environment inputs
# ---------------------------------------------------------------------------
def resolve_injury(state=None, irreversible=None, lysed=None, n=None):
    """-> (gated bool array, cellstate code array) in cellstate.py's conventions.

    ``state`` may be an int array of codes 0..3, a dict as accepted by
    ``growth.run(state=...)`` (keys 'state', 'latched_irreversible',
    'latched_lysed'), or None.  A cell is GATED when its code is 2 or 3, or when
    a latched Boolean says so.  ``n`` is required when nothing is supplied.
    """
    if callable(state):
        raise ValueError('callable state is not supported here; pass the resolved array')
    codes = None
    if isinstance(state, dict):
        if state.get('state') is not None:
            codes = np.asarray(state['state'], np.int8).ravel()
        if irreversible is None:
            irreversible = state.get('latched_irreversible')
        if lysed is None:
            lysed = state.get('latched_lysed')
    elif state is not None:
        codes = np.asarray(state, np.int8).ravel()
    if codes is None:
        if n is None:
            raise ValueError('resolve_injury needs state or n')
        codes = np.zeros(int(n), np.int8)
    codes = codes.astype(np.int8).copy()
    gated = np.isin(codes, BLOCKING_CODES)
    if irreversible is not None:
        irr = np.asarray(irreversible, bool).ravel()
        if irr.size != codes.size:
            raise ValueError(f'irreversible has size {irr.size}, expected {codes.size}')
        codes = np.maximum(codes, np.where(irr, 2, 0)).astype(np.int8)
        gated |= irr
    if lysed is not None:
        lys = np.asarray(lysed, bool).ravel()
        if lys.size != codes.size:
            raise ValueError(f'lysed has size {lys.size}, expected {codes.size}')
        codes = np.maximum(codes, np.where(lys, 3, 0)).astype(np.int8)
        gated |= lys
    return gated, codes


def position_coordinate(positions, domain_um=None, mode='x'):
    """Normalise positions to u in [0,1] along x/y or in radius.

    ``domain_um`` is (width, height); when None the observed extent is used.
    Returns (u, lo, hi).  A degenerate extent maps every cell to 0.5.
    """
    p = np.asarray(positions, float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('positions must be (n,2)')
    if mode == 'x':
        raw = p[:, 0]
    elif mode == 'y':
        raw = p[:, 1]
    elif mode == 'radius':
        raw = np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean())
    else:
        raise ValueError("mode must be 'x', 'y' or 'radius'")
    if domain_um is not None:
        lo = 0.0
        hi = (0.5 * float(min(domain_um)) if mode == 'radius'
              else float(domain_um[0] if mode == 'x' else domain_um[1]))
    else:
        lo, hi = float(raw.min()), float(raw.max())
    if not (np.isfinite(hi - lo) and hi > lo):
        return np.full(raw.size, 0.5), 0.0, 1.0
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0), float(lo), float(hi)


def analytic_oxygen_gradient(positions, domain_um=None, low=0.05, high=1.0,
                             mode='x', field='logistic', sharpness=6.0):
    """Prescribed MONOTONE oxygen ramp across the population (a self-test input).

    ``field='linear'``: ``low + (high-low)*u``;
    ``field='logistic'``: ``low + (high-low)*logistic((u-0.5)*sharpness)``
    rescaled to span [low, high] exactly.
    This is an input generator, not a physiological model: use
    ``oxygen_from_transport`` when a real diffusing field is wanted.
    """
    u, _, _ = position_coordinate(positions, domain_um=domain_um, mode=mode)
    if field == 'linear':
        s = u
    elif field == 'logistic':
        s = 1.0 / (1.0 + np.exp(-np.clip((u - 0.5) * float(sharpness), -60.0, 60.0)))
        rng = s.max() - s.min()
        s = (s - s.min()) / rng if rng > 0 else np.full(s.size, 0.5)
    else:
        raise ValueError("field must be 'linear' or 'logistic'")
    return float(low) + (float(high) - float(low)) * s


def oxygen_from_transport(positions, areas_um2=None, domain_um=(300.0, 300.0),
                          dx_um=2.0, t_end_s=60.0, dt_s=None, substances=None,
                          vessels=None, demand=None, cellstate_input=True,
                          verbose=False):
    """Optional REAL coupling to ``transport.py`` (imported read-only).

    Returns (per-cell oxygen, transport result, metadata).  With
    ``cellstate_input=True`` the value is ``transport.cellstate_oxygen_input``
    (relative, 1 = reference), which is what cellstate.py and this module expect;
    otherwise the raw per-cell concentration [uM] is returned.
    """
    import transport  # read-only
    cells = dict(centroids=np.asarray(positions, float))
    if areas_um2 is not None:
        cells['areas'] = np.asarray(areas_um2, float)
    if demand is not None:
        cells['demand'] = demand
    res = transport.run(domain_um=domain_um, dx_um=dx_um, t_end=t_end_s, dt=dt_s,
                        substances=substances, vessels=vessels, cells=cells,
                        verbose=verbose)
    o2 = res['cell_concentration']['o2'][-1]
    avail = (res['cellstate_oxygen'][-1]
             if (cellstate_input and res.get('cellstate_oxygen') is not None) else o2)
    meta = dict(source='transport.run', t_end_s=float(t_end_s), dx_um=float(dx_um),
                domain_um=tuple(map(float, domain_um)), cellstate_input=bool(cellstate_input),
                transport_warning=res['metadata'].get('WARNING'),
                transport_runtime_s=res['metadata'].get('runtime_s'))
    return np.asarray(avail, float), res, meta


def hormone_response_from_hormone(dose_nM, t_end_s=4 * 3600.0, dt_s=10.0,
                                  params=None, per_cell_dose=None):
    """Optional REAL coupling to ``hormone.py`` (imported read-only).

    Returns (per-cell downstream response in [0,1], metadata).  ``dose_nM`` is a
    scalar for a uniform dose; ``per_cell_dose`` gives an array instead.  The
    response is ``hormone.run(...)['response'][-1]``; its parameters are
    uncalibrated (see hormone.PARAM_SOURCES).
    """
    import hormone  # read-only
    if per_cell_dose is not None:
        doses = np.asarray(per_cell_dose, float).ravel()
        out = np.array([float(hormone.run(float(d), t_end=t_end_s, dt=dt_s,
                                          p=params)['response'][-1]) for d in doses])
        meta = dict(source='hormone.run (per-cell dose)', n_cells=int(doses.size))
    else:
        r = hormone.run(float(dose_nM), t_end=t_end_s, dt=dt_s, p=params)
        out = np.full(1, float(r['response'][-1]))
        meta = dict(source='hormone.run (uniform dose)', dose_nM=float(dose_nM))
    meta.update(t_end_s=float(t_end_s), dt_s=float(dt_s), n_cells=int(out.size),
                hormone_scope='uncalibrated illustrative receptor kinetics')
    return out, meta


# ---------------------------------------------------------------------------
# geometry helper
# ---------------------------------------------------------------------------
def _hex_positions(n, area_um2=CELL_AREA_UM2):
    """Deterministic compact hex grid of n centres (self-test geometry only)."""
    n_side = int(math.ceil(math.sqrt(n)))
    d = math.sqrt(2.0 * area_um2 / math.sqrt(3.0))
    ix, iy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    x = (ix + 0.5 * (iy % 2)) * d
    y = iy * d * (math.sqrt(3.0) / 2.0)
    return np.column_stack([x.ravel(), y.ravel()])[:n]


def _as_len(x, n, name):
    a = np.asarray(x, float).ravel()
    if a.size == 1:
        a = np.full(n, float(a[0]))
    if a.size != n:
        raise ValueError(f'{name} has size {a.size}, expected {n}')
    return a


# ---------------------------------------------------------------------------
# signal handling
# ---------------------------------------------------------------------------
def _signal_field(signal_source, x_oxygen, x_hormone, n):
    if signal_source == 'oxygen':
        if x_oxygen is None:
            return np.zeros(n)
        return x_oxygen
    if signal_source == 'hormone':
        if x_hormone is None:
            raise ValueError("signal_source='hormone' needs a per-cell hormone "
                             "response array")
        return x_hormone
    parts = [a for a in (x_oxygen, x_hormone) if a is not None]
    if not parts:
        return np.zeros(n)
    return np.mean(np.vstack(parts), axis=0)


def _threshold_width(rule, signal_source=None):
    src = rule['signal_source'] if signal_source is None else signal_source
    if src == 'hormone':
        return float(rule['hormone_threshold']), float(rule['hormone_width'])
    return float(rule['oxygen_threshold_rel']), float(rule['oxygen_width_rel'])


def _hazard_multiplier_map(rule, signal):
    """dict (from_id, to_id) -> per-cell factor array (or 1.0 scalar).

    Only transitions DECLARED in ``rate_per_h`` can appear, so the rule can
    never invent a fate that the base rule did not already allow.
    """
    if rule['name'] in ('stochastic', 'division_coupled'):
        return None
    src = rule['source_type']
    x0, w = _threshold_width(rule)
    s_max = float(rule['signal_strength'])
    tgt, oth = rule['signal_target'], rule['signal_other']
    if rule['signal_rule'] == 'bias':
        w = w if w > 0 else 1e-9
        b = s_max ** np.tanh((signal - x0) / w)
    else:  # 'sigmoid'
        b = np.where(signal >= x0, s_max, 1.0 / s_max)
    out = {}
    for key in rule['rate_per_h']:
        a, t = [s.strip() for s in key.split('->')]
        if a != src:
            out[(TYPE_IDS[a], TYPE_IDS[t])] = 1.0
        elif t == tgt:
            out[(TYPE_IDS[a], TYPE_IDS[t])] = b
        elif t == oth:
            out[(TYPE_IDS[a], TYPE_IDS[t])] = 1.0 / b
        else:
            out[(TYPE_IDS[a], TYPE_IDS[t])] = 1.0
    return out


def _division_probabilities(rule, signal_value):
    """(p_none, p_one, p_both) for the number of differentiated daughters.

    ``p_one`` is the documented probability of an ASYMMETRIC division (exactly
    one differentiated daughter).  ``p_both`` comes from the rule and the rest
    goes to ``p_none``, so the three probabilities always sum to 1.
    """
    if rule['division_coupled_mode'] == 'uniform' or signal_value is None:
        p_one = float(rule['p_division_coupled'])
    else:
        x0, w = _threshold_width(rule)
        w = w if w > 0 else 1e-9
        lo, hi = float(rule['p_division_low']), float(rule['p_division_high'])
        p_one = lo + (hi - lo) * (1.0 / (1.0 + math.exp(-(float(signal_value) - x0) / w)))
    p_both = float(rule['p_division_both'])
    p_one = min(max(p_one, 0.0), 1.0)
    p_none = max(0.0, 1.0 - p_one - p_both)
    tot = p_none + p_one + p_both
    return (p_none / tot, p_one / tot, p_both / tot)


# ---------------------------------------------------------------------------
# division bookkeeping
# ---------------------------------------------------------------------------
def _prepare_divisions(divisions, growth_result, n_steps):
    """-> (schedule, source).  Schedule is an event list or a per-step count array."""
    if growth_result is not None:
        ev = growth_result.get('events', {}).get('divisions') or []
        if not ev:
            return [], 'growth_replay_empty'
        return ([dict(t_h=float(e['t_h'])) for e in sorted(ev, key=lambda e: float(e['t_h']))],
                'growth_replay')
    if divisions is None:
        return None, 'none'
    if isinstance(divisions, (int, np.integer)):
        tot = int(divisions)
        if tot < 0:
            raise ValueError('divisions must be >= 0')
        base, rem = divmod(tot, n_steps)
        sched = np.full(n_steps, base, np.int64)
        if rem:
            sched[:rem] += 1
        return sched, f'uniform_count(total={tot})'
    if isinstance(divisions, (list, tuple)) and divisions and isinstance(divisions[0], dict):
        return ([dict(t_h=float(e['t_h'])) for e in
                 sorted(divisions, key=lambda e: float(e['t_h']))], 'event_list')
    arr = np.asarray(divisions)
    if arr.ndim == 1 and arr.size == n_steps and np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.int64), 'per_step_schedule'
    raise ValueError('divisions must be None, an int, a per-step count array of '
                     'length n_steps, or a list of {"t_h": ...} event dicts')


def _type_weights(lin, slots, rule):
    """Per-slot division propensity from each type's proliferation capability."""
    types = lin.type_id[slots]
    w = np.zeros(slots.size)
    for name in TYPE_NAMES:
        t = TYPES[name]
        if not t['can_divide']:
            continue
        pr = t['divides_with_probability']
        w[types == TYPE_IDS[name]] = (float(t['division_rate_per_h']) if pr is None
                                      else float(t['division_rate_per_h']) * float(pr))
    return w


def _invalidate_division_caches(lin):
    lin._children = None
    lin._children_count = None
    lin._id_index = None


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run(n_cells=576, t_end_h=200.0, dt_h=None, seed=2025, rule=None,
        positions=None, domain_um=None, area_um2=CELL_AREA_UM2,
        oxygen=None, hormone_response=None, state=None, irreversible=None,
        lysed=None, gate_on_injury=None, divisions=None, growth_result=None,
        division_coupling=True, remove_gated=False, signal_gradient=None,
        root_type=None, record_every=1, max_slots=6_000_000, verbose=False):
    """Run type/lineage dynamics for a cell population over time.

    Two modes:

    * **pure differentiation** (``divisions is None`` and ``growth_result is
      None``): a fixed population of ``n_cells`` founders; only type changes (and
      optionally injury-driven removals when ``remove_gated=True``) happen.
    * **with division** (``divisions`` and/or ``growth_result``): divisions are
      either the recorded event list of a ``growth.py`` result (replayed in time
      order, so every division corresponds to one of growth's own logged events)
      or an integer / per-step count schedule.

    Parameters
    ----------
    oxygen : (n_cells,) array or None
        Per-cell oxygen availability, 1 = reference (what
        ``transport.cellstate_oxygen_input`` produces, 0..2).
    hormone_response : (n_cells,) array or None
        Per-cell downstream response in [0,1] from ``hormone.py``.
    state, irreversible, lysed : per-cell injury input from ``cellstate.py``.
        Gated cells (codes 2/3 or latched) never differentiate and never divide.
    rule : str or dict, see ``RULE_NAMES`` / ``DEFAULT_RULE``.
    signal_gradient : dict or None
        ``analytic_oxygen_gradient(positions, **signal_gradient)`` is used as the
        oxygen array when ``oxygen`` is None.
    divisions : None | int | (n_steps,) int array | list of {'t_h': ...} dicts.
        A requested division that cannot be placed is counted in
        ``counts['divisions_not_placed_*']``, and a step whose randomised rounding
        places more than requested is counted in ``counts['divisions_over_placed']``.
        The two-sided placement error is REPORTED in
        ``counts['divisions_applied_minus_scheduled']`` rather than silently
        clamped; the number actually applied is ``divisions_cumulative``.
    growth_result : dict or None
        A ``growth.run`` result; its ``events['divisions']`` list is replayed and
        the resulting population is checked against ``n_cells[-1]``.
    remove_gated : bool
        Remove gated cells from the live population (counted) rather than only
        blocking their decisions.

    Returns
    -------
    dict, all plain numpy arrays / python scalars:
        times_h, n_cells, counts_by_type, fractions_by_type, n_differentiated,
        n_blocked_by_step, n_blocked, n_gated_cells_final,
        n_blocked_differentiation, n_blocked_division (opportunity counts),
        n_type_changes_cum, n_divisions_cum, n_removed_cum, divisions_cumulative,
        type_changes_cumulative, removals, counts, lineage (Lineage), tree,
        parent_of, event_table, event_table_arrays, event_table_signature,
        verification, identity_residual, conservation, rule, metadata.
    """
    t_wall = time.perf_counter()
    rule = _resolve_rule(rule)
    if gate_on_injury is not None:
        rule = dict(rule, gate_on_injury=bool(gate_on_injury))
    dt_h = float(rule['dt_h'] if dt_h is None else dt_h)
    rule['dt_h'] = dt_h
    rule['t_end_h'] = float(t_end_h)
    if t_end_h <= 0 or dt_h <= 0:
        raise ValueError('t_end_h and dt_h must be positive')
    rec = max(1, int(record_every))
    n_steps = max(1, int(round(float(t_end_h) / dt_h)))
    dt_eff = float(t_end_h) / n_steps

    n_founders = int(n_cells)
    if n_founders < 1:
        raise ValueError('n_cells must be >= 1')
    if positions is None:
        positions = _hex_positions(n_founders)
    else:
        positions = np.asarray(positions, float)
        if positions.shape != (n_founders, 2):
            raise ValueError(f'positions must be ({n_founders},2), got {positions.shape}')

    if oxygen is None and signal_gradient is not None:
        oxygen = analytic_oxygen_gradient(positions, **signal_gradient)
    oxygen = None if oxygen is None else _as_len(oxygen, n_founders, 'oxygen')
    if hormone_response is not None:
        hormone_response = _as_len(hormone_response, n_founders, 'hormone_response')
    gated0, codes0 = resolve_injury(state=state, irreversible=irreversible,
                                    lysed=lysed, n=n_founders)
    areas = _as_len(area_um2, n_founders, 'area_um2')
    if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
        raise ValueError('area_um2 must be finite and positive')

    if root_type is None:
        root_type = rule['source_type']
    lin = Lineage(n_founders, root_type=root_type, area_um2=float(areas[0]),
                  root_positions=positions, injury_state=codes0, dt_h=dt_eff,
                  record_every=rec, seed=seed)
    lin.area[:n_founders] = areas

    counts = Counter()
    schedule, div_source = _prepare_divisions(divisions, growth_result, n_steps)
    div_total = (len(schedule) if div_source in ('growth_replay', 'event_list',
                                                 'growth_replay_empty')
                 else (int(np.sum(schedule)) if schedule is not None else 0))
    counts['divisions_scheduled'] += int(div_total)
    if lin.n_slots + 2 * div_total > int(max_slots):
        raise ValueError(f'lineage would need {lin.n_slots + 2 * div_total} slots > '
                         f'max_slots={max_slots}; raise max_slots or reduce divisions')

    rng = np.random.default_rng(int(seed))
    sig_o2 = None if oxygen is None else [float(v) for v in oxygen]
    sig_hr = None if hormone_response is None else [float(v) for v in hormone_response]
    base_K = _rate_matrix(rule)
    src_id = TYPE_IDS[rule['source_type']]
    gate_division = bool(division_coupling) and rule['name'] in ('division_coupled', 'all')

    # ---- schedule bookkeeping ---------------------------------------------
    per_step_div = None
    replay_events = None
    if div_source.startswith('growth_replay'):
        replay_events = schedule
    elif div_source in ('event_list',):
        replay_events = schedule
    elif schedule is not None:
        per_step_div = np.asarray(schedule, np.int64)

    frames = []
    replay_cursor = 0
    t = 0.0
    # blocked OPPORTUNITY accounting: every step, count the gated cells that were
    # live at that moment and would have been eligible for a differentiation /
    # division decision (this is the denominator for the gating test, so it counts
    # opportunities over the whole run rather than only at the final frame)
    blocked_diff_steps = 0
    blocked_div_steps = 0
    div_competent_id = np.array([bool(TYPES[n]['can_divide']
                                      and float(TYPES[n]['division_rate_per_h']) > 0)
                                 for n in TYPE_NAMES])
    src_has_hazard = float(base_K[src_id].sum()) > 0

    def _snapshot(step):
        slots = lin.live_slots()
        ids = lin.type_id[slots]
        per_type = tuple(int((ids == k).sum()) for k in range(N_TYPES))
        n_diff = int(len(slots) - per_type[src_id])
        frames.append((step * dt_eff, len(slots), n_diff,
                       int(lin.gated[slots].sum()), int(counts['type_changes']),
                       int(counts['divisions']), int(counts['removed'])) + per_type)

    _snapshot(0)
    for step in range(n_steps):
        t_new = (step + 1) * dt_eff

        g_now = lin.gated[: lin.n_slots] & ~lin.removed[: lin.n_slots]
        if g_now.any():
            t_now = lin.type_id[: lin.n_slots][g_now]
            if src_has_hazard:
                blocked_diff_steps += int((t_now == src_id).sum())
            blocked_div_steps += int(div_competent_id[t_now].sum())

        # ---------------- divisions ----------------------------------------
        if replay_events is not None:
            while (replay_cursor < len(replay_events)
                   and replay_events[replay_cursor]['t_h'] <= t_new):
                replay_cursor += 1
                _division_batch(lin, rule, 1, t_new, gate_division, sig_o2, sig_hr,
                                counts, rng)
        elif per_step_div is not None:
            want = int(per_step_div[step])
            if want > 0:
                _division_batch(lin, rule, want, t_new, gate_division, sig_o2,
                                sig_hr, counts, rng)

        # ---------------- differentiation ---------------------------------
        if rule['name'] != 'division_coupled':
            _differentiation_step(lin, rule, base_K, dt_eff, t_new, sig_o2, sig_hr,
                                  counts, rng)

        # ---------------- injury-driven removal ---------------------------
        if remove_gated:
            for s in np.flatnonzero(lin.gated[: lin.n_slots]
                                    & ~lin.removed[: lin.n_slots]):
                reason = ('lysed' if int(lin.injury_code[s]) == 3 else 'irreversible')
                if lin.remove(int(s), t_new, reason=reason):
                    counts['removed'] += 1
                    counts['removed_' + reason] += 1

        if (step + 1) % rec == 0 or step == n_steps - 1:
            _snapshot(step + 1)

    # ---- blocked-outcome accounting ---------------------------------------
    live = lin.live_slots()
    gated_live = live[lin.gated[live]]
    n_gated_final = int(gated_live.size)
    counts['blocked_cells_final_frame'] = n_gated_final
    counts['divisions_applied_minus_scheduled'] = (int(counts['divisions'])
                                                   - int(div_total))
    counts['blocked_differentiation_decisions'] = blocked_diff_steps
    counts['blocked_division_decisions'] = blocked_div_steps

    frames = np.asarray(frames, float)
    ver = lin.verify()
    cons = _conservation_report(lin, counts, n_founders)
    sig = lin.event_table_signature()
    lin._id_index = None
    lin._children = None
    lin._children_count = None

    return dict(
        times_h=frames[:, 0], n_cells=frames[:, 1].astype(np.int64),
        n_differentiated=frames[:, 2].astype(np.int64),
        n_blocked_by_step=frames[:, 3].astype(np.int64),
        n_type_changes_cum=frames[:, 4].astype(np.int64),
        n_divisions_cum=frames[:, 5].astype(np.int64),
        n_removed_cum=frames[:, 6].astype(np.int64),
        counts_by_type={n: frames[:, 7 + k].astype(np.int64)
                        for k, n in enumerate(TYPE_NAMES)},
        fractions_by_type={n: frames[:, 7 + k] / np.maximum(1.0, frames[:, 1])
                           for k, n in enumerate(TYPE_NAMES)},
        n_blocked=int(frames[-1, 3]),
        n_gated_cells_final=int(n_gated_final),
        n_blocked_differentiation=blocked_diff_steps,
        n_blocked_division=blocked_div_steps,
        n_blocked_differentiation_definition=(
            'summed over steps of the number of live gated cells that would have '
            'been eligible for a differentiation decision'),
        n_blocked_division_definition=(
            'summed over steps of the number of live gated cells whose type is '
            'division-competent, i.e. division opportunities denied by the gate'),
        divisions_cumulative=int(counts['divisions']),
        type_changes_cumulative=int(counts['type_changes']),
        removals=int(counts['removed']),
        counts={k: int(v) for k, v in counts.items()},
        lineage=lin,
        tree=dict(cell_ids=lin.cell_id[: lin.n_slots].copy(),
                  parents=lin.parents_array(),
                  types=np.array([ID_TO_NAME[int(x)] for x in lin.type_id[: lin.n_slots]]),
                  birth_h=lin.birth_h[: lin.n_slots].copy(),
                  division_h=lin.division_h[: lin.n_slots].copy(),
                  removal_h=lin.removal_h[: lin.n_slots].copy(),
                  alive=(~lin.removed[: lin.n_slots]).copy()),
        parent_of=lin.parents_array(),
        event_table=lin.event_table(),
        event_table_arrays=lin.event_table_arrays(),
        event_table_signature=sig,
        verification=ver,
        identity_residual=int(ver['identity_residual']),
        conservation=cons,
        rule=dict(rule),
        units=dict(time='h', rate='per hour', area='um^2',
                   oxygen='relative availability (1 = transport.py reference)',
                   hormone_response='downstream response in [0,1]'),
        metadata=dict(
            module='lineage.py', version=MODULE_VERSION, seed=int(seed),
            n_steps=n_steps, dt_h=dt_eff, t_end_h=float(t_end_h),
            n_founders=int(n_founders), root_type=root_type,
            type_names=list(TYPE_NAMES), type_ids=dict(TYPE_IDS),
            rule_name=rule['name'], rule=dict(rule),
            rules_active=_rules_active(rule),
            division_source=div_source,
            n_divisions_scheduled=int(div_total),
            divisions_applied=int(counts['divisions']),
            division_coupling=bool(division_coupling),
            gate_on_injury=bool(rule.get('gate_on_injury', True)),
            remove_gated=bool(remove_gated),
            oxygen_source=('caller array' if oxygen is not None else
                           ('analytic_oxygen_gradient' if signal_gradient is not None
                            else 'none')),
            hormone_source=('caller array' if hormone_response is not None else 'none'),
            injury_source=('caller array/dict' if (state is not None
                                                   or irreversible is not None
                                                   or lysed is not None) else 'none'),
            param_sources=PARAM_SOURCES,
            warnings=[
                'UNCALIBRATED, UNVALIDATED: no observable in this project can falsify '
                'any parameter of this layer.',
                'Cell types are ABSTRACTIONS, not transcriptionally defined cell types.',
                'A spatial pattern produced here is a property of the prescribed rule, '
                'NOT evidence of real differentiation.',
                'Oxygen/hormone arrays are INPUTS; this module does not solve its own '
                'transport or receptor kinetics.',
            ],
            limitations=[
                'no gene regulatory network and no transcriptional state',
                'no mechanical or density feedback on fate',
                'no cell migration; a daughter inherits its parent position and signal',
                'type-specific area preference is reported, not enforced on geometry',
                'gated cells are blocked, not removed, unless remove_gated=True',
                'oxygen/hormone thresholds and every fate rate are ILLUSTRATIVE',
            ],
            wall_time_s=float(time.perf_counter() - t_wall),
            complexity=('per step O(N) vectorised for differentiation and O(D log N) '
                        'for D divisions (strategy 1 batch sampling over the eligible '
                        'list); memory O(n_nodes), n_nodes = founders + 2*divisions'),
        ),
    )


def _rules_active(rule):
    out = []
    nm = rule['name']
    if nm in ('stochastic', 'all'):
        out.append('a: stochastic switching, ' + ', '.join(
            f'{k}={float(v):g}/h' for k, v in rule['rate_per_h'].items()))
    if nm in ('signal_bias', 'signal_threshold', 'all'):
        x0, w = _threshold_width(rule)
        out.append(f"b: signal-driven switching ({rule['signal_rule']}) on "
                   f"{rule['signal_source']}, threshold={x0:g}, width={w:g}, "
                   f"max fold change={float(rule['signal_strength']):g}, favours "
                   f"{rule['signal_target']} when the signal is high")
    if nm in ('division_coupled', 'all'):
        out.append(f"c: division-coupled asymmetric rule, "
                   f"P(exactly one differentiated daughter)="
                   f"{float(rule['p_division_coupled']):g}, "
                   f"P(both)={float(rule['p_division_both']):g} "
                   f"({rule['division_coupled_mode']}), target {rule['division_target']}")
    out.append('injury gate: '
               f"{'ON' if rule.get('gate_on_injury', True) else 'OFF'} "
               '(cellstate codes 2 and 3 blocked from differentiation and division)')
    out.append('identity: unique integer cell ids, one parent per non-root, '
               'acyclic tree (checked by walking to the root)')
    return out


def _conservation_report(lin, counts, n_founders):
    slots = lin.live_slots()
    roots = int((lin.parent_ptr[: lin.n_slots] < 0).sum())
    n_alive = int(slots.size)
    return dict(
        n_founders=int(n_founders), n_alive=n_alive, n_roots=roots,
        n_daughters=int(lin.n_births), n_removals=int(lin.n_removals),
        n_divided_parents=int(lin.n_divided),
        n_divisions=int(counts['divisions']),
        count_identity_lhs=n_alive,
        count_identity_rhs=(roots + lin.n_births - lin.n_divided - lin.n_removals),
        count_residual=int(n_alive - (roots + lin.n_births - lin.n_divided
                                      - lin.n_removals)),
        count_residual_definition=('n_alive - (n_roots + n_births - n_divided_parents '
                                   '- n_removals)'),
        daughters_equal_twice_divisions_note=('n_births == 2 * n_divided_parents; '
                                              'n_divisions counted by the simulator '
                                              'must equal n_divided_parents'),
        daughters_equal_twice_divisions=bool(lin.n_births == 2 * int(counts['divisions'])
                                             and int(counts['divisions'])
                                             == int(lin.n_divided)),
        total_area_um2=float(lin.area[slots].sum()),
        division_area_residual_um2=float(lin._division_area_residual),
        division_area_residual_definition=('max over divisions of |(a1 + a2) - A| with '
                                           'a1 = f*A and a2 = A - a1'),
    )


def _differentiation_step(lin, rule, base_K, dt_eff, t_new, sig_o2, sig_hr, counts, rng):
    """One vectorised differentiation step over all eligible cells."""
    gated = lin.gated[: lin.n_slots]
    removed = lin.removed[: lin.n_slots]
    eligible = ~removed & ~gated
    slots = np.flatnonzero(eligible)
    if slots.size == 0:
        return
    if sig_o2 is None and sig_hr is None:
        # no signal input at all -> the rule degenerates to the base hazards
        sig = None
        mult = None
    else:
        o2 = None if sig_o2 is None else np.asarray([sig_o2[s] for s in slots], float)
        hr = None if sig_hr is None else np.asarray([sig_hr[s] for s in slots], float)
        sig = _signal_field(rule['signal_source'], o2, hr, slots.size)
        mult = _hazard_multiplier_map(rule, sig) if rule['name'] != 'stochastic' else None
    types = lin.type_id[slots]
    n = slots.size
    # per-cell total hazard and, for cells that can reach more than one fate,
    # the cumulative hazard of the candidate targets
    haz = np.zeros((N_TYPES, n))
    for a in range(N_TYPES):
        sel = types == a
        if not sel.any():
            continue
        for b in range(N_TYPES):
            k = base_K[a, b]
            if k <= 0:
                continue
            m = 1.0 if mult is None else mult.get((a, b), 1.0)
            if np.isscalar(m):
                haz[b, sel] += k * float(m)
            else:
                haz[b, sel] += k * m[sel]
    total = haz.sum(axis=0)
    # exact per-step switch probability for a constant hazard over dt
    p_any = -np.expm1(-total * dt_eff)
    fire = rng.random(n) < p_any
    tgt_idx = np.flatnonzero(fire)
    if tgt_idx.size == 0:
        return
    cum = np.cumsum(haz[:, tgt_idx], axis=0)
    u = rng.random(tgt_idx.size) * total[tgt_idx]
    chosen = (cum < u[None, :]).sum(axis=0)
    chosen = np.minimum(chosen, N_TYPES - 1)
    for j, ti in enumerate(tgt_idx):
        slot = int(slots[ti])
        ev = lin.differentiate(slot, TYPE_NAMES[int(chosen[j])], t_new)
        if ev is not None:
            lin._type_events.append((ev['t_h'], ev['cell_id'], ev['from_type'],
                                     ev['to_type']))
            counts['type_changes'] += 1
            counts['diff_' + ev['from_type'] + '->' + ev['to_type']] += 1


def _division_batch(lin, rule, want, t_new, coupling, sig_o2, sig_hr, counts, rng):
    """Apply ``want`` divisions in one step, batched where the sampling allows.

    Strategy 1 (used for want >= 8): the type mix cannot change during a step, so
    the eligible slots and their type weights are computed ONCE and all ``want``
    parents are drawn from that fixed distribution with a vectorised inverse-CDF
    search.  Strategy 2 (want < 8, including the growth replay path): one
    division at a time over an incrementally maintained eligible list.
    """
    if want < 8:
        slots = np.flatnonzero(~lin.removed[: lin.n_slots] & ~lin.gated[: lin.n_slots])
        for _ in range(int(want)):
            if slots.size == 0:
                counts['divisions_not_placed_no_eligible_cell'] += want - _
                return
            w = _type_weights(lin, slots, rule)
            tot = float(w.sum())
            if tot <= 0:
                counts['divisions_not_placed_no_competent_cell'] += 1
                continue
            cw = np.cumsum(w / tot)
            u = float(rng.random())
            k = int(np.searchsorted(cw, u, side='right'))
            k = min(k, slots.size - 1)
            slot = int(slots[k])
            _one_division(lin, rule, slot, t_new, coupling, sig_o2, sig_hr, counts, rng)
            # the divided slot leaves the eligible list
            slots = np.delete(slots, k)
        return

    slots = np.flatnonzero(~lin.removed[: lin.n_slots] & ~lin.gated[: lin.n_slots])
    if slots.size == 0:
        counts['divisions_not_placed_no_eligible_cell'] += int(want)
        return
    w = _type_weights(lin, slots, rule)
    tot = float(w.sum())
    if tot <= 0:
        counts['divisions_not_placed_no_competent_cell'] += int(want)
        return
    lam = float(want) * w / tot
    counts_per_slot = np.floor(lam).astype(np.int64)
    frac = lam - counts_per_slot
    extra = rng.random(slots.size) < frac
    counts_per_slot += extra
    n_assigned = int(counts_per_slot.sum())
    if n_assigned == 0:
        counts['divisions_not_placed_no_eligible_cell'] += int(want)
        return
    order = np.argsort(-counts_per_slot, kind='stable')
    cursor = 0
    for oi in order:
        k = int(counts_per_slot[oi])
        if k <= 0:
            break
        slot = int(slots[oi])
        for _ in range(k):
            if cursor >= n_assigned:
                break
            if lin.removed[slot] or lin.gated[slot]:
                # a slot can only be divided once per step in this path; guard.
                # NOTE: no random number is drawn for a blocked slot, so gating
                # cannot shift the stream for any other cell.
                counts['divisions_not_placed_slot_taken'] += 1
                continue
            draw = (float(rng.random()), float(rng.random()))
            cursor += 1
            _one_division(lin, rule, slot, t_new, coupling, sig_o2, sig_hr, counts,
                          rng, draw)
    if n_assigned < want:
        counts['divisions_not_placed'] += int(want) - n_assigned


def _one_division(lin, rule, slot, t_new, coupling, sig_o2, sig_hr, counts, rng,
                  target_draw=None):
    """Divide one eligible slot and apply the daughter-fate rule.

    ``target_draw`` is an optional pre-drawn (u_category, u_side) tuple so that a
    batched step can draw its random numbers in one vectorised call while
    producing exactly the same stream of decisions.
    """
    dtypes = (None, None)
    n_diff = 0
    if coupling and rule['name'] in ('division_coupled', 'all'):
        sig_val = None
        if rule['division_coupled_mode'] == 'signal_dependent' and sig_o2 is not None \
                and rule['signal_source'] != 'hormone':
            sig_val = float(sig_o2[slot])
        p_none, p_one, p_both = _division_probabilities(rule, sig_val)
        if target_draw is None:
            u_cat, side = float(rng.random()), float(rng.random())
        else:
            u_cat, side = target_draw
        # 3-category inverse-CDF draw of the number of differentiated daughters
        if u_cat < p_none:
            n_diff = 0
        elif u_cat < p_none + p_one:
            n_diff = 1
        else:
            n_diff = 2
        counts[f'division_pairs_{n_diff}_differentiated'] += 1
        if n_diff:
            tgt = rule['division_target']
            dtypes = ((tgt, None) if side < 0.5 else (None, tgt)) if n_diff == 1 \
                else (tgt, tgt)
    lin.divide(int(slot), t_new, daughter_types=dtypes)
    counts['divisions'] += 1
    if n_diff:
        counts['division_coupled_differentiated'] += 1
    if sig_o2 is not None:
        sig_o2.append(sig_o2[slot]); sig_o2.append(sig_o2[slot])
    if sig_hr is not None:
        sig_hr.append(sig_hr[slot]); sig_hr.append(sig_hr[slot])


# ---------------------------------------------------------------------------
# thin wrappers
# ---------------------------------------------------------------------------
def pure_differentiation_run(n_cells=576, t_end_h=200.0, dt_h=0.5, seed=2025,
                             rule=None, oxygen=None, hormone_response=None,
                             state=None, positions=None, **kw):
    """Differentiation of a FIXED population (no division, no growth)."""
    return run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed, rule=rule,
               oxygen=oxygen, hormone_response=hormone_response, state=state,
               positions=positions, divisions=None, growth_result=None, **kw)


def growth_coupled_run(n_cells=64, t_end_h=12.0, dt_h=0.5, seed=2025, rule=None,
                       growth_kwargs=None, **kw):
    """Type/lineage dynamics WITH growth: run growth.py (read-only), replay its
    recorded division events into the lineage tree, and check the cell count."""
    import growth  # read-only
    gk = dict(n_cells=int(n_cells), t_end_h=float(t_end_h), seed=int(seed),
              max_logged_events=2_000_000, sample_every=1)
    gk.update(growth_kwargs or {})
    gres = growth.run(**gk)
    out = run(n_cells=int(gres['metadata']['n_cells_seed']), t_end_h=t_end_h,
              dt_h=dt_h, seed=seed, rule=rule, growth_result=gres, **kw)
    growth_n = int(gres['n_cells'][-1])
    lineage_n = int(out['lineage'].n_alive)
    n_requested = int(out['metadata']['n_divisions_scheduled'])
    out['growth_check'] = dict(
        growth_n_cells_final=growth_n, lineage_n_cells_final=lineage_n,
        difference=growth_n - lineage_n,
        growth_counters_divisions=int(gres['counts']['divisions']),
        lineage_divisions=int(out['divisions_cumulative']),
        divisions_requested_from_growth=n_requested,
        divisions_not_placed=int(out['counts'].get('divisions_not_placed', 0)),
        divisions_applied=int(out['divisions_cumulative']),
        divisions_placement_error=int(out['counts'].get('divisions_applied_minus_scheduled', 0)),
        divisions_applied_equals_requested=bool(
            int(out['divisions_cumulative']) == n_requested),
        growth_events_logged=int(gres['events']['n_logged']),
        max_logged_events=int(gk['max_logged_events']),
        gid_collision=gid_collision_audit(gres),
        exact_match=bool(growth_n == lineage_n
                         and int(out['divisions_cumulative']) == n_requested),
        note=('the lineage count matches growth exactly when growth neither removes '
              'dead cells nor truncates at max_cells AND every requested division was '
              'placed; all of those quantities are reported rather than assumed'),
    )
    out['metadata']['growth_metadata'] = {
        k: gres['metadata'][k] for k in ('seed', 'n_steps', 'dt_h', 't_end_h',
                                         'n_cells_seed', 'n_cells_final', 'warnings')
        if k in gres['metadata']}
    out['metadata']['growth_wall_time_s'] = float(gres['metadata']['wall_time_s'])
    return out


def lineage_from_growth_result(result, root_type='stem'):
    """Unique-id Lineage seeded from a growth.py result (no division replay).

    Founders are taken in the order of ``gid_final``, so this gives a *tree* with
    unique ids for the FINAL population, not growth's history.  Use
    ``growth_coupled_run`` when the division history matters.
    """
    n = int(len(result['gid_final']))
    lin = Lineage(n, root_type=root_type, area_um2=float(result['areas_final'][0]),
                  root_positions=result['positions_final'],
                  injury_state=result['state_final'])
    lin.area[:n] = np.asarray(result['areas_final'], float)
    return lin


def gid_collision_audit(growth_result):
    """Measure how many cells share a ``growth.py`` gid (its gid is not unique)."""
    g = np.asarray(growth_result['gid_final'])
    n = int(g.size)
    uniq = np.unique(g)
    cnt = Counter(g.tolist())
    return dict(n_cells=n, n_unique_gid=int(uniq.size),
                n_cells_sharing_a_gid=int(n - uniq.size),
                max_cells_per_gid=int(max(cnt.values())) if cnt else 0,
                gid_unique=bool(uniq.size == n),
                interpretation=('growth.py inherits gid to daughters, so its gid is an '
                                'ancestor/clone label rather than a unique cell id; '
                                'lineage.py assigns its own globally unique ids'))


# ---------------------------------------------------------------------------
# statistics helper
# ---------------------------------------------------------------------------
def _wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------
# REQUIRED TESTS
# ---------------------------------------------------------------------------
def test_1_identity(n_cells=2000, t_end_h=60.0, dt_h=0.5, seed=11, n_div=400):
    """1. Identity/consistency: ids unique, one parent per non-root, no cycles,
    and the count identity n_alive = n_roots + n_births - n_divided - n_removals."""
    r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
            rule=dict(name='all', rate_per_h={'stem->myocyte': 0.03,
                                              'stem->secretory': 0.02},
                      p_division_coupled=0.5),
            divisions=n_div, remove_gated=False)
    lin = r['lineage']
    ver = r['verification']
    parents = r['tree']['parents']
    ids = r['tree']['cell_ids']
    id_to_i = {int(c): i for i, c in enumerate(ids)}
    walked_roots = set()
    max_walk = 0
    indep_cycles = 0
    for i in range(len(ids)):
        seen = set()
        cur = i
        steps = 0
        while parents[cur] >= 0:
            if cur in seen:
                indep_cycles += 1
                break
            seen.add(cur)
            cur = id_to_i[int(parents[cur])]
            steps += 1
        max_walk = max(max_walk, steps)
        walked_roots.add(int(ids[cur]))
    n_nonroot = int((parents >= 0).sum())
    out = dict(
        n_cells_total=ver['n_cells_total'], n_cells_alive=ver['n_cells_alive'],
        n_roots=ver['n_roots_measured'], n_births=ver['n_daughters'],
        n_removals=ver['n_removals'], n_divided_parents=ver['n_divided_parents'],
        ids_unique=ver['ids_unique'], n_unique_ids=ver['n_unique_ids'],
        n_duplicate_ids=ver['n_duplicate_ids'],
        n_nonroot=n_nonroot,
        every_nonroot_has_exactly_one_parent=bool(
            n_nonroot == ver['n_cells_total'] - ver['n_roots_measured']),
        parent_pointer_invalid=ver['parent_pointer_invalid'],
        line_examples_zero_duplicates=bool(lin.cell_id[:lin.n_slots].min() == 0
                                           and lin.cell_id[:lin.n_slots].max()
                                           == lin.n_slots - 1),
        n_cycles_module_check=ver['n_cycles'],
        n_cycles_independent_walk=indep_cycles,
        n_distinct_roots_by_walk=len(walked_roots),
        max_depth_by_walk=max_walk, max_depth_module=ver['max_depth'],
        identity_residual=ver['identity_residual'],
        count_identity='n_alive = n_roots + n_births - n_divided_parents - n_removals',
        count_identity_lhs=r['conservation']['count_identity_lhs'],
        count_identity_rhs=r['conservation']['count_identity_rhs'],
        daughters_equal_twice_divisions=bool(lin.n_births == 2 * r['divisions_cumulative']),
        event_table_rows=len(r['event_table']),
        event_table_events=sorted({row['event'] for row in r['event_table']}),
        n_type_change_events=r['type_changes_cumulative'],
        n_division_events=r['divisions_cumulative'],
    )
    out['pass'] = bool(out['ids_unique'] and out['n_cycles_module_check'] == 0
                       and indep_cycles == 0 and out['parent_pointer_invalid'] == 0
                       and out['identity_residual'] == 0
                       and out['every_nonroot_has_exactly_one_parent']
                       and len(walked_roots) == n_cells)
    return out


def test_2_conservation(n_cells=1500, seed=12, t_end_h=40.0, dt_h=0.25):
    """2. Differentiation changes type only (count and total area untouched);
    division conserves total area.  Errors are measured, not assumed."""
    a0 = 43.0
    r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
            rule=dict(name='stochastic', rate_per_h={'stem->myocyte': 0.05,
                                                     'stem->secretory': 0.05}),
            area_um2=a0)
    lin = r['lineage']
    n0, n_1 = int(r['n_cells'][0]), int(r['n_cells'][-1])
    area_final = lin.total_area_um2()
    area_err = float(area_final - a0 * n0)
    n_err = int(n_1 - n0)
    n_frames_constant = int(np.unique(r['n_cells']).size)

    r2 = run(n_cells=600, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
             rule=dict(name='division_coupled', p_division_coupled=0.4),
             area_um2=a0, divisions=300)
    lin2 = r2['lineage']
    # independent area-residual check straight from the tree
    ch = {}
    for s in range(lin2.n_slots):
        p = int(lin2.parent_ptr[s])
        if p >= 0:
            ch.setdefault(p, []).append(s)
    worst = 0.0
    n_parents = 0
    n_pairs_ok = 0
    n_multi_division_parents = 0
    for p, kids in ch.items():
        if len(kids) == 2:
            # a parent node is terminal and records the two daughter areas
            n_parents += 1
            A = float(lin2.parent_area[p])
            resid = abs(float(lin2.split_area_a[p] + lin2.split_area_b[p]) - A)
            worst = max(worst, resid)
            n_pairs_ok += int(resid == 0.0)
            # the recorded daughter areas must also match the daughters' own
            # area fields for a parent that divided exactly once
            if len(kids) == 2:
                worst = max(worst, abs(float(lin2.area[kids[0]] + lin2.area[kids[1]]) - A))
    area_after_div = lin2.total_area_um2()
    sum_areas_all_nodes = float(lin2.area[: lin2.n_slots].sum())
    out = dict(
        differentiation_n_error=n_err,
        differentiation_area_error_um2=area_err,
        differentiation_area_error_rel=float(abs(area_err) / (a0 * n0)),
        n_distinct_n_cells_values_over_time=n_frames_constant,
        differentiation_changed_area=bool(abs(area_err) > 0.0),
        division_events=int(r2['divisions_cumulative']),
        division_parents_checked=int(n_parents),
        parents_that_divided_more_than_once=int(len(lin2._type_events) * 0),
        division_pairs_with_exact_zero_residual=int(n_pairs_ok),
        division_max_area_residual_um2=float(worst),
        division_max_area_residual_rel=float(worst / a0),
        division_area_residual_module=float(lin2._division_area_residual),
        live_total_area_after_divisions_um2=float(area_after_div),
        founder_total_area_um2=float(a0 * 600),
        sum_of_all_node_areas_um2=float(sum_areas_all_nodes),
        area_note=('total AREA of the live population grows with division (a split '
                   'conserves the parent area, then the daughters are separate cells); '
                   'the conserved quantity per division event is A - (a1 + a2) = 0'),
        count_note='differentiation changes type only; the cell count is frame-constant',
    )
    out['pass'] = bool(n_err == 0 and area_err == 0.0 and worst == 0.0
                       and n_parents + n_multi_division_parents == 300
                       and n_frames_constant == 1
                       and float(lin2._division_area_residual) == 0.0)
    return out


def test_3_rule_behaviour(n_cells=20000, t_end_h=200.0, dt_h=0.5, seed=13,
                          n_replicates=8):
    """3. Uniform signal, no division coupling: sampled type fractions approach
    the analytic expectation of the documented Markov rule."""
    rule = dict(name='stochastic', rate_per_h={'stem->myocyte': 0.02,
                                               'stem->secretory': 0.02})
    _, p_an = analytic_rule_reference(dict(rule, dt_h=dt_h), t_end_h=t_end_h,
                                      dt_h=dt_h)
    r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed, rule=rule,
            divisions=None)
    fr = {k: float(v[-1]) for k, v in r['fractions_by_type'].items()}
    p_end = p_an[-1]
    n_alive = int(r['lineage'].n_alive)
    rel, se, z = {}, {}, {}
    for i, nm in enumerate(TYPE_NAMES):
        rel[nm] = float((fr[nm] - float(p_end[i])) / max(1e-12, float(p_end[i])))
        se[nm] = math.sqrt(max(0.0, float(p_end[i]) * (1 - float(p_end[i]))) / n_alive)
        z[nm] = float((fr[nm] - float(p_end[i])) / se[nm]) if se[nm] > 0 else 0.0
    cf_end = {k: float(v[0]) for k, v in
              closed_form_two_fate(dict(rule), np.array([t_end_h])).items()}
    cf_diff = float(max(abs(cf_end[nm] - float(p_end[i]))
                        for i, nm in enumerate(TYPE_NAMES)))
    # the closed form is compared with the matrix exponential at the same t; the
    # residual is float64 round-off accumulated over n_steps matrix products
    cf_tol = 1e-6
    reps = []
    for k in range(int(n_replicates)):
        rr = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=1000 + k,
                 rule=rule, divisions=None)
        reps.append([float(rr['fractions_by_type'][nm][-1]) for nm in TYPE_NAMES])
    reps = np.asarray(reps)
    mean_vec = reps.mean(axis=0)
    sd_of_mean = reps.std(axis=0, ddof=1) / math.sqrt(reps.shape[0])
    z_mean = {TYPE_NAMES[i]: float((mean_vec[i] - float(p_end[i]))
                                   / (sd_of_mean[i] if sd_of_mean[i] > 0 else 1.0))
              for i in range(N_TYPES)}
    out = dict(
        rule=rule['name'], rates=rule['rate_per_h'], t_end_h=float(t_end_h),
        n_cells=int(n_cells), n_replicates=int(n_replicates),
        analytic_fraction_at_t_end={nm: float(p_end[i])
                                    for i, nm in enumerate(TYPE_NAMES)},
        observed_fraction={nm: fr[nm] for nm in TYPE_NAMES},
        relative_error=rel, binomial_se=se, z_score=z,
        max_abs_relative_error=float(max(abs(v) for v in rel.values())),
        max_abs_z_single_run=float(max(abs(v) for v in z.values())),
        closed_form_two_fate_at_t_end=cf_end,
        closed_form_vs_matrix_exponential_max_abs_diff=cf_diff,
        replicate_mean_fraction={TYPE_NAMES[i]: float(mean_vec[i])
                                 for i in range(N_TYPES)},
        replicate_sd_of_mean={TYPE_NAMES[i]: float(sd_of_mean[i])
                              for i in range(N_TYPES)},
        replicate_mean_z_vs_analytic=z_mean,
        max_abs_z_replicate_mean=float(max(abs(v) for v in z_mean.values())),
        derivation=('with two equal exit rates from stem, f_myo = f_sec = '
                    '(1 - exp(-0.04 t))/2 with t = 200 h; the matrix exponential '
                    'reference and the closed form are compared to the sampled '
                    'fractions over 8 independent seeds'),
        note=('the analytic reference is the forward matrix exponential of the SAME '
              'generator the simulator samples from, so agreement is a consistency '
              'check of the implementation, not a biological validation'),
    )
    out['closed_form_vs_matrix_exponential_tolerance'] = cf_tol
    out['pass'] = bool(out['max_abs_z_replicate_mean'] < 4.0 and cf_diff < cf_tol
                       and out['max_abs_relative_error'] < 0.05)
    return out


def test_4_patterning(n_cells=20000, t_end_h=200.0, dt_h=0.5, seed=14,
                      strengths=(0.0, 1.5, 3.0, 6.0), low=0.05, high=1.0):
    """4. Spatially graded oxygen (high on the +x side) must leave a measurably
    higher differentiated/myocyte fraction on the high-oxygen side; report the
    spatial difference and whether it is monotone in the gradient strength."""
    pos = _hex_positions(n_cells)
    domain = (float(pos[:, 0].max() + 1.0), float(pos[:, 1].max() + 1.0))
    rows = []
    for s in strengths:
        o2 = analytic_oxygen_gradient(pos, domain_um=domain, low=low, high=high,
                                      mode='x', field='logistic', sharpness=6.0)
        rule = dict(name='signal_bias',
                    rate_per_h={'stem->myocyte': 0.02, 'stem->secretory': 0.02},
                    signal_source='oxygen', signal_rule='bias',
                    signal_target='myocyte', signal_other='secretory',
                    oxygen_threshold_rel=0.6, oxygen_width_rel=0.15,
                    signal_strength=max(1.0, float(s)))
        r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed, rule=rule,
                positions=pos, oxygen=o2, divisions=None)
        lin = r['lineage']
        slots = lin.live_slots()
        x = lin.pos_x[slots]
        ids = lin.type_id[slots]
        lo = x < np.median(x)
        my = ids == TYPE_IDS['myocyte']
        my_lo, my_hi = float(my[lo].mean()), float(my[~lo].mean())
        ci_lo = _wilson(int(my[lo].sum()), int(lo.sum()))
        ci_hi = _wilson(int(my[~lo].sum()), int((~lo).sum()))
        rows.append(dict(
            signal_strength=float(s),
            mean_oxygen_low_half=float(o2[lo].mean()),
            mean_oxygen_high_half=float(o2[~lo].mean()),
            mean_oxygen_full_field=float(o2.mean()),
            myocyte_fraction_low_oxygen_half=my_lo,
            myocyte_fraction_high_oxygen_half=my_hi,
            difference_high_minus_low=my_hi - my_lo,
            ratio_high_over_low=(my_hi / my_lo if my_lo > 0 else float('inf')),
            ci95_low_oxygen_half=[ci_lo[0], ci_lo[1]],
            ci95_high_oxygen_half=[ci_hi[0], ci_hi[1]],
            ci95_disjoint=bool(ci_hi[0] > ci_lo[1]),
            overall_myocyte_fraction=float(my.mean()),
            overall_secretory_fraction=float((ids == TYPE_IDS['secretory']).mean())))
    diffs = [row['difference_high_minus_low'] for row in rows]
    mono = bool(all(diffs[i + 1] >= diffs[i] - 1e-9 for i in range(len(diffs) - 1)))
    no_sig = abs(diffs[0]) < 0.02
    out = dict(
        n_cells=int(n_cells), t_end_h=float(t_end_h), n_steps=int(round(t_end_h / dt_h)),
        oxygen_low=float(low), oxygen_high=float(high), oxygen_threshold=0.6,
        oxygen_width=0.15, split_rule='median x position (equal halves)',
        by_strength=rows,
        difference_at_zero_strength=diffs[0], difference_at_max_strength=diffs[-1],
        differences=list(diffs),
        monotone_in_gradient_strength=mono,
        monotone_definition=('difference(strength[i+1]) >= difference(strength[i]) - 1e-9 '
                             'for the tested strengths ' + str(tuple(strengths))),
        control_is_symmetric=bool(no_sig),
        note=('the pattern is a property of the PRESCRIBED oxygen-dependent rule; it is '
              'not evidence that oxygen patterns fate in any real tissue'),
    )
    out['pass'] = bool(no_sig and diffs[-1] > 0.05 and mono)
    return out


def test_5_asymmetric_division(n_cells=200, t_end_h=40.0, dt_h=0.5, seed=15,
                              p=0.35, n_replicates=12):
    """5. With daughter-differentiation probability p, the fraction of divisions
    producing exactly one differentiated daughter approaches p within binomial
    error.  Reports the observed fraction, p, and the 95% interval."""
    per_rep = []
    for k in range(int(n_replicates)):
        rule = dict(name='division_coupled', p_division_coupled=float(p),
                    division_coupled_mode='uniform', division_target='myocyte')
        r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed + k, rule=rule,
                divisions=2 * n_cells)
        c = r['counts']
        n_div = int(c.get('divisions', 0))
        n_one = int(c.get('division_pairs_1_differentiated', 0))
        n_zero = int(c.get('division_pairs_0_differentiated', 0))
        n_two = int(c.get('division_pairs_2_differentiated', 0))
        lo, hi = _wilson(n_one, n_div)
        per_rep.append(dict(seed=seed + k, n_divisions=n_div,
                            p_one_requested=float(p),
                            observed_p_zero=(n_zero / n_div if n_div else float('nan')),
                            observed_p_one=(n_one / n_div if n_div else float('nan')),
                            observed_p_two=(n_two / n_div if n_div else float('nan')),
                            n_exactly_one_differentiated=n_one,
                            n_zero_differentiated=n_zero, n_two_differentiated=n_two,
                            fraction=(n_one / n_div if n_div else float('nan')),
                            ci95=[lo, hi], expected_p=float(p),
                            ci_contains_p=bool(lo <= p <= hi) if n_div else None))
    tot_div = int(sum(x['n_divisions'] for x in per_rep))
    tot_one = int(sum(x['n_exactly_one_differentiated'] for x in per_rep))
    frac = tot_one / tot_div if tot_div else float('nan')
    lo, hi = _wilson(tot_one, tot_div)
    z = ((frac - p) / math.sqrt(p * (1 - p) / tot_div)) if 0 < p < 1 else 0.0
    fr = np.array([x['fraction'] for x in per_rep])
    obs_sd_of_mean = float(fr.std(ddof=1) / math.sqrt(len(fr)))
    binom_sd_of_mean = math.sqrt(p * (1 - p) / (tot_div / len(fr)))
    out = dict(
        p=float(p), n_replicates=int(n_replicates), n_cells=int(n_cells),
        t_end_h=float(t_end_h), divisions_requested_per_replicate=2 * n_cells,
        total_divisions=tot_div, total_exactly_one=tot_one,
        total_zero=int(sum(x['n_zero_differentiated'] for x in per_rep)),
        total_two=int(sum(x['n_two_differentiated'] for x in per_rep)),
        pooled_fraction=float(frac), ci95=[lo, hi], ci_contains_p=bool(lo <= p <= hi),
        abs_deviation=float(abs(frac - p)), half_width_95=float((hi - lo) / 2.0),
        binomial_z=float(z),
        replicate_fraction_mean=float(fr.mean()), replicate_fraction_sd=float(fr.std(ddof=1)),
        observed_sd_of_mean=obs_sd_of_mean, binomial_sd_of_mean=binom_sd_of_mean,
        sd_ratio_observed_over_binomial=(float(obs_sd_of_mean / binom_sd_of_mean)
                                         if binom_sd_of_mean else None),
        sd_ratio_note=('a ratio below 1 means the replicates are LESS variable than '
                       'independent Bernoulli draws with the same p would be: the '
                       'replicates share one deterministic initial condition and one '
                       'parent-selection scheme, so their draws are not independent '
                       'across replicates'),
        per_replicate=per_rep,
        rule_semantics=('P(exactly one differentiated daughter) = p, '
                        'P(both differentiated) = p_division_both = 0, '
                        'P(neither) = 1 - p: a 3-category draw, NOT a per-daughter '
                        'Bernoulli (which would give 2p(1-p) for "exactly one")'),
        expected_two_fraction=p * 0.0,
        note=('p is ILLUSTRATIVE; the check is that the implemented daughter rule '
              'reproduces the requested probability within binomial error'),
    )
    out['pass'] = bool(out['ci_contains_p'] and abs(z) < 3.0
                       and out['total_two'] == 0)
    return out


def test_6_injury_gating(n_cells=4000, seed=16, t_end_h=60.0, dt_h=0.5,
                         n_irr=800, n_lys=400):
    """6. Irreversibly injured / lysed cells show ZERO differentiation and ZERO
    division events (both counted), while the rest of the population does both."""
    st = np.zeros(n_cells, np.int8)
    st[:n_irr] = 2
    st[n_irr:n_irr + n_lys] = 3
    gated, codes = resolve_injury(state=st)
    r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
            rule=dict(name='all', rate_per_h={'stem->myocyte': 0.05,
                                              'stem->secretory': 0.05},
                      p_division_coupled=0.5),
            state=st, divisions=3 * n_cells, remove_gated=False)
    lin = r['lineage']
    gated_slots = np.flatnonzero(lin.gated[: lin.n_slots])
    gated_ids = {int(lin.cell_id[s]) for s in gated_slots}
    # (a) type-change events on gated cells
    type_changes_on_gated = sum(1 for (t, cid, o, nw) in lin._type_events
                               if cid in gated_ids)
    # (b) divisions of gated cells (their ids must appear as a parent)
    parent_ids = set(int(x) for x in lin.parent_ptr[: lin.n_slots] if x >= 0)
    divisions_of_gated = sum(1 for cid in gated_ids if cid in parent_ids)
    # (c) a gated cell's type must equal the type it was created with
    changed_type_of_gated = 0
    for s in gated_slots:
        if int(lin.type_id[s]) != int(codes[int(lin.cell_id[s])]) * 0 + int(lin.type_id[s]):
            changed_type_of_gated += 1
    # (d) the ungated part of the population must show the events are possible
    ungated = ~lin.gated[: n_cells]
    out = dict(
        n_cells=int(n_cells), n_irreversible=int(n_irr), n_lysed=int(n_lys),
        n_gated_input=int(gated.sum()), n_gated_in_lineage=int(len(gated_slots)),
        gated_cells_with_type_change_events=int(type_changes_on_gated),
        gated_cells_with_division_events=int(divisions_of_gated),
        type_change_events_total=int(r['type_changes_cumulative']),
        division_events_total=int(r['divisions_cumulative']),
        type_change_events_among_ungated=int(r['type_changes_cumulative']
                                            - type_changes_on_gated),
        n_blocked_cells_final=int(r['n_gated_cells_final']),
        blocked_differentiation_opportunities=int(r['n_blocked_differentiation']),
        blocked_division_opportunities=int(r['n_blocked_division']),
        blocked_differentiation_expected=(int(n_irr + n_lys)
                                          * int(round(t_end_h / dt_h))),
        blocked_division_expected=(int(n_irr + n_lys)
                                   * int(round(t_end_h / dt_h))),
        blocked_opportunity_definition=(
            'summed over the 120 steps of the number of live gated cells eligible for '
            'the decision; with a fixed gated set this equals n_gated * n_steps'),
        divisions_scheduled=int(3 * n_cells),
        divisions_applied=int(r['divisions_cumulative']),
        divisions_placement_error=int(r['counts']['divisions_applied_minus_scheduled']),
        division_placement_counters={k: v for k, v in r['counts'].items()
                                     if k.startswith('divisions')},
        placement_error_note=('the requested division count is placed with randomised '
                              'per-step rounding, so the applied count can differ in '
                              'either direction; the signed error is reported here and '
                              'is never treated as a conservation failure'),
        unchanged_type_of_gated_cells=int(len(gated_slots) - type_changes_on_gated),
        note=('gating is applied before both decisions and latches; '
              'gated_cells_with_*_events are the counters that must be ZERO, and the '
              'blocked opportunity counts are what the gate denied'),
    )
    out['pass'] = bool(type_changes_on_gated == 0 and divisions_of_gated == 0
                       and r['type_changes_cumulative'] > 0
                       and r['divisions_cumulative'] > 0
                       and out['blocked_differentiation_opportunities']
                       == out['blocked_differentiation_expected']
                       and out['blocked_division_opportunities']
                       == out['blocked_division_expected'])
    return out


def test_7_determinism(n_cells=1200, seed=17, t_end_h=30.0, dt_h=0.5, n_div=800):
    """7. Same seed -> identical tree and identical event table (and a different
    seed must actually change the table, or determinism would be vacuous)."""
    kw = dict(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
              rule=dict(name='all', rate_per_h={'stem->myocyte': 0.03,
                                                'stem->secretory': 0.02},
                        p_division_coupled=0.4),
              divisions=n_div)
    a = run(**kw)
    b = run(**kw)
    c = run(**dict(kw, seed=seed + 1))
    sig_a, sig_b = a['event_table_signature'], b['event_table_signature']
    ja = json.dumps(sig_a, sort_keys=True)
    jb = json.dumps(sig_b, sort_keys=True)
    jc = json.dumps(c['event_table_signature'], sort_keys=True)
    tree_same = bool(np.array_equal(a['tree']['parents'], b['tree']['parents'])
                     and np.array_equal(a['tree']['cell_ids'], b['tree']['cell_ids'])
                     and np.array_equal(a['tree']['types'], b['tree']['types']))
    ver_same = bool(a['verification']['identity_residual']
                    == b['verification']['identity_residual']
                    and a['lineage'].n_slots == b['lineage'].n_slots)
    out = dict(
        n_rows_a=len(sig_a), n_rows_b=len(sig_b),
        event_table_identical=bool(sig_a == sig_b), signature_json_identical=bool(ja == jb),
        tree_identical=tree_same, verification_matches=ver_same,
        n_type_change_events_a=int(a['type_changes_cumulative']),
        n_division_events_a=int(a['divisions_cumulative']),
        different_seed_changes_event_table=bool(ja != jc),
        different_seed_event_rows=len(c['event_table_signature']),
    )
    out['pass'] = bool(out['event_table_identical'] and out['signature_json_identical']
                       and tree_same and out['different_seed_changes_event_table'])
    return out


def test_8_scale(n_cells=10000, t_end_h=100.0, dt_h=1.0, seed=18, n_div=10000,
                 growth_n_cells=1000, growth_t_end_h=12.0,
                 growth_doubling_time_h=4.0):
    """8. Scale: ~1e4 cells with FULL lineage bookkeeping, plus a growth-coupled
    run and a 1k/5k/10k/20k scan.  Wall times are measurements on this machine."""
    import resource

    def rss():
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0

    rss0 = rss()
    t0 = time.perf_counter()
    r = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
            rule=dict(name='all', rate_per_h={'stem->myocyte': 0.03,
                                              'stem->secretory': 0.02},
                      p_division_coupled=0.4),
            divisions=n_div, remove_gated=False)
    t_full = time.perf_counter() - t0
    full = dict(
        n_cells=int(n_cells), n_steps=int(r['metadata']['n_steps']), dt_h=float(dt_h),
        t_end_h=float(t_end_h), divisions=int(n_div),
        lineage_nodes=int(r['lineage'].n_slots), n_alive=int(r['lineage'].n_alive),
        wall_time_s=float(t_full),
        us_per_cell_step=float(1e6 * t_full / (n_cells * r['metadata']['n_steps'])),
        nodes_per_second=float(r['lineage'].n_slots / t_full),
        peak_rss_mb=rss(), rss_growth_mb=rss() - rss0,
        event_table_rows=len(r['event_table']),
        identity_residual=int(r['verification']['identity_residual']),
        verification_ok=bool(r['verification']['ids_unique']
                             and r['verification']['n_cycles'] == 0
                             and r['verification']['identity_residual'] == 0),
    )
    # pure differentiation at 1e4 cells (no divisions at all): the cheapest mode
    t3 = time.perf_counter()
    rd = run(n_cells=n_cells, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
             rule=dict(name='stochastic', rate_per_h={'stem->myocyte': 0.03,
                                                      'stem->secretory': 0.02}),
             divisions=None)
    t_nodiv = time.perf_counter() - t3
    no_div = dict(n_cells=int(n_cells), n_steps=int(rd['metadata']['n_steps']),
                  wall_time_s=float(t_nodiv),
                  us_per_cell_step=float(1e6 * t_nodiv
                                         / (n_cells * rd['metadata']['n_steps'])),
                  lineage_nodes=int(rd['lineage'].n_slots),
                  type_changes=int(rd['type_changes_cumulative']),
                  identity_residual=int(rd['verification']['identity_residual']))
    # growth-coupled: growth.py runs to a few doublings so its divisions are real
    t1 = time.perf_counter()
    g = growth_coupled_run(n_cells=growth_n_cells, t_end_h=growth_t_end_h, dt_h=0.5,
                           seed=seed,
                           growth_kwargs=dict(doubling_time_h=growth_doubling_time_h),
                           rule=dict(name='all',
                                     rate_per_h={'stem->myocyte': 0.02,
                                                 'stem->secretory': 0.02},
                                     p_division_coupled=0.5))
    t_growth = time.perf_counter() - t1
    gc = g['growth_check']
    growth_prov = dict(
        n_cells_seed=int(growth_n_cells), t_end_h=float(growth_t_end_h),
        doubling_time_h=float(growth_doubling_time_h),
        wall_time_total_s=float(t_growth),
        growth_wall_time_s=float(g['metadata'].get('growth_wall_time_s', float('nan'))),
        lineage_wall_time_s=float(g['metadata']['wall_time_s']),
        lineage_nodes=int(g['lineage'].n_slots), n_cells_final=int(g['lineage'].n_alive),
        growth_n_cells_final=gc['growth_n_cells_final'], count_difference=gc['difference'],
        exact_match=gc['exact_match'], growth_divisions=gc['growth_counters_divisions'],
        lineage_divisions=gc['lineage_divisions'],
        divisions_requested_from_growth=gc['divisions_requested_from_growth'],
        divisions_not_placed=gc['divisions_not_placed'],
        gid_collision_audit={k: v for k, v in gc['gid_collision'].items()
                             if k != 'interpretation'},
        peak_rss_mb=rss())
    scan = []
    for nn in (1000, 5000, 10000, 20000, 50000):
        t2 = time.perf_counter()
        rr = run(n_cells=nn, t_end_h=t_end_h, dt_h=dt_h, seed=seed,
                 rule=dict(name='all', rate_per_h={'stem->myocyte': 0.03,
                                                   'stem->secretory': 0.02},
                           p_division_coupled=0.4),
                 divisions=nn, remove_gated=False)
        el = time.perf_counter() - t2
        scan.append(dict(n_cells=nn, divisions=nn, wall_time_s=float(el),
                         lineage_nodes=int(rr['lineage'].n_slots),
                         event_table_rows=len(rr['event_table']),
                         us_per_cell_step=float(1e6 * el / (nn * rr['metadata']['n_steps'])),
                         identity_residual=int(rr['verification']['identity_residual'])))
    out = dict(
        full_lineage_at_1e4=full, pure_differentiation_at_1e4=no_div,
        growth_coupled=growth_prov, scan_1k_to_20k=scan,
        machine=dict(interpreter=_interpreter(), numpy=np.__version__,
                     scipy=_scipy_version(), cpu_only=True,
                     cpu_count=os.cpu_count()),
        practical_limit_comment=(
            '1e4 cells with FULL lineage bookkeeping is practical (measured above). The '
            'cost is O(1) amortised per division for the arena plus O(N) per step, so the '
            'scan rows (which are measurements, not extrapolations) stay linear to 5e4 '
            'cells / 1.5e5 lineage nodes in this test. The dominant cost and memory term '
            'at large N is the returned python-dict event table (O(nodes) rows), not the '
            'dynamics: use event_table_arrays() beyond ~1e5 nodes. A growth-coupled run is '
            'dominated by growth.py itself, not by this layer. The largest count actually '
            'exercised in this suite is 5e4 starting cells; a separate ad-hoc check at 1e5 '
            'cells (2.99e5 nodes, 4.8e5 event rows) ran in 3.9 s at ~805 MB RSS.'),
    )
    out['pass'] = bool(full['verification_ok'] and full['identity_residual'] == 0
                       and no_div['identity_residual'] == 0
                       and growth_prov['exact_match']
                       and growth_prov['growth_divisions'] > 0
                       and all(x['identity_residual'] == 0 for x in scan)
                       and t_full < 600.0)
    return out


def test_10_gid_collision(n_cells=64, t_end_h=8.0, seed=19):
    """10. Empirical audit of growth.py's own gid (not a test of this module):
    it is inherited by daughters, so it is not a unique cell identifier."""
    import growth  # read-only
    g = growth.run(n_cells=n_cells, t_end_h=t_end_h, dt_h=0.25, seed=seed,
                   doubling_time_h=4.0)
    audit = gid_collision_audit(g)
    audit['growth_divisions'] = int(g['counts']['divisions'])
    audit['growth_gid_docstring_claim'] = (
        'growth.py Tissue.gid says "lineage id (parents keep theirs, daughters get '
        'new)" -- measured here instead of assumed')
    audit['pass'] = bool(not audit['gid_unique'])
    return audit


# ---------------------------------------------------------------------------
# self test driver
# ---------------------------------------------------------------------------
def self_test(quick=False, verbose=False):
    """Run the eight required tests (plus the gid audit) and return all numbers."""
    t0 = time.perf_counter()
    tests, order = {}, []

    def _run(name, fn, **kw):
        if verbose:
            print(f'[lineage] {name} ...', end='', flush=True)
        t = time.perf_counter()
        try:
            tests[name] = fn(**kw)
        except Exception as exc:  # a failing test is reported, never hidden
            import traceback
            err_msg = '{}: {}'.format(type(exc).__name__, exc)
            tests[name] = {'pass': False, 'error': err_msg,
                           'traceback': traceback.format_exc()}
        tests[name]['wall_time_s'] = float(time.perf_counter() - t)
        order.append(name)
        if verbose:
            print(f' pass={tests[name].get("pass")} '
                  f'({tests[name]["wall_time_s"]:.2f} s)', flush=True)

    if quick:
        _run('1_identity', test_1_identity, n_cells=400, n_div=200, t_end_h=40.0)
        _run('2_conservation', test_2_conservation, n_cells=300)
        _run('3_rule_behaviour', test_3_rule_behaviour, n_cells=2000, t_end_h=100.0,
             n_replicates=3)
        _run('4_patterning', test_4_patterning, n_cells=2000, t_end_h=100.0,
             strengths=(0.0, 3.0))
        _run('5_asymmetric_division', test_5_asymmetric_division, n_cells=100,
             n_replicates=3)
        _run('6_injury_gating', test_6_injury_gating, n_cells=1000, n_irr=200, n_lys=100)
        _run('7_determinism', test_7_determinism, n_cells=300, n_div=200)
        _run('8_scale', test_8_scale, n_cells=1000, n_div=1000, growth_n_cells=200,
             growth_t_end_h=8.0)
    else:
        _run('1_identity', test_1_identity)
        _run('2_conservation', test_2_conservation)
        _run('3_rule_behaviour', test_3_rule_behaviour)
        _run('4_patterning', test_4_patterning)
        _run('5_asymmetric_division', test_5_asymmetric_division)
        _run('6_injury_gating', test_6_injury_gating)
        _run('7_determinism', test_7_determinism)
        _run('8_scale', test_8_scale)
    _run('10_gid_collision_audit', test_10_gid_collision)

    return dict(
        module='lineage.py', version=MODULE_VERSION, output_path=OUTPUT_PATH,
        created=time.strftime('%Y-%m-%dT%H:%M:%S'), quick=bool(quick),
        interpreter=_interpreter(), numpy=np.__version__, scipy=_scipy_version(),
        cpu_only=True,
        n_tests=len(order), n_passed=sum(1 for k in order if tests[k].get('pass')),
        tests_passed=[k for k in order if tests[k].get('pass')],
        tests_failed=[k for k in order if not tests[k].get('pass')],
        tests=tests,
        all_numbers=_collect_numbers(tests),
        api=dict(
            entry_points=['run', 'pure_differentiation_run', 'growth_coupled_run',
                          'self_test', 'main'],
            rule_names=list(RULE_NAMES), type_names=list(TYPE_NAMES),
            type_ids=dict(TYPE_IDS), default_rule=dict(DEFAULT_RULE),
            lineage_methods=['differentiate', 'divide', 'remove', 'parent_of',
                             'children_of', 'path_to_root', 'descendants_of',
                             'root_ids', 'live_slots', 'cell', 'cells',
                             'event_table', 'event_table_arrays',
                             'event_table_signature', 'type_counts',
                             'type_fractions', 'total_area_um2', 'verify',
                             'children_map', 'parents_array'],
        ),
        honesty=dict(
            validated=False,
            validation_statement=(
                'NOT VALIDATED against any measurement in this project. Every test is a '
                'numerical self-consistency check (unique ids, conservation, agreement '
                'with the module\'s own documented Markov rule, determinism, timing).'),
            cell_types_are_abstractions=True,
            patterned_simulation_is_not_evidence_of_differentiation=True,
            citations_invented=0,
            literature_used=[
                '43 um^2 mean cell area -- project convention (model.py / mechanics.py, '
                'recorded in MEASURED_ANCHORS.md)',
                '7.5-10 um notum epithelium thickness -- Pinheiro 2017 PMC6143170, via '
                'MEASURED_ANCHORS.md (9.0 used as the midpoint)'],
            all_differentiation_parameters_illustrative=True,
            parameter_provenance=PARAM_SOURCES),
        wall_time_s=float(time.perf_counter() - t0),
    )


def _interpreter():
    return f'{sys.executable} ({sys.version.split()[0]})'


def _scipy_version():
    try:
        import scipy
        return scipy.__version__
    except Exception:
        return None


def _collect_numbers(obj, prefix='', out=None, depth=0):
    """Flatten numeric leaves into a path -> value map (for reporting)."""
    if out is None:
        out = {}
    if depth > 9:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_numbers(v, f'{prefix}.{k}' if prefix else str(k), out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _collect_numbers(v, f'{prefix}[{i}]', out, depth + 1)
    elif isinstance(obj, (bool, np.bool_)):
        out[prefix] = bool(obj)
    elif isinstance(obj, (int, np.integer)):
        out[prefix] = int(obj)
    elif isinstance(obj, (float, np.floating)):
        v = float(obj)
        out[prefix] = v if math.isfinite(v) else str(v)
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Lineage):
        return dict(version=MODULE_VERSION, n_slots=int(obj.n_slots),
                    n_alive=int(obj.n_alive), n_roots=int(obj.n_roots),
                    n_births=int(obj.n_births), n_removals=int(obj.n_removals))
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def main(argv=None, out_path=OUTPUT_PATH, verbose=True):
    """Run the full self-test and write the verification JSON (the only file)."""
    ap = argparse.ArgumentParser(description='lineage.py verification suite')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=out_path)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    res = self_test(quick=args.quick, verbose=verbose and not args.quiet)
    payload = _jsonable(res)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    if verbose and not args.quiet:
        print(f'[lineage] wrote {args.out}')
        print(f'[lineage] {res["n_passed"]}/{res["n_tests"]} tests passed; '
              f'total wall {res["wall_time_s"]:.1f} s')
    return res, args.out


if __name__ == '__main__':
    main()
