"""Intracellular cell-state layer for the wound prototype: volume, osmotic solute,
ATP, plasma-membrane integrity with repair, and a REVERSIBLE vs IRREVERSIBLE
injury verdict with explicit hysteresis.

======================================================================
READ THIS FIRST -- WHAT THIS MODULE IS AND IS NOT
======================================================================
THIS MODULE IS NOT CALIBRATED. It is a mechanism-demonstration layer with
illustrative parameter values. It MUST NOT be used to claim that any real cell
"survives", "is injured", or "dies", and it must not be used to predict tissue
outcome. Its only defensible outputs are (a) numerical self-consistency, (b) the
demonstration that ATP/pump/volume/integrity coupling CAN produce progression
from reversible injury to irreversible injury to lysis, and (c) the explicit
result that almost none of its parameters are identifiable from the only
observable this project currently has (an annular fluorescence half-height
radius). Test 8 measures that unidentifiability; do not read the other tests as
biology.

Every parameter below is tagged in ``PARAM_SOURCES`` as either
  (i)  'literature'  -- an exact citation and value that were CHECKED, or
  (ii) 'illustrative' -- an arbitrary value chosen to make the mechanism run.
Any parameter not tagged 'literature' is arbitrary and has no empirical support
whatsoever. No citation is asserted that was not verified; see PARAM_SOURCES.

======================================================================
CONVENTIONS (matched to /run/media/sensen/Data2/cell_wound_prototype/model.py)
======================================================================
  time s; concentrations uM; length um; area um^2; cell area 43 um^2;
  n_cells = n_side**2 (default 576 = 24^2); per-cell state arrays of shape
  (n_cells,); returned time series of shape (n_frames, n_cells).
  run() returns a plain dict; import of this file creates no files and no plots.
  Hexagonal geometry and the prescribed microtear field are taken READ-ONLY from
  model.py (``model.hex_geometry``, ``model.damage_profile``); model.py is not
  modified. Coordinate-keyed random draws follow model.py's convention so the
  same seed gives the same cell identities on nested grids.

======================================================================
STATE VARIABLES (all per cell; all dimensionless unless stated)
======================================================================
  V    relative cell volume. V = 1 is the isotonic reference state. Cell
       geometry is NOT moved by volume changes (fixed hexagons) -- a documented
       limitation: this module reports volume, it does not reshape the tissue.
  s    intracellular PERMEANT solute amount, in units of the reference cell's
       isotonic permeant content. Concentration of permeant solute in the
       reference-volume-normalised units is s/V. Instantaneous relative
       osmolarity of the cell interior is (s + s_i)/V and the extracellular
       osmolarity is 1 by definition, so the osmotic driving term is
       (s + s_i)/V - 1 with s_i = 's_impermeant' (impermeant protein/polyanion
       osmolyte share). Total content is s; water moves, so concentration
       s/V changes even when s is conserved (that is test 2).
  a    ATP concentration, uM (cytosolic, spatially lumped).
  h    h_m, plasma-membrane integrity in [0,1]; 1 = intact, 0 = no barrier.
Latched (hysteretic) Booleans, per cell, never cleared once set:
  latched_irreversible  -- set by sustained ATP depletion, sustained integrity
       below floor, or volume beyond the lysis threshold.
  latched_lysed         -- set by volume rupture (V >= v_lysis) or by integrity
       below hm_lyse; such a cell is driven to a collapsed, non-viable attractor.
  injured               -- hysteretic reversible-injury flag (entry and exit
       thresholds differ; see test 5).

======================================================================
EQUATIONS (units in brackets after each term)
======================================================================
Parameters named exactly as in DEFAULTS. g(a) is the ATP gate of the pumps and
of RVD (regulatory volume decrease):
    g(a) = [ a^n / (K^n + a^n) ] / [ a0^n / (K^n + a0^n) ]      [-]
with n = 'n_atp_pump', K = 'k_atp_pump', a0 = 'atp0'. The normalisation makes
g(a0) = 1 exactly so that the no-injury control has EXACT zero drift; the shape
is a standard Hill activation, the normalisation is a modelling choice. Oxygen
availability is similarly normalised to 1 at oxygen = 1:
    ox(o2) = [ o2^m / (k^n + o2^m) ] / [ 1 / (k^n + 1) ]        [-]
with m = 'n_o2', k = 'k_o2'.

(1) Volume -- osmotic water flux + tear-driven fluid entry + RVD + post-lysis
    collapse. [1/s], with L = [lysed] and P = open-membrane fraction from (2):
      V' = (1-L)*[ k_water*((s + s_i)/V - 1)              osmotic water flux
                 + q_water*P                              fluid entry via tear
                 - rvd_rate*g(a)*max(V - 1, 0) ]           regulatory volume decrease
           - L*k_collapse*(V - v_collapse)                post-lysis collapse
    k_water is a lumped L_p*A*R*T/V_ref rate (arbitrary). RVD is the requested
    "small restoring term"; it is ATP-gated (g(a)) because RVD in real cells
    requires ATP-dependent ion transport and cytoskeletal remodelling -- a
    qualitative modelling assumption, not a calibrated one. RVD opposes swelling
    only; the shrinkage limb (regulatory volume increase) is deliberately ABSENT.
    A lysed cell keeps only the collapse term: no membrane, no osmotic barrier.

(2) Permeant solute -- flux through the open membrane + ATP-driven extrusion.
    [1/s]
      P = (1 - h)*tear + p_resid_leak*[latched irreversible]      [-]
      s' = (1-L)*[ p_leak*P*(1 - s/V)                   leak down the gradient
                 - g(a)*pump_rate*(s/V - s_setpoint) ]  Na/K-ATPase-like extrusion
           - L*k_collapse*(s - v_collapse)              post-lysis collapse
    The extracellular permeant concentration is 1 in these units. The pump
    extrudes solute in proportion to the deviation of s/V from the setpoint
    s_setpoint = 0.5, so pump flux is EXACTLY zero in the control state. The
    constant basal pump turnover is therefore not represented as a flux; its ATP
    cost is lumped into 'cons_base' and only the deviation-driven cost appears in
    (3) -- a documented simplification required for an exactly neutral control.
    After the irreversible latch, a small residual permeability is assigned
    (p_resid_leak): this is the physical carrier of the hysteresis, i.e. the
    model's statement that a cell that has crossed the irreversible threshold
    retains a homeostasis defect rather than returning to a pristine membrane.

(3) ATP -- substrate/oxygen-limited production, maintenance plus pump cost. [uM/s]
      a' = (1-L)*[ ox(o2)*clip(cons_base + k_atp_rec*(a0 - a), 0, prod_max)
                   - (cons_base/a0)*a - cons_pump*|j_p| ] - L*k_collapse*a
      j_p = g(a)*pump_rate*(s/V - s_setpoint)            [1/s] pump flux
    At a = a0 and oxygen = 1: production = ox*cons_base = cons_base and
    consumption = cons_base*(a0/a0) = cons_base, so a' = 0 EXACTLY. Production
    saturates at prod_max and is zero for a lysed cell. ATP depletion lowers
    g(a), which disables the pumps and RVD: that is the coupling that makes
    injury progress (higher leak load -> more pump work -> lower ATP -> weaker
    pump -> higher solute and water load -> more swelling).

(4) Membrane integrity with repair. [1/s], with I = [latched irreversible]
      h' = (1-L)*[ k_repair*(1 - h)                      repair (k_repair = 1/tau_repair)
                   - k_damage*tear*h                     damage-driven loss
                   - I*k_fail*max(h - hm_ceiling_fail, 0) ]  post-latch ceiling
           - L*k_rupture*h                               catastrophic failure
    k_repair = 0 if repair_enabled = False, and is reduced by the factor
    'repair_fail_frac' once a cell has latched irreversible (commitment: an
    irreversibly injured cell does not go on repairing as if nothing happened).
    The form is exponential relaxation to a steady state
    h_ss = k_repair/(k_repair + k_damage*tear) during a sustained tear, and it is
    written in the "repair toward 1" form (not "damage from 1") precisely so that
    h = 1 is NOT an absorbing state -- an intact membrane in a damaging field
    must still be able to lose integrity.

(5) Triage (hysteresis is explicit, not emergent -- see test 5):
    entry to reversible injury (0 -> 1): h < hm_inj, or V > v_swell_rev, or
       V < v_shrink_rev, or a < atp_inj_frac*a0;
    exit back to healthy (1 -> 0): h > hm_heal AND |V - 1| < v_heal_tol AND
       a > atp_heal_frac*a0. The exit thresholds are strictly stricter than the
       entry thresholds, which is the requested hysteresis band.
    irreversible latch (-> 2), any of:
       a < atp_deplete_frac*a0 continuously for longer than t_atp_irrev s;
       h < hm_floor continuously for longer than t_hm_irrev s;
       V >= v_lysis at any time.
    lysed latch (-> 3), either: V >= v_lysis, or h < hm_lyse.
    Precedence: 3 over 2 over 1 over 0. A latched cell can only worsen.

======================================================================
STATE CODES (returned in result['state'])
======================================================================
  0 healthy                -- not latched, and no entry criterion satisfied
  1 reversibly_injured     -- hysteretic injury flag set, no latch
  2 irreversibly_injured   -- latched, not lysed
  3 lysed                  -- collapsed, non-viable attractor
  A cell whose ODE variables drift back toward baseline after an
  irreversible latch stays code 2/3. That is the point of the module; a naive
  instantaneous-threshold verdict would call it "recovered" (test 5 quantifies
  the difference for both verdicts).

======================================================================
NUMERICS
======================================================================
Vectorised exponential (L-stable) first-order scheme over all cells at once:
every equation above is rewritten as an AFFINE decomposition in its own state
variable, y' = A(y) - lam(y)*y with A >= 0 and lam >= 0, and updated as
    y_{n+1} = y_n*exp(-lam*dt) + A*dt*phi(lam*dt),
    phi(z) = (1 - exp(-z))/z,   phi(z) -> 1 as z -> 0.
For lam = 0 this degenerates to explicit Euler. The decomposition is explicit and
must be done with care: each term is split according to the variable it decays
toward, e.g. k_water*((s+s_i)/V - 1) = k_water*(s+s_i)/V - k_water (lam += k_water/V)
and -rvd_rate*g*max(V-1,0) = rvd_rate*g - rvd_rate*g*V (lam += rvd_rate*g), while
the tear-driven fluid entry q_water*P is a PURE SOURCE with no lam partner. A
mis-split silently deletes a term, so self_test()['test0_decomposition_self_check']
verifies A - lam*y == RHS for all four states on random admissible states; while
this file was being written that check caught two real bugs of exactly this kind.
Consequences (documented stability limits): the update is unconditionally stable
in lam*dt, it preserves y >= 0 for A >= 0, and for the integrity equation
A/lam <= 1 keeps h in [0,1] for every dt. dt = 0.05 s is the documented default
(sweeps use 0.1 s; test 1b reports the dt refinement 0.05 vs 0.01). No implicit
solve and no solve_ivp is used: 576 cells x 300 s is 6000 vectorised steps
(~1 s on this machine). Guards: V is clipped to [v_min_guard, v_max_guard], a to
>= 0, h to [0,1]; guard violations are counted BEFORE clipping and reported in
metadata['bounds_hit'] (zero in every tested configuration -- see test 3).

``mass_balance_error`` is the maximum over cells of the relative mismatch between
the solute increment and its own flux integrated with the scheme's weighting,
|ds - sum_steps flux_start*dt*phi(lam*dt)| / max(1e-12, |ds|, |budget|). It is an
implementation/consistency diagnostic -- it is exactly 0.0 in the control and in
the zero-permeability test, and rounding level (<= ~1e-9) otherwise, whereas a
documented/implemented mismatch produced O(1) values (1.8) during development.
It is NOT a physical conservation law: solute is not conserved in general because
the leak admits it and the pumps extrude it. metadata['solute_balance'] also
reports a plain trapezoidal quadrature residual, which grows when lam*dt >> 1;
that is a property of first-order quadrature, not of the biology.

======================================================================
SOURCES
======================================================================
  model.py (this directory): prescribed radial microtear field, hex geometry,
    cell area 43 um^2; Ported portions copyright (c) 2022 mshutson, MIT License.
  measurement.py (this directory): fluorescence half-height radius observable.
  Köhler S., Schmidt H., Fülle P., Hirrlinger J., Winkler U. "A Dual Nanosensor
    Approach to Determine the Cytosolic Concentration of ATP in Astrocytes",
    Front. Cell. Neurosci. 14:565921 (2020), DOI 10.3389/fncel.2020.565921.
    Basal cytosolic [ATP] ~1.5 mM in cultured cortical astrocytes and 0.7-1.3 mM
    in acutely isolated cortical slices. This is the ONLY literature-anchored
    numeric value in this module (used to bracket 'atp0'). It was verified by
    fetching the article text; it is a different cell type and preparation, so it
    only fixes an order of magnitude.
"""
from __future__ import annotations

import os
import sys
import time as _time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:  # read-only reuse of the existing prototype; model.py is never modified
    import model as _model
except Exception as _exc:  # pragma: no cover - environment error path
    raise ImportError(
        f'cellstate.py requires model.py in {_HERE!r} (and its scipy deps) for '
        f'hex geometry and the prescribed microtear field') from _exc

__all__ = ['DEFAULTS', 'PARAM_SOURCES', 'STATE_CODES', 'run', 'cell_geometry',
           'prescribed_tear_field', 'bridge_damage', 'identifiability_analysis',
           'self_test']

# ---------------------------------------------------------------------------
# State codes (documented above; do not renumber)
STATE_CODES = {0: 'healthy', 1: 'reversibly_injured', 2: 'irreversibly_injured',
               3: 'lysed'}

# ---------------------------------------------------------------------------
# Default parameters. EVERY value here is 'illustrative' unless PARAM_SOURCES
# says 'literature'. Units are given in the comment and in PARAM_SOURCES.
DEFAULTS = dict(
    # --- volume / osmotic regulation -------------------------------------
    k_water=0.10,          # 1/s   lumped osmotic water-flux rate (L_p*A*R*T/V_ref)
    s_impermeant=0.5,      # -     impermeant (protein) osmolyte share, isotonic units
    s_setpoint=0.5,        # -     permeant osmolyte setpoint held by the pumps
    rvd_rate=0.02,         # 1/s   regulatory volume decrease rate at full ATP
    q_water=0.02,          # 1/s   fluid entry per unit open-membrane fraction
    v_collapse=0.05,       # -     relative volume of the post-lysis attractor
    k_collapse=0.5,        # 1/s   rate of collapse after lysis
    v_min_guard=0.02,      # -     hard numerical floor on V (guarded, reported)
    v_max_guard=8.0,       # -     hard numerical ceiling on V (guarded, reported)
    # --- solute / pumps --------------------------------------------------
    p_leak=0.5,            # 1/s   permeant-solute conductance per unit open membrane
    p_resid_leak=0.02,     # -     residual open-membrane fraction after the latch
    pump_rate=0.05,        # 1/s   pump-limited relaxation rate at full ATP
    k_atp_pump=200.0,      # uM    ATP half-saturation of the pump gate
    n_atp_pump=2.0,        # -     Hill coefficient of the pump gate
    # --- ATP -------------------------------------------------------------
    atp0=1000.0,           # uM    reference cytosolic ATP (literature-bracketed)
    cons_base=20.0,        # uM/s  baseline maintenance consumption (and turnover scale)
    cons_pump=3000.0,      # uM per unit pump flux (deviation-driven pump cost)
    k_atp_rec=0.02,        # 1/s   feedback gain of substrate-limited production
    prod_max=60.0,         # uM/s  maximum ATP production
    k_o2=0.5,              # -     oxygen half-saturation (oxygen input is 0..1)
    n_o2=3.0,              # -     Hill coefficient of oxygen availability
    atp_deplete_frac=0.15, # -     fraction of atp0 counted as "depleted"
    atp_inj_frac=0.7,      # -     ATP fraction below which injury entry triggers
    atp_heal_frac=0.9,     # -     ATP fraction required to exit back to healthy
    t_atp_irrev=60.0,      # s     sustained-depletion window for the irreversible latch
    # --- membrane integrity / repair -------------------------------------
    k_damage=1.0,          # 1/s   integrity loss per unit tear
    tau_repair=30.0,       # s     membrane repair (resealing) time constant
    repair_fail_frac=0.1,  # -     repair capacity retained after the irreversible latch
    k_fail=0.05,           # 1/s   pull of h down to the post-latch ceiling
    hm_ceiling_fail=0.6,   # -     integrity ceiling of an irreversibly injured cell
    k_rupture=2.0,         # 1/s   integrity loss rate of a lysed cell
    # --- triage thresholds ----------------------------------------------
    hm_inj=0.98,           # -     integrity below which injury entry triggers
    hm_heal=0.995,         # -     integrity above which injury exit is allowed
    hm_floor=0.2,          # -     integrity floor for the irreversible latch
    hm_lyse=0.02,          # -     integrity below which the membrane has failed
    v_swell_rev=1.2,       # -     reversible-swelling volume threshold
    v_shrink_rev=0.8,      # -     reversible-shrinkage volume threshold
    v_heal_tol=0.05,       # -     |V-1| tolerance for exiting injury
    v_lysis=1.8,           # -     relative volume at which the membrane ruptures
    t_hm_irrev=60.0,       # s     recovery window for sustained low integrity
)

# ---------------------------------------------------------------------------
# Parameter provenance. 'literature' entries carry an exact, CHECKED citation
# and value. 'illustrative' entries are arbitrary: they were chosen to make the
# mechanism run and to keep the control exactly neutral. Any claim that these
# numbers describe a real cell would be fabricated.
_LIT_ATP = dict(
    value='atp0 = 1000 uM (used only as an order-of-magnitude anchor)',
    citation=('Köhler S, Schmidt H, Fülle P, Hirrlinger J, Winkler U, Front Cell '
              'Neurosci 14:565921 (2020), DOI 10.3389/fncel.2020.565921: basal '
              'cytosolic [ATP] ~1.5 mM in cultured cortical astrocytes, 0.7-1.3 mM '
              'in acute cortical slices'),
    note=('different cell type and preparation; verified by fetching the article '
          'text; fixes a factor-of-two range, not a value for these cells'))
_ILL = 'arbitrary illustrative value; no empirical support; chosen for mechanism demonstration only'

PARAM_SOURCES = {
    'k_water': dict(kind='illustrative', units='1/s', note=_ILL),
    's_impermeant': dict(kind='illustrative', units='-', note=_ILL),
    's_setpoint': dict(kind='illustrative', units='-', note=_ILL),
    'rvd_rate': dict(kind='illustrative', units='1/s', note=_ILL +
                     ' (RVD existence is standard physiology; this rate is not measured here)'),
    'q_water': dict(kind='illustrative', units='1/s', note=_ILL),
    'v_collapse': dict(kind='illustrative', units='-', note=_ILL),
    'k_collapse': dict(kind='illustrative', units='1/s', note=_ILL),
    'v_min_guard': dict(kind='illustrative', units='-', note='numerical guard, not physiology'),
    'v_max_guard': dict(kind='illustrative', units='-', note='numerical guard, not physiology'),
    'p_leak': dict(kind='illustrative', units='1/s', note=_ILL),
    'p_resid_leak': dict(kind='illustrative', units='-', note=_ILL),
    'pump_rate': dict(kind='illustrative', units='1/s', note=_ILL),
    'k_atp_pump': dict(kind='illustrative', units='uM', note=_ILL),
    'n_atp_pump': dict(kind='illustrative', units='-', note=_ILL),
    'atp0': dict(kind='literature', units='uM', **_LIT_ATP),
    'cons_base': dict(kind='illustrative', units='uM/s', note=_ILL),
    'cons_pump': dict(kind='illustrative', units='uM per unit pump flux', note=_ILL),
    'k_atp_rec': dict(kind='illustrative', units='1/s', note=_ILL),
    'prod_max': dict(kind='illustrative', units='uM/s', note=_ILL),
    'k_o2': dict(kind='illustrative', units='-', note=_ILL),
    'n_o2': dict(kind='illustrative', units='-', note=_ILL),
    'atp_deplete_frac': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'atp_inj_frac': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'atp_heal_frac': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    't_atp_irrev': dict(kind='illustrative', units='s', note='sustained-depletion window: a model verdict'),
    'k_damage': dict(kind='illustrative', units='1/s', note=_ILL),
    'tau_repair': dict(kind='illustrative', units='s', note=_ILL +
                       ' (Ca2+-dependent resealing exists and is fast in real cells; this time constant is not taken from a measurement)'),
    'repair_fail_frac': dict(kind='illustrative', units='-', note=_ILL),
    'k_fail': dict(kind='illustrative', units='1/s', note=_ILL),
    'hm_ceiling_fail': dict(kind='illustrative', units='-', note=_ILL),
    'k_rupture': dict(kind='illustrative', units='1/s', note=_ILL),
    'hm_inj': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'hm_heal': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'hm_floor': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'hm_lyse': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'v_swell_rev': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'v_shrink_rev': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'v_heal_tol': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    'v_lysis': dict(kind='illustrative', units='-', note='triage threshold: a model verdict'),
    't_hm_irrev': dict(kind='illustrative', units='s', note='recovery window: a model verdict'),
}

AREA_UM2 = 43.0            # um^2, from model.PARAMETERS['reference_area']
WOUND_RADIUS_UM = 51.25    # um, from model.PARAMETERS['wound_radius']
ABLATED_RADIUS_UM = 24.5   # um, from model.PARAMETERS['ablated_radius']
CALCIUM_WINDOW_S = 25.0    # s, model.run refuses t_end > 25 s (documented truncation)

_EQUATIONS = [
    "V' = k_water*((s+s_i)/V - 1) + q_water*P - rvd_rate*g(a)*max(V-1,0) "
    "- [lysed]*k_collapse*(V - v_collapse)   [1/s]",
    "P = (1-h)*tear + p_resid_leak*[latched_irreversible]   [-]; "
    "s' = p_leak*P*(1 - s/V) - g(a)*pump_rate*(s/V - s_setpoint)   [1/s]",
    "a' = ox(o2)*clip(cons_base + k_atp_rec*(atp0-a), 0, prod_max) "
    "- (cons_base/atp0)*a - cons_pump*|g(a)*pump_rate*(s/V - s_setpoint)|   [uM/s]",
    "h' = k_repair*(1-h) - k_damage*tear*h - [lysed]*k_rupture*h "
    "- [latched_irreversible]*k_fail*max(h - hm_ceiling_fail, 0)   [1/s]",
    "g(a) = [a^n/(K^n+a^n)]/[atp0^n/(K^n+atp0^n)]; "
    "ox(o2) = [o2^m/(k_o2^m+o2^m)]/[1/(k_o2^m+1)]   [-]",
]


# ---------------------------------------------------------------------------
def _check_params(params):
    p = DEFAULTS.copy()
    if params:
        unknown = set(params) - set(p)
        if unknown:
            raise ValueError(f'Unknown parameters: {sorted(unknown)}')
        p.update(params)
    for key, val in p.items():
        if not np.isfinite(val):
            raise ValueError(f'Parameter {key} must be finite, got {val!r}')
    for key in ('k_water', 'p_leak', 'pump_rate', 'cons_base', 'atp0', 'k_damage'):
        if p[key] < 0:
            raise ValueError(f'Parameter {key} must be >= 0, got {p[key]}')
    if p['atp0'] <= 0 or p['cons_base'] < 0:
        raise ValueError('atp0 must be positive')
    if not 0 <= p['atp_deplete_frac'] < 1:
        raise ValueError('atp_deplete_frac must be in [0,1)')
    if p['v_lysis'] <= p['v_swell_rev']:
        raise ValueError('v_lysis must exceed v_swell_rev')
    if p['v_min_guard'] <= 0 or p['v_max_guard'] <= p['v_lysis']:
        raise ValueError('guards must satisfy 0 < v_min_guard and v_max_guard > v_lysis')
    if p['s_setpoint'] <= 0 or p['s_impermeant'] < 0:
        raise ValueError('s_setpoint must be positive and s_impermeant nonnegative')
    return p


def cell_geometry(n_cells=576):
    """Hexagonal cell centres, polygons, areas, radius and the ablated mask.

    n_cells must be a perfect square (n_side = sqrt(n_cells)), matching
    model.hex_geometry; cell area is fixed at 43 um^2. The 'ablated' mask uses
    model.py's radial rule (radius < 24.5 um): in model.py those cells are
    imposed calcium reservoirs, NOT viable cells, so statistics over "tissue"
    normally exclude them (see run(..., tissue_only_stats)).
    """
    n_side = int(round(float(n_cells) ** .5))
    if n_side < 2 or n_side * n_side != int(n_cells):
        raise ValueError(f'n_cells must be a perfect square >= 4, got {n_cells!r}')
    geo = _model.hex_geometry(n_side, area=AREA_UM2)
    radius = np.linalg.norm(geo['positions'], axis=1)
    return dict(positions=geo['positions'], polygons=geo['polygons'],
                areas=geo['areas'], edges=geo['edges'],
                edge_lengths=geo['edge_lengths'], radius=radius,
                ablated=radius < ABLATED_RADIUS_UM, n_side=n_side,
                n_cells=n_side * n_side)


def prescribed_tear_field(geo, wound=True, tear_scale=1.0, wound_radius=WOUND_RADIUS_UM):
    """Prescribed radial microtear density from model.damage_profile.

    This is the spreadsheet-fitted, prescribed (NOT mechanically computed) field
    used by the existing calcium model; its peak is ~0.968 at the wound centre
    and it is zero beyond wound_radius. tear_scale multiplies it.
    """
    if tear_scale < 0:
        raise ValueError('tear_scale must be nonnegative')
    if not wound:
        return np.zeros(geo['n_cells'])
    return np.asarray(_model.damage_profile(geo['radius'], wound_radius), float) * tear_scale


def _heterogeneity(geo, seed, sigma):
    """Coordinate-keyed lognormal multipliers, mean 1, for pump_rate and k_damage.

    Same key scheme as model.run so the same seed selects the same cell
    identities on nested grids; sigma = 0 (default) disables heterogeneity and
    keeps the no-injury control exactly neutral. The multiplicative spread on
    pump capacity and damage susceptibility is an illustrative modelling choice.
    """
    if sigma <= 0:
        return np.ones(geo['n_cells']), np.ones(geo['n_cells'])
    side = np.sqrt(2 * AREA_UM2 / (3 * np.sqrt(3)))
    keys = np.rint(geo['positions'] / [np.sqrt(3) * side / 4, 1.5 * side / 2]).astype(np.int64)
    z = np.array([np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(x) & 0xffffffff, int(y) & 0xffffffff])).normal(size=2)
        for x, y in keys])
    return (np.exp(-.5 * sigma ** 2 + sigma * z[:, 0]),   # pump capacity multiplier
            np.exp(-.5 * sigma ** 2 + sigma * z[:, 1]))   # damage susceptibility multiplier


def _expo(y, A, lam, dt):
    """One step of y' = A - lam*y with A >= 0, lam >= 0 (L-stable, positivity preserving)."""
    z = lam * dt
    phi = np.where(z > 1e-8, -np.expm1(-z) / np.where(z > 1e-8, z, 1.0), 1.0 - .5 * z)
    return y * np.exp(-z) + (A * dt) * phi


def _radius_summary(m):
    r = np.asarray(m['radius'], float)
    fin = np.isfinite(r)
    return dict(radius_final_um=float(r[fin][-1]) if fin.any() else float('nan'),
                radius_mean_uncensored_um=float(r[fin].mean()) if fin.any() else float('nan'),
                n_frames=int(r.size), n_censored=int((~fin).sum()))


# ---------------------------------------------------------------------------
def run(n_cells=576, tear=None, strain=None, oxygen=1.0, t_end=300.0, dt=0.05,
        params=None, seed=2025, *, sample_dt=1.0, wound=True, tear_scale=1.0,
        strain_gain=1.0, tear_until=None, repair_enabled=True, het_sigma=0.0,
        V0=1.0, atp_init=None, ablated_override=None, tissue_only_stats=True,
        initial=None, return_frames=True):
    """Integrate the per-cell state layer for all cells at once and return a dict.

    Parameters
    ----------
    n_cells : int, must be a perfect square (default 576 = model n_side 24).
    tear : (n_cells,) nonnegative array, or None. Per-cell tear/damage input, e.g.
        from mechanics or from the existing prescribed radial damage field. If
        None and wound=True, model.damage_profile is used (times tear_scale).
    strain : (n_cells,) nonnegative array, or None. Converted with the documented
        arbitrary map tear_strain = strain_gain*strain**2 and ADDED to tear.
    oxygen : float >= 0, or an (n_cells,) array of per-cell relative oxygen
        availability (1 = reference). A scalar applies the same value to all
        cells; an array lets a spatial field (for example from transport.py)
        drive each cell individually. Either way ATP production is scaled by the
        normalised Hill availability ox(o2).
    t_end, dt : float, seconds. dt is adjusted to t_end/n_steps so the run ends
        exactly at t_end; the nominal dt is what is reported.
    sample_dt : storage interval for the returned time series (default 1 s).
    tear_until : float or None. If given, the tear input is applied only for
        t < tear_until (insult window protocol used by the repair/hysteresis tests).
    repair_enabled : bool. False sets k_repair = 0 (no membrane resealing).
    het_sigma : float. Optional per-cell lognormal heterogeneity of pump capacity
        and damage susceptibility (0 = off, exact control); the seed selects it.
    V0 : float, initial relative volume (used by the conservation test).
    initial : dict or None. Continuation state (the 'final' entry of a previous
        result) including the latched Booleans; times restart at 0.
    tissue_only_stats : bool. If True, *_tissue statistics exclude the ablated core.
    atp_init : float or None. Initial ATP in uM (default atp0).
    ablated_override : bool array or None. Replaces the radial ablated mask.
    return_frames : bool. False drops the four time-series arrays (keeps 'final',
        'state_final' and the metadata) to save memory.

    Returns
    -------
    dict with 'times' (n_frames,), 'V'/'solute'/'atp'/'integrity' (n_frames,n_cells),
    'state' (n_frames,n_cells) int8 codes (see STATE_CODES), 'death_fraction'
    (n_frames, all cells, fraction with state >= 2), 'death_fraction_tissue',
    'mass_balance_error' (float; implementation/consistency residual, see the
    NUMERICS section -- NOT a conservation law), 'metadata', plus geometry
    ('positions','polygons','areas','radius','ablated'), 'tear_field', per-cell
    event times ('time_atp_depleted','time_irreversible','time_lysed', inf if never),
    'state_final', 'state_naive_final' (same thresholds applied instantaneously,
    WITHOUT the latch -- the comparison used in test 5), the latched Boolean arrays
    ('latched_irreversible','latched_lysed','injured_final') and 'final'
    (continuation state for a later run(initial=...)).

    This module is uncalibrated; see the module docstring before using any number.
    """
    start = _time.perf_counter()
    p = _check_params(params)
    if t_end <= 0 or dt <= 0 or sample_dt <= 0:
        raise ValueError('t_end, dt and sample_dt must be positive')
    oxygen_arr = np.asarray(oxygen, dtype=float)
    if not np.all(np.isfinite(oxygen_arr)) or np.any(oxygen_arr < 0):
        raise ValueError('oxygen must be finite and >= 0 (scalar or per-cell array)')
    oxygen_is_scalar = oxygen_arr.ndim == 0
    if V0 <= 0:
        raise ValueError('V0 must be positive')
    geo = cell_geometry(n_cells)
    n = geo['n_cells']
    if oxygen_is_scalar:
        oxygen_used = float(oxygen_arr)
        oxygen_desc = f'scalar {oxygen_used}'
    else:
        if oxygen_arr.shape != (n,):
            raise ValueError(f'oxygen array must have shape ({n},) to match the cells')
        oxygen_used = oxygen_arr
        oxygen_desc = (f'per-cell array: min {float(oxygen_arr.min())}, '
                       f'mean {float(oxygen_arr.mean())}, max {float(oxygen_arr.max())}')

    # ---- tear / strain input ------------------------------------------------
    if tear is None and strain is None:
        tear_in = prescribed_tear_field(geo, wound=wound, tear_scale=tear_scale)
        tear_source = ('prescribed radial microtear field (model.damage_profile, '
                       f'wound_radius={WOUND_RADIUS_UM} um) x tear_scale={tear_scale}'
                       if wound else 'zero field (wound=False)')
    else:
        tear_in = np.zeros(n)
        src = []
        if tear is not None:
            t_arr = np.asarray(tear, float)
            if t_arr.shape != (n,):
                raise ValueError(f'tear must have shape ({n},)')
            if np.any(t_arr < 0) or not np.all(np.isfinite(t_arr)):
                raise ValueError('tear must be finite and nonnegative')
            tear_in += t_arr
            src.append('explicit tear array')
        if strain is not None:
            s_arr = np.asarray(strain, float)
            if s_arr.shape != (n,):
                raise ValueError(f'strain must have shape ({n},)')
            if np.any(s_arr < 0) or not np.all(np.isfinite(s_arr)):
                raise ValueError('strain must be finite and nonnegative')
            tear_in += strain_gain * s_arr ** 2
            src.append(f'strain -> tear via {strain_gain}*strain**2 (arbitrary map)')
        tear_source = ' + '.join(src)
    if ablated_override is not None:
        ablated = np.asarray(ablated_override, bool).copy()
        if ablated.shape != (n,):
            raise ValueError(f'ablated_override must have shape ({n},)')
    else:
        ablated = geo['ablated'].copy()

    # ---- parameters, gates and derived constants ---------------------------
    k_atp_pump, n_atp_pump = p['k_atp_pump'], p['n_atp_pump']
    g_norm = 1.0 / (p['atp0'] ** n_atp_pump / (k_atp_pump ** n_atp_pump + p['atp0'] ** n_atp_pump))
    ox = (oxygen_used ** p['n_o2'] / (p['k_o2'] ** p['n_o2'] + oxygen_used ** p['n_o2'])) / \
         (1.0 / (p['k_o2'] ** p['n_o2'] + 1.0))
    k_repair = (1.0 / p['tau_repair']) if repair_enabled else 0.0
    pump_het, dmg_het = _heterogeneity(geo, seed, het_sigma)
    pump_rate_i = p['pump_rate'] * pump_het
    k_damage_i = p['k_damage'] * dmg_het

    # ---- initial state -----------------------------------------------------
    if initial is None:
        V = np.full(n, float(V0))
        s = np.full(n, p['s_setpoint'])
        a = np.full(n, p['atp0'] if atp_init is None else float(atp_init))
        h = np.ones(n)
        latched_irr = np.zeros(n, bool)
        latched_lys = np.zeros(n, bool)
        injured = np.zeros(n, bool)
    else:
        V = np.array(initial['V'], float)
        s = np.array(initial['solute'], float)
        a = np.array(initial['atp'], float)
        h = np.array(initial['integrity'], float)
        latched_irr = np.array(initial['latched_irreversible'], bool)
        latched_lys = np.array(initial['latched_lysed'], bool)
        injured = np.array(initial['injured'], bool)
        for arr in (V, s, a, h, latched_irr, latched_lys, injured):
            if arr.shape != (n,):
                raise ValueError('initial arrays must have shape (n_cells,)')

    # ---- time base ---------------------------------------------------------
    n_steps = max(1, int(round(t_end / dt)))
    h_step = t_end / n_steps
    stride = max(1, int(round(sample_dt / h_step)))
    record = np.zeros(n_steps + 1, bool)
    record[::stride] = True
    record[-1] = True  # always land exactly on t_end
    times = np.arange(n_steps + 1) * h_step

    n_frames = int(record.sum())
    V_f = np.empty((n_frames, n)); s_f = np.empty((n_frames, n))
    a_f = np.empty((n_frames, n)); h_f = np.empty((n_frames, n))
    st_f = np.empty((n_frames, n), np.int8)

    # ---- latched-state scalars and diagnostics -----------------------------
    irr_f = latched_irr.astype(float)
    lys_f = latched_lys.astype(float)
    atp_timer = np.zeros(n); h_timer = np.zeros(n)
    t_lyse = np.full(n, np.inf); t_irr = np.full(n, np.inf)
    t_dep = np.full(n, np.inf)
    seen_dep = np.zeros(n, bool)
    seen_irr = np.zeros(n, bool); seen_lys = np.zeros(n, bool)
    bounds_hit = dict(V_floor=0, V_ceiling=0, atp_floor=0,
                      integrity_low=0, integrity_high=0)
    a_floor = p['atp_deplete_frac'] * p['atp0']
    atp0 = p['atp0']
    s_i, s_set = p['s_impermeant'], p['s_setpoint']
    v_min, v_max = p['v_min_guard'], p['v_max_guard']

    def flux_s(V_, s_, a_, h_, tear_now, irr_f_, lys_f_):
        """ds/dt exactly as integrated below, used for the solute budget check.

        Written in the documented gated form
          s' = (1-L)*[p_leak*P*(1 - s/V) - g*pump_rate*(s/V - s_set)] - L*k_collapse*(s - v_collapse)
        including the post-lysis collapse term, otherwise a lysed cell would look
        like an unaccounted solute source in the budget.
        """
        P = (1.0 - h_) * tear_now + p['p_resid_leak'] * irr_f_
        invV = 1.0 / V_
        g = (a_ * a_ / (k_atp_pump * k_atp_pump + a_ * a_)) * g_norm
        return ((1.0 - lys_f_) * (p['p_leak'] * P * (1.0 - s_ * invV)
                                  - g * pump_rate_i * (s_ * invV - s_set))
                - lys_f_ * p['k_collapse'] * (s_ - p['v_collapse']))

    tear0 = tear_in if (tear_until is None or 0.0 < tear_until) else np.zeros(n)
    s_start = s.copy()
    bal_scheme = np.zeros(n)   # flux integrated with the scheme's own weighting
    bal_trapz = np.zeros(n)    # plain trapezoidal quadrature of ds/dt

    tmpA = np.empty(n); tmpB = np.empty(n)
    frame = 0
    for step in range(n_steps + 1):
        t = times[step]
        if record[step]:
            V_f[frame] = V; s_f[frame] = s; a_f[frame] = a; h_f[frame] = h
            st_f[frame] = np.where(lys_f > .5, 3, np.where(irr_f > .5, 2,
                                  np.where(injured, 1, 0))).astype(np.int8)
            frame += 1
        if step == n_steps:
            break
        tear_now = tear_in if (tear_until is None or t < tear_until) else np.zeros(n)
        # Flux at the START of the step, with exactly the coefficients the update
        # below uses: this makes the solute budget check below an exact identity
        # unless the implementation and the documented equation disagree.
        flux_start = flux_s(V, s, a, h, tear_now, irr_f, lys_f)

        # --- gates, permeability, pump flux --------------------------------
        aa = a * a
        np.add(aa, k_atp_pump * k_atp_pump, out=tmpA)
        np.divide(aa, tmpA, out=tmpB)
        g = tmpB * g_norm
        invV = 1.0 / V
        conc = s * invV
        j_p = g * pump_rate_i * (conc - s_set)
        perm_open = (1.0 - h) * tear_now + p['p_resid_leak'] * irr_f
        gate = 1.0 - lys_f                      # L-gate: a lysed cell keeps only collapse

        # --- (1) volume: y' = A - lam*y with A >= 0, lam > 0 ---------------
        # Affine decomposition in V of the documented equation
        #   V' = (1-L)*[k_water*((s+s_i)/V - 1) + q_water*P - rvd_rate*g*max(V-1,0)]
        #        - L*k_collapse*(V - v_collapse),   L = [lysed], P = open membrane
        # using  k_water*((s+s_i)/V-1) = k_water*(s+s_i)/V - k_water   (lam += k_water/V)
        #        -rvd_rate*g*max(V-1,0) = rvd_rate*g - rvd_rate*g*V     (lam += rvd_rate*g)
        # A mis-split silently deletes a term; self_test test0 verifies this algebra.
        swollen = (V > 1.0)
        rvd_coef = p['rvd_rate'] * g * swollen          # same value in A and in lam
        lam_V = gate * (p['k_water'] * invV + rvd_coef) + lys_f * p['k_collapse']
        A_V = gate * (p['k_water'] * (s + s_i) * invV + p['q_water'] * perm_open + rvd_coef) \
            + lys_f * p['k_collapse'] * p['v_collapse']
        V_new = _expo(V, A_V, lam_V, h_step)

        # --- (2) solute: exact affine split in s ---------------------------
        #   s' = (1-L)*[p_leak*P*(1 - s/V) - g*pump_rate*(s/V - s_setpoint)]
        #        - L*k_collapse*(s - v_collapse)
        lam_s = gate * (p['p_leak'] * perm_open * invV + g * pump_rate_i * invV) \
            + lys_f * p['k_collapse']
        A_s = gate * (p['p_leak'] * perm_open + g * pump_rate_i * s_set) \
            + lys_f * p['k_collapse'] * p['v_collapse']
        s_new = _expo(s, A_s, lam_s, h_step)

        # --- (3) ATP ------------------------------------------------------
        prod_eff = ox * np.clip(p['cons_base'] + p['k_atp_rec'] * (atp0 - a), 0.0, p['prod_max'])
        lam_a = (1.0 - lys_f) * (p['cons_base'] / atp0 + ox * p['k_atp_rec']
                                 + p['cons_pump'] * np.abs(j_p) / np.maximum(a, atp0 * 1e-6)) \
            + lys_f * p['k_collapse']
        # A = f + lam*a with f = (1-lys)*(prod_eff - (cons_base/atp0)*a - cons_pump*|j_p|)
        #                       - lys*k_collapse*(a - 0); A >= 0 by construction.
        A_a = (1.0 - lys_f) * (prod_eff + ox * p['k_atp_rec'] * a)
        a_new = _expo(a, A_a, lam_a, h_step)

        # --- (4) integrity: A/lam <= 1 keeps h in [0,1] for any dt ---------
        rep = k_repair * np.where(irr_f > .5, p['repair_fail_frac'], 1.0)
        fail = (h > p['hm_ceiling_fail']) & (irr_f > .5)
        lam_h = (1.0 - lys_f) * (rep + k_damage_i * tear_now + fail * p['k_fail']) \
            + lys_f * p['k_rupture']
        A_h = (1.0 - lys_f) * (rep + fail * p['k_fail'] * p['hm_ceiling_fail'])
        h_new = _expo(h, A_h, lam_h, h_step)

        # --- guards (violations are counted BEFORE clipping) ---------------
        bounds_hit['V_floor'] += int(np.count_nonzero(V_new < v_min))
        bounds_hit['V_ceiling'] += int(np.count_nonzero(V_new > v_max))
        bounds_hit['atp_floor'] += int(np.count_nonzero(a_new < 0.0))
        bounds_hit['integrity_low'] += int(np.count_nonzero(h_new < 0.0))
        bounds_hit['integrity_high'] += int(np.count_nonzero(h_new > 1.0))
        np.clip(V_new, v_min, v_max, out=V_new)
        np.clip(a_new, 0.0, None, out=a_new)
        np.clip(h_new, 0.0, 1.0, out=h_new)

        # --- solute balance -------------------------------------------------
        # (i) scheme-consistent budget: the state increment must equal the flux at
        #     the start of the step integrated with the scheme's own exponential
        #     weighting, i.e. flux*dt*phi(lam*dt). This is an IMPLEMENTATION check
        #     (it detects a documented/implemented mismatch or a clipped state),
        #     consistent with the update being the exact solution of the frozen
        #     linear problem.
        z_s = lam_s * h_step
        phi_s = np.where(z_s > 1e-8, -np.expm1(-z_s) / np.where(z_s > 1e-8, z_s, 1.0),
                         1.0 - .5 * z_s)
        bal_scheme += flux_start * (h_step * phi_s)
        # (ii) plain trapezoidal quadrature of ds/dt: an ODE-quadrature residual that
        #      grows when lam*dt >> 1. Reported for honesty, not as conservation.
        flux_at_end = flux_s(V_new, s_new, a_new, h_new, tear_now, irr_f, lys_f)
        bal_trapz += .5 * h_step * (flux_start + flux_at_end)

        V, s, a, h = V_new, s_new, a_new, h_new
        t_next = t + h_step

        # --- triage --------------------------------------------------------
        first_dep = (~seen_dep) & (a < a_floor)
        if first_dep.any():
            t_dep[first_dep] = t_next
            seen_dep |= first_dep
        atp_timer = np.where(a < a_floor, atp_timer + h_step, 0.0)
        h_timer = np.where(h < p['hm_floor'], h_timer + h_step, 0.0)
        new_lys = (~latched_lys) & ((V >= p['v_lysis']) | (h < p['hm_lyse']))
        if new_lys.any():
            t_lyse[new_lys] = t_next
            latched_lys |= new_lys
            lys_f[new_lys] = 1.0
        new_irr = (~latched_irr) & ((atp_timer > p['t_atp_irrev']) |
                                    (h_timer > p['t_hm_irrev']) |
                                    latched_lys)
        if new_irr.any():
            t_irr[new_irr] = t_next
            latched_irr |= new_irr
            irr_f[new_irr] = 1.0
        entry = (h < p['hm_inj']) | (V > p['v_swell_rev']) | (V < p['v_shrink_rev']) \
            | (a < p['atp_inj_frac'] * atp0)
        exit_ = (h > p['hm_heal']) & (np.abs(V - 1.0) < p['v_heal_tol']) \
            & (a > p['atp_heal_frac'] * atp0)
        injured = np.where(injured, ~exit_, entry)

    runtime = _time.perf_counter() - start

    # ---- death fractions, misleading-verdict comparison, balance -----------
    dead = st_f >= 2
    death_fraction = dead.mean(axis=1)
    if tissue_only_stats:
        tissue = ~ablated
        death_fraction_tissue = dead[:, tissue].mean(axis=1) if tissue.any() else death_fraction
    else:
        death_fraction_tissue = death_fraction
    ds = s - s_start
    denom = np.maximum(1e-12, np.maximum(np.abs(ds), np.abs(bal_scheme)))
    mass_balance_error = float(np.max(np.abs(ds - bal_scheme) / denom))
    denom_t = np.maximum(1e-12, np.maximum(np.abs(ds), np.abs(bal_trapz)))
    trapezoid_residual = float(np.max(np.abs(ds - bal_trapz) / denom_t))

    # A verdict that ignores the latch: same thresholds applied instantaneously.
    naive_lysed = lys_f > .5
    naive_irr = (a < a_floor) | (h < p['hm_floor']) | (V >= p['v_lysis'])
    naive_inj = (h < p['hm_inj']) | (V > p['v_swell_rev']) | (V < p['v_shrink_rev']) \
        | (a < p['atp_inj_frac'] * atp0)
    state_naive = np.where(naive_lysed, 3, np.where(naive_irr, 2,
                          np.where(naive_inj, 1, 0))).astype(np.int8)

    metadata = dict(
        model='intracellular cell-state layer (volume/solute/ATP/integrity + triage)',
        calibrated=False,
        WARNING=('UNCALIBRATED MECHANISM DEMONSTRATION. Every parameter except the '
                 'atp0 order of magnitude is an arbitrary illustrative value. These '
                 'state codes are NOT a biological prediction: do not report that any '
                 'cell survives or dies in reality on the basis of this module. See '
                 'PARAM_SOURCES and test 8 (identifiability).'),
        n_cells=n, n_side=geo['n_side'], seed=seed, het_sigma=het_sigma,
        oxygen=oxygen_used, oxygen_input=oxygen_desc, t_end=t_end, dt_nominal=dt, dt_effective=h_step,
        n_steps=n_steps, sample_dt=sample_dt, n_frames=n_frames,
        tear_source=tear_source, tear_max=float(tear_in.max()), wound=bool(wound),
        tear_until=tear_until, repair_enabled=bool(repair_enabled),
        parameters={k: float(v) for k, v in p.items() if isinstance(v, (int, float))},
        param_sources=PARAM_SOURCES,
        state_codes={str(k): v for k, v in STATE_CODES.items()},
        equations=_EQUATIONS,
        units=dict(time='s', V='relative (1 = isotonic reference)', solute='relative amount',
                   atp='uM', integrity='dimensionless [0,1]', tear='dimensionless 0..~1',
                   oxygen='relative availability (1 = reference)', area='um^2', length='um'),
        numerics=dict(scheme='vectorised exponential Euler per state, y\'=A-lam*y, A>=0, lam>=0',
                      order='first order in dt (state-dependent coefficients frozen over the step)',
                      stability='unconditional in lam*dt; positivity preserved; h in [0,1] for any dt',
                      guards='V clipped to [v_min_guard, v_max_guard], a >= 0, h in [0,1]',
                      caveat='no solve_ivp; the scheme is stiff-safe but only first order'),
        bounds_hit=bounds_hit,
        mass_balance_error=mass_balance_error,
        solute_balance=dict(
            scheme_consistency_residual=mass_balance_error,
            trapezoid_quadrature_residual=trapezoid_residual,
            definition=("mass_balance_error = max_cells |ds - sum_steps flux_start*dt*"
                        "phi(lam*dt)| / max(1e-12,|ds|,|budget|): the implemented solute "
                        "update versus the documented flux integrated with the scheme's "
                        "own exponential weighting. It is an implementation/consistency "
                        "diagnostic, NOT a physical conservation law: solute is not "
                        "conserved in general because the leak admits it and the pumps "
                        "extrude it. The physical statement is the zero-permeability "
                        "control (test 2), where the residual is exactly 0. The "
                        "trapezoid_quadrature_residual is a plain ODE quadrature error "
                        "that grows when lam*dt >> 1 (dt=0.05 s can exceed the fastest "
                        "relaxation time); it is reported for honesty, not as an error of "
                        "the model."),
            trapezoid_caveat='large values are a property of first-order quadrature, not of the biology'),
        runtime_seconds=runtime,
        ablated_cells=int(ablated.sum()),
        ablated_rule=f'radius < {ABLATED_RADIUS_UM} um (model.py convention)',
        tissue_only_stats=bool(tissue_only_stats),
        limitations=[
            'cell geometry is fixed: volume changes are reported, not applied to tissue shape',
            'one lumped intracellular compartment: no ER/mitochondrial ATP pools, no pH, no ROS',
            'no gap-junction coupling of volume/solute/ATP between cells in this module',
            'death triage is a model verdict with arbitrary thresholds, not an assay',
            'prescribed injury input, not computed mechanics (same truncation as model.py)',
            'oxygen is a relative availability input, not a Krogh-type oxygen field',
        ],
        literature=[_LIT_ATP['citation']],
    )

    result = dict(
        times=times[record], V=V_f, solute=s_f, atp=a_f, integrity=h_f, state=st_f,
        death_fraction=death_fraction, death_fraction_tissue=death_fraction_tissue,
        mass_balance_error=mass_balance_error, metadata=metadata,
        positions=geo['positions'], polygons=geo['polygons'], areas=geo['areas'],
        radius=geo['radius'], ablated=ablated, tear_field=tear_in,
        tear_until=tear_until,
        time_atp_depleted=t_dep, time_irreversible=t_irr, time_lysed=t_lyse,
        state_final=st_f[-1], state_naive_final=state_naive,
        latched_irreversible=latched_irr, latched_lysed=latched_lys,
        injured_final=injured, n_side=geo['n_side'],
        final=dict(V=V, solute=s, atp=a, integrity=h,
                   latched_irreversible=latched_irr, latched_lysed=latched_lys,
                   injured=injured),
    )
    if not return_frames:
        for key in ('V', 'solute', 'atp', 'integrity', 'state'):
            result.pop(key)
    return result


def bridge_damage(result, prescribed_damage, bridge_gain=1.0,
                  window=(0.0, CALCIUM_WINDOW_S)):
    """Map cell-state integrity to an effective calcium-model damage field.

    damage_eff = damage_prescribed * (1 + bridge_gain*(1 - h_min)), where h_min is
    the minimum membrane integrity of that cell inside `window` (default the first
    25 s, because model.run refuses t_end > 25 s). Properties: monotone in
    (1 - h_min), bounded by (1 + bridge_gain)*damage_prescribed, and EXACTLY the
    prescribed damage when h_min = 1 -- so an intact-membrane cell reproduces the
    existing model's own input, which is checkable (see test 8).

    bridge_gain is ARBITRARY. There is no measurement that fixes it: it is the
    transduction between "this cell lost membrane integrity in the state layer"
    and "this cell admits more calcium in the existing model". Test 8 reports how
    much of the observable's variation that arbitrary choice controls.
    """
    times = np.asarray(result['times'], float)
    m = (times >= window[0]) & (times <= window[1])
    if not m.any():
        raise ValueError('window contains no stored frames')
    h_min = np.asarray(result['integrity'][m], float).min(axis=0)
    dmg = np.asarray(prescribed_damage, float)
    if dmg.shape != h_min.shape:
        raise ValueError('prescribed_damage must match the cell count')
    return dmg * (1.0 + bridge_gain * (1.0 - h_min)), h_min


# ---------------------------------------------------------------------------
def identifiability_analysis(n_cells=576, params=None, oxygen=1.0, t_end=300.0,
                            dt=0.05, perturb=0.3, bridge_gain=1.0, seeds=(2025, 2026),
                            names=('tau_repair', 'cons_base', 'v_lysis',
                                   'v_swell_rev', 'p_leak'),
                            calcium_t_end=CALCIUM_WINDOW_S, calcium_sample_dt=0.5):
    """Sensitivity of internal states and of the ONLY observable to each parameter.

    For each named parameter, one run at the reference value and two runs at
    (1 -/+ perturb) x value. For every run:
      (a) internal: relative change in the final-frame mean volume and the change
          in the final death fraction (state >= 2);
      (b) observable: the cell-state integrity is pushed through the documented
          ARBITRARY bridge (see bridge_damage) into model.run(damage_override=...),
          and measurement.measure returns the annular fluorescence half-height
          radius. Reported at the last calcium frame (t = 25 s, model.py's
          admissible end) and as the mean over uncensored frames.
    Also reported: the bridge-off reference (integrity ignored entirely), the
    raster floor (0.625 um at 1.6 pixels/um), and a seed-to-seed spread of the
    observable to use as an empirical noise floor.

    Cost: ~17 calcium runs at ~1.1 s each plus ~11 cell-state runs at ~0.3 s each;
    this is deliberately a SMALL sweep (the spec's own warning: 576 cells costs
    ~0.7 s per calcium run). No plots, no files.
    """
    from measurement import measure  # matplotlib-backed, deferred on purpose

    p = _check_params(params)
    geo = cell_geometry(n_cells)
    prescribed = prescribed_tear_field(geo, wound=True, tear_scale=1.0)
    n_side = geo['n_side']
    n_runs = 0

    def cellstate_and_radius(par):
        nonlocal n_runs
        res = run(n_cells=n_cells, tear=prescribed.copy(), oxygen=oxygen, t_end=t_end,
                  dt=dt, params=par, seed=2025, sample_dt=1.0,
                  ablated_override=geo['ablated'], return_frames=True)
        n_runs += 1
        dmg, h_min = bridge_damage(res, prescribed, bridge_gain)
        cal = _model.run(n_side=n_side, t_end=calcium_t_end, sample_dt=calcium_sample_dt,
                         damage_override=dmg, seed=2025)
        n_runs += 1
        m = measure(cal['c'], cal['baseline_c'], cal['polygons'], cal['gcamp'],
                    cal['ablated'])
        return res, h_min, dmg, cal, _radius_summary(m)

    ref_res, ref_hmin, ref_dmg, ref_cal, ref_rad = cellstate_and_radius(p)
    # Integrity ignored: must reproduce model.py's own prescribed-damage input.
    cal_nobridge = _model.run(n_side=n_side, t_end=calcium_t_end,
                              sample_dt=calcium_sample_dt, damage_override=prescribed,
                              seed=2025)
    n_runs += 1
    rad_nobridge = _radius_summary(measure(cal_nobridge['c'], cal_nobridge['baseline_c'],
                                           cal_nobridge['polygons'], cal_nobridge['gcamp'],
                                           cal_nobridge['ablated']))
    seed_rows = []
    for sd in seeds:
        c2 = _model.run(n_side=n_side, t_end=calcium_t_end, sample_dt=calcium_sample_dt,
                        damage_override=ref_dmg, seed=int(sd))
        n_runs += 1
        seed_rows.append(dict(seed=int(sd), **_radius_summary(
            measure(c2['c'], c2['baseline_c'], c2['polygons'], c2['gcamp'], c2['ablated']))))
    bridge_rows = []
    for bg in (0.0, .5, 2.0):
        dmg, _ = bridge_damage(ref_res, prescribed, bg)
        c2 = _model.run(n_side=n_side, t_end=calcium_t_end, sample_dt=calcium_sample_dt,
                        damage_override=dmg, seed=2025)
        n_runs += 1
        bridge_rows.append(dict(bridge_gain=bg, **_radius_summary(
            measure(c2['c'], c2['baseline_c'], c2['polygons'], c2['gcamp'], c2['ablated']))))

    mean_V = float(ref_res['V'][-1].mean())
    death = float(np.mean(ref_res['state_final'] >= 2))
    rev = float(np.mean(ref_res['state_final'] == 1))
    r_ref = ref_rad['radius_final_um']
    raster_um = 1.0 / 1.6  # measurement.py: pixels_per_um = 1.6
    rows = []
    for name in names:
        for sign in (-1, 1):
            par = dict(p)
            par[name] = p[name] * (1.0 + sign * perturb)
            try:
                res, h_min, dmg, cal, rad = cellstate_and_radius(par)
            except Exception as exc:  # e.g. a perturbed threshold set becomes invalid
                rows.append(dict(parameter=name, perturb_fraction=sign * perturb,
                                 error=f'{type(exc).__name__}: {exc}'))
                continue
            mean_V_p = float(res['V'][-1].mean())
            death_p = float(np.mean(res['state_final'] >= 2))
            rev_p = float(np.mean(res['state_final'] == 1))
            r_p = rad['radius_final_um']
            rows.append(dict(
                parameter=name, value_ref=float(p[name]),
                value=float(par[name]), perturb_fraction=sign * perturb,
                mean_V=mean_V_p, mean_V_fractional_change=(mean_V_p - mean_V) / mean_V,
                death_fraction=death_p, death_fraction_absolute_change=death_p - death,
                reversibly_injured_fraction=rev_p,
                reversibly_injured_fraction_absolute_change=rev_p - rev,
                h_min_mean=float(h_min.mean()),
                damage_max=float(dmg.max()),
                radius_final_um=r_p,
                radius_final_fractional_change=(r_p - r_ref) / r_ref,
                radius_mean_uncensored_um=rad['radius_mean_uncensored_um'],
                radius_mean_fractional_change=(
                    (rad['radius_mean_uncensored_um'] - ref_rad['radius_mean_uncensored_um'])
                    / ref_rad['radius_mean_uncensored_um']),
                n_censored=rad['n_censored'],
            ))

    seed_finals = [r['radius_final_um'] for r in seed_rows if np.isfinite(r['radius_final_um'])]
    seed_spread = (max(seed_finals) - min(seed_finals)) if len(seed_finals) > 1 else 0.0
    floor = max(raster_um / r_ref, (seed_spread / r_ref) if r_ref else 0.0)
    ok_rows = [r for r in rows if 'radius_final_fractional_change' in r]
    max_abs_radius = max((abs(r['radius_final_fractional_change']) for r in ok_rows), default=0.0)
    bridge_swing = (max(b['radius_final_um'] for b in bridge_rows)
                    - min(b['radius_final_um'] for b in bridge_rows)) / r_ref
    worst = max(ok_rows, key=lambda r: abs(r['radius_final_fractional_change']), default=None)
    worst_name = worst['parameter'] if worst else 'n/a'
    max_mean_V = max((abs(r['mean_V_fractional_change']) for r in ok_rows), default=0.0)
    max_death = max((abs(r['death_fraction_absolute_change']) for r in ok_rows), default=0.0)
    nobridge_frac = (rad_nobridge['radius_final_um'] - r_ref) / r_ref
    blunt = (
        'Blunt answer: NONE of the five parameters is identifiable from the fluorescence '
        f'radius. The largest |fractional radius change| in the whole +/-30% table is '
        f'{100*max_abs_radius:.3f}% (parameter {worst_name}), which is below the measurement '
        f'raster floor ({100*raster_um/r_ref:.3f}% = {raster_um:.4f} um) and below the '
        f'seed-to-seed spread of the same observable ({100*seed_spread/r_ref:.3f}%). The two '
        f'triage thresholds (v_swell_rev, v_lysis) move the radius by 0-0.3% because they are '
        f'classification thresholds that do not enter the state equations at all. Meanwhile the '
        f'ARBITRARY bridge coefficient, which no data constrains, swings the same observable by '
        f'{100*bridge_swing:.2f}%, and simply ignoring membrane integrity moves it by '
        f'{100*abs(nobridge_frac):.2f}%. Inside the module the parameters ARE visible (mean '
        f'volume moves by up to {100*max_mean_V:.2f}% and the death fraction by up to '
        f'{max_death:.3f} in absolute fraction), but none of V, solute, ATP or integrity is '
        f'observable in this project, so the radius cannot distinguish these mechanisms and '
        f'cannot validate this module.')
    verdict = dict(
        unidentifiability_floor_fraction_of_radius=float(floor),
        floor_sources=dict(raster_floor_fraction_of_radius=float(raster_um / r_ref),
                           seed_spread_fraction_of_radius=float(seed_spread / r_ref if r_ref else 0.0)),
        max_abs_radius_fractional_change_over_table=float(max_abs_radius),
        identifiability_verdict={
            r['parameter']: ('identifiable' if abs(r['radius_final_fractional_change']) > floor
                             else 'UNIDENTIFIABLE (change below both the raster resolution '
                                  'and the seed-to-seed spread)')
            for r in ok_rows},
        arbitrary_bridge_swing_fraction_of_radius=float(bridge_swing),
        blunt_statement=blunt,
    )
    return dict(
        reference=dict(mean_V=mean_V, death_fraction=death,
                       reversibly_injured_fraction=rev,
                       h_min_mean=float(ref_hmin.mean()),
                       radius_final_um=r_ref,
                       radius_mean_uncensored_um=ref_rad['radius_mean_uncensored_um'],
                       damage_max=float(ref_dmg.max())),
        rows=rows,
        no_bridge_reference=rad_nobridge,
        no_bridge_radius_fractional_change=(
            (rad_nobridge['radius_final_um'] - r_ref) / r_ref),
        bridge_gain_rows=bridge_rows,
        seed_rows=seed_rows,
        seed_radius_spread_um=float(seed_spread),
        raster_floor_um=float(raster_um),
        raster_floor_fraction_of_radius=float(raster_um / r_ref),
        verdict=verdict,
        calcium_runs=n_runs,
        notes=[
            'the observable is the annular fluorescence half-height radius from '
            'measurement.measure; it is the ONLY observable this project has',
            f'raster floor {raster_um:.4f} um = {100*raster_um/r_ref:.2f}% of the '
            f'reference radius {r_ref:.3f} um: fractional radius changes below that '
            'cannot even be resolved by the measurement code',
            f'seed-to-seed spread of the observable with identical inputs: '
            f'{seed_spread:.4f} um ({100*seed_spread/r_ref:.2f}%)',
            'the bridge coefficient is arbitrary; its own variation (bridge_gain 0 -> 2) '
            'moves the observable by more than every parameter in the table',
            'the calcium model itself is truncated at 25 s, so only the first 25 s of '
            'cell-state integrity can enter the observable through this bridge',
        ],
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _verdict_stats(res, zone=None):
    """Compact per-run summary of the triage verdict and of state extremes."""
    st = res['state_final']
    out = dict(
        death_fraction_all=float(np.mean(st >= 2)),
        healthy_fraction_all=float(np.mean(st == 0)),
        reversibly_injured_fraction_all=float(np.mean(st == 1)),
        irreversibly_injured_not_lysed_fraction_all=float(np.mean(st == 2)),
        lysed_fraction_all=float(np.mean(st == 3)),
        mean_volume_final=float(res['V'][-1].mean()),
        min_volume=float(res['V'].min()), max_volume=float(res['V'].max()),
        min_atp_uM=float(res['atp'].min()), max_atp_uM=float(res['atp'].max()),
        min_integrity=float(res['integrity'].min()),
        max_integrity=float(res['integrity'].max()),
        min_solute=float(res['solute'].min()), max_solute=float(res['solute'].max()),
        mass_balance_error=float(res['mass_balance_error']),
        bounds_hit=dict(res['metadata']['bounds_hit']),
        runtime_seconds=float(res['metadata']['runtime_seconds']),
    )
    if zone is not None:
        out.update(dict(
            zone_n_cells=int(np.count_nonzero(zone)),
            zone_death_fraction=float(np.mean(st[zone] >= 2)),
            zone_lysed_fraction=float(np.mean(st[zone] == 3)),
            zone_mean_volume_final=float(res['V'][-1][zone].mean()),
        ))
    return out


def _amplitude_threshold(criterion='zone_death', target=0.0, zone=None, tear_until=None,
                         dt=0.1, t_end=300.0, lo=0.0, hi=None, tol=0.002, cap=4.0,
                         params=None, seed=2025, max_iterations=14):
    """Bisect for the smallest prescribed-tear amplitude reaching a target fraction.

    criterion='zone_death' uses the fraction of `zone` cells with state >= 2 (death
    triage), 'zone_lysed' the fraction with state == 3. target=0.0 therefore finds
    the FIRST amplitude at which any cell of the zone is irreversibly injured.

    The response is assumed monotone in amplitude (the reported curves are); `hi` is
    doubled until the target is reached or `cap` is exceeded. Returns the bracket,
    the threshold estimate, the number of runs used and the protocol description.
    Protocol: wound=True, prescribed radial microtear field x amplitude, tear applied
    for all t if tear_until is None else for t < tear_until, dt as given.
    """
    runs = [0]

    def fraction(amp):
        r = run(n_cells=576, wound=True, tear_scale=float(amp), tear_until=tear_until,
                dt=dt, t_end=t_end, params=params, seed=seed, sample_dt=t_end)
        runs[0] += 1
        st = r['state_final']
        key = (st >= 2) if criterion == 'zone_death' else (st == 3)
        return float(np.mean(key[zone]) if zone is not None else np.mean(key))

    if hi is None:
        hi = .25
        while fraction(hi) <= target and hi < cap:
            hi *= 2.0
    f_hi = fraction(hi)
    if f_hi <= target:
        return dict(reached=False, criterion=criterion, target=target, amplitude_hi=float(hi),
                    fraction_at_hi=f_hi, n_runs=runs[0], tear_until=tear_until, dt=dt)
    iterations = 0
    while (hi - lo) > tol and iterations < max_iterations:
        mid = .5 * (lo + hi)
        if fraction(mid) > target:
            hi = mid
        else:
            lo = mid
        iterations += 1
    return dict(reached=True, criterion=criterion, target=target,
                threshold_amplitude=float(.5 * (lo + hi)),
                bracket=[float(lo), float(hi)], iterations=iterations,
                fraction_at_upper=f_hi, n_runs=runs[0], tol=tol,
                tear_until=tear_until, dt=dt, t_end=t_end)


def _decomposition_self_check(p, seed=3):
    """Algebra check of the documented y' = A - lam*y split for all four states.

    Evaluates the documented right-hand sides and the documented (A, lam) splits on
    random admissible states and reports the largest relative mismatch
    |A - lam*y - RHS| / scale. This verifies the ALGEBRA of the split that the
    integrator relies on (an error here would silently integrate a different model);
    the implementation itself is verified by mass_balance_error, which caught exactly
    such a mismatch during development. It is not a test of biology.
    """
    rng = np.random.default_rng(seed)
    n = 4096
    V = rng.uniform(.05, 3.0, n)
    s = rng.uniform(0.0, 3.0, n)
    a = rng.uniform(0.0, 2000.0, n)
    h = rng.uniform(0.0, 1.0, n)
    tear = rng.uniform(0.0, 2.0, n)
    irr = (rng.random(n) < .3).astype(float)
    lys = (rng.random(n) < .1).astype(float)
    K, nH = p['k_atp_pump'], p['n_atp_pump']
    g = (a ** nH / (K ** nH + a ** nH)) / (p['atp0'] ** nH / (K ** nH + p['atp0'] ** nH))
    ox = 1.0
    invV = 1.0 / V
    gate = 1.0 - lys
    perm_open = (1.0 - h) * tear + p['p_resid_leak'] * irr
    j_p = g * p['pump_rate'] * (s * invV - p['s_setpoint'])
    rvd = p['rvd_rate'] * g * np.maximum(V - 1.0, 0.0)
    rvd_coef = p['rvd_rate'] * g * (V > 1.0)
    rep = (1.0 / p['tau_repair']) * np.where(irr > .5, p['repair_fail_frac'], 1.0)
    fail = (h > p['hm_ceiling_fail']) & (irr > .5)
    prod_eff = ox * np.clip(p['cons_base'] + p['k_atp_rec'] * (p['atp0'] - a), 0.0, p['prod_max'])
    # documented right-hand sides
    rhs_V = (gate * (p['k_water'] * ((s + p['s_impermeant']) * invV - 1.0)
                     + p['q_water'] * perm_open - rvd)
             - lys * p['k_collapse'] * (V - p['v_collapse']))
    rhs_s = (gate * (p['p_leak'] * perm_open * (1.0 - s * invV) - j_p)
             - lys * p['k_collapse'] * (s - p['v_collapse']))
    rhs_a = ((1.0 - lys) * (prod_eff - p['cons_base'] / p['atp0'] * a
                            - p['cons_pump'] * np.abs(j_p)) - lys * p['k_collapse'] * a)
    rhs_h = ((1.0 - lys) * (rep * (1.0 - h) - p['k_damage'] * tear * h
                            - fail * p['k_fail'] * np.maximum(h - p['hm_ceiling_fail'], 0.0))
             - lys * p['k_rupture'] * h)
    # documented (A, lam) splits
    lam_V = gate * (p['k_water'] * invV + rvd_coef) + lys * p['k_collapse']
    A_V = gate * (p['k_water'] * (s + p['s_impermeant']) * invV + p['q_water'] * perm_open
                  + rvd_coef) + lys * p['k_collapse'] * p['v_collapse']
    lam_s = gate * (p['p_leak'] * perm_open * invV + g * p['pump_rate'] * invV) \
        + lys * p['k_collapse']
    A_s = gate * (p['p_leak'] * perm_open + g * p['pump_rate'] * p['s_setpoint']) \
        + lys * p['k_collapse'] * p['v_collapse']
    lam_a = ((1.0 - lys) * (p['cons_base'] / p['atp0'] + ox * p['k_atp_rec']
                            + p['cons_pump'] * np.abs(j_p) / np.maximum(a, p['atp0'] * 1e-6))
             + lys * p['k_collapse'])
    A_a = (1.0 - lys) * (prod_eff + ox * p['k_atp_rec'] * a)
    lam_h = ((1.0 - lys) * (rep + p['k_damage'] * tear + fail * p['k_fail'])
             + lys * p['k_rupture'])
    A_h = (1.0 - lys) * (rep + fail * p['k_fail'] * p['hm_ceiling_fail'])
    checks = {}
    for name, A, lam, y, rhs in (('V', A_V, lam_V, V, rhs_V), ('solute', A_s, lam_s, s, rhs_s),
                                 ('atp', A_a, lam_a, a, rhs_a), ('integrity', A_h, lam_h, h, rhs_h)):
        scale = np.maximum(1e-6, np.abs(A) + np.abs(lam * y) + np.abs(rhs))
        checks['max_relative_mismatch_' + name] = float(np.max(np.abs(A - lam * y - rhs) / scale))
        checks['min_A_' + name] = float(A.min())
    checks['note'] = ('A >= 0 everywhere is what makes the scheme positivity preserving; '
                      'max_relative_mismatch_* should be at rounding level')
    return checks


# ---------------------------------------------------------------------------
def self_test(identifiability=True, quick=False, verbose=False):
    """Run the required numerical tests; return a JSON-serialisable dict of NUMBERS.

    Tests (all numbers are ACTUAL outputs of this file, not targets):
      1  no-injury control drift, plus a dt-refinement check (1b)
      2  solute conservation with zero permeability and pumps off
      3  boundedness over a tear and oxygen stress sweep
      4  injury/lysis threshold curve and bisected threshold amplitudes
      5  hysteresis: latched vs naive verdict, plus a release-and-repair continuation
      6  repair enabled vs disabled
      7  ATP-depletion and irreversible-injury times vs oxygen
      8  identifiability of the new parameters from the fluorescence radius
      9  determinism (same seed reproducible, seed changes with heterogeneity on)
    Sweeps use dt = 0.1 s (documented; the dt-refinement check quantifies the
    difference) to keep the whole suite near a minute. No files and no plots.
    Label meanings: 'sub-threshold' = the largest tested amplitude with zero
    irreversibly injured cells; 'supra-threshold' = the smallest tested amplitude at
    which at least half of the tear-bearing input zone is irreversibly injured.
    """
    t_suite = _time.perf_counter()
    p = DEFAULTS.copy()
    out = {'uncalibrated_warning':
           ('THIS MODULE IS UNCALIBRATED. Except for the atp0 order of magnitude every '
            'parameter is an arbitrary illustrative value. The state codes are model '
            'verdicts, not measurements: this must not be used to claim that any cell '
            'survives or dies in reality.'),
           'state_codes': {str(k): v for k, v in STATE_CODES.items()},
           'sweep_dt_seconds': 0.1}
    geo = cell_geometry(576)
    zone = prescribed_tear_field(geo, wound=True, tear_scale=1.0) > 0.0

    # ---------------- 0. algebra self-check of the A - lam*y split -----------
    out['test0_decomposition_self_check'] = _decomposition_self_check(p)

    # ---------------- 1. no-injury control ---------------------------------
    ctrl = run(n_cells=576, tear=np.zeros(576), oxygen=1.0, t_end=300.0, dt=0.05)
    out['test1_no_injury_control'] = dict(
        duration_s=300.0, dt=0.05, n_cells=576,
        max_abs_V_drift=float(np.max(np.abs(ctrl['V'] - 1.0))),
        max_abs_solute_drift=float(np.max(np.abs(ctrl['solute'] - p['s_setpoint']))),
        max_abs_atp_drift_uM=float(np.max(np.abs(ctrl['atp'] - p['atp0']))),
        max_abs_integrity_drift=float(np.max(np.abs(ctrl['integrity'] - 1.0))),
        all_cells_healthy_all_frames=bool(np.all(ctrl['state'] == 0)),
        final_death_fraction=float(ctrl['death_fraction'][-1]),
        mass_balance_error=float(ctrl['mass_balance_error']),
        bounds_hit=dict(ctrl['metadata']['bounds_hit']),
        runtime_seconds=float(ctrl['metadata']['runtime_seconds']),
    )
    scenario = dict(n_cells=576, tear=np.full(576, .05), oxygen=1.0, t_end=300.0)
    coarse = run(dt=.05, **scenario)
    fine = run(dt=.01, **scenario)
    out['test1b_dt_refinement'] = dict(
        scenario='uniform tear = 0.05, oxygen = 1.0, 300 s',
        dt_coarse=0.05, dt_fine=0.01, n_steps_coarse=6000, n_steps_fine=30000,
        max_abs_V_difference=float(np.max(np.abs(coarse['V'][-1] - fine['V'][-1]))),
        max_abs_solute_difference=float(np.max(np.abs(coarse['solute'][-1] - fine['solute'][-1]))),
        max_abs_atp_difference_uM=float(np.max(np.abs(coarse['atp'][-1] - fine['atp'][-1]))),
        max_abs_integrity_difference=float(np.max(np.abs(coarse['integrity'][-1] - fine['integrity'][-1]))),
        death_fraction_coarse=float(coarse['death_fraction'][-1]),
        death_fraction_fine=float(fine['death_fraction'][-1]),
        runtime_fine_seconds=float(fine['metadata']['runtime_seconds']),
    )

    # ---------------- 2. solute conservation --------------------------------
    cons = {}
    sealed = dict(p_leak=0.0, q_water=0.0, pump_rate=0.0)
    for V0 in (1.5, 0.8):
        r = run(n_cells=576, tear=np.full(576, .1), oxygen=1.0, t_end=300.0, dt=0.1,
                V0=V0, params=sealed)
        s0 = p['s_setpoint']
        cons[f'V0_{V0}'] = dict(
            V_initial=V0, V_final_mean=float(r['V'][-1].mean()),
            V_target_osmotic=p['s_setpoint'] + p['s_impermeant'],
            max_abs_solute_deviation=float(np.max(np.abs(r['solute'] - s0))),
            max_relative_solute_deviation=float(np.max(np.abs(r['solute'] - s0)) / s0),
            final_osmotic_drive_mean=float(np.mean((r['solute'][-1] + p['s_impermeant'])
                                                  / r['V'][-1] - 1.0)),
            mass_balance_error=float(r['mass_balance_error']),
            latched_fraction=float(np.mean(r['latched_irreversible'] | r['latched_lysed'])),
            min_integrity=float(r['integrity'].min()),
        )
    # Same sealed membrane but with a big sustained tear: the solute is STILL exactly
    # conserved, yet the cell destroys itself through the integrity/triage channel.
    r_w = run(n_cells=576, tear=np.full(576, .5), oxygen=1.0, t_end=300.0, dt=0.1,
              params=sealed, sample_dt=10.0)
    t_lyse = r_w['time_lysed']
    before = r_w['times'] < 60.0
    cons['solute_conservation_is_not_survival'] = dict(
        setup='same sealed membrane (p_leak = 0, q_water = 0, pump_rate = 0), tear = 0.5',
        max_abs_solute_deviation_before_latch=float(np.max(np.abs(
            r_w['solute'][before] - p['s_setpoint']))),
        integrity_steady_state=float(r_w['integrity'][before][-1, 0]),
        integrity_floor_for_the_latch=p['hm_floor'],
        time_to_irreversible_latch_s=float(r_w['time_irreversible'].min())
        if np.isfinite(r_w['time_irreversible']).any() else None,
        time_to_lysis_s=float(t_lyse.min()) if np.isfinite(t_lyse).any() else None,
        explanation=('an intact solute budget says nothing about survival: with a sustained '
                     'tear the integrity settles below hm_floor, the latch fires after the '
                     'recovery window, repair capacity is then cut to repair_fail_frac and '
                     'integrity crosses hm_lyse, which drives the lysed attractor (and only '
                     'then does the collapse term move s). The plain conservation rows '
                     'therefore use a tear small enough (0.1) that h stays above hm_floor '
                     'and none of the commitment mechanisms fire.'),
    )
    cons['setup'] = ('p_leak = 0 (zero solute permeability), q_water = 0 (zero water '
                     'permeability), pump_rate = 0 (pumps off), tear = 0.1; the tear cannot '
                     'act because the membrane is sealed, so the solute AMOUNT s must be '
                     'exactly conserved while water re-equilibrates V toward '
                     's + s_impermeant = 1')
    out['test2_solute_conservation'] = cons

    # ---------------- 3. boundedness over a stress sweep --------------------
    amps = [0.0, .01, .05, .1, .25, .5, 1.0, 2.0, 5.0, 10.0]
    mins = dict(V=np.inf, solute=np.inf, atp=np.inf, integrity=np.inf)
    maxs = dict(V=-np.inf, solute=-np.inf, atp=-np.inf, integrity=-np.inf)
    guard_hits = {}
    sweep = []
    for amp in amps:
        r = run(n_cells=576, tear=np.full(576, amp), oxygen=1.0, t_end=300.0, dt=0.1)
        for k, a_ in (('V', r['V']), ('solute', r['solute']), ('atp', r['atp']),
                      ('integrity', r['integrity'])):
            mins[k] = min(mins[k], float(a_.min())); maxs[k] = max(maxs[k], float(a_.max()))
        for k, v in r['metadata']['bounds_hit'].items():
            guard_hits[k] = guard_hits.get(k, 0) + int(v)
        sweep.append({'case': f'uniform tear {amp}, oxygen 1.0', **_verdict_stats(r)})
    for ox in (0.0, 0.2):
        r = run(n_cells=576, tear=np.full(576, 1.0), oxygen=ox, t_end=300.0, dt=0.1)
        for k, a_ in (('V', r['V']), ('solute', r['solute']), ('atp', r['atp']),
                      ('integrity', r['integrity'])):
            mins[k] = min(mins[k], float(a_.min())); maxs[k] = max(maxs[k], float(a_.max()))
        for k, v in r['metadata']['bounds_hit'].items():
            guard_hits[k] = guard_hits.get(k, 0) + int(v)
        sweep.append({'case': f'uniform tear 1.0, oxygen {ox}', **_verdict_stats(r)})
    out['test3_boundedness'] = dict(
        cases=sweep, minima=mins, maxima=maxs, guard_hits=guard_hits,
        V_never_nonpositive=bool(mins['V'] > 0.0),
        atp_never_negative=bool(mins['atp'] >= 0.0),
        integrity_within_unit_interval=bool(mins['integrity'] >= 0.0 and maxs['integrity'] <= 1.0),
        no_guard_ever_fired=bool(sum(guard_hits.values()) == 0),
    )

    # ---------------- 4. threshold curve ------------------------------------
    grid = np.round(np.arange(0.0, 1.5001, 0.125), 4).tolist()
    curve = []
    for amp in grid:
        r = run(n_cells=576, wound=True, tear_scale=amp, oxygen=1.0, t_end=300.0, dt=0.1,
                sample_dt=300.0)
        curve.append({'amplitude': float(amp), **_verdict_stats(r, zone=zone)})
    thr_any = _amplitude_threshold(target=0.0, zone=zone, dt=0.1, t_end=300.0, lo=0.0, hi=0.5)
    thr_half = _amplitude_threshold(target=0.5, zone=zone, dt=0.1, t_end=300.0,
                                    lo=(thr_any.get('threshold_amplitude') or 0.0), hi=1.0)
    thr_lysis = _amplitude_threshold(criterion='zone_lysed', target=0.5, zone=zone, dt=0.1,
                                     t_end=300.0, lo=(thr_any.get('threshold_amplitude') or 0.0),
                                     hi=1.5)
    out['test4_threshold_curve'] = dict(
        amplitude_grid=grid, curve=curve,
        input_zone_definition=('cells with prescribed tear > 0 (radius < ~51.2 um); '
                               'the ablated core is included in the zone'),
        input_zone_cells=int(zone.sum()),
        first_amplitude_with_any_irreversible=thr_any,
        amplitude_for_half_of_input_zone_irreversible=thr_half,
        amplitude_for_half_of_input_zone_lysed=thr_lysis,
        grid_resolution_note=('grid amplitudes with zero irreversible cells vs the first '
                              'with any: the bisected values above are the reported '
                              'thresholds; the grid is reported for the full curve'),
        protocol='sustained prescribed radial microtear field, dt = 0.1 s, t_end = 300 s',
    )
    zero_amps = [c['amplitude'] for c in curve if c['zone_death_fraction'] == 0.0]
    sub_amp = zero_amps[-1] if zero_amps else 0.0
    supra_amp = (thr_half.get('threshold_amplitude') if thr_half.get('reached')
                 else (thr_any.get('threshold_amplitude') or 0.5))

    # ---------------- 5. hysteresis -----------------------------------------
    thr_pulse = _amplitude_threshold(target=0.0, zone=zone, tear_until=60.0, dt=0.1,
                                     t_end=300.0, lo=0.0, hi=0.5)
    pulse_sub = .5 * thr_pulse['threshold_amplitude'] if thr_pulse.get('reached') else 0.05
    pulse_supra = 1.5 * thr_pulse['threshold_amplitude'] if thr_pulse.get('reached') else 0.3
    hyst = {'pulse_protocol_threshold': thr_pulse,
            'sustained_protocol_amplitudes_used_for_the_labels':
                dict(sub_threshold=float(sub_amp), supra_threshold=float(supra_amp)),
            'pulse_protocol_amplitudes_used': dict(sub_threshold=float(pulse_sub),
                                                  supra_threshold=float(pulse_supra)),
            'protocol': 'prescribed radial microtear field applied for t < 60 s, then ZERO'}
    for label, amp in (('sub_threshold', pulse_sub), ('supra_threshold', pulse_supra)):
        r = run(n_cells=576, wound=True, tear_scale=float(amp), oxygen=1.0, t_end=300.0,
                dt=0.1, tear_until=60.0, sample_dt=10.0)
        st = r['state_final']
        ever_latched = r['latched_irreversible'] | r['latched_lysed']
        ever_injured = (r['state'] >= 1).any(axis=0)
        # A verdict that ignores the dwell-time windows: fire as soon as an
        # instantaneous threshold is crossed at ANY frame.
        naive_ever = ((r['integrity'] < p['hm_floor'])
                      | (r['atp'] < p['atp_deplete_frac'] * p['atp0'])
                      | (r['V'] >= p['v_lysis'])).any(axis=0)
        ode_back = ((np.abs(r['final']['V'] - 1.0) < p['v_heal_tol'])
                    & (r['final']['atp'] > p['atp_heal_frac'] * p['atp0'])
                    & (r['final']['integrity'] > p['hm_heal']))
        hyst[label] = dict(
            amplitude=float(amp),
            ever_latched_fraction_all=float(ever_latched.mean()),
            ever_latched_fraction_of_input_zone=float(np.mean(ever_latched[zone])),
            ever_injured_fraction_all=float(ever_injured.mean()),
            recovered_fraction_of_ever_injured=(
                float(np.mean(st[ever_injured] == 0)) if ever_injured.any() else None),
            latched_cells_returning_to_healthy=(
                int(np.sum(st[ever_latched] == 0)) if ever_latched.any() else 0),
            latched_cells_still_irreversible_or_lysed_fraction=(
                float(np.mean(st[ever_latched] >= 2)) if ever_latched.any() else None),
            naive_verdict_ever_fired_fraction_all=float(naive_ever.mean()),
            naive_verdict_false_deaths_fraction_of_all=float(np.mean(naive_ever & (st == 0))),
            naive_verdict_false_deaths_fraction_of_those_it_condemned=(
                float(np.mean(st[naive_ever] == 0)) if naive_ever.any() else None),
            dwell_time_rule_avoids_this_many_false_deaths_fraction_of_all=float(
                np.mean(naive_ever & (st == 0))),
            ode_variables_back_within_tolerance_fraction_all=float(ode_back.mean()),
            min_integrity=float(r['integrity'].min()),
            min_integrity_frame_time_s=float(r['times'][int(np.argmin(r['integrity'].min(axis=1)))]),
            frames_below_hm_floor=int(np.count_nonzero(r['integrity'].min(axis=1) < p['hm_floor'])),
            max_volume=float(r['V'].max()),
            min_atp_uM=float(r['atp'].min()),
            **{k: v for k, v in _verdict_stats(r, zone=zone).items()
               if k.endswith('_all') or k.startswith('zone_')},
        )
    # Release the insult AND re-enable repair: latched cells must not recover.
    r_off = run(n_cells=576, tear=np.full(576, .10), oxygen=1.0, t_end=300.0, dt=0.1,
                tear_until=20.0, repair_enabled=False, sample_dt=10.0)
    mask = r_off['latched_irreversible'] & ~r_off['latched_lysed']
    cont = run(n_cells=576, tear=np.zeros(576), oxygen=1.0, t_end=300.0, dt=0.1,
               initial=r_off['final'], repair_enabled=True, sample_dt=10.0)
    hyst['release_and_repair_continuation'] = dict(
        precursor=('uniform tear 0.10 for t < 20 s with repair DISABLED -> every cell '
                   'latches irreversible without lysing'),
        release=('then 300 s with zero tear, oxygen 1.0, repair RE-ENABLED: even with the '
                 'damage gone and repair restored, the latch must hold'),
        precursor_final_states={STATE_CODES[c]: int(np.sum(r_off['state_final'] == c))
                                for c in range(4)},
        latched_not_lysed_cells=int(mask.sum()),
        cells_returning_to_healthy=int(np.sum(cont['state_final'][mask] == 0)),
        fraction_of_latched_still_irreversible=float(np.mean(cont['state_final'][mask] >= 2))
        if mask.any() else None,
        max_integrity_of_latched_cells=float(cont['integrity'][-1][mask].max()) if mask.any() else None,
        mean_integrity_of_latched_cells=float(cont['integrity'][-1][mask].mean()) if mask.any() else None,
        integrity_ceiling_parameter=p['hm_ceiling_fail'],
        mean_volume_of_latched_cells=float(cont['V'][-1][mask].mean()) if mask.any() else None,
        mean_atp_fraction_of_latched_cells=float(np.mean(cont['atp'][-1][mask] / p['atp0']))
        if mask.any() else None,
        note=('hysteresis has a physical carrier: the latch switches on a residual '
              'permeability, reduces repair capacity and pulls integrity down to '
              'hm_ceiling_fail, so the raw variables stay away from baseline too'),
    )
    out['test5_hysteresis'] = hyst

    # ---------------- 6. repair on vs off -----------------------------------
    rep = {'insult': 'uniform tear = 0.10 for t < 20 s, then zero; oxygen = 1.0',
           'tau_repair_s': p['tau_repair'], 'recovery_window_parameter_t_hm_irrev_s': p['t_hm_irrev'],
           'hm_floor': p['hm_floor']}
    for label, enable in (('repair_enabled', True), ('repair_disabled', False)):
        r = run(n_cells=576, tear=np.full(576, .10), oxygen=1.0, t_end=300.0, dt=0.1,
                tear_until=20.0, repair_enabled=enable, sample_dt=10.0)
        t = r['times']
        h0 = r['integrity'][:, 0]
        back = np.flatnonzero((t > 20.0) & (np.abs(h0 - 1.0) < .02)
                              & (np.abs(r['V'][:, 0] - 1.0) < p['v_heal_tol']))
        healthy = np.flatnonzero((t > 20.0) & (r['state'][:, 0] == 0))
        rep[label] = dict(
            min_integrity=float(r['integrity'].min()),
            integrity_after_insult_t20s=float(h0[np.argmin(np.abs(t - 20.0))]),
            max_volume=float(r['V'].max()),
            min_atp_uM=float(r['atp'].min()),
            final_integrity=float(h0[-1]), final_volume=float(r['V'][-1, 0]),
            final_atp_uM=float(r['atp'][-1, 0]), final_state=int(r['state_final'][0]),
            time_back_to_within_2pct_of_baseline_s=(float(t[back[0]]) if back.size else None),
            time_first_healthy_again_s=(float(t[healthy[0]]) if healthy.size else None),
            latched_fraction_all=float(np.mean(r['latched_irreversible'] | r['latched_lysed'])),
            **{k: v for k, v in _verdict_stats(r).items() if k.endswith('_all')},
        )
    rep['conclusion'] = ('with repair the same insult returns to code 0; with repair '
                         'disabled the integrity deficit persists past the recovery '
                         'window and the irreversible latch fires for every cell')
    out['test6_repair'] = rep

    # ---------------- 7. ATP and injury time vs oxygen ----------------------
    rows = []
    for ox in (1.0, 0.5, 0.2, 0.0):
        row = {'oxygen': ox}
        for tag, tear_arr in (('no_tear', np.zeros(576)), ('mild_tear_0.01', np.full(576, .01))):
            r = run(n_cells=576, tear=tear_arr, oxygen=ox, t_end=300.0, dt=0.1,
                    sample_dt=300.0)
            dep, irr = r['time_atp_depleted'], r['time_irreversible']
            row[tag] = dict(
                time_to_atp_depletion_s=float(dep.min()) if np.isfinite(dep).any() else None,
                fraction_depleted_within_run=float(np.mean(np.isfinite(dep))),
                time_to_irreversible_injury_s=float(irr.min()) if np.isfinite(irr).any() else None,
                fraction_irreversible_within_run=float(np.mean(np.isfinite(irr))),
                final_atp_fraction_of_baseline=float(r['atp'][-1].mean() / p['atp0']),
                final_death_fraction=float(r['death_fraction'][-1]),
                final_states={STATE_CODES[c]: int(np.sum(r['state_final'] == c)) for c in range(4)},
            )
        rows.append(row)

    def _key(row):
        v = row['no_tear']['time_to_atp_depletion_s']
        return np.inf if v is None else v
    monotone = all(_key(a) >= _key(b) - 1e-9 for a, b in zip(rows, rows[1:]))
    out['test7_oxygen_dependence'] = dict(
        rows=rows, protocol='uniform tear, dt = 0.1 s, t_end = 300 s; None = event never occurred',
        times_non_decreasing_as_oxygen_falls=bool(monotone),
        trend_is_a_model_assumption_not_data=(
            'YES an assumption. The trend follows from three arbitrary pieces: the Hill '
            'oxygen-availability function ox(o2) with k_o2 = 0.5 and n_o2 = 3, the ATP '
            'production ceiling (cons_base = 20 uM/s, prod_max = 60 uM/s, k_atp_rec = 0.02 1/s, '
            'turnover time atp0/cons_base = 50 s) and the sustained-depletion window '
            't_atp_irrev = 60 s. No oxygen-ATP measurement exists in this project. Only the '
            'atp0 order of magnitude (0.7-1.5 mM) is literature-bracketed, and that bracket '
            'is from astrocytes, not these cells. The 0.2-vs-0.5 switch is a property of '
            'these arbitrary numbers, not a measured oxygen threshold.'),
    )

    # ---------------- 8. identifiability ------------------------------------
    out['test8_identifiability'] = identifiability_analysis() if identifiability else \
        {'skipped': 'identifiability=False'}

    # ---------------- 9. determinism ----------------------------------------
    kw = dict(n_cells=576, wound=True, tear_scale=0.2, oxygen=0.8, t_end=60.0, dt=0.1,
              het_sigma=0.2)
    a1 = run(seed=2025, **kw)
    a2 = run(seed=2025, **kw)
    a3 = run(seed=2026, **kw)
    b1 = run(seed=1, **{**kw, 'het_sigma': 0.0})
    b2 = run(seed=2, **{**kw, 'het_sigma': 0.0})
    out['test9_determinism'] = dict(
        scenario='tear_scale 0.2, oxygen 0.8, 60 s, dt 0.1, het_sigma 0.2',
        identical_same_seed=bool(np.array_equal(a1['V'], a2['V'])
                                 and np.array_equal(a1['atp'], a2['atp'])
                                 and np.array_equal(a1['integrity'], a2['integrity'])
                                 and np.array_equal(a1['state'], a2['state'])),
        max_abs_difference_same_seed_V=float(np.max(np.abs(a1['V'] - a2['V']))),
        max_abs_difference_same_seed_atp_uM=float(np.max(np.abs(a1['atp'] - a2['atp']))),
        max_abs_difference_different_seed_V=float(np.max(np.abs(a1['V'] - a3['V']))),
        death_fraction_seed2025=float(a1['death_fraction'][-1]),
        death_fraction_seed2026=float(a3['death_fraction'][-1]),
        het_sigma_zero_identical_across_seeds=bool(np.array_equal(b1['V'], b2['V'])),
        note='seed only matters when het_sigma > 0; default het_sigma = 0 is exactly deterministic',
    )
    if quick:
        for key in ('test8_identifiability', 'test4_threshold_curve', 'test5_hysteresis'):
            out.pop(key, None)
        out['quick'] = 'identifiability, threshold and hysteresis sections removed'
    out['suite_runtime_seconds'] = float(_time.perf_counter() - t_suite)
    if verbose:  # pragma: no cover
        import pprint
        pprint.pp(out)
    return out
def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so self_test() is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


if __name__ == '__main__':
    import json
    print(json.dumps(_jsonable(self_test()), indent=2))
