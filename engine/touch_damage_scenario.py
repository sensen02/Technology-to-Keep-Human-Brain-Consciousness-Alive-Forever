"""ONE scenario that puts the data-anchored neck cut and the micrometre-electrode
damage model into the same paired experiment.

THE END-TO-END QUESTION
-----------------------
Touch on the body -> ascending fibres -> brain.  What reaches the brain

    intact;  sham (probe inserted, ZERO damage);  neck_cut;
    electrode_sheath_intact;  electrode_sheath_removed;  neck_cut_plus_electrode

with the answer reported PER ROUTE and never as one aggregate brain-drive number.

WHAT THIS MODULE IS
-------------------
Two round-1 components joined, each unchanged in its own file:

* ``engine.neck_cut_data``    -- the explicit fibre set (3,686 annotated
  neck-crossing neurons; 279,796 edges / 2,606,043 synapses severed) and the
  bounded 16,458-neuron coarse conductance tier.
* ``engine.electrode_damage`` -- the sheath layer, the sheath-intact /
  sheath-removed preparation variants, the UNMEASURED free parenchymal radius,
  the species guard, and the exact cable / chemistry coupling.

This module adds only the integration: the six paired scenarios, the coupling of
the measured local electrode damage into the point tier, the exact accounting of
that coupling, the route-resolved vs aggregate demonstration, the declared
placement sweeps, and the explicitly hypothetical surgery term.

HONESTY RULES (properties of the data, not caveats to be softened)
-----------------------------------------------------------------
1.  The crossing set is the DATASET'S ANNOTATION (``ann_super_class``, cross
    checked against ``ann_class``).  No soma or arbor coordinates exist in this
    cache, so NO single cell's ascending/descending polarity is verified here.
2.  ALL EDGES ARE UNSIGNED.  ``nt_pair`` is -1 on every cached row, so every
    route number below is a SYNAPSE COUNT (a bandwidth), not excitation.
3.  The cut is an ANNOTATED-CLASS EDGE REMOVAL, not a cut plane reconstructed
    from anatomy.
4.  There is NO quantitative insect electrode-damage measurement, so every
    threshold in the damage model is a CROSS-SPECIES PROXY, and the species
    guard still fires on foreign MEASURED parameters.
5.  The parenchymal damage radius AND the sheath breach criterion are UNMEASURED.
    They are swept, never fitted.
6.  The electrode's fibre placement (fixture geometry, track position, clearance,
    and the share of the ascending fibre population the damage cylinder is
    declared to reach) is DECLARED, not measured, because the cache has no
    coordinates.  Every declaration is reported together with the number it
    produces, and the ones that matter are swept as what-ifs.
7.  THE SURGERY TERM IS UNQUANTIFIED.  In real fly preparations the sheath is
    usually removed or enzymatically breached first, so insertion damage may be
    dominated by the SURGERY, and NO insect study quantifies surgery damage.  No
    value is invented: the term is registered with ``value = None``, it is not
    modelled in any scenario, and the sweep over an assumed surgery damage radius
    is labelled WHAT-IF everywhere and is never called a result.
8.  The bounded coarse tier holds 16,458 of 153,962 neurons and carries only
    ~20% of the cut bandwidth.
9.  NO consciousness, viability, survival, recovery or rescue claim is made.
10. REACHABILITY IS NOT EVIDENCE.  Nothing here counts hops, shortest paths,
    connected components or graph distances; ``audit_reachability`` proves that
    no such quantity appears in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from .cable import CableNeuron, Morphology
from . import neck_cut_data as ncd
from .electrode_damage import (
    CableDamageConfig, CavityExpansionField, CROSS_SPECIES_CAVEAT, FlyBrainGeometry,
    NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION, STRAIN_THRESHOLD_CHOICES, SheathLayer,
    SpeciesGuard, build_registry, chemistry_config_for_variant, injury_threshold_options,
    parenchymal_damage_estimate, preparation_variants, run_paired_cable,
    run_paired_chemistry)
from .neck_cut_data import (
    BANC_RELATIVE_PATH, COUNTERFACTUAL_NOTE, NetworkRunConfig, SIGN_MODES, TouchTierConfig,
    _arrivals_iter as arrivals_iter, _stable_digest as stable_digest, build_cut_set,
    build_edges, conductance_ledger, derive_crossing_set, load_banc,
    mechanosensory_partition, presynaptic_groups, replay_severed_conductance, route_budget,
    select_touch_tier)
from .neural_cond import ConductanceNetwork, ConductanceParams
from .profile import DROSOPHILA_CNS

# ---------------------------------------------------------------------------
# scenario vocabulary
# ---------------------------------------------------------------------------
PROBE_MODES = ('none', 'inserted')
PREPARATIONS = ('sheath_intact', 'sheath_removed')
#: how the measured LOCAL electrode damage is carried into the point tier
COUPLING_MODES = ('none', 'owner_shunt', 'drive_scaled')
ACTIVATED_ROUTES = ('body_only', 'head_only', 'both')
CUT_MODES_USED = ('intact', 'neck_cut')
PRIMARY_SIGN_MODE = 'counterfactual_ach'
PRIMARY_COUPLING = 'owner_shunt'
SECONDARY_COUPLING = 'drive_scaled'

SCENARIO_NAMES = ('intact', 'sham', 'neck_cut', 'electrode_sheath_intact',
                  'electrode_sheath_removed', 'neck_cut_plus_electrode')

INTERVENTION_OF = {
    'intact': 'none',
    'sham': 'probe_inserted_with_ZERO_damage',
    'neck_cut': 'annotated_class_edge_removal',
    'electrode_sheath_intact': 'electrode_inserted_through_intact_sheath',
    'electrode_sheath_removed': 'electrode_inserted_into_sheath_removed_preparation',
    'neck_cut_plus_electrode': 'annotated_class_edge_removal_AND_electrode_inserted',
}

ELECTRODE_SCENARIOS = ('electrode_sheath_intact', 'electrode_sheath_removed',
                       'neck_cut_plus_electrode')

HONESTY_LINES = [
    'CROSSING SET = DATASET ANNOTATION (ann_super_class), cross-checked against ann_class; no '
    'soma/arbor coordinates exist in this cache, so NO single-cell polarity is verified.',
    'ALL EDGES UNSIGNED: nt_pair = -1 on every cached row; every route number is a SYNapse '
    'COUNT (a bandwidth), not excitation.',
    'The cut is an ANNOTATED-CLASS EDGE REMOVAL, not a cut plane reconstructed from anatomy.',
    'NO insect electrode-damage measurement exists, so every threshold in the damage model is a '
    'CROSS-SPECIES PROXY.',
    'The parenchymal damage radius and the sheath breach criterion are UNMEASURED; they are '
    'swept, never fitted to anything.',
    'The electrode fibre placement (fixture geometry, track position, clearance, and the share '
    'of the ascending fibre population reached) is DECLARED, not measured: the cache has no '
    'coordinates.',
    'THE SURGERY TERM IS UNQUANTIFIED: in real fly preparations the sheath is removed or '
    'enzymatically breached first, insertion damage may be dominated by the SURGERY, and no '
    'insect study quantifies surgery damage. It is NOT modelled and no value is invented for it; '
    'the sweep over an assumed surgery radius is a WHAT-IF, never a result.',
    'The species guard still fires on foreign MEASURED parameters (53 of them).',
    'The bounded coarse tier holds 16,458 of 153,962 neurons and carries only ~20% of the cut '
    'bandwidth, so the paired simulation measures about a fifth of the cut.',
    'REACHABILITY IS NOT EVIDENCE: no hop count, shortest path, component size or graph distance '
    'is used anywhere in the report.',
    'The sham is bit-identical to intact BY CONSTRUCTION because surgical damage is not modelled; '
    'a real sheath-removal surgery is exactly what would break that identity.',
    'MASK SEMANTICS: group masks are projected onto the tier with in_tier_only, so a neuron '
    'OUTSIDE the tier is EXCLUDED and can never mark a local row. The round-1 wraparound defect '
    '(local_of_global == -1 wrapping onto local row n-1) survives ONLY as the explicitly named '
    'contrast arm round1_wraparound_legacy, and the old-vs-new difference is reported in step11.',
    'No consciousness, viability, survival, recovery or rescue claim is made.',
]

#: the unquantified surgery term.  ``value`` MUST stay None.
SURGERY_TERM_NAME = 'surgery_damage_radius_um'
SURGERY_TERM = {
    'name': SURGERY_TERM_NAME,
    'status': 'UNQUANTIFIED',
    'value': None,
    'unit': 'um',
    'is_a_measurement': False,
    'is_a_fitted_illustrative_value': False,
    'modelled_in_any_scenario': False,
    'what_if_only': True,
    'why_it_exists': (
        'In real fly preparations the perineural sheath is usually removed or enzymatically '
        'breached before the electrode is inserted ("perineural sheath gently removed"; 0.5% '
        'collagenase), so the insertion damage may be a SMALLER insult than the surgery that '
        'preceded it -- and no insect study quantifies surgery damage either.'),
    'no_insect_quantification': (
        NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION +
        ' The same negative result applies to sheath removal and collagenase treatment: no '
        'insect study reports a damaged volume, a radius, or even a histology count for the '
        'surgical step alone.'),
    'why_not_modelled': (
        'Modelling it would require inventing a number. Every scenario in this report therefore '
        'contains ZERO surgical damage, which is why the sham is bit-identical to intact. In '
        'reality the sheath-removal surgery is exactly the manipulation that would break that '
        'identity, and the size of the broken identity is UNKNOWN and is not estimated here.'),
    'what_it_would_break': (
        'The sham == intact bit-identity. A surgical insult present in every probe-inserted '
        'preparation would lower the sham arm too, so "the electrode did it" could not be '
        'separated from "the surgery did it" without a measurement that does not exist.'),
    'where_it_is_swept': 'step3_surgery_term.what_if_sweep',
    'what_if_disclaimer': (
        'WHAT-IF ONLY: the sweep shows how a conclusion WOULD change as a function of an ASSUMED '
        'surgery damage radius. It is not a result, not fitted, not calibrated, and no entry of '
        'it may be quoted as a measured or illustrative value.'),
    'cross_species_context': CROSS_SPECIES_CAVEAT,
}


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------
def _nonneg(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite number')
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + ' must be finite and nonnegative')
    return float(value)


def _positive(value, name):
    v = _nonneg(value, name)
    if v <= 0:
        raise ValueError(name + ' must be strictly positive')
    return v


def _unit(value, name, *, allow_zero=True):
    v = _nonneg(value, name)
    if v > 1.0:
        raise ValueError(name + ' must lie in [0, 1]')
    if not allow_zero and v <= 0:
        raise ValueError(name + ' must be strictly positive')
    return v


def _fmt(v):
    if v is None:
        return 'NOT MODELLED'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return 'not modelled' if not math.isfinite(f) else '%.6f' % f


# ---------------------------------------------------------------------------
# group masks: the CORRECT semantics by default, and the ROUND-1 WRAPAROUND DEFECT
# ---------------------------------------------------------------------------
#: The ONE definition of the tier-local mask semantics lives in
#: ``engine.neck_cut_data.local_mask``, next to the ``local_of_global`` map it
#: consumes.  ``local_of_global`` is -1 for every neuron OUTSIDE the tier, and the
#: round-1 idiom ``m[local_of_global[np.flatnonzero(global_mask)]] = True`` therefore
#: resolves those neurons to local index -1, which NumPy wraps onto the LAST row:
#: every out-of-tier member of a group spuriously marked local row n-1.
#:
#: ROUND 1 SHIPPED THAT DEFECT.  This module now DEFAULTS to the correct semantics
#: (``in_tier_only``: out-of-tier global ids are EXCLUDED, never wrapped) for every
#: number it reports.  The legacy behaviour survives ONLY as the explicitly named
#: contrast arm ``round1_wraparound_legacy``, so that the old-vs-new difference stays
#: measurable (``mask_wraparound_audit`` and ``mask_semantics_contrast``) instead of
#: being deleted.
MASK_SEMANTICS = ncd.MASK_SEMANTICS
DEFAULT_MASK_SEMANTICS = ncd.DEFAULT_MASK_SEMANTICS
LEGACY_MASK_SEMANTICS = ncd.LEGACY_MASK_SEMANTICS


def local_mask(tier, global_mask, semantics=DEFAULT_MASK_SEMANTICS):
    """Tier-LOCAL boolean mask for a global neuron mask.

    Delegates to ``engine.neck_cut_data.local_mask``, the only definition: under the
    default ``in_tier_only`` semantics an out-of-tier global id is EXCLUDED, so the
    number of masked rows equals the number of in-tier members, and local row ``n-1``
    is marked only if it is a genuine member.
    """
    return ncd.local_mask(tier, global_mask, semantics)


def mask_membership_table(tier, groups):
    """Per named mask: membership under the correct and the legacy semantics."""
    return ncd.mask_membership_table(tier, groups)


def mask_semantics_contrast(a, tier, cut, groups, scenario=None, cfg=None, tier_cfg=None):
    """OLD-vs-NEW numbers for one arm differing ONLY in the mask semantics."""
    return ncd.mask_semantics_contrast(a, tier, cut, groups, scenario, cfg, tier_cfg)


def mask_wraparound_audit(a, tier, groups, semantics=None):
    """MEASURED size of the round-1 wraparound defect, per named mask.

    ``a`` is accepted and unused for signature continuity with round 1; the audit
    reads only the tier index map and the global group masks.  The default semantics
    ARE now the correct ones; ``masks_differ`` says which masks the legacy arm would
    have changed, and ``rows_wrapped_to_last_local_index`` is how many out-of-tier
    neurons round 1 quietly folded onto local row ``n-1``.
    """
    del a, semantics
    return ncd.mask_semantics_audit(tier, groups)


# ---------------------------------------------------------------------------
# scenario specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TouchDamageSpec:
    """One arm of the paired grid.  Only the intervention differs between arms."""

    name: str
    cut_mode: str = 'intact'
    probe: str = 'none'
    preparation: str = 'sheath_removed'
    coupling: str = 'none'
    zero_damage: bool = False
    route: str = 'both'
    sign_mode: str = 'annotation'

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.name not in SCENARIO_NAMES:
            raise ValueError('unknown scenario name %r; use %r' % (self.name, SCENARIO_NAMES))
        if self.cut_mode not in ('intact', 'sham', 'neck_cut', 'descending_only'):
            raise ValueError('unknown cut_mode %r' % (self.cut_mode,))
        if self.probe not in PROBE_MODES:
            raise ValueError('probe must be one of %r' % (PROBE_MODES,))
        if self.preparation not in PREPARATIONS:
            raise ValueError('preparation must be one of %r' % (PREPARATIONS,))
        if self.coupling not in COUPLING_MODES:
            raise ValueError('coupling must be one of %r' % (COUPLING_MODES,))
        if not isinstance(self.zero_damage, bool):
            raise ValueError('zero_damage must be a bool')
        if self.route not in ACTIVATED_ROUTES:
            raise ValueError('route must be one of %r' % (ACTIVATED_ROUTES,))
        if self.sign_mode not in SIGN_MODES:
            raise ValueError('sign_mode must be one of %r' % (SIGN_MODES,))
        if self.probe == 'none' and (self.coupling != 'none' or self.zero_damage):
            raise ValueError('a scenario with no probe cannot carry electrode damage')
        if self.zero_damage and self.coupling == 'none':
            raise ValueError('a zero-damage sham must still run the coupling code path')
        return self

    @property
    def has_electrode(self):
        return self.probe == 'inserted' and self.coupling != 'none'

    @property
    def cut_is_active(self):
        return self.cut_mode == 'neck_cut'

    @property
    def arm(self):
        return arm_name(self.name, self.coupling if self.has_electrode else 'none')

    def as_dict(self):
        return {'name': self.name, 'cut_mode': self.cut_mode, 'probe': self.probe,
                'preparation': self.preparation, 'coupling': self.coupling,
                'zero_damage': self.zero_damage, 'route': self.route,
                'sign_mode': self.sign_mode, 'arm': self.arm,
                'electrode_modelled': self.has_electrode,
                'cut_modelled': self.cut_is_active,
                'intervention': INTERVENTION_OF[self.name]}


SCENARIOS = {
    'intact': TouchDamageSpec('intact', 'intact', 'none', 'sheath_removed', 'none'),
    'sham': TouchDamageSpec('sham', 'intact', 'inserted', 'sheath_removed', 'owner_shunt',
                            zero_damage=True),
    'neck_cut': TouchDamageSpec('neck_cut', 'neck_cut', 'none', 'sheath_removed', 'none'),
    'electrode_sheath_intact': TouchDamageSpec(
        'electrode_sheath_intact', 'intact', 'inserted', 'sheath_intact', 'owner_shunt'),
    'electrode_sheath_removed': TouchDamageSpec(
        'electrode_sheath_removed', 'intact', 'inserted', 'sheath_removed', 'owner_shunt'),
    'neck_cut_plus_electrode': TouchDamageSpec(
        'neck_cut_plus_electrode', 'neck_cut', 'inserted', 'sheath_removed', 'owner_shunt'),
}
for _s in SCENARIOS.values():
    _s.validate()


def arm_name(scenario_name, coupling_mode):
    """Report key of one (scenario, coupling) arm.

    Only the ELECTRODE scenarios carry the coupling suffix, so the six required
    scenario names are themselves valid arm keys.
    """
    if scenario_name not in ELECTRODE_SCENARIOS or coupling_mode == 'none':
        return scenario_name
    return '%s@%s' % (scenario_name, coupling_mode)


def base_scenario_of(arm):
    return arm.split('@', 1)[0]


def with_arm(spec, *, route=None, sign_mode=None, coupling=None):
    """A copy of ``spec`` carrying an explicit route, sign mode and/or coupling."""
    new = replace(spec, route=spec.route if route is None else route,
                  sign_mode=spec.sign_mode if sign_mode is None else sign_mode,
                  coupling=spec.coupling if coupling is None else coupling)
    new.validate()
    return new


# ---------------------------------------------------------------------------
# declared cable fixture: the ONE fibre the track is declared to meet
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CableFixture:
    """A DECLARED fibre segment crossed by the DECLARED track.

    Nothing here is measured in a fly: the cache has no coordinates, so the fibre
    and the track are both declarations.  What IS computed from them -- the damage
    field, the reached compartments, the added leak conductance and the resulting
    local attenuation -- are model outputs with provenance attached, not fly
    measurements.
    """

    name: str
    morphology: Morphology
    shaft_point_um: tuple
    shaft_direction: tuple
    fibre_length_um: float
    fibre_diameter_um: float
    compartments: int
    crossing_fraction_of_length: float
    perpendicular_clearance_um: float
    shaft_radius_um: float
    declaration: str

    def as_dict(self):
        return {'name': self.name, 'schematic_only': True, 'declared_not_measured': True,
                'fibre_length_um': self.fibre_length_um,
                'fibre_diameter_um': self.fibre_diameter_um,
                'compartments': self.compartments,
                'shaft_radius_um': self.shaft_radius_um,
                'track_axis_point_um': list(self.shaft_point_um),
                'track_axis_direction': list(self.shaft_direction),
                'track_is_perpendicular_to_fibre': True,
                'crossing_fraction_of_length': self.crossing_fraction_of_length,
                'track_meets_fibre_at_um_along_fibre':
                    self.crossing_fraction_of_length * self.fibre_length_um,
                'perpendicular_clearance_um': self.perpendicular_clearance_um,
                'driven_end_index': 0,
                'output_end_index': self.compartments - 1,
                'declaration': self.declaration}


def declared_crossing_fixture(shaft_radius_um, *, fibre_diameter_um=0.5, length_um=500.0,
                              compartments=250, crossing_fraction_of_length=0.75,
                              extra_clearance_um=0.0, name='declared_crossing_fixture'):
    """The declared fixture: an unbranched neurite crossed by the track.

    The fibre is driven at its proximal end (index 0, the VNC side) and read at its
    terminal (index n-1, the brain side), so the reported transmission factor is the
    fraction of the drive that still arrives at the fibre's OUTPUT.
    """
    a = _positive(shaft_radius_um, 'shaft_radius_um')
    _positive(fibre_diameter_um, 'fibre_diameter_um')
    _positive(length_um, 'length_um')
    if not 0.0 <= crossing_fraction_of_length <= 1.0:
        raise ValueError('crossing_fraction_of_length must lie in [0, 1]')
    _nonneg(extra_clearance_um, 'extra_clearance_um')
    if isinstance(compartments, bool) or not isinstance(compartments, int) or compartments < 2:
        raise ValueError('compartments must be an integer >= 2')
    morph = Morphology.cylinder(float(length_um), float(fibre_diameter_um), compartments)
    clearance = a + fibre_diameter_um / 2.0 + float(extra_clearance_um)
    point = (float(crossing_fraction_of_length) * float(length_um), clearance, 0.0)
    return CableFixture(
        name=name, morphology=morph, shaft_point_um=point, shaft_direction=(0.0, 0.0, 1.0),
        fibre_length_um=float(length_um), fibre_diameter_um=float(fibre_diameter_um),
        compartments=int(compartments),
        crossing_fraction_of_length=float(crossing_fraction_of_length),
        perpendicular_clearance_um=clearance, shaft_radius_um=a,
        declaration=(
            'DECLARED, not measured: an idealised unbranched neurite of the neck-connective '
            'length scale, crossed perpendicularly by the shaft axis at %.2f of its length with '
            'the fibre lying against the shaft wall (clearance %.3f um = shaft radius + fibre '
            'radius). It stands for "the most-damaged ascending fibre the track can meet", so '
            'the attenuation it yields is an UPPER BOUND on the per-fibre effect. The cache has '
            'no coordinates to place this fibre or this track, and FAFB cannot represent these '
            'fibres at all (truncated at the neck), so no reconstructed morphology is used.'
            % (crossing_fraction_of_length, clearance)))


def threshold_for_radius(field, radius_um):
    """Back-solve the ASSUMED field's own law for the threshold that yields a radius.

    ``|eps| = delta * a / r^2`` with ``delta = wall_displacement_fraction * a``, so
    ``r = sqrt(delta * a / eps)``.  Passing an effective threshold lets the caller
    hand the cable EXACTLY a chosen radius (for example the sheath-variant
    parenchymal radius) without introducing any new number; the inversion is exact
    and every cable record reports its round-trip error.
    """
    r = _positive(radius_um, 'radius_um')
    if r < field.shaft_radius_um:
        raise ValueError(
            'the assumed field cannot represent a damage radius below the shaft radius '
            '(%.4f um): its own strain_radius_um() floors at the shaft radius, so the radius '
            'would silently not be the requested one. Radii below it are outside this fixture.'
            % field.shaft_radius_um)
    return float(field.wall_displacement_um * field.shaft_radius_um / (r * r))


# ---------------------------------------------------------------------------
# the point tier with a static per-neuron shunt
# ---------------------------------------------------------------------------
class ShuntConductanceNetwork(ConductanceNetwork):
    """``engine.neural_cond.ConductanceNetwork`` plus a static per-neuron shunt.

    The parent owns no per-neuron leak, so the measured extra leak conductance at
    the compartments the damage field reaches cannot be delivered through the
    parent's public API.  Rather than edit ``engine/neural_cond.py`` (not owned by
    this round) the integration step is re-stated here with ONE additional term::

        G_i   = leak + g_shunt_i + ge_i + gi_i
        rhs_i = (C/dt) v_i + leak * rest + g_shunt_i * E_shunt_i
                + ge_i E_exc + gi_i E_inh + I_i - adaptation_i

    Two exactness properties are asserted by ``run_touch_damage_selftest.py`` so this
    copy cannot drift away from the parent:
      * with ``g_shunt == 0`` everywhere, an identical run is bit-for-bit equal to
        the parent network's run;
      * with a UNIFORM shunt at ``E_shunt == rest``, an identical run is bit-for-bit
        equal to the parent built with ``leak_uS = leak + g_shunt``.
    """

    def __init__(self, n_neurons, params=None, dt_ms=0.1):
        super().__init__(n_neurons, params, dt_ms)
        self.g_shunt_uS = np.zeros(self.n)
        self.E_shunt_mV = np.full(self.n, self.p.rest_mV)

    def set_shunt(self, node_indices, g_uS, E_rev_mV=None):
        """Set (not increment) the static shunt at chosen point neurons.

        A zero conductance removes the shunt.  ``g_uS`` may be a scalar or one value
        per index.  ``E_rev_mV = None`` means the tier's own rest potential, which
        isolates the CONDUCTANCE effect; a real mechanoporation pore has a
        nonspecific reversal near 0 mV and would also depolarise, and that is NOT
        modelled here.
        """
        idx = np.asarray(node_indices, dtype=int)
        if idx.ndim != 1:
            raise ValueError('shunt indices must be one-dimensional')
        if idx.size and (idx.min() < 0 or idx.max() >= self.n):
            raise ValueError('shunt index out of range')
        if np.unique(idx).size != idx.size:
            raise ValueError('duplicate shunt indices would double-count the damage')
        g = np.asarray(g_uS, dtype=float)
        if g.ndim == 0:
            g = np.full(idx.size, float(g))
        if g.shape != idx.shape or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError('shunt conductance must be finite, nonnegative and match indices')
        rev = self.p.rest_mV if E_rev_mV is None else float(E_rev_mV)
        if not math.isfinite(rev):
            raise ValueError('shunt reversal must be finite')
        self.g_shunt_uS = np.zeros(self.n)
        self.E_shunt_mV = np.full(self.n, self.p.rest_mV)
        if idx.size:
            self.g_shunt_uS[idx] = g
            self.E_shunt_mV[idx] = rev
        return self

    def step(self, current_nA=0.0, exc_events_uS=0.0, inh_events_uS=0.0, enabled=None):
        I = self._vector(current_nA, 'current')
        ee = self._vector(exc_events_uS, 'exc events', True)
        ei = self._vector(inh_events_uS, 'inh events', True)
        own = np.ones(self.n, bool) if enabled is None else np.asarray(enabled)
        if own.shape != (self.n,) or own.dtype != np.dtype(bool):
            raise ValueError('ownership mask must be a boolean vector of neuron count')
        p, dt = self.p, self.dt
        arrivals = self._ring[self._ring_i].copy()
        self._ring[self._ring_i] = False
        self.ge += self.We @ arrivals + ee
        self.gi += self.Wi @ arrivals + ei
        self.ge[~own] = 0
        self.gi[~own] = 0
        active = own & (self.t_ms + 1e-10 >= self.refractory_until_ms)
        G = p.leak_uS + self.g_shunt_uS + self.ge + self.gi
        rhs = (p.capacitance_nF / dt * self.v + p.leak_uS * p.rest_mV
               + self.g_shunt_uS * self.E_shunt_mV
               + self.ge * p.exc_reversal_mV + self.gi * p.inh_reversal_mV
               + I - self.adaptation)
        self.v[active] = (rhs / (p.capacitance_nF / dt + G))[active]
        self.v[own & ~active] = p.reset_mV
        spiked = active & (self.v >= p.threshold_mV)
        self.steps += 1
        self.t_ms = self.steps * dt
        self.v[spiked] = p.reset_mV
        self.refractory_until_ms[spiked] = self.t_ms + p.refractory_ms
        self.spike_count += spiked
        self.adaptation[own] *= np.exp(-dt / p.adaptation_tau_ms)
        self.adaptation[spiked] += p.adaptation_increment_nA
        self.ge *= np.exp(-dt / p.exc_tau_ms)
        self.gi *= np.exp(-dt / p.inh_tau_ms)
        self._ring[self._ring_i] = spiked
        self._ring_i = (self._ring_i + 1) % len(self._ring)
        return spiked


# ---------------------------------------------------------------------------
# the measured electrode damage
# ---------------------------------------------------------------------------
DEFAULT_CABLE_KWARGS = dict(Cm_uF_cm2=1.0, g_leak_S_cm2=1e-4, Ra_ohm_cm=150.0,
                            E_leak_mV=-65.0)


def cable_reference():
    """Fixtures and provenance-bearing parameters, adopted once, guard-first."""
    guard_default_refusal = None
    try:
        SpeciesGuard(DROSOPHILA_CNS, build_registry(), allow_cross=False)
    except Exception as exc:                      # the refusal IS the evidence
        guard_default_refusal = '%s: %s' % (type(exc).__name__, str(exc)[:400])
    if not guard_default_refusal:
        raise AssertionError('the species guard did NOT refuse the cross-species import')
    guard = SpeciesGuard(DROSOPHILA_CNS, build_registry(), allow_cross=True)
    adopt = guard.adopt
    shaft_diameter = adopt('assumed_shaft_diameter_um').value
    return {
        'guard': guard,
        'guard_default_refused_with': guard_default_refusal,
        'adopt': adopt,
        'geometry': FlyBrainGeometry.from_adopt(adopt),
        'thresholds': injury_threshold_options(guard),
        'sheath': SheathLayer(
            thickness_um=adopt('fly_sheath_thickness_um').value,
            critical_opening_radius_um=adopt(
                'assumed_sheath_critical_opening_radius_um').value,
            delamination_multiple=adopt('assumed_sheath_delamination_multiple').value),
        'variants': preparation_variants(adopt),
        'field': CavityExpansionField(
            shaft_radius_um=shaft_diameter / 2.0,
            wall_displacement_fraction=adopt('assumed_wall_displacement_fraction').value,
            shear_modulus_Pa=adopt('assumed_tissue_shear_modulus_Pa').value),
        'shaft_diameter_um': shaft_diameter,
        'shaft_radius_um': shaft_diameter / 2.0,
        'fixture': None,   # filled by build_electrode_plan
    }


def measure_cable_damage(reference, fixture, damage_radius_um, *, leak_density_S_cm2,
                        dt_ms=0.05, duration_ms=50.0, cable_kwargs=None,
                        threshold_label=None, radius_source=None):
    """Inserted-vs-sham on the declared fixture, with every number kept exact.

    Returns the reached compartments, the extra leak conductance added AT those
    compartments (per compartment and total), the resulting local attenuation, and
    the fraction of the drive that still arrives at the fibre's OUTPUT.
    """
    field = reference['field']
    geometry = reference['geometry']
    _positive(damage_radius_um, 'damage_radius_um')
    _nonneg(leak_density_S_cm2, 'leak_density_S_cm2')
    kwargs = dict(DEFAULT_CABLE_KWARGS if cable_kwargs is None else cable_kwargs)
    radius = float(damage_radius_um)
    eps_eff = threshold_for_radius(field, radius)
    out = run_paired_cable(fixture.morphology, field, fixture.shaft_point_um,
                           fixture.shaft_direction, eps_eff,
                           config=CableDamageConfig(leak_density_S_cm2=leak_density_S_cm2),
                           dt_ms=dt_ms, duration_ms=duration_ms, cable_kwargs=kwargs)
    if abs(out['damage_radius_um'] - radius) > 1e-9 * max(1.0, radius):
        raise AssertionError('threshold/radius round trip is not exact')

    morph = fixture.morphology
    idx = np.asarray(out['damaged_node_indices'])
    cable = CableNeuron(morph, dt_ms=dt_ms, **kwargs)
    leak_per_node = kwargs['g_leak_S_cm2'] * cable.area_cm2 * 1e6
    sham_final, ins_final = out['sham'][-1], out['inserted'][-1]
    E = out['E_leak_mV']
    sham_deflection = np.abs(sham_final - E)
    ins_deflection = np.abs(ins_final - E)
    transmission = np.ones(morph.n)
    nz = sham_deflection > 1e-9
    transmission[nz] = ins_deflection[nz] / sham_deflection[nz]
    terminal = int(morph.n - 1)
    g_end = out['g_end_uS']
    rec = {
        'fixture': fixture.name,
        'threshold_label': threshold_label,
        'radius_source': radius_source,
        'damage_radius_um': radius,
        'effective_strain_threshold_used': eps_eff,
        'round_trip_radius_error_um': abs(out['damage_radius_um'] - radius),
        'leak_density_S_cm2': float(leak_density_S_cm2),
        'reached_compartments': int(idx.size),
        'compartment_index_span': [int(idx.min()), int(idx.max())] if idx.size else None,
        'reached_fraction_of_fibre_compartments': float(idx.size / morph.n),
        'extra_leak_conductance_uS_per_compartment': (
            float(g_end[idx][0]) if idx.size else 0.0),
        'extra_leak_conductance_uS_total': float(g_end.sum()),
        'intact_membrane_leak_uS_per_compartment': (
            float(leak_per_node[idx][0]) if idx.size else float(leak_per_node[0])),
        'extra_leak_over_intact_membrane_leak': (
            float(g_end[idx][0] / leak_per_node[idx][0]) if idx.size else 0.0),
        'sham_final_voltage_mV_at_output': float(sham_final[terminal]),
        'inserted_final_voltage_mV_at_output': float(ins_final[terminal]),
        'output_transmission_factor_q': float(transmission[terminal]),
        'min_per_compartment_transmission_at_reached': (
            float(transmission[idx].min()) if idx.size else 1.0),
        'max_abs_delta_mV': float(out['max_abs_delta_mV']),
        'local_attenuation_fraction_at_reached_compartments':
            float(out['max_local_attenuation_fraction']),
        'fraction_of_fibre_attenuated_below_half': float(np.mean(transmission < 0.5)),
        'input_resistance_Mohm': float(out['input_resistance_Mohm']),
        'drive_nA': float(out['drive_nA']),
        'E_leak_mV': float(E),
        'damage_radius_over_shaft_radius': float(radius / fixture.shaft_radius_um),
        'declared_track_section_area_um2': float(math.pi * radius * radius),
        'declared_full_thickness_track_volume_um3': float(
            math.pi * radius * radius * geometry.thickness_um),
        'fly_brain_volume_fraction_of_track': float(geometry.damage_fraction(radius)),
        'track_diameter_over_brain_thickness': float(
            2.0 * radius / geometry.thickness_um),
        'provenance': ('MODEL OUTPUT, not a measurement: the radius comes from a cross-species '
                       'strain proxy or from an unmeasured free parameter, the leak density is '
                       'assumed, and the fixture is a declaration.'),
    }
    return {'record': rec, 'transmission_per_compartment': transmission,
            'reached_index': idx, 'g_end_uS_per_compartment': g_end.copy(),
            'sham_final_mV': sham_final.copy(), 'inserted_final_mV': ins_final.copy(),
            'sham_traces': out['sham'].copy(), 'inserted_traces': out['inserted'].copy(),
            'distances_to_shaft_axis_um': out['distances_to_shaft_axis_um'].copy(),
            'membrane_damage_record': out['membrane_damage_record']}


def variant_damage_table(reference, fixture, *, threshold_key='optimal', leak_density=None,
                         free_radius_um=None, dt_ms=0.05, duration_ms=50.0,
                         cable_kwargs=None):
    """The sheath-intact vs sheath-removed damage table on the declared fixture."""
    adopt = reference['adopt']
    leak = (adopt('assumed_damaged_leak_density_S_cm2').value if leak_density is None
            else _nonneg(leak_density, 'leak_density_S_cm2'))
    if threshold_key not in STRAIN_THRESHOLD_CHOICES:
        raise ValueError('threshold_key must be one of %r' % (tuple(STRAIN_THRESHOLD_CHOICES),))
    record = reference['thresholds'][threshold_key]
    free_default = adopt('assumed_parenchymal_free_radius_um').value
    out = {'threshold_key': threshold_key, 'threshold_record': record.to_dict(),
           'leak_density_S_cm2': leak, 'variants': {}}
    for name, variant in reference['variants'].items():
        est = parenchymal_damage_estimate(reference['field'], record.value,
                                         reference['geometry'], reference['sheath'], variant,
                                         threshold_record=record)
        est_free = parenchymal_damage_estimate(
            reference['field'], record.value, reference['geometry'], reference['sheath'],
            variant, free_radius_um=(free_default if free_radius_um is None
                                     else free_radius_um), threshold_record=record)
        cab = measure_cable_damage(
            reference, fixture, est['parenchymal_radius_mechanistic_um'],
            leak_density_S_cm2=leak, dt_ms=dt_ms, duration_ms=duration_ms,
            cable_kwargs=cable_kwargs, threshold_label=threshold_key,
            radius_source='parenchymal_radius_mechanistic_um (mechanistic x herniation)')
        cab_free = measure_cable_damage(
            reference, fixture, est_free['parenchymal_radius_free_parameter_um'],
            leak_density_S_cm2=leak, dt_ms=dt_ms, duration_ms=duration_ms,
            cable_kwargs=cable_kwargs, threshold_label=threshold_key,
            radius_source='parenchymal_radius_free_parameter_um (UNMEASURED free parameter)')
        out['variants'][name] = {
            'variant': variant.to_dict(),
            'parenchymal_mechanistic': est,
            'parenchymal_free_parameter': est_free,
            'sheath_delamination_radius_um': reference['sheath'].delamination_radius_um(
                reference['field'].shaft_radius_um),
            'sheath_breach_area_um2': reference['sheath'].breach_area_um2(
                reference['field'].shaft_radius_um),
            'cable_mechanistic': cab['record'],
            'cable_free_parameter': cab_free['record'],
        }
        out.setdefault('_arrays', {})[name] = {'mechanistic': cab, 'free_parameter': cab_free}
    return out


def chemical_damage_fraction(radius_um, element_side_um=8.0):
    """Fraction of a DECLARED local tissue element inside the damage cylinder.

    Same 8x8x8 um element as round 1, and the same cap at 1.0 that
    ``ChemistryDamageConfig.scaled_values`` enforces.  Saturation is reported rather
    than hidden: above ~4.51 um the whole element is inside the cylinder.
    """
    r = _positive(radius_um, 'radius_um')
    side = _positive(element_side_um, 'element_side_um')
    raw = math.pi * r * r / (side * side)
    return {'element_side_um': side, 'element_area_um2': side * side,
            'raw_fraction': float(raw), 'fraction_used': float(min(1.0, raw)),
            'saturated': bool(raw >= 1.0),
            'saturation_radius_um': float(side / math.sqrt(math.pi)),
            'note': ('DECLARED element size; the fraction is a free parameter of the coupling, '
                     'not a fly measurement')}


def measure_chemistry(reference, *, damage_fraction_by_variant, dt_s=0.05, duration_s=20.0,
                      fractions_sweep=(0.1, 0.5, 1.0)):
    """The K / barrier chemistry change through the existing injury_tissue API."""
    out = {'api': 'engine.injury_tissue, driven by engine.electrode_damage.run_paired_chemistry',
           'barrier_edge': [1, 2], 'dt_s': dt_s, 'duration_s': duration_s,
           'per_variant': {}, 'fraction_sweep': {},
           'fly_note': ('for Drosophila this is a glial/tracheal sheath restriction, NOT a '
                        'vertebrate blood-brain barrier')}
    for name, variant in reference['variants'].items():
        frac = _unit(damage_fraction_by_variant[name], 'damage_fraction')
        cfg = chemistry_config_for_variant(variant, reference['adopt'])
        res = run_paired_chemistry(frac, config=cfg, dt_s=dt_s, duration_s=duration_s,
                                   include_supported=True)
        closed = res['variants']['closed']
        supported = res['variants']['supported']
        out['per_variant'][name] = {
            'damage_fraction_used': frac,
            'chemistry_config': cfg.to_dict(),
            'variant_definition': variant.to_dict(),
            'closed': {
                'sham_coupling': closed['sham_coupling'],
                'inserted_coupling': closed['inserted_coupling'],
                'sham_permeability_um_s': closed['sham_permeability_um_s'],
                'inserted_permeability_um_s': closed['inserted_permeability_um_s'],
                'permeability_change_um_s': (closed['inserted_permeability_um_s']
                                             - closed['sham_permeability_um_s']),
                'sheath_conductance_change_um3_s': (
                    closed['inserted_coupling']['sheath_conductance_um3_s']
                    - closed['sham_coupling']['sheath_conductance_um3_s']),
                'final_Ki_sham_mM': closed['final_Ki_sham_mM'],
                'final_Ki_inserted_mM': closed['final_Ki_inserted_mM'],
                'final_Ke_sham_mM': closed['final_Ke_sham_mM'],
                'final_Ke_inserted_mM': closed['final_Ke_inserted_mM'],
                'delta_Ke_mM': closed['final_Ke_inserted_mM'] - closed['final_Ke_sham_mM'],
                'final_Ek_sham_mV': closed['final_Ek_sham_mV'],
                'final_Ek_inserted_mV': closed['final_Ek_inserted_mV'],
                'delta_Ek_mV': closed['final_Ek_inserted_mV'] - closed['final_Ek_sham_mV'],
                'sham_far_ligand_nM': closed['sham_far_ligand_nM'],
                'inserted_far_ligand_nM': closed['inserted_far_ligand_nM'],
                'delta_far_ligand_nM': (closed['inserted_far_ligand_nM']
                                        - closed['sham_far_ligand_nM']),
                'max_potassium_mass_error': closed['max_potassium_mass_error'],
                'max_ligand_mass_error': closed['max_ligand_mass_error'],
                'sham_sheath_events': closed['sham_sheath_events'],
                'inserted_sheath_events': closed['inserted_sheath_events'],
                'sham_sheath_event_count': len(closed['sham_sheath_events']),
                'inserted_sheath_event_count': len(closed['inserted_sheath_events']),
                'sham_has_exactly_zero_exchange': bool(
                    closed['sham_permeability_um_s'] == 0.0
                    and closed['sham_coupling']['sheath_conductance_um3_s'] == 0.0)
                if variant.sheath_present else bool(
                    closed['sham_permeability_um_s'] == 0.0),
            },
            'supported': {
                'delta_Ke_mM': supported['final_Ke_inserted_mM'] - supported['final_Ke_sham_mM'],
                'delta_Ek_mV': (supported['final_Ek_inserted_mV']
                                - supported['final_Ek_sham_mV']),
                'delta_far_ligand_nM': (supported['inserted_far_ligand_nM']
                                        - supported['sham_far_ligand_nM']),
                'max_potassium_mass_error': supported['max_potassium_mass_error'],
                'max_ligand_mass_error': supported['max_ligand_mass_error'],
            },
            '_arrays': {k: v['traces'] for k, v in res['variants'].items()},
            'closed_note': ('the closed variant is a reduced two-pool model with no pumps, no '
                            'buffer and no reservoir, so its long-time K redistribution is a '
                            'documented model artefact; the buffered+reservoir variant is '
                            'reported next to it'),
        }
    for frac in fractions_sweep:
        f = _unit(frac, 'chemistry fraction', allow_zero=False)
        row = {}
        for name, variant in reference['variants'].items():
            res = run_paired_chemistry(f, config=chemistry_config_for_variant(
                variant, reference['adopt']), dt_s=dt_s, duration_s=duration_s,
                include_supported=False)
            closed = res['variants']['closed']
            row[name] = {
                'damage_fraction': f,
                'delta_Ke_mM': closed['final_Ke_inserted_mM'] - closed['final_Ke_sham_mM'],
                'delta_Ek_mV': closed['final_Ek_inserted_mV'] - closed['final_Ek_sham_mV'],
                'sheath_conductance_change_um3_s': (
                    closed['inserted_coupling']['sheath_conductance_um3_s']
                    - closed['sham_coupling']['sheath_conductance_um3_s']),
                'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
            }
        out['fraction_sweep']['%.3f' % f] = row
    return out


# ---------------------------------------------------------------------------
# declared placement: which tier neurons the damage field is declared to reach
# ---------------------------------------------------------------------------
def reached_owner_local(tier, groups, route_group='crossing_ascending'):
    """Tier-LOCAL indices of the fibre population the track is DECLARED to meet.

    DECLARED: the track is placed on the ascending-fibre population -- the
    dataset-annotated ``crossing_ascending`` cells -- because that population is
    the pathway under test and the cache has no coordinates with which to place the
    track anywhere else.  Any other placement would be equally arbitrary, so the
    choice is named, its size is reported, and the share of it the damage cylinder
    is declared to reach is swept as a what-if.

    Only neurons that really are inside the tier are returned (``in_tier_only``
    semantics), so the count can be compared directly with
    ``tier['counts']['crossing_neurons_in_tier_ascending_side']``.
    """
    overlapping = groups['route_overlapping'].get(route_group)
    if overlapping is None:
        raise ValueError('unknown route group %r' % (route_group,))
    loc = np.asarray(tier['local_of_global'])
    gi = np.flatnonzero(np.asarray(overlapping))
    local = loc[gi]
    # ONLY in-tier neurons: local_of_global is -1 outside the tier and indexing
    # with -1 would wrap onto local row n-1 (the round-1 defect measured below).
    return np.unique(local[local >= 0]).astype(int)


def deterministic_share_subset(local_idx, share):
    """A deterministic prefix of the sorted owner list, of size ceil(share*N).

    The ordering carries NO biological meaning; it exists only so that a run is
    reproducible.  Which fibres a real cylinder intersects is unknown, so the share
    is a declared parameter.
    """
    idx = np.sort(np.asarray(local_idx, dtype=int))
    s = _unit(share, 'share', allow_zero=False)
    if idx.size == 0:
        raise ValueError('no reached owners to subsample')
    return idx[:max(1, min(idx.size, int(math.ceil(s * idx.size))))]


# ---------------------------------------------------------------------------
# coupling the measured electrode damage into the tier
# ---------------------------------------------------------------------------
def build_coupling(mode, reached_local, *, tier_n, per_compartment_g_uS=None,
                   fibre_total_g_uS=None, fibre_reached_compartments=None, q=None,
                   E_shunt_mV=None):
    """One exact coupling of the measured local damage into the point tier."""
    if mode not in COUPLING_MODES:
        raise ValueError('coupling mode must be one of %r' % (COUPLING_MODES,))
    idx = np.asarray(reached_local, dtype=int)
    if idx.ndim != 1 or np.unique(idx).size != idx.size:
        raise ValueError('reached_local must be a unique one-dimensional index array')
    if idx.size and (idx.min() < 0 or idx.max() >= int(tier_n)):
        raise ValueError('reached local index out of range')
    if mode == 'none':
        return {'mode': 'none', 'reached_local': idx, 'g_shunt_uS': 0.0, 'q': 1.0,
                'tier_n': int(tier_n), 'E_shunt_mV': None, 'no_op': True, 'stage': 'phase',
                'declaration': 'no electrode damage in this arm'}
    if mode == 'owner_shunt':
        per_comp = _nonneg(per_compartment_g_uS, 'per_compartment_g_uS')
        total = per_comp if fibre_total_g_uS is None else _nonneg(fibre_total_g_uS,
                                                                 'fibre_total_g_uS')
        g = total
        return {'mode': mode, 'reached_local': idx, 'g_shunt_uS': g, 'q': 1.0,
                'tier_n': int(tier_n), 'E_shunt_mV': E_shunt_mV, 'no_op': bool(g == 0.0),
                'stage': 'phase',
                'g_shunt_basis': 'fibre total over its reached compartments',
                'g_shunt_per_compartment_uS': per_comp,
                'fibre_reached_compartments': (None if fibre_reached_compartments is None
                                               else int(fibre_reached_compartments)),
                'g_shunt_over_tier_cell_leak': float(g / 0.001) if g else 0.0,
                'declaration': ('one reached point owner stands for its WHOLE fibre, so it '
                                'receives the MEASURED total extra leak conductance summed over '
                                'that fibre\'s reached compartments (as a static shunt at the '
                                'tier rest potential). The per-compartment value is reported too; '
                                'using only one compartment would understate the leak by the '
                                'reached-compartment count.')}
    qv = _unit(q, 'q')
    return {'mode': mode, 'reached_local': idx, 'g_shunt_uS': 0.0, 'q': qv,
            'tier_n': int(tier_n), 'E_shunt_mV': None, 'no_op': bool(qv == 1.0),
            'stage': 'phase',
            'declaration': ('EVERY row out of a reached owner -- excitatory and inhibitory alike '
                            '-- is multiplied by the measured output transmission factor q of the '
                            'declared fibre, because a quieter fibre releases less transmitter of '
                            'whatever sign its verified label gives. It is a pure rescaling of '
                            'the drive those fibres deliver and adds nothing to the membrane.')}


def reached_column_mask(matrix, reach):
    """CSR column mask: True at stored entries whose presynaptic index is reached."""
    reach = np.asarray(reach, dtype=int)
    if reach.size == 0:
        return np.zeros(matrix.data.size, bool)
    return np.isin(matrix.indices, reach)


def scale_columns_in_place(matrix, column_mask, factor):
    """A new matrix with the SAME sparsity pattern and ``data[column_mask]*factor``.

    Scaling the DATA cannot add, remove or reorder a stored entry, which is what
    makes the pattern-preservation claim in the selftest exact.
    """
    data = matrix.data.copy()
    data[column_mask] *= factor
    out = sparse.csr_matrix((data, matrix.indices.copy(), matrix.indptr.copy()),
                            shape=matrix.shape)
    out.has_canonical_format = matrix.has_canonical_format
    return out


def couple_edge_set(edges, coupling):
    """Apply one coupling to an edge set: exact, and a strict no-op when q == 1.

    ``drive_scaled`` multiplies EVERY row out of a reached owner by the measured
    output transmission factor q -- excitatory and inhibitory alike, because a
    quieter fibre releases less transmitter of whatever sign its verified label
    gives.  The unscaled matrices (same objects, same bytes) are returned whenever
    the coupling is a no-op or reaches nothing, which is what makes the sham
    bit-identical to intact and what makes the neck-cut arm's reached set provably
    empty.
    """
    reach = np.asarray(coupling['reached_local'], dtype=int)
    We, Wi = edges['We'], edges['Wi']
    hit_e = reached_column_mask(We, reach)
    hit_i = reached_column_mask(Wi, reach)
    total = float(We.data.sum() + Wi.data.sum())
    reach_uS = float(We.data[hit_e].sum() + Wi.data[hit_i].sum())
    out = {'We': We, 'Wi': Wi, 'rows_scaled': 0, 'rows_scaled_inhibitory': 0,
           'conductance_before_uS': total, 'conductance_after_uS': total,
           'damage_reached_conductance_uS': reach_uS,
           'damage_unreached_conductance_uS': total - reach_uS,
           'scaled_matrix_is_new_object': False, 'weight_identity_verified': True,
           'linear_identity_residual_uS': 0.0,
           'reached_columns_with_no_stored_row': bool(not hit_e.any() and not hit_i.any())}
    if coupling['mode'] == 'drive_scaled' and not coupling['no_op'] \
            and (hit_e.any() or hit_i.any()):
        q = coupling['q']
        We_s = scale_columns_in_place(We, hit_e, q)
        Wi_s = scale_columns_in_place(Wi, hit_i, q)
        after = float(We_s.data.sum() + Wi_s.data.sum())
        expected = q * reach_uS + (total - reach_uS)
        out.update({'We': We_s, 'Wi': Wi_s,
                    'rows_scaled': int(np.count_nonzero(hit_e)),
                    'rows_scaled_inhibitory': int(np.count_nonzero(hit_i)),
                    'conductance_after_uS': after,
                    'scaled_matrix_is_new_object': True,
                    'weight_identity_verified': bool(
                        abs(after - expected) <= 1e-9 * max(1.0, abs(after))),
                    'linear_identity_residual_uS': abs(after - expected)})
    return out


# ---------------------------------------------------------------------------
# the electrode plan: measurements -> couplings
# ---------------------------------------------------------------------------
#: declared share of the ascending fibre population the cylinder is declared to
#: reach.  1.0 is the MAXIMUM-DAMAGE declaration (an upper bound); the geometric
#: volume fraction of the track is reported next to it as a reference point.
DEFAULT_REACH_SHARE = 1.0
#: shares used by the declared-placement what-if sweep
REACH_SHARE_SWEEP = (0.03116659378561303, 0.1, 0.25, 0.5, 1.0)


def build_electrode_plan(reference, *, threshold_key='optimal', leak_density=None,
                         dt_ms=0.05, duration_ms=50.0, cable_kwargs=None,
                         fixture=None):
    """Measure the electrode damage once, on the declared fixture."""
    fixture = fixture or declared_crossing_fixture(reference['shaft_radius_um'])
    filled = dict(reference)
    filled['fixture'] = fixture
    table = variant_damage_table(filled, fixture, threshold_key=threshold_key,
                                leak_density=leak_density, dt_ms=dt_ms,
                                duration_ms=duration_ms, cable_kwargs=cable_kwargs)
    return {'reference': filled, 'fixture': fixture, 'table': table}


def zero_damage_sham_cable(reference, fixture, damage_radius_um, *, dt_ms=0.05,
                           duration_ms=50.0, cable_kwargs=None):
    """The sham arm: the SAME geometry and code path with a leak density of ZERO.

    The radius is the paired electrode arm's own modelled radius, so the sham
    differs from that arm in exactly one number -- the leak density -- and the
    returned zeros are measured zeros rather than an assumption.
    """
    rec = measure_cable_damage(
        reference, fixture, damage_radius_um, leak_density_S_cm2=0.0, dt_ms=dt_ms,
        duration_ms=duration_ms, cable_kwargs=cable_kwargs, threshold_label='sham',
        radius_source=('sham: the paired electrode arm\'s own modelled radius with a damaged '
                       'leak density of exactly zero S/cm2'))['record']
    if rec['extra_leak_conductance_uS_total'] != 0.0:
        raise AssertionError('the sham cable must add exactly zero conductance')
    if rec['output_transmission_factor_q'] != 1.0:
        raise AssertionError('the sham cable must transmit exactly 1.0')
    return rec


def build_couplings(reference, fixture, cable_table, tier, groups, *, reach_share, tier_n):
    """All couplings used by the grid, from the measured cable numbers only."""
    owners_all = reached_owner_local(tier, groups)
    owners = deterministic_share_subset(owners_all, reach_share)
    out = {'reached_owners_all': int(owners_all.size), 'reached_owners_used': int(owners.size),
           'reach_share': float(reach_share),
           'reached_owners_are_declared': True,
           'reached_owner_group': 'crossing_ascending (dataset annotation), tier-local',
           'by_scenario': {}}
    for name in ELECTRODE_SCENARIOS:
        variant = SCENARIOS[name].preparation
        rec = cable_table['variants'][variant]['cable_mechanistic']
        out['by_scenario'][name] = {
            'preparation': variant,
            'cable_mechanistic': rec,
            'owner_shunt': build_coupling(
                'owner_shunt', owners, tier_n=tier_n,
                per_compartment_g_uS=rec['extra_leak_conductance_uS_per_compartment'],
                fibre_total_g_uS=rec['extra_leak_conductance_uS_total'],
                fibre_reached_compartments=rec['reached_compartments']),
            'drive_scaled': build_coupling(
                'drive_scaled', owners, tier_n=tier_n,
                q=rec['output_transmission_factor_q']),
        }
    sham_radius = cable_table['variants'][SCENARIOS['sham'].preparation][
        'cable_mechanistic']['damage_radius_um']
    out['by_scenario']['sham'] = {
        'preparation': SCENARIOS['sham'].preparation,
        'cable_mechanistic': zero_damage_sham_cable(reference, fixture, sham_radius),
        'owner_shunt': build_coupling('owner_shunt', owners, tier_n=tier_n,
                                      per_compartment_g_uS=0.0),
    }
    for name in ('intact', 'neck_cut'):
        noop = build_coupling('none', np.zeros(0, int), tier_n=tier_n)
        out['by_scenario'][name] = {'owner_shunt': noop, 'none': noop}
    return out


# ---------------------------------------------------------------------------
# one paired scenario run
# ---------------------------------------------------------------------------
def run_damage_scenario(a, tier, cut, groups, spec, coupling, cfg=None, tier_cfg=None,
                        keep_spikes=False, mask_semantics=DEFAULT_MASK_SEMANTICS):
    """Run one arm of the grid with the SAME initialisation and seed as every other.

    The pre-phase matrices are always the INTACT ones, so the pre-phase spike train
    is bit-identical across every arm; the intervention (annotated-class edge
    removal and/or the measured electrode coupling) is applied at the phase step.
    """
    c = cfg or NetworkRunConfig()
    c.validate()
    spec.validate()
    tcfg = tier_cfg or TouchTierConfig()
    nn = int(tier['n'])
    steps, phase_step, settle_step = c.steps, c.phase_step, c.settle_step
    delay_steps = c.delay_steps

    intact = build_edges(a, tier, cut, spec.sign_mode, 'intact', tcfg)
    active_edges = (build_edges(a, tier, cut, spec.sign_mode, 'neck_cut', tcfg)
                    if spec.cut_mode == 'neck_cut' else intact)
    no_coupling = {'mode': 'none', 'reached_local': np.zeros(0, int), 'q': 1.0,
                   'g_shunt_uS': 0.0, 'no_op': True}
    active_coupled = couple_edge_set(active_edges, coupling if spec.has_electrode
                                     else no_coupling)

    def _mask(global_mask):
        # the CORRECT semantics by default; the round-1 defect only when named
        # explicitly through mask_semantics=LEGACY_MASK_SEMANTICS
        return local_mask(tier, global_mask, mask_semantics)

    mech = mechanosensory_partition(a)
    body_local = np.flatnonzero(_mask(mech['body_route']))
    head_local = np.flatnonzero(_mask(mech['head_route']))
    group_local = {name: np.flatnonzero(_mask(m)) for name, m in groups['disjoint'].items()}
    spike_groups = {name: np.flatnonzero(_mask(m))
                    for name, m in groups['route_overlapping'].items()}
    target_local = spike_groups['brain_side']
    ascending_local = spike_groups['crossing_ascending']

    params = ConductanceParams()
    net = ShuntConductanceNetwork(nn, params, c.dt_ms)
    net.set_connectivity(intact['We'], intact['Wi'])
    rng = np.random.default_rng(c.seed)
    background = rng.normal(c.background_mean_nA, c.background_sd_nA, nn)
    body_cur = np.zeros(nn)
    head_cur = np.zeros(nn)
    if spec.route in ('both', 'body_only'):
        body_cur[body_local] = c.body_touch_current_nA
    if spec.route in ('both', 'head_only'):
        head_cur[head_local] = c.head_touch_current_nA

    init_digest = stable_digest(net.v, net.ge, net.gi, net.adaptation,
                               intact['We'].indptr, intact['We'].indices, intact['We'].data,
                               intact['Wi'].indptr, intact['Wi'].indices, intact['Wi'].data,
                               background, body_cur, head_cur)

    spike_matrix = np.zeros((steps, nn), dtype=np.uint8)
    events = []
    shunt_owners = 0
    swap_connectivity = spec.cut_mode == 'neck_cut' or (
        spec.has_electrode and coupling['mode'] == 'drive_scaled' and not coupling['no_op'])
    for k in range(steps):
        t = k * c.dt_ms
        cur = background.copy()
        if t >= c.stim_start_ms and ((t - c.stim_start_ms) % c.stim_period_ms) < c.stim_duration_ms:
            cur = cur + body_cur + head_cur
        spike_matrix[k] = net.step(cur)
        if k != phase_step:
            continue
        if swap_connectivity:
            net.set_connectivity(active_coupled['We'], active_coupled['Wi'])
        if spec.cut_mode == 'neck_cut':
            events.append({'event_id': 'phase:0', 'time_ms': t,
                           'kind': 'annotated_class_edge_removal_ascending_and_descending',
                           'rows_removed': int(
                               active_edges['excluded']['rows_removed_by_cut_mode']),
                           'note': ('outgoing rows of annotated neck-crossing neurons are '
                                    'absent from the edge set from here on; NOT a cut plane')})
        if spec.has_electrode and coupling['mode'] == 'owner_shunt' and not coupling['no_op']:
            net.set_shunt(coupling['reached_local'], coupling['g_shunt_uS'],
                          coupling['E_shunt_mV'])
            shunt_owners = int(coupling['reached_local'].size)
            events.append({'event_id': 'phase:1', 'time_ms': t,
                           'kind': 'electrode_measured_leak_shunt_on_reached_owners',
                           'owners_shunted': shunt_owners,
                           'g_shunt_uS_per_owner': float(coupling['g_shunt_uS']),
                           'note': ('the extra leak conductance measured at the compartments '
                                    'the damage field reaches')})
        elif spec.has_electrode and coupling['mode'] == 'drive_scaled' and not coupling['no_op']:
            events.append({'event_id': 'phase:1', 'time_ms': t,
                           'kind': 'electrode_measured_output_transmission_scaling',
                           'rows_scaled': active_coupled['rows_scaled'],
                           'q': float(coupling['q']),
                           'note': ('rows whose presynaptic owner is reached are scaled by the '
                                    'measured output transmission factor')})
        elif spec.cut_mode != 'neck_cut':
            if spec.name == 'sham':
                kind, note = 'sham_noop', ('the scheduled phase was reached and no row and no '
                                           'conductance changed')
            elif spec.has_electrode:
                kind, note = ('electrode_reached_zero_compartments_noop',
                              'the damage radius reaches no compartment of the declared fibre, '
                              'so the electrode arm is a no-op')
            else:
                kind, note = 'no_intervention', 'intact reference run'
            events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': kind,
                           'rows_removed': 0, 'note': note})

    windows = {'baseline': (0, phase_step), 'post': (phase_step, steps),
               'settled': (settle_step, steps)}
    spike_counts = {}
    for wname, (lo, hi) in windows.items():
        seg = spike_matrix[lo:hi]
        spike_counts[wname] = {
            'total': int(seg.sum()),
            'per_group': {name: int(seg[:, ix].sum()) for name, ix in spike_groups.items()},
            'per_neuron': seg.sum(axis=0).astype(np.int64),
        }
    ledger, ledger_asc = {}, {}
    for wname, (lo, hi) in windows.items():
        ledger[wname] = conductance_ledger(spike_matrix, active_coupled['We'],
                                          active_coupled['Wi'], target_local, group_local,
                                          delay_steps, step_slice=slice(lo, hi))
        ledger_asc[wname] = conductance_ledger(spike_matrix, active_coupled['We'],
                                              active_coupled['Wi'], ascending_local,
                                              group_local, delay_steps,
                                              step_slice=slice(lo, hi))

    result = {
        'scenario': spec.as_dict(),
        'arm': spec.arm,
        'coupling': {k: v for k, v in coupling.items() if k != 'reached_local'},
        'coupling_reached_owners': int(coupling['reached_local'].size),
        'coupling_no_op': bool(coupling['no_op']),
        'shunt_owners_at_phase': shunt_owners,
        'neural_steps': int(steps),
        'neurons': nn,
        'edges': {'exc_nnz': int(active_coupled['We'].nnz),
                  'inh_nnz': int(active_coupled['Wi'].nnz),
                  'exc_uS': float(active_coupled['We'].data.sum()),
                  'inh_uS': float(active_coupled['Wi'].data.sum()),
                  'exc_uS_uncoupled': float(active_edges['sum_exc_uS']),
                  'rows_removed_by_cut_mode':
                      active_edges['excluded']['rows_removed_by_cut_mode'],
                  'rows_removed_synapses':
                      active_edges['excluded']['rows_removed_by_cut_mode_synapses']},
        'edge_coupling_accounting': {k: v for k, v in active_coupled.items()
                                    if k not in ('We', 'Wi')},
        'excluded_edge_accounting': active_edges['excluded'],
        'spike_counts': spike_counts,
        'conductance_at_brain_side_uS': ledger,
        'conductance_at_ascending_side_uS': ledger_asc,
        'brain_side_total_spikes_settled':
            spike_counts['settled']['per_group'].get('brain_side', 0),
        'events': events,
        'init_digest_sha256_32': init_digest,
        'baseline_spike_digest_sha256_32': stable_digest(spike_matrix[:phase_step]),
        'full_spike_digest_sha256_32': stable_digest(spike_matrix),
        'counterfactual': (COUNTERFACTUAL_NOTE if spec.sign_mode == 'counterfactual_ach'
                           else None),
        'conductance_params': {'leak_uS': params.leak_uS, 'rest_mV': params.rest_mV,
                               'threshold_mV': params.threshold_mV,
                               'capacitance_nF': params.capacitance_nF},
    }
    if keep_spikes:
        result['_spike_matrix'] = spike_matrix
    return result


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------
def run_grid(a, tier, cut, groups, couplings, *, sign_modes=SIGN_MODES,
             routes=ACTIVATED_ROUTES, cfg=None, tier_cfg=None,
             mask_semantics=DEFAULT_MASK_SEMANTICS, extra_spike_scenarios=(),
             extra_spike_sign_mode=PRIMARY_SIGN_MODE, extra_spike_route='both'):
    """Run every arm of the six-scenario grid for every active route.

    ``couplings[scenario_name][coupling_mode]`` supplies the coupling dict.  The
    electrode scenarios are run under BOTH exact couplings, so no conclusion can be
    an artefact of one coupling choice; the other scenarios have a single arm.

    The intact spike train of every (sign mode, route) is kept, because the exact
    attribution replays it.  ``extra_spike_scenarios`` additionally keeps the spike
    matrices of the given scenarios for one (sign mode, route) pair, so byte-level
    comparisons are possible without retaining the whole grid.
    """
    results, spike_mats, extra_mats = {}, {}, {}
    plan = []
    for name, spec in SCENARIOS.items():
        modes = ([PRIMARY_COUPLING, SECONDARY_COUPLING] if name in ELECTRODE_SCENARIOS
                 else [spec.coupling])
        for mode in modes:
            plan.append((name, mode))
    for sign_mode in sign_modes:
        for route in routes:
            for name, mode in plan:
                spec = with_arm(SCENARIOS[name], route=route, sign_mode=sign_mode,
                                coupling=mode)
                coupling = couplings['by_scenario'][name][mode]
                keep = (name == 'intact' or (
                    name in extra_spike_scenarios and sign_mode == extra_spike_sign_mode
                    and route == extra_spike_route))
                res = run_damage_scenario(a, tier, cut, groups, spec, coupling, cfg, tier_cfg,
                                          keep_spikes=keep, mask_semantics=mask_semantics)
                results[(sign_mode, route, spec.arm)] = res
                if keep:
                    matrix = res.pop('_spike_matrix')
                    if name == 'intact':
                        spike_mats[(sign_mode, route)] = matrix
                    else:
                        extra_mats[(sign_mode, route, spec.arm)] = matrix
                elif '_spike_matrix' in res:
                    res.pop('_spike_matrix')
    return {'results': results, 'intact_spike_matrices': spike_mats,
            'extra_spike_matrices': extra_mats, 'plan': plan}


def exact_attribution(a, tier, cut, groups, spike_matrix, intact_coupling, cfg=None,
                      tier_cfg=None, sign_mode=PRIMARY_SIGN_MODE,
                      mask_semantics=DEFAULT_MASK_SEMANTICS):
    """Both exact attributions with ACTIVITY HELD FIXED, from one recorded train.

    * the neck cut: the parent's ``replay_severed_conductance``;
    * the electrode: this module's ``replay_electrode_conductance``.
    """
    return {
        'neck_cut': replay_severed_conductance(a, tier, cut, groups, spike_matrix,
                                              config=cfg, tier_cfg=tier_cfg,
                                              sign_mode=sign_mode),
        'electrode_drive_scaled': replay_electrode_conductance(
            a, tier, cut, groups, spike_matrix, intact_coupling, config=cfg,
            tier_cfg=tier_cfg, sign_mode=sign_mode, mask_semantics=mask_semantics),
    }


# ---------------------------------------------------------------------------
# the pipeline: one call builds everything the report and the tests need
# ---------------------------------------------------------------------------
def build_context(root, *, reach_share=DEFAULT_REACH_SHARE, threshold_key='optimal',
                  leak_density=None, sign_modes=SIGN_MODES, routes=ACTIVATED_ROUTES,
                  run_cfg=None, tier_cfg=None, dt_ms=0.05, duration_ms=50.0,
                  cable_kwargs=None, extra_spike_scenarios=()):
    """Load the data and build every fixture, measurement, coupling and run."""
    root = Path(root)
    cfg = run_cfg or NetworkRunConfig()
    tcfg = tier_cfg or TouchTierConfig()
    a = load_banc(root / BANC_RELATIVE_PATH)
    mech = mechanosensory_partition(a)
    crossing = derive_crossing_set(a)
    cut = build_cut_set(a, crossing)
    budget = route_budget(a, cut, mech)
    tier = select_touch_tier(a, mech, cut, tcfg)
    groups = presynaptic_groups(a, mech, cut)
    reference = cable_reference()
    plan = build_electrode_plan(reference, threshold_key=threshold_key,
                               leak_density=leak_density, dt_ms=dt_ms,
                               duration_ms=duration_ms, cable_kwargs=cable_kwargs)
    reference, fixture, table = plan['reference'], plan['fixture'], plan['table']
    couplings = build_couplings(reference, fixture, table, tier, groups,
                               reach_share=reach_share, tier_n=tier['n'])
    grid = run_grid(a, tier, cut, groups, couplings, sign_modes=sign_modes, routes=routes,
                    cfg=cfg, tier_cfg=tcfg, extra_spike_scenarios=extra_spike_scenarios)
    if not ncd.tier_index_map_is_consistent(tier):
        raise AssertionError('the tier local_of_global map is not a bijection onto [0, n)')
    return {'root': root, 'a': a, 'mech': mech, 'crossing': crossing, 'cut': cut,
            'budget': budget, 'tier': tier, 'groups': groups, 'cfg': cfg,
            'tier_cfg': tcfg, 'reference': reference, 'fixture': fixture,
            'cable_table': table, 'couplings': couplings, 'grid': grid,
            'reach_share': float(reach_share),
            'mask_semantics': DEFAULT_MASK_SEMANTICS,
            'legacy_mask_semantics_arm': LEGACY_MASK_SEMANTICS,
            'mask_membership_table': ncd.mask_membership_table(tier, groups),
            'mask_semantics_audit': ncd.mask_semantics_audit(tier, groups)}


def route_drive_table(grid, routes=ACTIVATED_ROUTES):
    """route -> arm -> the numbers the aggregate-vs-route demonstration needs."""
    table = {}
    for (sign_mode, route, arm), res in grid['results'].items():
        table.setdefault(sign_mode, {}).setdefault(route, {})[arm] = {
            'brain_side_drive_uS_settled': float(
                sum(res['conductance_at_brain_side_uS']['settled'].values())),
            'brain_side_drive_uS_post': float(
                sum(res['conductance_at_brain_side_uS']['post'].values())),
            'ascending_side_drive_uS_settled': float(
                sum(res['conductance_at_ascending_side_uS']['settled'].values())),
            'brain_side_spikes_settled': res['spike_counts']['settled']['per_group']['brain_side'],
            'brain_side_spikes_post': res['spike_counts']['post']['per_group']['brain_side'],
            'total_spikes_settled': res['spike_counts']['settled']['total'],
            'body_touch_sensor_spikes_settled':
                res['spike_counts']['settled']['per_group']['body_touch_sensor'],
            'head_touch_sensor_spikes_settled':
                res['spike_counts']['settled']['per_group']['head_touch_sensor'],
            'crossing_ascending_spikes_settled':
                res['spike_counts']['settled']['per_group']['crossing_ascending'],
            'crossing_descending_spikes_settled':
                res['spike_counts']['settled']['per_group']['crossing_descending'],
            'rows_removed_by_cut_mode': res['edges']['rows_removed_by_cut_mode'],
            'synapses_removed_by_cut_mode': res['edges']['rows_removed_synapses'],
            'init_digest_sha256_32': res['init_digest_sha256_32'],
            'baseline_spike_digest_sha256_32': res['baseline_spike_digest_sha256_32'],
            'full_spike_digest_sha256_32': res['full_spike_digest_sha256_32'],
            'events': res['events'],
        }
    return table


# ---------------------------------------------------------------------------
# exact attribution with activity held fixed
# ---------------------------------------------------------------------------
REPLAY_WINDOWS = ('full', 'post', 'settled')


def replay_electrode_conductance(a, tier, cut, groups, spike_matrix, coupling, config=None,
                                tier_cfg=None, sign_mode='counterfactual_ach',
                                mask_semantics=DEFAULT_MASK_SEMANTICS, window='full'):
    """EXACT conductance the reached fibres deliver, with ACTIVITY HELD FIXED.

    Only defined for the ``drive_scaled`` coupling, which is a matrix change:
    replaying one recorded spike train through the unscaled and the scaled matrix
    isolates the electrode's effect on delivered drive from every network-feedback
    effect.  The linear identity
    ``q * reached_at_q1 + unreached == electrode_total`` is returned so the caller
    can assert it.
    """
    if sign_mode not in SIGN_MODES:
        raise ValueError('sign_mode must be one of %r' % (SIGN_MODES,))
    if coupling['mode'] != 'drive_scaled':
        raise ValueError('exact replay is defined only for the drive_scaled coupling')
    c = config or NetworkRunConfig()
    c.validate()
    tcfg = tier_cfg or TouchTierConfig()
    base = build_edges(a, tier, cut, sign_mode, 'intact', tcfg)
    scaled = couple_edge_set(base, coupling)
    nn = int(tier['n'])

    def _mask(global_mask):
        # the CORRECT semantics by default; the round-1 defect only when named
        # explicitly through mask_semantics=LEGACY_MASK_SEMANTICS
        return local_mask(tier, global_mask, mask_semantics)

    group_local = {name: np.flatnonzero(_mask(m)) for name, m in groups['disjoint'].items()}
    brain_local = np.flatnonzero(_mask(groups['route_overlapping']['brain_side']))
    reach = np.asarray(coupling['reached_local'], dtype=int)
    mask = np.zeros(base['We'].shape[1])
    mask[reach] = 1.0
    masked = sparse.diags(mask, format='csc')
    We_reached = (base['We'] @ masked).tocsr()
    Wi_reached = (base['Wi'] @ masked).tocsr()
    if window not in REPLAY_WINDOWS:
        raise ValueError('window must be one of %r' % (REPLAY_WINDOWS,))
    steps = spike_matrix.shape[0]
    sl = {'full': slice(0, steps), 'post': slice(c.phase_step, steps),
          'settled': slice(c.settle_step, steps)}[window]
    l_base = conductance_ledger(spike_matrix, base['We'], base['Wi'], brain_local, group_local,
                               c.delay_steps, step_slice=sl)
    # BOTH matrices: the coupling scales every row out of a reached owner, excitatory and
    # inhibitory alike, so replaying the scaled run with the UNSCALED inhibitory matrix
    # would break the identity below (this was a real defect, found by T4.4).
    l_scaled = conductance_ledger(spike_matrix, scaled['We'], scaled['Wi'], brain_local,
                                 group_local, c.delay_steps, step_slice=sl)
    l_reached = conductance_ledger(spike_matrix, We_reached, Wi_reached, brain_local,
                                  group_local, c.delay_steps, step_slice=sl)
    l_unreached = {k: l_base[k] - l_reached[k] for k in l_base}
    total_base = float(sum(l_base.values()))
    total_scaled = float(sum(l_scaled.values()))
    total_reached = float(sum(l_reached.values()))
    predicted = coupling['q'] * total_reached + (total_base - total_reached)
    return {
        'method': ('one recorded spike train replayed through the unscaled and the scaled matrix; '
                   'the difference is the drive the reached fibres deliver, with activity fixed'),
        'sign_mode': sign_mode, 'coupling_mode': coupling['mode'], 'q': float(coupling['q']),
        'window': window, 'window_is_the_full_run': window == 'full',
        'reached_owners': int(reach.size),
        'rows_touched_excitatory': int(np.count_nonzero(reached_column_mask(base['We'], reach))),
        'rows_touched_inhibitory': int(np.count_nonzero(reached_column_mask(base['Wi'], reach))),
        'at_brain_side_uS': {
            'intact_total': total_base,
            'electrode_total': total_scaled,
            'reached_rows_at_q1_total': total_reached,
            'unreached_rows_total': total_base - total_reached,
            'attenuated_by_reached_rows_total': total_base - total_scaled,
            'predicted_electrode_total': predicted,
            'per_group_intact': l_base, 'per_group_electrode': l_scaled,
            'per_group_reached_rows_at_q1': l_reached,
            'per_group_unreached_rows': l_unreached,
            'linear_identity_residual_uS': abs(predicted - total_scaled),
            'linear_identity_relative_residual': (abs(predicted - total_scaled)
                                                  / max(1e-12, abs(total_scaled))),
        },
        'reachability_used': False,
    }


# ---------------------------------------------------------------------------
# exact per-owner attribution, and the declared-share what-if in retention terms
# ---------------------------------------------------------------------------
def brain_side_owner_contributions(a, tier, cut, groups, spike_matrix, config=None,
                                   tier_cfg=None, sign_mode=PRIMARY_SIGN_MODE,
                                   window='settled', mask_semantics=DEFAULT_MASK_SEMANTICS):
    """EXACT conductance delivered to brain-side cells, split by PRESYNAPTIC owner.

    The parent network's own delay ring is reused, so
    ``sum(contributions)`` reproduces ``conductance_ledger``'s total exactly (the
    selftest asserts it).  It exists so that a DECLARED reached subset can be
    re-scored without re-simulating: the ledger is linear in the arrivals.
    """
    c = config or NetworkRunConfig()
    c.validate()
    tcfg = tier_cfg or TouchTierConfig()
    if sign_mode not in SIGN_MODES:
        raise ValueError('sign_mode must be one of %r' % (SIGN_MODES,))
    edges = build_edges(a, tier, cut, sign_mode, 'intact', tcfg)
    nn = int(tier['n'])

    def _mask(global_mask):
        # the CORRECT semantics by default; the round-1 defect only when named
        # explicitly through mask_semantics=LEGACY_MASK_SEMANTICS
        return local_mask(tier, global_mask, mask_semantics)

    brain_local = np.flatnonzero(_mask(groups['route_overlapping']['brain_side']))
    group_local = {name: np.flatnonzero(_mask(m)) for name, m in groups['disjoint'].items()}
    col_e = np.asarray(edges['We'][brain_local, :].sum(axis=0)).ravel()
    col_i = np.asarray(edges['Wi'][brain_local, :].sum(axis=0)).ravel()
    windows = {'baseline': (0, c.phase_step), 'post': (c.phase_step, c.steps),
               'settled': (c.settle_step, c.steps)}
    if window not in windows:
        raise ValueError('window must be one of %r' % (tuple(windows),))
    lo, hi = windows[window]
    counts = np.zeros(nn, dtype=np.int64)
    for k, arr in enumerate(arrivals_iter(spike_matrix, c.delay_steps)):
        if k < lo or k >= hi:
            continue
        counts += arr
    per_owner = (col_e + col_i) * counts
    by_group = {name: float(per_owner[ix].sum()) for name, ix in group_local.items()}
    return {'window': window, 'sign_mode': sign_mode,
            'brain_side_local': brain_local, 'group_local': group_local,
            'per_owner_uS': per_owner, 'total_uS': float(per_owner.sum()),
            'per_disjoint_group_uS': by_group,
            'arrivals_inside_window_per_owner': counts}


def retention_vs_reached_owners(contrib, owners_all, q, *, shares=REACH_SHARE_SWEEP,
                                share_note='', reference_total=None):
    """Route-drive retention as a function of the DECLARED reached share.

    WHAT-IF only: ``q`` is the measured output transmission factor of the declared
    fibre and ``owners_all`` is the declared fibre population, but WHICH fibres a
    real cylinder intersects is unknown, so every row carries ``what_if: True``.
    The arithmetic is exact: the ledger is linear in the arrivals, so
    ``retained = (q * reached + (total - reached)) / total``.
    """
    per_owner = np.asarray(contrib['per_owner_uS'], dtype=float)
    idx = np.sort(np.asarray(owners_all, dtype=int))
    if idx.size == 0:
        raise ValueError('no reached owners')
    total = float(per_owner.sum()) if reference_total is None else float(reference_total)
    qv = _unit(q, 'q')
    rows = []
    for s in shares:
        sv = _unit(s, 'share', allow_zero=True)
        k = int(math.ceil(sv * idx.size))
        reached = idx[:k]
        reached_uS = float(per_owner[reached].sum()) if k else 0.0
        retained = ((qv * reached_uS + (total - reached_uS)) / total) if total else None
        rows.append({'declared_reached_share': sv,
                     'declared_reached_owners': int(k),
                     'reached_owner_drive_uS': reached_uS,
                     'reached_share_of_brain_side_drive': (reached_uS / total) if total else None,
                     'body_route_drive_retained': retained,
                     'body_route_loss_fraction': None if retained is None else 1.0 - retained,
                     'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
                     'is_a_fitted_illustrative_value': False,
                     'note': share_note})
    return {'q_used': qv, 'owners_available': int(idx.size), 'brain_side_drive_uS': total,
            'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
            'why': ('which ascending fibres a real damage cylinder intersects is unknown: the '
                    'cache has no coordinates, so the reached share is a declaration and is '
                    'swept instead of assumed'),
            'rows': rows}


# ---------------------------------------------------------------------------
# the aggregate-vs-route demonstration
# ---------------------------------------------------------------------------
def aggregate_hides_route_loss(sign_mode, route_table, scenario_arm, baseline_arm='intact'):
    """Aggregate retention vs the route-resolved pair, with the hiding quantified.

    ``route_table[route][arm]['brain_side_drive_uS_settled']`` is the delivered
    drive at brain-side cells.  The aggregate arm drives BOTH routes; the
    route-resolved pair is the same intervention with only one route driven.  The
    aggregate is dominated by the head route and therefore understates the
    body-route loss.
    """
    def drive(route, arm):
        entry = route_table.get(route, {}).get(arm)
        return None if entry is None else entry['brain_side_drive_uS_settled']

    out = {'intervention_arm': scenario_arm, 'sign_mode': sign_mode, 'baseline_arm': baseline_arm}
    for route in ACTIVATED_ROUTES:
        base = drive(route, baseline_arm)
        cur = drive(route, scenario_arm)
        out['%s_drive_uS' % route] = {
            'intact': base, 'intervened': cur,
            'retained_fraction': ((cur / base) if base not in (None, 0.0) else None)}
    agg = out['both_drive_uS']['retained_fraction']
    body = out['body_only_drive_uS']['retained_fraction']
    head = out['head_only_drive_uS']['retained_fraction']
    out['aggregate_is_both_routes_driven'] = True
    out['route_resolved_pair'] = {'body_route_retained': body, 'head_route_retained': head}
    if agg is not None and body is not None:
        out['aggregate_minus_body_retained'] = agg - body
        out['aggregate_understates_the_body_route_loss_by_percentage_points'] = 100.0 * (
            agg - body)
        # the flag says: the aggregate reads more than half, the body route alone keeps
        # less than half, and the two differ by more than 25 percentage points -- which is
        # exactly the situation in which quoting the aggregate changes the conclusion.
        out['aggregate_hides_the_body_route_loss'] = bool(
            agg > 0.5 and body < 0.5 and (agg - body) > 0.25)
        out['body_route_loss_fraction'] = 1.0 - body
        out['aggregate_loss_fraction'] = 1.0 - agg
        out['aggregate_understates_the_loss_by_percentage_points'] = 100.0 * (
            (1.0 - body) - (1.0 - agg))
    if agg is not None and head is not None:
        out['aggregate_minus_head_retained'] = agg - head
    out['statement'] = (
        'an aggregate "total brain drive retained" is NOT sufficient evidence: with BOTH routes '
        'driven it reads %s, while the body route alone retains %s and the head route retains %s. '
        'The aggregate therefore hides a %.6f body-route loss behind a %.6f aggregate loss.'
        % (_fmt(agg), _fmt(body), _fmt(head),
           0.0 if body is None else 1.0 - body, 0.0 if agg is None else 1.0 - agg))
    return out


# ---------------------------------------------------------------------------
# sweeps of the DECLARED placement and of the ASSUMED parameters
# ---------------------------------------------------------------------------
def declared_parameter_sweeps(reference, fixture, *, dt_ms=0.05, duration_ms=50.0,
                              cable_kwargs=None):
    """Sensitivities of the model's own declarations and assumed parameters.

    These are geometric and parameter sensitivities, NOT measurements, results or
    fits: every row carries ``what_if: True`` and ``is_a_result: False``, and none of
    them may be quoted as a fly number.
    """
    adopt = reference['adopt']
    leak = adopt('assumed_damaged_leak_density_S_cm2').value
    sheath = reference['sheath']
    a = reference['field'].shaft_radius_um
    variants = reference['variants']
    optimum = reference['thresholds']['optimal']
    radii = {name: parenchymal_damage_estimate(
        reference['field'], optimum.value, reference['geometry'], sheath, v,
        threshold_record=optimum)['parenchymal_radius_mechanistic_um']
        for name, v in variants.items()}
    brief = ('reached_compartments', 'extra_leak_conductance_uS_per_compartment',
             'extra_leak_conductance_uS_total', 'output_transmission_factor_q',
             'local_attenuation_fraction_at_reached_compartments')

    def cab_row(radius, leak_density_value, fixture_used=None):
        rec = measure_cable_damage(reference, fixture_used or fixture, radius,
                                  leak_density_S_cm2=leak_density_value, dt_ms=dt_ms,
                                  duration_ms=duration_ms,
                                  cable_kwargs=cable_kwargs)['record']
        return {k: rec[k] for k in brief}

    out = {}

    # 1. the ASSUMED damaged leak density: the parameter that decides q
    dens = [1e-3, 1e-2, 5e-2, 1e-1, leak]
    out['assumed_leak_density_S_cm2'] = {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': ('the damaged leak density is ASSUMED (round 1); it is the single parameter that '
                'decides whether a reached fibre is merely attenuated or functionally blocked, '
                'so it is swept instead of quoted'),
        'primary_value_S_cm2': leak, 'values_S_cm2': dens, 'by_variant': {}}
    for name, radius in radii.items():
        out['assumed_leak_density_S_cm2']['by_variant'][name] = [
            dict(leak_density_S_cm2=d, damage_radius_um=radius,
                 what_if=True, is_a_result=False, is_a_measurement=False,
                 **cab_row(radius, d)) for d in dens]

    # 2. the DECLARED clearance between fibre and shaft axis
    clear = [0.0, 2.25, 5.0, 10.0, 15.0]
    out['declared_fibre_clearance'] = {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': ('how far the fibre lies from the shaft axis is unknown (no coordinates); the '
                'primary declaration puts the fibre against the wall, which maximises the '
                'reached compartment count and therefore the attenuation'),
        'extra_clearance_um': clear, 'by_variant': {}}
    for name, radius in radii.items():
        rows = []
        for extra in clear:
            alt = declared_crossing_fixture(a, extra_clearance_um=extra, name=fixture.name)
            rows.append({'extra_clearance_um': extra,
                         'perpendicular_clearance_um': alt.perpendicular_clearance_um,
                         'damage_radius_um': radius,
                         'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
                         **cab_row(radius, leak, alt)})
        out['declared_fibre_clearance']['by_variant'][name] = rows

    # 3. the DECLARED position of the track along the fibre
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    out['declared_track_position_along_fibre'] = {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': 'the track position along the fibre is unknown; the primary declaration is 0.75',
        'crossing_fraction_of_length': pos, 'primary_value': fixture.crossing_fraction_of_length,
        'by_variant': {}}
    for name, radius in radii.items():
        rows = []
        for q in pos:
            alt = declared_crossing_fixture(a, crossing_fraction_of_length=q,
                                            name=fixture.name)
            rows.append({'crossing_fraction_of_length': q, 'damage_radius_um': radius,
                         'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
                         **cab_row(radius, leak, alt)})
        out['declared_track_position_along_fibre']['by_variant'][name] = rows

    # 4. the CROSS-SPECIES strain threshold bracket
    out['cross_species_strain_threshold'] = {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': ('every strain threshold is a rodent proxy and there is no insect measurement, so '
                'the bracket is reported instead of one number'),
        'rows': []}
    for key, record in reference['thresholds'].items():
        if key.startswith('_'):
            continue
        for name, v in variants.items():
            est = parenchymal_damage_estimate(reference['field'], record.value,
                                             reference['geometry'], sheath, v,
                                             threshold_record=record)
            row = {'variant': name, 'threshold_key': key, 'threshold': record.value,
                   'threshold_species': record.species, 'threshold_source': record.source,
                   'parenchymal_radius_um': est['parenchymal_radius_mechanistic_um'],
                   'fly_brain_volume_fraction': est['parenchymal_fraction_mechanistic'],
                   'what_if': True, 'is_a_result': False, 'is_a_measurement': False}
            row.update(cab_row(est['parenchymal_radius_mechanistic_um'], leak))
            out['cross_species_strain_threshold']['rows'].append(row)

    # 5. the ASSUMED herniation multiplier: exactly 1 collapses the two variants
    mults = [1.0, 2.0, 5.0, 10.0]
    out['assumed_sheath_herniation_multiplier'] = {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': ('the herniation multiplier is ASSUMED and is the ONLY reason the intact-sheath '
                'variant differs from the removed-sheath variant; at exactly 1.0 the two '
                'variants coincide by construction, which the selftest asserts'),
        'values': mults, 'rows': []}
    for m in mults:
        for name in ('sheath_intact', 'sheath_removed'):
            v = variants[name]
            mult = m if v.sheath_present else 1.0
            est = parenchymal_damage_estimate(reference['field'], optimum.value,
                                             reference['geometry'], sheath, v,
                                             threshold_record=optimum)
            radius = est['mechanistic_damage_radius_um'] * mult
            row = {'assumed_multiplier': m, 'variant': name, 'multiplier_applied': mult,
                   'parenchymal_radius_um': radius,
                   'what_if': True, 'is_a_result': False, 'is_a_measurement': False}
            row.update(cab_row(radius, leak))
            out['assumed_sheath_herniation_multiplier']['rows'].append(row)
    return out


# ---------------------------------------------------------------------------
# the surgery what-if (HYPOTHETICAL ONLY, never a result)
# ---------------------------------------------------------------------------
#: purely hypothetical radii; NO value here is measured, fitted or claimed.
#: The lower bound is the declared fixture's own representability limit (a radius
#: below the 5 um shaft radius cannot be expressed through the field's radius law),
#: NOT a claim about surgery.
SURGERY_WHAT_IF_RADII_UM = (0.0, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)


def surgery_what_if(reference, fixture, *, radii_um=SURGERY_WHAT_IF_RADII_UM,
                    leak_density_S_cm2=None, dt_ms=0.05, duration_ms=50.0,
                    cable_kwargs=None):
    """WHAT-IF ONLY: retention vs an ASSUMED surgery damage radius.

    The mechanism is the SAME declared cable fixture used for the electrode, driven
    by an ASSUMED radius instead of a modelled one.  No number in the returned block
    is a measurement or a fitted illustrative value: every row carries
    ``what_if: True``, ``is_a_result: False`` and ``is_a_fitted_illustrative_value:
    False``, and the term it sweeps has ``value = None`` because no insect study
    quantifies surgical damage.
    """
    leak = (reference['adopt']('assumed_damaged_leak_density_S_cm2').value
            if leak_density_S_cm2 is None else _nonneg(leak_density_S_cm2,
                                                      'leak_density_S_cm2'))
    rows = []
    for r in radii_um:
        rr = _nonneg(r, 'assumed surgery radius')
        if rr == 0.0:
            rows.append({'assumed_surgery_damage_radius_um': 0.0,
                         'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
                         'is_a_fitted_illustrative_value': False,
                         'reached_compartments': 0,
                         'extra_leak_conductance_uS_total': 0.0,
                         'output_transmission_factor_q': 1.0,
                         'local_attenuation_fraction': 0.0,
                         'fly_brain_volume_fraction_of_track': 0.0,
                         'body_route_drive_retained_if_this_were_true': 1.0,
                         'reads_as': ('assumed ZERO surgical damage, which is what every '
                                      'modelled scenario in this report contains')})
            continue
        rec = measure_cable_damage(
            reference, fixture, rr, leak_density_S_cm2=leak, dt_ms=dt_ms,
            duration_ms=duration_ms, cable_kwargs=cable_kwargs,
            radius_source='ASSUMED surgery damage radius (WHAT-IF, not a measurement)')['record']
        q = rec['output_transmission_factor_q']
        rows.append({'assumed_surgery_damage_radius_um': rr,
                     'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
                     'is_a_fitted_illustrative_value': False,
                     'reached_compartments': rec['reached_compartments'],
                     'extra_leak_conductance_uS_total': rec[
                         'extra_leak_conductance_uS_total'],
                     'output_transmission_factor_q': q,
                     'local_attenuation_fraction': rec[
                         'local_attenuation_fraction_at_reached_compartments'],
                     'fly_brain_volume_fraction_of_track': rec[
                         'fly_brain_volume_fraction_of_track'],
                     'body_route_drive_retained_if_this_were_true': q,
                     'reads_as': ('IF the surgical step alone added a %.1f um damage radius to '
                                  'the same declared fibre, the drive that fibre delivers would '
                                  'be scaled by the measured q of that geometry' % rr)})
    return {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'is_a_fitted_illustrative_value': False,
        'term': SURGERY_TERM_NAME, 'term_value_used': None,
        'mechanism': ('the same declared cable fixture, driven by an ASSUMED radius instead of a '
                      'modelled one; the output transmission factor q of that geometry is what '
                      'is reported'),
        'lowest_representable_radius_um': float(reference['field'].shaft_radius_um),
        'lower_bound_is_NOT_a_biological_claim': (
            'the sweep starts at the shaft radius only because the assumed field floors its own '
            'damage radius at the shaft radius; a surgical insult smaller than the electrode '
            'footprint is simply not representable in this fixture, which is a limitation of the '
            'fixture and not a statement about surgery'),
        'what_the_sweep_shows': ('how far an assumed surgical insult would have to reach before '
                                 'it reproduced the electrode\'s own effect on the body-touch '
                                 'route. It does NOT say how far real surgery reaches.'),
        'rows': rows,
    }


# ---------------------------------------------------------------------------
# report audits
# ---------------------------------------------------------------------------
REACHABILITY_PATTERNS = ('reachab', 'hop_count', 'n_hops', 'shortest_path', 'path_length',
                         'path_len', 'component_size', 'connected_component', 'betweenness',
                         'eccentricity', 'graph_distance', 'graph_hops')


def audit_reachability(obj, path='$'):
    """Every NUMERIC value under a reachability-named key is a violation.

    Booleans and ``None`` are allowed: the report must be able to *state* that no
    reachability quantity is used.  A number is not allowed.
    """
    violations = []

    def walk(node, p):
        if isinstance(node, dict):
            for key, value in node.items():
                kp = '%s.%s' % (p, key)
                low = str(key).lower()
                if any(pat in low for pat in REACHABILITY_PATTERNS) and \
                        isinstance(value, (int, float)) and not isinstance(value, bool):
                    violations.append({'path': kp, 'value': value,
                                       'why': 'a numeric reachability quantity may not be '
                                              'evidence'})
                walk(value, kp)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, '%s[%d]' % (p, i))
    walk(obj, path)
    return violations


SURGERY_KEY_PATTERNS = ('surg', 'collagenase', 'sheath_removal_damage', 'surgical')


def audit_surgery_claims(obj, path='$'):
    """No numeric surgery damage may be claimed as measured or illustrative-fitted.

    A numeric value under a surgery-named key is allowed ONLY inside a subtree that
    declares ``what_if: True``, and no dict inside such a subtree may declare
    ``is_a_result: True`` or ``is_a_measurement: True``.
    """
    violations = []

    def walk(node, p, inside):
        if isinstance(node, dict):
            here = inside or node.get('what_if') is True
            if here:
                if node.get('is_a_result') is True:
                    violations.append({'path': p, 'why': 'a what-if entry may not claim to be a '
                                                         'result'})
                if node.get('is_a_measurement') is True:
                    violations.append({'path': p, 'why': 'a what-if entry may not claim to be a '
                                                         'measurement'})
                if node.get('is_a_fitted_illustrative_value') is True:
                    violations.append({'path': p, 'why': 'a what-if entry may not claim to be '
                                                         'fitted'})
            for key, value in node.items():
                kp = '%s.%s' % (p, key)
                low = str(key).lower()
                named = any(pat in low for pat in SURGERY_KEY_PATTERNS)
                if named and isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and not here:
                    violations.append({'path': kp, 'value': value,
                                       'why': 'a numeric surgery damage value outside a what-if '
                                              'subtree would read as measured or fitted'})
                walk(value, kp, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, '%s[%d]' % (p, i), inside)
    walk(obj, path, False)
    return violations


def audit_surgery_term(term=None):
    """The registered surgery term must stay unquantified."""
    violations = []
    t = SURGERY_TERM if term is None else term
    if t.get('name') != SURGERY_TERM_NAME:
        violations.append('the surgery term must be registered under its own name')
    if t.get('value') is not None:
        violations.append('the surgery term must carry value=None: no insect study quantifies '
                          'surgical damage')
    if 'UNQUANTIFIED' not in str(t.get('status', '')):
        violations.append('the surgery term status must contain UNQUANTIFIED')
    for flag in ('is_a_measurement', 'is_a_fitted_illustrative_value',
                 'modelled_in_any_scenario'):
        if t.get(flag) is not False:
            violations.append('surgery term flag %s must be False' % flag)
    if t.get('what_if_only') is not True:
        violations.append('the surgery term must be flagged what_if_only')
    if not str(t.get('no_insect_quantification', '')).strip():
        violations.append('the surgery term must carry the no-insect-quantification evidence')
    if not str(t.get('why_not_modelled', '')).strip():
        violations.append('the surgery term must state why it is not modelled')
    return violations
