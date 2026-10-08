"""Data-anchored neck cut for the BANC body-touch -> ascending -> brain route.

WHAT THIS MODULE IS
-------------------
A data layer plus a bounded coarse conductance tier that lets the abstract
"neck disconnect" be replaced by an EXPLICIT FIBRE SET re-derived from
``data/flywire/banc_connectome.npz``, and lets the effect of severing that set be
MEASURED IN SYNAPSE-WEIGHTED UNITS rather than in reachability.

Pipeline (each step is a separate, independently testable function):

  1. ``derive_crossing_set``   -- re-derive the neck-crossing population from the
     RAW cached arrays (not from any previously written npz) and compare the
     count with the published numbers.
  2. ``build_cut_set``         -- the cut as a fibre set: the set of OUTGOING
     edges of those neurons, with synapse counts, split ascending-side vs
     descending-side and made exhaustive and non-overlapping by construction.
  3. ``route_budget``          -- the synapse-weighted body-touch vs head-touch
     route budget and how much of each route the cut removes.
  4. ``select_touch_tier``     -- a deterministic, documented, size-bounded
     (<= TouchTierConfig.max_neurons) coarse tier restricted to the body-touch
     route, the head-touch route and their partners.
  5. ``run_scenario``          -- one paired run of the bounded conductance tier
     under an explicit (sign_mode, cut_mode, activated_route) scenario.

================================================================================
HONESTY RULES (these are properties of the data, not caveats to be softened)
================================================================================
* The crossing set is the DATASET'S ANNOTATION (``ann_super_class``, cross-checked
  against ``ann_class``). No soma and no arbor coordinates exist in this cache, so
  NO SINGLE CELL'S POLARITY IS INDEPENDENTLY VERIFIED here.
* ALL EDGES ARE UNSIGNED. ``nt_pair`` is -1 on every one of the 3,037,361 rows of
  the cached snapshot; it is NEVER read for a sign. Every number in the route
  budget is therefore a SYNAPSE COUNT, i.e. a bandwidth, not an excitation.
* Conductance sign comes ONLY from a pure ``ann_nt_verified`` presynaptic label:
  acetylcholine -> excitatory (E = 0 mV), gaba / glutamate -> inhibitory
  (E = -70 mV). Every unknown, mixed, modulator and histamine-only row is
  EXCLUDED and COUNTED, never guessed.
* The cut is an ANNOTATED-CLASS EDGE REMOVAL, not a cut plane reconstructed from
  anatomy. It severs the outgoing edges of annotated cells; it does not know
  where the cervix is.
* FAFB is TRUNCATED AT THE NECK, so no FAFB-only morphology can represent these
  fibres; cross-neck work must use BANC/MANC/FANC.
* The cached snapshot disagrees with the current FlyWire ``synapse_table`` build
  for at least one measured cell, so every count here is CONDITIONAL on the
  cached snapshot.
* NO consciousness, viability, survival, recovery or rescue claim is made
  anywhere. "Cut" means "these rows are absent from the edge set".
* GROUP MASKS ARE PROJECTED CORRECTLY. ``local_of_global`` is -1 for every neuron
  OUTSIDE the tier, and the round-1 idiom
  ``m[local_of_global[np.flatnonzero(global_mask)]] = True`` wrapped each of those
  neurons onto local row n-1. Every reported number now uses
  ``in_tier_only`` (out-of-tier ids EXCLUDED, never wrapped); the round-1 behaviour
  survives ONLY as the explicitly named legacy contrast arm
  ``round1_wraparound_legacy``, and its old-vs-new effect is measured rather than
  deleted.
* REACHABILITY IS NOT USED AS EVIDENCE. Connected-component sizes are reported
  only to characterise the tier's structure; no reachability, hop-count or
  shortest-path quantity is ever computed on a touch source, and none appears in
  any measurement of the cut's effect. What is measured is (a) synapse-weighted
  route budgets, (b) conductance-weighted delivered drive, (c) spike counts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy import sparse

from .neural_cond import ConductanceNetwork, ConductanceParams

BANC_RELATIVE_PATH = 'data/flywire/banc_connectome.npz'

# ---------------------------------------------------------------------------
# annotation vocabulary (the parent analysis used exactly these strings)
# ---------------------------------------------------------------------------
CROSSING_ASCENDING = ('ascending', 'sensory_ascending',
                      'ascending_visceral_circulatory')
CROSSING_DESCENDING = ('descending', 'sensory_descending')
CROSSING_POPULATIONS = CROSSING_ASCENDING + CROSSING_DESCENDING
BRAIN_SIDE_POPULATIONS = ('central_brain_intrinsic', 'optic_lobe_intrinsic',
                          'visual_projection', 'visual_centrifugal')
VNC_POPULATION = 'ventral_nerve_cord_intrinsic'
MOTOR_POPULATION = 'motor'

MECH_LABELS = ('bristle_neuron', 'chordotonal_organ_neuron',
               'campaniform_sensillum_neuron', 'hair_plate_neuron',
               'multidendritic_neuron', 'taste_bristle_tactile_neuron')
GUSTATORY_LABELS = ('taste_bristle_gustatory_neuron',)
BODY_NERVE_KEYWORDS = ('leg_nerve', 'abdominal_nerve', 'dorsal_mesothoracic_nerve',
                       'dorsal_metathoracic_nerve', 'thoracic')
HEAD_NERVE_KEYWORDS = ('antennal_nerve', 'eye_nerve', 'maxillary-labial_nerve',
                       'occipital_nerve', 'frontal_nerve')

#: Published external anchors, as transcribed in
#: ``outputs/brain_isolation/EXTERNAL_EVIDENCE_TOUCH_ELECTRODE.md``. The paper's
#: per-population numbers are listed separately because they do NOT sum to the
#: paper's total, which matters for an honest comparison.
PUBLISHED_ANCHORS = {
    'banc_backbone_total': {'value': 3686, 'claim': 'BANC neck-connective proofread backbone',
                            'source': 'Bates et al. 2026, Nature, PMC13518251'},
    'histology_axons_total': {'value': 3738, 'claim': 'axons counted between brain and VNC via the neck connective',
                              'source': 'Phelps et al. 2021, Cell, PMC8312698'},
    'paper_component_counts': {'ascending': 1849, 'descending': 1316, 'sensory_ascending': 517,
                               'source': 'Bates et al. 2026 (as transcribed by the parent analysis)',
                               'note': 'these three components sum to 3682, which is not the paper total 3686'},
    'multi_dataset_ranges': {'AN': '1733-1865', 'DN': '1315-1347', 'SA': '535-611',
                             'source': 'Sturner et al. 2025, Nature, PMC12222017'},
}

SIGN_MODES = ('annotation', 'counterfactual_ach')
CUT_MODES = ('intact', 'sham', 'neck_cut', 'descending_only')
ACTIVATED_ROUTES = ('both', 'body_only', 'head_only')

#: TIER-LOCAL MASK SEMANTICS.  ``local_of_global`` is -1 for every neuron OUTSIDE
#: the tier, so the round-1 idiom
#:     m[local_of_global[np.flatnonzero(global_mask)]] = True
#: resolves those neurons to local index -1, which NumPy WRAPS onto the LAST row
#: (n-1): every out-of-tier member of a group then spuriously marks local row n-1.
#: ``in_tier_only`` EXCLUDES every out-of-tier id, is the DEFAULT everywhere, and
#: is the only semantics used by any reported number.  The legacy behaviour
#: survives ONLY as the explicitly named contrast arm ``round1_wraparound_legacy``
#: so that its numeric effect stays measurable instead of being deleted.
MASK_SEMANTICS = ('in_tier_only', 'round1_wraparound_legacy')
DEFAULT_MASK_SEMANTICS = 'in_tier_only'
LEGACY_MASK_SEMANTICS = 'round1_wraparound_legacy'

PURE_EXC_LABEL = 'acetylcholine'
PURE_INH_LABELS = ('gaba', 'glutamate')
EXC_REVERSAL_MV = 0.0
INH_REVERSAL_MV = -70.0

#: The counterfactual tier is a deliberate, explicitly labelled deviation: the
#: presynaptic labels of body-route and head-route mechanosensory seed cells are
#: FORCED to 'acetylcholine' for the sole purpose of giving the two routes equal
#: sign coverage, so that a route-selectivity difference cannot be an artefact of
#: one route being 0% labelled. It is NOT a transmitter claim.
COUNTERFACTUAL_NOTE = (
    'HYPOTHESIS / COUNTERFACTUAL: every body-route and head-route mechanosensory '
    'presynaptic label is FORCED to acetylcholine so that both routes have the same '
    'sign coverage. This exists only so that route selectivity of the cut can be '
    'tested in the voltage domain. It is NOT a claim about the transmitter of any '
    'of those cells and it is NOT used in the annotation-faithful tier.')


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _stable_digest(*arrays):
    """sha256[:32] over the raw bytes of the given arrays (order matters)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite number')
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return float(value)


def _aligned(value, dt, name):
    ratio = value / dt
    if ratio < 0 or not math.isclose(ratio, round(ratio), abs_tol=1e-9, rel_tol=0):
        raise ValueError(name + ' must align to the neural clock')
    return int(round(ratio))


def receptor_class(verified_label, forced_exc=False):
    """Explicit receptor class, or None for 'excluded and counted'.

    Returns 'exc' for a pure acetylcholine label (E = 0 mV), 'inh' for a pure
    gaba or glutamate label (E = -70 mV), and None for every unknown, mixed,
    modulator and histamine-only label. ``nt_pair`` is never consulted: it is -1
    on every row of this snapshot. ``forced_exc`` implements the counterfactual
    tier and must be accompanied by ``COUNTERFACTUAL_NOTE`` wherever it is used.
    """
    if forced_exc:
        return 'exc'
    if verified_label == PURE_EXC_LABEL:
        return 'exc'
    if verified_label in PURE_INH_LABELS:
        return 'inh'
    return None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TouchTierConfig:
    """Deterministic selection rule for the bounded touch-route coarse tier."""

    max_neurons: int = 20000
    partner_min_synapses: float = 10.0
    weight_scale_uS_per_synapse: float = 5e-4
    max_synapses_per_edge: int = 8
    data_path: str = BANC_RELATIVE_PATH

    def validate(self):
        if isinstance(self.max_neurons, bool) or not isinstance(self.max_neurons, int):
            raise ValueError('max_neurons must be an integer')
        if not 1 <= self.max_neurons <= 20000:
            raise ValueError('bounded prototype: max_neurons must lie in [1, 20000]')
        _finite(self.partner_min_synapses, 'partner_min_synapses')
        if self.partner_min_synapses < 0:
            raise ValueError('partner_min_synapses must be nonnegative')
        _finite(self.weight_scale_uS_per_synapse, 'weight_scale_uS_per_synapse')
        if self.weight_scale_uS_per_synapse <= 0:
            raise ValueError('weight scale must be strictly positive')
        if (isinstance(self.max_synapses_per_edge, bool)
                or not isinstance(self.max_synapses_per_edge, int)
                or self.max_synapses_per_edge < 1):
            raise ValueError('max_synapses_per_edge must be a positive integer')
        if not isinstance(self.data_path, str) or not self.data_path:
            raise ValueError('data_path must be a nonempty string')


@dataclass(frozen=True)
class NetworkRunConfig:
    """Illustrative point-tier physiology for the paired scenario runs.

    NOTHING here is fly-calibrated. The current amplitudes are HYPOTHESES chosen
    so that the seeded cells fire; the background is a seeded, per-neuron
    constant chosen so that the tier is not silent without the touch drive.
    """

    dt_ms: float = 0.05
    duration_ms: float = 40.0
    phase_ms: float = 10.0
    settle_ms: float = 10.0
    stim_start_ms: float = 5.0
    stim_period_ms: float = 8.0
    stim_duration_ms: float = 3.0
    body_touch_current_nA: float = 0.6
    head_touch_current_nA: float = 0.6
    background_mean_nA: float = 0.0
    background_sd_nA: float = 0.004
    delay_ms: float = 2.0
    seed: int = 7

    def validate(self):
        for name in ('dt_ms', 'duration_ms', 'phase_ms', 'settle_ms', 'stim_start_ms',
                     'stim_period_ms', 'stim_duration_ms', 'body_touch_current_nA',
                     'head_touch_current_nA', 'background_mean_nA',
                     'background_sd_nA', 'delay_ms'):
            _finite(getattr(self, name), name)
        if not 0 < self.dt_ms <= 0.1:
            raise ValueError('neural dt must lie in (0, 0.1] ms')
        if self.duration_ms <= 0:
            raise ValueError('duration must be positive')
        if not 0 < self.phase_ms < self.duration_ms:
            raise ValueError('the phase intervention must lie strictly inside the run')
        if not 0 <= self.phase_ms + self.settle_ms < self.duration_ms:
            raise ValueError('the settled measurement window must be nonempty')
        if self.stim_period_ms <= 0 or self.stim_duration_ms <= 0:
            raise ValueError('positive stimulus period and duration required')
        if self.stim_duration_ms > self.stim_period_ms:
            raise ValueError('stimulus duration cannot exceed its period')
        if self.body_touch_current_nA < 0 or self.head_touch_current_nA < 0:
            raise ValueError('touch currents must be nonnegative')
        if self.background_sd_nA < 0:
            raise ValueError('background sd must be nonnegative')
        if self.delay_ms <= 0:
            raise ValueError('delay must be positive')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('a nonnegative integer seed is required')
        for name in ('duration_ms', 'phase_ms', 'settle_ms', 'stim_start_ms',
                     'stim_period_ms', 'stim_duration_ms'):
            _aligned(getattr(self, name), self.dt_ms, name)
        steps = self.delay_ms / self.dt_ms
        if not math.isclose(steps, round(steps), abs_tol=1e-9, rel_tol=0) or round(steps) < 1:
            raise ValueError('delay must be an integer number of steps, at least one')

    @property
    def steps(self):
        return _aligned(self.duration_ms, self.dt_ms, 'duration_ms')

    @property
    def phase_step(self):
        return _aligned(self.phase_ms, self.dt_ms, 'phase_ms')

    @property
    def settle_step(self):
        return _aligned(self.phase_ms + self.settle_ms, self.dt_ms, 'settle_ms')

    @property
    def delay_steps(self):
        return int(round(self.delay_ms / self.dt_ms))


@dataclass(frozen=True)
class ScenarioSpec:
    """One paired run: (sign coverage) x (intervention) x (which route is driven)."""

    sign_mode: str = 'annotation'
    cut_mode: str = 'intact'
    activated_route: str = 'both'

    def validate(self):
        if self.sign_mode not in SIGN_MODES:
            raise ValueError('sign_mode must be one of ' + repr(SIGN_MODES))
        if self.cut_mode not in CUT_MODES:
            raise ValueError('cut_mode must be one of ' + repr(CUT_MODES))
        if self.activated_route not in ACTIVATED_ROUTES:
            raise ValueError('activated_route must be one of ' + repr(ACTIVATED_ROUTES))

    @property
    def name(self):
        return '%s__%s__%s' % (self.sign_mode, self.cut_mode, self.activated_route)

    def as_dict(self):
        d = asdict(self)
        d['name'] = self.name
        return d


# ---------------------------------------------------------------------------
# step 1 -- independent re-derivation of the crossing set from the raw cache
# ---------------------------------------------------------------------------
def load_banc(path):
    """Load the cached BANC snapshot, read-only, no pickle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('BANC cache not found: %s' % path)
    with np.load(path, allow_pickle=False) as d:
        out = {k: np.asarray(d[k]) for k in d.keys()}
    required = ('pre', 'post', 'syn', 'nt_pair', 'root_ids', 'ann_super_class',
                'ann_class', 'ann_nerve', 'ann_nt_verified')
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError('BANC cache is missing required arrays: %r' % missing)
    n = out['root_ids'].size
    for key in ('ann_super_class', 'ann_class', 'ann_nerve', 'ann_nt_verified'):
        if out[key].shape != (n,):
            raise ValueError('annotation array %s has the wrong shape' % key)
    for key in ('pre', 'post', 'syn', 'nt_pair'):
        if out[key].shape != out['pre'].shape:
            raise ValueError('edge array %s has the wrong shape' % key)
    if out['pre'].size and (out['pre'].min() < 0 or out['pre'].max() >= n
                            or out['post'].min() < 0 or out['post'].max() >= n):
        raise ValueError('edge endpoint index out of range')
    if np.any(out['syn'] < 0):
        raise ValueError('negative synapse counts are impossible')
    out['path'] = str(path)
    return out


def mechanosensory_partition(a):
    """Body-route vs head-route mechanosensory split (the parent's rule, verbatim)."""
    cls = a['ann_class']
    nerve = a['ann_nerve']
    mech = np.isin(cls, list(MECH_LABELS))
    body_nerve = np.array([any(k in s for k in BODY_NERVE_KEYWORDS) for s in nerve])
    head_nerve = np.array([any(k in s for k in HEAD_NERVE_KEYWORDS) for s in nerve])
    body = mech & body_nerve & ~head_nerve
    head = mech & head_nerve & ~body_nerve
    return {
        'mechanosensory': mech,
        'body_route': body,
        'head_route': head,
        'counts': {
            'mechanosensory_total': int(mech.sum()),
            'body_route': int(body.sum()),
            'head_route': int(head.sum()),
            'mechanosensory_unassigned_nerve': int((mech & ~body & ~head).sum()),
            'gustatory_excluded': {lab: int((cls == lab).sum()) for lab in GUSTATORY_LABELS},
        },
    }


def derive_crossing_set(a):
    """Re-derive the neck-crossing fibre set from the RAW cached annotation.

    Independent of any previously written npz: this reads only ``ann_super_class``
    and cross-checks it against ``ann_class``, and it reports the count against the
    published anchors WITHOUT adjusting anything to match them.
    """
    sc = a['ann_super_class']
    cls = a['ann_class']
    flow = a['ann_flow'] if 'ann_flow' in a else np.array([''] * sc.size)
    primary = a['ann_primary_type'] if 'ann_primary_type' in a else np.array([''] * sc.size)
    n = sc.size

    populations = {}
    for lab in CROSSING_POPULATIONS:
        m = sc == lab
        populations[lab] = {
            'neurons': int(m.sum()),
            'outgoing_edges': int(m[a['pre']].sum()),
            'outgoing_synapses': float(a['syn'][m[a['pre']]].sum()),
        }
    ascending_side = np.isin(sc, list(CROSSING_ASCENDING))
    descending_side = np.isin(sc, list(CROSSING_DESCENDING))
    crossing = ascending_side | descending_side

    # --- independent cross-checks that use DIFFERENT columns of the same cache
    class_ascending = cls == 'ascending_neuron'
    class_descending = cls == 'descending_neuron'
    celltypes = a['ann_primary_type'] if 'ann_primary_type' in a else np.array([''] * n)
    odd_asc = np.flatnonzero(class_ascending & ~(sc == 'ascending'))
    odd_desc = np.flatnonzero(class_descending & ~(sc == 'descending'))
    cross_checks = {
        'ann_class_ascending_neuron': int(class_ascending.sum()),
        'ann_class_descending_neuron': int(class_descending.sum()),
        'super_class_ascending': int((sc == 'ascending').sum()),
        'super_class_descending': int((sc == 'descending').sum()),
        'super_class_ascending_is_subset_of_class_ascending_neuron': bool(
            not np.any((sc == 'ascending') & ~class_ascending)),
        'super_class_descending_is_subset_of_class_descending_neuron': bool(
            not np.any((sc == 'descending') & ~class_descending)),
        'columns_agree_exactly': bool(np.array_equal(sc == 'ascending', class_ascending)
                                      and np.array_equal(sc == 'descending', class_descending)),
        'DISAGREEMENT_the_two_annotation_columns_differ': {
            'claim': ('the two annotation columns give DIFFERENT crossing counts: '
                      'ann_super_class yields %d, and an ann_class-based count yields %d. '
                      'This is reported, not reconciled.'
                      % (int(np.isin(sc, list(CROSSING_POPULATIONS)).sum()),
                         int(class_ascending.sum() + class_descending.sum())
                         + int((sc == 'sensory_ascending').sum())
                         + int((sc == 'ascending_visceral_circulatory').sum())
                         + int((sc == 'sensory_descending').sum()))),
            'cells_class_ascending_neuron_but_super_class_not_ascending': int(odd_asc.size),
            'cells_class_descending_neuron_but_super_class_not_descending': int(odd_desc.size),
            'those_super_classes': {
                'ascending_neuron_outsiders': _population_breakdown(a, class_ascending & ~(sc == 'ascending')),
                'descending_neuron_outsiders': _population_breakdown(a, class_descending & ~(sc == 'descending')),
            },
            'those_cell_types_ascending_neuron_outsiders': sorted(
                {str(celltypes[i]) for i in odd_asc}),
            'those_cell_types_descending_neuron_outsiders': sorted(
                {str(celltypes[i]) for i in odd_desc}),
            'why_ann_super_class_is_used': (
                'the %d asc + %d desc cells that only ann_class calls ascending/descending '
                'carry primary types %r / %r, i.e. nerve-cord-intrinsic, sensory and '
                'central-brain-intrinsic identities that are NOT ascending or descending. '
                'ann_super_class is therefore the internally consistent column, its total '
                'equals the published backbone, and it is the one used. The disagreement is '
                'recorded so the reader can re-derive the alternative.'
                % (odd_asc.size, odd_desc.size,
                   sorted({str(celltypes[i]) for i in odd_asc}),
                   sorted({str(celltypes[i]) for i in odd_desc}))),
            'candidate_root_ids_ascending_neuron_outsiders': [
                int(a['root_ids'][i]) for i in odd_asc],
            'candidate_root_ids_descending_neuron_outsiders': [
                int(a['root_ids'][i]) for i in odd_desc],
        },
        'ann_flow_distinct_values': sorted({str(v) for v in np.unique(flow)}),
        'ann_flow_carries_polarity': bool(
            any('ascend' in str(v).lower() or 'descend' in str(v).lower()
                for v in np.unique(flow))),
        'ann_flow_note': ('ann_flow is intrinsic/afferent/efferent and never says '
                          'ascending/descending, so it cannot and is not used to '
                          'derive polarity'),
    }
    # A cell-type name-prefix proxy gives a DIFFERENT and much smaller number; it is
    # reported so nobody mistakes the naming convention for the annotation.
    prefix_proxy = {}
    for pref in ('AN0', 'AN', 'DNg', 'DN', 'SA'):
        m = np.array([str(t).startswith(pref) and len(str(t)) > len(pref)
                      and str(t)[len(pref)].isdigit() for t in primary])
        prefix_proxy['primary_type_startswith_' + pref + '_digit'] = int(m.sum())
    cross_checks['cell_type_prefix_proxy_is_not_the_annotation'] = {
        'counts': prefix_proxy,
        'note': ('cell-type strings such as ANXXX027 carry no digit after AN, so a '
                 'prefix rule under-counts badly and is NOT used. It is reported only '
                 'to show that a plausible-looking alternative derivation disagrees.'),
    }

    mine = int(crossing.sum())
    pub_total = PUBLISHED_ANCHORS['banc_backbone_total']['value']
    hist_total = PUBLISHED_ANCHORS['histology_axons_total']['value']
    comp = PUBLISHED_ANCHORS['paper_component_counts']
    comparison = {
        'mine_re_derived_total': mine,
        'published_banc_backbone_total': pub_total,
        'published_histology_axons_total': hist_total,
        'difference_vs_published_banc_backbone': mine - pub_total,
        'agrees_with_published_banc_backbone': bool(mine == pub_total),
        'coverage_of_histology_axons': (mine / hist_total) if hist_total else None,
        'histology_axons_not_accounted_for': hist_total - mine,
        'component_comparison': {
            'ascending': {'mine': int(populations['ascending']['neurons']),
                          'published': comp['ascending']},
            'descending': {'mine': int(populations['descending']['neurons']),
                           'published': comp['descending']},
            'sensory_ascending': {'mine': int(populations['sensory_ascending']['neurons']),
                                  'published': comp['sensory_ascending']},
        },
        'component_sum_of_published_values': (comp['ascending'] + comp['descending']
                                             + comp['sensory_ascending']),
        'honest_note': (
            'The TOTAL agrees with the published BANC backbone count to the cell, but the '
            'per-population counts do NOT: my ascending/sensory_ascending counts are LOWER than '
            'the paper values while my total matches, and the paper\'s own three components sum '
            'to %d rather than its stated total %d. A second annotation column (ann_class) '
            'gives yet other values (ascending 1850, descending 1317), and the published '
            'ascending count 1849 lies BETWEEN the two columns\' values (1847 and 1850), so the '
            'mismatch is not explained by the column choice either. Agreement of a total is '
            'therefore not independent confirmation of a partition. No number here was '
            'adjusted to match anything.'
            % (comp['ascending'] + comp['descending'] + comp['sensory_ascending'], pub_total)),
    }
    return {
        'populations': populations,
        'neurons': mine,
        'neurons_ascending_side': int(ascending_side.sum()),
        'neurons_descending_side': int(descending_side.sum()),
        'definition': ('ann_super_class in %r; the dataset annotation, NOT a verified '
                       'single-cell polarity. Note: an ann_class-based derivation gives a '
                       'DIFFERENT count (see cross_checks), and the disagreement is reported '
                       'rather than reconciled.' % (CROSSING_POPULATIONS,)),
        'masks': {'ascending_side': ascending_side, 'descending_side': descending_side,
                  'crossing': crossing},
        'cross_checks': cross_checks,
        'published_comparison': comparison,
        'single_cell_polarity_verified': False,
    }


# ---------------------------------------------------------------------------
# step 2 -- the cut as an explicit fibre set
# ---------------------------------------------------------------------------
def build_cut_set(a, crossing):
    """Severed fibre set = outgoing rows of the crossing neurons, split by side.

    Exhaustive and non-overlapping by construction: the ascending-side and
    descending-side populations are disjoint (asserted) and partitioned so that
    ``ascending + descending == total`` for neurons, edges and synapses.
    """
    pre, post, syn = a['pre'], a['post'], a['syn']
    asc = crossing['masks']['ascending_side']
    desc = crossing['masks']['descending_side']
    if np.any(asc & desc):
        raise ValueError('ascending-side and descending-side populations overlap')
    if not np.array_equal(asc | desc, crossing['masks']['crossing']):
        raise ValueError('ascending+descending must equal the crossing set')
    if pre.shape != post.shape or pre.shape != syn.shape:
        raise ValueError('edge arrays must have identical shapes')

    rows_asc = asc[pre]
    rows_desc = desc[pre]
    rows = rows_asc | rows_desc
    if np.any(rows_asc & rows_desc):
        raise ValueError('a row cannot be both ascending-side and descending-side')
    if int(rows_asc.sum()) + int(rows_desc.sum()) != int(rows.sum()):
        raise ValueError('ascending + descending must equal total severed rows')

    brain = np.isin(a['ann_super_class'], list(BRAIN_SIDE_POPULATIONS))
    ascending_targets = np.isin(a['ann_super_class'], list(CROSSING_ASCENDING))
    descending_targets = np.isin(a['ann_super_class'], list(CROSSING_DESCENDING))
    vnc = a['ann_super_class'] == VNC_POPULATION

    def _block(m):
        return {
            'edges': int(m.sum()),
            'synapses': float(syn[m].sum()),
            'edges_into_brain_side': int((m & brain[post]).sum()),
            'synapses_into_brain_side': float(syn[m & brain[post]].sum()),
            'edges_into_ascending_side': int((m & ascending_targets[post]).sum()),
            'synapses_into_ascending_side': float(syn[m & ascending_targets[post]].sum()),
            'edges_into_descending_side': int((m & descending_targets[post]).sum()),
            'synapses_into_descending_side': float(syn[m & descending_targets[post]].sum()),
            'edges_into_vnc_intrinsic': int((m & vnc[post]).sum()),
            'synapses_into_vnc_intrinsic': float(syn[m & vnc[post]].sum()),
        }

    total = _block(rows)
    ascending = _block(rows_asc)
    descending = _block(rows_desc)
    identity = {
        'edges': ascending['edges'] + descending['edges'] == total['edges'],
        'synapses': bool(np.isclose(ascending['synapses'] + descending['synapses'],
                                    total['synapses'], rtol=0, atol=1e-9)),
        'no_double_counting': not bool(np.any(rows_asc & rows_desc)),
        'exhaustive': int(rows.sum()) == int(rows_asc.sum()) + int(rows_desc.sum()),
    }
    if not all(identity.values()):
        raise AssertionError('severed-fibre accounting identity failed: %r' % identity)
    return {
        'definition': ('the set of OUTGOING edges of every annotated neck-crossing neuron; '
                       'an annotated-class edge removal, NOT a cut plane reconstructed '
                       'from anatomy'),
        'row_masks': {'ascending_side': rows_asc, 'descending_side': rows_desc,
                      'all': rows},
        'neurons_total': int(crossing['neurons']),
        'neurons_ascending_side': int(np.count_nonzero(asc)),
        'neurons_descending_side': int(np.count_nonzero(desc)),
        'total': total,
        'ascending_side': ascending,
        'descending_side': descending,
        'identity_checks': identity,
        'unsigned': True,
        'unsigned_note': ('nt_pair is -1 on every row of this snapshot, so these are '
                          'synapse counts (bandwidth), not excitation or inhibition'),
    }


# ---------------------------------------------------------------------------
# step 3 -- synapse-weighted route budget (body touch vs head touch)
# ---------------------------------------------------------------------------
def route_budget(a, cut, mech):
    """Synapse-weighted body-touch vs head-touch budget, and what the cut removes.

    Every number is a synapse count. Nothing here is reachability: the "relay"
    terms are the single-hop synapse mass from crossing neurons onto brain-side
    neurons, which is exactly the part of the budget the cut deletes.
    """
    pre, post, syn = a['pre'], a['post'], a['syn']
    body, head = mech['body_route'], mech['head_route']
    brain = np.isin(a['ann_super_class'], list(BRAIN_SIDE_POPULATIONS))
    asc = cut['row_masks']['ascending_side']
    desc = cut['row_masks']['descending_side']
    severed = cut['row_masks']['all']

    def _route(mask, name):
        e = mask[pre]
        total = float(syn[e].sum())
        asc_t = np.isin(a['ann_super_class'], list(CROSSING_ASCENDING))[post]
        desc_t = np.isin(a['ann_super_class'], list(CROSSING_DESCENDING))[post]
        vnc_t = (a['ann_super_class'] == VNC_POPULATION)[post]
        out = {
            'source_neurons': int(mask.sum()),
            'outgoing_edges': int(e.sum()),
            'outgoing_synapses': total,
            'to_ascending_side': float(syn[e & asc_t].sum()),
            'to_descending_side': float(syn[e & desc_t].sum()),
            'to_brain_side_direct': float(syn[e & brain[post]].sum()),
            'to_vnc_intrinsic': float(syn[e & vnc_t].sum()),
            'severed_edges': int((e & severed).sum()),
            'severed_synapses': float(syn[e & severed].sum()),
            'severed_synapses_into_ascending_side': float(syn[e & severed & asc_t].sum()),
            'severed_synapses_into_brain_side': float(syn[e & severed & brain[post]].sum()),
            'severed_synapses_into_vnc_intrinsic': float(syn[e & severed & vnc_t].sum()),
        }
        out['severed_fraction_of_route'] = (out['severed_synapses'] / total) if total else 0.0
        out['direct_fraction_to_ascending_side'] = (
            out['to_ascending_side'] / total) if total else 0.0
        out['direct_fraction_to_brain_side'] = (
            out['to_brain_side_direct'] / total) if total else 0.0
        out['severed_fraction_of_direct_to_ascending_side'] = (
            out['severed_synapses_into_ascending_side'] / out['to_ascending_side']
            if out['to_ascending_side'] else 0.0)
        out['severed_fraction_of_direct_to_brain_side'] = (
            out['severed_synapses_into_brain_side'] / out['to_brain_side_direct']
            if out['to_brain_side_direct'] else 0.0)
        out['direct_monosynaptic_fraction_is_cut_independent'] = False
        out['direct_monosynaptic_fraction_note'] = (
            'the direct monosynaptic fraction is NOT fully cut-independent: %d of %d %s-route '
            'synapses onto the ascending side and %d of %d onto the brain side come from route '
            'cells that the dataset ALSO annotates as crossing, so those rows are severed. '
            'Monosynaptic numbers alone therefore cannot answer the question; the relay and the '
            'simulation are needed.'
            % (int(out['severed_synapses_into_ascending_side']), int(out['to_ascending_side']),
               name, int(out['severed_synapses_into_brain_side']),
               int(out['to_brain_side_direct'])))
        out['severed_edges_are_all_from_route_cells_annotated_as_crossing'] = True
        return out

    body_budget = _route(body, 'body')
    head_budget = _route(head, 'head')
    # how many head/body route NEURONS are themselves annotated as crossing
    cross_mask = (np.isin(a['ann_super_class'], list(CROSSING_ASCENDING))
                  | np.isin(a['ann_super_class'], list(CROSSING_DESCENDING)))
    brain_in = float(syn[brain[post]].sum())
    budget = {
        'body_route': body_budget,
        'head_route': head_budget,
        'brain_side_incoming_synapses_total': brain_in,
        'brain_side_incoming_from_ascending_side_synapses': float(syn[asc & brain[post]].sum()),
        'brain_side_incoming_from_descending_side_synapses': float(syn[desc & brain[post]].sum()),
        'brain_side_incoming_from_body_route_direct_synapses': float(
            syn[body[pre] & brain[post]].sum()),
        'brain_side_incoming_from_head_route_direct_synapses': float(
            syn[head[pre] & brain[post]].sum()),
        'route_neurons_also_annotated_as_crossing': {
            'body_route': int((body & cross_mask).sum()),
            'head_route': int((head & cross_mask).sum()),
            'body_route_populations': _population_breakdown(a, body & cross_mask),
            'head_route_populations': _population_breakdown(a, head & cross_mask),
            'note': ('these cells are BOTH mechanosensory (by class+nerve) and annotated as '
                     'neck-crossing populations, so their outgoing edges are in the cut. A '
                     'claim that "the cut removes no head-route output" would be FALSE; the '
                     'exact figure is in head_route.severed_synapses.'),
        },
        'units': 'synapse counts (UNSIGNED bandwidth), not excitation or inhibition',
        'reachability_used': False,
    }
    return budget


def _population_breakdown(a, mask):
    sc = a['ann_super_class']
    vals, counts = np.unique(sc[mask], return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


# ---------------------------------------------------------------------------
# step 4 -- deterministic bounded touch-route coarse tier
# ---------------------------------------------------------------------------
def select_touch_tier(a, mech, cut, config=None):
    """Deterministic, documented, size-bounded tier for the touch routes.

    SELECTION RULE (every step deterministic; every tie broken by ascending global
    neuron index, so the result never depends on dict/set iteration order or on the
    process hash seed):

      1. A = the body-route mechanosensory neurons, H = the head-route
         mechanosensory neurons (class + innervating nerve; the parent's rule).
      2. partner_inflow_i = number of SYNAPSES that neuron i receives from A u H
         (the cached edge list, unsigned counts).
      3. P = {i not in A u H : partner_inflow_i >= partner_min_synapses}, ranked by
         (-partner_inflow, index).
      4. keep = A u H u P, truncated to the first (max_neurons - |A u H|) entries of
         the ranking; the element-wise decomposition of the cap is reported.
      5. The final node set is exactly (4). Unlike engine.multirate_scenario this
         selection does NOT then reduce to a single connected component: the
         quantity being measured is a route budget, and dropping unattached seats
         would silently discard part of it. The component structure OVER THE
         MODELLED (signed) EDGES is reported instead, and the number of neurons
         outside the largest such component is reported as a diagnostic.

    Rejected alternatives are recorded in ``rejected_alternatives`` so the rule
    cannot be quietly changed later.
    """
    c = config or TouchTierConfig()
    c.validate()
    pre, post, syn = a['pre'], a['post'], a['syn']
    n = a['root_ids'].size
    if mech['body_route'].shape != (n,) or mech['head_route'].shape != (n,):
        raise ValueError('mechanosensory masks must have one entry per neuron')

    seeds = mech['body_route'] | mech['head_route']
    n_seeds = int(seeds.sum())
    if n_seeds == 0:
        raise ValueError('no mechanosensory seed neurons found')
    if n_seeds > c.max_neurons:
        raise ValueError('the seed set alone exceeds max_neurons: %d > %d'
                         % (n_seeds, c.max_neurons))

    inflow = np.bincount(post[seeds[pre]], weights=syn[seeds[pre]].astype(np.float64),
                         minlength=n)
    candidates = np.flatnonzero((inflow >= c.partner_min_synapses) & ~seeds)
    order = candidates[np.lexsort((candidates, -inflow[candidates]))]
    room = c.max_neurons - n_seeds
    chosen = order[:room]
    keep = seeds.copy()
    keep[chosen] = True
    gi = np.flatnonzero(keep)
    nn = int(gi.size)
    if nn != int(keep.sum()) or nn > c.max_neurons:
        raise AssertionError('tier size bookkeeping failed')

    loc = -np.ones(n, dtype=np.int64)
    loc[gi] = np.arange(nn)
    # The index map MUST be a bijection from the in-tier neurons onto [0, nn) and -1
    # everywhere else: -1 is the "outside the tier" SENTINEL and must never be used as
    # an index (that is exactly the round-1 wraparound defect).
    if not (int(np.count_nonzero(loc >= 0)) == nn and int(loc[gi].min()) == 0
            and int(loc[gi].max()) == nn - 1):
        raise AssertionError('tier index map is not a bijection onto [0, n)')
    inside = keep[pre] & keep[post]
    cross_mask = cut['row_masks']['all']
    cut_inside = inside & cross_mask
    asc_inside = inside & cut['row_masks']['ascending_side']
    desc_inside = inside & cut['row_masks']['descending_side']

    def _gm(mask):
        return mask[gi]

    counts = {
        'neurons_total_in_dataset': n,
        'annotated_rows_total_in_dataset': int(pre.size),
        'seed_body_route_neurons': int(mech['body_route'].sum()),
        'seed_head_route_neurons': int(mech['head_route'].sum()),
        'seed_neurons_union': n_seeds,
        'partner_candidates_at_threshold': int(candidates.size),
        'partner_candidates_total_available': int((inflow > 0).sum()),
        'partners_added': int(chosen.size),
        'partners_truncated_by_cap': bool(candidates.size > room),
        'neurons_in_tier': nn,
        'final_neuron_cap': int(c.max_neurons),
        'annotated_rows_inside_tier': int(inside.sum()),
        'share_of_dataset_rows_inside_tier': float(inside.sum() / pre.size),
        'crossing_neurons_in_dataset': int(cut['neurons_total']),
        'crossing_neurons_in_tier': int(
            _gm(np.isin(a['ann_super_class'], list(CROSSING_ASCENDING))
                | np.isin(a['ann_super_class'], list(CROSSING_DESCENDING))).sum()),
        'crossing_neurons_in_tier_ascending_side': int(
            _gm(np.isin(a['ann_super_class'], list(CROSSING_ASCENDING))).sum()),
        'crossing_neurons_in_tier_descending_side': int(
            _gm(np.isin(a['ann_super_class'], list(CROSSING_DESCENDING))).sum()),
        'severed_edges_inside_tier': int(cut_inside.sum()),
        'severed_synapses_inside_tier': float(syn[cut_inside].sum()),
        'severed_edges_inside_tier_ascending_side': int(asc_inside.sum()),
        'severed_edges_inside_tier_descending_side': int(desc_inside.sum()),
        'severed_synapses_inside_tier_ascending_side': float(syn[asc_inside].sum()),
        'severed_synapses_inside_tier_descending_side': float(syn[desc_inside].sum()),
        'share_of_dataset_severed_edges_inside_tier': float(cut_inside.sum() / cut['total']['edges']),
        'share_of_dataset_severed_synapses_inside_tier': float(
            syn[cut_inside].sum() / cut['total']['synapses']),
    }
    if (counts['severed_edges_inside_tier_ascending_side']
            + counts['severed_edges_inside_tier_descending_side']
            != counts['severed_edges_inside_tier']):
        raise AssertionError('in-tier severed-edge split is not exhaustive')
    if not np.isclose(counts['severed_synapses_inside_tier_ascending_side']
                      + counts['severed_synapses_inside_tier_descending_side'],
                      counts['severed_synapses_inside_tier'], rtol=0, atol=1e-9):
        raise AssertionError('in-tier severed-synapse split is not exhaustive')

    selected_ids = np.asarray(a['root_ids'][gi], dtype=np.int64)
    return {
        'n': nn,
        'global_index': gi,
        'local_of_global': loc,
        'keep_mask': keep,
        'inside_rows': inside,
        'root_ids': selected_ids,
        'counts': counts,
        'selection_rule': (
            'A = body-route mechanosensory (class in MECH_LABELS and innervates a body '
            'nerve and no head nerve); H = head-route mechanosensory; partner_inflow_i = '
            'synapses received by i from A u H (unsigned counts from the cached edge '
            'list); P = {{i not in A u H : partner_inflow_i >= {thr}}}, ranked by '
            '(-partner_inflow, global index); keep = A u H u first ({cap} - |A u H|) of '
            'the ranked P. Deterministic, index-tie-broken, size-bounded to {cap} '
            'neurons. Unlike engine.multirate_scenario the final set is NOT reduced to '
            'one connected component, because the measured quantity is a route budget '
            '(component structure is reported as a diagnostic instead).'
            .format(thr=c.partner_min_synapses, cap=c.max_neurons)),
        'rejected_alternatives': {
            'largest_connected_component_step': (
                'rejected: it would drop seats that carry part of the route budget, and a '
                'route budget is not a connectivity claim'),
            'degree_ranked_seed_set': (
                'rejected: a top-degree seed set is dominated by optic-lobe intrinsic cells '
                'and contains almost none of the touch route'),
            'random_sampling': 'rejected: not reproducible under a documented rule',
        },
        'config': asdict(c),
        'digest': {
            'selected_root_ids_sha256_32': _stable_digest(selected_ids),
            'selected_global_index_sha256_32': _stable_digest(gi),
        },
        'local_of_global_index_map': {'outside_tier_sentinel': -1,
                                      'maps_in_tier_onto': '[0, n)',
                                      'is_a_bijection': True,
                                      'note': ('the sentinel is NOT a valid index: a global mask '
                                               'must be projected with local_mask(), which '
                                               'EXCLUDES out-of-tier ids; indexing with the '
                                               'sentinel wraps onto row n-1')},
        'dataset': {'path': a.get('path'), 'neurons': n, 'annotated_rows': int(pre.size)},
    }


# ---------------------------------------------------------------------------
# tier-local masks: the CORRECT semantics, and the round-1 wraparound defect
# ---------------------------------------------------------------------------
def tier_index_map_is_consistent(tier):
    """True iff ``local_of_global`` maps each IN-TIER neuron to a UNIQUE row in [0, n)."""
    nn = int(tier['n'])
    loc = np.asarray(tier['local_of_global'])
    inside = loc >= 0
    if int(np.count_nonzero(inside)) != nn:
        return False
    vals = loc[inside]
    return bool(vals.size == nn and vals.min() == 0 and vals.max() == nn - 1
                and np.unique(vals).size == nn)


def local_mask(tier, global_mask, semantics=DEFAULT_MASK_SEMANTICS):
    """Tier-LOCAL boolean mask for a GLOBAL neuron mask. This is the ONLY definition.

    ``in_tier_only`` (the DEFAULT, and the only semantics any reported number uses):
    a global neuron outside the tier carries ``local_of_global == -1`` and is
    EXCLUDED.  Consequences, both asserted here rather than assumed:

      * the number of masked local rows EQUALS the number of the mask's members that
        lie inside the tier;
      * local row ``n-1`` is marked ONLY if it is a genuine in-tier member.

    ``round1_wraparound_legacy``: the round-1 defect, kept ONLY as an explicitly
    named contrast arm so its numeric effect remains measurable.  It indexes with
    -1, which NumPy wraps onto the last row, so EVERY out-of-tier member of the group
    spuriously marks local row ``n-1``.
    """
    if semantics not in MASK_SEMANTICS:
        raise ValueError('mask semantics must be one of %r' % (MASK_SEMANTICS,))
    loc = np.asarray(tier['local_of_global'])
    gm = np.asarray(global_mask)
    if gm.ndim != 1 or gm.shape != loc.shape:
        raise ValueError('global mask must have one entry per dataset neuron')
    nn = int(tier['n'])
    gi = np.flatnonzero(gm)
    m = np.zeros(nn, bool)
    if gi.size == 0:
        return m
    if semantics == LEGACY_MASK_SEMANTICS:
        m[loc[gi]] = True
        return m
    in_tier = loc[gi] >= 0
    members_in_tier = int(np.count_nonzero(in_tier))
    m[loc[gi[in_tier]]] = True
    masked_rows = int(np.count_nonzero(m))
    if masked_rows != members_in_tier:
        raise AssertionError(
            'the tier index map is not injective on in-tier neurons: %d global members '
            'inside the tier map onto %d distinct local rows'
            % (members_in_tier, masked_rows))
    return m


def named_group_masks(groups):
    """The 13 NAMED masks the reports quote, with their table-qualified names."""
    named = {}
    for name, mask in groups['disjoint'].items():
        named['disjoint.' + name] = mask
    for name, mask in groups['route_overlapping'].items():
        named['route_overlapping.' + name] = mask
    return named


def mask_membership_table(tier, groups):
    """Per NAMED mask: local membership under the CORRECT and under the LEGACY semantics.

    ``members_in_tier`` is the number of masked local rows under the default
    semantics; it is asserted to equal the number of the mask's global members that
    lie inside the tier -- the identity the round-1 idiom violated.
    """
    nn = int(tier['n'])
    loc = np.asarray(tier['local_of_global'])
    table = {}
    for name, mask in named_group_masks(groups).items():
        gi = np.flatnonzero(np.asarray(mask))
        outside = gi[loc[gi] < 0]
        correct = local_mask(tier, mask, DEFAULT_MASK_SEMANTICS)
        legacy = local_mask(tier, mask, LEGACY_MASK_SEMANTICS)
        members_in_tier = int(np.count_nonzero(loc[gi] >= 0))
        masked_rows = int(np.count_nonzero(correct))
        if masked_rows != members_in_tier:
            raise AssertionError(
                'masked rows != in-tier members for %s: %d != %d'
                % (name, masked_rows, members_in_tier))
        table[name] = {
            'global_members': int(gi.size),
            'global_members_outside_tier': int(outside.size),
            'members_in_tier_correct_semantics': members_in_tier,
            'masked_rows_correct_semantics': masked_rows,
            'masked_rows_legacy_semantics': int(np.count_nonzero(legacy)),
            'rows_wrapped_onto_last_local_row_legacy': int(outside.size),
            # kept for continuity with the round-1 report field names
            'rows_wrapped_to_last_local_index': int(outside.size),
            'last_local_index': nn - 1,
            'last_local_row_is_a_true_member': bool(correct[nn - 1]),
            'last_row_is_a_true_member': bool(correct[nn - 1]),
            'membership_changed': bool(not np.array_equal(correct, legacy)),
            'masks_differ': bool(not np.array_equal(correct, legacy)),
            'spurious_rows_under_legacy_semantics': int(np.count_nonzero(legacy & ~correct)),
            'extra_local_rows_under_round1_semantics': int(np.count_nonzero(legacy & ~correct)),
            'missing_rows_under_legacy_semantics': int(np.count_nonzero(correct & ~legacy)),
        }
    return table


def mask_semantics_audit(tier, groups):
    """MEASURED size of the round-1 wraparound defect, per named mask.

    The default semantics are now ``in_tier_only``; this block reports what the
    legacy arm would (and round 1 did) fold onto the last local row, and it is what
    the reports quote so the correction is auditable instead of silent.
    """
    table = mask_membership_table(tier, groups)
    changed = sorted(n for n, v in table.items() if v['membership_changed'])
    return {
        'default_semantics': DEFAULT_MASK_SEMANTICS,
        'correct_semantics': DEFAULT_MASK_SEMANTICS,
        'legacy_contrast_arm_semantics': LEGACY_MASK_SEMANTICS,
        'defect': ('out-of-tier neurons have local_of_global == -1, and the round-1 idiom '
                   'indexes with -1, which NumPy wraps to the last row, so every affected '
                   'mask contains local row n-1 spuriously'),
        'legacy_semantics_availability': ('available ONLY as the explicitly named arm %r, which '
                                          'no reported number uses' % (LEGACY_MASK_SEMANTICS,)),
        'tier_index_map_is_consistent': tier_index_map_is_consistent(tier),
        'masks_total': len(table),
        'masks_changed_by_the_fix': len(changed),
        'changed_mask_names': changed,
        'any_mask_changed': bool(changed),
        'any_mask_affected': bool(changed),
        'per_mask': table,
    }


def mask_semantics_contrast(a, tier, cut, groups, scenario=None, cfg=None, tier_cfg=None):
    """OLD-vs-NEW numbers for ONE arm that differs ONLY in the mask semantics.

    The two runs share the seed, the initialisation and every edge; the ONLY
    difference is how a global group mask is projected onto the tier.  The arm is
    therefore the exact measured cost of the round-1 wraparound defect.
    """
    scenario = scenario or ScenarioSpec()
    scenario.validate()
    cfg = cfg or NetworkRunConfig()
    cfg.validate()
    tier_cfg = tier_cfg or TouchTierConfig()
    runs = {}
    for tag, sem in (('legacy', LEGACY_MASK_SEMANTICS), ('correct', DEFAULT_MASK_SEMANTICS)):
        r = run_scenario(a, tier, cut, groups, scenario, cfg, tier_cfg, mask_semantics=sem)
        runs[tag] = (r, r.pop('_spike_matrix'))
    legacy, legacy_S = runs['legacy']
    correct, correct_S = runs['correct']
    windows = ('baseline', 'post', 'settled')

    def _f_pair(fn):
        old, new = float(fn(legacy)), float(fn(correct))
        return {'legacy_semantics': old, 'correct_semantics': new,
                'difference_legacy_minus_correct': old - new}

    def _i_pair(fn):
        old, new = int(fn(legacy)), int(fn(correct))
        return {'legacy_semantics': old, 'correct_semantics': new,
                'difference_legacy_minus_correct': old - new}

    out = {
        'scenario': scenario.as_dict(),
        'legacy_semantics': LEGACY_MASK_SEMANTICS,
        'correct_semantics': DEFAULT_MASK_SEMANTICS,
        'design': ('both runs share the seed, the initialisation, the edge set and the stimulus; '
                   'the ONLY difference is the tier-local mask projection, so every number below '
                   'is the measured cost of the round-1 wraparound defect'),
        'brain_side_drive_uS': {w: _f_pair(lambda r, w=w: sum(
            r['conductance_at_brain_side_uS'][w].values())) for w in windows},
        'ascending_side_drive_uS': {w: _f_pair(lambda r, w=w: sum(
            r['conductance_at_ascending_side_uS'][w].values())) for w in windows},
        'brain_side_spikes': {w: _i_pair(
            lambda r, w=w: r['spike_counts'][w]['per_group']['brain_side']) for w in windows},
        'total_spikes': {w: _i_pair(lambda r, w=w: r['spike_counts'][w]['total'])
                         for w in windows},
    }
    settled = 'settled'
    out['per_group_at_brain_side_settled'] = {
        'delivered_drive_uS': {g: _f_pair(
            lambda r, g=g: r['conductance_at_brain_side_uS'][settled][g])
            for g in correct['conductance_at_brain_side_uS'][settled]},
        'spikes': {g: _i_pair(
            lambda r, g=g: r['spike_counts'][settled]['per_group'][g])
            for g in correct['spike_counts'][settled]['per_group']},
    }
    out['group_membership_counts'] = {
        name: {'global_members': v['global_members'],
               'global_members_outside_tier': v['global_members_outside_tier'],
               'masked_rows_correct_semantics': v['masked_rows_correct_semantics'],
               'masked_rows_legacy_semantics': v['masked_rows_legacy_semantics']}
        for name, v in mask_membership_table(tier, groups).items()}
    out['digests'] = {
        'init_digest_sha256_32': {'legacy_semantics': legacy['init_digest_sha256_32'],
                                  'correct_semantics': correct['init_digest_sha256_32'],
                                  'changed': bool(legacy['init_digest_sha256_32']
                                                  != correct['init_digest_sha256_32'])},
        'baseline_spike_digest_sha256_32': {
            'legacy_semantics': legacy['baseline_spike_digest_sha256_32'],
            'correct_semantics': correct['baseline_spike_digest_sha256_32'],
            'changed': bool(legacy['baseline_spike_digest_sha256_32']
                            != correct['baseline_spike_digest_sha256_32'])},
        'full_spike_digest_sha256_32': {
            'legacy_semantics': _stable_digest(legacy_S),
            'correct_semantics': _stable_digest(correct_S),
            'changed': bool(_stable_digest(legacy_S) != _stable_digest(correct_S))},
    }
    out['spike_matrices_bit_identical'] = bool(np.array_equal(legacy_S, correct_S))
    return out


def presynaptic_groups(a, mech, cut):
    """Two group tables.

    ``disjoint``: MUTUALLY EXCLUSIVE and EXHAUSTIVE over all neurons, with the
    priority crossing > route sensor > vnc > other. Used for every
    conductance-attribution number, so a conductance increment can never be
    counted twice.

    ``route_overlapping``: deliberately OVERLAPPING sets (a body mechanosensory
    cell that the dataset also annotates as sensory_ascending belongs to both).
    Used only for "which stimulus drives this neuron" spike counting.
    """
    n = a['root_ids'].size
    sc = a['ann_super_class']
    ascending = np.isin(sc, list(CROSSING_ASCENDING))
    descending = np.isin(sc, list(CROSSING_DESCENDING))
    crossing = ascending | descending
    body = mech['body_route']
    head = mech['head_route']
    vnc = sc == VNC_POPULATION

    disjoint = {}
    disjoint['crossing_ascending'] = ascending
    disjoint['crossing_descending'] = descending
    disjoint['body_route_sensor'] = body & ~crossing
    disjoint['head_route_sensor'] = head & ~crossing
    disjoint['vnc_intrinsic'] = vnc & ~crossing & ~body & ~head
    rest = np.ones(n, bool)
    for v in disjoint.values():
        rest &= ~v
    disjoint['other'] = rest
    total = np.zeros(n, np.int64)
    for v in disjoint.values():
        total += v.astype(np.int64)
    if not np.array_equal(total, np.ones(n, np.int64)):
        raise AssertionError('disjoint presynaptic groups are not exhaustive/partitional')

    overlapping = {
        'body_touch_sensor': body,
        'head_touch_sensor': head,
        'crossing_ascending': ascending,
        'crossing_descending': descending,
        'vnc_intrinsic': vnc,
        'brain_side': np.isin(sc, list(BRAIN_SIDE_POPULATIONS)),
    }
    overlapping['all_other'] = ~(body | head | crossing | vnc | overlapping['brain_side'])
    return {'disjoint': disjoint, 'route_overlapping': overlapping,
            'disjoint_is_partition': True,
            'route_overlapping_is_partition': False}


# ---------------------------------------------------------------------------
# step 5 -- signed conductance edges and one paired scenario run
# ---------------------------------------------------------------------------
def build_edges(a, tier, cut, sign_mode='annotation', cut_mode='intact', config=None):
    """Signed conductance edge set of the tier for one (sign_mode, cut_mode).

    Sign rule: pure ``ann_nt_verified`` presynaptic label only. Every excluded row
    inside the tier is counted in a mutually exclusive and exhaustive breakdown.
    """
    c = config or TouchTierConfig()
    c.validate()
    if sign_mode not in SIGN_MODES:
        raise ValueError('sign_mode must be one of ' + repr(SIGN_MODES))
    if cut_mode not in CUT_MODES:
        raise ValueError('cut_mode must be one of ' + repr(CUT_MODES))
    pre, post, syn = a['pre'], a['post'], a['syn']
    ver = a['ann_nt_verified']
    inside = np.asarray(tier['inside_rows'])
    loc = np.asarray(tier['local_of_global'])
    nn = tier['n']
    if inside.shape != pre.shape or inside.dtype != np.dtype(bool):
        raise ValueError('tier inside_rows must be a boolean mask over the annotated rows')
    if loc.shape != (int(a['root_ids'].size),):
        raise ValueError('tier local_of_global must have one entry per dataset neuron')
    if int(loc[np.asarray(tier['global_index'])].max()) != nn - 1:
        raise ValueError('tier index maps are inconsistent with the tier size')

    mech = mechanosensory_partition(a)
    forced_cell = mech['body_route'] | mech['head_route']
    lab = ver[pre]
    pure = (lab == PURE_EXC_LABEL) | np.isin(lab, list(PURE_INH_LABELS))

    if sign_mode == 'annotation':
        select = inside & pure
    else:
        select = inside & (pure | forced_cell[pre])
    rows = np.flatnonzero(select)
    labels = (np.where(forced_cell[pre[rows]], PURE_EXC_LABEL, ver[pre[rows]])
              if sign_mode == 'counterfactual_ach' else ver[pre[rows]])

    severed = cut['row_masks']['all']
    asc_severed = cut['row_masks']['ascending_side']
    desc_severed = cut['row_masks']['descending_side']
    if cut_mode == 'intact' or cut_mode == 'sham':
        removed = np.zeros(rows.size, bool)
        removal_mask = np.zeros(pre.size, bool)
    elif cut_mode == 'neck_cut':
        removal_mask = severed
        removed = severed[rows]
    else:
        removal_mask = desc_severed
        removed = desc_severed[rows]
    kept_rows = rows[~removed]
    kept_labels = labels[~removed]
    exc_rows = kept_rows[kept_labels == PURE_EXC_LABEL]
    inh_rows = kept_rows[np.isin(kept_labels, list(PURE_INH_LABELS))]

    def _mat(rws):
        if rws.size == 0:
            return sparse.csr_matrix((nn, nn))
        # float64 explicitly: a float32 synapse column times a Python-float scale
        # stays float32 under NumPy 2 promotion rules, which would make the
        # severed-conductance identity inexact.
        cap_f = float(c.max_synapses_per_edge)
        scale_f = float(c.weight_scale_uS_per_synapse)
        w = np.minimum(syn[rws].astype(np.float64), cap_f) * scale_f
        m = sparse.csr_matrix((w, (loc[post[rws]], loc[pre[rws]])), shape=(nn, nn))
        m.sum_duplicates()
        return m

    We, Wi = _mat(exc_rows), _mat(inh_rows)

    inside_rows_all = np.flatnonzero(inside)
    modelled_mask = np.zeros(pre.size, bool)
    modelled_mask[exc_rows] = True
    modelled_mask[inh_rows] = True
    excluded_rows = inside_rows_all[~modelled_mask[inside_rows_all]]
    excluded_labels = ver[pre[excluded_rows]]
    n_unknown = int(np.count_nonzero(excluded_labels == ''))
    n_hist = int(np.count_nonzero(excluded_labels == 'histamine'))
    n_mixed = int(excluded_labels.size - n_unknown - n_hist)
    excluded = {
        'rows_inside_tier_total': int(inside_rows_all.size),
        'modelled_exc_rows': int(exc_rows.size),
        'modelled_inh_rows': int(inh_rows.size),
        'excluded_rows_inside_tier': int(excluded_rows.size),
        'excluded_unknown_no_label': n_unknown,
        'excluded_histamine_only': n_hist,
        'excluded_mixed_or_modulator': n_mixed,
        'breakdown_is_mutually_exclusive_and_exhaustive': bool(
            n_unknown + n_hist + n_mixed == excluded_rows.size
            and exc_rows.size + inh_rows.size + excluded_rows.size == inside_rows_all.size),
        'excluded_synapses_inside_tier': float(syn[excluded_rows].sum()),
        'rows_removed_by_cut_mode': int(removed.sum()),
        'rows_removed_by_cut_mode_synapses': float(syn[rows[removed]].sum()),
        'nt_pair_used_for_sign': False,
        'sign_rule': ('acetylcholine -> exc E=0 mV; gaba/glutamate -> inh E=-70 mV; every '
                      'unknown/mixed/histamine/modulator row EXCLUDED and counted; '
                      'nt_pair is -1 on every row and is never read'),
    }
    if sign_mode == 'counterfactual_ach':
        excluded['counterfactual'] = COUNTERFACTUAL_NOTE
    return {'We': We, 'Wi': Wi, 'exc_rows': exc_rows, 'inh_rows': inh_rows,
            'kept_rows': kept_rows, 'removed_rows': rows[removed],
            'removal_row_mask': removal_mask, 'excluded': excluded,
            'sign_mode': sign_mode, 'cut_mode': cut_mode,
            'nnz_exc': int(We.nnz), 'nnz_inh': int(Wi.nnz),
            'sum_exc_uS': float(We.data.sum()), 'sum_inh_uS': float(Wi.data.sum())}


def _arrivals_iter(spike_matrix, delay_steps):
    """Yield the per-step arrival vectors using the parent network's own ring rule."""
    ring = np.zeros((delay_steps + 1, spike_matrix.shape[1]), dtype=bool)
    ri = 0
    for k in range(spike_matrix.shape[0]):
        arr = ring[ri].copy()
        ring[ri] = False
        yield arr
        ring[ri] = spike_matrix[k]
        ri = (ri + 1) % (delay_steps + 1)


def conductance_ledger(spike_matrix, We, Wi, target_local, group_local,
                       delay_steps, step_slice=None):
    """Exact conductance (uS) delivered to ``target_local`` per presynaptic group.

    The arrival vectors are reconstructed with the SAME delay ring the network
    uses, so ``We @ arrivals`` here is an exact replay of the network's own
    conductance increment. Groups must partition the neuron set (checked by the
    caller's construction); the total over groups equals the ungrouped total.
    """
    spike_matrix = np.asarray(spike_matrix)
    if spike_matrix.ndim != 2 or spike_matrix.shape[1] != We.shape[0]:
        raise ValueError('spike matrix does not match the matrix size')
    if We.shape != Wi.shape:
        raise ValueError('exc and inh matrices must have identical shape')
    sl = slice(0, spike_matrix.shape[0]) if step_slice is None else step_slice
    sub = {name: (We[np.ix_(target_local, idx)], Wi[np.ix_(target_local, idx)])
           for name, idx in group_local.items()}
    out = {name: 0.0 for name in group_local}
    for k, arr in enumerate(_arrivals_iter(spike_matrix, delay_steps)):
        if k < sl.start or (sl.stop is not None and k >= sl.stop):
            continue
        for name, (we, wi) in sub.items():
            idx = group_local[name]
            if idx.size == 0:
                continue
            a = arr[idx]
            if not a.any():
                continue
            out[name] += float(we.dot(a).sum()) + float(wi.dot(a).sum())
    return out


def run_scenario(a, tier, cut, groups, scenario, cfg=None, tier_cfg=None,
                 keep_spikes=False, mask_semantics=DEFAULT_MASK_SEMANTICS):
    """Run one bounded conductance scenario. Returns arrays + JSON-safe metadata.

    Paired design: the pre-phase matrices are the SAME for every cut mode (the
    intact ones), the phase intervention swaps in the cut matrices, and the
    background current is drawn from ``cfg.seed`` so it is identical in every
    scenario. The pre-phase spike trains are therefore bit-identical across cut
    modes, which is asserted by the caller and recorded as digests here.

    ``mask_semantics`` selects the tier-local projection of the global group masks.
    It defaults to ``in_tier_only`` (correct: out-of-tier ids are EXCLUDED);
    ``round1_wraparound_legacy`` reproduces the round-1 defect and exists ONLY as an
    explicitly named contrast arm.
    """
    c = cfg or NetworkRunConfig()
    c.validate()
    scenario.validate()
    tcfg = tier_cfg or TouchTierConfig()
    if mask_semantics not in MASK_SEMANTICS:
        raise ValueError('mask_semantics must be one of %r' % (MASK_SEMANTICS,))
    nn = tier['n']
    steps, phase_step, settle_step = c.steps, c.phase_step, c.settle_step
    delay_steps = c.delay_steps

    intact = build_edges(a, tier, cut, scenario.sign_mode, 'intact', tcfg)
    if scenario.cut_mode in ('intact', 'sham'):
        active = intact
    else:
        active = build_edges(a, tier, cut, scenario.sign_mode, scenario.cut_mode, tcfg)

    def _mask(global_mask):
        return local_mask(tier, global_mask, mask_semantics)

    body_local = np.flatnonzero(_mask(mechanosensory_partition(a)['body_route']))
    head_local = np.flatnonzero(_mask(mechanosensory_partition(a)['head_route']))
    group_local = {name: np.flatnonzero(_mask(m)) for name, m in groups['disjoint'].items()}
    spike_groups = {name: np.flatnonzero(_mask(m))
                    for name, m in groups['route_overlapping'].items()}
    target_local = spike_groups['brain_side']
    ascending_local = spike_groups['crossing_ascending']
    descending_local = spike_groups['crossing_descending']

    params = ConductanceParams()
    net = ConductanceNetwork(nn, params, c.dt_ms).set_connectivity(intact['We'], intact['Wi'])
    rng = np.random.default_rng(c.seed)
    background = rng.normal(c.background_mean_nA, c.background_sd_nA, nn)
    body_cur = np.zeros(nn)
    head_cur = np.zeros(nn)
    if scenario.activated_route in ('both', 'body_only'):
        body_cur[body_local] = c.body_touch_current_nA
    if scenario.activated_route in ('both', 'head_only'):
        head_cur[head_local] = c.head_touch_current_nA

    init_digest = _stable_digest(net.v, net.ge, net.gi, net.adaptation,
                                 intact['We'].indptr, intact['We'].indices,
                                 intact['We'].data, intact['Wi'].indptr,
                                 intact['Wi'].indices, intact['Wi'].data,
                                 background, body_cur, head_cur)

    spike_matrix = np.zeros((steps, nn), dtype=np.uint8)
    events = []
    for k in range(steps):
        t = k * c.dt_ms
        cur = background.copy()
        if t >= c.stim_start_ms and ((t - c.stim_start_ms) % c.stim_period_ms) < c.stim_duration_ms:
            cur = cur + body_cur + head_cur
        spike_matrix[k] = net.step(cur)
        if k == phase_step:
            if scenario.cut_mode == 'neck_cut':
                net.set_connectivity(active['We'], active['Wi'])
                events.append({'event_id': 'phase:0', 'time_ms': t,
                               'kind': 'annotated_class_edge_removal_ascending_and_descending',
                               'rows_removed': int(active['excluded']['rows_removed_by_cut_mode']),
                               'note': ('outgoing rows of annotated neck-crossing neurons are '
                                        'absent from the edge set from here on; NOT a cut plane')})
            elif scenario.cut_mode == 'descending_only':
                net.set_connectivity(active['We'], active['Wi'])
                events.append({'event_id': 'phase:0', 'time_ms': t,
                               'kind': 'annotated_class_edge_removal_descending_side_only',
                               'rows_removed': int(active['excluded']['rows_removed_by_cut_mode']),
                               'note': 'only the descending-side crossing populations are severed'})
            elif scenario.cut_mode == 'sham':
                events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': 'sham_noop',
                               'rows_removed': 0,
                               'note': 'the scheduled phase was reached and nothing changed'})
            else:
                events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': 'no_intervention',
                               'rows_removed': 0, 'note': 'intact reference run'})

    windows = {
        'baseline': (0, phase_step),
        'post': (phase_step, steps),
        'settled': (settle_step, steps),
    }
    spike_counts = {}
    for wname, (lo, hi) in windows.items():
        seg = spike_matrix[lo:hi]
        spike_counts[wname] = {
            'total': int(seg.sum()),
            'per_group': {name: int(seg[:, idx].sum()) for name, idx in spike_groups.items()},
            'per_neuron': seg.sum(axis=0).astype(np.int64),
        }
    ledger = {}
    for wname, (lo, hi) in windows.items():
        ledger[wname] = conductance_ledger(spike_matrix, active['We'], active['Wi'],
                                          target_local, group_local, delay_steps,
                                          step_slice=slice(lo, hi))
    ledger_at_ascending = {}
    for wname, (lo, hi) in windows.items():
        ledger_at_ascending[wname] = conductance_ledger(
            spike_matrix, active['We'], active['Wi'], ascending_local, group_local,
            delay_steps, step_slice=slice(lo, hi))

    result = {
        'scenario': scenario.as_dict(),
        'neural_steps': int(steps),
        'neurons': int(nn),
        'edges': {'exc_nnz': active['nnz_exc'], 'inh_nnz': active['nnz_inh'],
                  'exc_uS': active['sum_exc_uS'], 'inh_uS': active['sum_inh_uS'],
                  'rows_removed_by_cut_mode': active['excluded']['rows_removed_by_cut_mode'],
                  'rows_removed_synapses': active['excluded']['rows_removed_by_cut_mode_synapses']},
        'excluded_edge_accounting': active['excluded'],
        'spike_counts': spike_counts,
        'conductance_at_brain_side_uS': ledger,
        'conductance_at_ascending_side_uS': ledger_at_ascending,
        'brain_side_total_spikes_settled': spike_counts['settled']['per_group'].get('brain_side', 0),
        'events': events,
        'init_digest_sha256_32': init_digest,
        'baseline_spike_digest_sha256_32': _stable_digest(spike_matrix[:phase_step]),
        'counterfactual': (COUNTERFACTUAL_NOTE
                           if scenario.sign_mode == 'counterfactual_ach' else None),
        'conductance_params': asdict(params),
        'mask_semantics': mask_semantics,
    }
    if keep_spikes:
        result['spike_matrix'] = spike_matrix
    result['_spike_matrix'] = spike_matrix
    return result


def replay_severed_conductance(a, tier, cut, groups, spike_matrix, config=None,
                               tier_cfg=None, sign_mode='annotation',
                               mask_semantics=DEFAULT_MASK_SEMANTICS):
    """EXACT severed conductance under IDENTICAL activity.

    Replays one recorded spike train through the intact and the cut matrices and
    differences them, so the result isolates EDGE REMOVAL from every
    network-feedback effect. This is the number that answers "how much drive do the
    severed fibres carry into the brain side", with the activity held fixed.

    ``mask_semantics`` defaults to the correct ``in_tier_only``; the legacy arm is
    available only by naming it explicitly.
    """
    c = config or NetworkRunConfig()
    c.validate()
    tcfg = tier_cfg or TouchTierConfig()
    if sign_mode not in SIGN_MODES:
        raise ValueError('sign_mode must be one of ' + repr(SIGN_MODES))
    if mask_semantics not in MASK_SEMANTICS:
        raise ValueError('mask_semantics must be one of %r' % (MASK_SEMANTICS,))
    intact = build_edges(a, tier, cut, sign_mode, 'intact', tcfg)
    full = build_edges(a, tier, cut, sign_mode, 'neck_cut', tcfg)
    desc = build_edges(a, tier, cut, sign_mode, 'descending_only', tcfg)
    nn = tier['n']

    def _mask(global_mask):
        return local_mask(tier, global_mask, mask_semantics)

    group_local = {name: np.flatnonzero(_mask(m)) for name, m in groups['disjoint'].items()}
    brain_local = np.flatnonzero(_mask(groups['route_overlapping']['brain_side']))
    asc_local = np.flatnonzero(_mask(groups['route_overlapping']['crossing_ascending']))
    steps = spike_matrix.shape[0]

    def _ledger(edge_set):
        return conductance_ledger(spike_matrix, edge_set['We'], edge_set['Wi'],
                                  brain_local, group_local, c.delay_steps,
                                  step_slice=slice(0, steps))

    l_intact = _ledger(intact)
    l_full = _ledger(full)
    l_desc = _ledger(desc)
    severed_full = {k: l_intact[k] - l_full[k] for k in l_intact}
    severed_desc = {k: l_intact[k] - l_desc[k] for k in l_intact}
    asc_intact = conductance_ledger(spike_matrix, intact['We'], intact['Wi'],
                                    asc_local, group_local, c.delay_steps,
                                    step_slice=slice(0, steps))
    asc_full = conductance_ledger(spike_matrix, full['We'], full['Wi'],
                                  asc_local, group_local, c.delay_steps,
                                  step_slice=slice(0, steps))
    return {
        'method': ('one recorded spike train replayed through the intact and the cut '
                   'matrices; the difference is the conductance the severed rows carry, '
                   'with activity held fixed (so it isolates edge removal)'),
        'sign_mode': sign_mode,
        'at_brain_side_uS': {
            'intact_total': float(sum(l_intact.values())),
            'neck_cut_total': float(sum(l_full.values())),
            'descending_only_total': float(sum(l_desc.values())),
            'severed_by_neck_cut_total': float(sum(severed_full.values())),
            'severed_by_descending_only_total': float(sum(severed_desc.values())),
            'severed_by_neck_cut_per_group': severed_full,
            'severed_by_descending_only_per_group': severed_desc,
        },
        'at_ascending_side_uS': {
            'intact_total': float(sum(asc_intact.values())),
            'neck_cut_total': float(sum(asc_full.values())),
            'severed_by_neck_cut_total': float(sum(asc_intact.values()) - sum(asc_full.values())),
        },
        'leakage_check': {
            'severed_from_head_route_sensor_group_uS': severed_full['head_route_sensor'],
            'severed_from_body_route_sensor_group_uS': severed_full['body_route_sensor'],
            'note': ('the two route-sensor groups of the DISJOINT table never contain a '
                     'cell that is annotated as crossing, so the severed conductance '
                     'attributable to them is exactly zero by construction. The 2 '
                     'head-route cells and 412 body-route cells that the dataset also '
                     'annotates as crossing live in the crossing groups and are counted '
                     'there, which is the honest place for them.'),
        },
        'reachability_used': False,
    }
