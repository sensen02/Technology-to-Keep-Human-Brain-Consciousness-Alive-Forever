"""Bounded multirate scenario: BANC coarse tier + histamine vision + injury + HH neck.

FOUR previously separate experiments composed into ONE run:

  (a) coarse tier   -- a DETERMINISTIC CONNECTED SUBNETWORK of the real BANC
      connectome (data/flywire/banc_connectome.npz); receptor sign taken ONLY
      from pure `ann_nt_verified` labels,
  (b) visual drive -- engine.graded_vision continuous graded photoreceptor
      release, routed into the DEDICATED histamine conductance channel of
      engine.neural_hist (no longer borrowing the inhibitory channel),
  (c) injury        -- engine.injury_tissue InjuryPotassium (reduced
      electroneutral K exchange) + InjuryLigand (barrier rewrite) on clocks
      slower than the neural clock,
  (d) fine tier     -- engine.neural_active classical-HH cable for the neck,
      linking the brain-side coarse tier to a body-side output,
  (e) electrode     -- engine.electrode bipolar field on the fine cable through
      an explicit rigid transform, plus an ARTIFACT-ONLY recorder.

================================================================================
HONESTY / SCOPE (read before using any number from here)
================================================================================
* The composed eye->brain->neck->body path is a MIXTURE: the coarse tier is a
  real BANC subgraph, while the eye-side and body-side interface cells
  (IDEAL_EYE_IF, IDEAL_NECK_RELAY_IF, IDEAL_BODY_IF) are INVENTED placeholder
  cells. They are NOT anatomy, NOT reconstructed neurons and NOT present in the
  BANC dataset. Every region carries a label string saying exactly that.
* The BANC subnetwork is a SELECTION of at most ~20000 neurons drawn from
  153962 annotated neurons and 3037361 annotated connection rows. It is NOT the
  whole nervous system, NOT the whole brain, and its spiking statistics are not
  a physiological prediction.
* Every physiology parameter here is ILLUSTRATIVE and UNCALIBRATED: the point
  ConductanceParams, the synapse->microsiem scale, the per-edge synapse cap, the
  graded-release scale, the cable geometry, the HH temperature, the electrode
  conductivity/geometry and the injury gains.
* Both chemistry->neural gain parameters (`injury_k_target_gain_uS`,
  `injury_ligand_target_gain_uS`) default to ZERO and are labelled HYPOTHESIS.
* The reduced potassium chemistry is an externally clamped electroneutral salt
  exchange with an explicit reservoir ledger. It is NOT electrodiffusion, has no
  pumps, no ATP, no charge/energy ledger, and no survival/viability meaning.
* There is NO consciousness, viability, survival, recovery or "rescue" claim
  anywhere in this module. A "neck cut" is a sealed axial disconnect of an ideal
  cylinder section, not tissue damage.
* The recorder output contains electrode artifact (+ optional noise) ONLY. It is
  never Vm, never a neural recording, and no neural source is reconstructed.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import math
import numpy as np
from scipy import sparse

from .neural_hist import HistamineNetwork, HistamineParams
from .neural_active import ActiveRuntime, IdealCableSpec
from .graded_vision import GradedVision, pure_histamine_mask
from .injury_tissue import InjuryPotassium, InjuryLigand, PotassiumConfig
from .electrode import Contact, BipolarField, VoltageRecorder

BANC_RELATIVE_PATH = 'data/flywire/banc_connectome.npz'
MASSLESS_TOL = 1e-12

SCENARIOS = ('intact', 'sham', 'neck_cut', 'eye_loss', 'injury_at_phase')
COARSE_REGIONS = ('IDEAL_EYE_IF', 'IDEAL_NECK_RELAY_IF', 'IDEAL_BODY_IF')
FINE_REGION = 'neck_cable_distal'

#: Explicit, mandatory labels for the invented cells. These strings are written
#: into every output so no reader can mistake them for real anatomy.
IDEALIZED_LABELS = {
    'IDEAL_EYE_IF': 'IDEALIZED interface cell - NOT anatomy, NOT in the BANC dataset',
    'IDEAL_NECK_RELAY_IF': 'IDEALIZED interface cell - NOT anatomy, NOT in the BANC dataset',
    'IDEAL_BODY_IF': 'IDEALIZED interface cell - NOT anatomy, NOT in the BANC dataset',
    'BANC_SELECTED': 'real BANC root_id present in the deterministic subnetwork selection',
    'neck_cable': 'IDEALIZED classical-HH cylinder cable - NOT reconstructed anatomy',
}

# ---------------------------------------------------------------------------
# receptor mapping (explicit; never inferred from nt_pair, which is all -1)
# ---------------------------------------------------------------------------
PURE_EXC_LABEL = 'acetylcholine'
PURE_INH_LABELS = ('gaba', 'glutamate')
EXC_REVERSAL_MV = 0.0
INH_REVERSAL_MV = -70.0


def receptor_class(verified_label):
    """Map a pure `ann_nt_verified` label onto an explicit receptor class.

    Returns 'exc' (ACh, E=0 mV), 'inh' (GABA/glutamate, E=-70 mV) or None for
    every mixed label, every modulator and every unannotated cell. A None result
    means the edge is EXCLUDED and counted, never guessed.
    """
    if verified_label == PURE_EXC_LABEL:
        return 'exc'
    if verified_label in PURE_INH_LABELS:
        return 'inh'
    return None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubnetworkConfig:
    """Deterministic selection rule for the bounded BANC coarse tier."""
    max_neurons: int = 20000
    data_path: str = BANC_RELATIVE_PATH
    weight_scale_uS_per_synapse: float = 5e-4
    max_synapses_per_edge: int = 8
    include_pure_histamine_photoreceptors: bool = True
    max_pure_histamine_photoreceptors: int = 400

    def validate(self):
        if isinstance(self.max_neurons, bool) or not isinstance(self.max_neurons, int):
            raise ValueError('max_neurons must be an integer')
        if not 1 <= self.max_neurons <= 20000:
            raise ValueError('bounded prototype: max_neurons in [1, 20000]')
        if not np.isfinite(self.weight_scale_uS_per_synapse) or self.weight_scale_uS_per_synapse <= 0:
            raise ValueError('weight scale must be finite positive')
        if (isinstance(self.max_synapses_per_edge, bool)
                or not isinstance(self.max_synapses_per_edge, int)
                or self.max_synapses_per_edge < 1):
            raise ValueError('per-edge synapse cap must be a positive integer')
        if (isinstance(self.max_pure_histamine_photoreceptors, bool)
                or not isinstance(self.max_pure_histamine_photoreceptors, int)
                or self.max_pure_histamine_photoreceptors < 0):
            raise ValueError('photoreceptor budget must be a nonnegative integer')


@dataclass(frozen=True)
class MultirateConfig:
    scenario: str = 'intact'
    duration_ms: float = 40.0
    dt_ms: float = 0.05
    phase_ms: float = 15.0
    seed: int = 7

    # (b) vision
    light_start_ms: float = 5.0
    light_level: float = 1.0
    light_step_ms: float = 2.5
    visual_period_ms: float = 10.0
    visual_amplitude_nA: float = 0.6
    visual_scale_uS_per_synapse: float = 5e-3
    # Idealized placeholder currents (NOT measured drives):
    # `eye_pointer_gain_uS` scales the graded visual command into an injected
    # current on the IDEAL_EYE_IF placeholder cell; `brain_pointer_gain_uS` is a
    # constant arousal current spread over the selected BANC neurons so that the
    # coarse tier is not silent. Both are illustrative and explicitly labelled.
    eye_pointer_gain_uS: float = 0.05
    brain_pointer_gain_uS: float = 0.05

    # (c) slow chemistry / injury
    chemistry_dt_ms: float = 2.5
    potassium_dt_ms: float = 5.0
    injury_permeability_um_s: float = 0.05
    injury_reservoir_conductance_um3_s: float = 0.0
    injury_barrier_conductance_um3_s: float = 5.0
    # HYPOTHESIS gains mapping chemistry onto neural conductance. BOTH DEFAULT 0.
    injury_k_target_gain_uS: float = 0.0
    injury_ligand_target_gain_uS: float = 0.0
    injury_hist_shift_gain: float = 0.0

    # (d) fine neck tier
    cut_child_section: int = 1
    neck_sections: int = 3
    neck_nseg_per_section: int = 11
    neck_length_um: float = 600.0
    neck_diameter_um: float = 20.0
    neck_drive_uS_per_spike: float = 0.05
    neck_drive_pulse_nA: float = 20.0
    neck_drive_first_ms: float = 5.0
    neck_drive_period_ms: float = 10.0
    neck_drive_duration_ms: float = 0.5

    # (e) electrode / recorder
    stimulus_start_ms: float = 20.0
    stimulus_duration_ms: float = 2.0
    stimulus_nA: float = 200.0
    electrode_conductivity_S_m: float = 0.3
    recorder_noise_sd_mV: float = 0.0

    def validate(self):
        if self.scenario not in SCENARIOS:
            raise ValueError('unknown scenario')
        numeric = ('duration_ms', 'dt_ms', 'phase_ms', 'light_start_ms', 'light_level',
                   'light_step_ms', 'visual_period_ms', 'visual_amplitude_nA',
                   'visual_scale_uS_per_synapse', 'chemistry_dt_ms', 'potassium_dt_ms',
                   'injury_permeability_um_s', 'injury_reservoir_conductance_um3_s',
                   'injury_barrier_conductance_um3_s', 'injury_k_target_gain_uS',
                   'injury_ligand_target_gain_uS', 'injury_hist_shift_gain',
                   'neck_length_um', 'neck_diameter_um', 'neck_drive_uS_per_spike',
                   'neck_drive_pulse_nA', 'neck_drive_first_ms', 'neck_drive_period_ms',
                   'neck_drive_duration_ms', 'stimulus_start_ms', 'stimulus_duration_ms',
                   'stimulus_nA', 'electrode_conductivity_S_m', 'recorder_noise_sd_mV',
                   'eye_pointer_gain_uS', 'brain_pointer_gain_uS')
        for name in numeric:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + ' must be finite')
        if not 0 < self.dt_ms <= 0.1:
            raise ValueError('neural dt must be in (0, 0.1] ms')
        if not 0 < self.phase_ms < self.duration_ms:
            raise ValueError('phase must lie strictly inside the run')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('nonnegative integer seed required')
        if self.light_step_ms < self.dt_ms:
            raise ValueError('graded-vision interval cannot be shorter than the neural step')
        if self.chemistry_dt_ms <= self.dt_ms or self.potassium_dt_ms <= self.dt_ms:
            raise ValueError('chemistry and potassium clocks must be SLOWER than the neural clock')
        if self.potassium_dt_ms < self.chemistry_dt_ms:
            raise ValueError('potassium clock must not be faster than the chemistry clock')
        for name in ('duration_ms', 'phase_ms', 'light_start_ms', 'light_step_ms',
                     'chemistry_dt_ms', 'potassium_dt_ms', 'stimulus_start_ms',
                     'stimulus_duration_ms', 'neck_drive_first_ms', 'neck_drive_period_ms',
                     'neck_drive_duration_ms', 'visual_period_ms'):
            ratio = getattr(self, name) / self.dt_ms
            if ratio < 0 or not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
                raise ValueError(name + ' must align to the neural clock')
        for slow in ('chemistry_dt_ms', 'potassium_dt_ms'):
            step_ms = getattr(self, slow)
            span = self.duration_ms / step_ms
            if not math.isclose(span, round(span), abs_tol=1e-8, rel_tol=0):
                raise ValueError('duration must align to ' + slow)
            phase = self.phase_ms / step_ms
            if not math.isclose(phase, round(phase), abs_tol=1e-8, rel_tol=0):
                raise ValueError('phase must align to ' + slow)
        for name in ('light_level', 'visual_amplitude_nA', 'injury_barrier_conductance_um3_s',
                     'injury_k_target_gain_uS', 'injury_ligand_target_gain_uS',
                     'injury_hist_shift_gain', 'neck_drive_uS_per_spike',
                     'neck_drive_pulse_nA', 'stimulus_nA', 'recorder_noise_sd_mV',
                     'injury_permeability_um_s', 'injury_reservoir_conductance_um3_s',
                     'visual_scale_uS_per_synapse', 'neck_drive_period_ms',
                     'neck_drive_duration_ms', 'stimulus_duration_ms', 'light_step_ms',
                     'eye_pointer_gain_uS', 'brain_pointer_gain_uS'):
            if getattr(self, name) < 0:
                raise ValueError(name + ' must be nonnegative')
        if self.neck_drive_period_ms <= 0 or self.stimulus_duration_ms <= 0:
            raise ValueError('positive pulse period and stimulus duration required')
        if type(self.cut_child_section) is not int or not 0 < self.cut_child_section < self.neck_sections:
            raise ValueError('cut child must be a nonroot neck section index')


# ---------------------------------------------------------------------------
# (a) deterministic bounded subnetwork selection
# ---------------------------------------------------------------------------
def _stable_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:32]


def _induced_components(pre, post, node_mask, n, edge_rows=None):
    """Connected components (undirected) of an edge-induced subgraph.

    `edge_rows` selects the row universe (default: every annotated row). Every
    selected row with BOTH endpoints inside `node_mask` is an undirected edge.
    Returns (n_components, sizes, largest_component_node_indices).
    """
    inside = node_mask[pre] & node_mask[post]
    if edge_rows is not None:
        inside = inside & edge_rows
    rows = np.flatnonzero(inside)
    nodes = np.flatnonzero(node_mask)
    loc = -np.ones(n, dtype=np.int64)
    loc[nodes] = np.arange(nodes.size)
    g = sparse.coo_matrix((np.ones(rows.size), (loc[pre[rows]], loc[post[rows]])),
                          shape=(nodes.size, nodes.size))
    n_comp, comp = sparse.csgraph.connected_components(g, directed=False)
    sizes = np.bincount(comp, minlength=n_comp)
    return n_comp, sizes, nodes[comp == int(np.argmax(sizes))]


def select_banc_subnetwork(config=None, root=None):
    """Deterministic, documented, bounded, CONNECTED BANC subnetwork.

    SELECTION RULE (every step is deterministic; every tie is broken by ascending
    neuron index, so the result never depends on dict/set iteration order or on the
    process hash seed):

      1. annotated_degree_i = number of annotated connection rows whose pre OR post
         index is i (duplicate annotated rows are counted separately).
      2. Rank all neurons by (-annotated_degree, index); take the top `max_neurons`
         as the degree seed set S1 (= 20000 by default).
      3. Photosensitive seeds: rank every photoreceptor whose `ann_nt_verified` label
         is EXACTLY 'histamine' by (-annotated_degree, index) and take the first
         `max_pure_histamine_photoreceptors`. Mixed labels (e.g.
         'acetylcholine,histamine') and unannotated photoreceptors are NEVER added;
         they stay in the excluded-row accounting. These seeds are low-degree cells
         that the degree ranking in step 2 never reaches, so their ROWS (not just
         their endpoints) are carried forward explicitly.
      4. Seat closure: take the MODELED edge universe (every annotated row whose
         presynaptic label is pure ACh / pure GABA / pure glutamate, i.e. the rows
         the explicit receptor mapping can actually turn into a conductance edge)
         intersected with S1, plus the seed rows from step 3, and add both endpoints
         of every such row. Both step-2 neurons and step-3 visual seeds therefore
         come with all of their modelled edges.
      5. Largest connected component (undirected) of that induced graph. Neurons that
         the induced subgraph leaves unattached are dropped, so the final count can
         be well below the cap (a sparse photoreceptor whose only modelled partners
         sit outside the degree core is dropped here, and that is reported).
      6. If the result exceeds `final_cap` = max(max_neurons, 20000) +
         max_pure_histamine_photoreceptors, keep the `final_cap` highest-ranked
         neurons by (-annotated_degree, index) and re-impose step 5.

    Step 5 always runs last, so the returned subnetwork IS connected by construction
    over the modelled edges. `counts` and `excluded` are recomputed from the FINAL
    node set, never from an intermediate one, and the anatomical connectivity (over
    ALL annotated rows) is reported separately from the modelled connectivity.

    `counts` and `excluded` are recomputed from the FINAL node set, never from an
    intermediate one.
    """
    c = config or SubnetworkConfig()
    c.validate()
    path = c.data_path if root is None else str(root / c.data_path)
    with np.load(path, allow_pickle=False) as d:
        pre = np.asarray(d['pre'], dtype=np.int64)
        post = np.asarray(d['post'], dtype=np.int64)
        syn = np.asarray(d['syn'], dtype=np.float64)
        nt_pair = np.asarray(d['nt_pair'])
        cls = np.asarray(d['ann_class'])
        body_part = np.asarray(d['ann_body_part'])
        ver = np.asarray(d['ann_nt_verified'])
        root_ids = np.asarray(d['root_ids'])
    n = int(root_ids.size)
    degree = np.bincount(pre, minlength=n) + np.bincount(post, minlength=n)

    # step 2: degree-ranked seed set
    order = np.lexsort((np.arange(n), -degree))
    keep = np.zeros(n, bool)
    keep[order[:c.max_neurons]] = True
    n_seed = int(keep.sum())

    # Edge universes. `modeled` = rows the receptor mapping can actually turn into a
    # conductance edge (pure ACh / pure GABA / pure glutamate). `all_rows` = every
    # annotated row. Anatomical connectivity of the induced subgraph and MODELLED
    # connectivity are NOT the same thing, so both are measured and reported.
    labels_all = ver[pre]
    modeled_rows = (labels_all == PURE_EXC_LABEL) | np.isin(labels_all, list(PURE_INH_LABELS))

    # step 3: photosensitive seeds. A pure-histamine photoreceptor is a low-degree
    # cell whose visual rows are the ONLY way it enters the modelled graph, so the
    # seed set is an explicit list of row indices rather than a set of neurons.
    visual_global = np.flatnonzero(cls == 'photoreceptor_neuron')
    pure_hist = np.zeros(n, bool)
    pure_hist[visual_global] = ver[visual_global] == 'histamine'
    photo_candidates = np.flatnonzero(pure_hist)
    photo_order = photo_candidates[np.lexsort((photo_candidates, -degree[photo_candidates]))]
    photo_chosen = photo_order[:c.max_pure_histamine_photoreceptors]
    rows_of_photo = {}
    seed_rows = np.zeros(pre.shape, bool)
    if c.include_pure_histamine_photoreceptors:
        for idx in photo_chosen:
            rows = np.flatnonzero((pre == idx) | (post == idx))
            rows_of_photo[int(idx)] = rows
            seed_rows[rows] = True
            if rows.size:
                keep[idx] = True
                keep[pre[rows]] = True
                keep[post[rows]] = True
    n_after_photo = int(keep.sum())

    # step 4/5. The photosensitive seat set is the seed rows UNION every edge of the
    # induced modelled subgraph, so a visual seed and its targets cannot end up in a
    # different component or be split by the truncation below.
    edge_set = seed_rows | (modeled_rows & keep[pre] & keep[post])
    reachable = np.zeros(n, bool)
    reachable[pre[edge_set]] = True
    reachable[post[edge_set]] = True
    seat_set = (keep | reachable)
    n_comp, sizes, final = _induced_components(pre, post, seat_set, n, modeled_rows)
    reserved = pure_hist & seat_set
    photo_skipped = int(photo_chosen.size - np.count_nonzero(reserved))
    photo_reattached = int(np.count_nonzero(reserved))

    # step 7: size cap, then re-impose the largest component
    final_cap = max(c.max_neurons, 20000) + c.max_pure_histamine_photoreceptors
    truncated = False
    if final.size > final_cap:
        order3 = np.lexsort((final, -degree[final]))
        keep_cap = np.zeros(n, bool)
        keep_cap[final[order3[:final_cap]]] = True
        n_comp, sizes, final = _induced_components(pre, post, keep_cap, n, modeled_rows)
        truncated = True

    keep = np.zeros(n, bool)
    keep[final] = True
    final_index = np.flatnonzero(keep)
    new_local = -np.ones(n, dtype=np.int64)
    new_local[final_index] = np.arange(final_index.size)

    # receptor classification over the FINAL induced edge set
    rc = np.full(pre.shape, 'x', dtype='<U1')
    rc[labels_all == PURE_EXC_LABEL] = 'e'
    rc[np.isin(labels_all, list(PURE_INH_LABELS))] = 'i'
    kept_index = np.flatnonzero(keep[pre] & keep[post])
    exc_rows = kept_index[rc[kept_index] == 'e']
    inh_rows = kept_index[rc[kept_index] == 'i']
    other_rows = kept_index[rc[kept_index] == 'x']

    def _matrix(rows):
        if rows.size == 0:
            return sparse.csr_matrix((final_index.size, final_index.size))
        counts = np.minimum(syn[rows], c.max_synapses_per_edge)
        w = counts * c.weight_scale_uS_per_synapse
        return sparse.csr_matrix((w, (new_local[post[rows]], new_local[pre[rows]])),
                                 shape=(final_index.size, final_index.size))

    We, Wi = _matrix(exc_rows), _matrix(inh_rows)

    total_rows = int(pre.size)
    retained = int(exc_rows.size + inh_rows.size)
    outside = int(np.count_nonzero(~(keep[pre] & keep[post])))
    other_labels = ver[pre[other_rows]]
    # MUTUALLY EXCLUSIVE and EXHAUSTIVE classification of the excluded rows that lie
    # inside the subnetwork:
    #   unknown      -- no verified transmitter label at all
    #   histamine_only -- the verified label is exactly 'histamine'
    #   mixed        -- some other non-empty label (mixed transmitters, modulators, ...)
    # so unknown + histamine_only + mixed == number of inside-subnetwork excluded rows,
    # and outside + inside == total excluded rows.
    n_unknown = int(np.count_nonzero(other_labels == ''))
    n_histamine = int(np.count_nonzero(other_labels == 'histamine'))
    n_mixed = int(other_labels.size - n_unknown - n_histamine)
    excluded = {
        'total_annotated_rows_in_dataset': total_rows,
        'excluded_endpoint_outside_subnetwork': outside,
        'excluded_inside_subnetwork_rows': int(other_labels.size),
        'excluded_inside_subnetwork_rows_with_pure_histamine_label': n_histamine,
        'excluded_nonpure_label_breakdown': {
            'unknown': n_unknown,
            'histamine_only': n_histamine,
            'mixed': n_mixed,
            'breakdown_is_mutually_exclusive_and_exhaustive': bool(
                n_unknown + n_histamine + n_mixed == other_labels.size),
        },
        'excluded_inside_subnetwork_unknown_label': n_unknown,
        'excluded_inside_subnetwork_histamine_only': n_histamine,
        'excluded_inside_subnetwork_mixed_label': n_mixed,
        'excluded_mixed_histamine_rows_in_dataset':
            int(np.count_nonzero(ver[pre] == 'acetylcholine,histamine')),
        'retained_exc_rows': int(exc_rows.size),
        'retained_inh_rows': int(inh_rows.size),
        'retained_rows': retained,
        'excluded_rows_total': total_rows - retained,
        'accounting_identity_ok': bool(outside + other_labels.size == total_rows - retained),
        'nt_pair_unique_values': [int(v) for v in np.unique(nt_pair)],
        'nt_pair_used_for_sign': False,
        'sign_rule': 'acetylcholine->exc E=0 mV; gaba/glutamate->inh E=-70 mV; '
                     'mixed/unknown/histamine/modulator rows excluded and counted',
    }

    # pure-histamine visual projection restricted to the final set
    hist = pure_histamine_mask(ver, pre)
    visual_and_hist = hist & np.isin(pre, visual_global)
    vis_rows = np.flatnonzero(visual_and_hist)
    vis_inside = vis_rows[keep[pre[vis_rows]] & keep[post[vis_rows]]]
    visual_local = (np.sort(new_local[np.unique(pre[vis_inside])].astype(np.int64))
                    if vis_inside.size else np.zeros(0, np.int64))
    visual = {
        'pure_histamine_visual_rows_total': int(vis_rows.size),
        'pure_histamine_visual_rows_inside_subnetwork': int(vis_inside.size),
        'pure_histamine_visual_rows_outside_subnetwork': int(vis_rows.size - vis_inside.size),
        'pure_histamine_visual_presynaptic_selected': int(visual_local.size),
        'pure_histamine_visual_targets_selected':
            int(np.unique(new_local[post[vis_inside]]).size) if vis_inside.size else 0,
        'excluded_mixed_ach_histamine_visual_rows':
            int(np.count_nonzero((cls[pre] == 'photoreceptor_neuron')
                                 & (ver[pre] == 'acetylcholine,histamine'))),
        'excluded_unannotated_visual_rows':
            int(np.count_nonzero((cls[pre] == 'photoreceptor_neuron') & (ver[pre] == ''))),
        'pre_edges': new_local[pre[vis_inside]].copy(),
        'post_edges': new_local[post[vis_inside]].copy(),
        'global_pre_edges': pre[vis_inside].copy(),
        'global_post_edges': post[vis_inside].copy(),
        'syn_edges': syn[vis_inside].copy(),
        'visual_local_indices': visual_local,
    }

    # final-set recomputed counts (no stale intermediate numbers)
    n_comp_final, sizes_final, _ = _induced_components(pre, post, keep, n, modeled_rows)
    n_comp_anatomy, sizes_anatomy, _ = _induced_components(pre, post, keep, n)
    selected_ids = np.asarray(root_ids[final_index], dtype=np.int64)
    is_photo = pure_hist[final_index]
    selected_body_parts = body_part[final_index]
    body_part_counts = {str(part): int(np.count_nonzero(selected_body_parts == part))
                        for part in np.unique(selected_body_parts)}
    counts = {
        'neurons_total_in_dataset': n,
        'annotated_rows_total_in_dataset': total_rows,
        'seed_top_degree_neurons': n_seed,
        'photosensitive_seeds_requested': int(photo_chosen.size),
        'photosensitive_seeds_added_with_their_rows': int(np.count_nonzero(keep & pure_hist)),
        'photosensitive_seed_seats_reserved': int(np.count_nonzero(reserved)),
        'photosensitive_seeds_in_final_graph': int(np.count_nonzero(pure_hist[final_index])),
        'photosensitive_seeds_dropped_by_component_step': int(
            photo_chosen.size - np.count_nonzero(pure_hist[final_index])),
        'photosensitive_seeds_reattached_after_component_step': int(photo_reattached),
        'photosensitive_seeds_skipped_no_target_in_main_component': int(photo_skipped),
        'neurons_after_photo_seed_addition': n_after_photo,
        'seats_after_modelled_closure': int(seat_set.sum()),
        'components_of_candidate_set_before_last_step': int(n_comp),
        'largest_component_size_before_last_step': int(sizes.max()),
        'neurons_in_final_subnetwork': int(final_index.size),
        'final_selection_connected_components_modeled_edges': int(n_comp_final),
        'final_selection_is_connected_over_modeled_edges': bool(n_comp_final <= 1),
        'final_selected_nodes_not_reachable_from_largest_component': int(
            final.size - sizes_final.max()),
        'final_selection_components_all_annotated_rows': int(n_comp_anatomy),
        'final_selection_largest_component_all_annotated_rows': int(sizes_anatomy.max()),
        'modeled_edge_universe_rows': int(modeled_rows.sum()),
        'all_annotated_row_universe_rows': int(pre.size),
        'truncated_to_max_neurons': bool(truncated),
        'final_neuron_cap': int(final_cap),
        'pure_histamine_photoreceptors_in_final': int(is_photo.sum()),
        'other_photoreceptor_cells_in_final': int(np.count_nonzero(
            (cls[final_index] == 'photoreceptor_neuron') & ~is_photo)),
        'exc_matrix_nnz': int(We.nnz),
        'inh_matrix_nnz': int(Wi.nnz),
        'max_exc_weight_uS': float(We.data.max()) if We.nnz else 0.0,
        'max_inh_weight_uS': float(Wi.data.max()) if Wi.nnz else 0.0,
        'sum_exc_weight_uS': float(We.data.sum()) if We.nnz else 0.0,
        'sum_inh_weight_uS': float(Wi.data.sum()) if Wi.nnz else 0.0,
    }
    return {'n': int(final_index.size), 'global_index': final_index, 'root_ids': selected_ids,
            'We': We, 'Wi': Wi, 'visual': visual, 'counts': counts, 'excluded': excluded,
            'photo_seed_info': photo_seed_info(n, visual_global, pure_hist, photo_chosen),
            'body_part_counts': body_part_counts,
            'selection_rule': (
                'annotated_degree = # annotated rows touching the neuron (pre OR post); rank all '
                f'neurons by (-degree, index) and take the top {c.max_neurons}; rank '
                'pure-histamine photoreceptors by (-degree, index), take the first '
                f'{c.max_pure_histamine_photoreceptors} and add them plus their row endpoints; '
                'take the MODELLED edge universe (rows the explicit receptor mapping can turn '
                'into a conductance edge) plus the seed rows of the chosen pure-histamine '
                'photoreceptors, and close it over both endpoints; take the largest connected '
                'component of that induced graph (undirected); if it exceeds '
                f'{final_cap} neurons, keep the {final_cap} highest-ranked neurons and re-impose '
                'the largest connected component. Deterministic, index-tie-broken, size-bounded, '
                'and connected by construction OVER THE MODELLED EDGES.'),
            'dataset': {'path': str(path), 'neurons': n, 'annotated_rows': total_rows},
            'digest': {'selected_root_ids_sha256_32': _stable_digest(selected_ids),
                       'exc_matrix_sha256_32': _stable_digest(
                           We.indptr, We.indices, We.data.astype(np.float64)),
                       'inh_matrix_sha256_32': _stable_digest(
                           Wi.indptr, Wi.indices, Wi.data.astype(np.float64))}}


def photo_seed_info(n, visual_global, pure_hist, photo_chosen):
    """Descriptive block about the photosensitive-seed step of the selection."""
    return {
        'photoreceptors_total': int(visual_global.size),
        'pure_histamine_photoreceptors_total': int(pure_hist.sum()),
        'mixed_or_unannotated_photoreceptors_excluded':
            int(visual_global.size - int(pure_hist.sum())),
        'pure_histamine_photoreceptors_requested': int(photo_chosen.size),
        'rule': 'rank pure-histamine photoreceptors by (-annotated_degree, index), take the first '
                'max_pure_histamine_photoreceptors, add them and their annotated row endpoints; '
                'a pure-histamine photoreceptor is a low-degree cell that never survives the '
                'top-degree seed set on its own',
    }


# ---------------------------------------------------------------------------
# (c) slow chemistry / injury layer with completed-interval semantics
# ---------------------------------------------------------------------------
CHEM_SHAPE = (7, 7, 1)


def barrier_edges(conductance_um3_s, shape=CHEM_SHAPE):
    """Leaky x/y face barrier map for the injury ligand barrier rewrite."""
    edges = {}
    nx, ny, nz = shape
    for j in range(ny):
        for k in range(nz):
            for i in (0, nx - 2):
                a = np.ravel_multi_index((i, j, k), shape)
                b = np.ravel_multi_index((i + 1, j, k), shape)
                edges[(min(a, b), max(a, b))] = float(conductance_um3_s)
    for i in range(nx):
        for k in range(nz):
            for j in (0, ny - 2):
                a = np.ravel_multi_index((i, j, k), shape)
                b = np.ravel_multi_index((i, j + 1, k), shape)
                edges[(min(a, b), max(a, b))] = float(conductance_um3_s)
    return edges


class SlowInjuryLayer:
    """Owns the slow clocks. Advances ONLY at completed intervals.

    Semantics (no future leakage):
      * neural step k uses the ligand response and potassium state produced by the
        last COMPLETED slow interval, i.e. from slow time <= k*dt;
      * the slow models advance only after the neural step that ends a slow
        interval;
      * the injury event rewrites RATES ONLY (permeability, reservoir
        conductance, barrier map, hypothesis gains). It moves no state array, so
        both mass jumps are exactly zero and are asserted, not clipped.
    """

    def __init__(self, config):
        c = config
        self.c = c
        self.chem_every = round(c.chemistry_dt_ms / c.dt_ms)
        self.k_every = round(c.potassium_dt_ms / c.dt_ms)
        self.chem_dt_s = c.chemistry_dt_ms / 1000.0
        self.k_dt_s = c.potassium_dt_ms / 1000.0
        self.phase_step = round(c.phase_ms / c.dt_ms)
        initial = np.zeros(CHEM_SHAPE)
        initial[0, 0, 0] = 4.0
        initial[0, 1, 0] = 1.0
        self.ligand = InjuryLigand(shape=CHEM_SHAPE, initial_nM=initial,
                                   receptor_capacity_nM=2.0, kon_nM_inv_s=50.0,
                                   koff_s=10.0, response_tau_s=0.02)
        self.potassium = InjuryPotassium(PotassiumConfig())
        self.ek_baseline_mV = float(self.potassium.ek_mV)
        self.injury_applied = False
        self.injury_step = None
        self.events = []
        self.history = {name: [value] for name, value in (
            ('union_time_s', 0.0), ('chem_time_s', 0.0), ('k_time_s', 0.0),
            ('response', float(self.ligand.model.response[0, 0, 0])),
            ('free_nM', float(self.ligand.model.free_nM[0, 0, 0])),
            ('outside_mM', float(self.potassium.outside_mM)),
            ('ek_mV', float(self.potassium.ek_mV)),
            ('permeability_um_s', float(self.potassium.permeability_um_s)),
            ('barrier_active', False),
            ('ligand_mass_error', float(self.ligand.mass_balance_error())),
            ('k_mass_error', float(self.potassium.mass_balance_error())))}

    def neural_inputs(self, k_gain_uS, ligand_gain_uS, hist_shift_gain):
        """State handed to one neural step, built ONLY from completed intervals."""
        response = float(self.ligand.model.response[0, 0, 0])
        ek = float(self.potassium.ek_mV)
        exc_factor = -np.expm1(-self.c.dt_ms / 8.0)
        return {
            'response': response,
            'ek_mV': ek,
            'k_increment_uS': float(k_gain_uS * (ek - self.ek_baseline_mV) * exc_factor),
            'ligand_increment_uS': float(ligand_gain_uS * response * exc_factor),
            'k_shift_mV': float(hist_shift_gain * (ek - self.ek_baseline_mV)),
            'permeability_um_s': float(self.potassium.permeability_um_s),
            'barrier_active': bool(self.ligand.events),
            'chemistry_time_s': float(self.ligand.time_s),
            'potassium_time_s': float(self.potassium.time_s),
        }

    def apply_injury(self, step_index):
        """RATE-ONLY injury event: rewrite permeability/reservoir and ligand barrier."""
        if self.injury_applied:
            raise RuntimeError('injury applied twice')
        c = self.c
        k_before = self.potassium.total_amount()
        l_before = self.ligand.total_amount()
        self.potassium.apply_injury(c.injury_permeability_um_s,
                                    c.injury_reservoir_conductance_um3_s)
        self.ligand.switch_barrier(barrier_edges(c.injury_barrier_conductance_um3_s))
        k_jump = self.potassium.total_amount() - k_before
        l_jump = self.ligand.total_amount() - l_before
        for jump, scale, name in ((k_jump, k_before, 'potassium'), (l_jump, l_before, 'ligand')):
            if abs(jump) > MASSLESS_TOL * max(1.0, abs(scale)):
                raise FloatingPointError(name + ' injury event changed mass')
        self.injury_applied = True
        self.injury_step = int(step_index)
        record = {'kind': 'injury_rates_rewritten', 'step': int(step_index),
                  'time_ms': step_index * c.dt_ms,
                  'slow_time_s': {'chemistry': self.ligand.time_s,
                                  'potassium': self.potassium.time_s},
                  'permeability_um_s': float(self.potassium.permeability_um_s),
                  'reservoir_conductance_um3_s': float(self.potassium.reservoir_conductance_um3_s),
                  'barrier_edges_rewritten': len(barrier_edges(c.injury_barrier_conductance_um3_s)),
                  'potassium_mass_jump': float(k_jump),
                  'ligand_mass_jump': float(l_jump),
                  'potassium_events': list(self.potassium.events),
                  'ligand_events': list(self.ligand.events)}
        self.events.append(record)
        return record

    def advance(self, steps_done):
        """Advance whichever slow clock completed an interval at this neural-step end."""
        record = {'chemistry_advanced': False, 'potassium_advanced': False,
                  'chemistry_time_s_before': float(self.ligand.time_s),
                  'potassium_time_s_before': float(self.potassium.time_s)}
        if steps_done % self.chem_every == 0:
            self.ligand.step(self.chem_dt_s)
            record['chemistry_advanced'] = True
        if steps_done % self.k_every == 0:
            self.potassium.step(self.k_dt_s)
            record['potassium_advanced'] = True
        if record['chemistry_advanced'] or record['potassium_advanced']:
            # The two clocks run at different rates, so ONE history array carries the
            # union of their completed-interval times; each series is then a pure
            # step function on its own clock and must be read against its own time
            # array (chem_time_s for the ligand chemistry, k_time_s for potassium).
            h = self.history
            h['union_time_s'].append(steps_done * self.c.dt_ms / 1000.0)
            h['chem_time_s'].append(float(self.ligand.time_s))
            h['k_time_s'].append(float(self.potassium.time_s))
            h['response'].append(float(self.ligand.model.response[0, 0, 0]))
            h['free_nM'].append(float(self.ligand.model.free_nM[0, 0, 0]))
            h['outside_mM'].append(float(self.potassium.outside_mM))
            h['ek_mV'].append(float(self.potassium.ek_mV))
            h['permeability_um_s'].append(float(self.potassium.permeability_um_s))
            h['barrier_active'].append(bool(self.ligand.events))
            h['ligand_mass_error'].append(float(self.ligand.mass_balance_error()))
            h['k_mass_error'].append(float(self.potassium.mass_balance_error()))
        return record


# ---------------------------------------------------------------------------
# (e) explicit transform for the electrode field
# ---------------------------------------------------------------------------
#: The neck cable's local +x is mapped into world space by this explicit rigid
#: transform. The placement is INVENTED: the ideal cable has no real anatomy.
NECK_ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
NECK_TRANSLATION_UM = np.array([100.0, -600.0, 50.0])
FIELD_REFERENCE_UM = (100.0, 1500.0, 50.0)
FIELD_CONTACTS = (Contact((140.0, -500.0, 50.0), 20.0), Contact((140.0, 500.0, 50.0), 20.0))
RECORDER_POSITION_UM = (140.0, -400.0, 50.0)


def neck_segment_world_um(cable):
    """Local segment centers -> world um through the explicit rigid transform."""
    return cable.segment_xyz_um() @ NECK_ROTATION.T + NECK_TRANSLATION_UM


# ---------------------------------------------------------------------------
# the composed run
# ---------------------------------------------------------------------------
PROBE_NAMES = ('eye_if_mV', 'relay_if_mV', 'body_if_mV',
               'neck_proximal_mV', 'neck_distal_mV',
               'chemical_response', 'chemical_free_target_nM',
               'outside_potassium_mM', 'ek_mV',
               'chemical_clock_s', 'potassium_clock_s',
               'histamine_target_mean_mV', 'histamine_target_g_hist_uS',
               'histamine_target_spike_fraction')
INPUT_NAMES = ('light_level', 'visual_command_nA', 'histamine_increment_uS',
               'histamine_steady_target_uS', 'k_increment_uS', 'ligand_increment_uS',
               'hist_reversal_shift_mV', 'extra_exc_events_uS', 'extra_exc_events_raw_uS',
               'eye_pointer_current_nA',
               'brain_pointer_current_nA', 'coarse_recurrent_conductance_uS', 'neck_drive_pulse_nA',
               'cable_in_event_uS', 'field_proximal_mV', 'field_distal_mV',
               'artifact_unfiltered_mV', 'artifact_recorded_mV', 'recorder_neural_unfiltered_mV',
               'response_used', 'ek_used_mV',
               'chemical_clock_used_s', 'potassium_clock_used_s')


def run_multirate_scenario(config=None, subnetwork=None, subnetwork_config=None, root=None):
    """Run one scenario. Returns arrays + JSON-compatible metadata; writes nothing."""
    c = config or MultirateConfig()
    c.validate()
    s = subnetwork if subnetwork is not None else select_banc_subnetwork(subnetwork_config, root)
    n = round(c.duration_ms / c.dt_ms)
    phase_step = round(c.phase_ms / c.dt_ms)
    light_every = round(c.light_step_ms / c.dt_ms)
    light_start_step = round(c.light_start_ms / c.dt_ms)

    # ---- coarse tier (a)+(b): dedicated histamine channel ------------------
    params = HistamineParams()
    net = HistamineNetwork(s['n'], params, c.dt_ms).set_connectivity(s['We'], s['Wi'])
    visual_local = np.asarray(s['visual']['visual_local_indices'], dtype=np.int64)
    visual_pre = np.asarray(s['visual']['pre_edges'], dtype=np.int64)
    visual_post = np.asarray(s['visual']['post_edges'], dtype=np.int64)
    visual_syn = np.asarray(s['visual']['syn_edges'], dtype=np.float64)
    # Every edge handed to GradedVision is already an exact pure-histamine row
    # (selected by pure_histamine_mask over ann_nt_verified), so the mask is all-True
    # and GradedVision performs no further filtering of its own.
    visual_pure = np.ones(visual_pre.size, bool)
    gv = GradedVision(s['n'], visual_local, visual_pre, visual_post, visual_syn, visual_pure,
                      dt_ms=c.light_step_ms,
                      scale_uS_per_synapse=c.visual_scale_uS_per_synapse,
                      e_hist_mV=params.hist_reversal_mV,
                      tau_release_ms=params.hist_tau_ms)
    hist_targets = np.unique(gv.post) if gv.post.size else np.zeros(0, np.int64)

    # ---- (c) slow layer and (d) fine cable ---------------------------------
    slow = SlowInjuryLayer(c)
    spec = IdealCableSpec(length_um=c.neck_length_um, diameter_um=c.neck_diameter_um,
                          sections=c.neck_sections, nseg_per_section=c.neck_nseg_per_section)
    field = BipolarField(FIELD_CONTACTS, c.electrode_conductivity_S_m, FIELD_REFERENCE_UM)
    recorder = VoltageRecorder(RECORDER_POSITION_UM, FIELD_REFERENCE_UM,
                               c.electrode_conductivity_S_m, 1000.0,
                               c.recorder_noise_sd_mV, c.seed)

    probes = {name: np.zeros(n + 1) for name in PROBE_NAMES}
    inputs = {name: np.zeros(n) for name in INPUT_NAMES}
    events, injuries = [], []
    emissions, coarse_emissions, fine_emissions = [], [], []

    with ActiveRuntime(c.dt_ms, {'exc': (params.exc_reversal_mV, params.exc_tau_ms),
                                 'inh': (params.inh_reversal_mV, params.inh_tau_ms)}) as rt:
        cable = rt.add_cable('neck', spec)
        rt.reset()
        world = neck_segment_world_um(cable)
        unit_field = field.potential_mV(world, (1.0, -1.0))
        if not np.any(unit_field):
            raise RuntimeError('electrode field transform produced an all-zero field')
        drive = np.zeros(len(cable.segments))
        drive[0] = c.neck_drive_pulse_nA
        # IDEALIZED placeholder drive: a uniform current over the selected BANC
        # neurons so the coarse tier is not silent. ILLUSTRATIVE, not measured.
        drive_current = np.zeros(net.n)
        drive_current[3:] = c.brain_pointer_gain_uS

        def sample(i):
            probes['eye_if_mV'][i] = net.v[0]
            probes['relay_if_mV'][i] = net.v[1]
            probes['body_if_mV'][i] = net.v[2]
            volts = cable.voltage_mV()
            probes['neck_proximal_mV'][i] = volts[0]
            probes['neck_distal_mV'][i] = volts[-1]
            probes['chemical_response'][i] = float(slow.ligand.model.response[0, 0, 0])
            probes['chemical_free_target_nM'][i] = float(slow.ligand.model.free_nM[0, 0, 0])
            probes['outside_potassium_mM'][i] = float(slow.potassium.outside_mM)
            probes['ek_mV'][i] = float(slow.potassium.ek_mV)
            probes['chemical_clock_s'][i] = float(slow.ligand.time_s)
            probes['potassium_clock_s'][i] = float(slow.potassium.time_s)

        def sample_targets(i):
            if hist_targets.size:
                probes['histamine_target_mean_mV'][i] = float(np.mean(net.v[hist_targets]))
                probes['histamine_target_g_hist_uS'][i] = float(np.sum(net.g_hist[hist_targets]))
                probes['histamine_target_spike_fraction'][i] = float(
                    np.count_nonzero(net.refractory_until_ms[hist_targets] == net.t_ms)
                    / hist_targets.size)

        sample(0)
        sample_targets(0)
        pending_cable_uS = 0.0
        for k in range(n):
            t = k * c.dt_ms
            post = k >= phase_step

            # ---- graded vision at its own (slower) interval -----------------
            light_now = c.light_level if k >= light_start_step else 0.0
            visual_command = (c.visual_amplitude_nA * (1.0 + math.sin(2 * math.pi * t / c.visual_period_ms)) / 2.0
                              if k >= light_start_step else 0.0)
            if c.scenario == 'eye_loss' and post:
                light_now = 0.0
                visual_command = 0.0
            if k % light_every == 0:
                gv.step(light_now, c.light_step_ms)
                steady_target = np.asarray(gv.last_target_uS, float)
            else:
                steady_target = np.maximum(0.0, np.asarray(gv.last_target_uS, float)
                                           * np.exp(-c.dt_ms / gv.tau))
            gv.last_target_uS = steady_target
            hist_events = steady_target * (1.0 - np.exp(-c.dt_ms / gv.tau))

            # ---- scheduled phase intervention ------------------------------
            if k == phase_step:
                if c.scenario == 'neck_cut':
                    cable.cut_axial(c.cut_child_section)
                    events.append({'event_id': 'phase:0', 'time_ms': t,
                                   'kind': 'axial_disconnect_sealed_ends',
                                   'child_section': c.cut_child_section,
                                   'note': 'sealed ideal-section disconnect, NOT tissue damage'})
                elif c.scenario == 'eye_loss':
                    events.append({'event_id': 'phase:0', 'time_ms': t,
                                   'kind': 'ideal_eye_interface_drive_zeroed',
                                   'child_section': None,
                                   'note': 'graded histamine drive zeroed; chemistry and K untouched'})
                elif c.scenario == 'sham':
                    events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': 'sham_noop',
                                   'child_section': None,
                                   'note': 'scheduled phase reached, nothing changed'})
                else:
                    events.append({'event_id': 'phase:0', 'time_ms': t, 'kind': 'phase_noop',
                                   'child_section': None, 'note': 'no intervention scheduled'})
                if c.scenario == 'injury_at_phase':
                    injuries.append({'event_id': 'injury:0', **slow.apply_injury(k)})

            # ---- electrode field on the fine cable (explicit transform) -----
            stim = (c.stimulus_nA
                    if c.stimulus_start_ms <= t < c.stimulus_start_ms + c.stimulus_duration_ms
                    else 0.0)
            cable.set_extracellular_voltage_mV(unit_field * stim)

            # ---- fixed fine-cable drive pulse train (HYPOTHESIS amplitude) --
            pulse = 1.0 if (t >= c.neck_drive_first_ms
                            and ((t - c.neck_drive_first_ms) % c.neck_drive_period_ms)
                            < c.neck_drive_duration_ms) else 0.0
            cable.set_inward_current_nA(drive * pulse)

            # ---- coarse tier step ------------------------------------------
            # currents: constant idealized "brain pointer" on every BANC neuron
            # (so the coarse tier is not silent) plus a graded visual pointer on
            # the IDEAL_EYE_IF placeholder. BOTH are illustrative placeholder
            # drives, not measured phototransduction or brain activity.
            si = slow.neural_inputs(c.injury_k_target_gain_uS,
                                    c.injury_ligand_target_gain_uS,
                                    c.injury_hist_shift_gain)
            raw_extra = (si['k_increment_uS'] + si['ligand_increment_uS']) / (
                -np.expm1(-c.dt_ms / params.exc_tau_ms))
            # A conductance increment is nonnegative by definition. If the hypothesis
            # gain makes the reduced K diagnostic drive negative, that K effect is NOT
            # expressible in the excitatory channel: it is clamped to zero and recorded
            # in the raw diagnostic array rather than silently flipped into inhibition.
            extra = max(0.0, float(raw_extra))
            point_current = drive_current.copy()
            point_current[0] += visual_command
            spiked = net.step(point_current, extra, 0.0, hist_events)
            for region, idx in zip(COARSE_REGIONS, (0, 1, 2)):
                if spiked[idx]:
                    coarse_emissions.append((float(net.t_ms), region))
                    emissions.append({'event_id': 'spike:%d' % len(emissions),
                                      'time_ms': float(net.t_ms), 'region': region,
                                      'site': 'coarse point tier (IDEALIZED interface cell)'})

            # ---- cross-tier coupling into the fine cable --------------------
            cable_in_event = 0.0
            if spiked[1]:
                pending_cable_uS += c.neck_drive_uS_per_spike
            if pending_cable_uS > 0:
                cable.add_conductance_uS('exc', pending_cable_uS)
                cable_in_event = pending_cable_uS
                pending_cable_uS = 0.0
            cable_spikes = rt.step()
            if 'neck' in cable_spikes:
                fine_emissions.append(float(rt.t_ms))
                emissions.append({'event_id': 'spike:%d' % len(emissions),
                                  'time_ms': float(rt.t_ms), 'region': FINE_REGION,
                                  'site': 'ideal HH cable distal end (body side)'})

            rec = recorder.step(c.dt_ms, stimulus_field=field,
                                stimulus_currents_nA=(stim, -stim))
            values = {
                'light_level': light_now, 'visual_command_nA': visual_command,
                'histamine_increment_uS': float(hist_events.sum()),
                'histamine_steady_target_uS': float(steady_target.sum()),
                'k_increment_uS': si['k_increment_uS'],
                'ligand_increment_uS': si['ligand_increment_uS'],
                'hist_reversal_shift_mV': si['k_shift_mV'],
                'extra_exc_events_uS': float(extra),
                'extra_exc_events_raw_uS': float(raw_extra),
                'eye_pointer_current_nA': visual_command,
                'brain_pointer_current_nA': float(drive_current[3]) if net.n > 3 else 0.0,
                'coarse_recurrent_conductance_uS': float(net.ge.sum() + net.gi.sum()),
                'neck_drive_pulse_nA': float(drive[0] * pulse),
                'cable_in_event_uS': float(cable_in_event),
                'field_proximal_mV': float(unit_field[0] * stim),
                'field_distal_mV': float(unit_field[-1] * stim),
                'artifact_unfiltered_mV': rec['artifact_unfiltered_mV'],
                'recorder_neural_unfiltered_mV': rec['neural_unfiltered_mV'],
                'artifact_recorded_mV': rec['measured_mV'],
                'response_used': si['response'], 'ek_used_mV': si['ek_mV'],
                'chemical_clock_used_s': si['chemistry_time_s'],
                'potassium_clock_used_s': si['potassium_time_s'],
            }
            for name in INPUT_NAMES:
                inputs[name][k] = values[name]

            slow.advance(k + 1)
            sample(k + 1)
            sample_targets(k + 1)

    metrics = _metrics(c, net, n, coarse_emissions, fine_emissions, slow, hist_targets, gv,
                       cable.cut_children, s)
    regions = {
        'coarse_banc_subnetwork': {
            'label': IDEALIZED_LABELS['BANC_SELECTED'],
            'honesty': 'a deterministic bounded SELECTION, not the whole nervous system',
            'neurons': int(s['n'])},
        'IDEAL_EYE_IF': {'label': IDEALIZED_LABELS['IDEAL_EYE_IF'],
                         'note': 'explicit placeholder used where no real anatomy exists'},
        'IDEAL_NECK_RELAY_IF': {'label': IDEALIZED_LABELS['IDEAL_NECK_RELAY_IF'],
                                'note': 'explicit placeholder used where no real anatomy exists'},
        'IDEAL_BODY_IF': {'label': IDEALIZED_LABELS['IDEAL_BODY_IF'],
                          'note': 'explicit placeholder used where no real anatomy exists'},
        'neck_cable': {'label': IDEALIZED_LABELS['neck_cable'],
                       'note': 'the fine tier representing the neck; body side = distal section end'},
    }
    return {'config': asdict(c), 'times_ms': np.arange(n + 1) * c.dt_ms,
            'input_times_ms': np.arange(n) * c.dt_ms,
            'probes': probes, 'inputs': inputs, 'events': events + injuries,
            'injuries': injuries, 'emissions': emissions,
            'metrics': metrics, 'regions': regions,
            'provenance': _provenance(c, s, gv, slow, spec, unit_field, hist_targets),
            'slow_history': {k: np.asarray(v) for k, v in slow.history.items()},
            'histamine': {'targets': hist_targets, 'visual_local_indices': visual_local,
                          'last_steady_target_uS': np.asarray(gv.last_target_uS, float)}}


def _metrics(c, net, steps, coarse_emissions, fine_emissions, slow, hist_targets, gv,
             cut_children, s):
    metrics = {}
    for region in COARSE_REGIONS:
        for phase, lo, hi in (('baseline', 0.0, c.phase_ms), ('post', c.phase_ms, c.duration_ms)):
            metrics[region + '_' + phase + '_spikes'] = int(
                sum(1 for t, r in coarse_emissions if lo < t <= hi and r == region))
    for phase, lo, hi in (('baseline', 0.0, c.phase_ms), ('post', c.phase_ms, c.duration_ms)):
        metrics[FINE_REGION + '_' + phase + '_spikes'] = int(
            sum(1 for t in fine_emissions if lo < t <= hi))
    metrics['fine_tier_baseline_spikes'] = metrics[FINE_REGION + '_baseline_spikes']
    metrics['fine_tier_post_spikes'] = metrics[FINE_REGION + '_post_spikes']
    metrics['body_side_post_spikes'] = metrics[FINE_REGION + '_post_spikes']
    metrics['coarse_interface_baseline_spikes'] = int(
        sum(1 for t, _ in coarse_emissions if t <= c.phase_ms))
    metrics['coarse_interface_post_spikes'] = int(
        sum(1 for t, _ in coarse_emissions if t > c.phase_ms))
    metrics['whole_coarse_spike_count'] = int(net.spike_count.sum())
    metrics['interface_cell_spike_counts'] = {name: int(net.spike_count[i])
                                              for i, name in enumerate(COARSE_REGIONS)}
    metrics['banc_neuron_spike_count'] = int(net.spike_count[3:].sum())
    metrics['banc_neurons'] = int(s['n'])
    metrics['neural_steps'] = int(steps)
    metrics['visual_target_posts'] = int(hist_targets.size)
    metrics['visual_edge_count'] = int(gv.post.size)
    metrics['histamine_increment_total_uS'] = float(net.hist_increment_total_uS)
    metrics['final_histamine_target_uS'] = float(gv.last_target_uS.sum())
    metrics['histamine_delivered_total_uS'] = float(net.hist_increment_total_uS)
    metrics['cut_children'] = sorted(cut_children)
    metrics['potassium_events'] = list(slow.potassium.events)
    metrics['ligand_events'] = list(slow.ligand.events)
    metrics['injury_applied'] = bool(slow.injury_applied)
    metrics['injury_step'] = slow.injury_step
    metrics['potassium_mass_balance_error_mM_um3'] = float(slow.potassium.mass_balance_error())
    metrics['ligand_mass_balance_error_nM_um3'] = float(slow.ligand.mass_balance_error())
    metrics['final_chemical_time_s'] = float(slow.ligand.time_s)
    metrics['final_potassium_time_s'] = float(slow.potassium.time_s)
    metrics['ek_baseline_mV'] = float(slow.ek_baseline_mV)
    metrics['ek_final_mV'] = float(slow.potassium.ek_mV)
    metrics['outside_potassium_final_mM'] = float(slow.potassium.outside_mM)
    metrics['permeability_um_s_final'] = float(slow.potassium.permeability_um_s)
    metrics['reservoir_conductance_final_um3_s'] = float(slow.potassium.reservoir_conductance_um3_s)
    metrics['barrier_rewrites'] = len(slow.ligand.events)
    return metrics


def _provenance(c, s, gv, slow, spec, unit_field, hist_targets):
    return {
        'status': 'BOUNDED COMPOSED PROTOTYPE - illustrative and uncalibrated',
        'composition': ('real BANC subgraph (coarse) + invented idealized interface cells + '
                        'invented classical-HH cable (neck) + invented bipolar electrode field + '
                        'reduced injury chemistry'),
        'mixes_real_and_invented': True,
        'coarse_tier': {
            'source': 'data/flywire/banc_connectome.npz',
            'selection_rule': s['selection_rule'],
            'selection_counts': s['counts'],
            'excluded_edge_accounting': s['excluded'],
            'receptor_mapping': {
                'acetylcholine': 'exc, E=0 mV (explicit assumption)',
                'gaba': 'inh, E=-70 mV (explicit assumption)',
                'glutamate': 'inh, E=-70 mV (explicit assumption)',
                'everything else (unknown/mixed/histamine/modulator)': 'EXCLUDED and counted'},
            'sign_inference': 'never from nt_pair (all values -1); annotation labels only',
            'edge_weight': 'min(syn, cap) * scale_uS_per_synapse, ILLUSTRATIVE reduction; '
                           'synapse counts are NOT conductances',
            'parameters': HistamineParams().provenance()},
        'visual_drive': {
            'source': 'engine.graded_vision pure-histamine edges',
            'channel': 'engine.neural_hist dedicated g_hist (own E_hist/tau_hist)',
            'graded_interval_ms': c.light_step_ms,
            'between_interval_rule': 'steady target propagated by exact exp(-dt/tau_hist); '
                                     'within-interval event = target*(1-exp(-dt/tau_hist))',
            'inhibitory_channel_borrowed': False,
            'describe': gv.describe(),
            'visual_scale_uS_per_synapse': c.visual_scale_uS_per_synapse,
            'targets': int(hist_targets.size),
            'presynaptic_visual_cells_selected': int(s['visual']['visual_local_indices'].size),
            'biology_caveat': ('BANC photoreceptor output rows are histamine/mixed/unknown; only exact '
                               'pure-histamine labels are used, so the projection is explicitly partial')},
        'injury': {
            'potassium': 'InjuryPotassium: externally clamped electroneutral reduced K exchange, NOT '
                         'electrodiffusion; no pumps/ATP/charge ledger; reservoir ledger explicit',
            'ligand': 'InjuryLigand: full barrier rewrite with zero mass jump',
            'k_gain_uS': c.injury_k_target_gain_uS,
            'ligand_gain_uS': c.injury_ligand_target_gain_uS,
            'hist_shift_gain': c.injury_hist_shift_gain,
            'gains_status': 'HYPOTHESIS; all three default to 0.0 = no chemistry->neural coupling',
            'clocks': {'neural_dt_ms': c.dt_ms, 'chemistry_dt_ms': c.chemistry_dt_ms,
                       'potassium_dt_ms': c.potassium_dt_ms,
                       'semantics': 'completed intervals only; neural step k uses slow state from '
                                    'intervals that ENDED at or before k*dt; slow models advance after '
                                    'the neural step that closes the interval'},
            'ek_mV_baseline': float(slow.ek_baseline_mV),
            'ek_mV_final': float(slow.potassium.ek_mV),
            'ek_diagnostic_caveat': ('the Nernst value is a diagnostic of the reduced model; with a '
                                     'depleting closed extracellular pool it can go extreme, which is a '
                                     'model-insufficiency artifact, NOT resting potential and NOT glial '
                                     'protection')},
        'fine_tier': {
            'spec': spec.provenance(),
            'cut': 'sealed axial disconnect of one ideal cylinder section; no membrane damage, no wound leak',
            'drive': 'fixed focal IClamp pulse train at segment 0 (HYPOTHESIS amplitude) plus '
                     'hypothesis-weighted relay events from the coarse tier',
            'body_side_definition': 'distal end of the last cable section',
            'anatomy_caveat': 'ideal straight uniform cylinder, NOT the fly neck'},
        'electrode': {
            'field': 'homogeneous %.3g S/m bipolar spherical-volume contacts' % c.electrode_conductivity_S_m,
            'contacts': [asdict(x) for x in FIELD_CONTACTS],
            'reference_um': list(FIELD_REFERENCE_UM),
            'rotation_local_to_world': NECK_ROTATION.tolist(),
            'translation_um': NECK_TRANSLATION_UM.tolist(),
            'unit_field_proximal_mV': float(unit_field[0]),
            'unit_field_distal_mV': float(unit_field[-1]),
            'field_values_finite': bool(np.all(np.isfinite(unit_field))),
            'recorder': ('ARTIFACT-ONLY point sample: no membrane source contacts are supplied, so '
                         'neural_unfiltered_mV is exactly 0 and every recorded value is electrode '
                         'artifact + optional noise. NOT Vm, NOT a neural recording, no source '
                         'reconstruction.')},
        'honesty': {
            'mixes_real_and_invented': True,
            'interface_cells': 'IDEAL_EYE_IF / IDEAL_NECK_RELAY_IF / IDEAL_BODY_IF are NOT anatomy and '
                               'are absent from the BANC dataset',
            'banc_subnetwork': 'a bounded deterministic SELECTION, not the whole nervous system or brain',
            'parameters': 'all physiology parameters illustrative and uncalibrated',
            'claims_not_made': ['consciousness', 'viability', 'survival', 'rescue', 'recovery',
                                'electrodiffusion', 'validated visual pathway', 'whole-brain model'],
            'path_caveat': ('the composed eye->brain->neck->body path mixes a real BANC subgraph with '
                            'invented interface cells and an invented cable; only the '
                            'intra-subnetwork connectivity is data-derived')}}


def scenario_set(base=None, scenarios=SCENARIOS):
    """The intervention set: identical fields, only `scenario` differs."""
    b = base or MultirateConfig()
    return [replace(b, scenario=name) for name in scenarios]


def compare_scenarios(configs, subnetwork=None, subnetwork_config=None, root=None):
    """Run several scenarios on ONE shared subnetwork/initialization and collect."""
    s = subnetwork if subnetwork is not None else select_banc_subnetwork(subnetwork_config, root)
    return [run_multirate_scenario(cfg, s, subnetwork_config, root) for cfg in configs]
