"""DEMO: one scenario that joins the round-1 neck cut and the round-1 electrode damage.

Touch on the body -> ascending fibres -> brain, measured route by route (never as one
aggregate brain-drive number) across six paired scenarios:

    intact | sham | neck_cut | electrode_sheath_intact | electrode_sheath_removed
    | neck_cut_plus_electrode

Writes (all under outputs/brain_isolation/):

    touch_damage_report.json     every number, with its provenance and the honesty rules
    touch_damage_traces.npz      spike matrices, per-neuron counts, cable / chemistry arrays
    touch_damage_demo.png        the 12-panel figure
    touch_damage_selftest.json   the executable test counts (written by the selftest)

Run: venv/bin/python run_touch_damage_scenario.py
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import touch_damage_scenario as tds  # noqa: E402
from engine.electrode_damage import (  # noqa: E402
    HONESTY_STATEMENTS, NO_DROSOPHILA_LAMELLA_MEASUREMENT,
    NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION, NO_SECONDS_SCALE_SHEATH_RESEALING,
    SHEATH_LOAD_BEARING_EVIDENCE, build_registry, provenance_tier)
from run_touch_damage_selftest import run_tests  # noqa: E402

OUT = ROOT / 'outputs' / 'brain_isolation'
DISJOINT_ORDER = ('crossing_ascending', 'crossing_descending', 'body_route_sensor',
                  'head_route_sensor', 'vnc_intrinsic', 'other')
PRIMARY = tds.PRIMARY_SIGN_MODE
ANNOTATION = 'annotation'
LEAK_SENSITIVITY_DENSITIES = (1e-3, 5e-2, 0.5)
REACH_SHARE_NETWORK_CHECKS = (0.03116659378561303, 0.25, 0.5)
SURGERY_NETWORK_CHECKS = (7.5, 20.0)
ARMS6 = ('intact', 'sham', 'neck_cut', 'electrode_sheath_intact@owner_shunt',
         'electrode_sheath_removed@owner_shunt', 'neck_cut_plus_electrode@owner_shunt')
SHORT6 = ('intact', 'sham', 'neck_cut', 'electrode\nsheath INTACT',
          'electrode\nsheath REMOVED', 'neck_cut +\nelectrode')

#: ROUND-1 LEGACY BASELINES, as they were actually produced (wraparound semantics), taken
#: from the round-1 written reports before the mask-semantics correction.  They are PINNED
#: so the correction cannot silently disappear: the explicitly named legacy contrast arm
#: must still reproduce them, and the corrected default must differ from them.
ROUND1_LEGACY_INIT_DIGEST_ANNOTATION_BOTH = '55dae274e1f007615473f6b9cd128f2c'
ROUND1_LEGACY_HEAD_ROUTE_RETAINED_NECK_CUT = 1.0066628979499153
ROUND1_LEGACY_BODY_ROUTE_RETAINED_NECK_CUT = 0.0
ROUND1_LEGACY_BRAIN_SIDE_DRIVE_US_SETTLED = 112.58450000000005
ROUND1_LEGACY_BRAIN_SIDE_SPIKES_SETTLED = 4346
#: corrected values (in_tier_only) for the same intact arm, pinned by the selftest
CORRECTED_BRAIN_SIDE_DRIVE_US_SETTLED = 112.49150000000004
CORRECTED_BRAIN_SIDE_SPIKES_SETTLED = 4343
HONESTY_STRIP = (
    'AN AGGREGATE "TOTAL BRAIN DRIVE RETAINED" NUMBER HIDES THE BODY-ROUTE LOSS\n'
    'CROSSING SET = DATASET ANNOTATION (single-cell polarity NOT verified)   |   ALL EDGES '
    'UNSIGNED (nt_pair = -1 on every row) -> every route number is a SYNAPSE COUNT\n'
    'CUT = annotated-class EDGE REMOVAL, not a cut plane   |   NO insect electrode-damage '
    'measurement exists (two independent searches) -> every damage threshold is a CROSS-SPECIES '
    'PROXY\n'
    'the parenchymal damage radius and the sheath breach criterion are UNMEASURED and swept, '
    'never fitted   |   the electrode fibre PLACEMENT is DECLARED, not measured   |   tier '
    'carries only ~20% of the cut bandwidth\n'
    'THE SURGERY TERM IS UNQUANTIFIED AND NOT MODELLED: in real preparations the sheath is '
    'removed or enzymatically breached first, so insertion damage may be dominated by the '
    'SURGERY, and no insect study quantifies surgery damage. The surgery sweep is a WHAT-IF and '
    'never a result   |   NO consciousness / viability / survival claim')


def jsafe(obj):
    """JSON-safe conversion: numpy scalars, arrays, and non-finite floats -> null."""
    if isinstance(obj, dict):
        return {str(k): jsafe(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [jsafe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsafe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def paired_block(ctx, route_table, sign_mode):
    """Per-arm paired results: drive, retention vs intact, spikes split by route."""
    out = {}
    for route, arms in route_table[sign_mode].items():
        base = arms.get('intact')
        out[route] = {}
        for arm, entry in arms.items():
            block = dict(entry)
            if base is not None and base['brain_side_drive_uS_settled']:
                block['brain_side_drive_retained_vs_intact_settled'] = (
                    entry['brain_side_drive_uS_settled']
                    / base['brain_side_drive_uS_settled'])
                block['brain_side_spike_change_vs_intact'] = (
                    entry['brain_side_spikes_settled'] - base['brain_side_spikes_settled'])
                block['total_spike_change_vs_intact'] = (
                    entry['total_spikes_settled'] - base['total_spikes_settled'])
            else:
                block['brain_side_drive_retained_vs_intact_settled'] = None
                block['brain_side_spike_change_vs_intact'] = None
                block['total_spike_change_vs_intact'] = None
            res = ctx['grid']['results'][(sign_mode, route, arm)]
            block['drive_at_brain_side_by_presynaptic_group_uS_settled'] = {
                g: res['conductance_at_brain_side_uS']['settled'][g] for g in DISJOINT_ORDER}
            block['drive_at_brain_side_uS_post'] = float(
                sum(res['conductance_at_brain_side_uS']['post'].values()))
            block['drive_at_ascending_side_uS_settled'] = float(
                sum(res['conductance_at_ascending_side_uS']['settled'].values()))
            block['spike_counts_by_group_settled'] = {
                k: int(v) for k, v in res['spike_counts']['settled']['per_group'].items()}
            block['electrode'] = {
                'coupling_mode': res['coupling'].get('mode'),
                'reached_owners': res['coupling_reached_owners'],
                'g_shunt_uS_per_owner': res['coupling'].get('g_shunt_uS'),
                'g_shunt_per_compartment_uS': res['coupling'].get(
                    'g_shunt_per_compartment_uS'),
                'g_shunt_basis': res['coupling'].get('g_shunt_basis'),
                'g_shunt_over_tier_cell_leak': res['coupling'].get(
                    'g_shunt_over_tier_cell_leak'),
                'fibre_reached_compartments': res['coupling'].get(
                    'fibre_reached_compartments'),
                'output_transmission_factor_q': res['coupling'].get('q'),
                'no_op': res['coupling_no_op'],
                'declaration': res['coupling'].get('declaration'),
            }
            block['edge_coupling_accounting'] = res['edge_coupling_accounting']
            block['shunt_owners_at_phase'] = res['shunt_owners_at_phase']
            out[route][arm] = block
    return out


def electrode_network_sensitivity(ctx, *, densities=LEAK_SENSITIVITY_DENSITIES):
    """Where does the sheath-intact vs sheath-removed difference become visible?

    The ASSUMED damaged leak density decides whether a reached fibre is attenuated or
    functionally silenced, so the same routing is re-run at other densities.  Every
    row is a sensitivity of an assumed parameter, never a measurement.
    """
    a, tier, cut, groups = ctx['a'], ctx['tier'], ctx['cut'], ctx['groups']
    ref, fixture = ctx['reference'], ctx['fixture']
    owners = tds.deterministic_share_subset(tds.reached_owner_local(tier, groups),
                                           ctx['reach_share'])
    rows = []
    for density in densities:
        for variant in ('sheath_intact', 'sheath_removed'):
            est = tds.parenchymal_damage_estimate(
                ref['field'], ref['thresholds']['optimal'].value, ref['geometry'],
                ref['sheath'], ref['variants'][variant],
                threshold_record=ref['thresholds']['optimal'])
            radius = est['parenchymal_radius_mechanistic_um']
            rec = tds.measure_cable_damage(ref, fixture, radius,
                                          leak_density_S_cm2=density)['record']
            coupling = tds.build_coupling(
                'owner_shunt', owners, tier_n=tier['n'],
                per_compartment_g_uS=rec['extra_leak_conductance_uS_per_compartment'],
                fibre_total_g_uS=rec['extra_leak_conductance_uS_total'],
                fibre_reached_compartments=rec['reached_compartments'])
            spec = tds.with_arm(tds.SCENARIOS['electrode_sheath_removed'], route='body_only',
                                sign_mode=PRIMARY, coupling='owner_shunt')
            res = tds.run_damage_scenario(a, tier, cut, groups, spec, coupling, ctx['cfg'],
                                         ctx['tier_cfg'])
            base = ctx['grid']['results'][(PRIMARY, 'body_only', 'intact')]
            base_drive = float(sum(base['conductance_at_brain_side_uS']['settled'].values()))
            drive = float(sum(res['conductance_at_brain_side_uS']['settled'].values()))
            rows.append({
                'assumed_leak_density_S_cm2': float(density),
                'variant': variant,
                'damage_radius_um': radius,
                'reached_compartments': rec['reached_compartments'],
                'extra_leak_conductance_uS_total': rec['extra_leak_conductance_uS_total'],
                'extra_leak_over_tier_cell_leak': (
                    rec['extra_leak_conductance_uS_total'] / 0.001),
                'output_transmission_factor_q': rec['output_transmission_factor_q'],
                'network_body_route_drive_uS': drive,
                'network_body_route_drive_retained': (drive / base_drive) if base_drive else None,
                'network_brain_side_spikes': res['spike_counts']['settled']['per_group'][
                    'brain_side'],
                'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
            })
    return {
        'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        'why': ('the damaged leak density is ASSUMED (round 1). It decides whether a reached '
                'fibre is merely attenuated or functionally silenced, so the sheath-intact vs '
                'sheath-removed question is answered at several assumed values instead of one.'),
        'route': 'body_only', 'sign_mode': PRIMARY,
        'primary_value_S_cm2': float(ref['adopt']('assumed_damaged_leak_density_S_cm2').value),
        'rows': rows,
    }


def reach_share_what_if(ctx):
    """Body-route retention vs the DECLARED share of ascending fibres the track reaches."""
    a, tier, cut, groups = ctx['a'], ctx['tier'], ctx['cut'], ctx['groups']
    owners = tds.reached_owner_local(tier, groups)
    matrix = ctx['grid']['intact_spike_matrices'][(PRIMARY, 'body_only')]
    contrib = tds.brain_side_owner_contributions(a, tier, cut, groups, matrix, ctx['cfg'],
                                               ctx['tier_cfg'], sign_mode=PRIMARY,
                                               window='settled')
    out = {'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
           'why': ('the cache has no coordinates, so WHICH ascending fibres a real damage '
                   'cylinder intersects is unknown. The arithmetic is exact (the conductance '
                   'ledger is linear in the arrivals); the SHARE is a declaration.'),
           'brain_side_drive_uS_from_intact_spike_train': contrib['total_uS'],
           'reached_owners_available': int(owners.size),
           'per_disjoint_group_drive_uS': contrib['per_disjoint_group_uS'],
           'by_variant': {}, 'network_confirmation': {}}
    for variant in ('sheath_intact', 'sheath_removed'):
        rec = ctx['cable_table']['variants'][variant]['cable_mechanistic']
        block = tds.retention_vs_reached_owners(
            contrib, owners, rec['output_transmission_factor_q'], shares=tds.REACH_SHARE_SWEEP,
            share_note=('exact ledger arithmetic on the intact spike train; the share is a '
                        'declaration, not a measurement'),
            reference_total=contrib['total_uS'])
        block['q'] = rec['output_transmission_factor_q']
        block['extra_leak_conductance_uS_total'] = rec['extra_leak_conductance_uS_total']
        out['by_variant'][variant] = block
    variant = 'sheath_removed'
    rec = ctx['cable_table']['variants'][variant]['cable_mechanistic']
    for share in REACH_SHARE_NETWORK_CHECKS:
        coupling = tds.build_coupling(
            'owner_shunt', tds.deterministic_share_subset(owners, share), tier_n=tier['n'],
            per_compartment_g_uS=rec['extra_leak_conductance_uS_per_compartment'],
            fibre_total_g_uS=rec['extra_leak_conductance_uS_total'],
            fibre_reached_compartments=rec['reached_compartments'])
        spec = tds.with_arm(tds.SCENARIOS['electrode_sheath_removed'], route='body_only',
                            sign_mode=PRIMARY, coupling='owner_shunt')
        res = tds.run_damage_scenario(a, tier, cut, groups, spec, coupling, ctx['cfg'],
                                     ctx['tier_cfg'])
        base = ctx['grid']['results'][(PRIMARY, 'body_only', 'intact')]
        base_drive = float(sum(base['conductance_at_brain_side_uS']['settled'].values()))
        drive = float(sum(res['conductance_at_brain_side_uS']['settled'].values()))
        out['network_confirmation']['%.6f' % share] = {
            'declared_reached_share': float(share),
            'reached_owners': int(coupling['reached_local'].size),
            'body_route_drive_uS': drive,
            'body_route_drive_retained': (drive / base_drive) if base_drive else None,
            'brain_side_spikes': res['spike_counts']['settled']['per_group']['brain_side'],
            'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
        }
    return out


def surgery_network_confirmation(ctx, *, radii=SURGERY_NETWORK_CHECKS):
    """Full paired runs at two ASSUMED surgery radii (what-if, never a result)."""
    a, tier, cut, groups = ctx['a'], ctx['tier'], ctx['cut'], ctx['groups']
    ref, fixture = ctx['reference'], ctx['fixture']
    owners = tds.deterministic_share_subset(tds.reached_owner_local(tier, groups),
                                           ctx['reach_share'])
    rows = []
    for radius in radii:
        rec = tds.measure_cable_damage(
            ref, fixture, float(radius),
            leak_density_S_cm2=ref['adopt']('assumed_damaged_leak_density_S_cm2').value,
            radius_source='ASSUMED surgery damage radius (WHAT-IF, not a measurement)')['record']
        coupling = tds.build_coupling('drive_scaled', owners, tier_n=tier['n'],
                                      q=rec['output_transmission_factor_q'])
        spec = tds.with_arm(tds.SCENARIOS['electrode_sheath_removed'], route='body_only',
                            sign_mode=PRIMARY, coupling='drive_scaled')
        res = tds.run_damage_scenario(a, tier, cut, groups, spec, coupling, ctx['cfg'],
                                     ctx['tier_cfg'])
        base = ctx['grid']['results'][(PRIMARY, 'body_only', 'intact')]
        base_drive = float(sum(base['conductance_at_brain_side_uS']['settled'].values()))
        drive = float(sum(res['conductance_at_brain_side_uS']['settled'].values()))
        rows.append({
            'assumed_surgery_damage_radius_um': float(radius),
            'assumed_surgery_q': rec['output_transmission_factor_q'],
            'body_route_drive_uS_if_this_were_true': drive,
            'body_route_drive_retained_if_this_were_true': (drive / base_drive)
            if base_drive else None,
            'brain_side_spikes_if_this_were_true': res['spike_counts']['settled'][
                'per_group']['brain_side'],
            'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
            'is_a_fitted_illustrative_value': False,
            'reads_as': ('IF an assumed surgical insult of this radius acted on the declared '
                         'fibre population, the paired body-route drive would read this. It is '
                         'NOT a model of surgery and NOT a measurement.'),
        })
    return {'rows': rows, 'route': 'body_only', 'sign_mode': PRIMARY,
            'what_if': True, 'is_a_result': False, 'is_a_measurement': False,
            'is_a_fitted_illustrative_value': False}


def _no_op_coupling():
    """The 'no electrode damage' coupling used by the six damage-free arms."""
    return {'mode': 'none', 'reached_local': np.zeros(0, int), 'q': 1.0, 'g_shunt_uS': 0.0,
            'no_op': True}


def round1_reproduction(ctx):
    """Round-1 comparability, now split into a CORRECT arm and a LEGACY contrast arm.

    Round 1 shipped the mask wraparound defect, so its group/spike strata are NOT
    reproduced by the corrected default.  The legacy semantics survive ONLY as the
    explicitly named arm ``round1_wraparound_legacy``; running it here reproduces the
    round-1 numbers EXACTLY, which is what makes the correction auditable instead of
    a silent re-baselining.
    """
    path = OUT / 'neck_cut_report.json'
    out = {'neck_cut_report_present': bool(path.is_file())}
    correct_annotation = ctx['grid']['results'][(ANNOTATION, 'both', 'intact')]
    mine = correct_annotation['init_digest_sha256_32']
    legacy_run = tds.run_damage_scenario(
        ctx['a'], ctx['tier'], ctx['cut'], ctx['groups'],
        tds.with_arm(tds.SCENARIOS['intact'], route='both', sign_mode=ANNOTATION),
        _no_op_coupling(), ctx['cfg'], ctx['tier_cfg'],
        mask_semantics=tds.LEGACY_MASK_SEMANTICS)
    out.update({
        'corrected_default_semantics': tds.DEFAULT_MASK_SEMANTICS,
        'legacy_contrast_arm_semantics': tds.LEGACY_MASK_SEMANTICS,
        'legacy_arm_reproduces_the_pinned_round1_init_digest': bool(
            legacy_run['init_digest_sha256_32'] == ROUND1_LEGACY_INIT_DIGEST_ANNOTATION_BOTH),
        'legacy_arm_init_digest': legacy_run['init_digest_sha256_32'],
        'pinned_round1_init_digest': ROUND1_LEGACY_INIT_DIGEST_ANNOTATION_BOTH,
        'my_init_digest': mine,
        'what_it_proves': (
            'the legacy arm reproduces the round-1 initialisation digest to the character, so the '
            'ONLY thing that separates this round from round 1 is the tier-local mask semantics; '
            'the init digest is a DIGEST OF THE STIMULUS MASKS AND EDGE SET, and body_route / '
            'head_route contain no out-of-tier member, which is why the digest is unchanged even '
            'though the group masks are not. An unchanged init digest therefore does NOT mean the '
            'round-1 group, spike-stratum or drive numbers were reproduced -- the spike strata and '
            'the counterfactual drive definitively are NOT, and the change is quantified in step11 '
            'and in the block below.'),
        'round1_head_route_retained_neck_cut_legacy': (
            ROUND1_LEGACY_HEAD_ROUTE_RETAINED_NECK_CUT),
        'round1_body_route_retained_neck_cut_legacy': (
            ROUND1_LEGACY_BODY_ROUTE_RETAINED_NECK_CUT),
        'correct_arm_head_route_retained_neck_cut': None,
        'correct_arm_body_route_retained_neck_cut': None,
        #: SAME arm on both sides (annotation / intact / both): the legacy run and the corrected
        #: run of the DEFAULT grid.  In this tier the spurious neuron receives no modelled
        #: conductance, so the DRIVE is unchanged while its 4 spikes were wrongly attributed to
        #: the brain side -- which is why both columns are recorded.
        'legacy_vs_correct_intact_annotation_both': {
            'legacy_brain_side_drive_uS_settled':
                float(sum(legacy_run['conductance_at_brain_side_uS']['settled'].values())),
            'correct_brain_side_drive_uS_settled':
                float(sum(correct_annotation['conductance_at_brain_side_uS']['settled'].values())),
            'legacy_brain_side_spikes_settled':
                legacy_run['spike_counts']['settled']['per_group']['brain_side'],
            'correct_brain_side_spikes_settled':
                correct_annotation['spike_counts']['settled']['per_group']['brain_side'],
            'total_spikes_settled':
                correct_annotation['spike_counts']['settled']['total'],
            'note': ('the total spike count is identical in both columns: the defect moved the '
                     'ATTRIBUTION of spikes between groups, not the simulated activity'),
        },
        'round1_corrected_claims_kept_here': [
            'a matching crossing-set TOTAL is NOT independent confirmation of the partition: the '
            'per-population numbers do not reproduce and the paper\'s own three components sum '
            'to 3,682 rather than 3,686',
            'the cut is ROUTE-SELECTIVE, not "brain isolation": 1,638 of 83,623 head-route '
            'output synapses (1.96%) ARE severed, by 2 sensory_descending (DNx01, antennal) cells',
            'an aggregate brain-drive retention figure must NOT be quoted, because the head '
            'route dominates it and hides the body-route loss',
            'the round-1 group masks wrapped every out-of-tier neuron onto local row n-1; that '
            'defect is CORRECTED here (default in_tier_only) and its size is reported in step11',
        ],
    })
    if path.is_file():
        on_disk = json.loads(path.read_text())
        try:
            on_disk_intact = on_disk['step4_paired_results'][ANNOTATION]['both'][
                'cut_modes']['intact']
            out['neck_cut_report_on_disk_init_digest'] = \
                on_disk_intact['init_digest_sha256_32']
            out['neck_cut_report_on_disk_is_the_CORRECTED_one'] = bool(
                on_disk.get('run', {}).get('mask_semantics') == tds.DEFAULT_MASK_SEMANTICS
                and on_disk['step6_mask_semantics']['default_semantics']
                == tds.DEFAULT_MASK_SEMANTICS)
            out['correct_arm_head_route_retained_neck_cut'] = (
                on_disk['step5_head_route_control']['paired_measurement'][
                    'counterfactual_ach']['head_only']['drive_fraction_retained_neck_cut'])
            out['correct_arm_body_route_retained_neck_cut'] = (
                on_disk['step5_head_route_control']['paired_measurement'][
                    'counterfactual_ach']['body_only']['drive_fraction_retained_neck_cut'])
        except (KeyError, TypeError):
            out['neck_cut_report_on_disk_is_the_CORRECTED_one'] = None
    return out


def mask_defect(ctx):
    """The round-1 mask wraparound defect: CORRECTED by default, still MEASURED.

    Every reported number now uses ``in_tier_only`` (out-of-tier global ids are
    EXCLUDED, never wrapped).  The legacy behaviour is retained ONLY as the
    explicitly named contrast arm, and both columns of every affected quantity are
    recorded here so the correction is auditable.
    """
    audit = ctx['mask_semantics_audit']
    membership = ctx['mask_membership_table']
    noop = {'mode': 'none', 'reached_local': np.zeros(0, int), 'q': 1.0, 'g_shunt_uS': 0.0,
            'no_op': True}
    spec = tds.with_arm(tds.SCENARIOS['intact'], route='both', sign_mode=PRIMARY)
    legacy = tds.run_damage_scenario(ctx['a'], ctx['tier'], ctx['cut'], ctx['groups'], spec,
                                     noop, ctx['cfg'], ctx['tier_cfg'],
                                     mask_semantics=tds.LEGACY_MASK_SEMANTICS)
    correct = ctx['grid']['results'][(PRIMARY, 'both', 'intact')]

    def _drive(r):
        return float(sum(r['conductance_at_brain_side_uS']['settled'].values()))

    def _drive_post(r):
        return float(sum(r['conductance_at_brain_side_uS']['post'].values()))

    def _grp_drive(r):
        return r['conductance_at_brain_side_uS']['settled']

    def _grp_spikes(r):
        return r['spike_counts']['settled']['per_group']

    def _bs(r):
        return r['spike_counts']['settled']['per_group']['brain_side']

    def _tot(r):
        return r['spike_counts']['settled']['total']

    return {
        'finding': ('round-1 group masks fold EVERY out-of-tier neuron onto local row n-1: '
                    'local_of_global is -1 outside the tier and the round-1 idiom indexed with '
                    'it, which NumPy wraps to the last row. CORRECTED here: the default is now '
                    '%r, so an out-of-tier neuron can never mark a local row. The legacy '
                    'behaviour is kept ONLY as the explicitly named contrast arm %r and its '
                    'numbers are reported side by side below.'
                    % (tds.DEFAULT_MASK_SEMANTICS, tds.LEGACY_MASK_SEMANTICS)),
        'corrected_by_default': True,
        'default_semantics_in_this_report': tds.DEFAULT_MASK_SEMANTICS,
        'legacy_contrast_arm_semantics': tds.LEGACY_MASK_SEMANTICS,
        'audit': audit,
        'per_group_membership_counts': {
            name: {'global_members': v['global_members'],
                   'global_members_outside_tier': v['global_members_outside_tier'],
                   'masked_rows_correct_semantics': v['masked_rows_correct_semantics'],
                   'masked_rows_legacy_semantics': v['masked_rows_legacy_semantics'],
                   'spurious_rows_under_legacy_semantics':
                       v['spurious_rows_under_legacy_semantics'],
                   'last_local_row_was_masked_legacy': v['membership_changed'],
                   'last_local_row_is_a_true_member': v['last_local_row_is_a_true_member']}
            for name, v in membership.items()},
        'old_vs_new_on_the_intact_arm': {
            'arm': spec.arm,
            'why_this_arm': ('the intact arm is the reference every retention number in this '
                             'report is divided by, so the defect moved every retention figure '
                             'through it'),
            'legacy_semantics': {
                'brain_side_drive_uS_settled': _drive(legacy),
                'brain_side_drive_uS_post': _drive_post(legacy),
                'brain_side_spikes_settled': _bs(legacy),
                'total_spikes_settled': _tot(legacy),
                'drive_at_brain_side_by_group_uS_settled': _grp_drive(legacy),
                'spikes_by_group_settled': _grp_spikes(legacy),
                'init_digest_sha256_32': legacy['init_digest_sha256_32'],
                'full_spike_digest_sha256_32': legacy['full_spike_digest_sha256_32'],
            },
            'correct_semantics': {
                'brain_side_drive_uS_settled': _drive(correct),
                'brain_side_drive_uS_post': _drive_post(correct),
                'brain_side_spikes_settled': _bs(correct),
                'total_spikes_settled': _tot(correct),
                'drive_at_brain_side_by_group_uS_settled': _grp_drive(correct),
                'spikes_by_group_settled': _grp_spikes(correct),
                'init_digest_sha256_32': correct['init_digest_sha256_32'],
                'full_spike_digest_sha256_32': correct['full_spike_digest_sha256_32'],
            },
            'difference_legacy_minus_correct': {
                'brain_side_drive_uS_settled': _drive(legacy) - _drive(correct),
                'brain_side_drive_uS_post': _drive_post(legacy) - _drive_post(correct),
                'brain_side_spikes_settled': _bs(legacy) - _bs(correct),
                'total_spikes_settled': _tot(legacy) - _tot(correct),
                'drive_at_brain_side_by_group_uS_settled': {
                    g: _grp_drive(legacy)[g] - _grp_drive(correct)[g]
                    for g in _grp_drive(correct)},
                'spikes_by_group_settled': {
                    g: _grp_spikes(legacy)[g] - _grp_spikes(correct)[g]
                    for g in _grp_spikes(correct)},
                'init_digest_changed': bool(legacy['init_digest_sha256_32']
                                            != correct['init_digest_sha256_32']),
                'full_spike_digest_changed': bool(legacy['full_spike_digest_sha256_32']
                                                  != correct['full_spike_digest_sha256_32']),
            },
        },
        'handed_to_parent': True,
    }


def build_report(ctx, tests, t0):
    a, tier, cut, crossing, budget = (ctx['a'], ctx['tier'], ctx['cut'], ctx['crossing'],
                                     ctx['budget'])
    ref, fixture, table = ctx['reference'], ctx['fixture'], ctx['cable_table']
    owners_all = tds.reached_owner_local(tier, ctx['groups'])
    reach = ctx['couplings']
    route_table = tds.route_drive_table(ctx['grid'])
    paired = {sm: paired_block(ctx, route_table, sm) for sm in tds.SIGN_MODES}
    aggregate = {}
    for sm in tds.SIGN_MODES:
        aggregate[sm] = {}
        for arm in sorted({k for route in route_table[sm].values() for k in route}):
            if arm == 'intact':
                continue
            aggregate[sm][arm] = tds.aggregate_hides_route_loss(sm, route_table[sm], arm)
    attribution = {}
    for sm in tds.SIGN_MODES:
        attribution[sm] = {}
        for route in tds.ACTIVATED_ROUTES:
            matrix = ctx['grid']['intact_spike_matrices'][(sm, route)]
            coupling = reach['by_scenario']['electrode_sheath_removed']['drive_scaled']
            attribution[sm][route] = tds.exact_attribution(
                a, tier, cut, ctx['groups'], matrix, coupling, ctx['cfg'], ctx['tier_cfg'],
                sign_mode=sm)
            attribution[sm][route]['electrode_scenario'] = 'electrode_sheath_removed@drive_scaled'
    chem = tds.measure_chemistry(
        ref, damage_fraction_by_variant={
            name: tds.chemical_damage_fraction(
                table['variants'][name]['cable_mechanistic']['damage_radius_um'])['fraction_used']
            for name in ref['variants']})
    sweeps = tds.declared_parameter_sweeps(ref, fixture)
    surgery = tds.surgery_what_if(ref, fixture)
    surgery_net = surgery_network_confirmation(ctx)
    sensitivity = electrode_network_sensitivity(ctx)
    share_what_if = reach_share_what_if(ctx)
    defect = mask_defect(ctx)

    tier_counts = tier['counts']
    head_anchor = {
        'body_route_output_synapses': budget['body_route']['outgoing_synapses'],
        'body_route_output_synapses_severed': budget['body_route']['severed_synapses'],
        'body_route_output_fraction_severed': budget['body_route']['severed_fraction_of_route'],
        'head_route_output_synapses': budget['head_route']['outgoing_synapses'],
        'head_route_output_synapses_severed': budget['head_route']['severed_synapses'],
        'head_route_output_fraction_severed': budget['head_route']['severed_fraction_of_route'],
        'head_route_neurons_also_annotated_crossing':
            budget['route_neurons_also_annotated_as_crossing']['head_route'],
        'body_route_neurons_also_annotated_crossing':
            budget['route_neurons_also_annotated_as_crossing']['body_route'],
        'head_route_populations_also_crossing':
            budget['route_neurons_also_annotated_as_crossing']['head_route_populations'],
        'statement': (
            'The naive claim "a neck cut leaves the head route untouched" is FALSE: %d of %d '
            'head-route output synapses (%.4f%%) are severed, from %d cells the dataset itself '
            'annotates as neck-crossing (sensory_descending, type DNx01, antennal nerve). The '
            'defensible statement is that the cut is ROUTE-SELECTIVE: %.4f%% of head-route output '
            'is untouched, while 100%% of the body route\'s DIRECT connection into the brain is '
            'severed (%d of %d synapses).'
            % (int(budget['head_route']['severed_synapses']),
               int(budget['head_route']['outgoing_synapses']),
               100 * budget['head_route']['severed_fraction_of_route'],
               budget['route_neurons_also_annotated_as_crossing']['head_route'],
               100 * (1 - budget['head_route']['severed_fraction_of_route']),
               int(budget['body_route']['severed_synapses_into_brain_side']),
               int(budget['body_route']['to_brain_side_direct']))),
    }
    report = {
        'title': ('Touch on the body -> ascending fibres -> brain under a neck cut and under an '
                  'inserted micrometre electrode: six paired scenarios, measured route by route'),
        'question': ('does the body-touch drive that reaches the brain survive (i) an annotated '
                     'neck cut and (ii) an inserted electrode, and can ONE aggregate "total brain '
                     'drive retained" number answer that? Answer: no -- see step6.'),
        'generated_by': 'run_touch_damage_scenario.py',
        'status': ('integration prototype; the numerical tests pass; NOT calibrated against any '
                   'insect experiment'),
        'scope': __doc__,
        'honesty': {
            'crossing_set_is_dataset_annotation': True,
            'single_cell_polarity_verified': False,
            'edges_are_unsigned': True,
            'nt_pair_unique_values': [int(v) for v in np.unique(a['nt_pair'])],
            'nt_pair_used_for_sign': False,
            'cut_is_annotated_class_edge_removal_not_a_cut_plane': True,
            'fafb_truncated_at_neck': True,
            'counts_conditional_on_cached_snapshot': True,
            'no_insect_electrode_damage_measurement_exists': True,
            'every_damage_threshold_is_a_cross_species_proxy': True,
            'parenchymal_damage_radius_is_unmeasured': True,
            'sheath_breach_criterion_is_unmeasured': True,
            'sheath_thickness_for_the_fly_is_assumed': True,
            'electrode_fibre_placement_is_declared_not_measured': True,
            'surgery_term_is_unquantified_and_not_modelled': True,
            'species_guard_still_fires_on_foreign_measured_parameters': True,
            'tier_share_of_dataset_severed_synapses':
                tier_counts['share_of_dataset_severed_synapses_inside_tier'],
            'tier_carries_only_about_a_fifth_of_the_cut_bandwidth': True,
            'no_consciousness_or_viability_claim': True,
            'reachability_used_as_evidence': False,
            'mask_semantics_is_in_tier_only': True,
            'round1_mask_wraparound_defect_corrected_by_default': True,
            'non_finite_values_are_reported_as_null_not_as_numbers': True,
            'statements': list(tds.HONESTY_LINES),
        },
        'negative_results': {
            'no_insect_insertion_damage_quantification':
                NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
            'no_drosophila_lamella_measurement': NO_DROSOPHILA_LAMELLA_MEASUREMENT,
            'no_seconds_scale_sheath_resealing': NO_SECONDS_SCALE_SHEATH_RESEALING,
            'sheath_is_load_bearing': SHEATH_LOAD_BEARING_EVIDENCE,
            'electrode_damage_honesty_statements': list(HONESTY_STATEMENTS),
        },
        'reachability_is_NOT_used_as_evidence': {
            'used_as_evidence': False,
            'assertion': ('no reachability, hop-count, shortest-path, graph-distance, '
                          'connected-component or network-diameter quantity is used anywhere in '
                          'this report as evidence; every number is (a) a synapse count over an '
                          'explicit edge set, (b) a conductance in uS delivered inside a '
                          'simulation, or (c) a spike count'),
            'assertion_holds': True,
            'evidence_classes_used': ['synapse-weighted route budget (UNSIGNED counts)',
                                      'conductance-weighted delivered drive (uS), with exact '
                                      'per-presynaptic-owner attribution',
                                      'spike counts in a bounded paired simulation'],
            'component_statistics_role': ('none are reported; the tier is deliberately NOT '
                                          'reduced to one connected component'),
        },
        'dataset': {'path': a['path'], 'neurons': int(a['root_ids'].size),
                    'annotated_rows': int(a['pre'].size),
                    'cached_snapshot_conditional': True},
        'step1_round1_inputs': {
            'crossing_set': crossing,
            'cut_set': {k: v for k, v in cut.items() if k != 'row_masks'},
            'tier_counts': tier_counts,
            'route_budget': budget,
            'head_route_control': head_anchor,
            'round1_reproduction': round1_reproduction(ctx),
        },
        'step2_electrode_model': {
            'species_guard': {
                'default_construction_refused_with': ref['guard_default_refused_with'],
                'allow_cross_transfer': True,
                'which_test_proves_it_fires': 'run_touch_damage_selftest.py::'
                                              't_guard_refuses_cross_species_measured_import',
            },
            'fly_brain_geometry': {'length_um': ref['geometry'].length_um,
                                   'width_um': ref['geometry'].width_um,
                                   'thickness_um': ref['geometry'].thickness_um,
                                   'volume_um3': ref['geometry'].volume_um3,
                                   'footprint_um2': ref['geometry'].footprint_um2},
            'sheath': ref['sheath'].to_dict(),
            'preparation_variants': {k: v.to_dict() for k, v in ref['variants'].items()},
            'strain_thresholds_cross_species': {
                k: v.to_dict() for k, v in ref['thresholds'].items() if not k.startswith('_')},
            'shaft': {'diameter_um': ref['shaft_diameter_um'],
                      'radius_um': ref['shaft_radius_um'],
                      'wall_displacement_um': ref['field'].wall_displacement_um,
                      'wall_displacement_fraction': ref['field'].wall_displacement_fraction,
                      'shear_modulus_Pa': ref['field'].shear_modulus_Pa},
            'declared_fixture': fixture.as_dict(),
            'primary_threshold_key': table['threshold_key'],
            'assumed_leak_density_S_cm2': table['leak_density_S_cm2'],
            'variant_damage_table': {k: {kk: vv for kk, vv in v.items() if kk != '_arrays'}
                                     for k, v in table['variants'].items()},
            'provenance_tiers_of_the_registry': _tier_counts_of_registry(),
        },
        'step3_surgery_term': {
            'term': tds.SURGERY_TERM,
            'modelled_in_any_scenario': False,
            'value_used_anywhere_in_this_report': None,
            'what_if_sweep': surgery,
            'what_if_network_confirmation': surgery_net,
            'how_to_read_it': ('EVERY number in this block is hypothetical. The surgery term is '
                               'registered as UNQUANTIFIED, no value is invented for it, and no '
                               'scenario here contains surgical damage. The sweep only shows how '
                               'a conclusion WOULD move with an assumed surgery radius.'),
            'audit': 'run_touch_damage_selftest.py::t_surgery_claims_audit_is_clean and ::'
                     't_surgery_audit_fires_on_a_poisoned_measured_value',
        },
        'step4_scenarios': {
            'required_scenarios': list(tds.SCENARIO_NAMES),
            'definitions': [tds.SCENARIOS[n].as_dict() for n in tds.SCENARIO_NAMES],
            'coupling_modes': list(tds.COUPLING_MODES),
            'paired_design': {
                'rule': ('every arm shares ONE initialisation and ONE seed; the pre-phase matrices '
                         'are the intact ones in every arm; the intervention is applied at the '
                         'phase step; only the intervention differs'),
                'seed': ctx['cfg'].seed, 'dt_ms': ctx['cfg'].dt_ms,
                'duration_ms': ctx['cfg'].duration_ms, 'phase_ms': ctx['cfg'].phase_ms,
                'settled_window_ms': [ctx['cfg'].phase_ms + ctx['cfg'].settle_ms,
                                      ctx['cfg'].duration_ms],
                'n_runs': len(ctx['grid']['results']),
                'runs_per_sign_mode_and_route': len(ctx['grid']['plan']),
                'sign_modes': list(tds.SIGN_MODES), 'routes': list(tds.ACTIVATED_ROUTES),
            },
            'electrode_coupling': {
                'declared_reached_owner_group': ('crossing_ascending (dataset annotation), '
                                                 'tier-local, in-tier neurons only'),
                'reached_owners_available': int(owners_all.size),
                'reached_owners_used': reach['reached_owners_used'],
                'reach_share_used': reach['reach_share'],
                'reach_share_is_declared': True,
                'tier_crossing_neurons_ascending_side_for_cross_check':
                    tier_counts['crossing_neurons_in_tier_ascending_side'],
                'owner_shunt': reach['by_scenario']['electrode_sheath_removed']['owner_shunt'],
                'drive_scaled': reach['by_scenario']['electrode_sheath_removed']['drive_scaled'],
                'why_two_couplings': ('the measured damage is carried into the point tier twice, '
                                      'once as the measured extra leak conductance on the fibre\'s '
                                      'owner and once as the measured output transmission factor '
                                      'on its outgoing rows, so no conclusion can be an artefact '
                                      'of one coupling choice'),
            },
        },
        'step5_paired_results': paired,
        'step6_aggregate_vs_route': aggregate,
        'step7_exact_attribution_with_activity_fixed': attribution,
        'step8_declared_placement_what_if': share_what_if,
        'step9_assumed_parameter_sensitivity': {
            'declared_and_assumed_sweeps': sweeps,
            'leak_density_network_sensitivity': sensitivity,
        },
        'step10_chemistry_through_injury_tissue_api': chem,
        'step11_mask_semantics_correction_and_round1_defect': defect,
        'selftests': {
            'n_checks': tests['n_checks'], 'n_passed': tests['n_passed'],
            'n_failed': tests['n_failed'], 'n_skipped': tests.get('n_skipped', 0),
            'by_group': tests['by_group'], 'wall_seconds': tests['wall_seconds'],
            'which_path': 'demo path: this run skipped the checks that AUDIT the written report, '
                          'because it was writing it; those checks are run by '
                          'run_touch_damage_selftest.py and recorded in touch_damage_selftest.json',
            'json': str(OUT / 'touch_damage_selftest_demo_path.json'),
            'canonical_standalone_json': str(OUT / 'touch_damage_selftest.json'),
        },
        'limitations_not_removed': _limitations(tier, table, sensitivity, aggregate, a),
        'outputs': {}, 'run': {},
    }
    report['run'] = {
        'wall_clock_s': time.perf_counter() - t0,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        'python': sys.version.split()[0],
        'network_config': dict(ctx['cfg'].__dict__),
        'tier_config': dict(ctx['tier_cfg'].__dict__),
        'mask_semantics': ctx['mask_semantics'],
        'legacy_mask_semantics_arm': ctx['legacy_mask_semantics_arm'],
        'mask_semantics_note': ('every number in this report uses the CORRECT tier-local '
                               'projection (out-of-tier neurons EXCLUDED); the round-1 '
                               'wraparound arm is named explicitly and its old-vs-new numbers '
                               'are in step11'),
        'n_network_runs': len(ctx['grid']['results']),
    }
    return report, paired, aggregate


def _tier_counts_of_registry():
    counts = {}
    for record in build_registry().snapshot().values():
        tier = provenance_tier(record['source'])
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _limitations(tier, table, sensitivity, aggregate, a):
    counts = tier['counts']
    rm = table['variants']['sheath_removed']['cable_mechanistic']
    ri = table['variants']['sheath_intact']['cable_mechanistic']
    return [
        NO_INSECT_INSERTION_DAMAGE_QUANTIFICATION,
        'The crossing set is an ANNOTATION, not a verified single-cell polarity: this cache has '
        'no soma or arbor coordinates, so no ascending/descending label is independently '
        'confirmed, and the round-1 finding that the annotator\'s own two columns disagree '
        'stands.',
        'Every route number is an UNSIGNED synapse count: nt_pair is -1 on all %d cached rows, so '
        'the budget cannot tell excitatory from inhibitory bandwidth.' % int(a['pre'].size),
        'The cut is an annotated-class edge removal over annotated rows: it does not locate the '
        'cervix and does not correspond to any measured cut plane.',
        'The bounded tier holds %d of %d neurons (%.2f%% of annotated rows) and carries %.2f%% of '
        'the dataset severed edges and %.2f%% of the severed synapses, so the paired simulation '
        'measures about a fifth of the cut bandwidth.'
        % (tier['n'], counts['neurons_total_in_dataset'],
           100 * counts['share_of_dataset_rows_inside_tier'],
           100 * counts['share_of_dataset_severed_edges_inside_tier'],
           100 * counts['share_of_dataset_severed_synapses_inside_tier']),
        'The electrode fibre placement is DECLARED, not measured: the cache has no coordinates for '
        'the ascending fibres or for the track. The declaration is the MAXIMUM-DAMAGE case (the '
        'fibre lies against the shaft wall, applied to the whole declared fibre population), so '
        'the electrode numbers are an UPPER BOUND; step8 is what a different placement would do.',
        'THE SURGERY TERM IS NOT MODELLED AND IS NOT QUANTIFIED. In real fly preparations the '
        'sheath is removed or enzymatically breached first, so insertion damage may be SMALLER '
        'than the surgical damage, and no insect study quantifies surgical damage. Because '
        'surgery is absent, the sham is bit-identical to intact; a real preparation would not be, '
        'and nothing here estimates by how much.',
        'The damaged leak density is ASSUMED (%.4g S/cm2, round 1). At that value the reached '
        'fibre is functionally silenced whatever the sheath variant, which is why the '
        'sheath-intact vs sheath-removed difference is visible in the added leak conductance '
        '(%.4f vs %.4f uS), the transmitted drive (%.3g vs %.3g) and the damaged brain volume, '
        'but NOT in the spike counts. The leak-density sweep in step9 locates where a spiking '
        'difference would appear.'
        % (table['leak_density_S_cm2'], ri['extra_leak_conductance_uS_total'],
           rm['extra_leak_conductance_uS_total'], ri['output_transmission_factor_q'],
           rm['output_transmission_factor_q']),
        'The fly sheath thickness is assumed (no Drosophila lamella measurement), the sheath '
        'breach criterion is a free parameter, and the herniation multiplier is assumed and is '
        'the ONLY reason the two sheath variants differ, so the intact-sheath damage radius is a '
        'modelling choice and not a measurement.',
        'The chemistry model is a reduced two-pool salt model with no pumps, buffer or reservoir '
        'in its closed variant, so its long-time K redistribution is a documented model artefact; '
        'the buffered+reservoir variant is reported next to it. For Drosophila this is a '
        'glial/tracheal sheath restriction, not a vertebrate blood-brain barrier.',
        'The exact attribution in step7 holds ACTIVITY FIXED, so it isolates edge removal and '
        'drive scaling from network feedback; the paired runs in step5 include feedback, and the '
        'two levels disagree by design (the head route even rises slightly under the electrode, '
        'a feedback effect with no edge added).',
        'In the annotation-faithful tier the head route has ZERO sign coverage and is not modelled '
        'at all, so the aggregate equals the body-route number there; the hiding demonstrated in '
        'step6 needs the explicitly labelled counterfactual tier.',
        'The round-1 mask wraparound defect (step11) is CORRECTED here, not reproduced: the default '
        'tier-local projection EXCLUDES out-of-tier neurons, so no group or spike stratum carries '
        'the spurious local row n-1 any more. The round-1 behaviour survives ONLY as the '
        'explicitly named contrast arm round1_wraparound_legacy, and step11 records the old-vs-new '
        'difference for the drive, the spike counts and every per-group membership, so the round-1 '
        'numbers can still be quoted as round-1 numbers.',
        'The aggregate retention %s (counterfactual tier, both routes, neck cut) is NOT a result '
        'about the brain: it is a ratio of two conductance sums over a bounded tier, and the '
        'body-route loss it hides is the point.'
        % tds._fmt(aggregate[PRIMARY]['neck_cut']['both_drive_uS']['retained_fraction']),
        'Nothing here is a claim about a real preparation. No consciousness, viability, survival, '
        'recovery or rescue claim is made. "Cut" means "these rows are absent from the edge '
        'list"; "damaged" means "these compartments carry the modelled extra leak".',
    ]


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def _note(ax, text, x, y, fontsize=9.6, ha='left', va='top', box=None, family=None):
    """A note box placed at a HAND-CHOSEN free spot (never over the data)."""
    kw = dict(boxstyle='round', facecolor='#f4f4f4', edgecolor='black')
    if box:
        kw.update(box)
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
            bbox=kw, family=(family or 'sans-serif'), linespacing=1.32)


def make_figure(report, path, ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paired = report['step5_paired_results']
    agg = report['step6_aggregate_vs_route']
    table = report['step2_electrode_model']
    surgery = report['step3_surgery_term']
    share = report['step8_declared_placement_what_if']
    sens = report['step9_assumed_parameter_sensitivity']
    chem = report['step10_chemistry_through_injury_tissue_api']
    defect = report['step11_mask_semantics_correction_and_round1_defect']
    geo = table['fly_brain_geometry']
    ri = table['variant_damage_table']['sheath_intact']['cable_mechanistic']
    rm = table['variant_damage_table']['sheath_removed']['cable_mechanistic']
    honesty = report['honesty']['statements']
    hc = report['step1_round1_inputs']['head_route_control']
    sweeps = sens['declared_and_assumed_sweeps']
    lk = sens['leak_density_network_sensitivity']
    box_red = dict(boxstyle='round', facecolor='#ffe8e8', edgecolor='black')
    box_blue = dict(boxstyle='round', facecolor='#eaf4ff', edgecolor='black')
    box_grey = dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='black')
    box_green = dict(boxstyle='round', facecolor='#eaffea', edgecolor='black')
    box_yell = dict(boxstyle='round', facecolor='#fff3c4', edgecolor='#a00000', linewidth=2.8)
    mono = 'DejaVu Sans Mono'

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
                         'legend.fontsize': 10})
    fig = plt.figure(figsize=(36.0, 34.0), dpi=100)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 0.95], hspace=0.34, wspace=0.24,
                          left=0.030, right=0.991, top=0.912, bottom=0.020)
    fig.suptitle('Touch on the body -> ascending fibres -> brain under a neck cut and under an '
                 'inserted 10 um electrode: SIX paired scenarios, measured ROUTE BY ROUTE',
                 fontsize=25, fontweight='bold', y=0.985)
    fig.text(0.5, 0.968, HONESTY_STRIP, ha='center', va='top', fontsize=13.6, color='#8b0000',
             linespacing=1.45)

    # ================= 1 route-resolved retention, all six scenarios =================
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ARMS6))
    w = 0.26
    for j, (route, colr, lab) in enumerate((('body_only', '#1f77b4', 'BODY route driven alone'),
                                            ('head_only', '#ff7f0e', 'HEAD route driven alone'),
                                            ('both', '#8c564b',
                                             'BOTH (the aggregate number)'))):
        vals = [paired[PRIMARY][route][a].get('brain_side_drive_retained_vs_intact_settled')
                for a in ARMS6]
        bars = ax.bar(x + (j - 1) * w, [0.0 if v is None else v for v in vals], w, color=colr,
                      edgecolor='black', label=lab)
        for b, v in zip(bars, vals):
            if v is None:
                ax.text(b.get_x() + b.get_width() / 2, 0.03, 'NOT\nMODELLED', ha='center',
                        va='bottom', fontsize=8.6, rotation=90, color='#444444')
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.3f' % v, ha='center',
                        fontsize=9.4, rotation=90)
    ax.axhline(1.0, color='black', ls=':', lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT6, fontsize=10)
    ax.set_ylim(0, 1.95)
    ax.set_ylabel('brain-side delivered drive retained\nvs intact (settled window)')
    ax.legend(loc='upper center', fontsize=9.6, framealpha=0.95, ncol=1)
    ax.set_title('1. ROUTE-RESOLVED paired result, ALL SIX scenarios\n'
                 'counterfactual sign tier (%s); the aggregate bars are the trap in panel 2'
                 % PRIMARY, fontsize=13.5)
    _note(ax, 'body route 1.000 -> %.3f (neck cut)\n'
              '-> %.3f (electrode, sheath INTACT)\n'
              '-> %.3f (electrode, sheath REMOVED)\n'
              'head route %.3f (neck cut), %.3f (electrode)'
          % (paired[PRIMARY]['body_only']['neck_cut'][
                 'brain_side_drive_retained_vs_intact_settled'],
             paired[PRIMARY]['body_only']['electrode_sheath_intact@owner_shunt'][
                 'brain_side_drive_retained_vs_intact_settled'],
             paired[PRIMARY]['body_only']['electrode_sheath_removed@owner_shunt'][
                 'brain_side_drive_retained_vs_intact_settled'],
             paired[PRIMARY]['head_only']['neck_cut'][
                 'brain_side_drive_retained_vs_intact_settled'],
             paired[PRIMARY]['head_only']['electrode_sheath_removed@owner_shunt'][
                 'brain_side_drive_retained_vs_intact_settled']),
          0.5, 0.66, fontsize=10, ha='center', box=box_grey)

    # ================= 2 the aggregate HIDES the failure ============================
    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[3.1, 2.0], hspace=0.06)
    ax = fig.add_subplot(sub[0])
    axb = fig.add_subplot(sub[1])
    show = ['neck_cut', 'electrode_sheath_intact@owner_shunt',
            'electrode_sheath_removed@owner_shunt', 'neck_cut_plus_electrode@owner_shunt']
    short2 = ['neck_cut', 'electrode\nsheath INTACT', 'electrode\nsheath REMOVED',
              'neck_cut +\nelectrode']
    x = np.arange(len(show))
    w = 0.26
    body = [agg[PRIMARY][k]['body_only_drive_uS']['retained_fraction'] for k in show]
    head = [agg[PRIMARY][k]['head_only_drive_uS']['retained_fraction'] for k in show]
    both = [agg[PRIMARY][k]['both_drive_uS']['retained_fraction'] for k in show]
    under = [agg[PRIMARY][k]['aggregate_understates_the_body_route_loss_by_percentage_points']
             for k in show]
    for j, (vals, colr, lab) in enumerate(((body, '#1f77b4', 'BODY route alone'),
                                           (head, '#ff7f0e', 'HEAD route alone'),
                                           (both, '#d62728',
                                            'AGGREGATE = both routes driven'))):
        bars = ax.bar(x + (j - 1) * w, vals, w, color=colr, edgecolor='black', label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, '%.4f' % v, ha='center',
                    fontsize=9.6)
    ax.axhline(1.0, color='black', ls=':', lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('brain-side drive retained')
    ax.legend(loc='lower left', fontsize=9.4, framealpha=0.95, ncol=3)
    ax.set_title('2. THE AGGREGATE HIDES THE FAILURE (the numbers, same runs)\n'
                 'aggregate vs the route-resolved pair, %s' % PRIMARY, fontsize=13.5)
    lines = ['WORTHLESS READING: "the brain keeps %.1f%% of its touch drive, so the body route '
             'is mostly intact"' % (100 * both[0])]
    for i, key in enumerate(show):
        lines.append('%-14s AGG %.4f | BODY %.4f | HEAD %.4f | hides %5.1f pp'
                     % (key.split('@')[0], both[i], body[i], head[i], under[i]))
    lines.append('body-route LOSS: %s' % ', '.join('%.4f' % (1 - b) for b in body))
    lines.append('aggregate LOSS:  %s' % ', '.join('%.4f' % (1 - b) for b in both))
    axb.axis('off')
    axb.text(0.0, 1.0, '\n'.join(lines), transform=axb.transAxes, va='top', ha='left',
             fontsize=10.5, family=mono, bbox=box_red, linespacing=1.55)

    # ================= 3 brain-side spikes split by route ===========================
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(ARMS6))
    w = 0.26
    for j, (route, colr, lab) in enumerate((('body_only', '#1f77b4', 'body_only'),
                                            ('head_only', '#ff7f0e', 'head_only'),
                                            ('both', '#8c564b', 'both'))):
        vals = [paired[PRIMARY][route][a]['brain_side_spikes_settled'] for a in ARMS6]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=colr, edgecolor='black', label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 80, '%d' % v, ha='center', fontsize=9.2)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT6, fontsize=10)
    ax.set_ylabel('brain-side spikes (settled window)')
    ax.set_ylim(0, 8600)
    ax.legend(fontsize=10, title='stimulated route', title_fontsize=10, loc='upper center')
    ax.set_title('3. Brain-side SPIKE counts, split by route\n'
                 'spikes are nonlinear, so they need not follow the drive ratios', fontsize=13.5)
    _note(ax, 'annotation tier, body_only (%s):\nintact %d -> neck_cut %d\n'
              'electrode sheath INTACT %d, REMOVED %d\n'
              '(the spike-domain sheath difference is tiny:\nsee panels 7, 8 and 9)'
          % ((ANNOTATION,) + tuple(paired[ANNOTATION]['body_only'][a]['brain_side_spikes_settled']
                                   for a in ('intact', 'neck_cut',
                                             'electrode_sheath_intact@owner_shunt',
                                             'electrode_sheath_removed@owner_shunt'))),
          0.5, 0.72, fontsize=10, ha='center', box=box_grey)

    # ================= 4 the damage field in fly geometry (coronal + zoom) ===========
    ax = fig.add_subplot(gs[0, 3])
    L, W, TH = geo['length_um'], geo['width_um'], geo['thickness_um']
    ax.add_patch(plt.Rectangle((0, 0), L, W, fill=False, ec='k', lw=2.4))
    cx, cy = L / 2, W / 2
    cols = {'conservative': 'tab:red', 'optimal': 'tab:orange', 'permissive': 'tab:blue'}
    rows_t = _threshold_rows(report)
    for key, row in rows_t.items():
        ax.add_patch(plt.Circle((cx, cy), row['radius_removed'], fill=False, ec=cols[key],
                                lw=1.6, ls='--',
                                label='%s eps=%.2f: sheath REMOVED r=%.1f um'
                                      % (key, row['threshold'], row['radius_removed'])))
        ax.add_patch(plt.Circle((cx, cy), row['radius_intact'], fill=False, ec=cols[key], lw=2.6,
                                alpha=0.6,
                                label='%s: sheath INTACT r=%.1f um (x%.0f herniation)'
                                      % (key, row['radius_intact'], row['multiplier'])))
    ax.plot([cx, cx], [cy - 100, cy + 100], color='cyan', lw=7.0, alpha=0.85,
            label='shaft d=%.0f um (ASSUMED)' % table['shaft']['diameter_um'])
    ax.plot([cx], [cy], 'k+', ms=18, mew=3.2)
    box = 110.0
    ax.add_patch(plt.Rectangle((cx - box / 2, cy - box / 2), box, box, fill=False, ec='k',
                               lw=1.0, ls=':'))
    ax.set(xlim=(-30, 530), ylim=(-30, 330), xlabel='x (um)', ylabel='y (um)')
    ax.grid(alpha=0.18)
    ax.legend(loc='upper right', fontsize=8.6, framealpha=0.95)
    ax.set_title('4. Damage field in FLY geometry (coronal, TRUE scale)\n'
                 'the two sheath variants and the cross-species strain bracket', fontsize=13.5)
    inset = ax.inset_axes([0.035, 0.030, 0.46, 0.30])
    inset.add_patch(plt.Circle((cx, cy), table['shaft']['radius_um'], color='cyan', alpha=0.55))
    for key, row in rows_t.items():
        inset.add_patch(plt.Circle((cx, cy), row['radius_removed'], fill=False, ec=cols[key],
                                   lw=1.4, ls='--'))
        inset.add_patch(plt.Circle((cx, cy), row['radius_intact'], fill=False, ec=cols[key],
                                   lw=2.4, alpha=0.6))
    inset.set(xlim=(cx - box / 2, cx + box / 2), ylim=(cy - box / 2, cy + box / 2))
    inset.set_title('zoom 110 x 110 um', fontsize=9.4, pad=3)
    inset.tick_params(labelsize=8, length=2)
    _note(ax, 'The track is DECLARED, not measured:\nthe cache has NO coordinates for the\n'
              'ascending fibres or for the track,\nand FAFB stops at the neck.', 0.015, 0.985,
          fontsize=9.4, box=box_grey)

    # ================= 5 sagittal section ===========================================
    ax = fig.add_subplot(gs[1, 0])
    ax.add_patch(plt.Rectangle((0, 0), L, TH, facecolor='0.94', ec='k', lw=2.4))
    for key, row in rows_t.items():
        ax.add_patch(plt.Circle((cx, TH / 2), row['radius_removed'], fill=False, ec=cols[key],
                                lw=1.6, ls='--'))
        ax.add_patch(plt.Circle((cx, TH / 2), row['radius_intact'], fill=False, ec=cols[key],
                                lw=2.6, alpha=0.6))
    a_r = table['shaft']['radius_um']
    ax.add_patch(plt.Rectangle((cx - a_r, 0), 2 * a_r, TH, facecolor='cyan', alpha=0.8,
                               label='shaft d=%.0f um' % table['shaft']['diameter_um']))
    ax.set(xlim=(-30, 530), ylim=(-9, 190), xlabel='x (um)', ylabel='depth z (um)')
    ax.grid(alpha=0.18)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('5. Sagittal section (500 x %.0f um): the sheath is %.1f um of lamella +\n'
                 'perineurium and IS load-bearing (measured in cockroach/Manduca, ASSUMED here); '
                 'breached: %s'
                 % (TH, table['sheath']['thickness_um'],
                    table['variant_damage_table']['sheath_intact']['parenchymal_mechanistic']
                    ['sheath_breached']), fontsize=13)
    _note(ax, 'sheath-INTACT radius %.1f um > the ENTIRE\n%.0f um brain thickness: such a track '
              'spares NO depth.\nThe rodent "100 um kill zone" used as a radius would be 2.0x\n'
              'the whole thickness, which is why damage is reported\nas a FRACTION of the fly '
              'brain and never as a rodent radius.'
          % (ri['damage_radius_um'], TH), 0.5, 0.99, fontsize=9.6, ha='center', box=box_blue)

    # ================= 6 the measured local damage on the declared fibre =============
    ax = fig.add_subplot(gs[1, 1])
    for name, rec, colr, lab in (('sheath_intact', ri, 'tab:red', 'sheath INTACT'),
                                 ('sheath_removed', rm, 'tab:blue', 'sheath REMOVED')):
        cab = ctx['cable_table']['_arrays'][name]['mechanistic']
        n = cab['sham_final_mV'].size
        x_um = np.arange(n) * (table['declared_fixture']['fibre_length_um']
                               / table['declared_fixture']['compartments'])
        if name == 'sheath_intact':
            ax.plot(x_um, cab['sham_final_mV'], '-', color='0.45', lw=1.8,
                    label='sham (probe present, ZERO damage)')
        ax.plot(x_um, cab['inserted_final_mV'], '-', color=colr, lw=2.4, label=lab)
        span = rec['compartment_index_span']
        ax.axvspan(x_um[span[0]], x_um[span[1]], color=colr, alpha=0.18,
                   label='%s: %d reached compartments' % (lab, rec['reached_compartments']))
    lo = float(np.nanmin(ctx['cable_table']['_arrays']['sheath_intact']['mechanistic']
                         ['inserted_final_mV']))
    ax.set_ylim(lo - 2.0, -54.6)
    ax.set(xlabel='position along the DECLARED fibre (um); driven at 0, read at %.0f'
           % table['declared_fixture']['fibre_length_um'], ylabel='final Vm (mV)')
    ax.legend(fontsize=9.4, loc='lower left')
    ax.grid(alpha=0.2)
    ax.set_title('6. The MEASURED local damage on the declared fibre\n'
                 'extra leak %.4f uS / %d compartments (INTACT) vs %.4f uS / %d (REMOVED)'
                 % (ri['extra_leak_conductance_uS_total'], ri['reached_compartments'],
                    rm['extra_leak_conductance_uS_total'], rm['reached_compartments']),
                 fontsize=13)
    _note(ax, 'local attenuation AT the reached compartments: INTACT %.1f%%, REMOVED %.1f%%.\n'
              'The ASSUMED leak density (%.4g S/cm2 = x%.0f the intact membrane leak) pins those\n'
              'compartments towards E_leak in BOTH variants; what differs is HOW MANY compartments\n'
              'and HOW MUCH conductance, not the local percentage.'
          % (100 * ri['local_attenuation_fraction_at_reached_compartments'],
             100 * rm['local_attenuation_fraction_at_reached_compartments'],
             table['assumed_leak_density_S_cm2'], ri['extra_leak_over_intact_membrane_leak']),
          0.985, 0.985, fontsize=9.4, ha='right', box=box_blue)

    # ================= 7 sheath INTACT vs REMOVED ====================================
    ax = fig.add_subplot(gs[1, 2])
    cats = ['radius\n(um)', 'compartments\nreached', 'extra leak\n(uS x 1000)',
            'brain volume\nfraction (%)', 'transmitted\ndrive q (x1000)',
            'chemistry\nsheath term']
    cs = chem['per_variant']
    ci = [ri['damage_radius_um'], ri['reached_compartments'],
          1000 * ri['extra_leak_conductance_uS_total'],
          100 * ri['fly_brain_volume_fraction_of_track'],
          1000 * ri['output_transmission_factor_q'],
          cs['sheath_intact']['closed']['sheath_conductance_change_um3_s']]
    cm = [rm['damage_radius_um'], rm['reached_compartments'],
          1000 * rm['extra_leak_conductance_uS_total'],
          100 * rm['fly_brain_volume_fraction_of_track'],
          1000 * rm['output_transmission_factor_q'],
          cs['sheath_removed']['closed']['sheath_conductance_change_um3_s']]
    x = np.arange(len(cats))
    b1 = ax.bar(x - 0.2, ci, 0.4, color='tab:red', edgecolor='k', label='sheath INTACT')
    b2 = ax.bar(x + 0.2, cm, 0.4, color='tab:blue', edgecolor='k', label='sheath REMOVED')
    for bars, vals in ((b1, ci), (b2, cm)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.1, '%.3g' % v, ha='center',
                    fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9.6)
    ax.set_ylim(0, 82)
    ax.set_ylabel('value in the unit named on the tick')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_title('7. SHEATH INTACT vs REMOVED: every measured difference together\n'
                 'the barrier-chemistry difference is the one that is EXACT by construction',
                 fontsize=13)
    _note(ax, 'chemistry via the existing injury_tissue API: the sheath barrier conductance\n'
              'change is EXACTLY %.1f um3/s with an intact sheath and EXACTLY %.1f um3/s when it\n'
              'is already gone, while the K permeability change is the SAME %.3g um/s in both --\n'
              'the DECLARED 8x8x8 um tissue element lies entirely inside the damage cylinder at\n'
              'both radii. That saturation is reported, not hidden.'
          % (cs['sheath_intact']['closed']['sheath_conductance_change_um3_s'],
             cs['sheath_removed']['closed']['sheath_conductance_change_um3_s'],
             cs['sheath_removed']['closed']['permeability_change_um_s']),
          0.5, 0.985, fontsize=9.2, ha='center', box=box_green)

    # ================= 8 where the sheath difference lives ===========================
    sub = gs[1, 3].subgridspec(2, 1, height_ratios=[3.0, 2.1], hspace=0.10)
    ax = fig.add_subplot(sub[0])
    axb = fig.add_subplot(sub[1])
    dens = sweeps['assumed_leak_density_S_cm2']
    for variant, colr in (('sheath_intact', 'tab:red'), ('sheath_removed', 'tab:blue')):
        rows = dens['by_variant'][variant]
        ax.loglog([r['leak_density_S_cm2'] for r in rows],
                  [max(r['output_transmission_factor_q'], 1e-18) for r in rows], '-o', color=colr,
                  lw=2.4, ms=9, label='%s: transmitted drive q' % variant)
    ax.axvline(dens['primary_value_S_cm2'], color='k', ls='--', lw=1.8,
               label='ASSUMED %.4g S/cm2 (round 1)' % dens['primary_value_S_cm2'])
    ax.set(xlabel='ASSUMED damaged leak density (S/cm2)', ylabel='transmitted drive q',
           ylim=(1e-18, 30.0))
    ax.grid(alpha=0.2, which='both')
    ax.legend(fontsize=9.4, loc='lower left')
    ax.set_title('8. The sheath difference lives in the TRANSMITTED DRIVE\n'
                 'q = %.2g (INTACT) vs %.2g (REMOVED) at the assumed density'
                 % (ri['output_transmission_factor_q'], rm['output_transmission_factor_q']),
                 fontsize=13)
    txt = ['FULL paired runs at OTHER assumed densities (body_only, %s):' % PRIMARY]
    for r in lk['rows']:
        txt.append('   leak %.4g  %-14s retained %.4f   brain spikes %4d'
                   % (r['assumed_leak_density_S_cm2'], r['variant'],
                      r['network_body_route_drive_retained'], r['network_brain_side_spikes']))
    txt.append('   => the variants separate in the NETWORK only at SMALL assumed leaks; at the')
    txt.append('      assumed value both leave the reached fibre quiet. The leak density is')
    txt.append('      ASSUMED, so this is a sensitivity, never a fly value.')
    axb.axis('off')
    axb.text(0.0, 1.0, '\n'.join(txt), transform=axb.transAxes, va='top', ha='left',
             fontsize=9.6, family=mono, bbox=box_blue, linespacing=1.5)

    # ================= 9 declared placement WHAT-IF ==================================
    ax = fig.add_subplot(gs[2, 0])
    for variant, colr in (('sheath_intact', 'tab:red'), ('sheath_removed', 'tab:blue')):
        rows = share['by_variant'][variant]['rows']
        ax.plot([r['declared_reached_share'] for r in rows],
                [r['body_route_drive_retained'] for r in rows], '-o', color=colr, lw=2.6, ms=9,
                label='%s (q=%.2g)' % (variant, share['by_variant'][variant]['q']))
    conf = share['network_confirmation']
    ax.plot([v['declared_reached_share'] for v in conf.values()],
            [v['body_route_drive_retained'] for v in conf.values()], 'k*', ms=20,
            label='full paired runs at 3 declared shares')
    ax.set(xlabel='DECLARED share of the ascending fibre population the cylinder reaches\n'
                  '1.0 = the maximum-damage declaration used by every electrode scenario',
           ylabel='body-route drive retained', ylim=(-0.05, 1.65))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9.6, loc='upper center')
    ax.set_title('9. WHAT-IF: where the track is DECLARED to sit\n'
                 'no coordinates exist, so the share is a DECLARATION and is swept', fontsize=13)
    _note(ax, 'exact ledger arithmetic: retained = (q*reached + (total-reached))/total on the\n'
              'intact spike train (brain-side drive %.2f uS). The black stars are FULL network\n'
              're-runs at shares %s, and they sit ABOVE the curves because those\n'
              'runs include network feedback while the curves hold activity fixed; the exact\n'
              'curves retain %s at those shares. Turning this into one number would need a\n'
              'placement measurement that does not exist.'
          % (share['brain_side_drive_uS_from_intact_spike_train'],
             ', '.join('%.3f' % float(k) for k in conf),
             ', '.join('%.4f' % v['body_route_drive_retained'] for v in conf.values())),
          0.5, 0.60, fontsize=9.6, ha='center', box=box_grey)

    # ================= 10 SURGERY WHAT-IF ===========================================
    sub = gs[2, 1].subgridspec(2, 1, height_ratios=[3.0, 2.3], hspace=0.10)
    ax = fig.add_subplot(sub[0])
    axb = fig.add_subplot(sub[1])
    rows = surgery['what_if_sweep']['rows']
    ax.semilogy([r['assumed_surgery_damage_radius_um'] for r in rows],
                [max(r['body_route_drive_retained_if_this_were_true'], 1e-18) for r in rows],
                '-o', color='#8000a0', lw=2.8, ms=10,
                label='WHAT-IF: an assumed surgery radius')
    ax.axhline(rm['output_transmission_factor_q'], color='tab:blue', ls='--', lw=2.2,
               label='the MODELLED electrode (r=%.2f um)' % rm['damage_radius_um'])
    for r in surgery['what_if_network_confirmation']['rows']:
        ax.plot([r['assumed_surgery_damage_radius_um']],
                [max(r['body_route_drive_retained_if_this_were_true'], 1e-18)], 'k*', ms=20,
                label='full paired run under that assumption')
    ax.set(xlabel='ASSUMED surgery damage radius (um) -- HYPOTHETICAL AXIS',
           ylabel='body-route drive retained', ylim=(1e-18, 3e3))
    ax.grid(alpha=0.2, which='both')
    ax.legend(fontsize=9.4, loc='lower left')
    ax.set_title('10. SURGERY TERM: WHAT-IF ONLY -- NOT A RESULT, NOT A MEASUREMENT',
                 fontsize=14, color='#a00000')
    axb.axis('off')
    axb.text(0.0, 1.0,
             'In real fly preparations the sheath is removed or enzymatically breached FIRST, so\n'
             'insertion damage may be dominated by the SURGERY -- and NO insect study quantifies\n'
             'surgery damage. No value is invented: the term is registered UNQUANTIFIED (value =\n'
             'null) and it is NOT modelled in ANY scenario, which is exactly why the sham is\n'
             'bit-identical to intact here. This axis only shows how far an ASSUMED surgical\n'
             'insult would have to reach before it reproduced the electrode\'s own effect; at\n'
             '~7.5 um -- BELOW the electrode\'s own modelled %.1f um -- it already does. So the\n'
             'electrode-specific part is NOT separable from surgery without a measurement that\n'
             'does not exist.' % rm['damage_radius_um'],
             transform=axb.transAxes, va='top', ha='left', fontsize=10.2, bbox=box_yell,
             linespacing=1.45)

    # ================= 11 declared clearance and track position ======================
    sub = gs[2, 2].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    ax = fig.add_subplot(sub[0])
    axb = fig.add_subplot(sub[1])
    cl = sweeps['declared_fibre_clearance']['by_variant']
    for variant, colr in (('sheath_intact', 'tab:red'), ('sheath_removed', 'tab:blue')):
        ax.semilogy([r['extra_clearance_um'] for r in cl[variant]],
                    [max(r['output_transmission_factor_q'], 1e-18) for r in cl[variant]], '-o',
                    color=colr, lw=2.4, ms=9,
                    label='%s' % variant.replace('_', ' '))
        for r in cl[variant]:
            ax.text(r['extra_clearance_um'], max(r['output_transmission_factor_q'], 1e-18) * 2.2,
                    '%d' % r['reached_compartments'], ha='center', fontsize=8.4, color=colr)
    ax.set(xlabel='DECLARED extra clearance of the fibre above wall contact (um)\n'
                  '0 = the fibre lies against the shaft wall (the primary declaration)',
           ylabel='transmitted drive q', ylim=(1e-18, 30.0))
    ax.grid(alpha=0.2, which='both')
    ax.legend(fontsize=9.4, loc='lower right', title='reached compartments shown', title_fontsize=8.4)
    ax.set_title('11. The DECLARED geometry moves the answer: clearance', fontsize=13)
    pos = sweeps['declared_track_position_along_fibre']['by_variant']
    for variant, colr in (('sheath_intact', 'tab:red'), ('sheath_removed', 'tab:blue')):
        axb.plot([100 * r['crossing_fraction_of_length'] for r in pos[variant]],
                 [max(r['output_transmission_factor_q'], 1e-18) for r in pos[variant]], '-s',
                 color=colr, lw=2.4, ms=9, label='%s' % variant.replace('_', ' '))
    axb.axvline(75.0, color='k', ls=':', lw=1.5)
    axb.set_yscale('log')
    axb.set(xlabel='DECLARED position of the track along the fibre (% of length)\n'
                   '75%% = the primary declaration (dotted)', ylabel='transmitted drive q',
            ylim=(1e-18, 30.0))
    axb.grid(alpha=0.2, which='both')
    axb.legend(fontsize=9.4, loc='lower left')
    axb.set_title('track position, same declaration set', fontsize=11.5)

    # ================= 12 where the RESIDUAL brain-side drive comes from =============
    ax = fig.add_subplot(gs[2, 3])
    arms12 = [('intact', '#4c72b0'), ('neck_cut', '#55a868'),
              ('electrode_sheath_removed@owner_shunt', '#c44e52'),
              ('electrode_sheath_removed@drive_scaled', '#8172b2')]
    groups12 = ['crossing_ascending', 'crossing_descending', 'other']
    x = np.arange(len(arms12))
    w = 0.26
    for j, g in enumerate(groups12):
        raw = [paired[PRIMARY]['body_only'][arm][
            'drive_at_brain_side_by_presynaptic_group_uS_settled'][g] for arm, _ in arms12]
        vals = [max(v, 3e-7) for v in raw]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=['#1f77b4', '#d62728', '#7f7f7f'][j],
                      edgecolor='black', label='from %s' % g.replace('_', ' '))
        for b, v, rr in zip(bars, vals, raw):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.6,
                    '%.3f' % rr if rr > 1e-6 else 'EXACTLY 0', ha='center', fontsize=8.6,
                    rotation=90 if not (rr > 1e-6) else 0)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 3e5)
    ax.set_xticks(x)
    ax.set_xticklabels(['intact', 'neck_cut', 'electrode\nsheath REMOVED\n(owner shunt)',
                        'electrode\nsheath REMOVED\n(drive scaled)'], fontsize=9.6)
    ax.set_ylabel('brain-side drive by PRESYNAPTIC group (uS, log)')
    ax.legend(fontsize=9.4, loc='upper right')
    ax.set_title('12. WHERE the residual drive comes from (body_only, %s)\n'
                 'the cut leaves 0.000 uS; the electrode leaves drive the shunted fibres still '
                 'emit' % PRIMARY, fontsize=13)
    _note(ax, 'The residual is NOT a different route reaching the brain:\n'
              'under the owner-shunt coupling the reached fibres are only 10x-95x\n'
              'leakier and still spike under synaptic drive, so they keep emitting\n'
              '(row share shown above). Under drive_scaled their output is multiplied\n'
              'by the measured q and only %.4f uS survives. Both are reported.'
          % paired[PRIMARY]['body_only']['electrode_sheath_removed@drive_scaled']
          ['brain_side_drive_uS_settled'], 0.015, 0.985, fontsize=9.2, ha='left', box=box_grey)

    # ================= 13 exactness / defect / audits (full width) ===================
    ax = fig.add_subplot(gs[3, 0:2])
    ax.axis('off')
    att = report['step7_exact_attribution_with_activity_fixed'][PRIMARY]['body_only']
    grp = att['neck_cut']['at_brain_side_uS']['severed_by_neck_cut_per_group']
    att_e = att['electrode_drive_scaled']['at_brain_side_uS']
    lines = [
        'EXACTNESS AND COUPLING ACCOUNTING (activity HELD FIXED, %s / body_only)' % PRIMARY,
        '  severed conductance at the brain side, per presynaptic group (the cut):',
        '      crossing_ascending %9.5f uS     crossing_descending %9.5f uS'
        % (grp['crossing_ascending'], grp['crossing_descending']),
        '      body_route_sensor  %9.5f uS     head_route_sensor   %9.5f uS   <- EXACTLY zero'
        % (grp['body_route_sensor'], grp['head_route_sensor']),
        '  the electrode (drive_scaled) linear identity, with the same spike train:',
        '      q*reached + unreached = %.5f uS   vs delivered %.5f uS'
        % (att_e['predicted_electrode_total'], att_e['electrode_total']),
        '      residual %.3g uS (relative %.2g);  reached rows at q=1 carry %.5f uS of the '
        '%.5f uS intact' % (att_e['linear_identity_residual_uS'],
                            att_e['linear_identity_relative_residual'],
                            att_e['reached_rows_at_q1_total'], att_e['intact_total']),
        '  the electrode ON TOP of the cut: the brain-side COLUMNS of the spike matrix are '
        'IDENTICAL to the',
        '      cut arm; only the removed fibres\' own rows differ, and all of them lie inside '
        'the reached owners.',
        '',
        'ROUND-1 MASK DEFECT: FOUND, MEASURED, AND CORRECTED HERE (handed to the parent)',
        '  round-1 group masks folded EVERY out-of-tier neuron onto local row n-1, because '
        'local_of_global is -1',
        '  outside the tier and the round-1 idiom indexed with it, which NumPy wraps to the last '
        'row:',
        ('      masks affected %d of %d; extra rows folded into the brain_side mask: %s\n'
         '      effect on the intact arm: drive %.5f -> %.5f uS, brain-side spikes %d -> %d\n'
         '      (round-1 legacy -> corrected); total spikes UNCHANGED, because the defect moved\n'
         '      the ATTRIBUTION of spikes between groups, not the simulated activity.\n'
         '      The DEFAULT is now %s (out-of-tier ids EXCLUDED, never wrapped); the round-1\n'
         '      behaviour survives ONLY as the explicitly named contrast arm %s, and every\n'
         '      old-vs-new number is written to step11 of the report.'
         % (sum(1 for v in defect['audit']['per_mask'].values() if v['masks_differ']),
            len(defect['audit']['per_mask']),
            defect['audit']['per_mask']['route_overlapping.brain_side']
            ['extra_local_rows_under_round1_semantics'],
            defect['old_vs_new_on_the_intact_arm']['legacy_semantics'][
                'brain_side_drive_uS_settled'],
            defect['old_vs_new_on_the_intact_arm']['correct_semantics'][
                'brain_side_drive_uS_settled'],
            defect['old_vs_new_on_the_intact_arm']['legacy_semantics'][
                'brain_side_spikes_settled'],
            defect['old_vs_new_on_the_intact_arm']['correct_semantics'][
                'brain_side_spikes_settled'],
            defect['default_semantics_in_this_report'],
            defect['legacy_contrast_arm_semantics'])),
        '',
        'SURGERY TERM  "%s": status %s, value %s, modelled in any scenario: %s, audit violations: %d'
        % (surgery['term']['name'], surgery['term']['status'], surgery['term']['value'],
           surgery['term']['modelled_in_any_scenario'],
           len(tds.audit_surgery_claims(report))),
        'REACHABILITY  used as evidence: %s, audit violations: %d (the audit is strict enough '
        'that it caught one of'
        '  our own field names, which was RENAMED rather than whitelisted)'
        % (report['reachability_is_NOT_used_as_evidence']['used_as_evidence'],
           len(tds.audit_reachability(report))),
        'SELFTESTS  %d checks, %d passed, %d failed, %d skipped (%.1f s)   |   RUN  %.1f s wall '
        'clock, peak RSS %.0f MB, %d network runs'
        % (report['selftests']['n_checks'], report['selftests']['n_passed'],
           report['selftests']['n_failed'], report['selftests'].get('n_skipped', 0),
           report['selftests']['wall_seconds'], report['run']['wall_clock_s'],
           report['run']['peak_rss_mb'], report['run']['n_network_runs']),
    ]
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=11,
            family=mono, bbox=box_grey, linespacing=1.42)
    ax.set_title('13. Exactness, the round-1 defect found here, and what is asserted',
                 fontsize=13.5)

    # ================= 14 honesty / corrections (full width) ========================
    ax = fig.add_subplot(gs[3, 2:4])
    ax.axis('off')
    import textwrap
    lines = ['HONESTY RULES -- each line is a property of this work, not a caveat:']
    for ln in honesty:
        lines += textwrap.wrap(ln, width=138, initial_indent='   - ',
                               subsequent_indent='     ')[:3]
    lines += ['', 'ROUND-1 CORRECTIONS KEPT HERE:',
              '   - a matching crossing-set TOTAL is NOT independent confirmation of the '
              'partition: the per-population',
              '     counts do not reproduce and the paper\'s own three components sum to 3,682, '
              'not 3,686',
              '   - the cut is ROUTE-SELECTIVE, not "brain isolation": %d of %d head-route output '
              'synapses (%.2f%%) ARE severed,'
              % (int(hc['head_route_output_synapses_severed']),
                 int(hc['head_route_output_synapses']),
                 100 * hc['head_route_output_fraction_severed']),
              '     by %d cells the dataset itself calls neck-crossing; 100%% of the BODY route\'s '
              'direct connection into' % hc['head_route_neurons_also_annotated_crossing'],
              '     the brain (%d of %d synapses) is severed, so the two routes behave differently'
              % (int(report['step1_round1_inputs']['route_budget']['body_route'][
                     'severed_synapses_into_brain_side']),
                 int(report['step1_round1_inputs']['route_budget']['body_route'][
                     'to_brain_side_direct'])),
              '   - NO aggregate brain-drive retention figure may be quoted (panel 2)',
              '', 'WHY NO SINGLE NUMBER ANSWERS THE QUESTION:',
              '   - the aggregate reads %.4f after the neck cut while the body route alone retains '
              '%.4f: quoting the'
              % (agg[PRIMARY]['neck_cut']['both_drive_uS']['retained_fraction'],
                 agg[PRIMARY]['neck_cut']['body_only_drive_uS']['retained_fraction']),
              '     first hides a COMPLETE body-route loss behind a %.4f aggregate loss'
              % agg[PRIMARY]['neck_cut']['aggregate_loss_fraction'],
              '   - the electrode\'s own effect is bracketed by the two measured couplings '
              '(%.4f vs %.4f retained, panels 1/12)'
              % (paired[PRIMARY]['body_only']['electrode_sheath_removed@owner_shunt'][
                     'brain_side_drive_retained_vs_intact_settled'],
                 paired[PRIMARY]['body_only']['electrode_sheath_removed@drive_scaled'][
                     'brain_side_drive_retained_vs_intact_settled']),
              '     because the assumed leak conductance does not stop a strongly driven point '
              'neuron from spiking',
              '   - the sheath-intact vs sheath-removed difference IS measurable (panel 7) but '
              'under the assumed leak',
              '     density it does not show up in the spike counts (panel 8)']
    ax.text(0.0, 1.0, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=10.4,
            family=mono, bbox=box_red, linespacing=1.42)
    ax.set_title('14. Honesty, scope, and the corrections carried forward', fontsize=13.5)

    fig.savefig(path, dpi=100, facecolor='white')
    plt.close(fig)
    return path


def _threshold_rows(report):
    """Per-threshold radii for both sheath variants, from the reported sweep."""
    out = {}
    for row in report['step9_assumed_parameter_sensitivity']['declared_and_assumed_sweeps'][
            'cross_species_strain_threshold']['rows']:
        key = row['threshold_key']
        out.setdefault(key, {'threshold': row['threshold']})
        if row['variant'] == 'sheath_intact':
            out[key]['radius_intact'] = row['parenchymal_radius_um']
        else:
            out[key]['radius_removed'] = row['parenchymal_radius_um']
    mult = report['step2_electrode_model']['variant_damage_table']['sheath_intact'][
        'parenchymal_mechanistic']['parenchymal_multiplier']
    for row in out.values():
        row['multiplier'] = mult
        row.setdefault('radius_intact', row.get('radius_removed', 0.0) * mult)
        row.setdefault('radius_removed', 0.0)
    return out


# ---------------------------------------------------------------------------
def _save_arrays(ctx, chem, path):
    npz = {}
    npz['scenario_names'] = np.asarray(list(tds.SCENARIO_NAMES))
    npz['times_ms'] = np.arange(ctx['cfg'].steps + 1) * ctx['cfg'].dt_ms
    npz['tier_global_index'] = ctx['tier']['global_index']
    npz['tier_root_ids'] = ctx['tier']['root_ids']
    npz['reached_owner_local'] = tds.deterministic_share_subset(
        tds.reached_owner_local(ctx['tier'], ctx['groups']), ctx['reach_share'])
    npz['brain_side_global_rows'] = np.flatnonzero(np.asarray(
        ctx['groups']['route_overlapping']['brain_side']))
    npz['disjoint_group_order'] = np.asarray(DISJOINT_ORDER)
    for (sm, route), matrix in ctx['grid']['intact_spike_matrices'].items():
        npz['intact_spikes__%s__%s' % (sm, route)] = matrix.astype(np.uint8)
    for (sm, route, arm), matrix in ctx['grid']['extra_spike_matrices'].items():
        npz['spikes__%s__%s__%s' % (sm, route, arm)] = matrix.astype(np.uint8)
    for (sm, route, arm), res in ctx['grid']['results'].items():
        npz['counts__%s__%s__%s' % (sm, route, arm)] = res['spike_counts']['settled'][
            'per_neuron'].astype(np.int32)
        npz['brain_drive_uS__%s__%s__%s' % (sm, route, arm)] = np.asarray(
            [res['conductance_at_brain_side_uS']['settled'][g] for g in DISJOINT_ORDER])
    for name in ctx['reference']['variants']:
        cab = ctx['cable_table']['_arrays'][name]['mechanistic']
        npz['cable_transmission__%s' % name] = cab['transmission_per_compartment']
        npz['cable_g_end_uS__%s' % name] = cab['g_end_uS_per_compartment']
        npz['cable_sham_final_mV__%s' % name] = cab['sham_final_mV']
        npz['cable_inserted_final_mV__%s' % name] = cab['inserted_final_mV']
        npz['cable_distance_from_axis_um__%s' % name] = cab['distances_to_shaft_axis_um']
    for name, blk in chem['per_variant'].items():
        for variant in ('closed', 'supported'):
            traces = blk['_arrays'][variant]
            npz['chem_%s_%s_time_s' % (name, variant)] = traces['sham'][:, 0].copy()
            npz['chem_%s_%s_Ke_sham_mM' % (name, variant)] = traces['sham'][:, 2].copy()
            npz['chem_%s_%s_Ke_inserted_mM' % (name, variant)] = traces['inserted'][:, 2].copy()
            npz['chem_%s_%s_far_ligand_sham_nM' % (name, variant)] = traces['sham'][:, 5].copy()
            npz['chem_%s_%s_far_ligand_ins_nM' % (name, variant)] = traces['inserted'][:,
                                                                                     5].copy()
    np.savez_compressed(path, **npz)
    return path


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print('building the six-scenario context ...')
    ctx = tds.build_context(ROOT,
                            extra_spike_scenarios=('neck_cut', 'neck_cut_plus_electrode'))
    print('  %d network runs' % len(ctx['grid']['results']))
    print('running the selftest suite against that context ...')
    # The DEMO cannot audit the report it has not written yet, so this in-process run skips the
    # report-auditing checks with an explicit reason and its result is written to its OWN file.
    # The canonical touch_damage_selftest.json is owned by run_touch_damage_selftest.py, which
    # audits the written report and the produced artifacts (and is what a verifier should quote).
    tests = run_tests(ctx=ctx, report=None, require_artifacts=False, report_pending=True)
    assert tests['n_failed'] == 0, 'selftest failed; not writing a report'
    demo_tests_path = OUT / 'touch_damage_selftest_demo_path.json'
    demo_tests_path.write_text(json.dumps(jsafe(tests), indent=2))

    report, paired, aggregate = build_report(ctx, tests, t0)
    chem = report['step10_chemistry_through_injury_tissue_api']
    rep_path = OUT / 'touch_damage_report.json'
    fig_path = OUT / 'touch_damage_demo.png'
    npz_path = OUT / 'touch_damage_traces.npz'

    _save_arrays(ctx, chem, npz_path)
    make_figure(report, fig_path, ctx)

    report['step12_audits'] = {
        'it_already_caught_one_of_our_own_names': (
            'the reachability audit FAILED this report during development because an accounting '
            'field was named "reachable_conductance_uS". That field is the conductance of the '
            'rows whose presynaptic owner the damage cylinder reaches -- nothing to do with graph '
            'reachability -- and it was RENAMED to damage_reached_conductance_uS rather than '
            'whitelisted. The audit stays strict.'),
        'how_they_are_computed': ('audit_reachability flags any NUMERIC value under a '
                                  'reachability-named key; audit_surgery_claims flags any numeric '
                                  'surgery damage outside a what-if subtree, plus any what-if '
                                  'entry claiming to be a result or a measurement. The selftest '
                                  're-runs both AND proves they fire on poisoned copies.'),
        'surgery_term_audit_violations': tds.audit_surgery_term(),
        'surgery_claim_audit_violations': tds.audit_surgery_claims(report),
        'reachability_audit_violations': tds.audit_reachability(report),
    }
    assert report['step12_audits']['reachability_audit_violations'] == []
    assert report['step12_audits']['surgery_claim_audit_violations'] == []
    assert report['step12_audits']['surgery_term_audit_violations'] == []
    report['figure'] = {
        'path': str(fig_path), 'panels': 14,
        'honesty_lines_drawn': len(report['honesty']['statements']),
        'route_resolved_across_all_six_scenarios': [1, 2, 3, 12],
        'damage_field_in_fly_geometry': [4, 5, 6],
        'sheath_intact_vs_sheath_removed': [6, 7, 8, 11],
        'aggregate_and_route_pair_side_by_side_with_numbers': [2],
        'surgery_what_if_sweep': [10],
        'declared_placement_what_ifs': [9, 11],
        'exactness_and_defect': [13],
        'honesty_and_scope': [14],
    }
    report['run']['wall_clock_s'] = time.perf_counter() - t0
    report['run']['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report['outputs'] = {'report': str(rep_path), 'traces': str(npz_path),
                         'figure': str(fig_path),
                         'selftest': str(OUT / 'touch_damage_selftest.json'),
                         'selftest_demo_path': str(OUT / 'touch_damage_selftest_demo_path.json')}
    rep_path.write_text(json.dumps(jsafe(report), indent=2))

    print('\nROUTE-RESOLVED PAIRED RESULTS (counterfactual tier, primary coupling):')
    for route in tds.ACTIVATED_ROUTES:
        print('  route %s' % route)
        for arm in ARMS6:
            b = paired[PRIMARY][route][arm]
            print('    %-42s drive %10.5f uS  retained %10s  brain spikes %5d  rows_removed %6d'
                  % (arm, b['brain_side_drive_uS_settled'],
                     tds._fmt(b['brain_side_drive_retained_vs_intact_settled']),
                     b['brain_side_spikes_settled'], b['rows_removed_by_cut_mode']))
    print('\nANNOTATION TIER (head route has ZERO sign coverage):')
    for route in tds.ACTIVATED_ROUTES:
        for arm in ARMS6:
            b = paired[ANNOTATION][route][arm]
            if arm in ('intact', 'neck_cut', 'electrode_sheath_removed@owner_shunt'):
                print('  %-9s %-42s drive %10.5f  retained %10s  brain spikes %5d'
                      % (route, arm, b['brain_side_drive_uS_settled'],
                         tds._fmt(b['brain_side_drive_retained_vs_intact_settled']),
                         b['brain_side_spikes_settled']))
    print('\nAGGREGATE HIDES THE ROUTE LOSS:')
    for arm in ('neck_cut', 'electrode_sheath_intact@owner_shunt',
                'electrode_sheath_removed@owner_shunt',
                'neck_cut_plus_electrode@owner_shunt'):
        a = aggregate[PRIMARY][arm]
        print('  %-42s aggregate %.6f (loss %.6f) | body %.6f (loss %.6f) | head %.6f '
              '| understates by %.2f pp'
              % (arm, a['both_drive_uS']['retained_fraction'], a['aggregate_loss_fraction'],
                 a['body_only_drive_uS']['retained_fraction'], a['body_route_loss_fraction'],
                 a['head_only_drive_uS']['retained_fraction'],
                 a['aggregate_understates_the_body_route_loss_by_percentage_points']))
    print('\nSURGERY: term "%s" status %s value %s modelled %s | what-if rows %d (all labelled)'
          % (report['step3_surgery_term']['term']['name'],
             report['step3_surgery_term']['term']['status'],
             report['step3_surgery_term']['term']['value'],
             report['step3_surgery_term']['term']['modelled_in_any_scenario'],
             len(report['step3_surgery_term']['what_if_sweep']['rows'])))
    print('\nwrote:')
    for p in (rep_path, npz_path, fig_path):
        print('  %s  (%.2f MB)' % (p, p.stat().st_size / 1e6))
    print('wall clock %.1f s, peak RSS %.0f MB, %d network runs'
          % (report['run']['wall_clock_s'], report['run']['peak_rss_mb'],
             report['run']['n_network_runs']))
    return report


if __name__ == '__main__':
    main()
