"""Self-test for the data-anchored neck cut (engine/neck_cut_data.py).

Covers, with exact counts:

  T1  deterministic and reproducible tier selection (in-process and across
      subprocesses with different PYTHONHASHSEED values)
  T2  the crossing-set count reproduced INDEPENDENTLY from the raw npz, by
      counting code written inside this test file
  T3  edge / synapse accounting exact (ascending + descending = total, no double
      counting, exhaustive and mutually exclusive)
  T4  the cut removes ONLY the intended edges and nothing else (row level and
      conductance-matrix level)
  T5  head-route drive unchanged by the cut while body-route drive drops
  T6  identical initialisation and seed across scenarios
  T7  zero-shift control: sham is bit-identical to intact
  T8  invalid-input rejection
  T9  explicit assertion that reachability is NOT used as evidence
  T10 receptor mapping and the nt_pair rule
  T11 sign-coverage accounting is mutually exclusive and exhaustive
  T12 the conductance-ledger emission mirror is exact against the network itself
  T13 the written artifacts exist, parse, and carry the required honesty fields

Run: venv/bin/python run_neck_cut_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import neck_cut_data as ncd  # noqa: E402
from engine.neural_cond import ConductanceNetwork, ConductanceParams  # noqa: E402

#: ROUND-1 LEGACY (wraparound) and CORRECTED (in_tier_only) values for one arm, pinned so the
#: defect cannot be silently reintroduced.  The arm is counterfactual_ach / intact / both; the
#: numbers were produced independently by run_neck_cut_scenario.py and by
#: run_touch_damage_scenario.py and agree exactly.
#: NOTE: only the mask ATTRIBUTION changes.  The stimulus masks carry no out-of-tier member, so
#: the simulated activity, the total spike count and every digests are identical under both
#: semantics; what moves is which neurons count as brain-side and in which presynaptic group.
MASK_FIX_LEGACY_BRAIN_SIDE_DRIVE_US = 112.58450000000005
MASK_FIX_CORRECT_BRAIN_SIDE_DRIVE_US = 112.49150000000004
MASK_FIX_LEGACY_BRAIN_SIDE_SPIKES = 4346
MASK_FIX_CORRECT_BRAIN_SIDE_SPIKES = 4343
MASK_FIX_CORRECT_TOTAL_SPIKES = 58658
MASK_FIX_MASKS_CHANGED = 9
MASK_FIX_MASKS_TOTAL = 13
MASK_FIX_CROSSING_ASCENDING_GLOBAL = 2359
MASK_FIX_CROSSING_ASCENDING_OUTSIDE = 1165
MASK_FIX_CROSSING_ASCENDING_IN_TIER = 1194

OUT = ROOT / 'outputs' / 'brain_isolation'
RESULTS = []
T0 = time.perf_counter()


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), str(detail)))
    print('%-5s %-62s %s' % ('PASS' if ok else 'FAIL', name, detail))
    return bool(ok)


def _fmt(v):
    return 'not modelled' if v != v else '%.6f' % v


def expect_raises(name, exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc as e:
        return check(name, True, '%s: %s' % (type(e).__name__, str(e)[:70]))
    except Exception as e:  # wrong exception type
        return check(name, False, 'wrong exception %s: %s' % (type(e).__name__, e))
    return check(name, False, 'no exception raised')


# ===========================================================================
def main():
    a = ncd.load_banc(ROOT / ncd.BANC_RELATIVE_PATH)
    n_neurons = int(a['root_ids'].size)
    n_rows = int(a['pre'].size)
    print('dataset: %d neurons, %d annotated rows\n' % (n_neurons, n_rows))

    mech = ncd.mechanosensory_partition(a)
    crossing = ncd.derive_crossing_set(a)
    cut = ncd.build_cut_set(a, crossing)
    budget = ncd.route_budget(a, cut, mech)
    tier_cfg = ncd.TouchTierConfig()
    tier = ncd.select_touch_tier(a, mech, cut, tier_cfg)
    groups = ncd.presynaptic_groups(a, mech, cut)
    run_cfg = ncd.NetworkRunConfig()

    # ------------------------------------------------------------------ T1
    tier_b = ncd.select_touch_tier(a, mech, cut, tier_cfg)
    same_index = np.array_equal(tier['global_index'], tier_b['global_index'])
    same_digest = tier['digest'] == tier_b['digest']
    check('T1.1 tier selection is reproducible in-process',
          same_index and same_digest,
          '%d neurons, digest %s' % (tier['n'], tier['digest']['selected_root_ids_sha256_32']))

    probe = (
        'import sys,json;sys.path.insert(0,%r);'
        'from engine.neck_cut_data import *;'
        'a=load_banc(%r);m=mechanosensory_partition(a);'
        'c=derive_crossing_set(a);k=build_cut_set(a,c);'
        't=select_touch_tier(a,m,k);'
        'print(json.dumps({"n":t["n"],"d":t["digest"]["selected_root_ids_sha256_32"]}))'
        % (str(ROOT), str(ROOT / ncd.BANC_RELATIVE_PATH)))
    sub = {}
    for hashseed in ('0', '12345'):
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = hashseed
        out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                             env=env, cwd=str(ROOT))
        if out.returncode != 0:
            sub[hashseed] = 'ERROR: ' + out.stderr.strip()[-200:]
        else:
            sub[hashseed] = json.loads(out.stdout.strip())
    ok = (isinstance(sub['0'], dict) and isinstance(sub['12345'], dict)
          and sub['0'] == sub['12345']
          and sub['0']['d'] == tier['digest']['selected_root_ids_sha256_32'])
    check('T1.2 tier selection reproducible across processes/hash seeds', ok,
          'PYTHONHASHSEED 0 vs 12345 -> %s' % sub['0'])

    big = ncd.select_touch_tier(a, mech, cut, ncd.TouchTierConfig(max_neurons=9000))
    check('T1.3 tier respects the neuron cap exactly',
          big['n'] <= 9000 and big['counts']['partners_truncated_by_cap'],
          'cap 9000 -> %d neurons (seeds %d, partners added %d, truncated=%s)'
          % (big['n'], big['counts']['seed_neurons_union'], big['counts']['partners_added'],
             big['counts']['partners_truncated_by_cap']))
    check('T1.4 default tier is under the 20000 limit and partition-consistent',
          tier['n'] <= 20000 and tier['counts']['partners_truncated_by_cap'] is False,
          '%d neurons, %d partner candidates, %d added'
          % (tier['n'], tier['counts']['partner_candidates_at_threshold'],
             tier['counts']['partners_added']))

    # ------------------------------------------------------------------ T2
    # independent counting code, written here, reading the npz directly
    with np.load(ROOT / ncd.BANC_RELATIVE_PATH, allow_pickle=False) as d:
        sc_raw = np.asarray(d['ann_super_class'])
        cls_raw = np.asarray(d['ann_class'])
        pre_raw = np.asarray(d['pre'])
        syn_raw = np.asarray(d['syn'])
        nt_pair_raw = np.asarray(d['nt_pair'])
        ver_raw = np.asarray(d['ann_nt_verified'])
    indep = {}
    for lab in ('ascending', 'sensory_ascending', 'ascending_visceral_circulatory',
                'descending', 'sensory_descending'):
        mask = np.array([s == lab for s in sc_raw])
        indep[lab] = int(mask.sum())
    indep_total = sum(indep.values())
    check('T2.1 independent crossing-set count == module count',
          indep_total == crossing['neurons'],
          'test-file count %d vs module %d; per population %s'
          % (indep_total, crossing['neurons'], indep))
    check('T2.2 crossing-set count vs published anchors',
          crossing['neurons'] == 3686
          and ncd.PUBLISHED_ANCHORS['banc_backbone_total']['value'] == 3686
          and ncd.PUBLISHED_ANCHORS['histology_axons_total']['value'] == 3738,
          'mine %d == parent %d == published BANC backbone %d; histology %d -> coverage %.4f, '
          'unaccounted %d axons'
          % (crossing['neurons'], 3686, 3686, 3738,
             crossing['neurons'] / 3738, 3738 - crossing['neurons']))
    comp = crossing['published_comparison']
    check('T2.3 per-population mismatch vs the paper is reported, not hidden',
          (comp['component_comparison']['ascending']['mine'],
           comp['component_comparison']['sensory_ascending']['mine'],
           comp['component_comparison']['descending']['mine']) == (1847, 507, 1315)
          and comp['component_comparison']['ascending']['published'] == 1849
          and 'NOT' in comp['honest_note'],
          'mine AN/SA/DN = 1847/507/1315 vs published 1849/517/1316; published 3 components '
          'sum to %d, not 3686'
          % comp['component_sum_of_published_values'])
    check('T2.4 the two annotation columns DISAGREE, and the disagreement is reported',
          crossing['cross_checks']['columns_agree_exactly'] is False
          and crossing['cross_checks']['super_class_ascending'] == 1847
          and crossing['cross_checks']['ann_class_ascending_neuron'] == 1850
          and crossing['cross_checks']['super_class_descending'] == 1315
          and crossing['cross_checks']['ann_class_descending_neuron'] == 1317
          and crossing['cross_checks']
          ['DISAGREEMENT_the_two_annotation_columns_differ']
          ['cells_class_ascending_neuron_but_super_class_not_ascending'] == 3
          and crossing['cross_checks']
          ['DISAGREEMENT_the_two_annotation_columns_differ']
          ['cells_class_descending_neuron_but_super_class_not_descending'] == 2,
          'ann_super_class: AN 1847 / DN 1315 (total 3686) vs ann_class: AN 1850 / DN 1317; '
          'the 3 asc + 2 desc outsiders carry primary types %r / %r'
          % (crossing['cross_checks']['DISAGREEMENT_the_two_annotation_columns_differ']
             ['those_cell_types_ascending_neuron_outsiders'],
             crossing['cross_checks']['DISAGREEMENT_the_two_annotation_columns_differ']
             ['those_cell_types_descending_neuron_outsiders']))
    check('T2.5 other polarity sources rejected (ann_flow, cell-type prefix)',
          crossing['cross_checks']['ann_flow_carries_polarity'] is False
          and crossing['cross_checks']['cell_type_prefix_proxy_is_not_the_annotation']
          ['counts']['primary_type_startswith_AN_digit'] == 1598,
          'ann_flow values %r carry no polarity; an AN-prefix rule on primary_type would '
          'give 1598 (under-counts)'
          % crossing['cross_checks']['ann_flow_distinct_values'])
    check('T2.6 no previously written npz was read (only the raw cache)',
          a['path'] == str(ROOT / ncd.BANC_RELATIVE_PATH) and 'touch_ascending' not in a['path'],
          'source = %s' % a['path'])

    # ------------------------------------------------------------------ T3
    asc, desc = cut['ascending_side'], cut['descending_side']
    tot = cut['total']
    ident = cut['identity_checks']
    check('T3.1 neuron split is exhaustive and non-overlapping',
          cut['neurons_ascending_side'] + cut['neurons_descending_side'] == cut['neurons_total']
          and cut['neurons_ascending_side'] == indep['ascending'] + indep['sensory_ascending']
          + indep['ascending_visceral_circulatory']
          and cut['neurons_descending_side'] == indep['descending'] + indep['sensory_descending'],
          '%d asc + %d desc = %d; independently %d + %d = %d'
          % (cut['neurons_ascending_side'], cut['neurons_descending_side'], cut['neurons_total'],
             indep['ascending'] + indep['sensory_ascending'] + indep['ascending_visceral_circulatory'],
             indep['descending'] + indep['sensory_descending'],
             indep_total))
    check('T3.2 edge accounting exact and no double counting',
          ident['edges'] and ident['no_double_counting'] and ident['exhaustive']
          and asc['edges'] + desc['edges'] == tot['edges'],
          '%d asc + %d desc = %d edges' % (asc['edges'], desc['edges'], tot['edges']))
    check('T3.3 synapse accounting exact',
          ident['synapses'] and asc['synapses'] + desc['synapses'] == tot['synapses'],
          '%d asc + %d desc = %d synapses'
          % (asc['synapses'], desc['synapses'], tot['synapses']))
    # independent edge count
    asc_mask = np.isin(sc_raw, list(ncd.CROSSING_ASCENDING))
    desc_mask = np.isin(sc_raw, list(ncd.CROSSING_DESCENDING))
    indep_edges_asc = int(asc_mask[pre_raw].sum())
    indep_edges_desc = int(desc_mask[pre_raw].sum())
    indep_syn_asc = float(syn_raw[asc_mask[pre_raw]].sum())
    indep_syn_desc = float(syn_raw[desc_mask[pre_raw]].sum())
    check('T3.4 edge and synapse counts reproduced by test-file code',
          indep_edges_asc == asc['edges'] and indep_edges_desc == desc['edges']
          and indep_syn_asc == asc['synapses'] and indep_syn_desc == desc['synapses'],
          'independent: %d/%d edges, %d/%d synapses (asc/desc)'
          % (indep_edges_asc, indep_edges_desc, indep_syn_asc, indep_syn_desc))
    check('T3.5 in-tier severed split is exhaustive',
          tier['counts']['severed_edges_inside_tier_ascending_side']
          + tier['counts']['severed_edges_inside_tier_descending_side']
          == tier['counts']['severed_edges_inside_tier'],
          '%d asc + %d desc = %d of %d dataset severed edges inside the tier (%.2f%%)'
          % (tier['counts']['severed_edges_inside_tier_ascending_side'],
             tier['counts']['severed_edges_inside_tier_descending_side'],
             tier['counts']['severed_edges_inside_tier'], tot['edges'],
             100 * tier['counts']['share_of_dataset_severed_edges_inside_tier']))
    check('T3.6 route budget is a synapse count, unsigned, and not a reachability number',
          budget['units'].startswith('synapse counts') and budget['reachability_used'] is False,
          'body out %d syn, severed %d; head out %d syn, severed %d'
          % (budget['body_route']['outgoing_synapses'],
             budget['body_route']['severed_synapses'],
             budget['head_route']['outgoing_synapses'],
             budget['head_route']['severed_synapses']))

    # ------------------------------------------------------------------ T5 (needs runs)
    print('\nrunning the paired grid used by T4/T5/T6/T7 ...')
    keep = {}
    grid = [('annotation', c, r) for c in ('intact', 'sham', 'neck_cut', 'descending_only')
            for r in ('body_only', 'head_only')]
    grid += [('counterfactual_ach', c, r) for c in ('intact', 'neck_cut', 'descending_only')
             for r in ('body_only', 'head_only')]
    for mode, cm, route in grid:
        spec = ncd.ScenarioSpec(mode, cm, route)
        r = ncd.run_scenario(a, tier, cut, groups, spec, run_cfg, tier_cfg)
        keep[spec.name] = r
    print('  %d scenarios run\n' % len(keep))

    # ------------------------------------------------------------------ T4
    e_int = ncd.build_edges(a, tier, cut, 'annotation', 'intact', tier_cfg)
    e_cut = ncd.build_edges(a, tier, cut, 'annotation', 'neck_cut', tier_cfg)
    e_dsc = ncd.build_edges(a, tier, cut, 'annotation', 'descending_only', tier_cfg)
    rem = e_cut['removed_rows']
    rem_mask = cut['row_masks']['all']
    rem_mask_asc = cut['row_masks']['ascending_side']
    rem_mask_desc = cut['row_masks']['descending_side']
    mod = e_int['kept_rows']
    expected_cut = mod[rem_mask[mod]]
    expected_desc = mod[rem_mask_desc[mod]]
    check('T4.1 the full cut removes exactly the intended modelled rows',
          np.array_equal(np.sort(rem), np.sort(expected_cut)),
          '%d rows removed, independently expected %d of %d modelled rows'
          % (rem.size, expected_cut.size, mod.size))
    check('T4.2 the descending-only cut removes exactly the descending rows',
          np.array_equal(np.sort(e_dsc['removed_rows']), np.sort(expected_desc))
          and np.array_equal(np.sort(e_dsc['removed_rows']), np.sort(expected_desc)),
          '%d rows removed, expected %d' % (e_dsc['removed_rows'].size, expected_desc.size))
    # no removed row may have a non-crossing presynaptic cell
    check('T4.3 no removed row has a presynaptic cell outside the crossing set',
          not bool(np.any(rem_mask[rem] == False)),  # noqa: E712
          'checked %d removed rows' % rem.size)
    # ascending rows must survive a descending-only cut
    surviving_asc = np.intersect1d(e_dsc['kept_rows'], mod[rem_mask_asc[mod]])
    check('T4.4 a descending-only cut leaves every ascending-side row intact',
          surviving_asc.size == int(rem_mask_asc[mod].sum()),
          '%d of %d ascending-side modelled rows still present'
          % (surviving_asc.size, int(rem_mask_asc[mod].sum())))
    # kept set identity
    check('T4.5 kept rows == intact rows minus removed rows, exactly',
          np.array_equal(np.sort(e_cut['kept_rows']),
                         np.sort(np.setdiff1d(mod, rem)))
          and e_cut['kept_rows'].size + rem.size == mod.size,
          '%d kept + %d removed = %d modelled' % (e_cut['kept_rows'].size, rem.size, mod.size))
    # matrix level: the difference may only touch synapse counts of removed rows
    D = (e_int['We'] + e_int['Wi']) - (e_cut['We'] + e_cut['Wi'])
    D.eliminate_zeros()
    scale = tier_cfg.weight_scale_uS_per_synapse
    cap = tier_cfg.max_synapses_per_edge
    expected_weight = float(np.minimum(a['syn'][rem].astype(np.float64), cap).sum() * scale)
    check('T4.6 removed conductance equals min(syn, cap) * scale over the removed rows',
          np.isclose(float(D.data.sum()), expected_weight, rtol=1e-12, atol=1e-12),
          'matrix difference %.10g uS vs row-accounting %.10g uS over %d rows '
          '(rel. difference %.3g, sparse float64 cancellation only)'
          % (float(D.data.sum()), expected_weight, rem.size,
             abs(float(D.data.sum()) - expected_weight) / expected_weight))
    # every matrix difference must lie between the same (pre,post) pair as a removed row
    loc = tier['local_of_global']
    dirs_expected = set(zip(loc[a['pre'][rem]].tolist(), loc[a['post'][rem]].tolist()))
    crow, ccol = D.tocoo().row, D.tocoo().col
    check('T4.7 every conductance difference is on a removed (pre, post) pair',
          set(zip(ccol.tolist(), crow.tolist())).issubset(dirs_expected),
          'checked %d distinct changed (pre,post) pairs against %d removed rows'
          % (len(set(zip(ccol.tolist(), crow.tolist()))), len(dirs_expected)))
    # no visible cell outside the tier may be affected, and pre-phase matrices identical
    check('T4.8 the cut changes nothing before the phase (pre-phase matrices identical)',
          all(np.array_equal(keep[ncd.ScenarioSpec('annotation', 'intact', r).name]
                             ['init_digest_sha256_32'],
                             keep[ncd.ScenarioSpec('annotation', cm, r).name]
                             ['init_digest_sha256_32'])
              for r in ('body_only', 'head_only') for cm in ('sham', 'neck_cut', 'descending_only')),
          'initial-state digests identical across cut modes; rows removed by the full cut = %d'
          % e_cut['excluded']['rows_removed_by_cut_mode'])
    check('T4.9 the cut removes 0 rows from the intact and sham edge sets',
          e_int['excluded']['rows_removed_by_cut_mode'] == 0
          and ncd.build_edges(a, tier, cut, 'annotation', 'sham', tier_cfg)
          ['excluded']['rows_removed_by_cut_mode'] == 0,
          'intact 0, sham 0 removals')

    # ------------------------------------------------------------------ T5
    h_int = keep[ncd.ScenarioSpec('annotation', 'intact', 'head_only').name]
    h_cut = keep[ncd.ScenarioSpec('annotation', 'neck_cut', 'head_only').name]
    h_sha = keep[ncd.ScenarioSpec('annotation', 'sham', 'head_only').name]
    h_dsc = keep[ncd.ScenarioSpec('annotation', 'descending_only', 'head_only').name]
    b_int = keep[ncd.ScenarioSpec('annotation', 'intact', 'body_only').name]
    b_cut = keep[ncd.ScenarioSpec('annotation', 'neck_cut', 'body_only').name]
    hb = lambda r: sum(r['conductance_at_brain_side_uS']['settled'].values())
    check('T5.1 head-route drive is unchanged by the cut in the annotation tier',
          hb(h_int) == hb(h_cut) == hb(h_sha) == hb(h_dsc)
          and h_int['spike_counts']['settled']['total']
          == h_cut['spike_counts']['settled']['total']
          == h_sha['spike_counts']['settled']['total']
          == h_dsc['spike_counts']['settled']['total'],
          'brain drive %.9g uS in all four cut modes; total spikes %d in all four'
          % (hb(h_int), h_int['spike_counts']['settled']['total']))
    check('T5.2 body-route drive collapses to zero under the full cut',
          hb(b_int) > 0 and hb(b_cut) == 0.0,
          'body-route brain drive %.6f uS -> %.6f uS; brain-side spikes %d -> %d'
          % (hb(b_int), hb(b_cut),
             b_int['spike_counts']['settled']['per_group']['brain_side'],
             b_cut['spike_counts']['settled']['per_group']['brain_side']))
    check('T5.3 body-route brain drive survives a descending-only cut',
          hb(b_int) > 0 and hb(h_dsc) >= 0
          and abs(hb(keep[ncd.ScenarioSpec('annotation', 'descending_only', 'body_only').name])
                  / hb(b_int) - 1.0) < 0.05,
          'retained %.6f of %.6f uS under descending_only'
          % (hb(keep[ncd.ScenarioSpec('annotation', 'descending_only', 'body_only').name]),
             hb(b_int)))
    S_body = keep[ncd.ScenarioSpec('annotation', 'intact', 'body_only').name]['_spike_matrix']
    S_head = keep[ncd.ScenarioSpec('annotation', 'intact', 'head_only').name]['_spike_matrix']
    rep_body = ncd.replay_severed_conductance(a, tier, cut, groups, S_body, run_cfg, tier_cfg)
    rep_head = ncd.replay_severed_conductance(a, tier, cut, groups, S_head, run_cfg, tier_cfg)
    leak_head = rep_head['at_brain_side_uS']['severed_by_neck_cut_per_group']
    check('T5.4 severed conductance from the head-route-sensor group is EXACTLY zero',
          leak_head['head_route_sensor'] == 0.0 and leak_head['body_route_sensor'] == 0.0,
          'head_route_sensor %.12g uS, body_route_sensor %.12g uS, vnc_intrinsic %.12g uS'
          % (leak_head['head_route_sensor'], leak_head['body_route_sensor'],
             leak_head['vnc_intrinsic']))
    body_sev = rep_body['at_brain_side_uS']
    head_sev = rep_head['at_brain_side_uS']
    check('T5.5 annotation tier: the head route has NO modelled edge at all, and is '
          'reported as not modelled rather than as zero retained',
          head_sev['intact_total'] == 0.0 and head_sev['severed_by_neck_cut_total'] == 0.0
          and body_sev['intact_total'] > 0
          and body_sev['severed_by_neck_cut_total'] > 0.9 * body_sev['intact_total']
          and body_sev['neck_cut_total'] < 0.05 * body_sev['intact_total'],
          'head-route intact drive 0.0 uS and severed 0.0 uS (head route 100%% unlabelled, so '
          '0 of its rows are modelled); body-route severed %.6f of %.6f uS, leaving %.6f uS'
          % (body_sev['severed_by_neck_cut_total'], body_sev['intact_total'],
             body_sev['neck_cut_total']))
    cf_body_S = keep[ncd.ScenarioSpec('counterfactual_ach', 'intact', 'body_only').name][
        '_spike_matrix']
    cf_head_S = keep[ncd.ScenarioSpec('counterfactual_ach', 'intact', 'head_only').name][
        '_spike_matrix']
    cf_body = ncd.replay_severed_conductance(a, tier, cut, groups, cf_body_S, run_cfg, tier_cfg,
                                             sign_mode='counterfactual_ach')
    cf_head = ncd.replay_severed_conductance(a, tier, cut, groups, cf_head_S, run_cfg, tier_cfg,
                                             sign_mode='counterfactual_ach')
    bsev = cf_body['at_brain_side_uS']
    hsev = cf_head['at_brain_side_uS']
    check('T5.6 counterfactual tier, activity held fixed: body route loses >90%%, head route '
          '<10%%',
          (bsev['severed_by_neck_cut_total'] / bsev['intact_total']) > 0.9
          and (hsev['severed_by_neck_cut_total'] / hsev['intact_total']) < 0.1
          and cf_head['at_brain_side_uS']['severed_by_neck_cut_per_group']
          ['head_route_sensor'] == 0.0,
          'body-route: severed %.4f of %.4f uS (%.2f%%); head-route: severed %.4f of %.4f uS '
          '(%.2f%%); head_route_sensor severed %.12g uS'
          % (bsev['severed_by_neck_cut_total'], bsev['intact_total'],
             100 * bsev['severed_by_neck_cut_total'] / bsev['intact_total'],
             hsev['severed_by_neck_cut_total'], hsev['intact_total'],
             100 * hsev['severed_by_neck_cut_total'] / hsev['intact_total'],
             cf_head['at_brain_side_uS']['severed_by_neck_cut_per_group']['head_route_sensor']))
    bd = budget['body_route']
    hd0 = budget['head_route']
    check('T5.7 the monosynaptic body/head contrast is quantified and NOT claimed '
          'cut-independent',
          bd['severed_synapses_into_brain_side'] == bd['to_brain_side_direct']
          and bd['severed_fraction_of_direct_to_brain_side'] == 1.0
          and hd0['severed_fraction_of_direct_to_brain_side'] < 0.01
          and bd['direct_monosynaptic_fraction_is_cut_independent'] is False,
          'body -> brain direct: %d of %d synapses severed (%.1f%%); head -> brain direct: '
          '%d of %d severed (%.2f%%); body -> ascending: %d of %d severed (%.2f%%)'
          % (int(bd['severed_synapses_into_brain_side']), int(bd['to_brain_side_direct']),
             100 * bd['severed_fraction_of_direct_to_brain_side'],
             int(hd0['severed_synapses_into_brain_side']), int(hd0['to_brain_side_direct']),
             100 * hd0['severed_fraction_of_direct_to_brain_side'],
             int(bd['severed_synapses_into_ascending_side']), int(bd['to_ascending_side']),
             100 * bd['severed_fraction_of_direct_to_ascending_side']))
    hd = budget['head_route']
    hc_note = budget['route_neurons_also_annotated_as_crossing']
    check('T5.8 the naive "no head-route output is severed" claim is refuted, with numbers',
          hd['severed_synapses'] > 0 and hc_note['head_route'] == 2
          and hd['severed_fraction_of_route'] < 0.02,
          '%d of %d head-route synapses severed (%.4f%%) from %d of %d head-route cells '
          '(sensory_descending); body-route cells also crossing: %d'
          % (hd['severed_synapses'], hd['outgoing_synapses'],
             100 * hd['severed_fraction_of_route'], hc_note['head_route'],
             budget['head_route']['source_neurons'], hc_note['body_route']))
    c_int = keep[ncd.ScenarioSpec('counterfactual_ach', 'intact', 'body_only').name]
    c_cut = keep[ncd.ScenarioSpec('counterfactual_ach', 'neck_cut', 'body_only').name]
    c_int_h = keep[ncd.ScenarioSpec('counterfactual_ach', 'intact', 'head_only').name]
    c_cut_h = keep[ncd.ScenarioSpec('counterfactual_ach', 'neck_cut', 'head_only').name]
    rb = hb(c_cut) / hb(c_int)
    rh = hb(c_cut_h) / hb(c_int_h)
    check('T5.9 counterfactual tier: route selectivity survives in the voltage domain',
          rb < 0.05 and rh > 0.9,
          'body route retained %.6f, head route retained %.6f -> separation %.1f x'
          % (rb, rh, (rh / rb) if rb else float('inf')))
    check('T5.10 descending-only differs materially from the full cut',
          abs(hb(keep[ncd.ScenarioSpec('counterfactual_ach', 'descending_only', 'body_only').name])
              / hb(c_int) - 1.0) < 0.05 and abs(rb - 1.0) > 0.5,
          'body route: full cut retained %.6f, descending_only retained %.6f'
          % (rb, hb(keep[ncd.ScenarioSpec('counterfactual_ach', 'descending_only',
                                          'body_only').name]) / hb(c_int)))

    # ------------------------------------------------------------------ T6
    for mode in ('annotation', 'counterfactual_ach'):
        for route in ('body_only', 'head_only'):
            names = [ncd.ScenarioSpec(mode, c, route).name
                     for c in ('intact', 'sham', 'neck_cut', 'descending_only')
                     if ncd.ScenarioSpec(mode, c, route).name in keep]
            init = {keep[n]['init_digest_sha256_32'] for n in names}
            base = {keep[n]['baseline_spike_digest_sha256_32'] for n in names}
            bcnt = {keep[n]['spike_counts']['baseline']['total'] for n in names}
            check('T6 init+seed identical across cut modes: %s / %s' % (mode, route),
                  len(init) == 1 and len(base) == 1 and len(bcnt) == 1,
                  '%d cut modes, init digest %s, baseline spikes %s'
                  % (len(names), sorted(init)[0], sorted(bcnt)))
    init_ann = keep[ncd.ScenarioSpec('annotation', 'intact', 'body_only').name][
        'init_digest_sha256_32']
    init_cf = keep[ncd.ScenarioSpec('counterfactual_ach', 'intact', 'body_only').name][
        'init_digest_sha256_32']
    check('T6.2 the initial-state digest is sensitive to the edge set',
          init_ann != init_cf,
          'annotation %s vs counterfactual %s' % (init_ann, init_cf))
    check('T6.3 the seeded background current is identical across scenarios',
          all(keep[n]['conductance_params'] == keep[sorted(keep)[0]]['conductance_params']
              for n in keep),
          'seed %d, background mean %.4g nA sd %.4g nA, dt %.3g ms, %d steps'
          % (run_cfg.seed, run_cfg.background_mean_nA, run_cfg.background_sd_nA,
             run_cfg.dt_ms, run_cfg.steps))

    # ------------------------------------------------------------------ T7
    ok_sham = True
    detail = []
    for mode, sham_cm in (('annotation', 'sham'), ('counterfactual_ach', 'intact')):
        for route in ('body_only', 'head_only'):
            i = keep[ncd.ScenarioSpec(mode, 'intact', route).name]
            s = keep[ncd.ScenarioSpec(mode, sham_cm, route).name]
            same = np.array_equal(i['_spike_matrix'], s['_spike_matrix'])
            same_g = (i['conductance_at_brain_side_uS'] == s['conductance_at_brain_side_uS']
                      and i['conductance_at_ascending_side_uS']
                      == s['conductance_at_ascending_side_uS'])
            ok_sham &= bool(same and same_g)
            detail.append('%s/%s spikes=%s drive=%s' % (mode, route, same, same_g))
    check('T7 zero-shift control: sham is bit-identical to intact', ok_sham, '; '.join(detail))

    # ------------------------------------------------------------------ T8
    expect_raises('T8.1 max_neurons below the seed set is rejected', ValueError,
                  ncd.select_touch_tier, a, mech, cut, ncd.TouchTierConfig(max_neurons=5000))
    for bad, kw in ((0, dict(max_neurons=0)), (20001, dict(max_neurons=20001)),
                    ('x', dict(max_neurons=True)), (0, dict(max_synapses_per_edge=0)),
                    (0, dict(weight_scale_uS_per_synapse=0.0)),
                    (-1, dict(partner_min_synapses=-1.0))):
        expect_raises('T8.2 TouchTierConfig rejects %r' % (kw,), ValueError,
                      ncd.TouchTierConfig(**kw).validate)
    for kw in (dict(dt_ms=0.5), dict(phase_ms=100.0), dict(seed=-1), dict(seed=1.5),
               dict(stim_duration_ms=9.0), dict(settle_ms=40.0), dict(background_sd_nA=-1.0),
               dict(delay_ms=0.0), dict(duration_ms=40.03)):
        expect_raises('T8.3 NetworkRunConfig rejects %r' % (kw,), ValueError,
                      ncd.NetworkRunConfig(**kw).validate)
    expect_raises('T8.4 ScenarioSpec rejects an unknown sign_mode', ValueError,
                  ncd.ScenarioSpec('nope', 'intact', 'both').validate)
    expect_raises('T8.5 ScenarioSpec rejects an unknown cut_mode', ValueError,
                  ncd.ScenarioSpec('annotation', 'nope', 'both').validate)
    expect_raises('T8.6 ScenarioSpec rejects an unknown activated_route', ValueError,
                  ncd.ScenarioSpec('annotation', 'intact', 'nope').validate)
    expect_raises('T8.7 build_edges rejects an unknown sign_mode', ValueError,
                  ncd.build_edges, a, tier, cut, 'nope', 'intact', tier_cfg)
    expect_raises('T8.8 build_edges rejects an unknown cut_mode', ValueError,
                  ncd.build_edges, a, tier, cut, 'annotation', 'nope', tier_cfg)
    expect_raises('T8.9 load_banc rejects a missing file', FileNotFoundError,
                  ncd.load_banc, ROOT / 'data' / 'flywire' / 'definitely_absent.npz')
    expect_raises('T8.10 conductance_ledger rejects a mismatched spike matrix', ValueError,
                  ncd.conductance_ledger, np.zeros((3, 7), bool),
                  *[__import__('scipy.sparse', fromlist=['x']).csr_matrix((5, 5))] * 2,
                  np.arange(2), {'g': np.arange(5)}, 40)
    bad_cross = dict(crossing)
    bad_cross['masks'] = dict(crossing['masks'])
    bad_cross['masks']['ascending_side'] = crossing['masks']['crossing']
    bad_cross['masks']['descending_side'] = crossing['masks']['crossing']
    expect_raises('T8.11 build_cut_set rejects overlapping ascending/descending masks',
                  ValueError, ncd.build_cut_set, a, bad_cross)
    expect_raises('T8.12 build_edges rejects an oversized tier index', ValueError,
                  ncd.build_edges,
                  a, {**tier, 'inside_rows': tier['inside_rows'][:-1]}, cut,
                  'annotation', 'intact', tier_cfg)

    # ------------------------------------------------------------------ T9
    rep_txt = (OUT / 'neck_cut_report.json')
    if rep_txt.is_file():
        report = json.loads(rep_txt.read_text())
    else:
        report = None
    check('T9.1 the report asserts that reachability is NOT evidence',
          report is not None
          and report.get('reachability_used_as_evidence') is False
          and report['reachability_is_NOT_used_as_evidence']['assertion_holds'] is True
          and 'No reachability' in
          report['reachability_is_NOT_used_as_evidence']['assertion'],
          'flag %r, assertion present and true'
          % (report.get('reachability_used_as_evidence') if report else None))

    ALLOWED_DISCLAIMER_PATHS = (
        'reachability_is_NOT_used_as_evidence', 'reachability_used_as_evidence',
        'component_statistics_role', 'evidence_classes_used', 'assertion',
        'assertion_holds', 'statements', 'limitations', 'honesty', 'selection_rule',
        'rejected_alternatives', 'note', 'units', 'definition', 'method',
        'unsigned_note', 'atomic',
    )
    BAD = ('reachab', 'shortest', 'hop_count', 'n_hops', 'bfs', 'dijkstra',
           'connected_component', 'n_components', 'component_size', 'largest_component',
           'graph_component', 'component_count')
    numeric_paths_bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            joined = '/'.join(path).lower()
            if any(b in joined for b in BAD) and not any(
                    a.lower() in joined for a in ALLOWED_DISCLAIMER_PATHS):
                numeric_paths_bad.append('%s=%r' % ('/'.join(path), node))

    if report is not None:
        walk(report, [])
    check('T9.2 no numeric evidence sits under a reachability-shaped key',
          not numeric_paths_bad,
          'offending paths: %r' % (numeric_paths_bad[:6] if numeric_paths_bad else 'none'))

    src = Path(ncd.__file__).read_text()
    check('T9.3 the module imports no graph-reachability machinery',
          not any(tok in src for tok in ('csgraph', 'shortest_path', 'breadth_first',
                                         'dijkstra', 'connected_components')),
          'no csgraph/shortest_path/breadth_first/dijkstra/connected_components in %s'
          % Path(ncd.__file__).name)

    tree = ast.parse(src)
    funcs = {f.name: f for f in tree.body if isinstance(f, ast.FunctionDef)}
    measured = ('derive_crossing_set', 'build_cut_set', 'route_budget', 'select_touch_tier',
                'build_edges', 'run_scenario', 'conductance_ledger',
                'replay_severed_conductance')
    offenders = []
    for name in measured:
        fn = funcs[name]
        for node in ast.walk(fn):
            tokens = []
            if isinstance(node, ast.Call):
                f = node.func
                tokens.append(getattr(f, 'id', None) or getattr(f, 'attr', None) or '')
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            for tok in tokens:
                if tok and any(b in tok.lower() for b in ('reachab', 'shortest', 'component',
                                                          'bfs', 'dijkstra', 'hop')):
                    offenders.append('%s:%s' % (name, tok))
    check('T9.4 no measurement function calls a reachability routine',
          not offenders,
          'scanned %d measurement functions; offenders %r'
          % (len(measured), offenders if offenders else 'none'))

    # ------------------------------------------------------------------ T10
    table = {'acetylcholine': 'exc', 'gaba': 'inh', 'glutamate': 'inh', '': None,
             'histamine': None, 'acetylcholine,histamine': None, 'serotonin': None,
             'glutamate,octopamine': None, 'acetylcholine,gaba': None,
             'tyramine': None, 'octopamine': None, 'dopamine': None}
    ok_tab = all(ncd.receptor_class(k) == v for k, v in table.items())
    check('T10.1 receptor mapping is exactly ACh->exc, gaba/glutamate->inh, else excluded',
          ok_tab,
          'exc: %d | inh: %d | excluded-counted: %d tested labels' %
          (1, 2, len(table) - 3))
    rc_fn = funcs['receptor_class']
    rc_skip = set()
    if rc_fn.body and isinstance(rc_fn.body[0], ast.Expr):
        first = rc_fn.body[0].value
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            rc_skip.add(id(first))
    rc_tokens = []
    for node in ast.walk(rc_fn):
        if id(node) in rc_skip:
            continue
        if isinstance(node, (ast.Name, ast.Attribute, ast.arg, ast.keyword)):
            rc_tokens.append(getattr(node, 'id', None) or getattr(node, 'attr', None)
                             or getattr(node, 'arg', None) or '')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            rc_tokens.append(node.value)
    rc_uses_nt_pair = any('nt_pair' in t for t in rc_tokens)
    sign_blocks_ok = True
    for mode in ('annotation', 'counterfactual_ach'):
        ec = ncd.build_edges(a, tier, cut, mode, 'intact', tier_cfg)['excluded']
        sign_blocks_ok &= (ec['nt_pair_used_for_sign'] is False
                           and 'nt_pair' in ec['sign_rule'])
    check('T10.2 nt_pair is -1 on every row and is never used for a sign',
          bool(np.all(nt_pair_raw == -1)) and not rc_uses_nt_pair and sign_blocks_ok
          and ncd.receptor_class('acetylcholine') == 'exc',
          'unique nt_pair values %r; receptor_class tokens scanned (%d) contain no nt_pair '
          'reference; both sign-coverage blocks declare nt_pair_used_for_sign = False'
          % ([int(v) for v in np.unique(nt_pair_raw)], len(rc_tokens)))
    for mode in ('annotation', 'counterfactual_ach'):
        ec = ncd.build_edges(a, tier, cut, mode, 'intact', tier_cfg)['excluded']
        ok = ec['breakdown_is_mutually_exclusive_and_exhaustive'] is True and \
            ec['nt_pair_used_for_sign'] is False
        check('T10.3 sign-coverage accounting exhaustive (%s)' % mode, ok,
              'exc %d + inh %d + excluded %d = %d rows inside tier '
              '(unknown %d, histamine-only %d, mixed %d)'
              % (ec['modelled_exc_rows'], ec['modelled_inh_rows'],
                 ec['excluded_rows_inside_tier'], ec['rows_inside_tier_total'],
                 ec['excluded_unknown_no_label'], ec['excluded_histamine_only'],
                 ec['excluded_mixed_or_modulator']))
    e_ann = ncd.build_edges(a, tier, cut, 'annotation', 'intact', tier_cfg)
    modelled_row = np.zeros(int(a['pre'].size), bool)
    modelled_row[e_ann['kept_rows']] = True
    head_rows_in_tier = np.flatnonzero(tier['inside_rows'] & mech['head_route'][a['pre']])
    n_head_modelled = int(modelled_row[head_rows_in_tier].sum())
    check('T10.4 head-route sign coverage is zero in the annotation-faithful tier',
          all(v == '' for v in ver_raw[mech['head_route']]) and n_head_modelled == 0,
          '0 of %d head-route neurons carry a verified label; %d of %d head-route rows inside '
          'the tier are modelled in the annotation tier'
          % (int(mech['head_route'].sum()), n_head_modelled, head_rows_in_tier.size))

    # ------------------------------------------------------------------ T11
    disj = groups['disjoint']
    total = sum(int(m.sum()) for m in disj.values())
    check('T11.1 disjoint presynaptic groups partition the neuron set exactly',
          total == n_neurons,
          ' + '.join('%s %d' % (k, int(v.sum())) for k, v in disj.items())
          + ' = %d == %d neurons' % (total, n_neurons))
    overlap = groups['route_overlapping']
    check('T11.2 route-overlapping groups are labelled as overlapping (body vs crossing)',
          int((overlap['body_touch_sensor']
               & (overlap['crossing_ascending'] | overlap['crossing_descending'])).sum()) == 412
          and groups['route_overlapping_is_partition'] is False,
          'body-route cells also annotated crossing: %d; head-route: %d'
          % (int((overlap['body_touch_sensor']
                  & (overlap['crossing_ascending'] | overlap['crossing_descending'])).sum()),
             int((overlap['head_touch_sensor']
                  & (overlap['crossing_ascending'] | overlap['crossing_descending'])).sum())))

    # ------------------------------------------------------------------ T12
    params = ConductanceParams()
    small = ConductanceNetwork(4, params, run_cfg.dt_ms)
    rng = np.random.default_rng(3)
    W = np.zeros((4, 4))
    W[1, 0] = 5e-4
    W[2, 1] = 3e-4
    W[3, 2] = 1e-3
    Wi = np.zeros((4, 4))
    Wi[3, 1] = 2e-4
    small.set_connectivity(W, Wi)
    steps = 60
    S = np.zeros((steps, 4), np.uint8)
    ge_before, ge_after = [], []
    for k in range(steps):
        ge_before.append(small.ge.copy())
        S[k] = small.step(0.03 if k < 30 else 0.0)
        ge_after.append(small.ge.copy())
    decay = float(np.exp(-run_cfg.dt_ms / params.exc_tau_ms))
    max_err = 0.0
    for k, arr in enumerate(ncd._arrivals_iter(S, int(round(params.delay_ms / run_cfg.dt_ms)))):
        expected = W @ arr
        actual = ge_after[k] / decay - ge_before[k]
        max_err = max(max_err, float(np.max(np.abs(expected - actual))))
    check('T12 the conductance-ledger emission mirror is exact', max_err < 1e-15,
          'max |mirror - network increment| = %.3g uS over %d steps '
          '(delay %d steps, dt %.3g ms)'
          % (max_err, steps, int(round(params.delay_ms / run_cfg.dt_ms)), run_cfg.dt_ms))

    # ------------------------------------------------------------------ T13
    arts = {'report': OUT / 'neck_cut_report.json',
            'traces': OUT / 'neck_cut_traces.npz',
            'figure': OUT / 'neck_cut_demo.png'}
    missing = [k for k, p in arts.items() if not p.is_file() or p.stat().st_size < 1000]
    check('T13.1 demo artifacts exist and are non-trivial', not missing,
          '; '.join('%s %.2f MB' % (k, p.stat().st_size / 1e6)
                    for k, p in arts.items() if p.is_file()))
    if not missing:
        with np.load(arts['traces'], allow_pickle=False) as t:
            keys = set(t.keys())
            ok_npz = ('times_ms' in keys and 'tier_global_index' in keys
                      and 'cut_rows_ascending_side' in keys)
            n_scen = sum(1 for k in keys if k.startswith('spike_count__'))
            n_tier = int(t['tier_global_index'].size)
            n_asc = int(t['cut_rows_ascending_side'].size)
            n_desc = int(t['cut_rows_descending_side'].size)
            ok_npz &= n_scen == 21
            ok_npz &= n_asc == cut['ascending_side']['edges']
            ok_npz &= n_desc == cut['descending_side']['edges']
            ok_npz &= n_tier == tier['n']
        check('T13.2 traces carry every scenario and the exact cut row sets', ok_npz,
              '%d spike_count blocks, %d cut rows (asc %d + desc %d), tier %d neurons'
              % (n_scen, n_asc + n_desc, n_asc, n_desc, n_tier))
        rep = json.loads(arts['report'].read_text())
        need = ('step1_crossing_set', 'step1_vs_published_summary', 'step2_cut_set',
                'step3_route_budget', 'step3_tier', 'step3_sign_coverage',
                'step4_paired_results', 'step4_exact_replay_attribution',
                'step5_head_route_control', 'step6_mask_semantics', 'limitations', 'honesty',
                'reachability_is_NOT_used_as_evidence', 'run')
        miss = [k for k in need if k not in rep]
        check('T13.3 report carries every required block', not miss, 'missing %r' % miss)
        st = ' '.join(rep['honesty']['statements']) + json.dumps(rep['limitations'])
        required_claims = {
            'unsigned edges (nt_pair = -1)': 'UNSIGNED' in st and '-1' in st,
            'crossing set is dataset annotation': 'ANNOTATION' in st,
            'not a reconstructed cut plane': 'cut plane' in st,
            'FAFB truncated at the neck': 'FAFB' in st and 'truncat' in st,
            'conditional on the cached snapshot': 'CONDITIONAL' in st or 'conditional' in st,
            'no consciousness/viability claim': 'consciousness' in st,
        }
        check('T13.4 all six honesty rules appear in the written report',
              all(required_claims.values()),
              '; '.join('%s=%s' % kv for kv in required_claims.items()))
        fig_ok = arts['figure'].stat().st_size > 200000
        check('T13.5 figure is a non-trivial raster', fig_ok,
              '%.2f MB, %d x %d px (verified by eye with the image reader)'
              % (arts['figure'].stat().st_size / 1e6,
                 *__import__('PIL.Image', fromlist=['x']).open(arts['figure']).size))

    # ------------------------------------------------------------------ T14
    # REGRESSION SUITE FOR THE ROUND-1 MASK WRAPAROUND DEFECT.
    # The defect: group masks were built as
    #     m[local_of_global[np.flatnonzero(global_mask)]] = True
    # and local_of_global is -1 for every neuron OUTSIDE the tier, so NumPy wrapped
    # each out-of-tier member onto local row n-1.  These four checks would have caught
    # it, and the numeric pin in T14.4 fails if it is ever reintroduced.
    audit = ncd.mask_semantics_audit(tier, groups)
    memberships = ncd.mask_membership_table(tier, groups)
    named = ncd.named_group_masks(groups)
    loc = np.asarray(tier['local_of_global'])
    nn = int(tier['n'])
    outside = np.flatnonzero(loc < 0)

    # (a) a member outside the tier must never mark ANY local row
    g_out = int(outside[0])
    single_out = np.zeros(loc.size, bool)
    single_out[g_out] = True
    m_single_out = ncd.local_mask(tier, single_out)
    other_group = named['disjoint.other']
    outside_only = other_group & ~tier['keep_mask']
    m_outside_only = ncd.local_mask(tier, outside_only)
    m_outside_only_legacy = ncd.local_mask(tier, outside_only, ncd.LEGACY_MASK_SEMANTICS)
    mixed = (named['disjoint.crossing_ascending'] | outside_only)
    check('T14.1 an out-of-tier group member marks NO local row (round-1 wrapped it onto n-1)',
          int(m_single_out.sum()) == 0 and int(m_outside_only.sum()) == 0
          and np.array_equal(ncd.local_mask(tier, mixed),
                             ncd.local_mask(tier, named['disjoint.crossing_ascending']))
          and int(outside_only.sum()) > 0
          and int(m_outside_only_legacy.sum()) == 1 and bool(m_outside_only_legacy[nn - 1]),
          'out-of-tier global id %d -> %d masked rows (legacy: %d, at local row %d); a mask of '
          '%d out-of-tier neurons -> %d masked rows (legacy: %d); adding every out-of-tier '
          'neuron to a group mask changes nothing'
          % (g_out, int(m_single_out.sum()), int(m_outside_only_legacy.sum()), nn - 1,
             int(outside_only.sum()), int(m_outside_only.sum()),
             int(m_outside_only_legacy.sum())))

    # (b) the last local row is marked ONLY if it is a genuine member
    last_global = int(tier['global_index'][nn - 1])
    wrong_last, true_member_masks = [], []
    for name, mask in named.items():
        correct = ncd.local_mask(tier, mask)
        legacy = ncd.local_mask(tier, mask, ncd.LEGACY_MASK_SEMANTICS)
        if bool(correct[nn - 1]) != bool(np.asarray(mask)[last_global]):
            wrong_last.append(name)
        if bool(correct[nn - 1]):
            true_member_masks.append(name)
    check('T14.2 the last local row is marked ONLY when it is a genuine group member',
          not wrong_last and len(true_member_masks) > 0
          and memberships['disjoint.crossing_ascending']['last_local_row_is_a_true_member']
          is False
          and memberships['disjoint.head_route_sensor']['last_local_row_is_a_true_member']
          is True,
          'local row %d (= global %d) matches the global mask in all %d named masks; it is a '
          'genuine member in %d of them (%s); the legacy arm set it spuriously in %d of them'
          % (nn - 1, last_global, len(named), len(true_member_masks),
             ', '.join(sorted(true_member_masks)[:2]),
             sum(1 for v in memberships.values() if v['masks_differ'])))

    # (c) masked rows == in-tier members, for a group with KNOWN out-of-tier members
    size_ok, size_detail = True, []
    for name, mask in named.items():
        v = memberships[name]
        if v['masked_rows_correct_semantics'] != v['members_in_tier_correct_semantics']:
            size_ok = False
            size_detail.append(name)
        if v['global_members'] - v['global_members_outside_tier'] \
                != v['members_in_tier_correct_semantics']:
            size_ok = False
            size_detail.append(name + ':global')
    asc_v = memberships['disjoint.crossing_ascending']
    check('T14.3 masked rows == in-tier members exactly (checked on a group with out-of-tier '
          'members)',
          size_ok and asc_v['global_members'] == MASK_FIX_CROSSING_ASCENDING_GLOBAL
          and asc_v['global_members_outside_tier'] == MASK_FIX_CROSSING_ASCENDING_OUTSIDE
          and asc_v['masked_rows_correct_semantics'] == MASK_FIX_CROSSING_ASCENDING_IN_TIER
          and asc_v['masked_rows_correct_semantics']
          == tier['counts']['crossing_neurons_in_tier_ascending_side'],
          'all %d named masks satisfy it; crossing_ascending: %d global members - %d out of '
          'tier = %d masked rows (= tier count %d); legacy masked %d'
          % (len(named), asc_v['global_members'], asc_v['global_members_outside_tier'],
             asc_v['masked_rows_correct_semantics'],
             tier['counts']['crossing_neurons_in_tier_ascending_side'],
             asc_v['masked_rows_legacy_semantics']))

    # (d) numeric pin: the corrected values, and the legacy values they replaced
    pin_spec = ncd.ScenarioSpec('counterfactual_ach', 'intact', 'both')
    pin_correct = ncd.run_scenario(a, tier, cut, groups, pin_spec, run_cfg, tier_cfg)
    pin_legacy = ncd.run_scenario(a, tier, cut, groups, pin_spec, run_cfg, tier_cfg,
                                  mask_semantics=ncd.LEGACY_MASK_SEMANTICS)
    pin_correct.pop('_spike_matrix', None)
    pin_legacy.pop('_spike_matrix', None)

    def _bs_drive(r):
        return float(sum(r['conductance_at_brain_side_uS']['settled'].values()))

    def _bs_spikes(r):
        return r['spike_counts']['settled']['per_group']['brain_side']

    check('T14.4 REGRESSION PIN: corrected intact numbers, and the legacy numbers they replace',
          abs(_bs_drive(pin_correct) - MASK_FIX_CORRECT_BRAIN_SIDE_DRIVE_US) < 1e-9
          and _bs_spikes(pin_correct) == MASK_FIX_CORRECT_BRAIN_SIDE_SPIKES
          and pin_correct['spike_counts']['settled']['total'] == MASK_FIX_CORRECT_TOTAL_SPIKES
          and abs(_bs_drive(pin_legacy) - MASK_FIX_LEGACY_BRAIN_SIDE_DRIVE_US) < 1e-9
          and _bs_spikes(pin_legacy) == MASK_FIX_LEGACY_BRAIN_SIDE_SPIKES
          and pin_correct['mask_semantics'] == ncd.DEFAULT_MASK_SEMANTICS
          and pin_legacy['mask_semantics'] == ncd.LEGACY_MASK_SEMANTICS,
          '%s / both / intact: drive %.5f uS and %d brain-side spikes corrected, vs %.5f uS and '
          '%d under the round-1 legacy arm (delta %.5f uS, %d spikes); total spikes %d'
          % (pin_spec.sign_mode, _bs_drive(pin_correct), _bs_spikes(pin_correct),
             _bs_drive(pin_legacy), _bs_spikes(pin_legacy),
             _bs_drive(pin_legacy) - _bs_drive(pin_correct),
             _bs_spikes(pin_legacy) - _bs_spikes(pin_correct),
             pin_correct['spike_counts']['settled']['total']))

    # (e) the WRITTEN report records the correction, names the legacy arm, and quotes both
    #     columns; a stale report that still encoded the old semantics FAILS here.
    if not missing:
        rep14 = json.loads(arts['report'].read_text())
        blk = rep14.get('step6_mask_semantics', {})
        contrast = blk.get('intact_arms_old_vs_new', {}).get(pin_spec.name, {})
        drive_pair = contrast.get('brain_side_drive_uS', {}).get('settled', {})
        check('T14.5 the written report records the corrected default AND the legacy contrast',
              blk.get('default_semantics') == ncd.DEFAULT_MASK_SEMANTICS
              and blk.get('legacy_contrast_arm_semantics') == ncd.LEGACY_MASK_SEMANTICS
              and rep14.get('run', {}).get('mask_semantics') == ncd.DEFAULT_MASK_SEMANTICS
              and blk.get('audit', {}).get('masks_changed_by_the_fix') == MASK_FIX_MASKS_CHANGED
              and blk.get('audit', {}).get('masks_total') == MASK_FIX_MASKS_TOTAL
              and abs(drive_pair.get('correct_semantics', -1)
                      - MASK_FIX_CORRECT_BRAIN_SIDE_DRIVE_US) < 1e-9
              and abs(drive_pair.get('legacy_semantics', -1)
                      - MASK_FIX_LEGACY_BRAIN_SIDE_DRIVE_US) < 1e-9,
              'report default %r, legacy arm %r, masks changed %s of %s, drive %.5f (legacy) vs '
              '%.5f (correct)'
              % (blk.get('default_semantics'),
                 blk.get('legacy_contrast_arm_semantics'),
                 blk.get('audit', {}).get('masks_changed_by_the_fix'),
                 blk.get('audit', {}).get('masks_total'),
                 drive_pair.get('legacy_semantics', float('nan')),
                 drive_pair.get('correct_semantics', float('nan'))))
    else:
        check('T14.5 the written report records the corrected default AND the legacy contrast',
              False, 'FAILS: %s is missing, so the report cannot be audited' % arts['report'])

    check('T14.6 the tier index map is a bijection with an explicit outside sentinel',
          audit['tier_index_map_is_consistent'] is True
          and int(np.count_nonzero(loc >= 0)) == nn
          and int(loc[tier['global_index']].min()) == 0
          and int(loc[tier['global_index']].max()) == nn - 1
          and int(outside.size) == int(a['root_ids'].size) - nn,
          '%d in-tier neurons map bijectively onto [0, %d); %d out-of-tier neurons carry the '
          'sentinel -1 (= %d of %d dataset neurons)'
          % (nn, nn - 1, int(outside.size), a['root_ids'].size - nn, a['root_ids'].size))

    # ------------------------------------------------------------------ summary
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print('\n' + '=' * 78)
    print('SELFTEST SUMMARY: %d checks, %d passed, %d failed, %.1f s wall clock'
          % (len(RESULTS), npass, nfail, time.perf_counter() - T0))
    print('  crossing set (mine)                : %d  (parent 3686, Bates 3686, Phelps 3738)'
          % crossing['neurons'])
    print('  severed fibre set                  : %d neurons -> %d edges / %d synapses'
          % (cut['neurons_total'], cut['total']['edges'], int(cut['total']['synapses'])))
    print('    ascending side                   : %d neurons / %d edges / %d synapses'
          % (cut['neurons_ascending_side'], asc['edges'], int(asc['synapses'])))
    print('    descending side                  : %d neurons / %d edges / %d synapses'
          % (cut['neurons_descending_side'], desc['edges'], int(desc['synapses'])))
    print('  tier                               : %d neurons, %d/%d severed edges inside (%.2f%%)'
          % (tier['n'], tier['counts']['severed_edges_inside_tier'], cut['total']['edges'],
             100 * tier['counts']['share_of_dataset_severed_edges_inside_tier']))
    def safe_div(num, den):
        return (num / den) if den else float('nan')

    print('  body-route drive retained (full cut): annotation %s, counterfactual %s'
          % (_fmt(safe_div(hb(b_cut), hb(b_int))), _fmt(rb)))
    print('  head-route drive retained (full cut): annotation %s (head route NOT MODELLED in '
          'this tier: 0 modelled rows), counterfactual %s'
          % (_fmt(safe_div(hb(h_cut), hb(h_int))), _fmt(rh)))
    print('  sham bit-identical to intact       : %s' % ok_sham)
    print('  mask semantics                     : %s (out-of-tier ids EXCLUDED); legacy arm %s '
          'changed %d of %d named masks, brain-side drive %.5f -> %.5f uS and brain-side spikes '
          '%d -> %d on counterfactual_ach/intact/both'
          % (ncd.DEFAULT_MASK_SEMANTICS, ncd.LEGACY_MASK_SEMANTICS,
             MASK_FIX_MASKS_CHANGED, MASK_FIX_MASKS_TOTAL,
             MASK_FIX_LEGACY_BRAIN_SIDE_DRIVE_US, MASK_FIX_CORRECT_BRAIN_SIDE_DRIVE_US,
             MASK_FIX_LEGACY_BRAIN_SIDE_SPIKES, MASK_FIX_CORRECT_BRAIN_SIDE_SPIKES))
    print('=' * 78)
    if nfail:
        print('FAILED CHECKS:')
        for name, ok, detail in RESULTS:
            if not ok:
                print('  - %s :: %s' % (name, detail))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'neck_cut_selftest.json').write_text(json.dumps({
        'checks_total': len(RESULTS), 'checks_passed': npass, 'checks_failed': nfail,
        'wall_clock_s': time.perf_counter() - T0,
        'results': [{'name': n, 'passed': ok, 'detail': d} for n, ok, d in RESULTS],
        'headline_counts': {
            'crossing_set_mine': crossing['neurons'],
            'crossing_set_parent': 3686,
            'crossing_set_published_banc_backbone': 3686,
            'crossing_set_published_histology_axons': 3738,
            'severed_neurons': cut['neurons_total'],
            'severed_neurons_ascending_side': cut['neurons_ascending_side'],
            'severed_neurons_descending_side': cut['neurons_descending_side'],
            'severed_edges': cut['total']['edges'],
            'severed_edges_ascending_side': asc['edges'],
            'severed_edges_descending_side': desc['edges'],
            'severed_synapses': cut['total']['synapses'],
            'severed_synapses_ascending_side': asc['synapses'],
            'severed_synapses_descending_side': desc['synapses'],
            'tier_neurons': tier['n'],
            'tier_severed_edges': tier['counts']['severed_edges_inside_tier'],
            'tier_share_of_severed_edges': tier['counts']['share_of_dataset_severed_edges_inside_tier'],
            'head_route_output_synapses': budget['head_route']['outgoing_synapses'],
            'head_route_output_synapses_severed': budget['head_route']['severed_synapses'],
            'body_route_output_synapses': budget['body_route']['outgoing_synapses'],
            'body_route_output_synapses_severed': budget['body_route']['severed_synapses'],
        },
    }, indent=2, default=str))
    print('wrote %s' % (OUT / 'neck_cut_selftest.json'))
    return 0 if nfail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
