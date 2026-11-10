"""Independent verification of the fine_phase run.

This script does NOT import run.py. It re-reads the sealed inputs and the produced artifacts and
recomputes every published number from the saved native records:

  V1  frozen inputs unchanged (re-hash INPUT_HASHES_BEFORE.json)
  V2  F0 bitwise control vs the sealed dt_0p0625ms stage (WARM.npz, the five BODY/NATIVE/NEURAL
      releases, 1.3 s J = 3.2364115081859093)
  V3  J, its components, foot support, min COM, the D1-D8 booleans and the first positive non-foot
      contact time recomputed from NATIVE.npz for all 20 releases and compared exactly to RESULT.json
  V4  release aggregate neural metrics (total spikes, window-mean pool rates, per-cell rates)
      recomputed from NEURAL.npz and compared exactly to RESULT.json
  V5  the shift rule re-derived from TRUE_CURRENTS.npz and compared to the stored applied_block_index
  V6  the four conditions really are different (warm and release neural state), so the experiment is
      not vacuous
  V7  the pre-registered reading rule re-applied from RESULT.json only, and the verdict compared to
      CROSS_CONDITION.json / SUMMARY.json
  V8  accounting, finiteness, warnings and scope-claim checks (no selection, no walking claim)

Exit code 0 only if every check passes.
"""
import sys
import json
import pickle
import hashlib
from pathlib import Path
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]

O = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/fine_phase')
S = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
NUM = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics')

DT = 0.0000625
NSUB = 16
CDT = 0.001
WARM_CMDS = 1200
REL_CMDS = 200
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
CONDITIONS = [('F0', 0), ('F1', 1), ('F2', 2), ('F3', 4)]
SEALED_J_1P3 = 3.2364115081859093
SEALED_NAME = 'dt_0p0625ms'
D_GATE_ORDER = ['D1_no_nonfoot_ground_force', 'D2_no_free_fall', 'D3_foot_support_not_below_assist',
                'D4_foot_support_never_collapses', 'D4b_nonfoot_is_not_carrying_the_body',
                'D5_com_height_not_below_assist', 'D6_force_balance_consistent', 'D7_tripod_redundancy',
                'D8_still_above_absolute_floor']


def js(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    import dynamic_support_metrics as dsm
    import train_dynamic_support as dst
    checks = []

    def check(name, ok, detail=None):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))
        print(('PASS ' if ok else 'FAIL ') + name, '' if detail is None else json.dumps(detail)[:300], flush=True)
        return ok

    # ------------------------------------------------------------------ V1 frozen inputs
    before = js(O / 'INPUT_HASHES_BEFORE.json')
    bad = []
    for p, h in before.items():
        if not Path(p).exists() or sha(p) != h:
            bad.append(p)
    check('V1_frozen_inputs_unchanged', not bad, dict(files=len(before), changed_or_missing=bad))

    summary = js(O / 'SUMMARY.json')
    cross = js(O / 'CROSS_CONDITION.json')
    prereg = js(O / 'PREREGISTRATION.json')
    results = {}
    for cond, _ in CONDITIONS:
        for ph in PHASES:
            results[(cond, ph)] = js(O / 'RUNS' / ('%s_paired_%s' % (cond, ph)) / 'RESULT.json')
    check('V0_all_20_results_present', len(results) == 20, dict(found=len(results)))

    # ------------------------------------------------------------------ V2 F0 bitwise control
    z = arrays(O / 'CONFIGS/F0/WARM.npz')
    sz = arrays(NUM / 'CONFIGS' / SEALED_NAME / 'WARM.npz')
    warm_same = {k: bool(np.array_equal(z[k], sz[k])) for k in sz}
    check('V2a_F0_warm_npz_bitwise_vs_sealed', all(warm_same.values()),
          dict(n_arrays=len(sz), not_exact=[k for k, v in warm_same.items() if not v],
               extra_keys=[k for k in z if k not in sz]))
    rel_same = {}
    for ph in PHASES:
        old = js(NUM / 'RUNS' / (SEALED_NAME + '_paired_' + str(ph)) / 'RESULT.json')
        mine = results[('F0', ph)]
        per = {}
        for f in ['BODY.npz', 'NATIVE.npz', 'NEURAL.npz']:
            a = arrays(Path(old['folder']) / f)
            b = arrays(O / 'RUNS' / ('F0_paired_' + str(ph)) / f)
            per[f] = bool(set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a))
        per['J_exactly_equal'] = bool(mine['J'] == old['J'])
        per['D_checks_equal'] = bool(mine['validation']['checks'] == old['validation']['checks'])
        per['first_nonfoot_exactly_equal'] = bool(mine['first_positive_nonfoot_s'] == old['first_positive_nonfoot_s'])
        rel_same[str(ph)] = per
    check('V2b_F0_five_releases_bitwise_vs_sealed',
          all(all(v.values()) for v in rel_same.values()), rel_same)
    check('V2c_F0_J_at_1p3_exactly_3p2364115081859093',
          results[('F0', 1.3)]['J'] == SEALED_J_1P3, dict(J=results[('F0', 1.3)]['J']))

    # ------------------------------------------------------------------ V3 recompute the physical numbers
    with (S / 'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:
        physical = pickle.load(f)
    recompute = {}
    j_mismatch, parts_mismatch, gate_mismatch, nf_mismatch = [], [], [], []
    for (cond, ph), r in sorted(results.items()):
        native = arrays(O / 'RUNS' / r['label'] / 'NATIVE.npz')['rows']
        scored = native.copy()
        scored[:, 0] -= ph - 1.3
        old_duration = dst.RELEASE
        dst.RELEASE = .2
        try:
            J, parts = dst.objective(scored, physical[ph]['assist'])
        finally:
            dst.RELEASE = old_duration
        gate = dsm.evaluate_native(physical[ph]['assist'], native)
        gate_checks = dict(gate['checks'])
        gate_checks['D1_no_nonfoot_ground_force'] = bool(np.max(native[:, 7]) == 0)
        pos = np.flatnonzero(native[:, 7] > 1e-9)
        first_nf = float(native[pos[0], 0] - ph) if len(pos) else None
        if J != r['J']:
            j_mismatch.append(r['label'])
        if json.dumps(parts, sort_keys=True, default=float) != json.dumps(r['parts'], sort_keys=True, default=float):
            parts_mismatch.append(r['label'])
        if gate_checks != r['validation']['checks']:
            gate_mismatch.append(dict(label=r['label'], recomputed=gate_checks, stored=r['validation']['checks']))
        if first_nf != r['first_positive_nonfoot_s']:
            nf_mismatch.append(dict(label=r['label'], recomputed=first_nf, stored=r['first_positive_nonfoot_s']))
        recompute[r['label']] = dict(J=J, foot_support_mean=parts['foot_support_mean'],
                                     com_z_min=parts['com_z_min'], first_positive_nonfoot_s=first_nf,
                                     D_gates_passed=int(sum(gate_checks.values())))
    check('V3a_J_recomputed_from_NATIVE_exact', not j_mismatch, dict(mismatch=j_mismatch, n=len(recompute)))
    check('V3b_J_components_recomputed_exact', not parts_mismatch, dict(mismatch=parts_mismatch))
    check('V3c_D1_D8_booleans_recomputed_exact', not gate_mismatch, dict(mismatch=gate_mismatch))
    check('V3d_first_nonfoot_recomputed_exact', not nf_mismatch, dict(mismatch=nf_mismatch))

    # ------------------------------------------------------------------ V4 recompute neural aggregates
    agg_bad = []
    for (cond, ph), r in sorted(results.items()):
        n = arrays(O / 'RUNS' / r['label'] / 'NEURAL.npz')
        # NEURAL.npz stores one row per 1 ms command, each holding this command's 16 sub-step bitmaps.
        packed = n['spikes12000_packed']
        counts = np.zeros(12000, np.int64)
        for row in packed:
            counts += np.unpackbits(row, axis=1, count=12000).sum(0).astype(np.int64)
        pool = n['rates'].astype(float)
        a = r['aggregate_neural']
        ok = (int(counts.sum()) == a['release_total_spikes']
              and np.array_equal(counts / .2, np.asarray(a['release_window_mean_cell_hz']))
              and np.array_equal(pool.mean(0), np.asarray(a['release_window_mean_pool_rates']))
              and packed.shape == (REL_CMDS, NSUB, 1500) and pool.shape == (REL_CMDS, 66))
        if not ok:
            agg_bad.append(dict(label=r['label'], packed_shape=list(packed.shape),
                                total=int(counts.sum()), stored_total=a['release_total_spikes'],
                                cell_hz_equal=bool(np.array_equal(counts / .2,
                                                                 np.asarray(a['release_window_mean_cell_hz']))),
                                pool_equal=bool(np.array_equal(pool.mean(0),
                                                               np.asarray(a['release_window_mean_pool_rates'])))))
    check('V4_release_neural_aggregates_recomputed_exact', not agg_bad, dict(mismatch=agg_bad))

    # ------------------------------------------------------------------ V5 shift rule re-derived
    cur = arrays(S / 'TRUE_CURRENTS.npz')['currents']
    s_all = np.arange(WARM_CMDS * NSUB)
    shift_bad = []
    for cond, N in CONDITIONS:
        w = arrays(O / 'CONFIGS' / cond / 'WARM.npz')
        expect = (np.maximum(s_all - N, 0) // NSUB).astype(np.int64)
        if not np.array_equal(np.asarray(w['applied_block_index']), expect):
            shift_bad.append(cond)
        if int(w['shift_substeps']) != N or float(w['shift_ms']) != N * DT * 1000:
            shift_bad.append(cond + '_decl')
    # direct spot check: at sub-step s the applied row really is currents[expect[s]]
    spot = {}
    for cond, N in CONDITIONS:
        w = arrays(O / 'CONFIGS' / cond / 'WARM.npz')
        bidx = np.asarray(w['applied_block_index'])
        ss = [0, N, N + 15, N + 16, 5000, 19199]
        spot[cond] = dict(substeps=N, samples=[dict(s=int(x), block=int(bidx[x]),
                                                    matches_rule=bool(np.array_equal(cur[bidx[x]], cur[max(int(x) - N, 0) // NSUB])))
                                               for x in ss])
    check('V5_shift_rule_rederived_from_currents', not shift_bad,
          dict(mismatch=shift_bad, spot=spot))

    # ------------------------------------------------------------------ V6 conditions really differ
    f0agg = arrays(O / 'CONFIGS/F0/WARM_AGG.npz')
    differ = {}
    for cond, N in CONDITIONS:
        if cond == 'F0':
            continue
        a = arrays(O / 'CONFIGS' / cond / 'WARM_AGG.npz')
        w = arrays(O / 'CONFIGS' / cond / 'WARM.npz')
        w0 = arrays(O / 'CONFIGS/F0/WARM.npz')
        differ[cond] = dict(
            warm_spikes_differ=bool(not np.array_equal(w['spikes12000_packed'], w0['spikes12000_packed'])),
            warm_pool_rates_differ=bool(not np.array_equal(w['pool_rates'], w0['pool_rates'])),
            capture_rates_differ_all_phases=bool(all(
                not np.array_equal(a['cap_rates_' + str(ph)], f0agg['cap_rates_' + str(ph)]) for ph in PHASES)),
            release_neural_differs_all_phases=bool(all(
                results[(cond, ph)]['aggregate_neural']['release_total_spikes']
                != results[('F0', ph)]['aggregate_neural']['release_total_spikes'] or
                not np.array_equal(np.asarray(results[(cond, ph)]['aggregate_neural']['release_window_mean_pool_rates']),
                                   np.asarray(results[('F0', ph)]['aggregate_neural']['release_window_mean_pool_rates']))
                for ph in PHASES)))
    check('V6_every_shift_changed_the_neural_state',
          all(all(v.values()) for v in differ.values()), differ)

    # ------------------------------------------------------------------ V7 reading rule re-applied
    def spreads(ph):
        J = {c: results[(c, ph)]['J'] for c, _ in CONDITIONS}
        foot = {c: results[(c, ph)]['parts']['foot_support_mean'] for c, _ in CONDITIONS}
        com = {c: results[(c, ph)]['parts']['com_z_min'] for c, _ in CONDITIONS}
        nf = {c: (.2 if results[(c, ph)]['first_positive_nonfoot_s'] is None
                  else results[(c, ph)]['first_positive_nonfoot_s']) for c, _ in CONDITIONS}
        return dict(J=J, foot=foot, com=com, nf=nf,
                    J_spread=float(max(J.values()) - min(J.values())),
                    foot_spread=float(max(foot.values()) - min(foot.values())),
                    com_spread=float(max(com.values()) - min(com.values())),
                    nf_spread=float(max(nf.values()) - min(nf.values())),
                    any_all_D=[c for c, _ in CONDITIONS if results[(c, ph)]['validation']['passed']],
                    n_D_max=max(results[(c, ph)]['D_gates_passed'] for c, _ in CONDITIONS))
    sp = {str(ph): spreads(ph) for ph in PHASES}
    m = sp['1.3']
    if m['J_spread'] <= 0.2 and m['foot_spread'] <= 0.1:
        verdict = 'FINE_STABLE'
    elif m['J_spread'] > 1.0:
        verdict = 'FINE_FRAGILE'
    else:
        verdict = 'INTERMEDIATE'
    check('V7a_verdict_reapplied_matches_CROSS_CONDITION', verdict == cross['verdict'],
          dict(recomputed=verdict, stored=cross['verdict']))
    vn = cross['verdict_numbers']
    check('V7b_spread_numbers_match',
          vn['J_spread_at_1p3'] == m['J_spread'] and vn['foot_support_spread_at_1p3'] == m['foot_spread'],
          dict(recomputed=dict(J=m['J_spread'], foot=m['foot_spread']),
               stored=dict(J=vn['J_spread_at_1p3'], foot=vn['foot_support_spread_at_1p3'])))
    other = [ph for ph in PHASES if ph != 1.3]
    smaller = [str(ph) for ph in other if sp[str(ph)]['J_spread'] < m['J_spread']]
    check('V7c_all_other_phase_spread_claim_matches',
          bool(vn['all_other_phases_smaller_J_spread']) == bool(all(
              sp[str(ph)]['J_spread'] < m['J_spread'] for ph in other)),
          dict(recomputed_smaller=smaller, stored_flag=vn['all_other_phases_smaller_J_spread']))
    check('V7d_any_condition_passes_all_D_gates_matches',
          bool(vn['any_condition_passes_all_D_gates_at_any_phase']) == bool(
              any(results[(c, ph)]['validation']['passed'] for c, _ in CONDITIONS for ph in PHASES)),
          dict(recomputed_any=any(results[(c, ph)]['validation']['passed']
                                  for c, _ in CONDITIONS for ph in PHASES),
               stored=vn['any_condition_passes_all_D_gates_at_any_phase']))
    check('V7e_reading_rule_frozen_in_prereg', prereg['reading_rule'] == cross['reading_rule'],
          dict(frozen_bands=dict(stable=prereg['reading_rule']['FINE_STABLE']['condition'],
                                 fragile=prereg['reading_rule']['FINE_FRAGILE']['condition'])))

    # ------------------------------------------------------------------ V8 accounting / scope
    expect_steps = 4 * WARM_CMDS * NSUB + 20 * REL_CMDS * NSUB
    check('V8a_neural_step_accounting', summary['neural_steps'] == expect_steps,
          dict(stored=summary['neural_steps'], expected=expect_steps))
    check('V8b_budget_matches_prereg',
          summary['warm_replays'] == 4 and summary['physical_releases'] == 20
          and summary['new_teacher_recordings'] == 0,
          dict(warm=summary['warm_replays'], releases=summary['physical_releases'],
               new_teacher=summary['new_teacher_recordings']))
    finite_ok = True
    warn_total = 0
    nan_files = []
    for (cond, ph), r in results.items():
        folder = O / 'RUNS' / r['label']
        n = arrays(folder / 'NATIVE.npz')['rows']
        b = arrays(folder / 'BODY.npz')
        if not (np.isfinite(n).all() and all(np.isfinite(v).all() for v in b.values())):
            nan_files.append(r['label'])
        if not (np.isfinite(r['J']) and np.isfinite(np.asarray(list(r['parts'].values()), dtype=float)).all()):
            nan_files.append(r['label'] + '_J')
        warn_total += r['warnings']
    finite_ok = not nan_files
    check('V8c_all_records_finite_and_no_mujoco_warnings', finite_ok and warn_total == 0,
          dict(nan_files=nan_files, warnings=warn_total))
    check('V8d_no_selection_and_scope_limits',
          summary['selected_model'] is None and summary['selected_timestep'] is None
          and summary['walking_certified'] is False and summary['autonomous_support_certified'] is False
          and summary['absolute_forces'] == 'NOT_ESTABLISHED'
          and summary['biological_validity'] == 'NOT_ESTABLISHED'
          and summary['brain_model_invalid_claim'] == 'NOT_MADE',
          {k: summary[k] for k in ['selected_model', 'selected_timestep', 'walking_certified',
                                   'absolute_forces', 'biological_validity', 'brain_model_invalid_claim']})
    coverage_left = (O / 'COVERAGE.json').exists()
    ledger = js(O / 'CLEANUP_LEDGER.json') if (O / 'CLEANUP_LEDGER.json').exists() else {'deleted': []}
    check('V8e_legacy_constructor_output_cleaned_and_logged',
          not coverage_left and any(d['path'].endswith('COVERAGE.json') for d in ledger['deleted']),
          dict(coverage_present=coverage_left, ledger_entries=len(ledger['deleted'])))

    passed = all(c['passed'] for c in checks)
    out = dict(verifier=str(Path(__file__).resolve()), independent_of_run_py=True,
               verified_dir=str(O), sealed_reference=str(NUM),
               checks=checks, recomputed_physical=recompute, recomputed_spreads=sp,
               verdict_recomputed=verdict, n_checks=len(checks),
               n_passed=int(sum(c['passed'] for c in checks)), all_passed=passed,
               note='Every published J, component, foot support, min COM, first non-foot time and D boolean '
                    'was recomputed from the saved 12.5 us native records; every aggregate neural metric was '
                    'recomputed from the saved 0.0625 ms spike records; the F0 control was re-compared '
                    'bitwise to the sealed stage.')
    dump(O / 'VERIFICATION.json', out)
    print(('VERIFIED %d/%d' % (out['n_passed'], out['n_checks'])), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
