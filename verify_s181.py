"""S181 saved-data INDEPENDENT check. Read-only: no engine, no forward, no optimizer, no import of mujoco.

Verifies only what is already on disk in the frozen run directory: CANDIDATES.json, SUMMARY.json,
PREREG.json, INPUT_HASHES.json, INPUT_HASHES_AFTER.json, SOURCE_FREEZE/, S181_SELECTED.png.
Producer files (run_s181_coordinated.py, plot_s181.py), the tracker and any history are opened
READ-ONLY and never written. The output JSON is refused if it already exists.
"""
import argparse
import hashlib
import json
import struct
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

# frozen constants taken from the producer's declared prereg (re-declared here, not imported)
CAP, RESERVED, LOAD_CAP = 2200, 200, 3
DEPTH_CAP = 0.03442910871489635
BASELINE_MINIMAX = 3.308620867791414
ROOT_TOL = 1e-8
STRICT = 1e-9
DEFAULT_RUN = Path('/home/sensen/Desktop/engine修复验收/s181-coordinated-search/20261025T120000000000Z')
DEFAULT_OUT = Path('/home/sensen/Desktop/engine修复验收/s181-coordinated-search/S181_INDEPENDENT_CHECK.json')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=DEFAULT_RUN)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    run, out = a.run, a.out  # literal requested paths, not symlink-resolved
    assert run.is_absolute() and out.is_absolute(), 'both paths must be absolute'
    assert run.is_dir(), f'run dir missing: {run}'
    if out.exists():
        raise SystemExit(f'REFUSED: output already exists, will not overwrite: {out}')
    checks, disc = [], []

    def add(name, ok, numbers, detail=''):
        checks.append(dict(name=name, status='PASS' if ok else 'FAIL', numbers=numbers, detail=detail))

    def js(p):
        return json.loads((run / p).read_text())

    cands, summ, prereg = js('CANDIDATES.json'), js('SUMMARY.json'), js('PREREG.json')
    base = cands[0]
    base_mm = float(base['minimax'])
    n = len(cands)
    p1 = [c for c in cands if c['tag'].startswith('P1_')]
    p2 = [c for c in cands if c['tag'].startswith('P2_')]
    bl = [c for c in cands if c['tag'].startswith('BASELINE')]

    # ---------- 1. candidate inventory + phase-2 degeneration ----------
    rank = summ['phase1_ranking']
    top2 = [(r[1], r[2]) for r in rank[:2]]
    same_leg_top2 = top2[0][0] == top2[1][0]
    p2tags = [c['tag'] for c in p2]
    dup = None
    if len(p2) == 1:
        best_match = min(p1, key=lambda c: (abs(c['minimax'] - p2[0]['minimax']),
                                            max(abs(x - y) for x, y in zip(c['qpos'], p2[0]['qpos']))))
        qd = max(abs(x - y) for x, y in zip(p2[0]['qpos'], best_match['qpos']))
        dup = dict(matches=best_match['tag'], max_abs_qpos_diff=qd,
                   minimax_diff=p2[0]['minimax'] - best_match['minimax'],
                   activation_identical=bool(p2[0]['activation'] == best_match['activation']),
                   unreachable_identical=bool(p2[0]['unreachable'] == best_match['unreachable']))
        if qd == 0.0 and abs(dup['minimax_diff']) == 0.0:
            disc.append(dict(id='PHASE2_DEGENERATE_DUPLICATE', severity='design',
                             note='phase 2 produced 1 candidate that re-applies the same single joint '
                                  'set of leg lm and is bit-identical to ' + best_match['tag'] +
                                  ' (max|dqpos|=0, dminimax=0); it adds no new information.'))
    if same_leg_top2:
        disc.append(dict(id='PHASE2_TOP2_SAME_LEG', severity='design',
                         note=f'phase1_ranking top-2 are {top2[0]} and {top2[1]}: same leg "{top2[0][0]}", '
                              'so "combine the two lowest candidates" combines one leg with itself.'))
    disc.append(dict(id='PHASE2_RULE_FLAW', severity='design',
                     note='producer loops `for i in range(2): for j in range(i+1,2)`, i.e. only the pair '
                          '(0,1) is ever formed and no distinct-leg guard exists; combining is applied as '
                          'q[k]=clip(q0[k]+delta) for delta_i then delta_j on the SAME joint set, so the '
                          'second delta overwrites the first instead of adding to it. PREREG budget says '
                          'phase2=4 but 1 was run.'))
    add('1_candidate_inventory_and_phase2_degeneration',
        n == 18 and len(bl) == 1 and len(p1) == 16 and len(p2) == 1 and same_leg_top2 and dup is not None and dup['max_abs_qpos_diff'] == 0.0,
        dict(n_candidates=n, n_baseline=len(bl), n_phase1=len(p1), n_phase2=len(p2),
             phase1_ranking_pairs=[(m, l, d) for m, l, d in rank], ranking_len=len(rank),
             top2=top2, top2_same_leg=same_leg_top2, phase2_tags=p2tags, phase2_duplicate=dup,
             phase1_legs=sorted({c['tag'][3:5] for c in p1}),
             phase1_deltas=sorted({c['tag'].split('_')[2] for c in p1})),
        'phase 2 degenerated because the phase-1 top-2 were both leg "lm" (deltas +0.15/+0.07); '
        'the fused candidate equals P1_lm_+0.07 exactly.')

    # ---------- 2. valid set + best + no candidate beats baseline ----------
    def geom_rule(c):
        g = c['geometry']
        return bool(g['foot_depth'] <= DEPTH_CAP and g['nonfoot_depth'] <= 1e-12
                    and g['bearing_legs'] >= 3 and g['friction_margin'] >= -1e-10)

    valid = [c for c in cands if c['root_ok'] and c['geom_ok']]
    revalid = [c for c in cands if bool(float(c['root_norm']) <= ROOT_TOL) and geom_rule(c)]
    gm_bad = [c['tag'] for c in cands if bool(geom_rule(c)) != bool(c['geom_ok'])]
    ro_bad = [c['tag'] for c in cands if bool(float(c['root_norm']) <= ROOT_TOL) != bool(c['root_ok'])]
    best = min(valid, key=lambda c: c['minimax'])
    all_min = min(float(c['minimax']) for c in cands)
    better = [(c['tag'], c['minimax']) for c in cands if float(c['minimax']) < BASELINE_MINIMAX - STRICT]
    sb = summ['best']
    add('2_valid_set_and_best_equals_baseline',
        len(valid) == 14 and len(revalid) == 14 and not gm_bad and not ro_bad and not better
        and abs(best['minimax'] - BASELINE_MINIMAX) == 0.0 and best['tag'].startswith('BASELINE')
        and abs(base_mm - BASELINE_MINIMAX) == 0.0 and summ['n_valid'] == 14
        and sb['tag'] == best['tag'] and float(sb['minimax']) == float(best['minimax'])
        and sum(1 for c in cands if not c['root_ok']) == 4
        and sum(1 for c in cands if not c['geom_ok']) == 4,
        dict(n_valid=len(valid), n_valid_recomputed=len(revalid), n_invalid=len(cands) - len(valid),
             best_tag=best['tag'], best_minimax=float(best['minimax']),
             declared_baseline_minimax=base_mm, min_over_all_candidates=all_min,
             candidates_below_baseline_minus_1e9=better,
             n_minimax_lt_baseline=len(better),
             minimax_ge_baseline=all(float(c['minimax']) >= BASELINE_MINIMAX - STRICT for c in cands),
             geom_rule_vs_declared_mismatches=gm_bad, root_ok_rule_vs_declared_mismatches=ro_bad,
             n_root_ok=sum(1 for c in cands if c['root_ok']), n_geom_ok=sum(1 for c in cands if c['geom_ok']),
             summary_n_valid=summ['n_valid'], summary_best_tag=sb['tag'],
             summary_best_minimax=float(sb['minimax']),
             valid_equals_root_ok=bool([c['tag'] for c in valid] == [c['tag'] for c in cands if c['root_ok']])),
        'root_ok, geom_ok and the frozen accept rule were recomputed from the saved numbers only.')
    if not gm_bad and not [c for c in cands if c['root_ok'] and not c['geom_ok']]:
        disc.append(dict(id='GEOM_GATE_REDUNDANT_HERE', severity='scope',
                         note='every candidate that fails geometry also fails the root solve, so the '
                              '{root_ok AND geom_ok} gate equals {root_ok} in this run (14 = 14); geometry '
                              'never independently removed a root-balanced candidate.'))

    # ---------- 3. exactly-5-unreachable-row candidates ----------
    five = [(c['tag'], float(c['minimax'])) for c in cands if len(c['unreachable_rows']) == 5]
    five_worse = all(m > BASELINE_MINIMAX + STRICT for _, m in five)
    un_bad = [c['tag'] for c in cands
              if sorted(x - 6 for x in c['unreachable']) != sorted(c['unreachable_rows'])
              or len(c['unreachable_rows']) != len(set(c['unreachable_rows']))
              or any(not (0 <= r < 42) for r in c['unreachable_rows'])]
    base_rows = base['unreachable_rows']
    add('3_five_unreachable_row_candidates_all_worse',
        len(five) == 4 and five_worse and not un_bad,
        dict(n_with_exactly_5_unreachable_rows=len(five),
             candidates=[dict(tag=t, minimax=m, delta_vs_baseline=m - BASELINE_MINIMAX) for t, m in five],
             all_strictly_worse_than_baseline=five_worse,
             baseline_unreachable_rows=base_rows, baseline_n_unreachable=len(base_rows),
             min_minimax_among_5=five and min(m for _, m in five),
             unreachable_index_bookkeeping_mismatches=un_bad,
             row_count_histogram={str(k): sum(1 for c in cands if len(c['unreachable_rows']) == k)
                                  for k in sorted({len(c['unreachable_rows']) for c in cands})}),
        'best coverage (5 unreachable rows) exists but only at minimax 4.387964...: no candidate reaches a '
        '48-row balanced point and none has fewer than 5 unreachable rows.')

    # ---------- 4. budget + call ledger ----------
    call = summ['calls']
    pcj = call['per_candidate_jacobians']
    usable = CAP - RESERVED
    fwd = int(call['forward'])
    per = pcj
    jac_sum = sum(int(v) for v in per.values())
    jac_ok = jac_sum == int(call['root_jacobians'])
    pcj_issues = sorted(set(per) ^ {c['tag'] for c in cands})
    fd_ok = int(call['fd_probes']) == 3 * int(call['root_jacobians'])
    iter_ok = int(call['root_jacobians']) == int(call['root_newton_iterations']) + n
    resid = 2 * (n + int(call['fd_probes']) + int(call['line_search_trials'])) + 2 * n - fwd
    implied_cache_hits = resid / 2 if resid >= 0 and resid % 2 == 0 else None
    add('4_budget_and_call_ledger',
        fwd <= usable and int(call['model_load']) == 1 and int(call['model_load']) <= LOAD_CAP
        and int(call['inner_lp']) == n and int(call['candidate_evaluations']) == n
        and jac_ok and not pcj_issues and fd_ok and iter_ok and implied_cache_hits is not None
        and min(per.values()) == 1 and max(per.values()) == 9,
        dict(forward=fwd, forward_cap=CAP, reserved=RESERVED, usable_forward_cap=usable,
             forward_headroom=usable - fwd, forward_within_cap=fwd <= usable,
             model_load=int(call['model_load']), load_cap=LOAD_CAP,
             root_jacobians=int(call['root_jacobians']),
             root_newton_iterations=int(call['root_newton_iterations']),
             final_decomposition_jacobians=int(call['root_jacobians']) - int(call['root_newton_iterations']),
             inner_lp=int(call['inner_lp']), candidate_evaluations=int(call['candidate_evaluations']),
             fd_probes=int(call['fd_probes']), line_search_trials=int(call['line_search_trials']),
             per_candidate_jacobians=per, per_candidate_min=min(per.values()),
             per_candidate_max=max(per.values()),
             n_candidates_ge_8=[t for t, v in per.items() if int(v) >= 8],
             n_candidates_ge_8_count=sum(1 for v in per.values() if int(v) >= 8),
             jacobians_sum_equals_ledger=jac_ok, per_candidate_tag_mismatches=pcj_issues,
             fd_probes_equals_3x_jacobians=fd_ok, jacobians_equals_iterations_plus_final=iter_ok,
             implied_cache_hits_from_forward_identity=implied_cache_hits),
        'forward ledger is internally consistent only if 9 root evaluations were cache hits, which '
        'SUMMARY does not report; also only 4/18 candidates spent >=8 Jacobians.')

    # ---------- 5. root unknowns + native window ----------
    root = str(prereg['root'])
    win_txt = str(prereg['window'])
    win = json.loads(win_txt.split('frozen): ', 1)[1])
    hashes = js('INPUT_HASHES.json')
    xml_path = next(p for p in hashes if p.endswith('MODEL_CALIB_H_VERIFIED.xml')
                    and 's27-graded-dose' in p)
    idx = {p: i for i, (p, _h) in enumerate(sorted(hashes.items()))}
    xml_copy = run / 'SOURCE_FREEZE' / f"{idx[xml_path]:04d}_{Path(xml_path).name}"
    joints = [j for j in ET.parse(xml_copy).getroot().find('worldbody').iter('joint')]
    jn = [(j.get('name'), [float(x) for x in j.get('range').split()]) for j in joints[1:]]
    win_match = ({k: v for k, v in jn} == win) and len(jn) == 42
    frozen_mismatch = [k for k, v in jn if win.get(k) != v]
    add('5_root_unknowns_and_native_window',
        root.strip().startswith('z, roll, pitch only') and 'x,y,yaw' in root.replace(' ', '')
        and win_txt.startswith('native XML limits') and win_match and len(win) == 42
        and not frozen_mismatch and 's27-graded-dose' in xml_path,
        dict(prereg_root=root, root_unknowns_declared=['z', 'roll', 'pitch'],
             root_held=['x', 'y', 'yaw'], n_window_joints=len(win),
             window_source='native XML limits of the frozen model',
             xml_checked=str(xml_copy), xml_joint_count=len(jn),
             window_equals_frozen_xml_ranges=bool(win_match), range_mismatches=frozen_mismatch,
             n_other_root_dofs_declared_free=0),
        'the window was re-read from the frozen XML copy in SOURCE_FREEZE, not from the prereg text.')

    # ---------- 6. hashes + figure ----------
    after = js('INPUT_HASHES_AFTER.json')
    eq = hashes == after
    rehash_bad, missing = [], []
    for p, h in hashes.items():
        if not Path(p).exists():
            missing.append(p)
            continue
        if sha(p) != h:
            rehash_bad.append(p)
    freeze = sorted(x.name for x in (run / 'SOURCE_FREEZE').iterdir())
    expect_freeze = [f"{i:04d}_{Path(p).name}" for i, (p, _h) in enumerate(sorted(hashes.items()))]
    freeze_bad = [nm for (p, h), nm in zip(sorted(hashes.items()), expect_freeze)
                  if nm not in freeze or sha(run / 'SOURCE_FREEZE' / nm) != h]
    fig = run / 'S181_SELECTED.png'
    raw = fig.read_bytes()
    sig_ok = raw[:8] == b'\x89PNG\r\n\x1a\n'
    w = hgt = 0
    idat = b''
    if sig_ok:
        w, hgt = struct.unpack('>II', raw[16:24])
        i = 8
        while i + 8 <= len(raw):
            ln = int.from_bytes(raw[i:i + 4], 'big')
            typ = raw[i + 4:i + 8]
            if typ == b'IDAT':
                idat += raw[i + 8:i + 8 + ln]
            i += 12 + ln
    px = zlib.decompress(idat) if idat else b''
    distinct = len(set(px))
    fig_ok = (sig_ok and len(raw) > 10000 and w >= 1000 and hgt >= 400
              and len(px) == w * hgt * 4 + hgt and distinct > 32)
    add('6_hashes_freeze_and_figure',
        eq and not missing and not rehash_bad and not freeze_bad and len(freeze) == len(hashes) and fig_ok,
        dict(n_input_hashes=len(hashes), input_hashes_equals_after=eq,
             n_rehashed=len(hashes) - len(missing), missing_inputs=missing,
             rehash_mismatches=rehash_bad, n_source_freeze_files=len(freeze),
             source_freeze_mismatches=freeze_bad,
             figure=dict(path=str(fig), bytes=len(raw), sha256=sha(fig), png_signature_ok=sig_ok,
                         width=w, height=hgt, pixels_bytes=len(px),
                         distinct_pixel_bytes=distinct,
                         raw_stream_is_wxhx4_plus_filter_rows=(len(px) == w * hgt * 4 + hgt))),
        'INPUT_HASHES == INPUT_HASHES_AFTER and every declared file, including the 403 SOURCE_FREEZE '
        'copies, re-hashes to its declared value.')

    # ---------- 7. scope / claim boundaries ----------
    prod = Path('/run/media/sensen/Data2/cell_wound_prototype/run_s181_coordinated.py')
    src = prod.read_text()
    no_engine_calls = ('mj_step' not in src) and ('mj_neural' not in src) and ('mj_dyn' not in src)
    flags_ok = all(prereg.get(k) is True for k in ('no_step', 'no_neural', 'no_training', 'no_dynamics'))
    si_ok = ('SI NOT_ESTABLISHED' in str(summ.get('units', ''))) and ('SI NOT_ESTABLISHED' in str(prereg.get('units', '')))
    no48 = len(base['unreachable_rows']) > 0 and all(len(c['unreachable_rows']) > 0 for c in cands)
    claim_txt = json.dumps([summ, prereg]).lower()
    no_claim_words = not any(w in claim_txt for w in ('standing', 'walking', 'walk', 'neural claim', 'si_established'))
    add('7_scope_and_claim_boundaries',
        no_engine_calls and flags_ok and si_ok and no48 and no_claim_words,
        dict(producer_source=str(prod), producer_sha256=sha(prod),
             producer_contains_mj_step=('mj_step' in src), producer_contains_mj_neural=('mj_neural' in src),
             prereg_no_step_no_neural_no_training_no_dynamics=flags_ok,
             units_si_not_established=si_ok,
             no_candidate_has_zero_unreachable_rows=no48,
             best_is_baseline=best['tag'].startswith('BASELINE'),
             n_candidates_below_baseline=len(better)),
        'the run makes no dynamics, standing, walking or neural claim and no 48-row balanced point.')

    out_json = dict(
        verified=all(c['status'] == 'PASS' for c in checks),
        run_dir=str(run),
        checker=dict(file=str(Path(__file__).resolve()), sha256=sha(Path(__file__).resolve()),
                     mode='saved-data only; no engine, no forward, no optimizer', engine_imported=False),
        constants=dict(forward_cap=CAP, reserved=RESERVED, load_cap=LOAD_CAP, depth_cap=DEPTH_CAP,
                       baseline_minimax=BASELINE_MINIMAX, root_tol=ROOT_TOL),
        checks=checks,
        discrepancies=disc,
        checker_revisions=[dict(
            note='the first execution of this same checker exited 0 but marked checks 5 and 6 FAIL for two '
                 'checker-side reasons (prereg writes "x,y,yaw" without spaces; figure non-emptiness was '
                 'sampled only from the head of the inflated PNG stream, which is uniform background). Both '
                 'saved-data values were correct. That defective output (sha256 7deb78a5f6484d3f6504f21f2d'
                 '8a6e98dabd7520df43f68ae8297b969ac9df38) was deleted and the checker corrected; this file is '
                 'the single accepted run. No producer file, tracker or history was touched.')],
        calls=dict(model_load=int(call['model_load']), forward=fwd, step=0, neural=0),
        statement=(
            'S181 saved data re-checked without running the engine: 18 candidates (1 baseline + 16 phase-1 + '
            '1 phase-2), valid set = 14, best minimax = 3.308620867791414 = BASELINE_no_leg_movement, and no '
            'candidate is below baseline - 1e-9. Phase 2 degenerated because the phase-1 top-2 were both leg '
            '"lm" (+0.15 and +0.07), so the pair-combination rule fused one leg with itself and produced a '
            'bit-identical duplicate of P1_lm_+0.07. Every candidate carrying exactly 5 unreachable rows '
            '(4.387964246, 4.387985855, 5.147548774, 4.387985855) is strictly worse than the baseline. '
            'Forward = 968 <= 2200 - 200 = 2000, model_load = 1 <= 3, root_jacobians = 109 '
            '(91 Newton iterations + 18 final-decomposition Jacobians), inner_lp = 18, per-candidate '
            'Jacobians 1..9 with only 4/18 at or above 8. Root unknowns are z/roll/pitch only and the '
            'window is the frozen native XML range of all 42 leg joints. There is NO 48-row balanced point, '
            'NO candidate that beats the baseline, and NO dynamics, standing or neural claim; '
            'SI NOT_ESTABLISHED.'))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'REFUSED: output already exists, will not overwrite: {out}')
    out.write_text(json.dumps(out_json, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(written=str(out), verified=out_json['verified'],
                          status={c['name']: c['status'] for c in checks},
                          forward=fwd, model_load=int(call['model_load'])),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
