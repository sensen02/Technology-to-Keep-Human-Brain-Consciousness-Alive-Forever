"""Independent verification of the S5/P2 numerics_shift stage.

Everything here is recomputed from the saved arrays and from the sealed read-only inputs with
separately written code: the shift rule, the applied stream, an exact prefix replay, every metric,
the aggregate contrast, the growth description and the pre-registered decision rule.
Never writes into the sealed numerics / numerics_followup / sensory stages.
"""
import sys, json, pickle, copy, hashlib, struct, time, traceback
from pathlib import Path
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]
from engine.neural_cond import ConductanceNetwork

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics_shift')
SRC = Path(__file__).resolve().parent
NUM = OUT.parent / 'numerics'
FOL = OUT.parent / 'numerics_followup'

DT = 0.015625
SUB = 64
NMS = 200
NSTEP = NMS * SUB
CELLS = 12000
FIELDS = ['v', 'ge', 'gi', 'adaptation', 'refractory_until_ms', 'spike_count']
SHIFTS = [0, 1, 2, 4, 8]
D_DT = 0.287739
FINE_TOL = 1e-12


def js(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def label(s):
    return 'shift_%03d_substeps' % s


def load(s):
    with np.load(OUT / 'RUNS' / label(s) / 'NEURAL.npz') as z:
        return {k: z[k] for k in z.files}


def png_size(p):
    d = Path(p).read_bytes()[:33]
    assert d[:8] == b'\x89PNG\r\n\x1a\n'
    assert d[12:16] == b'IHDR'
    return struct.unpack('>II', d[16:24])


# ------------------------------------------------------------------ independent implementations

def rule_blocks(shift):
    """Independent explicit construction of the per-sub-step 1 ms source block (no numpy shortcut)."""
    out = np.empty(NSTEP, np.int64)
    for s in range(NSTEP):
        k = s - shift
        if k < 0:
            k = 0
        blk = 0
        while (blk + 1) * SUB <= k:
            blk += 1
        out[s] = blk
    return out


def nrms(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b ** 2))
    return float(num / den) if den > 1e-9 else float(num / 1e-9)


def window_mean_nrms(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx = x.sum(0) / x.shape[0]
    my = y.sum(0) / y.shape[0]
    return nrms(mx, my), mx, my


def spikes_bits(packed):
    """Yield (row offset, boolean matrix) for the packed spike stream."""
    for a in range(0, packed.shape[0], 400):
        b = min(a + 400, packed.shape[0])
        yield a, np.unpackbits(packed[a:b], axis=1, count=CELLS).astype(bool)


def sealed_prefix_replay():
    """Control and shifted streams replayed side by side for 96 sub-steps from the sealed initial state."""
    with (FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl').open('rb') as f:
        ini = pickle.load(f)
    with np.load(NUM / 'RUNS/dt_0p5ms_paired_1.3/NEURAL.npz') as z:
        cur = z['sensory_current'].copy()
    with np.load(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz') as z:
        sealed_pack = z['spikes12000_packed'][:96].copy()
    out = {}
    horizon = 96
    for shift in SHIFTS:
        a = copy.deepcopy(ini['net'])
        b = copy.deepcopy(ini['net'])
        sb = rule_blocks(shift)
        ctrl_rows = []
        first_state = None
        first_any = None
        for s in range(horizon):
            sa = a.step(cur[s // SUB])          # independent: the sealed application, not rule_blocks(0)
            sbi = b.step(cur[sb[s]])
            ctrl_rows.append(np.packbits(sa))
            same_state = True
            for k in FIELDS:
                if not np.array_equal(np.asarray(getattr(a, k)), np.asarray(getattr(b, k))):
                    same_state = False
                    break
            same_spike = bool(np.array_equal(sa, sbi))
            if first_any is None and not (same_state and same_spike):
                first_any = s
            if first_state is None and not same_state:
                first_state = s
                break
        ctrl_prefix_matches_sealed = bool(np.array_equal(np.asarray(ctrl_rows), sealed_pack[:len(ctrl_rows)]))
        out[str(shift)] = dict(first_differing_substep=first_any,
                               first_differing_continuous_substep=first_state,
                               control_prefix_matches_sealed=ctrl_prefix_matches_sealed,
                               steps_replayed=len(ctrl_rows),
                               control_applied_block=int(rule_blocks(shift)[first_any]) if first_any is not None else None)
        del a, b
    return out


# ---------------------------------------------------------------------------------------- checks

def main():
    t0 = time.monotonic()
    rec = {}
    summ = js(OUT / 'SUMMARY.json')
    pre = js(OUT / 'PREREGISTRATION.json')
    tests = js(OUT / 'MECHANISM_TESTS.json')
    probe = js(OUT / 'PREFIX_PROBE.json')
    z = {s: load(s) for s in SHIFTS}

    # C1 run inventory and declared metadata
    inv = {}
    ok = True
    for s in SHIFTS:
        shape = {k: list(np.asarray(z[s][k]).shape) for k in
                 ['spikes12000_packed', 'pool_rates', 'raw_command', 'clipped_command', 'counts12000',
                  'applied_block_index', 'sensory_current', 'neural_clock']}
        # 66 pools for the rates; 42 readout command channels, both as in the sealed stage; for every shared
        # key the bitwise equality check C2 against the sealed run implies the same shape.
        want = dict(spikes12000_packed=[NSTEP, CELLS // 8], pool_rates=[NMS, 66], raw_command=[NMS, 42],
                    clipped_command=[NMS, 42], counts12000=[NMS, CELLS], applied_block_index=[NSTEP],
                    sensory_current=[NMS, CELLS], neural_clock=[NMS, 2])   # clock is sampled per 1 ms block
        good = shape == want
        dts = float(np.asarray(z[s]['neural_dt_ms'])) == DT and int(np.asarray(z[s]['shift_substeps'])) == s
        good = good and dts
        ok = ok and good
        inv[str(s)] = dict(shapes=shape, shapes_expected_ok=bool(shape == want), shift_meta_ok=bool(dts),
                           total_spikes=int(np.asarray(z[s]['counts12000']).sum()),
                           shift_ms=float(np.asarray(z[s]['shift_ms'])))
    r = js(OUT / 'RUNS' / label(0) / 'RESULT.json')
    ok = ok and r['shift_substeps'] == 0 and r['neural_steps'] == NSTEP and r['physical_runs'] == 0
    rec['C1_run_inventory'] = dict(passed=bool(ok), runs=inv, control_result=r,
                                   note='all five runs carry the declared 200 x 12000 / 12800-step shapes and dt')

    # C2 control bitwise against the sealed numerics_followup run
    with np.load(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz') as zz:
        sealed = {k: zz[k] for k in zz.files}
    keys = ['spikes12000_packed', 'pool_rates', 'raw_command', 'clipped_command', 'command_source_rates',
            'counts12000', 'sensory_current', 'initial_pool_rates', 'neural_dt_ms', 'neural_clock',
            'time_s', 'command_time_s'] + FIELDS
    eq = {k: bool(np.array_equal(np.asarray(z[0][k]), np.asarray(sealed[k]))) for k in keys}
    with (FOL / 'RUNS/backward_euler_0p015625ms/FINAL_STATE.pkl').open('rb') as f:
        sfin = pickle.load(f)
    with (FOL / 'RUNS/backward_euler_0p015625ms/INITIAL_STATE.pkl').open('rb') as f:
        sini = pickle.load(f)
    mine_fin = pickle.load((OUT / 'RUNS' / label(0) / 'FINAL_STATE.pkl').open('rb'))
    mine_ini = pickle.load((OUT / 'RUNS' / label(0) / 'INITIAL_STATE.pkl').open('rb'))
    feq = {k: bool(np.array_equal(getattr(mine_fin['net'], k), getattr(sfin['net'], k))) for k in FIELDS}
    ieq = {k: bool(np.array_equal(getattr(mine_ini['net'], k), getattr(sini['net'], k))) for k in FIELDS}
    rec['C2_control_bitwise'] = dict(
        passed=bool(all(eq.values()) and all(feq.values()) and all(ieq.values())
                    and np.array_equal(mine_fin['rates'], sfin['rates'])
                    and np.array_equal(mine_ini['rates'], sini['rates'])
                    and np.array_equal(mine_fin['net']._ring, sfin['net']._ring)
                    and mine_fin['net']._ring_i == sfin['net']._ring_i),
        arrays_exact=eq, final_state_exact=feq, initial_state_exact=ieq,
        final_ring_exact=bool(np.array_equal(mine_fin['net']._ring, sfin['net']._ring)),
        sealed=str(FOL / 'RUNS/backward_euler_0p015625ms/NEURAL.npz'),
        note='control must reproduce the sealed backward_euler_0p015625ms 200 ms result bitwise')

    # C3 the stored block index equals the declared rule, recomputed with an explicit loop
    rule = {}
    ok = True
    for s in SHIFTS:
        want = rule_blocks(s)
        got = np.asarray(z[s]['applied_block_index'], np.int64)
        same = bool(np.array_equal(got, want))
        ok = ok and same
        changes = np.flatnonzero(np.diff(want) != 0) + 1
        rule[str(s)] = dict(passed=same, first_change_substep=int(changes[0]), n_boundaries=int(changes.size),
                            block_at_step_64=int(want[64]), min_block=int(want.min()), max_block=int(want.max()))
    rec['C3_shift_rule_recomputed'] = dict(
        passed=bool(ok), per_shift=rule,
        rule='k=max(s-N,0); block=k//64; applied_current(s)=sensory_current[block] (delay by N sub-step indices, '
             'first sample held; identical for every run)',
        note='the exact per-sub-step source block stored with each run matches the documented rule')

    # C4 applied stream equals the independent delayed reconstruction (subset of columns, full time axis)
    cur = np.asarray(z[0]['sensory_current'], float)
    assert cur.shape == (NMS, CELLS)
    cols = np.r_[0:8, 12000 - 8:12000, 6000:6004]
    held = np.repeat(cur[:, cols], SUB, axis=0)
    ok = True
    detail = {}
    for s in SHIFTS:
        want = held if s == 0 else np.concatenate([np.repeat(cur[0:1, cols], s, axis=0), held])[:NSTEP]
        got = cur[np.asarray(z[s]['applied_block_index'], np.int64)][:, cols]
        same = bool(np.array_equal(got, want))
        ok = ok and same
        rows = np.flatnonzero(np.any(got != held, axis=1)) if s else np.empty(0, np.int64)
        detail[str(s)] = dict(passed=same, columns=cols.tolist(),
                              first_row_where_got_differs_from_undelayed=(int(rows[0]) if rows.size else None))
    rec['C4_applied_stream_reconstruction'] = dict(passed=bool(ok), per_shift=detail,
                                                   note='the applied stream stored in each run reproduces the '
                                                        'declared sub-step delay with the first sample held, on the '
                                                        'full 200 ms time axis for a fixed column subset')

    # C5 independent exact prefix replay from the sealed initial state
    replay = sealed_prefix_replay()
    ok = True
    for s in SHIFTS:
        rep = replay[str(s)]
        same = rep['control_prefix_matches_sealed']
        if s == 0:
            same = same and rep['first_differing_substep'] is None
        else:
            rec_probe = probe[str(s)]
            same = (same and rep['first_differing_continuous_substep'] == rec_probe['first_differing_continuous_substep']
                    and rep['first_differing_substep'] == rec_probe['first_differing_substep'])
        ok = ok and same
        replay[str(s)]['matches_recorded_probe'] = bool(same)
    rec['C5_prefix_replay_independent'] = dict(
        passed=bool(ok), per_shift=replay, horizon_steps=96,
        note='two networks replayed side by side from the sealed initial state with full float64 equality of all six '
             'state arrays at every sub-step; the control prefix also reproduces the sealed spike stream. This '
             'independently confirms whether any threshold flip occurs and the first sub-step at which the '
             'continuous state differs')

    # C6 metrics recomputed from the saved arrays
    ok = True
    per = {}
    for s in SHIFTS:
        if s == 0:
            continue
        c = summ['comparisons'][str(s)]
        e = {}
        for w in [1, 5, 10, 20, 50, 100, 200]:
            got = {k: nrms(z[s][k][:w], z[0][k][:w]) for k in ['pool_rates', 'raw_command', 'clipped_command']}
            want = {k: c['windows'][str(w)][k]['normalized_RMS'] for k in got}
            good = all(abs(got[k] - want[k]) <= FINE_TOL * max(1.0, abs(want[k])) for k in got)
            ok = ok and good
            e[str(w)] = dict(pool_rates=got['pool_rates'], raw_command=got['raw_command'],
                             clipped_command=got['clipped_command'], matches_summary=bool(good))
        per[str(s)] = e
    rec['C6_metrics_recomputed'] = dict(passed=bool(ok), per_shift=per,
                                        formula='nRMS = sqrt(mean((a-b)^2)) / max(sqrt(mean(b^2)), 1e-9), same '
                                                'windows and formula as the sealed stages, b = control',
                                        note='independently rewritten metric code reproduces every stored window value')

    # C7 spike statistics recomputed
    ok = True
    per = {}
    for s in SHIFTS:
        if s == 0:
            continue
        total_diff = 0
        first = None
        for a, sp in spikes_bits(np.asarray(z[s]['spikes12000_packed'])):
            b = a + sp.shape[0]
            ctrl = np.unpackbits(np.asarray(z[0]['spikes12000_packed'][a:b]), axis=1, count=CELLS).astype(bool)
            d = (sp != ctrl).sum(1)
            total_diff += int(d.sum())
            if first is None and d.any():
                first = int(a + int(np.flatnonzero(d)[0]))
        cp = np.asarray(z[s]['counts12000']).astype(np.int64).sum(0)
        cc = np.asarray(z[0]['counts12000']).astype(np.int64).sum(0)
        pear = float(np.corrcoef(cp, cc)[0, 1])
        st = summ['comparisons'][str(s)]
        good = (total_diff == st['windows']['200']['differing_spikes_cumulative']
                and first == st['discrete_divergence']['first_differing_spike_step']
                and int(cp.sum() - cc.sum()) == st['final_spike_count']['difference_total']
                and abs(pear - st['final_spike_count']['per_cell_pearson']) < 1e-12)
        ok = ok and good
        per[str(s)] = dict(differing_spikes_cumulative_200ms=total_diff,
                           first_differing_spike_step=first,
                           first_differing_spike_time_ms=(None if first is None else (first + 1) * DT),
                           total_spike_count=int(cp.sum()), control_total=int(cc.sum()),
                           total_difference=int(cp.sum() - cc.sum()),
                           per_cell_pearson=pear, matches_summary=bool(good),
                           any_threshold_flip=bool(first is not None))
    rec['C7_spike_statistics_recomputed'] = dict(passed=bool(ok), per_shift=per)

    # C8 aggregate window-mean contrast, including the sealed anchors
    ok = True
    anchors = {}
    for tag, fine, coarse in [('fine_0p03125_vs_0p015625', 'backward_euler_0p03125ms', 'backward_euler_0p015625ms'),
                              ('coarse_0p0625_vs_0p03125', 'backward_euler_0p0625ms', 'backward_euler_0p03125ms')]:
        with np.load(FOL / 'RUNS' / fine / 'NEURAL.npz') as zf:
            pf = zf['pool_rates'].copy()
        with np.load(FOL / 'RUNS' / coarse / 'NEURAL.npz') as zc:
            pc = zc['pool_rates'].copy()
        got, _, _ = window_mean_nrms(pf, pc)
        want = summ['sealed_aggregate_anchors'][tag]['window_mean']['normalized_RMS']
        good = abs(got - want) <= 1e-12 * max(1.0, want)
        ok = ok and good
        anchors[tag] = dict(recomputed=got, summary=want, matches=bool(good))
    per = {}
    for s in SHIFTS:
        if s == 0:
            continue
        got, mx, my = window_mean_nrms(z[s]['pool_rates'], z[0]['pool_rates'])
        want = summ['comparisons'][str(s)]['aggregate_200ms']['window_mean_pool_rates']['normalized_RMS']
        rel = np.abs(mx - my) / np.maximum(np.abs(my), 1e-12)
        within = int((rel <= 0.10).sum())
        want_within = summ['comparisons'][str(s)]['aggregate_200ms']['window_mean_pool_rates']['pools_within_10pct']
        good = abs(got - want) <= 1e-12 * max(1.0, want) and within == want_within
        ok = ok and good
        per[str(s)] = dict(recomputed_window_mean_nrms=got, summary=want, pools_within_10pct=within,
                           pools_total=int(rel.size), matches=bool(good),
                           pearson=float(np.corrcoef(mx, my)[0, 1]))
    rec['C8_aggregate_recomputed'] = dict(passed=bool(ok), sealed_anchors=anchors, per_shift=per,
                                          note='window-mean (aggregate) metric recomputed for every shift and for '
                                               'the sealed dt pairs, with the same formula')

    # C9 growth description and pre-registered decision rule
    pt = {s: (0.0 if s == 0 else nrms(z[s]['pool_rates'][:NMS], z[0]['pool_rates'][:NMS])) for s in SHIFTS}
    ph = 0.5 * D_DT
    it = 0.1 * D_DT
    if pt[SHIFTS[-1]] >= ph:
        reading = 'PHASE_EXPLAINS_DT'
    elif pt[SHIFTS[-1]] <= it and D_DT >= 0.25:
        reading = 'INTEGRATOR_EXPLAINS_DT'
    else:
        reading = 'MIXED'
    stored_pt = {s: (0.0 if s == 0 else summ['comparisons'][str(s)]['windows']['200']['pool_rates']['normalized_RMS'])
                 for s in SHIFTS}
    per_shift_ratio = {str(s): pt[s] / s for s in SHIFTS if s > 0}
    vals = np.asarray([pt[s] for s in SHIFTS if s > 0], float)
    shifts_pos = np.asarray([s for s in SHIFTS if s > 0], float)
    slope = float(np.polyfit(np.log(shifts_pos), np.log(np.maximum(vals, 1e-300)), 1)[0])
    good = (all(abs(pt[s] - stored_pt[s]) <= 1e-12 * max(1.0, stored_pt[s]) for s in SHIFTS)
            and reading == summ['verdict']['reading']
            and abs(summ['verdict']['rule']['phase_threshold'] - ph) < 1e-15
            and abs(summ['verdict']['rule']['integrator_threshold'] - it) < 1e-15)
    rec['C9_decision_rule'] = dict(
        passed=bool(good), recomputed_reading=reading, stored_reading=summ['verdict']['reading'],
        D_dt=D_DT, D_shift_0125ms=pt[SHIFTS[-1]], shift_over_dt=pt[SHIFTS[-1]] / D_DT,
        phase_threshold=ph, integrator_threshold=it, sealed_floor=0.25,
        per_shift_pointwise=per_shift_ratio, loglog_slope=slope,
        growth_summary=dict(roughly_proportional=summ['growth']['proportionality']['roughly_proportional'],
                            per_substep=summ['growth']['proportionality']['per_substep_normalized'],
                            crossing_measured=summ['growth']['crossing_measured'],
                            crossing_extrapolated=summ['growth']['crossing_extrapolated']),
        recomputed_slope_matches=bool(abs(slope - summ['growth']['loglog_fit']['slope']) < 1e-12),
        note='the decision rule is re-evaluated from independently recomputed nRMS values; the recorded reading must '
             'match the pre-registered rule and the thresholds')

    # C10 sealed read-only inputs untouched; this stage's own sources are reported separately because two of them
    # were corrected after the run (analysis/verification only) and are covered by POSTFREEZE_CHANGE_LEDGER.json
    before = js(OUT / 'INPUT_HASHES_BEFORE.json')
    after = {p: sha(p) for p in before}
    sealed = [p for p in before if not str(p).startswith(str(SRC))]
    own = [p for p in before if str(p).startswith(str(SRC))]
    bad_sealed = [p for p in sealed if before[p] != after[p]]
    changed_own = [p for p in own if before[p] != after[p]]
    ledger = js(OUT / 'POSTFREEZE_CHANGE_LEDGER.json') if (OUT / 'POSTFREEZE_CHANGE_LEDGER.json').exists() else {}
    led_files = {e.get('file') for e in ledger.get('edits', [])} | {Path(e.get('old', '')).name
                                                                    for e in ledger.get('edits', [])}
    rec['C10_sealed_inputs_unchanged'] = dict(
        passed=not bad_sealed, sealed_files_checked=len(sealed), sealed_mismatches=bad_sealed,
        own_sources_checked=len(own), own_sources_changed_after_freeze=changed_own,
        own_sources_documented_in_ledger=all(Path(p).name in led_files for p in changed_own),
        ledger=str(OUT / 'POSTFREEZE_CHANGE_LEDGER.json'),
        at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        note='the sealed numerics / numerics_followup / engine inputs are byte identical to the pre-registered '
             'hashes; this stage rewritten two of its own post-run analysis scripts (plot and verifier only), which '
             'are logged in POSTFREEZE_CHANGE_LEDGER.json and never touched a neural replay or a stored number')

    # C11 no physics / stochastic imports in this stage's sources
    import ast
    mods = {}
    ok = True
    for p in sorted(SRC.glob('*.py')):
        tree = ast.parse(p.read_text())
        got = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                got.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                got.add(node.module.split('.')[0])
        forbidden = {'mujoco', 'mujoco_viewer', 'torch', 'random'}
        bad_m = sorted(got & forbidden)
        ok = ok and not bad_m
        mods[p.name] = dict(imports=sorted(got), forbidden_found=bad_m)
    rec['C11_no_physics_no_stochastic'] = dict(passed=bool(ok), per_file=mods)

    # C12 figure present and consistent
    fig = js(OUT / 'FIGURE.json')
    p = Path(fig['figure'])
    w, h = png_size(p)
    ok = bool(p.exists() and w == fig['width'] and h == fig['height'] and p.stat().st_size > 20000 and w >= 900)
    rec['C12_figure'] = dict(passed=ok, path=str(p), width=w, height=h, size_bytes=p.stat().st_size,
                             declared=dict(width=fig['width'], height=fig['height'], size_bytes=fig['size_bytes']),
                             panels=fig['panels'])

    # C13 accounting and forbidden-claim guards
    forbidden_claims = []
    if summ.get('selected_timestep') is not None:
        forbidden_claims.append('selected_timestep')
    if summ.get('model_selection') is not None:
        forbidden_claims.append('model_selection')
    if summ.get('brain_model_invalid'):
        forbidden_claims.append('brain_model_invalid')
    if summ.get('biological_validity') != 'NOT_ESTABLISHED':
        forbidden_claims.append('biological_validity')
    ok = (not forbidden_claims and summ['physical_runs'] == 0 and summ['physical_steps'] == 0
          and summ['new_warm_runs'] == 0 and summ['full_neural_runs'] == 5
          and pre['physics_runs'] == 0 and pre['new_warm_runs'] == 0
          and pre.get('selected_timestep') is None)
    rec['C13_accounting_and_claim_guards'] = dict(
        passed=bool(ok), forbidden_claims=forbidden_claims, physical_runs=summ['physical_runs'],
        physical_steps=summ['physical_steps'], new_warm_runs=summ['new_warm_runs'],
        full_neural_runs=summ['full_neural_runs'], neural_steps_total=summ['neural_steps_total'],
        prefix_probe_streams=summ['prefix_probe_streams'],
        selected_timestep=summ['selected_timestep'], model_selection=summ['model_selection'],
        brain_model_invalid=summ['brain_model_invalid'], biological_validity=summ['biological_validity'])

    # C14 mechanism tests passed and no failure records remain
    fails = sorted(SRC.glob('FAILURE_*.json')) + sorted(OUT.glob('FAILURE_*.json'))
    rec['C14_tests_and_no_failures'] = dict(passed=bool(tests['passed'] and not fails),
                                            mechanism_tests_passed=tests['passed'],
                                            failure_files=[str(f) for f in fails],
                                            preregistered=bool((OUT / 'PREREGISTRATION.json').exists()),
                                            source_freeze=js(OUT / 'SOURCE_FREEZE.json'))

    failures = {k: v for k, v in rec.items() if not v['passed']}
    out = dict(stage='S5/P2 numerics_shift independent verification',
               verifier=str(SRC / 'verify.py'), verifier_sha256=sha(SRC / 'verify.py'),
               verified_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               wall_s=time.monotonic() - t0, checks=rec, failed_checks=sorted(failures),
               passed=not failures, scope='numeric reproduction, shift rule, bitwise control, metrics, aggregate '
               'contrast, decision rule and accounting guards; not a physical or biological acceptance',
               biological_validity='NOT_ESTABLISHED', selected_timestep=None, model_selection=None)
    (OUT / 'VERIFICATION.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + '\n')
    print('VERIFY', 'PASS' if out['passed'] else 'FAIL', sorted(failures), round(out['wall_s'], 2), flush=True)
    if failures:
        raise SystemExit(3)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ('FAILURE_verify_' + str(time.time_ns()) + '.json')).write_text(
            json.dumps(dict(traceback=traceback.format_exc()), indent=2) + '\n')
        raise
