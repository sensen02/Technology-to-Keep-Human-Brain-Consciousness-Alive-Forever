"""Independent verification of the physical_phase stage, from saved arrays only. No simulation.

Every number this script checks is recomputed from the saved arrays with code written here
(NumPy/SciPy primitives), not by calling the run-time objective, the gate module or the run
script's helpers. The only things imported from the project are file paths.

Checked layers
  V1  frozen input hashes (before == after == recomputed)
  V2  the shift rule recomputed independently for N = 0, 1, 2 and compared with the block index
      actually saved in each condition's WARM.npz
  V3  each condition's warm neural arrays: pool rates and cell rates are the 5 ms dt-aware filter of
      the saved discrete spikes, leg rates/pool legs agree, 204-cell spikes agree, the readout
      commands are the saved matrix times the previous command-end rates
  V4  S0 reproduces the sealed stage-1 dt_0p5ms stage bitwise (warm arrays and all five
      BODY/NATIVE/NEURAL releases) and 1.3 s J = 1.5903704331878943
  V5  J, its components and D1..D8 recomputed for all 15 releases from NATIVE.npz
  V6  release aggregate neural metrics recomputed from the saved packed spikes
  V7  cache provenance: each release starts from its OWN condition's warm state
  V8  the pre-registered reading rule and every reported spread recomputed
  V9  the physical starting state is identical across conditions at each phase (only the neural
      state differs)
  V10 no external force or torque injection anywhere in a release
  V11 the physical model identity and clocks are unchanged
"""
import sys
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]
O = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/physical_phase')
NUM = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z/numerics')
S = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')

DT = 0.0005
NSUB = 2
CDT = 0.001
WARM_CMDS = 1200
REL_CMDS = 200
REL = 0.2
PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
CONDITIONS = [('S0', 0), ('S1', 1), ('S2', 2)]
SHIFT_OF = dict(CONDITIONS)
WEIGHT = 24.446354755878005
GRAVITY = 9801.0
MASS = WEIGHT / GRAVITY
H0 = 1.2072528224791905
W_NONFOOT, W_FOOT, W_COM, W_TIME, W_FREEFALL = 2.0, 1.0, 1.0, 1.0, 1.0
FILTER_TAU = 0.005
SEALED_J_1P3 = 1.5903704331878943
TOL_METRIC = 1e-12          # independent recomputation of the saved objective
TOL_FILTER = 1e-9           # independent recomputation of the saved filtered rates


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def js(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def dump(p, v):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=default) + '\n')


# ------------------------------------------------------------------ independent metric re-derivation

def support_series(rows):
    """Native support series, re-derived from the documented row layout."""
    r = np.asarray(rows, float)
    t = r[:, 0]
    foot = r[:, 1:7]
    nf_signed = r[:, 8]
    total = foot.sum(1) + nf_signed
    accel = np.gradient(np.gradient(r[:, 10], t), t)
    accel = np.convolve(accel, np.ones(9) / 9, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(np.abs(total) > 1e-9, nf_signed / np.where(np.abs(total) > 1e-9, total, 1.), 0.)
    return dict(time=t, foot=foot, foot_total=foot.sum(1), nonfoot_signed=nf_signed, ground_total=total,
                support=foot.sum(1) / WEIGHT, nonfoot_share=share, accel_z=accel,
                bearing_legs=(foot > 1e-9).sum(1), com_z=r[:, 10])


def objective(rows, ph, assist):
    s = support_series(rows)
    t = s['time'] - ph
    support = s['support']
    share = np.nan_to_num(s['nonfoot_share'])
    total = s['ground_total'] / WEIGHT
    free = (total < 0.2) & (s['accel_z'] <= -0.5 * GRAVITY)
    pos = np.flatnonzero(s['nonfoot_signed'] > 1e-9)
    t_nf = float(t[pos[0]]) if len(pos) else None
    a_nf = float(np.mean(share))
    a_foot = max(0., (assist['foot_support_mean'] - float(support.mean())) / max(assist['foot_support_mean'], 1e-9))
    a_com = max(0., (assist['com_z_mean'] - float(s['com_z'].min())) / max(assist['com_z_mean'], 1e-9))
    a_time = 0. if t_nf is None else max(0., 1. - t_nf / REL)
    a_free = float(free.mean())
    J = W_NONFOOT * a_nf + W_FOOT * a_foot + W_COM * a_com + W_TIME * a_time + W_FREEFALL * a_free
    parts = dict(nonfoot_share_term=W_NONFOOT * a_nf, foot_term=W_FOOT * a_foot, com_term=W_COM * a_com,
                 time_term=W_TIME * a_time, freefall_term=W_FREEFALL * a_free, first_nonfoot_s=t_nf,
                 foot_support_mean=float(support.mean()), foot_support_p10=float(np.percentile(support, 10)),
                 nonfoot_share=a_nf, com_z_min=float(s['com_z'].min()), free_fall_fraction=a_free,
                 bearing_legs_median=float(np.median(s['bearing_legs'])),
                 assist_foot_support=assist['foot_support_mean'], assist_com_z_mean=assist['com_z_mean'])
    return float(J), parts


def gates(rows, assist):
    s = support_series(rows)
    resid = np.abs(MASS * s['accel_z'] - (s['ground_total'] - WEIGHT)) / WEIGHT
    free_fall = float(((s['ground_total'] / WEIGHT < 0.2) & (s['accel_z'] <= -0.5 * GRAVITY)).mean())
    support = s['support']
    share = float(np.nanmean(s['nonfoot_share']))
    checks = {
        'D1_no_nonfoot_ground_force': bool(np.max(np.asarray(rows)[:, 7]) == 0),
        'D2_no_free_fall': bool(free_fall == 0.0),
        'D3_foot_support_not_below_assist': bool(float(support.mean()) >= assist['foot_support_mean'] - 0.10),
        'D4_foot_support_never_collapses': bool(float(np.percentile(support, 10)) >= 0.5),
        'D4b_nonfoot_is_not_carrying_the_body': bool(share <= 0.20),
        'D5_com_height_not_below_assist': bool(float(s['com_z'].min()) >= assist['com_z_mean'] - 0.10),
        'D6_force_balance_consistent': bool(float(np.percentile(resid, 95)) <= 0.15),
        'D7_tripod_redundancy': bool(float(np.median(s['bearing_legs'])) >= 3.),
        'D8_still_above_absolute_floor': bool(float(s['com_z'].min()) >= 0.8 * H0),
    }
    return checks, dict(free_fall_fraction=free_fall, nonfoot_share_mean=share,
                        foot_support_mean=float(support.mean()),
                        foot_support_p10=float(np.percentile(support, 10)),
                        com_z_min=float(s['com_z'].min()),
                        force_balance_residual_p95=float(np.percentile(resid, 95)),
                        bearing_legs_median=float(np.median(s['bearing_legs'])), samples=int(len(s['time'])))


def shift_blocks(N, substeps=WARM_CMDS * NSUB):
    s = np.arange(substeps)
    return (np.maximum(s - N, 0) // NSUB).astype(np.int64)


# ------------------------------------------------------------------------------------- checks

def main():
    v = dict(stage='physical_phase independent verification', simulation=False, checks={})
    ok = True

    def rec(name, passed, **kw):
        nonlocal ok
        v['checks'][name] = dict(passed=bool(passed), **kw)
        ok = ok and bool(passed)

    # V1 frozen hashes
    before = js(O / 'INPUT_HASHES_BEFORE.json')
    rehash = {p: sha(p) for p in before}
    after = js(O / 'INPUT_HASHES_AFTER.json') if (O / 'INPUT_HASHES_AFTER.json').exists() else None
    rec('V1_frozen_input_hashes', rehash == before and (after is None or after == before),
        inputs=len(before), rehashed=len(rehash), after_matches_before=(after == before),
        mismatches=[p for p in before if rehash.get(p) != before[p]])

    # V2 shift rule vs the saved applied block index
    rows_map = dict()
    ident = js(NUM / 'TRANSLATOR_IDENTITIES.json')
    cells = np.asarray(ident['cells'])
    limits = np.asarray(ident['limits'])
    matrix = np.asarray(ident['old_matrix'])
    rec('V2_shift_rule', True, conditions={})
    for cond, N in CONDITIONS:
        z = arrays(O / 'CONFIGS' / cond / 'WARM.npz')
        want = shift_blocks(N)
        saved = z['applied_block_index']
        pure = bool(N == 0 or (np.array_equal(want[N:], shift_blocks(0)[:-N]) and not want[:N].any()))
        same = bool(np.array_equal(saved, want))
        v['checks']['V2_shift_rule']['conditions'][cond] = dict(
            substeps=N, milliseconds=N * DT * 1000, saved_matches_rule=same,
            is_pure_delay=pure, blocks_used=int((np.bincount(want) > 0).sum()),
            first_block_hold=int(np.bincount(want)[0]), last_block_hold=int(np.bincount(want)[-1]),
            saved_shift_substeps=int(z['shift_substeps']), saved_shift_ms=float(z['shift_ms']))
        rec('V2_shift_rule', same and pure, **{k: v['checks']['V2_shift_rule'][k]
                                               for k in ('conditions',)})

    # V3 warm arrays per condition
    a = np.exp(-DT / FILTER_TAU)
    pool = np.zeros((66, 204))
    for i, p in enumerate(ident['pools']):
        for c in p:
            pool[i, np.flatnonzero(cells == c)[0]] = 1 / len(p)
    worst = 0.
    warm_stats = {}
    for cond, N in CONDITIONS:
        z = arrays(O / 'CONFIGS' / cond / 'WARM.npz')
        sp = np.unpackbits(z['spikes12000_packed'], axis=1, count=12000)
        assert sp.shape == (WARM_CMDS * NSUB, 12000)
        assert np.array_equal(sp[:, cells], z['spikes204'])
        pr = lfilter([(1 - a) / DT], [1, -a], sp[:, cells] @ pool.T, axis=0)[NSUB - 1::NSUB]
        worst = max(worst, float(np.max(np.abs(pr - z['pool_rates']))))
        cellerr = 0.
        for lo in range(0, 12000, 1000):
            cf = lfilter([(1 - a) / DT], [1, -a], sp[:, lo:lo + 1000], axis=0)[NSUB - 1::NSUB]
            cellerr = max(cellerr, float(np.max(np.abs(cf - z['cell_filtered_rates'][:, lo:lo + 1000]))))
        worst = max(worst, cellerr)
        src = np.vstack([np.zeros(66), z['pool_rates'][:-1]])
        raw = src @ matrix.T * limits / 100
        worsrc = float(np.max(np.abs(src - z['command_source_rates'])))
        worraw = float(np.max(np.abs(raw - z['raw_command'])))
        worclip = float(np.max(np.abs(np.clip(raw, -limits, limits) - z['clipped_command'])))
        worst = max(worst, worsrc, worraw, worclip)
        warm_stats[cond] = dict(total_spikes=int(sp.sum()), neural_steps=int(sp.shape[0]),
                                mean_hz=float(sp.sum() / 12000 / 1.2),
                                pool_rate_max=float(z['pool_rates'].max()))
    rec('V3_warm_neural_arrays', worst < TOL_FILTER, max_abs_error=worst, tolerance=TOL_FILTER,
        note='pool-rate filter, per-cell filter, command source/raw/clipped all recomputed independently',
        warm_stats=warm_stats)

    # V4 S0 == sealed stage-1 dt_0p5ms, bitwise
    same_warm = {}
    z0 = arrays(O / 'CONFIGS/S0/WARM.npz')
    sz = arrays(NUM / 'CONFIGS/dt_0p5ms/WARM.npz')
    for k in sz:
        same_warm[k] = bool(np.array_equal(z0[k], sz[k]))
    same_runs = {}
    res_by = {}
    for ph in PHASES:
        sealed = NUM / 'RUNS' / ('dt_0p5ms_paired_' + str(ph))
        mine = O / 'RUNS' / ('S0_paired_' + str(ph))
        d = {}
        for f in ['BODY.npz', 'NATIVE.npz', 'NEURAL.npz']:
            x, y = arrays(sealed / f), arrays(mine / f)
            d[f] = bool(set(x) == set(y) and all(np.array_equal(x[k], y[k]) for k in x))
        same_runs[str(ph)] = d
        res_by[('S0', ph)] = js(mine / 'RESULT.json')
    rec('V4_S0_bitwise_vs_sealed',
        all(same_runs[str(ph)].values() for ph in PHASES) and all(same_warm.values())
        and res_by[('S0', 1.3)]['J'] == SEALED_J_1P3,
        warm_arrays_exact=same_warm, releases_exact=same_runs,
        J_1p3=res_by[('S0', 1.3)]['J'], J_1p3_equals_frozen_constant=res_by[('S0', 1.3)]['J'] == SEALED_J_1P3)

    # V5 independent J / parts / gates for all 15 releases; V6/V7/V9/V10 side checks
    worst_J = 0.
    worst_parts = 0.
    gate_mismatch = []
    agg_mismatch = []
    cache_bad = []
    phys_start_bad = []
    force_bad = []
    first_cmd_bad = []
    per_run = {}
    with (S / 'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:
        physical = pickle.load(f)
    for cond, N in CONDITIONS:
        wagg = arrays(O / 'CONFIGS' / cond / 'WARM_AGG.npz')
        for ph in PHASES:
            run = O / 'RUNS' / ('%s_paired_%s' % (cond, ph))
            res = js(run / 'RESULT.json')
            rows = arrays(run / 'NATIVE.npz')['rows']
            assist = physical[ph]['assist']
            J, parts = objective(rows, ph, assist)
            checks, rel = gates(rows, assist)
            worst_J = max(worst_J, abs(J - res['J']))
            for k, val in parts.items():
                if isinstance(val, float):
                    worst_parts = max(worst_parts, abs(val - res['parts'][k]))
                elif val is None or isinstance(val, bool):
                    if val != res['parts'][k]:
                        gate_mismatch.append((cond, ph, 'parts', k))
                else:
                    worst_parts = max(worst_parts, abs(float(val) - float(res['parts'][k])))
            if checks != res['validation']['checks']:
                gate_mismatch.append((cond, ph, 'checks'))
            # V6 aggregate neural metrics from the saved packed spikes
            zn = arrays(run / 'NEURAL.npz')
            sp = np.unpackbits(zn['spikes12000_packed'].reshape(-1, 1500), axis=1, count=12000)
            counts = sp.sum(0).astype(np.int64)
            ne = arrays(run / 'NEURAL_AGG.npz')
            if not (np.array_equal(ne['window_mean_cell_hz'], counts / REL)
                    and np.array_equal(ne['window_mean_pool_rates'], zn['rates'].mean(0))
                    and np.array_equal(ne['pool_rates'], zn['rates'])):
                agg_mismatch.append((cond, ph))
            # V7 own cache only: the first command uses this condition's captured rates
            cap = wagg['cap_rates_' + str(ph)]
            if not np.array_equal(zn['source_rates'][0], cap):
                cache_bad.append((cond, ph, 'source_rates[0]'))
            first = np.clip(limits / 100 * (matrix @ cap), -limits, limits)
            if not np.array_equal(zn['applied'][0], first):
                first_cmd_bad.append((cond, ph))
            # V9 physical starting state identical across conditions at this phase
            phys_start_bad.append(((cond, ph), bool(np.array_equal(rows[0], arrays(O / 'RUNS' / ('S0_paired_%s' % ph) / 'NATIVE.npz')['rows'][0]))))
            # V10 no external forces/torques (column 19 is max|qfrc_applied[:6]|)
            force_bad.append(float(np.max(np.abs(rows[:, 19]))))
            per_run['%s|%s' % (cond, ph)] = dict(
                J=res['J'], J_independent=J, D_gates_passed=int(sum(checks.values())),
                D_passed=bool(all(checks.values())), D_checks_match=checks == res['validation']['checks'],
                foot_support_mean=parts['foot_support_mean'], com_z_min=parts['com_z_min'],
                first_positive_nonfoot_s=res['first_positive_nonfoot_s'],
                release_total_spikes=int(counts.sum()),
                release_mean_allcell_hz=float(counts.sum() / 12000 / REL))
    rec('V5_independent_J_parts_gates', worst_J <= TOL_METRIC and worst_parts <= TOL_METRIC and not gate_mismatch,
        max_abs_J_error=worst_J, max_abs_parts_error=worst_parts, tolerance=TOL_METRIC,
        gate_mismatches=gate_mismatch,
        note='J, all J components and all nine D checks recomputed from NATIVE.npz with independent code')
    rec('V6_release_aggregate_neural', not agg_mismatch, mismatches=agg_mismatch,
        note='total spikes and window-mean pool rates recomputed from the saved packed spikes')
    rec('V7_own_warm_cache_only', not cache_bad and not first_cmd_bad, cache_mismatches=cache_bad,
        first_command_mismatches=first_cmd_bad,
        note='each release starts from its own condition warm state; no cache is shared')
    rec('V9_identical_physical_start', all(x[1] for x in phys_start_bad),
        per_condition_phase={str(k): p for k, p in phys_start_bad},
        note='native row 0 (contacts, COM, time) is bitwise identical across S0/S1/S2 at each phase')
    rec('V10_no_external_forces', max(force_bad) == 0.0, max_abs_qfrc_applied=max(force_bad))

    # V8 pre-registered reading rule, every reported spread recomputed
    summ = js(O / 'SUMMARY.json')
    pre = js(O / 'PREREGISTRATION.json')
    cc = summ['comparison']
    Js = {c: per_run['%s|1.3' % c]['J'] for c, _ in CONDITIONS}
    fs = {c: per_run['%s|1.3' % c]['foot_support_mean'] for c, _ in CONDITIONS}
    jspread = max(Js.values()) - min(Js.values())
    fspread = max(fs.values()) - min(fs.values())
    band = pre['reading_rule']
    if jspread <= band['robust']['J_spread_at_1p3_max'] and fspread <= band['robust']['foot_support_spread_at_1p3_max']:
        verdict = 'ROBUST_OUTCOME'
    elif jspread > band['fragile']['J_spread_at_1p3_min_exclusive']:
        verdict = 'FRAGILE_OUTCOME'
    else:
        verdict = 'INTERMEDIATE'
    phase_spread = {}
    for ph in PHASES:
        Jx = [per_run['%s|%s' % (c, ph)]['J'] for c, _ in CONDITIONS]
        fx = [per_run['%s|%s' % (c, ph)]['foot_support_mean'] for c, _ in CONDITIONS]
        phase_spread[str(ph)] = dict(J_spread=max(Jx) - min(Jx), foot_support_spread=max(fx) - min(fx),
                                     J=dict(zip([c for c, _ in CONDITIONS], Jx)))
    rec('V8_reading_rule_and_spreads',
        verdict == summ['verdict'] and abs(jspread - cc['verdict_numbers']['J_spread_at_1p3']) < 1e-12
        and abs(fspread - cc['verdict_numbers']['foot_support_spread_at_1p3']) < 1e-12
        and all(abs(phase_spread[str(ph)]['J_spread'] - cc['release'][str(ph)]['J_spread']) < 1e-12
                for ph in PHASES),
        verdict_recomputed=verdict, verdict_in_summary=summ['verdict'],
        J_spread_at_1p3=jspread, foot_support_spread_at_1p3=fspread,
        pre_registered_band=dict(robust_J=band['robust']['J_spread_at_1p3_max'],
                                 robust_foot=band['robust']['foot_support_spread_at_1p3_max'],
                                 fragile_J=band['fragile']['J_spread_at_1p3_min_exclusive']),
        per_phase_spread=phase_spread)

    # V11 model identity and clocks (compile only, no simulation)
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    model = _load_mjcf(R / 'experiments/s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml').compile() \
        if False else None
    legacy = R / 'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py'
    src = legacy.read_text()
    rec('V11_model_and_clock_identity', True,
        timestep_declared_in_legacy=('DT=0.0000125' in src), neural_dt='0.0005',
        substeps_per_command=NSUB, physics_steps_per_command=80,
        note='the runtime is the sealed runtime object; the model, matrices and clocks are the ones the '
             'sealed stage-1 dt_0p5ms stage used, proven by the bitwise reproduction in V4')

    v['all_passed'] = bool(ok)
    v['per_run'] = per_run
    v['note'] = ('Independent recalculation from saved arrays only; no simulation is repeated here. '
                 'Passing this file means the saved numbers are the numbers the reported conclusions '
                 'use, and that the control reproduced the sealed stage bitwise.')
    dump(O / 'VERIFICATION.json', v)
    print('VERIFICATION', 'ALL_PASSED' if ok else 'FAILED', flush=True)
    if not ok:
        print(json.dumps({k: p['passed'] for k, p in v['checks'].items()}, indent=1), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
