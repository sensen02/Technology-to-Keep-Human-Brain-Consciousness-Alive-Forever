"""Independent verification of the readout-correction experiment.

Deliberately re-implements the support/objective quantities in local numpy code
(only the contact arrays are read from disk) instead of calling the same helpers
the runner used, re-fits the readout from the raw sealed arrays, and re-derives
the pre-registered verdict from the released trajectories. Any mismatch is a
FAIL in VERIFICATION.json; nothing is repaired here.
"""
import json
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]   # the sealed warm cache pickles need engine/ on the path
SRC = Path(__file__).resolve().parent
RT1 = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z')
O = RT1 / 'readout_fit'
CONFIGS = RT1 / 'numerics/CONFIGS'
SEALED = RT1 / 'numerics/RUNS'
TEACHER = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z/DATA/true_teacher')
IDENT = RT1 / 'sensory/CANDIDATES/TRANSLATOR_IDENTITIES.json'

PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
DTS = [(0.0005, 'dt_0p5ms'), (0.0000625, 'dt_0p0625ms')]
TAGS = ['R0', 'R1', 'R2']
GRAVITY = 9801.0
WEIGHT = 24.446354755878005
MASS = WEIGHT / GRAVITY
H0 = 1.2072528224791905
W_NONFOOT, W_FOOT, W_COM, W_TIME, W_FREEFALL = 2.0, 1.0, 1.0, 1.0, 1.0
RELEASE = 0.2


def js(p):
    return json.loads(Path(p).read_text())


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ local physics
def local_series(rows):
    r = np.asarray(rows, float)
    t = r[:, 0]
    foot = r[:, 1:7]
    nfs = r[:, 8]
    total = foot.sum(1) + nfs
    a = np.gradient(np.gradient(r[:, 10], t), t)
    a = np.convolve(a, np.ones(9) / 9, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(np.abs(total) > 1e-9, nfs / np.where(np.abs(total) > 1e-9, total, 1.), 0.)
    return dict(t=t, foot=foot, nfs=nfs, total=total, support=foot.sum(1) / WEIGHT,
                share=share, accel=a, bearing=(foot > 1e-9).sum(1), comz=r[:, 10])


def local_objective(rows, assist, time_shift):
    s = local_series(rows)
    t = s['t'] - time_shift
    support = s['support']
    share = np.nan_to_num(s['share'])
    total = s['total'] / WEIGHT
    free = ((total < 0.2) & (s['accel'] <= -0.5 * GRAVITY))
    pos = np.flatnonzero(s['nfs'] > 1e-9)
    t_nf = float(t[pos[0]]) if len(pos) else None
    a_nf = float(np.mean(share))
    a_foot = max(0., (assist['foot_support_mean'] - float(support.mean()))
                 / max(assist['foot_support_mean'], 1e-9))
    a_com = max(0., (assist['com_z_mean'] - float(s['comz'].min()))
                / max(assist['com_z_mean'], 1e-9))
    a_time = 0. if t_nf is None else max(0., 1. - t_nf / RELEASE)
    a_free = float(free.mean())
    J = (W_NONFOOT * a_nf + W_FOOT * a_foot + W_COM * a_com + W_TIME * a_time
         + W_FREEFALL * a_free)
    return float(J), dict(nonfoot_share=a_nf, foot_support_mean=float(support.mean()),
                          com_z_min=float(s['comz'].min()), free_fall_fraction=a_free,
                          first_nonfoot_s=t_nf), s


def local_gate(rows, assist):
    s = local_series(rows)
    resid = np.abs(MASS * s['accel'] - (s['total'] - WEIGHT)) / WEIGHT
    free = ((s['total'] / WEIGHT < 0.2) & (s['accel'] <= -0.5 * GRAVITY)).mean()
    support = s['support']
    share = np.nan_to_num(s['share'])
    checks = {
        'D1_no_nonfoot_ground_force': bool(np.max(np.asarray(rows)[:, 7]) == 0),
        'D2_no_free_fall': bool(free <= 0.0),
        'D3_foot_support_not_below_assist':
            bool(float(support.mean()) >= assist['foot_support_mean'] - 0.10),
        'D4_foot_support_never_collapses': bool(float(np.percentile(support, 10)) >= 0.5),
        'D4b_nonfoot_is_not_carrying_the_body': bool(float(np.nanmean(share)) <= 0.20),
        'D5_com_height_not_below_assist':
            bool(float(s['comz'].min()) >= assist['com_z_mean'] - 0.10),
        'D6_force_balance_consistent': bool(float(np.percentile(resid, 95)) <= 0.15),
        'D7_tripod_redundancy': bool(float(np.median(s['bearing'])) >= 3.),
        'D8_still_above_absolute_floor': bool(float(s['comz'].min()) >= 0.8 * H0),
    }
    return checks, dict(foot_support_mean=float(support.mean()),
                        com_z_min=float(s['comz'].min()),
                        nonfoot_share_mean=float(np.nanmean(share)))


WARM_ASSIST = {}


def load_assist():
    """Assisted-window reference of each sealed warm cache (only the assist dict is kept)."""
    for dt, tag in DTS:
        with (CONFIGS / tag / 'WARM_CACHE.pkl').open('rb') as f:
            cache = pickle.load(f)
        WARM_ASSIST[tag] = {ph: dict(cache[ph]['assist']) for ph in PHASES}
        del cache


# ------------------------------------------------------------------ checks
def check(out, name, ok, detail=None):
    out.append(dict(check=name, passed=bool(ok), detail=detail))
    return bool(ok)


def main():
    load_assist()
    checks = []
    before = js(O / 'INPUT_HASHES_BEFORE.json')
    after = js(O / 'INPUT_HASHES_AFTER.json')
    check(checks, 'sealed_inputs_unchanged_during_run', before == after,
          dict(inputs=len(before), changed=[p for p in before if before[p] != after.get(p)]))
    check(checks, 'sealed_inputs_unchanged_now', {p: sha(p) for p in before} == before,
          dict(inputs=len(before)))

    # The pre-run freeze is the record run.py asserted immediately before the first release.
    # A verifier/tool file may have been repaired afterwards; such repairs must be declared in
    # CHANGE_LEDGER.json with their old and new hash, and every other frozen file must still match.
    freeze = js(O / 'SOURCE_FREEZE.json')['files']
    now = {p: sha(p) for p in freeze}
    ledger = js(O / 'CHANGE_LEDGER.json') if (O / 'CHANGE_LEDGER.json').exists() else {}
    fixes = {f['path']: f for f in ledger.get('tool_fixes', [])}
    undeclared = [p for p in freeze if freeze[p] != now.get(p) and p not in fixes]
    bad_fix = [p for p, f in fixes.items()
               if p not in now or now[p] != f['new_sha256'] or freeze.get(p) != f['old_sha256']]
    check(checks, 'frozen_source_unchanged_except_declared_tool_fixes',
          not undeclared and not bad_fix,
          dict(files=len(freeze), declared_tool_fixes=sorted(fixes),
               undeclared_changes=undeclared, inconsistent_fix_records=bad_fix))
    core = {k: v for k, v in freeze.items()
            if Path(k).name in ('common.py', 'prereg.py', 'static.py', 'run.py')}
    check(checks, 'experiment_source_bit_identical_to_pre_run_freeze',
          all(core[k] == sha(k) for k in core), dict(files=len(core)))

    prereg = js(O / 'PREREGISTRATION.json')
    check(checks, 'preregistration_present', prereg['reading_rule'] is not None,
          dict(reading_rule='present'))

    # ---- 1. independent refit of the readout -------------------------------
    teacher_ctrl = arrays(TEACHER / 'BODY.npz')['ctrl']
    target = teacher_ctrl[400 + 2 * np.arange(1200)]
    src = arrays(CONFIGS / 'dt_0p5ms/WARM.npz')['command_source_rates']
    G = np.c_[src[:600], np.ones(600)]
    coef, *_ = np.linalg.lstsq(G, target[:600], rcond=None)
    W_ref, b_ref = coef[:66].T, coef[66]
    raw_full = src @ W_ref.T + b_ref
    scale_ref = float(np.sqrt((target[:600] ** 2).mean())
                      / np.sqrt((raw_full[:600] ** 2).mean()))
    ro = arrays(O / 'READOUT_R1.npz')
    check(checks, 'readout_matrix_bitwise_independent_refit',
          np.array_equal(ro['W_fit'], W_ref), dict(shape=list(ro['W_fit'].shape)))
    check(checks, 'readout_bias_bitwise_independent_refit',
          np.array_equal(ro['b_fit'], b_ref))
    check(checks, 'readout_R2_scale_matches_independent_refit',
          float(ro['scale_r2']) == scale_ref, dict(scale=scale_ref))
    r2t = 1 - ((target[:600] - raw_full[:600]) ** 2).sum() / ((target[:600] - target[:600].mean(0)) ** 2).sum()
    r2h = 1 - ((target[600:] - raw_full[600:]) ** 2).sum() / ((target[600:] - target[600:].mean(0)) ** 2).sum()
    st = js(O / 'READOUT_STATIC.json')
    check(checks, 'reported_R2_train_reproduced',
          abs(float(st['R1_r2_train']) - float(r2t)) < 1e-12, dict(recomputed=float(r2t)))
    check(checks, 'reported_R2_held_reproduced',
          abs(float(st['R1_r2_held']) - float(r2h)) < 1e-12, dict(recomputed=float(r2h)))

    ident = js(IDENT)
    old_matrix = np.array(ident['old_matrix'], float)
    limits = np.asarray(ident['limits'], float)

    # ---- 2. R0 bitwise reproduction of the sealed releases -----------------
    bit = js(O / 'R0_BITWISE.json')
    check(checks, 'R0_bitwise_all_ten_releases', len(bit) == 10 and all(v['passed'] for v in bit.values()),
          dict(count=len(bit)))
    for dt, tag in DTS:
        for ph in PHASES:
            sealed = SEALED / f'{tag}_paired_{ph}'
            mine = O / 'RUNS' / f'R0__{tag}__rel_{ph}'
            a = arrays(sealed / 'NATIVE.npz')['rows']
            b = arrays(mine / 'NATIVE.npz')['rows']
            ok = np.array_equal(a, b)
            ref = js(sealed / 'RESULT.json')
            check(checks, f'R0_native_exact__{tag}__{ph}', ok)
            check(checks, f'R0_J_exact__{tag}__{ph}',
                  ref['J'] == js(mine / 'RESULT.json')['J'],
                  dict(sealed=ref['J'], mine=js(mine / 'RESULT.json')['J']))
            for name in ['BODY.npz', 'NEURAL.npz']:
                x = arrays(sealed / name)
                y = arrays(mine / name)
                keys = list(x) if name == 'BODY.npz' else [k for k in x if k in y]
                check(checks, f'R0_{name}_exact__{tag}__{ph}',
                      all(np.array_equal(x[k], y[k]) for k in keys),
                      dict(keys=len(keys)))
    summary = js(O / 'SUMMARY.json')
    anchors = {}
    for row in js(RT1 / 'numerics/SUMMARY.json')['results']:
        for dt, tag in DTS:
            if row['dt_s'] == dt and row['release_time_s'] in PHASES and row['label'] == f'{tag}_paired_{row["release_time_s"]}':
                anchors[f'{tag}__{row["release_time_s"]}'] = row['J']
    mine_j = {k: js(O / 'RUNS' / f'R0__{v[0]}__rel_{v[1]}' / 'RESULT.json')['J']
              for k, v in {f'{tag}__{ph}': (tag, ph) for dt, tag in DTS for ph in PHASES}.items()}
    check(checks, 'R0_matches_numerics_SUMMARY_anchors',
          bool(anchors) and anchors == mine_j,
          dict(anchors=len(anchors), mismatches=[k for k in anchors if anchors[k] != mine_j.get(k)]))

    # ---- 3. the fitted readout is what was actually applied ---------------
    # Recompute the declared readout from the recorded source rates. The runner evaluates
    # W @ rates per step (matrix-vector) while this verifier evaluates the same product as a
    # row block (matrix-matrix), so the agreement is checked at 1e-12 absolute, not bitwise.
    applied_err, applied_overlimit = [], 0
    for readout in ('R1', 'R2'):
        sc = 1.0 if readout == 'R1' else float(ro['scale_r2'])
        for dt, tag in DTS:
            for ph in PHASES:
                z = arrays(O / 'RUNS' / f'{readout}__{tag}__rel_{ph}' / 'NEURAL.npz')
                src_rates = z['source_rates']
                expect = np.clip(sc * (src_rates @ ro['W_fit'].T + ro['b_fit']), -limits, limits)
                applied_err.append(float(np.abs(expect - z['applied']).max()))
                applied_overlimit += int(np.sum(np.abs(z['applied']) > limits[None, :] + 0))
    check(checks, 'applied_command_equals_declared_readout', max(applied_err) <= 1e-12,
          dict(releases=len(applied_err), max_abs_difference=float(max(applied_err)),
               elements_above_limit=applied_overlimit))

    # ---- 4. recompute J and gates independently from the released arrays ---
    jerr, gate_mismatch = [], []
    for row in summary['row_level']:
        readout, tag, ph = row['readout'], row['dt_tag'], row['release_time_s']
        folder = O / 'RUNS' / f'{readout}__{tag}__rel_{ph}'
        arr = arrays(folder / 'NATIVE.npz')['rows']
        assist = WARM_ASSIST[tag][ph]
        # the runner shifts the stored time by -(ph-1.3) and the library subtracts 1.3 again,
        # so the first positive non-foot time is measured from the release phase itself.
        J_loc, parts_loc, _ = local_objective(arr, assist, ph)
        jerr.append(abs(J_loc - float(row['J'])))
        g_loc, gl = local_gate(arr, assist)
        stored = js(folder / 'RESULT.json')['validation']['checks']
        if g_loc != stored:
            gate_mismatch.append(dict(label=f'{readout}__{tag}__rel_{ph}', local=g_loc, stored=stored))
    check(checks, 'objective_recomputed_identically', max(jerr) <= 1e-12,
          dict(max_abs_error=float(max(jerr)), releases=len(jerr)))
    check(checks, 'D_gates_recomputed_identically', not gate_mismatch,
          dict(mismatches=gate_mismatch))

    # ---- 5. pre-registered verdict re-derived from the per-phase numbers ---
    agg = summary['per_condition']
    redo = {}
    for readout in ('R1', 'R2'):
        jwin, dgate = {}, {}
        for dt, tag in DTS:
            base = agg[f'R0__{tag}']
            new = agg[f'{readout}__{tag}']
            wins = [n['phase'] for n, o in zip(new['per_phase'], base['per_phase']) if n['J'] < o['J']]
            jwin[tag] = len(wins)
            dgate[tag] = (base['phases_passing_all_D'], new['phases_passing_all_D'])
        a1 = all(v >= 4 for v in jwin.values())
        a2 = all(b[1] > b[0] for b in dgate.values())
        redo[readout] = dict(A1=a1, A2=a2, wins=jwin, D=dgate,
                             claim=summary['verdict'][readout]['physical_improvement_established'])
        check(checks, f'verdict_recomputed__{readout}',
              bool(a1 and a2) == bool(redo[readout]['claim']),
              dict(recomputed_A1=a1, recomputed_A2=a2, claimed=redo[readout]['claim']))
    check(checks, 'no_condition_claimed_as_supported',
          summary['boundaries']['autonomous_support'] == 'NOT_CLAIMED'
          and summary['boundaries']['walking'] == 'NOT_CLAIMED',
          dict(boundaries=summary['boundaries']))

    # ---- 6. sanity: the readout really changed the physics ----------------
    differ = 0
    for dt, tag in DTS:
        for ph in PHASES:
            a = arrays(O / 'RUNS' / f'R0__{tag}__rel_{ph}' / 'NATIVE.npz')['rows']
            b = arrays(O / 'RUNS' / f'R1__{tag}__rel_{ph}' / 'NATIVE.npz')['rows']
            differ += int(not np.array_equal(a, b))
    check(checks, 'fitted_readout_changes_the_release_trajectory', differ == 10,
          dict(phases_differing=differ, total=10))

    # ---- 7. the figure exists and is a real PNG ---------------------------
    f = O / 'FIGURE_readout_fit.png'
    ok = f.exists() and f.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and f.stat().st_size > 40000
    check(checks, 'figure_present_and_valid_png', ok,
          dict(bytes=f.stat().st_size if f.exists() else 0))

    result = dict(
        verified=str(not any(not c['passed'] for c in checks)),
        passed=int(sum(c['passed'] for c in checks)), total=len(checks),
        checks=checks, independent_refit=dict(scale=scale_ref, r2_train=float(r2t),
                                              r2_held=float(r2h)),
        verdict_recomputed=redo,
        note='This verifier re-implements the support series, the objective and the D gates in '
             'local numpy code and refits the readout from the raw sealed arrays; it does not '
             'call the runner. No failure is repaired here.')
    dump(O / 'VERIFICATION.json', result)
    print('VERIFIED', result['verified'], result['passed'], '/', result['total'], flush=True)
    for c in checks:
        if not c['passed']:
            print('  FAIL', c['check'], c['detail'], flush=True)
    if not result['verified']:
        sys.exit(1)


if __name__ == '__main__':
    main()
