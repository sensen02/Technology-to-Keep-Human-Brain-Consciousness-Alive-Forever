"""Independent verification of the whole-population readout test.

Deliberately re-implements the readout refit, the 5 ms cell filter, the support/objective
quantities and the D gates in local numpy code (only the release contact arrays are read
from disk), and re-derives the pre-registered verdict.  Any mismatch is a FAIL in
VERIFICATION.json; nothing is repaired here.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path[:0] = [str(R), str(R / 'tools')]   # the sealed warm cache pickles need engine/ on the path
SRC = Path(__file__).resolve().parent
RT1 = Path('/home/sensen/Desktop/engine修复验收/s5-model-correction/20261106T063313Z')
O = RT1 / 'pop_readout'
CONFIGS = RT1 / 'numerics/CONFIGS'
SEALED = RT1 / 'numerics/RUNS'
PRIOR = RT1 / 'readout_fit/RUNS'
TEACHER = Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z/DATA/true_teacher')

PHASES = [1.225, 1.275, 1.3, 1.325, 1.375]
DTS = [(0.0005, 'dt_0p5ms'), (0.0000625, 'dt_0p0625ms')]
TAGS = ['R0', 'P1', 'P2', 'P3']
GAMMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
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
                                  default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                                  else x.item()) + '\n')


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
    for dt, tag in DTS:
        with (CONFIGS / tag / 'WARM_CACHE.pkl').open('rb') as f:
            cache = pickle.load(f)
        WARM_ASSIST[tag] = {ph: dict(cache[ph]['assist']) for ph in PHASES}
        del cache


# ------------------------------------------------------------------ checks
def check(out, name, ok, detail=None):
    out.append(dict(check=name, passed=bool(ok), detail=detail))
    return bool(ok)


# ------------------------------------------------------------------ independent refit
def ridge_dual(X, Y, alpha):
    n = len(X)
    A = X @ X.T
    A[np.diag_indices_from(A)] += float(alpha)
    return (X.T @ np.linalg.solve(A, Y)).T.copy()


def cv_select(Xfit, Yfit, folds=5):
    """Independent blocked-CV alpha selection; returns (gamma, curve)."""
    s = float((Xfit * Xfit).sum() / len(Xfit))
    n = len(Xfit)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    curve = []
    for g in GAMMAS:
        alpha = g * s
        pred = np.zeros_like(Yfit)
        for f in range(folds):
            tr = np.r_[np.arange(0, bounds[f]), np.arange(bounds[f + 1], n)]
            va = np.arange(bounds[f], bounds[f + 1])
            W = ridge_dual(Xfit[tr], Yfit[tr], alpha)
            pred[va] = Xfit[va] @ W.T
        ss = ((Yfit - pred) ** 2).sum()
        tot = ((Yfit - Yfit.mean(0)) ** 2).sum()
        curve.append(dict(gamma=g, alpha=alpha, cv_r2=float(1 - ss / tot)))
    best = max(curve, key=lambda c: c['cv_r2'])
    return best['gamma'], curve, s


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    load_assist()
    checks = []
    before = js(O / 'INPUT_HASHES_BEFORE.json')
    after = js(O / 'INPUT_HASHES_AFTER.json')
    check(checks, 'sealed_inputs_unchanged_during_run', before == after,
          dict(inputs=len(before), changed=[p for p in before if before[p] != after.get(p)]))
    check(checks, 'sealed_inputs_unchanged_now', {p: sha(p) for p in before} == before,
          dict(inputs=len(before)))

    freeze = js(O / 'SOURCE_FREEZE.json')['files']
    now = {p: sha(p) for p in freeze}
    ledger = js(O / 'CHANGE_LEDGER.json') if (O / 'CHANGE_LEDGER.json').exists() else {}
    fixes = {f['path']: f for f in ledger.get('tool_fixes', [])}
    undeclared, bad_fix = [], []
    for p in freeze:
        if freeze[p] != now.get(p):
            f = fixes.get(p)
            if f is None or f.get('old_sha256') != freeze[p] or f.get('new_sha256') != now[p]:
                undeclared.append(p)
    for p, f in fixes.items():
        if p in now and f.get('new_sha256') is not None and now[p] != f['new_sha256']:
            bad_fix.append(p)
    check(checks, 'frozen_source_unchanged_except_declared_tool_fixes',
          not undeclared and not bad_fix,
          dict(files=len(freeze), declared_tool_fixes=sorted(Path(p).name for p in fixes),
               undeclared_changes=undeclared, inconsistent_fix_records=bad_fix))

    prereg = js(O / 'PREREGISTRATION.json')
    check(checks, 'preregistration_and_amendments_present',
          'amendments' in prereg and 'reading_rule' in prereg,
          dict(amendments=[a['id'] for a in prereg.get('amendments', [])],
               physics_run_before_amendment=all(
                   a['physics_run_before'] is False for a in prereg.get('amendments', []))))
    amend = js(O / 'AMENDMENT_APPLIED.json')
    check(checks, 'amendment_recorded_before_any_release',
          amend['physics_releases_run_before'] == 0
          and amend['heldout_used_for_any_choice'] is False,
          dict(physics_releases_run_before=amend['physics_releases_run_before']))
    check(checks, 'pre_amendment_artifacts_retained',
          all((O / 'PREAMENDMENT' / n).exists() for n in
              ('READOUT_STATIC.json', 'READOUT_FIT.npz', 'static_PREAMENDMENT.log')),
          dict(dir=str(O / 'PREAMENDMENT')))

    # ---- 1. independent refit of all three fitted readouts -----------------------
    teacher_ctrl = arrays(TEACHER / 'BODY.npz')['ctrl']
    target = teacher_ctrl[400 + 2 * np.arange(1200)]
    z05 = arrays(CONFIGS / 'dt_0p5ms/WARM.npz')
    src = np.vstack([np.zeros((1, 12000)), z05['cell_filtered_rates'][:-1]])
    ro = arrays(O / 'READOUT_FIT.npz')
    limits = ro['limits']
    Xf, Yf = src[:600], target[:600]
    m_fit, Ybar = Xf.mean(0), Yf.mean(0)
    refit = {}
    gamma1, curve1, s1 = cv_select(Xf, Yf)
    W1r = ridge_dual(Xf, Yf, gamma1 * s1)
    refit['P1'] = dict(gamma=gamma1, held=r2(target[600:], (src @ W1r.T)[600:]))
    cells = np.asarray(ro['cells204'], int)
    X204 = src[:, cells]
    m3 = X204[:600].mean(0)
    gamma2, curve2, s2 = cv_select(Xf - m_fit, Yf - Ybar)
    W2r = ridge_dual(Xf - m_fit, Yf - Ybar, gamma2 * s2)
    b2r = Ybar - W2r @ m_fit
    refit['P2'] = dict(gamma=gamma2, held=r2(target[600:], (src @ W2r.T + b2r)[600:]))
    gamma3, curve3, s3 = cv_select(X204[:600] - m3, Yf - Ybar)
    W3r = ridge_dual(X204[:600] - m3, Yf - Ybar, gamma3 * s3)
    b3r = Ybar - W3r @ m3
    refit['P3'] = dict(gamma=gamma3,
                       held=r2(target[600:], (X204 @ W3r.T + b3r)[600:]))
    stat = js(O / 'READOUT_STATIC.json')['fits']
    for name, key in (('P1', 'W_P1'), ('P2', 'W_P2'), ('P3', 'W_P3')):
        Wr = {'P1': W1r, 'P2': W2r, 'P3': W3r}[name]
        d = float(np.abs(np.asarray(ro[key]) - Wr).max() / max(np.abs(Wr).max(), 1e-12))
        check(checks, f'{name}_readout_matrix_matches_independent_refit', d < 1e-8,
              dict(max_relative_abs_difference=d, shape=list(ro[key].shape)))
        check(checks, f'{name}_chosen_gamma_matches_independent_cv',
              float(stat[name]['gamma']) == float(refit[name]['gamma']),
              dict(reported=stat[name]['gamma'], recomputed=refit[name]['gamma']))
        check(checks, f'{name}_reported_held_r2_reproduced',
              abs(float(stat[name]['r2_held']) - refit[name]['held']) < 1e-9,
              dict(reported=stat[name]['r2_held'], recomputed=refit[name]['held']))
    check(checks, 'P2_offset_reproduced', np.allclose(ro['b_P2'], b2r, atol=1e-7, rtol=1e-9),
          dict(max_abs_diff=float(np.abs(ro['b_P2'] - b2r).max())))

    # ---- 2. cell-filter convention and the applied command, recomputed independently
    filt, cmd_err, over = {}, [], 0
    m_fit_v, m3_v = ro['mean_fit_P2'], ro['mean_fit_P3']
    b2_v, b3_v = ro['b_P2'], ro['b_P3']
    for dt, tag in DTS:
        z = arrays(CONFIGS / tag / 'WARM.npz')
        decay = float(np.exp(-dt / .005))
        nstep = round(0.001 / dt)
        f = np.zeros(12000)
        packed_all = z['spikes12000_packed']
        assert len(packed_all) % nstep == 0
        rec = np.empty((len(packed_all) // nstep, 12000))
        for i, p in enumerate(packed_all):
            f = decay * f + (1 - decay) * np.unpackbits(p, count=12000) / dt
            if (i + 1) % nstep == 0:
                rec[(i + 1) // nstep - 1] = f
        filt[tag] = bool(np.array_equal(rec, z['cell_filtered_rates']))
        for ph in PHASES:
            ip = round((ph - 0.2) / 0.001) - 1
            warm = z['cell_filtered_rates'][ip]
            for readout in TAGS:
                zz = arrays(O / 'RUNS' / f'{readout}__{tag}__rel_{ph}' / 'NEURAL.npz')
                packed = zz['spikes12000_packed']
                applied = zz['applied']
                state = warm.copy()
                expect = np.empty_like(applied)
                Wm = {'P1': ro['W_P1'], 'P2': ro['W_P2'], 'P3': ro['W_P3']}.get(readout)
                for k in range(len(applied)):
                    if readout == 'R0':
                        raw = limits / 100.0 * (ro['old_matrix'] @ zz['source_rates'][k])
                    elif readout == 'P1':
                        raw = Wm @ state
                    elif readout == 'P2':
                        raw = Wm @ (state - m_fit_v) + b2_v
                    else:
                        raw = Wm @ (state[cells] - m3_v) + b3_v
                    expect[k] = np.clip(raw, -limits, limits)
                    for p in packed[k]:
                        state = decay * state + (1 - decay) * np.unpackbits(p, count=12000) / dt
                cmd_err.append(dict(label=f'{readout}__{tag}__rel_{ph}',
                                    max_abs=float(np.abs(expect - applied).max())))
                over += int(np.sum(np.abs(applied) > limits[None, :]))
        del z, rec
    check(checks, 'cell_filter_reproduces_sealed_warm_trace', all(filt.values()), filt)
    worst = max(cmd_err, key=lambda d: d['max_abs'])
    check(checks, 'applied_command_reproduced_from_saved_spikes', worst['max_abs'] <= 1e-9,
          dict(releases=len(cmd_err), worst=worst, elements_above_limit=over))

    # ---- 3. R0 bitwise reproduction of the sealed releases -----------------------
    bit = js(O / 'R0_BITWISE.json')
    check(checks, 'R0_bitwise_all_ten_releases',
          len(bit) == 10 and all(v['passed'] for v in bit.values()), dict(count=len(bit)))
    prior = js(RT1 / 'readout_fit/R0_BITWISE.json')
    same_as_prior = all(prior.get(k, {}).get('passed') is True for k in bit)
    for dt, tag in DTS:
        for ph in PHASES:
            sealed = SEALED / f'{tag}_paired_{ph}'
            mine = O / 'RUNS' / f'R0__{tag}__rel_{ph}'
            check(checks, f'R0_native_exact__{tag}__{ph}',
                  np.array_equal(arrays(sealed / 'NATIVE.npz')['rows'],
                                 arrays(mine / 'NATIVE.npz')['rows']))
            ref, got = js(sealed / 'RESULT.json'), js(mine / 'RESULT.json')
            check(checks, f'R0_J_exact__{tag}__{ph}', ref['J'] == got['J'],
                  dict(sealed=ref['J'], mine=got['J']))
            for name in ['BODY.npz', 'NEURAL.npz']:
                x, y = arrays(sealed / name), arrays(mine / name)
                keys = list(x) if name == 'BODY.npz' else [k for k in x if k in y]
                check(checks, f'R0_{name}_exact__{tag}__{ph}',
                      all(np.array_equal(x[k], y[k]) for k in keys), dict(keys=len(keys)))

    # ---- 4. independent J and D gates for all 40 releases ----------------------
    summary = js(O / 'SUMMARY.json')
    jerr, gate_mismatch, grid = [], [], set()
    for row in summary['row_level']:
        readout, tag, ph = row['readout'], row['dt_tag'], row['release_time_s']
        grid.add((readout, tag, ph))
        folder = O / 'RUNS' / f'{readout}__{tag}__rel_{ph}'
        arr = arrays(folder / 'NATIVE.npz')['rows']
        assist = WARM_ASSIST[tag][ph]
        J_loc, _, _ = local_objective(arr, assist, ph)
        jerr.append(abs(J_loc - float(row['J'])))
        g_loc, _ = local_gate(arr, assist)
        if g_loc != js(folder / 'RESULT.json')['validation']['checks']:
            gate_mismatch.append(dict(label=f'{readout}__{tag}__rel_{ph}'))
    check(checks, 'release_grid_complete_40', len(grid) == 40,
          dict(releases=len(grid), expected=4 * 2 * 5))
    check(checks, 'objective_recomputed_identically', max(jerr) <= 1e-12,
          dict(max_abs_error=float(max(jerr)), releases=len(jerr)))
    check(checks, 'D_gates_recomputed_identically', not gate_mismatch,
          dict(mismatches=gate_mismatch))

    # ---- 5. pre-registered verdict re-derived -------------------------------
    agg = summary['per_condition']
    redo = {}
    for readout in ('P1', 'P2', 'P3'):
        sup, part = {}, {}
        for _, tag in DTS:
            base, new = agg[f'R0__{tag}'], agg[f'{readout}__{tag}']
            sup[tag] = sum(1 for p in new['per_phase'] if p['D_passed'])
            part[tag] = sum(1 for n, o in zip(new['per_phase'], base['per_phase'])
                            if n['D_pass_count'] > o['D_pass_count'])
        achieved = all(v >= 4 for v in sup.values())
        partial = all(v >= 4 for v in part.values())
        redo[readout] = dict(support_phases=sup, improved_phases=part,
                             achieved=achieved, partial=partial,
                             claim=summary['verdict'][readout]['statement'])
        expect = 'SUPPORT_ACHIEVED' if achieved else ('PARTIAL' if partial else 'NO_IMPROVEMENT')
        check(checks, f'verdict_recomputed__{readout}', expect == redo[readout]['claim'],
              dict(recomputed=expect, claimed=redo[readout]['claim'],
                   support_phases=sup, improved_phases=part))
    overall = ('SUPPORT_ACHIEVED' if any(r['achieved'] for r in redo.values())
               else ('PARTIAL' if any(r['partial'] for r in redo.values())
                     else 'NO_IMPROVEMENT'))
    check(checks, 'overall_verdict_recomputed', overall == summary['overall_verdict'],
          dict(recomputed=overall, claimed=summary['overall_verdict']))
    check(checks, 'no_support_claim_outside_the_rule',
          summary['boundaries']['autonomous_support'] == 'NOT_CLAIMED'
          and summary['boundaries']['walking'] == 'NOT_CLAIMED',
          dict(boundaries=summary['boundaries']))

    # ---- 6. the readout really changed the physics ---------------------------
    differ = 0
    for tag in ('dt_0p5ms', 'dt_0p0625ms'):
        for ph in PHASES:
            a = arrays(O / 'RUNS' / f'R0__{tag}__rel_{ph}' / 'NATIVE.npz')['rows']
            for readout in ('P1', 'P2', 'P3'):
                b = arrays(O / 'RUNS' / f'{readout}__{tag}__rel_{ph}' / 'NATIVE.npz')['rows']
                differ += int(not np.array_equal(a, b))
    check(checks, 'fitted_readouts_change_the_release_trajectory', differ == 30,
          dict(phases_differing=differ, total=30))

    # ---- 7. interface statistics and annotation -----------------------------
    st = js(O / 'READOUT_STATIC.json')
    r0 = st['per_timestep']['dt_0p5ms']['R0']
    check(checks, 'R0_interface_reproduces_parent_facts',
          abs(r0['overall_correlation_with_teacher'] + 0.0831) < 5e-4
          and abs(r0['saturation_fraction_overall'] - 0.784) < 5e-3
          and abs(r0['sign_agreement'] - 0.4653) < 5e-4,
          dict(r=r0['overall_correlation_with_teacher'],
               saturation=r0['saturation_fraction_overall'],
               sign=r0['sign_agreement']))
    r0chk = st['r0_reconstruction']
    check(checks, 'sealed_R0_command_reconstructed_from_stored_rates',
          r0chk['clipped_max_abs_diff'] < 1e-12, r0chk)

    ann = js(O / 'CELL_ANNOTATION.json')
    check(checks, 'annotation_join_complete_by_banc_888_id',
          ann['join']['matched'] == 12000 and ann['join']['unmatched'] == 0,
          ann['join'])
    check(checks, 'annotation_top50_share_reported',
          all(len(ann['per_readout'][k]['top50']) == 50 for k in ('P1', 'P2')),
          dict(P1_top50_share=ann['per_readout']['P1']['top50_shared_energy_share'],
               P2_top50_share=ann['per_readout']['P2']['top50_shared_energy_share']))

    # ---- 8. figure ----------------------------------------------------------
    figs = [p for p in O.glob('FIGURE_*.png')]
    ok = bool(figs) and all(p.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and p.stat().st_size > 40000
                            for p in figs)
    check(checks, 'figure_present_and_valid_png', ok,
          dict(files=[str(p) for p in figs],
               bytes={p.name: p.stat().st_size for p in figs}))

    result = dict(
        verified=str(not any(not c['passed'] for c in checks)),
        passed=int(sum(c['passed'] for c in checks)), total=len(checks),
        checks=checks, independent_refit=refit,
        verdict_recomputed=redo, overall_verdict_recomputed=overall,
        note='This verifier re-implements the ridge refit and its CV alpha choice, the 5 ms cell '
             'filter, the support series, the objective and the D gates in local numpy code; it '
             'does not call the runner or static.py. No failure is repaired here.')
    dump(O / 'VERIFICATION.json', result)
    print('VERIFIED', result['verified'], result['passed'], '/', result['total'], flush=True)
    for c in checks:
        if not c['passed']:
            print('  FAIL', c['check'], c['detail'], flush=True)
    if not result['verified']:
        sys.exit(1)


if __name__ == '__main__':
    main()
