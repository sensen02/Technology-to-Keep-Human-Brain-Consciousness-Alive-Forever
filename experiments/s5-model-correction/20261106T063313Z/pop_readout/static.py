"""Cheap static part: fit P1/P2/P3 (and the declared diagnostics) and describe them.

Reads only sealed arrays (WARM.npz, teacher BODY.npz, TRANSLATOR_IDENTITIES.json).
Writes the fitted readout as arrays plus a JSON description and every interface
property the task asks for.  Prints the go/no-go held-out R^2.

alpha is parameterised scale-free: alpha = gamma * s, s = mean squared feature norm of
the fit window (see common.GAMMAS).  gamma is chosen by blocked CV INSIDE the fit
window only; the held-out window never enters the choice.
"""
import numpy as np

import common as C


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ ridge machinery
def fold_spec(Xfit, folds=C.CV_FOLDS):
    """One spectral solve per contiguous block of the FIT window; reused by every alpha."""
    n = len(Xfit)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    spec = []
    for f in range(folds):
        va = slice(bounds[f], bounds[f + 1])
        tr = np.r_[np.arange(0, bounds[f]), np.arange(bounds[f + 1], n)]
        Xtr = Xfit[tr]
        s, U = np.linalg.eigh(Xtr @ Xtr.T)
        spec.append(dict(va=va, Kva=Xfit[va] @ Xtr.T, U=U, s=s, tr=tr))
    return spec


def cv_from_spec(spec, Yfit, alphas):
    for sp in spec:
        sp['UtY'] = sp['U'].T @ Yfit[sp['tr']]
    curve = []
    for alpha in alphas:
        pred = np.zeros_like(Yfit)
        for sp in spec:
            pred[sp['va']] = sp['Kva'] @ (sp['U'] @ (sp['UtY'] / (sp['s'] + float(alpha))[:, None]))
        curve.append(dict(alpha=float(alpha), cv_r2=float(C.r2_score(Yfit, pred))))
    return curve


def fit_ridge(Xfit, Yfit, tag):
    """Scale-free gamma grid + blocked CV inside the fit window + final fit on all 600."""
    s = C.alpha_scale(Xfit)
    alphas = [g * s for g in C.GAMMAS]
    spec = fold_spec(Xfit)
    curve = cv_from_spec(spec, Yfit, alphas)
    del spec
    best = max(curve, key=lambda g: g['cv_r2'])
    alpha = best['alpha']
    gamma = alpha / s
    for c in curve:
        c['gamma'] = c['alpha'] / s
    W, _ = C.ridge_solve(Xfit, Yfit, alpha)
    log('  %-28s s=%.6g gamma=%.3g alpha=%.6g cv_r2=%.4f' % (tag, s, gamma, alpha, best['cv_r2']))
    return dict(W=W, alpha=float(alpha), gamma=float(gamma), scale=float(s),
                cv_curve=curve, cv_best=float(best['cv_r2']))


# ------------------------------------------------------------------ trend removal
def rm_trend(A, win, chunk=2000):
    """Subtract a centred `win`-sample moving average from every column of A."""
    n, d = A.shape
    pad = win // 2
    out = np.empty_like(A)
    k = win - 1 - pad
    for j in range(0, d, chunk):
        B = A[:, j:j + chunk]
        Ap = np.vstack([B[:1].repeat(pad, 0), B, B[-1:].repeat(k, 0)])
        cs = np.cumsum(Ap, axis=0)
        cs = np.vstack([np.zeros((1, B.shape[1])), cs])
        out[:, j:j + chunk] = B - (cs[win:] - cs[:-win])[:n] / win
    return out


def describe(tag, raw, clipped, limits, target):
    st = C.scale_free_stats(raw, clipped, limits, target, C.FIT_LO, C.FIT_HI)
    st['tag'] = tag
    return st


def r2(Y, P):
    return float(C.r2_score(np.asarray(Y, float), np.asarray(P, float)))


def main():
    if not (C.O / 'PREREGISTRATION.json').exists():
        raise SystemExit('run prereg.py first')
    teacher = C.teacher_torque()
    matrix, limits = C.decoder_identity()
    cells = C.motor_cells_204()

    W = C.warm('dt_0p5ms')
    Xall = W['source']                       # 1200 x 12000
    Xfit, Yfit = Xall[C.FIT_LO:C.FIT_HI], teacher[C.FIT_LO:C.FIT_HI]
    Xheld, Yheld = Xall[C.HELD_LO:C.HELD_HI], teacher[C.HELD_LO:C.HELD_HI]
    m_fit = Xfit.mean(0)                     # per-cell fit-window mean (the declared offset)
    Ybar = Yfit.mean(0)                      # per-joint fit-window mean torque
    Xfit_c = Xfit - m_fit
    Xheld_c = Xheld - m_fit
    Yc_fit = Yfit - Ybar
    Yc_held = Yheld - Ybar
    out = {}

    # ---------------- P1: all 12000 raw rates, NO constant term
    log('fitting P1 (all 12000, no constant term) ...')
    f1 = fit_ridge(Xfit, Yfit, 'P1')
    W1, a1, b1 = f1['W'], f1['alpha'], np.zeros(42)
    raw1 = C.ridge_predict(W1, b1, Xall)
    out['P1'] = dict(alpha=a1, gamma=f1['gamma'], scale=f1['scale'], cv_best=f1['cv_best'],
                     cv_curve=f1['cv_curve'],
                     r2_fit=r2(Yfit, raw1[C.FIT_LO:C.FIT_HI]),
                     r2_held=r2(Yheld, raw1[C.HELD_LO:C.HELD_HI]),
                     n_features=12000, constant_term=False)

    # ---------------- P2: the same 12000 features with every cell's fit-window mean removed
    # The offset removal is applied to the features, and the target's fit-window mean is added
    # back as the single fixed offset b2: without that, de-meaned features cannot carry the mean
    # torque at all (see P2z).  This is the standard ridge-with-intercept shape.
    log('fitting P2 (all 12000, per-cell fit-window mean removed) ...')
    f2 = fit_ridge(Xfit_c, Yc_fit, 'P2')
    W2, a2 = f2['W'], f2['alpha']
    b2 = Ybar - (W2 @ m_fit)
    raw2 = C.ridge_predict(W2, b2, Xall)
    out['P2'] = dict(alpha=a2, gamma=f2['gamma'], scale=f2['scale'], cv_best=f2['cv_best'],
                     cv_curve=f2['cv_curve'],
                     r2_fit=r2(Yfit, raw2[C.FIT_LO:C.FIT_HI]),
                     r2_held=r2(Yheld, raw2[C.HELD_LO:C.HELD_HI]),
                     n_features=12000, constant_term=True,
                     offset_rms=float(np.sqrt((b2 ** 2).mean())),
                     de_meaning='features: r - mean_fit_per_cell; offset: per-joint fit-window '
                                'mean torque added back as one fixed constant')

    # ---------------- P2z: de-meaned features with NO offset supplied (declared degenerate form)
    log('fitting P2z (de-meaned features, no offset supplied; interface-only) ...')
    f2z = fit_ridge(Xfit_c, Yfit, 'P2z')
    W2z, a2z = f2z['W'], f2z['alpha']
    raw2z = C.ridge_predict(W2z, -(W2z @ m_fit), Xall)
    out['P2z_no_offset_diagnostic'] = dict(
        alpha=a2z, gamma=f2z['gamma'], scale=f2z['scale'], cv_best=f2z['cv_best'],
        cv_curve=f2z['cv_curve'],
        r2_fit=r2(Yfit, raw2z[C.FIT_LO:C.FIT_HI]),
        r2_held=r2(Yheld, raw2z[C.HELD_LO:C.HELD_HI]),
        n_features=12000, constant_term=False,
        note='de-meaned features with the offset tied to -W@m_fit instead of a free constant. '
             'This cannot represent the mean torque (its prediction averages to ~0 while the '
             'target averages to 2.43), so it is reported as the diagnostic of WHY an offset is '
             'needed, and it is NOT released.')

    # ---------------- P3: the 204 motor cells, same protocol as P2
    log('fitting P3 (204 motor cells, de-meaned, offset supplied) ...')
    X204 = Xall[:, cells]
    m3 = X204[C.FIT_LO:C.FIT_HI].mean(0)
    X204_fit = X204[C.FIT_LO:C.FIT_HI] - m3
    f3 = fit_ridge(X204_fit, Yc_fit, 'P3')
    W3, a3 = f3['W'], f3['alpha']
    b3 = Ybar - (W3 @ m3)
    raw3 = C.ridge_predict(W3, b3, X204)
    out['P3'] = dict(alpha=a3, gamma=f3['gamma'], scale=f3['scale'], cv_best=f3['cv_best'],
                     cv_curve=f3['cv_curve'],
                     r2_fit=r2(Yfit, raw3[C.FIT_LO:C.FIT_HI]),
                     r2_held=r2(Yheld, raw3[C.HELD_LO:C.HELD_HI]),
                     n_features=204, constant_term=True)

    # ---------------- P3b: 204 cells, raw rates, no constant term (parent 0.27-0.41 bracket)
    log('fitting P3b (204 motor cells, raw rates, no constant term) ...')
    f3b = fit_ridge(X204[C.FIT_LO:C.FIT_HI], Yfit, 'P3b')
    W3b, a3b = f3b['W'], f3b['alpha']
    b3b = np.zeros(42)
    raw3b = C.ridge_predict(W3b, b3b, X204)
    out['P3b_raw_no_constant_diagnostic'] = dict(
        alpha=a3b, gamma=f3b['gamma'], scale=f3b['scale'], cv_best=f3b['cv_best'],
        cv_curve=f3b['cv_curve'],
        r2_fit=r2(Yfit, raw3b[C.FIT_LO:C.FIT_HI]),
        r2_held=r2(Yheld, raw3b[C.HELD_LO:C.HELD_HI]),
        n_features=204, constant_term=False,
        note='used to bracket the parent published 0.27-0.41 held-out R^2 for the 204 motor '
             'cells; interface-only, NOT released.')

    # ---------------- P4: parent's 100 ms trend-removal probe (interface-only)
    log('fitting P4 (100 ms trend removed from features and target; interface-only) ...')
    Xt = rm_trend(Xall, C.TREND_MS)
    Yt = rm_trend(teacher, C.TREND_MS)
    mt = Xt[C.FIT_LO:C.FIT_HI].mean(0)
    Ytbar = Yt[C.FIT_LO:C.FIT_HI].mean(0)
    f4 = fit_ridge(Xt[C.FIT_LO:C.FIT_HI] - mt, Yt[C.FIT_LO:C.FIT_HI] - Ytbar, 'P4_trend')
    W4, a4 = f4['W'], f4['alpha']
    raw4 = C.ridge_predict(W4, Ytbar - (W4 @ mt), Xt)
    out['P4_trend_removed_diagnostic'] = dict(
        alpha=a4, gamma=f4['gamma'], scale=f4['scale'], cv_best=f4['cv_best'],
        cv_curve=f4['cv_curve'],
        r2_fit=r2(Yt[C.FIT_LO:C.FIT_HI], raw4[C.FIT_LO:C.FIT_HI]),
        r2_held=r2(Yt[C.HELD_LO:C.HELD_HI], raw4[C.HELD_LO:C.HELD_HI]),
        n_features=12000, constant_term=True, trend_window_ms=C.TREND_MS,
        note='a centred 100 ms moving average is removed from features and target; this cannot '
             'be a causal physical readout (edge effects use the future) and is reported only to '
             'locate the parent published 0.841.')
    del Xt, Yt

    # ---------------- shuffle control
    log('shuffle control (time-permuted target, same features) ...')
    rng = np.random.default_rng(20261106)
    s2, U2 = np.linalg.eigh(Xfit_c @ Xfit_c.T)
    Kheld = Xheld_c @ Xfit_c.T
    sh = []
    for _ in range(3):
        UtY = U2.T @ Yc_fit[rng.permutation(len(Yc_fit))]
        sh.append(float(C.r2_score(Yc_held, Kheld @ (U2 @ (UtY / (s2 + a2)[:, None])))))
    shuf = dict(alpha=a2, repeats=sh, mean=float(np.mean(sh)),
                target='fit-window-centred target, held-out-centred target')
    del s2, U2, Kheld

    # ---------------- interface statistics on both sealed caches
    log('interface statistics on both sealed caches ...')
    variants = [('P1', W1, b1, None), ('P2', W2, b2, None), ('P2z', W2z, -(W2z @ m_fit), None),
                ('P3', W3, b3, cells), ('P3b', W3b, b3b, cells)]
    per_ts = {}
    for dt, tag in C.DTS:
        z = C.arrays(C.CONFIGS / tag / 'WARM.npz')
        src = np.vstack([np.zeros((1, 12000)), z['cell_filtered_rates'][:-1]])
        per_ts[tag] = {}
        rawR0 = limits / 100.0 * (z['command_source_rates'] @ matrix.T)
        per_ts[tag]['R0'] = describe('R0', rawR0, np.clip(rawR0, -limits, limits), limits, teacher)
        for name, Wm, bm, cols in variants:
            raw = C.ridge_predict(Wm, bm, src if cols is None else src[:, cols])
            per_ts[tag][name] = describe(name, raw, np.clip(raw, -limits, limits), limits, teacher)
        del z, src

    z05 = C.arrays(C.CONFIGS / 'dt_0p5ms/WARM.npz')
    r0con = limits / 100.0 * (z05['command_source_rates'] @ matrix.T)
    r0chk = dict(
        clipped_max_abs_diff=float(np.abs(np.clip(r0con, -limits, limits)
                                          - z05['clipped_command']).max()),
        raw_max_abs_diff=float(np.abs(r0con - z05['raw_command']).max()))

    # ---------------- matrix stability (fit window vs held window) for P2
    Wh, _ = C.ridge_solve(Xheld_c, Yc_held, a2)
    stab = dict(P2=dict(row_vector_correlation_with_held_fit=float(
        np.corrcoef(W2.ravel(), Wh.ravel())[0, 1]),
        relative_frobenius_difference=float(np.linalg.norm(W2 - Wh)
                                            / max(np.linalg.norm(W2), 1e-12))))
    del Wh

    np.savez_compressed(
        C.O / 'READOUT_FIT.npz',
        W_P1=W1, b_P1=b1, W_P2=W2, b_P2=b2, W_P3=W3, b_P3=b3,
        W_P2z=W2z, b_P2z=-(W2z @ m_fit), W_P3b=W3b, b_P3b=b3b,
        mean_fit_P2=m_fit, mean_fit_P3=m3, target_mean_fit=Ybar,
        alpha=np.array([a1, a2, a3, a3b, a2z, a4], float),
        gamma=np.array([f1['gamma'], f2['gamma'], f3['gamma'], f3b['gamma'], f2z['gamma'],
                        f4['gamma']], float),
        limits=limits, old_matrix=matrix, cells204=cells,
        fit_window=np.array([C.FIT_LO, C.FIT_HI]), held_window=np.array([C.HELD_LO, C.HELD_HI]),
        teacher_rms=np.array([float(np.sqrt((Yfit ** 2).mean())),
                              float(np.sqrt((Yheld ** 2).mean()))]))

    C.dump(C.O / 'READOUT_DESCRIPTION.json', dict(
        file=str(C.O / 'READOUT_FIT.npz'),
        arrays=dict(W_P1=[42, 12000], b_P1=[42], W_P2=[42, 12000], b_P2=[42], W_P3=[42, 204],
                    b_P3=[42], mean_fit_P2=[12000], mean_fit_P3=[204], target_mean_fit=[42],
                    cells204=[204], limits=[42], old_matrix=[42, 66]),
        rules=dict(R0='command = clip(limits/100 * (old_matrix @ pool_rates_66), +/- limits)',
                   P1='command = clip(W_P1 @ r_12000, +/- limits)   (no constant term)',
                   P2='command = clip(W_P2 @ (r_12000 - mean_fit_P2) + b_P2, +/- limits), '
                      'b_P2 = target_mean_fit - W_P2 @ mean_fit_P2',
                   P3='command = clip(W_P3 @ (r_204 - mean_fit_P3) + b_P3, +/- limits)'),
        features='5 ms-filtered per-cell rates at the 1 ms command-interval start '
                 '(WARM.npz cell_filtered_rates shifted by one interval, zeros at 0.2 s)',
        fitted_on='sealed dt_0p5ms warm cache, 0.2-0.8 s only; one matrix per readout used for '
                  'both neural timesteps and all five phases',
        alpha_parameterisation='alpha = gamma * s, s = mean squared feature norm of the fit '
                               'window; gamma grid %r; gamma chosen by %d blocked folds inside '
                               'the fit window only' % (C.GAMMAS, C.CV_FOLDS),
        alphas=dict(P1=dict(gamma=f1['gamma'], alpha=a1),
                    P2=dict(gamma=f2['gamma'], alpha=a2),
                    P3=dict(gamma=f3['gamma'], alpha=a3),
                    P3b=dict(gamma=f3b['gamma'], alpha=a3b),
                    P2z=dict(gamma=f2z['gamma'], alpha=a2z),
                    P4=dict(gamma=f4['gamma'], alpha=a4)),
        label='OFFLINE ENGINEERING FITTING of the readout interface, NOT biological calibration. '
              'The neurons still generate the command.',
        importance='核心必需'))

    go = dict(P1_held_r2=out['P1']['r2_held'], P2_held_r2=out['P2']['r2_held'],
              P3_held_r2=out['P3']['r2_held'],
              P3b_held_r2=out['P3b_raw_no_constant_diagnostic']['r2_held'],
              P2z_held_r2=out['P2z_no_offset_diagnostic']['r2_held'],
              P4_trend_removed_held_r2=out['P4_trend_removed_diagnostic']['r2_held'],
              threshold=0.5,
              go=bool(max(out['P1']['r2_held'], out['P2']['r2_held']) >= 0.5))
    C.dump(C.O / 'READOUT_STATIC.json', dict(
        per_timestep=per_ts, fits=out, shuffle_control=shuf, stability=stab,
        r0_reconstruction=r0chk,
        teacher_rms=dict(full=float(np.sqrt((teacher ** 2).mean())),
                         fit=float(np.sqrt((Yfit ** 2).mean())),
                         held=float(np.sqrt((Yheld ** 2).mean())),
                         static_full=float(np.sqrt((teacher.mean(0) ** 2).mean())),
                         fluctuation_full=float(np.sqrt(((teacher - teacher.mean(0)) ** 2).mean()))),
        data_check=dict(
            cells_with_nonzero_rate_fit_window=int((Xfit > 0).any(0).sum()),
            cells_with_nonzero_rate_held_window=int((Xheld > 0).any(0).sum()),
            cells_with_nonzero_rate_full_window=int((Xall > 0).any(0).sum()),
            fit_feature_mean_hz=float(m_fit.mean()),
            fit_feature_rms_hz=float(np.sqrt((Xfit ** 2).mean())),
            alpha_scale_s=float(C.alpha_scale(Xfit))),
        go_no_go=go,
        importance='核心必需'))
    log('GO_NO_GO ' + str(go))
    for k, v in out.items():
        log('  %-32s gamma %-9.3g alpha %-12.6g r2_fit %7.4f r2_held %7.4f'
            % (k, v['gamma'], v['alpha'], v['r2_fit'], v['r2_held']))
    log('shuffle %.4f' % shuf['mean'])


if __name__ == '__main__':
    main()
