"""S165 corrected saved-data verifier. No engine/producer/neural imports or calls.

Certificates concern the saved, independently reconstructed binary64 matrices in
native model units, not SI calibration or dynamic/neural attainability. A rigid
contact certificate is only about the declared static relaxation.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import numpy as np
from scipy.optimize import linprog


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    err = float(np.max(np.abs(a-b), initial=0.))
    assert err <= tol, (label, err, tol)
    return err


def exact_dot(a, b):
    return sum((Fraction.from_float(float(x))*Fraction.from_float(float(y))
                for x, y in zip(a, b)), Fraction(0))


def separation(A, b, lower, upper):
    """Find y with b.y > sup_{lower<=x<=upper} (A.T y).x.

    Infinite upper bounds require A_j.y<=0 EXACTLY. The final certificate
    checks rational arithmetic on the binary64 inputs; solver tolerances are
    never used to excuse positive coefficients of unbounded variables.
    """
    A, b = np.asarray(A), np.asarray(b)
    lower, upper = np.asarray(lower), np.asarray(upper)
    assert np.all(lower == 0), 'certificate implementation requires zero lower bounds'
    finite = np.flatnonzero(np.isfinite(upper))
    free = np.flatnonzero(~np.isfinite(upper))
    # Row scaling only preconditions the search; certificate is native-unit.
    scale = np.maximum(np.max(np.abs(A), axis=1), np.abs(b))
    scale = np.maximum(scale, 1e-12)
    As, bs = A/scale[:, None], b/scale
    m, k = A.shape[0], len(finite)
    c = np.r_[-bs, upper[finite]]
    bounds = [(-1., 1.)]*m + [(0., None)]*k
    for margin in (0., 1e-10, 1e-8, 1e-6):
        rows = []; rhs = []
        if k:
            rows.append(np.c_[As[:, finite].T, -np.eye(k)])
            rhs.extend([0.]*k)
        if len(free):
            rows.append(np.c_[As[:, free].T, np.zeros((len(free), k))])
            rhs.extend([-margin]*len(free))
        result = linprog(c, A_ub=np.vstack(rows) if rows else None,
                         b_ub=np.asarray(rhs) if rows else None,
                         bounds=bounds, method='highs')
        if not result.success:
            continue
        y = result.x[:m]/scale
        dots = [exact_dot(A[:, j], y) for j in range(A.shape[1])]
        if any(dots[j] > 0 for j in free):
            continue
        support = sum((Fraction.from_float(float(upper[j]))*max(dots[j], 0)
                       for j in finite), Fraction(0))
        target = exact_dot(b, y)
        gap = target-support
        if gap > 0:
            return y, dict(validated=True, arithmetic='exact rational on binary64 inputs',
                proof='b.y > sum_j finite(upper_j*max(A_j.y,0)); all infinite-upper columns A_j.y<=0',
                gap=float(gap), gap_numerator=str(gap.numerator),
                gap_denominator=str(gap.denominator), target=float(target),
                support=float(support), free_column_max=float(max((dots[j] for j in free), default=Fraction(0))),
                search_margin=margin)
    return None, dict(validated=False, interpretation='Numerical infeasibility only; no exact separating certificate found')


def scalar_rate(act, ctrl, dyn):
    # MuJoCo muscle activation: optional smooth switching uses dynprm[2].
    tau_act, tau_deact, smooth = map(float, dyn[:3])
    clipped_ctrl = min(1., max(0., ctrl))
    clipped_act = min(1., max(0., act))
    scale = .5 + 1.5*clipped_act
    ta, td = tau_act*scale, tau_deact/scale
    delta = clipped_ctrl-act
    if smooth >= 1e-15:
        x = min(1., max(0., delta/smooth+.5))
        switch = x*x*x*(3*x*(2*x-5)+10)
        tau = td+(ta-td)*switch
    else:
        tau = ta if delta > 0 else td
    return delta/max(tau, 1e-15)


def muscle(length, velocity, act, gain, bias, lengthrange):
    # Independent scalar piecewise muscle law; no existing verifier import.
    def active_curve(l, low, high):
        if l < low or l > high: return 0.
        midlow, midhigh = .5*(low+1), .5*(high+1)
        if l <= midlow: return .5*((l-low)/(midlow-low))**2
        if l <= 1: return 1-.5*((1-l)/(1-midlow))**2
        if l <= midhigh: return 1-.5*((l-1)/(midhigh-1))**2
        return .5*((high-l)/(high-midhigh))**2
    g, p = gain, bias
    assert g[2] > 0 and p[2] > 0
    l0 = (lengthrange[1]-lengthrange[0])/(g[1]-g[0])
    l = g[0]+(length-lengthrange[0])/l0
    v = velocity/(l0*g[6]); y = g[8]-1
    fv = 0. if v <= -1 else (v+1)**2 if v <= 0 else g[8]-(y-v)**2/y if v <= y else g[8]
    gainvalue = -g[2]*active_curve(l, g[4], g[5])*fv
    l0b = (lengthrange[1]-lengthrange[0])/(p[1]-p[0])
    lb = p[0]+(length-lengthrange[0])/l0b; mid = .5*(1+p[5])
    biasvalue = 0. if lb <= 1 else -p[2]*p[7]*.5*((lb-1)/(mid-1))**2 if lb <= mid else -p[2]*p[7]*(.5+(lb-mid)/(mid-1))
    return gainvalue*act+biasvalue, gainvalue, biasvalue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    prereg = json.loads((out/'PREREG.json').read_text())
    source = Path(prereg['source'])
    hashes = json.loads((out/'INPUT_HASHES.json').read_text())
    assert all(sha(p) == h for p, h in hashes.items()), 'source hashes changed'
    afterhash = json.loads((out/'INPUT_HASHES_AFTER.json').read_text())
    assert hashes == afterhash
    freezes = json.loads((out/'SOURCE_FREEZE_MANIFEST.json').read_text())
    assert all(sha(v['copy']) == v['sha256'] == hashes[p] for p, v in freezes.items())
    summary = json.loads((out/'SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETED_DIAGNOSTIC_ONLY'
    ledger = json.loads((out/'CALL_LEDGER.json').read_text())
    assert ledger == dict(model_load=1, forward=6, dynamic_forward=3, zeroV_forward=3, step=0, neural_ticks=0)
    assert prereg['selected_indices'] == [0, 15, 83]
    assert prereg['activation_upper_bounds'] == [.5, 1.]
    assert prereg['model']['tau_act'] == .001 and prereg['model']['tau_deact'] == .0004
    model = load(source/'init80ms/NATIVE_MODEL_ARRAYS.npz')
    a = load(source/'init80ms/TRAJECTORY.npz')
    r = load(source/'init80ms/CONSTRAINT_ROWS.npz')
    affinity = json.loads((out/'AFFINITY_PROOF.json').read_text())
    dyn = np.asarray(affinity['dynprm'])
    assert np.all(dyn[:, 0] == .001) and np.all(dyn[:, 1] == .0004)
    def modelarr(key):
        return model[key] if key in model else model['model.'+key]
    for key in ('gainprm', 'biasprm'):
        close(np.asarray(affinity[key]), modelarr('actuator_'+key), key, 0.)
    for key in ('actuator_forcelimited', 'actuator_actlimited', 'jnt_actfrclimited', 'tendon_actfrclimited'):
        if key in model or 'model.'+key in model:
            assert not np.any(modelarr(key)), key
    assert affinity['no_force_limits']
    native_dyn = modelarr('actuator_dynprm').copy()
    native_dyn[:, 0] = .001
    close(dyn, native_dyn, 'runtime tau-only numeric patch', 0.)
    assert np.array_equal(np.asarray(affinity['actadr']), np.arange(90))
    proof = json.loads((out/'MODEL_PROOF.json').read_text())
    source_proof = json.loads((source/'init80ms/MODEL_PROVENANCE.json').read_text())
    assert proof['before'] == source_proof['before'] and proof['after'] == source_proof['after']
    if 'opt.cone' in model:
        assert int(model['opt.cone']) == 0, 'native cone must be pyramidal'
    meta = json.loads((source/'init80ms/COM_CONTACT_METADATA.json').read_text())
    floors = set(meta['floor_geom_ids'])
    legs = list(meta['foot_order'])
    geomleg = {int(k): legs.index(v) for k, v in meta['geom_map'].items()}
    mapping_paths = [p for p in hashes if Path(p).name == 'S149_ACTUATOR_MAP.json']
    assert len(mapping_paths) == 1
    mapping = json.loads(Path(mapping_paths[0]).read_text())
    assert sorted(e['actuator'] for e in mapping) == list(range(90))
    assert all(sum(e['leg'] == leg for e in mapping) == 15 for leg in legs)
    errors = {}; solutions = []; certificates = {}; plotstates = {}
    def decode(c, edges, label, checkraw=False):
        force = np.zeros((6, 3)); margins = []
        for ci, addr in enumerate(c['contact_efc_address']):
            e = edges[int(addr):int(addr)+4]
            mu = c['contact_friction'][ci, :2]
            assert len(e) == 4 and np.all(mu > 0)
            assert e.min() >= -1e-8, (label, 'negative edge')
            local = np.array([e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])])
            margin = local[0]-np.abs(local[1:]/mu).sum()
            assert margin >= -1e-8, (label, margin)
            margins.append(float(margin))
            if checkraw:
                close(local, c['contact_force_raw6'][ci, :3], label+' local')
            g1, g2 = map(int, c['contact_geom_ids'][ci])
            assert (g1 in floors) != (g2 in floors)
            foot = g2 if g1 in floors else g1
            assert foot in geomleg
            force[geomleg[foot]] += (1 if g1 in floors else -1)*(c['contact_frame'][ci].T@local)
        return force, margins
    for i in (0, 15, 83):
        d = load(out/f'DYNAMIC_{i:03d}.npz')
        s = load(out/f'ZERO_V_{i:03d}.npz')
        affine = load(out/f'STATIC_AFFINE_{i:03d}.npz')
        checks = {}
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
            close(d[key], a[key][i], key, 0.)
            close(s[key], np.zeros_like(d[key]) if key == 'qvel' else d[key], 'zeroV '+key, 0.)
        close(d['qacc_warmstart_input'], a['native_qacc_warmstart'][i], 'dynamic warmstart', 0.)
        close(s['qacc_warmstart_input'], a['native_qacc_warmstart'][i], 'zeroV warmstart', 0.)
        for c, tag in ((d, 'dynamic'), (s, 'zeroV')):
            assert c['M'].shape == (48, 48) and c['actuator_moment'].shape == (90, 48)
            close(c['M'], c['M'].T, tag+' symmetric mass')
            assert np.linalg.eigvalsh(c['M']).min() > 0
            Ma = c['M']@c['qacc']
            checks[tag+'_Ma'] = close(Ma, c['Ma'], tag+' Ma')
            checks[tag+'_balance48'] = close(Ma, c['qfrc_actuator']+c['qfrc_passive']+c['qfrc_constraint']+c['qfrc_applied']-c['qfrc_bias'], tag+' force balance')
            checks[tag+'_JTF'] = close(c['efc_J_dense'].T@c['efc_force'], c['qfrc_constraint'], tag+' JTF')
            assert not c['qfrc_applied'].any() and not c['xfrc_applied'].any()
            close(c['actuator_moment']@c['qvel'], c['actuator_velocity'], tag+' tendon velocity')
            rates = np.array([scalar_rate(float(c['act'][j]), float(c['ctrl'][j]), dyn[j]) for j in range(90)])
            close(rates, c['act_dot'], tag+' actdot', 1e-9)
            assert c['efc_J_dense'].shape == (72, 48)
            assert np.all(c['efc_type'] == 6) and len(c['contact_dim']) == 18 and np.all(c['contact_dim'] == 3)
            assert np.all(c['contact_row_count'] == 4)
            for ci, addr in enumerate(c['contact_efc_address']):
                assert np.array_equal(np.flatnonzero(c['efc_id'] == ci), np.arange(addr, addr+4))
            foot, _ = decode(c, c['efc_force'], tag, True)
            close(foot, c['foot_force'], tag+' foot force')
        for key in ('qacc', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias', 'qfrc_constraint', 'Ma', 'actuator_force', 'actuator_length', 'actuator_velocity', 'actuator_moment', 'com_acceleration_world', 'foot_force'):
            checks['saved_'+key] = close(d[key], a[key][i], 'saved '+key)
        prefix = f'sample{i:03d}_'
        for key in ('efc_J_dense', 'efc_force', 'efc_id', 'efc_type', 'contact_efc_address'):
            close(d[key], r[prefix+key], 'saved rows '+key)
        gains = []; biases = []
        for j in range(90):
            _, gain, bias = muscle(s['actuator_length'][j], 0., 0., modelarr('actuator_gainprm')[j], modelarr('actuator_biasprm')[j], modelarr('actuator_lengthrange')[j])
            gains.append(gain); biases.append(bias)
            for c in (d, s):
                f, _, _ = muscle(c['actuator_length'][j], c['actuator_velocity'][j], c['act'][j], modelarr('actuator_gainprm')[j], modelarr('actuator_biasprm')[j], modelarr('actuator_lengthrange')[j])
                close(np.asarray(f), np.asarray(c['actuator_force'][j]), 'muscle force')
        gains, biases = np.asarray(gains), np.asarray(biases)
        assert not s['qvel'].any()
        close(s['actuator_velocity'], np.zeros(90), 'zero muscle velocity', 1e-12)
        moment = s['actuator_moment']; J = s['efc_J_dense']
        B = moment.T*gains[None, :]
        base = moment.T@biases+s['qfrc_passive']-s['qfrc_bias']
        f0 = np.maximum(s['efc_D']*s['efc_aref'], 0.)
        close(moment.T@(gains*s['act']+biases), s['qfrc_actuator'], 'affine native torque')
        for key, value in dict(B=B, base=base, gain=gains, bias=biases, f0=f0, J=J, moment=moment).items():
            close(value, affine[key], 'saved affine '+key)
        close(B@s['act']+base+J.T@f0, affine['actual_soft_static_residual'], 'actual static residual')
        softfoot, softmargins = decode(s, f0, 'soft f0')
        stateplot = dict(actual_act=s['act'].tolist(), actual_legFz=s['foot_force'][:, 2].tolist(), soft_legFz=softfoot[:, 2].tolist(), solutions=[])
        for upper in (.5, 1.):
            for kind in ('soft', 'rigid_upper'):
                label = f'{kind}_{i:03d}_upper{upper:.1f}'
                lp = load(out/(label+'.npz'))
                info = json.loads((out/(label+'.json')).read_text())
                A = np.column_stack((B, J.T)) if kind == 'rigid_upper' else B
                b = -base if kind == 'rigid_upper' else -base-J.T@f0
                n = A.shape[1]
                close(lp['Aeq'], A, label+' A', 1e-12)
                close(lp['beq'], b, label+' b', 1e-12)
                bounds = np.c_[np.zeros(n), np.r_[np.full(90, upper), np.full(n-90, np.inf)]]
                assert np.array_equal(lp['bounds'], bounds)
                x = lp['x']; feasible = bool(np.isfinite(x).all() and np.max(np.abs(A@x-b)) <= 1e-8 and np.min(x) >= -1e-8 and np.all(x[:90] <= upper+1e-8))
                assert feasible == info['feasible_absolute_native']
                result = dict(sample=i, kind=kind, upper=upper, solver_status=info['solver_status'], feasible=feasible)
                if feasible:
                    residual = A@x-b
                    close(lp['residual'], residual, label+' residual')
                    result['residual_maxabs'] = float(np.abs(residual).max())
                    control = load(out/(label+'_POSSIBLE_CONTROL_NOT_APPLIED.npz'))
                    ctrl = x[:90].copy()
                    ranges = np.asarray(affinity['ctrlrange']); limited = np.asarray(affinity['ctrllimited'], bool)
                    clamped = np.where(limited, np.clip(ctrl, ranges[:, 0], ranges[:, 1]), ctrl)
                    rates = np.array([scalar_rate(float(x[j]), float(clamped[j]), dyn[j]) for j in range(90)])
                    close(control['activation'], x[:90], label+' possible act', 0.)
                    close(control['ctrl'], ctrl, label+' possible ctrl', 0.)
                    close(rates, control['offline_act_dot'], label+' possible actdot', 0.)
                    assert np.array_equal(rates, np.zeros(90))
                    edges = x[90:] if kind == 'rigid_upper' else f0
                    legforce, margins = decode(s, edges, label)
                    result['min_pyramid_margin'] = min(margins)
                    stateplot['solutions'].append(dict(kind=kind, upper=upper, activation=x[:90].tolist(), legFz=legforce[:, 2].tolist(), residuals=residual.tolist()))
                else:
                    y, cert = separation(A, b, bounds[:, 0], bounds[:, 1])
                    result['certificate'] = cert
                    if y is not None: certificates[label+'_y'] = y
                solutions.append(result)
        errors[str(i)] = checks
        plotstates[str(i)] = stateplot
    assert all(sha(p) == h for p, h in hashes.items()), 'source identity changed during offline audit'
    report = dict(passed=True, model_load=0, forward=0, step=0, neural_ticks=0,
                  source_hashes_preserved=True, errors=errors, solutions=solutions,
                  verifier_sha256=sha(__file__), physical_units='NOT_ESTABLISHED',
                  limitations='Static matrix feasibility only; rigid relaxation does not certify native-soft feasibility, attainable neural control, prewarm, or autonomous motion.')
    with (out/'PARENT_STATIC_INDEPENDENT_CHECK.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    with (out/'PARENT_SEPARATING_CERTIFICATES.npz').open('xb') as f:
        np.savez_compressed(f, **certificates)
    with (out/'PLOT_DATA.json').open('x') as f:
        json.dump(dict(states=plotstates, legs=legs), f, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
