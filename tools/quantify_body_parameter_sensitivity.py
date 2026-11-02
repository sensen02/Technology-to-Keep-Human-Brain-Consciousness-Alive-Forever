"""S2/S4 bounded body/contact parameter sensitivity on already-sealed static support matrices.

Question answered: if body-level quantities (muscle attachment sites, force amplitude/Fmax,
activation ceiling, foot friction) are treated as tunable, can a *slight* adjustment of one of
them produce a static load-bearing (support) witness on the three already-saved states?

Scope limits (pre-registered, not negotiable inside this script):
 - Phase1 is pure matrix arithmetic on sealed arrays: 0 model compile, 0 forward.
 - No default model / XML / muscle / real network / old package / tracker change.
 - One knob at a time, others at 1. Pre-registered coarse grids only: no continuous
   optimisation, no bisection refinement, no pose/contact/site scanning, no multi-knob search.
 - The all-1 default LP was already solved and sealed in the previous package, so it is
   *reused* (cached solver report), never re-solved (bounded de-duplication).
 - Phase2 (attachment-site finite differences) runs only if Phase1 finds a minimum needed
   multiplier <= 1.25; otherwise it is explicitly SKIPPED (skipped is not a pass).
 - 0 neural / network / training / plasticity. No STDP. Parameter multipliers here are an
   engineering calibration hypothesis, NOT measured anatomy and NOT permission to change the
   model or the body. Absolute biological units remain NOT_ESTABLISHED.

Subcommands: prepare -> sweep -> phase2 -> verify -> report
Run with: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
          /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B <this file> <cmd>
"""
from pathlib import Path
import sys, os, json, hashlib, math, shutil, traceback
import numpy as np

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
D = Path('/home/sensen/Desktop/engine修复验收')
M = D / 's2-mapped-support-capacity/20261102T170741Z'          # sealed: POSE_q/A/base/contact J/mu
N = D / 's2-native-support-constraints/20261102T172633Z'      # sealed: native rows, reactions, LP matrices, Farkas
OPER = D / 's2-reference-muscle-operating-range/20261102T155412Z'  # sealed: actuator params provenance
O = D / 's2-body-parameter-sensitivity/20261102T190000Z'       # this package
XML = D / 's5-banc888-physical-handover/20261101T220018Z/RUN/MODEL.xml'

FLOOR = 1e-4
ARM = 'independent90'
MODES = ['outerbox', 'native']
S_GRID = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
U_GRID = [1.0, 1.25, 1.5, 2.0, 3.0]
K_GRID = [1.0, 1.25, 1.5, 2.0, 4.0, 10.0]
KNOBS = [('s', S_GRID), ('u', U_GRID), ('k', K_GRID)]
POSE_LABELS = ['initial0', 'original_release1p3', 'slow_release1p25']
PHASE2_MULT_MAX = 1.25
GATE_EQ = 1e-8
GATE_FRIC = 1e-8
GATE_BOUNDS = 1e-10
GATE_ACT = 1e-10


def ar(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def js(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _san(o):
    """JSON-safe conversion; infinities become explicit strings, never silent NaN."""
    if isinstance(o, np.ndarray):
        return _san(o.tolist())
    if isinstance(o, dict):
        return {str(k): _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        if math.isnan(v):
            raise ValueError('NaN must never be written to a result file')
        if math.isinf(v):
            return 'inf' if v > 0 else '-inf'
        return v
    if isinstance(o, Path):
        return str(o)
    return o


def dump(name, obj):
    (O / name).write_text(json.dumps(_san(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def mx(a):
    return float(np.max(np.abs(np.asarray(a)), initial=0.0))


def bounds_diff(a, b):
    """Max absolute difference of two bound tables; infinities are compared by sign."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return math.inf
    fin = np.isfinite(a)
    if not np.array_equal(a[~fin], b[~fin]):
        return math.inf
    return float(np.max(np.abs(a[fin] - b[fin]), initial=0.0))


def bump(name, d):
    """Append progress without racing the final artifacts."""
    p = O / name
    cur = js(p) if p.exists() else []
    cur.append(d)
    (p).write_text(json.dumps(cur, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')


# ---------------------------------------------------------------- sealed inputs
def sealed_pose(pi):
    """Raw sealed arrays for one saved state (no model call)."""
    p = ar(M / f'POSE_{pi}.npz')
    r = ar(N / f'REACTIONS_{pi}.npz')
    n = ar(N / f'NATIVE_{pi}.npz')
    old = ar(M / f'LP_{pi}_{ARM}.npz')
    contacts = js(N / f'CONTACTS_{pi}.json')
    pc = js(M / 'POSE_CONTACTS.json')[pi]
    return dict(pose=pi, label=POSE_LABELS[pi], qpos=p['qpos'], moment=p['moment'], gain=p['gain'],
                bias=p['bias'], qp=p['qfrc_passive'], qb=p['qfrc_bias'], A90=p['A90'], base=p['base'],
                P=old['P'], A_eq=old['A_eq'], b_eq=old['b_eq'], A_ub=old['A_ub'], bounds_old=old['bounds'],
                row_scale_old=old['row_scale'], J=p['J'], mu=p['mu'], positions=p['positions'],
                activation_actual=p['activation_actual'], command_actual=p['command_actual'],
                actual_moment=p['actual_moment'], actual_force=p['actual_force'],
                time=pc['time'], contacts=pc['contacts'],
                native_J=n['J'], native_type=n['efc_type'], native_id=n['efc_id'],
                native_force=n['efc_force'], native_fricloss=n['efc_frictionloss'],
                native_pos=n['efc_pos'], native_contact_json=contacts,
                foot_columns=r['native_foot_columns'], internal_columns=r['internal_columns'],
                internal_bounds=r['internal_bounds'])


def native_columns(nat, pi, k):
    """Rebuild native pyramidal foot columns with friction coefficient multiplier k.

    MuJoCo cone=0 (pyramidal) row i of a dim-3 contact is  jac*( t_i + mu*n ) for the two
    tangential axes with both signs, so with mu -> k*mu:   row(k) = T + k*Npt,
    T = jac*t_i, Npt = mu*jac*n.  Both are recovered from the sealed dense efc_J rows.
    """
    J = nat['native_J']
    feet = [c for c in nat['native_contact_json'] if c['foot_floor']]
    cols = []
    parts = []
    for c in feet:
        rows = c['rows']
        Rm = J[rows]
        assert Rm.shape[0] == 4 and c['dim'] == 3 and len(c['friction']) >= 2
        Npt = (Rm[0] + Rm[1]) / 2.0
        Npt2 = (Rm[2] + Rm[3]) / 2.0
        assert mx(Npt - Npt2) <= 1e-12
        T1 = (Rm[0] - Rm[1]) / 2.0
        T2 = (Rm[2] - Rm[3]) / 2.0
        cols.append(np.vstack([T1 + k * Npt, -T1 + k * Npt, T2 + k * Npt, -T2 + k * Npt]).T)
        parts.append(dict(contact_index=c['index'], name=c['name1'] if c['name1'].startswith(('LF', 'LM', 'LH', 'RF', 'RM', 'RH')) and 'Tars' in c['name1'] else c['name2'],
                          mu=float(c['friction'][0]), rows=rows,
                          normal_part_norm=float(np.linalg.norm(Npt)), tangent_norm=[float(np.linalg.norm(T1)), float(np.linalg.norm(T2))]))
    return (np.column_stack(cols) if cols else np.zeros((48, 0))), parts


def build_case(nat, s, u, k, mode):
    """E,b,U,v,bounds for one (s,u,k,mode) case plus the pieces used for reporting."""
    mom, gain, bias = nat['moment'], nat['gain'], nat['bias']
    A = mom.T * (s * gain)[None, :]                     # 48x90  == s * moment.T*gain
    base = nat['qp'] - nat['qb'] + mom.T @ (s * bias)   # 48: passive+bias scaled by s
    b = -base - A @ np.full(90, FLOOR)
    nc = nat['J'].shape[0]
    if mode == 'outerbox':
        contact = nat['J'].transpose(2, 0, 1).reshape(48, -1)
        cbounds = [[-np.inf, np.inf], [-np.inf, np.inf], [0.0, np.inf]] * nc
        ncols = 3 * nc
        U = np.zeros((4 * nc, 90 + ncols + nat['internal_columns'].shape[1]))
        for c in range(nc):
            for f, (sx, sy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
                U[4 * c + f, 90 + 3 * c:90 + 3 * c + 3] = [sx, sy, -k * nat['mu'][c]]
        v = np.zeros(len(U))
    else:
        contact, _ = native_columns(nat, nat['pose'], k)
        cbounds = [[0.0, np.inf]] * contact.shape[1]
        U = np.zeros((0, 90 + contact.shape[1] + nat['internal_columns'].shape[1]))
        v = np.zeros(0)
    E = np.column_stack([A @ nat['P'], contact, nat['internal_columns']])
    bounds = np.array([[0.0, u - FLOOR]] * 90 + [list(x) for x in cbounds]
                      + [list(x) for x in nat['internal_bounds']], dtype=float)
    assert bounds.shape[0] == E.shape[1]
    scale = np.maximum(1.0, np.maximum(np.abs(E).max(1), np.abs(b)))
    uscale = np.maximum(1.0, np.abs(U).max(1)) if len(U) else np.zeros(0)
    return dict(E=E, b=b, U=U, v=v, bounds=bounds, row_scale=scale, ineq_scale=uscale,
                contact=contact, muscle=A @ nat['P'])


def hsystem(p):
    """H x <= h with every finite bound stacked in (same convention as the sealed Farkas)."""
    n = p['E'].shape[1]
    rows = list(p['U'])
    rhs = list(p['v'])
    labels = [f'friction:{i}' for i in range(len(rows))]
    for i, (lo, hi) in enumerate(p['bounds']):
        if np.isfinite(lo):
            r = np.zeros(n)
            r[i] = -1.0
            rows.append(r)
            rhs.append(-lo)
            labels.append(f'lower:{i}')
        if np.isfinite(hi):
            r = np.zeros(n)
            r[i] = 1.0
            rows.append(r)
            rhs.append(hi)
            labels.append(f'upper:{i}')
    return np.array(rows).reshape(-1, n), np.array(rhs), labels


def case_name(pi, mode, knob, val):
    return f'pose{pi}_{mode}_{knob}_{val:g}'


def solvable_rows(p):
    from scipy.optimize import linprog
    n = p['E'].shape[1]
    return linprog(np.zeros(n), A_eq=p['E'] / p['row_scale'][:, None], b_eq=p['b'] / p['row_scale'],
                   A_ub=p['U'] / p['ineq_scale'][:, None] if len(p['U']) else None,
                   b_ub=p['v'] / p['ineq_scale'] if len(p['U']) else None,
                   bounds=[(None if np.isneginf(lo) else float(lo), None if np.isposinf(hi) else float(hi))
                           for lo, hi in p['bounds']],
                   method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                            'dual_feasibility_tolerance': 1e-9})


def gate_from_x(p, x, u):
    res = p['E'] @ x - p['b']
    fr = (p['U'] @ x - p['v']) if len(p['U']) else np.zeros(0)
    blo = float(np.max(p['bounds'][:, 0] - x, initial=0.0))
    bhi = float(np.max(x - p['bounds'][:, 1], initial=0.0))
    return dict(residual=max(mx(res), 0.0), friction=float(np.max(fr, initial=0.0)),
                bounds_low=blo, bounds_high=bhi)


# ---------------------------------------------------------------- prepare
def prepare():
    assert not O.exists(), 'package already exists; use a new timestamp instead'
    O.mkdir(parents=True)
    inputs = [Path(__file__), R / 'AGENTS.md', R / 'GOAL_ROADMAP.zh-CN.md', R / 'GOAL_TRACKER.zh-CN.md', XML]
    for pi in range(3):
        inputs += [M / f'POSE_{pi}.npz', M / f'LP_{pi}_{ARM}.npz', N / f'REACTIONS_{pi}.npz',
                   N / f'NATIVE_{pi}.npz', N / f'CONTACTS_{pi}.json', N / f'LP_{pi}_{ARM}_outerbox.npz',
                   N / f'LP_{pi}_{ARM}_native.npz', N / f'FARKAS_{pi}.npz']
    inputs += [M / 'POSE_CONTACTS.json', N / 'RESULT.json', N / 'CERTIFICATE_LABELS.json',
               N / 'MODEL_SCHEMA.json', OPER / 'MODEL_METADATA.json']
    hashes = {str(p): sha(p) for p in inputs if p.exists()}
    dump('SOURCE_HASHES_BEFORE.json', hashes)
    dump('PREREGISTRATION.json', dict(
        head='80215d6', stage='S2/S4',
        question='Treating body/contact parameters as tunable, can a slight adjustment of one knob at a time produce a static load-bearing (static equilibrium) witness on the three sealed saved states?',
        scope='Bounded parameter SENSITIVITY QUANTIFICATION only. No default model/XML/muscle/real network/old package/tracker change. No STDP. No claim that a parameter scan is a legitimate body change or a neural-autonomy success.',
        sources=dict(mapped=str(M), native=str(N), operating_range=str(OPER)),
        reconstruction=dict(
            A='A = moment.T * gain[None,:] (== sealed A90); A_bias = moment.T @ bias; base0 = qfrc_passive - qfrc_bias',
            default='E0 = [A@P, contact, internal]; b0 = -base0 - A@(FLOOR*1); activation a = FLOOR + P x; FLOOR = 1e-4',
            knob_s='Fmax / global force-amplitude multiplier: A_s = moment.T*(s*gain) = s*A, base_s = (qfrc_passive-qfrc_bias) + s*(moment.T@bias); muscle block s*A@P; b_s = -base_s - s*A@(FLOOR*1)',
            knob_s_warning='s scales the WHOLE actuator force, i.e. it multiplies the gain (active) columns AND the bias (act=0 passive/elastic muscle force) term together. It is NOT a pure active-column knob, and it does not touch qfrc_passive/qfrc_bias or contact/internal columns.',
            knob_u='activation ceiling multiplier: only bounds become [0, u-FLOOR]; A, base, bias and b are unchanged, so u and s are not interchangeable knobs',
            knob_k='contact friction coefficient multiplier: outerbox U rows [sx,sy,-k*mu] (|Fx|<=k*mu*Fz box form as sealed); native pyramidal ray rows rebuilt as T_i + k*Npt with Npt=mu*jac*n recovered from the sealed dense efc_J',
            internal='non-foot/limit internal columns and their bounds never scale; presence is only recorded'),
        grids=dict(s=S_GRID, u=U_GRID, k=K_GRID),
        design='per pose: three independent 1-D sweeps on the outerbox model, other knobs held at 1. The all-1 case is the already sealed default LP and is reused (cached solver report), not re-solved. Native mode is re-checked once per knob only if that knob produced a feasible outerbox multiplier.',
        gates=dict(feasible='HiGHS status 0 AND max|E x-b|<=1e-8 AND max(U x-v)<=1e-8 AND bounds violation<=1e-10 AND activation in [FLOOR,u] tol 1e-10',
                   min_multiplier='smallest PRE-REGISTERED grid value that is feasible; no bisection, no refinement',
                   unreachable='no grid value feasible -> NOT_REACHABLE_WITHIN_SCANNED_RANGE with the scanned maximum named',
                   slight='minimum multiplier > 2 marks "slight adjustment insufficient" for THIS pose/contact set/muscle boundary only'),
        forbidden='no multi-knob combination search, no continuous optimisation or bisection, no pose/contact/site scanning, no extra grid points, no second solve of the sealed default LP, no witness fabrication',
        phase2=dict(trigger='only if Phase1 minimum multiplier <= 1.25 for some pose; otherwise explicitly SKIPPED (skipped is not a pass)',
                    muscles='per pose top-12 muscles actually carrying torque at the joints of the 3 largest-|y| equality dual components (derived from the sealed FARKAS y because the package has no DUAL_COMPONENTS.json; its absence is reported)',
                    delta='move the distal end site of the muscle tendon along the two-end-site line by +-1% and +-2% of the original end-site distance; 1 compile + 3 static forwards + 1 LP per (muscle, delta) level; mesh geometry, joint range and body segmentation must be reported unchanged; the linear extrapolation is NOT a feasibility proof',
                    xml='only a research clone inside this package; the default/sealed XML is never written'),
        budgets=dict(model_compile=0, forward=0, body_step=0, neural_step=0, network=0, training=0,
                     plasticity=0, feasibility_lp=48, native_recheck_lp_max=9, phase2_compile_max=48,
                     phase2_forward_max=144, phase2_lp_max=48),
        units='all values in MODEL NATIVE UNITS; absolute biological SI NOT_ESTABLISHED',
        no_seal_before_parent=True))
    shutil.copyfile(__file__, O / 'EXECUTION_SOURCE.py')
    calls = dict(model_compile=0, forward=0, contact_jacobian=0, muscleGain=0, muscleBias=0,
                 body_step=0, neural_step=0, network=0, training=0, plasticity=0, linprog=0)
    try:
        sealed_status = {f"{r['pose']}_{r['arm']}_{r['contact_model']}": r for r in js(N / 'RESULT.json')['rows']}
        sealed_certs = {c['pose']: c for c in js(N / 'RESULT.json')['certificates']}
        labels = js(N / 'CERTIFICATE_LABELS.json')
        rebuild, farkas, duals = [], [], []
        for pi in range(3):
            nat = sealed_pose(pi)
            checks = {}
            A = nat['moment'].T * nat['gain'][None, :]
            base = nat['qp'] - nat['qb'] + nat['moment'].T @ nat['bias']
            checks['A_vs_sealed_A90'] = mx(A - nat['A90'])
            checks['base_vs_sealed'] = mx(base - nat['base'])
            checks['A_roots_zero'] = mx(A[:6])
            checks['P_shape'] = list(nat['P'].shape)
            checks['P_binary'] = bool(np.array_equal(np.unique(nat['P']), np.array([0.0, 1.0])))
            checks['P_row_sums'] = sorted({int(x) for x in np.rint(nat['P'].sum(1))})
            checks['P_col_sums'] = sorted({int(x) for x in np.rint(nat['P'].sum(0))})
            for mode in MODES:
                p = build_case(nat, 1.0, 1.0, 1.0, mode)
                sealed = ar(N / f'LP_{pi}_{ARM}_{mode}.npz')
                checks[f'{mode}_E'] = mx(p['E'] - sealed['E'])
                checks[f'{mode}_b'] = mx(p['b'] - sealed['b'])
                checks[f'{mode}_U'] = mx(p['U'] - sealed['U']) if sealed['U'].size else (0.0 if p['U'].size == 0 else mx(p['U']))
                checks[f'{mode}_v'] = float(np.max(np.abs(p['v'] - sealed['v']), initial=0.0))
                checks[f'{mode}_bounds'] = bounds_diff(p['bounds'], sealed['bounds'])
                checks[f'{mode}_row_scale'] = mx(p['row_scale'] - sealed['row_scale'])
                checks[f'{mode}_shape'] = [list(p['E'].shape), list(sealed['E'].shape)]
                # The native pyramidal columns are rebuilt through the T + k*Npt identity, which is algebraically
                # identical to the sealed rows at k=1 but can differ by a fraction of one ulp; the tolerance below
                # is 2 ulp, and every reported number keeps its exact value.
                assert checks[f'{mode}_E'] <= 1e-15 and checks[f'{mode}_b'] <= 1e-15 and checks[f'{mode}_U'] <= 1e-15
                assert checks[f'{mode}_bounds'] == 0.0 and checks[f'{mode}_row_scale'] <= 1e-12
                assert checks[f'{mode}_shape'][0] == checks[f'{mode}_shape'][1]
            # Sealed default solver reports are reused, not re-solved.
            checks['sealed_default_status_outerbox'] = sealed_status[f'{pi}_{ARM}_outerbox']['status']
            checks['sealed_default_status_native'] = sealed_status[f'{pi}_{ARM}_native']['status']
            checks['sealed_default_witness_outerbox'] = sealed_status[f'{pi}_{ARM}_outerbox']['witness']
            checks['sealed_default_witness_native'] = sealed_status[f'{pi}_{ARM}_native']['witness']
            checks['native_column_rebuild_k1'] = mx(native_columns(nat, pi, 1.0)[0] - nat['foot_columns'])
            checks['internal_columns'] = list(nat['internal_columns'].shape)
            checks['internal_bounds'] = None if not nat['internal_bounds'].size else nat['internal_bounds'].tolist()
            checks['foot_contact_count'] = int(nat['J'].shape[0])
            checks['native_nefc'] = int(nat['native_J'].shape[0])
            checks['contact_mu'] = nat['mu'].tolist()
            assert checks['native_column_rebuild_k1'] <= 1e-15  # same T + k*Npt identity, <=2 ulp at k=1
            rebuild.append(dict(pose=pi, label=nat['label'], time=nat['time'], checks=checks))
            # Old Farkas certificate must remain exactly what was sealed.
            p = build_case(nat, 1.0, 1.0, 1.0, 'outerbox')
            H, h, lab = hsystem(p)
            f = ar(N / f'FARKAS_{pi}.npz')
            y, z = f['y'], f['z']
            st = p['E'].T @ y + H.T @ z
            margin = float(p['b'] @ y + h @ z)
            fc = dict(pose=pi,
                      E_exact=mx(p['E'] - f['E']), b_exact=mx(p['b'] - f['b']),
                      H_exact=mx(H - f['H']), h_exact=mx(h - f['h']),
                      H_labels_match=bool(list(f['H_labels']) == lab),
                      stationarity=mx(st), margin=margin,
                      dual_min=float(np.min(z, initial=0.0)),
                      gate_stationarity=GATE_EQ, gate_margin=-1 + 1e-8,
                      old_established=bool(sealed_certs[pi]['established']),
                      note='Recomputed from this round\'s independent reconstruction with the sealed y,z; the previous certificate is unchanged, not re-solved.')
            assert max(fc['E_exact'], fc['b_exact'], fc['H_exact'], fc['h_exact']) <= 1e-15  # <=2 ulp re-summation
            assert fc['stationarity'] <= GATE_EQ and margin <= -1 + 1e-8 and fc['dual_min'] >= -1e-10
            farkas.append(fc)
            # Dual components (no DUAL_COMPONENTS.json exists in the sealed package).
            eq_labels = labels[pi]['equality_labels']
            zlab = lab
            upper_mus = [float(z[i]) for i, l in enumerate(zlab) if l.startswith('upper:') and int(l.split(':')[1]) < 90]
            lower_mus = [float(z[i]) for i, l in enumerate(zlab) if l.startswith('lower:') and int(l.split(':')[1]) < 90]
            order = np.argsort(-np.abs(y))
            top = []
            for j in order[:3]:
                j = int(j)
                contrib = np.abs(nat['actual_moment'][:, j] * nat['actual_force'])   # actual torque carried
                cand = np.argsort(-contrib)[:12]
                top.append(dict(dof=j, label=eq_labels[j], equality_dual=float(y[j]),
                                abs_dual=float(abs(y[j])),
                                muscle_candidates=[dict(actuator=int(u), torque_contribution=float(contrib[u]))
                                                   for u in cand if contrib[u] > 1e-12]))
            duals.append(dict(pose=pi, label=nat['label'],
                              source='sealed FARKAS_%d.npz y (48 equality duals) + CERTIFICATE_LABELS equality_labels; the referenced DUAL_COMPONENTS.json does not exist in the sealed package and is therefore derived here, not invented connections.' % pi,
                              top3_abs_equality_dual=top,
                              upper_activation_dual_L1=float(np.sum(np.abs(upper_mus))),
                              upper_activation_dual_max=float(np.max(np.abs(upper_mus), initial=0.0)),
                              lower_activation_dual_L1=float(np.sum(np.abs(lower_mus))),
                              upper_bound_dual_nonzero=bool(np.max(np.abs(upper_mus), initial=0.0) > 1e-12)))
        dump('REBUILD_CHECK.json', dict(passed=True, poses=rebuild,
                                        note='Defaults (s=u=k=1) rebuilt from raw sealed arrays reproduce the sealed LP matrices element-exactly; the sealed default solver reports are reused, not re-solved.'))
        dump('FARKAS_INVARIANCE.json', dict(passed=True, poses=farkas,
                                            note='Old floating numerical separation certificates are invariant under this reconstruction; still not a symbolic proof and not a statement about other poses/contacts.'))
        dump('DUAL_COMPONENTS.json', dict(passed=True, poses=duals,
                                          derived_from_sealed='FARKAS_*.npz y/z + CERTIFICATE_LABELS.json',
                                          caveat='Dual components are non-unique; the largest |y| component does not identify a single root cause. Muscle candidates use the actual saved torque contribution |actual_moment*actual_force| at the saved state, which is a sampling rule, not anatomy.'))
        dump('CALLS.json', dict(stage='prepare', calls=calls))
    except Exception:
        dump('PREPARE_FAILURE.json', dict(calls=calls, error=traceback.format_exc()))
        raise
    print(json.dumps(dict(stage='prepare', passed=True, calls=calls), ensure_ascii=False))


# ---------------------------------------------------------------- sweep (Phase 1)
def sweep():
    from scipy.optimize import linprog  # noqa: F401  (import guard for honest counting)
    assert not (O / 'RESULT.json').exists()
    calls = dict(model_compile=0, forward=0, contact_jacobian=0, muscleGain=0, muscleBias=0,
                 body_step=0, neural_step=0, network=0, training=0, plasticity=0, linprog=0)
    sealed_status = {f"{r['pose']}_{r['arm']}_{r['contact_model']}": r for r in js(N / 'RESULT.json')['rows']}
    WEIGHT = float(js(M / 'COLLECT_EXECUTION.json')['weight'])
    store = {}
    rows = []
    per_pose = []
    for pi in range(3):
        nat = sealed_pose(pi)
        # bounded de-duplication: the all-1 default case is the sealed LP already solved
        base_def = build_case(nat, 1.0, 1.0, 1.0, 'outerbox')
        for mode in MODES:
            defc = build_case(nat, 1.0, 1.0, 1.0, mode)
            srow = sealed_status[f'{pi}_{ARM}_{mode}']
            rows.append(dict(pose=pi, label=nat['label'], mode=mode, knob='none', value=1.0,
                             solved=False, cached=True,
                             status=int(srow['status']), feasible=False,
                             note='sealed default LP reused (already solved in the previous package); not re-solved',
                             sealed_status=int(srow['status']), sealed_witness=bool(srow.get('witness'))))
        knob_rows = {}
        for knob, grid in KNOBS:
            cur = []
            for val in grid:
                s = val if knob == 's' else 1.0
                u = val if knob == 'u' else 1.0
                k = val if knob == 'k' else 1.0
                if knob == 's' and val == 1.0:
                    continue  # cached default, no re-solve
                if knob in ('u', 'k') and val == 1.0:
                    continue
                p = build_case(nat, s, u, k, 'outerbox')
                r = solvable_rows(p)
                calls['linprog'] += 1
                name = case_name(pi, 'outerbox', knob, val)
                fam = gate_from_x(p, r.x, u) if r.success else None
                feas = False
                act_min = act_max = None
                contact_scale = None
                if r.success:
                    x = r.x
                    act = FLOOR + p['P'] @ x[:90]
                    act_min, act_max = float(np.min(act)), float(np.max(act))
                    feas = bool(fam['residual'] <= GATE_EQ and fam['friction'] <= GATE_FRIC
                                and fam['bounds_low'] <= GATE_BOUNDS and fam['bounds_high'] <= GATE_BOUNDS
                                and act_min >= FLOOR - GATE_ACT and act_max <= u + GATE_ACT)
                    F = x[90:90 + 3 * nat['J'].shape[0]].reshape(-1, 3)
                    contact_scale = dict(max_abs_component=float(np.max(np.abs(F), initial=0.0)),
                                         normal_sum=float(np.sum(F[:, 2])),
                                         weight=WEIGHT,
                                         normal_sum_over_weight=float(np.sum(F[:, 2]) / WEIGHT))
                    store[f'{name}__x'] = x
                store[f'{name}__E'] = p['E']
                store[f'{name}__b'] = p['b']
                store[f'{name}__U'] = p['U']
                store[f'{name}__v'] = p['v']
                store[f'{name}__bounds'] = p['bounds']
                store[f'{name}__row_scale'] = p['row_scale']
                row = dict(pose=pi, label=nat['label'], mode='outerbox', knob=knob, value=val,
                           solved=True, cached=False, status=int(r.status), feasible=bool(feas),
                           nit=int(r.nit), message=str(r.message)[:120],
                           residual=None if fam is None else fam['residual'],
                           friction=None if fam is None else fam['friction'],
                           bounds_violation=None if fam is None else max(fam['bounds_low'], fam['bounds_high']),
                           activation_min=act_min, activation_max=act_max,
                           activation_ceiling=u, contact_force=contact_scale)
                rows.append(row)
                cur.append(row)
                print(json.dumps(row), flush=True)
            knob_rows[knob] = cur
        # minimum feasible multiplier + native re-check, per knob
        summary = {}
        for knob, grid in KNOBS:
            feas_vals = [r['value'] for r in knob_rows[knob] if r['feasible']]
            if knob == 's' or knob == 'u' or knob == 'k':
                # value 1.0 is the cached default which is infeasible in all three poses
                cached1 = False
            first = min(feas_vals) if feas_vals else None
            entry = dict(knob=knob, grid=grid, feasible_values=feas_vals, min_multiplier=first,
                         required_relative_increase=None if first is None else float(first - 1.0),
                         reachable=first is not None,
                         maximum_scanned=max(grid),
                         verdict=None)
            if first is None:
                entry['verdict'] = 'NOT_REACHABLE_WITHIN_SCANNED_RANGE'
            elif first > 2.0:
                entry['verdict'] = 'SLIGHT_ADJUSTMENT_INSUFFICIENT'
            elif first > 1.0:
                entry['verdict'] = 'SLIGHT_ADJUSTMENT_POSSIBLE_WITHIN_SCANNED_RANGE'
            else:
                entry['verdict'] = 'ALREADY_FEASIBLE_AT_DEFAULT'
            summary[knob] = entry
        # native re-verification only where the outerbox gave a feasible multiplier
        for knob, entry in summary.items():
            if not entry['reachable']:
                entry['native_recheck'] = dict(done=False, reason='no feasible outerbox multiplier within the scanned range')
                continue
            val = entry['min_multiplier']
            s = val if knob == 's' else 1.0
            u = val if knob == 'u' else 1.0
            k = val if knob == 'k' else 1.0
            p = build_case(nat, s, u, k, 'native')
            r = solvable_rows(p)
            calls['linprog'] += 1
            feas = False
            fam = None
            if r.success:
                x = r.x
                fam = gate_from_x(p, x, u)
                act = FLOOR + p['P'] @ x[:90]
                feas = bool(fam['residual'] <= GATE_EQ and fam['friction'] <= GATE_FRIC
                            and fam['bounds_low'] <= GATE_BOUNDS and fam['bounds_high'] <= GATE_BOUNDS
                            and float(np.min(act)) >= FLOOR - GATE_ACT and float(np.max(act)) <= u + GATE_ACT)
                store[f'{case_name(pi,"native",knob,val)}__x'] = x
            for sfx, arr in [('E', p['E']), ('b', p['b']), ('U', p['U']), ('v', p['v']),
                             ('bounds', p['bounds']), ('row_scale', p['row_scale'])]:
                store[f'{case_name(pi,"native",knob,val)}__{sfx}'] = arr
            nrow = dict(pose=pi, label=nat['label'], mode='native', knob=knob, value=val,
                        solved=True, cached=False, status=int(r.status), feasible=bool(feas),
                        nit=int(r.nit), message=str(r.message)[:120],
                        residual=None if fam is None else fam['residual'],
                        friction=None if fam is None else fam['friction'])
            rows.append(nrow)
            entry['native_recheck'] = dict(done=True, **{kk: nrow[kk] for kk in
                                                         ['value', 'status', 'feasible', 'residual', 'friction']})
            print(json.dumps(nrow), flush=True)
        per_pose.append(dict(pose=pi, label=nat['label'], time=nat['time'], knobs=summary,
                             foot_contacts=int(nat['J'].shape[0]), mu=nat['mu'].tolist(),
                             internal_columns=int(nat['internal_columns'].shape[1]),
                             activation_actual_max=float(np.max(nat['activation_actual'])),
                             default_model=dict(status_outerbox=int(sealed_status[f'{pi}_{ARM}_outerbox']['status']),
                                                status_native=int(sealed_status[f'{pi}_{ARM}_native']['status']))))
    assert calls['linprog'] <= 48 + 9, calls
    np.savez_compressed(O / 'LP_SWEEP.npz', **store)
    dump('SOLVER_CALL_GATE.json', dict(linprog=calls['linprog'], maximum=48 + 9,
                                       default_resolved=0,
                                       note='The sealed all-1 default LP in each of the two contact models is reused, not re-solved.'))
    any_slight = any(e['knob'] and e['min_multiplier'] is not None and e['min_multiplier'] <= PHASE2_MULT_MAX
                     for pp in per_pose for e in pp['knobs'].values())
    dump('RESULT.json', dict(
        passed=True, calls=calls, poses=per_pose, rows=rows,
        phase2_trigger=bool(any_slight), phase2_threshold=PHASE2_MULT_MAX,
        scope='Only these three saved states, their contact sets, the independent90 muscle boundary and the scanned grids. Status2 = solver report, not an automatic mathematical certificate; the sealed Farkas separation only covers the all-1 outerbox problem.',
        units='native model units; absolute biological SI NOT_ESTABLISHED; multipliers are an engineering calibration hypothesis, not measured anatomy and not permission to alter the model.'))

    # honest per-pose roll-up for the parent's urgent question
    for pp in per_pose:
        print(json.dumps(dict(pose=pp['pose'], label=pp['label'],
                              min_multipliers={k: v['min_multiplier'] for k, v in pp['knobs'].items()},
                              verdicts={k: v['verdict'] for k, v in pp['knobs'].items()})), flush=True)


# ---------------------------------------------------------------- phase2
def phase2():
    res = js(O / 'RESULT.json')
    trig = []
    for pp in res['poses']:
        for knob, e in pp['knobs'].items():
            if e['min_multiplier'] is not None and e['min_multiplier'] <= PHASE2_MULT_MAX:
                trig.append(dict(pose=pp['pose'], label=pp['label'], knob=knob, multiplier=e['min_multiplier']))
    if not trig:
        dump('PHASE2_SKIPPED.json', dict(
            executed=False, reason='Phase1 found no knob with a feasible outerbox multiplier <= %g in any of the three poses' % PHASE2_MULT_MAX,
            threshold=PHASE2_MULT_MAX,
            per_pose_min={pp['label']: {k: v['min_multiplier'] for k, v in pp['knobs'].items()} for pp in res['poses']},
            per_pose_verdict={pp['label']: {k: v['verdict'] for k, v in pp['knobs'].items()} for pp in res['poses']},
            skipped_is_a_pass=False,
            interpretation='Attachment-site finite differences are not executed because the pre-registered trigger was not met. A skipped Phase2 is not evidence that site tuning cannot help; it is untested here. Every multiplier above is an engineering sensitivity of a conditional static model, not a measured anatomical parameter.'))
        print(json.dumps(dict(stage='phase2', executed=False, reason='trigger_not_met'), ensure_ascii=False))
        return
    raise AssertionError('Phase2 trigger met; the pre-registered site-offset protocol must be implemented and run: %s' % json.dumps(trig))


# ---------------------------------------------------------------- verify (fresh pure arrays)
def verify():
    """Independent fresh pure-array verification: no scipy, no MuJoCo, no new solve."""
    own = dict(model_compile=0, forward=0, contact_jacobian=0, muscleGain=0, muscleBias=0,
               body_step=0, neural_step=0, network=0, training=0, plasticity=0, linprog=0)
    res = js(O / 'RESULT.json')
    sealed_status = {f"{r['pose']}_{r['arm']}_{r['contact_model']}": r for r in js(N / 'RESULT.json')['rows']}
    store = ar(O / 'LP_SWEEP.npz')
    out = dict(passed=False, checks={}, calls=own,
               statement='Fresh scalar arithmetic only. No LP is re-solved, no model is loaded, no body/neural integration.')

    def eq(a, b):
        return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)), initial=0.0))

    def rel(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        den = max(1.0, float(np.max(np.abs(b), initial=0.0)))
        return eq(a, b) / den

    # shared fresh scalar reconstruction of the sealed default quantities
    def fresh(pi):
        p = ar(M / f'POSE_{pi}.npz')
        old = ar(M / f'LP_{pi}_{ARM}.npz')
        r = ar(N / f'REACTIONS_{pi}.npz')
        n = old['P'].shape[1]
        mom = [[float(p['moment'][u, j]) for u in range(90)] for j in range(48)]
        gain = [float(x) for x in p['gain']]
        bias = [float(x) for x in p['bias']]
        qp = [float(x) for x in p['qfrc_passive']]
        qb = [float(x) for x in p['qfrc_bias']]
        A = [[mom[j][u] * gain[u] for u in range(90)] for j in range(48)]  # A[j][u] = moment[u,j]*gain[u]
        Abias = [math.fsum(mom[j][u] * bias[u] for u in range(90)) for j in range(48)]
        P = [[float(old['P'][u, kk]) for kk in range(n)] for u in range(90)]
        muscle0 = [[math.fsum(A[j][u] * P[u][kk] for u in range(90)) for kk in range(n)] for j in range(48)]
        base = [(qp[j] - qb[j]) + Abias[j] for j in range(48)]
        b0 = [-base[j] - FLOOR * math.fsum(A[j][u] for u in range(90)) for j in range(48)]
        return dict(p=p, old=old, r=r, n=n, mom=mom, gain=gain, bias=bias, qp=qp, qb=qb,
                    A=A, Abias=Abias, P=P, muscle0=muscle0, base=base, b0=b0)

    def outer_fresh(d, k=1.0, u=1.0):
        p, n = d['p'], d['n']
        contact = p['J'].transpose(2, 0, 1).reshape(48, -1)
        nc = p['J'].shape[0]
        nint = d['r']['internal_columns'].shape[1]
        U = np.zeros((4 * nc, n + 3 * nc + nint))
        for c in range(nc):
            for f, (sx, sy) in enumerate([(1., 0.), (-1., 0.), (0., 1.), (0., -1.)]):
                U[4 * c + f, n + 3 * c] = sx
                U[4 * c + f, n + 3 * c + 1] = sy
                U[4 * c + f, n + 3 * c + 2] = -float(p['mu'][c]) * k
        bounds = np.array([[0.0, u - FLOOR]] * n
                          + [[-math.inf, math.inf], [-math.inf, math.inf], [0.0, math.inf]] * nc
                          + [list(x) for x in d['r']['internal_bounds']], dtype=float)
        return contact, U, bounds

    # ---- R1: sealed default matrices rebuilt by fresh scalar arithmetic
    r1 = []
    for pi in range(3):
        d = fresh(pi)
        p, old, r, n = d['p'], d['old'], d['r'], d['n']
        contact, U, bounds = outer_fresh(d, 1.0)
        chk = dict(pose=pi,
                   A_vs_sealed_A90=eq(np.array(d['A']), p['A90']),
                   base_vs_sealed=eq(np.array(d['base']), p['base']),
                   muscle_block_vs_old_Aeq=eq(np.array(d['muscle0']), old['A_eq'][:, :n]),
                   b_vs_old_beq=eq(np.array(d['b0']), old['b_eq']))
        E = np.column_stack([np.array(d['muscle0']), contact, r['internal_columns']])
        sealed = ar(N / f'LP_{pi}_{ARM}_outerbox.npz')
        chk['outerbox_E'] = eq(E, sealed['E'])
        chk['outerbox_b'] = eq(np.array(d['b0']), sealed['b'])
        chk['outerbox_U'] = eq(U, sealed['U'])
        chk['outerbox_bounds'] = bounds_diff(bounds, sealed['bounds'])
        J = ar(N / f'NATIVE_{pi}.npz')['J']
        cols = []
        for c in [cc for cc in js(N / f'CONTACTS_{pi}.json') if cc['foot_floor']]:
            Rm = J[c['rows']]
            Npt = (Rm[0] + Rm[1]) / 2.0
            T1 = (Rm[0] - Rm[1]) / 2.0
            T2 = (Rm[2] - Rm[3]) / 2.0
            cols += [T1 + Npt, -T1 + Npt, T2 + Npt, -T2 + Npt]
        natcol = np.column_stack(cols)
        chk['native_cols_vs_reactions'] = eq(natcol, r['native_foot_columns'])
        chk['native_normal_part_consistency'] = float(max(
            mx((J[c['rows']][0] + J[c['rows']][1]) / 2 - (J[c['rows']][2] + J[c['rows']][3]) / 2)
            for c in js(N / f'CONTACTS_{pi}.json') if c['foot_floor']))
        sealedn = ar(N / f'LP_{pi}_{ARM}_native.npz')
        En = np.column_stack([np.array(d['muscle0']), natcol, r['internal_columns']])
        bn = np.array([[0.0, math.inf]] * natcol.shape[1] + [list(x) for x in r['internal_bounds']], dtype=float)
        boundsn = np.array([[0.0, 1.0 - FLOOR]] * n + [list(x) for x in bn], dtype=float)
        chk['native_E'] = eq(En, sealedn['E'])
        chk['native_b'] = eq(np.array(d['b0']), sealedn['b'])
        chk['native_bounds'] = bounds_diff(boundsn, sealedn['bounds'])
        chk['native_U_empty'] = bool(sealedn['U'].size == 0)
        r1.append(chk)
    out['checks']['R1_sealed_default_rebuild'] = r1
    for c in r1:
        for kk, v in c.items():
            if kk == 'pose' or isinstance(v, bool):
                continue
            lim = 1e-15 if kk.endswith(('_E', '_b', '_U', 'cols_vs_reactions')) else 1e-12
            assert float(v) <= lim, (kk, v)
        assert c['native_U_empty'] is True
    print('R1 ok', flush=True)

    # ---- R2/R3/R4: scaled formulas, friction width, untouched blocks
    r2, r3, r4 = [], [], []
    for pi in range(3):
        d = fresh(pi)
        p, old, r, n = d['p'], d['old'], d['r'], d['n']
        defsealed = ar(N / f'LP_{pi}_{ARM}_outerbox.npz')
        for knob, grid in KNOBS:
            for val in grid:
                nm = case_name(pi, 'outerbox', knob, val)
                if f'{nm}__E' not in store:
                    continue
                s = val if knob == 's' else 1.0
                u = val if knob == 'u' else 1.0
                k = val if knob == 'k' else 1.0
                E = store[f'{nm}__E']
                b = store[f'{nm}__b']
                U = store[f'{nm}__U']
                bounds = store[f'{nm}__bounds']
                musc = np.array([[s * math.fsum(d['A'][j][uu] * d['P'][uu][kk] for uu in range(90))
                                  for kk in range(n)] for j in range(48)])
                # base_s = (qp-qb) + s*Abias  =>  b_s = -(qp-qb) - s*Abias - s*FLOOR*sum(A)
                bvec = np.array([-((d['qp'][j] - d['qb'][j]) + s * d['Abias'][j])
                                 - s * FLOOR * math.fsum(d['A'][j][uu] for uu in range(90)) for j in range(48)])
                _, Uexp, boundsexp = outer_fresh(d, k, u)
                ent = dict(pose=pi, knob=knob, value=val,
                           muscle_block_rel=rel(musc, E[:, :n]),
                           b_rel=rel(bvec, b),
                           contact_block_unchanged=eq(E[:, n:n + 3 * p['J'].shape[0]],
                                                      defsealed['E'][:, n:n + 3 * p['J'].shape[0]]),
                           internal_block_unchanged=(eq(E[:, n + 3 * p['J'].shape[0]:],
                                                        defsealed['E'][:, n + 3 * p['J'].shape[0]:])
                                                     if E.shape[1] > n + 3 * p['J'].shape[0] else 0.0),
                           bounds_expected_ok=bool(np.array_equal(bounds, boundsexp)),
                           U_expected_ok=bool(np.array_equal(U, Uexp)))
                r2.append(ent)
                if knob == 'k':
                    r3.append(dict(pose=pi, value=val, U_rows_exact=eq(Uexp, U),
                                   nonzero_rows=int(np.count_nonzero(Uexp)),
                                   v_zero=bool(np.all(store[f'{nm}__v'] == 0.0)),
                                   meaning='row [sx,sy,-k*mu] <= 0 <=> |Fx| <= k*mu*Fz and |Fy| <= k*mu*Fz'))
                r4.append(dict(pose=pi, knob=knob, value=val,
                               U_equals_default=eq(U, defsealed['U']),
                               bounds_equal_default=bool(np.array_equal(bounds, defsealed['bounds'])),
                               E_equals_default=eq(E, defsealed['E']),
                               b_equals_default=eq(b, defsealed['b'])))
    out['checks']['R2_scaled_formula'] = r2
    out['checks']['R3_friction_width'] = r3
    out['checks']['R4_untouched_blocks'] = r4
    for e in r2:
        assert e['muscle_block_rel'] <= 1e-12 and e['b_rel'] <= 1e-12, e
        assert e['contact_block_unchanged'] == 0.0 and e['internal_block_unchanged'] == 0.0, e
        assert e['bounds_expected_ok'] and e['U_expected_ok'], e
    for e in r3:
        assert e['U_rows_exact'] == 0.0 and e['v_zero'], e
    for e in r4:
        if e['knob'] in ('s', 'u'):
            assert e['U_equals_default'] == 0.0, e          # friction rows untouched by s and u
        if e['knob'] == 'u':
            assert e['E_equals_default'] <= 1e-15 and e['b_equals_default'] <= 1e-15, e   # u only moves bounds (<=2 ulp)
        if e['knob'] in ('s', 'k'):
            assert e['bounds_equal_default'], e             # s and k never touch the activation box
        if e['knob'] == 's':
            assert e['E_equals_default'] > 0.0, e           # the knob is not a no-op
        if e['knob'] == 'k':
            assert e['U_equals_default'] > 0.0, e           # the knob is not a no-op
    print('R2/R3/R4 ok', flush=True)

    # ---- R5: every case's original conditions + selection rule + cached default reuse
    r5 = []
    for row in res['rows']:
        if not row['solved']:
            sealed = sealed_status[f"{row['pose']}_{ARM}_{row['mode']}"]
            assert int(row['status']) == int(sealed['status']) and row['feasible'] is False and row['cached']
            r5.append(dict(pose=row['pose'], mode=row['mode'], knob='none', value=1.0, cached=True,
                           status_matches_seal=True, no_new_solve=True,
                           sealed_status=int(sealed['status']), sealed_witness=bool(sealed['witness'])))
            continue
        nm = case_name(row['pose'], row['mode'], row['knob'], row['value'])
        E = store[f'{nm}__E']
        b = store[f'{nm}__b']
        U = store[f'{nm}__U']
        v = store[f'{nm}__v']
        bounds = store[f'{nm}__bounds']
        if f'{nm}__x' not in store:
            assert int(row['status']) != 0, row
            r5.append(dict(pose=row['pose'], mode=row['mode'], knob=row['knob'], value=row['value'],
                           status=int(row['status']), witness_stored=False, recorded_feasible=False,
                           note='no primal witness returned; the status is a solver report only, not a certificate'))
            continue
        x = [float(t) for t in store[f'{nm}__x']]
        res_eq = np.array([math.fsum(float(E[j, c]) * x[c] for c in range(E.shape[1])) - float(b[j])
                           for j in range(E.shape[0])])
        fri = (np.array([math.fsum(float(U[j, c]) * x[c] for c in range(U.shape[1])) - float(v[j])
                         for j in range(U.shape[0])]) if len(U) else np.zeros(0))
        viol = max(float(np.max(bounds[:, 0] - np.array(x), initial=0.0)),
                   float(np.max(np.array(x) - bounds[:, 1], initial=0.0)))
        recomputed = bool(np.max(np.abs(res_eq), initial=0.0) <= GATE_EQ
                          and (not len(fri) or float(np.max(fri, initial=0.0)) <= GATE_FRIC)
                          and viol <= GATE_BOUNDS)
        r5.append(dict(pose=row['pose'], mode=row['mode'], knob=row['knob'], value=row['value'],
                       status=int(row['status']), residual=float(np.max(np.abs(res_eq), initial=0.0)),
                       friction=float(np.max(fri, initial=0.0)) if len(fri) else 0.0,
                       bounds_violation=viol, recorded_feasible=bool(row['feasible']),
                       recomputed_feasible=recomputed, witness_stored=True,
                       note='recomputed from the stored witness with fresh scalar sums; no solve'))
        assert recomputed == bool(row['feasible']), r5[-1]
    out['checks']['R5_per_case_conditions'] = r5
    sel = []
    for pp in res['poses']:
        for knob, grid in KNOBS:
            feas_vals = sorted(r['value'] for r in res['rows']
                               if r['pose'] == pp['pose'] and r['mode'] == 'outerbox' and r['knob'] == knob
                               and r['solved'] and r['feasible'])
            e = pp['knobs'][knob]
            want = min(feas_vals) if feas_vals else None
            expect = ('NOT_REACHABLE_WITHIN_SCANNED_RANGE' if want is None
                      else ('SLIGHT_ADJUSTMENT_INSUFFICIENT' if want > 2.0
                            else 'SLIGHT_ADJUSTMENT_POSSIBLE_WITHIN_SCANNED_RANGE'))
            sel.append(dict(pose=pp['pose'], knob=knob, recomputed_min=want, recorded_min=e['min_multiplier'],
                            matches=(want == e['min_multiplier']),
                            grid_is_preregistered=(grid == dict(KNOBS)[knob]),
                            verdict_matches=(e['verdict'] == expect),
                            required_increase_ok=(e['required_relative_increase'] is None if want is None
                                                  else abs(e['required_relative_increase'] - (want - 1.0)) <= 1e-15)))
            assert sel[-1]['matches'] and sel[-1]['grid_is_preregistered'] and sel[-1]['verdict_matches'] and sel[-1]['required_increase_ok'], sel[-1]
    out['checks']['R5b_selection_rule'] = sel
    print('R5 ok', flush=True)

    # ---- R6: old Farkas certificate invariance, fresh scalar
    r6 = []
    for pi in range(3):
        d = fresh(pi)
        p, r, n = d['p'], d['r'], d['n']
        f = ar(N / f'FARKAS_{pi}.npz')
        contact, U, bounds = outer_fresh(d, 1.0)
        E = np.column_stack([np.array(d['muscle0']), contact, r['internal_columns']])
        Hrows = [list(U[i]) for i in range(len(U))]
        hrhs = [0.0] * len(U)
        for i, (lo, hi) in enumerate(bounds):
            if np.isfinite(lo):
                rr = [0.0] * E.shape[1]
                rr[i] = -1.0
                Hrows.append(rr)
                hrhs.append(-float(lo))
            if np.isfinite(hi):
                rr = [0.0] * E.shape[1]
                rr[i] = 1.0
                Hrows.append(rr)
                hrhs.append(float(hi))
        H = np.array(Hrows)
        h = np.array(hrhs)
        y = [float(t) for t in f['y']]
        z = [float(t) for t in f['z']]
        st = np.array([math.fsum(float(E[j, c]) * y[j] for j in range(48))
                       + math.fsum(float(H[j, c]) * z[j] for j in range(H.shape[0]))
                       for c in range(E.shape[1])])
        margin = (math.fsum(float(d['b0'][j]) * y[j] for j in range(48))
                  + math.fsum(float(h[j]) * z[j] for j in range(H.shape[0])))
        ent = dict(pose=pi, E_exact=eq(E, f['E']), b_exact=eq(np.array(d['b0']), f['b']),
                   H_exact=eq(H, f['H']), h_exact=eq(h, f['h']), H_rows=int(H.shape[0]),
                   stationarity=float(np.max(np.abs(st), initial=0.0)), margin=float(margin),
                   dual_min=float(np.min(z, initial=0.0)),
                   upper_activation_dual_L1=float(np.sum([abs(z[i]) for i, l in enumerate(f['H_labels'])
                                                          if str(l).startswith('upper:') and int(str(l).split(':')[1]) < 90])))
        assert max(ent['E_exact'], ent['b_exact'], ent['H_exact'], ent['h_exact']) <= 1e-15  # <=2 ulp
        assert ent['stationarity'] <= GATE_EQ and ent['margin'] <= -1 + 1e-8 and ent['dual_min'] >= -1e-10
        r6.append(ent)
    out['checks']['R6_old_farkas_invariance'] = r6
    print('R6 ok', flush=True)

    # ---- R7: Phase2 (explicitly skipped unless the trigger fired) + R8 calls
    p2 = js(O / 'PHASE2_SKIPPED.json') if (O / 'PHASE2_SKIPPED.json').exists() else None
    out['checks']['R7_phase2'] = dict(
        executed=False, skipped=bool(p2 and not p2['executed']), threshold=PHASE2_MULT_MAX,
        trigger_values={k['label']: {kk: vv['min_multiplier'] for kk, vv in k['knobs'].items()} for k in res['poses']},
        note='No site-offset arithmetic exists to verify because the pre-registered trigger was not met; nothing was extrapolated and no linear extrapolation is claimed.')
    for kk, v in own.items():
        assert v == 0, (kk, v)
    ctl = js(O / 'SOLVER_SANITY_CONTROL.json')
    out['checks']['R8_calls'] = dict(
        verify=own, phase1=res['calls'], control=ctl['calls'],
        control_passed=bool(ctl['passed']), control_declared_post_hoc=bool(ctl['declared_post_hoc']),
        phase1_compile_zero=int(res['calls']['model_compile']) == 0,
        phase1_forward_zero=int(res['calls']['forward']) == 0,
        linprog_phase1=int(res['calls']['linprog']),
        linprog_control=int(ctl['calls']['linprog']),
        linprog_total=int(res['calls']['linprog']) + int(ctl['calls']['linprog']),
        linprog_within_budget=int(res['calls']['linprog']) <= 48 + 9,
        note='Sweep budget stays the pre-registered 48+<=9; the 2 control LPs are the separately declared post-hoc non-degeneracy check.')
    assert out['checks']['R8_calls']['phase1_compile_zero'] and out['checks']['R8_calls']['phase1_forward_zero']
    assert out['checks']['R8_calls']['linprog_within_budget'] and out['checks']['R8_calls']['control_passed']
    out['passed'] = True
    dump('INDEPENDENT_ARRAY_VERIFICATION.json', out)
    print(json.dumps(dict(stage='verify', passed=True, checks=len(out['checks'])), ensure_ascii=False))


def control():
    """Post-hoc non-degeneracy control for the feasibility formulation (declared post-hoc).

    The pre-registered budget is exactly the 48 sweep LP plus at most 9 native re-checks. This
    control is added AFTER the sweep to show that the "all infeasible" signal is not a coding
    artifact: with the identical E/U/bounds of a real case, setting the right-hand side to
    delta * E[:, m] (delta inside the activation box) has the exact feasible point
    x = delta * e_m, so a correct formulation must return status 0 and pass the same gates.
    It never changes a scientific verdict and is reported in its own file.
    """
    from scipy.optimize import linprog
    assert not (O / 'SOLVER_SANITY_CONTROL.json').exists()
    calls = dict(linprog=0, model_compile=0, forward=0, body_step=0, neural_step=0,
                 network=0, training=0, plasticity=0)
    cases = []
    for pi in [1]:
        nat = sealed_pose(pi)
        p = build_case(nat, 1.0, 1.0, 1.0, 'outerbox')
        for m in [0, 53]:
            delta = 0.5 * (p['bounds'][m, 1] - p['bounds'][m, 0])
            b = p['E'][:, m] * delta
            r = linprog(np.zeros(p['E'].shape[1]), A_eq=p['E'] / p['row_scale'][:, None], b_eq=b / p['row_scale'],
                        A_ub=p['U'] / p['ineq_scale'][:, None], b_ub=p['v'] / p['ineq_scale'],
                        bounds=[(None if np.isneginf(lo) else float(lo), None if np.isposinf(hi) else float(hi))
                                for lo, hi in p['bounds']], method='highs',
                        options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
            calls['linprog'] += 1
            known = np.zeros(p['E'].shape[1])
            known[m] = delta
            known_res = mx(p['E'] @ known - b)
            known_fric = float(np.max(p['U'] @ known - p['v'], initial=0.0))
            ent = dict(pose=pi, label=nat['label'], muscle_variable=m, delta=delta,
                       solver_status=int(r.status),
                       solver_residual=None if not r.success else max(mx(p['E'] @ r.x - b), 0.0),
                       solver_friction=None if not r.success else float(np.max(p['U'] @ r.x - p['v'], initial=0.0)),
                       analytic_witness_residual=known_res, analytic_witness_friction=known_fric,
                       analytic_witness_bounds_ok=bool(delta <= p['bounds'][m, 1] + 1e-12),
                       passes=bool(r.success and max(mx(p['E'] @ r.x - b), 0.0) <= GATE_EQ
                                   and float(np.max(p['U'] @ r.x - p['v'], initial=0.0)) <= GATE_FRIC
                                   and known_res <= GATE_EQ and known_fric <= GATE_FRIC))
            cases.append(ent)
            print(json.dumps(ent), flush=True)
    passed = all(c['passes'] for c in cases)
    dump('SOLVER_SANITY_CONTROL.json', dict(
        passed=bool(passed), cases=cases, calls=calls, declared_post_hoc=True,
        scope='Non-degeneracy control only. It shows the shared E/U/bounds plumbing can return status 0 on a trivially feasible right-hand side; it is not a body change, not a witness for any real pose, and it does not alter any sweep verdict.',
        default_case_remains=dict(pose=1, sealed_status=2,
                                  note='the real (gravity plus internal passive load) right-hand side stays infeasible in every scanned case')))
    assert passed, 'control failed: the formulation could never report feasibility'
    print(json.dumps(dict(stage='control', passed=passed, lp=calls['linprog']), ensure_ascii=False))


# ---------------------------------------------------------------- report
def report():
    res = js(O / 'RESULT.json')
    rb = js(O / 'REBUILD_CHECK.json')
    fk = js(O / 'FARKAS_INVARIANCE.json')
    du = js(O / 'DUAL_COMPONENTS.json')
    p2 = js(O / 'PHASE2_SKIPPED.json')
    ve = js(O / 'INDEPENDENT_ARRAY_VERIFICATION.json')
    ctl = js(O / 'SOLVER_SANITY_CONTROL.json')
    L = []
    A = L.append
    A('# S2/S4 身体参数灵敏度：三个已封姿态上的有界倍率量化（手动研究实施，非成品）')
    A('')
    A('包：`%s`' % O)
    A('工具：`%s`' % (R / 'tools/quantify_body_parameter_sensitivity.py'))
    A('起点 HEAD `80215d6`。父独立复核包：`%s`；父无界锥互补探针：`%s`。' % (
        D / 's2-body-parameter-parent-review/20261102T190000Z',
        D / 's2-body-parameter-reachability/20261102T190000Z'))
    A('')
    A('本轮只回答一个问题：**把身体参数（肌肉力幅/Fmax、activation 上限、体重经接触摩擦倍率）当作可调对象时，'
      '“略微”调节能否让这三个已保存姿态出现**静态**承重（静态平衡）见证。**')
    A('不修改默认模型/XML/肌肉/真实网络/旧包/tracker；不启用 STDP；不把参数扫描当神经自主成功；'
      '也不把任何扫描结果当作给身体的合法改动。')
    A('')
    A('## 0. 调用与去重（据实）')
    A('')
    c = res['calls']
    A('| 项 | 次数 |')
    A('|---|---|')
    for k in ['model_compile', 'forward', 'contact_jacobian', 'muscleGain', 'muscleBias', 'body_step',
              'neural_step', 'network', 'training', 'plasticity', 'linprog']:
        A('| %s | %d |' % (k, c[k]))
    A('| 事后非退化对照 LP（单列，不入预登记预算） | %d |' % ctl['calls']['linprog'])
    A('')
    A('- 执行源诚信：预登记后本工具新增了事后对照阶段并加固了 verifier/report/finalize；'
      '`SOURCE_STAGE_LEDGER.json` 用 AST 逐函数比对证明**科学关键函数（sealed_pose/native_columns/build_case/solvable_rows/gate_from_x 等）与预登记时刻完全相同**，'
      '变更仅限 `control/verify/report/finalize` 且已声明，没有为了结果去改矩阵重建或求解设定。')
    A('')
    A('去重：原 18 个 feasibility LP（3pose×3arm×2mode）与 3 个 Farkas LP **没有重解**；'
      '本轮 all-1 默认点直接复用已封 solver 报告（`cached=True`，不重解），只解预登记的缩放值 48 个。'
      '本轮全部阶段神经/网络/训练/可塑性/身体积分 = 0，模型 compile/forward = 0。')
    A('')
    A('## 1. 重建公式与核对（0 模型调用）')
    A('')
    A('- `A = moment.T*gain[None,:]`（=已封 A90）；`A_bias = moment.T@bias`；`base0 = qfrc_passive − qfrc_bias`。')
    A('- 默认：`E0=[A@P, contact, internal]`，`b0 = −base0 − A@(FLOOR*1)`，`a = FLOOR + P x`，`x∈[0,u−FLOOR]`，`FLOOR=1e-4`。')
    A('- **s（力幅/Fmax 倍率）**：`A_s = moment.T*(s*gain) = s*A`，`base_s = (qfrc_passive−qfrc_bias) + s*(moment.T@bias)`，'
      '故肌肉块 `s*A@P`、`b_s = −base_s − s*A@(FLOOR*1)`。**s 同时缩放主动列与 bias（act=0 时肌肉被动/弹性力）项，即整个执行器力输出，'
      '不是只放大主动列**；`qfrc_passive`/`qfrc_bias`（重力+关节被动）与接触/限位列不缩放。')
    A('- **u（activation 上限倍率）**：只把 bounds 改成 `[0,u−FLOOR]`，A/b/bias 全不动 —— 与 s 不可混成一个旋钮。')
    A('- **k（接触摩擦系数倍率）**：outerbox 用封存 box 形式 `[sx,sy,−k·mu]`（`|Fx|≤k·mu·Fz`、`|Fy|≤k·mu·Fz`）；'
      'native 金字塔 ray 行按 `T_i + k·Npt` 重建（`Npt = mu·jac·n` 由封存 dense `efc_J` 恢复；k=1 时与封存行相差 ≤2 ulp）。')
    A('- 非足/限位内部列不参与任何缩放，仅记录（pose1 有 1 条 `joint_RMCoxa_roll` 下限行，`internal_bounds=[0,inf]`）。')
    A('')
    A('| pose | A vs A90 | base | outerbox E | outerbox b | outerbox U | native E | native b | native 列重建(k=1) |')
    A('|---|---|---|---|---|---|---|---|---|')
    for e in rb['poses']:
        c2 = e['checks']
        A('| %d %s | %g | %g | %g | %g | %g | %g | %g | %g |' % (
            e['pose'], e['label'], c2['A_vs_sealed_A90'], c2['base_vs_sealed'], c2['outerbox_E'],
            c2['outerbox_b'], c2['outerbox_U'], c2['native_E'], c2['native_b'], c2['native_column_rebuild_k1']))
    A('')
    A('全部差值在 machine epsilon 量级（outerbox E/b/U/bounds 与 pose0 完全为 0；native 接触列经 `T+k·Npt` 恒等式重建，k=1 时 2.2e-16）。'
      '`P` 为 0/1 权限矩阵（每行和为 1），故 `a=FLOOR+x` 的上限恰为 `u`。')
    A('')
    A('## 2. 每姿态每 knob 的最小可行倍率（核心表）')
    A('')
    A('可行 = HiGHS status0 且 raw 残差 ≤1e-8、摩擦 ≤1e-8、bounds ≤1e-10、activation∈[FLOOR,u]。'
      '最小倍率 = **预登记网格中**最小的可行值（无二分、无细化、无多 knob 组合）。')
    A('')
    A('| pose | knob | 预登记网格 | 可行值 | 最小可行倍率 | 所需相对增量 | 判定 | native 复验 |')
    A('|---|---|---|---|---|---|---|---|')
    for pp in res['poses']:
        for knob in ['s', 'u', 'k']:
            e = pp['knobs'][knob]
            nr = e['native_recheck']
            nrt = ('未做（%s）' % nr['reason']) if not nr['done'] else ('feasible=%s' % nr['feasible'])
            A('| %d %s | %s | %s | %s | %s | %s | %s | %s |' % (
                pp['pose'], pp['label'], knob, e['grid'], e['feasible_values'] or '无可行值',
                ('%g×' % e['min_multiplier']) if e['min_multiplier'] is not None else '不可达',
                ('+%g×' % e['required_relative_increase']) if e['required_relative_increase'] is not None else '—',
                e['verdict'], nrt))
    A('')
    A('**三个姿态、三个 knob 全部 48 个缩放 case 都是 status2 不可行**，因此每个 pose×knob 都标记 '
      '`NOT_REACHABLE_WITHIN_SCANNED_RANGE`，扫描上界分别是 s=10×、u=3×、k=10×。没有任何可行倍率存在，'
      '也就没有 “所需相对增量” 可给（表中写 —），native 复验因无可行倍率而按预登记不做。')
    A('')
    A('### 2.1 逐值求解明细（outerbox，全部实解）')
    A('')
    A('| pose | knob 与取值 | status | feasible | 备注 |')
    A('|---|---|---|---|---|')
    for r in res['rows']:
        if not r['solved']:
            A('| %d | 默认 all-1（%s） | %d | false | 复用已封 solver 报告，未重解 |' % (r['pose'], r['mode'], r['status']))
            continue
        A('| %d | %s=%g | %d | %s | %s |' % (r['pose'], r['knob'], r['value'], r['status'], r['feasible'],
                                            r['message'].replace('|', '/')))
    A('')
    A('## 3. 结论（措辞按父独立复核的合并结论，限界明确）')
    A('')
    A('> **在这三个已保存姿态上，“只调幅度”无效，问题在方向/几何。**')
    A('')
    A('- 本包网格：s 到 10×、u 到 3×、k 到 10× 全部不可行（48/48）。')
    A('- 父无界锥探针（`s2-body-parameter-reachability`）：力幅放大到 1e4×、取消 activation 上限、取消接触力上限（只留非负+摩擦锥）仍不可行；'
      '同代码路径的可行对照（零载荷、沿单条真实肌肉力矩列）status0，说明不是代码假象。')
    A('- 父独立复核（`%s`）独立重建并自行重解 s=10/u=3/k=10 共 12 LP，结论一致，Phase2 SKIP 正确。' %
      (D / 's2-body-parameter-parent-review/20261102T190000Z'))
    A('')
    A('**“略微调节不足”**：按预登记判据，最小可行倍率 >2× 或不可达即判“略微调节不足”。本轮更强——'
      '不仅 ≤2× 不行，连 10×/3×/10× 都不行，且父的 1e4× 与无界锥也不行。因此：'
      '**在“略微调节”的语义下（≤1.25×~2×）不足；即使放宽到 10× 甚至无上界，单靠幅度仍不足。**')
    A('')
    A('**此结论严格限于**：这 3 个已保存姿态（0s / 1.3s / slow 1.25s）、它们各自的实际足地接触集合、'
      'independent90 的肌肉边界、本条件静态模型（`qvel=0`、`qacc=0` 假设、接触反力独立于 soft solver/compliance/complementarity）。'
      '**不证明动态行走不可能**（动态步态不要求每一瞬间静态平衡），**不覆盖其它姿态/接触/肌肉边界**，'
      '**不是解剖标定**，**不授权改 Fmax/activation 上限/摩擦/体重**。')
    A('')
    A('## 4. dual 证书上界与本次扫描的一致性')
    A('')
    A('| pose | upper activation dual L1 | upper activation dual max | 非零 | 本轮 s 扫描 | 本轮 u 扫描 |')
    A('|---|---|---|---|---|---|')
    for d, pp in zip(du['poses'], res['poses']):
        A('| %d %s | %.6g | %.6g | %s | %s | %s |' % (
            d['pose'], d['label'], d['upper_activation_dual_L1'], d['upper_activation_dual_max'],
            d['upper_bound_dual_nonzero'], pp['knobs']['s']['verdict'], pp['knobs']['u']['verdict']))
    A('')
    A('旧 Farkas 分离证书中 activation **上界**分量非 0（L1 0.4004/0.5357/1.0266，最大单项 0.116/0.189/0.145），'
      '说明那次数值分离**用到了幅度/边界一侧的信息，不只是方向**（若幅度完全无关，上界 duel 不会被激活）。'
      '它与本轮扫描**一致地**指向：**不是靠小幅（甚至大幅）调幅就能翻成可行**——这与父的 1e4×/无界锥探针同向。'
      '但 dual 非唯一、Farkas 只覆盖 all-1 的 outerbox 问题，且“上界 dual 非零”本身并不能推出“幅度完全无关”，'
      '所以**不下“幅度是唯一原因/方向已足够”这类唯因结论**；能确定的只是：在这 3 个姿态上，幅度类改动在本轮与父的探针里都没有产生见证。')
    A('')
    A('### 4.1 旧证书不变性（本轮独立重算）')
    A('')
    A('| pose | stationarity | margin | z_min | E/b/H/h 逐项 |')
    A('|---|---|---|---|---|')
    for e in fk['poses']:
        A('| %d | %g | %g | %g | E%g / b%g / H%g / h%g |' % (
            e['pose'], e['stationarity'], e['margin'], e['dual_min'],
            e['E_exact'], e['b_exact'], e['H_exact'], e['h_exact']))
    A('')
    A('## 5. Phase2（附着位点灵敏度）')
    A('')
    A('**明确 %s**：预登记触发条件（Phase1 任一姿态出现 ≤1.25× 最小可行倍率）**未满足**，因此按规则跳过，**skipped 不算通过**。' %
      ('已跳过' if not p2['executed'] else '已执行'))
    A('')
    A('理由：%s。' % p2['reason'])
    A('')
    A('各姿态最小可行倍率（触发判据输入）：`%s`。' % json.dumps(p2['per_pose_min'], ensure_ascii=False))
    A('')
    A('这**不是**“位点/几何调节无用”的证据（父探针与本包只否定幅度类改动）；本轮也没有做任何'
      '“把等效倍率降到 1 所需 δ”的线性外推——外推本身也不构成可行证明。要回答“方向/几何类改动是否有效”，'
      '需另起预登记（新 XML 研究 clone、位点 δ 网格、接触布局/内部约束方向），不得用本轮结果替代。')
    A('')
    A('## 6. 非退化对照（事后追加，明确标注）')
    A('')
    A('`SOLVER_SANITY_CONTROL.json`：用**同一套** E/U/bounds 与同一个 solver 设置，把 RHS 换成 `delta*E[:,m]`'
      '（`delta` 落在 activation box 内，`x=delta*e_m` 是精确可行点）。结果 status0，残差 %s、%s，'
      '解析见证残差 0、摩擦 0 → **“全不可行”不是公式/代码假象**。此对照是**预登记预算之外的事后追加**，'
      '只作非退化检查，不改变任何扫描判定，也不是任何真实姿态的见证。' % (
          '%.3g' % ctl['cases'][0]['solver_residual'], '%.3g' % ctl['cases'][1]['solver_residual']))
    A('')
    A('## 7. 独立纯数组核验')
    A('')
    A('`INDEPENDENT_ARRAY_VERIFICATION.json`：%d 组检查，`passed=%s`，0 solve / 0 模型 / 0 身体 / 0 神经。' % (
        len(ve['checks']), ve['passed']))
    for kk in ve['checks']:
        A('- `%s`' % kk)
    A('')
    A('覆盖：R1 已封默认矩阵的标量重建；R2 缩放后 `E(s)/b(s)` 逐元素公式（最大相对差 %s）；R3 `U(k)` 宽度与摩擦方向；'
      'R4 各 knob 未触碰块的恒等性 + knob 非空操作（s 改 E、k 改 U、u 只改 bounds）；R5 每个 case 的原始条件与最小倍率选择规则（含默认点不重解）；'
      'R5b 选择规则/判定/相对增量逐 pose 复核；R6 旧 Farkas 证书不变；R7 Phase2 算术（本轮 skipped）；R8 调用记账。' % (
          '%.3g' % max(e['muscle_block_rel'] for e in ve['checks']['R2_scaled_formula'])))
    A('')
    A('## 8. 与“学习拟合数据中的标准姿态”的分工')
    A('')
    A('本轮**不涉及**姿态/参考学习。父的 `s1-pose-data-resolution` 已明确：现有数据只有关键点/派生角，没有肌肉级实测姿态；'
      '父的 `s2-body-parameter-reachability` 专管“幅度上界无效”的锥证据。本包只提供 3pose × 各 knob 的网格最小倍率表。'
      '“方向/几何类”（姿态、附着点、接触布局、内部约束）要真正回答，需要新预登记的新一轮，不能由本轮推断。')
    A('')
    A('## 9. 失败源')
    A('')
    A('- 第 1 次 `prepare`：本包自己的断言在 `bounds` 含 ±inf 时做了 `inf−inf` 比较产生 NaN 而失败；'
      '发生在任何科学数组/求解/模型加载之前（0 LP、0 compile、0 forward），已改为符号感知比较 `bounds_diff` 后重跑。')
    A('- 开发期（未进入本包记录的科学阶段）：`verify` 初版标量重建把 `A` 写成对 u 的二次求和、`outer_fresh` 未把 `u` 传进 bounds、'
      'R4 误把 k 的 U 当作应与默认相同；均在 verify 自己的断言处暴露并修正，**没有改动任何科学数组或求解结果**，'
      '也没有为通过核验而放宽任何科学门（只把“同一式子的不同求和顺序”放到 ≤2 ulp）。')
    A('- 无 solver 失败、无超时、无后台 job、无子代理委派。')
    A('')
    A('## 10. 清理')
    A('')
    cl = js(O / 'CLEANUP_LEDGER.json') if (O / 'CLEANUP_LEDGER.json').exists() else None
    A('%s' % (('实际删除 %d 个文件（见 `CLEANUP_LEDGER.json`）。' % cl['actual_deleted_count']) if cl
              else '报告生成阶段未删除任何文件。'))
    A('已封只读来源包（s2-mapped-support-capacity / s2-native-support-constraints / s2-reference-muscle-operating-range）一个字节未写，'
      '封存后 `SOURCE_HASH_RECHECK.json` 复核。')
    A('')
    A('## 11. 重要程度')
    A('')
    A('| 文件 | 重要程度 | 用途 |')
    A('|---|---|---|')
    for pth, imp, use in [
            ('RESULT.json', '核心必需', '全部 48+6 个 case 的求解状态与最小倍率判定'),
            ('LP_SWEEP.npz', '核心必需', '每个 case 的 E/b/U/v/bounds/row_scale 数组'),
            ('REBUILD_CHECK.json', '核心必需', '默认点与已封矩阵的逐项重建核对'),
            ('FARKAS_INVARIANCE.json', '核心必需', '旧证书不变性（非新证书）'),
            ('DUAL_COMPONENTS.json', '核心必需', 'dual 分量 + 肌肉候选（Phase2 判据输入）'),
            ('PHASE2_SKIPPED.json', '核心必需', 'Phase2 明确 skipped 与理由'),
            ('SOLVER_SANITY_CONTROL.json', '核心必需', '事后非退化对照'),
            ('INDEPENDENT_ARRAY_VERIFICATION.json', '核心必需', '独立纯数组核验'),
            ('PARENT_REVIEW.json', '核心必需', '父独立复核（父写入）'),
            ('PREREGISTRATION.json', '核心必需', '跑前预登记（网格/门/禁止项）'),
            ('EXECUTION_SOURCE.py', '核心必需', '执行源冻结副本'),
            ('SOURCE_HASHES_BEFORE.json', '重要辅助', '来源哈希（预登记时刻）'),
            ('SOURCE_STAGE_LEDGER.json', '重要辅助', '执行源跨阶段变更台账（科学函数 AST 逐项相同）'),
            ('SOURCE_HASH_RECHECK.json', '重要辅助', '封存后来源未变复核'),
            ('CALLS.json', '重要辅助', '阶段调用账'),
            ('SOLVER_CALL_GATE.json', '重要辅助', 'LP 预算门'),
            ('RESOURCE_ACCOUNTING.json', '重要辅助', '资源/调用/时间账'),
            ('JOB_LEDGER.json', '重要辅助', 'job 清单（全前台，无后台）'),
            ('FAILURE_SOURCES.json', '重要辅助', '失败源清单'),
            ('CLEANUP_LEDGER.json', '重要辅助', '清理台账'),
            ('FILE_INVENTORY.json', '重要辅助', '文件清单与重要程度'),
            ('MANIFEST_SHA256.json', '重要辅助', '封存清单'),
            ('SEAL_STATE.json', '重要辅助', '封存状态'),
            ('SEAL_ONCE.py', '重要辅助', '唯一封存脚本（封存后不可再跑）'),
            ('REPORT.zh-CN.md', '重要辅助', '本报告')]:
        A('| `%s` | %s | %s |' % (pth, imp, use))
    A('')
    A('## 12. 单位与不可宣称项')
    A('')
    A('- 所有数值都是 **model native units**；绝对力值与生物 SI **NOT_ESTABLISHED**。')
    A('- 参数改变只是**工程标定假设**，不是测得解剖，不获得使用许可，也不写入任何默认/旧模型。')
    A('- status2 只是 solver 报告不是证书；除 all-1 outerbox 的旧 Farkas 数值分离外，本轮没有新证书。')
    A('- 本轮 0 神经/网络/训练/可塑性；自主支撑与行走目标仍未达。')
    (O / 'REPORT.zh-CN.md').write_text('\n'.join(L) + '\n')
    dump('CLEANUP_LEDGER.json', dict(
        deleted=[], actual_deleted_count=0,
        reason='No temporary or superseded file qualified for deletion in this round; the sealed source packages are read-only and untouched.',
        retained=['all package files are referenced by the report/manifest',
                  'sealed source packages untouched']))
    print(json.dumps(dict(stage='report', passed=True), ensure_ascii=False))


def finalize():
    """Accounting + review-ready hashes. Sealing is a separate single step (SEAL_ONCE.py)."""
    tool = R / 'tools/quantify_body_parameter_sensitivity.py'
    res = js(O / 'RESULT.json')
    ctl = js(O / 'SOLVER_SANITY_CONTROL.json')
    calls = dict(res['calls'])
    calls['linprog_control_post_hoc'] = int(ctl['calls']['linprog'])
    calls['linprog_total'] = int(res['calls']['linprog']) + int(ctl['calls']['linprog'])
    dump('RESOURCE_ACCOUNTING.json', dict(
        calls=calls,
        budget=dict(feasibility_lp=48, native_recheck_lp_max=9, native_recheck_used=0,
                    control_lp_post_hoc=int(ctl['calls']['linprog']), model_compile=0, forward=0),
        reuse='Sealed default LP reports (6 cases) and the 3 Farkas certificates were reused, never re-solved; the 18 sealed feasibility LPs were not repeated.',
        runtime='All stages are foreground, single process, sub-second to a few seconds; no background job and no sub-agent delegation.',
        notes='Sensitivity multipliers are engineering hypotheses on a conditional static model, not measured anatomy. Native units only; biological SI NOT_ESTABLISHED.'))
    dump('JOB_LEDGER.json', dict(
        jobs=[], background_jobs=0, subagent_delegation=0, foreground_stages=[
            'prepare', 'sweep', 'control', 'phase2', 'verify', 'report', 'finalize', 'SEAL_ONCE'],
        note='Every stage ran in the foreground of one shell; no background job id was created and no child agent was used, so there is nothing to collect.'))
    dump('FAILURE_SOURCES.json', dict(failures=[
        dict(stage='prepare', attempt=1, before_science=True,
             cause='own assertion compared bound tables containing +-inf, so inf-inf produced NaN and the check failed',
             calls=dict(model_compile=0, forward=0, linprog=0),
             fix='sign-aware bounds comparator bounds_diff; package directory recreated fresh, no science array or solver result was touched'),
        dict(stage='verify', attempt='development', before_science=False,
             cause='draft verification code had three own bugs (A written as a double sum over u, u not passed into the rebuilt bounds, k-rows wrongly expected equal to defaults)',
             calls=dict(model_compile=0, forward=0, linprog=0),
             fix='fixed under the verifier\'s own assertions; no scientific gate was relaxed except the mathematically identical re-summation tolerance (2 ulp)'),
        dict(stage='sweep/control/phase2', attempt=1, before_science=False, cause='none; no solver failure, no timeout',
             calls=dict(linprog=50), fix='n/a')],
        unrelated_notes='Pre-existing untracked tools in the repo belong to sibling rounds (parent cross-check/reachability); this round neither edited nor deleted them.'))
    hashes = js(O / 'SOURCE_HASHES_BEFORE.json')
    # The tool itself was extended after pre-registration (post-hoc control, verifier hardening,
    # report/finalize). Science-critical definitions must stay AST-identical; anything else is
    # declared explicitly instead of pretending the whole file hash is unchanged.
    import ast
    SCIENCE = ['ar', 'js', 'sha', 'dump', 'mx', 'bounds_diff', 'sealed_pose', 'native_columns',
               'build_case', 'case_name', 'solvable_rows', 'gate_from_x']
    ALLOWED_EXTRA_CHANGES = {'control', 'verify', 'report', 'finalize'}

    def funcs(path):
        tree = ast.parse(Path(path).read_text())
        return {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest()
                for n in tree.body if isinstance(n, ast.FunctionDef)}

    frozen, live = funcs(O / 'EXECUTION_SOURCE.py'), funcs(tool)
    changed = sorted([k for k in frozen if k in live and frozen[k] != live[k]])
    added = sorted(set(live) - set(frozen))
    removed = sorted(set(frozen) - set(live))
    sci_changed = [k for k in changed + added + removed if k in SCIENCE]
    undeclared = [k for k in changed + added + removed if k not in SCIENCE and k not in ALLOWED_EXTRA_CHANGES]
    dump('SOURCE_STAGE_LEDGER.json', dict(
        frozen=str(O / 'EXECUTION_SOURCE.py'), frozen_sha256=sha(O / 'EXECUTION_SOURCE.py'),
        live=str(tool), live_sha256=sha(tool),
        prepare_time_sha256=hashes[str(tool)],
        science_functions=SCIENCE, changed=changed, added=added, removed=removed,
        science_effects=sci_changed, undeclared_changes=undeclared,
        conclusion='Every science-critical definition is AST-identical to the pre-registered source; only the '
                   'separately declared stages changed after pre-registration.',
        note='The tool file hash therefore differs from SOURCE_HASHES_BEFORE on purpose; this ledger is the honest record.'))
    assert not sci_changed, sci_changed
    assert not undeclared, undeclared
    drift = {pth: (h, sha(Path(pth))) for pth, h in hashes.items()
             if pth != str(tool) and Path(pth).exists() and sha(Path(pth)) != h}
    fixed = {pth: h for pth, h in hashes.items() if not Path(pth).exists() and pth != str(tool)}
    dump('SOURCE_HASH_RECHECK.json', dict(unchanged=not drift, drift=drift, missing=fixed,
                                          execution_source=dict(
                                              changed_after_preregistration=True,
                                              prepare_time_sha256=hashes[str(tool)],
                                              live_sha256=sha(tool),
                                              science_functions_ast_identical=True,
                                              see='SOURCE_STAGE_LEDGER.json'),
                                          note='Sealed input packages and the frozen XML must be byte-identical after '
                                               'this round; the execution source is accounted for stage by stage.'))
    assert not drift, drift
    snap = {str(pth): dict(sha256=sha(pth), bytes=pth.stat().st_size) for pth in sorted(O.iterdir())
            if pth.is_file() and pth.name not in ('MANIFEST_SHA256.json', 'REVIEW_READY_HASHES.json')}
    (O / 'REVIEW_READY_HASHES.json').write_text(json.dumps(snap, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(stage='finalize', review_ready=len(snap), not_sealed=True,
                          linprog_total=calls['linprog_total']), ensure_ascii=False))


CMDS = {'prepare': prepare, 'sweep': sweep, 'control': control, 'phase2': phase2,
        'verify': verify, 'report': report, 'finalize': finalize}
if __name__ == '__main__':
    CMDS[sys.argv[1]]()
