"""S177/P1-S2 six-row root wrench Gauss-Newton solve (engine run; no mj_step, no neural).

Root rows of the 48-row static balance have zero activation authority (S175), so with the
leg joint angles held at the S154 release values the root wrench balance is 6 equations in
6 unknowns (x, y, z, roll, pitch, yaw). This runner solves exactly that system and reports
the 42-row joint subsystem separately; the two are never merged into one score.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil, traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton, Budget, RefusedJacobian

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S173 = BASE/'s173-accounted-near-initial-search/20261024T235544196037334Z/RUN'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP = 600
RESERVED = 20
LEGS = ('lf','lm','lh','rf','rm','rh')
STATE = ('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied')
OPTIONAL = ('mocap_pos','mocap_quat','eq_active','plugin_state','userdata')
DEPTH_CAP = 0.03442910871489635

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sanitize(v):
    if isinstance(v, dict): return {str(k): sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [sanitize(x) for x in v]
    if isinstance(v, np.generic): v = v.item()
    if isinstance(v, float) and not np.isfinite(v): return None
    if isinstance(v, (int, float, bool, str)) or v is None: return v
    return str(v)
def dump(p, v): p.write_text(json.dumps(sanitize(v), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
def extract(p, names):
    t = ast.parse(p.read_text())
    return '\n'.join(ast.unparse(next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == x)) for x in names)+'\n'

def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def quat_from_xyz(rx, ry, rz):
    cx, sx = np.cos(rx/2), np.sin(rx/2); cy, sy = np.cos(ry/2), np.sin(ry/2); cz, sz = np.cos(rz/2), np.sin(rz/2)
    return np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])

def main(out):
    assert out.is_absolute() and not out.exists(), 'fresh timestamp only'
    out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    ledger = dict(model_load=0, gate_forward=0, confirmation_forward=0)
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)),
             (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
             (ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))]
    files = [Path(__file__).resolve(), ROOT/'root_newton_core.py', ROOT/'test_root_newton_core.py', XML,
             S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz', S154/'init80ms/MODEL_PROVENANCE.json',
             S154/'init80ms/NATIVE_MODEL_ARRAYS.npz', S154/'init80ms/COM_CONTACT_METADATA.json',
             S165/'STATIC_AFFINE_000.npz', S165/'ZERO_V_000.npz', S166/'MAPPING.npz', S173/'EVAL_001.npz']+[p for p, _ in specs]
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files})
    dump(out/'INPUT_HASHES.json', hashes)
    manifest = {}
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
        manifest[p] = dict(sha256=h, copy=str(t))
    dump(out/'SOURCE_FREEZE_MANIFEST.json', manifest)
    text = '\n'.join(extract(p, n) for p, n in specs)
    (out/'SOURCE_FREEZE'/'INERT_FUNCTIONS.py').write_text(text)
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile(text, str(out/'SOURCE_FREEZE'/'INERT_FUNCTIONS.py'), 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T = load(S166/'MAPPING.npz')['T']; missing = np.flatnonzero(np.load(S166/'MAPPING.npz')['actuator_pool'] < 0)
    assert T.shape == (90, 66) and np.count_nonzero(T) == 74 and len(missing) == 16
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]; nodes = list(tree.getroot().find('actuator'))
    assert joints[0].get('type') == 'free' and len(joints) == 43 and len(q0) == 49
    q0z = float(q0[2]); q0xy = q0[:2].copy(); quat0 = q0[3:7].copy()
    ANG = 0.35; XYPOS = 0.30
    lower = np.array([q0xy[0]-XYPOS, q0xy[1]-XYPOS, q0z*(1-.15), -ANG, -ANG, -ANG])
    upper = np.array([q0xy[0]+XYPOS, q0xy[1]+XYPOS, q0z*(1+.15), ANG, ANG, ANG])
    p0 = np.array([q0xy[0], q0xy[1], q0z, 0., 0., 0.])
    assert np.all(lower <= p0) and np.all(p0 <= upper)
    dump(out/'PREREG.json', dict(
        task='S177/P1-S2 six-row root wrench Gauss-Newton solve, joint subsystem reported separately',
        fixed='all 42 leg joints at S154 release values; real 66-pool mapping with 16 missing slots zero; native soft contact f0 recomputed per pose',
        variables='root x,y,z and roll,pitch,yaw deltas applied in world frame; 6 unknowns, 6 equations',
        frozen_bounds=dict(x=[float(lower[0]), float(upper[0])], y=[float(lower[1]), float(upper[1])],
                           z=[float(lower[2]), float(upper[2])], roll_pitch_yaw_rad=[-ANG, ANG],
                           note='owner authorised this window; frozen before load, never widened after results'),
        residual='residual6 = -(base + J.T f0)[0:6]; root rows carry zero activation authority (S175)',
        solver='Gauss-Newton with 6-column forward-difference Jacobian, lstsq step, bounded line search halving up to 5 times, monotone accept on |residual6|inf only',
        tolerance=1e-8, budget=dict(search_forward_cap=CAP, reserved_for_confirmation=RESERVED, max_iterations=60, fd_step=1e-4),
        geometry_gates=dict(foot_depth_cap=DEPTH_CAP, nonfoot_cap=1e-12, min_bearing_legs=3, friction_margin_min=-1e-10),
        joint_subsystem='interval reachability plus one small LP over real 66 pools with x in [0,1]; reported separately, never merged with residual6',
        accounting='forwards, cache hits, jacobians, FD probes, line-search trials, contact-set changes and model loads are counted separately',
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    import mujoco as mj
    assert mj.__version__ == '3.9.0'
    for name in ('control', 'act_dyn', 'act_gain', 'act_bias', 'passive', 'sensor'):
        g = getattr(mj, 'get_mjcb_'+name, None); assert g is None or g() is None
    ledger['model_load'] += 1; m = mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    floors = set(meta['floor_geom_ids']); geomleg = {int(k): LEGS.index(v) for k, v in meta['geom_map'].items()}
    def forward(q, act90):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0; d.act[:] = act90; d.ctrl[:] = act90; d.qacc_warmstart[:] = 0
        init = {n: np.asarray(getattr(d, n)).copy() for n in (*STATE, *OPTIONAL)}
        mj.mj_forward(m, d)
        assert all(np.array_equal(getattr(d, n), init[n]) for n in (*STATE, *OPTIONAL))
        c = ns['constraint_record'](mj, m, d, 0); c.update(ns['moment_record'](m, d))
        for n in ('qacc', 'act_dot', 'actuator_force', 'actuator_length', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias',
                  'efc_D', 'efc_aref', 'efc_type', 'efc_frictionloss'):
            c[n] = np.asarray(getattr(d, n)).copy()
        c['contact_dist'] = np.array([v.dist for v in d.contact]); c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
        return c
    def affine(c):
        types = c['efc_type']; assert np.all(np.isin(types, [0, 3, 4, 5, 6])), 'unsupported native row type'
        f0 = c['efc_D']*c['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        gain = np.empty(90); bias = np.empty(90)
        for j, node in enumerate(nodes): _, gain[j], bias[j] = ns['evaluate'](c['actuator_length'][j], 0., 0., node)
        moment = c['actuator_moment']; B = moment.T*gain[None, :]; base = moment.T@bias+c['qfrc_passive']-c['qfrc_bias']
        return B, base, f0
    def geometry(c, f0):
        depth = 0.; nonfoot = 0.; bearing = set(); margins = []
        for ci, (g1, g2) in enumerate(c['contact_geom']):
            g1 = int(g1); g2 = int(g2); leg = geomleg.get(g2 if g1 in floors else g1) if ((g1 in floors) != (g2 in floors)) else None
            pen = max(0., -float(c['contact_dist'][ci])); addr = int(c['contact_efc_address'][ci]); dim = int(c['contact_dim'][ci])
            if leg is None: nonfoot = max(nonfoot, pen)
            else: depth = max(depth, pen)
            if addr < 0: continue
            count = int(c['contact_row_count'][ci])
            if dim == 1: local = np.r_[f0[addr], 0., 0.]
            else:
                assert dim == 3 and count == 4
                e = f0[addr:addr+4]; mu = c['contact_friction'][ci, :2]
                local = np.r_[e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])]
            margins.append(float(local[0]-sum(abs(local[1:]/mu))))
            if leg is not None and local[0] > 0 and c['contact_dist'][ci] <= 0: bearing.add(leg)
        sig = '|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in c['contact_geom']))+f'#nefc{len(c["efc_type"])}'
        return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot), bearing_legs=int(len(bearing)),
                    friction_margin=float(min(margins, default=0.)), contact_signature=sig, contacts=int(len(c['contact_geom'])))
    def p_to_qpos(p):
        q = q0.copy(); q[0:2] = p[0:2]; q[2] = p[2]
        dq = quat_from_xyz(p[3], p[4], p[5]); qw = quat_mul(dq, quat0); q[3:7] = qw/np.linalg.norm(qw)
        return q
    def callback(p):
        c = forward(p_to_qpos(p), np.zeros(90))
        B, base, f0 = affine(c); bb = -(base+c['efc_J_dense'].T@f0)
        g = geometry(c, f0)
        return dict(residual6=bb[:6].copy(), B=B, base=base, f0=f0, J=c['efc_J_dense'], qpos=p_to_qpos(p), **g)
    solver = RootNewton(callback, lower, upper, cap=CAP, output=out/'NEWTON_LEDGER.json', name='root6')
    result = None
    try:
        gate = solver.evaluate(p0, 'equivalence_gate', 0)
        ledger['gate_forward'] += 1
        ref = load(S173/'EVAL_001.npz')
        gate_err = dict(qpos=float(np.max(abs(gate['qpos']-ref['qpos']))), B=float(np.max(abs(gate['B']-ref['B']))),
                        base=float(np.max(abs(gate['base']-ref['base']))), f0=float(np.max(abs(gate['f0']-ref['f0']))))
        dump(out/'EQUIVALENCE_GATE.json', dict(reference='S173 RUN/EVAL_001.npz', errors=gate_err, tolerance=1e-10,
                                               passed=bool(max(gate_err.values()) <= 1e-10)))
        assert max(gate_err.values()) <= 1e-10, 'root runner does not reproduce frozen production probe'
        result = solver.solve(p0)
        stop = result['stop']
        dump(out/'NEWTON_SOLVE_RAW.json', dict(stop=stop, norm_inf=result['norm_inf'], residual6=result['residual6'],
             final_decomposition=result.get('final_decomposition'), ledger=result['ledger'], history=result['history']))
    except Budget:
        stop = 'BUDGET_STOP'
    except Exception:
        dump(out/'FAILED.json', dict(traceback=traceback.format_exc(), ledger=solver.ledger, no_retry=True)); raise
    rec = result['record']; p = np.asarray(result['p'])
    B = rec['B']; base = rec['base']; f0 = rec['f0']
    A = (B@T)[6:, :]; tgt = (-(base+rec['J'].T@f0))[6:]
    interval = []
    for d in range(42):
        lo = float(np.minimum(A[d], 0).sum()); hi = float(np.maximum(A[d], 0).sum()); t = float(tgt[d])
        interval.append(dict(joint_row=d, target=t, min_possible=lo, max_possible=hi, reachable=bool(lo-1e-9 <= t <= hi+1e-9),
                             violation=float(max(lo-t, 0., t-hi))))
    n = A.shape[1]
    lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                 b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs',
                 options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    assert lp.success, lp.message
    t_star = float(lp.x[-1]); pools = lp.x[:n]
    unreachable = [r for r in interval if not r['reachable']]
    dump(out/'JOINT_SUBSYSTEM_42.json', dict(note='reported separately from residual6; never merged',
        unreachable_rows=[r['joint_row'] for r in unreachable], worst_single_row_violation=float(max((r['violation'] for r in interval), default=0.)),
        lp_minimax_t=t_star, rows=interval, joint_rows_exactly_balanced=bool(t_star <= 1e-8)))
    with (out/'JOINT_SUBSYSTEM_42_ROWS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(interval[0])); w.writeheader(); w.writerows(interval)
    geom_ok = bool(rec['foot_depth'] <= DEPTH_CAP and rec['nonfoot_depth'] <= 1e-12 and rec['bearing_legs'] >= 3 and rec['friction_margin'] >= -1e-10)
    root_ok = bool(result['norm_inf'] <= 1e-8)
    decomp = result.get('final_decomposition')
    dump(out/'ROOT_RESULT.json', dict(p=p.tolist(), qpos=rec['qpos'].tolist(), residual6=result['residual6'], norm_inf=result['norm_inf'],
        rank_decomposition=decomp,
        root_rows_balanced=root_ok, geometry=dict(foot_depth=rec['foot_depth'], nonfoot_depth=rec['nonfoot_depth'], bearing_legs=rec['bearing_legs'],
        friction_margin=rec['friction_margin'], contact_signature=rec['contact_signature'], geometry_gates_pass=geom_ok),
        contact_set_changes=solver.ledger['contact_set_changes'], stop=stop, history=solver.history))
    np.savez_compressed(out/'ROOT_BALANCE_STATE.npz', p=p, qpos=rec['qpos'], residual6=result['residual6'], B=B, base=base, f0=f0, J=rec['J'],
                        T=T, lp_activation=pools, contact_dist=rec.get('contact_dist', np.zeros(0)))
    unctrl = None if decomp is None else decomp['uncontrollable_inf']
    if root_ok and geom_ok:
        act = T@pools; c = forward(rec['qpos'], act); ledger['confirmation_forward'] += 1
        Bc, basec, f0c = affine(c); act90 = T@pools
        res48 = Bc@act90+basec+c['efc_J_dense'].T@f0c
        g = geometry(c, f0c)
        ok48 = bool(abs(res48).max() <= 1e-8)
        dump(out/'CONFIRMATION.json', dict(max_abs_residual48=float(abs(res48).max()), max_abs_qacc=float(abs(c['qacc']).max()),
            all48_balanced=ok48, act_dot_all_zero=bool(not np.any(c['act_dot'])), geometry=g))
        np.savez_compressed(out/'CONFIRMATION.npz', qpos=rec['qpos'], act90=act90, res48=res48, qacc=c['qacc'], act_dot=c['act_dot'])
        status = 'ROOT_AND_JOINTS_BALANCED_CONFIRMED' if ok48 else 'ROOT_BALANCED_JOINT_NOT_BALANCED_PROGRESS'
    elif root_ok:
        status = 'ROOT_BALANCED_BUT_GEOMETRY_VIOLATION'
    elif unctrl is not None and decomp['controllable_inf'] <= 1e-8 and unctrl > 1e-8:
        status = 'ROOT_CONTROLLABLE_PART_BALANCED_UNCONTROLLABLE_FLOOR'
    else:
        status = 'ROOT_NOT_BALANCED_SCREEN_FAILED'
    dump(out/'SUMMARY.json', dict(status=status, stop=stop, root_norm_inf=result['norm_inf'], root_rows_balanced=root_ok,
        joint_lp_minimax=t_star, joint_rows_balanced=bool(t_star <= 1e-8), unreachable_joint_rows=[r['joint_row'] for r in unreachable],
        root_rank=None if decomp is None else decomp['rank'], root_controllable_inf=None if decomp is None else decomp['controllable_inf'],
        root_uncontrollable_inf=unctrl,
        geometry_gates_pass=geom_ok, newton_ledger=solver.ledger, runner_ledger=ledger, history=solver.history,
        units='native; SI NOT_ESTABLISHED', trajectory=False, integration=False, neural_controller=False, deployment=False))
    after = {k: sha(k) for k in hashes}
    dump(out/'INPUT_HASHES_AFTER.json', after)
    print(json.dumps(dict(status=status, root_norm_inf=result['norm_inf'], joint_lp=t_star, geometry_ok=geom_ok, ledger=solver.ledger)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    main(a.output or BASE/'s177-root-newton-search'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
