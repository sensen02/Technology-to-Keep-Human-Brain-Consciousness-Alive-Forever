"""S180 Stage B: 3-unknown root solve + targeted leg-joint candidates (native limits)."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
from s179_core import build_row_dof_map, classify_root, select_best

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
S180A = BASE/'s180-dof-authority-audit/20261025T096000000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP, RESERVED, LOAD_CAP = 2200, 200, 3
DEPTH_CAP = 0.03442910871489635
STATE = ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied')
OPTIONAL = ('mocap_pos', 'mocap_quat', 'eq_active', 'plugin_state', 'userdata')

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
    assert out.is_absolute() and not out.exists(); out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    files = [Path(__file__).resolve(), ROOT/'s179_core.py', ROOT/'root_newton_core.py', XML, S154/'INPUT_HASHES.json',
             S154/'init80ms/TRAJECTORY.npz', S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/NATIVE_MODEL_ARRAYS.npz',
             S154/'init80ms/COM_CONTACT_METADATA.json', S166/'MAPPING.npz', S177/'ROOT_BALANCE_STATE.npz',
             S180A/'NATIVE_LIMIT_FLIP.csv', S180A/'DEMAND_DECOMPOSITION.csv', S180A/'ROW_DOF_MAP_REUSED.json']
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files}); dump(out/'INPUT_HASHES.json', hashes)
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)), (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',))]
    text = '\n'.join(extract(p, n) for p, n in specs)
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile(text, '<inert>', 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T = load(S166/'MAPPING.npz')['T']; pool = load(S166/'MAPPING.npz')['actuator_pool']
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    jname = [j.get('name') for j in joints[1:]]
    nat = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    qleg0 = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:].copy()
    p177 = np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'], float)
    rowmap = build_row_dof_map(['x']*6+jname)
    calls = dict(model_load=0, forward=0, root_newton_iterations=0, root_jacobians=0, fd_probes=0,
                 candidate_evaluations=0, inner_lp=0, line_search_trials=0, per_candidate_jacobians={})
    import mujoco as mj
    calls['model_load'] += 1; m = mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    floors = set(meta['floor_geom_ids']); gleg = {int(k): v for k, v in meta['geom_map'].items()}
    def forward(q, act90=None):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0
        a = np.zeros(90) if act90 is None else act90
        d.act[:] = a; d.ctrl[:] = a; d.qacc_warmstart[:] = 0
        mj.mj_forward(m, d); calls['forward'] += 1
        c = dict(actuator_force=np.asarray(d.actuator_force).copy(), qacc=np.asarray(d.qacc).copy(),
                 qfrc_actuator=np.asarray(d.qfrc_actuator).copy(), qfrc_passive=np.asarray(d.qfrc_passive).copy(),
                 qfrc_bias=np.asarray(d.qfrc_bias).copy(), actuator_length=np.asarray(d.actuator_length).copy(),
                 efc_D=np.asarray(d.efc_D).copy(), efc_aref=np.asarray(d.efc_aref).copy(),
                 efc_type=np.asarray(d.efc_type).copy(), efc_J_dense=np.asarray(ns['constraint_record'](mj, m, d, 0)['efc_J_dense']).copy(),
                 contact_dist=np.array([v.dist for v in d.contact]))
        c['actuator_moment'] = np.asarray(ns['moment_record'](m, d)['actuator_moment']).copy()
        c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
        return c
    def gain_bias(q):
        f1 = np.asarray(forward(q, np.ones(90))['actuator_force']); f0 = np.asarray(forward(q, np.zeros(90))['actuator_force'])
        return f1-f0, f0
    def model_at(q):
        f1 = forward(q, np.ones(90)); f0v = forward(q, np.zeros(90))
        g = np.asarray(f1['actuator_force'])-np.asarray(f0v['actuator_force']); b = np.asarray(f0v['actuator_force'])
        mom = f0v['actuator_moment']
        B = mom.T*g[None, :]; base = mom.T@b+c['qfrc_passive']-c['qfrc_bias'] if False else mom.T@b+f0v['qfrc_passive']-f0v['qfrc_bias']
        types = f0v['efc_type']; f0 = f0v['efc_D']*f0v['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        return f0v, B, base, f0
    def geometry(c, f0):
        depth = 0.; nonfoot = 0.; bearing = set(); margins = []
        for ci, (g1, g2) in enumerate(ns_geom(c)):
            pass
        return None
    def geom(c, f0):
        depth = 0.; nonfoot = 0.; bearing = set(); margins = []
        for ci, (g1, g2) in enumerate(c['contact_geom']):
            g1 = int(g1); g2 = int(g2); leg = gleg.get(g2 if g1 in floors else g1) if ((g1 in floors) != (g2 in floors)) else None
            pen = max(0., -float(c['contact_dist'][ci])); addr = int(c['contact_efc_address'][ci]); dim = int(c['contact_dim'][ci])
            if leg is None: nonfoot = max(nonfoot, pen)
            else: depth = max(depth, pen)
            if addr < 0: continue
            cnt = int(c['contact_row_count'][ci])
            if dim == 1: local = np.r_[f0[addr], 0., 0.]
            else:
                e = f0[addr:addr+4]; mu = c['contact_friction'][ci, :2]; local = np.r_[e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])]
            margins.append(float(local[0]-sum(abs(local[1:]/mu))))
            if leg is not None and local[0] > 0 and c['contact_dist'][ci] <= 0: bearing.add(leg)
        return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot), bearing_legs=len(bearing),
                    friction_margin=float(min(margins, default=0.)),
                    contact_signature='|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in c['contact_geom']))+f'#nefc{len(c["efc_type"])}')
    # constraint_record must be attached to the dict; add the pieces the helpers need
    _orig_forward = forward
    def forward_with_contact(q, act90=None):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0
        a = np.zeros(90) if act90 is None else act90
        d.act[:] = a; d.ctrl[:] = a; d.qacc_warmstart[:] = 0
        mj.mj_forward(m, d); calls['forward'] += 1
        c = dict(actuator_force=np.asarray(d.actuator_force).copy(), qacc=np.asarray(d.qacc).copy(),
                 qfrc_actuator=np.asarray(d.qfrc_actuator).copy(), qfrc_passive=np.asarray(d.qfrc_passive).copy(),
                 qfrc_bias=np.asarray(d.qfrc_bias).copy(),
                 efc_D=np.asarray(d.efc_D).copy(), efc_aref=np.asarray(d.efc_aref).copy(),
                 efc_type=np.asarray(d.efc_type).copy(),
                 contact_dist=np.array([v.dist for v in d.contact]),
                 contact_friction=np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5))
        c.update({k: np.asarray(v).copy() for k, v in ns['constraint_record'](mj, m, d, 0).items()})
        c['actuator_moment'] = np.asarray(ns['moment_record'](m, d)['actuator_moment']).copy()
        return c
    forward = forward_with_contact
    def model_at(q):
        f1 = forward(q, np.ones(90)); f0v = forward(q, np.zeros(90))
        g = np.asarray(f1['actuator_force'])-np.asarray(f0v['actuator_force']); b = np.asarray(f0v['actuator_force'])
        mom = f0v['actuator_moment']
        B = mom.T*g[None, :]; base = mom.T@b+f0v['qfrc_passive']-f0v['qfrc_bias']
        types = f0v['efc_type']; f0 = f0v['efc_D']*f0v['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        return f0v, B, base, f0
    dump(out/'PREREG.json', dict(
        task='S180 Stage B: 3-unknown root solve (z, roll, pitch) + targeted leg joints inside NATIVE XML limits',
        root_unknowns='z, roll, pitch only; x, y, yaw fixed at the S154 release values (root 6-row Jacobian rank 3, null space = x,y,yaw)',
        window='native XML limits, frozen: ' + json.dumps({jname[i]: nat[i].tolist() for i in range(42)}),
        trigger='Stage A: all six unreachable rows are interval-flippable inside native limits with 0.017-0.48 rad, and their demand is contact-dominated',
        candidates='the six per-row cheapest single-joint moves from Stage A, plus six pairwise combinations',
        per_candidate='root Newton with >=8 Jacobians and per-candidate Jacobian counting',
        baseline='no leg movement, same root solver, reported on the same table',
        budget=dict(forward_cap=CAP, reserved=RESERVED, load_cap=LOAD_CAP),
        best_rule='rank only inside {root_ok and geom_ok}; minimax and unreachable rows reported separately',
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    def qpos_of(qleg, p3):
        q = q0.copy(); q[7:] = qleg; q[0:2] = q0[0:2]; q[2] = p3[0]
        dq = quat_from_xyz(p3[1], p3[2], q0[5]); qw = quat_mul(dq, q0[3:7]); q[3:7] = qw/np.linalg.norm(qw)
        return q
    lo3 = np.array([p177[2]*.85, -.35, -.35]); hi3 = np.array([p177[2]*1.15, .35, .35])
    def evaluate(qleg, tag):
        hist = []; contact_changed = {'v': False}; jac_start = calls['root_jacobians']
        def cb(p3):
            if calls['forward'] >= CAP-RESERVED-6: raise RuntimeError('budget guard')
            q = qpos_of(qleg, p3); c = forward(q, np.zeros(90))
            g = geom(c, f0_placeholder)
            return dict(residual6=np.zeros(6))
        def cb2(p3):
            if calls['forward'] >= CAP-RESERVED-8: raise RuntimeError('budget guard')
            q = qpos_of(qleg, p3); f1 = forward(q, np.ones(90)); f0v = forward(q, np.zeros(90))
            gg = np.asarray(f1['actuator_force'])-np.asarray(f0v['actuator_force']); bb = np.asarray(f0v['actuator_force'])
            mom = f0v['actuator_moment']
            types = f0v['efc_type']; f0 = f0v['efc_D']*f0v['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
            base = mom.T@bb+f0v['qfrc_passive']-f0v['qfrc_bias']
            r6 = -(base+f0v['efc_J_dense'].T@f0)[:6]
            g = geom(f0v, f0)
            if hist and hist[-1] != g['contact_signature']: contact_changed['v'] = True
            hist.append(g['contact_signature'])
            return dict(residual6=r6)
        nr = RootNewton(cb2, lo3, hi3, cap=10**6, max_iter=8, tol=1e-8, name='inner3')
        try:
            res = nr.solve(np.array([p177[2], p177[3], p177[4]])); rn = float(res['norm_inf']); p3 = np.array(res['p']); stop = res['stop']
        except Exception as exc:
            rn = float('inf'); p3 = np.array([p177[2], p177[3], p177[4]]); stop = f'EXC:{exc}'
        calls['root_newton_iterations'] += nr.ledger['newton_iterations']; calls['root_jacobians'] += nr.ledger['jacobians']
        calls['fd_probes'] += nr.ledger['fd_probes']; calls['line_search_trials'] += nr.ledger['line_search_trials']
        calls['per_candidate_jacobians'][tag] = nr.ledger['jacobians']
        f0v, B, base, f0 = model_at(qpos_of(qleg, p3))
        g = geom(f0v, f0); A = (B@T)[6:, :]; tgt = -(base+f0v['efc_J_dense'].T@f0)[6:]
        n = A.shape[1]
        lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                     b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs')
        calls['inner_lp'] += 1; calls['candidate_evaluations'] += 1
        lo_r = np.minimum(A, 0.).sum(axis=1); hi_r = np.maximum(A, 0.).sum(axis=1)
        reach = (lo_r-1e-9 <= tgt) & (tgt <= hi_r+1e-9)
        return dict(tag=tag, root_norm=rn, root_ok=bool(rn <= 1e-8), root_stop=stop,
                    root_failure_class=classify_root([1.0, rn], rn, contact_changed=contact_changed['v']),
                    minimax=float(lp.x[-1]), activation=(T@lp.x[:n]).tolist(), qpos=qpos_of(qleg, p3).tolist(),
                    unreachable=[d for d in range(6, 48) if not reach[d-6]], unreachable_rows=[d-6 for d in range(6, 48) if not reach[d-6]],
                    geometry=g, geom_ok=bool(g['foot_depth'] <= DEPTH_CAP and g['nonfoot_depth'] <= 1e-12 and g['bearing_legs'] >= 3 and g['friction_margin'] >= -1e-10),
                    per_candidate_jacobians=nr.ledger['jacobians'])
    f0_placeholder = np.zeros(1)
    recs = []
    SIX = [19, 22, 26, 40, 42, 47]
    dump(out/'PREREG.json', dict(
        task='S182/P1-S2 pure geometric feasibility screen (no root Newton)',
        question='does any pose inside the native joint limits make all six unreachable rows interval-reachable at once',
        why='S179-S181 could not lower the 42-row minimax by leg reallocation; this asks the cheaper necessary question first',
        cost='2 forwards per pose (act=0 and act=1 give the exact engine gain); no Newton, no LP requirement',
        variables='42 leg joints inside native XML limits + root z (S177 +/-15%), roll, pitch (+/-0.35 rad); x, y, yaw fixed at release',
        sampling='Sobol sequence, fixed scramble seed 20261025, 300 poses; then coordinate pattern search on the best pose (45 coords x 2 signs x step 0.10 then 0.05)',
        score_columns='n_target_rows_reachable, max_violation_six, n_unreachable_all, root_norm_ungated (reported, NOT a gate here), geometry gates',
        note='root six-row residual is recorded but is not a constraint in this screen; any pose found here still needs a root-balance check before it can be called feasible',
        budget=dict(forward_cap=CAP, reserved=RESERVED, load_cap=LOAD_CAP),
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    state = dict(best=None)
    def geom_eval(qleg, p3, tag):
        q = qpos_of(qleg, p3)
        c1 = forward(q, np.ones(90)); c0 = forward(q, np.zeros(90))
        gv = np.asarray(c1['actuator_force'])-np.asarray(c0['actuator_force']); b = np.asarray(c0['actuator_force'])
        mom = c0['actuator_moment']
        B = mom.T*gv[None, :]; base = mom.T@b+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])
        types = c0['efc_type']; f0 = c0['efc_D']*c0['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        gw = c0['efc_J_dense'].T@f0
        A = (B@T)[6:, :]; tgt = -(base+gw)[6:]
        lo_r = np.minimum(A, 0.).sum(axis=1); hi_r = np.maximum(A, 0.).sum(axis=1)
        reach = (lo_r-1e-9 <= tgt) & (tgt <= hi_r+1e-9)
        viol = np.maximum.reduce([lo_r-tgt, np.zeros(42), tgt-hi_r])
        g = geom(c0, f0)
        calls['candidate_evaluations'] += 1
        v6 = [float(viol[d-6]) for d in SIX]
        rec = dict(tag=tag, p3=[float(x) for x in p3], n_target_rows_reachable=int(sum(1 for d in SIX if reach[d-6])),
                   max_violation_six=max(v6), violations_six=v6, n_unreachable_all=int(42-int(reach.sum())),
                   unreachable=[int(d) for d in range(6, 48) if not reach[d-6]],
                   root_norm_ungated=float(np.abs(-(base+gw)[:6]).max()), geometry=g,
                   geom_ok=bool(g['foot_depth'] <= DEPTH_CAP and g['nonfoot_depth'] <= 1e-12 and g['bearing_legs'] >= 3 and g['friction_margin'] >= -1e-10),
                   qleg=[float(x) for x in qleg])
        recs.append(rec)
        b = state['best']
        if b is None or (rec['n_target_rows_reachable'], -rec['max_violation_six']) > (b['n_target_rows_reachable'], -b['max_violation_six']):
            state['best'] = rec
        return rec
    from scipy.stats import qmc
    lo45 = np.r_[nat[:, 0], p177[2]*.85, -.35, -.35]; hi45 = np.r_[nat[:, 1], p177[2]*1.15, .35, .35]
    sampler = qmc.Sobol(d=45, scramble=True, seed=20261025)
    pts = sampler.random(300)
    for i in range(300):
        if calls['forward'] >= CAP-RESERVED-40: break
        u = pts[i]; x45 = lo45+u*(hi45-lo45)
        geom_eval(x45[:42], x45[42:], f'sobol{i:03d}')
        if i % 100 == 0: dump(out/'CANDIDATES.json', sorted(recs, key=lambda r: (-r['n_target_rows_reachable'], r['max_violation_six'])))
    base_pose = np.r_[qleg0, p177[2], p177[3], p177[4]]
    geom_eval(base_pose[:42], base_pose[42:], 'BASELINE_s177_pose')
    best = state['best']
    if best is not None:
        cur = np.r_[best['qleg'], best['p3']]
        for step in (0.10, 0.05):
            improved = True
            while improved:
                improved = False
                for j in range(45):
                    for sign in (+1, -1):
                        if calls['forward'] >= CAP-RESERVED-20: break
                        cand = cur.copy()
                        cand[j] = float(np.clip(cand[j]+sign*step*(hi45[j]-lo45[j]), lo45[j], hi45[j]))
                        r = geom_eval(cand[:42], cand[42:], f'refine_j{j}_{sign:+d}_{step:g}')
                        if (r['n_target_rows_reachable'], -r['max_violation_six']) > (best['n_target_rows_reachable'], -best['max_violation_six']):
                            cur = cand; best = r; state['best'] = r; improved = True
                    if calls['forward'] >= CAP-RESERVED-20: break
    best = state['best']
    ranked = sorted(recs, key=lambda r: (-r['n_target_rows_reachable'], r['max_violation_six']))
    dump(out/'CANDIDATES.json', ranked)
    summary = dict(poses=len(recs), best=best, baseline=[r for r in recs if r['tag'] == 'BASELINE_s177_pose'][0] if any(r['tag'] == 'BASELINE_s177_pose' for r in recs) else None,
                   histogram_n_reachable={k: sum(1 for r in recs if r['n_target_rows_reachable'] == k) for k in range(7)},
                   any_pose_all_six_reachable=bool(best is not None and best['n_target_rows_reachable'] == 6),
                   crosstab_n6_by_geom={g: {k: sum(1 for r in recs if r['geom_ok'] == g and r['n_target_rows_reachable'] == k) for k in range(7)} for g in (True, False)},
                   best_geom_valid=max([r for r in recs if r['geom_ok']] or [None], key=lambda r: (r['n_target_rows_reachable'], -r['max_violation_six'])) if any(r['geom_ok'] for r in recs) else None,
                   n_geom_valid=sum(1 for r in recs if r['geom_ok']),
                   calls=calls, units='native; SI NOT_ESTABLISHED', note='root six-row residual is recorded but NOT gated here')
    dump(out/'SUMMARY.json', summary)
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(dict(poses=len(recs), best_n=best['n_target_rows_reachable'] if best else None,
        best_viol=best['max_violation_six'] if best else None, forward=calls['forward'],
        any_all_six=summary['any_pose_all_six_reachable'], base_n=(summary['baseline'] or {}).get('n_target_rows_reachable')))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path)
    a = ap.parse_args(); main(a.output or BASE/'s182-geometric-feasibility'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
