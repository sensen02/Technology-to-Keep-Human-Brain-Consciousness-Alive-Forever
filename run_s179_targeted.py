"""S179/P1-S2 targeted leg-joint search: fix pipeline semantics, then one directed search.

Scope: only the right-middle (RM) leg's 7 joints may move (Stage A showed DOF 42 /
joint_row 36 / RMCoxa_yaw is the only row with a sign-flip channel inside the window).
Root balance is an inner six-row Newton with >=8 Jacobians per candidate; failures are
classified as DIVERGE / CONTACT_SET_CHANGE / RESIDUAL_FLOOR. Root residual and the
42-row LP minimax are reported as two independent columns and are never merged.
No mj_step, no neural ticks, no training, no dynamics.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil, traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
from s179_core import (build_row_dof_map, leg_of_name, validate_identity, classify_root,
                       select_best, row_of_dof)

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP, RESERVED, LOAD_CAP = 2500, 200, 3
DEPTH_CAP = 0.03442910871489635
TARGET_ROW = 36          # joint_row; DOF 42
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
    files = [Path(__file__).resolve(), ROOT/'s179_core.py', ROOT/'test_s179_core.py', ROOT/'root_newton_core.py', XML,
             S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz', S154/'init80ms/MODEL_PROVENANCE.json',
             S154/'init80ms/NATIVE_MODEL_ARRAYS.npz', S154/'init80ms/COM_CONTACT_METADATA.json', S166/'MAPPING.npz',
             S177/'ROOT_BALANCE_STATE.npz', S177/'ROOT_RESULT.json', BASE/'s178-legjoint-release-search/CONCLUSION.zh-CN.md']
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
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',)), (ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))]
    text = '\n'.join(extract(p, n) for p, n in specs)
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile(text, '<inert>', 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T = load(S166/'MAPPING.npz')['T']
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    nodes = list(tree.getroot().find('actuator')); jnames = [j.get('name') for j in joints[1:]]
    rng = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    lower = np.maximum(q0[7:]-0.6, rng[:, 0]); upper = np.minimum(q0[7:]+0.6, rng[:, 1])
    assert np.all(lower < upper)
    base177 = json.loads((S177/'ROOT_RESULT.json').read_text()); p177 = np.array(base177['p'], float)
    qleg0 = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:].copy()
    calls = dict(model_load=0, forward=0, root_newton_iterations=0, root_jacobians=0, fd_probes=0,
                 candidate_evaluations=0, inner_lp=0, line_search_trials=0)
    import mujoco as mj
    assert mj.__version__ == '3.9.0'
    calls['model_load'] += 1; m = mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m) == proof['before']
    m.actuator_dynprm[:, 0] = .001; m.actuator_dynprm[:, 1] = .0004
    m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; m.opt.timestep = .000025
    assert ns['snapshot'](m) == proof['after']
    # ---- row/DOF identity from the model itself, cross-checked against the XML text ----
    dof_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or f'dof{j}' for j in range(m.njnt)]
    dof_of_joint = {j: int(m.jnt_dofadr[j]) for j in range(m.njnt)}
    name_by_dof = {}
    for j in range(m.njnt):
        adr = int(m.jnt_dofadr[j]); nd = 3 if int(m.jnt_type[j]) == int(mj.mjtJoint.mjJNT_BALL) else 1
        for k in range(nd):
            name_by_dof[adr+k] = dof_names[j]+(f'dof{k}' if nd > 1 else '')
    for d in range(6):
        name_by_dof.setdefault(d, f'root_dof{d}')
    assert set(name_by_dof) == set(range(48)), sorted(name_by_dof)
    rowmap = build_row_dof_map([name_by_dof[d] for d in range(48)])
    for k, v in enumerate(rowmap):
        assert v['dof'] == k+6 and v['dof_name'] == jnames[k], f'row/DOF identity broken at row {k}'
    dump(out/'ROW_DOF_MAP.json', dict(note='42 balance rows <-> real DOFs, from model.jnt_dofadr + names, cross-checked against XML joint order',
         rows=rowmap, leg_blocks={leg: [v['dof'] for v in rowmap if v['leg'] == leg] for leg in ('lf','lh','lm','rf','rh','rm')}))
    assert row_of_dof(rowmap, 42) == TARGET_ROW
    assert [v for v in rowmap if v['dof'] == 42][0]['dof_name'] == 'joint_RMCoxa_pitch'
    assert [v for v in rowmap if v['dof'] == 41][0]['dof_name'] == 'joint_RMCoxa_yaw'
    validate_identity(rowmap, [v['joint_row'] for v in rowmap], [v['dof'] for v in rowmap], 'rowmap self')
    floors = set(meta['floor_geom_ids']); geomleg = {int(k): leg_of_name(v.upper().replace('LEG', '')+'_') if False else v for k, v in meta['geom_map'].items()}
    def forward(q, act90=None):
        d = mj.MjData(m); d.time = float(z['time'][0])
        for n in STATE: getattr(d, n)[:] = z[n][0]
        for n in OPTIONAL:
            if n in z: getattr(d, n)[:] = z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0
        a = np.zeros(90) if act90 is None else act90
        d.act[:] = a; d.ctrl[:] = a; d.qacc_warmstart[:] = 0
        init = {n: np.asarray(getattr(d, n)).copy() for n in (*STATE, *OPTIONAL)}
        mj.mj_forward(m, d); calls['forward'] += 1
        assert all(np.array_equal(getattr(d, n), init[n]) for n in (*STATE, *OPTIONAL))
        c = ns['constraint_record'](mj, m, d, 0); c.update(ns['moment_record'](m, d))
        for n in ('qacc', 'act_dot', 'actuator_force', 'actuator_length', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias',
                  'efc_D', 'efc_aref', 'efc_type', 'efc_frictionloss'):
            c[n] = np.asarray(getattr(d, n)).copy()
        c['contact_dist'] = np.array([v.dist for v in d.contact]); c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
        return c
    def affine(c):
        types = c['efc_type']; assert np.all(np.isin(types, [0, 3, 4, 5, 6]))
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
            cnt = int(c['contact_row_count'][ci])
            if dim == 1: local = np.r_[f0[addr], 0., 0.]
            else:
                e = f0[addr:addr+4]; mu = c['contact_friction'][ci, :2]; local = np.r_[e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])]
            margins.append(float(local[0]-sum(abs(local[1:]/mu))))
            if leg is not None and local[0] > 0 and c['contact_dist'][ci] <= 0: bearing.add(leg)
        return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot), bearing_legs=len(bearing),
                    friction_margin=float(min(margins, default=0.)),
                    contact_signature='|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in c['contact_geom']))+f'#nefc{len(c["efc_type"])}')
    def build_qpos(qleg, p):
        q = q0.copy(); q[7:] = qleg; q[0:2] = p[0:2]; q[2] = p[2]
        dq = quat_from_xyz(p[3], p[4], p[5]); qw = quat_mul(dq, q0[3:7]); q[3:7] = qw/np.linalg.norm(qw)
        return q
    lo6 = np.array([p177[0]-.3, p177[1]-.3, p177[2]*.85, -.35, -.35, -.35])
    hi6 = np.array([p177[0]+.3, p177[1]+.3, p177[2]*1.15, .35, .35, .35])
    def evaluate(qleg, p_start):
        hist = []
        contact_changed = {'v': False}
        def cb(pv):
            if calls['forward'] >= CAP-RESERVED-3:
                raise RuntimeError('forward budget guard')
            c = forward(build_qpos(qleg, pv)); B, base, f0 = affine(c); g = geometry(c, f0)
            if hist:
                if g['contact_signature'] != hist[-1]['sig']:
                    contact_changed['v'] = True
            hist.append(dict(norm=None, sig=g['contact_signature']))
            bb = -(base+c['efc_J_dense'].T@f0)
            hist[-1]['norm'] = float(abs(bb[:6]).max())
            return dict(residual6=bb[:6].copy())
        nr = RootNewton(cb, lo6, hi6, cap=10**6, max_iter=8, tol=1e-8, name='inner')
        try:
            res = nr.solve(p_start)
            root_norm = float(res['norm_inf']); root_p = np.array(res['p']); stop = res['stop']
        except Exception as exc:
            root_norm = float('inf'); root_p = np.array(p_start); stop = f'EXC:{exc}'
        calls['root_newton_iterations'] += nr.ledger['newton_iterations']; calls['root_jacobians'] += nr.ledger['jacobians']
        calls['fd_probes'] += nr.ledger['fd_probes']; calls['line_search_trials'] += nr.ledger['line_search_trials']
        c = forward(build_qpos(qleg, root_p)); B, base, f0 = affine(c); geom = geometry(c, f0)
        bb = -(base+c['efc_J_dense'].T@f0); A = (B@T)[6:, :]; tgt = bb[6:]
        n = A.shape[1]
        lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                     b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs',
                     options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        assert lp.success; calls['inner_lp'] += 1
        lo_row = np.minimum(A, 0.).sum(axis=1); hi_row = np.maximum(A, 0.).sum(axis=1)
        reach = (lo_row-1e-9 <= tgt) & (tgt <= hi_row+1e-9)
        viol = np.maximum.reduce([lo_row-tgt, np.zeros(42), tgt-hi_row])
        norms = [h['norm'] for h in hist]
        cls = classify_root(norms, root_norm, tol=1e-8, contact_changed=contact_changed['v'],
                            nonfinite=any(not np.isfinite(x) for x in norms))
        returns = dict(
            root_norm=root_norm, root_ok=bool(root_norm <= 1e-8), root_p=root_p.tolist(), root_stop=stop,
            root_failure_class=cls, root_residual_history=norms, root_contact_changed=bool(contact_changed['v']),
            minimax=float(lp.x[-1]), unreachable=[int(v['dof']) for v in rowmap if not reach[v['joint_row']]],
            unreachable_rows=[int(v['joint_row']) for v in rowmap if not reach[v['joint_row']]],
            target_row_reachable=bool(reach[TARGET_ROW]), row_violation=viol.tolist(),
            row_lo=lo_row.tolist(), row_hi=hi_row.tolist(), row_target=tgt.tolist(),
            geometry=geom, geom_ok=bool(geom['foot_depth'] <= DEPTH_CAP and geom['nonfoot_depth'] <= 1e-12
                                       and geom['bearing_legs'] >= 3 and geom['friction_margin'] >= -1e-10),
            activation=(T@lp.x[:n]).tolist(), qpos=build_qpos(qleg, root_p).tolist(),
            contact_signature=geom['contact_signature'])
        validate_identity(rowmap, list(range(42)), [v['dof'] for v in rowmap], 'candidate_table')
        calls['candidate_evaluations'] += 1
        return returns
    # ---- pre-registered candidate schedule: only the RM leg's 7 joints move ----
    rm_dofs = [v['dof'] for v in rowmap if v['leg'] == 'rm']
    rm_idx = [d-6 for d in rm_dofs]
    yaw = [i for i in rm_idx if rowmap[i]['dof_name'] == 'joint_RMCoxa_yaw'][0]   # rm_idx is joint_row space
    sched = []
    for j in rm_idx:
        for dq in (+0.075, -0.075):
            sched.append(([(j, dq)], f'RM {rowmap[j]["dof_name"]} {dq:+.3f}'))
    for dq in (+0.02, -0.02, +0.028, -0.028, +0.04, -0.04, +0.06, -0.06):
        sched.append(([(yaw, dq)], f'RMCoxa_yaw {dq:+.3f} fine'))
    tib = [i for i in rm_idx if 'Tibia' in rowmap[i]['dof_name']][0]   # joint_row space
    for dq in (+0.05, -0.05):
        sched.append(([(yaw, dq), (tib, dq)], f'RMCoxa_yaw+Tibia {dq:+.3f}'))
    sched = sched[:24]
    dump(out/'PREREG.json', dict(
        task='S179/P1-S2 pipeline-fixed targeted leg-joint search (RM leg only)',
        identity='every table/ledger row carries joint_row, dof, dof_name, leg, joint_name; ROW_DOF_MAP.json built from model.jnt_dofadr and cross-checked against XML order',
        best_rule='rank ONLY inside {root_ok and geom_ok}; empty set => NO_ROOT_BALANCED_GEOM_VALID_POINT; root and joint residuals always two columns',
        root_failure_classes=['ROOT_BALANCED', 'ROOT_DIVERGE', 'ROOT_CONTACT_SET_CHANGE', 'ROOT_RESIDUAL_FLOOR'],
        frozen_ranges=dict(leg_joint='native XML limits AND release +/- 0.6 rad', lower=lower.tolist(), upper=upper.tolist(),
                           root='x,y within +/-0.30 of S177 root; z within +/-15%; roll,pitch,yaw within +/-0.35 rad'),
        targeted='RM leg 7 joints only; target joint_row %d (dof 42, RMCoxa_yaw, Stage A need 0.0278 rad)' % TARGET_ROW,
        other_rows='the other five unreachable rows may only change via contact/target with foot placement; if unchanged, report as the limited conclusion',
        schedule=sched, budget=dict(forward_cap=CAP, reserved=RESERVED, model_load_cap=LOAD_CAP, per_candidate_root_jacobians_min=8),
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    cands = []
    p_warm = p177.copy()
    for spec, note in sched:
        if calls['forward'] >= CAP-RESERVED-40:
            break
        qleg = qleg0.copy()
        for j, dq in spec:
            qleg[j] = float(np.clip(qleg0[j]+dq, lower[j], upper[j]))
        rec = evaluate(qleg, p_warm)
        if rec['root_ok']:
            p_warm = np.array(rec['root_p'])
        cands.append(dict(index=len(cands)+1, note=note, moves=[[int(j), float(dq)] for j, dq in spec],
                          joint_offset_from_release={rowmap[j]['dof_name']: float(qleg[j]-qleg0[j]) for j, _ in spec},
                          **{k: v for k, v in rec.items() if k not in ('activation', 'qpos')}))
        dump(out/'CANDIDATES.json', cands)
    sel = select_best(cands)
    counts = {k: sum(1 for c in cands if c['root_failure_class'] == k) for k in ('ROOT_BALANCED', 'ROOT_DIVERGE', 'ROOT_CONTACT_SET_CHANGE', 'ROOT_RESIDUAL_FLOOR')}
    reps = {}
    for k in counts:
        sub = [c for c in cands if c['root_failure_class'] == k]
        if sub: reps[k] = dict(candidate=sub[0]['index'], note=sub[0]['note'], root_norm=sub[0]['root_norm'],
                               residual_history=sub[0]['root_residual_history'])
    dump(out/'ROOT_FAILURE_CLASSES.json', dict(counts=counts, representatives=reps,
         contact_set_change_count=sum(1 for c in cands if c['root_contact_changed']),
         note='classes are physically distinct; a root-unbalanced candidate is never accepted for a better joint minimax'))
    rows_tbl = []
    for c in cands:
        for i in range(42):
            v = rowmap[i]
            rows_tbl.append(dict(candidate=c['index'], joint_row=v['joint_row'], dof=v['dof'], dof_name=v['dof_name'],
                                 leg=v['leg'], joint_name=v['dof_name'], target=c['row_target'][i] if 'row_target' in c else None,
                                 reachable=None, is_target_row=bool(v['joint_row'] == TARGET_ROW)))
    with (out/'CANDIDATES.csv').open('w', newline='') as f:
        keys = [k for k in cands[0] if k not in ('row_lo', 'row_hi', 'row_target', 'row_violation', 'activation', 'qpos', 'geometry')]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for c in cands: w.writerow({k: sanitize(c[k]) for k in keys})
    reach_rows = []
    for c in cands:
        for i in range(42):
            v = rowmap[i]
            reach_rows.append(dict(candidate=c['index'], joint_row=v['joint_row'], dof=v['dof'], dof_name=v['dof_name'],
                                   leg=v['leg'], joint_name=v['dof_name'], target=c['row_target'][i],
                                   lo=c['row_lo'][i], hi=c['row_hi'][i], violation=c['row_violation'][i],
                                   reachable=bool(c['row_lo'][i]-1e-9 <= c['row_target'][i] <= c['row_hi'][i]+1e-9),
                                   is_target_row=bool(v['joint_row'] == TARGET_ROW)))
    with (out/'ROW_REACHABILITY.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(reach_rows[0])); w.writeheader(); w.writerows(reach_rows)
    best = sel['best']
    conf = None
    if best is not None and best['minimax'] <= 1e-8:
        act = np.array(best['activation']); q = np.array(best['qpos'])
        cf = forward(q, act); Bc, basec, f0c = affine(cf); res48 = Bc@act+basec+cf['efc_J_dense'].T@f0c
        g = cf and geometry(cf, f0c)
        types = cf['efc_type']; f0l = np.where(types == 0, cf['efc_D']*cf['efc_aref'], np.maximum(cf['efc_D']*cf['efc_aref'], 0.))
        nsum = 0.0
        for ci in range(len(cf['contact_geom'])):
            addr = int(cf['contact_efc_address'][ci]); dim = int(cf['contact_dim'][ci])
            if addr < 0: continue
            nsum += float(f0l[addr] if dim == 1 else f0l[addr:addr+4].sum())
        weight = float(np.load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass'].sum()*9801.)
        conf = dict(max_abs_residual48=float(abs(res48).max()), max_abs_qacc=float(abs(cf['qacc']).max()),
                    act_dot_all_zero=bool(not np.any(cf['act_dot'])),
                    vertical_identity=dict(total_contact_normal=nsum, body_weight=weight, difference=nsum-weight,
                                           holds=bool(abs(nsum-weight) <= 1e-6*max(weight, 1.))),
                    geometry=g, all48_balanced=bool(abs(res48).max() <= 1e-8))
        dump(out/'CONFIRMATION.json', conf)
        np.savez_compressed(out/'CANDIDATE_STATE.npz', qpos=q, act90=act, res48=res48, qacc=cf['qacc'], act_dot=cf['act_dot'])
    target_tbl = [dict(candidate=c['index'], note=c['note'], root_failure_class=c['root_failure_class'],
                       root_norm=c['root_norm'], minimax=c['minimax'], target_row_reachable=c['target_row_reachable'],
                       unreachable_dofs=c['unreachable'], n_unreachable=len(c['unreachable'])) for c in cands]
    with (out/'TARGET_ROW_TRACK.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(target_tbl[0])); w.writeheader(); w.writerows(target_tbl)
    summary = dict(task='S179', candidates=len(cands), selection=sel['selection'],
                   best=({k: v for k, v in best.items() if k not in ('row_lo', 'row_hi', 'row_target', 'row_violation', 'activation', 'qpos', 'geometry')} if best else None),
                   root_failure_counts=counts, target_row=TARGET_ROW,
                   target_row_reachable_any=bool(any(c['target_row_reachable'] for c in cands)),
                   n_unreachable_first=len(cands[0]['unreachable']), n_unreachable_min=min(len(c['unreachable']) for c in cands),
                   unreachable_dofs_first=cands[0]['unreachable'],
                   unreachable_dofs_min=min((c['unreachable'] for c in cands), key=len),
                   confirmation=conf, calls=calls, residual_definition='root |residual6|inf and 42-row LP minimax are reported separately and never merged',
                   units='native; SI NOT_ESTABLISHED', dynamics=False, neural=False, training=False)
    dump(out/'SUMMARY.json', summary)
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(summary)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    main(a.output or BASE/'s179-targeted-legjoint-search'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
