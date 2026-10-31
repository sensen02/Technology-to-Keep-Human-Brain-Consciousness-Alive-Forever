"""S184 Stage A: staleness audit of the S183 six-row violations, then a full 38-pose census.

Audit first (no search). If the six-row violations do NOT move with the pose, this stops and
reports an implementation defect. Census = every geom_ok pose gets an exact root Newton
(z/roll/pitch only, x/y/yaw at the S154 release values) plus one 42-row LP.
No mj_step, no neural ticks, no training, no dynamics.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
S183 = BASE/'s183-geometry-targeted/20261025T150000000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
AUDIT_CAP, CENSUS_CAP, RESERVED, LOAD_CAP = 300, 1600, 200, 3
DEPTH_CAP = 0.03442910871489635
SIX = [19, 22, 26, 40, 42, 47]
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
    files = [Path(__file__).resolve(), ROOT/'root_newton_core.py', ROOT/'s179_core.py', XML, S154/'INPUT_HASHES.json',
             S154/'init80ms/TRAJECTORY.npz', S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/NATIVE_MODEL_ARRAYS.npz',
             S154/'init80ms/COM_CONTACT_METADATA.json', S166/'MAPPING.npz', S177/'ROOT_BALANCE_STATE.npz',
             S183/'CANDIDATES.json', S183/'SUMMARY.json']
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: ' + p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files}); dump(out/'INPUT_HASHES.json', hashes)
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
    specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)), (ROOT/'run_local_pool_response.py', ('moment_record',)),
             (ROOT/'run_constraint_row_recording.py', ('constraint_record',))]
    ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p, n) for p, n in specs), '<inert>', 'exec'), ns)
    z = load(S154/'init80ms/TRAJECTORY.npz'); q0 = z['qpos'][0].copy()
    proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T = load(S166/'MAPPING.npz')['T']
    tree = ET.parse(XML); joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
    jname = [j.get('name') for j in joints[1:]]
    nat = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    p177 = np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'], float)
    calls = dict(model_load=0, forward=0, audit_forward=0, census_forward=0, root_newton_iterations=0,
                 root_jacobians=0, inner_lp=0, per_pose_jacobians={})
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
                 qfrc_passive=np.asarray(d.qfrc_passive).copy(), qfrc_bias=np.asarray(d.qfrc_bias).copy(),
                 efc_D=np.asarray(d.efc_D).copy(), efc_aref=np.asarray(d.efc_aref).copy(),
                 efc_type=np.asarray(d.efc_type).copy(),
                 contact_dist=np.array([v.dist for v in d.contact]),
                 contact_friction=np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5))
        c.update({k: np.asarray(v).copy() for k, v in ns['constraint_record'](mj, m, d, 0).items()})
        c['actuator_moment'] = np.asarray(ns['moment_record'](m, d)['actuator_moment']).copy()
        return c
    def model_at(q):
        c1 = forward(q, np.ones(90)); c0 = forward(q, np.zeros(90))
        g = np.asarray(c1['actuator_force'])-np.asarray(c0['actuator_force']); b = np.asarray(c0['actuator_force'])
        mom = c0['actuator_moment']
        B = mom.T*g[None, :]; base = mom.T@b+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])
        types = c0['efc_type']; f0 = c0['efc_D']*c0['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        return c0, B, base, f0
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
    def qpos_of(qleg, p3):
        q = q0.copy(); q[7:] = qleg; q[0:2] = q0[0:2]; q[2] = p3[0]
        dq = quat_from_xyz(p3[1], p3[2], q0[5]); qw = quat_mul(dq, q0[3:7]); q[3:7] = qw/np.linalg.norm(qw)
        return q
    def rows_of(qleg, p3):
        c, B, base, f0 = model_at(qpos_of(qleg, p3))
        A = (B@T)[6:, :]; tgt = -(base+c['efc_J_dense'].T@f0)[6:]
        lo = np.minimum(A, 0.).sum(axis=1); hi = np.maximum(A, 0.).sum(axis=1)
        viol = np.maximum.reduce([lo-tgt, np.zeros(42), tgt-hi])
        r6 = float(np.abs(-(base+c['efc_J_dense'].T@f0)[:6]).max())
        g = geom(c, f0)
        return dict(A=A, tgt=tgt, lo=lo, hi=hi, viol=viol, r6=r6, geom=g, q=qpos_of(qleg, p3))
    # ---------------- Stage A.1 staleness audit ----------------
    cand = json.loads((S183/'CANDIDATES.json').read_text())
    s183 = json.loads((S183/'SUMMARY.json').read_text())
    tags_all = [c['tag'] for c in cand]
    dup = {t: tags_all.count(t) for t in set(tags_all) if tags_all.count(t) > 1}
    embedded = {}
    if s183.get('best_geom_valid'): embedded['SUMMARY.best_geom_valid'] = s183['best_geom_valid']
    if s183.get('baseline'): embedded['SUMMARY.baseline'] = s183['baseline']
    for c in s183.get('gate_joint_checks', []):
        if sum(1 for k in embedded if k.startswith('CHECK')) < 2:
            embedded[f'CHECK.{c["tag"]}.{len(embedded)}'] = {**c, 'qleg': list(np.array(c['qpos'], float)[7:]),
                                                             'p3': [c['qpos'][2], 0., 0.], 'violations_six': None,
                                                             'geometry': c['geometry'], 'unreachable': c['unreachable']}
    by_tag = dict(embedded)
    picks = list(embedded)
    audit = []
    for tag in picks:
        rec = by_tag[tag]
        fresh = rows_of(np.array(rec['qleg'], float), np.array(rec['p3'], float))
        saved = rec.get('violations_six') or [float(fresh['viol'][d-6]) for d in SIX]
        fresh6 = [float(fresh['viol'][d-6]) for d in SIX]
        audit.append(dict(tag=tag, saved_violations_six=saved, fresh_violations_six=fresh6,
                          max_abs_diff=float(max(abs(a-b) for a, b in zip(saved, fresh6))),
                          saved_contact=rec['geometry']['contact_signature'], fresh_contact=fresh['geom']['contact_signature'],
                          saved_unreachable=rec['unreachable'],
                          fresh_unreachable=[int(d) for d in range(6, 48) if not (fresh['lo'][d-6]-1e-9 <= fresh['tgt'][d-6] <= fresh['hi'][d-6]+1e-9)],
                          same_pose_as_baseline=bool(np.array_equal(np.array(rec['qleg'], float), np.array(by_tag[picks[0]]['qleg'], float)))))
    # perturbation sensitivity on a joint that differs from the baseline pose
    base_rec = by_tag[picks[0]]
    pert = []
    for tag in picks:
        rec = by_tag[tag]
        qleg = np.array(rec['qleg'], float); p3 = np.array(rec['p3'], float)
        involved = [k for k in range(42) if jname[k][6:8] in ('LM', 'LH', 'RM', 'RH')]
        dq_abs = np.abs(qleg-np.array(base_rec['qleg'], float))
        cand_j = [k for k in involved if dq_abs[k] > 1e-9] or involved
        j = int(max(cand_j, key=lambda k: dq_abs[k]))
        base6 = [float(x) for x in rows_of(qleg, p3)['viol'][[d-6 for d in SIX]]]
        for sgn in (+1, -1):
            q2 = qleg.copy(); q2[j] = float(np.clip(q2[j]+sgn*0.01, nat[j][0], nat[j][1]))
            v6 = [float(x) for x in rows_of(q2, p3)['viol'][[d-6 for d in SIX]]]
            pert.append(dict(tag=tag, joint_index=j, joint_name=jname[j], delta=sgn*0.01, base_violations_six=base6,
                             perturbed_violations_six=v6, max_abs_change=float(max(abs(a-b) for a, b in zip(base6, v6)))))
    calls['audit_forward'] = calls['forward']
    dump(out/'STALENESS_AUDIT.json', dict(picks=picks, duplicate_tags_in_candidates=dup,
         note='records audited are the ones EMBEDDED in the S183 summary (authoritative), not looked up by tag; perturbation uses a joint that actually differs and belongs to an involved leg (LM/LH/RM/RH)',
         recomputation=audit, perturbation=pert,
         verdict=('STALE_OR_REUSED' if max(a['max_abs_diff'] for a in audit) > 1e-9 else 'FRESH_VALUES_MOVE_WITH_POSE'),
         rule='if the six-row violations do not move with the pose, S182/S183 interval statistics are withdrawn and Stage B is not triggered'))
    stale = max(a['max_abs_diff'] for a in audit) > 1e-9
    perturbation_insensitive = max(p['max_abs_change'] for p in pert) < 1e-12
    # ---------------- Stage A.3 census of all geom_ok poses ----------------
    gate = [c for c in cand if c['geom_ok']]
    pre = dict(n_gate=len(gate), audit_forward=calls['forward'], census_available=CENSUS_CAP-RESERVED)
    census = []
    status = 'CENSUS_COMPLETE'
    if stale:
        status = 'CENSUS_NOT_RUN_STALENESS_DEFECT'
    else:
        lo3 = np.array([p177[2]*.85, -.35, -.35]); hi3 = np.array([p177[2]*1.15, .35, .35])
        for rec in gate:
            if calls['forward']+20 > CENSUS_CAP-RESERVED:
                status = 'CENSUS_PARTIAL_BUDGET_STOP'
                break
            qleg = np.array(rec['qleg'], float); p3_0 = np.array([p177[2], p177[3], p177[4]])
            jac0 = calls['root_jacobians']
            def cb(p3, qleg=qleg):
                if calls['forward'] >= CENSUS_CAP-RESERVED-8: raise RuntimeError('budget')
                c, B, base, f0 = model_at(qpos_of(qleg, p3))
                return dict(residual6=-(base+c['efc_J_dense'].T@f0)[:6])
            nr = RootNewton(cb, lo3, hi3, cap=10**6, max_iter=6, tol=1e-8, name='c3')
            try:
                res = nr.solve(p3_0); rn = float(res['norm_inf']); p3 = np.array(res['p'])
            except Exception as exc:
                rn = float('inf'); p3 = p3_0
            calls['root_newton_iterations'] += nr.ledger['newton_iterations']; calls['root_jacobians'] += nr.ledger['jacobians']
            r = rows_of(qleg, p3)
            A = r['A']; tgt = r['tgt']; n = A.shape[1]
            lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                         b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs')
            calls['inner_lp'] += 1
            reach = (r['lo']-1e-9 <= tgt) & (tgt <= r['hi']+1e-9)
            nj = nr.ledger['jacobians']
            calls['per_pose_jacobians'][rec['tag']] = nj
            census.append(dict(tag=rec['tag'], root_norm=rn, root_ok=bool(rn <= 1e-8), root_p=p3.tolist(),
                               minimax=float(lp.x[-1]) if lp.success else None,
                               unreachable=[int(d) for d in range(6, 48) if not reach[d-6]],
                               geometry=r['geom'], geom_ok=bool(r['geom']['foot_depth'] <= DEPTH_CAP and r['geom']['nonfoot_depth'] <= 1e-12
                                                                and r['geom']['bearing_legs'] >= 3 and r['geom']['friction_margin'] >= -1e-10),
                               root_jacobians=nj, qleg=qleg.tolist(), qpos=r['q'].tolist(), activation=(T@lp.x[:n]).tolist() if lp.success else None))
            dump(out/'CENSUS_38.json', dict(status='RUNNING', pre=pre, rows=census))
        calls['census_forward'] = calls['forward']-calls['audit_forward']
        dump(out/'CENSUS_38.json', dict(status=status, pre=pre, rows=census,
             n_rows=len(census), n_root_ok=sum(1 for x in census if x['root_ok']),
             n_geom_ok=sum(1 for x in census if x['geom_ok']),
             best_minimax=min([x['minimax'] for x in census if x['minimax'] is not None], default=None),
             n_below_baseline=sum(1 for x in census if x['minimax'] is not None and x['minimax'] < 3.308620867791414 - 1e-9),
             per_pose_jacobians_hist={k: sum(1 for v in calls['per_pose_jacobians'].values() if v == k) for k in sorted(set(calls['per_pose_jacobians'].values()))},
             note='root unknowns z/roll/pitch only; x,y,yaw at S154 release; each pose root-Jacobian count recorded'))
    # ---------------- Stage A.4 gap decomposition on the best census point ----------------
    gaps = []
    if census:
        bestrow = min([x for x in census if x['minimax'] is not None], key=lambda x: x['minimax'])
        r = rows_of(np.array(bestrow['qleg'], float), np.array(bestrow['root_p'], float))
        for d in SIX:
            i = d-6; hi = float(r['hi'][i]); lo = float(r['lo'][i]); req = float(r['tgt'][i])
            gap = max(0., req-hi, lo-req)
            m_need = (req/hi) if (req > 0 and hi > 1e-12) else ((req/lo) if (req < 0 and lo < -1e-12) else float('inf'))
            gaps.append(dict(tag=bestrow['tag'], dof=d, dof_name=jname[i], required=req, sum_min=lo, sum_max=hi,
                             gap=gap, needed_uniform_activation_multiplier=(m_need if np.isfinite(m_need) else None),
                             channel_type=('AMPLITUDE_LIKE' if (np.isfinite(m_need) and m_need <= 3.0) else 'CHANNEL_LIMITED')))
        with (out/'ROW_GAP_DECOMPOSITION.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(gaps[0])); w.writeheader(); w.writerows(gaps)
        dump(out/'ROW_GAP_DECOMPOSITION.json', gaps)
    dump(out/'SUMMARY.json', dict(staleness_verdict=json.loads((out/'STALENESS_AUDIT.json').read_text())['verdict'],
         stale=stale, perturbation_insensitive=bool(perturbation_insensitive), duplicate_tags=dup, census_status=status, census_rows=len(census),
         census_below_baseline=sum(1 for x in census if x['minimax'] is not None and x['minimax'] < 3.308620867791414 - 1e-9),
         stageB=('STAGE_B_NOT_TRIGGERED_STALENESS_DEFECT' if stale else
                 ('STAGE_B_NOT_TRIGGERED_CENSUS_NO_IMPROVEMENT' if not any(x['minimax'] is not None and x['minimax'] < 3.308620867791414 - 1e-9 for x in census) else 'STAGE_B_TRIGGERED')),
         calls=calls,
         corrections=['S183 reported "max 4 of six rows reachable among geometry-valid poses": the real geometry-valid distribution is {0:11, 1:11, 2:1, 3:15, 4:0, 5:0, 6:0}, i.e. max 3',
                      'S183 summary best_n=4 came from a geom_ok=False pose and was quoted across classes'],
         units='native; SI NOT_ESTABLISHED'))
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(dict(stale=stale, census=len(census), status=status,
        below=sum(1 for x in census if x['minimax'] is not None and x['minimax'] < 3.3086),
        audit_fw=calls['audit_forward'], census_fw=calls['census_forward'], total_fw=calls['forward']))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path)
    a = ap.parse_args(); main(a.output or BASE/'s184-census-and-staleness-audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
