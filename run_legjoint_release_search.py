"""S178/P1-S2: release the 42 leg joints from the S177 root-balanced baseline.

Stage A (<=500 forwards): per-row sign-flip pre-check on the six unreachable joint rows.
Stage B (<=3000 forwards, <=3 loads, 200 reserved): block hill-climb over leg joints with an
inner six-row root Newton (warm start), 42-row LP minimax as the only search objective,
geometry as a hard constraint, contact-set changes counted. No mj_step, no neural ticks.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, hashlib, json, shutil, traceback
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
from legjoint_release_core import BlockHillClimb, row_intervals, sign_flip_estimate

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
STAGE_A_CAP, STAGE_B_CAP, STAGE_B_RESERVED, STAGE_B_LOADS = 500, 3000, 200, 3
DEPTH_CAP = 0.03442910871489635
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
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

class Engine:
    def __init__(self, out, stage, load_cap):
        self.out = out; self.calls = dict(model_load=0, forward=0); self.load_cap = load_cap
        specs = [(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)),
                 (ROOT/'run_local_pool_response.py', ('moment_record',)),
                 (ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
                 (ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))]
        text = '\n'.join(extract(p, n) for p, n in specs)
        self.ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
        exec(compile(text, '<inert>', 'exec'), self.ns)
        self.z = load(S154/'init80ms/TRAJECTORY.npz'); self.q0 = self.z['qpos'][0].copy()
        self.proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
        self.meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
        self.mapping = load(S166/'MAPPING.npz'); self.T = self.mapping['T']
        tree = ET.parse(XML); self.joints = [n for n in tree.getroot().find('worldbody').iter('joint')]
        self.nodes = list(tree.getroot().find('actuator'))
        self.names = [j.get('name') for j in self.joints[1:]]
        rng = np.array([list(map(float, j.get('range').split())) for j in self.joints[1:]])
        self.lower = np.maximum(self.q0[7:]-0.6, rng[:, 0]); self.upper = np.minimum(self.q0[7:]+0.6, rng[:, 1])
        assert np.all(self.lower < self.upper) and np.all(self.q0[7:] >= self.lower) and np.all(self.q0[7:] <= self.upper)
        import mujoco as mj
        assert mj.__version__ == '3.9.0'
        for name in ('control', 'act_dyn', 'act_gain', 'act_bias', 'passive', 'sensor'):
            g = getattr(mj, 'get_mjcb_'+name, None); assert g is None or g() is None
        assert self.calls['model_load'] < self.load_cap
        self.calls['model_load'] += 1
        self.mj = mj; self.m = mj.MjModel.from_xml_path(str(XML))
        assert self.ns['snapshot'](self.m) == self.proof['before']
        self.m.actuator_dynprm[:, 0] = .001; self.m.actuator_dynprm[:, 1] = .0004
        self.m.opt.integrator = mj.mjtIntegrator.mjINT_RK4; self.m.opt.timestep = .000025
        assert self.ns['snapshot'](self.m) == self.proof['after']
        self.floors = set(self.meta['floor_geom_ids']); self.geomleg = {int(k): LEGS.index(v) for k, v in self.meta['geom_map'].items()}
    def forward(self, q, act90=None):
        mj = self.mj; d = mj.MjData(self.m); d.time = float(self.z['time'][0])
        for n in STATE: getattr(d, n)[:] = self.z[n][0]
        for n in OPTIONAL:
            if n in self.z: getattr(d, n)[:] = self.z[n][0]
            else: assert np.asarray(getattr(d, n)).size == 0
        d.qpos[:] = q; d.qvel[:] = 0
        a = np.zeros(90) if act90 is None else act90
        d.act[:] = a; d.ctrl[:] = a; d.qacc_warmstart[:] = 0
        init = {n: np.asarray(getattr(d, n)).copy() for n in (*STATE, *OPTIONAL)}
        mj.mj_forward(self.m, d); self.calls['forward'] += 1
        assert all(np.array_equal(getattr(d, n), init[n]) for n in (*STATE, *OPTIONAL))
        c = self.ns['constraint_record'](mj, self.m, d, 0); c.update(self.ns['moment_record'](self.m, d))
        for n in ('qacc', 'act_dot', 'actuator_force', 'actuator_length', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias',
                  'efc_D', 'efc_aref', 'efc_type', 'efc_frictionloss'):
            c[n] = np.asarray(getattr(d, n)).copy()
        c['contact_dist'] = np.array([v.dist for v in d.contact]); c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
        return c
    def affine(self, c):
        types = c['efc_type']; assert np.all(np.isin(types, [0, 3, 4, 5, 6]))
        f0 = c['efc_D']*c['efc_aref']; f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
        gain = np.empty(90); bias = np.empty(90)
        for j, node in enumerate(self.nodes): _, gain[j], bias[j] = self.ns['evaluate'](c['actuator_length'][j], 0., 0., node)
        moment = c['actuator_moment']; B = moment.T*gain[None, :]; base = moment.T@bias+c['qfrc_passive']-c['qfrc_bias']
        return B, base, f0
    def geometry(self, c, f0):
        depth = 0.; nonfoot = 0.; bearing = set(); margins = []
        for ci, (g1, g2) in enumerate(c['contact_geom']):
            g1 = int(g1); g2 = int(g2); leg = self.geomleg.get(g2 if g1 in self.floors else g1) if ((g1 in self.floors) != (g2 in self.floors)) else None
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
        return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot), bearing_legs=int(len(bearing)),
                    friction_margin=float(min(margins, default=0.)),
                    contact_signature='|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in c['contact_geom']))+f'#nefc{len(c["efc_type"])}')
    def build_qpos(self, qleg, p):
        q = self.q0.copy(); q[7:] = qleg; q[0:2] = p[0:2]; q[2] = p[2]
        dq = quat_from_xyz(p[3], p[4], p[5]); qw = quat_mul(dq, self.q0[3:7]); q[3:7] = qw/np.linalg.norm(qw)
        return q

def main(out, stage):
    assert out.is_absolute() and not out.exists()
    out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    files = [Path(__file__).resolve(), ROOT/'legjoint_release_core.py', ROOT/'test_legjoint_core.py',
             ROOT/'root_newton_core.py', XML, S154/'INPUT_HASHES.json', S154/'init80ms/TRAJECTORY.npz',
             S154/'init80ms/MODEL_PROVENANCE.json', S154/'init80ms/NATIVE_MODEL_ARRAYS.npz',
             S154/'init80ms/COM_CONTACT_METADATA.json', S166/'MAPPING.npz', S177/'ROOT_BALANCE_STATE.npz', S177/'ROOT_RESULT.json']
    old = json.loads((S154/'INPUT_HASHES.json').read_text()); old = old.get('hashes', old)
    hashes = {}
    for p, h in old.items():
        if isinstance(h, str) and len(h) == 64:
            assert sha(p) == h, 'S154 binding changed: '+p
            hashes[p] = h
    hashes.update({str(p): sha(p) for p in files})
    dump(out/'INPUT_HASHES.json', hashes)
    for i, (p, h) in enumerate(sorted(hashes.items())):
        t = out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p, t); assert sha(t) == h
    base177 = json.loads((S177/'ROOT_RESULT.json').read_text())
    p177 = np.array(base177['p'], float); qleg0 = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:].copy()
    dump(out/'PREREG.json', dict(
        task='S178/P1-S2 release leg joints from the S177 root-balanced baseline',
        baseline=dict(root_p=p177.tolist(), leg_joints_from=str(S177/'ROOT_BALANCE_STATE.npz'), root_norm_inf=base177['norm_inf']),
        frozen_ranges=dict(leg_joint='intersection(native XML limits, S154 release +/- 0.6 rad)', lower=None, upper=None,
                           root='x,y within +/-0.30 of release; z within +/-15%; roll,pitch,yaw within +/-0.35 rad (S177 window)'),
        candidate_definition='root residual6<=1e-8 AND 42-row LP minimax<=1e-8 AND all geometry gates pass AND act_dot==0; otherwise diagnostic only',
        reporting='never merged: root residual vector, 42-row minimax + unreachable rows, per-gate geometry; no scalar score, no weighting, no max-combination',
        stageA='six unreachable rows [13,16,20,34,36,41]; 7 joints of the row leg x 9 grid points; one forward per point; row interval and target only; sign-flip estimate from the two scan neighbours of the release point',
        stageB='block hill-climb over the six leg blocks (7 joints each) with inner warm-start root Newton (6-row) and one 42-row LP per candidate; accept only if root_ok and geometry pass and minimax strictly decreases',
        budget=dict(stageA_forward_cap=STAGE_A_CAP, stageB_forward_cap=STAGE_B_CAP, stageB_reserved_for_verification=STAGE_B_RESERVED,
                    stageB_model_load_cap=STAGE_B_LOADS, whole_block_refused_if_insufficient=True),
        verification='keep 200 forwards in reserve for an independent reload check of any candidate',
        no_step=True, no_neural=True, no_training=True, no_dynamics=True, units='native; SI NOT_ESTABLISHED'))
    eng = Engine(out, stage, load_cap=(1 if stage == 'a' else STAGE_B_LOADS))
    dump(out/'FROZEN_RANGES.json', dict(joint_names=eng.names, lower=eng.lower.tolist(), upper=eng.upper.tolist(),
                                        release=eng.q0[7:].tolist(), leg_of_joint=[(i, LEGS[(i)//7]) for i in range(42)]))
    rows42 = np.arange(6, 48)
    def evaluate_pose(qleg, p, env=None, newton_iters=60):
        def cb(pv):
            if eng.calls['forward'] >= STAGE_B_CAP - STAGE_B_RESERVED - 2:
                raise RuntimeError('stage B forward budget exhausted inside inner Newton')
            c = eng.forward(eng.build_qpos(qleg, pv)); B, base, f0 = eng.affine(c)
            bb = -(base+c['efc_J_dense'].T@f0)
            if env is not None: env.ledger['forward'] = eng.calls['forward']
            return dict(residual6=bb[:6].copy(), B=B, base=base, f0=f0, J=c['efc_J_dense'])
        lo = np.array([p177[0]-.3, p177[1]-.3, p177[2]*.85, -.35, -.35, -.35])
        hi = np.array([p177[0]+.3, p177[1]+.3, p177[2]*1.15, .35, .35, .35])
        nr = RootNewton(cb, lo, hi, cap=40, max_iter=4, name='inner')
        try:
            res = nr.solve(p)
        except (RuntimeError, Exception) as exc:
            return dict(root_p=list(p), root_norm=float('inf'), root_ok=False, root_residual6=[0.]*6,
                        root_stop=str(exc), residual6=[0.]*6, minimax=float('inf'), unreachable=[], row_violation=[0.]*42,
                        row_lo=[0.]*42, row_hi=[0.]*42, row_target=[0.]*42, geom_ok=False,
                        geometry=dict(foot_depth=0., nonfoot_depth=0., bearing_legs=0, friction_margin=0., contact_signature='BUDGET'),
                        activation=[0.]*90, qpos=eng.build_qpos(qleg, p).tolist(), qacc_max=None, contact_signature='BUDGET')
        if env is not None:
            env.ledger['root_newton_iterations'] += nr.ledger['newton_iterations']
            env.ledger['root_jacobians'] += nr.ledger['jacobians']
            env.ledger['fd_probes'] += nr.ledger['fd_probes']
            env.ledger['forward'] = eng.calls['forward']
        pr = np.array(res['p'])
        c = eng.forward(eng.build_qpos(qleg, pr)); B, base, f0 = eng.affine(c)
        bb = -(base+c['efc_J_dense'].T@f0); geom = eng.geometry(c, f0)
        A = (B@eng.T)[6:, :]; tgt = bb[6:]
        lo_i, hi_i, viol, reach = row_intervals(A, tgt)
        n = A.shape[1]
        lp = linprog(np.r_[np.zeros(n), 1.], A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                     b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)]*n+[(0, None)], method='highs',
                     options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        assert lp.success
        if env is not None: env.ledger['inner_lp'] += 1; env.ledger['forward'] = eng.calls['forward']
        geom_ok = bool(geom['foot_depth'] <= DEPTH_CAP and geom['nonfoot_depth'] <= 1e-12 and geom['bearing_legs'] >= 3 and geom['friction_margin'] >= -1e-10)
        return dict(root_p=pr.tolist(), root_norm=float(res['norm_inf']), root_ok=bool(res['norm_inf'] <= 1e-8),
                    root_residual6=res['residual6'], root_stop=res['stop'], residual6=bb.tolist(),
                    minimax=float(lp.x[-1]), unreachable=[int(rows42[i]) for i in range(42) if not reach[i]],
                    row_violation=viol.tolist(), row_lo=lo_i.tolist(), row_hi=hi_i.tolist(), row_target=tgt.tolist(),
                    geom_ok=geom_ok, geometry=geom, activation=(eng.T@lp.x[:n]).tolist(),
                    qpos=eng.build_qpos(qleg, pr).tolist(), qacc_max=None, contact_signature=geom['contact_signature'])
    summary = dict(stage=stage, engine_calls=eng.calls)
    if stage in ('a', 'both'):
        # NOTE: S177 reported 42-row subsystem INDICES; the physical DOFs are +6.
        rows6 = [19, 22, 26, 40, 42, 47]
        assert all(6 <= r <= 47 for r in rows6)
        pA = p177
        qA = eng.q0.copy(); qA[7:] = qleg0; qA[0:2] = pA[0:2]; qA[2] = pA[2]
        cA = eng.forward(eng.build_qpos(qleg0, pA)); BA, baseA, f0A = eng.affine(cA)
        bbA = -(baseA+cA['efc_J_dense'].T@f0A); A0 = (BA@eng.T)[6:, :]; tgt0 = bbA[6:]
        lo0, hi0, vio0, rea0 = row_intervals(A0, tgt0)
        dump(out/'STAGE_A_BASELINE_ROWS.json', dict(note='S177 root-balanced baseline, leg joints at release values',
             rows=[dict(dof=int(r), reachable=bool(rea0[int(r)-6]), target=float(tgt0[int(r)-6]),
                        lo=float(lo0[int(r)-6]), hi=float(hi0[int(r)-6])) for r in rows6]))
        table = []
        def leg_name_of(joint_index):
            nm = eng.names[joint_index]
            for pref, leg in (('LFCoxa','lf'),('LMCoxa','lm'),('LHCoxa','lh'),('RFCoxa','rf'),('RMCoxa','rm'),('RHCoxa','rh')):
                if nm.startswith('joint_'+pref): return leg
            return 'unlabeled'
        for r in rows6:
            i = rows42.tolist().index(r)
            leg = (r-6)//7
            joint_ids = list(range(leg*7, leg*7+7))
            leg_label = leg_name_of(joint_ids[0])
            dom = int(np.argmax(np.abs(A0[i])))
            for j in joint_ids:
                ang = np.linspace(eng.lower[j], eng.upper[j], 9)
                cols = []; infos = []
                for a in ang:
                    if eng.calls['forward'] >= STAGE_A_CAP:
                        break
                    qleg = qleg0.copy(); qleg[j] = a
                    cc = eng.forward(eng.build_qpos(qleg, pA)); Bc, basec, f0c = eng.affine(cc)
                    Ac = (Bc@eng.T)[6:, :]
                    t = float(-(basec+cc['efc_J_dense'].T@f0c)[6:][i])
                    l = float(np.minimum(Ac[i], 0).sum()); h = float(np.maximum(Ac[i], 0).sum())
                    cols.append(float(Ac[i, dom])); infos.append(dict(angle=float(a), target=t, lo=l, hi=h, reachable=bool(l-1e-9 <= t <= h+1e-9)))
                est = sign_flip_estimate(cols, ang[:len(cols)], base_index=min(4, len(cols)-1))
                flip_possible = any(x['reachable'] for x in infos)
                table.append(dict(row=r, dof=r, leg=leg_label, leg_index=leg, joint=j, joint_name=eng.names[j],
                                  dominant_column=dom, release_angle=float(qleg0[j]), range=[float(eng.lower[j]), float(eng.upper[j])],
                                  row_reachable_at_some_grid=bool(flip_possible), slope_dm_dq=est['slope'],
                                  angle_needed_for_sign_flip=est['angle_needed'], span_available=est.get('span_available'),
                                  flippable=bool(est['flippable'] and est['angle_needed'] is not None and est['angle_needed'] <= 0.6),
                                  grid=infos))
        with (out/'STAGE_A_SIGNFLIP.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in table[0] if k != 'grid']); w.writeheader()
            for t in table: w.writerow({k: v for k, v in t.items() if k != 'grid'})
        per_row = []
        for r in rows6:
            i = rows42.tolist().index(r)
            sub = [t for t in table if t['row'] == r]
            emp = [t for t in sub if t['row_reachable_at_some_grid']]
            cheapest = min(emp, key=lambda t: abs(t['release_angle'])) if emp else None
            per_row.append(dict(dof=r, leg=sub[0]['leg'], any_joint_makes_row_reachable=bool(emp),
                                best_joint=None if cheapest is None else cheapest['joint'],
                                min_angle_needed=min([t['angle_needed_for_sign_flip'] for t in sub if t['angle_needed_for_sign_flip'] is not None], default=None),
                                baseline=dict(target=float(tgt0[i]), lo=float(lo0[i]), hi=float(hi0[i]), reachable=bool(rea0[i]))))
        dump(out/'STAGE_A_RESULT.json', dict(per_row=per_row, table=table, forward=eng.calls['forward'], model_load=eng.calls['model_load']))
        summary['stage_a'] = dict(forward=eng.calls['forward'], per_row=per_row)
    if stage in ('b', 'both') and eng.calls['forward'] < STAGE_A_CAP:
        env = BlockHillClimb(lambda qleg, e: evaluate_pose(qleg, np.array(p177), e), eng.lower, eng.upper,
                             blocks=[np.arange(i*7, i*7+7) for i in range(6)], cap=STAGE_B_CAP,
                             reserved=STAGE_B_RESERVED, output=out/'STAGE_B_LEDGER.json', name='legblocks')
        res = env.run(qleg0)
        fin = res['final']
        dump(out/'STAGE_B_RESULT.json', dict(stop=res['stop'], sweeps=res['sweeps'], ledger=res['ledger'],
             final={k: v for k, v in fin.items() if k not in ('rows',)}, history=res['history'],
             joint_offset_from_release=[float(a-b) for a, b in zip(res['q'], eng.q0[7:])],
             note='large joint offsets do NOT imply neural reachability; not claimed'))
        if fin['root_ok'] and fin['geom_ok']:
            act = np.array(fin['activation']); q = np.array(fin['qpos'])
            cf = eng.forward(q, act)
            res48 = None
            Bc, basec, f0c = eng.affine(cf); res48 = Bc@act+basec+cf['efc_J_dense'].T@f0c
            g = eng.geometry(cf, f0c)
            # vertical identity: sum of contact normal forces must equal body weight when root z balances
            types = cf['efc_type']; f0c_local = np.where(types == 0, cf['efc_D']*cf['efc_aref'], np.maximum(cf['efc_D']*cf['efc_aref'], 0.))
            normal_sum = 0.0
            for ci in range(len(cf['contact_geom'])):
                addr = int(cf['contact_efc_address'][ci]); dim = int(cf['contact_dim'][ci]); cnt = int(cf['contact_row_count'][ci])
                if addr < 0: continue
                normal_sum += float(f0c_local[addr] if dim == 1 else f0c_local[addr:addr+4].sum())
            weight = float(np.load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass'].sum()*9801.)
            dump(out/'STAGE_B_CONFIRMATION.json', dict(max_abs_residual48=float(abs(res48).max()),
                 max_abs_qacc=float(abs(cf['qacc']).max()), act_dot_all_zero=bool(not np.any(cf['act_dot'])),
                 vertical_identity=dict(total_contact_normal=normal_sum, body_weight=weight, difference=normal_sum-weight,
                                        identity_holds=bool(abs(normal_sum-weight) <= 1e-6*max(weight, 1.))),
                 geometry=g, all48_balanced=bool(abs(res48).max() <= 1e-8)))
            np.savez_compressed(out/'CANDIDATE_STATE.npz', qpos=q, act90=act, res48=res48, qacc=cf['qacc'], act_dot=cf['act_dot'])
        summary['stage_b'] = dict(stop=res['stop'], minimax=fin['minimax'], root_norm=fin['root_norm'],
                                  root_ok=fin['root_ok'], geom_ok=fin['geom_ok'], unreachable=fin['unreachable'])
    summary['engine_calls'] = eng.calls
    dump(out/'SUMMARY.json', summary)
    dump(out/'INPUT_HASHES_AFTER.json', {k: sha(k) for k in hashes})
    print(json.dumps(sanitize(summary)))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path); ap.add_argument('--stage', default='a', choices=['a', 'b', 'both'])
    a = ap.parse_args()
    main(a.output or BASE/'s178-legjoint-release-search'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'), a.stage)
