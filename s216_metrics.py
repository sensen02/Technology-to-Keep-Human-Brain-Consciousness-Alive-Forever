"""S216 Part 0/A/B/C: three metrics, per-variant base persistence, consistent-pairing cone."""
import json
import datetime
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s216-metric-and-cone-consistent')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
DESK = Path('/home/sensen/Desktop/engine修复验收')
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
S215 = DESK / 's215-mu-truth-and-two-blockers/20261025T172134983637Z'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
EPS = 1e-12
CL = dict(model_loads=0, position_stage_calls=0, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0, lp_solves=0)

prereg = dict(round='S216', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              metrics=dict(rowwise_lb='req_r / max same-sign column of row r  (LOWER BOUND ONLY)',
                           lp_smin_leg='inf{s : minimax_leg(s) <= 1e-9} over the joint 7-row leg LP (bisection, 26 iters)',
                           minimax_leg='min t s.t. |A_leg x - req_leg| <= t, 0 <= x <= s'),
              rule='feasibility verdicts may cite ONLY lp_smin_leg and minimax_leg(2); rowwise_lb is a labelled lower bound',
              labels=['TWO_BLOCKERS_AT_CEILING', 'REAL_CONE_FEASIBLE_CONSISTENT', 'REAL_CONE_INFEASIBLE_CONSISTENT'],
              hard_gates=['consistent lp_smin {LF .357447, RF .670430, LM 4.670348, LH 1.332850, RH 1.519474, RM inf}',
                          'consistent minimax(2) {0,0,1.322814,0,1.667540,0}',
                          'mixed minimax(2) {0,0,1.212754,0,1.738366,0} and mixed LM lp_smin 4.448172',
                          'frozen lp_smin LF .353883 RF .761004 ; rowwise 20.704108/17.651270/19.096199',
                          'frozen@s1 cone rho = 0.5098701026551923'],
              budget=dict(model_loads_max=4, position_stage_calls_max=4, constraint_stage_calls_max=2, lp_max=120,
                          forward_calls=0, mj_step_calls=0, training_steps=0, xml_changes=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

z15 = np.load(S215 / 'AUTH_MATRICES.npz')
A_frozen, A_full, A_24 = z15['A_frozen'], z15['A_full'], z15['A_port24']
B_frozen, B_full = z15['B_frozen'], z15['B_full']
req_frozen, req_full, req_24 = z15['req_frozen'], z15['req_full'], z15['req_port24']
T = z15['T']
J_s, f0_s, base_s = z15['J'], z15['f0'], z15['base']
leg_of = [str(x) for x in z15['leg_of']]
row_names = [str(x) for x in z15['row_names']]
pool = z15['actuator_pool']
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
jn = row_names

def solve(Amat, reqv, box):
    n, mm = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm))), np.column_stack((-Amat, -np.ones(mm))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

def leg_rows_cols(Amat, L):
    rows = [i for i in range(42) if leg_of[i] == L]
    cols = [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > EPS
            and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= EPS]
    return rows, cols

def minimax_leg(Amat, reqv, L, s):
    rows, cols = leg_rows_cols(Amat, L)
    return solve(Amat[rows][:, cols], reqv[rows], s)

def lp_smin(Amat, reqv, L, hi=64.0, iters=26):
    rows, cols = leg_rows_cols(Amat, L)
    top = solve(Amat[rows][:, cols], reqv[rows], hi)
    if top is None or top > 1e-9:
        return None
    lo = 1e-6
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = solve(Amat[rows][:, cols], reqv[rows], mid)
        if v is not None and v <= 1e-9:
            hi = mid
        else:
            lo = mid
    return float(hi)

def rowwise(Amat, reqv):
    out = {}
    for i in range(42):
        a = Amat[i, :]
        r = reqv[i]
        if abs(r) <= EPS:
            out[jn[i]] = 0.0
            continue
        same = a[np.sign(a) == np.sign(r)]
        same = same[np.abs(same) > EPS]
        out[jn[i]] = None if same.size == 0 else float(abs(r) / np.max(np.sign(r) * same))
    return out

# ---------------------------------------------------------------- Part 0: three metrics
metrics = {}
for tag, A, rq in (('consistent_full', A_full, req_full), ('mixed_full', A_full, req_frozen),
                   ('frozen', A_frozen, req_frozen)):
    metrics[tag] = {}
    for L in LEGS:
        metrics[tag][L] = dict(rowwise_lb=max([v for k, v in rowwise(A, rq).items()
                                               if leg_of[jn.index(k)] == L and v is not None] or [None])
                               if any(leg_of[jn.index(k)] == L and rowwise(A, rq)[k] is not None for k in jn) else None,
                               lp_smin=lp_smin(A, rq, L),
                               minimax_s1=minimax_leg(A, rq, L, 1.0),
                               minimax_s2=minimax_leg(A, rq, L, 2.0))
cons = metrics['consistent_full']
mix = metrics['mixed_full']
gate = dict(
    consistent_lp_smin=all(abs((cons[L]['lp_smin'] or -1) - e) <= 1e-5 for L, e in
                           (('LF', 0.357447), ('RF', 0.670430), ('LM', 4.670348), ('LH', 1.332850), ('RH', 1.519474)))
    and cons['RM']['lp_smin'] is None,
    consistent_minimax_s2=all(abs((cons[L]['minimax_s2'] or 0) - e) <= 1e-5 for L, e in
                              (('LF', 0.0), ('RF', 0.0), ('LM', 1.322814), ('LH', 0.0), ('RM', 1.667540), ('RH', 0.0))),
    mixed_minimax_s2=all(abs((mix[L]['minimax_s2'] or 0) - e) <= 1e-5 for L, e in
                         (('LM', 1.212754), ('RM', 1.738366))),
    mixed_lm_lp_smin=abs((mix['LM']['lp_smin'] or -1) - 4.448172) <= 1e-5,
    frozen_lp_smin=all(abs((metrics['frozen'][L]['lp_smin'] or -1) - e) <= 1e-5 for L, e in
                       (('LF', 0.353883), ('RF', 0.761004))))
(DIR / 'METRIC_TABLE.json').write_text(json.dumps(dict(metrics=metrics, gates=gate,
                                                       definitions=prereg['metrics']), indent=2, default=str))
with open(DIR / 'METRIC_TABLE.csv', 'w') as f:
    f.write('pairing,leg,rowwise_lb,lp_smin,minimax_s1,minimax_s2\n')
    for tag in metrics:
        for L in LEGS:
            m = metrics[tag][L]
            f.write('%s,%s,%s,%s,%s,%s\n' % (tag, L, m['rowwise_lb'], m['lp_smin'], m['minimax_s1'], m['minimax_s2']))
blockers = dict(
    LM=dict(kind='AMPLITUDE', lp_smin=cons['LM']['lp_smin'], minimax_s2=cons['LM']['minimax_s2'],
            note='s_min 4.670348 > 2, so LM is infeasible at the shipped-equivalent ceiling (consistent pairing)'),
    RM=dict(kind='DIRECTION', lp_smin=cons['RM']['lp_smin'], minimax_s2=cons['RM']['minimax_s2'],
            unreachable_rows=[k for k, v in rowwise(A_full, req_full).items() if v is None],
            note='row 36 has no same-sign column at all; no box can fix it'),
    label='TWO_BLOCKERS_AT_CEILING')
(DIR / 'BLOCKER_TABLE.json').write_text(json.dumps(blockers, indent=2, default=str))

# ---------------------------------------------------------------- Part A: per-variant base
def variant_base(path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    qrel = np.asarray(np.load(S177)['qpos'], float)
    d = mujoco.MjData(m); d.qpos[:] = qrel
    d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    nodes = list(ET.parse(str(path)).getroot().find('actuator'))
    arm = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownnz[a]); adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            arm[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    def fl(l, lo, hi):
        if l < lo or l > hi:
            return 0.0
        aa = (lo + 1) * 0.5; bb = (1 + hi) * 0.5
        if l <= aa:
            return 0.5 * ((l - lo) / max(1e-15, aa - lo)) ** 2
        if l <= 1:
            return 1 - 0.5 * ((1 - l) / max(1e-15, 1 - aa)) ** 2
        if l <= bb:
            return 1 - 0.5 * ((l - 1) / max(1e-15, bb - 1)) ** 2
        return 0.5 * ((hi - l) / max(1e-15, hi - bb)) ** 2
    def gain_of(length, node):
        g = [float(x) for x in node.get('gainprm').split()]
        lr = [float(x) for x in node.get('lengthrange').split()]
        l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
        l = g[0] + (length - lr[0]) / max(1e-15, l0)
        return -g[2] * fl(l, g[4], g[5])
    g = np.array([gain_of(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    momT = arm.T
    base = momT @ np.asarray(d.actuator_force, float) + np.asarray(d.qfrc_passive, float) - np.asarray(d.qfrc_bias, float)
    return base, arm, g, d

bases = {'frozen': variant_base(FROZEN)}
variant_paths = {'port24': S215 / 'VARIANT_port24.xml', 'full': S215 / 'VARIANT_full_repair.xml'}
for tag, p in variant_paths.items():
    bases[tag] = variant_base(p)
asserts = {}
for tag in ('frozen', 'port24', 'full'):
    b, arm, g, d = bases[tag]
    rq = (-(b + J_s.T @ f0_s))[6:]
    ref = {'frozen': req_frozen, 'port24': req_24, 'full': req_full}[tag]
    asserts[tag] = dict(max_abs_delta=float(np.max(np.abs(rq - ref))), identity_ok=bool(np.max(np.abs(rq - ref)) <= 1e-12))
np.savez_compressed(DIR / 'AUTH_MATRICES_V2.npz',
                    A_frozen=A_frozen, A_port24=A_24, A_full=A_full,
                    B_frozen=B_frozen, B_port24=z15['B_port24'], B_full=B_full,
                    base_frozen=bases['frozen'][0], base_port24=bases['port24'][0], base_full=bases['full'][0],
                    moment_frozen=bases['frozen'][1], moment_port24=bases['port24'][1], moment_full=bases['full'][1],
                    gain_frozen=bases['frozen'][2], gain_port24=bases['port24'][2], gain_full=bases['full'][2],
                    req_frozen=req_frozen, req_port24=req_24, req_full=req_full,
                    f0=f0_s, J=J_s, T=T, leg_of=np.array(leg_of), row_names=np.array(jn), actuator_pool=pool)

# ---------------------------------------------------------------- Part B: cone, consistent pairing
def cone_structure(path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m); d.qpos[:] = np.asarray(np.load(S177)['qpos'], float); d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    mujoco.mj_fwdConstraint(m, d); CL['constraint_stage_calls'] += 1
    nefc = int(d.nefc)
    et = np.asarray(d.efc_type[:nefc]).copy()
    eid = np.asarray(d.efc_id[:nefc]).copy()
    eJ = np.asarray(d.efc_J).reshape(nefc, m.nv).copy()
    groups = {}
    for r in range(nefc):
        if int(et[r]) in (5, 6, 7):
            groups.setdefault(int(eid[r]), []).append(r)
    gfr = np.asarray(m.geom_friction)
    csets = []
    for ci, rows in sorted(groups.items()):
        g1, g2 = int(d.contact[ci].geom1), int(d.contact[ci].geom2)
        csets.append((min(float(gfr[g1, 0]), float(gfr[g2, 0])), rows))
    return dict(nefc=nefc, ncon=int(d.ncon), efc_type=et, efc_id=eid, efc_J=eJ, csets=csets, geom_friction=gfr)

cf = cone_structure(FROZEN)
cv = cone_structure(S215 / 'VARIANT_full_repair.xml')
same = bool(np.array_equal(cf['efc_type'], cv['efc_type']) and np.array_equal(cf['efc_id'], cv['efc_id'])
            and np.max(np.abs(cf['efc_J'] - cv['efc_J'])) <= 1e-12)
np.savez_compressed(DIR / 'CONE_STRUCTURE.npz', nefc=cf['nefc'], ncon=cf['ncon'], efc_type=cf['efc_type'],
                    efc_id=cf['efc_id'], efc_J=cf['efc_J'], geom_friction=cf['geom_friction'],
                    f0=f0_s, J=J_s, mu=np.array([c[0] for c in cf['csets']]),
                    row_groups=np.array([len(c[1]) for c in cf['csets']]))
cone_meta = dict(nefc=cf['nefc'], ncon=cf['ncon'], n_groups=len(cf['csets']),
                 mu=list({c[0] for c in cf['csets']}), same_constraints_in_variant=same,
                 f0_norm=float(np.linalg.norm(f0_s)),
                 f0_norm_matches_expected=bool(abs(float(np.linalg.norm(f0_s)) - 3.2099080163009863) <= 1e-12),
                 friction_attr_count=int(FROZEN.read_text().count('friction=')))

def cone_solve(cc, Amat, reqv, box, fix_f0=False):
    n_cols = Amat.shape[1]
    n_f = cc['nefc']
    n = n_cols + (0 if fix_f0 else n_f)
    N = n + 1
    Jj = cc['efc_J'][:, 6:]
    rhs = reqv + (Jj.T @ f0_s) if fix_f0 else reqv
    Aub, bub = [], []
    for i in range(42):
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:n_cols] = sgn * Amat[i]
            if not fix_f0:
                row[n_cols:n] = -sgn * Jj.T[i]
            row[n] = -1.0
            Aub.append(row); bub.append(sgn * rhs[i])
    if not fix_f0:
        for mu, rows in cc['csets']:
            normal, tans = rows[0], rows[1:]
            for t in tans:
                for s in (1.0, -1.0):
                    row = np.zeros(N)
                    row[n_cols + t] = s
                    row[n_cols + normal] = -mu / np.sqrt(len(tans))
                    Aub.append(row); bub.append(0.0)
        for mu, rows in cc['csets']:
            row = np.zeros(N); row[n_cols + rows[0]] = -1.0
            Aub.append(row); bub.append(0.0)
        bnds = [(0, box)] * n_cols + [(None, None)] * n_f + [(0, None)]
    else:
        bnds = [(0, box)] * n_cols + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

cone_res = {}
for tag, cc, A, rq in (('frozen', cf, A_frozen, req_frozen), ('full', cv, A_full, req_full)):
    for box in (1.0, 2.0):
        cone_res['%s_s%g' % (tag, box)] = dict(
            joint_f=cone_solve(cc, A, rq, box, False), fixed_f0=cone_solve(cc, A, rq, box, True))
(DIR / 'REAL_CONE_CONSISTENT.json').write_text(json.dumps(dict(cone=cone_meta, results=cone_res,
    gate_frozen_s1_equals_0p5098701026551923=bool(abs((cone_res['frozen_s1']['joint_f'] or -1) - 0.5098701026551923) <= 1e-9),
    label=('REAL_CONE_FEASIBLE_CONSISTENT' if (cone_res['full_s2']['joint_f'] or 1) <= 1e-9
           else 'REAL_CONE_INFEASIBLE_CONSISTENT'),
    note='S214 REAL_CONE_AT_CEILING.json full rows (0.0/0.0) have no reproducible req pairing on record; this round '
         'supersedes it for the record and S214 archival files are untouched'), indent=2, default=str))

# ---------------------------------------------------------------- Part C: row 36
r36 = 36
rw = {tag: rowwise(A, rq) for tag, A, rq in (('frozen', A_frozen, req_frozen), ('full', A_full, req_full))}
maxpos = {}
for tag, A in (('frozen', A_frozen), ('full', A_full)):
    a = A[r36, :]
    maxpos[tag] = float(a[a > 0].max()) if (a > 0).any() else 0.0
row36 = dict(frozen=dict(req=float(req_frozen[r36]), maxpos=maxpos['frozen'], unreachable=rw['frozen'][jn[r36]] is None),
             full=dict(req=float(req_full[r36]), maxpos=maxpos['full'], unreachable=rw['full'][jn[r36]] is None),
             ratio_of_maxpos=float(maxpos['full'] / maxpos['frozen']) if maxpos['frozen'] else None,
             both_below_eps=bool(maxpos['frozen'] < EPS and maxpos['full'] < EPS),
             dof_name=jn[r36], leg=leg_of[r36])
mirror = json.loads((REPO / 'outputs/muscles_six_legs/mirror_decisions.json').read_text())
cands = [e for e in mirror['non_one_to_one_mappings'] if 'pleural_remotor' in e['site']]
(DIR / 'ROW36_FULL.json').write_text(json.dumps(dict(row36=row36, mirror_candidates=cands,
    coverage_statement='the mirror table has pleural_remotor_and_abductor entries only on other legs/sides; this is a DATA '
                       'COVERAGE fact (no right-mid annotation), not a body fact. The generator script was NOT executed.'), indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), gates=gate, blockers=blockers['label'],
                      cons={L: (None if cons[L]['lp_smin'] is None else round(cons[L]['lp_smin'], 6)) for L in LEGS},
                      cons_mm2={L: round(cons[L]['minimax_s2'] or 0, 6) for L in LEGS},
                      mix_mm2={L: round(mix[L]['minimax_s2'] or 0, 6) for L in LEGS},
                      cone=cone_res, row36=row36, base_asserts=asserts, cone_meta=cone_meta, calls=CL), indent=1, default=str))
