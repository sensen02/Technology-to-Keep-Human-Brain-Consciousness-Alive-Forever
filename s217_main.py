"""S217 MAIN_SCRIPT: base rebuild (position+velocity+actuation), corrected metrics, cone closure."""
import json
import datetime
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
DESK = Path('/home/sensen/Desktop/engine修复验收')
S215 = DESK / 's215-mu-truth-and-two-blockers/20261025T172134983637Z'
S216 = DESK / 's216-metric-and-cone-consistent/20261025T172410020231Z'
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
EPS = 1e-12
CL = dict(model_loads=0, fwd_position=0, fwd_velocity=0, fwd_actuation=0, fwd_constraint=0,
          mj_forward=0, mj_step=0, lp_solves=0)

prereg = dict(round='S217', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              why_new_stages='base = moment.T@bias_force + qfrc_passive - qfrc_bias, and qfrc_bias is only computed by '
                             'mj_fwdVelocity; S216 read it after mj_fwdPosition alone and got an all-zero base',
              invariants='qpos/qvel/act/ctrl are asserted bitwise unchanged across the three stage calls; mj_forward and '
                         'mj_step are never called',
              hard_gates=dict(base_frozen_vs_b177=1e-12,
                              B_matches_s216=0.0,
                              full_global=dict(s1=1.8401142035682785, s2=1.7383659885871001),
                              full_lp_smin=dict(LF=0.353883, RF=0.761004, LM=4.448172, LH=1.266683, RH=1.429992, RM=None),
                              full_minimax2=dict(LM=1.212754, RM=1.738366),
                              cone_eight_cells=0.05932941438013683,
                              cone_dropout_row36=0.0,
                              sigma_lambda=24.4464, f_fixed_minimax=3.308620867791373),
              labels=['S216_CONSISTENT_PAIRING_WAS_BASE_ZERO', 'TWO_BLOCKERS_AT_CEILING',
                      'REAL_CONE_INFEASIBLE', 'CONE_PARAMETERIZATION_CORRECTED'],
              budget=dict(model_loads_max=4, fwd_position_max=4, fwd_velocity_max=4, fwd_actuation_max=4,
                          mj_forward=0, mj_step=0, lp_max=200, training_steps=0, xml_changes=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

z16 = np.load(S216 / 'AUTH_MATRICES_V2.npz')
T = z16['T']
J_s = z16['J']
f0_s = z16['f0']
leg_of = [str(x) for x in z16['leg_of']]
jn = [str(x) for x in z16['row_names']]
pool = z16['actuator_pool']
b177 = np.asarray(np.load(S177)['base'], float)
qrel = np.asarray(np.load(S177)['qpos'], float)
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

# ---------------------------------------------------------------- Part 0: base=0 defect
norms_old = {k: float(np.linalg.norm(z16['base_' + k])) for k in ('frozen', 'port24', 'full')}
req_id_old = {}
for k, A in (('frozen', 'req_frozen'), ('port24', 'req_port24'), ('full', 'req_full')):
    recomputed = (-(z16['base_' + k] + J_s.T @ f0_s))[6:]
    req_id_old[k] = float(np.max(np.abs(recomputed - z16[A])))
nreq_delta = float(np.max(np.abs(z16['req_port24'] - z16['req_full'])))
(DIR / 'BASE_ZERO_DEFECT.json').write_text(json.dumps(dict(
    base_norms=norms_old, all_zero=bool(all(v == 0.0 for v in norms_old.values())),
    req_identity_maxabs=req_id_old,
    port24_equals_full_req_maxabs=nreq_delta,
    root_cause='base = moment.T@bias_force + qfrc_passive - qfrc_bias (run_root_newton_search.py:140 region); qfrc_bias is '
               'computed only by mj_fwdVelocity, so after mj_fwdPosition alone it is still zero and the base came out as 0',
    consequence='the port24/full "own req" lost the entire base term, so their req identity held trivially; the frozen branch '
                'differed by 0.3745 because it kept S177 base'), indent=2, default=str))

# ---------------------------------------------------------------- Part A: rebuild base per variant
def fl(l, lo, hi):
    if l < lo or l > hi:
        return 0.0
    a = (lo + 1) * 0.5
    b = (1 + hi) * 0.5
    if l <= a:
        return 0.5 * ((l - lo) / max(1e-15, a - lo)) ** 2
    if l <= 1:
        return 1 - 0.5 * ((1 - l) / max(1e-15, 1 - a)) ** 2
    if l <= b:
        return 1 - 0.5 * ((l - 1) / max(1e-15, b - 1)) ** 2
    return 0.5 * ((hi - l) / max(1e-15, hi - b)) ** 2

def gain_from_node(length, node):
    g = [float(x) for x in node.get('gainprm').split()]
    lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    return -g[2] * fl(l, g[4], g[5])

def stages(path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m)
    d.qpos[:] = qrel; d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = 0.0
    q0 = d.qpos.copy(); v0 = d.qvel.copy(); a0 = d.act.copy(); c0 = d.ctrl.copy()
    mujoco.mj_fwdPosition(m, d); CL['fwd_position'] += 1
    mom = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownnz[a]); adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            mom[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    mujoco.mj_fwdVelocity(m, d); CL['fwd_velocity'] += 1
    mujoco.mj_fwdActuation(m, d); CL['fwd_actuation'] += 1
    force0 = np.asarray(d.actuator_force, float).copy()
    passive = np.asarray(d.qfrc_passive, float).copy()
    bias = np.asarray(d.qfrc_bias, float).copy()
    nodes = list(ET.parse(str(path)).getroot().find('actuator'))
    gain = np.array([gain_from_node(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    base = mom.T @ force0 + passive - bias
    unchanged = dict(qpos=bool(np.array_equal(q0, d.qpos)), qvel=bool(np.array_equal(v0, d.qvel)),
                     act=bool(np.array_equal(a0, d.act)), ctrl=bool(np.array_equal(c0, d.ctrl)))
    return dict(m=m, d=d, mom=mom, gain=gain, base=base, force0=force0, passive=passive,
                bias=bias, unchanged=unchanged, B=mom.T * gain[None, :])

variants = {'frozen': stages(FROZEN),
            'port24': stages(S215 / 'VARIANT_port24.xml'),
            'full': stages(S215 / 'VARIANT_full_repair.xml')}
base_check = {}
for tag, v in variants.items():
    base_check[tag] = dict(norm=float(np.linalg.norm(v['base'])),
                           B_matches_s216=float(np.max(np.abs(v['B'] - z16['B_' + tag]))),
                           unchanged=v['unchanged'])
base_check['frozen']['vs_b177'] = float(np.max(np.abs(variants['frozen']['base'] - b177)))
(DIR / 'BASE_REBUILD.json').write_text(json.dumps(base_check, indent=2, default=str))

A = {tag: (v['B'] @ T)[6:, :] for tag, v in variants.items()}
A48 = {tag: (v['B'] @ T) for tag, v in variants.items()}
rv48 = {tag: (-(v['base'] + J_s.T @ f0_s)) for tag, v in variants.items()}
reqv = {tag: (-(v['base'] + J_s.T @ f0_s))[6:] for tag, v in variants.items()}
req_delta = {tag: float(np.max(np.abs(reqv[tag] - reqv['frozen']))) for tag in reqv}
np.savez_compressed(DIR / 'AUTH_MATRICES_V3.npz',
                    **{'A_' + k: v for k, v in A.items()}, **{'B_' + k: variants[k]['B'] for k in variants},
                    **{'base_' + k: variants[k]['base'] for k in variants},
                    **{'req_' + k: reqv[k] for k in variants},
                    **{'moment_' + k: variants[k]['mom'] for k in variants},
                    **{'gain_' + k: variants[k]['gain'] for k in variants},
                    **{'force0_' + k: variants[k]['force0'] for k in variants},
                    **{'bias_' + k: variants[k]['bias'] for k in variants},
                    **{'passive_' + k: variants[k]['passive'] for k in variants},
                    f0=f0_s, J=J_s, T=T, leg_of=np.array(leg_of), row_names=np.array(jn), actuator_pool=pool)

def solve(Amat, rv, box, rows=None):
    if rows is not None:
        Amat, rv = Amat[rows], rv[rows]
    n, mm = Amat.shape[1], rv.size
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm))), np.column_stack((-Amat, -np.ones(mm))))),
                 b_ub=np.r_[rv, -rv], bounds=[(0, box)] * n + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

def leg_cols(Amat, L):
    rows = [i for i in range(42) if leg_of[i] == L]
    return [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > EPS
            and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= EPS]

def minimax_leg(tag, L, box):
    rows, cols = [i for i in range(42) if leg_of[i] == L], leg_cols(A[tag], L)
    return solve(A[tag][rows][:, cols], reqv[tag][rows], box)

def lp_smin(tag, L, hi=64.0, iters=30):
    rows, cols = [i for i in range(42) if leg_of[i] == L], leg_cols(A[tag], L)
    top = solve(A[tag][rows][:, cols], reqv[tag][rows], hi)
    if top is None or top > 1e-9:
        return None
    lo = 1e-6
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = solve(A[tag][rows][:, cols], reqv[tag][rows], mid)
        if v is not None and v <= 1e-9:
            hi = mid
        else:
            lo = mid
    return float(hi)

def rowwise(tag):
    out = {}
    for i in range(42):
        a, r = A[tag][i, :], reqv[tag][i]
        if abs(r) <= EPS:
            out[jn[i]] = 0.0
            continue
        same = a[np.sign(a) == np.sign(r)]
        same = same[np.abs(same) > EPS]
        out[jn[i]] = None if same.size == 0 else float(abs(r) / np.max(np.sign(r) * same))
    return out

metrics = {}
for tag in variants:
    metrics[tag] = dict(global_s1=solve(A[tag], reqv[tag], 1.0), global_s2=solve(A[tag], reqv[tag], 2.0),
                        rowwise=rowwise(tag),
                        per_leg={L: dict(lp_smin=lp_smin(tag, L), minimax_s1=minimax_leg(tag, L, 1.0),
                                         minimax_s2=minimax_leg(tag, L, 2.0)) for L in LEGS})
(DIR / 'METRIC_V3.json').write_text(json.dumps(dict(metrics=metrics, req_delta=req_delta,
                                                    base_check=base_check), indent=2, default=str))
with open(DIR / 'METRIC_V3.csv', 'w') as f:
    f.write('variant,leg,rowwise_lb,lp_smin,minimax_s1,minimax_s2\n')
    for tag in metrics:
        for L in LEGS:
            m = metrics[tag]['per_leg'][L]
            lb = max([v for k, v in metrics[tag]['rowwise'].items()
                      if leg_of[jn.index(k)] == L and v is not None] or [0])
            f.write('%s,%s,%s,%s,%s,%s\n' % (tag, L, lb, m['lp_smin'], m['minimax_s1'], m['minimax_s2']))
gate_metrics = dict(
    full_global_s1=bool(abs(metrics['full']['global_s1'] - 1.8401142035682785) <= 1e-9),
    full_global_s2=bool(abs(metrics['full']['global_s2'] - 1.7383659885871001) <= 1e-9),
    full_lp_smin=all(abs((metrics['full']['per_leg'][L]['lp_smin'] or -1) - e) <= 1e-5 for L, e in
                     (('LF', 0.353883), ('RF', 0.761004), ('LM', 4.448172), ('LH', 1.266683), ('RH', 1.429992)))
    and metrics['full']['per_leg']['RM']['lp_smin'] is None,
    full_minimax2=all(abs((metrics['full']['per_leg'][L]['minimax_s2'] or 0) - e) <= 1e-5 for L, e in
                      (('LM', 1.212754), ('RM', 1.738366))),
    base_frozen_vs_b177=bool(base_check['frozen']['vs_b177'] <= 1e-12),
    B_unchanged=bool(all(base_check[t]['B_matches_s216'] == 0.0 for t in ('frozen', 'port24', 'full'))),
    state_unchanged=bool(all(v['unchanged'][k] for v in variants.values() for k in ('qpos', 'qvel', 'act', 'ctrl'))))

# ---------------------------------------------------------------- Part B: cone closure
def cone_param(path):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m)
    d.qpos[:] = qrel; d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['fwd_position'] += 1
    mujoco.mj_fwdVelocity(m, d); CL['fwd_velocity'] += 1
    mujoco.mj_fwdConstraint(m, d); CL['fwd_constraint'] += 1
    nefc = int(d.nefc)
    et = np.asarray(d.efc_type[:nefc]).copy()
    eid = np.asarray(d.efc_id[:nefc]).copy()
    eJ = np.asarray(d.efc_J).reshape(nefc, m.nv).copy()
    return dict(m=m, nefc=nefc, ncon=int(d.ncon), efc_type=et, efc_id=eid, efc_J=eJ,
                cone_opt=int(m.opt.cone), condim=np.asarray(m.geom_condim).copy(),
                gfr=np.asarray(m.geom_friction).copy())

cp = cone_param(FROZEN)
cpv = cone_param(S215 / 'VARIANT_full_repair.xml')
# decomposition: 4 rows per contact -> (r0+r1)/2 = (r2+r3)/2 = J_n ; differences are J_t
per = {}
groups = {}
for r in range(cp['nefc']):
    groups.setdefault(int(cp['efc_id'][r]), []).append(r)
decomp = {}
for ci, rows in groups.items():
    if len(rows) == 4:
        r0, r1, r2, r3 = rows
        Jn1 = (cp['efc_J'][r0] + cp['efc_J'][r1]) / 2
        Jn2 = (cp['efc_J'][r2] + cp['efc_J'][r3]) / 2
        decomp[ci] = dict(Jn_diff=float(np.max(np.abs(Jn1 - Jn2))),
                          Jt1_norm=float(np.linalg.norm((cp['efc_J'][r0] - cp['efc_J'][r1]) / 2)),
                          Jt2_norm=float(np.linalg.norm((cp['efc_J'][r2] - cp['efc_J'][r3]) / 2)))
groups_ok = len(groups) == cp['ncon'] == 16 and all(len(v) == 4 for v in groups.values())
Jn_diff_max = max(v['Jn_diff'] for v in decomp.values()) if decomp else None
np.savez_compressed(DIR / 'CONE_STRUCTURE_V2.npz', nefc=cp['nefc'], ncon=cp['ncon'],
                    efc_type=cp['efc_type'], efc_id=cp['efc_id'], efc_J=cp['efc_J'],
                    cone_opt=cp['cone_opt'], condim=cp['condim'], geom_friction=cp['gfr'],
                    f0=f0_s, J=J_s, row_groups=np.array([len(v) for v in groups.values()]))

def build_Jmu(mu):
    """rebuild the pyramidal rows at friction coefficient mu from the 4-row decomposition."""
    rows = []
    for ci, rr in groups.items():
        if len(rr) != 4:
            continue
        r0, r1, r2, r3 = rr
        Jn = (cp['efc_J'][r0] + cp['efc_J'][r1]) / 2
        Jt1 = (cp['efc_J'][r0] - cp['efc_J'][r1]) / 2
        Jt2 = (cp['efc_J'][r2] - cp['efc_J'][r3]) / 2
        rows.extend([Jn + mu * Jt1, Jn - mu * Jt1, Jn + mu * Jt2, Jn - mu * Jt2])
    return np.array(rows)


def cone_lp(tag, box, nrows, drop_row36=False, mu=1.0, free_lam=True):
    Amat = A48[tag] if nrows == 48 else A[tag]
    rv = (rv48[tag] if nrows == 48 else reqv[tag]).copy()
    Jmu = build_Jmu(mu)                       # (64, 48)
    Jmap = Jmu[:, 6:] if nrows == 42 else Jmu
    n_cols = Amat.shape[1]
    n_f = Jmu.shape[0]
    n = n_cols + (n_f if free_lam else 0)
    N = n + 1
    if not free_lam:
        rhs = rv + (Jmap.T @ f0_s)            # A x = req + J^T f0  (lambda pinned to f0)
    else:
        rhs = rv
    Aub, bub = [], []
    rows = [i for i in range(nrows) if not (drop_row36 and i == 36)]
    for i in rows:
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:n_cols] = sgn * Amat[i]
            if free_lam:
                row[n_cols:n] = -sgn * Jmap.T[i]
            row[n] = -1.0
            Aub.append(row); bub.append(sgn * rhs[i])
    if free_lam:
        for r in range(n_f):
            row = np.zeros(N); row[n_cols + r] = -1.0
            Aub.append(row); bub.append(0.0)
        bnds = [(0, box)] * n_cols + [(None, None)] * n_f + [(0, None)]
    else:
        bnds = [(0, box)] * n_cols + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None), (lp.x[n_cols:n] if (free_lam and lp.success) else None)


eight = {}
for tag in ('frozen', 'full'):
    for box in (1.0, 2.0):
        for nrows in (42, 48):
            eight['%s_box%g_%drows' % (tag, box, nrows)] = cone_lp(tag, box, nrows, mu=1.0)[0]
dropout = {}
for tag in ('frozen', 'full'):
    for nrows in (42, 48):
        dropout['%s_%drows' % (tag, nrows)] = dict(with_row36=cone_lp(tag, 1.0, nrows)[0],
                                                   without_row36=cone_lp(tag, 1.0, nrows, drop_row36=True)[0])
leg_cone = {}
for L in LEGS:
    rows = [i for i in range(42) if leg_of[i] == L]
    leg_cone[L] = cone_lp('full', 1.0, 42, drop_row36=False)[0] if False else None
mu_curve = {}
for mu in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    mu_curve['%g' % mu] = cone_lp('frozen', 1.0, 42, mu=mu)[0]
f_fixed = cone_lp('frozen', 1.0, 48, free_lam=False, mu=1.0)[0]
lam_free, lam = cone_lp('frozen', 1.0, 48, mu=1.0)
sigma_lambda = float(np.sum(lam)) if lam is not None else None
body_weight = float(-b177[2])
cone_doc = dict(parameterization=dict(cone=mujoco.mjtCone(cp['cone_opt']).name, condim_unique=sorted({int(x) for x in cp['condim']}),
                                      nefc=cp['nefc'], ncon=cp['ncon'], groups_ok=bool(groups_ok),
                                      four_rows_per_contact=bool(groups_ok),
                                      Jn_equals_maxdiff=Jn_diff_max,
                                      statement='each contact contributes four PYRAMIDAL rows: (r0+r1)/2 = (r2+r3)/2 = J_n and the '
                                                'two differences are the two tangential directions; at mu=1 this reproduces the live '
                                                'efc_J. The constraint is therefore simply lambda >= 0 on all 64 rows.'),
                eight_cells=eight, dropout=dropout, mu_curve=mu_curve,
                f_fixed_minimax=f_fixed, lambda_free_minimax=lam_free, sigma_lambda=sigma_lambda, body_weight=body_weight,
                sigma_matches_weight=bool(sigma_lambda is not None and abs(sigma_lambda - body_weight) <= 1e-3),
                drop_out_row36_zero=bool(abs((dropout['frozen_42rows']['without_row36'] or 1)) <= 1e-12),
                gates=dict(eight_cells_all=bool(all(abs((v or -1) - 0.05932941438013683) <= 1e-9 for v in eight.values())),
                           f_fixed_reproduces_3_3086=bool(abs((f_fixed or 0) - 3.308620867791373) <= 1e-9)),
                labels=['CONE_PARAMETERIZATION_CORRECTED', 'REAL_CONE_INFEASIBLE'],
                withdrawn='S216 REAL_CONE_FEASIBLE_CONSISTENT (both columns were computed with a wrong cone parameterization and a '
                          'zero base); S208 fixed_f0 column was actually the "contact term deleted" value')
(DIR / 'CONE_CLOSURE.json').write_text(json.dumps(cone_doc, indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), base_norms=norms_old, base_check=base_check, req_delta=req_delta,
                      gate_metrics=gate_metrics,
                      full_metrics=dict(global_s1=metrics['full']['global_s1'], global_s2=metrics['full']['global_s2'],
                                        lp_smin={L: (None if metrics['full']['per_leg'][L]['lp_smin'] is None
                                                     else round(metrics['full']['per_leg'][L]['lp_smin'], 6)) for L in LEGS},
                                        minimax2={L: round(metrics['full']['per_leg'][L]['minimax_s2'] or 0, 6) for L in LEGS}),
                      cone=dict(eight=eight, dropout=dropout, mu=mu_curve, f_fixed=f_fixed, sigma_lambda=sigma_lambda,
                                weight=body_weight, gates=cone_doc['gates'], groups_ok=groups_ok, Jn=Jn_diff_max),
                      calls=CL), indent=1, default=str))
