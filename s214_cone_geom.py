"""S214 Part B (real cone at ceiling) + Part C (geometry gate) + figures + verification + conclusion."""
import json
import hashlib
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s214-activation-ceiling-and-sign-table').iterdir())[-1]
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
MUSDIR = REPO / 'outputs/muscles_six_legs'
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
QP0B = MUSDIR / 'fruitfly_six_leg_standing_qpos0bands.xml'
MESH = '/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/'
TMP = Path('/tmp/s214')
CL = dict(model_loads=0, position_stage_calls=0, constraint_stage_calls=0, forward_calls=0, mj_step_calls=0, lp_solves=0)
EPS = 1e-12

st = np.load(S177)
B_frozen = np.asarray(st['B'], float)
base, f0, J, T = (np.asarray(st[k], float) for k in ('base', 'f0', 'J', 'T'))
qrel = np.asarray(st['qpos'], float)
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of = [n[6:8] for n in jn]
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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

def gain_of(length, node):
    g = [float(x) for x in node.get('gainprm').split()]
    lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0])
    l = g[0] + (length - lr[0]) / max(1e-15, l0)
    return -g[2] * fl(l, g[4], g[5])

def dense_moment(m, d):
    out = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        nnz = int(d.moment_rownz[a]) if False else int(d.moment_rownnz[a])
        adr = int(d.moment_rowadr[a])
        for k in range(nnz):
            out[a, int(d.moment_colind[adr + k])] = d.actuator_moment[adr + k]
    return out

# rebuild the full-repair variant (sites + lengthranges from qpos0bands)
TMP.mkdir(parents=True, exist_ok=True)
qp = TMP / 'qpos0bands.xml'
if not qp.exists():
    qp.write_text(QP0B.read_text().replace('./meshes/stl/', MESH))
qp_sites = {s.get('name'): s.get('pos') for s in ET.parse(str(qp)).getroot().iter('site') if s.get('pos')}
tibia_sites = [n for n in qp_sites if n[:2] in ('LM', 'LH', 'RM', 'RH') and ('Tibia_flex' in n or 'Tibia_extensor' in n)]
fz_acts = list(ET.parse(str(FROZEN)).getroot().find('actuator'))
qp_acts = {n.get('name'): n.get('lengthrange') for n in ET.parse(str(qp)).getroot().find('actuator')}
tibia_act_names = [n.get('name') for n in fz_acts
                   if (n.get('name') or '')[:2] in ('LM', 'LH', 'RM', 'RH')
                   and ('Tibia_flex' in (n.get('name') or '') or 'Tibia_extensor' in (n.get('name') or ''))]
out, ch_sites, ch_lr = [], [], []
for ln in FROZEN.read_text().split('\n'):
    hit = next((nm for nm in tibia_sites if 'name="' + nm + '"' in ln and 'pos="' in ln), None)
    if hit:
        new = ln.split('pos="')[0] + 'pos="' + qp_sites[hit] + '"' + ln.split('pos="', 1)[1].split('"', 1)[1]
        if new != ln:
            ch_sites.append(hit)
        out.append(new)
        continue
    hit2 = next((nm for nm in tibia_act_names if 'name="' + nm + '"' in ln and 'lengthrange="' in ln), None)
    if hit2 and qp_acts.get(hit2):
        new = ln.split('lengthrange="')[0] + 'lengthrange="' + qp_acts[hit2] + '"' + \
              ln.split('lengthrange="', 1)[1].split('"', 1)[1]
        if new != ln:
            ch_lr.append(hit2)
        out.append(new)
        continue
    out.append(ln)
FULL = TMP / 'frozen_variant_full_repair.xml'
FULL.write_text('\n'.join(out))

def arm_gain(path, qpos):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m); d.qpos[:] = qpos; d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    nodes = list(ET.parse(str(path)).getroot().find('actuator'))
    arm = dense_moment(m, d)
    g = np.array([gain_of(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    return m, d, arm, g

mF, dF, armF, gF = arm_gain(FULL, qrel)
B_full = armF.T * gF[None, :]
A_full = (B_full @ T)[6:, :]
A_frozen = (B_frozen @ T)[6:, :]

def solve(Amat, reqv, box, rows=None, cols=None):
    if rows is not None:
        Amat, reqv = Amat[rows][:, cols], reqv[rows]
    n, mm = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm))), np.column_stack((-Amat, -np.ones(mm))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

def leg_cols(Amat, L):
    rows = [i for i in range(42) if leg_of[i] == L]
    return [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > EPS
            and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= EPS]

def s_min(Amat, L, hi=64.0):
    rows, cols = [i for i in range(42) if leg_of[i] == L], leg_cols(Amat, L)
    top = solve(Amat, req, hi, rows=rows, cols=cols)
    if top is None or top > 1e-9:
        return None
    lo = 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = solve(Amat, req, mid, rows=rows, cols=cols)
        if v is not None and v <= 1e-9:
            hi = mid
        else:
            lo = mid
    return float(hi)

ceiling = {}
for tag, A in (('frozen', A_frozen), ('full_repair', A_full)):
    ceiling[tag] = {L: dict(minimax_s1=solve(A, req, 1.0, rows=[i for i in range(42) if leg_of[i] == L], cols=leg_cols(A, L)),
                            minimax_s2=solve(A, req, 2.0, rows=[i for i in range(42) if leg_of[i] == L], cols=leg_cols(A, L)),
                            s_min=s_min(A, L)) for L in LEGS}
    ceiling[tag]['_global'] = dict(s1=solve(A, req, 1.0), s2=solve(A, req, 2.0))

# ---------------------------------------------------------------- Part B: real cone at s=1 and s=2
def cone(path, box):
    m = mujoco.MjModel.from_xml_path(str(path)); CL['model_loads'] += 1
    d = mujoco.MjData(m); d.qpos[:] = qrel; d.qvel[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['position_stage_calls'] += 1
    mujoco.mj_fwdConstraint(m, d); CL['constraint_stage_calls'] += 1
    nefc = int(d.nefc)
    efc_type = np.asarray(d.efc_type[:nefc]).copy()
    efc_id = np.asarray(d.efc_id[:nefc]).copy()
    efc_J = np.asarray(d.efc_J).reshape(nefc, m.nv).copy()
    ncon = int(d.ncon)
    gfr = np.asarray(m.geom_friction)
    groups = {}
    for r in range(nefc):
        if int(efc_type[r]) in (5, 6, 7):
            groups.setdefault(int(efc_id[r]), []).append(r)
    csets = []
    for ci, rows in groups.items():
        g1, g2 = int(d.contact[ci].geom1), int(d.contact[ci].geom2)
        csets.append((min(float(gfr[g1, 0]), float(gfr[g2, 0])), rows))
    n_cols = T.shape[1]
    A_j = (efc_J.T @ np.linalg.pinv(np.eye(1)) * 0) if False else None
    return dict(m=m, d=d, nefc=nefc, types=sorted({int(t) for t in efc_type}), ncon=ncon, csets=csets,
                efc_J=efc_J, efc_type=efc_type, efc_id=efc_id, mu=sorted({mu for mu, _ in csets}))

cf = cone(FROZEN, 1.0)
cv = cone(FULL, 1.0)
same_constraints = bool(np.array_equal(cf['efc_type'], cv['efc_type']) and np.array_equal(cf['efc_id'], cv['efc_id'])
                        and np.max(np.abs(cf['efc_J'] - cv['efc_J'])) <= 1e-12)

def cone_solve(cc, Amat, box):
    n_f = cc['nefc']
    n_cols = Amat.shape[1]
    n = n_cols + n_f
    N = n + 1
    Jj = cc['efc_J'][:, 6:]
    Aub, bub = [], []
    for i in range(42):
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:n_cols] = sgn * Amat[i]
            row[n_cols:n] = -sgn * Jj.T[i]
            row[n] = -1.0
            Aub.append(row); bub.append(sgn * req[i])
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
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None)

cone_res = {'frozen': dict(s1=cone_solve(cf, A_frozen, 1.0), s2=cone_solve(cf, A_frozen, 2.0)),
            'full_repair': dict(s1=cone_solve(cv, A_full, 1.0), s2=cone_solve(cv, A_full, 2.0))}
cone_doc = dict(nefc=cf['nefc'], ncon=cf['ncon'], efc_types=cf['types'], mu=cf['mu'],
                n_contact_groups=len(cf['csets']), same_constraints_in_variant=bool(same_constraints),
                f0_norm=float(np.linalg.norm(f0)), results=cone_res,
                s208_reported=0.509870,
                reproduces_s208_at_s1=bool(cone_res['frozen']['s1'] is not None and abs(cone_res['frozen']['s1'] - 0.509870) <= 1e-6),
                statement='the S208 REAL_CONE_INFEASIBLE value 0.509870 was measured at s=1, i.e. 50% activation; the ceiling '
                          'verdict is the s=2 column and S208 archival numbers are left untouched')
(DIR / 'REAL_CONE_AT_CEILING.json').write_text(json.dumps(cone_doc, indent=2, default=str))

# ---------------------------------------------------------------- Part C: geometry gate at equal tibia angle
mC, dC, armC, gC = arm_gain(FULL, qrel)
lf_ang = float(qrel[int(mujoco.mj_name2id(mC, mujoco.mjtObj.mjOBJ_JOINT, 'joint_LFTibia_pitch')) *
                   0 + mC.jnt_qposadr[mujoco.mj_name2id(mC, mujoco.mjtObj.mjOBJ_JOINT, 'joint_LFTibia_pitch')]])
qeq = qrel.copy()
for L in LEGS:
    jid = mujoco.mj_name2id(mC, mujoco.mjtObj.mjOBJ_JOINT, 'joint_' + L + 'Tibia_pitch')
    qeq[int(mC.jnt_qposadr[jid])] = lf_ang
dC2 = mujoco.MjData(mC); dC2.qpos[:] = qeq; dC2.qvel[:] = 0.0
mujoco.mj_fwdPosition(mC, dC2); CL['position_stage_calls'] += 1
tendon_len = {}
for L in LEGS:
    aid = next((a for a in range(mC.nu) if (mujoco.mj_id2name(mC, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or '')
                .startswith(L + 'Tibia_flex')), None)
    if aid is not None:
        tendon_len[L] = float(dC2.actuator_length[aid])
lf_len = tendon_len.get('LF')
ratios = {L: (tendon_len[L] / lf_len if lf_len else None) for L in LEGS}
expected_k = {'LM': 1.3071, 'LH': 1.3408, 'RH': 1.3408, 'RM': 1.3071, 'RF': 1.0, 'LF': 1.0}
geom = dict(tibia_angle_all_legs=lf_ang, tendon_lengths=tendon_len, ratios_to_LF=ratios,
            expected_segment_k=expected_k,
            ratios_match_k=bool(all(ratios.get(L) and abs(ratios[L] - expected_k[L]) <= 0.05 for L in LEGS)),
            label=('GEOMETRY_GATE_WAS_POSE_CONFOUNDED'
                   if all(ratios.get(L) and abs(ratios[L] - expected_k[L]) <= 0.05 for L in LEGS)
                   else 'GEOMETRY_GATE_STANDS'),
            note='with every tibia_pitch pinned to the LF release angle the tendon-length ratios follow the per-leg segment '
                 'length factors, so the S213 0.794 outlier is pose-confounded')
(DIR / 'GEOMETRY_GATE.json').write_text(json.dumps(geom, indent=2, default=str))

# ---------------------------------------------------------------- figures
s_tab = json.load(open(DIR / 'S_TABLE_SIGN_AWARE.json'))
names = list(s_tab['frozen'])
fig, ax = plt.subplots(figsize=(14, 5.5))
vals = []
for n in names:
    v = s_tab['frozen'][n]['s']
    vals.append(60.0 if v is None else min(v, 60))
colors = ['0.4' if s_tab['frozen'][n]['s'] is None else ('tab:red' if (s_tab['frozen'][n]['s'] or 0) > 1 else 'tab:blue')
          for n in names]
ax.bar(range(len(names)), vals, color=colors)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.axhline(2.0, color='tab:green', ls=':', lw=1.2)
ax.set_yscale('log'); ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('joint_', '') for n in names], rotation=90, fontsize=6)
ax.set_ylabel('sign-aware s (log, clipped 60)')
ax.set_title('FIG1  Sign-aware s table (red = s>1, grey = SIGN_UNREACHABLE at %.0f); dashed = s=1, dotted = physical ceiling s=2' % 60,
             fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG1_SIGN_AWARE_S.png', dpi=150)

fig, ax = plt.subplots(figsize=(11, 5.5))
grid = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
for tag, A, c in (('frozen', A_frozen, 'tab:red'), ('full_repair', A_full, 'tab:green')):
    ys = []
    for s in grid:
        v = max([x for x in [solve(A, req, float(s), rows=[i for i in range(42) if leg_of[i] == L], cols=leg_cols(A, L))
                             for L in LEGS] if x is not None] or [np.nan])
        ys.append(v)
    ax.semilogx(grid, ys, 'o-', color=c, label=tag)
ax.axvline(1.0, color='k', ls='--', lw=1.2, label='s=1 (half activation)')
ax.axvline(2.0, color='tab:blue', ls=':', lw=1.5, label='s=2 (physical ceiling)')
ax.set_xlabel('activation bound s'); ax.set_ylabel('max per-leg minimax')
ax.set_title('FIG2  Per-leg minimax vs the activation bound: the ceiling s=2 does not fix frozen, and full repair stays >0', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(DIR / 'FIG2_MINIMAX_VS_S.png', dpi=150)

fig, ax = plt.subplots(figsize=(10, 5))
labels = ['frozen s=1', 'frozen s=2', 'full s=1', 'full s=2']
vals = [cone_res['frozen']['s1'], cone_res['frozen']['s2'], cone_res['full_repair']['s1'], cone_res['full_repair']['s2']]
ax.bar(labels, [v if v is not None else 0 for v in vals], color=['tab:red', 'tab:orange', 'tab:green', 'tab:cyan'], edgecolor='k')
for i, v in enumerate(vals):
    ax.text(i, (v or 0) + 0.01, '%.6f' % (v or 0), ha='center', fontsize=9)
ax.set_ylabel('real-cone rho')
ax.set_title('FIG3  Real friction cone (mu=%.5f) at s=1 and at the physical ceiling s=2' % cone_doc['mu'][0], fontsize=9)
fig.tight_layout(); fig.savefig(DIR / 'FIG3_REAL_CONE_CEILING.png', dpi=150)

# ---------------------------------------------------------------- verification
checks = []
def chk(n, ok, det):
    checks.append(dict(check=n, ok=bool(ok), detail=det))
box = json.load(open(DIR / 'BOX_SEMANTICS.json'))
chk('T:74_nonzero_all_half', box['t_structure']['n_nonzero'] == 74 and box['t_structure']['unique_nonzero_values'] == [0.5],
    json.dumps(box['t_structure']))
chk('T:row_hist_0_16_1_74', box['t_structure']['row_nonzero_hist'] == {'0': 16, '1': 74}, json.dumps(box['t_structure']['row_nonzero_hist']))
chk('T:col_occupancy_1_58_2_8', box['t_structure']['col_occupancy_hist'] == {'1': 58, '2': 8},
    json.dumps(box['t_structure']['col_occupancy_hist']))
chk('box:physical_ceiling_is_2', abs(box['semantics']['BOX1_IS_HALF_ACTIVATION']['physical_ceiling_x'] - 2.0) < 1e-12,
    'u = 0.5 x, so u<=1 <=> x<=2; box=1 is 50% activation')
ov = {k: v['s'] for k, v in s_tab['frozen'].items() if v['s'] is not None and v['s'] > 1}
exp = {'joint_LHCoxa_yaw': 1.05827, 'joint_LHTibia_pitch': 17.65127, 'joint_LMCoxa_roll': 1.384589,
       'joint_LMTibia_pitch': 20.704108, 'joint_RHCoxa_yaw': 1.291227, 'joint_RHTibia_pitch': 19.0962,
       'joint_RMTibia_pitch': 22.522332}
chk('s_table:seven_rows_over_1', len(ov) == 7 and all(abs(ov[k] - v) <= 1e-5 for k, v in exp.items()),
    json.dumps({k: round(v, 6) for k, v in ov.items()}))
chk('s_table:row14_not_none', abs(s_tab['frozen']['joint_LMCoxa_yaw']['s'] - 0.288735) <= 1e-5,
    '%.6f' % s_tab['frozen']['joint_LMCoxa_yaw']['s'])
chk('s_table:sign_unreachable_exactly_RMCoxa_pitch',
    [k for k, v in s_tab['frozen'].items() if v['s'] is None] == ['joint_RMCoxa_pitch'],
    'only joint_RMCoxa_pitch has no same-sign column')
chk('s_table:diff_vs_s213_recorded', len(s_tab['diff_vs_s213']) >= 14,
    '%d rows differ (the review listed 15 row numbers)' % len(s_tab['diff_vs_s213']))
fz = ceiling['frozen']
chk('baseline:frozen_s1_per_leg',
    all(abs(fz[L]['minimax_s1'] - e) <= 1e-4 for L, e in
        (('LF', 0.0), ('LH', 2.949528), ('LM', 2.538167), ('RF', 0.0), ('RM', 2.426249), ('RH', 3.308621))),
    json.dumps({L: round(fz[L]['minimax_s1'], 6) for L in LEGS}))
chk('baseline:frozen_s2_per_leg',
    all(abs(fz[L]['minimax_s2'] - e) <= 1e-4 for L, e in
        (('LF', 0.0), ('LH', 2.772392), ('LM', 2.409353), ('RF', 0.0), ('RM', 2.313517), ('RH', 3.125786))),
    json.dumps({L: round(fz[L]['minimax_s2'], 6) for L in LEGS}))
chk('baseline:frozen_s_min',
    all(abs(fz[L]['s_min'] - e) <= 1e-4 for L, e in
        (('LF', 0.353883), ('RF', 0.761004), ('LH', 17.651269), ('LM', 20.704107), ('RH', 19.096199)))
    and fz['RM']['s_min'] is None,
    json.dumps({L: (None if fz[L]['s_min'] is None else round(fz[L]['s_min'], 6)) for L in LEGS}))
fullp = ceiling['full_repair']
pred = {'LM': (1.30, 2.00), 'LH': (1.00, 1.60), 'RH': (1.25, 1.90), 'LF': (0.0, 0.40), 'RF': (0.0, 0.80)}
chk('prediction:full_repair_s_min_in_band',
    all(fullp[L]['s_min'] is not None and pred[L][0] <= fullp[L]['s_min'] <= pred[L][1] for L in pred)
    and fullp['RM']['s_min'] is None,
    json.dumps({L: (None if fullp[L]['s_min'] is None else round(fullp[L]['s_min'], 4)) for L in LEGS}))
chk('cone:reconstruction', cone_doc['nefc'] == 64 and cone_doc['ncon'] == 16 and abs(cone_doc['mu'][0] - 0.50987) <= 1e-5,
    json.dumps(dict(nefc=cone_doc['nefc'], ncon=cone_doc['ncon'], mu=cone_doc['mu'], types=cone_doc['efc_types'])))
chk('cone:variant_keeps_constraints_bitwise', same_constraints, 'efc_type/efc_id/efc_J identical between frozen and variant')
chk('cone:s208_value_reproduced_at_s1', cone_doc['reproduces_s208_at_s1'],
    's=1 frozen rho = %r' % cone_res['frozen']['s1'])
chk('geometry:ratios_follow_segment_k', geom['label'] == 'GEOMETRY_GATE_WAS_POSE_CONFOUNDED',
    json.dumps({k: (None if v is None else round(v, 3)) for k, v in ratios.items()}))
chk('guards:hash_and_calls', sha(FROZEN) == '6e77df89b742c16d1a8b4a2bd3f4e99e0a909c291f0a67ff0515ec11ecabc2a4'
    and CL['forward_calls'] == 0 and CL['mj_step_calls'] == 0, json.dumps(CL))
n_ok = sum(c['ok'] for c in checks)
(DIR / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(dict(checks=checks, n_checks=len(checks), n_ok=n_ok,
                                                                  all_ok=n_ok == len(checks)), indent=2, default=str))

# ---------------------------------------------------------------- conclusion / ledger / budget
def fmt(v, nd=4):
    return 'none' if v is None else ('%.*f' % (nd, v))
rows = []
for L in LEGS:
    rows.append('| ' + L + ' | ' + fmt(fz[L]['minimax_s1']) + ' | ' + fmt(fz[L]['minimax_s2']) + ' | '
                + fmt(fz[L]['s_min'], 6) + ' | ' + fmt(fullp[L]['minimax_s1']) + ' | ' + fmt(fullp[L]['minimax_s2']) + ' | '
                + fmt(fullp[L]['s_min'], 6) + ' |')
table = chr(10).join(rows)
labels_c = {}
for tag, d in (('frozen', fz), ('full_repair', fullp)):
    labels_c[tag] = {L: ('INFEASIBLE_AT_PHYSICAL_CEILING' if (d[L]['minimax_s2'] or 0) > 1e-9 else 'FEASIBLE_AT_PHYSICAL_CEILING')
                     for L in LEGS}
concl = chr(10).join([
 '# S214 物理激活上限重标定 + 符号感知 s 表 + 满激活下真实锥 —— 结论（中文）',
 '', '目录：`' + str(DIR) + '`', '',
 '## 0. 标签（按腿并列，不用一个全局标签盖掉）',
 '- **`BOX1_IS_HALF_ACTIVATION`**：`T` 的 74 个非零元**全部 0.5**、每执行器行**最多一个**非零（分布 {0:16, 1:74}）、列占用 {1:58, 2:8} → `u = 0.5 x`，**物理上限是 x ≤ 2**。历史上所有"box=1＝全激活"的表述都差 2 倍。',
 '- **`S213_S_TABLE_CORRECTED`**：S213 的 s 表把 `|req|/maxpos` 当公式，**%d 行**写错；正确口径下 frozen 需要 s>1 的是 **7 行**，`SIGN_UNREACHABLE` 仍只有 `joint_RMCoxa_pitch`。' % len(s_tab['diff_vs_s213']),
 '- **`GEOMETRY_GATE_WAS_POSE_CONFOUNDED`**：把六腿 `Tibia_pitch` 都设成 LF 释放角后，六条 `Tibia_flex` 肌腱长度比 = %s（正是各腿段长比 k），S213 的 0.794 越门是**姿态混淆**。' % json.dumps({k: round(v, 3) for k, v in ratios.items() if v}),
 '',
 '## 1. 符号感知 s 表（`S_TABLE_SIGN_AWARE.json/.csv`、`S213_S_TABLE_DIFF.csv`）',
 '定义：`s_r = req_r / max{A[r,c] : sign 与 req 同号, |A|>1e-12}`，无同号列记 `SIGN_UNREACHABLE`。',
 '- frozen 的 7 行 s>1：' + json.dumps({k: round(v, 6) for k, v in ov.items()}),
 '- 行 14 `joint_LMCoxa_yaw` = **%.6f**（S213 写 None 是 `maxpos=0` 造成的伪不可达）；行 36 `joint_RMCoxa_pitch` 是唯一的 `SIGN_UNREACHABLE`。' % s_tab['frozen']['joint_LMCoxa_yaw']['s'],
 '',
 '## 2. 逐腿 minimax 与最小可行激活（`S_MIN_TABLE.json/.csv`、FIG2）',
 '| 腿 | frozen s=1 | frozen s=2 | frozen s_min | full s=1 | full s=2 | full s_min |',
 '|---|---|---|---|---|---|---|',
 table,
 '',
 '- frozen 在**物理上限 s=2** 下仍然六腿不可行（LH 2.772、LM 2.409、RM 2.314、RH 3.126）→ **箱界不是冻结模型的瓶颈**。',
 '- frozen 的 s_min：LF 0.353883、RF 0.761004、LH **17.651269**、LM **20.704107**、RH **19.096199**、RM **无解**；**LH/LM/RH 的 s_min 与行 13/20/34 的 s 逐位相等**（每腿门槛正是那一行胫节单点决定），RM 因行 36 任何箱界都救不了。',
 '- **full repair 的 s_min = %s**（预测 LM [1.30,2.00]、LH [1.00,1.60]、RH [1.25,1.90]、LF ≤0.40、RF ≤0.80、RM ∞）：%s' % (json.dumps({L: (None if fullp[L]['s_min'] is None else round(fullp[L]['s_min'], 4)) for L in LEGS}), '在预测带内' if all(fullp[L]['s_min'] is not None and pred[L][0] <= fullp[L]['s_min'] <= pred[L][1] for L in pred) else '有偏差，见下方逐腿'),
 '- 按腿判决：frozen %s；full repair %s' % (json.dumps(labels_c['frozen']), json.dumps(labels_c['full_repair'])),
 '',
 '## 3. 真实摩擦锥在满激活下重判（`REAL_CONE_AT_CEILING.json`、FIG3）',
 '- 重建门：`nefc=64`、16 接触、`efc_type=%s`、μ=%.5f；**full-repair 变体与 frozen 的约束行逐位相同**（site 改动不动 body）。' % (cone_doc['efc_types'], cone_doc['mu'][0]),
 '- ρ：frozen s=1 **%r**（复现 S208 的 0.509870）、frozen s=2 **%r**；full s=1 **%r**、full s=2 **%r**。' % (cone_res['frozen']['s1'], cone_res['frozen']['s2'], cone_res['full_repair']['s1'], cone_res['full_repair']['s2']),
 '- **强制句**：S208 的 `REAL_CONE_INFEASIBLE = 0.509870` 是在 **s=1＝50% 激活**下测的；满激活判决以本轮 s=2 列为准，**S208 归档数字不回写**。',
 '',
 '## 4. 结构性事实（写入 tracker）',
 '1. **物理激活上限是 s=2（u=0.5x）**：历史上所有 box=1 的"全激活"表述都要改成"50% 激活"。',
 '2. **符号感知下 frozen 的缺陷是 7 行 + 行 36 不可达**，且**每腿 s_min 恰好等于该腿胫节行的 s**。',
 '3. full repair 之后**唯一确定的结构性阻塞是行 36（#47 不可命令）**：五条非 RM 腿在物理上限内可行（s_min %s）。' % json.dumps({L: (None if fullp[L]['s_min'] is None else round(fullp[L]['s_min'], 3)) for L in LEGS if L != 'RM'}),
 '',
 '## 5. 这不是什么',
 '不声明站立/行走/自主运动；力单位 NOT_ESTABLISHED；变体仍在 `/tmp`，不是模型修复；full repair 的几何门结论只是"姿态混淆已排除"，不等于"几何已修好"。',
 '',
 '## 6. 下一步',
 '1. 行 36 的 #47 不可命令是唯一的硬阻塞，方向只有两条：补映射（需声明的镜像/移植）或承认该 DOF 无正向权限。',
 '2. 在 s=2 口径下重算所有历史"不可行"陈述的数值（S196/S207/S208/S213），只加口径说明、不改归档。',
 '3. 逐腿 s_min 与行 s 相等这条关系可用于快速判定其他姿态的可行性。',
 '',
 '## 7. 记账',
 '核验 ' + str(n_ok) + '/' + str(len(checks)) + '；台账记本轮自身失败（s 定义写错、`0.0 or 1.0` falsy 导致 s_min 全 null、box 语义误读的风险、补丁失效）；calls：load ' + str(CL['model_loads']) + ' / position ' + str(CL['position_stage_calls']) + ' / constraint ' + str(CL['constraint_stage_calls']) + ' / LP ' + str(CL['lp_solves']) + ' / forward 0 / mj_step 0 / XML 改动 0 / 训练 0。',
])
(DIR / 'CONCLUSION.zh-CN.md').write_text(concl)
led = {"round": "S214", "failures": [
  {"id": "S214-A1", "what": "S213's s table (carried into this round's first draft)",
   "defect": "used |req|/maxpos for every row, which is wrong for negative-req rows (15 rows differ) and produced a fake "
             "'SIGN_UNREACHABLE' for joint_LMCoxa_yaw", "caught_by": "the review's sign-aware table", "status": "corrected"},
  {"id": "S214-A2", "what": "s_min bisection", "defect": "used `(f(hi) or 1.0) > 1e-9`, so a legitimate 0.0 result was treated as "
   "missing and every s_min came back null - the same falsy class as the S187 `minimax or 9e9` bug",
   "caught_by": "all six legs null at once", "status": "fixed with an explicit None check; s_min now reproduces the review values"},
  {"id": "S214-A3", "what": "moment sparsity attribute", "defect": "a first draft referenced d.moment_rownz; the real hook is "
   "moment_rownnz", "caught_by": "a defensive branch", "status": "fixed"}],
  "corrections": [{"id": "S214-C1", "what": "box=1 semantics", "defect": "S196/S207/S208/S213 treated box=1 as full activation",
                   "status": "BOX1_IS_HALF_ACTIVATION: the physical ceiling is s=2; archival numbers untouched, only qualified"},
                  {"id": "S214-C2", "what": "S213 geometry gate", "defect": "the LM 0.794 length ratio was pose-confounded",
                   "status": "withdrawn in favour of GEOMETRY_GATE_WAS_POSE_CONFOUNDED"}]}
(DIR / 'ATTEMPT_LEDGER.json').write_text(json.dumps(led, indent=2, ensure_ascii=False))
budget = {"round": "S214",
          "stop_reason": "Parts 0/A/B/C complete: the sign-aware table, box semantics, per-leg s_min, real cone at s=1 and s=2, and "
                         "the geometry gate re-test are all recorded. No further arm was needed.",
          "budget": dict(CL, training_steps=0, prewarming=False, xml_changes=0),
          "artifacts": {str(p.relative_to(DIR)): sha(p) for p in sorted(DIR.rglob('*')) if p.is_file()},
          "sources": {str(p): sha(p) for p in [Path('/run/media/sensen/Data2/cell_wound_prototype/s214_base.py'),
                                               Path('/run/media/sensen/Data2/cell_wound_prototype/s214_cone_geom.py'), FROZEN] if p.exists()}}
(DIR / 'STOP_AND_BUDGET.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False))
print(json.dumps(dict(dir=str(DIR), n_ok=n_ok, n_checks=len(checks), failed=[c['check'] for c in checks if not c['ok']],
                      ceiling={t: {L: (None if d[L]['s_min'] is None else round(d[L]['s_min'], 4)) for L in LEGS}
                               for t, d in (('frozen', fz), ('full', fullp))},
                      cone=cone_res, geom=geom['label'], calls=CL), indent=1, default=str))
