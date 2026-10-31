"""S221 Part 0 corrections + Part A pose-robust LP sweep (1 load)."""
import json, datetime, hashlib
import numpy as np, mujoco
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s221-pose-robust-and-first-dynamics')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
S217 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
S220 = Path('/home/sensen/Desktop/engine修复验收/s220-certificate-and-pose-robustness/20261025T174741799647Z')
S177 = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CL = dict(model_loads=0, fwd_position=0, fwd_velocity=0, fwd_actuation=0, fwd_constraint=0,
          mj_forward=0, mj_step=0, lp_solves=0)
W_BODY = 24.446354755878
T0 = 0.05932941438013683
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

prereg = dict(round='S221', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              corrections=['INFEASIBILITY_POSE_ROBUST replaces RESIDUAL_POSE_BOUNDARY',
                           'CERT_POSE_SPECIFIC only describes one w vector',
                           'sum(lambda) is vertex-dependent; the invariant is |sum(lambda) - W| = t',
                           'column-side details are vertex-dependent; only "no positive projection" is invariant',
                           'bookkeeping: LP counts and check counts must be single-sourced'],
              pose_grid='42 joints x {+2,-2,+10,-10,+20} mrad = 210 poses, per pose position->constraint->velocity->actuation',
              labels=['INFEASIBILITY_POSE_ROBUST', 'INFEASIBILITY_POSE_BOUNDARY', 'W_CERTIFICATE_POSE_SPECIFIC'],
              budget=dict(model_loads_max=2, lp_solves_max=400, mj_forward_max=3, mj_step_max=600, training_steps=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

z = np.load(S217 / 'AUTH_MATRICES_V3.npz'); c = np.load(S217 / 'CONE_STRUCTURE_V2.npz')
T = z['T']; J0 = z['J']; f0 = z['f0']
jn = [str(x) for x in z['row_names']]
NAME48 = ['root_%d' % i for i in range(6)] + jn
B0 = z['B_frozen']; base0 = z['base_frozen']
pool = np.asarray(z['actuator_pool'])
CENSUS = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z/READOUT_CENSUS.json')
act_names = [x['actuator'] for x in json.load(open(CENSUS))['rows']]
qrel = np.asarray(np.load(S177)['qpos'], float)
groups = {}
for r in range(int(c['nefc'])):
    groups.setdefault(int(c['efc_id'][r]), []).append(r)

# -------------------------------------------------- Part 0: corrections (recompute the disputed numbers)
lam_opt = None
def t_lp(Amat, bvec, s=1.0, Jm=None, want_lam=False):
    Jm = J0 if Jm is None else Jm
    n_cols = Amat.shape[1]; n_lam = Jm.shape[0]; n = n_cols + n_lam; N = n + 1
    Aub, bub = [], []
    for i in range(Amat.shape[0]):
        for sgn in (1.0, -1.0):
            row = np.zeros(N); row[:n_cols] = sgn * Amat[i]; row[n_cols:n] = sgn * Jm.T[i]; row[n] = -1.0
            Aub.append(row); bub.append(sgn * (-bvec[i]))
    for r in range(n_lam):
        row = np.zeros(N); row[n_cols + r] = -1.0; Aub.append(row); bub.append(0.0)
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub),
                 bounds=[(0, s)] * n_cols + [(None, None)] * n_lam + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    if not lp.success:
        return None, None, None
    return float(lp.x[-1]), (lp.x[n_cols:n] if want_lam else None), lp.x[:n_cols]

A0_48 = B0 @ T
t0, lam0, x0 = t_lp(A0_48, base0, 1.0, want_lam=True)
sig0 = float(np.sum(lam0))
# certificate at q0 (48-var LP) and L1 minimal certificate (96-var LP)
Aub_w = np.vstack((A0_48.T, J0, base0[None, :]))
lpw = linprog(np.zeros(48), A_ub=Aub_w, b_ub=np.concatenate((np.zeros(66), np.zeros(64), np.array([-1.0]))),
              bounds=[(None, None)] * 48, method='highs')
CL['lp_solves'] += 1
w = lpw.x / (-(base0 @ lpw.x))
l1 = linprog(np.ones(96), A_ub=np.vstack((np.column_stack((A0_48.T, -A0_48.T)),
                                          np.column_stack((J0, -J0)),
                                          np.concatenate((base0, -base0))[None, :])),
             b_ub=np.concatenate((np.zeros(66), np.zeros(64), np.array([-1.0]))),
             bounds=[(0, None)] * 96, method='highs')
CL['lp_solves'] += 1
wl1 = l1.x[:48] - l1.x[48:]
proj_w = A0_48.T @ w
proj_l1 = A0_48.T @ (wl1 / -(base0 @ wl1))
corr = dict(
    sum_lambda_vertex=dict(measured=float(sig0), W=W_BODY, t=t0,
                           W_plus_t=float(W_BODY + t0), W_minus_t=float(W_BODY - t0),
                           invariant_holds=bool(min(abs(sig0 - (W_BODY + t0)), abs(sig0 - (W_BODY - t0))) <= 1e-6),
                           statement='sum(lambda) is not unique: |sum(lambda) - W| = t is the invariant; both W+t and W-t are optimal vertices'),
    column_side=dict(
        n_negative_from_single_w=int((proj_w < -1e-9).sum()),
        negative_single_w=[dict(column=int(i), projection=float(proj_w[i])) for i in np.where(proj_w < -1e-9)[0]],
        negative_l1_w=[dict(column=int(i), projection=float(proj_l1[i])) for i in np.where(proj_l1 < -1e-9)[0]],
        invariant='no commandable column has a POSITIVE projection on w; which columns are strictly negative is vertex-dependent'),
    w_norm_vs_l1=dict(sum_abs_w=float(np.abs(w).sum()), l1_objective=float(l1.fun),
                      same_solution=bool(abs(np.abs(w).sum() - l1.fun) <= 1e-6),
                      note='the archived S220 w came from the 48-variable LP while l1_objective came from the 96-variable one')
)
(DIR / 'S220_CORRECTIONS.json').write_text(json.dumps(corr, indent=2, default=str))

# -------------------------------------------------- Part A: per-pose LP re-solve on 210 poses
m = mujoco.MjModel.from_xml_path(str(FROZEN)); CL['model_loads'] += 1
nodes = list(ET.parse(str(FROZEN)).getroot().find('actuator'))
d = mujoco.MjData(m)

def fl(l, lo, hi):
    if l < lo or l > hi: return 0.0
    a = (lo + 1) * .5; b = (1 + hi) * .5
    if l <= a: return 0.5 * ((l - lo) / max(1e-15, a - lo)) ** 2
    if l <= 1: return 1 - 0.5 * ((1 - l) / max(1e-15, 1 - a)) ** 2
    if l <= b: return 1 - 0.5 * ((l - 1) / max(1e-15, b - 1)) ** 2
    return 0.5 * ((hi - l) / max(1e-15, hi - b)) ** 2

def gain_of(length, node):
    g = [float(x) for x in node.get('gainprm').split()]; lr = [float(x) for x in node.get('lengthrange').split()]
    l0 = (lr[1] - lr[0]) / max(1e-15, g[1] - g[0]); l = g[0] + (length - lr[0]) / max(1e-15, l0)
    return -g[2] * fl(l, g[4], g[5])

def state(q):
    d.qpos[:] = q; d.qvel[:] = 0.0; d.act[:] = 0.0; d.ctrl[:] = 0.0
    mujoco.mj_fwdPosition(m, d); CL['fwd_position'] += 1
    mom = np.zeros((m.nu, m.nv))
    for a in range(m.nu):
        for k in range(int(d.moment_rownnz[a])):
            adr = int(d.moment_rowadr[a]) + k
            mom[a, int(d.moment_colind[adr])] = d.actuator_moment[adr]
    mujoco.mj_fwdVelocity(m, d); CL['fwd_velocity'] += 1
    mujoco.mj_fwdActuation(m, d); CL['fwd_actuation'] += 1
    mujoco.mj_fwdConstraint(m, d); CL['fwd_constraint'] += 1
    g = np.array([gain_of(float(d.actuator_length[a]), nodes[a]) for a in range(m.nu)])
    Bq = mom.T * g[None, :]
    baseq = mom.T @ np.asarray(d.actuator_force, float) + np.asarray(d.qfrc_passive, float) - np.asarray(d.qfrc_bias, float)
    nefc = int(d.nefc)
    Jq = np.asarray(d.efc_J).reshape(nefc, m.nv).copy()
    contacts = tuple(sorted((int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(int(d.ncon))))
    return dict(B=Bq, base=baseq, J=Jq, contacts=contacts, ncon=int(d.ncon), nefc=nefc,
                min_dist=float(min((d.contact[i].dist for i in range(int(d.ncon))), default=0.0)))

st0 = state(qrel)
joints = [j for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('joint_')]
rows = []
for jid in joints:
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    adr = int(m.jnt_qposadr[jid])
    lo, hi = m.jnt_range[jid]
    for dlt in (0.002, -0.002, 0.010, -0.010, 0.020):
        ang = qrel[adr] + dlt
        if m.jnt_limited[jid] and not (lo <= ang <= hi):
            continue
        q = qrel.copy(); q[adr] = ang
        st = state(q)
        tq, _, _ = t_lp(st['B'] @ T, st['base'], 1.0)
        rows.append(dict(joint=name, delta=dlt, t=tq, contacts_same=bool(st['contacts'] == st0['contacts']),
                         ncon=st['ncon'], min_dist=st['min_dist']))
with open(DIR / 'POSE_ROBUST_LP.csv', 'w') as f:
    f.write('joint,delta,dcontact,t,contacts_same,ncon,min_dist\n')
    for r in rows:
        f.write('%s,%.4f,%d,%r,%d,%d,%.6e\n' % (r['joint'], r['delta'], int(not r['contacts_same']), r['t'],
                                                r['contacts_same'], r['ncon'], r['min_dist']))
same = [r for r in rows if r['contacts_same']]
changed = [r for r in rows if not r['contacts_same']]
immovable = {}
for jid in joints:
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    ts = [r['t'] for r in same if r['joint'] == nm]
    if ts:
        immovable[nm] = dict(n=len(ts), min=float(min(ts)), max=float(max(ts)),
                             spread=float(max(ts) - min(ts)), invariant=bool(len(set(round(x, 15) for x in ts)) == 1))
rm_joints = sorted([k for k in immovable if k.startswith('joint_RM')])
others = sorted([k for k in immovable if not k.startswith('joint_RM')])
pose = dict(n_rows=len(rows), n_same=len(same), n_changed=len(changed),
            same_min_t=float(min(r['t'] for r in same)) if same else None,
            same_max_t=float(max(r['t'] for r in same)) if same else None,
            changed_min_t=float(min(r['t'] for r in changed)) if changed else None,
            changed_max_t=float(max(r['t'] for r in changed)) if changed else None,
            no_zero_t=bool(all(r['t'] is None or r['t'] > 1e-9 for r in rows)),
            n_invariant_joints=len([k for k, v in immovable.items() if v['invariant']]),
            invariant_joints=sorted([k for k, v in immovable.items() if v['invariant']]),
            noninvariant_joints={k: v for k, v in immovable.items() if not v['invariant']},
            rm_joints=rm_joints, other_joints_n=len(others),
            label=('INFEASIBILITY_POSE_ROBUST' if (same and min(r['t'] for r in same) > 1e-9
                                                   and all(r['t'] > 1e-9 for r in rows)) else 'INFEASIBILITY_POSE_BOUNDARY'),
            q0=dict(t=st0['t'] if 't' in st0 else t0, ncon=st0['ncon'], nefc=st0['nefc'], min_dist=st0['min_dist']))
(DIR / 'POSE_ROBUST_LP.json').write_text(json.dumps(pose, indent=2, default=str))
print(json.dumps(dict(dir=str(DIR), corrections={k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items()
                                                    if not isinstance(vv, (list, dict))}) for k, v in corr.items()},
                      pose={k: v for k, v in pose.items() if k in ('n_rows','n_same','n_changed','same_min_t','same_max_t',
                                                                    'changed_min_t','changed_max_t','no_zero_t',
                                                                    'n_invariant_joints','label')},
                      invariant_sample=pose['invariant_joints'][:6], rm=rm_joints,
                      noninvariant={k: (round(v['min'],6), round(v['max'],6)) for k, v in pose['noninvariant_joints'].items()},
                      calls=CL), indent=1, default=str))
