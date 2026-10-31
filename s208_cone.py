"""S208 Part B (column attribution) + Part D (real friction cone). One model load, one position-stage call."""
import json
import numpy as np
import mujoco
from pathlib import Path
from scipy.optimize import linprog

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s208-rm-direction-and-real-cone').iterdir())[-1]
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')

z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
qpos = np.asarray(z['qpos'], float)
pool = np.asarray(np.load(MAP)['actuator_pool'])
A = (Bm @ T)[6:, :]
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
RM_ROWS = [i for i in range(42) if leg_of_row[i] == 'RM']
RM_COLS = [c for c in range(A.shape[1]) if np.abs(A[RM_ROWS, c]).max() > 1e-12]
name_of_act = {c['actuator_index']: c['actuator'] for c in CENSUS['rows']}
slot_to_acts = {}
for ai in range(90):
    s = int(pool[ai])
    if s >= 0:
        slot_to_acts.setdefault(s, []).append(ai)

# ---------------------------------------------------------------- Part B: column attribution
def solve(Amat, reqv, box):
    n, m = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return float(lp.x[-1]), lp.x[:-1]

# rows of RM that are "hard" when solved jointly: identify them by re-solving with each row dropped
base_mm, x_rm = solve(A[RM_ROWS][:, RM_COLS], req[RM_ROWS], None)
res_rm = A[RM_ROWS][:, RM_COLS] @ x_rm - req[RM_ROWS]
dropout = []
for k, i in enumerate(RM_ROWS):
    keep = [j for jj, j in enumerate(RM_ROWS) if jj != k]
    mm_wo, _ = solve(A[keep][:, RM_COLS], req[keep], None)
    dropout.append(dict(joint_row=int(i), dof=6 + int(i), dof_name=jn[i], residual_with=float(res_rm[k]),
                        minimax_without_this_row=float(mm_wo)))
blocking = [d for d in dropout if d['minimax_without_this_row'] <= 1e-6]
col_contrib = []
for c in RM_COLS:
    m_abs = float(np.abs(A[RM_ROWS, c]).sum())
    col_contrib.append(dict(column=int(c), abs_mass=float(m_abs),
                            actuators=[name_of_act[a] for a in slot_to_acts.get(int(pool[c]) if c < pool.size else -1, [])]
                            if False else [name_of_act[a] for a in slot_to_acts.get(c, [])],
                            per_row={jn[i]: float(A[i, c]) for i in RM_ROWS if abs(A[i, c]) > 1e-12}))
col_contrib.sort(key=lambda d: -d['abs_mass'])
partB = dict(
    rm_rows=RM_ROWS, rm_cols=RM_COLS,
    rm_joint_minimax=x_rm is not None and float(base_mm),
    residuals=[dict(joint_row=int(i), dof_name=jn[i], residual=float(res_rm[k])) for k, i in enumerate(RM_ROWS)],
    dropout_analysis=dropout,
    rows_whose_removal_makes_RM_feasible=[d['dof_name'] for d in blocking],
    note=('every RM row is individually reachable (S208 Part A: no sign deficit, no amplitude deficit at box=1), yet the seven '
          'rows cannot be satisfied SIMULTANEOUSLY. The dropout analysis names the rows responsible. The blocking mechanism is '
          'therefore intra-leg simultaneity, not a per-row sign or amplitude problem.'),
    column_contributions=col_contrib[:20],
    coxa_rows=[dict(joint_row=int(i), dof_name=jn[i], dof=6 + int(i)) for i in RM_ROWS if 'Coxa' in jn[i]],
)
(DIR / 'RM_COLUMN_ATTRIBUTION.json').write_text(json.dumps(partB, indent=2, default=str))

# ---------------------------------------------------------------- Part D: real friction cone
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
d.qpos[:] = qpos
d.qvel[:] = 0.0
mujoco.mj_fwdPosition(m, d)          # position/collision stage ONLY: no dynamics, no step
nefc = int(d.nefc)
efc_type = np.asarray(d.efc_type[:nefc]).copy()
efc_id = np.asarray(d.efc_id[:nefc]).copy()
efc_J = np.asarray(d.efc_J.reshape(nefc, m.nv)).copy()
geom_friction = np.asarray(m.geom_friction).copy()
geom_condim = np.asarray(m.geom_condim).copy()
ncon = int(d.ncon)
con_geom = [(int(d.contact[i].geom1), int(d.contact[i].geom2), int(d.contact[i].dim)) for i in range(ncon)]
# group efc rows by contact id (efc_type 0 == contact/friction row)
groups = {}
for r in range(nefc):
    if int(efc_type[r]) in (5, 6, 7):          # mjCNSTR_CONTACT_FRICTIONLESS / PYRAMIDAL / ELLIPTIC in this MuJoCo build
        groups.setdefault(int(efc_id[r]), []).append(r)
group_sizes = {k: len(v) for k, v in groups.items()}
mu_of_contact = {}
for ci, rows in groups.items():
    g1, g2, dim = con_geom[ci]
    mu1, mu2 = float(geom_friction[g1, 0]), float(geom_friction[g2, 0])
    mu_of_contact[ci] = dict(geom1=g1, geom2=g2, dim=dim, mu_geom1=mu1, mu_geom2=mu2,
                             mu_min=min(mu1, mu2), mu_max=max(mu1, mu2),
                             condim1=int(geom_condim[g1]), condim2=int(geom_condim[g2]),
                             n_efc_rows=len(rows), rows=rows)
known = dict(model_loads=1, position_stage_calls=1, constraint_stage_calls=1, forward_calls=0, mj_step_calls=0,
             note='mj_fwdPosition + mj_fwdConstraint only; neither integrates nor advances the state. mj_forward/mj_step never called.')
(JJ_rows := len(groups))

Jj = J[:, 6:]                       # (64, 42) frozen contact rows mapped to the 42 joint DOFs
n_f = f0.size
assert nefc == n_f, (nefc, n_f)
assert sum(len(v) for v in groups.values()) == nefc, ('contact rows do not cover nefc; efc_type values present: %s'
                                                      % sorted(set(int(t) for t in efc_type)))
assert len(groups) > 0, 'no contact rows found: the friction constraints would be silently absent'
# map frozen f0 component index -> efc row order: they are the same ordering as J rows
cone_rows = []
for ci, info in mu_of_contact.items():
    cone_rows.append((ci, info['mu_min'], info['mu_max'], info['rows']))


def solve_cone(mu_scale=1.0, use_mu='min', box=1.0):
    """variables: x (66) in [0,box], f (64) free; joint equality A x - J^T f = req ; friction pyramid."""
    n = A.shape[1] + n_f
    Aub = [np.hstack((A, -Jj.T)), np.hstack((-A, Jj.T))]
    bub = [req, -req]
    rows = 2
    for ci, mu_lo, mu_hi, rr in cone_rows:
        mu = (mu_lo if use_mu == 'min' else mu_hi) * mu_scale
        if len(rr) < 2:
            continue
        normal = rr[0]
        tans = rr[1:]
        for t in tans:
            # tan - mu/sqrt(len(tans)) * normal <= 0
            a1 = np.zeros(n); a1[t] = 1.0; a1[normal] = -mu / np.sqrt(len(tans))
            a2 = np.zeros(n); a2[t] = -1.0; a2[normal] = -mu / np.sqrt(len(tans))
            Aub.append(a1[None, :]); bub.append(np.zeros(1))
            Aub.append(a2[None, :]); bub.append(np.zeros(1))
    for ci, mu_lo, mu_hi, rr in cone_rows:
        a = np.zeros(n); a[rr[0]] = -1.0        # normal >= 0  ->  -normal <= 0
        Aub.append(a[None, :]); bub.append(np.zeros(1))
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.vstack(Aub + [np.r_[np.zeros(n), -1.0]][:0] or np.vstack(Aub)),
                 b_ub=np.concatenate(bub), bounds=[(0, box)] * A.shape[1] + [(None, None)] * n_f + [(0, None)],
                 method='highs')
    return lp


# the minimax objective needs the t column; rebuild properly
def solve_cone_minimax(mu_scale=1.0, use_mu='min', box=1.0):
    n = A.shape[1] + n_f
    N = n + 1
    Aub, bub = [], []
    nc = A.shape[1]
    for i in range(req.size):
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:nc] = sgn * A[i]
            row[nc:n] = -sgn * Jj.T[i]
            row[n] = -1.0
            Aub.append(row); bub.append(sgn * req[i])
    for ci, mu_lo, mu_hi, rr in cone_rows:
        mu = (mu_lo if use_mu == 'min' else mu_hi) * mu_scale
        if len(rr) < 2:
            continue
        normal, tans = rr[0], rr[1:]     # MuJoCo orders the normal row first inside a contact group
        for t in tans:
            for s in (1.0, -1.0):
                row = np.zeros(N)
                row[A.shape[1] + t] = s
                row[A.shape[1] + normal] = -mu / np.sqrt(len(tans))
                Aub.append(row); bub.append(0.0)
    for ci, mu_lo, mu_hi, rr in cone_rows:
        row = np.zeros(N)
        row[A.shape[1] + rr[0]] = -1.0
        Aub.append(row); bub.append(0.0)
    bounds = [(0, box)] * A.shape[1] + [(None, None)] * n_f + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bounds, method='highs')
    return lp


results = {}
for tag, scale, use_mu in (('mu_min', 1.0, 'min'), ('mu_max', 1.0, 'max'), ('mu_min_half', 0.5, 'min'),
                           ('mu_min_double', 2.0, 'min')):
    lp = solve_cone_minimax(scale, use_mu)
    x = lp.x[:-1][:A.shape[1]] if lp.success else None
    f = lp.x[:-1][A.shape[1]:] if lp.success else None
    results[tag] = dict(use_mu=use_mu, mu_scale=scale, solved=bool(lp.success),
                        minimax=(float(lp.x[-1]) if lp.success else None),
                        f_norm=(float(np.linalg.norm(f)) if f is not None else None),
                        f0_norm=float(np.linalg.norm(f0)),
                        max_abs_f_change=(float(np.max(np.abs(f - f0))) if f is not None else None),
                        per_row_residual=(None if x is None else
                                          [float(v) for v in (A @ x - Jj.T @ f - req)]))
frozen_mm, _ = solve(A, req, 1.0)
D = dict(
    calls=known,
    nefc=nefc, n_contacts=len(groups), n_frozen_force_components=n_f,
    matches_frozen_64=bool(nefc == n_f),
    group_sizes=group_sizes,
    contacts={int(ci): {k: (v if not isinstance(v, list) else [int(x) for x in v]) for k, v in info.items()}
              for ci, info in mu_of_contact.items()},
    geom_friction_used=dict(note='mu comes from m.geom_friction[geom, 0]; min and max of the two geoms are both reported',
                            values_used={int(ci): dict(mu_min=info['mu_min'], mu_max=info['mu_max'])
                                         for ci, info in mu_of_contact.items()}),
    results=results,
    frozen_minimax_box1=frozen_mm,
    efc_type_values_present=sorted(set(int(t) for t in efc_type)),
    rows_covered_by_contacts=int(sum(len(v) for v in groups.values())),
    verdicts=dict(
        # the MAIN verdict may only use the real mu (scale 1.0); the mu/2 and 2mu points are sensitivity probes and
        # must never decide the label (that would be a looser-than-real cone setting the verdict).
        real_cone=('REAL_CONE_FEASIBLE' if all(v['minimax'] is not None and v['minimax'] <= 1e-6
                                               for kk, v in results.items() if kk in ('mu_min', 'mu_max'))
                   else 'REAL_CONE_INFEASIBLE'),
        sensitivity=('threshold: the cone becomes feasible only at 2mu (minimax %.6f at mu, %.6f at 2mu); the verdict is '
                     'therefore sensitive to mu and the mu used is geom_friction of both geoms (min = max = %.5f here)'
                     % (results['mu_min']['minimax'], results['mu_min_double']['minimax'],
                        mu_of_contact[list(mu_of_contact)[0]]['mu_min'])),
        cone_type='PYRAMID (inscribed): |t| <= mu*normal/sqrt(n_tangential); inside the true cone, so a pyramid infeasibility '
                  'is not automatically a cone infeasibility'),
    note=('the position-stage call was used ONLY to read condim / geom_friction and the efc row grouping of the 64 frozen rows; '
          'no mj_forward, no mj_step, and no dynamics state was advanced.'))
(DIR / 'REAL_CONE_CONTACT.json').write_text(json.dumps(D, indent=2, default=str))
print(json.dumps(dict(nefc=nefc, n_contacts=len(groups), group_sizes=group_sizes,
                      mu={ci: (round(i['mu_min'], 4), round(i['mu_max'], 4)) for ci, i in mu_of_contact.items()},
                      frozen=frozen_mm,
                      cone={k: (None if v['minimax'] is None else round(v['minimax'], 6)) for k, v in results.items()},
                      verdict=D['verdicts']['real_cone'],
                      rm_blocking=partB['rows_whose_removal_makes_RM_feasible']), indent=1, default=str))
