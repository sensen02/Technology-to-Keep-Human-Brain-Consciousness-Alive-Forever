"""S219 MAIN_SCRIPT: basis protocol, dropout table, MIS (order-independent), 3x3 sharing, synth-47, mu*."""
import json
import datetime
import hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s219-mis-rm-coxa-triad')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
S218 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CL = dict(model_loads=0, mj_forward=0, mj_step=0, lp_solves=0)
WEIGHT = 24.446354755878
EPS = 1e-12

prereg = dict(round='S219', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              basis={'48row': 'row r <-> DOF r (r = 0..47)', '42row': 'row r <-> DOF r+6 (r = 0..41)',
                     'traps': ['J is (64,48): contact-row x DOF; the LP must use J[:, i]',
                               'lambda has 64 entries, efc_type all 6 (PYRAMIDAL)']},
              gates=dict(free48_s1=0.059329414380136829, free48_s2=0.059329414380136829,
                         port24=0.059332809878909779, full=0.059329414380136836,
                         pinned42_s1=[3.3086208677913729, 2.3054670145597509, 1.8401142035682785],
                         triad_t=0.059329414380136836,
                         zeroing_set=['joint_RMCoxa_yaw', 'joint_RMCoxa_pitch', 'joint_RMCoxa_roll'],
                         drop_tibia_pitch_unchanged=0.05932941438013683,
                         synth47_x={'T0.5': 0.333398, 'T1.0': 0.1667, 'T0.25': 0.6668, 'T0.10': 1.000},
                         synth_product=0.16670),
              labels=['CONE_OBSTRUCTION_IS_RM_COXA_TRIAD', 'SINGLE_CAUSE_MISSING_47', 'REAL_CONE_INFEASIBLE_AT_MU1',
                      'FILTER_IMPLEMENTATION_BUG', 'CONE_AND_SIGNGAP_SAME_ROOT', 'UNDETERMINED'],
              budget=dict(model_loads_max=1, mj_forward=0, mj_step=0, lp_max=250, training_steps=0, xml_changes=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

z = np.load(S218 / 'AUTH_MATRICES_V3.npz')
c = np.load(S218 / 'CONE_STRUCTURE_V2.npz')
T = z['T']; J = z['J']; f0 = z['f0']; pool = np.asarray(z['actuator_pool'])
jn = [str(x) for x in z['row_names']]
act_names = [str(x) for x in z['actuator_names']] if 'actuator_names' in z.files else [str(a) for a in range(90)]
A42 = {k: z['A_' + k] for k in ('frozen', 'port24', 'full')}
B = {k: z['B_' + k] for k in ('frozen', 'port24', 'full')}
base = {k: z['base_' + k] for k in ('frozen', 'port24', 'full')}
A48 = {k: B[k] @ T for k in B}
NAME48 = ['root_%d' % i for i in range(6)] + jn

# ---------------------------------------------------------------- Part 0: basis names
groups = {}
for r in range(int(c['nefc'])):
    groups.setdefault(int(c['efc_id'][r]), []).append(r)
basis = dict(n48=48, n42=42, names48=NAME48,
             probes=dict(r36=NAME48[36], r41=NAME48[41], r42=NAME48[42], r47=NAME48[47],
                         r42_of_42row=jn[36], r41_of_42row=jn[35]),
             traps=dict(J_shape=list(J.shape), efc_J_equal=float(np.max(np.abs(J - c['efc_J']))),
                        f0_equal=float(np.max(np.abs(f0 - c['f0']))), nefc=int(c['nefc']),
                        ncon=int(c['ncon']), groups_4rows=bool(all(len(v) == 4 for v in groups.values())),
                        lambda_size=int(J.shape[0])))
(DIR / 'BASIS_NAMES.json').write_text(json.dumps(basis, indent=2, default=str))


def build_Jmu(mu):
    rows = []
    for ci, rr in sorted(groups.items()):
        r0, r1, r2, r3 = rr
        Jn = (J[r0] + J[r1]) / 2
        Jt1 = (J[r0] - J[r1]) / 2
        Jt2 = (J[r2] - J[r3]) / 2
        rows.extend([Jn + mu * Jt1, Jn - mu * Jt1, Jn + mu * Jt2, Jn - mu * Jt2])
    return np.array(rows)


Jmu_ok = float(np.max(np.abs(build_Jmu(1.0) - J)))


def cone(nrows, s, tag='frozen', free=True, mu=1.0, drop=(), extra_col=None):
    Amat = A48[tag] if nrows == 48 else A42[tag]
    bvec = base[tag] if nrows == 48 else base[tag][6:]
    if extra_col is not None:
        Amat = np.column_stack((Amat, (B[tag] @ extra_col)[(0 if nrows == 48 else 6):]))
    Jm = build_Jmu(mu)
    Jb = Jm if nrows == 48 else Jm[:, 6:]
    rows = [i for i in range(nrows) if i not in drop]
    n_cols = Amat.shape[1]
    n_lam = Jm.shape[0] if free else 0
    n = n_cols + n_lam
    N = n + 1
    Aub, bub = [], []
    for i in rows:
        for sgn in (1.0, -1.0):
            row = np.zeros(N)
            row[:n_cols] = sgn * Amat[i]
            if free:
                row[n_cols:n] = sgn * Jb.T[i]
            row[n] = -1.0
            Aub.append(row)
            rhs = -bvec[i] if free else (-bvec[i] - Jb.T[i] @ f0)
            bub.append(sgn * rhs)
    if free:
        for r in range(n_lam):
            row = np.zeros(N); row[n_cols + r] = -1.0
            Aub.append(row); bub.append(0.0)
        bnds = [(0, s)] * n_cols + [(None, None)] * n_lam + [(0, None)]
    else:
        bnds = [(0, s)] * n_cols + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    CL['lp_solves'] += 1
    if not lp.success:
        return None, None, None
    lam = lp.x[n_cols:n] if free else f0
    return float(lp.x[-1]), lam, lp.x[:n_cols]


# ---------------------------------------------------------------- Part A: baselines + dropout table
base_gates = {}
for tag, s in (('frozen', 1.0), ('frozen', 2.0)):
    base_gates['free48_%s_s%g' % (tag, s)] = cone(48, s, tag, True)[0]
base_gates['port24_free48_s1'] = cone(48, 1.0, 'port24', True)[0]
base_gates['full_free48_s1'] = cone(48, 1.0, 'full', True)[0]
for tag in ('frozen', 'port24', 'full'):
    base_gates['pinned42_s1_' + tag] = cone(42, 1.0, tag, False)[0]
    base_gates['pinned42_s2_' + tag] = cone(42, 2.0, tag, False)[0]
base_gates['free42_s1'] = cone(42, 1.0, 'frozen', True)[0]
dropout = {}
for r in range(48):
    t, _, _ = cone(48, 1.0, 'frozen', True, drop=(r,))
    dropout[NAME48[r]] = dict(row48=r, row42=(r - 6 if r >= 6 else None), t=t)
zeros = sorted([k for k, v in dropout.items() if v['t'] is not None and v['t'] <= 1e-9])
with open(DIR / 'DROPOUT_TABLE.csv', 'w') as f:
    f.write('row48,name,t\n')
    for k, v in dropout.items():
        f.write('%d,%s,%s\n' % (v['row48'], k, v['t']))
(DIR / 'DROPOUT_TABLE.json').write_text(json.dumps(dict(table=dropout, zeroing_set=zeros), indent=2, default=str))

# ---------------------------------------------------------------- Part B: MIS by deletion filter (3 orders)
def feasible(nrows, drop, tag='frozen'):
    t, _, _ = cone(nrows, 1.0, tag, True, drop=tuple(sorted(drop)))
    return t is not None and t <= 1e-9


def deletion_filter(nrows, order, tag='frozen'):
    cur = list(range(nrows))
    steps = []
    for i in order:
        if i not in cur:
            continue
        trial = [r for r in cur if r != i]
        if not feasible(nrows, trial, tag):
            steps.append(dict(removed=i, name=(NAME48[i] if nrows == 48 else jn[i]), t_after=cone(nrows, 1.0, tag, True,
                                                                                                 drop=tuple(sorted(trial)))[0]))
            cur = trial
    return cur, steps


mis_runs = {}
for label, order in (('forward', list(range(48))), ('reverse', list(range(47, -1, -1))),
                     ('rm_first', [42, 41, 43] + [i for i in range(48) if i not in (41, 42, 43)])):
    cur, steps = deletion_filter(48, order)
    mis_runs[label] = dict(mis_rows=cur, mis_names=[NAME48[r] for r in cur], n=len(cur), steps=steps)
triad_rows = [41, 42, 43]
mono = dict(triad_t=cone(48, 1.0, 'frozen', True, drop=())[0])
mono['triad_only'] = cone(48, 1.0, 'frozen', True, drop=tuple(i for i in range(48) if i not in triad_rows))[0]
mono['pairs'] = {('%d,%d' % (a, b)): cone(48, 1.0, 'frozen', True,
                                           drop=tuple(i for i in range(48) if i not in (a, b)))[0]
                 for a in triad_rows for b in triad_rows if a < b}
mono['singles'] = {str(a): cone(48, 1.0, 'frozen', True,
                                drop=tuple(i for i in range(48) if i != a))[0] for a in triad_rows}
supersets = {}
for extra in (0, 5, 20):
    keep = triad_rows + [i for i in range(48) if i not in triad_rows][:extra]
    supersets[str(len(keep))] = cone(48, 1.0, 'frozen', True, drop=tuple(sorted(set(range(48)) - set(keep))))[0]
mono['supersets'] = supersets
(DIR / 'MIS.json').write_text(json.dumps(dict(runs=mis_runs, monotonicity=mono,
                                              order_independent=bool(len({tuple(v['mis_rows']) for v in mis_runs.values()}) == 1)),
                                         indent=2, default=str))

# ---------------------------------------------------------------- Part C: 3x3 sharing and synth-47
pool_cols = {}
for row48 in triad_rows:
    ar = A42['frozen'][row48 - 6, :] if row48 >= 6 else None
    nz = [(int(cidx), float(ar[cidx])) for cidx in np.where(np.abs(ar) > EPS)[0]] if ar is not None else []
    pool_cols[NAME48[row48]] = [dict(column=cidx, value=val,
                                     actuators=[act_names[a] for a in range(90) if int(pool[a]) == cidx],
                                     shared=bool(sum(1 for a in range(90) if int(pool[a]) == cidx) > 1)) for cidx, val in nz]
sign_matrix = {}
for name, cols in pool_cols.items():
    sign_matrix[name] = {d['column']: ('+' if d['value'] > 0 else '-') for d in cols}
(DIR / 'COLUMN_SHARING_3X3.json').write_text(json.dumps(dict(pool_columns=pool_cols, signs=sign_matrix,
                                                              pitch_row_all_negative=bool(
                                                                  all(v == '-' for v in sign_matrix.get('joint_RMCoxa_pitch', {}).values()))
                                                              and bool(sign_matrix.get('joint_RMCoxa_pitch'))), indent=2, default=str))

synth = {}
for a in range(90):
    if int(pool[a]) >= 0:
        continue
    col = np.zeros(90); col[a] = 0.5
    t, _, x = cone(48, 1.0, 'frozen', True, extra_col=col)
    synth['#%d %s' % (a, act_names[a])] = dict(t=t, x=x[-1] if x is not None else None)
(DIR / 'SYNTH_47_SWEEP.json').write_text(json.dumps(dict(sweep=synth,
                                                         only_47_zeroes=sorted([k for k, v in synth.items()
                                                                                if v['t'] is not None and v['t'] <= 1e-9])),
                                                    indent=2, default=str))
xs = {}
for val in (0.5, 1.0, 0.25, 0.10):
    col = np.zeros(90); col[47] = val
    t, _, x = cone(48, 1.0, 'frozen', True, extra_col=col)
    xs[str(val)] = dict(t=t, x_syn=float(x[-1]) if x is not None else None,
                        product=float(x[-1] * val) if x is not None else None)
(DIR / 'SYNTH_47_XS.json').write_text(json.dumps(xs, indent=2, default=str))

# ---------------------------------------------------------------- Part D: mu*
def t_of_mu(mu, extra=None):
    return cone(48, 1.0, 'frozen', True, mu=mu, extra_col=extra)[0]


mu_grid = {str(m): t_of_mu(m) for m in (1.0, 1.25, 1.5, 1.75, 2.0)}
lo, hi = 1.0, 2.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    v = t_of_mu(mid)
    if v is not None and v <= 1e-9:
        hi = mid
    else:
        lo = mid
mu_star = hi
col47 = np.zeros(90); col47[47] = 0.5
mu_star_with47 = None
if t_of_mu(1.0, extra=col47) is not None and t_of_mu(1.0, extra=col47) <= 1e-9:
    mu_star_with47 = 1.0
(DIR / 'MU_STAR.json').write_text(json.dumps(dict(grid=mu_grid, mu_star=mu_star,
                                                  mu_star_with_47=mu_star_with47,
                                                  t_s1_equals_s2=bool(abs((cone(48, 1.0, 'frozen', True)[0] or 0) -
                                                                          (cone(48, 2.0, 'frozen', True)[0] or 0)) <= 1e-12),
                                                  Jmu_reconstruction_error=Jmu_ok), indent=2, default=str))

gates = dict(
    free48_s1=bool(abs((base_gates['free48_frozen_s1'] or 0) - 0.059329414380136829) <= 1e-12),
    free48_s2=bool(abs((base_gates['free48_frozen_s2'] or 0) - 0.059329414380136829) <= 1e-12),
    port24=bool(abs((base_gates['port24_free48_s1'] or 0) - 0.059332809878909779) <= 1e-9),
    full=bool(abs((base_gates['full_free48_s1'] or 0) - 0.059329414380136836) <= 1e-12),
    pinned42=bool(all(abs((base_gates['pinned42_s1_' + t] or 0) - e) <= 1e-9 for t, e in
                      (('frozen', 3.3086208677913729), ('port24', 2.3054670145597509), ('full', 1.8401142035682785)))),
    zeroing_set=bool(zeros == ['joint_RMCoxa_roll', 'joint_RMCoxa_pitch', 'joint_RMCoxa_yaw']),
    drop_tibia_unchanged=bool(abs((dropout['joint_RMTibia_pitch']['t'] or 0) - 0.05932941438013683) <= 1e-12),
    triad=bool(abs((mono['triad_only'] or 0) - 0.059329414380136836) <= 1e-12),
    subsets_feasible=bool(all((v or 1) <= 1e-12 for v in mono['pairs'].values()) and all((v or 1) <= 1e-12 for v in mono['singles'].values())),
    supersets_infeasible=bool(all((v or 0) > 1e-9 for v in mono['supersets'].values())),
    order_independent=bool(len({tuple(v['mis_rows']) for v in mis_runs.values()}) == 1 and mis_runs['forward']['n'] == 3),
    only_47_zeroes=bool(len([k for k, v in synth.items() if v['t'] is not None and v['t'] <= 1e-9]) == 1),
    Jmu_ok=bool(Jmu_ok <= 1e-12))
print(json.dumps(dict(dir=str(DIR), gates=gates, zeroing=zeros, baseline={k: (None if v is None else round(v, 12)) for k, v in base_gates.items()},
                      mis={k: v['mis_names'] for k, v in mis_runs.items()}, mono={k: (v if not isinstance(v, dict) else
                      {kk: (None if vv is None else round(vv, 12)) for kk, vv in v.items()}) for k, v in mono.items()},
                      synth47={k: (None if v['t'] is None else round(v['t'], 12)) for k, v in synth.items() if v['t'] is not None and v['t'] <= 1e-9},
                      n_zero_synth=len([k for k, v in synth.items() if v['t'] is not None and v['t'] <= 1e-9]),
                      xs=xs, mu=dict(grid=mu_grid, mu_star=mu_star, with47=mu_star_with47),
                      Jmu=Jmu_ok, calls=CL), indent=1, default=str))
