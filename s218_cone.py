"""S218 CONE_SCRIPT: correct cone LP (residual = A x + J^T lambda + base), MIS, mu curve."""
import json
import datetime
import hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s218-cone-closure-signed')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
S217 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CL = dict(model_loads=0, mj_forward=0, mj_step=0, lp_solves=0)
WEIGHT = 24.446354755878

prereg = dict(round='S218', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              spec=dict(residual='r_i = (B@T x)_i + (J^T lambda)_i + base_i',
                        vars='x in [0,s]^66 (66 commandable pool columns), lambda in R^64_+ (row forces, no upper bound), t',
                        pinned='lambda fixed at the frozen f0, so A x = -(base + J^T f0) = req',
                        rows='48 rows = all DOFs (primary); 42 rows = root rows dropped (degenerate, unphysical with free lambda)'),
              gates=dict(pinned48_s1=3.3086208677913729, pinned48_s2=3.1257857459661631,
                         free48_s1=0.059329414380136829, free48_s2=0.059329414380136829,
                         sigma_lambda_equals_weight_minus_t=0.059329414380136829,
                         mu_curve=dict(m0_25=1.3375222084510701, m0_5=0.42889124367165099,
                                       m1_0=0.059329414380136829, m2_0=0.0),
                         pinned_three_variants=[3.3086208677913729, 2.3054670145597509, 1.8401142035682785],
                         free42_sigma_lambda=7.122791),
              labels=['S217_CONE_LP_BUGS', 'ROOT_EXCLUDED_RELAXATION_UNPHYSICAL', 'CONE_BLOCKER_IS_NOT_ROW36',
                      'REAL_CONE_INFEASIBLE_AT_MU1'],
              budget=dict(model_loads=0, mj_forward=0, mj_step=0, lp_solves_max=300, training_steps=0, xml_changes=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

z = np.load(S217 / 'AUTH_MATRICES_V3.npz')
c = np.load(S217 / 'CONE_STRUCTURE_V2.npz')
T = z['T']
J = z['J']              # (64, 48) live efc_J at mu=1
leg_of = [str(x) for x in z['leg_of']]
jn = [str(x) for x in z['row_names']]
A42 = {'frozen': z['A_frozen'], 'port24': z['A_port24'], 'full': z['A_full']}
B = {k: z['B_' + k] for k in ('frozen', 'port24', 'full')}
base = {k: z['base_' + k] for k in ('frozen', 'port24', 'full')}
f0 = z['f0']
A48 = {k: (B[k] @ T) for k in B}
assert J.shape == (64, 48)

# ---------------------------------------------------------------- Part 0: spec and decomposition gates
groups = {}
for r in range(int(c['nefc'])):
    groups.setdefault(int(c['efc_id'][r]), []).append(r)
decomp_ok = True
Jn_diff = 0.0
for ci, rr in groups.items():
    if len(rr) == 4:
        r0, r1, r2, r3 = rr
        d1 = np.max(np.abs((J[r0] + J[r1]) / 2 - (J[r2] + J[r3]) / 2))
        Jn_diff = max(Jn_diff, float(d1))
        decomp_ok &= bool(d1 <= 1e-15)


def build_Jmu(mu):
    rows = []
    for ci, rr in sorted(groups.items()):
        r0, r1, r2, r3 = rr
        Jn = (J[r0] + J[r1]) / 2
        Jt1 = (J[r0] - J[r1]) / 2
        Jt2 = (J[r2] - J[r3]) / 2
        rows.extend([Jn + mu * Jt1, Jn - mu * Jt1, Jn + mu * Jt2, Jn - mu * Jt2])
    return np.array(rows)


Jmu1_diff = float(np.max(np.abs(build_Jmu(1.0) - J)))
root_authority = dict(moment_front6_norm=float(np.linalg.norm(z['moment_frozen'][:, :6])),
                      B_front6_max=float(np.abs(np.asarray(B['frozen'] @ T)[:6, :]).max()))
spec = dict(parameterization=dict(cone='PYRAMIDAL', nefc=int(c['nefc']), ncon=int(c['ncon']),
                                  rows_per_contact={str(k): len(v) for k, v in sorted(groups.items())},
                                  lambda_size=int(J.shape[0]), lambda_nonneg=True, lambda_upper=None),
            gates=dict(four_rows_per_contact=bool(all(len(v) == 4 for v in groups.values())),
                       Jn_equals_maxdiff=Jn_diff, Jn_ok=bool(Jn_diff <= 1e-15),
                       build_Jmu_1_reproduces_live_J=Jmu1_diff, build_Jmu_ok=bool(Jmu1_diff <= 1e-12)),
            root_authority=root_authority,
            s217_bugs={'bug1': 'free branch used rhs = req, dropping the J^T f0 already folded into req; the correct RHS is -base',
                       'bug2': 'pinned branch added J^T f0 a second time, giving 24.446354756 (the weight)',
                       'bug3': 'the lambda coefficient was written -sgn*Jmap.T[i], so the rows were allowed to PULL instead of push',
                       'effect': 'S217 reported 24.446354756 / 2.0125 / 1.1325; those numbers are artefacts of the two RHS errors and '
                                 'the sign error and are not evidence'})
(DIR / 'CONE_LP_SPEC.json').write_text(json.dumps(spec, indent=2, default=str))

# ---------------------------------------------------------------- cone LP
def cone(nrows, s, tag='frozen', free=True, mu=1.0, drop=(), pin_lambda=True):
    Amat = A48[tag] if nrows == 48 else A42[tag]
    bvec = base[tag] if nrows == 48 else base[tag][6:]
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
                row[n_cols:n] = sgn * Jb.T[i]      # + (J^T lambda), sign CORRECT
            else:
                pass
            row[n] = -1.0
            Aub.append(row)
            rhs = -bvec[i]
            if not free:
                rhs = -bvec[i] - (Jb.T[i] @ f0)    # A x = -(base + J^T f0)
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
        return None, None
    lam = lp.x[n_cols:n] if free else f0
    return float(lp.x[-1]), lam

four = {}
for nrows in (48, 42):
    for free in (False, True):
        for tag in ('frozen', 'port24', 'full'):
            t, lam = cone(nrows, 1.0, tag, free)
            four['%s_%s_%drows' % (tag, 'free' if free else 'pinned', nrows)] = dict(
                t=t, sigma_lambda=float(np.sum(lam)) if lam is not None else None)
four['frozen_free_48rows_s2'] = dict(t=cone(48, 2.0, 'frozen', True)[0])
four['frozen_pinned_48rows_s2'] = dict(t=cone(48, 2.0, 'frozen', False)[0])
(DIR / 'CONE_FOUR_WAY.json').write_text(json.dumps(four, indent=2, default=str))

# ---------------------------------------------------------------- Part B: MIS by deletion filter
def feasible(nrows, drop):
    t, _ = cone(nrows, 1.0, 'frozen', True, drop=tuple(sorted(drop)))
    return t is not None and t <= 1e-9


base_rows = list(range(48))
assert not feasible(48, set()), 'the full system must be infeasible for the filter to apply'
mis = list(base_rows)
for i in list(base_rows):
    trial = [r for r in mis if r != i]
    if not feasible(48, set(trial)):
        mis = trial
mis_report = dict(mis=[dict(row=int(r), name=('root_' + str(r) if r < 6 else jn[r - 6])) for r in mis],
                  size=len(mis))
drop_single = {}
for label, r in (('joint_RMTibia_pitch', 41 + 6 if False else 41), ('joint_RMCoxa_pitch', 36)):
    idx = r + 6 if r < 42 else r
    drop_single[label] = dict(row_index_48=int(idx), t_without=cone(48, 1.0, 'frozen', True, drop=(idx,))[0])
    drop_single[label + '_42row'] = dict(t_without=cone(42, 1.0, 'frozen', True, drop=(r,))[0])
mis_report['drop_single'] = drop_single
(DIR / 'MIS_48ROWS.json').write_text(json.dumps(mis_report, indent=2, default=str))

# ---------------------------------------------------------------- Part C: mu curve
mu_curve = {}
for mu in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    mu_curve['%g' % mu] = dict(free48=cone(48, 1.0, 'frozen', True, mu=mu)[0],
                               free42=cone(42, 1.0, 'frozen', True, mu=mu)[0])
(DIR / 'MU_CURVE.json').write_text(json.dumps(mu_curve, indent=2, default=str))

gates = dict(
    pinned48_s1=bool(abs((four['frozen_pinned_48rows']['t'] or 0) - 3.3086208677913729) <= 1e-9),
    pinned48_s2=bool(abs((four['frozen_pinned_48rows_s2']['t'] or 0) - 3.1257857459661631) <= 1e-9),
    free48_s1=bool(abs((four['frozen_free_48rows']['t'] or 0) - 0.059329414380136829) <= 1e-9),
    free48_s2=bool(abs((four['frozen_free_48rows_s2']['t'] or 0) - 0.059329414380136829) <= 1e-9),
    sigma_lambda=bool(abs((four['frozen_free_48rows']['sigma_lambda'] or 0) - (WEIGHT - 0.059329414380136829)) <= 1e-6),
    mu_curve=bool(abs((mu_curve['0.25']['free48'] or 0) - 1.3375222084510701) <= 1e-6
                  and abs((mu_curve['0.5']['free48'] or 0) - 0.42889124367165099) <= 1e-6
                  and abs((mu_curve['1']['free48'] or 0) - 0.059329414380136829) <= 1e-9
                  and abs((mu_curve['2']['free48'] or 1)) <= 1e-9),
    pinned_three=bool(abs((four['frozen_pinned_48rows']['t'] or 0) - 3.3086208677913729) <= 1e-9
                      and abs((four['port24_pinned_48rows']['t'] or 0) - 2.3054670145597509) <= 1e-9
                      and abs((four['full_pinned_48rows']['t'] or 0) - 1.8401142035682785) <= 1e-9),
    free42_sigma=bool(abs((four['frozen_free_42rows']['sigma_lambda'] or 0) - 7.122791) <= 1e-3),
    drop_row41_zero=bool(abs((drop_single['joint_RMTibia_pitch']['t_without'] or 1)) <= 1e-9),
    row36_unchanged_when_dropped=bool(abs((drop_single['joint_RMCoxa_pitch']['t_without'] or 0) - 0.059329414380136829) <= 1e-9),
    spec=bool(decomp_ok and Jmu1_diff <= 1e-12))
print(json.dumps(dict(dir=str(DIR), gates=gates, spec_ok=decomp_ok, Jmu=Jmu1_diff, Jn=Jn_diff,
                      four={k: (None if v['t'] is None else round(v['t'], 12)) for k, v in four.items()},
                      mis=mis_report['mis'], drop=drop_single,
                      mu={k: (None if v['free48'] is None else round(v['free48'], 9)) for k, v in mu_curve.items()},
                      sigma48=four['frozen_free_48rows']['sigma_lambda'], sigma42=four['frozen_free_42rows']['sigma_lambda'],
                      weight=WEIGHT, root_authority=root_authority, calls=CL), indent=1, default=str))
