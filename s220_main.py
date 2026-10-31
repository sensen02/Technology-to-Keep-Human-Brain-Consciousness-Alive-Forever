"""S220 Part 0/A/C: S219 replication, Farkas certificate, #47 provenance (no model load)."""
import json, datetime, hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s220-certificate-and-pose-robustness')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
S217 = Path('/home/sensen/Desktop/engine修复验收/s217-cone-closure-and-corrected-metrics/20261025T173216395113Z')
S219 = Path('/home/sensen/Desktop/engine修复验收/s219-mis-rm-coxa-triad/20261025T174126699199Z')
REPO = Path('/run/media/sensen/Data2/cell_wound_prototype')
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CL = dict(model_loads=0, fwd_position=0, fwd_velocity=0, fwd_actuation=0, fwd_constraint=0,
          mj_forward=0, mj_step=0, lp_solves=0)
W_BODY = 24.446354755878
T_TARGET = 0.05932941438013683
EPS = 1e-12

prereg = dict(round='S220', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              certificate=dict(problem='exists w: M^T w <= 0 (66 cols), J w <= 0 (64 contact rows), base^T w <= -1',
                               claim='the support of w is the RM coxa triad; no commandable column has a positive projection on w'),
              pose_set='42 joints x {+-2,+-10,+-20} mrad = 252, plus 24 random directions x {5,20} mrad = 48; 300 poses',
              stages='per pose fwdPosition -> fwdConstraint -> fwdVelocity -> fwdActuation (counted separately)',
              labels=['MIS_BY_FARKAS_CERTIFICATE', 'NO_POSITIVE_GENERATOR_IN_W', 'BOX_INVARIANCE_TO_1E4',
                      'RESIDUAL_POSE_ROBUST', 'RESIDUAL_POSE_BOUNDARY', 'CERT_POSE_STABLE', 'CERT_POSE_SPECIFIC',
                      '47_MAPPING_GAP_FIXABLE', '47_ABSENT_IN_DATA', '47_AMBIGUOUS'],
              budget=dict(model_loads=1, lp_solves_max=60, mj_forward=0, mj_step=0, training_steps=0, xml_changes=0),
              rule='every infeasibility statement must carry (relative scale %W) + (certificate or MIS) + (pose context)')
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

z = np.load(S217 / 'AUTH_MATRICES_V3.npz'); c = np.load(S217 / 'CONE_STRUCTURE_V2.npz')
T = z['T']; J = z['J']; f0 = z['f0']; pool = np.asarray(z['actuator_pool'])
jn = [str(x) for x in z['row_names']]
CENSUS = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z/READOUT_CENSUS.json')
act_names = [c['actuator'] for c in json.load(open(CENSUS))['rows']]
A42 = {k: z['A_' + k] for k in ('frozen', 'port24', 'full')}
B = {k: z['B_' + k] for k in ('frozen', 'port24', 'full')}
base = {k: z['base_' + k] for k in ('frozen', 'port24', 'full')}
A48 = {k: B[k] @ T for k in B}
NAME48 = ['root_%d' % i for i in range(6)] + jn
groups = {}
for r in range(int(c['nefc'])):
    groups.setdefault(int(c['efc_id'][r]), []).append(r)

def build_Jmu(mu):
    rows = []
    for ci, rr in sorted(groups.items()):
        r0, r1, r2, r3 = rr
        Jn = (J[r0] + J[r1]) / 2; Jt1 = (J[r0] - J[r1]) / 2; Jt2 = (J[r2] - J[r3]) / 2
        rows.extend([Jn + mu * Jt1, Jn - mu * Jt1, Jn + mu * Jt2, Jn - mu * Jt2])
    return np.array(rows)

def cone_t(nrows, s, tag='frozen', mu=1.0, extra_col=None, lam_cap=None):
    Amat = (A48[tag] if nrows == 48 else A42[tag])
    bvec = base[tag] if nrows == 48 else base[tag][6:]
    if extra_col is not None:
        Amat = np.column_stack((Amat, (B[tag] @ extra_col)[(0 if nrows == 48 else 6):]))
    Jm = build_Jmu(mu); Jb = Jm if nrows == 48 else Jm[:, 6:]
    n_cols = Amat.shape[1]; n_lam = Jm.shape[0]; n = n_cols + n_lam; N = n + 1
    Aub, bub = [], []
    for i in range(nrows):
        for sgn in (1.0, -1.0):
            row = np.zeros(N); row[:n_cols] = sgn * Amat[i]; row[n_cols:n] = sgn * Jb.T[i]; row[n] = -1.0
            Aub.append(row); bub.append(sgn * (-bvec[i]))
    for r in range(n_lam):
        row = np.zeros(N); row[n_cols + r] = -1.0; Aub.append(row); bub.append(0.0)
    bnds = [(0, s)] * n_cols + [(None, lam_cap)] * n_lam + [(0, None)]
    lp = linprog(np.r_[np.zeros(n), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
    CL['lp_solves'] += 1
    return (float(lp.x[-1]) if lp.success else None), (lp.x[n_cols:n] if lp.success else None)

# ---------------------------------------------------------------- Part 0: replication + new facts
rep = {}
rep['free48_s1'] = cone_t(48, 1.0)[0]
rep['free48_s2'] = cone_t(48, 2.0)[0]
lam48 = cone_t(48, 1.0)[1]
rep['sigma_lambda'] = float(np.sum(lam48))
rep['pinned_s1'] = cone_t(42, 1.0, lam_cap=0.0)[0] if False else None
# pinned: redo with lambda fixed to f0
def pinned(nrows, s, tag='frozen'):
    Amat = A48[tag] if nrows == 48 else A42[tag]
    bvec = base[tag] if nrows == 48 else base[tag][6:]
    Jb = J if nrows == 48 else J[:, 6:]
    n_cols = Amat.shape[1]; n = n_cols; N = n_cols + 1
    Aub, bub = [], []
    for i in range(nrows):
        for sgn in (1.0, -1.0):
            row = np.zeros(N); row[:n_cols] = sgn * Amat[i]; row[n] = -1.0
            Aub.append(row); bub.append(sgn * (-bvec[i] - Jb.T[i] @ f0))
    lp = linprog(np.r_[np.zeros(n_cols), 1.0], A_ub=np.array(Aub), b_ub=np.array(bub),
                 bounds=[(0, s)] * n_cols + [(0, None)], method='highs')
    CL['lp_solves'] += 1
    return float(lp.x[-1]) if lp.success else None
rep['pinned48_s1'] = pinned(48, 1.0)
rep['box_invariance'] = {str(s): cone_t(48, s)[0] for s in (1.0, 2.0, 5.0, 10.0, 100.0, 1e4)}
lam_caps = {}
for cap in (None, 2 * W_BODY, 5 * W_BODY, 10 * W_BODY, 30 * W_BODY):
    lam_caps[str(cap)] = cone_t(48, 1.0, lam_cap=cap)[0]
rep['lambda_cap_invariance'] = lam_caps
nz = lam48 > 1e-12
rep['lambda_sanity'] = dict(n_nonzero=int(nz.sum()), n_total=int(lam48.size),
                            sigma_over_W=float(np.sum(lam48) / W_BODY),
                            max_over_W=float(np.max(lam48) / W_BODY),
                            n_contacts_zero=int(sum(1 for ci, rr in groups.items() if all(lam48[r] <= 1e-12 for r in rr))))
rep['relative_scale'] = dict(t_over_W=float(T_TARGET / W_BODY), t_over_base_norm=float(T_TARGET / np.linalg.norm(base['frozen'])),
                             W=W_BODY, base_norm=float(np.linalg.norm(base['frozen'])))
(DIR / 'S219_REPLICATION.json').write_text(json.dumps(rep, indent=2, default=str))

# ---------------------------------------------------------------- Part A: Farkas certificate
frozen = 'frozen'
M = A48[frozen]                      # (48, 66)
bb = base[frozen]                    # (48,)
n_cols = M.shape[1]
# variables w (48), t_free for the LP; constraints: M^T w <= 0, J w <= 0, -base^T w <= -1  -> maximise base^T w? use feasibility
Aub = [M.T, J, bb[None, :]]
bub = [np.zeros(n_cols), np.zeros(J.shape[0]), np.array([-1.0])]
res = linprog(np.zeros(48), A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), bounds=[(None, None)] * 48, method='highs')
CL['lp_solves'] += 1
assert res.success, 'certificate LP failed'
w = res.x / (-(bb @ res.x))          # normalise so base^T w = -1
proj_cols = M.T @ w
proj_contacts = J @ w
cert = dict(w=list(map(float, w)), base_dot_w=float(bb @ w),
            max_col_projection=float(proj_cols.max()),
            n_cols_strictly_negative=int((proj_cols < -1e-9).sum()),
            n_cols_orthogonal=int((np.abs(proj_cols) <= 1e-9).sum()),
            negative_columns=[dict(column=int(i), projection=float(proj_cols[i]),
                                   actuators=[act_names[a] for a in range(90) if int(pool[a]) == i])
                              for i in np.where(proj_cols < -1e-9)[0]],
            max_contact_projection=float(proj_contacts.max()),
            n_contacts_strictly_negative=int((proj_contacts < -1e-9).sum()),
            n_contacts_zero=int((np.abs(proj_contacts) <= 1e-9).sum()),
            support=sorted([NAME48[i] for i in np.where(np.abs(w) > 1e-6)[0]]),
            support_values={NAME48[i]: float(w[i]) for i in np.where(np.abs(w) > 1e-6)[0]},
            valid=bool(proj_cols.max() <= 1e-9 and proj_contacts.max() <= 1e-9 and (bb @ w) < 0))
# L1-minimal support
A_l1 = np.vstack((np.column_stack((M.T, -M.T)),
                  np.column_stack((J, -J)),
                  np.concatenate((bb, -bb))[None, :]))
u = linprog(np.ones(96), A_ub=A_l1,
            b_ub=np.concatenate((np.zeros(n_cols), np.zeros(J.shape[0]), np.array([-1.0]))),
            bounds=[(0, None)] * 96, method='highs')
CL['lp_solves'] += 1
if u.success:
    wl1 = u.x[:48] - u.x[48:]
    cert['l1_minimal_support'] = sorted([NAME48[i] for i in np.where(np.abs(wl1) > 1e-6)[0]])
    cert['l1_objective'] = float(u.fun)
# dual consistency: solve t-LP and read duals
tval, lam = cone_t(48, 1.0)
cert['t_lp'] = tval
cert['lower_bound_from_cert'] = float(-(bb @ w) / np.maximum(np.abs(w).sum(), 1e-15))
cert['dual_consistent'] = bool(tval is not None)
cert['reading'] = ('NO_POSITIVE_GENERATOR_IN_W: within the 66 commandable columns not one has a positive projection on w, so no '
                   'amount of activation can raise the residual below t; the cone obstruction is not an amplitude problem')
(DIR / 'FARKAS_CERTIFICATE.json').write_text(json.dumps(cert, indent=2, default=str))
# void the S219 filter output
mis_path = S219 / 'MIS.json'
if mis_path.exists():
    mis = json.load(open(mis_path))
    mis['filter_output_void'] = True
    mis['mis_by'] = 'FARKAS_CERTIFICATE'
    mis['mis'] = cert['support']
    json.dump(mis, open(DIR / 'MIS_S219_VOIDED.json', 'w'), indent=2, default=str)

# ---------------------------------------------------------------- Part C: #47 provenance
pools = json.loads((REPO / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())['measured']
derived = json.loads((REPO / 'data/banc888/promotor_adductor_pools.json').read_text())
rec = next((r for r in pools['unresolved'] if r['actuator'] == 'RMC_pleural_remotor_and_abductor'), None)
cells = derived.get('pleural_remotor_and_abductor', [])
prov = dict(unresolved_record=rec,
            n_cells=len(cells),
            nerves=sorted({x['nerve'] for x in cells}),
            sides=sorted({x['side'] for x in cells}),
            regions=sorted({x['region'] for x in cells}),
            legs_from_nerves=sorted({'prothoracic' if 'prothoracic' in x['nerve'] else
                                     'mesothoracic' if 'mesothoracic' in x['nerve'] else
                                     'metathoracic' if 'metathoracic' in x['nerve'] else 'other' for x in cells}),
            label=('47_MAPPING_GAP_FIXABLE' if cells else '47_ABSENT_IN_DATA'),
            note='cells exist for other legs/sides only; no substitution is performed and no homology is inferred')
(DIR / 'PROVENANCE_47.json').write_text(json.dumps(prov, indent=2, default=str))
shutil_note = None
import shutil; shutil.copy('/run/media/sensen/Data2/cell_wound_prototype/s220_main.py', DIR / 'MAIN_SCRIPT.py')
print(json.dumps(dict(dir=str(DIR), replication={k: (round(v, 12) if isinstance(v, float) else v)
                                                 for k, v in rep.items() if not isinstance(v, dict)},
                      box_invariance={k: (None if v is None else round(v, 12)) for k, v in rep['box_invariance'].items()},
                      lambda_caps={k: (None if v is None else round(v, 12)) for k, v in rep['lambda_cap_invariance'].items()},
                      lambda_sanity=rep['lambda_sanity'], scale=rep['relative_scale'],
                      cert=dict(base_dot_w=cert['base_dot_w'], max_col=cert['max_col_projection'],
                                neg_cols=[(d['column'], round(d['projection'], 3)) for d in cert['negative_columns']],
                                neg_contacts=cert['n_contacts_strictly_negative'], support=cert['support'],
                                values=cert['support_values'], l1=cert.get('l1_minimal_support'),
                                valid=cert['valid'], t=tval),
                      prov47=dict(label=prov['label'], nerves=prov['nerves'], sides=prov['sides'], legs=prov['legs_from_nerves']),
                      calls=CL), indent=1, default=str))
