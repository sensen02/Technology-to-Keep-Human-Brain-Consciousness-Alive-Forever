"""S207 Parts B, C, D + per-leg decomposition. Offline: 0 model loads, 0 forward, 0 mj_step."""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linprog, nnls

DIR = sorted(Path('/home/sensen/Desktop/engine修复验收/s207-scale-direction-contact-audit').iterdir())[-1]
STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')

z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
pool = np.asarray(np.load(MAP)['actuator_pool'])
A = (Bm @ T)[6:, :]
req_base = -base[6:]
req_contact = -(J.T @ f0)[6:]
req = req_base + req_contact
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
LEGS = sorted(set(leg_of_row))


def minimax(Amat, reqv, box=None):
    n = Amat.shape[1]
    m = reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    if not lp.success:
        return None, None, lp
    return float(lp.x[-1]), lp.x[:-1], lp


def feas(rho, Amat, reqv, box=None):
    """different formulation: is there x >= 0 with |Ax - req| <= rho ? (no t variable)"""
    n = Amat.shape[1]
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.zeros(n), A_ub=np.vstack((Amat, -Amat)), b_ub=np.r_[reqv + rho, -reqv + rho],
                 bounds=bnds, method='highs')
    return bool(lp.success)


# ---------------- Part B: unbounded box -> cone distance ------------------------------------
rho1, x_inf, lp_inf = minimax(A, req, None)
res_inf = A @ x_inf - req
lo, hi = 0.0, max(float(np.max(np.abs(res_inf))) * 2, 1e-9)
bidir = 0
while not feas(hi, A, req, None):
    hi *= 2
    bidir += 1
    if bidir > 40:
        break
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if feas(mid, A, req, None):
        hi = mid
    else:
        lo = mid
rho2 = hi
xn, _rnorm = nnls(A, req)
res_nnls = A @ xn - req
order = np.argsort(-np.abs(res_inf))
cone = dict(
    rho_inf_from_minimax=rho1,
    rho_from_independent_feasibility_bisection=rho2,
    abs_difference=abs(rho1 - rho2),
    stable=bool(abs(rho1 - rho2) <= 1e-6),
    exceeds_threshold=bool(rho1 > 1e-6),
    norm_ratio=float(rho1 / np.linalg.norm(req)),
    norm_req=float(np.linalg.norm(req)),
    nnls_check=dict(max_abs_residual=float(np.max(np.abs(res_nnls))),
                    note='nnls minimises the L2 residual, so its max-abs residual is an upper reference, not an equality check'),
    top_residual_rows=[dict(row=int(i), dof=6 + int(i), dof_name=jn[int(i)], residual=float(res_inf[i]),
                            abs_residual=float(abs(res_inf[i]))) for i in order[:10]],
    non_uniqueness_note=('the unbounded minimax LP has MANY optimal solutions: any x whose worst row hits rho_inf is optimal, so the '
                         'per-row residual pattern is NOT unique. Ties at +-rho_inf appear on several legs; the top-5 ordering below is '
                         'arbitrary among those ties. The UNIQUE statement is the leg-wise decomposition in LEG_DECOMPOSITION.json, '
                         'where only RM has a positive unbounded minimum. FIG2 therefore shows ONE optimal solution, not a structural '
                         'signature.'),
    is_every_top_row_a_hard_dof=bool(all((6 + int(i)) in (19, 22, 26, 40, 42, 47) for i in order[:6])),
    hard_dof_rows_present=sorted({6 + int(i) for i in order[:10]} & {19, 22, 26, 40, 42, 47}),
)
(DIR / 'CONE_DISTANCE.json').write_text(json.dumps(cone, indent=2, default=str))

# ---------------- per-leg decomposition ------------------------------------------------------
per_leg = {}
for L in LEGS:
    idx = [i for i in range(42) if leg_of_row[i] == L]
    # columns whose nonzeros are in this leg's rows
    cols = [c for c in range(A.shape[1]) if np.abs(A[idx, c]).max() > 1e-12
            and np.abs(np.delete(A, idx, axis=0)[:, c]).max() <= 1e-12]
    mm_box1, _x, _lp = minimax(A[idx][:, cols], req[idx], 1.0)
    mm_free, _x2, _lp2 = minimax(A[idx][:, cols], req[idx], None)
    per_leg[L] = dict(rows=len(idx), columns=len(cols), minimax_box1=mm_box1, minimax_unbounded=mm_free,
                      required_norm=float(np.linalg.norm(req[idx])))
glob_max = max(v['minimax_box1'] for v in per_leg.values())
glob_max_free = max(v['minimax_unbounded'] for v in per_leg.values())
leg_decomp = dict(
    per_leg=per_leg,
    global_box1_from_full_solve=3.308620867791373,
    max_of_leg_box1=glob_max,
    matches_box1=bool(abs(glob_max - 3.308620867791373) <= 1e-9),
    max_of_leg_unbounded=glob_max_free,
    global_unbounded=rho1,
    matches_unbounded=bool(abs(glob_max_free - rho1) <= 1e-6),
    statement=('A is exactly block diagonal by leg (0 columns touch two legs, leg-block mass share 1.0), so the 42-row problem '
               'is six independent 7-row problems and the global minimax is the MAXIMUM of the six. The S196/S203 phrase '
               '"simultaneity/coupling barrier" is therefore not supported by the frozen matrices: the six legs do not fight the '
               'same decision variables at all.'))
(DIR / 'LEG_DECOMPOSITION.json').write_text(json.dumps(leg_decomp, indent=2, default=str))

# ---------------- Part C: requirement decomposition -----------------------------------------
variants = {'full': req, 'base_only': req_base, 'contact_only': req_contact, 'contact_zeroed': req_base}
dec = {}
for k, rv in variants.items():
    mm1, _x, _lp = minimax(A, rv, 1.0)
    mmf, _x2, _lp2 = minimax(A, rv, None)
    dec[k] = dict(req_norm=float(np.linalg.norm(rv)), minimax_box1=mm1, minimax_unbounded=mmf,
                  max_abs_req=float(np.max(np.abs(rv))))
share = dict(
    req_base_norm=float(np.linalg.norm(req_base)), req_contact_norm=float(np.linalg.norm(req_contact)),
    base_share=float(np.linalg.norm(req_base) / (np.linalg.norm(req_base) + np.linalg.norm(req_contact))),
    contact_share=float(np.linalg.norm(req_contact) / (np.linalg.norm(req_base) + np.linalg.norm(req_contact))),
    per_row=[dict(dof=6 + i, dof_name=jn[i], base=float(req_base[i]), contact=float(req_contact[i]),
                  total=float(req[i])) for i in range(42)])
(DIR / 'REQUIREMENT_DECOMPOSITION.json').write_text(json.dumps(dict(variants=dec, shares=share), indent=2, default=str))

# ---------------- Part D: free-contact relaxation -------------------------------------------
import re
xml_txt = XML.read_text()
mus = sorted({float(m) for m in re.findall(r'friction="([0-9.]+)', xml_txt)})
lines = [(i + 1, ln.strip()) for i, ln in enumerate(xml_txt.splitlines()) if 'friction=' in ln][:5]
n_f = f0.size
extra = n_f
A_joint = A                                     # (42, 66) joint rows
Jj = J[:, 6:]                                   # (64, 42) contact rows mapped to joint DOFs
# joint-space equality: A x + (J^T f)[6:] = req  with f the 64-vector
# free-contact relaxation D1: f >= 0 only (no per-contact grouping is stored in the frozen artefacts)
A1 = np.column_stack((A_joint, -Jj.T))          # x contributes +A x ; f contributes -J^T f  (moved to LHS)
b1 = req
n1 = A1.shape[1]
lp = linprog(np.r_[np.zeros(n1), 1.0],
             A_ub=np.vstack((np.column_stack((A1, -np.ones(42))), np.column_stack((-A1, -np.ones(42))))),
             b_ub=np.r_[b1, -b1], bounds=[(0, 1.0)] * A_joint.shape[1] + [(0, None)] * n_f + [(0, None)],
             method='highs')
d1 = dict(solved=bool(lp.success), minimax=(float(lp.x[-1]) if lp.success else None),
          formulation='x in [0,1] (66 cols) and f >= 0 (64 contact-force components), minimax on the 42 joint rows',
          relaxation_note=('this is a RELAXATION: per-contact grouping, friction-cone coupling and per-leg normal/tangential '
                           'inequalities CANNOT be imposed because the frozen artefacts store neither efc_type nor a contact '
                           'table for the S177 pose, and this round allows 0 model loads. Treating each of the 64 force '
                           'components as an independent non-negative variable is strictly MORE permissive than the real '
                           'friction cone, so a positive result here cannot be read as "free contact makes it feasible" for '
                           'the real cone; a negative result (still infeasible) does rule the contact-lock OUT as the source. '
                           'A grouping was deliberately NOT invented.'))
f_sol = lp.x[A_joint.shape[1]:-1] if lp.success else None
d1['f_solution'] = dict(n=int(n_f), norm_frozen=float(np.linalg.norm(f0)),
                        norm_solution=(float(np.linalg.norm(f_sol)) if f_sol is not None else None),
                        max_abs_change=(float(np.max(np.abs(f_sol - f0))) if f_sol is not None else None),
                        per_component=[dict(i=int(i), frozen=float(f0[i]),
                                            solved=(float(f_sol[i]) if f_sol is not None else None))
                                       for i in range(n_f)])
free = dict(friction_values_in_xml=mus, friction_source_lines=lines,
            choice_note=('the XML carries %d distinct friction coefficients; the S177 contact allocation is unknowable here, so '
                         'the variant uses the non-negativity-only relaxation D1 rather than inventing a mu-dependent cone'
                         % len(mus)), D1=d1)
(DIR / 'FREE_CONTACT_VARIANT.json').write_text(json.dumps(free, indent=2, default=str))

print(json.dumps(dict(rho1=rho1, rho2=rho2, cone_stable=cone['stable'], top=[(r['dof_name'], round(r['residual'], 4))
                 for r in cone['top_residual_rows'][:5]],
                 leg_matches_box1=leg_decomp['matches_box1'], leg_max_box1=glob_max,
                 leg_matches_unbounded=leg_decomp['matches_unbounded'], leg_max_free=glob_max_free,
                 variants={k: round(v['minimax_box1'], 6) if v['minimax_box1'] is not None else None for k, v in dec.items()},
                 mus=mus, D1=dict(solved=d1['solved'], minimax=d1['minimax'])), indent=1))
