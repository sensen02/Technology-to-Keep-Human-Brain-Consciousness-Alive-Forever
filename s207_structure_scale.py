"""S207 Part 0 (structure pin-down) + Part A (scale scan). Offline: 0 model loads, 0 forward, 0 mj_step."""
import json
import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

RUNROOT = Path('/home/sensen/Desktop/engine修复验收/s207-scale-direction-contact-audit')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = RUNROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)

STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
BASE_GATE = 3.308620867791373
S_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

prereg = dict(
    round='S207', directory=str(DIR), written_before_runs=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    title='scale / direction / contact-lock decomposition of the frozen static infeasibility',
    structure_claims_to_pin=['decision variables are the %d columns of A, NOT the 90 actuators',
                             'f0 is a frozen constant and is never a decision variable in the S195/S196 formulation',
                             'rows 0-5 (root translation/yaw) are excluded by the [6:] slice'],
    part_A=dict(grid=S_GRID, hard_gate='s=1 must reproduce %r' % BASE_GATE,
                starget='bisection on s for minimax <= 1e-9', equivalence='A<-s*A with box=1 must match, rel diff < 1e-9'),
    part_B=dict(unbounded='x >= 0, no upper bound; rho_inf = distance of req to cone(A)',
                threshold=1e-6, cross_method='scipy.optimize.nnls on A vs req'),
    part_C=dict(variants=['full', 'base_only', 'contact_only', 'contact_zeroed']),
    part_D=dict(relaxation='f0 becomes a decision variable; grouping per contact is NOT available in the frozen artefacts '
                           '(no efc_type/contact table is stored), so the relaxations are documented explicitly and the '
                           'label is qualified accordingly'),
    gates_lexicographic=['UNDETERMINED if the baseline gate fails',
                         'DIRECTIONAL_OBSTRUCTION if rho_inf > 1e-6',
                         'CONTACT_LOCKED_INFEASIBILITY if the free-contact relaxation solves to <= 1e-6 at box=1',
                         'SCALE_GAP_QUANTIFIED if s* exists and s* > 1',
                         'FROZEN_FORMULATION_ALREADY_FEASIBLE -> UNDETERMINED (self-contradiction)'],
    budget=dict(model_loads=0, forward=0, mj_step=0, lp_max=40),
    forbidden=dict(words=['改善', 'improved'], note='no movement claim; static feasibility only'),
)
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
mp = np.load(MAP)
pool = np.asarray(mp['actuator_pool'])
A_full = Bm @ T
A = A_full[6:, :]
req_base = -base[6:]
req_contact = -(J.T @ f0)[6:]
req = req_base + req_contact

# ---------------- Part 0: structure ---------------------------------------------------------
shape_asserts = dict(B=Bm.shape, T=T.shape, base=base.shape, J=J.shape, f0=f0.shape, BT=A_full.shape,
                     A=A.shape, req=req.shape)
ok_shapes = (Bm.shape == (48, 90) and T.shape == (90, 66) and base.shape == (48,) and J.shape == (64, 48)
             and f0.shape == (64,) and A_full.shape == (48, 66) and A.shape == (42, 66) and req.shape == (42,))

# what are the 66 variables? the mapping file names the pool dimension
pool_ids = sorted({int(v) for v in pool.tolist() if int(v) >= 0})
n_slots_minus_one = int((pool < 0).sum())
nonzero_cols = int((np.abs(A).max(axis=0) > 0).sum())
col_norm = np.abs(A).sum(axis=0)

# per-row (42) and per-actuator (90) grouping into legs
rowdof = json.load(open(ROWDOF))
jn = [r['dof_name'] for r in rowdof['rows']]
leg_of_row = [n[6:8] for n in jn]                      # e.g. 'LF'
chk_legs = sorted(set(leg_of_row))

# block structure: for each of the 66 columns, which leg-blocks of rows does it touch?
blocks = {}
for c in range(A.shape[1]):
    touched = {}
    for i in range(42):
        if abs(A[i, c]) > 1e-12:
            touched[leg_of_row[i]] = touched.get(leg_of_row[i], 0) + 1
    blocks[c] = touched
n_cross_leg_columns = sum(1 for c in blocks if len(blocks[c]) > 1)
leg_rows = {L: [i for i in range(42) if leg_of_row[i] == L] for L in chk_legs}

# A restricted to each leg's rows: how much of the total |A| mass is inside the same-leg block?
same_leg_mass = 0.0
total_mass = float(np.abs(A).sum())
for c, touched in blocks.items():
    for L, cnt in touched.items():
        same_leg_mass += float(np.abs(A[leg_rows[L], c]).sum())
structure = dict(
    shapes=shape_asserts, shapes_ok=bool(ok_shapes),
    n_rows=int(A.shape[0]), n_variables=int(A.shape[1]),
    variable_identity=dict(
        answer=('the decision variables are the %d COLUMNS of A. Each column is one entry of the frozen actuator pool used by '
                'the s166 mapping T (actuator_pool has %d non-negative slot ids and %d entries equal to -1). A column is a '
                'FRACTION applied to that pool\'s activations (x in [0, s] multiplies the mapped activation), so the box bound s '
                'is a dimensionless multiplier on a pool activation, not a force and not an actuator count.'
                % (A.shape[1], len(pool_ids), n_slots_minus_one)),
        n_pool_slots=int(pool.size), n_nonnegative_slots=len(pool_ids), n_minus_one_slots=n_slots_minus_one,
        columns_with_nonzero_impact=nonzero_cols,
        columns_all_zero=int(A.shape[1] - nonzero_cols)),
    actuator_relation=('T maps the pool dimension to the 90 actuators; the 16 structurally uncommandable actuators occupy '
                       'pool slots of -1 and their rows of T are exactly zero (S206). Hence 90 actuators -> %d decision columns '
                       'and no actuator is itself a decision variable.' % A.shape[1]),
    rows_excluded_answer=('rows 0..5 of B are the six ROOT DOFs (3 translation + 3 rotation in the qvel ordering) and are '
                          'dropped by the [6:] slice because a free-floating root cannot be balanced by muscle forces: the frozen '
                          'balance state S177 already puts those six rows in its residual6 (which is <= 1e-8 by construction of '
                          'that round), and the LP only asks the 42 LEG-JOINT rows. The same convention is stated in '
                          's185_reconcile.py (row_target uses [6+i]) and used by run_s196_dual_fix_ab.py.'),
    residuals_six_root=dict(residual6=np.asarray(z['residual6'], float).tolist(),
                           max_abs=float(np.max(np.abs(np.asarray(z['residual6'], float))))),
    row_grouping=dict(legs=chk_legs, rows_per_leg={L: len(v) for L, v in leg_rows.items()},
                      n_columns_touching_more_than_one_leg=n_cross_leg_columns,
                      leg_block_mass_share=same_leg_mass / total_mass if total_mass else None,
                      interpretation=('if nearly all |A| mass sits inside its own leg block, the 42-row problem is close to six '
                                      'independent 7-row problems and the "simultaneity barrier" wording must be revised')),
    column_summary=[dict(column=c, pool_slot=(int(pool[c]) if c < pool.size else None),
                         abs_mass=float(col_norm[c]), legs_touched=sorted(blocks[c])) for c in range(A.shape[1])],
)
(DIR / 'LP_STRUCTURE.json').write_text(json.dumps(structure, indent=2, default=str))

# ---------------- Part A: scale scan --------------------------------------------------------
def solve(Amat, reqv, box, unbounded=False):
    n = Amat.shape[1]
    bnds = [(0, None)] * n if unbounded else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(42))), np.column_stack((-Amat, -np.ones(42))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), lp

rows = []
residuals_by_s = {}
for s in S_GRID:
    mm, lp = solve(A, req, s)
    x = lp.x[:-1]
    res = A @ x - req
    residuals_by_s[s] = res.tolist()
    rows.append(dict(s=s, minimax=mm, max_abs_residual=float(np.max(np.abs(res))),
                     n_rows_within_1e_6=int((np.abs(res) <= 1e-6).sum()),
                     worst_row=int(np.argmax(np.abs(res))), worst_dof_name=jn[int(np.argmax(np.abs(res)))],
                     sum_x=float(x.sum()), max_x=float(x.max())))
gate_s1 = abs(rows[1]['minimax'] - BASE_GATE) <= 1e-9

lo, hi = None, None
if rows[0]['minimax'] is not None:
    for s in S_GRID:
        pass
if rows[-1]['minimax'] is not None and rows[-1]['minimax'] <= 1e-9:
    lo, hi = 1.0, S_GRID[-1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mm, _lp = solve(A, req, mid)
        if mm is None:
            lo = mid
            continue
        if mm <= 1e-9:
            hi = mid
        else:
            lo = mid
    s_star = hi
else:
    s_star = None

# scaling equivalence: A <- s*A with box 1 must equal box s on A
equiv = {}
for s in (2.0, 8.0):
    mm_box, _ = solve(A, req, s)
    mm_scaled, _ = solve(s * A, req, 1.0)
    equiv[str(s)] = dict(box_s=mm_box, scaled_A_box1=mm_scaled,
                         rel_diff=(abs(mm_box - mm_scaled) / abs(mm_box)) if mm_box else None)
equiv_ok = all(v['rel_diff'] is not None and v['rel_diff'] < 1e-9 for v in equiv.values())

(DIR / 'SCALE_SCAN.json').write_text(json.dumps(dict(
    grid=S_GRID, gate=dict(expected=BASE_GATE, measured=rows[1]['minimax'], passed=bool(gate_s1)),
    rows=rows, s_star=s_star, exists=s_star is not None,
    equivalence=dict(checks=equiv, passed=bool(equiv_ok)),
    residuals_per_s=residuals_by_s,
    note='s is a dimensionless activation multiplier; it is NOT a force and has no physical unit'), indent=2, default=str))
with open(DIR / 'SCALE_SCAN.csv', 'w') as f:
    f.write('s,minimax,max_abs_residual,n_rows_within_1e-6,worst_row,worst_dof_name,sum_x,max_x\n')
    for r in rows:
        f.write('%g,%r,%r,%d,%d,%s,%r,%r\n' % (r['s'], r['minimax'], r['max_abs_residual'],
                                               r['n_rows_within_1e_6'], r['worst_row'], r['worst_dof_name'],
                                               r['sum_x'], r['max_x']))
print(json.dumps(dict(dir=str(DIR), structure_ok=ok_shapes,
                      n_vars=int(A.shape[1]), nonzero_cols=nonzero_cols,
                      cross_leg_cols=n_cross_leg_columns,
                      leg_mass_share=round(structure['row_grouping']['leg_block_mass_share'] or 0, 4),
                      gate=bool(gate_s1), s_star=s_star, equiv_ok=bool(equiv_ok),
                      scan=[(r['s'], None if r['minimax'] is None else round(r['minimax'], 6)) for r in rows]),
                  indent=1))
