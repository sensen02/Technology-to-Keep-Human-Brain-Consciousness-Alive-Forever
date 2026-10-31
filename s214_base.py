"""S214 Part 0 (sign-aware s table) + Part A (activation ceiling re-calibration). No model loads."""
import json
import datetime
import hashlib
import numpy as np
from pathlib import Path
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s214-activation-ceiling-and-sign-table')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)
DESK = Path('/home/sensen/Desktop/engine修复验收')
S177 = DESK / 's177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
S213 = sorted(Path(DESK / 's213-full-repair-lp').iterdir())[-1]
MAP = DESK / 's166-mapped-support/20261024T231418078692647Z/MAPPING.npz'
ROWDOF = sorted(DESK.glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
FROZEN = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
EPS = 1e-12

prereg = dict(round='S214', directory=str(DIR), written_before_runs=True,
              written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
              s_definition='s_r = req_r / max{ A[r,c] : sign(A[r,c]) = sign(req_r), |A[r,c]| > 1e-12 }; '
                           'if no same-sign column exists -> SIGN_UNREACHABLE. |req|/maxpos is forbidden.',
              predictions=dict(
                  full_repair_s_min=dict(LM=[1.30, 2.00], LH=[1.00, 1.60], RH=[1.25, 1.90], LF=[0.0, 0.40],
                                         RF=[0.0, 0.80], RM='infinite'),
                  statement='five non-RM legs become feasible at the physical ceiling; RM stays infeasible at ANY box'),
              labels=['FEASIBLE_AT_PHYSICAL_CEILING', 'INFEASIBLE_AT_PHYSICAL_CEILING',
                      'BOX1_IS_HALF_ACTIVATION', 'S213_S_TABLE_CORRECTED',
                      'GEOMETRY_GATE_WAS_POSE_CONFOUNDED', 'GEOMETRY_GATE_STANDS'],
              budget=dict(model_loads_max=4, position_stage_calls_max=8, constraint_stage_calls_max=4,
                          forward_calls=0, mj_step_calls=0, lp_max=150, training_steps=0, xml_changes=0))
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

st = np.load(S177)
B = np.asarray(st['B'], float)
base, f0, J, T = (np.asarray(st[k], float) for k in ('base', 'f0', 'J', 'T'))
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of = [n[6:8] for n in jn]
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

# ---------------------------------------------------------------- Part A(0): T structure and box semantics
nz_per_row = (np.abs(T) > 0).sum(axis=1)
nz_per_col = (np.abs(T) > 0).sum(axis=0)
tstruct = dict(n_nonzero=int((np.abs(T) > 0).sum()),
               unique_nonzero_values=sorted({float(v) for v in T[T != 0]}),
               row_nonzero_hist={int(k): int(v) for k, v in zip(*np.unique(nz_per_row, return_counts=True))},
               col_occupancy_hist={int(k): int(v) for k, v in zip(*np.unique(nz_per_col[nz_per_col > 0], return_counts=True))},
               n_columns=int(T.shape[1]), n_actuators=int(T.shape[0]))
boxes = dict(BOX1_IS_HALF_ACTIVATION=dict(
    statement='every row of T has at most one nonzero entry and every nonzero entry is exactly 0.5, so the actuator activation is '
              'u_a = 0.5 * x_c; the physical muscle activation limit u <= 1 therefore corresponds to x <= 2',
    row_max_nonzero=int(nz_per_row.max()), value=0.5,
    physical_ceiling_x=2.0, box1_activation_fraction=0.5))
(DIR / 'BOX_SEMANTICS.json').write_text(json.dumps(dict(t_structure=tstruct, semantics=boxes), indent=2, default=str))

# ---------------------------------------------------------------- Part 0: sign-aware s table
def s_row(Amat, i):
    a = Amat[i, :]
    r = req[i]
    if abs(r) <= EPS:
        return 0.0
    same = a[np.sign(a) == np.sign(r)]
    same = same[np.abs(same) > EPS]
    if same.size == 0:
        return None
    return float(abs(r) / np.max(np.sign(r) * same))

def table(Amat):
    out = {}
    for i in range(42):
        out[jn[i]] = dict(row=i, req=float(req[i]), s=s_row(Amat, i))
    return out

A_frozen = (B @ T)[6:, :]
tables = {'frozen': table(A_frozen)}
# port24 / full matrices come from S213 archival artefacts (rebuilt below from the stored variant xml if present)
s213csv = S213 / 'FULL_REPAIR_LP.csv'
diff_rows = []
if s213csv.exists():
    import csv
    with open(s213csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        nm = r['dof_name']
        old = r['frozen_s']
        new = tables['frozen'][nm]['s']
        try:
            oldv = None if old in ('None', '', 'null') else float(old)
        except ValueError:
            oldv = None
        if (oldv is None) != (new is None) or (oldv is not None and new is not None and abs(oldv - new) > 1e-6):
            diff_rows.append(dict(row_no=int(r['joint_row']), dof_name=nm, s213_value=oldv, correct_value=new))
(DIR / 'S213_S_TABLE_DIFF.csv').write_text('row_no,dof_name,s213_value,correct_value\n' +
                                           '\n'.join('%d,%s,%s,%s' % (d['row_no'], d['dof_name'], d['s213_value'],
                                                                      d['correct_value']) for d in diff_rows) + '\n')
(DIR / 'S_TABLE_SIGN_AWARE.json').write_text(json.dumps(dict(definition=prereg['s_definition'], eps=EPS,
                                                             frozen=tables['frozen'], diff_vs_s213=diff_rows),
                                                        indent=2, default=str))
with open(DIR / 'S_TABLE_SIGN_AWARE.csv', 'w') as f:
    f.write('row,dof_name,req,sign_aware_s,unreachable\n')
    for nm, v in tables['frozen'].items():
        sv = '' if v['s'] is None else ('%.9g' % v['s'])
        f.write('%d,%s,%.9g,%s,%d\n' % (v['row'], nm, v['req'], sv, int(v['s'] is None)))
over1 = {nm: v['s'] for nm, v in tables['frozen'].items() if v['s'] is not None and v['s'] > 1.0}
unreach = [nm for nm, v in tables['frozen'].items() if v['s'] is None]

# ---------------------------------------------------------------- Part A: minimax + s_min per leg
def solve(Amat, reqv, box):
    n, mm = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(mm))), np.column_stack((-Amat, -np.ones(mm))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None)

def leg_rows(L):
    return [i for i in range(42) if leg_of[i] == L]

def leg_cols(Amat, L):
    rows = leg_rows(L)
    return [c for c in range(Amat.shape[1]) if np.abs(Amat[rows, c]).max() > EPS
            and np.abs(np.delete(Amat, rows, axis=0)[:, c]).max() <= EPS]

def leg_sys(Amat, L):
    rows, cols = leg_rows(L), leg_cols(Amat, L)
    return Amat[rows][:, cols], req[rows]

def s_min(Amat, L, lo=1e-6, hi=64.0):
    Asub, rsub = leg_sys(Amat, L)
    f = lambda s: solve(Asub, rsub, s)
    top = f(hi)
    if top is None or top > 1e-9:      # explicit None check: a 0.0 result must NOT be treated as missing
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if v is not None and v <= 1e-9:
            hi = mid
        else:
            lo = mid
    return float(hi)

res = {'frozen': {}}
for L in LEGS:
    Asub, rsub = leg_sys(A_frozen, L)
    res['frozen'][L] = dict(minimax_s1=solve(Asub, rsub, 1.0), minimax_s2=solve(Asub, rsub, 2.0), s_min=s_min(A_frozen, L))
(DIR / 'S_MIN_TABLE.json').write_text(json.dumps(dict(frozen=res['frozen'], predictions=prereg['predictions']),
                                                  indent=2, default=str))
with open(DIR / 'S_MIN_TABLE.csv', 'w') as f:
    f.write('leg,frozen_minimax_s1,frozen_minimax_s2,frozen_s_min\n')
    for L in LEGS:
        r = res['frozen'][L]
        f.write('%s,%s,%s,%s\n' % (L, r['minimax_s1'], r['minimax_s2'], r['s_min']))
print(json.dumps(dict(dir=str(DIR), t_structure=tstruct,
                      n_over1=len(over1), over1={k: round(v, 6) for k, v in over1.items()},
                      unreachable=unreach, n_diff_rows=len(diff_rows),
                      diff=[(d['dof_name'], d['s213_value'], None if d['correct_value'] is None else round(d['correct_value'], 6)) for d in diff_rows],
                      frozen=res['frozen']), indent=1, default=str))
