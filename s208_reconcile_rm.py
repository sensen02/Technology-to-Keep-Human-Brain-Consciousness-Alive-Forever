"""S208 Part 0 (variable-space reconciliation) + Part A (RM per-row signed intervals) + Part C (symbol variant).

No model load, no forward, no mj_step: everything from the frozen S177 state, S166 mapping and S204 census.
"""
import json
import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s208-rm-direction-and-real-cone')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)

STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
BASE_GATE = 3.308620867791373

prereg = dict(
    round='S208', directory=str(DIR), written_before_runs=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    title='RM row-level direction deficit, variable-space reconciliation, and the real friction cone',
    reconciles=['S207 mis-stated the pool dimension; this round must explain 74 commandable actuators, 66 slots and 16 -1 entries '
                'with one mapping or declare VARIABLE_SPACE_UNRECONCILED'],
    retraction='S207 said the S196 numbers need NO contact-lock qualification. That sentence is RETRACTED; see MANDATORY_CLAUSE.',
    labels=dict(real_cone=['REAL_CONE_FEASIBLE', 'REAL_CONE_INFEASIBLE', 'REAL_CONE_UNCOMPUTABLE'],
                rm_deficit=['RM_AMPLITUDE_DEFICIT', 'RM_SIGN_UNREACHABLE'],
                reconciliation='VARIABLE_SPACE_UNRECONCILED',
                rule='both label families are reported side by side and neither may mask the other; the MAIN label is taken from '
                     'REAL_CONE_* only, because that is the only version that relaxes nothing physical. If the reconciliation '
                     'fails, VARIABLE_SPACE_UNRECONCILED becomes the main label and everything else drops to appendix.'),
    mandatory_clause=('regardless of the main label, the conclusion must state that the S196 static infeasibility holds under the '
                      'premise that the release-pose contact force allocation is FROZEN, and that under that f0 the current '
                      'commandable system is infeasible only on RM (by index row); the S205/S206 coverage results are untouched.'),
    load_policy=('exactly ONE model load, used only to read geom_friction / geom_condim and the contact grouping of the 64 frozen '
                 'constraint rows at the S177 pose. Because the grouping is not stored in any frozen artefact, that read requires '
                 'the POSITION stage (mujoco.mj_fwdPosition) once; no mj_forward, no mj_step, no dynamics, and the call is '
                 'counted separately as position_stage_calls=1.'),
    part_D_cone=('normal >= 0 and per-contact |tangential| <= mu * normal, implemented as the inscribed friction PYRAMID '
                 '(|t_i| <= mu*normal/sqrt(2)) so that every solve stays an LP; the pyramid is INSIDE the cone, so a pyramid '
                 'infeasibility is not automatically a cone infeasibility - this is stated wherever it matters. mu is read from '
                 'geom_friction of the two geoms of each contact, and min / max / mu/2 / 2mu are all reported.'),
    budget=dict(model_loads=1, position_stage_calls=1, forward_calls=0, mj_step=0, lp_max=60),
    forbidden=dict(words=['改善', 'improved'], note='no movement claim; static feasibility only; no force unit is claimed'),
)
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

z = np.load(STATE)
Bm, base, f0, J, T = (np.asarray(z[k], float) for k in ('B', 'base', 'f0', 'J', 'T'))
pool = np.asarray(np.load(MAP)['actuator_pool'])
A = (Bm @ T)[6:, :]
req = (-(base + J.T @ f0))[6:]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
leg_of_row = [n[6:8] for n in jn]
RM_ROWS = [i for i in range(42) if leg_of_row[i] == 'RM']
assert len(RM_ROWS) == 7
RM_COLS = [c for c in range(A.shape[1]) if np.abs(A[RM_ROWS, c]).max() > 1e-12]
assert all(np.abs(np.delete(A, RM_ROWS, axis=0)[:, c]).max() <= 1e-12 for c in RM_COLS)

# ---------------------------------------------------------------- Part 0: reconciliation
cmd_rows = [c for c in CENSUS['rows'] if c['pool_size'] > 0]
empty_rows = [c for c in CENSUS['rows'] if c['pool_size'] == 0]
T_rows_zero = [i for i in range(90) if float(np.abs(T[i, :]).max()) == 0.0]
slot_to_acts = {}
for ai in range(90):
    s = int(pool[ai])
    if s >= 0:
        slot_to_acts.setdefault(s, []).append(ai)
shared = {s: v for s, v in slot_to_acts.items() if len(v) > 1}
name_of_act = {c['actuator_index']: c['actuator'] for c in CENSUS['rows']}
N_CMD = len(cmd_rows)
N_SHARED = len(shared)
ARITH_STATEMENT = (
    '90 actuators = 74 commandable + 16 structurally uncommandable. The 74 commandable actuators map onto '
    + str(len(slot_to_acts)) + ' distinct pool slots; ' + str(N_SHARED) + ' slots are shared by 2 or more actuators, and 74 - '
    + str(N_SHARED) + ' = ' + str(N_CMD - N_SHARED) + '. T is 90x' + str(T.shape[1])
    + ': row = actuator, column = pool slot. There is therefore NO 8-actuator loss: the 74 vs '
    + str(T.shape[1]) + ' difference is exactly the sharing. The S207 wording (66 entries, 16 of them -1) was wrong about the '
    + 'length: actuator_pool has 90 entries, 16 of which are -1, and the 74 non-negative entries take ' + str(len(slot_to_acts))
    + ' distinct values.')

recon = dict(
    actuator_pool_length=int(pool.size),
    n_minus_one=int((pool < 0).sum()),
    minus_one_indices=[int(i) for i in np.where(pool < 0)[0]],
    per_actuator_slot=[int(v) for v in pool.tolist()],
    n_nonnegative=int((pool >= 0).sum()),
    n_distinct_slots=len(slot_to_acts),
    slot_ids=sorted(slot_to_acts),
    T_shape=list(T.shape),
    t_rows_identically_zero=T_rows_zero,
    t_rows_zero_equals_minus_one=bool(sorted(T_rows_zero) == sorted(int(i) for i in np.where(pool < 0)[0])),
    shared_slots={int(sv): dict(n=len(v), actuators=[name_of_act[i] for i in v], actuator_indices=v)
                  for sv, v in sorted(shared.items())},
    n_shared_slots=N_SHARED,
    arithmetic=dict(n_actuators=90, n_commandable=N_CMD, n_uncommandable=len(empty_rows),
                    check_90=N_CMD + len(empty_rows), check_slots=N_CMD - N_SHARED,
                    statement=ARITH_STATEMENT),
)
cell_union_cmd = sorted({int(x) for c in cmd_rows for x in c['local_indices']})
recon['cell_reconciliation'] = dict(
    census_union_unique_cells=len(cell_union_cmd),
    census_reported=202,
    matches=bool(len(cell_union_cmd) == 202),
    note='the 202 cells belong to the 74 commandable actuators (their local_indices); the 66 slots are actuator-facing pool ids, a '
         'different object, so 202 is not expected to equal 66')
recon['reconciled'] = bool(recon['actuator_pool_length'] == 90 and recon['n_minus_one'] == 16
                           and recon['n_nonnegative'] == 74 and recon['n_distinct_slots'] == 66
                           and len(shared) == 8 and recon['t_rows_zero_equals_minus_one']
                           and recon['cell_reconciliation']['matches'])
(DIR / 'VARIABLE_SPACE_RECONCILIATION.json').write_text(json.dumps(recon, indent=2, default=str))

# ---------------------------------------------------------------- Part A: RM per-row intervals
def row_interval(row, box):
    """Signed reachable interval of one LP row.

    box is a finite activation bound s:  interval = [s*sum(min(a,0)), s*sum(max(a,0))].
    box is None means x >= 0 with NO upper bound, in which case the interval is genuinely
    unbounded on any side where the row has a coefficient of the matching sign:
      any a>0  -> hi = +inf ; any a<0 -> lo = -inf ; otherwise that side is exactly 0.
    (The first version of this function multiplied by 1.0 for the unbounded case, which silently
    returned the box=1 interval; that defect is recorded in ATTEMPT_LEDGER as S208-A2.)
    """
    a = A[row, :]
    if box is None:
        hi = float('inf') if np.any(a > 1e-15) else 0.0
        lo = float('-inf') if np.any(a < -1e-15) else 0.0
        return lo, hi
    return float(box * np.minimum(a, 0).sum()), float(box * np.maximum(a, 0).sum())

rm_rows = []
for i in RM_ROWS:
    lo1, hi1 = row_interval(i, 1.0)
    loF, hiF = row_interval(i, None)
    r = float(req[i])
    deficit_up_box = max(0.0, r - hi1)
    deficit_dn_box = max(0.0, lo1 - r)
    deficit_up_free = 0.0 if hiF == float('inf') else max(0.0, r - hiF)
    deficit_dn_free = 0.0 if loF == float('-inf') else max(0.0, loF - r)
    a = A[i, :]
    nz = a[np.abs(a) > 1e-12]
    rm_rows.append(dict(
        joint_row=int(i), dof=6 + int(i), dof_name=jn[i], required=r,
        lo_box1=lo1, hi_box1=hi1, lo_unbounded=loF, hi_unbounded=hiF,
        deficit_box1_up=float(deficit_up_box), deficit_box1_down=float(deficit_dn_box),
        deficit_free_up=float(deficit_up_free), deficit_free_down=float(deficit_dn_free),
        deficit_box1=max(deficit_up_box, deficit_dn_box), deficit_free=max(deficit_up_free, deficit_dn_free),
        kind_box1=('none' if max(deficit_up_box, deficit_dn_box) <= 1e-12
                   else ('sign' if ((r > 0 and not np.any(A[i, :] > 1e-15))
                                    or (r < 0 and not np.any(A[i, :] < -1e-15))) else 'amplitude')),
        kind_free=('none' if max(deficit_up_free, deficit_dn_free) <= 1e-12
                   else ('sign' if ((r > 0 and not np.any(A[i, :] > 1e-15))
                                    or (r < 0 and not np.any(A[i, :] < -1e-15))) else 'amplitude')),
        column_signs=dict(n_positive=int((nz > 0).sum()), n_negative=int((nz < 0).sum())),
        all_same_sign=bool(len(nz) > 0 and (np.all(nz > 0) or np.all(nz < 0))),
        sign_orientation=('+' if len(nz) and np.all(nz > 0) else ('-' if len(nz) and np.all(nz < 0) else 'mixed'))))
sign_rows = [r for r in rm_rows if r['kind_free'] == 'sign']
amp_rows = [r for r in rm_rows if r['kind_free'] == 'amplitude']
rm_summary = dict(
    leg='RM', n_rows=len(rm_rows), n_columns=len(RM_COLS), columns=RM_COLS,
    rows=rm_rows,
    deficit_vector_free=[r['required'] - np.clip(r['required'], r['lo_unbounded'], r['hi_unbounded']) if False else
                         (r['deficit_free_up'] - r['deficit_free_down']) for r in rm_rows],
    deficit_inf_norm_free=float(max(r['deficit_free'] for r in rm_rows)),
    deficit_inf_norm_box1=float(max(r['deficit_box1'] for r in rm_rows)),
    n_sign_rows_free=len(sign_rows), n_amplitude_rows_free=len(amp_rows),
    sign_rows=[r['dof_name'] for r in sign_rows], amplitude_rows=[r['dof_name'] for r in amp_rows],
    labels=dict(sign='RM_SIGN_UNREACHABLE' if sign_rows else None,
                amplitude='RM_AMPLITUDE_DEFICIT' if amp_rows else None,
                note='both are reported side by side; a row is "sign" when req falls OUTSIDE [lo,hi] on the side where the non-negative '
                     'activation cannot reach at all, and "amplitude" when more activation would reach it'))

def solve(Amat, reqv, box, lo_bound=0.0):
    n, m = Amat.shape[1], reqv.size
    bnds = [(lo_bound, None)] * n if box is None else [(lo_bound, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), (lp.x[:-1] if lp.success else None)

glob1, _ = solve(A, req, 1.0)
rm1, _x = solve(A[RM_ROWS][:, RM_COLS], req[RM_ROWS], 1.0)
rmF, _x2 = solve(A[RM_ROWS][:, RM_COLS], req[RM_ROWS], None)
gate = dict(global_box1=glob1, expected=BASE_GATE, global_ok=bool(abs(glob1 - BASE_GATE) <= 1e-9),
            rm_box1=rm1, rm_box1_ok=bool(abs(rm1 - 2.426215) <= 1e-4),
            rm_unbounded=rmF, rm_unbounded_ok=bool(abs(rmF - 1.7383659885871) <= 1e-6))
rm_summary['gates'] = gate
(DIR / 'RM_ROW_DEFICITS.json').write_text(json.dumps(rm_summary, indent=2, default=str))
with open(DIR / 'RM_ROW_DEFICITS.csv', 'w') as f:
    f.write('joint_row,dof,dof_name,required,lo_box1,hi_box1,lo_free,hi_free,deficit_box1,deficit_free,kind_box1,kind_free,sign_orientation,all_same_sign\n')
    for r in rm_rows:
        f.write('%d,%d,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s,%s,%d\n' % (
            r['joint_row'], r['dof'], r['dof_name'], r['required'], r['lo_box1'], r['hi_box1'],
            r['lo_unbounded'], r['hi_unbounded'], r['deficit_box1'], r['deficit_free'],
            r['kind_box1'], r['kind_free'], r['sign_orientation'], int(r['all_same_sign'])))

# ---------------------------------------------------------------- Part C: signed activation box
sym_rm, _ = solve(A[RM_ROWS][:, RM_COLS], req[RM_ROWS], 1.0, lo_bound=-1.0)
sym_glob, _ = solve(A, req, 1.0, lo_bound=-1.0)
sym = dict(
    variant='x in [-1, 1] (two-sided activation) on the same matrices; this is a RELAXATION with no physical mechanism '
            '(a muscle cannot have negative activation) and is reported only to separate "single-sided decoder" from "coverage"',
    rm_minimax=sym_rm, global_minimax=sym_glob,
    rm_becomes_feasible=bool(sym_rm is not None and sym_rm <= 1e-6),
    global_becomes_feasible=bool(sym_glob is not None and sym_glob <= 1e-6),
    reading=('if this variant is feasible, the RM barrier is the combination of a one-sided non-negative activation set and '
             'insufficient antagonist coverage on the blocking rows; it is NOT a physically executable fix'))
(DIR / 'SYMBOL_VARIANT.json').write_text(json.dumps(sym, indent=2, default=str))

print(json.dumps(dict(dir=str(DIR), reconciled=recon['reconciled'],
                      arithmetic=recon['arithmetic']['statement'][:120],
                      shared_slots={s: v['actuators'] for s, v in list(recon['shared_slots'].items())[:8]},
                      gates=gate, rm_sign_rows=rm_summary['sign_rows'],
                      rm_amp_rows=rm_summary['amplitude_rows'],
                      deficit_free=rm_summary['deficit_inf_norm_free'],
                      symbol=dict(rm=sym_rm, glob=sym_glob)), indent=1, default=str))
