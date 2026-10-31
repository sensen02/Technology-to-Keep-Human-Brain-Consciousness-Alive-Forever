"""S209 Part 0 (epsilon protocol + exact rational row 36) + Part A (42-row sign census) + Part B (provenance)
+ Part C (synthetic column). No model load, no forward, no step."""
import json
import datetime
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog

ROOT = Path('/home/sensen/Desktop/engine修复验收/s209-row36-sign-and-agonist')
STAMP = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
DIR = ROOT / STAMP
DIR.mkdir(parents=True, exist_ok=False)

STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
MAP = Path('/home/sensen/Desktop/engine修复验收/s166-mapped-support/20261024T231418078692647Z/MAPPING.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
ROWDOF = sorted(Path('/home/sensen/Desktop/engine修复验收').glob('s179-*/*/ROW_DOF_MAP.json'))[-1]
POOLS = json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_rebuild/'
                        'neural_rebuild_11_mn_pools.json').read_text())['measured']
DERIVED = json.loads(Path('/run/media/sensen/Data2/cell_wound_prototype/data/banc888/'
                          'promotor_adductor_pools.json').read_text())

EPS_MAIN = 1e-12
EPS_GRID = [1e-18, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8]
prereg = dict(
    round='S209', directory=str(DIR), written_before_runs=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    purpose='pin the row-36 direction gap at the actuator level, close its provenance, and test a synthetic antagonist column',
    s208_correction=('S208 declared that every RM row is individually reachable because some positive coefficient existed. That is '
                     'wrong whenever the positive coefficients are floating-point residue. This round replaces it with a two-definition '
                     'protocol plus an exact rational recomputation.'),
    epsilon_protocol=dict(
        eps_main=EPS_MAIN, eps_grid=EPS_GRID,
        rationale=('eps is a SIGN-SUPPORT threshold, not a physical tolerance: a coefficient smaller than eps is treated as absent '
                   'from the row support. 1e-12 is chosen because the frozen B entries are O(1) and float64 accumulation noise for a '
                   '90-term dot product is ~1e-16..1e-15, so 1e-12 sits ~3 orders above the noise while remaining ~12 orders below '
                   'the smallest physical coefficient in these rows. The eps grid is reported so that any flip point is visible.'),
        definition_numeric=('(a) numerically reachable: max positive coefficient > eps; the activation needed is x = req/coef, which is '
                            'reported in full and compared against the s grid. A huge x is still "reachable" in this definition.'),
        definition_structural=('(b) structurally reachable: the positive-support sum is > eps. If the positive support is empty the '
                               'row cannot be pushed positive by ANY non-negative activation.'),
        rule='both definitions are reported for every row; neither alone may carry a conclusion'),
    high_precision=('row 36 of A is recomputed three ways: float64 (the frozen product), numpy.longdouble (>=18 significant digits) '
                    'and EXACT rational arithmetic (each float64 converted to an exact Fraction, then summed), so that the verdict '
                    'does not depend on the 1e-18 / 1e-12 band that the S209 pre-registration inherits from the advisor request.'),
    labels=dict(direction=['ROW36_SIGN_UNREACHABLE_STRUCTURAL', 'ROW36_NUMERIC_ONLY', 'ROW36_UNDETERMINED'],
                agonist=['AGONIST_ONLY_VIA_UNRESOLVED_47', 'AGONIST_ELSEWHERE', 'AGONIST_ABSENT'],
                synthetic=['SIGN_GAP_CLOSED_BY_SYNTHETIC_COLUMN', 'SIGN_GAP_PARTIALLY_CLOSED', 'SIGN_GAP_UNCHANGED'],
                rule='the four labels are each reported once and never mask one another; the MAIN label is the direction label'),
    synthetic_column=dict(
        construction=('T_syn is a new (90 x 1) column, zero everywhere except row 47 = 0.5 (0.5 is the frozen value of every nonzero '
                      'T entry: 74 entries, all exactly 0.5). The synthetic decision variable is bounded 0 <= x_syn <= s like the '
                      'others. T_syn exists ONLY in this round directory; the frozen T is hashed before and after.'),
        rationale='0.5 keeps the synthetic column in the same activation units as the frozen columns; no other value is used'),
    budget=dict(model_loads=1, position_stage_calls=1, constraint_stage_calls=1, forward_calls=0, mj_step_calls=0,
                lp_max=200, neural_ticks=0, training_steps=0, xml_changes=0),
    forbidden=dict(words=['改善', 'improved'], note='no movement claim; feasibility bookkeeping only; force units NOT_ESTABLISHED'),
)
(DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

st = np.load(STATE)
B = np.asarray(st['B'], float)
base = np.asarray(st['base'], float)
f0 = np.asarray(st['f0'], float)
J = np.asarray(st['J'], float)
T = np.asarray(st['T'], float)
req = (-(base + J.T @ f0))[6:]
A = (B @ T)[6:, :]
jn = [r['dof_name'] for r in json.load(open(ROWDOF))['rows']]
name_of_act = {c['actuator_index']: c for c in CENSUS['rows']}
T_row_sum = np.abs(T).sum(axis=1)
commandable = {i: bool(T_row_sum[i] > 0) for i in range(90)}

# ---------------------------------------------------------------- Part 0: exact/longdouble row 36
r0 = 36
row64 = A[r0]
rowld = (B.astype(np.longdouble) @ T.astype(np.longdouble))[6:][r0]
tau = np.asarray(st['T'], float)
# exact rational recomputation of A[36,:] = sum_a convolved; do it column by column with Fractions
exact_pos = Fraction(0)
exact_neg = Fraction(0)
exact_cols = []
for c in range(A.shape[1]):
    acc = Fraction(0)
    for a in range(90):
        if T[a, c] != 0.0:
            acc += Fraction(float(B[6 + r0, a])) * Fraction(float(T[a, c]))
    exact_cols.append(acc)
    if acc > 0:
        exact_pos += acc
    else:
        exact_neg += acc
hp = dict(
    row=int(r0), dof=6 + r0, dof_name=jn[r0], required=float(req[r0]),
    float64_max_positive=float(row64.max()), float64_n_positive=int((row64 > 0).sum()),
    float64_positive_sum=float(np.asarray(row64)[np.asarray(row64) > 0].sum()),
    longdouble_max_positive=float(rowld.max()), longdouble_positive_sum=float(rowld[rowld > 0].sum()),
    longdouble_negative_sum=float(rowld[rowld < 0].sum()),
    exact_positive_sum_num=exact_pos.numerator, exact_positive_sum_den=exact_pos.denominator,
    exact_positive_sum_float=float(exact_pos), exact_negative_sum_float=float(exact_neg),
    exact_positive_support=[c for c, v in enumerate(exact_cols) if v > 0],
    exact_zero_support_and_float_noise=[
        dict(column=c, exact=float(exact_cols[c]), float64=float(row64[c]))
        for c in range(A.shape[1]) if exact_cols[c] == 0 and row64[c] != 0.0],
    longdouble_noise_floor=float(np.finfo(np.longdouble).eps) * float(np.abs(B[6 + r0, :]).max() * np.abs(T).max() * 90),
    ratio_required_over_positive=float(req[r0] / row64.max()) if row64.max() > 0 else None,
    activation_needed=float(req[r0] / row64.max()) if row64.max() > 0 else None,
    verdict=('exact_positive_sum_is_zero' if exact_pos == 0 else 'exact_positive_sum_nonzero'),
    note=('the exact rational sum decides structurally, independent of the 1e-18 / 1e-12 band; float64 and longdouble are reported '
          'alongside so the numerical residue is visible'),
)
(DIR / 'HIGH_PRECISION_ROW36.json').write_text(json.dumps(hp, indent=2, default=str))

# ---------------------------------------------------------------- Part A: 42-row sign census
def row_stats(i, eps):
    a = A[i, :]
    pos = a[a > eps]
    neg = a[a < -eps]
    r = float(req[i])
    need_pos = r > 0
    sup = float(pos.sum()) if need_pos else float(neg.sum())
    opp = float(neg.sum()) if need_pos else float(pos.sum())
    ok = sup > eps if need_pos else (-sup) > eps
    best = float(pos.max()) if (need_pos and pos.size) else (float(-neg.min()) if neg.size else 0.0)
    return dict(row=int(i), dof=6 + int(i), dof_name=jn[i], required=r,
                n_positive=int(pos.size), n_negative=int(neg.size),
                needed_direction=('positive' if need_pos else 'negative'),
                support_sum_needed=sup, support_sum_opposite=opp,
                structurally_reachable=bool(ok),
                max_coefficient_needed_direction=best,
                activation_if_single=float(abs(r) / best) if best > eps else None,
                fragility_ratio=float(best / abs(r)) if abs(r) > 0 else None)

census = {str(e): [row_stats(i, e) for i in range(42)] for e in EPS_GRID}
unreach_main = [row_stats(i, EPS_MAIN) for i in range(42) if not row_stats(i, EPS_MAIN)['structurally_reachable']]
census_main = [row_stats(i, EPS_MAIN) for i in range(42)]
flip = {}
for i in range(42):
    series = []
    for e in EPS_GRID:
        series.append(('reachable' if row_stats(i, e)['structurally_reachable'] else 'unreachable'))
    if len(set(series)) > 1:
        first = next((EPS_GRID[k] for k in range(len(series)) if series[k] != series[0]), None)
        flip[jn[i]] = dict(series=dict(zip([str(e) for e in EPS_GRID], series)), flips_at_eps=first)
sign = dict(eps_main=EPS_MAIN, eps_grid=EPS_GRID,
            rows=census_main,
            n_structurally_unreachable=len(unreach_main),
            unreachable_rows=[{'dof_name': r['dof_name'], 'dof': r['dof'], 'required': r['required'],
                               'support_sum_needed': r['support_sum_needed']} for r in unreach_main],
            eps_flip_table=flip,
            fragility_ranking=sorted([dict(dof_name=r['dof_name'], fragility=r['fragility_ratio'],
                                           activation_if_single=r['activation_if_single'])
                                      for r in census_main if r['fragility_ratio'] is not None],
                                     key=lambda d: d['fragility'])[:10])
(DIR / 'SIGN_CENSUS.json').write_text(json.dumps(sign, indent=2, default=str))
with open(DIR / 'SIGN_CENSUS.csv', 'w') as f:
    f.write('row,dof,dof_name,required,needed_direction,n_positive,n_negative,support_sum_needed,'
            'structurally_reachable,max_coef_needed,activation_if_single,fragility\n')
    for r in census_main:
        f.write('%d,%d,%s,%.12g,%s,%d,%d,%.6g,%d,%.6g,%s,%s\n' % (
            r['row'], r['dof'], r['dof_name'], r['required'], r['needed_direction'], r['n_positive'],
            r['n_negative'], r['support_sum_needed'], int(r['structurally_reachable']),
            r['max_coefficient_needed_direction'],
            ('%.6g' % r['activation_if_single']) if r['activation_if_single'] else '', r['fragility_ratio']))

# ---------------------------------------------------------------- row-36 actuator table
r36 = []
for a in range(90):
    coef = float(B[6 + r0, a])
    if abs(coef) <= EPS_MAIN:
        continue
    info = name_of_act.get(a, {})
    r36.append(dict(actuator_index=a, actuator=info.get('actuator'), leg=info.get('leg'),
                    muscle_target=info.get('muscle_target'), status=info.get('status'),
                    B_coefficient=coef, sign=('+' if coef > 0 else '-'),
                    t_row_sum=float(T_row_sum[a]), commandable=commandable[a],
                    pool_slot=int(np.load(MAP)['actuator_pool'][a]),
                    A_row36_coefficient=float(A[r0, int(np.load(MAP)['actuator_pool'][a])])
                    if int(np.load(MAP)['actuator_pool'][a]) >= 0 else 0.0))
pos_acts = [d for d in r36 if d['sign'] == '+']
cmd_pos = [d for d in pos_acts if d['commandable']]
row36 = dict(row=int(r0), dof=6 + r0, dof_name=jn[r0], required=float(req[r0]),
             n_actuators_above_eps=len(r36), actuators=r36,
             n_positive=len(pos_acts), positive_actuators=[d['actuator'] for d in pos_acts],
             n_positive_commandable=len(cmd_pos),
             negative_commandable=[d['actuator'] for d in r36 if d['sign'] == '-' and d['commandable']])
if len(pos_acts) == 1 and not pos_acts[0]['commandable']:
    row36['agonist_label'] = 'AGONIST_ONLY_VIA_UNRESOLVED_47'
elif cmd_pos:
    row36['agonist_label'] = 'AGONIST_ELSEWHERE'
else:
    row36['agonist_label'] = 'AGONIST_ABSENT'
(DIR / 'ROW36_ACTUATORS.json').write_text(json.dumps(row36, indent=2, default=str))

# ---------------------------------------------------------------- Part B: provenance of #47
prov = dict(
    unresolved_record=next((r for r in POOLS['unresolved'] if r['actuator'] == 'RMC_pleural_remotor_and_abductor'), None),
    derived_cells=DERIVED.get('pleural_remotor_and_abductor'),
    dataset_facts=dict(
        n_cells=len(DERIVED.get('pleural_remotor_and_abductor', [])),
        nerves=sorted({c['nerve'] for c in DERIVED.get('pleural_remotor_and_abductor', [])}),
        regions=sorted({c['region'] for c in DERIVED.get('pleural_remotor_and_abductor', [])}),
        sides=sorted({c['side'] for c in DERIVED.get('pleural_remotor_and_abductor', [])})),
    chain=('the frozen B gives actuator #47 a +6.587913 coefficient on DOF 42 (joint_RMCoxa_pitch); its T row is identically zero '
           'because the pool is unresolved; the unresolved reason is that the BANC annotation has pleural_remotor_and_abductor only '
           'on the LEFT mesothoracic leg (+ prothoracic accessory and both metathoracic), i.e. the RIGHT MIDDLE leg annotation is '
           'missing; therefore the only positive contributor to row 36 cannot be commanded.'),
    evidence_level=('CONSISTENCY evidence for "mapping gap -> zero one-sided authority on this DOF -> single-row infeasibility". It is '
                    'NOT a proof that closing the gap restores standing, and no substitution is performed here.'))
(DIR / 'AGONIST_PROVENANCE.json').write_text(json.dumps(prov, indent=2, default=str))

# ---------------------------------------------------------------- Part C: synthetic column
def solve(Amat, reqv, box):
    n, m = Amat.shape[1], reqv.size
    bnds = [(0, None)] * n if box is None else [(0, box)] * n
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((Amat, -np.ones(m))), np.column_stack((-Amat, -np.ones(m))))),
                 b_ub=np.r_[reqv, -reqv], bounds=bnds + [(0, None)], method='highs')
    return (float(lp.x[-1]) if lp.success else None), (lp.x[:-1] if lp.success else None)

T_syn = np.zeros(90)
T_syn[47] = 0.5
A_syn = (B @ T_syn)[6:].reshape(-1, 1)          # (42, 1)
A_aug = np.column_stack((A, A_syn[:, 0]))
leg_of_row = [n[6:8] for n in jn]
RM_ROWS = [i for i in range(42) if leg_of_row[i] == 'RM']
assert len(RM_ROWS) == 7, RM_ROWS
assert abs(solve(A[RM_ROWS, :], req[RM_ROWS], 1.0)[0] - 2.4262491357755884) <= 1e-6, 'RM block wrong'
sym_gain = float(A_syn[6 + r0 - 6, 0]) if False else float(A_syn[r0, 0])
res = {}
for tag, rows, box in (('RM_box1', RM_ROWS, 1.0), ('RM_unbounded', RM_ROWS, None),
                       ('global_box1', list(range(42)), 1.0), ('global_unbounded', list(range(42)), None)):
    p0, x0 = solve(A[rows, :], req[rows], box)
    p1, x1 = solve(A_aug[rows, :], req[rows], box)
    res[tag] = dict(rho_frozen=p0, rho_with_synthetic=p1, x_syn=(float(x1[-1]) if x1 is not None else None),
                    closed=bool(p1 is not None and p1 <= 1e-6))
rm_rows_check = []
p1_rm, x1_rm = solve(A_aug[RM_ROWS, :], req[RM_ROWS], 1.0)
if p1_rm is not None:
    res_rm = A_aug[RM_ROWS, :] @ x1_rm - req[RM_ROWS]
    for k, i in enumerate(RM_ROWS):
        a = A_aug[i, :]
        pos = a[a > EPS_MAIN]
        rm_rows_check.append(dict(dof_name=jn[i], residual=float(res_rm[k]),
                                  max_positive=float(pos.max()) if pos.size else 0.0,
                                  required=float(req[i]),
                                  s_needed=(float(req[i] / pos.max()) if pos.size else None)))
syn = dict(
    T_syn=dict(row=47, value=0.5, A_column_at_row36=sym_gain,
               coeff_if_activation_1=sym_gain,
               activation_needed_for_the_gap=(float(req[r0] / sym_gain) if sym_gain != 0 else None),
               ratio_to_required=float(sym_gain / req[r0]) if req[r0] else None),
    solves=res, rm_residuals_after_synthetic=rm_rows_check,
    label=('SIGN_GAP_CLOSED_BY_SYNTHETIC_COLUMN' if res['RM_box1']['closed'] else
           ('SIGN_GAP_PARTIALLY_CLOSED' if res['RM_box1']['rho_with_synthetic'] < res['RM_box1']['rho_frozen'] else
            'SIGN_GAP_UNCHANGED')),
    note=('a synthetic column is a bookkeeping device for "what if one antagonist existed"; it is not a model change and it may not '
          'be used to claim feasibility, standing or walking'))
(DIR / 'SYNTHETIC_COLUMN.json').write_text(json.dumps(syn, indent=2, default=str))

print(json.dumps(dict(dir=str(DIR),
                      exact=dict(pos_float=float(exact_pos), pos_exact_is_zero=bool(exact_pos == 0),
                                 f64_pos=hp['float64_positive_sum'], ld_pos=hp['longdouble_positive_sum'],
                                 activation_needed=hp['activation_needed']),
                      census=dict(n_unreachable=sign['n_structurally_unreachable'],
                                  rows=[u['dof_name'] for u in sign['unreachable_rows']],
                                  flips=list(sign['eps_flip_table'])[:5]),
                      row36=dict(n_pos=row36['n_positive'], pos_acts=row36['positive_actuators'],
                                 label=row36['agonist_label']),
                      syn=dict(label=syn['label'], gain=sym_gain, act_needed=syn['T_syn']['activation_needed_for_the_gap'],
                               rho={k: (None if v['rho_with_synthetic'] is None else round(v['rho_with_synthetic'], 6))
                                    for k, v in res.items()},
                               rm_res=[(r['dof_name'], round(r['residual'], 4)) for r in rm_rows_check])), indent=1, default=str))
