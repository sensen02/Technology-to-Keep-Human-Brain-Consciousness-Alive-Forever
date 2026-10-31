"""S206: does the 16-actuator commandability gap change the static simultaneous-feasibility verdict?

Fully offline: every ingredient is frozen in ROOT_BALANCE_STATE.npz (B, base, T), so this round
loads no model, runs no forward and steps no physics. PREREG is written by this process BEFORE the
LP solves, and the first solve must reproduce the frozen baseline gate 3.308620867791373 exactly.
"""
import json
import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

RUN = Path('/home/sensen/Desktop/engine修复验收/s206-commandability-gap-lp')
stamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')
B_DIR = RUN / stamp
B_DIR.mkdir(parents=True, exist_ok=False)

STATE = Path('/home/sensen/Desktop/engine修复验收/s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz')
CENSUS = json.load(open('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/'
                        '20261025T073450411660Z/READOUT_CENSUS.json'))
BASE_GATE = 3.308620867791373

prereg = dict(
    round='S206',
    title='static simultaneous feasibility recomputed with the 16 uncommandable actuators removed',
    directory=str(B_DIR),
    written_before_analysis=True,
    written_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    question=('S205 showed that 16 of 90 actuators can never receive a command and that they carry ~58%% of the |moment| on '
              'DOF 22 and DOF 42. The frozen static LP (S195/S196) allowed all 90 actuators. Does removing the 16 columns from '
              'the actuator-moment matrix change the minimax residual, i.e. is part of the recorded infeasibility an artefact of '
              'counting actuators that cannot be commanded?'),
    method=dict(
        source='ROOT_BALANCE_STATE.npz (frozen): B = (48,90) actuator-moment x gain, base = (48) bias term, T = (90,66) mapping',
        A='A_all = (B @ T)[6:,:] (42 x 66); A_cmd = (B with the 16 uncommandable columns zeroed @ T)[6:,:]',
        req='required[i] = -(base + J.T@f0)[6+i] (the frozen s185 definition, which includes the contact term)',
        lp='minimax: min t s.t. |A x - req| <= t, 0 <= x <= box, t >= 0 (the same formulation as S195/S196)',
        boxes=[1.0, 2.0]),
    gates=dict(
        baseline='the all-actuator frozen box [0,1] solve MUST reproduce %r; if it does not, the round is void' % BASE_GATE,
        verdicts=['GAP_IS_COVERAGE (minimax rises materially when the 16 columns are removed)',
                  'GAP_IS_ELSEWHERE (minimax essentially unchanged)', 'UNDETERMINED (baseline gate fails)'],
        threshold='materially = the minimax increases by more than 1%% relative'),
    budget=dict(model_loads=0, forward=0, mj_step=0, lp_solves=6),
    forbidden=dict(words=['改善', 'improved'], note='no movement claim; this is a static feasibility recomputation'),
)
(B_DIR / 'PREREG.json').write_text(json.dumps(prereg, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------ load frozen ingredients
z = np.load(STATE)
Bm = np.asarray(z['B'], float)          # (48, 90)
f0 = np.asarray(z['f0'], float)         # (64,) frozen contact force vector
J = np.asarray(z['J'], float)           # (64, 48)
base = np.asarray(z['base'], float)     # (48,)
T = np.asarray(z['T'], float)           # (90, 66)
assert Bm.shape == (48, 90) and T.shape == (90, 66) and base.shape == (48,)
empty = sorted(c['actuator_index'] for c in CENSUS['rows'] if c['pool_size'] == 0)
cmd = sorted(set(range(90)) - set(empty))
assert len(empty) == 16 and len(cmd) == 74

# frozen definition (s185_reconcile.py): required[i] = -(base + J.T@f0)[6+i]
req = -(base + J.T @ f0)[6:]
A_all = (Bm @ T)[6:, :]
Bm_cmd = Bm.copy()
Bm_cmd[:, empty] = 0.0
A_cmd = (Bm_cmd @ T)[6:, :]


def solve(A, box):
    n = A.shape[1]
    lp = linprog(np.r_[np.zeros(n), 1.0],
                 A_ub=np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42))))),
                 b_ub=np.r_[req, -req],
                 bounds=[(0, box)] * n + [(0, None)], method='highs')
    return float(lp.x[-1]) if lp.success else None, lp


res = {}
for tag, A in (('all_actuators', A_all), ('commandable_only', A_cmd)):
    for box in (1.0, 2.0):
        mm, lp = solve(A, box)
        res['%s_box%g' % (tag, box)] = dict(minimax=mm, box=box, status=int(lp.status),
                                            n_columns=int(A.shape[1]))

gate_ok = abs(res['all_actuators_box1']['minimax'] - BASE_GATE) <= 1e-9
delta_abs = res['commandable_only_box1']['minimax'] - res['all_actuators_box1']['minimax']
delta_rel = delta_abs / abs(res['all_actuators_box1']['minimax'])
materially = abs(delta_rel) > 0.01
verdict = 'UNDETERMINED' if not gate_ok else ('GAP_IS_COVERAGE' if materially else 'GAP_IS_ELSEWHERE')

# per-row authority loss (which of the 42 rows lose the most when the 16 columns go)
hi_all = np.maximum(A_all, 0).sum(axis=1)
hi_cmd = np.maximum(A_cmd, 0).sum(axis=1)
lo_all = np.minimum(A_all, 0).sum(axis=1)
lo_cmd = np.minimum(A_cmd, 0).sum(axis=1)
import glob
rm_path = sorted(glob.glob('/home/sensen/Desktop/engine修复验收/s179-targeted-legjoint-search/*/ROW_DOF_MAP.json'))[-1]
rowmap = json.load(open(rm_path))
jn = [r['dof_name'] for r in rowmap['rows']]
rows = []
for i in range(42):
    span_all = hi_all[i] - lo_all[i]
    span_cmd = hi_cmd[i] - lo_cmd[i]
    rows.append(dict(joint_row=i, dof=6 + i, dof_name=jn[i],
                     span_all=float(span_all), span_cmd=float(span_cmd),
                     span_lost=float(span_all - span_cmd),
                     span_lost_frac=float((span_all - span_cmd) / span_all) if span_all > 1e-15 else None,
                     required=float(req[i]),
                     required_exceeds_cmd=(bool(abs(req[i]) > abs(span_cmd) + 1e-12))))
rows_sorted = sorted([r for r in rows if r['span_lost_frac']], key=lambda r: -r['span_lost_frac'])

out = dict(
    gate=dict(expected=BASE_GATE, measured=res['all_actuators_box1']['minimax'], passed=bool(gate_ok)),
    results=res,
    verdict=verdict,
    delta=dict(absolute=float(delta_abs), relative=float(delta_rel),
               materially_threshold=0.01, materially=bool(materially)),
    rows_with_lost_authority=rows_sorted,
    n_rows_with_lost_authority=len(rows_sorted),
    uncommandable_actuators=[c['actuator'] for c in CENSUS['rows'] if c['pool_size'] == 0],
    reading=dict(
        verdict=('removing the 16 uncommandable actuators leaves the frozen minimax unchanged to within the 1%% threshold, so the '
                 'recorded static infeasibility is NOT explained by the commandability gap' if verdict == 'GAP_IS_ELSEWHERE' else
                 ('the frozen minimax rises by %.2f%% when the 16 uncommandable actuators are removed, so part of the recorded '
                  'infeasibility is attributable to the commandability gap' % (100 * delta_rel) if verdict == 'GAP_IS_COVERAGE' else
                  'the baseline gate failed, so nothing is concluded')),
        scope=('this is still the STATIC balance problem at one frozen pose with the frozen mapping; it says nothing about whether '
               'the body could stand after the gap is repaired')),
    calls=dict(model_loads=0, forward=0, mj_step=0, lp_solves=6, neural_ticks=0),
)
(B_DIR / 'COMMANDABILITY_GAP_LP.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(gate=out['gate'], verdict=verdict, delta_rel=round(delta_rel, 6),
                      res={k: (None if v['minimax'] is None else round(v['minimax'], 6)) for k, v in res.items()},
                      top_rows=[(r['dof_name'], round(r['span_lost_frac'], 4)) for r in rows_sorted[:6]]), indent=1))
print('DIR', B_DIR)
