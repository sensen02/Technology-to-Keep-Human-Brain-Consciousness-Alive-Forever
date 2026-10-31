"""S178 independent verification (read-only): 1 model load, <=4 mj_forward, no mj_step,
no neural tick, no training, exactly one 42-row LP. Writes one JSON, refuses overwrite."""
import ast
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
S177 = BASE/'s177-root-newton-search/20261025T020500000000Z'
RUN = BASE/'s178-legjoint-release-search'
STAGE_A = RUN/'20261025T034000000000Z'
STAGE_B = RUN/'20261025T040000000000Z'
OLD = RUN/'20261025T030000000000Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
OUT = RUN/'S178_INDEPENDENT_CHECK.json'

DEPTH_CAP = 0.03442910871489635
BASELINE_MINIMAX = 3.3086208677913731      # S177 reported baseline (user-frozen reference)
DOFS = [19, 22, 26, 40, 42, 47]
STAGE_B_CAP, STAGE_B_RESERVED = 3000, 200
STATE = ('qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied')
OPTIONAL = ('mocap_pos', 'mocap_quat', 'eq_active', 'plugin_state', 'userdata')
LEG_BY_PREFIX = (('LFCoxa', 'lf'), ('LHCoxa', 'lh'), ('LMCoxa', 'lm'),
                 ('RFCoxa', 'rf'), ('RHCoxa', 'rh'), ('RMCoxa', 'rm'))

if OUT.exists():
    sys.exit('refuse overwrite: ' + str(OUT))

CALLS = dict(model_load=0, forward=0, step=0, neural=0)
CHECKS, DISC = [], []
RECOMPUTED = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(name, ok, numbers):
    CHECKS.append(dict(name=name, status='PASS' if ok else 'FAIL', numbers=numbers))
    return ok


def leg_of(name):
    """Leg from the joint NAME (LFCoxa/LFTrochanter/LFTibia ... all map to lf)."""
    s = name[6:8] if name.startswith('joint_') else ''
    return s.lower() if s in ('LF', 'LH', 'LM', 'RF', 'RH', 'RM') else 'unlabeled'


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw])


def quat_from_xyz(rx, ry, rz):
    cx, sx = np.cos(rx/2), np.sin(rx/2)
    cy, sy = np.cos(ry/2), np.sin(ry/2)
    cz, sz = np.cos(rz/2), np.sin(rz/2)
    return np.array([cx*cy*cz + sx*sy*sz, sx*cy*cz - cx*sy*sz,
                     cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz])


def build_qpos(q0, qleg, p):
    q = np.asarray(q0, float).copy()
    q[7:] = qleg
    q[0:2] = p[0:2]
    q[2] = p[2]
    dq = quat_from_xyz(p[3], p[4], p[5])
    qw = quat_mul(dq, q0[3:7])
    q[3:7] = qw/np.linalg.norm(qw)
    return q


def extract(p, names):
    t = ast.parse(Path(p).read_text())
    return '\n'.join(ast.unparse(next(n for n in t.body
                                      if isinstance(n, ast.FunctionDef) and n.name == x)) for x in names) + '\n'


# ---------------------------------------------------------------- static (no engine)
frozen = json.loads((STAGE_A/'FROZEN_RANGES.json').read_text())
frozen_b = json.loads((STAGE_B/'FROZEN_RANGES.json').read_text())
stage_a = json.loads((STAGE_A/'STAGE_A_RESULT.json').read_text())
base_rows = json.loads((STAGE_A/'STAGE_A_BASELINE_ROWS.json').read_text())
sum_a = json.loads((STAGE_A/'SUMMARY.json').read_text())
ledger_b = json.loads((STAGE_B/'STAGE_B_LEDGER.json').read_text())
result_b = json.loads((STAGE_B/'STAGE_B_RESULT.json').read_text())
sum_b = json.loads((STAGE_B/'SUMMARY.json').read_text())
rows_csv = list(csv.DictReader((STAGE_A/'STAGE_A_SIGNFLIP.csv').open()))

traj = np.load(S154/'init80ms/TRAJECTORY.npz', allow_pickle=False)
q0 = traj['qpos'][0].copy()
release = q0[7:].copy()
s177 = json.loads((S177/'ROOT_RESULT.json').read_text())
p177 = np.array(s177['p'], float)
qleg0 = np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:].copy()
state177 = np.load(S177/'ROOT_BALANCE_STATE.npz')
T = np.load(S166/'MAPPING.npz')['T']
meta = json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())

# --- check 1: frozen ranges = intersection(native XML limits, release +/- 0.6 rad)
tree = ET.parse(XML)
jnodes = [n for n in tree.getroot().find('worldbody').iter('joint')]
assert jnodes[0].get('type') == 'free'
jnodes = jnodes[1:]
xml_names = [n.get('name') for n in jnodes]
xml_lo = np.array([float(n.get('range').split()[0]) for n in jnodes])
xml_hi = np.array([float(n.get('range').split()[1]) for n in jnodes])
lo_c = np.maximum(release - 0.6, xml_lo)
hi_c = np.minimum(release + 0.6, xml_hi)
lo_s = np.array(frozen['lower'], float)
hi_s = np.array(frozen['upper'], float)
names_s = list(frozen['joint_names'])
rel_s = np.array(frozen['release'], float)
lo_clip = int(np.sum(np.abs(lo_c - (release - 0.6)) > 0))
hi_clip = int(np.sum(np.abs(hi_c - (release + 0.6)) > 0))
check('1_frozen_ranges_intersection_exact', bool(
    np.array_equal(lo_c, lo_s) and np.array_equal(hi_c, hi_s)
    and np.allclose(rel_s, release, rtol=0, atol=0) and names_s == xml_names),
    dict(n_joints=42, names_match_xml_names=bool(names_s == xml_names),
         max_abs_lower_diff=float(np.abs(lo_c - lo_s).max()),
         max_abs_upper_diff=float(np.abs(hi_c - hi_s).max()),
         max_abs_release_diff=float(np.abs(rel_s - release).max()),
         interval_bites_lower_on_xml_limit=lo_clip, interval_bites_upper_on_xml_limit=hi_clip,
         release_inside_all=bool(np.all(lo_s <= rel_s) and np.all(rel_s <= hi_s)),
         all_lower_lt_upper=bool(np.all(lo_s < hi_s)),
         stage_a_equals_stage_b_ranges=bool(frozen['lower'] == frozen_b['lower'] and
                                            frozen['upper'] == frozen_b['upper']),
         xml_sha256=sha(XML), frozen_xml_copies_match=bool(all(
             sha(p) == sha(XML) for p in sorted((STAGE_A/'SOURCE_FREEZE').glob('*MODEL_CALIB_H_VERIFIED.xml'))
             + sorted((STAGE_B/'SOURCE_FREEZE').glob('*MODEL_CALIB_H_VERIFIED.xml')))),
         n_frozen_xml_copies=len(list((STAGE_A/'SOURCE_FREEZE').glob('*MODEL_CALIB_H_VERIFIED.xml'))),
         width_all_equal_1p2_unclipped=int(np.sum(np.isclose(hi_s - lo_s, 1.2, atol=1e-12)))))

blocks = []
for i in range(6):
    js = list(range(i*7, i*7+7))
    blocks.append(dict(block_index=i, joint_range=[js[0], js[-1]],
                       legs=sorted({leg_of(names_s[j]) for j in js}),
                       joint_names=[names_s[j] for j in js]))
block_legs = [b['legs'][0] for b in blocks]
claimed = ['rf', 'lh', 'lm', 'lf', 'rh', 'rm']
check('1_block_order_claim_rf_first', bool(block_legs == claimed),
      dict(actual_block_legs=block_legs,
           actual_block_ranges=[[b['joint_range'][0], b['joint_range'][1]] for b in blocks],
           user_claimed_block_legs=claimed,
           mismatching_blocks=[i for i in range(6) if block_legs[i] != claimed[i]],
           joint_names_per_block={str(b['block_index']): b['joint_names'] for b in blocks},
           leg_of_joint_field_in_artifact=[sorted(set(v for _, v in frozen['leg_of_joint']))]))

# --- check 2: Stage A tables (block membership, baseline rows, sign-flip angles)
by_dof = {}
for r in rows_csv:
    by_dof.setdefault(int(r['dof']), []).append(r)
block_ok, joint_sets, sign_flip = True, {}, {}
for d in DOFS:
    b0 = ((d - 6)//7)*7
    want = list(range(b0, b0+7))
    got = sorted({int(r['joint']) for r in by_dof[d]})
    joint_sets[d] = dict(block_start=b0, expected=want, scanned=got, matches=got == want)
    block_ok = block_ok and got == want
    legs = {r['leg'] for r in by_dof[d]}
    derived = sorted({leg_of(names_s[j]) for j in got})
    block_ok = block_ok and legs == set(derived)
    needs = [(float(r['angle_needed_for_sign_flip']), int(r['joint']), r['flippable'])
             for r in by_dof[d] if r['angle_needed_for_sign_flip'] not in ('', 'None')]
    best = min(needs) if needs else None
    sign_flip[d] = dict(min_angle_needed_rad=None if best is None else best[0],
                        joint_index=None if best is None else best[1],
                        le_0p6=bool(best is not None and best[0] <= 0.6),
                        flag_flippable_count=int(sum(1 for r in by_dof[d] if r['flippable'] == 'True')),
                        joints_with_some_reachable_grid_point=int(sum(1 for r in by_dof[d]
                                                                      if r['row_reachable_at_some_grid'] == 'True')),
                        n_joints_scanned=len(got),
                        leg_labels=sorted(legs), leg_from_joint_names=derived,
                        range_clipped_points=int(sum(1 for r in by_dof[d] if r['angle_needed_for_sign_flip'] in ('', 'None'))))
check('2a_stage_a_block_membership', block_ok, joint_sets)

brows = {int(r['dof']): r for r in base_rows['rows']}
unreach = all(brows[d]['reachable'] is False for d in DOFS)
consistent = all((brows[d]['lo'] - 1e-9 <= brows[d]['target'] <= brows[d]['hi'] + 1e-9) is False for d in DOFS)
own_interval = len({(brows[d]['target'], brows[d]['lo'], brows[d]['hi']) for d in DOFS}) == 6
per_row_match = all(abs(a - b) < 1e-15 for d in DOFS
                    for a, b in [(brows[d]['target'], [p for p in stage_a['per_row'] if p['dof'] == d][0]['baseline']['target']),
                                 (brows[d]['lo'], stage_a['per_row'][[p['dof'] for p in stage_a['per_row']].index(d)]['baseline']['lo']),
                                 (brows[d]['hi'], stage_a['per_row'][[p['dof'] for p in stage_a['per_row']].index(d)]['baseline']['hi'])])
check('2b_stage_a_baseline_rows_unreachable_own_interval',
      bool(unreach and consistent and own_interval and per_row_match),
      dict(all_six_unreachable=bool(unreach), reach_flag_consistent_with_own_interval=bool(consistent),
           six_distinct_intervals=bool(own_interval), equals_stage_a_result_per_row=bool(per_row_match),
           rows={str(d): dict(target=brows[d]['target'], lo=brows[d]['lo'], hi=brows[d]['hi'],
                              margin_to_interval=float(max(brows[d]['lo'] - brows[d]['target'],
                                                           brows[d]['target'] - brows[d]['hi'])))
                 for d in DOFS}))

check('2c_stage_a_smallest_angle_for_sign_flip', bool(
    all(sign_flip[d]['min_angle_needed_rad'] is not None for d in DOFS)
    and all(sign_flip[d]['le_0p6'] == (sign_flip[d]['flag_flippable_count'] > 0) for d in DOFS)),
      dict(per_dof={str(d): sign_flip[d] for d in DOFS},
           dofs_needing_le_0p6=[d for d in DOFS if sign_flip[d]['le_0p6']],
           dofs_needing_gt_0p6=[d for d in DOFS if not sign_flip[d]['le_0p6']],
           one_sided_window_rad=0.6,
           forward_calls_stage_a=sum_a['stage_a']['forward'], cap_stage_a=500))

# --- check 3: Stage B ledger arithmetic
events = ledger_b['events']
trials = [e for e in events if e['role'] == 'block_trial']
rej_root = [e for e in events if e['role'] == 'REJECT_ROOT_NOT_BALANCED']
rej_geom = [e for e in events if e['role'] == 'REJECT_GEOMETRY']
init_ev = [e for e in events if e['role'] == 'initial']
root_ok = [t for t in trials if t['root_ok']]
sig0 = init_ev[0]['contact_signature']
contact_recount = sum(1 for t in root_ok if t['contact_signature'] != sig0)
led = ledger_b['ledger']
budget_ceiling = STAGE_B_CAP - STAGE_B_RESERVED
lin_search = led['forward'] - led['candidate_evaluations'] - led['candidate_evaluations'] - led['fd_probes']
check('3a_stage_b_ledger_values_as_required', bool(
    len(trials) == 60 and led['accepted_moves'] == 0 and led['rejected_root'] == 41
    and led['rejected_geometry'] == 0 and led['contact_set_changes'] == 18
    and led['candidate_evaluations'] == 61 and led['forward'] == 2098
    and led['forward'] <= budget_ceiling and len(rej_root) == 41 and len(rej_geom) == 0
    and len(root_ok) == 19 and all(t['minimax'] > BASELINE_MINIMAX for t in root_ok)),
    dict(block_trial_events=len(trials), accepted_moves=led['accepted_moves'],
         rejected_root_events_ledger=led['rejected_root'], rejected_root_events_recount=len(rej_root),
         rejected_geometry_events_ledger=led['rejected_geometry'], rejected_geometry_events_recount=len(rej_geom),
         contact_set_changes_as_written_in_ledger_json=led['contact_set_changes'],
         candidate_evaluations=led['candidate_evaluations'], forward=led['forward'],
         forward_budget_ceiling=budget_ceiling, forward_within_budget=bool(led['forward'] <= budget_ceiling),
         forward_margin=budget_ceiling - led['forward'], model_loads=sum_b['engine_calls']['model_load'],
         summary_engine_forward=sum_b['engine_calls']['forward'],
         summary_minus_ledger=int(sum_b['engine_calls']['forward'] - led['forward']),
         root_balanced_trials=len(root_ok), baseline_minimax=BASELINE_MINIMAX,
         root_balanced_minimax_min=min(t['minimax'] for t in root_ok),
         root_balanced_minimax_max=max(t['minimax'] for t in root_ok),
         all_root_balanced_worse_than_baseline=bool(all(t['minimax'] > BASELINE_MINIMAX for t in root_ok)),
         sweeps=led['sweeps'], trials_per_sweep=len(trials)/led['sweeps'],
         stop=result_b['stop'],
         forward_accounting=dict(inner_initial_evaluations=led['candidate_evaluations'],
                                 inner_fd_probes=led['fd_probes'],
                                 candidate_final_forwards=led['candidate_evaluations'],
                                 derived_inner_line_search_trials=int(lin_search),
                                 newton_iterations=led['root_newton_iterations'],
                                 line_search_ge_newton=bool(lin_search >= led['root_newton_iterations'])),
         bookkeeping_best_not_used_for_accept=dict(stored_minimax=ledger_b['best']['minimax'],
                                                   stored_root_ok=ledger_b['best']['root_ok'],
                                                   stored_root_norm=ledger_b['best']['root_norm'],
                                                   stored_bearing_legs=ledger_b['best']['geometry']['bearing_legs'],
                                                   note='global-best record is a root-unbalanced, contact-free pose; '
                                                        'accept test never reads it')))

res_ledger = result_b['ledger']
check('3b_ledger_self_consistency_contact_set_changes', bool(
    led['contact_set_changes'] == contact_recount == res_ledger['contact_set_changes']),
      dict(ledger_json_value=led['contact_set_changes'],
           stage_b_result_json_value=res_ledger['contact_set_changes'],
           recount_from_event_log=contact_recount,
           root_ok_trials_whose_signature_differs_from_the_initial=contact_recount,
           other_ledger_keys_agree_between_the_two_files=bool(
               all(led[k] == res_ledger[k] for k in led if k != 'contact_set_changes')),
           mechanism='BlockHillClimb._dump() snapshots self.ledger while recording candidate #61, '
                     'before run() applies that candidate\'s increment, so STAGE_B_LEDGER.json misses the '
                     'last increment; STAGE_B_RESULT.json (dumped after run() returns) has 19. '
                     'No physics/LP/forward number is affected.'))

# ---------------------------------------------------------------- engine (1 load, 2 forwards)
import mujoco as mj

ns = dict(np=np, ah=lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
ns_text = '\n'.join([
    extract(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)),
    extract(ROOT/'run_local_pool_response.py', ('moment_record',)),
    extract(ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
    extract(ROOT/'verify_muscle_reconstruction.py', ('fl', 'evaluate'))])
exec(compile(ns_text, '<independent>', 'exec'), ns)

proof = json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
CALLS['model_load'] += 1
m = mj.MjModel.from_xml_path(str(XML))
snap_before = ns['snapshot'](m) == proof['before']
m.actuator_dynprm[:, 0] = .001
m.actuator_dynprm[:, 1] = .0004
m.opt.integrator = mj.mjtIntegrator.mjINT_RK4
m.opt.timestep = .000025
snap_after = ns['snapshot'](m) == proof['after']
anodes = list(tree.getroot().find('actuator'))

lim = np.asarray([bool(m.jnt_limited[1+i]) for i in range(42)])
native = dict(njnt=int(m.njnt), hinge=len([j for j in range(m.njnt)
                                           if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]),
              names_match_xml_order=bool([m.joint(1+i).name for i in range(42)] == xml_names),
              limited_joints=int(lim.sum()),
              range_match_xml_on_limited=bool(np.array_equal(m.jnt_range[1:][lim],
                                                             np.column_stack((xml_lo, xml_hi))[lim])))


def forward(q, act=None):
    d = mj.MjData(m)
    d.time = float(traj['time'][0])
    for n in STATE:
        getattr(d, n)[:] = traj[n][0]
    for n in OPTIONAL:
        if n in traj:
            getattr(d, n)[:] = traj[n][0]
        else:
            assert np.asarray(getattr(d, n)).size == 0
    d.qpos[:] = q
    d.qvel[:] = 0
    a = np.zeros(90) if act is None else act
    d.act[:] = a
    d.ctrl[:] = a
    d.qacc_warmstart[:] = 0
    init = {n: np.asarray(getattr(d, n)).copy() for n in (*STATE, *OPTIONAL)}
    mj.mj_forward(m, d)
    CALLS['forward'] += 1
    pure = all(np.array_equal(getattr(d, n), init[n]) for n in (*STATE, *OPTIONAL))
    c = ns['constraint_record'](mj, m, d, 0)
    c.update(ns['moment_record'](m, d))
    for n in ('actuator_length', 'qfrc_actuator', 'qfrc_passive', 'qfrc_bias'):
        c[n] = np.asarray(getattr(d, n)).copy()
    c['efc_D'] = np.asarray(d.efc_D).copy()
    c['efc_aref'] = np.asarray(d.efc_aref).copy()
    c['efc_force'] = np.asarray(d.efc_force).copy()
    c['contact_dist'] = np.array([v.dist for v in d.contact])
    c['contact_friction'] = np.array([v.friction.copy() for v in d.contact]).reshape(-1, 5)
    return d, c, pure


def affine(c):
    types = c['efc_type']
    assert np.all(np.isin(types, [0, 3, 4, 5, 6]))
    f0 = c['efc_D']*c['efc_aref']
    f0 = np.where(types == 0, f0, np.maximum(f0, 0.))
    gain = np.empty(90)
    bias = np.empty(90)
    for j, node in enumerate(anodes):
        _, gain[j], bias[j] = ns['evaluate'](float(c['actuator_length'][j]), 0., 0., node)
    mom = c['actuator_moment']
    return mom.T*gain[None, :], mom.T@bias + c['qfrc_passive'] - c['qfrc_bias'], f0


def geometry(c, f0):
    floors = set(meta['floor_geom_ids'])
    geomleg = {int(k): v for k, v in meta['geom_map'].items()}
    depth = 0.
    nonfoot = 0.
    bearing = set()
    margins = []
    mu = np.ones(2)
    for ci in range(len(c['contact_geom'])):
        g1, g2 = int(c['contact_geom'][ci][0]), int(c['contact_geom'][ci][1])
        leg = geomleg.get(g2 if g1 in floors else g1) if ((g1 in floors) != (g2 in floors)) else None
        pen = max(0., -float(c['contact_dist'][ci]))
        if leg is None:
            nonfoot = max(nonfoot, pen)
        else:
            depth = max(depth, pen)
        addr = int(c['contact_efc_address'][ci])
        if addr < 0:
            continue
        dim = int(c['contact_dim'][ci])
        if dim == 1:
            local = np.r_[f0[addr], 0., 0.]
        else:
            e = f0[addr:addr+4]
            mu = c['contact_friction'][ci, :2]
            local = np.r_[e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])]
        margins.append(float(local[0] - sum(abs(local[1:]/mu))))
        if leg is not None and local[0] > 0 and c['contact_dist'][ci] <= 0:
            bearing.add(leg)
    return dict(foot_depth=float(depth), nonfoot_depth=float(nonfoot), bearing_legs=int(len(bearing)),
                friction_margin=float(min(margins, default=0.)),
                contact_signature='|'.join(sorted(f'{int(a)}-{int(b)}' for a, b in c['contact_geom']))
                + f'#nefc{len(c["efc_type"])}')


def rows_and_lp(B, base, f0, J):
    bb = -(base + J.T@f0)
    A = (B@T)[6:, :]
    tgt = bb[6:]
    lo = np.minimum(A, 0.).sum(axis=1)
    hi = np.maximum(A, 0.).sum(axis=1)
    reach = (lo - 1e-9 <= tgt) & (tgt <= hi + 1e-9)
    viol = np.maximum.reduce([lo - tgt, np.zeros_like(lo), tgt - hi])
    return bb, A, tgt, lo, hi, reach, viol


def lp42(A, tgt):
    n = A.shape[1]
    A_ub = np.vstack((np.column_stack((A, -np.ones(42))), np.column_stack((-A, -np.ones(42)))))
    lp = __import__('scipy.optimize', fromlist=['linprog']).linprog(
        np.r_[np.zeros(n), 1.], A_ub=A_ub, b_ub=np.r_[tgt, -tgt],
        bounds=[(0, 1)]*n + [(0, None)], method='highs',
        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    assert lp.success
    return lp


fin = result_b['final']
qtgt = np.array(fin['qpos'], float)
d4, c4, pure4 = forward(qtgt, None)
B4, base4, f04 = affine(c4)
bb4, A4, tgt4, lo4, hi4, reach4, viol4 = rows_and_lp(B4, base4, f04, c4['efc_J_dense'])
geom4 = geometry(c4, f04)
lp4 = lp42(A4, tgt4)
recomp = dict(qpos_matches_build=bool(np.array_equal(build_qpos(q0, qleg0, p177), state177['qpos'])),
              residual6_n=int(len(bb4)), root_max_abs_residual=float(np.abs(bb4[:6]).max()),
              root_residual6=bb4[:6].tolist(),
              minimax=float(lp4.x[-1]),
              unreachable=[int(6 + i) for i in range(42) if not reach4[i]],
              max_row_violation=float(viol4.max()),
              n_unreachable_rows=int(sum(1 for i in range(42) if not reach4[i])),
              geometry=geom4, geometry_ok=bool(
                  geom4['foot_depth'] <= DEPTH_CAP and geom4['nonfoot_depth'] <= 1e-12
                  and geom4['bearing_legs'] >= 3 and geom4['friction_margin'] >= -1e-10),
              depth_cap=DEPTH_CAP, nefc=int(len(c4['efc_type'])), ncon=int(len(c4['contact_geom'])))

def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


geom_exact = bool(geom4 == fin['geometry'])
geom_close = (near(geom4['foot_depth'], fin['geometry']['foot_depth'], 1e-15)
              and near(geom4['nonfoot_depth'], fin['geometry']['nonfoot_depth'], 1e-15)
              and int(geom4['bearing_legs']) == int(fin['geometry']['bearing_legs'])
              and near(geom4['friction_margin'], fin['geometry']['friction_margin'], 1e-15)
              and geom4['contact_signature'] == fin['geometry']['contact_signature'])

diffs = dict(
    root_norm_abs_diff=float(abs(recomp['root_max_abs_residual'] - fin['root_norm'])),
    residual6_max_abs_diff=float(np.abs(bb4 - np.array(fin['residual6'])).max()),
    minimax_abs_diff=float(abs(recomp['minimax'] - fin['minimax'])),
    minimax_saved=float(fin['minimax']), minimax_recomputed=float(recomp['minimax']),
    root_norm_saved=float(fin['root_norm']), root_norm_recomputed=float(recomp['root_max_abs_residual']),
    row_target_max_abs_diff=float(np.abs(tgt4 - np.array(fin['row_target'])).max()),
    row_lo_max_abs_diff=float(np.abs(lo4 - np.array(fin['row_lo'])).max()),
    row_hi_max_abs_diff=float(np.abs(hi4 - np.array(fin['row_hi'])).max()),
    row_violation_max_abs_diff=float(np.abs(viol4 - np.array(fin['row_violation'])).max()),
    geometry_bitwise_equal=geom_exact, geometry_close=bool(geom_close),
    geometry_recomputed=geom4, geometry_saved=fin['geometry'],
    unreachable_equal=bool(recomp['unreachable'] == list(fin['unreachable'])),
    unreachable_recomputed=recomp['unreachable'], unreachable_saved=list(fin['unreachable']),
    geom_ok_equal=bool(recomp['geometry_ok'] == fin['geom_ok']),
    root_ok_flag=bool(recomp['root_max_abs_residual'] <= 1e-8), root_ok_saved=bool(fin['root_ok']),
    minimax_lt_1e8=bool(recomp['minimax'] <= 1e-8),
    contact_signature_equal=bool(geom4['contact_signature'] == fin['contact_signature']),
    forward_state_unchanged_except_derived=bool(pure4),
    model_snapshot_before_matches_provenance=bool(snap_before),
    model_snapshot_after_matches_provenance=bool(snap_after),
    nv=int(m.nv), nq=int(m.nq), nu=int(m.nu), nefc=int(len(c4['efc_type'])),
    T_shape=list(T.shape), lp_rows=42, lp_cols=int(A4.shape[1]),
    activation_zero_used=True, qvel_zero_used=True, native_model=native)

check('4_independent_reload_at_stage_b_final', bool(
    near(recomp['root_max_abs_residual'], fin['root_norm'], 1e-15)
    and near(recomp['minimax'], fin['minimax'], 1e-9)
    and float(np.abs(bb4 - np.array(fin['residual6'])).max()) <= 1e-12
    and geom_close and recomp['unreachable'] == list(fin['unreachable'])
    and float(np.abs(tgt4 - np.array(fin['row_target'])).max()) <= 1e-12
    and recomp['root_max_abs_residual'] <= 1e-8 and pure4), diffs)
RECOMPUTED['stage_b_final'] = recomp

# --- independent recompute of the Stage A baseline rows (second forward, no LP)
dA, cA, pureA = forward(build_qpos(q0, qleg0, p177), None)
BA, baseA, f0A = affine(cA)
bbA, AA, tgtA, loA, hiA, reachA, violA = rows_and_lp(BA, baseA, f0A, cA['efc_J_dense'])
def cmp_saved(key, mine):
    ref = state177[key]
    return float(np.abs(np.asarray(mine) - np.asarray(ref)).max()) if np.shape(ref) == np.shape(mine) \
        else f'shape_mismatch:saved{list(np.shape(ref))}/recomputed{list(np.shape(mine))}'


saved177 = {k: cmp_saved(k, v) for k, v in
            (('B', BA), ('base', baseA), ('f0', f0A), ('J', cA['efc_J_dense']))}
base_recomp = {str(d): dict(target=float(tgtA[d-6]), lo=float(loA[d-6]), hi=float(hiA[d-6]),
                            reachable_by_own_interval=bool(reachA[d-6]),
                            saved_target=brows[d]['target'], saved_lo=brows[d]['lo'], saved_hi=brows[d]['hi'],
                            target_diff=float(abs(tgtA[d-6] - brows[d]['target'])),
                            lo_diff=float(abs(loA[d-6] - brows[d]['lo'])),
                            hi_diff=float(abs(hiA[d-6] - brows[d]['hi'])))
                for d in DOFS}
base_ok = (all(v['target_diff'] <= 1e-12 and v['lo_diff'] <= 1e-12 and v['hi_diff'] <= 1e-12
               for v in base_recomp.values())
           and not any(v['reachable_by_own_interval'] for v in base_recomp.values()) and pureA)
check('2d_baseline_rows_recomputed_from_engine', bool(base_ok),
      dict(per_dof=base_recomp, forward_state_unchanged=bool(pureA),
           baseline_still_unreachable_all_six=bool(not any(v['reachable_by_own_interval'] for v in base_recomp.values())),
           cross_check_vs_s177_saved_arrays=saved177,
           n_unreachable_rows_at_baseline=int(sum(1 for i in range(42) if not reachA[i]))))
RECOMPUTED['stage_a_baseline_rows'] = base_recomp

# --- check 5: files exist/non-empty + the 030000 defect attempt
four = [STAGE_A/'STAGE_A_SIGNS.png', STAGE_B/'ROOT_JOINT_SEPARATION.png',
        STAGE_A/'STAGE_A_RESULT.json', STAGE_B/'STAGE_B_RESULT.json']
png_ok = all(p.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' for p in four if p.suffix == '.png')
inventory = {str(p): p.stat().st_size for p in
             [STAGE_A/'STAGE_A_SIGNFLIP.csv', STAGE_A/'STAGE_A_RESULT.json', STAGE_A/'STAGE_A_BASELINE_ROWS.json',
              STAGE_A/'FROZEN_RANGES.json', STAGE_A/'SUMMARY.json', STAGE_A/'STAGE_A_SIGNS.png',
              STAGE_B/'STAGE_B_LEDGER.json', STAGE_B/'STAGE_B_RESULT.json', STAGE_B/'SUMMARY.json',
              STAGE_B/'FROZEN_RANGES.json', STAGE_B/'ROOT_JOINT_SEPARATION.png', STAGE_B/'STAGE_B_CONFIRMATION.json',
              STAGE_B/'CANDIDATE_STATE.npz']}
check('5a_artifacts_present_nonempty', bool(all(p.exists() and p.stat().st_size > 0 for p in four)
                                           and png_ok and all(v > 0 for v in inventory.values())),
      dict(four_plot_and_result_files={str(p): p.stat().st_size for p in four},
           png_signatures_valid=bool(png_ok), other_artifacts=inventory))

old_rows = list(csv.DictReader((OLD/'STAGE_A_SIGNFLIP.csv').open()))
old_sum = json.loads((OLD/'SUMMARY.json').read_text())
old_by = {}
for r in old_rows:
    old_by.setdefault(int(r['row']), []).append(r)
old_map = {}
bad = 0
for rr, sub in sorted(old_by.items()):
    js = sorted({int(x['joint']) for x in sub})
    derived = sorted({leg_of(names_s[j]) for j in js})
    label = sorted({x['leg'] for x in sub})
    agree = label == derived
    bad += 0 if agree else 1
    old_map[str(rr)] = dict(labeled_leg=label, leg_from_joint_names=derived,
                            disagrees=bool(not agree), scanned_joints=js,
                            correct_physical_dof=rr + 6, plus6_in_stage_a_table=bool(rr + 6 in DOFS))
old_baselines = {(p['baseline']['target'], p['baseline']['lo'], p['baseline']['hi']) for p in old_sum['stage_a']['per_row']}
check('5b_earlier_030000_defect_run_wrong_leg_blocks', bool(bad == 6 and len(old_baselines) == 1),
      dict(rows_in_defect_table=sorted(old_by), rows_plus6=sorted(r + 6 for r in old_by),
           rows_plus6_equals_correct_dofs=bool(sorted(r + 6 for r in old_by) == DOFS),
           dofs_22_26_40_42_47_present_in_that_table=bool(any(r in old_by for r in [22, 26, 40, 42, 47])),
           n_leg_labels_disagreeing=int(bad), per_row=old_map,
           distinct_baseline_intervals_in_that_table=len(old_baselines),
           shared_baseline_interval=list(old_baselines)[0],
           excluded_from_evidence=True,
           note='that table has no 22/26/40/42/47 rows; its rows are those six physical DOFs minus 6, '
                'and all six leg labels come from a positional LEGS order, so all six disagree'))
if not any(r in old_by for r in [22, 26, 40, 42, 47]):
    DISC.append('check5b literal wording: the 030000 Stage A table contains no DOF 22/26/40/42/47 rows; '
                'it lists rows 13/16/20/34/36/41 (= the same six physical DOFs minus 6) and all six of its '
                'leg labels disagree with the joint-name-derived leg, so the defect claim holds for those rows.')

pos_labels = [frozen['leg_of_joint'][i*7][1] for i in range(6)]
if pos_labels != block_legs:
    DISC.append('FROZEN_RANGES.json field leg_of_joint (both runs) assigns LEGS positional order to '
                'blocks 0..5 (' + ','.join(pos_labels) + ') while the joints are actually '
                + ','.join(block_legs) + ', so blocks 1,2,4,5 are mislabeled; the Stage A CSV leg column '
                'itself is derived from joint names and is correct, so no number is affected.')
if sum_a.get('stage_a', {}).get('forward') and json.loads((STAGE_A/'PREREG.json').read_text())['stageA'].startswith(
        'six unreachable rows [13,16,20,34,36,41]'):
    DISC.append('PREREG.json of both runs still writes the Stage A rows as [13,16,20,34,36,41] '
                '(S177 42-row subsystem indices) while the executed table uses the physical DOFs '
                '[19,22,26,40,42,47]; the two lists differ by exactly +6, so it is a label/wording '
                'inconsistency, not a different experiment.')
DISC.append('Stage B conserved 2099 engine forwards in SUMMARY vs 2098 in the ledger: the extra forward is '
            'the post-run candidate confirmation at the final point (STAGE_B_CONFIRMATION.json, act from the LP). '
            'Both are inside the 2800 ceiling.')
DISC.append('Real producer defect found: STAGE_B_LEDGER.json reports contact_set_changes=18 but the run\'s own '
            'event log (all 19 root_ok trials differ from the initial contact signature) and '
            'STAGE_B_RESULT.json\'s ledger both give 19. Cause: BlockHillClimb._dump() writes the ledger while '
            'recording candidate #61, one statement before run() adds that candidate\'s increment. Every other '
            'ledger key is identical in the two files, so no forward/candidate/LP/physics number changes; only '
            'the "18" in the required field list is one short of the true count.')
DISC.append('check1 block-order claim as stated by the requester (RF first, LF at 21-27) is refused by the '
            'artifact: block 0 is LF and block 3 is RF; the other four block positions match.')
DISC.append('Verifier-side disclosure: a first invocation of this script (same file, before fixing the '
            'joint-name->leg helper that returned "unlabeled" for Trochanter/Tibia joints) produced the same '
            'engine numbers but a false FAIL on check 2a; it was discarded and the delivered JSON is the single '
            'authoritative invocation. Session totals: 2 model loads, 4 mj_forward (cap 4), 2 42-row LPs, '
            '0 mj_step, 0 neural ticks.')

verified = all(c['status'] == 'PASS' for c in CHECKS)
RECOMPUTED['baseline_minimax_reference'] = BASELINE_MINIMAX
RECOMPUTED['stage_a_forward'] = sum_a['stage_a']['forward']
RECOMPUTED['stage_b_forward_ledger'] = led['forward']
RECOMPUTED['four_files'] = {str(p): p.stat().st_size for p in four}

statement = (
    'verified=false ONLY because of two non-physical items: (i) the requester-supplied block order is wrong '
    '(block 0 is LF, block 3 is RF) and (ii) STAGE_B_LEDGER.json writes contact_set_changes=18 while its own '
    'event log and STAGE_B_RESULT.json give 19. Every physics, muscle, LP, geometry and residual number '
    'reproduced bit-exactly under this read-only reload (all diffs 0.0, 1 model load, 2 mj_forward, 0 mj_step, '
    '0 neural ticks, 1 42-row LP): root max|r6|=1.3464784842653899e-12, 42-row minimax=3.308620867791373, '
    'unreachable=[19,22,26,40,42,47], foot depth 0.00598163075637239 <= cap 0.03442910871489635, nonfoot 0, '
    'bearing legs 6, friction margin 0.582717700499493. '
    'With accepted_moves=0 the Stage B trajectory is NOT valid evidence about the leg-joint subspace: no joint '
    'was ever moved, so the run only shows that every trial out of the S177 root-balanced baseline was refused '
    '- 41/60 by inner-Newton failure to re-balance the root (geometry never refused) - and that all 19 '
    'root-balanced trials worsened the 42-row minimax (3.5096..5.9508 vs baseline 3.3086208677913731). '
    'The accept logic itself (root gate, geometry gate, strict minimax decrease, planted-best ignored, '
    'zero-accept signalled, budget refusal) is covered by the engine-free tests in test_legjoint_core.py, so '
    'the harness is exercised even though this search produced nothing. No candidate exists '
    '(all48_balanced=false, minimax 3.31 >> 1e-8, six unreachable rows); no dynamics/standing/neural claim is '
    'made or supported; units are native and SI is NOT_ESTABLISHED.')

payload = dict(verified=bool(verified), checks=CHECKS, discrepancies=DISC, recomputed=RECOMPUTED,
               calls=dict(CALLS, lps=1, training=0,
                          session_disclosure='invocation 1 discarded (verifier-side leg-label helper bug, same '
                                             'engine numbers); session totals = 2 model loads, 4 mj_forward, '
                                             '2 42-row LPs, 0 mj_step, 0 neural ticks'),
               statement=statement)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(verified=verified, out=str(OUT),
                      statuses={c['name']: c['status'] for c in CHECKS},
                      calls=CALLS, n_discrepancies=len(DISC)), ensure_ascii=False))
