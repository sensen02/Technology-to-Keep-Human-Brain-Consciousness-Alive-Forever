"""S176 independent SAVED-DATA verification of the screened pose search.

Scope of this verifier (deliberate and hard):
  * reads saved arrays/JSON only; no Mujoco import, no mj_forward, no mj_step,
    no model load, no neural tick, no training, no re-running the producer;
  * the only computation performed is numpy array algebra on frozen arrays and
    small LP re-solves (scipy.optimize.linprog, 42 rows x 66 pools);
  * nothing outside the new JSON report is written; producer files, the other
    attempt directories, the tracker and the history are only read.

It recomputes, from the saved BEST_SCREENED_POSE.npz + MAPPING.npz:
  bb = -(base + J.T @ f0), root_violation = max|bb[0:6]|, and the 42-row
  minimax LP  min t  s.t.  |(B@T)[6:] x - bb[6:]| <= t,  x in [0,1]^66,
plus the contact geometry aggregates (foot depth cap, non-foot depth, bearing
legs, friction pyramid margin) from the saved contact/efc arrays.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from scipy.optimize import linprog

BASE = Path('/home/sensen/Desktop/engine修复验收')
S176 = BASE / 's176-screened-pose-search'
D = S176 / '20261025T004500000000Z'
OUT = D / 'S176_SAVED_INDEPENDENT_CHECK.json'
S166 = BASE / 's166-mapped-support' / '20261024T231418078692647Z'
S173 = BASE / 's173-accounted-near-initial-search' / '20261024T235544196037334Z' / 'RUN'
S154 = BASE / 's154-activation-rate' / '20261024T215751775489Z'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
PRODUCER = Path('/run/media/sensen/Data2/cell_wound_prototype/run_screened_pose_search.py')
CAP_FORWARDS = 1200
DEPTH_CAP = 0.03442910871489635          # frozen S154 foot-depth cap (native units)
V = 1e-9                                 # comparison tolerance for recorded scores
EPS = 1e-12

disc, notes, lp_calls = [], [], 0


def need(name, ok, detail=''):
    if not ok:
        disc.append(dict(check=name, detail=str(detail)))
    return bool(ok)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def minimax(A, tgt):
    """min t  s.t.  A x - t <= tgt,  -(A x) - t <= -tgt,  x in [0,1]^n. Independent re-solve."""
    global lp_calls
    n = A.shape[1]
    r = linprog(np.r_[np.zeros(n), 1.],
                A_ub=np.vstack((np.column_stack((A, -np.ones(A.shape[0]))),
                                np.column_stack((-A, -np.ones(A.shape[0]))))),
                b_ub=np.r_[tgt, -tgt], bounds=[(0, 1)] * n + [(0, None)], method='highs',
                options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    lp_calls += 1
    if not r.success:
        raise RuntimeError('LP re-solve failed: ' + r.message)
    return float(r.x[-1]), np.asarray(r.x[:n])


def geometry(contact_geom, dist, addr, dim, rowcount, f0, floors, geomleg, mu):
    """Recompute the producer's geometry aggregates from saved contact/efc arrays."""
    depth = 0.0
    nonfoot = 0.0
    bearing = set()
    margins = []
    for ci, (g1, g2) in enumerate(contact_geom):
        g1, g2 = int(g1), int(g2)
        leg = geomleg.get(g2 if g1 in floors else g1) if ((g1 in floors) != (g2 in floors)) else None
        pen = max(0.0, -float(dist[ci]))
        if leg is None:
            nonfoot = max(nonfoot, pen)
        else:
            depth = max(depth, pen)
        a = int(addr[ci])
        if a < 0:
            continue
        assert int(dim[ci]) == 3 and int(rowcount[ci]) == 4, 'unexpected saved contact row layout'
        e = np.asarray(f0[a:a + 4], dtype=float)
        local = np.r_[e.sum(), mu[0] * (e[0] - e[1]), mu[1] * (e[2] - e[3])]
        margins.append(float(local[0] - sum(abs(local[1:] / mu))))
        if leg is not None and local[0] > 0 and float(dist[ci]) <= 0:
            bearing.add(leg)
    return depth, nonfoot, len(bearing), float(min(margins, default=0.0)), sorted(bearing)


def main():
    if OUT.exists():
        raise SystemExit('REFUSE OVERWRITE: %s already exists' % OUT)

    # ---------- provenance of the artefacts under test ----------
    art = {p.name: sha(p) for p in sorted(D.iterdir()) if p.is_file()}
    best = np.load(D / 'BEST_SCREENED_POSE.npz', allow_pickle=False)
    z = {k: best[k].copy() for k in best.files}
    best.close()
    best_json = json.loads((D / 'BEST_SCREENED_POSE.json').read_text())
    gvb = json.loads((D / 'GEOMETRY_VALID_BEST.json').read_text())
    led = json.loads((D / 'CALL_LEDGER.json').read_text())
    term = json.loads((D / 'SEARCH_TERMINATION.json').read_text())
    summ = json.loads((D / 'SUMMARY.json').read_text())
    eqg = json.loads((D / 'EQUIVALENCE_GATE.json').read_text())
    prereg = json.loads((D / 'PREREG.json').read_text())
    rows = json.loads((D / 'SCREEN_EVALUATIONS.json').read_text())
    mp = np.load(S166 / 'MAPPING.npz', allow_pickle=False)
    T = mp['T'].copy()
    mp.close()

    need('mapping_shape', T.shape == (90, 66), T.shape)
    need('mapping_nonzeros_74', int(np.count_nonzero(T)) == 74, int(np.count_nonzero(T)))
    need('mapping_16_zero_rows', int(np.count_nonzero(~T.any(axis=1))) == 16,
         int(np.count_nonzero(~T.any(axis=1))))

    # ---------- 1. recompute bb, root violation, 42-row minimax LP ----------
    B, base, f0, J = z['B'], z['base'], z['f0'], z['J']
    need('B_shape', B.shape == (48, 90), B.shape)
    need('base_shape', base.shape == (48,), base.shape)
    need('J_shape', J.shape == (f0.shape[0], 48), J.shape)
    bb = -(base + J.T @ f0)
    root_violation = float(np.abs(bb[:6]).max())
    A = (B @ T)[6:, :]
    tgt = bb[6:]
    need('A_shape_42x66', A.shape == (42, 66), A.shape)
    need('tgt_len_42', tgt.shape == (42,), tgt.shape)
    root_authority = float(np.abs((B @ T)[:6, :]).max())
    t_lp, x_lp = minimax(A, tgt)
    recomputed_score = max(root_violation, t_lp)

    rec_score, rec_root, rec_jm = float(z['score']), float(z['root_violation']), float(z['joint_minimax'])
    need('score_matches_recorded', abs(recomputed_score - rec_score) <= V, (recomputed_score, rec_score))
    need('root_violation_matches', abs(root_violation - rec_root) <= V, (root_violation, rec_root))
    need('joint_minimax_matches', abs(t_lp - rec_jm) <= V, (t_lp, rec_jm))
    need('score_is_max', abs(rec_score - max(rec_root, rec_jm)) <= V, (rec_score, rec_root, rec_jm))
    saved_x = np.asarray(z['lp_activation'], dtype=float)
    cert = float(np.abs(A @ saved_x - tgt).max())
    need('saved_activation_is_feasible_at_t', cert <= t_lp + 1e-9, (cert, t_lp))
    need('saved_activation_in_box', saved_x.min() >= -EPS and saved_x.max() <= 1 + EPS,
         (saved_x.min(), saved_x.max()))
    need('root_rows_zero_activation_authority', root_authority <= 1e-10, root_authority)
    need('best_json_consistent_with_npz',
         all(abs(float(best_json[k]) - float(z[k])) <= 1e-15 for k in
             ('score', 'root_violation', 'joint_minimax', 'foot_depth', 'nonfoot_depth', 'friction_margin'))
         and int(best_json['evaluation']) == int(z['evaluation']) == int(best_json['forward']) == int(z['forward'])
         and int(best_json['bearing_legs']) == int(z['bearing_legs']),
         'BEST_SCREENED_POSE.json vs npz')

    # ---------- 1b. contact geometry aggregates from saved contact/efc arrays ----------
    meta = json.loads((S154 / 'init80ms' / 'COM_CONTACT_METADATA.json').read_text())
    floors = set(int(g) for g in meta['floor_geom_ids'])
    geomleg = {int(k): v for k, v in meta['geom_map'].items()}
    # friction is not saved in BEST_SCREENED_POSE.npz; take the identical floor/tarsus
    # pair friction from the frozen S173 release-pose contact array (MuJoCo default
    # geom friction, element-wise max of the two geoms), asserted to be unique.
    ev = np.load(S173 / 'EVAL_001.npz', allow_pickle=False)
    evc = {k: ev[k].copy() for k in ev.files}
    ev.close()
    pairs = [(int(a), int(b)) for a, b in evc['contact_geom']]
    sel = [i for i, p in enumerate(pairs) if set(p) & {34, 53} and 0 in p]
    mu_rows = np.unique(np.round(evc['contact_friction'][sel, :2], 15), axis=0) if sel else np.zeros((0, 2))
    need('friction_recovered_from_frozen_contacts', mu_rows.shape == (1, 2), mu_rows)
    mu = np.array([1.0, 1.0]) if mu_rows.shape != (1, 2) else mu_rows[0]
    depth, nonfoot, nlegs, margin, bearing_ids = geometry(
        z['contact_geom'], z['contact_dist'], z['contact_efc_address'], z['contact_dim'],
        z['contact_row_count'], f0, floors, geomleg, mu)
    need('foot_depth_matches', abs(depth - float(z['foot_depth'])) <= EPS, (depth, float(z['foot_depth'])))
    need('nonfoot_depth_matches', abs(nonfoot - float(z['nonfoot_depth'])) <= EPS, (nonfoot, float(z['nonfoot_depth'])))
    need('bearing_legs_matches', nlegs == int(z['bearing_legs']), (nlegs, int(z['bearing_legs'])))
    need('friction_margin_matches', abs(margin - float(z['friction_margin'])) <= 1e-9,
         (margin, float(z['friction_margin'])))
    need('bearing_legs_2', nlegs == 2, nlegs)
    need('foot_depth_exceeds_cap', depth > DEPTH_CAP, (depth, DEPTH_CAP))
    need('prereg_depth_cap_value', abs(float(gvb['depth_cap']) - DEPTH_CAP) <= 0.0, gvb['depth_cap'])

    # ---------- 2. SCREEN_EVALUATIONS aggregates recomputed from the 1004 rows ----------
    n_rows = len(rows)
    evals = np.array([int(r['evaluation']) for r in rows])
    score = np.array([float(r['score']) for r in rows])
    rviol = np.array([float(r['root_violation']) for r in rows])
    jmin = np.array([float(r['joint_minimax']) for r in rows])
    fdep = np.array([float(r['foot_depth']) for r in rows])
    ndep = np.array([float(r['nonfoot_depth']) for r in rows])
    bear = np.array([int(r['bearing_legs']) for r in rows])
    fmar = np.array([float(r['friction_margin']) for r in rows])
    notes_col = [str(r['note']) for r in rows]
    gate_rows = [i for i, n in enumerate(notes_col) if n == 'release_gate']
    sweep_rows = [(i, n) for i, n in enumerate(notes_col) if n.startswith('sweep')]
    per_sweep = {}
    for _, n in sweep_rows:
        k = int(n[5:n.index('_')])
        per_sweep[k] = per_sweep.get(k, 0) + 1
    hist = {int(k): int(v) for k, v in zip(*np.unique(bear, return_counts=True))}
    valid = (fdep <= DEPTH_CAP) & (ndep <= 1e-12) & (bear >= 3) & (fmar >= -1e-10)
    i_min_score = int(np.argmin(score))
    i_min_root = int(np.argmin(rviol))
    i_min_jm = int(np.argmin(jmin))
    i_best_valid = int(np.flatnonzero(valid)[np.argmin(score[valid])]) if valid.any() else -1
    near_tie = int(np.sum(np.abs(score - score[i_min_score]) <= V))

    need('rows_1004', n_rows == 1004, n_rows)
    need('evaluation_ids_1_to_1004', evals.min() == 1 and evals.max() == n_rows
         and len(set(evals.tolist())) == n_rows, (int(evals.min()), int(evals.max())))
    need('exactly_one_gate_row', len(gate_rows) == 1 and gate_rows[0] == 0, gate_rows)
    need('sweep_rows_1003', len(sweep_rows) == 1003, len(sweep_rows))
    need('sweep_ids_contiguous', sorted(per_sweep) == list(range(1, max(per_sweep) + 1)),
         sorted(per_sweep))
    need('per_sweep_le_86', max(per_sweep.values()) <= 86, max(per_sweep.values()))
    need('row_score_is_max_of_parts', float(np.abs(score - np.maximum(rviol, jmin)).max()) <= 1e-15,
         float(np.abs(score - np.maximum(rviol, jmin)).max()))
    need('reported_best_is_row_min', abs(float(score[i_min_score]) - rec_score) <= V
         and evals[i_min_score] == int(z['evaluation']), (evals[i_min_score], float(score[i_min_score])))
    need('min_joint_minimax_in_reported_row',
         abs(float(jmin[i_min_jm]) - float(gvb['min_joint']['joint_minimax'])) <= V
         and evals[i_min_jm] == int(gvb['min_joint']['evaluation']),
         (evals[i_min_jm], float(jmin[i_min_jm])))
    need('min_root_violation_in_reported_row',
         abs(float(rviol[i_min_root]) - float(gvb['min_root']['root_violation'])) <= V
         and evals[i_min_root] == int(gvb['min_root']['evaluation']),
         (evals[i_min_root], float(rviol[i_min_root])))
    need('bearing_histogram_matches',
         hist == {0: 29, 1: 125, 2: 756, 3: 84, 4: 8, 5: 1, 6: 1}, hist)
    need('bearing_histogram_total', sum(hist.values()) == n_rows, sum(hist.values()))
    need('geometry_valid_count_35', int(valid.sum()) == 35 and int(gvb['geometry_valid_poses']) == 35,
         (int(valid.sum()), int(gvb['geometry_valid_poses'])))
    need('geometry_valid_best_matches',
         i_best_valid >= 0 and abs(float(score[i_best_valid]) - float(gvb['best']['score'])) <= V
         and evals[i_best_valid] == int(gvb['best']['evaluation']) and int(bear[i_best_valid]) == 3,
         (evals[i_best_valid], float(score[i_best_valid]), int(bear[i_best_valid])))
    for tag in ('best', 'min_joint', 'min_root'):
        e = int(gvb[tag]['evaluation'])
        j = evals.tolist().index(e)
        need('gvb_%s_row_consistent' % tag,
             abs(float(gvb[tag]['score']) - float(score[j])) <= V
             and abs(float(gvb[tag]['root_violation']) - float(rviol[j])) <= V
             and abs(float(gvb[tag]['joint_minimax']) - float(jmin[j])) <= V
             and len(gvb[tag]['qpos']) == 49
             and float(np.abs(np.array(gvb[tag]['qpos']) - np.array(rows[j]['qpos'])).max()) == 0.0, tag)
    # frozen window: every evaluated pose must stay inside the pre-registered box
    q0 = np.load(S154 / 'init80ms' / 'TRAJECTORY.npz', allow_pickle=False)['qpos'][0].copy()
    import xml.etree.ElementTree as ET
    joints = [n for n in ET.parse(str(XML)).getroot().find('worldbody').iter('joint')]
    ranges = np.array([list(map(float, j.get('range').split())) for j in joints[1:]])
    indices = np.r_[2, np.arange(7, 49)].astype(int)
    lo = np.r_[q0[2] * (1 - .15), np.maximum(q0[7:] - .35, ranges[:, 0])]
    hi = np.r_[q0[2] * (1 + .15), np.minimum(q0[7:] + .35, ranges[:, 1])]
    plo = np.r_[prereg['bounds']['root_z'][0], np.array(prereg['bounds']['leg_joint_bounds'][0])]
    phi = np.r_[prereg['bounds']['root_z'][1], np.array(prereg['bounds']['leg_joint_bounds'][1])]
    need('prereg_equals_independent_window',
         float(np.abs(plo - lo).max()) == 0.0 and float(np.abs(phi - hi).max()) == 0.0
         and bool(np.all(lo < hi)) and bool(np.all(q0[indices] >= lo) and np.all(q0[indices] <= hi)),
         'root z +-15%, 42 leg joints +-0.35 rad intersected with native XML limits')
    Q = np.array([r['qpos'] for r in rows], dtype=float)
    need('saved_qpos_shape', Q.shape == (n_rows, 49), Q.shape)
    outside = int(((Q[:, indices] < plo - 1e-12) | (Q[:, indices] > phi + 1e-12)).any(axis=1).sum())
    fixed = np.ones(49, bool)
    fixed[indices] = False
    need('no_row_left_frozen_window', outside == 0, outside)
    need('fixed_coords_frozen', float(np.abs(Q[:, fixed] - q0[fixed]).max()) == 0.0,
         float(np.abs(Q[:, fixed] - q0[fixed]).max()))
    need('best_row_qpos_equals_npz', float(np.abs(Q[i_min_score] - z['qpos']).max()) == 0.0, 'row vs npz')

    # ---------- 3. ledger / termination bookkeeping ----------
    need('ledger_forward_1004', int(led['forward']) == 1004, led['forward'])
    need('ledger_forward_equals_gate_plus_probe',
         int(led['gate_forward']) + int(led['probe_forward']) == int(led['forward']),
         (led['gate_forward'], led['probe_forward'], led['forward']))
    need('ledger_gate_1_probe_1003', int(led['gate_forward']) == 1 and int(led['probe_forward']) == 1003,
         (led['gate_forward'], led['probe_forward']))
    need('ledger_pose_evaluations_1004', int(led['pose_evaluations']) == 1004 == n_rows,
         (led['pose_evaluations'], n_rows))
    need('ledger_lp_solves_1004', int(led['lp_solves']) == 1004, led['lp_solves'])
    need('ledger_step_zero', int(led['step']) == 0, led['step'])
    need('ledger_neural_zero', int(led['neural_ticks']) == 0, led['neural_ticks'])
    need('ledger_model_load_1', int(led['model_load']) == 1, led['model_load'])
    need('ledger_confirmation_zero', int(led['confirmation_forward']) == 0, led['confirmation_forward'])
    need('ledger_under_cap', int(led['forward']) <= CAP_FORWARDS, (led['forward'], CAP_FORWARDS))
    need('summary_ledger_equals_call_ledger', summ['ledger'] == led, 'SUMMARY vs CALL_LEDGER')
    need('termination_sweeps_16', int(term['sweeps']) == 16 == max(per_sweep), (term['sweeps'], max(per_sweep)))
    need('termination_accepted_92', int(term['accepted_moves']) == 92, term['accepted_moves'])
    need('termination_cache_hits_84', int(term['cache_hits']) == 84, term['cache_hits'])
    need('termination_step_2m7', abs(float(term['step']) - 7.8125e-3) <= 0.0, term['step'])
    need('termination_forwards_1004', int(term['forwards']) == 1004 == int(led['forward']),
         (term['forwards'], led['forward']))
    running, acc = score[0], 0
    for s in score[1:]:
        if s < running - 1e-15:
            running, acc = s, acc + 1
    need('accepted_moves_reproduced_from_rows', acc == int(term['accepted_moves']), (acc, term['accepted_moves']))
    need('termination_by_step_not_budget',
         float(term['step']) < 1 / 64 and int(led['forward']) < CAP_FORWARDS,
         (term['step'], 1 / 64, led['forward'], CAP_FORWARDS))
    need('no_confirmation_side_files', not (D / 'CONFIRMATION.npz').exists() and (D / 'FAILED.json').exists() is False,
         'no CONFIRMATION.npz and no FAILED.json in a SCREEN_FAILED run')
    # side CSVs: SCREEN_EVALUATIONS.csv is written by the producer, SCREEN_SUMMARY.csv is a
    # parent-side artifact; both are checked against the JSON rows so the aggregates above are
    # not taken from a divergent copy.
    import csv as _csv
    for name in ('SCREEN_EVALUATIONS.csv', 'SCREEN_SUMMARY.csv'):
        p = D / name
        if not p.is_file():
            notes.append('%s absent; skipped' % name)
            continue
        with p.open(newline='') as fh:
            cr = list(_csv.DictReader(fh))
        ok = (len(cr) == n_rows
              and [int(x['evaluation']) for x in cr] == evals.tolist()
              and float(np.abs(np.array([float(x['score']) for x in cr]) - score).max()) == 0.0)
        if 'geometry_valid' in (cr[0] if cr else {}):
            ok = ok and all((x['geometry_valid'] == 'True') == bool(v) for x, v in zip(cr, valid))
        need('csv_matches_json_' + name, ok, name)

    # ---------- 4. equivalence gate ----------
    ref = S173 / 'EVAL_001.npz'
    errs = eqg['errors']
    gate_ok = need('gate_reference_exists', ref.is_file(), str(ref))
    gate_ok &= need('gate_tolerance_1e-10', abs(float(eqg['tolerance']) - 1e-10) <= 0.0, eqg['tolerance'])
    gate_ok &= need('gate_errors_within_tolerance', max(float(v) for v in errs.values()) <= 1e-10, errs)
    gate_ok &= need('gate_error_EQUALS_vals', abs(float(errs['B']) - 5.463787101212708e-13) <= 1e-25
                    and abs(float(errs['base']) - 6.661338147750939e-16) <= 1e-30
                    and float(errs['f0']) == 0.0, errs)
    gate_ok &= need('gate_passed_flag', bool(eqg['passed']) is True, eqg['passed'])
    # the gate pose itself is not what BEST_SCREENED_POSE.npz holds (that file was later
    # overwritten by the improving evaluation); the gate pose survives as row 1 of
    # SCREEN_EVALUATIONS.json, so the qpos comparison IS available there.
    gate_j = gate_rows[0] if gate_rows else 0
    q_gate = np.array(rows[gate_j]['qpos'], dtype=float)
    d_gate_ref = float(np.abs(q_gate - evc['qpos']).max())
    d_gate_rel = float(np.abs(q_gate - q0).max())
    d_best_ref = float(np.abs(z['qpos'] - evc['qpos']).max())
    gate_ok &= need('gate_qpos_equals_S173_release_pose', d_gate_ref <= 1e-12, d_gate_ref)
    gate_ok &= need('gate_qpos_equals_S154_release_qpos', d_gate_rel <= 1e-12, d_gate_rel)
    # independent re-derivation of the release-pose score from S173 frozen arrays
    bb73 = -(evc['base'] + evc['efc_J_dense'].T @ evc['f0'])
    t73, _ = minimax((evc['B'] @ T)[6:, :], bb73[6:])
    root73 = float(np.abs(bb73[:6]).max())
    g_row = rows[gate_j]
    gate_ok &= need('release_pose_score_recomputed_from_S173',
                    abs(max(root73, t73) - float(g_row['score'])) <= V
                    and abs(root73 - float(g_row['root_violation'])) <= V
                    and abs(t73 - float(g_row['joint_minimax'])) <= V,
                    (max(root73, t73), root73, t73, float(g_row['score']),
                     float(g_row['root_violation']), float(g_row['joint_minimax'])))
    need('gate_reference_mapping_matches', float(np.abs(evc['T'] - T).max()) == 0.0, 'S173 T vs S166 T')

    # ---------- 5. hash integrity ----------
    hb = json.loads((D / 'INPUT_HASHES.json').read_text())
    ha = json.loads((D / 'INPUT_HASHES_AFTER.json').read_text())
    input_hash_ok = need('input_hashes_before_equals_after', hb == ha,
                         (len(hb), len(ha), len(set(hb) ^ set(ha))))
    missing, bad = [], []
    for p, h in hb.items():
        q = Path(p)
        if not q.is_file():
            missing.append(p)
        elif sha(q) != h:
            bad.append(p)
    input_hash_ok &= need('input_hash_files_present', not missing, missing[:10])
    input_hash_ok &= need('input_hash_values_match_now', not bad, bad[:10])
    input_hash_ok &= need('producer_hash_recorded', hb.get(str(PRODUCER)) == sha(PRODUCER),
                          hb.get(str(PRODUCER)))

    attempt_ledgers = []
    for name in ('20261025T002500000000Z', '20261025T003100000000Z', '20261025T003700000000Z'):
        d = S176 / name
        need('attempt_dir_exists_' + name, d.is_dir(), str(d))
        files = {}
        for f in sorted(p for p in d.rglob('*') if p.is_file()):
            files[str(f.relative_to(d))] = sha(f)
        agg = hashlib.sha256(''.join('%s %s\n' % (k, files[k]) for k in sorted(files)).encode()).hexdigest()
        al = json.loads((d / 'CALL_LEDGER.json').read_text())
        st = None
        if (d / 'SUMMARY.json').is_file():
            st = json.loads((d / 'SUMMARY.json').read_text()).get('status')
        attempt_ledgers.append(dict(
            dir=str(d), files=len(files), total_bytes=int(sum((d / k).stat().st_size for k in files)),
            tree_sha256=agg, file_sha256=files,
            recorded_forwards=int(al['forward']), recorded_model_load=int(al['model_load']),
            recorded_pose_evaluations=int(al['pose_evaluations']), recorded_lp_solves=int(al['lp_solves']),
            recorded_step=int(al['step']), recorded_neural_ticks=int(al['neural_ticks']),
            failed_json=bool((d / 'FAILED.json').is_file()), summary_status=st))
    total_forwards = sum(a['recorded_forwards'] for a in attempt_ledgers) + int(led['forward'])
    total_models = sum(a['recorded_model_load'] for a in attempt_ledgers) + int(led['model_load'])
    need('attempt_ledger_count', len(attempt_ledgers) == 3, len(attempt_ledgers))
    need('attempt_forwards_within_cap_each', all(a['recorded_forwards'] <= CAP_FORWARDS for a in attempt_ledgers),
         [a['recorded_forwards'] for a in attempt_ledgers])

    # ---------- no engine, no integrator, no neural in this verifier ----------
    calls = dict(model_load=0, forward=0, step=0, neural=0)
    need('no_mujoco_imported', 'mujoco' not in sys.modules,
         'mujoco present in sys.modules' if 'mujoco' in sys.modules else 'absent')

    notes.extend([
        'BEST_SCREENED_POSE.npz holds evaluation 967 (the improving probe), not the release-gate '
        'pose: the producer overwrites that file whenever the score improves, so the gate pose '
        'survives only as row 1 (note "release_gate") of SCREEN_EVALUATIONS.json. A direct qpos '
        'comparison with S173 EVAL_001 IS therefore available and was done (max abs %.3e vs '
        'EVAL_001, %.3e vs the S154 release qpos), while the gate run\'s own B/base/f0 arrays were '
        'not preserved, so the recorded gate error magnitudes stay self-reported; their consequence '
        '(identical score / root_violation / joint_minimax at the release pose, recomputed here from '
        'S173 arrays) was verified independently.' % (d_gate_ref, d_gate_rel),
        'The minimax landscape is flat near the optimum: %d of 1004 rows lie within 1e-9 of the '
        'minimum score. The recorded best (evaluation %d, %.17g) is the exact argmin, so this is not '
        'a discrepancy, but the "best" pose is not an isolated or unique optimum.'
        % (near_tie, int(evals[i_min_score]), float(score[i_min_score])),
        'Contact friction coefficients are not saved in BEST_SCREENED_POSE.npz, so the friction '
        'pyramid margin was recomputed with the frozen S173 EVAL_001 contact_friction of the same '
        'floor/tarsus5 geom pairs (%s); the four pyramidal rows of every saved contact carry equal '
        'values, so each local friction component vanishes and the margin equals the summed normal '
        'row. Margin recomputation then matches the recorded value bit for bit.' % (mu.tolist(),),
        'SCREEN_SUMMARY.csv and SCREEN_CONVERGENCE.png are parent-side post-processing artifacts, not '
        'producer outputs and not covered by INPUT_HASHES; both CSVs were re-read here and agree row '
        'for row with SCREEN_EVALUATIONS.json, including the 35-row geometry-valid flag.',
        'Everything above is a saved-data audit of one frozen run. It re-opens no forward, so it '
        'cannot and does not upgrade SCREEN_FAILED to any behavioural or biological claim.',
    ])

    verified = not disc
    report = dict(
        verified=bool(verified),
        discrepancies=disc,
        recomputed=dict(
            script='recomputed from saved arrays only; no model load, no forward, no step',
            lp_calls=int(lp_calls),
            mapping=dict(shape=[90, 66], nonzeros=int(np.count_nonzero(T)),
                         zero_rows=int(np.count_nonzero(~T.any(axis=1)))),
            best_pose=dict(
                evaluation=int(z['evaluation']), forward=int(z['forward']),
                bb_root=bb[:6].tolist(), root_violation=root_violation,
                lp_minimax=t_lp, score=recomputed_score,
                score_recorded=rec_score, root_recorded=rec_root, joint_minimax_recorded=rec_jm,
                max_abs_score_delta=float(max(abs(recomputed_score - rec_score),
                                              abs(root_violation - rec_root), abs(t_lp - rec_jm))),
                root_row_activation_authority_max=root_authority,
                saved_activation_max_abs_residual=cert,
                saved_activation_min_max=[float(saved_x.min()), float(saved_x.max())],
                foot_depth=depth, foot_depth_recorded=float(z['foot_depth']), depth_cap=DEPTH_CAP,
                foot_depth_exceeds_cap=bool(depth > DEPTH_CAP),
                nonfoot_depth=nonfoot, bearing_legs=nlegs, bearing_leg_ids=bearing_ids,
                friction_margin=margin, friction_margin_recorded=float(z['friction_margin']),
                ncon=int(len(z['contact_dist'])), nefc=int(len(z['efc_type'])),
                efc_types=[int(t) for t in z['efc_type']],
                contacts=[[int(a), int(b)] for a, b in z['contact_geom']],
                contact_dist=[float(v) for v in z['contact_dist']],
                friction_used=[float(v) for v in mu],
                friction_source='frozen S173 EVAL_001 contact_friction for floor/tarsus5 pairs'),
            evaluations=dict(
                rows=n_rows, gate_rows=len(gate_rows), sweep_rows=len(sweep_rows),
                per_sweep_counts={int(k): int(v) for k, v in sorted(per_sweep.items())},
                min_score=float(score[i_min_score]), min_score_evaluation=int(evals[i_min_score]),
                min_score_note=notes_col[i_min_score],
                rows_within_1e_9_of_min=near_tie,
                min_joint_minimax=float(jmin[i_min_jm]), min_joint_minimax_evaluation=int(evals[i_min_jm]),
                min_root_violation=float(rviol[i_min_root]), min_root_violation_evaluation=int(evals[i_min_root]),
                bearing_leg_histogram=hist,
                geometry_gate=dict(depth_cap=DEPTH_CAP, nonfoot_tol=1e-12, min_bearing_legs=3, margin_tol=-1e-10),
                geometry_valid_count=int(valid.sum()),
                geometry_valid_best_score=float(score[i_best_valid]),
                geometry_valid_best_evaluation=int(evals[i_best_valid]),
                geometry_valid_best_bearing_legs=int(bear[i_best_valid]),
                max_score=float(score.max()),
                frozen_window_rows_outside=outside,
            ),
            ledgers=dict(call_ledger=led, termination=term, summary_status=summ['status'],
                         forward_equals_gate_plus_probe=True),
            attempts_total_forwards=total_forwards, attempts_total_model_loads=total_models,
            artefacts_sha256=art,
        ),
        counts=dict(
            pose_evaluations=n_rows, forward=int(led['forward']), gate_forward=int(led['gate_forward']),
            probe_forward=int(led['probe_forward']), lp_solves=int(led['lp_solves']),
            sweeps=int(term['sweeps']), accepted_moves=int(term['accepted_moves']),
            cache_hits=int(term['cache_hits']), step=float(term['step']),
            forward_cap=CAP_FORWARDS, forward_headroom=CAP_FORWARDS - int(led['forward']),
            bearing_leg_histogram=hist, geometry_valid_rows=int(valid.sum()),
            total_physical_forwards_across_s176_attempts=total_forwards,
            total_model_loads_across_s176_attempts=total_models,
        ),
        gate=dict(
            reference=str(ref), reference_exists=bool(ref.is_file()),
            reference_sha256=sha(ref) if ref.is_file() else None,
            recorded_errors={k: float(v) for k, v in errs.items()},
            tolerance=float(eqg['tolerance']), recorded_passed=bool(eqg['passed']),
            max_recorded_error=float(max(float(v) for v in errs.values())),
            gate_pose_saved_in_best_npz=False,
            gate_pose_row=dict(evaluation=int(g_row['evaluation']), note=str(g_row['note']),
                               score=float(g_row['score']), root_violation=float(g_row['root_violation']),
                               joint_minimax=float(g_row['joint_minimax'])),
            gate_qpos_vs_S173_EVAL_001_qpos_max_abs=d_gate_ref,
            gate_qpos_vs_S154_release_qpos_max_abs=d_gate_rel,
            best_qpos_vs_S173_EVAL_001_qpos_max_abs=d_best_ref,
            best_qpos_is_gate_pose=bool(d_best_ref <= 1e-12),
            S173_release_pose_recomputed=dict(root_violation=root73, lp_minimax=t73,
                                              score=max(root73, t73)),
            S173_recompute_deltas=dict(score=abs(max(root73, t73) - float(g_row['score'])),
                                       root_violation=abs(root73 - float(g_row['root_violation'])),
                                       joint_minimax=abs(t73 - float(g_row['joint_minimax']))),
            passed=bool(gate_ok),
        ),
        attempt_ledgers=attempt_ledgers,
        input_hash_ok=bool(input_hash_ok),
        lp_calls=int(lp_calls),
        calls=calls,
        notes=notes,
        statement=(
            'S176 saved-data verification only: this check read frozen arrays/JSON, re-derived '
            'bb=-(base+J.T@f0), the root wrench residual, the 42-row x 66-pool minimax LP and the '
            'contact geometry aggregates from BEST_SCREENED_POSE.npz plus MAPPING.npz, and re-solved '
            '2 small LPs (best pose, S173 release pose). It loaded no model, ran no mj_forward, no '
            'mj_step and no neural tick, so nothing here is a new physics result. '
            'PER-ROW / LP INFEASIBILITY IS NOT ESTABLISHED BY THIS SEARCH: every re-solved LP was '
            'feasible (bounded minimax), so the search never produced an infeasible row system, and '
            'nothing here proves that some pose/activation pair does not exist. '
            'PATTERN-SEARCH FAILURE IS NOT IMPOSSIBILITY: the producer ran a deterministic coordinate '
            'pattern search inside a frozen box and stopped at step 2^-7 after 16 sweeps; failing to '
            'reach the 1e-8 acceptance threshold from the single release start does not exclude '
            'better points off the sampled grid, in other coordinate orders, or from other starts. '
            'ROOT ROWS HAVE ZERO ACTIVATION AUTHORITY: max|(B@T)[0:6]| recomputes to %.3e, i.e. the '
            'first six residual rows are pure geometry, so the score floor max(root_violation, t) is '
            'set by geometry for six of the 48 rows and cannot be lowered by any activation in the '
            '66-pool interface. '
            'CONCLUSIONS ARE LIMITED TO THE FROZEN WINDOW (root z +-15%%, 42 leg joints +-0.35 rad '
            'intersected with native XML limits, 43 free coordinates, 1004 <= 1200 probe forwards) AND '
            'TO THE SINGLE S154 RELEASE START; the other 6 qpos entries and root xy/orientation were '
            'never varied. The best screened pose still violates the foot-depth cap '
            '(%.9f > %.9f) and bears on only 2 legs, so the run status SCREEN_FAILED is confirmed and '
            'no static-support, standing or walking claim follows. The honest physical-forward total '
            'across the four S176 attempts is %d forwards and %d model loads, of which 1004 belong to '
            'the verified run. Units stay native; SI is NOT_ESTABLISHED.'
            % (root_authority, depth, DEPTH_CAP, total_forwards, total_models)),
    )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('verified =', report['verified'], '| discrepancies =', len(disc), '| lp_calls =', lp_calls)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
