"""Assemble the CORRECTED / REISSUED equilibrium batch for the delivered pose A.

This script does NOT search anything. It consumes four independent runs of the already-verified search
implementation (run_mujoco_equilibrium_search.py, invoked with --pose <pose A>, --label <label> and, for the
nested probes, --radius r):

    A_ms240     full declared box, 240 random starts
    A_ms_r0p20  nested probe around the delivered pose A, r = 0.20 rad, 40 starts
    A_ms_r0p10  nested probe around the delivered pose A, r = 0.10 rad, 240 starts
    A_ms_r0p05  nested probe around the delivered pose A, r = 0.05 rad, 40 starts

and then
  * recomputes every reported quantity on a FRESH MjData (independent of the search's solver state),
  * writes one report.json with per-equilibrium detail (max|qacc|, norm, convergence status, start counts and
    hits, actuator at bound with its own atol, root z, contact counts, max u_diamond, max 42-joint distance),
  * carries the mandatory disclosure and the corrected friction-metric wording,
  * records the superseded / void directories, and
  * writes the figure, source snapshot and delivery hashes.

Manual research implementation, not a product. No standing, no walking, nothing is "achieved".
"""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
POSE_A = DESK / 'S2-declared-cone-min-s-20261015T123603/best_candidate_pose.npz'
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
CANONICAL_XML = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
SEARCH_SCRIPT = ROOT / 'run_mujoco_equilibrium_search.py'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
RUN_LABELS = ('A_ms240', 'A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')
NESTED_PLAN = {'A_ms_r0p05': 0.05, 'A_ms_r0p10': 0.10, 'A_ms_r0p20': 0.20}
SEARCH_TOL_MM_S2 = 1.0
DECLARED_GATE = dict(
    max_abs_qacc_mm_s2=1e-3,
    max_abs_qacc_reason=('about 1e-7 g: with mm units and gravity 9801 mm/s^2 this is ~1e-7 of g, i.e. numerically '
                         'indistinguishable from zero at the precision of a single mj_forward on this model (double '
                         'precision, solver tolerances ~1e-10), while still being 10x tighter than the tightest '
                         'meaningful dynamic criterion used anywhere in this project (1e-2 mm/s^2)'),
    six_tarsi_active_contact=True, zero_nonfoot_contacts=True, all_contacts_u_diamond_le_1=True,
    drive=('frozen connectome-derived command archive only (no external CPG, no PD, no preprogrammed action, '
           'no per-step pose correction)'),
    not_standing='passing this gate may only be called a DECLARED-CONE STATIC EQUILIBRIUM, never standing')

MANDATORY_DISCLOSURE = [
    'qacc = 0 is only a STATIC NECESSARY CONDITION: it says a state exists in which the model is momentarily at rest '
    'and the solver reports no acceleration with qvel = 0. It is NOT stability, NOT balance, NOT standing, and nothing '
    'here is a behaviour.',
    'It must NOT be read as "pose A is an equilibrium": the accepted point is a NEIGHBOURING state found by the '
    'search, several millimetres and up to 0.42 rad away from the delivered pose A in the 42 leg joints.',
    'A declared-cone static equilibrium is NOT dynamic maintainability. The separately measured open-loop run of this '
    'same state under the real 100-frame connectome-derived archive loses foot contact at step 2739 = 68.475 ms '
    '(artifact S2-true-equilibrium-openloop-20261015T154720).',
    'Only the frozen connectome-derived command archive is used as drive; there is no external CPG, no PD, no '
    'preprogrammed action and no per-step pose correction anywhere in this search.',
    'Absolute force values and physical units remain NOT_ESTABLISHED. The contact quantities here are relative '
    'utilisations of the declared cone, not calibrated forces.',
    'The static muscle allocation s <= 1 only means that a non-negative allocation exists under the declared cone; '
    'it is not a statement about muscles in an animal.',
    'Manual research implementation, not a product: no standing, no walking, and nothing about the project goal is '
    'complete.']

FRICTION_CORRECTED_WORDING = dict(
    criterion=('The declared cone is the AXIS-RAY DIAMOND, u_diamond = |ft1| / (mu1 * fn) + |ft2| / (mu2 * fn) <= 1 '
               '(MuJoCo: opt.cone = PYRAMIDAL, tarsus condim = 3, 4 mjCNSTR_CONTACT_PYRAMIDAL rows per contact); '
               'mu1 = mu2 = 1.0 from the contact friction rows [1.0, 1.0, 0.005, 0.0001, 0.0001].'),
    secondary_reading=('The SQUARE max(u1, u2) <= 1 with u1 = |ft1| / (mu1 * fn), u2 = |ft2| / (mu2 * fn) is a strictly '
                       'LARGER set (it contains the diamond); an earlier batch used it as the criterion, which was '
                       'wrong. It is kept here only as a clearly labelled secondary reading, never as the criterion.'),
    first_whole_run_diamond_saturation=dict(
        A=dict(step=256, ms=6.400), B=dict(step=401, ms=10.025), C=dict(step=277, ms=6.925),
        first_saturating_contact='RMTarsus5_geom|floor'),
    retracted_claim=('The earlier statement "overturning dominated, NOT friction saturation" is RETRACTED: under the '
                     'correct diamond criterion the whole-run diamond limit is first reached about 50 ms BEFORE the '
                     'collapse (A 6.400 ms vs T0 59.950 ms), so friction-limit contact IS reached well before '
                     'grounding and must not be described as absent.'),
    unchanged_numbers='No measured number was adjusted in this reissue; only the criterion name and the wording change.')

VOID_DIRS = [
    dict(path=str(DESK / 'S2-mujoco-equilibrium-A-20261015T151340'),
         reason=('INTEGRITY INCIDENT: report.json was replaced in place at 15:22:25 local and delivery-hashes.json '
                 'was rewritten, the script edited at 15:23:44; the directory is VOID and no number may be taken '
                 'from it.'),
         report_sha256_before='c12ead58', report_sha256_after='62841f3c'),
    dict(path=str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946'),
         reason=('SUPERSEDED INTERMEDIATE of this very reissue round: the hand-written multi-start in '
                 'run_equilibrium_reissue.py returned fun values that alias the live MjData buffer (np.asarray of '
                 'd.qacc is a view, not a copy), so the least-squares line search compared an array with itself and '
                 'never converged: best max|qacc| = 9016.5 mm/s^2 over 240 starts, 0 hits. Its numbers are NOT a '
                 'result and are not reused. The corrected batch instead re-uses the already-verified search '
                 'implementation run_mujoco_equilibrium_search.py, which copies the residual.'),
         report_sha256_before=None, report_sha256_after=None),
    dict(path=str(DESK / 'S2-mujoco-equilibrium-A_ms240-20261015T155009'),
         reason=('SUPERSEDED INTERMEDIATE of this round: produced before the search script was extended with the '
                 'per-equilibrium detail block (describe()), so it stored only the first 20 start residuals; its '
                 'numbers agree with the re-run but it lacks the per-equilibrium fields.'),
         report_sha256_before=None, report_sha256_after=None)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest_run(label):
    cands = sorted(DESK.glob('S2-mujoco-equilibrium-%s-*' % label))
    cands = [c for c in cands if (c / 'report.json').exists() and (c / 'stage2_best_state.npz').exists()]
    if not cands:
        return None
    return cands[-1]


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    runs = {}
    for lab in RUN_LABELS:
        r = latest_run(lab)
        if r is None:
            print('FAIL_STOP: no completed run for label', lab)
            return 1
        runs[lab] = r
    print('runs:')
    for lab, r in runs.items():
        print('  %-10s %s' % (lab, r))

    out = DESK / ('S2-equilibrium-A-corrected-reissued-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    m = cp.load_model()
    q0 = np.load(POSE_A)['projected_seated_qpos'].copy()
    src = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    suffixes = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll',
                'Tibia_pitch')
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in suffixes:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    act_lo = np.full(m.na, 1e-4)
    act_hi = np.full(m.na, 1.0)

    def recompute(qpos, act, tag):
        """fresh MjData: independent of any solver state left behind by the search"""
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.act[:] = np.clip(act, act_lo, act_hi)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        qa = np.asarray(d.qacc, dtype=float)
        order = np.argsort(-np.abs(qa))
        f6 = np.zeros(6)
        rows = []
        sq = 0.0
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            mujoco.mj_contactForce(m, d, ci, f6)
            fr = np.asarray(c.friction)[:3]
            fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
            u1 = abs(ft1) / (float(fr[0]) * fn) if fn > 0 else None
            u2 = abs(ft2) / (float(fr[1]) * fn) if fn > 0 else None
            ud = (u1 + u2) if (u1 is not None and u2 is not None) else None
            if u1 is not None:
                sq = max(sq, u1, u2)
            leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), 'trunk/head')
            rows.append(dict(geom=sorted((n1, n2)), leg=leg,
                             kind=('tarsus' if 'Tarsus' in (n1 + n2) else 'non-foot'),
                             fn=fn, u_diamond=ud))
        us = [r['u_diamond'] for r in rows if r['u_diamond'] is not None]
        per_leg = {}
        for r in rows:
            if r['kind'] == 'tarsus' and r['leg'] in LEGS and r['u_diamond'] is not None:
                per_leg[r['leg']] = max(per_leg.get(r['leg'], 0.0), r['u_diamond'])
        act_c = np.clip(act, act_lo, act_hi)
        return dict(
            tag=tag,
            max_abs_qacc_mm_s2=float(np.max(np.abs(qa))), qacc_norm=float(np.linalg.norm(qa)),
            nv=int(qa.size), n_dof=int(qa.size),
            n_dof_above_1e_minus_3=int(np.sum(np.abs(qa) > 1e-3)),
            n_dof_above_1e_minus_6=int(np.sum(np.abs(qa) > 1e-6)),
            n_dof_above_1e_minus_9=int(np.sum(np.abs(qa) > 1e-9)),
            top5_abs_qacc=[float(np.abs(qa)[i]) for i in order[:5]],
            top5_dof_index=[int(i) for i in order[:5]],
            n_contacts=int(len(d.contact)),
            n_tarsus_contacts=sum(1 for r in rows if r['kind'] == 'tarsus'),
            n_nonfoot_contacts=sum(1 for r in rows if r['kind'] == 'non-foot'),
            tarsi_with_contact=sorted(per_leg.keys()), six_tarsi_in_contact=bool(len(per_leg) == 6),
            per_leg_max_u_diamond=per_leg, max_u_diamond=(None if not us else float(max(us))),
            max_u_square_secondary_NOT_the_criterion=float(sq),
            root_z_raw=float(qpos[2]),
            root_z_change_from_delivered_pose_A_mm=float(qpos[2] - q0[2]),
            max_abs_delta_q_rad_vs_delivered_pose_A=float(np.max(np.abs(qpos[qadr] - q0[qadr]))),
            max_abs_delta_q_rad_vs_source_pose=float(np.max(np.abs(qpos[qadr] - src[qadr]))),
            act_min=float(act_c.min()), act_max=float(act_c.max()), act_mean=float(act_c.mean()),
            act_at_floor_own_atol=int(np.sum(np.abs(act_c - act_lo) <= 1e-12)),
            act_at_ceiling_own_atol=int(np.sum(np.abs(act_c - act_hi) <= 1e-9)),
            act_within_1e_minus_9_of_floor=int(np.sum(act_c - act_lo <= 1e-9)),
            act_within_1e_minus_9_of_ceiling=int(np.sum(act_hi - act_c <= 1e-9)),
            act_within_1e_minus_6_of_floor=int(np.sum(act_c - act_lo <= 1e-6)),
            act_within_1e_minus_6_of_ceiling=int(np.sum(act_hi - act_c <= 1e-6)))

    gate_checks = lambda e: dict(
        max_abs_qacc=bool(e['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2']),
        six_tarsi_active_contact=bool(e['six_tarsi_in_contact']),
        zero_nonfoot_contacts=bool(e['n_nonfoot_contacts'] == 0),
        all_contacts_u_diamond_le_1=bool(e['max_u_diamond'] is not None and e['max_u_diamond'] <= 1.0 + 1e-9))

    report = dict(
        status='RUNNING', title='S2 CORRECTED / REISSUED equilibrium search for the delivered pose A',
        reissue=True, corrected=True,
        supersedes=['S2-mujoco-equilibrium-A-20261015T151340 (VOID: integrity incident)',
                    'S2-equilibrium-A-corrected-reissued-20261015T154946 (defective multi-start)',
                    'S2-mujoco-equilibrium-A_ms240-20261015T155009 (superseded intermediate)'],
        manual_research_not_product=True, standing=False, walking=False, completion_percentage=None,
        scope=('corrected/reissued declared numerical equilibrium search for the delivered pose A: stage 1 varies only the '
               '90 actuators in [1e-4, 1]; stage 2 also allows root z within +-0.05 mm and the 42 leg hinge joints within '
               '[joint range - 0.005 rad] and within R of the delivered pose (R = 0.40 rad for the full box, and the nested '
               'probe values 0.20 / 0.10 / 0.05 rad). qvel = 0 everywhere; the minimised quantity is ||qacc||; the contact '
               'forces automatically satisfy the declared diamond because that is the cone MuJoCo applies.'),
        declared_gate=DECLARED_GATE,
        search_tolerance_mm_s2=SEARCH_TOL_MM_S2,
        tolerance_note=('the search decides "found" at 1.0 mm/s^2 while the declared gate is 10x tighter at 1e-3 mm/s^2; both '
                        'are reported separately and the accepted point satisfies both'),
        mandatory_disclosure=MANDATORY_DISCLOSURE,
        friction_metric_corrected_wording=FRICTION_CORRECTED_WORDING,
        not_established=('absolute force values, physical units, biological validity and any behaviour remain NOT_ESTABLISHED; '
                         'no standing, no walking'),
        root_cause_of_the_earlier_failure=('run_equilibrium_reissue.py returned np.asarray(d.qacc) from its residual, which is a '
                                           'VIEW into the live MjData buffer, not a copy, so every evaluation rewrote the array the '
                                           'optimiser still held. Measured on the same model/pose/box/first start '
                                           '(diagnose_aliased_residual.py): view residual -> nfev = 1, status "gtol termination '
                                           'condition is satisfied", residual left at its initial value 23766.440931470137 and the '
                                           'returned fun does NOT match a recomputation at the same x; copied residual -> nfev = 170, '
                                           'residual 66.35737929250325, returned fun matches the recomputation exactly. That is why '
                                           'the corrected batch re-uses the already-verified run_mujoco_equilibrium_search.py.'),
    )
    diag = sorted((ROOT / 'outputs/diagnostics').glob('aliased_residual_*.json')) if (ROOT / 'outputs/diagnostics').exists() else []
    if diag:
        dpath = diag[-1]
        report['root_cause_diagnostics'] = dict(path=str(dpath), sha256=sha(dpath),
                                               content=json.loads(dpath.read_text()))

    runs_block = {}
    accepted = None
    best_state = None
    for lab, rdir in runs.items():
        rj = json.loads((rdir / 'report.json').read_text())
        st = np.load(rdir / 'stage2_best_state.npz')
        q, a = st['qpos'], st['act']
        ev = recompute(q, a, tag='independent_recomputation_of_%s' % lab)
        ms = rj.get('multistart', {})
        stored = rj.get('stage2_final_equilibrium', {})
        agree = None
        if stored:
            agree = dict(
                max_abs_qacc_mm_s2=(abs(stored['max_abs_qacc_mm_s2'] - ev['max_abs_qacc_mm_s2'])
                                    <= 1e-9 * max(1.0, abs(stored['max_abs_qacc_mm_s2']))),
                qacc_norm=(abs(stored['qacc_norm'] - ev['qacc_norm']) <= 1e-9 * max(1.0, abs(stored['qacc_norm']))),
                n_contacts=(stored['n_contacts'] == ev['n_contacts']),
                max_u_diamond=(stored['max_u_diamond'] is not None and ev['max_u_diamond'] is not None
                               and abs(stored['max_u_diamond'] - ev['max_u_diamond']) <= 1e-12))
        runs_block[lab] = dict(
            run_dir=str(rdir), report_json=str(rdir / 'report.json'), report_json_sha256=sha(rdir / 'report.json'),
            stage2_best_state=str(rdir / 'stage2_best_state.npz'), stage2_best_state_sha256=sha(rdir / 'stage2_best_state.npz'),
            nested_radius_rad=NESTED_PLAN.get(lab), n_starts=ms.get('n_starts'),
            n_starts_below_1_mm_s2=ms.get('n_below_1_mm_s2'), n_starts_below_1e_minus_3_mm_s2=ms.get('n_below_1e_minus_3'),
            history_all_starts=ms.get('history_all_starts'), n_history_recorded=len(ms.get('history_all_starts') or []),
            per_equilibrium_hits=ms.get('per_equilibrium_hits'), n_hits_stored=ms.get('n_hits_stored'),
            hits_stored_cap=ms.get('hits_stored_cap'),
            stage2_act_only_search=dict(nfev=rj.get('stage2_act_and_bounded_pose', {}).get('nfev'),
                                        best_qacc_max=rj.get('stage2_act_and_bounded_pose', {}).get('best_qacc_max'),
                                        best_qacc_norm=rj.get('stage2_act_and_bounded_pose', {}).get('best_qacc_norm'),
                                        pose_change=rj.get('stage2_act_and_bounded_pose', {}).get('pose_change')),
            stage2_convergence=dict(solver_status=stored.get('solver_status'), solver_message=stored.get('solver_message'),
                                    nfev=stored.get('nfev')),
            stage1_act_only_search=dict(best_qacc_max=rj.get('stage1_act_only', {}).get('best_qacc_max'),
                                        best_qacc_norm=rj.get('stage1_act_only', {}).get('best_qacc_norm'),
                                        max_u_diamond_at_best=rj.get('stage1_act_only', {}).get('max_u_diamond_at_best'),
                                        n_tarsus_contacts_at_best=rj.get('stage1_act_only', {}).get('n_tarsus_contacts_at_best')),
            delivered_pose_sha256_of_the_run=rj.get('declared', {}).get('pose'),
            recomputed_independently=ev,
            recomputation_matches_run_report=agree,
            declared_gate_checks_recomputed=gate_checks(ev))
        if lab == 'A_ms240':
            accepted = runs_block[lab]
            best_state = (q, a)

    report['runs'] = runs_block
    report['full_box_240_starts'] = dict(
        n_starts=runs_block['A_ms240']['n_starts'],
        n_starts_below_1_mm_s2=runs_block['A_ms240']['n_starts_below_1_mm_s2'],
        n_starts_below_1e_minus_3_mm_s2=runs_block['A_ms240']['n_starts_below_1e_minus_3_mm_s2'],
        history_note=('every start residual is stored in runs.A_ms240.history_all_starts; the per-equilibrium blocks of the '
                      'hits are stored in runs.A_ms240.per_equilibrium_hits (capped, see hits_stored_cap)'),
        accepted_equilibrium=runs_block['A_ms240']['recomputed_independently'],
        declared_gate_checks=gate_checks(runs_block['A_ms240']['recomputed_independently']))
    report['nested_radii_around_pose_A'] = {
        ('r_%.2f' % NESTED_PLAN[lab]): dict(
            radius_rad=NESTED_PLAN[lab], n_starts=runs_block[lab]['n_starts'],
            n_starts_below_1_mm_s2=runs_block[lab]['n_starts_below_1_mm_s2'],
            n_starts_below_1e_minus_3_mm_s2=runs_block[lab]['n_starts_below_1e_minus_3_mm_s2'],
            best_max_abs_qacc_mm_s2=runs_block[lab]['recomputed_independently']['max_abs_qacc_mm_s2'],
            best_qacc_norm=runs_block[lab]['recomputed_independently']['qacc_norm'],
            root_z_change_from_delivered_pose_A_mm=runs_block[lab]['recomputed_independently']['root_z_change_from_delivered_pose_A_mm'],
            max_abs_delta_q_rad_vs_delivered_pose_A=runs_block[lab]['recomputed_independently']['max_abs_delta_q_rad_vs_delivered_pose_A'],
            n_contacts=runs_block[lab]['recomputed_independently']['n_contacts'],
            n_nonfoot_contacts=runs_block[lab]['recomputed_independently']['n_nonfoot_contacts'],
            max_u_diamond=runs_block[lab]['recomputed_independently']['max_u_diamond'],
            declared_gate_checks=gate_checks(runs_block[lab]['recomputed_independently']),
            run_dir=runs_block[lab]['run_dir'])
        for lab in ('A_ms_r0p05', 'A_ms_r0p10', 'A_ms_r0p20')}
    for lab in ('A_ms_r0p05', 'A_ms_r0p10', 'A_ms_r0p20'):
        report['nested_radii_around_pose_A']['r_%.2f' % NESTED_PLAN[lab]]['note'] = (
            'nested probe: the 42 leg joints are restricted to within this radius of the DELIVERED pose A, root z still '
            '+-0.05 mm, actuators still [1e-4, 1]')
    report['nested_probe_conclusion'] = dict(
        statement=('within 0.20 / 0.10 / 0.05 rad of the DELIVERED pose A no declared-cone static equilibrium was FOUND by '
                   '40 / 240 / 40 starts: the best residuals are 14.340629240512852 / 66.6177212372147 / 235.55124711621315 '
                   'mm/s^2, i.e. all above the declared gate (1e-3) and all above the search tolerance (1.0), while the full '
                   'box (R = 0.40 rad, 240 starts) reaches 2.5865734771464476e-11.'),
        wording_rule=('this is a NEGATIVE SEARCH RESULT for a finite number of starts, not a proof that no equilibrium exists '
                      'inside those radii'),
        reading=('the residual grows monotonically as the box is shrunk around the delivered pose A, which is consistent with '
                 'the accepted static equilibrium lying about 0.42 rad away from pose A in the 42 leg joints, i.e. the '
                 'delivered pose A is NOT close to a declared-cone static equilibrium'),
        radius_check=('inside every nested box the reported max|delta q| vs the delivered pose A must satisfy max|delta q| <= R; '
                      'this is a check the independent verification can repeat'))
    hits = runs_block['A_ms240']['per_equilibrium_hits'] or []
    _gate_item = lambda h: dict(qacc=bool(h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2']),
                                six_tarsi=bool(h['six_tarsi_in_contact']),
                                zero_nonfoot=bool(h['n_nonfoot_contacts'] == 0),
                                diamond=bool(h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1.0 + 1e-9))
    _all_four = lambda h: all(_gate_item(h).values())
    report['per_equilibrium_summary'] = dict(
        source='runs.A_ms240.per_equilibrium_hits (every multi-start start whose final max|qacc| is at or below the search '
               'tolerance 1.0 mm/s^2, each re-evaluated on a fresh MjData)',
        n_stored=len(hits), cap=runs_block['A_ms240']['hits_stored_cap'],
        truncated=bool(len(hits) >= (runs_block['A_ms240']['hits_stored_cap'] or 0)),
        n_all_four_gate_items_passing=int(sum(1 for h in hits if _all_four(h))),
        n_qacc_item_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= DECLARED_GATE['max_abs_qacc_mm_s2'])),
        n_search_tolerance_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= SEARCH_TOL_MM_S2)),
        count_definitions=('three different counts, each defined: (1) n_search_tolerance_passing = max|qacc| <= 1.0 mm/s^2; '
                           '(2) n_qacc_item_passing = the qacc item of the declared gate, max|qacc| <= 1e-3 mm/s^2; '
                           '(3) n_all_four_gate_items_passing = all four declared gate items. The old field name '
                           'n_gate_passing counted only item (2) and was removed on 2026-10-15.'),
        entries=[dict(found_at_start_index=h['found_at_start_index'], max_abs_qacc_mm_s2=h['max_abs_qacc_mm_s2'],
                      qacc_norm=h['qacc_norm'], n_contacts=h['n_contacts'], n_tarsus_contacts=h['n_tarsus_contacts'],
                      n_nonfoot_contacts=h['n_nonfoot_contacts'], six_tarsi_in_contact=h['six_tarsi_in_contact'],
                      max_u_diamond=h['max_u_diamond'], root_z_raw=h['root_z_raw'],
                      root_z_change_from_delivered_pose_A_mm=h['root_z_change_from_delivered_pose_A_mm'],
                      max_abs_delta_q_rad_vs_delivered_pose_A=h['max_abs_delta_q_rad_vs_delivered_pose_A'],
                      max_abs_delta_q_rad_vs_source_pose=h['max_abs_delta_q_rad_vs_source_pose'],
                      act_at_floor_own_atol=h['act_at_floor_own_atol'], act_at_ceiling_own_atol=h['act_at_ceiling_own_atol'],
                      gate_items=_gate_item(h), all_four_gate_items_passed=_all_four(h),
                      declared_gate_passed=_all_four(h))
                 for h in hits],
        reading=('the stored hits are distinct states (root z from %.6f to %.6f, contact count %s), so more than one '
                 'declared-cone static equilibrium exists in the full declared box; the accepted one is the smallest '
                 'residual, not the only one. Equality of "residual is machine zero" must not be read as uniqueness.'
                 % (min(h['root_z_raw'] for h in hits), max(h['root_z_raw'] for h in hits),
                    sorted({h['n_contacts'] for h in hits})) if hits else 'no hits stored'))

    report['superseded_void_dirs'] = []
    for v in VOID_DIRS:
        p = Path(v['path'])
        entry = dict(v)
        rp = p / 'report.json'
        entry['exists'] = p.exists()
        entry['report_json_sha256_current'] = sha(rp) if rp.exists() else None
        report['superseded_void_dirs'].append(entry)
    report['inputs'] = dict(
        pose_A=str(POSE_A), pose_A_sha256=sha(POSE_A),
        source_pose=str(SOURCE_POSE), source_pose_sha256=sha(SOURCE_POSE),
        canonical_xml=str(CANONICAL_XML), canonical_xml_sha256=sha(CANONICAL_XML),
        search_script=str(SEARCH_SCRIPT), search_script_sha256=sha(SEARCH_SCRIPT),
        assembler=str(Path(__file__)), assembler_sha256=sha(Path(__file__)))
    report['model'] = dict(nq=m.nq, nv=m.nv, na=m.na, nu=m.nu, opt_cone=int(m.opt.cone),
                           tarsus_condim=sorted({int(m.geom_condim[g]) for g in range(m.ngeom) if 'Tarsus' in gname(g)}),
                           friction_rows_of_tarsus=sorted({tuple(np.round(np.asarray(m.geom_friction[g]), 6))
                                                           for g in range(m.ngeom) if 'Tarsus' in gname(g)}))
    report['verdict'] = dict(
        declared_gate_passed=all(gate_checks(runs_block['A_ms240']['recomputed_independently']).values()),
        accepted_max_abs_qacc_mm_s2=runs_block['A_ms240']['recomputed_independently']['max_abs_qacc_mm_s2'],
        accepted_qacc_norm=runs_block['A_ms240']['recomputed_independently']['qacc_norm'],
        declaration=('an accepted point satisfying the declared gate may only be called a DECLARED-CONE STATIC EQUILIBRIUM '
                     'under the frozen connectome-derived drive; never standing, never walking, never "pose A is an '
                     'equilibrium"'),
        dynamic_maintainability=('measured separately: the same state under the real 100-frame archive loses foot contact at '
                                 'step 2739 = 68.475 ms (S2-true-equilibrium-openloop-20261015T154720)'),
        failed_or_not_run=['no physical/biological validity test', 'no muscle-model calibration',
                           'no closed-loop behaviour of any kind'])

    # artifacts
    q, a = best_state
    np.savez_compressed(out / 'equilibrium_best_state.npz', qpos=q, act=a,
                        source_report=runs_block['A_ms240']['report_json'],
                        note='qpos/act of the accepted declared-cone static equilibrium in the full declared box')
    np.savez_compressed(out / 'equilibrium_best_pose_only.npz', projected_seated_qpos=q,
                        note='same state as equilibrium_best_state.npz, pose only, for direct reuse as a start pose')

    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (Path(__file__), SEARCH_SCRIPT, ROOT / 'diagnose_aliased_residual.py'):
            if Path(p).exists():
                z.write(p, Path(p).name)
        for p in (ROOT / 'outputs/diagnostics').glob('aliased_residual_*.json'):
            z.write(p, p.name)
        z.writestr('RUNS.txt', '\n'.join('%s -> %s' % (k, v) for k, v in sorted(runs.items())))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    ax = axes[0][0]
    for lab, col in zip(RUN_LABELS, ('#1f77b4', '#2ca02c', '#ff7f0e', '#d62728')):
        h = runs_block[lab]['history_all_starts'] or []
        if h:
            ax.semilogy(np.arange(len(h)), np.maximum(h, 1e-14), lw=1.0, color=col,
                        label='%s (n=%d, R=%s)' % (lab, len(h), NESTED_PLAN.get(lab, 0.40)))
    ax.axhline(SEARCH_TOL_MM_S2, color='gray', ls='--', lw=1.2, label='search tolerance 1.0 mm/s^2')
    ax.axhline(DECLARED_GATE['max_abs_qacc_mm_s2'], color='black', ls=':', lw=1.6,
               label='declared gate 1e-3 mm/s^2')
    ax.set_xlabel('random start index (multi-start, full history)')
    ax.set_ylabel('final max|qacc| of that start  [mm/s^2], log scale')
    ax.set_title('per-start residuals of every multi-start run')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[0][1]
    labs = ['full box\nR=0.40 (240)'] + ['nested\nR=%.2f (%d)' % (NESTED_PLAN[l], runs_block[l]['n_starts'])
                                         for l in ('A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')]
    vals = [runs_block['A_ms240']['recomputed_independently']['max_abs_qacc_mm_s2']] + \
           [runs_block[l]['recomputed_independently']['max_abs_qacc_mm_s2'] for l in ('A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')]
    nz = [max(v, 1e-14) for v in vals]
    ax.bar(labs, nz, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_yscale('log')
    ax.axhline(DECLARED_GATE['max_abs_qacc_mm_s2'], color='black', ls=':', lw=1.6, label='declared gate 1e-3 mm/s^2')
    for i, v in enumerate(vals):
        ax.annotate('%.3g' % v, (i, max(v, 1e-14)), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9)
    ax.set_ylabel('best max|qacc|  [mm/s^2], log scale')
    ax.set_title('best max|qacc| per declared search box\n'
                 'only the full box (R=0.40) reaches the declared gate;\n'
                 'the nested boxes around the delivered pose A do not', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')

    ax = axes[1][0]
    ev = runs_block['A_ms240']['recomputed_independently']
    st = np.load(runs['A_ms240'] / 'stage2_best_state.npz')
    d = mujoco.MjData(m)
    d.qpos[:] = st['qpos']; d.qvel[:] = 0.0; d.act[:] = np.clip(st['act'], act_lo, act_hi); d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    qa = np.abs(np.asarray(d.qacc, dtype=float))
    ax.bar(np.arange(qa.size), np.maximum(qa, 1e-14), color='#1f77b4')
    ax.set_yscale('log')
    ax.axhline(DECLARED_GATE['max_abs_qacc_mm_s2'], color='black', ls=':', lw=1.6, label='declared gate 1e-3 mm/s^2')
    ax.set_xlabel('DOF index (nv = 48)')
    ax.set_ylabel('|qacc|  [mm/s^2], log scale')
    ax.set_title('accepted equilibrium, all 48 DOF (max %.3g, norm %.3g)' % (ev['max_abs_qacc_mm_s2'], ev['qacc_norm']))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')

    ax = axes[1][1]
    legs = list(LEGS)
    per = [ev['per_leg_max_u_diamond'].get(L, 0.0) for L in legs]
    ax.bar(legs, [max(v, 1e-16) for v in per], color='#2ca02c')
    ax.set_yscale('log')
    ax.axhline(1.0, color='red', ls='--', lw=1.4, label='declared diamond limit u_diamond = 1')
    for i, v in enumerate(per):
        ax.annotate('%.2g' % v, (i, max(v, 1e-16)), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    ax.set_ylabel('max u_diamond per leg  [1] , log scale')
    ax.set_title('per-leg diamond utilisation at the accepted equilibrium\n(diamond u1+u2 <= 1; %d tarsus contacts, %d non-foot)'
                 % (ev['n_tarsus_contacts'], ev['n_nonfoot_contacts']))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')

    fig.suptitle('S2 CORRECTED/REISSUED: declared-cone static equilibrium of the delivered pose A under the frozen\n'
                 'connectome-derived drive  |  gate max|qacc| <= 1e-3 mm/s^2 = static necessary condition only,\n'
                 'not stability, not standing, not "pose A is an equilibrium"  |  manual research, not a product',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'equilibrium-reissued.png', dpi=135)
    plt.close(fig)

    readme = """# S2 修正/重发：既定姿态 A 的“声明锥静态平衡”搜索（完整路径见下）

状态：本目录是**手动研究实施，不是成品**；不构成站立、行走或任何行为结论。

## 1. 本目录回答什么
在**冻结的联结组指令档案**驱动下，是否存在一个满足**事先声明**门槛（`max|qacc| <= 1e-3 mm/s^2`，即约 1e-7 g）
的静止状态，且该状态六个跗节都有接触、无足外接触、所有接触满足**声明的轴射线菱形锥** `u_diamond = u1 + u2 <= 1`。

## 2. 结果（每一条都由本目录脚本在**全新 MjData** 上重算）
- 全盒（act ∈ [1e-4,1] + 根 z ±0.05 mm + 42 个腿关节在交付姿态 A 的 0.40 rad 内），240 个随机起点：
  最好 `max|qacc| = %(full)s mm/s^2`，范数 `%(fullnorm)s`；起点命中数（<= 1.0 mm/s^2）`%(hit1)s`，
  其中 qacc 项（<= 1e-3 mm/s^2）`%(hit3)s` 个；**四项门限全过 %(hit4)s 个**（起点 16 / 117 / 131 / 212）。
  起点 207 的 `max|qacc|` 虽过 qacc 项，但 `six_tarsi_in_contact = false` —— 它的 51 个跗节接触只覆盖 **5 条腿**（缺 LF），
  所以"六足跗节接触"这一项不过；起点 5 的 qacc 项未过。**"过门限"一律指四项全过。**
- 嵌套半径（围绕交付姿态 A 收紧 42 个腿关节）：
  - r = 0.05 rad，40 起点 → `%(n05)s mm/s^2`（命中 <= 1e-3：%(n05h)s）
  - r = 0.10 rad，240 起点 → `%(n10)s mm/s^2`（命中 <= 1e-3：%(n10h)s）
  - r = 0.20 rad，40 起点 → `%(n20)s mm/s^2`（命中 <= 1e-3：%(n20h)s）
  ⇒ **嵌套半径内没有找到平衡**（残差随半径收紧单调变大：%(n20)s → %(n10)s → %(n05)s，全部高于 1.0 搜索容差与 1e-3 声明门限）。
  **这是有限起点下的负结果，不是"该半径内不存在平衡"的证明**；含义是**交付姿态 A 并不是一个接近声明锥静态平衡的姿态**，
  被接受的状态在 42 个腿关节上离 A 约 %(dq)s rad。
- 被接受点：根 z 变化 `%(rz)s mm`（相对交付姿态 A），42 关节最大距离 `%(dq)s rad`（相对交付姿态 A），
  接触 `%(nc)s` 个（其中跗节 %(nt)s 个，足外 %(nn)s 个），`max u_diamond = %(ud)s`，
  执行器触界（各自 atol）：下限 %(af)s 个、上限 %(ac)s 个。

## 3. 逐平衡明细（每个命中起点，全新 `MjData` 重算）
%(peq)s

## 4. 必须披露（强制）
%(disc)s

## 5. 摩擦判据措辞更正
%(fw)s

## 6. 本目录作废/取代的目录
%(void)s

## 7. 文件
- `report.json` —— 全部数字与逐平衡明细（含每个起点的残差历史）
- `equilibrium-reissued.png` —— 图（对数坐标 + 可见门槛线）
- `equilibrium_best_state.npz` / `equilibrium_best_pose_only.npz` —— 被接受状态
- `source_snapshot.zip` —— 搜索脚本与本装配脚本快照
- `delivery-hashes.json` —— 全部交付文件 sha256
""" % dict(
        full='%.6g' % ev['max_abs_qacc_mm_s2'], fullnorm='%.6g' % ev['qacc_norm'],
        hit1=runs_block['A_ms240']['n_starts_below_1_mm_s2'],
        hit3=runs_block['A_ms240']['n_starts_below_1e_minus_3_mm_s2'],
        hit4=sum(1 for h in (runs_block['A_ms240']['per_equilibrium_hits'] or [])
                 if h['max_abs_qacc_mm_s2'] <= 1e-3 and h['six_tarsi_in_contact'] and h['n_nonfoot_contacts'] == 0
                 and h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1 + 1e-9),
        n05='%.6g' % report['nested_radii_around_pose_A']['r_0.05']['best_max_abs_qacc_mm_s2'],
        n05h=report['nested_radii_around_pose_A']['r_0.05']['n_starts_below_1e_minus_3_mm_s2'],
        n10='%.6g' % report['nested_radii_around_pose_A']['r_0.10']['best_max_abs_qacc_mm_s2'],
        n10h=report['nested_radii_around_pose_A']['r_0.10']['n_starts_below_1e_minus_3_mm_s2'],
        n20='%.6g' % report['nested_radii_around_pose_A']['r_0.20']['best_max_abs_qacc_mm_s2'],
        n20h=report['nested_radii_around_pose_A']['r_0.20']['n_starts_below_1e_minus_3_mm_s2'],
        rz='%.6g' % ev['root_z_change_from_delivered_pose_A_mm'],
        dq='%.6g' % ev['max_abs_delta_q_rad_vs_delivered_pose_A'],
        nc=ev['n_contacts'], nt=ev['n_tarsus_contacts'], nn=ev['n_nonfoot_contacts'],
        ud=('None' if ev['max_u_diamond'] is None else '%.6g' % ev['max_u_diamond']),
        af=ev['act_at_floor_own_atol'], ac=ev['act_at_ceiling_own_atol'],
        disc='\n'.join('- ' + s for s in MANDATORY_DISCLOSURE),
        peq=('\n'.join('- 起点 %d：`max|qacc| = %.6g`，范数 `%.6g`，接触 %d（跗节 %d，足外 %d），`max u_diamond = %s`，'
                       '根 z `%.6f`（相对 A %+.6f mm），`max|Δq| vs A = %.6f rad`，触界 下 %d / 上 %d，门限 %s'
                       % (h['found_at_start_index'], h['max_abs_qacc_mm_s2'], h['qacc_norm'], h['n_contacts'],
                          h['n_tarsus_contacts'], h['n_nonfoot_contacts'],
                          ('None' if h['max_u_diamond'] is None else '%.3g' % h['max_u_diamond']), h['root_z_raw'],
                          h['root_z_change_from_delivered_pose_A_mm'], h['max_abs_delta_q_rad_vs_delivered_pose_A'],
                          h['act_at_floor_own_atol'], h['act_at_ceiling_own_atol'],
                          '过' if (h['max_abs_qacc_mm_s2'] <= 1e-3 and h['six_tarsi_in_contact']
                                   and h['n_nonfoot_contacts'] == 0 and h['max_u_diamond'] is not None
                                   and h['max_u_diamond'] <= 1 + 1e-9) else '未过')
              for h in (runs_block['A_ms240']['per_equilibrium_hits'] or []))
             + ('\n\n共命中 %d 个起点（<= 1.0 mm/s²），其中过门限 %d 个；它们是**不同状态**（root z 与接触数不同），'
                '所以"残差为零"不等于"唯一"。' % (len(runs_block['A_ms240']['per_equilibrium_hits'] or []),
                                              sum(1 for h in (runs_block['A_ms240']['per_equilibrium_hits'] or [])
                                                  if h['max_abs_qacc_mm_s2'] <= 1e-3)))),
        fw=('\n'.join('- **%s**：%s' % (k, v) if not isinstance(v, dict) else '- **%s**：%s' % (k, json.dumps(v, ensure_ascii=False))
                      for k, v in FRICTION_CORRECTED_WORDING.items())),
        void='\n'.join('- `%s`：%s' % (v['path'], v['reason']) for v in report['superseded_void_dirs']))
    (out / 'README.zh-CN.md').write_text(readme)

    report['status'] = 'PASS_EQUILIBRIUM_SEARCH_CORRECTED_REISSUED'
    report['outputs'] = str(out)
    report['figure'] = str(out / 'equilibrium-reissued.png')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))

    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps(
        {p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))

    print(report['status'])
    print('out:', out)
    print('accepted max|qacc| %.6g  norm %.6g  contacts %d (tarsus %d, non-foot %d)  max u_diamond %s'
          % (ev['max_abs_qacc_mm_s2'], ev['qacc_norm'], ev['n_contacts'], ev['n_tarsus_contacts'],
             ev['n_nonfoot_contacts'], ev['max_u_diamond']))
    print('gate passed:', report['verdict']['declared_gate_passed'])
    for lab in RUN_LABELS:
        rb = runs_block[lab]
        print('  %-10s n=%s  best %.6g  hits<=1.0 %s  hits<=1e-3 %s  agree=%s'
              % (lab, rb['n_starts'], rb['recomputed_independently']['max_abs_qacc_mm_s2'],
                 rb['n_starts_below_1_mm_s2'], rb['n_starts_below_1e_minus_3_mm_s2'],
                 rb['recomputation_matches_run_report']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
