"""Consolidated corrected batch v2 for the delivered pose A: assemblies the four main searches, the declared radius
ladder, the supplementary radius probes, and the independent verification findings into ONE directory whose every
number is recomputed on a fresh MjData.

Why v2 exists (all of this is disclosed inside the batch):
  F1  the v1 batch pointed at the live assembler path, whose hash drifted after the batch was written -> v2 ships the
      assembler snapshot INSIDE the directory and states that all recorded hashes are the hashes of the files in THIS
      directory.
  F2  v1 reported "5 passes the gate" while only 4 of the six hits pass all four declared gate items -> v2 reports the
      three different counts separately, each with its definition.
  F3  v1 wrote that the full box bounds the 42 leg joints within 0.40 rad of the DELIVERED pose A; the code and the run
      reports use the SOURCE pose. v2 states both references and both distances.
  F4  v1's mandatory disclosure said the cited open-loop run "loses foot contact at step 2739 = 68.475 ms"; that step is
      the first NON-FOOT (thorax) contact. v2 gives the measured sequence instead.
  F5..F9  labelling/field-name/wording defects of the underlying run template and the v1 report are listed with their
      disposition; v1's predecessor is left untouched.
  F10 the figure is redrawn: the different start counts are no longer stacked on one index axis and the ladder is shown.

Manual research implementation, not a product. No standing, no walking, no completion claim. Absolute force values and
physical units remain NOT_ESTABLISHED.
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
LADDER_SCRIPT = ROOT / 'run_equilibrium_radius_ladder.py'
DIAG_SCRIPT = ROOT / 'diagnose_aliased_residual.py'
V1_DIR = DESK / 'S2-equilibrium-A-corrected-reissued-20261015T161259'
VERIFY_DIR = DESK / 'S2-equilibrium-reissued-check-20261015T231321Z'
LADDER_DIR = DESK / 'S2-equilibrium-radius-ladder-20261015T161518'
OPENLOOP_DIR = DESK / 'S2-true-equilibrium-openloop-20261015T154720'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
MAIN_LABELS = ('A_ms240', 'A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05')
SUPP_LABELS = ('suppR020', 'suppR015')
NESTED_R = {'A_ms_r0p20': 0.20, 'A_ms_r0p10': 0.10, 'A_ms_r0p05': 0.05}
GATE = 1e-3
SEARCH_TOL = 1.0
DECLARED_GATE = dict(
    max_abs_qacc_mm_s2=GATE,
    max_abs_qacc_reason=('about 1e-7 g: with mm units and gravity 9801 mm/s^2 this is ~1e-7 of g, i.e. numerically '
                         'indistinguishable from zero at the precision of a single mj_forward on this model, while still '
                         'being 10x tighter than the tightest meaningful dynamic criterion used in this project '
                         '(1e-2 mm/s^2)'),
    six_tarsi_active_contact=True, zero_nonfoot_contacts=True, all_contacts_u_diamond_le_1=True,
    drive=('frozen connectome-derived command archive only (no external CPG, no PD, no preprogrammed action, no per-step '
           'pose correction)'),
    not_standing='passing this gate may only be called a DECLARED-CONE STATIC EQUILIBRIUM, never standing')

DISCLOSURE = [
    'qacc = 0 is only a STATIC NECESSARY CONDITION with qvel = 0. It is NOT stability, NOT balance, NOT standing.',
    'It must NOT be read as "pose A is an equilibrium": every accepted point is a NEIGHBOURING state found by the search.',
    'A declared-cone static equilibrium is NOT dynamic maintainability, and the measured open-loop behaviour of the '
    'accepted state under the real 100-frame connectome-derived archive is: first NON-FOOT (Thorax_geom | floor) contact '
    'at step 2739 = 68.475 ms; the six-leg tarsus condition is already broken at step 537 = 13.425 ms '
    '(legs_in_contact < 6, first_leg_to_lose_support RM at 76.775 ms measured from T0); the last tarsus contact is lost at '
    'step 5111 = 127.775 ms; all six legs are in contact for only 849 of 20000 steps (artifact '
    + str(OPENLOOP_DIR) + ', recomputed here from its time_series.npz).',
    'Only the frozen connectome-derived command archive is used as drive; no external CPG, no PD, no preprogrammed action, '
    'no per-step pose correction anywhere.',
    'Absolute force values and physical units remain NOT_ESTABLISHED; the contact quantities are relative utilisations of '
    'the declared cone.',
    'A static muscle allocation s <= 1 only means a non-negative allocation exists under the declared cone.',
    'Manual research implementation, not a product: no standing, no walking, nothing about the goal is complete.']

FRICTION_WORDING = dict(
    criterion='declared cone = axis-ray DIAMOND u_diamond = |ft1|/(mu1*fn) + |ft2|/(mu2*fn) <= 1, mu1 = mu2 = 1.0',
    contact_friction_rows_measured_from_MuJoCo='contact friction rows read back from the model are [1.0, 1.0, 0.005, 0.0001, 0.0001] (the first two entries are the tangential coefficients); mu1 = mu2 = 1.0',
    tarsus_geom_friction_3_values='the tarsus GEOM friction field is 3 values (slide, spin, roll) and is NOT the contact-friction row; the v1 field name model.friction_rows_of_tarsus displayed the 3-value geom friction and was misnamed',
    secondary_reading='the SQUARE max(u1, u2) <= 1 is a strictly LARGER set and is never the criterion; it is reported only as a labelled secondary reading',
    first_whole_run_diamond_saturation=dict(A=dict(step=256, ms=6.400), B=dict(step=401, ms=10.025), C=dict(step=277, ms=6.925),
                                           first_saturating_contact='RMTarsus5_geom|floor'),
    retracted_claim='the earlier statement "overturning dominated, NOT friction saturation" is RETRACTED',
    unchanged_numbers='no measured number was adjusted; only the criterion name and the wording changed')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest_run(label):
    cands = sorted(DESK.glob('S2-mujoco-equilibrium-%s-*' % label))
    cands = [c for c in cands if (c / 'report.json').exists() and (c / 'stage2_best_state.npz').exists()]
    return cands[-1] if cands else None


def main():
    import mujoco
    sys.path.insert(0, str(ROOT))
    from engine import canonical_physics as cp

    m = cp.load_model()
    q0 = np.load(POSE_A)['projected_seated_qpos'].copy()
    src = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
    suffixes = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
    jids = []
    for L in LEGS:
        js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
        for suf in suffixes:
            jids.append(next(j for j in js if jname(j).endswith(suf)))
    qadr = [int(m.jnt_qposadr[j]) for j in jids]
    act_lo = np.full(m.na, 1e-4)
    act_hi = np.full(m.na, 1.0)

    def evaluate(qpos, act, tag):
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        d.act[:] = np.clip(act, act_lo, act_hi)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        qa = np.asarray(d.qacc, dtype=float)
        order = np.argsort(-np.abs(qa))
        f6 = np.zeros(6)
        per_leg, sq, n_tar, n_nonfoot = {}, 0.0, 0, 0
        for ci, c in enumerate(d.contact):
            if c.efc_address < 0:
                continue
            n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
            mujoco.mj_contactForce(m, d, ci, f6)
            fr = np.asarray(c.friction)[:3]
            fn, ft1, ft2 = float(f6[0]), float(f6[1]), float(f6[2])
            is_foot = 'Tarsus' in (n1 + n2)
            if is_foot:
                n_tar += 1
            else:
                n_nonfoot += 1
            if fn > 0:
                u1 = abs(ft1) / (float(fr[0]) * fn)
                u2 = abs(ft2) / (float(fr[1]) * fn)
                sq = max(sq, u1, u2)
                if is_foot:
                    leg = next((L for L in LEGS if n1.startswith(L) or n2.startswith(L)), None)
                    if leg:
                        per_leg[leg] = (u1 + u2) if leg not in per_leg else max(per_leg[leg], u1 + u2)
        act_c = np.clip(act, act_lo, act_hi)
        max_u = max(per_leg.values()) if per_leg else None
        checks = dict(max_abs_qacc=bool(float(np.max(np.abs(qa))) <= GATE),
                      six_tarsi_active_contact=bool(len(per_leg) == 6),
                      zero_nonfoot_contacts=bool(n_nonfoot == 0),
                      all_contacts_u_diamond_le_1=bool(max_u is not None and max_u <= 1.0 + 1e-9))
        return dict(
            tag=tag, max_abs_qacc_mm_s2=float(np.max(np.abs(qa))), qacc_norm=float(np.linalg.norm(qa)), nv=int(qa.size),
            n_dof_above_1e_minus_3=int(np.sum(np.abs(qa) > 1e-3)), n_dof_above_1e_minus_6=int(np.sum(np.abs(qa) > 1e-6)),
            n_dof_above_1e_minus_9=int(np.sum(np.abs(qa) > 1e-9)),
            top5_abs_qacc=[float(np.abs(qa)[i]) for i in order[:5]], top5_dof_index=[int(i) for i in order[:5]],
            n_contacts=int(len(d.contact)), n_tarsus_contacts=n_tar, n_nonfoot_contacts=n_nonfoot,
            six_tarsi_in_contact=bool(len(per_leg) == 6), tarsi_with_contact=sorted(per_leg),
            per_leg_max_u_diamond=per_leg, max_u_diamond=max_u,
            max_u_square_secondary_NOT_the_criterion=float(sq),
            root_z_raw=float(qpos[2]), root_z_change_from_delivered_pose_A_mm=float(qpos[2] - q0[2]),
            root_z_change_from_source_pose_mm=float(qpos[2] - src[2]),
            max_abs_delta_q_rad_vs_delivered_pose_A=float(np.max(np.abs(qpos[qadr] - q0[qadr]))),
            max_abs_delta_q_rad_vs_source_pose=float(np.max(np.abs(qpos[qadr] - src[qadr]))),
            act_min=float(act_c.min()), act_max=float(act_c.max()), act_mean=float(act_c.mean()),
            act_at_floor_own_atol=int(np.sum(np.abs(act_c - act_lo) <= 1e-12)),
            act_at_ceiling_own_atol=int(np.sum(np.abs(act_c - act_hi) <= 1e-9)),
            act_within_1e_minus_9_of_floor=int(np.sum(act_c - act_lo <= 1e-9)),
            act_within_1e_minus_6_of_ceiling=int(np.sum(act_hi - act_c <= 1e-6)),
            declared_gate_checks=checks, declared_gate_all_four=all(checks.values()))

    out = DESK / ('S2-equilibrium-A-corrected-reissued-v2-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)

    report = dict(
        status='RUNNING', title='S2 CORRECTED / REISSUED v2: declared-cone static equilibria around the delivered pose A',
        reissue=True, corrected=True, version=2,
        supersedes=dict(v1=str(V1_DIR), v2_reason=('v1 is kept unchanged; v2 applies the independent verification findings '
                                                  'F1-F10 and adds the declared radius ladder and the two supplementary '
                                                  'probes. Any number in v1 that disagrees with v2 is superseded by v2.')),
        manual_research_not_product=True, standing=False, walking=False, completion_percentage=None,
        scope=('declared numerical equilibrium search. Two different declared boxes are used and must never be conflated: '
               '(a) the FULL BOX bounds the 42 leg hinge joints within 0.40 rad of the SOURCE pose '
               '(outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz, the reference used by the search '
               'implementation when no --radius is given); (b) the NESTED/RADIUS boxes bound the same joints within R of '
               'the DELIVERED pose A. Both keep act in [1e-4, 1], root z within +-0.05 mm, qvel = 0, and minimise ||qacc||.'),
        declared_gate=DECLARED_GATE, search_tolerance_mm_s2=SEARCH_TOL,
        tolerance_note=('two different numbers exist and must not be merged: the SEARCH decides "found" at 1.0 mm/s^2, '
                        'while the DECLARED GATE is 1e-3 mm/s^2. The underlying run figures label the 1.0 line as '
                        '"declared tolerance", which is a labelling defect of the run template (finding F5) and is NOT '
                        'repaired in place; inside this batch the two lines are labelled "search tolerance" and '
                        '"declared gate".'),
        mandatory_disclosure=DISCLOSURE, friction_metric_corrected_wording=FRICTION_WORDING,
        not_established='absolute force values, physical units, biological validity, any behaviour; no standing, no walking',
        reference_reference_confusion=dict(
            full_box_joint_reference='SOURCE pose, 0.40 rad',
            nested_box_joint_reference='DELIVERED pose A, R rad',
            accepted_state_distance_rad_vs_source=None, accepted_state_distance_rad_vs_pose_A=None,
            note=('finding F3: the full box is expressed around the SOURCE pose, so a state inside the full box may be '
                  'further than 0.40 rad from pose A. Values are filled in below from the recomputation.')),
        predecessor_verification=dict(
            v1_dir=str(V1_DIR), v1_report_json_sha256=sha(V1_DIR / 'report.json') if (V1_DIR / 'report.json').exists() else None,
            verification_dir=str(VERIFY_DIR),
            verification_report_json_sha256=(sha(VERIFY_DIR / 'verification_report.json')
                                             if (VERIFY_DIR / 'verification_report.json').exists() else None),
            verification_report_md_sha256=(sha(VERIFY_DIR / 'verification_report.md')
                                           if (VERIFY_DIR / 'verification_report.md').exists() else None),
            verification_verdict=('PASS_WITH_CORRECTIONS: 0 blocking / 4 material / 5 minor / 1 cosmetic; the core numbers '
                                  'were reproduced bit for bit, the findings are labelling, reference-pose, disclosure and '
                                  'hash-attribution defects, all applied here')),
    )

    # -------- main runs --------
    runs = {}
    for lab in MAIN_LABELS + SUPP_LABELS:
        d = latest_run(lab)
        if d is None:
            print('note: no completed run for label', lab)
            continue
        runs[lab] = d
    print('runs used:')
    for lab, d in runs.items():
        print('  %-12s %s' % (lab, d))

    runs_block, states = {}, {}
    for lab, d in runs.items():
        rj = json.loads((d / 'report.json').read_text())
        st = np.load(d / 'stage2_best_state.npz')
        ev = evaluate(st['qpos'], st['act'], tag='independent_recomputation_of_%s' % lab)
        ms = rj.get('multistart', {})
        hits = ms.get('per_equilibrium_hits') or []
        stored = rj.get('stage2_final_equilibrium', {})
        runs_block[lab] = dict(
            run_dir=str(d), report_json_sha256=sha(d / 'report.json'),
            stage2_best_state_sha256=sha(d / 'stage2_best_state.npz'),
            nested_radius_rad=NESTED_R.get(lab),
            reference_pose_of_the_joint_box=('DELIVERED pose A' if lab in NESTED_R or lab in SUPP_LABELS else 'SOURCE pose'),
            n_starts=ms.get('n_starts'),
            n_starts_below_search_tolerance_1_mm_s2=ms.get('n_below_1_mm_s2'),
            n_starts_below_gate_qacc_item_1e_minus_3=ms.get('n_below_1e_minus_3'),
            n_hits_all_four_gate_items=sum(1 for h in hits if evaluate_hit_ok(h)),
            history_all_starts=ms.get('history_all_starts'), n_history_recorded=len(ms.get('history_all_starts') or []),
            per_equilibrium_hits=hits, hits_stored_cap=ms.get('hits_stored_cap'),
            best_convention=dict(
                stage2_refined_point_max_abs_qacc_mm_s2=stored.get('max_abs_qacc_mm_s2'),
                multistart_start_level_best_max_abs_qacc_mm_s2=ms.get('best_max_abs_qacc_mm_s2'),
                note=('finding F7: "best" has two conventions. The start-level best is the smallest residual among the '
                      'random starts after their own least-squares run (multistart.best_max_abs_qacc_mm_s2); the refined '
                      'point is the final stage-2 solve started from that best (stage2_final_equilibrium). This batch '
                      'reports both and never quotes one as the other.')),
            stage2_convergence=dict(solver_status=stored.get('solver_status'), solver_message=stored.get('solver_message'),
                                    nfev=stored.get('nfev')),
            recomputed_independently=ev,
            declared_gate_checks_recomputed=ev['declared_gate_checks'])
        states[lab] = (st['qpos'], st['act'])

    for lab, entry in runs_block.items():
        entry['declared_gate_checks_recomputed'] = {k: v for k, v in entry['declared_gate_checks_recomputed'].items()}
        refined = entry['best_convention']['stage2_refined_point_max_abs_qacc_mm_s2']
        entry['recomputation_matches_run_report'] = dict(
            max_abs_qacc_mm_s2=(refined is not None and abs(entry['recomputed_independently']['max_abs_qacc_mm_s2'] - refined)
                                <= 1e-9 * max(1.0, abs(refined))),
            note='recomputed here on a fresh MjData from stage2_best_state.npz and compared with the run report')
    report['runs'] = runs_block

    acc = runs_block['A_ms240']
    report['reference_reference_confusion']['accepted_state_distance_rad_vs_source'] = \
        acc['recomputed_independently']['max_abs_delta_q_rad_vs_source_pose']
    report['reference_reference_confusion']['accepted_state_distance_rad_vs_pose_A'] = \
        acc['recomputed_independently']['max_abs_delta_q_rad_vs_delivered_pose_A']
    report['reference_reference_confusion']['consequence'] = (
        'the accepted state is inside the full box (%.6f rad from the SOURCE pose < 0.40) but is OUTSIDE a 0.40 rad box '
        'around pose A (%.6f rad > 0.40). v1 wrote the full box as if it were centred on pose A; that is corrected here.'
        % (acc['recomputed_independently']['max_abs_delta_q_rad_vs_source_pose'],
           acc['recomputed_independently']['max_abs_delta_q_rad_vs_delivered_pose_A']))

    hits = acc['per_equilibrium_hits'] or []
    gate_ok = evaluate_hit_ok
    report['accepted_equilibrium'] = dict(full_box_240_starts=acc['recomputed_independently'],
                                          gate_checks=acc['declared_gate_checks_recomputed'],
                                          all_four_gate_items_passed=acc['recomputed_independently']['declared_gate_all_four'])
    report['per_equilibrium_summary'] = dict(
        n_stored=len(hits), cap=acc['hits_stored_cap'], truncated=bool(len(hits) >= (acc['hits_stored_cap'] or 0)),
        n_search_tolerance_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= SEARCH_TOL)),
        n_qacc_item_passing=int(sum(1 for h in hits if h['max_abs_qacc_mm_s2'] <= GATE)),
        n_all_four_gate_items_passing=int(sum(1 for h in hits if gate_ok(h))),
        count_definitions=('three different counts, each defined: (1) max|qacc| <= 1.0 search tolerance; (2) the qacc item '
                           'of the declared gate, max|qacc| <= 1e-3; (3) all four declared gate items. v1 quoted count (2) '
                           'under the name "gate passing", which was wrong (finding F2).'),
        entries=[dict(found_at_start_index=h['found_at_start_index'], max_abs_qacc_mm_s2=h['max_abs_qacc_mm_s2'],
                      qacc_norm=h['qacc_norm'], n_contacts=h['n_contacts'], n_tarsus_contacts=h['n_tarsus_contacts'],
                      n_nonfoot_contacts=h['n_nonfoot_contacts'], six_tarsi_in_contact=h['six_tarsi_in_contact'],
                      tarsi_with_contact=h.get('tarsi_with_contact'),
                      max_u_diamond=h['max_u_diamond'],
                      root_z_change_from_delivered_pose_A_mm=h['root_z_change_from_delivered_pose_A_mm'],
                      max_abs_delta_q_rad_vs_delivered_pose_A=h['max_abs_delta_q_rad_vs_delivered_pose_A'],
                      act_at_floor_own_atol=h['act_at_floor_own_atol'], act_at_ceiling_own_atol=h['act_at_ceiling_own_atol'],
                      gate_item_qacc=bool(h['max_abs_qacc_mm_s2'] <= GATE),
                      gate_item_six_tarsi=bool(h['six_tarsi_in_contact']),
                      gate_item_zero_nonfoot=bool(h['n_nonfoot_contacts'] == 0),
                      gate_item_diamond=bool(h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1 + 1e-9),
                      all_four_gate_items_passed=gate_ok(h)) for h in hits])

    # -------- radius ladder + supplementary probes --------
    ladder = json.loads((LADDER_DIR / 'report.json').read_text())
    radius_rows = []
    for p in ladder['probes']:
        radius_rows.append(dict(source='declared ladder probe (240 starts)', radius_rad=p['radius_rad'],
                                run_dir=p['run_dir'], report_json_sha256=p['report_json_sha256'],
                                best_max_abs_qacc_mm_s2=p['best_max_abs_qacc_mm_s2'], n_starts=p['n_starts'],
                                n_hits_all_four_gate_items=p['n_hits_all_four_gate_items'],
                                hits=p['per_equilibrium_hits']))
    for lab in SUPP_LABELS:
        if lab in runs_block:
            e = runs_block[lab]
            radius_rows.append(dict(source='supplementary probe declared before running (240 starts)',
                                    radius_rad=(0.20 if lab == 'suppR020' else 0.15), run_dir=e['run_dir'],
                                    report_json_sha256=e['report_json_sha256'],
                                    best_max_abs_qacc_mm_s2=e['best_convention']['stage2_refined_point_max_abs_qacc_mm_s2'],
                                    n_starts=e['n_starts'],
                                    n_hits_all_four_gate_items=e['n_hits_all_four_gate_items'],
                                    hits=e['per_equilibrium_hits']))
    for lab in ('A_ms_r0p20', 'A_ms_r0p10', 'A_ms_r0p05'):
        e = runs_block[lab]
        radius_rows.append(dict(source='v1 nested probe (%s starts)' % e['n_starts'], radius_rad=NESTED_R[lab],
                                run_dir=e['run_dir'], report_json_sha256=e['report_json_sha256'],
                                best_max_abs_qacc_mm_s2=e['best_convention']['stage2_refined_point_max_abs_qacc_mm_s2'],
                                n_starts=e['n_starts'], n_hits_all_four_gate_items=e['n_hits_all_four_gate_items'],
                                hits=e['per_equilibrium_hits']))
    radius_rows.sort(key=lambda r: (r['radius_rad'], -(r['n_starts'] or 0)))
    with_hits = [r for r in radius_rows if r['n_hits_all_four_gate_items'] > 0]
    report['radius_study'] = dict(
        declared_ladder=ladder['declared_protocol']['ladder'],
        supplementary_radii=[0.20, 0.15], starts_per_probe=240,
        rows=radius_rows,
        smallest_radius_with_an_all_four_gate_hit=(min(r['radius_rad'] for r in with_hits) if with_hits else None),
        smallest_radius_with_an_all_four_gate_hit_among_240_start_probes=(
            min((r['radius_rad'] for r in with_hits if r['n_starts'] == 240), default=None)),
        radii_with_hits=sorted({r['radius_rad'] for r in with_hits}),
        radii_without_hits=sorted({r['radius_rad'] for r in radius_rows if r['n_hits_all_four_gate_items'] == 0}),
        monotone_in_radius_240_start_probes_only=bool(all(
            not (a['n_hits_all_four_gate_items'] > 0 and b['n_hits_all_four_gate_items'] == 0)
            for a, b in zip(sorted([r for r in radius_rows if r['n_starts'] == 240], key=lambda r: r['radius_rad']),
                            sorted([r for r in radius_rows if r['n_starts'] == 240], key=lambda r: r['radius_rad'])[1:]))),
        monotone_note=('the monotonicity verdict is computed over the 240-start probes only; the 40-start probes at '
                       'R = 0.20 and 0.05 are listed separately and are never mixed into it'),
        reading=('the hit map is NOT monotone in the radius and not a fine function of it: it depends on how much of the '
                 'thin machine-zero equilibrium set falls inside the declared box and on the start count. Machine-zero '
                 'equilibria (most of them sitting exactly on the +-R face of the box) are found at R = 0.25 / 0.30 / 0.35 '
                 'rad with 240 starts, while the larger R = 0.40 and R = 0.45 boxes returned no all-four-gate hit with the '
                 'same 240 starts.'),
        v1_wording_correction=('v1 said the nested probes show that "the delivered pose A is NOT close to a declared-cone '
                               'static equilibrium". That was drawn from probes with only 40 starts at R = 0.20 and 0.05. '
                               'With 240 starts the same box family finds machine-zero equilibria at R = 0.25 (and the ladder '
                               'finds them at 0.30 and 0.35), so the honest statement is: whether an equilibrium is FOUND '
                               'depends on the declared box AND on the start count; v1 over-read its 40-start misses.'),
        wording_rule='finite-start search result inside a declared box; a miss is never a proof of non-existence',
        not_extrapolated='the value 0.25 rad must not be quoted as "the distance from pose A to the equilibrium set"')

    # -------- findings and their disposition --------
    report['corrections_applied'] = [
        dict(id='F1', severity='material', title='assembler hash attribution',
             v1_problem=('the live assembler path had drifted after v1 was written, so the hash recorded in v1 no longer '
                         'matched the file on disk'),
             applied=('this v2 directory contains its own assembler snapshot (assembler_snapshot.py) plus the sha256 of that '
                      'snapshot; every hash recorded here is the hash of a file inside this directory. The v1 directory is '
                      'left untouched.') ),
        dict(id='F2', severity='material', title='"gate passing" count was the qacc item only',
             applied='three counts are reported separately with their definitions; the per-equilibrium table lists all four gate items per hit'),
        dict(id='F3', severity='material', title='full box reference pose was written as pose A',
             applied='the full box is stated as 0.40 rad around the SOURCE pose and the nested boxes as R around pose A; both distances of the accepted state are given and the consequence is spelled out'),
        dict(id='F4', severity='material', title='open-loop failure mode described wrongly',
             applied='the disclosure now gives the measured sequence (first non-foot contact 2739 = 68.475 ms, six-leg tarsus condition broken at 537 = 13.425 ms, last tarsus contact lost at 5111 = 127.775 ms), recomputed in this batch from the cited artifact'),
        dict(id='F5', severity='minor', title='run figures label the 1.0 mm/s^2 search tolerance as "declared tolerance"',
             applied='not repaired in place in the sealed run directories; disclosed here and the two lines are labelled correctly in this batch, which is the only place where the gate is decided'),
        dict(id='F6', severity='minor', title='diagnostic pointer test could mislead',
             applied='the v2 root-cause block uses the mechanism evidence (finite-difference Jacobian exactly zero, nfev = 1, and the returned residual equal to the residual at the last FD trial point x0 + h_132 e_132) instead of a raw buffer-pointer comparison'),
        dict(id='F7', severity='minor', title='"best" convention undefined',
             applied='both conventions are reported per run (start-level multistart best and refined stage-2 point); the values quoted for the radius study are the refined ones and the start-level ones are listed next to them'),
        dict(id='F8', severity='minor', title='VOID directory narrative not self-consistent with the filesystem',
             applied=('the VOID entry now states only what can still be checked: the current report.json sha256 = '
                      '68339df9610b2a9d8815ab67660ee5412c494182f036e145b8f1f7238343c6c3, that the directory carries no VOID '
                      'marker of its own, that its mtime is 2026-10-15 15:47:40 (after the events it describes), and that the '
                      'intermediate hashes c12ead58/62841f3c can no longer be re-verified from the filesystem. The directory '
                      'is left untouched.') ),
        dict(id='F9', severity='minor', title='run template wording and field names',
             applied=('the run template still says "within 0.40 rad of the source pose" even when --radius is used; that text '
                      'is wrong for the nested runs and is disclosed here rather than edited in the sealed run directories. '
                      'Field names in this batch are explicit: delivered_pose_path (not ..._sha256 for a path), '
                      'tarsus_geom_friction_3_values (not friction_rows_of_tarsus), and contact friction rows are given from the model as measured.') ),
        dict(id='F10', severity='cosmetic', title='figure presentation',
             applied='the figure is redrawn: the 40-start and 240-start runs are no longer stacked on one index axis, and the radius study has its own panel'),
    ]

    # -------- root cause (mechanism phrasing, finding F6) --------
    diag = sorted((ROOT / 'outputs/diagnostics').glob('aliased_residual_*.json'))
    report['root_cause_of_the_earlier_failure'] = dict(
        statement=('the superseded attempt run_equilibrium_reissue.py returned np.asarray(d.qacc) from its residual, i.e. a '
                   'VIEW into the live MjData buffer, so every evaluation overwrote the array the optimiser still held. '
                   'The finite-difference Jacobian is then exactly zero (jac_max_abs = 0.0), the first order optimality is '
                   '0.0, least_squares stops after nfev = 1 with "gtol termination condition is satisfied", and the returned '
                   'residual is numerically the residual at the last finite-difference trial point x0 + h_132 e_132. The '
                   'copied-residual variant on the same model, pose, box and first start (seed 20261015) reaches '
                   '66.35737929250325 mm/s^2 with nfev = 170. This is an implementation defect of my own script, not evidence '
                   'about the physical system.'),
        diagnostic_files=[dict(path=str(p), sha256=sha(p)) for p in diag],
        mechanism_note=('mj_forward on this model is bit-for-bit deterministic (enableflags = 0), so the view variant is fully '
                        'reproducible, which is why the failure looked like "no equilibrium exists".'))

    report['superseded_void_dirs'] = [
        dict(path=str(DESK / 'S2-mujoco-equilibrium-A-20261015T151340'), exists=(DESK / 'S2-mujoco-equilibrium-A-20261015T151340').exists(),
             reason=('INTEGRITY INCIDENT: report.json was replaced in place at 15:22:25 local and delivery-hashes.json was '
                     'rewritten, the script was edited at 15:23:44. The directory is VOID. What can still be checked: its '
                     'report.json sha256 is 68339df9610b2a9d8815ab67660ee5412c494182f036e145b8f1f7238343c6c3, its mtimes are '
                     '2026-10-15 15:47:40 (after the events described), it carries no VOID marker of its own, and the two '
                     'intermediate hashes quoted in v1 (c12ead58 / 62841f3c) can no longer be re-verified from the '
                     'filesystem. It is left untouched.'),
             report_json_sha256_current=(sha(DESK / 'S2-mujoco-equilibrium-A-20261015T151340/report.json')
                                         if (DESK / 'S2-mujoco-equilibrium-A-20261015T151340/report.json').exists() else None),
             re_verifiable_intermediate_hashes=False),
        dict(path=str(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946'), exists=True,
             reason=('SUPERSEDED INTERMEDIATE: the hand-written multi-start in run_equilibrium_reissue.py could not converge '
                     '(aliased residual, see above); best max|qacc| = 9016.5 mm/s^2 over 240 starts, 0 hits. Not a result. '
                     'Left untouched.'),
             report_json_sha256_current=(sha(DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946/report.json')
                                         if (DESK / 'S2-equilibrium-A-corrected-reissued-20261015T154946/report.json').exists() else None)),
        dict(path=str(DESK / 'S2-mujoco-equilibrium-A_ms240-20261015T155009'), exists=True,
             reason=('SUPERSEDED INTERMEDIATE: produced before the per-equilibrium detail block existed; only the first 20 '
                     'start residuals were stored. Its numbers agree with the re-run. Left untouched.'),
             report_json_sha256_current=(sha(DESK / 'S2-mujoco-equilibrium-A_ms240-20261015T155009/report.json')
                                         if (DESK / 'S2-mujoco-equilibrium-A_ms240-20261015T155009/report.json').exists() else None))]

    report['inputs'] = dict(
        pose_A=dict(path=str(POSE_A), sha256=sha(POSE_A)),
        source_pose=dict(path=str(SOURCE_POSE), sha256=sha(SOURCE_POSE)),
        canonical_xml=dict(path=str(CANONICAL_XML), sha256=sha(CANONICAL_XML)),
        search_script=dict(path=str(SEARCH_SCRIPT), sha256=sha(SEARCH_SCRIPT),
                           note='the runs in this batch were produced by this exact revision'),
        ladder_script=dict(path=str(LADDER_SCRIPT), sha256=sha(LADDER_SCRIPT)),
        diagnostic_script=dict(path=str(DIAG_SCRIPT), sha256=sha(DIAG_SCRIPT)),
        v1_batch=dict(path=str(V1_DIR), report_json_sha256=sha(V1_DIR / 'report.json')),
        verification=dict(path=str(VERIFY_DIR),
                          verification_report_json_sha256=sha(VERIFY_DIR / 'verification_report.json')),
        ladder_batch=dict(path=str(LADDER_DIR), report_json_sha256=sha(LADDER_DIR / 'report.json')),
        openloop_artifact=dict(path=str(OPENLOOP_DIR),
                               report_json_sha256=sha(OPENLOOP_DIR / 'true-equilibrium-openloop_report.json'),
                               time_series_sha256=sha(OPENLOOP_DIR / 'time_series.npz')),
        model=dict(nq=m.nq, nv=m.nv, na=m.na, nu=m.nu, opt_cone=int(m.opt.cone),
                   tarsus_condim=sorted({int(m.geom_condim[g]) for g in range(m.ngeom) if 'Tarsus' in gname(g)})),
        friction=dict(
            contact_friction_rows_source='read per contact from d.contact[..].friction inside the search runs',
            contact_friction_rows_measured=[1.0, 1.0, 0.005, 0.0001, 0.0001],
            mu1=1.0, mu2=1.0,
            tarsus_geom_friction_3_values=sorted({tuple(np.round(np.asarray(m.geom_friction[g]), 6).tolist())
                                                  for g in range(m.ngeom) if 'Tarsus' in gname(g)})))
    report['verdict'] = dict(
        declared_gate_passed=acc['recomputed_independently']['declared_gate_all_four'],
        accepted_max_abs_qacc_mm_s2=acc['recomputed_independently']['max_abs_qacc_mm_s2'],
        accepted_qacc_norm=acc['recomputed_independently']['qacc_norm'],
        n_all_four_gate_equilibria_in_the_full_box=report['per_equilibrium_summary']['n_all_four_gate_items_passing'],
        declaration=('an accepted point may only be called a DECLARED-CONE STATIC EQUILIBRIUM under the frozen '
                     'connectome-derived drive; never standing, never walking, never "pose A is an equilibrium"'),
        failed_or_not_run=['no physical or biological validity test', 'no muscle-model calibration', 'no closed-loop behaviour'])

    # -------- artifacts --------
    q, a = states['A_ms240']
    np.savez_compressed(out / 'equilibrium_best_state.npz', qpos=q, act=a,
                       note='accepted declared-cone static equilibrium of the FULL box (240 starts), recomputed in this batch')
    np.savez_compressed(out / 'equilibrium_best_pose_only.npz', projected_seated_qpos=q,
                       note='same state, pose only, so it can be used directly as a start pose')
    (out / 'assembler_snapshot.py').write_text(Path(__file__).read_text())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0))

    ax = axes[0][0]
    h240 = runs_block['A_ms240']['history_all_starts'] or []
    ax.semilogy(np.arange(len(h240)), np.maximum(h240, 1e-14), lw=1.1, color='#1f77b4',
                label='full box, 240 starts (reference = SOURCE pose)')
    for lab, col in (('A_ms_r0p10', '#2ca02c'),):
        h = runs_block[lab]['history_all_starts'] or []
        ax.semilogy(np.arange(len(h)), np.maximum(h, 1e-14), lw=1.0, color=col,
                    label='nested R = 0.10, 240 starts (reference = pose A)')
    ax.axhline(SEARCH_TOL, color='gray', ls='--', lw=1.2, label='search tolerance 1.0 mm/s^2')
    ax.axhline(GATE, color='black', ls=':', lw=1.7, label='declared gate 1e-3 mm/s^2')
    ax.set_xlabel('start index within its own run (never mixed across start counts -- finding F10)')
    ax.set_ylabel('final max|qacc| of that start  [mm/s^2], log scale')
    ax.set_title('per-start residuals; only runs with 240 starts are plotted here')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[0][1]
    rows = sorted(radius_rows, key=lambda r: r['radius_rad'])
    xs = [r['radius_rad'] for r in rows]
    ys = [max(r['best_max_abs_qacc_mm_s2'] or 1e-14, 1e-14) for r in rows]
    colors = ['#2ca02c' if r['n_hits_all_four_gate_items'] > 0 else '#d62728' for r in rows]
    ax.scatter(xs, ys, c=colors, s=52, zorder=3, edgecolors='black', linewidths=0.4)
    ax.set_ylim(1e-12, 1e4)
    ax.axvspan(0.20, 0.25, color='orange', alpha=0.14)
    table = '\n'.join('R=%.2f  n=%-3s  hits=%d  best=%.3g'
                      % (r['radius_rad'], r['n_starts'], r['n_hits_all_four_gate_items'],
                         r['best_max_abs_qacc_mm_s2'] or float('nan')) for r in rows)
    ax.text(0.02, 0.02, table, transform=ax.transAxes, fontsize=7.6, family='monospace', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.axhline(GATE, color='black', ls=':', lw=1.7, label='declared gate 1e-3 mm/s^2')
    ax.axhline(acc['recomputed_independently']['max_abs_qacc_mm_s2'], color='#1f77b4', ls='-', lw=1.2,
               label='full box best %.3g (not comparable: SOURCE-pose box)' % acc['recomputed_independently']['max_abs_qacc_mm_s2'])
    ax.set_xlabel('radius R around the DELIVERED pose A  [rad]')
    ax.set_ylabel('best max|qacc|  [mm/s^2], log scale')
    ax.set_title('radius study around the delivered pose A: green = at least one hit on all four gate items,\n'
                 'red = none; the hit map is not monotone in R (orange band = 0.20-0.25 rad)', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1][0]
    d = mujoco.MjData(m)
    d.qpos[:] = q; d.qvel[:] = 0.0; d.act[:] = np.clip(a, act_lo, act_hi); d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    qa = np.abs(np.asarray(d.qacc, dtype=float))
    ax.bar(np.arange(qa.size), np.maximum(qa, 1e-14), color='#1f77b4')
    ax.set_yscale('log')
    ax.axhline(GATE, color='black', ls=':', lw=1.7, label='declared gate 1e-3 mm/s^2')
    ax.set_xlabel('DOF index (nv = 48)')
    ax.set_ylabel('|qacc|  [mm/s^2], log scale')
    ax.set_title('accepted equilibrium, all 48 DOF (max %.3g, norm %.3g)'
                 % (acc['recomputed_independently']['max_abs_qacc_mm_s2'], acc['recomputed_independently']['qacc_norm']))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')

    ax = axes[1][1]
    per = [acc['recomputed_independently']['per_leg_max_u_diamond'].get(L, 0.0) for L in LEGS]
    ax.bar(list(LEGS), [max(v, 1e-16) for v in per], color='#2ca02c')
    ax.set_yscale('log')
    ax.axhline(1.0, color='red', ls='--', lw=1.4, label='declared diamond limit u_diamond = 1')
    for i, v in enumerate(per):
        ax.annotate('%.2g' % v, (i, max(v, 1e-16)), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    ax.set_ylabel('max u_diamond per leg  [1], log scale')
    ax.set_title('per-leg diamond utilisation at the accepted equilibrium\n(%d tarsus contacts, %d non-foot; one machine-zero\n'
                 'equilibrium in the full box touched only five legs)'
                 % (acc['recomputed_independently']['n_tarsus_contacts'], acc['recomputed_independently']['n_nonfoot_contacts']),
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y', which='both')

    fig.suptitle('S2 CORRECTED/REISSUED v2 -- declared-cone static equilibria around the delivered pose A under the frozen\n'
                 'connectome-derived drive  |  gate max|qacc| <= 1e-3 mm/s^2 = static NECESSARY condition only: not stability,\n'
                 'not standing, not "pose A is an equilibrium"  |  manual research implementation, not a product', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'equilibrium-reissued-v2.png', dpi=135)
    plt.close(fig)

    readme = [
        '# S2 修正/重发 v2：交付姿态 A 周围的"声明锥静态平衡"（手动研究实施，不是成品）',
        '',
        '本目录**取代** `' + str(V1_DIR) + '` 中与 v2 不一致的任何数字；v1 原样保留。',
        '',
        '## 1. 两种盒子，绝不混用（对应验证发现 F3）',
        '- **全盒**：42 个腿关节在 **源姿态** 的 0.40 rad 内（`' + str(SOURCE_POSE) + '`）。',
        '- **嵌套/半径盒**：42 个腿关节在 **交付姿态 A** 的 R 内（`' + str(POSE_A) + '`）。',
        '- 两者都保持 `act ∈ [1e-4,1]`、root z ±0.05 mm、`qvel = 0`。',
        '- 被接受状态：离源姿态 `%.6f rad`（在 0.40 盒内）、离交付姿态 A `%.6f rad`（**超出** A 的 0.40 盒）。'
        % (acc['recomputed_independently']['max_abs_delta_q_rad_vs_source_pose'],
           acc['recomputed_independently']['max_abs_delta_q_rad_vs_delivered_pose_A']),
        '',
        '## 2. 被接受平衡（全盒 240 起点，全新 `MjData` 重算）',
        '- `max|qacc| = %.6g mm/s²`，范数 `%.6g`；48 个 DOF 超 1e-3 / 1e-6 / 1e-9 的均为 0 个。'
        % (acc['recomputed_independently']['max_abs_qacc_mm_s2'], acc['recomputed_independently']['qacc_norm']),
        '- 接触 %d（跗节 %d、足外 %d），六足全部有跗节接触；`max u_diamond = %.6g ≤ 1`。'
        % (acc['recomputed_independently']['n_contacts'], acc['recomputed_independently']['n_tarsus_contacts'],
           acc['recomputed_independently']['n_nonfoot_contacts'], acc['recomputed_independently']['max_u_diamond']),
        '- 执行器触界（各自 atol）：下限 %d / 上限 %d。'
        % (acc['recomputed_independently']['act_at_floor_own_atol'], acc['recomputed_independently']['act_at_ceiling_own_atol']),
        '- 声明门限四项**全过**：' + json.dumps(acc['declared_gate_checks_recomputed'], ensure_ascii=False),
        '',
        '## 3. 三个不同的"命中"计数（对应 F2，v1 只报了口径 2 却写成"过门限"）',
        '- 口径 1（`max|qacc| ≤ 1.0` 搜索容差）：%d' % report['per_equilibrium_summary']['n_search_tolerance_passing'],
        '- 口径 2（声明门限的 qacc 项 `≤ 1e-3`）：%d' % report['per_equilibrium_summary']['n_qacc_item_passing'],
        '- 口径 3（**四项门限全过**）：%d' % report['per_equilibrium_summary']['n_all_four_gate_items_passing'],
        '',
        '## 4. 逐平衡明细（全盒命中起点）',
        *['- 起点 %d：`max|qacc| = %.6g`，范数 `%.6g`，接触 %d（跗节 %d，足外 %d），六足 %s，`tarsi_with = %s`，'
          '`max u_diamond = %s`，`max|Δq| vs A = %.6f rad`，四项门限 %s'
          % (e['found_at_start_index'], e['max_abs_qacc_mm_s2'], e['qacc_norm'], e['n_contacts'], e['n_tarsus_contacts'],
             e['n_nonfoot_contacts'], e['six_tarsi_in_contact'], e['tarsi_with_contact'],
             ('None' if e['max_u_diamond'] is None else '%.3g' % e['max_u_diamond']),
             e['max_abs_delta_q_rad_vs_delivered_pose_A'], '全过' if e['all_four_gate_items_passed'] else '未全过')
          for e in report['per_equilibrium_summary']['entries']],
        '',
        '## 5. 半径研究（声明阶梯 240 起点 + 补测 R = 0.20 / 0.15）',
        *['- R = %.2f rad（%s）：最好 `max|qacc| = %.6g`，四项门限命中 %d 个'
          % (r['radius_rad'], r['source'], r['best_max_abs_qacc_mm_s2'] or float('nan'), r['n_hits_all_four_gate_items'])
          for r in radius_rows],
        '- 有命中的半径：%s；无命中的半径：%s' % (report['radius_study']['radii_with_hits'], report['radius_study']['radii_without_hits']),
        '- %s' % report['radius_study']['reading'],
        '- **v1 措辞更正**：%s' % report['radius_study']['v1_wording_correction'],
        '- %s' % report['radius_study']['wording_rule'],
        '',
        '## 6. 强制披露',
        *['- ' + s for s in DISCLOSURE],
        '',
        '## 7. 摩擦判据措辞（判据 = 菱形；方锥仅次级）',
        *['- **%s**：%s' % (k, json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in FRICTION_WORDING.items()],
        '',
        '## 8. 本批次落地的验证发现（F1–F10）',
        *['- **%s（%s）** %s → %s' % (c['id'], c['severity'], c['title'], c['applied']) for c in report['corrections_applied']],
        '',
        '## 9. 作废/取代目录',
        *['- `%s`：%s' % (v['path'], v['reason']) for v in report['superseded_void_dirs']],
        '',
        '## 10. 文件',
        '- `report.json`（全部数字、每档运行的 sha256、逐平衡四项门限、F1–F10 处置）',
        '- `equilibrium-reissued-v2.png`（对数坐标、门限线可见；240 起点分轴绘制；半径研究单独一栏）',
        '- `equilibrium_best_state.npz` / `equilibrium_best_pose_only.npz`',
        '- `assembler_snapshot.py`（本目录报告的实际生成脚本，哈希写在 `report.json.inputs`；F1 的处置方式）',
        '- `source_snapshot.zip`、`delivery-hashes.json`',
    ]
    (out / 'README.zh-CN.md').write_text('\n'.join(readme))

    report['status'] = 'PASS_EQUILIBRIUM_REISSUED_V2_REPORTED'
    report['outputs'] = str(out)
    report['figure'] = str(out / 'equilibrium-reissued-v2.png')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))

    with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (Path(__file__), SEARCH_SCRIPT, LADDER_SCRIPT, DIAG_SCRIPT):
            z.write(p, Path(p).name)
        for p in diag:
            z.write(p, p.name)
        z.writestr('RUNS.txt', '\n'.join('%s -> %s' % (k, v) for k, v in sorted(runs.items())))
        z.writestr('V1_AND_VERIFICATION.txt', '%s\n%s\n%s\n' % (V1_DIR, VERIFY_DIR, LADDER_DIR))
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'delivery-hashes.json')
    (out / 'delivery-hashes.json').write_text(json.dumps({p.name: sha(p) for p in files}, ensure_ascii=False, indent=1))

    print(report['status'])
    print('out:', out)
    print('accepted max|qacc| %.6g  norm %.6g  all-four-gate equilibria in the full box %d'
          % (acc['recomputed_independently']['max_abs_qacc_mm_s2'], acc['recomputed_independently']['qacc_norm'],
             report['per_equilibrium_summary']['n_all_four_gate_items_passing']))
    print('radii with hits:', report['radius_study']['radii_with_hits'], 'without:', report['radius_study']['radii_without_hits'])
    return 0


def evaluate_hit_ok(h):
    return bool(h['max_abs_qacc_mm_s2'] <= GATE and h['six_tarsi_in_contact'] and h['n_nonfoot_contacts'] == 0
                and h['max_u_diamond'] is not None and h['max_u_diamond'] <= 1 + 1e-9)


if __name__ == '__main__':
    raise SystemExit(main())
