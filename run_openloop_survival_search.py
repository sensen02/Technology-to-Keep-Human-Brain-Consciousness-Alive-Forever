"""Declared search for a seated pose that SURVIVES the declared open-loop drive - criteria fixed in advance.

DECLARED GOAL (this round): move from "a pose whose static LP is feasible inside the declared cone" to "a pose that the MuJoCo
dynamics can hold under the declared frozen connectome command archive". The acceptance test is fixed here, before any search:

  1. DRIVE: the declared connectome-derived archive 100 x 90 frames at 5 ms cadence
     (/run/media/sensen/Data2/cell_wound_prototype/outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz), applied as
     ctrl = archive[min(k//200, 99)] at dt = 25 us (the only numerically valid branch), window >= 0.5 s (20000 steps).
  2. DISCIPLINE: qpos, qvel and act are written exactly ONCE before the first step; only ctrl is written inside the loop; no
     external CPG, no PD, no preprogrammed action, no per-step pose correction.
  3. SURVIVAL (declared): a pose survives iff, over the whole 0.5 s, there is NO non-foot (body-floor) contact.
  4. REPORTED FOR EVERY CANDIDATE, whatever the outcome: (a) its static s under the same declared pyramidal axis-ray diamond LP,
     (b) the T0 contact table (geom pair, normal force) if T0 exists, (c) the root z trajectory summary, (d) max |qvel|,
     (e) whether all six tarsi keep contact (and for what fraction of steps).
  5. CONCLUSION STYLE: always two separate sentences - "this pose is / is not statically feasible inside the declared cone
     (s = ...)" and "under this declared open-loop drive it is / is not maintainable for 0.5 s (first non-foot contact at ...)".
     The two sentences are never merged.
  6. IF a pose survives: do NOT call it a success. Report its s, its difference from A/B/C, and the provenance of its IK
     (seed file, residual weighting, solver) - because s is a function of seed and weighting, not only of the declared parameters.

Declared search grid (fixed before running): root z in {1.98, 2.00, 2.02} mm x footprint spread in {0.90, 0.95, 1.00} x coxa yaw
in {-8, -10, -12} deg, IK from the declared seed pose with the declared residual weighting (10.0, 0.5), contract = touch band
[-8, -2] um, all 41 non-foot geoms >= 0.40 mm, hinge >= 5e-3 rad inside limits, every joint <= 0.40 rad from the source pose.
Absolute force values and physical units remain NOT_ESTABLISHED; this is manual research, not a product, and it makes no
standing or walking claim.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares, linprog

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
DESK = Path('/home/sensen/Desktop/engine修复验收')
ARCHIVE = ROOT / 'outputs/experiment_platform/20261015_135150_a2fff3f0/traces.npz'
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
SEED_POSE = DESK / 'S2-seated-feasibility-search-20261015T122857/best_candidate_pose.npz'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
GROUPS = {'front': ('LF', 'RF'), 'mid': ('LM', 'RM'), 'hind': ('LH', 'RH')}
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
TOUCH_LOW, TOUCH_HIGH = -8e-3, -2e-3
H_DECLARED = 0.40
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
DIST_CAP = 4.0
DIAMOND = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
Z_GRID = (1.98, 2.00, 2.02)
SPREAD_GRID = (0.90, 0.95, 1.00)
YAW_GRID = (-8.0, -10.0, -12.0)
OFFSETS = dict(tp={'front': -10.0, 'mid': -5.0, 'hind': -10.0},
               cp={'front': 0.0, 'mid': -10.0, 'hind': -10.0},
               cr={'front': 0.0, 'mid': 0.0, 'hind': 0.0})
W_TOUCH, W_XY = 10.0, 0.5
DT = 2.5e-5
CMD_DT = 0.005
WINDOW = 0.5
S_CAP = 500.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cone_cols(contacts, dirs):
    cols = []
    for c in contacts:
        frame, mu, jac_t = c['frame'], c['mu'], c['J'].T
        normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
        for p, q in dirs:
            cols.append(jac_t @ (normal + float(p) * t1 + float(q) * t2))
    return np.asarray(cols).T


def threshold(muscle_cols, cone, b, lo, hi):
    def ok(sc):
        A = np.hstack([muscle_cols * sc, cone])
        scale = np.maximum(1.0, np.maximum(np.abs(b), np.max(np.abs(A), axis=1)))
        res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=b / scale,
                      bounds=list(zip(lo, hi)) + [(0, None)] * cone.shape[1], method='highs',
                      options=dict(dual_feasibility_tolerance=1e-10, primal_feasibility_tolerance=1e-10))
        return res.status == 0
    f1 = ok(1.0)
    low, high = (0.5, 1.0) if f1 else (1.0, 2.0)
    while high <= S_CAP and not ok(high):
        high *= 1.25
    if high > S_CAP:
        return dict(feasible_at_s_1=f1, threshold_midpoint=None, cap_reached=True, infeasible_up_to_s=high / 1.25)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high, threshold_midpoint=0.5 * (low + high), cap_reached=False)


def main():
    out = DESK / ('S2-openloop-survival-search-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  declared=dict(goal='find a seated pose that survives the declared open-loop drive, with the acceptance test fixed in advance',
                                drive=str(ARCHIVE), drive_sha256=sha(ARCHIVE), dt_s=DT, window_s=WINDOW,
                                indexing='ctrl = archive[min(k // 200, 99)]',
                                discipline='qpos/qvel/act written once; only ctrl inside the loop; no CPG, no PD, no preprogrammed action, no per-step pose correction',
                                survival_criterion='no non-foot (body-floor) contact at any step of the 0.5 s window',
                                grid=dict(root_z_mm=list(Z_GRID), footprint_spread=list(SPREAD_GRID), coxa_yaw_deg=list(YAW_GRID)),
                                ik=dict(seed=str(SEED_POSE), seed_sha256=sha(SEED_POSE), weights=[W_TOUCH, W_XY], solver='scipy least_squares (trf)'),
                                contract=dict(touch_band_um=[1000 * TOUCH_LOW, 1000 * TOUCH_HIGH], nonfoot_clearance_mm_min=H_DECLARED,
                                              joint_margin_rad=JOINT_MARGIN, max_deviation_rad=MAX_DEVIATION),
                                cone='declared pyramidal (axis-ray diamond)',
                                two_sentence_rule='static feasibility in the declared cone and open-loop maintainability are reported as two separate sentences'),
                  input_hashes={str(p): sha(p) for p in (Path(__file__), ARCHIVE, SEED_POSE, SOURCE_POSE) if Path(p).exists()},
                  absolute_force_values='NOT_ESTABLISHED')
    try:
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        m.opt.timestep = DT
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        other = [g for g in range(m.ngeom) if g != 0 and 'Tarsus' not in gname(g)]
        foot_by_leg = {L: [g for g in foot if gname(g).startswith(L)] for L in LEGS}
        leg_joints, joint_of = {}, {}
        for L in LEGS:
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + L)]
            leg_joints[L] = [next(j for j in js if jname(j).endswith(s)) for s in JOINT_SUFFIXES]
            for s in JOINT_SUFFIXES:
                joint_of[(L, s)] = next(j for j in js if jname(j).endswith(s))
        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        seed_pose = np.load(SEED_POSE)['projected_seated_qpos'].copy()
        controls = np.load(ARCHIVE)['normal_controls'].copy()
        hold = int(round(CMD_DT / DT))
        n = int(round(WINDOW / DT))

        def gdist(geom, points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, 0, geom, DIST_CAP, ft))
            return (r, ft) if points else r

        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_xy = {L: min((gdist(g, True) for g in foot_by_leg[L]), key=lambda t: t[0])[1][3:5] for L in LEGS}

        def build_pose(z0, spread, yaw):
            q = source.copy()
            q[2] = z0
            for L in LEGS:
                q[m.jnt_qposadr[joint_of[(L, 'Coxa_yaw')]]] = source[m.jnt_qposadr[joint_of[(L, 'Coxa_yaw')]]] + math.radians(yaw)
            for key, suf in (('tp', 'Trochanter_pitch'), ('cp', 'Coxa_pitch'), ('cr', 'Coxa_roll')):
                for grp, legs in GROUPS.items():
                    for L in legs:
                        j = joint_of[(L, suf)]
                        adr = m.jnt_qposadr[j]
                        q[adr] = float(np.clip(source[adr] + math.radians(OFFSETS[key][grp]),
                                               m.jnt_range[j, 0] + JOINT_MARGIN, m.jnt_range[j, 1] - JOINT_MARGIN))
            for L in LEGS:
                idx = [m.jnt_qposadr[j] for j in leg_joints[L]]
                lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION) for k, j in enumerate(leg_joints[L])])
                hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION) for k, j in enumerate(leg_joints[L])])
                x0 = np.clip(seed_pose[idx], lo, hi)
                target = spread * source_xy[L]

                def residual(x, L=L, q=q, idx=idx):
                    qq = q.copy()
                    qq[idx] = x
                    d.qpos[:] = qq
                    mujoco.mj_kinematics(m, d)
                    best = min((gdist(g, True) for g in foot_by_leg[L]), key=lambda t: t[0])
                    return np.array([W_TOUCH * (best[0] - 0.5 * (TOUCH_LOW + TOUCH_HIGH)),
                                     W_XY * (best[1][3] - target[0]), W_XY * (best[1][4] - target[1])])
                sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=3000)
                q[idx] = np.clip(sol.x, lo, hi)
            return q

        def evaluate_contract_and_s(q):
            d.qpos[:] = q
            d.qvel[:] = 0.0
            d.act[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            leg_dist = {L: float(min(gdist(g) for g in foot_by_leg[L])) for L in LEGS}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0], m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[L]]] -
                                           source[[m.jnt_qposadr[j] for j in leg_joints[L]]])) for L in LEGS))
            contacts, feet = [], set()
            for c in d.contact:
                if c.efc_address < 0:
                    continue
                jac = np.zeros((3, m.nv))
                for geom, sign in ((c.geom2, 1), (c.geom1, -1)):
                    body = int(m.geom_bodyid[geom])
                    jp, jr = np.zeros_like(jac), np.zeros_like(jac)
                    mujoco.mj_jac(m, d, jp, jr, c.pos, body)
                    jac += sign * jp
                names = [gname(int(c.geom1)), gname(int(c.geom2))]
                for nm in names:
                    if 'Tarsus' in nm:
                        feet.add(nm[:2])
                contacts.append(dict(frame=np.asarray(c.frame).reshape(3, 3), mu=np.asarray(c.friction)[:2], J=jac, names=names))
            nonfoot = sum(1 for c in contacts if not any('Tarsus' in nm for nm in c['names']))
            base = d.qfrc_actuator.copy()
            tgt = d.qfrc_bias - d.qfrc_passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                R[:, i] = d.qfrc_actuator - base
            contract = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in leg_dist.values()) and clearance >= H_DECLARED
                        and margin >= JOINT_MARGIN and deviation <= MAX_DEVIATION + 1e-9 and nonfoot == 0 and len(contacts) > 0)
            return dict(contract=bool(contract), leg_dist=leg_dist, clearance=clearance, deviation=deviation,
                        nonfoot=nonfoot, contacts=len(contacts), feet=sorted(feet), target=tgt, R=R, contactdict=contacts)

        def run_openloop(q):
            d2 = mujoco.MjData(m)
            d2.qpos[:] = q
            d2.qvel[:] = 0.0
            d2.act[:] = 0.0
            d2.ctrl[:] = controls[0]
            mujoco.mj_forward(m, d2)
            t0 = None
            rz = np.zeros(n)
            qv = np.zeros(n)
            leg_cnt = np.zeros((n, len(LEGS)))
            nonfoot_names = set()
            for k in range(n):
                d2.ctrl[:] = controls[min(k // hold, controls.shape[0] - 1)]
                mujoco.mj_step(m, d2)
                rz[k] = float(d2.qpos[2])
                qv[k] = float(np.max(np.abs(d2.qvel)))
                nf = 0
                for ci, c in enumerate(d2.contact):
                    if c.efc_address < 0:
                        continue
                    n1, n2 = gname(int(c.geom1)), gname(int(c.geom2))
                    if 'Tarsus' in (n1 + n2):
                        for li, L in enumerate(LEGS):
                            if (n1.startswith(L) and 'Tarsus' in n1) or (n2.startswith(L) and 'Tarsus' in n2):
                                leg_cnt[k, li] += 1
                    else:
                        nf += 1
                        if len(nonfoot_names) < 40 and k % 200 == 0:
                            nonfoot_names.add(' | '.join(sorted((n1, n2))))
                if nf > 0 and t0 is None:
                    t0 = k
            six = (leg_cnt > 0).sum(axis=1)
            return dict(survived_0p5s=bool(t0 is None), T0_step=t0,
                        T0_ms=(None if t0 is None else t0 * DT * 1000),
                        root_z_start=float(rz[0]), root_z_min=float(rz.min()), root_z_final=float(rz[-1]),
                        max_abs_qvel=float(qv.max()), six_leg_steps=int((six == 6).sum()),
                        six_leg_frac=float((six == 6).mean()), min_legs_in_contact=int(six.min()),
                        nonfoot_names=sorted(nonfoot_names))

        cands = []
        for z0 in Z_GRID:
            for spread in SPREAD_GRID:
                for yaw in YAW_GRID:
                    name = 'z%.2f_sp%.2f_yaw%+.0f' % (z0, spread, yaw)
                    print('candidate', name, flush=True)
                    q = build_pose(z0, spread, yaw)
                    ev = evaluate_contract_and_s(q)
                    entry = dict(name=name, root_z_mm=z0, footprint_spread=spread, coxa_yaw_deg=yaw,
                                 contract=ev['contract'], clearance_mm=ev['clearance'], deviation_rad=ev['deviation'],
                                 min_tarsus_mm=ev['leg_dist'], contacts=ev['contacts'], nonfoot=ev['nonfoot'],
                                 s=None, open_loop=None, conclusion=None)
                    if ev['contract']:
                        lo = m.actuator_ctrlrange[:, 0].copy()
                        hi = m.actuator_ctrlrange[:, 1].copy()
                        entry['s'] = threshold(ev['R'], cone_cols(ev['contactdict'], DIAMOND), ev['target'], lo, hi)
                        entry['open_loop'] = run_openloop(q)
                        np.savez_compressed(out / ('pose_%s.npz' % name), projected_seated_qpos=q)
                        s_val = entry['s'].get('threshold_midpoint')
                        ol = entry['open_loop']
                        entry['conclusion'] = dict(
                            static_sentence=('statically feasible inside the declared pyramidal axis-ray diamond cone with s = %.12g' % s_val
                                             if s_val is not None else
                                             'NOT statically feasible inside the declared cone up to s = %s' % entry['s'].get('infeasible_up_to_s')),
                            open_loop_sentence=('NOT maintainable under this declared open-loop drive: first non-foot contact at step %d = %.3f ms' % (ol['T0_step'], ol['T0_ms'])
                                                if not ol['survived_0p5s'] else
                                                'maintainable for the whole 0.5 s window under this declared open-loop drive (no non-foot contact)'),
                            merged=False)
                    cands.append(entry)
        report['candidates'] = cands
        surv = [c for c in cands if c['open_loop'] and c['open_loop']['survived_0p5s']]
        report['summary'] = dict(n_candidates=len(cands), n_contract_valid=sum(1 for c in cands if c['contract']),
                                 n_survived_0p5s=len(surv),
                                 best_by_T0=max((c for c in cands if c['open_loop'] and c['open_loop']['T0_step'] is not None),
                                                key=lambda c: c['open_loop']['T0_step'], default=None)['name']
                                 if any(c['open_loop'] and c['open_loop']['T0_step'] is not None for c in cands) else None,
                                 survivors=[c['name'] for c in surv])
        report['statement'] = ('manual research only: no standing, no walking, no completion claim; absolute force values and units remain '
                               'NOT_ESTABLISHED. If a survivor was found it must be reported with its s, its difference from A/B/C and the '
                               'provenance of its IK (seed file, residual weighting, solver), and it is still not a success claim')
        report['status'] = 'PASS_OPENLOOP_SURVIVAL_SEARCH_REPORTED'
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc()[-1500:])
    finally:
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print(json.dumps(report['summary'], indent=1))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
