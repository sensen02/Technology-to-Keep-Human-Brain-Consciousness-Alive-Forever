"""Declared-cone minimum-s search: minimise s under the DECLARED pyramidal (axis-ray diamond) cone.

OBJECTIVE (the only one): s = the smallest uniform multiplier on the muscle columns (equivalently on the
activation box) at which the ideal static relaxation becomes feasible, computed against the DECLARED
pyramidal cone = the non-negative hull of the axis rays n +- mu*t1, n +- mu*t2 (the four rows MuJoCo
emits; behavioural criterion: 45-degree sliding ratio 0.7071 = 1/sqrt(2), elliptic control 1.0000).
DECLARED CRITERION: s_best <= 1 means the pose is statically supportable inside this declared model
family; otherwise the shortfall in multiples and percent is reported. Relaxed cones (the diagonal rays
beyond the declared cone and beyond the Coulomb disk) and the unestablished absolute force scale are
NEVER used to rescue the result.

DECLARED SEARCH (coordinate descent, inherited parameters stated explicitly):
  inherited (from the previous rounds): root z = 2.02 mm, foot spread 0.95, uniform coxa-yaw -10 deg
  swept: front / mid / hind  trochanter-pitch offsets, then front / mid / hind coxa-pitch offsets,
         then front / mid / hind coxa-roll offsets, then a short root-z confirmation sweep
Each sweep uses the declared grid {-10, -5, 0, +5, +10} deg (z: {2.00, 2.02, 2.04} mm).

Pose search only: NOT neural control, NOT a preprogrammed action, NOT behaviour evidence, NOT standing.
Canonical XML, gains, friction, force scale and every historical hash are untouched.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares, linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
FLOOR_GEOM = 0
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
Z_BASE, SPREAD_BASE, YAW_BASE = 2.02, 0.95, -10.0
OFFSET_GRID = (-10.0, -5.0, 0.0, 5.0, 10.0)
Z_CONFIRM = (2.00, 2.02, 2.04)
S_CAP = 500.0
PREV = {'footprint_sweep': DESK / 'S2-footprint-dof-sweep-20261015T122249/report.json',
        'feasibility_search': DESK / 'S2-seated-feasibility-search-20261015T122857/report.json'}


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
    def ok(s):
        A = np.hstack([muscle_cols * s, cone])
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
        return dict(feasible_at_s_1=f1, threshold_midpoint=None, cap_reached=True,
                    infeasible_up_to_s=high / 1.25)
    for _ in range(26):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return dict(feasible_at_s_1=f1, infeasible_below=low, feasible_above=high,
                threshold_midpoint=0.5 * (low + high), cap_reached=False)


def main():
    out = DESK / ('S2-declared-cone-min-s-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  objective=('minimise s under the DECLARED pyramidal (axis-ray diamond) cone; s_best <= 1 means '
                             'statically supportable inside this declared family, otherwise the shortfall is '
                             'reported and never rescued by relaxed cones or the absolute force scale'),
                  declared_cone=dict(name='declared pyramidal (axis-ray diamond)',
                                     generators='non-negative hull of n+-mu*t1 and n+-mu*t2 (the four rows '
                                                'MuJoCo emits for a condim-3 pyramidal contact)',
                                     behavioural_criterion=('45-degree sliding ratio 0.7071 = 1/sqrt(2); elliptic '
                                                            'control 1.0000; the diagonal-ray block lies beyond '
                                                            'the declared cone and beyond the Coulomb disk')),
                  declared_search=dict(inherited=dict(root_z_mm=Z_BASE, footprint_scale=SPREAD_BASE,
                                                      coxa_yaw_deg=YAW_BASE),
                                       swept=['front/mid/hind trochanter-pitch offset',
                                              'front/mid/hind coxa-pitch offset',
                                              'front/mid/hind coxa-roll offset',
                                              'root z confirmation'],
                                       grid_deg=list(OFFSET_GRID), z_confirm_mm=list(Z_CONFIRM),
                                       contract=dict(touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                                                     nonfoot_clearance_mm_min=H_DECLARED,
                                                     joint_limit_margin_rad=JOINT_MARGIN,
                                                     max_joint_deviation_rad=MAX_DEVIATION)),
                  nature_of_this_step='DECLARED coordinate-descent pose search; NOT behaviour evidence',
                  declared_calls=dict(project_mj_steps=0),
                  absolute_force_values='NOT_ESTABLISHED')
    hashes = {str(p): sha(p) for p in (Path(__file__), SOURCE_POSE, PREV['footprint_sweep'],
                                       PREV['feasibility_search']) if Path(p).exists()}
    report['input_hashes'] = hashes
    try:
        report['previous_measured_numbers'] = {
            'footprint_dof_sweep': dict(
                source=str(PREV['footprint_sweep']), sha256=hashes.get(str(PREV['footprint_sweep'])),
                best=json.loads(PREV['footprint_sweep'].read_text())['conclusion']),
            'seated_feasibility_search': dict(
                source=str(PREV['feasibility_search']), sha256=hashes.get(str(PREV['feasibility_search'])),
                final=json.loads(PREV['feasibility_search'].read_text())['final_pose'])}
        import mujoco
        from engine import canonical_physics as cp
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        report['declared_calls']['project_model_loads'] = 1
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        other = [g for g in range(m.ngeom) if g != FLOOR_GEOM and 'Tarsus' not in gname(g)]
        leg_geoms = {p: [g for g in foot if gname(g).startswith(p)] for p in LEGS}
        leg_joints, joint_of = {}, {}
        for p in LEGS:
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + p)]
            leg_joints[p] = [next(j for j in js if jname(j).endswith(s)) for s in JOINT_SUFFIXES]
            for s in JOINT_SUFFIXES:
                joint_of[(p, s)] = next(j for j in js if jname(j).endswith(s))
        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()

        def gdist(geom, points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, geom, DIST_CAP, ft))
            return (r, ft) if points else r

        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_xy = {p: min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])[1][3:5] for p in LEGS}

        def build(params):
            q = source.copy()
            q[2] = params['z']
            for p in LEGS:
                q[m.jnt_qposadr[joint_of[(p, 'Coxa_yaw')]]] = source[m.jnt_qposadr[joint_of[(p, 'Coxa_yaw')]]] \
                    + math.radians(YAW_BASE)
            for group, legs in GROUPS.items():
                for p in legs:
                    for suffix, key in (('Trochanter_pitch', 'tp'), ('Coxa_pitch', 'cp'), ('Coxa_roll', 'cr')):
                        j = joint_of[(p, suffix)]
                        adr = m.jnt_qposadr[j]
                        q[adr] = float(np.clip(source[adr] + math.radians(params.get(key, {}).get(group, 0.0)),
                                               m.jnt_range[j, 0] + JOINT_MARGIN, m.jnt_range[j, 1] - JOINT_MARGIN))
            for p in LEGS:
                idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
                lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                x0 = np.clip(q[idx], lo, hi)
                target = params['spread'] * source_xy[p]

                def residual(x):
                    qq = q.copy()
                    qq[idx] = x
                    d.qpos[:] = qq
                    mujoco.mj_kinematics(m, d)
                    best = min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])
                    return np.concatenate([[10.0 * (best[0] - 0.5 * (TOUCH_LOW + TOUCH_HIGH))],
                                           [0.5 * (best[1][3] - target[0]), 0.5 * (best[1][4] - target[1])],
                                           0.05 * (x - x0)])
                sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                    max_nfev=3000)
                q[idx] = sol.x
            return q

        def evaluate(q):
            d.qpos[:] = q
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] = report['declared_calls'].get(
                'project_forward_calls', 0) + 1
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(gdist(g) for g in leg_geoms[p])) for p in LEGS}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]])) for p in LEGS))
            contacts = []
            for c in d.contact:
                if c.efc_address < 0:
                    continue
                jac = np.zeros((3, m.nv))
                for geom, sign in ((c.geom2, 1), (c.geom1, -1)):
                    body = int(m.geom_bodyid[geom])
                    jp, jr = np.zeros_like(jac), np.zeros_like(jac)
                    mujoco.mj_jac(m, d, jp, jr, c.pos, body)
                    jac += sign * jp
                contacts.append(dict(names=[gname(int(c.geom1)), gname(int(c.geom2))], J=jac,
                                     frame=np.asarray(c.frame).reshape(3, 3),
                                     mu=np.asarray(c.friction)[:2], dist=float(c.dist)))
            nonfoot = sum(1 for c in contacts if not any('Tarsus' in n for n in c['names']))
            feet = sorted({n[:2] for c in contacts for n in c['names'] if 'Tarsus' in n})
            base = d.qfrc_actuator.copy()
            target_b = d.qfrc_bias - d.qfrc_passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                report['declared_calls']['project_forward_calls'] += 1
                R[:, i] = d.qfrc_actuator - base
            contract = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in leg_dist.values())
                        and clearance >= H_DECLARED and margin >= JOINT_MARGIN
                        and deviation <= MAX_DEVIATION + 1e-9 and nonfoot == 0 and len(contacts) > 0)
            return dict(q=q, leg_dist=leg_dist, clearance=clearance, margin=margin, deviation=deviation,
                        contacts=len(contacts), nonfoot=nonfoot, feet=feet, target=target_b, R=R,
                        contract=bool(contract))

        params = dict(z=Z_BASE, spread=SPREAD_BASE, tp={}, cp={}, cr={})
        trace = []

        # --- the real scoring path (kept explicit for clarity)
        def score2(params, stage):
            q = build(params)
            d.qpos[:] = q
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] = report['declared_calls'].get(
                'project_forward_calls', 0) + 1
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(gdist(g) for g in leg_geoms[p])) for p in LEGS}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]])) for p in LEGS))
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
                for n in names:
                    if 'Tarsus' in n:
                        feet.add(n[:2])
                contacts.append(dict(frame=np.asarray(c.frame).reshape(3, 3),
                                     mu=np.asarray(c.friction)[:2], J=jac, names=names))
            nonfoot = sum(1 for c in contacts if not any('Tarsus' in n for n in c['names']))
            base = d.qfrc_actuator.copy()
            target_b = d.qfrc_bias - d.qfrc_passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                report['declared_calls']['project_forward_calls'] += 1
                R[:, i] = d.qfrc_actuator - base
            contract = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in leg_dist.values())
                        and clearance >= H_DECLARED and margin >= JOINT_MARGIN
                        and deviation <= MAX_DEVIATION + 1e-9 and nonfoot == 0 and len(contacts) > 0)
            entry = dict(params=json.loads(json.dumps(params)), contract=bool(contract), clearance=clearance,
                         margin=margin, deviation=deviation, contacts=len(contacts), nonfoot=nonfoot,
                         feet=sorted(feet), leg_dist=leg_dist, s=None)
            if contract:
                lo = m.actuator_ctrlrange[:, 0].copy()
                hi = m.actuator_ctrlrange[:, 1].copy()
                entry['s'] = threshold(R, cone_cols(contacts, DIAMOND), target_b, lo, hi)
            trace.append(dict(stage=stage, **entry))
            print('%-34s s=%s contract=%s clearance=%.4f feet=%s' % (
                stage, None if entry['s'] is None else entry['s'].get('threshold_midpoint'),
                entry['contract'], clearance, sorted(feet)), flush=True)
            return q, entry

        def best_of(entries):
            vals = [e for e in entries if e['s'] and e['s'].get('threshold_midpoint') is not None]
            return min(vals, key=lambda e: e['s']['threshold_midpoint']) if vals else None

        # --- coordinate descent
        base_q, base_e = score2(params, 'base (inherited z/spread/yaw)')
        best_e, best_q = base_e, base_q
        for key, suffix, label in (('tp', 'Trochanter_pitch', 'trochanter_pitch'),
                                   ('cp', 'Coxa_pitch', 'coxa_pitch'),
                                   ('cr', 'Coxa_roll', 'coxa_roll')):
            for group in GROUPS:
                trials = []
                for off in OFFSET_GRID:
                    p = json.loads(json.dumps(params))
                    p[key][group] = off
                    q, e = score2(p, '%s %s %+.0f deg' % (label, group, off))
                    trials.append((e, p))
                bestc = best_of([e for e, _ in trials])
                if bestc is not None:
                    params = [p for e, p in trials if e is bestc][0]
                    best_e, best_q = bestc, None
        for z in Z_CONFIRM:
            p = json.loads(json.dumps(params))
            p['z'] = z
            q, e = score2(p, 'z confirm %.2f mm' % z)
            if e['s'] and e['s'].get('threshold_midpoint') is not None:
                if best_e['s'] is None or e['s']['threshold_midpoint'] < best_e['s']['threshold_midpoint']:
                    best_e, best_q = e, q
        report['search_trace'] = trace
        report['best'] = best_e
        s_best = None if (best_e['s'] is None) else best_e['s'].get('threshold_midpoint')
        report['criterion_result'] = dict(
            s_best=s_best,
            statically_supportable_in_family=bool(s_best is not None and s_best <= 1.0),
            shortfall_multiple=(None if s_best is None else float(s_best)),
            shortfall_percent=(None if s_best is None else float(100 * (s_best - 1))),
            geometry=dict(min_nonfoot_clearance_mm=best_e['clearance'], contacts=best_e['contacts'],
                          nonfoot_contacts=best_e['nonfoot'], feet=best_e['feet'],
                          min_joint_limit_margin_rad=best_e['margin'],
                          max_joint_deviation_rad=best_e['deviation'],
                          per_leg_min_tarsus_distance_mm=best_e['leg_dist']),
            statement=('s_best <= 1: statically supportable inside the declared family (ideal static linear '
                       'relaxation only; not standing, not a MuJoCo equilibrium)'
                       if (s_best is not None and s_best <= 1.0) else
                       's_best > 1: NOT statically supportable inside the declared family; the shortfall is '
                       '%.4f times the declared muscle strength (%.2f percent) and is NOT rescued by relaxed '
                       'cones or by the unestablished absolute force scale'
                       % (s_best, 100 * (s_best - 1)) if s_best is not None else 'no contract-satisfying pose'))
        if best_q is not None:
            np.savez_compressed(out / 'best_candidate_pose.npz', projected_seated_qpos=best_q)
            report['best_candidate_npz'] = str(out / 'best_candidate_pose.npz')
            gate = subprocess.run([sys.executable, '-B', str(ROOT / 'run_s2_support_gate.py'),
                                   str(out / 'best_candidate_pose.npz')],
                                  capture_output=True, text=True, timeout=1800)
            report['gate_run'] = dict(returncode=gate.returncode,
                                      stdout_tail=gate.stdout.strip().splitlines()[-2:],
                                      stderr_tail=gate.stderr.strip().splitlines()[-2:])
        report['status'] = 'PASS_DECLARED_CONE_MIN_S_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for stage_prefix, style in (('trochanter_pitch', 'o-'), ('coxa_pitch', 's-'), ('coxa_roll', '^-')):
            for grp in GROUPS:
                xs, ys = [], []
                for e in trace:
                    if e['stage'].startswith('%s %s' % (stage_prefix, grp)) and e['s'] \
                            and e['s'].get('threshold_midpoint') is not None:
                        xs.append(float(e['stage'].split()[-2]))
                        ys.append(e['s']['threshold_midpoint'])
                if xs:
                    axes[0].plot(xs, ys, style, label='%s %s' % (stage_prefix, grp))
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        axes[0].set_xlabel('declared joint offset (deg)')
        axes[0].set_ylabel('s (declared pyramidal axis-ray diamond)')
        axes[0].set_title('per-group pitch/roll sweeps')
        axes[0].legend(fontsize=6)
        vals = [e['s']['threshold_midpoint'] for e in trace if e['s'] and e['s'].get('threshold_midpoint') is not None]
        axes[1].hist(vals, bins=10, color='#3060b0')
        axes[1].axvline(1.0, color='k', ls=':', label='s = 1')
        if vals:
            axes[1].axvline(min(vals), color='g', ls='--', label='s_best %.4f' % min(vals))
        axes[1].set_xlabel('s over the whole declared search')
        axes[1].set_ylabel('# poses')
        axes[1].set_title('how far the family gets')
        axes[1].legend(fontsize=8)
        axes[2].bar(range(len(GROUPS)), [best_e['leg_dist'][p] * 1000 for p in LEGS], color='#2f8f4f')
        axes[2].axhspan(1000 * TOUCH_LOW, 1000 * TOUCH_HIGH, color='0.85', label='declared touch band')
        axes[2].axhline(0, color='k', lw=0.6)
        axes[2].set_xticks(range(len(LEGS)))
        axes[2].set_xticklabels(LEGS)
        axes[2].set_ylabel('lowest tarsus penetration (um)')
        axes[2].set_title('best pose: per-leg contact')
        axes[2].legend(fontsize=8)
        fig.suptitle('Declared-cone minimum-s search (coordinate descent) - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'declared-cone-min-s.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1800:])
    finally:
        report['input_hashes_after'] = {p: sha(p) for p in hashes}
        report['hashes_unchanged'] = report['input_hashes_after'] == hashes
        if not report['hashes_unchanged']:
            report['status'] = 'FAIL_INPUT_MUTATED'
        with zipfile.ZipFile(out / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(Path(__file__), Path(__file__).name)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                                                    default=lambda o: o.tolist()) + '\n')
        (out / 'delivery-hashes.json').write_text(json.dumps(
            {f.name: sha(f) for f in sorted(out.iterdir())
             if f.is_file() and f.name != 'delivery-hashes.json'}, indent=2) + '\n')
    print(out)
    print(report['status'])
    if report['status'].startswith('PASS'):
        print('criterion:', json.dumps(report['criterion_result'], ensure_ascii=False)[:600])
        print('gate:', report.get('gate_run'))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
