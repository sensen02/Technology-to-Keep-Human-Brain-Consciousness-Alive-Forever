"""Single-DOF sweep on a declared, previously FIXED degree of freedom: the tarsus foot-contact position.

WHY THIS DOF: in the height sweep every pose was built with the foot contact target pinned to the SOURCE
footprint (the IK kept each leg's lowest tarsus point at the source (x, y)), so foot placement was a fixed
parameter rather than a searched one. This script declares it as the swept DOF: every leg's IK target in
the horizontal plane is scaled uniformly about the body axis by `spread`, which moves the six contact
points in/out together and therefore changes the support polygon and the moment arms of the leg muscles.

QUESTION: the declared cone (gen_axis = the cone MuJoCo actually realises, old name diamond) needs
s_K4 ~ 1.074 at the contract-optimal height (root z = 1.98 mm). Is that ~7.4 percent shortfall a POSE
ARTEFACT (removable by moving the feet) or a REAL FORCE-BUDGET GAP (not removable within the declared
family)?

DECLARED: pose search / parametric study of one declared DOF. NOT neural control, NOT a preprogrammed
action, NOT behaviour evidence, NOT standing. Only derived/search scripts are touched; the canonical XML,
the gains, the friction, the force scale and every historical hash are untouched.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares, linprog

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
FLOOR_GEOM = 0
LEG_PREFIXES = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
TOUCH_LOW, TOUCH_HIGH = -8e-3, -2e-3
H_DECLARED = 0.40
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
DIST_CAP = 4.0
Z_FIXED = 1.98                       # contract-optimal height from the fine-grid sweep
SPREADS = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
DIAMOND = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
TAN_HALF_DEN = 10 ** 6
S_CAP = 500.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ellipse_rays(k):
    pts = set()
    for i in range(k):
        th = 2 * math.pi * i / k
        if abs(math.sin(th)) < 1e-12:
            pts.add((1.0 if math.cos(th) > 0 else -1.0, 0.0))
        elif abs(math.cos(th)) < 1e-12:
            pts.add((0.0, 1.0 if math.sin(th) > 0 else -1.0))
        else:
            t = round(math.tan(th / 2) * TAN_HALF_DEN) / TAN_HALF_DEN
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return sorted(pts)


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
    out = DESK / ('S2-footprint-dof-sweep-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  sweep_question=('is the ~7.4 percent shortfall at the contract-optimal height a pose artefact '
                                  '(removable by moving the foot contact points) or a real force-budget gap?'),
                  swept_dof=dict(name='tarsus foot-contact position (uniform footprint scale about the body axis)',
                                 why_fixed_before=('the height sweep pinned each leg\'s IK target to the source '
                                                   'footprint, so foot placement was fixed, not searched'),
                                 declared_range=list(SPREADS), fixed_root_z_mm=Z_FIXED,
                                 contract=dict(touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                                               nonfoot_clearance_mm_min=H_DECLARED,
                                               joint_limit_margin_rad=JOINT_MARGIN,
                                               max_joint_deviation_rad=MAX_DEVIATION)),
                  nature_of_this_step=('DECLARED single-DOF pose search; NOT neural control, NOT a '
                                       'preprogrammed action, NOT behaviour evidence'),
                  declared_calls=dict(project_mj_steps=0),
                  absolute_force_values='NOT_ESTABLISHED')
    hashes = {str(p): sha(p) for p in (Path(__file__), SOURCE_POSE)}
    report['input_hashes'] = hashes
    try:
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
        leg_geoms = {p: [g for g in foot if gname(g).startswith(p)] for p in LEG_PREFIXES}
        leg_joints = {}
        for p in LEG_PREFIXES:
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + p)]
            leg_joints[p] = [next(j for j in js if jname(j).endswith(s)) for s in JOINT_SUFFIXES]
        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()

        def gdist(geom, points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, geom, DIST_CAP, ft))
            return (r, ft) if points else r

        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_xy = {p: min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])[1][3:5]
                     for p in LEG_PREFIXES}

        def build(spread):
            q = source.copy()
            q[2] = Z_FIXED
            for p in LEG_PREFIXES:
                idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
                lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                x0 = np.clip(q[idx], lo, hi)
                target = spread * source_xy[p]

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

        def system(q):
            d.qpos[:] = q
            d.qvel[:] = 0
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            report['declared_calls']['project_forward_calls'] = report['declared_calls'].get(
                'project_forward_calls', 0) + 1
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(gdist(g) for g in leg_geoms[p])) for p in LEG_PREFIXES}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]]))
                                  for p in LEG_PREFIXES))
            contacts, footprints = [], {}
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
                                     mu=np.asarray(c.friction)[:2], dist=float(c.dist),
                                     pos=np.asarray(c.pos).copy()))
            for p in LEG_PREFIXES:
                pts = [c['pos'] for c in contacts if any(n.startswith(p) for n in c['names'])]
                if pts:
                    footprints[p] = [float(np.mean([q[0] for q in pts])), float(np.mean([q[1] for q in pts]))]
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
            return dict(leg_dist=leg_dist, clearance=clearance, margin=margin, deviation=deviation,
                        contacts=contacts, nonfoot=nonfoot, target=target_b, R=R, contract=bool(contract),
                        footprint=footprints)

        rows = []
        for spread in SPREADS:
            q = build(spread)
            sysd = system(q)
            lo = m.actuator_ctrlrange[:, 0].copy()
            hi = m.actuator_ctrlrange[:, 1].copy()
            s4 = s32 = None
            if sysd['contract']:
                s4 = threshold(sysd['R'], cone_cols(sysd['contacts'], DIAMOND), sysd['target'], lo, hi)
                s32 = threshold(sysd['R'], cone_cols(sysd['contacts'], ellipse_rays(32)), sysd['target'], lo, hi)
            row = dict(spread=spread, contract_satisfied=sysd['contract'],
                       contacts=len(sysd['contacts']), nonfoot_contacts=sysd['nonfoot'],
                       min_nonfoot_clearance_mm=sysd['clearance'],
                       max_joint_deviation_rad=sysd['deviation'],
                       min_tarsus_distance_mm=sysd['leg_dist'],
                       footprint_xy=sysd['footprint'],
                       support_polygon_radius_mm=float(max(np.hypot(v[0], v[1]) for v in sysd['footprint'].values())
                                                       if sysd['footprint'] else None),
                       s_gen_axis=s4, s_K32=s32)
            rows.append(row)
            print('spread %.2f contract=%s contacts=%2d clearance=%.4f s_K4=%s s_K32=%s' % (
                spread, sysd['contract'], len(sysd['contacts']), sysd['clearance'],
                None if s4 is None else s4.get('threshold_midpoint'),
                None if s32 is None else s32.get('threshold_midpoint')), flush=True)
        report['sweep'] = rows
        ok_rows = [r for r in rows if r['contract_satisfied'] and r['s_gen_axis']
                   and r['s_gen_axis'].get('threshold_midpoint') is not None]
        if ok_rows:
            best = min(ok_rows, key=lambda r: r['s_gen_axis']['threshold_midpoint'])
            reachable = min(r['s_gen_axis']['threshold_midpoint'] for r in ok_rows)
            monotone = all(a['s_gen_axis']['threshold_midpoint'] <= b['s_gen_axis']['threshold_midpoint']
                           for a, b in zip(ok_rows, ok_rows[1:])) or \
                       all(a['s_gen_axis']['threshold_midpoint'] >= b['s_gen_axis']['threshold_midpoint']
                           for a, b in zip(ok_rows, ok_rows[1:]))
            report['conclusion'] = dict(
                best_spread=best['spread'], best_s_gen_axis=best['s_gen_axis']['threshold_midpoint'],
                best_s_K32=None if best['s_K32'] is None else best['s_K32'].get('threshold_midpoint'),
                minimum_over_the_swept_dof=reachable,
                monotone_within_the_contract=bool(monotone),
                s_1_reachable=bool(reachable <= 1.0),
                answer=('POSE ARTEFACT: moving the foot contact points removes the shortfall, s=1 becomes '
                        'reachable'
                        if reachable <= 1.0 else
                        'REAL FORCE-BUDGET GAP within this declared DOF: no footprint scale in the declared '
                        'range brings s=1 into the feasible region; the best still needs %.4f times the '
                        'declared muscle strength' % reachable))
        else:
            report['conclusion'] = dict(answer='no contract-satisfying pose in the declared range; nothing to '
                                               'conclude about the budget')
        report['status'] = 'PASS_FOOTPRINT_DOF_SWEEP_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        ok = [r for r in rows if r['s_gen_axis'] and r['s_gen_axis'].get('threshold_midpoint') is not None]
        bad = [r for r in rows if not r['contract_satisfied']]
        axes[0].semilogy([r['spread'] for r in ok], [r['s_gen_axis']['threshold_midpoint'] for r in ok], 'o-',
                         color='#3060b0', label='gen_axis (declared cone)')
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        if report['conclusion'].get('best_spread'):
            axes[0].plot([report['conclusion']['best_spread']], [report['conclusion']['best_s_gen_axis']], 'g*',
                         ms=14, label='best %.4f at spread %.2f' % (report['conclusion']['best_s_gen_axis'],
                                                                    report['conclusion']['best_spread']))
        if bad:
            axes[0].plot([r['spread'] for r in bad],
                         [1.0] * len(bad), 'rx', ms=10, label='contract violated')
        axes[0].set_xlabel('footprint scale (declared DOF; 1.0 = source footprint)')
        axes[0].set_ylabel('required muscle-force multiplier s')
        axes[0].set_title('single-DOF sweep: foot-contact position')
        axes[0].legend(fontsize=8)
        axes[1].plot([r['spread'] for r in rows], [r['min_nonfoot_clearance_mm'] for r in rows], 'o-')
        axes[1].axhline(H_DECLARED, color='r', ls=':', label='declared clearance floor')
        axes[1].set_xlabel('footprint scale')
        axes[1].set_ylabel('min non-foot clearance (mm)')
        axes[1].set_title('contract margin vs the swept DOF')
        axes[1].legend(fontsize=8)
        for r in rows:
            pts = list(r['footprint_xy'].values())
            if pts:
                axes[2].plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=3,
                             label='spread %.2f' % r['spread'])
        axes[2].set_aspect('equal')
        axes[2].set_xlabel('x (mm)')
        axes[2].set_ylabel('y (mm)')
        axes[2].set_title('support polygon per scale')
        axes[2].legend(fontsize=6, ncol=2)
        fig.suptitle('Single-DOF (foot placement) sweep under the declared contract - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'footprint-dof-sweep.png', dpi=130)
        plt.close(fig)
    except Exception as exc:
        import traceback
        report.update(status='FAIL_STOP', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc()[-1500:])
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
        print('conclusion:', json.dumps(report['conclusion'], ensure_ascii=False)[:400])
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
