"""Height sweep of the declared tarsus-only contract: how hard is each seated height to support?

Same declared contract and the same per-leg IK as the seated-pose step, but root height is swept
instead of maximised. For every height the script reports (a) whether the contract holds and (b) the
required muscle-force multiplier s at which the ideal static LP becomes feasible, for the declared
pyramidal cone (K=4) and for a fine inscribed Coulomb polygon (K=32).

CONE NAMING (unified 2026-10-15; single name, no second scheme):
  declared pyramidal (axis-ray diamond) = the non-negative hull of the axis rays n+-mu*t1, n+-mu*t2, i.e.
  the four rows MuJoCo emits for a condim-3 pyramidal contact, and the cone it actually realises.
  BEHAVIOURAL CRITERION: the 45-degree sliding ratio is 0.7071 = 1/sqrt(2) (elliptic control 1.0000),
  which excludes both the exact Coulomb disk (1.0) and the axis-aligned square (1.4142).
  The diagonal-ray block (formerly "outer"/"union") is a RELAXATION beyond the declared cone and beyond
  the Coulomb disk and must never be used to claim s<=1 feasibility. The K=32 column here is a finer
  inscribed polygon of the exact Coulomb disk.

DECLARED: this is a POSE SEARCH / parametric study of a declared family. It is NOT neural control,
NOT a preprogrammed action, NOT a behaviour test, and a feasible static LP at s<=1 would still NOT be
standing, balance or biology. No canonical XML, gain, friction or force-scale value is touched.
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
Z_GRID = [round(1.60 + 0.05 * i, 4) for i in range(13)]        # 1.60 .. 2.20 mm
# Optional declared override so the same code can run a finer grid under the same contract:
#   python -B run_seated_height_sweep.py --grid 1.85,1.90,1.94,...,2.10
if len(sys.argv) > 2 and sys.argv[1] == '--grid':
    Z_GRID = [round(float(v), 6) for v in sys.argv[2].split(',') if v.strip()]
    _GRID_SOURCE = 'declared argv grid: ' + sys.argv[2]
else:
    _GRID_SOURCE = 'default coarse grid 1.60..2.20 step 0.05'
Z_MAX_REACH = 2.2222                                           # accepted maximum from the pose step
TAN_HALF_DEN = 10 ** 6
S_CAP = 500.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ellipse_rays(k):
    pts = set()
    for i in range(k):
        theta = 2 * math.pi * i / k
        if abs(math.sin(theta)) < 1e-12:
            pts.add((1.0 if math.cos(theta) > 0 else -1.0, 0.0))
        elif abs(math.cos(theta)) < 1e-12:
            pts.add((0.0, 1.0 if math.sin(theta) > 0 else -1.0))
        else:
            t = round(math.tan(theta / 2) * TAN_HALF_DEN) / TAN_HALF_DEN
            pts.add(((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t)))
    return sorted(pts)


def main():
    started = time.time()
    out = DESK / ('S2-seated-height-sweep-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  nature_of_this_step=('DECLARED POSE SEARCH: 1-D root-height sweep with per-leg IK; '
                                       'NOT neural control, NOT preprogrammed action'),
                  declared_contract=dict(touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                                         nonfoot_clearance_mm_min=H_DECLARED,
                                         joint_limit_margin_rad=JOINT_MARGIN,
                                         max_joint_deviation_rad=MAX_DEVIATION),
                  declared_grid_mm=Z_GRID, grid_source=_GRID_SOURCE, s_cap=S_CAP,
                  cone_naming=dict(k4='declared pyramidal (axis-ray diamond), K=4',
                                   k32='finer inscribed polygon of the exact Coulomb disk'),
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0))
    hashes = {str(p): sha(p) for p in (Path(__file__), SOURCE_POSE)}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        m = cp.load_model()
        d = mujoco.MjData(m)
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

        def build(z):
            q = source.copy()
            q[2] = z
            for p in LEG_PREFIXES:
                idx = [m.jnt_qposadr[j] for j in leg_joints[p]]
                lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN, source[idx[k]] - MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN, source[idx[k]] + MAX_DEVIATION)
                               for k, j in enumerate(leg_joints[p])])
                if np.any(lo >= hi):
                    raise ValueError('empty joint box')
                x0 = np.clip(q[idx], lo, hi)

                def residual(x):
                    qq = q.copy()
                    qq[idx] = x
                    d.qpos[:] = qq
                    mujoco.mj_kinematics(m, d)
                    best = min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])
                    return np.concatenate([[10.0 * (best[0] - 0.5 * (TOUCH_LOW + TOUCH_HIGH))],
                                           [0.5 * (best[1][3] - source_xy[p][0]),
                                            0.5 * (best[1][4] - source_xy[p][1])],
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
            report['declared_calls']['project_forward_calls'] += 1
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(gdist(g) for g in leg_geoms[p])) for p in LEG_PREFIXES}
            clearance = float(min(gdist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]]) for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]]))
                                  for p in LEG_PREFIXES))
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
            base = d.qfrc_actuator.copy()
            target = d.qfrc_bias - d.qfrc_passive - base
            R = np.zeros((m.nv, m.nu))
            for i in range(m.nu):
                d.act[:] = 0
                d.act[i] = 1
                mujoco.mj_forward(m, d)
                report['declared_calls']['project_forward_calls'] += 1
                R[:, i] = d.qfrc_actuator - base
            return dict(q=q, leg_dist=leg_dist, clearance=clearance, margin=margin, deviation=deviation,
                        contacts=contacts, target=target, R=R)

        def multiplier(sysd, dirs):
            cols = []
            for c in sysd['contacts']:
                frame, mu, jac_t = c['frame'], c['mu'], c['J'].T
                normal, t1, t2 = frame[0], mu[0] * frame[1], mu[1] * frame[2]
                for p, q in dirs:
                    cols.append(jac_t @ (normal + p * t1 + q * t2))
            cone = np.asarray(cols).T
            lo = m.actuator_ctrlrange[:, 0].copy()
            hi = m.actuator_ctrlrange[:, 1].copy()
            target, R = sysd['target'], sysd['R']

            def ok(s):
                A = np.hstack([R * s, cone])
                scale = np.maximum(1.0, np.maximum(np.abs(target), np.max(np.abs(A), axis=1)))
                res = linprog(np.zeros(A.shape[1]), A_eq=A / scale[:, None], b_eq=target / scale,
                              bounds=list(zip(lo, hi)) + [(0, None)] * cone.shape[1], method='highs',
                              options=dict(dual_feasibility_tolerance=1e-10,
                                           primal_feasibility_tolerance=1e-10))
                return res.status == 0
            f1 = ok(1.0)
            low, high = (0.2, 1.0) if f1 else (1.0, 2.0)
            while high <= S_CAP and not ok(high):
                high *= 1.25
            if high > S_CAP:
                return dict(feasible_at_s_1=bool(f1), threshold_midpoint=None, cap_reached=True,
                            infeasible_up_to_s=high / 1.25)
            for _ in range(28):
                mid = 0.5 * (low + high)
                if ok(mid):
                    high = mid
                else:
                    low = mid
            return dict(feasible_at_s_1=bool(f1), cap_reached=False, infeasible_below=low,
                        feasible_above=high, threshold_midpoint=0.5 * (low + high))

        rows = []
        for z in sorted(set(Z_GRID + [Z_MAX_REACH])):
            try:
                q = build(z)
                sysd = system(q)
                contract = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in sysd['leg_dist'].values())
                            and sysd['clearance'] >= H_DECLARED and sysd['margin'] >= JOINT_MARGIN
                            and sysd['deviation'] <= MAX_DEVIATION + 1e-9)
                nonfoot = sum(1 for c in sysd['contacts']
                              if not any('Tarsus' in n for n in c['names']))
                if nonfoot:
                    contract = False
                row = dict(z_mm=z, contract_satisfied=bool(contract),
                           per_leg_min_tarsus_distance_mm=sysd['leg_dist'],
                           min_nonfoot_clearance_mm=sysd['clearance'],
                           min_joint_limit_margin_rad=sysd['margin'],
                           max_joint_deviation_rad=sysd['deviation'],
                           contact_count=len(sysd['contacts']), nonfoot_contact_count=nonfoot,
                           s_K4=multiplier(sysd, [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]),
                           s_K32=multiplier(sysd, ellipse_rays(32)))
                rows.append(row)
                print('z=%.4f contract=%s clearance=%.4f s_K4=%s s_K32=%s' % (
                    z, contract, sysd['clearance'], row['s_K4'].get('threshold_midpoint'),
                    row['s_K32'].get('threshold_midpoint')), flush=True)
            except Exception as exc:
                rows.append(dict(z_mm=z, contract_satisfied=False, error=type(exc).__name__ + ': ' + str(exc)))
                print('z=%.4f FAILED %s' % (z, exc), flush=True)
        report['sweep'] = rows
        valid = [r for r in rows if r.get('contract_satisfied') and r.get('s_K4', {}).get('threshold_midpoint')]
        if valid:
            best = min(valid, key=lambda r: r['s_K4']['threshold_midpoint'])
            report['best_in_contract'] = dict(z_mm=best['z_mm'],
                                              s_K4=best['s_K4']['threshold_midpoint'],
                                              s_K32=best['s_K32'].get('threshold_midpoint'),
                                              nonfoot_clearance_mm=best['min_nonfoot_clearance_mm'])
            report['contract_height_window_mm'] = [min(r['z_mm'] for r in valid), max(r['z_mm'] for r in valid)]
            report['reading'] = (
                'Within the declared contract family (per-leg IK, tarsus-only, clearance >= %.2f mm, '
                'joints >= %.3f rad inside limits, <= %.2f rad from the source pose) the required '
                'muscle-force multiplier varies strongly with root height. A value at or below 1 would '
                'mean only that the ideal static LP becomes feasible there; that is NOT standing, NOT a '
                'MuJoCo equilibrium and NOT biology, and no such claim is made here.'
                % (H_DECLARED, JOINT_MARGIN, MAX_DEVIATION))
        report['status'] = 'PASS_HEIGHT_SWEEP_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        good = [r for r in rows if r.get('contract_satisfied')]
        bad = [r for r in rows if not r.get('contract_satisfied')]
        for key, style, name in (('s_K4', 'o-', 'gen_axis = declared pyramidal cone (K=4); old name diamond'),
                                 ('s_K32', 's--', 'fine inscribed Coulomb polygon (K=32)')):
            xs = [r['z_mm'] for r in good if r.get(key, {}).get('threshold_midpoint')]
            ys = [r[key]['threshold_midpoint'] for r in good if r.get(key, {}).get('threshold_midpoint')]
            axes[0].semilogy(xs, ys, style, label=name)
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        axes[0].set_xlabel('root height z (mm)')
        axes[0].set_ylabel('required muscle-force multiplier s')
        axes[0].set_title('support difficulty vs seated height (contract-satisfying heights only)')
        axes[0].legend(fontsize=8)
        axes[1].plot([r['z_mm'] for r in rows if 'min_nonfoot_clearance_mm' in r],
                     [r['min_nonfoot_clearance_mm'] for r in rows if 'min_nonfoot_clearance_mm' in r],
                     'o-', color='#3060b0', label='min non-foot clearance')
        axes[1].axhline(H_DECLARED, color='r', ls=':', label='declared clearance floor %.2f mm' % H_DECLARED)
        if bad:
            axes[1].plot([r['z_mm'] for r in bad], [H_DECLARED] * len(bad), 'rx', ms=10,
                         label='contract violated')
        axes[1].set_xlabel('root height z (mm)')
        axes[1].set_ylabel('mm')
        axes[1].set_title('geometric margin vs height')
        axes[1].legend(fontsize=8)
        fig.suptitle('Seated height sweep under the declared tarsus-only contract - POSE SEARCH, '
                     'manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'height-sweep.png', dpi=130)
        plt.close(fig)
        report['wall_seconds'] = time.time() - started
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
    if report['status'].startswith('PASS') and 'best_in_contract' in report:
        print('best in contract:', report['best_in_contract'])
        print('contract height window (mm):', report['contract_height_window_mm'])
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
