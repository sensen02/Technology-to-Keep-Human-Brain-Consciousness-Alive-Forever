"""Build a new seated pose that satisfies an explicitly declared contact contract.

DECLARED NATURE OF THIS STEP
  This is a POSE SEARCH. Per-leg joint angles are chosen by damped least squares (IK) against the
  engine's own distance queries. It is NOT neural control, NOT a preprogrammed action sequence and
  NOT evidence about behaviour. Nothing in the canonical XML, the gains, the force scale or any
  archived directory is changed.

DECLARED CONTRACT (all checks use mj_kinematics + mj_geomDistance, never a "nearest contact" guess)
  C1 each of the six legs has at least one Tarsus geom whose distance to the floor plane lies in the
     declared touch band [-8e-3, -2e-3] mm (a 2-8 micrometre penetration, which also guarantees that
     MuJoCo generates an actual contact for the gate);
  C2 every one of the 41 non-tarsus geoms has floor clearance >= H_DECLARED = 0.40 mm;
  C3 every hinge joint sits at least 0.005 rad inside its range;
  C4 root quaternion = identity, root (x, y) = source values, root z maximised by bisection;
  C5 every joint stays within 0.40 rad of the source pose (so this is a local pose search).

OUTPUT: one npz (gate-compatible key 'projected_seated_qpos'), report.json, a figure and hashes.
Manual research implementation, not a product; nothing here demonstrates standing or balance.
"""
from pathlib import Path
from fractions import Fraction  # noqa: F401  (kept out of the hot loop, imported for symmetry)
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
DESK = Path('/home/sensen/Desktop/engine修复验收')
SOURCE_POSE = ROOT / 'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz'
FLOOR_GEOM = 0
LEG_PREFIXES = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
TOUCH_LOW, TOUCH_HIGH = -8e-3, -2e-3          # mm, declared touch band (penetration)
H_DECLARED = 0.40                             # mm, declared non-foot clearance floor
JOINT_MARGIN = 5e-3                           # rad, declared distance to every joint limit
MAX_DEVIATION = 0.40                          # rad, declared local-search bound
Z_SEARCH_SPAN = 0.30                          # mm, bisection upper offset from the source height
Z_RESOLUTION = 1e-4                           # mm
DIST_CAP = 4.0                                # mm, mj_geomDistance cap (values are clamped there)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    out = DESK / ('S2-seated-tarsus-only-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  nature_of_this_step=('DECLARED POSE SEARCH (per-leg IK against engine distance queries); '
                                       'NOT neural control, NOT preprogrammed action, NOT behaviour evidence'),
                  declared_contract=dict(
                      touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                      nonfoot_clearance_mm_min=H_DECLARED,
                      joint_limit_margin_rad=JOINT_MARGIN,
                      max_joint_deviation_rad=MAX_DEVIATION,
                      root_quaternion='identity', root_xy='source values',
                      root_z='maximised by bisection, resolution %.1e mm' % Z_RESOLUTION,
                      distance_queries='mj_kinematics + mj_geomDistance only'),
                  declared_calls=dict(project_mj_steps=0, project_forward_calls=0))
    inputs = [Path(__file__), SOURCE_POSE]
    hashes = {str(p): sha(p) for p in inputs}
    report['input_hashes'] = hashes
    try:
        import mujoco
        from engine import canonical_physics as cp
        report['canonical_xml'] = str(cp.CANONICAL_XML)
        report['canonical_xml_sha256'] = sha(cp.CANONICAL_XML)
        m = cp.load_model()
        d = mujoco.MjData(m)
        report['declared_calls']['project_model_loads'] = 1
        if (m.nq, m.nv, m.nu) != (49, 48, 90):
            raise ValueError('unexpected model dimensions')
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        foot = [g for g in range(m.ngeom) if 'Tarsus' in gname(g)]
        other = [g for g in range(m.ngeom) if g != FLOOR_GEOM and 'Tarsus' not in gname(g)]
        if len(foot) != 30 or len(other) != 41:
            raise ValueError('unexpected geom partition')
        leg_geoms = {p: [g for g in foot if gname(g).startswith(p)] for p in LEG_PREFIXES}
        if any(len(v) != 5 for v in leg_geoms.values()):
            raise ValueError('unexpected per-leg tarsus geom count')

        def leg_joints(prefix):
            js = [j for j in range(m.njnt) if jname(j).startswith('joint_' + prefix)]
            return [next(j for j in js if jname(j).endswith(s)) for s in JOINT_SUFFIXES]

        leg_joints = {p: leg_joints(p) for p in LEG_PREFIXES}

        def geom_dist(geom, with_points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, geom, DIST_CAP, ft))
            return (r, ft) if with_points else r

        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()
        if source.shape != (m.nq,):
            raise ValueError('unexpected source pose shape')
        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_leg_dist = {p: min(geom_dist(g) for g in leg_geoms[p]) for p in LEG_PREFIXES}
        source_clearance = {gname(g): geom_dist(g) for g in other}
        source_min_clearance = min(source_clearance.values())
        source_margin = {jname(j): float(min(source[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                            m.jnt_range[j, 1] - source[m.jnt_qposadr[j]]))
                         for j in range(1, m.njnt)}
        report['source_pose'] = dict(
            path=str(SOURCE_POSE), sha256=hashes[str(SOURCE_POSE)], root_z_mm=float(source[2]),
            per_leg_min_tarsus_distance_mm={p: float(v) for p, v in source_leg_dist.items()},
            min_nonfoot_clearance_mm=float(source_min_clearance),
            min_joint_limit_margin_rad=float(min(source_margin.values())),
            joints_at_or_inside_0p005_rad_of_a_limit=sorted(k for k, v in source_margin.items() if v < 5e-3),
            contact_band_ok=bool(all(TOUCH_LOW <= v <= TOUCH_HIGH for v in source_leg_dist.values())),
            nonfoot_clearance_ok=bool(source_min_clearance >= H_DECLARED))

        def solve_leg(prefix, qpos, xy_target, weight_xy=0.5, weight_reg=0.05):
            js = leg_joints[prefix]
            idx = [m.jnt_qposadr[j] for j in js]
            lo = np.array([max(m.jnt_range[j, 0] + JOINT_MARGIN,
                               source[idx[k]] - MAX_DEVIATION) for k, j in enumerate(js)])
            hi = np.array([min(m.jnt_range[j, 1] - JOINT_MARGIN,
                               source[idx[k]] + MAX_DEVIATION) for k, j in enumerate(js)])
            if np.any(lo >= hi):
                raise ValueError('empty joint box for leg %s' % prefix)
            x0 = np.clip(qpos[idx], lo, hi)

            def residual(x):
                q = qpos.copy()
                q[idx] = x
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                best = None
                for g in leg_geoms[prefix]:
                    r, ft = geom_dist(g, with_points=True)
                    if best is None or r < best[0]:
                        best = (r, ft)
                target = 0.5 * (TOUCH_LOW + TOUCH_HIGH)
                return np.concatenate([[10.0 * (best[0] - target)],
                                       [weight_xy * (best[1][3] - xy_target[0]),
                                        weight_xy * (best[1][4] - xy_target[1])],
                                       weight_reg * (x - x0)])
            sol = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                max_nfev=3000)
            return sol.x

        source_xy = {}
        for p in LEG_PREFIXES:
            best = min((geom_dist(g, with_points=True) for g in leg_geoms[p]), key=lambda t: t[0])
            source_xy[p] = (best[1][3], best[1][4])

        def build(z):
            q = source.copy()
            q[2] = z
            for p in LEG_PREFIXES:
                js = leg_joints[p]
                idx = [m.jnt_qposadr[j] for j in js]
                q[idx] = solve_leg(p, q, source_xy[p])
            return q

        def evaluate(q):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            leg_dist = {p: float(min(geom_dist(g) for g in leg_geoms[p])) for p in LEG_PREFIXES}
            clearance = float(min(geom_dist(g) for g in other))
            margin = float(min(min(q[m.jnt_qposadr[j]] - m.jnt_range[j, 0],
                                   m.jnt_range[j, 1] - q[m.jnt_qposadr[j]])
                               for j in range(1, m.njnt)))
            deviation = float(max(np.max(np.abs(q[[m.jnt_qposadr[j] for j in leg_joints[p]]]
                                           - source[[m.jnt_qposadr[j] for j in leg_joints[p]]]))
                                  for p in LEG_PREFIXES))
            ok = (all(TOUCH_LOW <= v <= TOUCH_HIGH for v in leg_dist.values())
                  and clearance >= H_DECLARED and margin >= JOINT_MARGIN
                  and deviation <= MAX_DEVIATION + 1e-9)
            return dict(root_z_mm=float(q[2]), per_leg_min_tarsus_distance_mm=leg_dist,
                        min_nonfoot_clearance_mm=clearance, min_joint_limit_margin_rad=margin,
                        max_joint_deviation_rad=deviation, contract_satisfied=bool(ok))

        # ---- bisection on root height
        z_lo = float(source[2])
        z_hi = z_lo + Z_SEARCH_SPAN
        low_eval = evaluate(build(z_lo))
        if not low_eval['contract_satisfied']:
            raise ValueError('contract not satisfied at the source height; nothing to search')
        high_eval = evaluate(build(z_hi))
        trace = [dict(z_mm=z_lo, feasible=True, **{k: v for k, v in low_eval.items() if k != 'root_z_mm'}),
                 dict(z_mm=z_hi, feasible=bool(high_eval['contract_satisfied']),
                      **{k: v for k, v in high_eval.items() if k != 'root_z_mm'})]
        if high_eval['contract_satisfied']:
            raise ValueError('search span exhausted: raise Z_SEARCH_SPAN')
        for _ in range(40):
            if z_hi - z_lo <= Z_RESOLUTION:
                break
            mid = 0.5 * (z_lo + z_hi)
            ev = evaluate(build(mid))
            if ev['contract_satisfied']:
                z_lo = mid
                trace.append(dict(z_mm=mid, feasible=True,
                                  **{k: v for k, v in ev.items() if k != 'root_z_mm'}))
            else:
                z_hi = mid
                trace.append(dict(z_mm=mid, feasible=False,
                                  **{k: v for k, v in ev.items() if k != 'root_z_mm'}))
        final = build(z_lo)
        final_eval = evaluate(final)

        # ---- one forward, only to check the active contact set (this is a static check, not a simulation)
        d.qpos[:] = final
        d.qvel[:] = 0
        d.act[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        report['declared_calls']['project_forward_calls'] = 1
        active = [dict(geom1=gname(int(c.geom1)), geom2=gname(int(c.geom2)), dist=float(c.dist),
                       dim=int(c.dim), efc_address=int(c.efc_address))
                  for c in d.contact if c.efc_address >= 0]
        nonfoot_contacts = [c for c in active if not ('Tarsus' in c['geom1'] or 'Tarsus' in c['geom2'])]
        seen = set()
        for c in active:
            for n in (c['geom1'], c['geom2']):
                if 'Tarsus' in n:
                    seen.add(n[:2])
        legs_with_contact = sorted(seen)
        if nonfoot_contacts:
            raise ValueError('a non-foot contact is active; the pose would be rejected by the gate')
        report['final_pose'] = dict(
            final_eval=final_eval, per_leg_tarsus_at_final=None,
            active_contacts=active, active_contact_count=len(active),
            nonfoot_active_contacts=0, legs_with_at_least_one_tarsus_contact=legs_with_contact,
            gate_precondition_ok=bool(len(legs_with_contact) == 6 and not nonfoot_contacts))
        report['search_trace'] = trace
        report['search_iterations'] = len(trace)
        report['height_gain_mm'] = float(final_eval['root_z_mm'] - source[2])
        report['final_pose']['per_leg_tarsus_at_final'] = {
            p: dict(min_distance_mm=final_eval['per_leg_min_tarsus_distance_mm'][p],
                    source_min_distance_mm=float(source_leg_dist[p])) for p in LEG_PREFIXES}
        report['nonfoot_clearance_table'] = sorted(
            ({'geom': gname(g), 'final_mm': geom_dist(g), 'source_mm': source_clearance[gname(g)]}
             for g in other), key=lambda r: r['final_mm'])
        np.savez_compressed(out / 'candidate_seated_pose.npz', projected_seated_qpos=final,
                            qpos=final, source_projected_seated_qpos=source)
        report['candidate_npz'] = dict(path=str(out / 'candidate_seated_pose.npz'),
                                       key_for_gate='projected_seated_qpos')
        report['wall_seconds'] = time.time() - started

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        labels = list(LEG_PREFIXES)
        src = [1e3 * source_leg_dist[p] for p in labels]
        fin = [1e3 * final_eval['per_leg_min_tarsus_distance_mm'][p] for p in labels]
        x = np.arange(len(labels))
        axes[0].axhspan(1e3 * TOUCH_LOW, 1e3 * TOUCH_HIGH, color='0.85', label='declared touch band 2-8 um')
        axes[0].bar(x - 0.2, src, 0.4, label='source pose', color='#b03030')
        axes[0].bar(x + 0.2, fin, 0.4, label='new pose', color='#3060b0')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('penetration of lowest tarsus geom (um)')
        axes[0].set_title('C1 per-leg tarsus contact')
        axes[0].legend(fontsize=8)
        order = [r for r in report['nonfoot_clearance_table']]
        axes[1].semilogy(range(len(order)), [max(r['final_mm'], 1e-6) for r in order], 'o-', ms=3,
                         label='new pose')
        axes[1].semilogy(range(len(order)), [max(r['source_mm'], 1e-6) for r in order], 's--', ms=3,
                         label='source pose')
        axes[1].axhline(H_DECLARED, color='r', ls=':', label='declared threshold %.2f mm' % H_DECLARED)
        axes[1].set_xlabel('non-foot geoms (sorted)')
        axes[1].set_ylabel('floor clearance (mm)')
        axes[1].set_title('C2 non-foot clearance')
        axes[1].legend(fontsize=8)
        zs = [t['z_mm'] for t in trace]
        axes[2].plot(range(len(trace)), zs, 'o-', color='#3060b0')
        for i, t in enumerate(trace):
            if not t['feasible']:
                axes[2].plot(i, t['z_mm'], 'rx', ms=8)
        axes[2].axhline(final_eval['root_z_mm'], color='g', ls='--',
                        label='accepted %.4f mm' % final_eval['root_z_mm'])
        axes[2].axhline(float(source[2]), color='k', ls=':', label='source %.4f mm' % source[2])
        axes[2].set_xlabel('bisection step (red x = contract violated)')
        axes[2].set_ylabel('root height z (mm)')
        axes[2].set_title('C4 height search')
        axes[2].legend(fontsize=8)
        fig.suptitle('New seated pose under a declared tarsus-only contract - '
                     'POSE SEARCH / manual research, NOT standing, NOT a product')
        fig.tight_layout()
        fig.savefig(out / 'seated-tarsus-only.png', dpi=130)
        plt.close(fig)
        report['status'] = 'PASS_SEATED_POSE_MEETS_DECLARED_TARSUS_ONLY_CONTRACT'
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
        print('accepted root z = %.6f mm (source %.6f, gain %.6f)' %
              (report['final_pose']['final_eval']['root_z_mm'], report['source_pose']['root_z_mm'],
               report['height_gain_mm']))
        print('per-leg tarsus distance (mm):', report['final_pose']['final_eval']['per_leg_min_tarsus_distance_mm'])
        print('min non-foot clearance = %.6f mm ; active contacts = %d ; non-foot = %d' %
              (report['final_pose']['final_eval']['min_nonfoot_clearance_mm'],
               report['final_pose']['active_contact_count'], report['final_pose']['nonfoot_active_contacts']))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
