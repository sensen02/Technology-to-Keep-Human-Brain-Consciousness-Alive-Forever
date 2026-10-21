"""Search the declared pose family for a pose that is statically FEASIBLE at s = 1 under gen_axis.

gen_axis = the non-negative hull of the axis rays n +- mu*t1, n +- mu*t2 = the DECLARED pyramidal cone
that MuJoCo actually realises (measured 45-degree sliding ratio 0.7071 = 1/sqrt(2); old name "diamond").
The earlier single-DOF sweeps left the family minimum at s_K4 = 1.0738 (height) and 1.0586 (footprint),
so this script searches the declared family in TWO dimensions and then refines a third:

  stage 1  root height z  x  footprint scale (declared grid)
  stage 2  uniform coxa-yaw offset applied to all six legs at the best stage-1 point

If a pose reaches s = 1 the script runs the canonical S2 support gate on it (active contacts, zero
non-foot contacts, joint-limit margins, one mj_forward) and archives the candidate; otherwise it reports
the lowest achievable s_K4 and the remaining shortfall.

DECLARED: pose search / parametric study. NOT neural control, NOT a preprogrammed action, NOT behaviour
evidence, NOT standing. Only derived/search scripts are touched; the canonical XML, the gains, the
friction, the force scale and every historical hash are untouched.
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
LEG_PREFIXES = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')
JOINT_SUFFIXES = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
                  'Trochanter_roll', 'Tibia_pitch')
TOUCH_LOW, TOUCH_HIGH = -8e-3, -2e-3
H_DECLARED = 0.40
JOINT_MARGIN = 5e-3
MAX_DEVIATION = 0.40
DIST_CAP = 4.0
Z_GRID = (1.90, 1.94, 1.98, 2.02, 2.06)
SPREAD_GRID = (0.90, 0.95, 1.00)
YAW_OFFSETS_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
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
    out = DESK / ('S2-seated-feasibility-search-' + time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', manual_research_not_product=True, standing=False, walking=False,
                  goal=('find a pose inside the DECLARED family that is statically feasible at s=1 under '
                        'gen_axis (K=4, the cone MuJoCo realises); if none exists, report the lowest s_K4 and '
                        'the shortfall'),
                  declared_search=dict(stage1=dict(root_z_mm=list(Z_GRID), footprint_scale=list(SPREAD_GRID)),
                                       stage2=dict(uniform_coxa_yaw_offset_deg=list(YAW_OFFSETS_DEG),
                                                   applied_at='the best stage-1 point'),
                                       contract=dict(touch_band_mm=[TOUCH_LOW, TOUCH_HIGH],
                                                     nonfoot_clearance_mm_min=H_DECLARED,
                                                     joint_limit_margin_rad=JOINT_MARGIN,
                                                     max_joint_deviation_rad=MAX_DEVIATION)),
                  nature_of_this_step=('DECLARED two-then-three-parameter pose search; NOT neural control, '
                                       'NOT preprogrammed action, NOT behaviour evidence'),
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
        yaw_idx = {}
        for p in LEG_PREFIXES:
            j = next(j for j in range(m.njnt) if jname(j) == 'joint_' + p + 'Coxa_yaw')
            yaw_idx[p] = (j, int(m.jnt_qposadr[j]))
        source = np.load(SOURCE_POSE)['projected_seated_qpos'].copy()

        def gdist(geom, points=False):
            ft = np.zeros(6)
            r = float(mujoco.mj_geomDistance(m, d, FLOOR_GEOM, geom, DIST_CAP, ft))
            return (r, ft) if points else r

        d.qpos[:] = source
        mujoco.mj_kinematics(m, d)
        source_xy = {p: min((gdist(g, True) for g in leg_geoms[p]), key=lambda t: t[0])[1][3:5]
                     for p in LEG_PREFIXES}

        def build(z, spread, yaw_deg):
            q = source.copy()
            q[2] = z
            for p in LEG_PREFIXES:
                j, adr = yaw_idx[p]
                q[adr] = source[adr] + math.radians(yaw_deg)
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

        def evaluate(q):
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
            legs = sorted({n[:2] for c in contacts for n in c['names'] if 'Tarsus' in n})
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
                        contacts=contacts, nonfoot=nonfoot, legs=legs, target=target_b, R=R,
                        contract=bool(contract))

        stage1 = []
        best = None
        for z in Z_GRID:
            for spread in SPREAD_GRID:
                q = build(z, spread, 0.0)
                sysd = evaluate(q)
                entry = dict(z_mm=z, spread=spread, yaw_deg=0.0, contract=sysd['contract'],
                             contacts=len(sysd['contacts']), nonfoot=sysd['nonfoot'],
                             clearance=sysd['clearance'], deviation=sysd['deviation'])
                if sysd['contract']:
                    lo = m.actuator_ctrlrange[:, 0].copy()
                    hi = m.actuator_ctrlrange[:, 1].copy()
                    s4 = threshold(sysd['R'], cone_cols(sysd['contacts'], DIAMOND), sysd['target'], lo, hi)
                    s32 = threshold(sysd['R'], cone_cols(sysd['contacts'], ellipse_rays(32)),
                                    sysd['target'], lo, hi)
                    entry.update(s_gen_axis=s4, s_K32=s32)
                    mid = s4.get('threshold_midpoint')
                    if mid is not None and (best is None
                                            or mid < best['entry']['s_gen_axis']['threshold_midpoint']):
                        best = dict(entry=entry, q=q)
                stage1.append(entry)
                print('stage1 z=%.2f spread=%.2f contract=%s s_K4=%s' % (
                    z, spread, sysd['contract'], entry.get('s_gen_axis', {}).get('threshold_midpoint')),
                    flush=True)
        report['stage1_grid'] = stage1
        report['stage1_best'] = None if best is None else best['entry']

        stage2 = []
        best2 = None
        if best is not None:
            z0 = best['entry']['z_mm']
            sp0 = best['entry']['spread']
            for yaw in YAW_OFFSETS_DEG:
                q = build(z0, sp0, yaw)
                sysd = evaluate(q)
                entry = dict(z_mm=z0, spread=sp0, yaw_deg=yaw, contract=sysd['contract'],
                             contacts=len(sysd['contacts']), nonfoot=sysd['nonfoot'],
                             clearance=sysd['clearance'], deviation=sysd['deviation'])
                if sysd['contract']:
                    lo = m.actuator_ctrlrange[:, 0].copy()
                    hi = m.actuator_ctrlrange[:, 1].copy()
                    entry['s_gen_axis'] = threshold(sysd['R'], cone_cols(sysd['contacts'], DIAMOND),
                                                    sysd['target'], lo, hi)
                    mid = entry['s_gen_axis'].get('threshold_midpoint')
                    if mid is not None and (best2 is None
                                            or mid < best2['entry']['s_gen_axis']['threshold_midpoint']):
                        best2 = dict(entry=entry, q=q)
                stage2.append(entry)
                print('stage2 yaw=%+.1f contract=%s s_K4=%s' % (
                    yaw, sysd['contract'], entry.get('s_gen_axis', {}).get('threshold_midpoint')), flush=True)
        report['stage2_grid'] = stage2
        report['stage2_best'] = None if best2 is None else best2['entry']

        winners = [b for b in (best, best2) if b is not None]
        if winners:
            final = min(winners, key=lambda b: b['entry']['s_gen_axis']['threshold_midpoint'])
            s_final = final['entry']['s_gen_axis']['threshold_midpoint']
            report['final_pose'] = dict(parameters=dict(z_mm=final['entry']['z_mm'],
                                                        spread=final['entry']['spread'],
                                                        yaw_deg=final['entry'].get('yaw_deg', 0.0)),
                                        s_gen_axis=final['entry']['s_gen_axis'],
                                        clearance_mm=final['entry']['clearance'],
                                        contacts=final['entry']['contacts'],
                                        answer=('FEASIBLE at s=1 under gen_axis: a declared-family pose was '
                                                'found' if s_final <= 1.0 else
                                                'NOT feasible at s=1 under gen_axis inside the searched family; '
                                                'the lowest requirement found is %.4f, i.e. still %.2f percent '
                                                'above the declared muscle strength' % (s_final, 100 * (s_final - 1))))
            np.savez_compressed(out / 'best_candidate_pose.npz', projected_seated_qpos=final['q'])
            report['best_candidate_npz'] = str(out / 'best_candidate_pose.npz')
            if s_final <= 1.0:
                gate = subprocess.run([sys.executable, '-B', str(ROOT / 'run_s2_support_gate.py'),
                                       str(out / 'best_candidate_pose.npz')],
                                      capture_output=True, text=True, timeout=1800)
                report['gate_run'] = dict(returncode=gate.returncode,
                                          stdout_tail=gate.stdout.strip().splitlines()[-3:],
                                          stderr_tail=gate.stderr.strip().splitlines()[-3:])
            else:
                report['gate_run'] = dict(skipped='s=1 is not feasible on gen_axis, so the support gate was not '
                                                  'run on this candidate')
        report['status'] = 'PASS_SEATED_FEASIBILITY_SEARCH_REPORTED'

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        for spread in SPREAD_GRID:
            pts = [e for e in stage1 if e['spread'] == spread and e.get('s_gen_axis')
                   and e['s_gen_axis'].get('threshold_midpoint') is not None]
            if pts:
                axes[0].plot([e['z_mm'] for e in pts], [e['s_gen_axis']['threshold_midpoint'] for e in pts],
                             'o-', label='spread %.2f' % spread)
        axes[0].axhline(1.0, color='k', ls=':', label='s = 1 (declared strength)')
        axes[0].set_xlabel('root z (mm)')
        axes[0].set_ylabel('s (gen_axis)')
        axes[0].set_title('stage 1: height x footprint')
        axes[0].legend(fontsize=8)
        pts = [e for e in stage2
               if e.get('s_gen_axis') and e['s_gen_axis'].get('threshold_midpoint') is not None]
        axes[1].plot([e['yaw_deg'] for e in pts], [e['s_gen_axis']['threshold_midpoint'] for e in pts], 's-',
                     color='#3060b0')
        axes[1].axhline(1.0, color='k', ls=':')
        for e in stage2:
            if not e['contract']:
                axes[1].plot([e['yaw_deg']], [1.0], 'rx', ms=8)
        axes[1].set_xlabel('uniform coxa-yaw offset (deg)')
        axes[1].set_ylabel('s (gen_axis)')
        axes[1].set_title('stage 2: yaw refinement at the stage-1 best')
        allpts = [e for e in stage1
                  if e.get('s_gen_axis') and e['s_gen_axis'].get('threshold_midpoint') is not None] + pts
        axes[2].hist([e['s_gen_axis']['threshold_midpoint'] for e in allpts], bins=8, color='#c08030')
        axes[2].axvline(1.0, color='k', ls=':', label='s = 1')
        if winners:
            axes[2].axvline(min(e['s_gen_axis']['threshold_midpoint'] for e in allpts), color='g', ls='--',
                            label='family best %.4f' % min(e['s_gen_axis']['threshold_midpoint'] for e in allpts))
        axes[2].set_xlabel('s (gen_axis) over the whole search')
        axes[2].set_ylabel('# poses')
        axes[2].set_title('how far the family gets')
        axes[2].legend(fontsize=8)
        fig.suptitle('Declared-family search for s=1 under gen_axis - manual research, NOT standing')
        fig.tight_layout()
        fig.savefig(out / 'seated-feasibility-search.png', dpi=130)
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
        print('stage1 best:', report['stage1_best'])
        print('stage2 best:', report['stage2_best'])
        print('final:', json.dumps(report.get('final_pose', {}).get('answer', ''), ensure_ascii=False))
        print('gate:', report.get('gate_run'))
    else:
        print(report.get('error', ''))
    return 0 if report['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
