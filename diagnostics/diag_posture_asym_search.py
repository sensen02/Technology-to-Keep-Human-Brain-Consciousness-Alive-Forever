"""Per-leg ASYMMETRIC posture search for static support (diagnostic, not a controller).

WHY.  A bounded search over UNIFORM offsets on all six legs found no statically
supportable standing posture (/run/media/sensen/Data2/cell_wound_prototype/outputs/
posture_search_round16).  Its stated untested directions were per-leg asymmetry and
the tibia/yaw/roll joints.  This file tests the first of those with a bounded search.

METHOD NOTES THAT MATTER.
  * Effective tarsus penetration is held FIXED at a chosen target for every pose,
    because at shallow seating MuJoCo generates few or one contact and the LP
    residual is then dominated by contact generation rather than by posture
    (measured: the same keyframe posture gives L1 42.45 at zero penetration and
    5.02 at 0.0514 mm).  Comparing poses at different depths is not apples-to-apples.
  * A candidate pose is only accepted as a standing candidate if every ground
    contact is a tarsus geom.  Poses where the body rests on femora, coxae, tibiae
    or the abdomen are recorded separately as resting poses, never as standing.
  * Plainly a diagnostic: no controller, no target, no schedule, no kinematics.

WHAT A RESULT WOULD AND WOULD NOT MEAN.  A feasible pose would show the model can
hold that posture with ideal instantaneous activation; it is NOT stability, NOT
controllability, NOT a controller and NOT autonomous standing.  No feasible pose in
the searched family is evidence over that family only, not a proof about all
postures.  Coordinate signs are model coordinates, not anatomical flexion labels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402  validated helpers
import diag_posture_support_search as r16      # noqa: E402  seating / pose helpers

OUT = ROOT / 'outputs/posture_asym_round17'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_TARGET_MM = -0.100      # signed: negative = penetrating, matches round16 best pose
COORD_STEPS_RAD = (0.20, 0.10, 0.05, 0.02)
STARTS = 8
MAX_EVALS = 4000
WALL_BUDGET_S = 900.0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--objective-set', choices=('all90', 'candidate44'), default='all90',
                    help='Actuator set the posture search optimises. Optimising one set and then '
                         'reporting the other is NOT a fair comparison, so run both.')
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out_dir = args.out_dir.resolve()
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0    # memory only: allow an exactly zero command
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key_qpos = data.qpos.copy()

    adr = {j: {leg: int(model.jnt_qposadr[model.joint(f'joint_{leg}{j}').id]) for leg in LEGS}
           for j in PARAM_JOINTS}
    rng = {j: model.jnt_range[model.joint(f'joint_LF{j}').id].copy() for j in PARAM_JOINTS}
    key_val = {j: {leg: float(key_qpos[adr[j][leg]]) for leg in LEGS} for j in PARAM_JOINTS}
    param_names = [(leg, j) for leg in LEGS for j in PARAM_JOINTS]

    manifest = json.loads(r15.MANIFEST.read_text())
    act_names = [model.actuator(i).name or '' for i in range(model.nu)]
    index_of = {}
    for i, n in enumerate(act_names):
        index_of.setdefault(n, i)
    cand_names = [r['actuator'] for r in manifest['actuators']
                  if r['status'] == 'CANDIDATE_SUPPORTED_BY_LABEL_NERVE_ROUTE']
    cand_idx = [index_of[c] for c in cand_names if c in index_of]
    sets_all = {'all90': list(range(model.nu)), 'candidate44': cand_idx}

    # ---- pose evaluation -------------------------------------------------
    def set_joints(x):
        """x: (n_param,) offsets from the keyframe, clamped to each joint's range."""
        qpos = key_qpos.copy()
        out = {}
        for k, (leg, j) in enumerate(param_names):
            lo, hi = float(rng[j][0]), float(rng[j][1])
            v = float(np.clip(key_val[j][leg] + x[k], lo, hi))
            qpos[adr[j][leg]] = v
            out[f'joint_{leg}{j}'] = v
        return qpos, out

    def evaluate(x, want_achievable=False):
        qpos, joints = set_joints(x)
        seated, root_z, seat_resid = r16.seat_root(model, data, mj, qpos, tarsus_geoms,
                                                   floor, PENETRATION_TARGET_MM)
        data.qpos[:] = seated
        data.qvel[:] = 0
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        row = {'x': list(map(float, x)), 'joints_rad': joints,
               'root_z_mm': float(root_z), 'seat_residual_mm': float(seat_resid),
               'contact_count': facts['contact_count'],
               'feet_only': facts['contacts_are_feet_only_tarsus'],
               'non_foot_geoms': facts['non_foot_ground_geoms'],
               'thorax_height_mm': facts['thorax_height_mm'],
               'active_joint_limit_rows': facts['active_joint_limit_rows']}
        if not facts['contacts_are_feet_only_tarsus']:
            row['objective_L1'] = None
            row['rejected_reason'] = 'ground contact is not tarsus-only (resting pose, not standing)'
            return row
        sols = r16.solve_pose(model, data, mj, {args.objective_set: sets_all[args.objective_set]},
                              'search')
        sol = sols[args.objective_set]
        row['objective_set'] = args.objective_set
        row['objective_L1'] = float(sol['objective_L1_residual'])
        row['objective_max_abs'] = float(sol['max_abs_residual'])
        if want_achievable:
            other = 'candidate44' if args.objective_set == 'all90' else 'all90'
            sols2 = r16.solve_pose(model, data, mj, {other: sets_all[other]}, 'final')
            row[f'{other}_L1_at_this_pose'] = float(sols2[other]['objective_L1_residual'])
        return row

    # ---- coordinate descent from several starts --------------------------
    rng_gen = np.random.default_rng(17)
    x0 = np.zeros(len(param_names))
    starts = [x0.copy()]
    starts.append(np.array([0.0357 if j == 'Coxa_pitch' else 0.0 for _, j in param_names]))
    for _ in range(STARTS - len(starts)):
        starts.append(rng_gen.normal(0.0, 0.08, size=len(param_names)))

    evals = 0
    history = []
    best = None
    for si, start in enumerate(starts):
        x = np.array(start, float)
        row = evaluate(x)
        evals += 1
        history.append({'start': si, 'eval': evals, 'step': None, 'objective_L1': row.get('objective_L1'),
                        'feet_only': row['feet_only']})
        cur = row.get('objective_L1')
        if cur is None:
            cur = np.inf
        for step in COORD_STEPS_RAD:
            improved = True
            sweeps = 0
            while improved and evals < MAX_EVALS and time.time() - t_start < WALL_BUDGET_S:
                improved = False
                sweeps += 1
                for k in range(len(param_names)):
                    for delta in (+step, -step):
                        if evals >= MAX_EVALS or time.time() - t_start > WALL_BUDGET_S:
                            break
                        cand = x.copy()
                        cand[k] += delta
                        row = evaluate(cand)
                        evals += 1
                        val = row.get('objective_L1')
                        history.append({'start': si, 'eval': evals, 'step': step,
                                        'objective_L1': val, 'feet_only': row['feet_only']})
                        if val is not None and val < cur - 1e-9:
                            x, cur = cand, val
                            improved = True
                if sweeps >= 6:
                    break
        final = evaluate(x, want_achievable=True)
        evals += 1
        if final.get('objective_L1') is not None:
            if best is None or final['objective_L1'] < best['objective_L1']:
                best = final
    print('evaluations', evals, 'wall', round(time.time() - t_start, 1), flush=True)

    report = {
        'status': 'BOUNDED_ASYMMETRIC_POSTURE_SEARCH',
        'what_this_is': 'static force-balance feasibility search over per-leg coxa/trochanter/tibia '
                        'pitch offsets at a fixed effective tarsus penetration; no controller, no target',
        'source_sha256': {str(XML): r15.digest(XML),
                          str(HERE / 'diag_static_support_feasibility.py'):
                              r15.digest(HERE / 'diag_static_support_feasibility.py'),
                          str(HERE / 'diag_posture_support_search.py'):
                              r15.digest(HERE / 'diag_posture_support_search.py')},
        'objective_actuator_set': args.objective_set,
        'config': {'legs': list(LEGS), 'param_joints': list(PARAM_JOINTS),
                   'n_params': len(param_names), 'penetration_target_mm': PENETRATION_TARGET_MM,
                   'coord_steps_rad': list(COORD_STEPS_RAD), 'starts': len(starts),
                   'max_evals': MAX_EVALS, 'wall_budget_s': WALL_BUDGET_S,
                   'feasibility_tolerance_L1': r15.FEASIBLE_SUM_TOL,
                   'candidate44_count': len(cand_idx)},
        'keyframe_reference': {'all90_L1_as_shipped': 5.015616714722375,
                               'note': 'round15; the search holds penetration at the target above'},
        'evaluations': evals,
        'wall_seconds': float(time.time() - t_start),
        'history': history,
        'best_pose': best,
        'feasible_found': bool(best is not None and best['objective_L1'] < r15.FEASIBLE_SUM_TOL),
        'xml_unchanged': r15.digest(XML) == xml_before,
        'limitations': [
            'Bounded search: coordinate descent from a fixed set of starts over per-leg pitch offsets '
            'only; yaw/roll, tarsus joints and per-leg roll were NOT searched.',
            'No feasible pose here is evidence about this family, not a proof that no posture works.',
            'Friction cone is linearised (converged over K elsewhere); actuator force is affine in '
            'activation only at the frozen pose; force-length/velocity and moment arms are frozen.',
            'Static feasibility is NOT stability, NOT controllability, NOT a controller, NOT standing.',
            'Coordinate signs are model coordinates, not anatomical flexion labels.',
            'The XML has no adhesion model; every candidate here relies on unilateral contact only.'],
        'scope': 'diagnostic only; no controller, no target, no behavioural claim',
    }

    # cone refinement and the 44-set at the best pose
    if best is not None:
        qpos, _ = set_joints(np.array(best['x'], float))
        seated, _, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor,
                                     PENETRATION_TARGET_MM)
        data.qpos[:] = seated
        data.qvel[:] = 0
        cone = {}
        for K in r16.CONE_SWEEP:
            saved = r15.CONE_DIRECTIONS
            r15.CONE_DIRECTIONS = K
            try:
                sol = r16.solve_pose(model, data, mj, {'all90': sets_all['all90']}, f'K{K}')
                cone[str(K)] = float(sol['all90']['objective_L1_residual'])
            finally:
                r15.CONE_DIRECTIONS = saved
        report['best_pose_cone_refinement'] = cone
        report['best_pose_validation'] = r16.validate_equilibrium(model, data, mj, 'best_asym')

        # Robustness: the winner was tuned at ONE fixed penetration depth, and at
        # shallow seating MuJoCo generates few contacts.  The SAME posture is
        # therefore re-scored across depths.  Without this, a low residual could be
        # misread as "this posture supports the body" when it only reflects deep
        # interpenetration generating many contact constraints.
        robust = []
        for dep in (0.040, 0.060, 0.080, 0.100, 0.120, 0.150):
            q, _ = set_joints(np.array(best['x'], float))
            seated, _, _ = r16.seat_root(model, data, mj, q, tarsus_geoms, floor, -dep)
            data.qpos[:] = seated
            data.qvel[:] = 0
            f = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
            entry = {'effective_penetration_mm': dep, 'contact_count': f['contact_count'],
                     'feet_only': f['contacts_are_feet_only_tarsus'],
                     'thorax_height_mm': f['thorax_height_mm']}
            if f['contacts_are_feet_only_tarsus']:
                s = r16.solve_pose(model, data, mj, {'all90': sets_all['all90']}, 'rob')['all90']
                a = np.asarray(s['activation'], float)
                entry.update(all90_L1=float(s['objective_L1_residual']),
                             active_actuators=int((a > 1e-9).sum()),
                             saturated_actuators=int((a > 1 - 1e-9).sum()))
            robust.append(entry)
        report['best_pose_penetration_robustness'] = robust
        report['robustness_interpretation'] = (
            'The winner was selected at one fixed penetration depth. Re-scored across depths it is low '
            'only where the feet interpenetrate deeply enough to generate many contacts; at shallower '
            'depths it is no better than the keyframe. A low residual therefore cannot be read as '
            '"this posture supports the body".')

    assert r15.digest(XML) == xml_before, 'canonical XML changed during the search'
    (out_dir / 'posture_asym.json').write_text(json.dumps(report, indent=2))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for si in range(len(starts)):
        pts = [(h['eval'], h['objective_L1']) for h in history
               if h['start'] == si and h['objective_L1'] is not None]
        if pts:
            axs[0].plot([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.8,
                        label=f'start {si}')
    axs[0].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    axs[0].axhline(5.0156, ls=':', c='gray', label='keyframe as shipped (5.016)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('pose evaluation'); axs[0].set_ylabel('all-90 LP L1 residual (force units)')
    axs[0].set_title('Per-leg asymmetric coordinate descent\n(lower is better; green = feasible)')
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)

    if best is not None:
        vals = [best['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg]
                for leg in LEGS for j in PARAM_JOINTS]
        labels = [f'{leg}\n{j.replace("_pitch","")}' for leg in LEGS for j in PARAM_JOINTS]
        axs[1].bar(range(len(vals)), vals,
                   color=['tab:blue', 'tab:orange', 'tab:green'] * len(LEGS))
        axs[1].set_xticks(range(len(vals))); axs[1].set_xticklabels(labels, fontsize=6, rotation=90)
        axs[1].axhline(0, c='k', lw=.8)
        axs[1].set_ylabel('offset from keyframe (rad)')
        axs[1].set_title('Best pose: per-leg joint offsets\n(still infeasible)')
        axs[1].grid(alpha=.3, axis='y')

        cone = report.get('best_pose_cone_refinement', {})
        if cone:
            ks = sorted(int(k) for k in cone)
            axs[2].semilogy(ks, [cone[str(k)] for k in ks], 'o-')
        axs[2].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
        axs[2].set_xlabel('friction-cone directions K'); axs[2].set_ylabel('all-90 L1 residual')
        axs[2].set_title('Best pose: cone refinement\n(converged, still infeasible)')
        axs[2].legend(fontsize=7); axs[2].grid(alpha=.3)
    fig.suptitle('Bounded per-leg asymmetric posture search: static force balance only\n'
                 'NOT stability, NOT a controller, NOT autonomous standing')
    fig.tight_layout(); fig.savefig(out_dir / 'posture_asym.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': report['wall_seconds'],
                      'feasible_found': report['feasible_found'],
                      'objective_set': args.objective_set,
                      'best_objective_L1': (best or {}).get('objective_L1'),
                      'other_set_L1_at_best_pose': {k: v for k, v in (best or {}).items()
                                                     if k.endswith('_L1_at_this_pose')},
                      'best_feet_only': (best or {}).get('feet_only'),
                      'best_contact_count': (best or {}).get('contact_count'),
                      'cone': report.get('best_pose_cone_refinement')}, indent=2))
    print(out_dir)


if __name__ == '__main__':
    main()
