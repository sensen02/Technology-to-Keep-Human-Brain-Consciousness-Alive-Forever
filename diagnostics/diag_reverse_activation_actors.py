"""Is the a/b actuator split the beam that blocks exact stance balance?

Round26 found a tarsus-only upright stance whose exact balance needs ONE actuator
(`LHF_trochanter_flexor_a`) to reach -5.1% activation.  Round27 showed that every pose in
that family is blocked on the ACTIVATION BOX rather than on adhesion, and that the geometric
evidence about a/b cannot settle whether those pairs are fibre clusters of one muscle.

This file answers the operational question directly, over a SAMPLE of the stance family:
  (1) with ideal sign-free contacts and all 90 actuators inside [0,1], what is the residual
      floor, and WHICH actuators would have to leave the box (aggregated over poses)?
  (2) with the twelve a/b actuators HELD AT ZERO - i.e. the mapping state this project is
      actually in, since none of them has an authorised motor pool - is exact balance
      reachable with ideal contacts?  If it becomes reachable, the a/b actuators are the
      blocking beam; if it stays blocked, the obstruction is elsewhere and is named.
  (3) the same two questions for the 45 strictly mapped actuators only.

Diagnostic only: fixed-pose force balance.  Feasibility is NOT stability, NOT a controller
and NOT walking; nothing here edits a model or authorises runtime use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

OUT = ROOT / 'outputs/reverse_activation_actors_round28'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050
SEED = ROOT / 'outputs/stance_with_adhesion_round26/stance_with_adhesion.json'
OVERLAY = ROOT / 'outputs/motor_overlay_v888/motor_overlay_v888.json'
N_POSES = 30
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--poses', type=int, default=N_POSES)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    overlay = json.loads(OVERLAY.read_text())
    ab_names = sorted(overlay['pair_level_cluster_only'])
    exact_names = sorted(overlay['exact_muscle_name'])
    manifest = json.loads((ROOT / 'outputs/muscle_anatomy_resolution/resolution.json').read_text())
    strict44_names = [r['actuator'] for r in manifest['actuators'] if r['candidates']]
    ab_idx = [names.index(n) for n in ab_names if n in names]
    strict_idx = sorted(set([names.index(n) for n in strict44_names if n in names]
                            + [names.index(n) for n in exact_names if n in names]))
    written_idx = [i for i in range(model.nu) if i not in set(strict_idx) | set(ab_idx)]

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key_qpos = data.qpos.copy()
    adr = {j: {leg: int(model.jnt_qposadr[model.joint(f'joint_{leg}{j}').id]) for leg in LEGS}
           for j in PARAM_JOINTS}
    rng = {j: model.jnt_range[model.joint(f'joint_LF{j}').id].copy() for j in PARAM_JOINTS}
    key_val = {j: {leg: float(key_qpos[adr[j][leg]]) for leg in LEGS} for j in PARAM_JOINTS}
    params = [(leg, j) for leg in LEGS for j in PARAM_JOINTS]
    seed = json.loads(SEED.read_text())['best']
    x_seed = np.array([seed['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg]
                       for leg, j in params])

    gen = np.random.default_rng(3)
    poses = [x_seed] + [x_seed + gen.normal(0, 0.04, size=len(params))
                        for _ in range(args.poses - 1)]

    def setup(x):
        qpos = key_qpos.copy()
        for k, (leg, j) in enumerate(params):
            lo, hi = float(rng[j][0]), float(rng[j][1])
            qpos[adr[j][leg]] = float(np.clip(key_val[j][leg] + x[k], lo, hi))
        seated, _, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, PENETRATION_MM)
        data.qpos[:] = seated; data.qvel[:] = 0
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        q = np.asarray(data.qpos[3:7], float)
        up_z = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        if not (facts['contacts_are_feet_only_tarsus'] and up_z >= 0.9):
            return None
        R, base = r15.actuator_moment_matrix(model, data, mj)
        target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
        terms = r15.contact_terms(model, data, mj)
        lims = r15.limit_terms(model, data, mj)
        lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims
                    else np.zeros((model.nv, 0)))
        free3 = np.hstack([t['J'].T for t in terms])
        return {'R': R, 'target': target, 'free3': free3, 'lim_cols': lim_cols,
                'facts': facts, 'up_z': up_z}

    def floor_with_free_activation(ctx, allowed_idx):
        """Minimal total activation excursion outside [0,1] for exact balance, ideal contacts."""
        nv, nu = model.nv, model.nu
        R = ctx['R'][:, allowed_idx]
        m = R.shape[1]
        A = np.hstack([R, R, -R, ctx['free3'], ctx['lim_cols']])
        c = np.concatenate([np.zeros(m), np.ones(2 * m),
                            np.zeros(ctx['free3'].shape[1]), np.zeros(ctx['lim_cols'].shape[1])])
        bounds = ([(0, 1)] * m + [(0, None)] * (2 * m)
                  + [(None, None)] * ctx['free3'].shape[1]
                  + [(0, None)] * ctx['lim_cols'].shape[1])
        r = linprog(c, A_eq=A, b_eq=ctx['target'], bounds=bounds, method='highs', options=LP_OPTS)
        if r.x is None:
            return {'reachable': False, 'message': r.message}
        z = r.x
        pos, neg = z[m:2 * m], z[2 * m:3 * m]
        outside = [(names[allowed_idx[i]], float(pos[i] + neg[i])) for i in range(m)
                   if pos[i] + neg[i] > 1e-9]
        return {'reachable': True, 'min_total_excursion': float(r.fun),
                'n_outside_box': len(outside), 'outside_box': outside}

    rows = []
    for k, x in enumerate(poses):
        ctx = setup(x)
        if ctx is None:
            rows.append({'pose': k, 'skipped': 'not tarsus-only/upright after perturbation'})
            continue
        all_idx = list(range(model.nu))
        strict_only = strict_idx
        no_ab = [i for i in all_idx if i not in set(ab_idx)]
        r_all = floor_with_free_activation(ctx, all_idx)
        r_noab = floor_with_free_activation(ctx, no_ab)
        r_strict = floor_with_free_activation(ctx, strict_only)
        rows.append({'pose': k, 'up_z': ctx['up_z'], 'n_contacts': ctx['facts']['contact_count'],
                     'all90': r_all, 'without_ab': r_noab, 'strict_with_written_off': r_strict})

    valid = [r for r in rows if 'all90' in r]
    agg = {}
    for key in ('all90', 'without_ab', 'strict_with_written_off'):
        reach = [r[key].get('reachable') for r in valid if r.get(key)]
        exc = [r[key]['min_total_excursion'] for r in valid
               if r.get(key) and r[key].get('reachable')]
        agg[key] = {'poses': len(exc),
                    'exact_balance_reachable_in_all': bool(reach and all(reach)),
                    'median_min_excursion': float(np.median(exc)) if exc else None,
                    'max_min_excursion': float(np.max(exc)) if exc else None}
    counts = {}
    for r in valid:
        for nm, _ in r['all90'].get('outside_box', []):
            counts[nm] = counts.get(nm, 0) + 1
    ab_counts = {nm: c for nm, c in counts.items() if nm in set(ab_names)}

    report = {
        'status': 'REVERSE_ACTIVATION_ACTOR_CENSUS',
        'question': 'which actuators block exact stance balance, and is the twelve-actuator a/b set the '
                    'blocking beam?',
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED),
                          str(OVERLAY): r15.digest(OVERLAY)},
        'setup': {'poses_requested': args.poses, 'poses_used': len(valid),
                  'penetration_mm': PENETRATION_MM,
                  'ab_actuators': ab_names, 'strict_mapped': [names[i] for i in strict_idx],
                  'written_off_never_mapped': [names[i] for i in written_idx]},
        'aggregate': agg,
        'actuator_outside_box_counts_over_poses': dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        'ab_actuator_counts': ab_counts,
        'per_pose': rows,
        'reading': {
            'all90': 'if exact balance is reachable for all poses with all 90 actuators free, the pose '
                     'family is fine and only the activation box is in the way',
            'without_ab': 'if it becomes unreachable or needs MORE excursion once the a/b actuators are '
                          'removed, the a/b actuators are load-bearing rather than the obstruction',
            'strict_with_written_off': 'the state this project is actually in: only the strictly mapped '
                                       'actuators driven, all others at zero'},
        'limitations': [
            'Fixed-pose static balance with idealized contacts: a bound, not a physical contact model.',
            'Actuator force affine in activation at the frozen pose; moment arms and force-length/velocity '
            'frozen.',
            'Excursion outside [0,1] is a diagnostic quantity and never a licence to drive activation out '
            'of range.',
            'No model file is modified; canonical asset stays byte-identical.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'reverse_activation_actors.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    keys = ['all90', 'without_ab', 'strict_with_written_off']
    exc = [[r[k]['min_total_excursion'] for r in valid if r.get(k) and r[k].get('reachable')]
           for k in keys]
    axs[0].boxplot(exc, tick_labels=['all 90', 'without a/b', 'strict only'])
    axs[0].set_ylabel('min total activation excursion for exact balance')
    axs[0].set_title('Does removing the a/b actuators make exact balance harder?')
    axs[0].grid(alpha=.3, axis='y')
    top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:14])
    axs[1].barh(range(len(top)), list(top.values()),
                color=['tab:orange' if n in set(ab_names) else 'tab:blue' for n in top])
    axs[1].set_yticks(range(len(top)))
    axs[1].set_yticklabels([n[:38] for n in top], fontsize=7)
    axs[1].set_xlabel('poses (of %d) where this actuator must leave [0,1]' % len(valid))
    axs[1].set_title('Actors blocking exact balance\n(orange = a/b actuator)')
    axs[1].grid(alpha=.3, axis='x')
    fig.suptitle('Reverse-activation census: diagnostic only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'reverse_activation_actors.png', dpi=110)

    print(json.dumps({'poses_used': len(valid), 'aggregate': agg,
                      'top_actors': dict(list(sorted(counts.items(), key=lambda kv: -kv[1])[:10])),
                      'ab_actors': ab_counts}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
