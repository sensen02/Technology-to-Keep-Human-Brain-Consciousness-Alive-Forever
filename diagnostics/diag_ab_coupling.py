"""Does coupling the a/b actuator pairs to one activation remove the reverse-activation blocker?

Round28's census found that the actuator blocking exact stance balance in EVERY sampled pose
is `LHF_trochanter_flexor_a`, with a handful of others close behind, and that removing the
twelve a/b actuators makes exact balance harder rather than easier - so the a/b set is
load-bearing, but a few individual actuators are in the way.

The model splits each of those muscles into an `_a` and a `_b` actuator.  If they are fibre
clusters of one muscle, the physically correct description is ONE activation driving both
clusters (with a capacity rule), not two independent activations.  This file measures what
that change would do to the blocker:

  independent  - the model as shipped: each actuator has its own activation in [0,1]
  coupled      - each a/b pair shares a single activation variable in [0,1], so the pair's
                 generalized force is the SUM of both columns times that one variable

For each pose the minimal total activation excursion outside [0,1] needed for exact balance
(with ideal sign-free contacts, so the answer bounds any contact model) is computed for both
descriptions, together with which actuators still need to leave the box.

This is a sensitivity test of a MODELLING CHOICE, not a repair and not a claim about the fly:
the shared-pool assumption still needs the upstream fibre-cluster definition to be
authoritative, and a reduced excursion is not walking.
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

OUT = ROOT / 'outputs/ab_coupling_round28'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050
SEED = ROOT / 'outputs/stance_with_adhesion_round26/stance_with_adhesion.json'
OVERLAY = ROOT / 'outputs/motor_overlay_v888/motor_overlay_v888.json'
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
# pairs to couple: the a/b trochanter flexor actuators, which is where the census put the blocker
PAIR_SUFFIX = ('_a', '_b')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--poses', type=int, default=12)
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

    # build the coupling groups: each trochanter flexor a/b pair becomes one variable
    groups, used = [], set()
    for n in ab_names:
        if n.endswith('_a'):
            b = n[:-2] + '_b'
            if b in names:
                groups.append((n, b))
                used.update({names.index(n), names.index(b)})
    free_idx = [i for i in range(model.nu) if i not in used]

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
    gen = np.random.default_rng(5)
    poses = [x_seed] + [x_seed + gen.normal(0, 0.03, size=len(params))
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
        return R, target, free3, lim_cols

    def excursion(R_eff, labels, target, free3, lim_cols):
        """Minimal total excursion outside [0,1] for exact balance; returns (value, outside list)."""
        m = R_eff.shape[1]
        A = np.hstack([R_eff, R_eff, -R_eff, free3, lim_cols])
        c = np.concatenate([np.zeros(m), np.ones(2 * m), np.zeros(free3.shape[1]),
                            np.zeros(lim_cols.shape[1])])
        bounds = ([(0, 1)] * m + [(0, None)] * (2 * m)
                  + [(None, None)] * free3.shape[1] + [(0, None)] * lim_cols.shape[1])
        r = linprog(c, A_eq=A, b_eq=target, bounds=bounds, method='highs', options=LP_OPTS)
        if r.x is None:
            return None, []
        z = r.x
        pos, neg = z[m:2 * m], z[2 * m:3 * m]
        outside = [(labels[i], float(pos[i] + neg[i])) for i in range(m)
                   if pos[i] + neg[i] > 1e-9]
        return float(r.fun), outside

    rows = []
    for k, x in enumerate(poses):
        ctx = setup(x)
        if ctx is None:
            rows.append({'pose': k, 'skipped': 'not tarsus-only/upright'})
            continue
        R, target, free3, lim_cols = ctx
        ind_val, ind_out = excursion(R, names, target, free3, lim_cols)
        cols, labels = [], []
        for n in sorted(used):
            cols.append(R[:, n].reshape(-1, 1)); labels.append(names[n])
        for a, b in groups:
            cols.append((R[:, names.index(a)] + R[:, names.index(b)]).reshape(-1, 1))
            labels.append(f'{a}|{b}')
        R_coupled = np.hstack(cols)
        cpl_val, cpl_out = excursion(R_coupled, labels, target, free3, lim_cols)
        # is the required generalized force inside the SPAN at all?  If not, the description
        # cannot represent exact balance before any activation bound is even considered.
        def span_residual(A):
            coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            return float(np.abs(A @ coef - target).max())
        span_ind = span_residual(np.hstack([R, free3, lim_cols]))
        span_cpl = span_residual(np.hstack([R_coupled, free3, lim_cols]))
        rank_ind = int(np.linalg.matrix_rank(np.hstack([R, free3, lim_cols])))
        rank_cpl = int(np.linalg.matrix_rank(np.hstack([R_coupled, free3, lim_cols])))
        rows.append({'pose': k, 'independent_excursion': ind_val,
                     'coupled_excursion': cpl_val,
                     'independent_outside': ind_out, 'coupled_outside': cpl_out,
                     'span_residual_independent': span_ind, 'span_residual_coupled': span_cpl,
                     'rank_independent': rank_ind, 'rank_coupled': rank_cpl,
                     'coupled_out_of_span': bool(span_cpl > 1e-6)})

    valid = [r for r in rows if r.get('independent_excursion') is not None
             and r.get('coupled_excursion') is not None]
    ind = np.array([r['independent_excursion'] for r in valid], float)
    cpl = np.array([r['coupled_excursion'] for r in valid], float)
    blocker_ind = sum(1 for r in valid
                      if any(n == 'LHF_trochanter_flexor_a' for n, _ in r['independent_outside']))
    blocker_cpl = sum(1 for r in valid
                      if any('LHF_trochanter_flexor_a' in n for n, _ in r['coupled_outside']))
    blocker_ind_zero = sum(1 for r in valid if r['independent_excursion'] <= 1e-9)
    blocker_cpl_zero = sum(1 for r in valid if r['coupled_excursion'] <= 1e-9)

    oos = sum(1 for r in rows if r.get('coupled_out_of_span'))
    all_rows = [r for r in rows if 'span_residual_coupled' in r]
    report = {
        'status': 'AB_COUPLING_TEST',
        'structural_finding': {
            'poses_where_the_coupled_description_cannot_represent_balance': oos,
            'poses_evaluated': len(all_rows),
            'meaning': 'coupling an a/b pair into ONE activation removes the DIFFERENTIAL mode between '
                       'the two columns. Where the required generalized force then falls outside the '
                       'reachable span, the coupled description cannot balance the body at all - before '
                       'any activation bound is considered. That is evidence that a and b are NOT '
                       'mechanically interchangeable, i.e. against the pure fibre-cluster reading.',
            'median_rank_independent': int(np.median([r['rank_independent'] for r in all_rows]))
            if all_rows else None,
            'median_rank_coupled': int(np.median([r['rank_coupled'] for r in all_rows]))
            if all_rows else None},
        'question': 'if the a/b pairs were one muscle with one activation, does the reverse-activation '
                    'blocker disappear?',
        'modeling_choice_under_test': {
            'independent': 'the shipped model: each a/b actuator has its own activation in [0,1]',
            'coupled': 'each a/b pair shares ONE activation in [0,1]; the pair generalized force is the '
                       'sum of both columns times that variable, i.e. the fibre-cluster reading',
            'coupled_pairs': [f'{a} + {b}' for a, b in groups],
            'actuators_left_independent': len(free_idx)},
        'poses': {'requested': args.poses, 'used': len(valid)},
        'result': {
            'independent_median_excursion': float(np.median(ind)) if ind.size else None,
            'coupled_median_excursion': float(np.median(cpl)) if cpl.size else None,
            'independent_zero_excursion_poses': blocker_ind_zero,
            'coupled_zero_excursion_poses': blocker_cpl_zero,
            'poses_where_the_LHF_trochanter_flexor_a_blocks_independent': blocker_ind,
            'poses_where_it_blocks_coupled': blocker_cpl},
        'per_pose': rows,
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED),
                          str(OVERLAY): r15.digest(OVERLAY)},
        'limitations': [
            'The coupled description is a MODELLING CHOICE that still needs the upstream fibre-cluster '
            'definition; this file only measures its effect.',
            'Ideal sign-free contacts bound any contact model; static feasibility is NOT stability, NOT a '
            'controller, NOT walking.',
            'Activation outside [0,1] is a diagnostic quantity, never a licence to drive out of range.',
            'No model file is modified.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'ab_coupling.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].hist(ind, bins=12, alpha=.6, label='independent (shipped)')
    axs[0].hist(cpl, bins=12, alpha=.6, label='coupled (one activation per pair)')
    axs[0].set_xlabel('min total activation excursion for exact balance')
    axs[0].set_ylabel('poses'); axs[0].legend(fontsize=8)
    axs[0].set_title('Does coupling the a/b pairs reduce the blocker?')
    axs[0].grid(alpha=.3)
    axs[1].bar(['zero-excursion\nposes: independent', 'zero-excursion\nposes: coupled'],
               [blocker_ind_zero, blocker_cpl_zero], color=['tab:blue', 'tab:orange'])
    axs[1].set_ylabel('poses (of %d)' % len(valid))
    axs[1].set_title('Poses where exact balance needs NO excursion')
    axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('a/b coupling sensitivity: modelling choice, NOT a repair, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'ab_coupling.png', dpi=110)

    print(json.dumps(report['result'], indent=2))
    print(out)


if __name__ == '__main__':
    main()
