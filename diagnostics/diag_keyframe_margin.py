"""How CLOSE is the standing keyframe to static force balance? (diagnostic, not a controller)

Round15/16/17 established that the keyframe cannot be statically supported and that
adhesion, posture and the bilateral mirror repair are not the cause.  This file asks
the sharper and more useful question: by how much does it miss, and in WHICH domain?

Three nested questions, all at the same frozen pose with MuJoCo's own contacts and
active joint-limit rows:
  (1) physical activation a in [0,1]                      -> minimal residual
  (2) ideal contact forces only, no muscles               -> minimal residual
  (3) ideal contact forces AND muscles allowed any real activation -> is exact
      balance reachable at all?
Then, since (3) shows exact balance IS reachable only outside the physical box, the
minimal total activation excursion beyond [0,1] needed for EXACT balance is computed,
together with the specific actuators that would have to leave the box.

A small excursion is not standing and not a licence to allow negative activation: it
localises the remaining obstruction.  Static feasibility is NOT stability, NOT
controllability, NOT a controller.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402  validated helpers

OUT = ROOT / 'outputs/keyframe_margin_round18'
XML = r15.XML
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}


def l1_min(A, bounds, nv, target):
    """Minimise the L1 generalized-force residual with slack columns."""
    n = A.shape[1]
    M = np.hstack([A, np.eye(nv), -np.eye(nv)])
    c = np.concatenate([np.zeros(n), np.ones(2 * nv)])
    r = linprog(c, A_eq=M, b_eq=target, bounds=list(bounds) + [(0, None)] * (2 * nv),
                method='highs', options=LP_OPTS)
    if r.x is None:
        return {'usable': False, 'message': r.message}
    return {'usable': True, 'L1': float(r.fun),
            'max_abs': float(np.abs(r.x[n:n + nv] - r.x[n + nv:]).max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0     # memory only: an exactly zero command must be legal
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    nv, nc = model.nv, 3 * len(terms)
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    # sign-free 3-component contact force per contact: strictly more permissive than
    # any friction or adhesion model of a point contact, so it is a valid floor.
    free3 = np.hstack([t['J'].T for t in terms])
    lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims else np.zeros((nv, 0)))

    contact_free_bounds = [(None, None)] * nc
    limit_bounds = [(0, None)] * lim_cols.shape[1]

    q1 = l1_min(np.hstack([R, free3, lim_cols]),
                [(0, 1)] * model.nu + contact_free_bounds + limit_bounds, nv, target)
    q2 = l1_min(np.hstack([free3, lim_cols]),
                contact_free_bounds + limit_bounds, nv, target)
    q3 = l1_min(np.hstack([R, free3, lim_cols]),
                [(None, None)] * model.nu + contact_free_bounds + limit_bounds, nv, target)

    report = {
        'status': 'KEYFRAME_STATIC_MARGIN_MEASUREMENT',
        'what_this_is': 'how far the XML standing keyframe is from static force balance, and in '
                        'which domain; frozen pose, MuJoCo contacts and active joint-limit rows; '
                        'no controller, no target, no schedule',
        'source_sha256': {str(XML): r15.digest(XML)},
        'pose': {'thorax_height_model': float(data.xpos[model.body("Thorax").id][2]),
                 'n_contacts': int(data.ncon),
                 'contacts_feet_only': bool(data.ncon and all(
                     'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(c.geom2)) or '')
                     or 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(c.geom1)) or '')
                     for c in data.contact)),
                 'active_joint_limit_rows': len(lims),
                 'body_weight_force_units': float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))},
        'contact_model_used': 'three SIGN-FREE unbounded force components per contact, i.e. strictly '
                              'more permissive than any friction or adhesion model of a point contact',
        'q1_physical_activation_with_ideal_contacts': q1,
        'q2_no_muscle_with_ideal_contacts': q2,
        'q3_unbounded_activation_with_ideal_contacts': q3,
        'limitations': [
            'Static force balance at ONE frozen pose: NOT stability, NOT controllability, NOT a '
            'controller and NOT standing.',
            'Actuator force is affine in activation only at this frozen pose; force-length/velocity '
            'and moment arms are frozen.',
            'Ideal sign-free contact forces are not physical; they only bound what any contact model '
            'could deliver.',
            'A small excursion outside [0,1] localises the obstruction; it is NOT permission to drive '
            'activation outside its physical range.'],
        'scope': 'diagnostic only',
    }

    # Exact balance: minimise the activation excursion beyond [0,1] with NO slack columns,
    # so a positive answer means the equality genuinely cannot hold inside the box.
    A = np.hstack([R, R, -R, free3, lim_cols])
    c = np.concatenate([np.zeros(model.nu), np.ones(2 * model.nu),
                        np.zeros(nc), np.zeros(lim_cols.shape[1])])
    bounds = ([(0, 1)] * model.nu + [(0, None)] * (2 * model.nu)
              + [(None, None)] * nc + [(0, None)] * lim_cols.shape[1])
    r = linprog(c, A_eq=A, b_eq=target, bounds=bounds, method='highs', options=LP_OPTS)
    if r.x is None:
        report['exact_balance'] = {'reachable': False, 'message': r.message}
    else:
        z = r.x
        a_box = z[:model.nu]
        pos = z[model.nu:2 * model.nu]
        neg = z[2 * model.nu:3 * model.nu]
        a = a_box + pos - neg
        outside = [{'actuator': names[i], 'required_activation': float(a[i]),
                    'excursion': float(pos[i] + neg[i])}
                   for i in range(model.nu) if pos[i] + neg[i] > 1e-9]
        report['exact_balance'] = {
            'reachable': True,
            'min_total_activation_excursion': float(r.fun),
            'n_actuators_outside_box': len(outside),
            'n_needing_reverse_action': int(sum(1 for o in outside if o['required_activation'] < 0)),
            'n_needing_over_activation': int(sum(1 for o in outside if o['required_activation'] > 1)),
            'actuators_outside_box': sorted(outside, key=lambda o: -o['excursion']),
            'interpretation': 'Exact static balance at this pose is reachable only if the listed '
                              'actuators leave the physical activation range. Because activation '
                              'cannot be negative, the obstruction is their zero-activation '
                              '(passive) contribution being slightly too large in the wrong '
                              'direction, with no other actuator able to compensate inside [0,1].'}
    report['summary'] = {
        'physical_residual_L1': q1.get('L1'),
        'physical_residual_as_fraction_of_body_weight': (q1.get('L1') or 0)
            / (float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))) if q1.get('usable') else None,
        'ideal_contact_floor_L1': q3.get('L1'),
        'no_muscle_residual_L1': q2.get('L1'),
        'exact_balance_needs_excursion': report['exact_balance'].get('min_total_activation_excursion'),
    }

    assert r15.digest(XML) == xml_before, 'canonical XML changed during the measurement'
    (OUT / 'keyframe_margin.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labels = ['physical a in [0,1]\n(ideal contacts)', 'no muscles\n(ideal contacts)',
              'unbounded a\n(ideal contacts)']
    vals = [q1.get('L1'), q2.get('L1'), max(q3.get('L1') or 0, 1e-12)]
    ax[0].bar(range(3), vals, color=['tab:blue', 'tab:orange', 'tab:green'])
    ax[0].axhline(report['pose']['body_weight_force_units'], ls=':', c='gray', label='one body weight')
    ax[0].set_yscale('log')
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel('L1 generalized-force residual')
    ax[0].set_title('Where the keyframe shortfall lives\n(a sign-free contact model is used, so these are floors)')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')

    ob = report['exact_balance'].get('actuators_outside_box', [])
    if ob:
        ax[1].barh([o['actuator'] for o in ob][::-1], [o['required_activation'] for o in ob][::-1],
                   color='tab:red')
        ax[1].axvline(0, c='k', lw=.8); ax[1].axvline(1, c='k', ls='--', lw=.8)
        ax[1].set_xlabel('activation required for EXACT balance (physical range is [0,1])')
        ax[1].set_title('Actuators that must leave the physical range\nfor exact static balance')
        ax[1].grid(alpha=.3, axis='x')
    fig.suptitle('Keyframe static margin: diagnostic only, NOT stability, NOT a controller, NOT standing')
    fig.tight_layout(); fig.savefig(OUT / 'keyframe_margin.png', dpi=110)

    print(json.dumps(report['summary'], indent=2))
    print(json.dumps({'exact_balance_reachable': report['exact_balance'].get('reachable'),
                      'n_outside_box': report['exact_balance'].get('n_actuators_outside_box'),
                      'min_excursion': report['exact_balance'].get('min_total_activation_excursion'),
                      'actuators': report['exact_balance'].get('actuators_outside_box')}, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
