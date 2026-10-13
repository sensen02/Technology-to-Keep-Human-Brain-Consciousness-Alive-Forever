"""Targeted anatomical check of the ONE actuator that blocks exact static balance.

Round18 measured that at the XML standing keyframe, with ideal (sign-free,
unbounded) contact forces and MuJoCo's active joint-limit rows, exact static
balance is reachable only if ONE actuator leaves its physical range:
``RFC_sternal_posterior_rotator`` at activation -0.005809 (0.58% of its range),
while the other 89 stay inside [0,1].

This file asks the narrow, falsifiable follow-up questions:
  (1) Is that actuator's tendon geometry inconsistent with the axis-derived
      sagittal mirror relation that round17/18 established (right legs are
      TRANSLATIONS of the left legs, not mirrors)?
  (2) Which term is responsible - the incremental (activation) column, or the
      zero-activation passive contribution that sits in the LP target?
  (3) How large a change in each would be needed, and is that change within the
      size of the measured mirror inconsistency, or far beyond it?

This is a diagnostic: it does NOT edit the model, does NOT authorise driving
activation outside [0,1], and static balance is NOT stability, NOT a controller
and NOT standing.  A counterfactual that achieves balance is a hypothesis about a
physical defect, not a repair and not evidence of autonomy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402  validated helpers

OUT = ROOT / 'outputs/targeted_rotator_check_round19'
XML = r15.XML
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
ACT = 'RFC_sternal_posterior_rotator'
ACT_L = 'LFC_sternal_posterior_rotator'


def l1_min(A, bounds, nv, target):
    from scipy.optimize import linprog
    n = A.shape[1]
    M = np.hstack([A, np.eye(nv), -np.eye(nv)])
    c = np.concatenate([np.zeros(n), np.ones(2 * nv)])
    r = linprog(c, A_eq=M, b_eq=target, bounds=list(bounds) + [(0, None)] * (2 * nv),
                method='highs', options=LP_OPTS)
    if r.x is None:
        return {'usable': False, 'message': r.message}
    return {'usable': True, 'L1': float(r.fun)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)

    names = [model.actuator(i).name or '' for i in range(model.nu)]
    ia, ial = names.index(ACT), names.index(ACT_L)
    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    nv = model.nv
    free3 = np.hstack([t['J'].T for t in terms])
    lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims else np.zeros((nv, 0)))
    nc, nlim = free3.shape[1], lim_cols.shape[1]

    def dofs_of(leg):
        out = {}
        for j in ('yaw', 'pitch', 'roll'):
            jid = model.joint(f'joint_{leg}Coxa_{j}').id
            out[j] = int(model.jnt_dofadr[jid])
        return out

    dR, dL = dofs_of('RF'), dofs_of('LF')

    # Axis-derived mirror requirement (round17): at the zero-leg configuration each
    # leg's axes are yaw=+x, pitch=+y, roll=+z.  Mirroring about the sagittal plane
    # sends an axial vector a -> -S a with S = diag(1,-1,1) for a plane normal to y,
    # so a compact way to state the requirement is that a genuinely mirrored pair
    # has OPPOSITE incremental generalized force at yaw and roll and the SAME at
    # pitch.  Report both the raw and the mirror-compared columns.
    arm = {
        'RF_incremental_force_on_RFCoxa': {j: float(R[dR[j], ia]) for j in dR},
        'LF_incremental_force_on_LFCoxa': {j: float(R[dL[j], ial]) for j in dL},
    }
    arm['mirror_relation_per_joint'] = {}
    for j in ('yaw', 'pitch', 'roll'):
        a_rf, a_lf = arm['RF_incremental_force_on_RFCoxa'][j], arm['LF_incremental_force_on_LFCoxa'][j]
        expected = 'SAME_sign (pitch)' if j == 'pitch' else 'OPPOSITE_sign (yaw/roll)'
        ok = (a_rf * a_lf > 0) if j == 'pitch' else (a_rf * a_lf < 0)
        arm['mirror_relation_per_joint'][j] = {
            'RF': a_rf, 'LF': a_lf, 'required': expected,
            'satisfies_axis_derived_mirror': bool(ok)}

    # which term is responsible: solve the physical LP with each actuator forced to a=0
    base_bounds = [(0, 1)] * model.nu
    at_physical = l1_min(np.hstack([R, free3, lim_cols]),
                         base_bounds + [(None, None)] * nc + [(0, None)] * nlim, nv, target)

    # (2a) counterfactual A: mirror ONLY this actuator's incremental column
    R_a = R.copy()
    R_a[dR['yaw'], ia] *= -1.0
    R_a[dR['roll'], ia] *= -1.0
    cfA = l1_min(np.hstack([R_a, free3, lim_cols]),
                 base_bounds + [(None, None)] * nc + [(0, None)] * nlim, nv, target)

    # (2b) counterfactual B: fix only the zero-activation (passive) contribution of
    # this actuator in the target, keeping the incremental column unchanged.  The
    # needed change is found by least squares on the single scalar, so the reported
    # delta is the MINIMAL passive-baseline change that the numbers themselves imply.
    t_a = target.copy()
    best = None
    for scale in np.linspace(-0.2, 0.2, 81):
        t_try = target - scale * (R[:, ia] * 0)
        # passive removal is a translation of the target along -R[:,ia] only when the
        # muscle is at a=0; express it as removing a fraction of the a=0 force:
        t_try = target + scale * np.abs(base)
        r = l1_min(np.hstack([R, free3, lim_cols]),
                   base_bounds + [(None, None)] * nc + [(0, None)] * nlim, nv, t_try)
        if r.get('usable') and (best is None or r['L1'] < best[1]):
            best = (float(scale), r['L1'])
    cfB = {'best_fraction_of_abs_base_added': best[0], 'L1': best[1]} if best else None

    report = {
        'status': 'TARGETED_ROTATOR_CHECK',
        'what_this_is': 'narrow anatomical/mechanical check of the single actuator that round18 found '
                        'must leave its physical activation range for exact static balance; '
                        'diagnostic only, no model edit, no controller',
        'source_sha256': {str(XML): r15.digest(XML)},
        'actuator': ACT,
        'left_homologue': ACT_L,
        'moment_arms': arm,
        'physical_LP_L1_with_ideal_contacts': at_physical.get('L1'),
        'counterfactual_A_mirror_only_this_actuator_column': cfA,
        'counterfactual_B_passive_baseline_scan': cfB,
        'limitations': [
            'Frozen single pose; contact forces treated as sign-free and unbounded (a floor, not a '
            'physical contact model).',
            'Counterfactuals are hypotheses about a physical defect, NOT repairs and NOT permission to '
            'edit the model or to drive activation outside [0,1].',
            'Static balance is NOT stability, NOT controllability, NOT a controller and NOT standing.',
            'Model coordinate signs are not anatomical flexion labels.',
            'The axis-derived mirror relation is derived from joint axes, not from anatomical '
            'registration.'],
        'scope': 'diagnostic only',
    }

    assert r15.digest(XML) == xml_before, 'canonical XML changed'
    (OUT / 'targeted_rotator_check.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    js = ['yaw', 'pitch', 'roll']
    x = np.arange(3)
    axs[0].bar(x - .2, [arm['LF_incremental_force_on_LFCoxa'][j] for j in js], .4, label='LF (left)')
    axs[0].bar(x + .2, [arm['RF_incremental_force_on_RFCoxa'][j] for j in js], .4, label='RF (right)')
    axs[0].axhline(0, c='k', lw=.8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(js)
    axs[0].set_ylabel('incremental generalized force per unit activation')
    axs[0].set_title(f'{ACT}\naxis-derived mirror needs OPPOSITE sign at yaw/roll, SAME at pitch')
    axs[0].legend(); axs[0].grid(alpha=.3, axis='y')
    vals = [report['physical_LP_L1_with_ideal_contacts'],
            cfA.get('L1') if cfA.get('usable') else np.nan,
            cfB['L1'] if cfB else np.nan]
    axs[1].bar(range(3), vals, color=['tab:blue', 'tab:orange', 'tab:green'])
    axs[1].set_yscale('log')
    axs[1].set_xticks(range(3))
    axs[1].set_xticklabels(['physical\n(a in [0,1])', 'counterfactual A\n(mirror this column)',
                            'counterfactual B\n(passive scan)'], fontsize=8)
    axs[1].set_ylabel('L1 residual (ideal contacts)')
    axs[1].set_title('Would either counterfactual close the measured shortfall?')
    axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('Single-actuator targeted check: diagnostic only, NOT a repair, NOT standing')
    fig.tight_layout(); fig.savefig(OUT / 'targeted_rotator_check.png', dpi=110)

    print(json.dumps({'physical_L1': report['physical_LP_L1_with_ideal_contacts'],
                      'moment_arms': arm, 'cfA_L1': cfA.get('L1'),
                      'cfB': cfB}, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
