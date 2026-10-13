"""Audit the feasible activation pattern at the standable stance: which muscles does it need?

Round30 found a tarsus-only, upright stance that is statically supportable with activations
inside [0,1] and a per-leg adhesion of 1.112 body weights (below the engine default 1.64 and
inside the measured 1-2 bw/leg range).  This file asks the operational follow-up: WHICH muscles
does that feasible solution actually use, and are they among the ones this project has a
motor-neuron mapping for?

For the stance it reports
  * the activation vector that minimises per-leg adhesion subject to exact balance under the
    friction cone, and the same with the budget fixed at the engine default;
  * the active set (activation above a stated tolerance) with each actuator's mapping status
    (strictly mapped / pair-level only / unresolved);
  * the residual when only the strictly mapped actuators may be used - if the stance becomes
    unsupportable then the missing motor pools are the bottleneck for holding a stance, and if
    it stays supportable the mapping gap is not what blocks standing.

Diagnostic only: this is a fixed-pose static solution, not a controller, and nothing here is
imported into the running loop.
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

OUT = ROOT / 'outputs/feasible_activation_audit_round31'
XML = r15.XML
LEGS = r16.LEGS
STANCE = ROOT / 'outputs/stance_joint_feasible_round30/stance_for_runner.json'
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
OVERLAY = ROOT / 'outputs/motor_overlay_v888/motor_overlay_v888.json'
ADHESION_BW = 1.64
PENETRATION_MM = -0.050
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--active-tol', type=float, default=1e-3)
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
    body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    stance = json.loads(STANCE.read_text())
    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    for nm, v in stance['joints_rad'].items():
        data.qpos[int(model.jnt_qposadr[model.joint(nm).id])] = float(v)
    data.qpos[2] = float(stance['root_z_mm'])
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)

    manifest = json.loads(MANIFEST.read_text())
    strict_names = {r['actuator'] for r in manifest['actuators'] if r['candidates']}
    overlay = json.loads(OVERLAY.read_text())
    exact = set(overlay['exact_muscle_name']); pair = set(overlay['pair_level_cluster_only'])

    def status_of(n):
        if n in strict_names:
            return 'strictly_mapped'
        if n in exact:
            return 'v888_exact_extra'
        if n in pair:
            return 'pair_level_only'
        return 'unresolved'

    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims
                else np.zeros((model.nv, 0)))
    B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
    nu, nv = model.nu, model.nv
    n_lam = sum(b.shape[1] for b in B)
    n_p = len(terms)
    nlim = lim_cols.shape[1]
    leg_of = []
    for t in terms:
        lg = None
        for cand in (t['geom2'], t['geom1']):
            if cand[:2] in LEGS:
                lg = cand[:2]; break
        leg_of.append(lg)
    present = sorted({l for l in leg_of if l})

    def solve(act_idx, adhesion_bw=ADHESION_BW, exact=True):
        """Minimise per-leg adhesion over the allowed actuators; optionally require exact balance."""
        cols, bnds = [R[:, act_idx]], [(0, 1)] * len(act_idx)
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b); bnds += [(0, None)] * b.shape[1]
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bnds += [(0, None)] * n_p + [(0, None)] * nlim
        A = np.hstack(cols + [lim_cols])
        A_ub = np.zeros((len(present) + n_p, A.shape[1]))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    A_ub[i, len(act_idx) + n_lam + k] = 1.0
        for k in range(n_p):
            A_ub[len(present) + k, len(act_idx) + n_lam + k] = 1.0
        b_ub = np.concatenate([np.full(len(present), adhesion_bw * body_weight),
                               np.full(n_p, adhesion_bw * body_weight * 10)])
        if exact:
            # minimise per-leg adhesion subject to exact balance
            A_eq = np.hstack([A, np.zeros((nv, 1))])
            c = np.zeros(A.shape[1] + 1); c[-1] = 1.0
            rows = np.zeros((len(present), A.shape[1] + 1))
            for i, lg in enumerate(present):
                for k, l2 in enumerate(leg_of):
                    if l2 == lg:
                        rows[i, len(act_idx) + n_lam + k] = 1.0
                rows[i, -1] = -1.0
            r = linprog(c, A_eq=A_eq, b_eq=target, A_ub=rows, b_ub=np.zeros(len(present)),
                        bounds=bnds + [(0, None)], method='highs', options=LP_OPTS)
            if r.x is None:
                return {'reachable': False, 'message': r.message}
            a = r.x[:len(act_idx)]
            legs_used = [float(r.x[len(act_idx) + n_lam + k]) / body_weight for k in range(n_p)]
            return {'reachable': True, 'min_adhesion_bw': float(r.x[-1]) / body_weight,
                    'activation': a.tolist(), 'contact_usage_bw': legs_used}
        # residual variant: allow slack, report the L1 residual
        n = A.shape[1]
        M = np.hstack([A, np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(n), np.ones(2 * nv)])
        A_ub = np.hstack([A_ub, np.zeros((A_ub.shape[0], 2 * nv))])
        bnds2 = list(bnds) + [(0, None)] * (2 * nv)
        r = linprog(c, A_eq=M, b_eq=target, A_ub=A_ub, b_ub=b_ub, bounds=bnds2,
                    method='highs', options=LP_OPTS)
        if r.x is None:
            return {'reachable': False, 'message': r.message}
        return {'reachable': True, 'L1_residual': float(r.fun),
                'activation': r.x[:len(act_idx)].tolist()}

    all_idx = list(range(nu))
    strict_idx = [i for i, n in enumerate(names) if status_of(n) == 'strictly_mapped']
    strict_plus_exact = [i for i, n in enumerate(names) if status_of(n) in ('strictly_mapped',
                                                                           'v888_exact_extra')]
    primary = solve(all_idx, ADHESION_BW, exact=True)
    fixed = solve(all_idx, ADHESION_BW, exact=False)
    only_strict = solve(strict_idx, ADHESION_BW * 10, exact=True)
    only_strict_plus = solve(strict_plus_exact, ADHESION_BW * 10, exact=True)

    active = []
    if primary.get('reachable'):
        a = np.asarray(primary['activation'], float)
        for i in np.argsort(-a):
            if a[i] <= args.active_tol:
                break
            active.append({'actuator': names[i], 'activation': float(a[i]),
                           'mapping_status': status_of(names[i])})
    by_status = {}
    for row in active:
        by_status[row['mapping_status']] = by_status.get(row['mapping_status'], 0) + 1
    total_activation = float(sum(r['activation'] for r in active))

    report = {
        'status': 'FEASIBLE_ACTIVATION_AUDIT',
        'stance': {'file': str(STANCE), 'joints': len(stance['joints_rad']),
                   'root_z_mm': stance['root_z_mm'],
                   'tarsus_only': facts['contacts_are_feet_only_tarsus'],
                   'n_contacts': facts['contact_count'],
                   'thorax_height_mm': facts['thorax_height_mm']},
        'source_sha256': {str(XML): r15.digest(XML), str(STANCE): r15.digest(STANCE),
                          str(MANIFEST): r15.digest(MANIFEST), str(OVERLAY): r15.digest(OVERLAY)},
        'adhesion_budget_bw_per_leg': ADHESION_BW,
        'solution_with_all_90': {k: v for k, v in primary.items() if k != 'activation'},
        'active_set': active, 'active_set_by_mapping_status': by_status,
        'total_active_activation': total_activation,
        'with_only_strictly_mapped': {k: v for k, v in only_strict.items() if k != 'activation'},
        'with_strict_plus_v888_exact': {k: v for k, v in only_strict_plus.items()
                                        if k != 'activation'},
        'reading': {
            'active_set_mapped': 'if the feasible solution uses mostly strictly mapped muscles, the '
                                 'missing motor pools are NOT what blocks holding a stance',
            'strict_only': 'if the stance stays supportable using only the strictly mapped actuators, the '
                           'mapping gap is not the bottleneck either'},
        'limitations': [
            'Fixed-pose static solution under a linearised cone and a bounded tensile normal force; NOT '
            'stability, NOT a controller, NOT walking.',
            'The activation vector is one feasible point, not the pattern a fly would use, and it is not '
            'imported into the running loop.',
            'Mapping status is this project annotations, not validation.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'feasible_activation_audit.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    if active:
        labs = [r['actuator'] for r in active][:25]
        vals = [r['activation'] for r in active][:25]
        cols = ['tab:green' if r['mapping_status'] in ('strictly_mapped', 'v888_exact_extra')
                else ('tab:orange' if r['mapping_status'] == 'pair_level_only' else 'tab:red')
                for r in active][:25]
        axs[0].barh(range(len(labs))[::-1], vals, color=cols)
        axs[0].set_yticks(range(len(labs))[::-1]); axs[0].set_yticklabels(labs, fontsize=6)
        axs[0].set_xlabel('activation in the feasible solution')
        axs[0].set_title('Muscles the feasible stance solution uses\n'
                         'green = mapped, orange = pair-level, red = unresolved')
        axs[0].grid(alpha=.3, axis='x')
    bars = ['all 90', 'strict + v888 exact', 'strict only']
    res = [primary.get('min_adhesion_bw'), only_strict_plus.get('min_adhesion_bw'),
           only_strict.get('min_adhesion_bw')]
    axs[1].bar(range(3), [np.nan if r is None else r for r in res],
               color=['tab:blue', 'tab:green', 'tab:orange'])
    axs[1].axhline(ADHESION_BW, ls='--', c='red', label='engine default 1.64 bw/leg')
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(bars, fontsize=8)
    axs[1].set_ylabel('min required per-leg adhesion (bw)')
    axs[1].set_title('Can the stance be held with fewer mapped actuators?')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('Feasible activation audit (static, diagnostic): NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'feasible_activation_audit.png', dpi=110)

    print(json.dumps({'solution': report['solution_with_all_90'],
                      'active_set_size': len(active), 'by_status': by_status,
                      'strict_only': report['with_only_strictly_mapped'],
                      'strict_plus_exact': report['with_strict_plus_v888_exact']}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
