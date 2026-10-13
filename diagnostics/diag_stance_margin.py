"""Why is the best tarsus-only stance still not supportable, and by how much?

Round26 found a posture family that is upright and tarsus-only with the engine's
documented adhesion, with the best LP residual 0.3084 force units (1.26% of body weight) -
far better than the 5.0156 of the keyframe without adhesion, but still not zero.

This file decomposes the remaining 0.3084 at THAT pose:
  * the FREE3 floor: with each contact allowed three unbounded sign-free force components
    (strictly more permissive than any friction or adhesion model), how much residual
    remains?  Whatever remains cannot be fixed by any contact model.
  * the adhesion sweep at that pose: how the residual falls as the per-leg tensile budget
    grows, so it is visible whether measured-range adhesion is the binding term.
  * the activation excursion: the smallest total amount by which actuators would have to
    leave [0,1] for EXACT balance, and which actuators those are.

Diagnostic only: no controller, no model edit, no parameter chosen to make a body stand.
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

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

XML = r15.XML
LEGS = r16.LEGS
BEST = ROOT / 'outputs/stance_with_adhesion_round26/stance_with_adhesion.json'
OUT = ROOT / 'outputs/stance_margin_round26'
DEGRADE_TOL = 1e-3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    best = json.loads(BEST.read_text())['best']
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

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    qpos = data.qpos.copy()
    for name, v in best['joints_rad'].items():
        qpos[model.joint(name).qposadr[0]] = v
    seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, -0.050)
    data.qpos[:] = seated
    data.qvel[:] = 0
    facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)

    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    nv, nu = model.nv, model.nu
    free3 = np.hstack([t['J'].T for t in terms])
    lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims else np.zeros((nv, 0)))
    nlim = lim_cols.shape[1]

    def l1(A, bounds):
        n = A.shape[1]
        M = np.hstack([A, np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(n), np.ones(2 * nv)])
        r = linprog(c, A_eq=M, b_eq=target, bounds=list(bounds) + [(0, None)] * (2 * nv),
                    method='highs', options={'presolve': True, 'time_limit': 60, 'maxiter': 20000})
        return (float(r.fun) if r.x is not None else None)

    floors = {
        'with_adhesion_1.64bw_per_leg': None,  # computed by the sweep below
        'ideal_signfree_contacts_with_activation_box':
            l1(np.hstack([R, free3, lim_cols]),
               [(0, 1)] * nu + [(None, None)] * free3.shape[1] + [(0, None)] * nlim),
        'ideal_signfree_contacts_unbounded_activation':
            l1(np.hstack([R, free3, lim_cols]),
               [(None, None)] * nu + [(None, None)] * free3.shape[1] + [(0, None)] * nlim),
        'no_muscle_ideal_contacts':
            l1(np.hstack([free3, lim_cols]),
               [(None, None)] * free3.shape[1] + [(0, None)] * nlim),
    }

    B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
    n_lam = sum(b.shape[1] for b in B)

    def legs_of(terms_):
        out = []
        for t in terms_:
            lg = None
            for cand in (t['geom2'], t['geom1']):
                if cand[:2] in LEGS:
                    lg = cand[:2]; break
            out.append(lg)
        return out

    leg_of = legs_of(terms)
    present = sorted({lg for lg in leg_of if lg})

    def with_adhesion(budget_bw):
        cols = [R]
        bounds = [(0.0, 1.0)] * nu
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b); bounds += [(0.0, None)] * b.shape[1]
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bounds += [(0.0, None)] * len(terms)
        A = np.hstack(cols + [lim_cols, np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(nu + n_lam + len(terms) + nlim), np.ones(2 * nv)])
        bounds += [(0, None)] * nlim + [(0, None)] * (2 * nv)
        A_ub = np.zeros((len(present), A.shape[1]))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    A_ub[i, nu + n_lam + k] = 1.0
        r = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub,
                    b_ub=np.full(len(present), budget_bw * body_weight), bounds=bounds,
                    method='highs', options={'presolve': True, 'time_limit': 60, 'maxiter': 20000})
        return (float(r.fun) if r.x is not None else None)

    sweep = {str(bw): with_adhesion(bw) for bw in (0.0, 0.5, 1.0, 1.64, 2.0, 4.0, 8.0, 16.0, 40.0)}
    floors['with_adhesion_1.64bw_per_leg'] = sweep['1.64']
    floors['with_adhesion_40bw_per_leg'] = sweep['40.0']

    # activation excursion for EXACT balance at this pose (ideal contacts, no slack columns)
    A = np.hstack([R, R, -R, free3, lim_cols])
    c = np.concatenate([np.zeros(nu), np.ones(2 * nu), np.zeros(free3.shape[1]), np.zeros(nlim)])
    bounds = ([(0, 1)] * nu + [(0, None)] * (2 * nu) + [(None, None)] * free3.shape[1]
              + [(0, None)] * nlim)
    r = linprog(c, A_eq=A, b_eq=target, bounds=bounds, method='highs',
                options={'presolve': True, 'time_limit': 60, 'maxiter': 20000})
    names = [model.actuator(i).name or f'act{i}' for i in range(nu)]
    if r.x is None:
        excursion = {'reachable': False}
    else:
        z = r.x
        a = z[:nu] + z[nu:2 * nu] - z[2 * nu:3 * nu]
        pos, neg = z[nu:2 * nu], z[2 * nu:3 * nu]
        outside = [{'actuator': names[i], 'required_activation': float(a[i]),
                    'excursion': float(pos[i] + neg[i])}
                   for i in range(nu) if pos[i] + neg[i] > 1e-9]
        excursion = {'reachable': True, 'min_total_excursion': float(r.fun),
                     'n_outside_box': len(outside), 'actuators_outside_box': outside}

    report = {
        'status': 'STANCE_MARGIN_DECOMPOSITION',
        'pose': {'source': str(BEST), 'tarsus_only': facts['contacts_are_feet_only_tarsus'],
                 'n_contacts': facts['contact_count'], 'thorax_height_mm': facts['thorax_height_mm'],
                 'root_z_mm': float(root_z)},
        'source_sha256': {str(XML): r15.digest(XML)},
        'body_weight_force_units': body_weight,
        'floors': floors,
        'floors_as_fraction_of_body_weight': {k: (v / body_weight if v is not None else None)
                                             for k, v in floors.items()},
        'adhesion_sweep_bw_per_leg': sweep,
        'activation_excursion_for_exact_balance': excursion,
        'interpretation': {
            'contact_side': 'if the ideal sign-free contact floor is ~0, the remaining residual at the '
                            'measured adhesion budget is a CONTACT-side shortfall, not an actuator one',
            'actuator_side': 'the excursion list names exactly which actuators would have to leave '
                             '[0,1] for exact balance with ideal contacts'},
        'limitations': [
            'Fixed pose, frozen force-length/velocity and moment arms; friction cone linearised.',
            'Ideal sign-free contacts are not physical: they only bound what any contact model could do.',
            'Static feasibility is NOT stability, NOT a controller, NOT walking.',
            'Nothing here is authorised to modify the canonical model.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (OUT / 'stance_margin.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ks = [k for k, v in sweep.items() if v is not None]
    axs[0].semilogy([float(k) for k in ks], [sweep[k] for k in ks], 'o-')
    axs[0].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    axs[0].axvline(1.64, ls=':', c='red', label='engine default 1.64 bw/leg')
    axs[0].set_xlabel('per-leg adhesion budget (body weights)')
    axs[0].set_ylabel('LP L1 residual (force units)')
    axs[0].set_title('Best tarsus-only stance: residual vs adhesion')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, which='both')
    labs = ['adhesion 1.64\n(engine default)', 'ideal contacts\n+box activation',
            'ideal contacts\n+free activation', 'no muscle\nideal contacts']
    vals = [floors['with_adhesion_1.64bw_per_leg'],
            floors['ideal_signfree_contacts_with_activation_box'],
            max(floors['ideal_signfree_contacts_unbounded_activation'] or 0, 1e-12),
            floors['no_muscle_ideal_contacts']]
    axs[1].bar(range(4), vals, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    axs[1].set_yscale('log')
    axs[1].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green')
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(labs, fontsize=7)
    axs[1].set_ylabel('L1 residual (force units)')
    axs[1].set_title('Where the remaining shortfall lives')
    axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('Best tarsus-only stance margin: diagnostic only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(OUT / 'stance_margin.png', dpi=110)

    print(json.dumps({'floors': floors,
                      'as_fraction_of_body_weight': report['floors_as_fraction_of_body_weight'],
                      'adhesion_sweep': sweep,
                      'excursion': {k: v for k, v in excursion.items()
                                    if k != 'actuators_outside_box'}}, indent=2))
    print('actuators outside box:', excursion.get('actuators_outside_box'))
    print(OUT)


if __name__ == '__main__':
    main()
