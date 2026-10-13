"""What does the ENGINE'S OWN documented leg adhesion do to the standing keyframe?

Round17 measured how much adhesion would be needed and compared the requirement with
measured fly biology (Ramdya et al. 2017 counted pretarsi holding an inverted fly for
>1 s, i.e. ~1 body weight per contact, and used 2 body weights per contacting leg for
gait optimisation; they warn that leg adhesion has NOT been formally measured for
D. melanogaster).  This project's engine has a documented adhesion default in another
branch: FlyGym's Fly.add_leg_adhesion(gain=40.0) = 40 uN per leg = 1.64 body weights
per leg, which sits INSIDE that measured range - while the musculoskeletal
(FlyMimic) branch this project actually uses documents "FlyMimic has no adhesion".

So the useful question is not "unlimited adhesion" (a floor) but: at the engine's own
documented, biology-consistent per-LEG budget, how much of the standing shortfall
goes away?  This is a static force-balance measurement, not a controller and not a
model edit.
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

OUT = ROOT / 'outputs/adhesion_per_leg_round19'
XML = r15.XML
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
BUDGETS_BW_PER_LEG = (0.0, 0.25, 0.5, 1.0, 1.64, 2.0, 4.0, 8.0, 16.0, 40.0)
ENGINE_DEFAULT_BW_PER_LEG = 1.64
MEASURED_RANGE_BW_PER_LEG = (1.0, 2.0)


def leg_of(geom_name):
    for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
        if geom_name.startswith(leg):
            return leg
    return None


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

    R, base = r15.actuator_moment_matrix(model, data, mj)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mj)
    lims = r15.limit_terms(model, data, mj)
    nv = model.nv
    body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))
    lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims else np.zeros((nv, 0)))
    nlim = lim_cols.shape[1]

    # attach each contact to a leg through whichever geom is a leg geom
    legs = []
    for t in terms:
        lg = leg_of(t['geom2']) or leg_of(t['geom1'])
        legs.append(lg)
    present = sorted({lg for lg in legs if lg})

    def solve_per_leg(budget_bw):
        """LP with per-leg tensile budget; friction bound tied to the compressive part."""
        cols, bounds, leg_rows = [], [], []
        A_base = np.hstack([R, lim_cols])
        bounds_base = [(0, 1)] * model.nu + [(0, None)] * nlim
        cols.append(A_base)
        # compressive pyramid generators, exactly as the validated helper builds them
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b)
            bounds.extend([(0, None)] * b.shape[1])
        n_lam = sum(b.shape[1] for b in B)
        assert len(bounds) == n_lam, (len(bounds), n_lam)
        # one tensile variable per contact, bounded by the leg budget through rows
        for t, lg in zip(terms, legs):
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        n_p = len(terms)
        A = np.hstack(cols + [np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(model.nu + nlim + n_lam), np.zeros(n_p), np.ones(2 * nv)])
        # the tensile variable is a FORCE magnitude in model force units, so it is
        # unbounded here and limited only through the per-leg row, which converts the
        # body-weight budget into force units.
        bnds = (bounds_base + bounds + [(0, None)] * n_p + [(0, None)] * (2 * nv))
        A_ub = np.zeros((len(present), A.shape[1]))
        for i, lg in enumerate(present):
            for k, (t, l2) in enumerate(zip(terms, legs)):
                if l2 == lg:
                    A_ub[i, model.nu + nlim + n_lam + k] = 1.0
        b_ub = np.full(len(present), budget_bw * body_weight)
        r = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub, b_ub=b_ub, bounds=bnds,
                    method='highs', options=LP_OPTS)
        if r.x is None:
            return {'usable': False, 'message': r.message}
        return {'usable': True, 'L1': float(r.fun),
                'L1_fraction_of_body_weight': float(r.fun) / body_weight}

    # budgets are expressed in BODY WEIGHTS per leg; the tensile variable is bounded
    # by that number directly, so the LP is unit-consistent with the validated helper.
    sweep = []
    for bw in BUDGETS_BW_PER_LEG:
        sol = solve_per_leg(bw)
        sweep.append({'per_leg_budget_body_weights': bw, **sol})
        print(bw, sol.get('L1'), flush=True)

    zero_budget = next(s for s in sweep if s['per_leg_budget_body_weights'] == 0.0)
    engine = next(s for s in sweep if s['per_leg_budget_body_weights'] == ENGINE_DEFAULT_BW_PER_LEG)
    top_measured = next(s for s in sweep if s['per_leg_budget_body_weights'] == MEASURED_RANGE_BW_PER_LEG[1])

    report = {
        'status': 'PER_LEG_ADHESION_BUDGET_SWEEP',
        'what_this_is': 'static force balance at the XML standing keyframe with a per-leg tensile '
                        'contact budget; no controller, no model edit',
        'source_sha256': {str(XML): r15.digest(XML)},
        'pose': {'thorax_height_model': float(data.xpos[model.body("Thorax").id][2]),
                 'n_contacts': int(data.ncon), 'legs_with_contact': present},
        'body_weight_force_units': body_weight,
        'adhesion_model': 'one bounded tensile normal force per contact plus a per-leg total budget; '
                          'friction bound tied to the compressive part only. NOT pulvillus contact '
                          'mechanics.',
        'engine_default_reference': {'source': 'FlyGym Fly.add_leg_adhesion(gain=40.0) = 40 uN per leg',
                                     'body_weights_per_leg': ENGINE_DEFAULT_BW_PER_LEG},
        'measured_reference': {'source': 'Ramdya et al. 2017 Nat Commun 8:14494',
                               'body_weights_per_leg': list(MEASURED_RANGE_BW_PER_LEG)},
        'sweep': sweep,
        'headline': {
            'residual_with_no_adhesion': zero_budget['L1'],
            'residual_at_engine_default': engine['L1'],
            'residual_at_top_of_measured_range': top_measured['L1'],
            'fraction_of_body_weight_at_engine_default': engine['L1_fraction_of_body_weight'],
        },
        'limitations': [
            'Static balance at one frozen pose: NOT stability, NOT controllability, NOT a controller, '
            'NOT standing.',
            'The adhesion model is a bounded tensile normal force, not pulvillus contact mechanics; a '
            'reduced residual identifies a candidate physical gap and does NOT authorise silently '
            'editing the model.',
            'Actuator force is affine in activation only at the frozen pose; force-length/velocity and '
            'moment arms frozen.',
            'The engine default comes from a different FlyGym branch; the musculoskeletal branch this '
            'project uses documents no adhesion at all, so adding it would be a real model change.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (OUT / 'adhesion_per_leg.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [s['per_leg_budget_body_weights'] for s in sweep if s.get('usable')]
    ys = [s['L1'] for s in sweep if s.get('usable')]
    ax.semilogy(xs, ys, 'o-', label='keyframe static residual')
    ax.axvspan(*MEASURED_RANGE_BW_PER_LEG, color='green', alpha=.15,
               label='measured fly range (1-2 bw/leg)')
    ax.axvline(ENGINE_DEFAULT_BW_PER_LEG, ls='--', color='tab:red',
               label='engine default = 1.64 bw/leg (40 uN)')
    ax.axhline(r15.FEASIBLE_SUM_TOL, ls=':', color='green', label='feasibility tolerance')
    ax.set_xlabel('adhesion budget per LEG (body weights)')
    ax.set_ylabel('L1 static-support residual (force units)')
    ax.set_title('Keyframe residual vs per-leg adhesion budget\n'
                 'Static balance only: NOT a controller, NOT standing')
    ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')
    fig.tight_layout(); fig.savefig(OUT / 'adhesion_per_leg.png', dpi=110)

    print(json.dumps(report['headline'], indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
