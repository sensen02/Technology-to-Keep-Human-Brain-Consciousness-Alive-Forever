"""Find the stance that needs the LEAST activation excursion, not the least residual.

Round28's census showed the tarsus-only upright stance family is blocked by the activation box
(a few actuators would have to leave [0,1] for exact balance, `LHF_trochanter_flexor_a` in
every sampled pose), and that the a/b coupling "fix" is not available because coupling removes
force directions the body needs (rank 48 -> 21-45, and the required force leaves the span).

So the remaining legitimate question is geometric: is there a posture in the family whose exact
balance needs NO activation outside [0,1]?  This file searches the pose space with the
EXCURSION as the objective, at ideal sign-free contacts (which bound any contact model), and
requires tarsus-only contact and an upright torso.  The best pose found is then re-scored under
the realistic contact model (friction cone plus the engine's documented per-leg adhesion).

Diagnostic only: fixed-pose force balance.  Zero excursion means "balanceable with activations
inside the physical range at this instant", which is NOT stability, NOT a controller and NOT
walking; a finite positive result is reported as the shortfall it is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

OUT = ROOT / 'outputs/stance_min_excursion_round29'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050
ADHESION_BW = 1.64
COORD_STEPS = (0.08, 0.04, 0.02, 0.01)
STARTS = 5
MAX_EVALS = 2500
WALL_BUDGET_S = 900.0
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
SEED = ROOT / 'outputs/stance_with_adhesion_round26/stance_with_adhesion.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

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

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key_qpos = data.qpos.copy()
    adr = {j: {leg: int(model.jnt_qposadr[model.joint(f'joint_{leg}{j}').id]) for leg in LEGS}
           for j in PARAM_JOINTS}
    rng = {j: model.jnt_range[model.joint(f'joint_LF{j}').id].copy() for j in PARAM_JOINTS}
    key_val = {j: {leg: float(key_qpos[adr[j][leg]]) for leg in LEGS} for j in PARAM_JOINTS}
    params = [(leg, j) for leg in LEGS for j in PARAM_JOINTS]

    def setup(x):
        qpos = key_qpos.copy()
        joints = {}
        for k, (leg, j) in enumerate(params):
            lo, hi = float(rng[j][0]), float(rng[j][1])
            v = float(np.clip(key_val[j][leg] + x[k], lo, hi))
            qpos[adr[j][leg]] = v
            joints[f'joint_{leg}{j}'] = v
        seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, PENETRATION_MM)
        data.qpos[:] = seated; data.qvel[:] = 0
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        q = np.asarray(data.qpos[3:7], float)
        up_z = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        if not facts['contacts_are_feet_only_tarsus']:
            return None, 'non-tarsus ground contact'
        if up_z < 0.9:
            return None, 'not upright'
        if facts['contact_count'] == 0:
            return None, 'no contact'
        R, base = r15.actuator_moment_matrix(model, data, mj)
        target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
        terms = r15.contact_terms(model, data, mj)
        lims = r15.limit_terms(model, data, mj)
        lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims
                    else np.zeros((model.nv, 0)))
        free3 = np.hstack([t['J'].T for t in terms])
        return {'R': R, 'target': target, 'free3': free3, 'lim_cols': lim_cols,
                'terms': terms, 'joints': joints, 'root_z': root_z, 'facts': facts, 'up_z': up_z}, None

    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    def excursion(ctx):
        nv, nu = model.nv, model.nu
        R, free3, lim_cols = ctx['R'], ctx['free3'], ctx['lim_cols']
        A = np.hstack([R, R, -R, free3, lim_cols])
        c = np.concatenate([np.zeros(nu), np.ones(2 * nu), np.zeros(free3.shape[1]),
                            np.zeros(lim_cols.shape[1])])
        bnds = ([(0, 1)] * nu + [(0, None)] * (2 * nu) + [(None, None)] * free3.shape[1]
                + [(0, None)] * lim_cols.shape[1])
        r = linprog(c, A_eq=A, b_eq=ctx['target'], bounds=bnds, method='highs', options=LP_OPTS)
        if r.x is None:
            return None, []
        z = r.x
        pos, neg = z[nu:2 * nu], z[2 * nu:3 * nu]
        outside = [(names[i], float(pos[i] + neg[i])) for i in range(nu)
                   if pos[i] + neg[i] > 1e-9]
        return float(r.fun), outside

    def real_contact_residual(ctx):
        """LP residual under the friction cone with the documented per-leg adhesion budget."""
        R, target = ctx['R'], ctx['target']
        terms, lims = ctx['terms'], r15.limit_terms(model, data, mj)
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        nu, nv = model.nu, model.nv
        n_lam = sum(b.shape[1] for b in B)
        n_p = len(terms)
        lim_cols = ctx['lim_cols']
        nlim = lim_cols.shape[1]
        cols = [R]
        bnds = [(0, 1)] * nu
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b); bnds += [(0, None)] * b.shape[1]
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bnds += [(0, None)] * n_p
        A = np.hstack(cols + [lim_cols, np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(nu + n_lam + n_p + nlim), np.ones(2 * nv)])
        bnds += [(0, None)] * nlim + [(0, None)] * (2 * nv)
        leg_of = []
        for t in terms:
            lg = None
            for cand in (t['geom2'], t['geom1']):
                if cand[:2] in LEGS:
                    lg = cand[:2]; break
            leg_of.append(lg)
        present = sorted({l for l in leg_of if l})
        A_ub = np.zeros((len(present), A.shape[1]))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    A_ub[i, nu + n_lam + k] = 1.0
        r = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub,
                    b_ub=np.full(len(present), ADHESION_BW * body_weight), bounds=bnds,
                    method='highs', options=LP_OPTS)
        return float(r.fun) if r.x is not None else None

    seed = json.loads(SEED.read_text())['best']
    x_seed = np.array([seed['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg] for leg, j in params])
    gen = np.random.default_rng(13)
    starts = [x_seed, np.zeros(len(params))]
    while len(starts) < STARTS:
        starts.append(x_seed + gen.normal(0, 0.06, size=len(params)))

    history, reasons, best = [], {}, None
    evals = 0
    for si, start in enumerate(starts):
        x = np.array(start, float)
        ctx, why = setup(x); evals += 1
        cur = np.inf
        if ctx is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            val, _ = excursion(ctx)
            if val is not None:
                cur = val
                if best is None or val < best['excursion']:
                    best = {'excursion': val, 'x': x.tolist(), 'joints_rad': ctx['joints'],
                            'up_z': ctx['up_z'], 'thorax_height_mm': ctx['facts']['thorax_height_mm']}
        history.append({'start': si, 'eval': evals, 'excursion': None if cur == np.inf else cur})
        for step in COORD_STEPS:
            improved = True
            sweeps = 0
            while improved and evals < MAX_EVALS and time.time() - t0 < WALL_BUDGET_S:
                improved = False
                sweeps += 1
                for k in range(len(params)):
                    for delta in (+step, -step):
                        if evals >= MAX_EVALS or time.time() - t0 > WALL_BUDGET_S:
                            break
                        cand = x.copy(); cand[k] += delta
                        ctx2, why2 = setup(cand); evals += 1
                        v = None
                        if ctx2 is None:
                            reasons[why2] = reasons.get(why2, 0) + 1
                        else:
                            v, _ = excursion(ctx2)
                        history.append({'start': si, 'eval': evals, 'excursion': v})
                        if v is not None and v < cur - 1e-9:
                            x, cur = cand, v
                            improved = True
                            if best is None or v < best['excursion']:
                                best = {'excursion': v, 'x': cand.tolist(), 'joints_rad': ctx2['joints'],
                                        'up_z': ctx2['up_z'],
                                        'thorax_height_mm': ctx2['facts']['thorax_height_mm']}
                if sweeps >= 6:
                    break

    best_detail = None
    if best is not None:
        ctx, why = setup(np.array(best['x'], float))
        if ctx is not None:
            val, outside = excursion(ctx)
            best_detail = dict(best, excursion_recheck=val, actuators_outside_box=outside,
                               real_contact_residual_with_adhesion=real_contact_residual(ctx),
                               n_contacts=ctx['facts']['contact_count'])

    report = {
        'status': 'STANCE_MIN_EXCURSION_SEARCH',
        'question': 'is there a tarsus-only upright posture whose exact balance needs NO activation '
                    'outside [0,1]?',
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED)},
        'config': {'penetration_mm': PENETRATION_MM, 'n_params': len(params),
                   'objective': 'minimum total activation excursion outside [0,1] for exact balance at '
                                'ideal sign-free contacts',
                   'adhesion_bw_per_leg_used_for_rescoring': ADHESION_BW,
                   'starts': len(starts), 'max_evals': MAX_EVALS},
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'rejection_reasons': reasons,
        'best': best_detail,
        'found_zero_excursion': bool(best_detail and best_detail['excursion'] <= 1e-9),
        'history': history,
        'limitations': [
            'Fixed-pose balance only: NOT stability, NOT controllability, NOT walking.',
            'Zero excursion would mean balanceable with in-range activations AT THIS INSTANT, not that a '
            'controller exists.',
            'Ideal sign-free contacts bound any contact model; the rescored real-contact residual uses a '
            'linearised cone and the documented adhesion budget.',
            'Bounded search: per-leg pitch offsets only.',
            'No model file is modified.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_min_excursion.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    pts = [(h['eval'], h['excursion']) for h in history if h.get('excursion') is not None]
    if pts:
        axs[0].plot([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    axs[0].axhline(1e-9, ls='--', c='green', label='zero excursion')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('pose evaluation'); axs[0].set_ylabel('min activation excursion')
    axs[0].set_title('Searching for a pose that needs no out-of-range activation')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, which='both')
    labs = list(reasons) or ['no rejections']
    axs[1].barh(range(len(labs)), [reasons.get(l, 0) for l in labs], color='tab:red')
    axs[1].set_yticks(range(len(labs)))
    axs[1].set_yticklabels([l[:44] for l in labs], fontsize=8)
    axs[1].set_xlabel('poses rejected')
    axs[1].set_title('Rejections')
    axs[1].grid(alpha=.3, axis='x')
    fig.suptitle('Minimum-excursion stance search: diagnostic only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'stance_min_excursion.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'rejection_reasons': reasons,
                      'best_excursion': (best_detail or {}).get('excursion'),
                      'best_outside_box': (best_detail or {}).get('actuators_outside_box'),
                      'best_real_contact_residual': (best_detail or {}).get('real_contact_residual_with_adhesion'),
                      'best_thorax_mm': (best_detail or {}).get('thorax_height_mm'),
                      'found_zero_excursion': report['found_zero_excursion']}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
