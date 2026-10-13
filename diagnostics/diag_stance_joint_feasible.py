"""Combined stance search: activations inside [0,1] AND minimal per-leg adhesion.

Round29 found the first tarsus-only upright stance whose exact balance needs NO activation
outside [0,1] (zero excursion) - the actuator-side blocker is gone at that pose.  Under the
realistic contact model its residual is still 0.81 with the engine's documented adhesion, so
the remaining question is now purely contact-side and well posed:

    among poses with ZERO activation excursion, which one needs the LEAST per-leg adhesion
    for exact balance under the friction cone?

For each pose:
  1. ideal sign-free contacts -> minimal total activation excursion for exact balance; poses
     with excursion > tolerance are rejected (the actuator box would have to be violated);
  2. friction cone plus a per-leg tensile budget B -> minimise B subject to EXACT balance;
     a pose with B <= 1.64 body weights needs no more adhesion than the engine's own default
     (and than the measured 1-2 bw/leg range), i.e. it is statically standable within the
     documented biology.

Diagnostic only: fixed-pose force balance.  A feasible pose is NOT stability, NOT a
controller and NOT walking; it is the prerequisite this project has been missing.
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

OUT = ROOT / 'outputs/stance_joint_feasible_round30'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050
ENGINE_DEFAULT_BW = 1.64
MEASURED_TOP_BW = 2.0
EXCURSION_TOL = 1e-9
COORD_STEPS = (0.06, 0.03, 0.015, 0.008)
STARTS = 5
MAX_EVALS = 2500
WALL_BUDGET_S = 900.0
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
SEED = ROOT / 'outputs/stance_min_excursion_round29/stance_min_excursion.json'


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
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

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
        return {'R': R, 'target': target, 'free3': free3, 'lim_cols': lim_cols, 'terms': terms,
                'joints': joints, 'root_z': root_z, 'facts': facts, 'up_z': up_z}, None

    def excursion(ctx):
        nv, nu = model.nv, model.nu
        A = np.hstack([ctx['R'], ctx['R'], -ctx['R'], ctx['free3'], ctx['lim_cols']])
        c = np.concatenate([np.zeros(nu), np.ones(2 * nu), np.zeros(ctx['free3'].shape[1]),
                            np.zeros(ctx['lim_cols'].shape[1])])
        bnds = ([(0, 1)] * nu + [(0, None)] * (2 * nu) + [(None, None)] * ctx['free3'].shape[1]
                + [(0, None)] * ctx['lim_cols'].shape[1])
        r = linprog(c, A_eq=A, b_eq=ctx['target'], bounds=bnds, method='highs', options=LP_OPTS)
        return (None if r.x is None else float(r.fun))

    def min_adhesion(ctx):
        """Minimal per-leg adhesion (body weights) for EXACT balance under the friction cone."""
        R, target, terms, lim_cols = ctx['R'], ctx['target'], ctx['terms'], ctx['lim_cols']
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        nu, nv = model.nu, model.nv
        n_lam = sum(b.shape[1] for b in B)
        n_p = len(terms)
        nlim = lim_cols.shape[1]
        cols, bnds = [R], [(0, 1)] * nu
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b); bnds += [(0, None)] * b.shape[1]
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bnds += [(0, None)] * n_p + [(0, None)] * nlim
        A = np.hstack(cols + [lim_cols])
        A_eq = np.hstack([A, np.zeros((nv, 1))])
        c = np.zeros(A.shape[1] + 1); c[-1] = 1.0
        leg_of = []
        for t in terms:
            lg = None
            for cand in (t['geom2'], t['geom1']):
                if cand[:2] in LEGS:
                    lg = cand[:2]; break
            leg_of.append(lg)
        present = sorted({l for l in leg_of if l})
        rows = np.zeros((len(present), A.shape[1] + 1))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    rows[i, nu + n_lam + k] = 1.0
            rows[i, -1] = -1.0
        r = linprog(c, A_eq=A_eq, b_eq=target, A_ub=rows, b_ub=np.zeros(len(present)),
                    bounds=bnds + [(0, None)], method='highs', options=LP_OPTS)
        if r.x is None:
            return None
        return float(r.x[-1]) / body_weight

    seed = json.loads(SEED.read_text())['best']
    x_seed = np.array([seed['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg] for leg, j in params])
    gen = np.random.default_rng(23)
    starts = [x_seed] + [x_seed + gen.normal(0, 0.05, size=len(params)) for _ in range(STARTS - 1)]

    def objective(ctx):
        """None if the actuator box would have to be violated; else required adhesion in bw."""
        ex = excursion(ctx)
        if ex is None or ex > EXCURSION_TOL:
            return None
        return min_adhesion(ctx)

    evals = 0
    history, reasons = [], {}
    best = None
    for si, start in enumerate(starts):
        x = np.array(start, float)
        ctx, why = setup(x); evals += 1
        cur = np.inf
        if ctx is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            v = objective(ctx)
            if v is not None:
                cur = v
                if best is None or v < best['min_adhesion_bw']:
                    best = {'min_adhesion_bw': v, 'x': x.tolist(), 'joints_rad': ctx['joints'],
                            'up_z': ctx['up_z'], 'thorax_height_mm': ctx['facts']['thorax_height_mm']}
        history.append({'start': si, 'eval': evals, 'min_bw': None if cur == np.inf else cur})
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
                            v = objective(ctx2)
                        history.append({'start': si, 'eval': evals, 'min_bw': v})
                        if v is not None and v < cur - 1e-6:
                            x, cur = cand, v
                            improved = True
                            if best is None or v < best['min_adhesion_bw']:
                                best = {'min_adhesion_bw': v, 'x': cand.tolist(),
                                        'joints_rad': ctx2['joints'], 'up_z': ctx2['up_z'],
                                        'thorax_height_mm': ctx2['facts']['thorax_height_mm']}
                if sweeps >= 6:
                    break

    detail = None
    if best is not None:
        ctx, _ = setup(np.array(best['x'], float))
        if ctx is not None:
            detail = dict(best, excursion_recheck=excursion(ctx),
                          adhesion_recheck_bw=min_adhesion(ctx),
                          n_contacts=ctx['facts']['contact_count'],
                          tarsus_only=ctx['facts']['contacts_are_feet_only_tarsus'])

    report = {
        'status': 'JOINT_STANCE_FEASIBILITY_SEARCH',
        'question': 'among tarsus-only upright poses whose exact balance needs NO activation outside '
                    '[0,1], which needs the least per-leg adhesion?',
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED)},
        'config': {'penetration_mm': PENETRATION_MM, 'n_params': len(params),
                   'excursion_tolerance': EXCURSION_TOL,
                   'engine_default_bw_per_leg': ENGINE_DEFAULT_BW,
                   'measured_top_bw_per_leg': MEASURED_TOP_BW,
                   'starts': len(starts), 'max_evals': MAX_EVALS},
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'rejection_reasons': reasons,
        'best': detail,
        'feasible_within_engine_default': bool(detail and detail['min_adhesion_bw'] <= ENGINE_DEFAULT_BW),
        'feasible_within_measured_range': bool(detail and detail['min_adhesion_bw'] <= MEASURED_TOP_BW),
        'history': history,
        'limitations': [
            'Fixed-pose static balance: NOT stability, NOT a controller, NOT walking.',
            'Ideal sign-free contacts are used only to screen poses on the ACTUATOR side; the adhesion '
            'number comes from the linearised friction cone plus a bounded tensile normal force, not from '
            'pulvillus contact mechanics.',
            'Bounded search over per-leg pitch offsets only.',
            'No model file is modified; the canonical asset stays byte-identical.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_joint_feasible.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    pts = [(h['eval'], h['min_bw']) for h in history if h.get('min_bw') is not None]
    if pts:
        axs[0].plot([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    axs[0].axhline(ENGINE_DEFAULT_BW, ls='--', c='red', label='engine default 1.64 bw/leg')
    axs[0].axhline(MEASURED_TOP_BW, ls=':', c='orange', label='measured top 2.0 bw/leg')
    axs[0].set_xlabel('pose evaluation'); axs[0].set_ylabel('required per-leg adhesion (bw)')
    axs[0].set_title('Poses with ZERO activation excursion only')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    labs = list(reasons) or ['no rejections']
    axs[1].barh(range(len(labs)), [reasons.get(l, 0) for l in labs], color='tab:red')
    axs[1].set_yticks(range(len(labs)))
    axs[1].set_yticklabels([l[:44] for l in labs], fontsize=8)
    axs[1].set_xlabel('poses rejected')
    axs[1].set_title('Rejections')
    axs[1].grid(alpha=.3, axis='x')
    fig.suptitle('Joint stance feasibility: diagnostic only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'stance_joint_feasible.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'rejection_reasons': reasons,
                      'best_min_adhesion_bw': (detail or {}).get('min_adhesion_bw'),
                      'excursion_recheck': (detail or {}).get('excursion_recheck'),
                      'best_thorax_mm': (detail or {}).get('thorax_height_mm'),
                      'n_contacts': (detail or {}).get('n_contacts'),
                      'feasible_within_engine_default': report['feasible_within_engine_default'],
                      'feasible_within_measured_range': report['feasible_within_measured_range']}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
