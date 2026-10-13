"""How much per-leg adhesion is actually REQUIRED for an exact, tarsus-only, upright stance?

Round26 found a tarsus-only upright posture family with the residual 0.3084 at the engine's
documented adhesion budget (1.64 body weights per leg), and showed that the residual falls
as the budget grows, down to the actuator-side floor 0.01574 (0.064% of body weight) and to
exactly 0 only if one actuator leaves [0,1].

That leaves the decisive question this file answers: is there a posture whose EXACT static
balance needs no more than the adhesion a fly actually has (1-2 body weights per leg)?
For each candidate pose the LP is
    minimise B   subject to   actuators in [0,1], friction cone, active joint limits,
                              per-leg tensile budget <= B, and EXACT force balance,
so B is the minimum per-leg adhesion the pose demands.  A pose with B <= 1.64 would be
statically standable within the documented, biology-consistent adhesion.

Diagnostic only: fixed-pose force balance; adhesion is a bounded tensile normal force per
leg, not pulvillus mechanics; feasibility is NOT stability, NOT a controller, NOT walking.
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

OUT = ROOT / 'outputs/stance_min_adhesion_round27'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050
ENGINE_DEFAULT_BW = 1.64
MEASURED_TOP_BW = 2.0
BIG_BW = 40.0                    # generous budget used to test whether exact balance is reachable at all
COORD_STEPS = (0.10, 0.05, 0.02)
STARTS = 4
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

    def leg_of_terms(terms):
        outl = []
        for t in terms:
            lg = None
            for cand in (t['geom2'], t['geom1']):
                if cand[:2] in LEGS:
                    lg = cand[:2]; break
            outl.append(lg)
        return outl

    def pose_qpos(x):
        qpos = key_qpos.copy()
        joints = {}
        for k, (leg, j) in enumerate(params):
            lo, hi = float(rng[j][0]), float(rng[j][1])
            v = float(np.clip(key_val[j][leg] + x[k], lo, hi))
            qpos[adr[j][leg]] = v
            joints[f'joint_{leg}{j}'] = v
        return qpos, joints

    def build(x):
        qpos, joints = pose_qpos(x)
        seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, PENETRATION_MM)
        data.qpos[:] = seated; data.qvel[:] = 0
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        q = np.asarray(data.qpos[3:7], float)
        up_z = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        return seated, root_z, joints, facts, up_z

    def solve_min_adhesion(x):
        """Minimal per-leg adhesion (body weights) for EXACT balance, or a reason it is impossible."""
        seated, root_z, joints, facts, up_z = build(x)
        if not facts['contacts_are_feet_only_tarsus']:
            return {'ok': False, 'why': 'non-tarsus ground contact'}
        if up_z < 0.9:
            return {'ok': False, 'why': 'not upright'}
        if facts['contact_count'] == 0:
            return {'ok': False, 'why': 'no contact'}
        R, base = r15.actuator_moment_matrix(model, data, mj)
        target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
        terms = r15.contact_terms(model, data, mj)
        lims = r15.limit_terms(model, data, mj)
        lim_cols = (np.column_stack([l['direction'] for l in lims]) if lims else np.zeros((model.nv, 0)))
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        nv, nu = model.nv, model.nu
        n_lam = sum(b.shape[1] for b in B)
        n_p = len(terms)
        nlim = lim_cols.shape[1]
        leg_of = leg_of_terms(terms)
        present = sorted({lg for lg in leg_of if lg})

        # stage 1: is exact balance reachable at all with a generous budget?
        cols = [R]
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b)
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        A1 = np.hstack(cols + [lim_cols, np.eye(nv), -np.eye(nv)])
        c1 = np.concatenate([np.zeros(nu + n_lam + n_p + nlim), np.ones(2 * nv)])
        bnds1 = ([(0, 1)] * nu + [(0, None)] * n_lam + [(0, None)] * n_p
                 + [(0, None)] * nlim + [(0, None)] * (2 * nv))
        A_ub = np.zeros((len(present) + n_p, A1.shape[1]))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    A_ub[i, nu + n_lam + k] = 1.0
        for k in range(n_p):
            A_ub[len(present) + k, nu + n_lam + k] = 1.0
        b_ub1 = np.concatenate([np.full(len(present), BIG_BW * body_weight),
                                np.full(n_p, BIG_BW * body_weight)])
        r1 = linprog(c1, A_eq=A1, b_eq=target, A_ub=A_ub, b_ub=b_ub1, bounds=bnds1,
                     method='highs', options=LP_OPTS)
        if r1.x is None:
            return {'ok': False, 'why': 'LP failed: ' + r1.message}
        resid = float(r1.fun)
        if resid > 1e-6:
            return {'ok': False, 'why': 'actuator-side: exact balance unreachable even with %g bw/leg' % BIG_BW,
                    'residual_at_big_budget': resid}

        # stage 2: minimise B subject to EXACT balance (no slack columns)
        A2 = np.hstack(cols + [lim_cols])
        c2 = np.zeros(A2.shape[1] + 1)
        c2[-1] = 1.0                                     # variable: B in force units
        A_eq = np.hstack([A2, np.zeros((nv, 1))])
        # per-leg tensile sums - B <= 0
        rows = np.zeros((len(present), A2.shape[1] + 1))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    rows[i, nu + n_lam + k] = 1.0
            rows[i, -1] = -1.0
        bnds2 = ([(0, 1)] * nu + [(0, None)] * n_lam + [(0, None)] * n_p
                 + [(0, None)] * nlim + [(0, None)])
        r2 = linprog(c2, A_eq=A_eq, b_eq=target, A_ub=rows, b_ub=np.zeros(len(present)),
                     bounds=bnds2, method='highs', options=LP_OPTS)
        if r2.x is None:
            return {'ok': False, 'why': 'stage-2 LP failed: ' + r2.message}
        B_bw = float(r2.x[-1]) / body_weight
        return {'ok': True, 'min_per_leg_adhesion_bw': B_bw,
                'within_engine_default': bool(B_bw <= ENGINE_DEFAULT_BW),
                'within_measured_range': bool(B_bw <= MEASURED_TOP_BW),
                'n_contacts': int(facts['contact_count']),
                'thorax_height_mm': float(facts['thorax_height_mm']),
                'root_z_mm': float(root_z), 'torso_up_z': up_z, 'joints_rad': joints}

    seed = json.loads(SEED.read_text())
    best_seed = seed['best']
    x_seed = np.zeros(len(params))
    for k, (leg, j) in enumerate(params):
        x_seed[k] = best_seed['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg]

    rng_gen = np.random.default_rng(11)
    starts = [x_seed, np.zeros(len(params))]
    while len(starts) < STARTS:
        starts.append(x_seed + rng_gen.normal(0.0, 0.05, size=len(params)))

    def score(res):
        return res['min_per_leg_adhesion_bw'] if res.get('ok') else np.inf

    evals = 0
    history, reasons = [], {}
    best = None
    for si, start in enumerate(starts):
        x = np.array(start, float)
        res = solve_min_adhesion(x); evals += 1
        cur = score(res)
        if not res.get('ok'):
            reasons[res['why']] = reasons.get(res['why'], 0) + 1
        history.append({'start': si, 'eval': evals, 'min_bw': res.get('min_per_leg_adhesion_bw'),
                        'ok': res.get('ok', False)})
        if res.get('ok') and (best is None or res['min_per_leg_adhesion_bw'] < best['min_per_leg_adhesion_bw']):
            best = res
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
                        r2 = solve_min_adhesion(cand); evals += 1
                        if not r2.get('ok'):
                            reasons[r2['why']] = reasons.get(r2['why'], 0) + 1
                        v = score(r2)
                        history.append({'start': si, 'eval': evals,
                                        'min_bw': r2.get('min_per_leg_adhesion_bw'),
                                        'ok': r2.get('ok', False)})
                        if v < cur - 1e-6:
                            x, cur = cand, v
                            improved = True
                            if r2.get('ok') and (best is None or
                                                 r2['min_per_leg_adhesion_bw'] < best['min_per_leg_adhesion_bw']):
                                best = r2
                if sweeps >= 5:
                    break

    report = {
        'status': 'MINIMUM_ADHESION_STANCE_SEARCH',
        'question': 'is there a tarsus-only upright posture whose EXACT static balance needs no more '
                    'per-leg adhesion than the documented engine default (1.64 body weights) or the top '
                    'of the measured range (2.0)?',
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED)},
        'config': {'penetration_mm': PENETRATION_MM, 'n_params': len(params),
                   'engine_default_bw_per_leg': ENGINE_DEFAULT_BW,
                   'measured_top_bw_per_leg': MEASURED_TOP_BW,
                   'big_budget_bw_per_leg': BIG_BW, 'starts': len(starts), 'max_evals': MAX_EVALS},
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'rejection_reasons': reasons,
        'best': best,
        'found_within_engine_default': bool(best and best.get('ok')
                                            and best['min_per_leg_adhesion_bw'] <= ENGINE_DEFAULT_BW),
        'found_within_measured_range': bool(best and best.get('ok')
                                            and best['min_per_leg_adhesion_bw'] <= MEASURED_TOP_BW),
        'history': history,
        'limitations': [
            'Fixed-pose static balance only: NOT stability, NOT controllability, NOT walking.',
            'Adhesion is a bounded tensile normal force per leg, not pulvillus contact mechanics.',
            'Actuator force affine in activation at the frozen pose; moment arms and force-length/velocity '
            'frozen; friction cone linearised.',
            'Bounded search over per-leg pitch offsets only; yaw/roll and tarsus joints not searched.',
            'No model file is modified.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_min_adhesion.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    pts = [(h['eval'], h['min_bw']) for h in history if h.get('ok') and h['min_bw'] is not None]
    if pts:
        axs[0].plot([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    axs[0].axhline(ENGINE_DEFAULT_BW, ls='--', c='red', label='engine default 1.64 bw/leg')
    axs[0].axhline(MEASURED_TOP_BW, ls=':', c='orange', label='measured top 2.0 bw/leg')
    axs[0].set_xlabel('pose evaluation'); axs[0].set_ylabel('minimum required per-leg adhesion (bw)')
    axs[0].set_title('Per-leg adhesion needed for EXACT balance')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    labs = list(reasons) or ['none rejected']
    axs[1].barh(range(len(labs)), [reasons.get(l, 0) for l in labs], color='tab:red')
    axs[1].set_yticks(range(len(labs)))
    axs[1].set_yticklabels([l[:46] for l in labs], fontsize=7)
    axs[1].set_xlabel('poses rejected for this reason')
    axs[1].set_title('Rejections')
    axs[1].grid(alpha=.3, axis='x')
    fig.suptitle('Minimum-adhesion stance search: diagnostic only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'stance_min_adhesion.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'rejection_reasons': reasons,
                      'best_min_bw': (best or {}).get('min_per_leg_adhesion_bw'),
                      'found_within_engine_default': report['found_within_engine_default'],
                      'found_within_measured_range': report['found_within_measured_range'],
                      'best_thorax_mm': (best or {}).get('thorax_height_mm'),
                      'best_contacts': (best or {}).get('n_contacts')}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
