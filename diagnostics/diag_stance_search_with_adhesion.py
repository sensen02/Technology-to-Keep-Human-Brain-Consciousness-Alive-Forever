"""Does a TARSUS-ONLY, upright, statically supportable stance exist WITH documented adhesion?

Round15/16 established that the XML keyframe cannot be statically supported and that no
feet-only posture was found WITHOUT adhesion.  Round19/22 showed that the engine's own
documented leg adhesion (1.64 body weights per leg, inside the measured 1-2 bw/leg range)
keeps all six feet planted for a full second, but the body still touches down on tibiae,
abdomen and a wing.

This file asks the question that walking actually depends on: with that adhesion present,
is there ANY posture - searched per leg, not uniformly - that is
  (a) statically supportable (LP residual within tolerance),
  (b) in contact with the floor through TARSI ONLY, and
  (c) upright (torso axis within ~25 degrees of vertical)?

Diagnostic only: fixed-pose force balance, no controller, no target, no model edit.  The
adhesion is a bounded tensile normal force per leg in the LP, i.e. a CAPABILITY bound, not
pulvillus contact mechanics, and static feasibility is NOT stability and NOT walking.
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

OUT = ROOT / 'outputs/stance_with_adhesion_round26'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_pitch', 'Trochanter_pitch', 'Tibia_pitch')
PENETRATION_MM = -0.050          # signed: negative = feet slightly into the floor
ADHESION_BW_PER_LEG = 1.64       # FlyGym Fly.add_leg_adhesion(gain=40.0) = 40 uN = 1.64 bw
UPRIGHT_MIN = 0.90               # |torso z-axis . world z|
COORD_STEPS = (0.15, 0.08, 0.04, 0.02)
STARTS = 6
MAX_EVALS = 3000
WALL_BUDGET_S = 900.0
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--adhesion-bw-per-leg', type=float, default=ADHESION_BW_PER_LEG)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

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

    def legs_of(terms):
        out_legs = []
        for t in terms:
            lg = None
            for cand in (t['geom2'], t['geom1']):
                if cand[:2] in LEGS:
                    lg = cand[:2]
                    break
            out_legs.append(lg)
        return out_legs

    def evaluate(x, want_detail=False):
        qpos = key_qpos.copy()
        joints = {}
        for k, (leg, j) in enumerate(params):
            lo, hi = float(rng[j][0]), float(rng[j][1])
            v = float(np.clip(key_val[j][leg] + x[k], lo, hi))
            qpos[adr[j][leg]] = v
            joints[f'joint_{leg}{j}'] = v
        seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, PENETRATION_MM)
        data.qpos[:] = seated
        data.qvel[:] = 0
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        q = np.asarray(data.qpos[3:7], float)
        up_z = float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
        row = {'x': list(map(float, x)), 'joints_rad': joints, 'root_z_mm': float(root_z),
               'n_contacts': facts['contact_count'],
               'tarsus_only': bool(facts['contacts_are_feet_only_tarsus']),
               'non_tarsus_geoms': facts['non_foot_ground_geoms'],
               'thorax_height_mm': facts['thorax_height_mm'], 'torso_up_z': up_z,
               'upright': bool(up_z >= UPRIGHT_MIN),
               'active_joint_limit_rows': facts['active_joint_limit_rows']}
        if not (row['tarsus_only'] and row['upright'] and facts['contact_count'] > 0):
            row['reject'] = ('non-tarsus ground contact' if not row['tarsus_only']
                             else 'not upright' if not row['upright'] else 'no contact')
            row['L1'] = None
            return row
        # LP with actuators, friction cone, active joint limits AND a per-leg adhesion budget
        R, base = r15.actuator_moment_matrix(model, data, mj)
        target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
        terms = r15.contact_terms(model, data, mj)
        lims = r15.limit_terms(model, data, mj)
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        nv, nu = model.nv, model.nu
        cols = [R]
        bounds = [(0.0, 1.0)] * nu
        for t, b in zip(terms, B):
            cols.append(t['J'].T @ b)
            bounds += [(0.0, None)] * b.shape[1]
        n_lam = sum(b.shape[1] for b in B)
        # one tensile variable per contact, limited per leg
        for t in terms:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        n_p = len(terms)
        bounds += [(0.0, None)] * n_p
        A = np.hstack(cols + [np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(nu + n_lam + n_p), np.ones(2 * nv)])
        bounds += [(0.0, None)] * (2 * nv)
        leg_of = legs_of(terms)
        present = sorted({lg for lg in leg_of if lg})
        A_ub = np.zeros((len(present), A.shape[1]))
        for i, lg in enumerate(present):
            for k, l2 in enumerate(leg_of):
                if l2 == lg:
                    A_ub[i, nu + n_lam + k] = 1.0
        b_ub = np.full(len(present), args.adhesion_bw_per_leg * body_weight)
        res = linprog(c, A_eq=A, b_eq=target, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options=LP_OPTS)
        if res.x is None:
            row['L1'] = None; row['reject'] = 'LP failed: ' + res.message
            return row
        row['L1'] = float(res.fun)
        row['feasible'] = bool(res.fun < r15.FEASIBLE_SUM_TOL)
        if want_detail:
            row['adhesion_used_bw'] = [float(res.x[nu + n_lam + k]) / body_weight
                                       for k in range(n_p)]
            row['activation'] = res.x[:nu].tolist()
        return row

    rng_gen = np.random.default_rng(7)
    x0 = np.zeros(len(params))
    starts = [x0, np.array([0.0357 if j == 'Coxa_pitch' else 0.0 for _, j in params])]
    while len(starts) < STARTS:
        starts.append(rng_gen.normal(0.0, 0.10, size=len(params)))

    evals = 0
    history = []
    best = None
    rejects = {'non-tarsus ground contact': 0, 'not upright': 0, 'no contact': 0}
    for si, start in enumerate(starts):
        x = np.array(start, float)
        row = evaluate(x); evals += 1
        history.append({'start': si, 'eval': evals, 'L1': row.get('L1'),
                        'tarsus_only': row['tarsus_only'], 'upright': row['upright']})
        if row.get('reject') in rejects:
            rejects[row['reject']] += 1
        cur = row.get('L1')
        if cur is None:
            cur = np.inf
        for step in COORD_STEPS:
            improved = True
            sweeps = 0
            while improved and evals < MAX_EVALS and time.time() - t_start < WALL_BUDGET_S:
                improved = False
                sweeps += 1
                for k in range(len(params)):
                    for delta in (+step, -step):
                        if evals >= MAX_EVALS or time.time() - t_start > WALL_BUDGET_S:
                            break
                        cand = x.copy(); cand[k] += delta
                        r2 = evaluate(cand); evals += 1
                        if r2.get('reject') in rejects:
                            rejects[r2['reject']] += 1
                        v = r2.get('L1')
                        history.append({'start': si, 'eval': evals, 'L1': v,
                                        'tarsus_only': r2['tarsus_only'], 'upright': r2['upright']})
                        if v is not None and v < cur - 1e-9:
                            x, cur = cand, v
                            improved = True
                if sweeps >= 5:
                    break
        final = evaluate(x, want_detail=True); evals += 1
        if final.get('L1') is not None and (best is None or final['L1'] < best['L1']):
            best = final

    report = {
        'status': 'STANCE_SEARCH_WITH_ADHESION',
        'question': 'with the engine documented leg adhesion present, is there a per-leg posture that is '
                    'statically supportable, tarsus-only and upright?',
        'source_sha256': {str(XML): r15.digest(XML)},
        'config': {'per_leg_params': [f'{l}:{j}' for l, j in params], 'n_params': len(params),
                   'penetration_mm': PENETRATION_MM,
                   'adhesion_bw_per_leg': args.adhesion_bw_per_leg,
                   'adhesion_reference': 'FlyGym Fly.add_leg_adhesion(gain=40.0) = 40 uN = 1.64 body '
                                         'weights per leg, inside the measured 1-2 bw/leg range',
                   'upright_min_up_z': UPRIGHT_MIN, 'coord_steps': list(COORD_STEPS),
                   'starts': len(starts), 'max_evals': MAX_EVALS, 'wall_budget_s': WALL_BUDGET_S},
        'evaluations': evals, 'wall_seconds': float(time.time() - t_start),
        'reject_counts': rejects,
        'best': best,
        'feasible_tarsus_only_upright_found': bool(best is not None and best.get('feasible', False)),
        'history': history,
        'limitations': [
            'Fixed-pose static force balance: NOT stability, NOT controllability, NOT walking.',
            'Adhesion is a bounded tensile normal force per leg in the LP, not pulvillus contact '
            'mechanics.',
            'Actuator force is affine in activation only at the frozen pose; force-length/velocity and '
            'moment arms frozen; friction cone linearised.',
            'Search is bounded: per-leg coxa/trochanter/tibia pitch only; yaw/roll and tarsus joints were '
            'not searched, so this is not a proof about all postures.',
            'No model file is modified; the canonical asset stays byte-identical.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_with_adhesion.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    pts = [(h['eval'], h['L1']) for h in history if h['L1'] is not None]
    if pts:
        axs[0].semilogy([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    axs[0].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    axs[0].set_xlabel('pose evaluation'); axs[0].set_ylabel('LP L1 residual (force units)')
    axs[0].set_title('Tarsus-only upright poses only\n(with %.2f bw/leg adhesion)'
                     % args.adhesion_bw_per_leg)
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, which='both')
    labels = list(rejects) + ['feasible']
    vals = [rejects[k] for k in rejects] + [1 if report['feasible_tarsus_only_upright_found'] else 0]
    axs[1].bar(range(len(vals)), vals, color=['tab:red', 'tab:orange', 'tab:gray', 'tab:green'])
    axs[1].set_xticks(range(len(vals)))
    axs[1].set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8)
    axs[1].set_ylabel('poses')
    axs[1].set_title('Where the candidates went')
    axs[1].grid(alpha=.3, axis='y')
    fig.suptitle('Stance search WITH documented adhesion: force balance only, NOT stability, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'stance_with_adhesion.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'reject_counts': rejects,
                      'best_L1': (best or {}).get('L1'),
                      'best_feasible': (best or {}).get('feasible'),
                      'best_tarsus_only': (best or {}).get('tarsus_only'),
                      'best_upright': (best or {}).get('upright'),
                      'best_thorax_mm': (best or {}).get('thorax_height_mm'),
                      'adhesion_used_bw': (best or {}).get('adhesion_used_bw')}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
