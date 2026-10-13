"""Can ANY tarsus-only upright stance be held if each a/b pair is ONE muscle?

The model splits several muscles into an `_a` and a `_b` actuator; whether those are fibre
clusters of one muscle (which cannot be activated independently in the animal) or separate
muscles is exactly what the annotations do not say.  Two earlier results bracket the question:
  * round28 showed that coupling each a/b pair into one activation removes force directions
    (rank 48 -> 21-45) and, at the sampled poses, puts exact balance outside the reachable span;
  * rounds 31-33 showed the stance solutions found so far depend on actuators with no mapped
    motor pool, and that the mapped actuators alone cannot hold any pose.

This file closes the bracket by asking the physical question directly:

    with every a/b pair constrained to ONE activation (the fibre-cluster reading), does any
    tarsus-only upright posture have a static solution inside [0,1] under the friction cone and
    an adhesion budget within the measured range?

  * If yes -> a physiologically self-consistent standable stance exists, and the a/b actuators
    need only one motor pool per pair (the documented pool can then be used as-is).
  * If no  -> the model cannot stand with physiological co-activation either, and the earlier
    stance solutions were relying on independent activation of what the model itself presents as
    subdivisions of one muscle.

Diagnostic only: fixed-pose force balance; NOT stability, NOT a controller, NOT walking.
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

OUT = ROOT / 'outputs/stance_ab_coupled_round33'
XML = r15.XML
LEGS = r16.LEGS
PARAM_JOINTS = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
                'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll',
                'Tibia_pitch')
PENETRATION_MM = -0.050
ADHESION_BW = 2.0
COORD_STEPS = (0.10, 0.05, 0.02)
STARTS = 6
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
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]

    # build the coupled column set: all actuators, but every `_a`/`_b` pair shares one variable
    pairs, used = [], set()
    for n in names:
        if n.endswith('_a'):
            b = n[:-2] + '_b'
            if b in names:
                pairs.append((names.index(n), names.index(b)))
                used.update({names.index(n), names.index(b)})
    singles = [i for i in range(model.nu) if i not in used]

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
        cols = [R[:, i].reshape(-1, 1) for i in singles]
        labels = [names[i] for i in singles]
        for a, b in pairs:
            cols.append((R[:, a] + R[:, b]).reshape(-1, 1))
            labels.append(f'{names[a]}+{names[b]}')
        return {'R': np.hstack(cols), 'labels': labels, 'target': target, 'terms': terms,
                'lim_cols': lim_cols, 'joints': joints, 'facts': facts, 'up_z': up_z}, None

    def residual(ctx):
        B = r15.cone_basis(model, data, ctx['terms'], r15.CONE_DIRECTIONS)
        ncsv = ctx['R'].shape[1]
        nv = model.nv
        n_lam = sum(b.shape[1] for b in B)
        n_p = len(ctx['terms'])
        nlim = ctx['lim_cols'].shape[1]
        cols, bnds = [ctx['R']], [(0, 1)] * ncsv
        for t, b in zip(ctx['terms'], B):
            cols.append(t['J'].T @ b); bnds += [(0, None)] * b.shape[1]
        for t in ctx['terms']:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bnds += [(0, None)] * n_p + [(0, None)] * nlim
        A = np.hstack(cols + [ctx['lim_cols'], np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(ncsv + n_lam + n_p + nlim), np.ones(2 * nv)])
        bnds += [(0, None)] * (2 * nv)
        leg_of = []
        for t in ctx['terms']:
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
                    A_ub[i, ncsv + n_lam + k] = 1.0
        r = linprog(c, A_eq=A, b_eq=ctx['target'], A_ub=A_ub,
                    b_ub=np.full(len(present), ADHESION_BW * body_weight), bounds=bnds,
                    method='highs', options=LP_OPTS)
        return None if r.x is None else float(r.fun)

    x_seed = np.zeros(len(params))
    try:
        s0 = json.loads(SEED.read_text())['best']
        for k, (leg, j) in enumerate(params):
            if f'joint_{leg}{j}' in s0['joints_rad']:
                x_seed[k] = s0['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg]
    except Exception:
        pass
    gen = np.random.default_rng(41)
    starts = [x_seed, np.zeros(len(params))]
    while len(starts) < STARTS:
        starts.append(x_seed + gen.normal(0, 0.10, size=len(params)))

    evals, history, reasons, best = 0, [], {}, None
    for si, start in enumerate(starts):
        x = np.array(start, float)
        ctx, why = setup(x); evals += 1
        cur = np.inf
        if ctx is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            v = residual(ctx)
            if v is not None:
                cur = v
                if best is None or v < best['L1']:
                    best = {'L1': v, 'x': x.tolist(), 'joints_rad': ctx['joints'],
                            'thorax_height_mm': ctx['facts']['thorax_height_mm'], 'up_z': ctx['up_z']}
        history.append({'start': si, 'eval': evals, 'L1': None if cur == np.inf else cur})
        for step in COORD_STEPS:
            improved, sweeps = True, 0
            while improved and evals < MAX_EVALS and time.time() - t0 < WALL_BUDGET_S:
                improved = False; sweeps += 1
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
                            v = residual(ctx2)
                        history.append({'start': si, 'eval': evals, 'L1': v})
                        if v is not None and v < cur - 1e-9:
                            x, cur = cand, v; improved = True
                            if best is None or v < best['L1']:
                                best = {'L1': v, 'x': cand.tolist(), 'joints_rad': ctx2['joints'],
                                        'thorax_height_mm': ctx2['facts']['thorax_height_mm'],
                                        'up_z': ctx2['up_z']}
                if sweeps >= 6:
                    break

    report = {
        'status': 'STANCE_SEARCH_AB_COUPLED',
        'question': 'with every a/b pair constrained to ONE activation, can any tarsus-only upright '
                    'stance be statically held?',
        'coupling': {'pairs': [f'{names[a]} + {names[b]}' for a, b in pairs],
                     'independent_actuators': len(singles),
                     'total_variables': len(singles) + len(pairs)},
        'config': {'penetration_mm': PENETRATION_MM, 'adhesion_bw_per_leg': ADHESION_BW,
                   'n_params': len(params), 'starts': len(starts), 'max_evals': MAX_EVALS},
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'rejection_reasons': reasons, 'best': best,
        'feasible_under_coupling': bool(best and best['L1'] < r15.FEASIBLE_SUM_TOL),
        'source_sha256': {str(XML): r15.digest(XML), str(SEED): r15.digest(SEED)},
        'history': history,
        'limitations': [
            'Fixed-pose static balance: NOT stability, NOT a controller, NOT walking.',
            'Coupling is a HYPOTHESIS about the a/b actuators (fibre clusters of one muscle); the '
            'annotations do not settle it.',
            'Friction cone linearised, actuator force affine in activation at the frozen pose.',
            'Bounded search over the leg joint offsets listed in config.',
            'No model file is modified.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_ab_coupled.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts = [(h['eval'], h['L1']) for h in history if h.get('L1') is not None]
    fig, ax = plt.subplots(figsize=(9, 5))
    if pts:
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    ax.axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    ax.set_xlabel('pose evaluation'); ax.set_ylabel('L1 residual with a/b coupled')
    ax.set_title('Can the body stand if each a/b pair is one muscle?\n'
                 'Static balance only: NOT stability, NOT walking')
    ax.legend(fontsize=8); ax.grid(alpha=.3, which='both')
    fig.tight_layout(); fig.savefig(out / 'stance_ab_coupled.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'rejection_reasons': reasons,
                      'best_L1': (best or {}).get('L1'),
                      'best_fraction_of_body_weight': ((best or {}).get('L1') or 0) / body_weight
                      if best else None,
                      'feasible_under_coupling': report['feasible_under_coupling'],
                      'variables': report['coupling']['total_variables']}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
