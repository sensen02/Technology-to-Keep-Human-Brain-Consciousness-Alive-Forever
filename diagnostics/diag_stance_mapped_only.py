"""Is there a standable stance whose solution uses ONLY the mapped actuators?

Round31 showed the round30 stance's feasible solution needs 39 active muscles, 21 of which have
no motor-neuron mapping in this project (two families: the merged Miller 28/30 promotor pool and
the a/b sterno-tergo extensor split), and that restricting to the mapped actuators makes the
stance infeasible.

So the question becomes: does ANY tarsus-only upright posture have a feasible solution using
only actuators this project can actually drive?  This file searches the pose space with the
actuator set RESTRICTED to the mapped ones (the 44 name-matched candidates plus the single
v888 exact addition), a per-leg adhesion budget within the measured range, and the friction
cone, minimising the static residual.

If a pose reaches the tolerance, standing becomes reachable with today's mapping (subject to
the neural side actually producing the pattern).  If none does, the shortfall is reported and
the missing pools are named as the requirement.

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

OUT = ROOT / 'outputs/stance_mapped_only_round32'
XML = r15.XML
LEGS = r16.LEGS
# Broader pose family: all three axes of the coxa and trochanter plus tibia pitch, so the
# search is not restricted to the sagittal plane the earlier searches used.
PARAM_JOINTS = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
                'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll',
                'Tibia_pitch')
PENETRATION_MM = -0.050
ADHESION_BW = 2.0            # top of the measured range, so the contact side is as favourable as biology allows
COORD_STEPS = (0.10, 0.05, 0.02, 0.01)
STARTS = 10
MAX_EVALS = 6000
WALL_BUDGET_S = 900.0
LP_OPTS = {'presolve': True, 'time_limit': 60, 'maxiter': 20000}
SEEDS = (ROOT / 'outputs/stance_joint_feasible_round30/stance_for_runner.json',
         ROOT / 'outputs/stance_with_adhesion_round26/stance_with_adhesion.json')
MANIFEST = ROOT / 'outputs/muscle_anatomy_resolution/resolution.json'
OVERLAY = ROOT / 'outputs/motor_overlay_v888/motor_overlay_v888.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--adhesion-bw', type=float, default=ADHESION_BW)
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

    manifest = json.loads(MANIFEST.read_text())
    mapped = {r['actuator'] for r in manifest['actuators'] if r['candidates']}
    overlay = json.loads(OVERLAY.read_text())
    mapped |= set(overlay['exact_muscle_name'])
    mapped_idx = [i for i, n in enumerate(names) if n in mapped]
    assert mapped_idx, 'no mapped actuators found'

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
        return {'R': R[:, mapped_idx], 'target': target, 'terms': terms, 'lim_cols': lim_cols,
                'joints': joints, 'facts': facts, 'up_z': up_z, 'root_z': root_z}, None

    def residual(ctx):
        """L1 residual with mapped actuators only, friction cone + per-leg adhesion budget."""
        B = r15.cone_basis(model, data, ctx['terms'], r15.CONE_DIRECTIONS)
        nu_m, nv = ctx['R'].shape[1], model.nv
        n_lam = sum(b.shape[1] for b in B)
        n_p = len(ctx['terms'])
        nlim = ctx['lim_cols'].shape[1]
        cols, bnds = [ctx['R']], [(0, 1)] * nu_m
        for t, b in zip(ctx['terms'], B):
            cols.append(t['J'].T @ b); bnds += [(0, None)] * b.shape[1]
        for t in ctx['terms']:
            cols.append(-t['J'].T @ t['normal'].reshape(-1, 1))
        bnds += [(0, None)] * n_p + [(0, None)] * nlim
        A = np.hstack(cols + [ctx['lim_cols'], np.eye(nv), -np.eye(nv)])
        c = np.concatenate([np.zeros(nu_m + n_lam + n_p + nlim), np.ones(2 * nv)])
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
                    A_ub[i, nu_m + n_lam + k] = 1.0
        r = linprog(c, A_eq=A, b_eq=ctx['target'], A_ub=A_ub,
                    b_ub=np.full(len(present), args.adhesion_bw * body_weight), bounds=bnds,
                    method='highs', options=LP_OPTS)
        return None if r.x is None else float(r.fun)

    x_seed = np.zeros(len(params))
    try:
        s0 = json.loads(SEEDS[0].read_text())
        for k, (leg, j) in enumerate(params):
            x_seed[k] = s0['joints_rad'][f'joint_{leg}{j}'] - key_val[j][leg]
    except Exception:
        pass
    gen = np.random.default_rng(31)
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
                            x, cur = cand, v
                            improved = True
                            if best is None or v < best['L1']:
                                best = {'L1': v, 'x': cand.tolist(), 'joints_rad': ctx2['joints'],
                                        'thorax_height_mm': ctx2['facts']['thorax_height_mm'],
                                        'up_z': ctx2['up_z']}
                if sweeps >= 6:
                    break

    report = {
        'status': 'STANCE_SEARCH_MAPPED_ACTUATORS_ONLY',
        'question': 'does any tarsus-only upright posture have a static solution using ONLY the '
                    'actuators this project has a motor-neuron mapping for?',
        'actuators_available': {'count': len(mapped_idx), 'names': [names[i] for i in mapped_idx],
                                'excluded_count': model.nu - len(mapped_idx)},
        'config': {'penetration_mm': PENETRATION_MM, 'adhesion_bw_per_leg': args.adhesion_bw,
                   'starts': len(starts), 'max_evals': MAX_EVALS},
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'rejection_reasons': reasons, 'best': best,
        'feasible_with_mapped_only': bool(best and best['L1'] < r15.FEASIBLE_SUM_TOL),
        'source_sha256': {str(XML): r15.digest(XML), str(MANIFEST): r15.digest(MANIFEST),
                          str(OVERLAY): r15.digest(OVERLAY)},
        'history': history,
        'limitations': [
            'Fixed-pose static balance: NOT stability, NOT a controller, NOT walking.',
            'Friction cone linearised; actuator force affine in activation at the frozen pose.',
            'Bounded search over per-leg pitch offsets only.',
            'No model file is modified.'],
        'scope': 'diagnostic only',
    }
    assert r15.digest(XML) == xml_before
    (out / 'stance_mapped_only.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts = [(h['eval'], h['L1']) for h in history if h.get('L1') is not None]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    if pts:
        axs[0].semilogy([p[0] for p in pts], [p[1] for p in pts], '.-', ms=3, lw=.7)
    axs[0].axhline(r15.FEASIBLE_SUM_TOL, ls='--', c='green', label='feasibility tolerance')
    axs[0].set_xlabel('pose evaluation')
    axs[0].set_ylabel('L1 residual, mapped actuators only')
    axs[0].set_title('Can the mapped actuators alone hold a stance?')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3, which='both')
    labs = list(reasons) or ['no rejections']
    axs[1].barh(range(len(labs)), [reasons.get(l, 0) for l in labs], color='tab:red')
    axs[1].set_yticks(range(len(labs))); axs[1].set_yticklabels([l[:44] for l in labs], fontsize=8)
    axs[1].set_xlabel('poses rejected'); axs[1].set_title('Rejections')
    axs[1].grid(alpha=.3, axis='x')
    fig.suptitle('Mapped-actuators-only stance search: diagnostic only, NOT walking')
    fig.tight_layout(); fig.savefig(out / 'stance_mapped_only.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_seconds': round(report['wall_seconds'], 1),
                      'rejection_reasons': reasons,
                      'best_L1': (best or {}).get('L1'),
                      'best_as_fraction_of_body_weight': ((best or {}).get('L1') or 0) / body_weight
                      if best else None,
                      'best_thorax_mm': (best or {}).get('thorax_height_mm'),
                      'feasible_with_mapped_only': report['feasible_with_mapped_only'],
                      'actuators_available': len(mapped_idx)}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
