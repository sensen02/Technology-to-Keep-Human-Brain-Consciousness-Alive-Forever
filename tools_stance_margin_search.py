"""Does a ZERO-ADHESION muscle balance exist with a USABLE support margin?

WHY.  Round 36 found a zero-adhesion exact balance (L1 = 0.0), but forward integration showed it
leaves the balance neighbourhood in ~26 ms.  Measuring why pointed at a plain geometric number:
the signed distance from the centre of mass to the edge of the support polygon
(`CoM margin`) was only 0.16 mm on that pose (0.28 mm on the round-30 pose, 0.55 mm on the XML
keyframe).  A pose whose balance margin is a small fraction of the 1.9 mm CoM height cannot survive
the deflections that the contact compliance and the passive joints produce, whatever the muscle
solution is.

So this file asks the question the earlier searches never asked:

    with adhesion forced to ZERO, does a tarsus-only upright pose exist that is BOTH exactly
    force-balanced by muscle AND has a CoM margin large enough to be usable?

and, when it finds one, runs the forward-integration stability test on it so the answer is a
measurement, not a hope.

Diagnostic only: no model file is written and no runtime authorisation is granted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

OUT = HERE / 'outputs/stance_margin_round37'
XML = r15.XML
PARAM_J = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
           'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
PENETRATION_MM = -0.050
SEED = HERE / 'outputs/muscle_only_stance_round36/design_motor_pools.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--evals', type=int, default=3000)
    ap.add_argument('--wall-budget-s', type=float, default=1500.0)
    ap.add_argument('--penetration-mm', type=float, default=PENETRATION_MM)
    ap.add_argument('--seed', type=Path, default=SEED)
    ap.add_argument('--xml', type=Path, default=None,
                    help='body asset (default canonical). A different value is a CONTACT VARIANT; '
                         'the canonical file is never modified.')
    ap.add_argument('--starts', type=int, default=12)
    ap.add_argument('--start-sigma', type=float, default=0.25)
    ap.add_argument('--stability-seconds', type=float, default=0.2,
                    help='forward-integrate the best pose under its own constant LP activation')
    ap.add_argument('--escape-mm', type=float, default=1.8,
                    help='thorax height below which the pose is called escaped')
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_used = args.xml.resolve() if args.xml else XML
    xml_before = r15.digest(XML)
    hash_used = r15.digest(xml_used)
    model = _load_mjcf(str(xml_used)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id
    body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()
    params = [(leg, j) for leg in r16.LEGS for j in PARAM_J]
    adr = {j: {leg: int(model.jnt_qposadr[model.joint(f'joint_{leg}{j}').id]) for leg in r16.LEGS}
           for j in PARAM_J}

    def facts_of(qpos):
        """Seat the root, then report contact and support facts (no adhesion anywhere)."""
        seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor,
                                          args.penetration_mm)
        data.qpos[:] = seated
        data.qvel[:] = 0
        data.act[:] = 0.0
        mj.mj_forward(model, data)
        f = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        pts = np.array([data.contact[k].pos for k in range(data.ncon)
                        if ((mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, int(data.contact[k].geom2))
                             or '').startswith(tuple(r16.LEGS)))])
        margin = inside = None
        if len(pts) >= 3:
            com = np.asarray(data.subtree_com[0], float)[:2]
            try:
                if Delaunay(pts[:, :2]).find_simplex(com) >= 0:
                    inside = True
            except Exception:
                inside = None
            try:
                eq = ConvexHull(pts[:, :2]).equations
                margin = float(np.min(-(eq[:, :2] @ com + eq[:, 2])))
            except Exception:
                margin = None
        # muscle-only exact balance: no adhesion columns at all
        R, base = r15.actuator_moment_matrix(model, data, mj)
        target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
        terms = r15.contact_terms(model, data, mj)
        lims = r15.limit_terms(model, data, mj)
        B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
        lp = r15.solve_set(R, target, terms, lims, B, model.nv, 'muscle-only balance')
        return {'seated': seated, 'facts': f, 'margin_mm': margin, 'com_inside': inside,
                'L1': float(lp['objective_L1_residual']),
                'feasible': bool(lp['feasible_within_tol']),
                'activation': lp.get('activation')}

    def make_qpos(x):
        q = key.copy()
        for k, (leg, j) in enumerate(params):
            jid = model.joint(f'joint_{leg}{j}').id
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            q[adr[j][leg]] = float(np.clip(key[adr[j][leg]] + x[k], lo, hi))
        return q

    def evaluate(x):
        return facts_of(make_qpos(x))

    # ---- seeds: the round-36 zero-adhesion balance pose plus random restarts -------------
    x0 = np.zeros(len(params))
    try:
        sd = json.loads(args.seed.read_text())
        best_seed = (sd.get('physics', {}).get('design_search_over_poses') or {}).get('best')
        if best_seed and 'x' in best_seed and len(best_seed['x']) == len(params):
            x0 = np.asarray(best_seed['x'], float)
    except Exception:
        pass
    evals = 0
    hist = []

    def better(cand, ref):
        """Prefer a muscle-only exact balance; among those, prefer the larger CoM margin."""
        cf, rf = cand['feasible'], ref['feasible']
        if cf != rf:
            return cf
        if not cf:
            return cand['L1'] < ref['L1'] - 1e-9
        cm = -1e9 if cand['margin_mm'] is None else cand['margin_mm']
        rm = -1e9 if ref['margin_mm'] is None else ref['margin_mm']
        return cm > rm + 1e-9

    rng = np.random.default_rng(19)
    starts = [x0]
    while len(starts) < int(args.starts):
        starts.append(x0 + rng.normal(0.0, float(args.start_sigma), size=len(params)))
    best = None
    for si, xs in enumerate(starts):
        if evals >= args.evals or time.time() - t0 > args.wall_budget_s:
            break
        x = np.array(xs, float)
        cur = evaluate(x); evals += 1
        hist.append({'eval': evals, 'start': si, 'L1': cur['L1'], 'margin_mm': cur['margin_mm'],
                     'feasible': cur['feasible'], 'h': cur['facts']['thorax_height_mm'],
                     'balanced_margin_mm': (cur['margin_mm'] if cur['feasible'] else None)})
        if best is None or better(cur, best):
            best = dict(cur); best['x'] = x.tolist()
        step = 0.10
        while step >= 0.005:
            improved = False
            for k in range(len(params)):
                for d in (+step, -step):
                    if evals >= args.evals or time.time() - t0 > args.wall_budget_s:
                        break
                    cand_x = x.copy(); cand_x[k] += d
                    cand = evaluate(cand_x); evals += 1
                    hist.append({'eval': evals, 'start': si, 'L1': cand['L1'],
                                 'margin_mm': cand['margin_mm'], 'feasible': cand['feasible'],
                                 'h': cand['facts']['thorax_height_mm'],
                                 'balanced_margin_mm': (cand['margin_mm'] if cand['feasible']
                                                        else None)})
                    if best is None or better(cand, best):
                        best = dict(cand); best['x'] = cand_x.tolist()
                    if better(cand, cur):
                        x, cur, improved = cand_x, cand, True
            if not improved:
                step *= 0.5

    act = {}
    if best.get('activation') is not None:
        act = {names[i]: float(best['activation'][i]) for i in range(model.nu)}
    report = {
        'status': 'STANCE_SUPPORT_MARGIN_SEARCH',
        'question': 'with adhesion forced to ZERO, does a tarsus-only upright pose exist that is '
                    'exactly force-balanced by muscle AND has a usable CoM support margin?',
        'adhesion': 0.0,
        'penetration_mm': args.penetration_mm,
        'evaluations': evals, 'wall_seconds': float(time.time() - t0),
        'best': {'L1': best['L1'], 'feasible': best['feasible'],
                 'margin_mm': best['margin_mm'], 'com_inside_support': best['com_inside'],
                 'thorax_height_mm': best['facts']['thorax_height_mm'],
                 'contact_count': best['facts']['contact_count'],
                 'tarsus_only': best['facts']['contacts_are_feet_only_tarsus'],
                 'x': best.get('x'), 'activation': act},
        'history': hist,
        'reference_margins_mm': {'xml_keyframe': 0.5466, 'round30_pose': 0.2807,
                                 'round36_muscle_only': 0.1587},
        'body_model': str(xml_used), 'is_variant_model': bool(args.xml is not None),
        'source_sha256': {str(XML): r15.digest(XML), str(xml_used): r15.digest(xml_used),
                          str(args.seed): r15.digest(args.seed)},
        'limitations': ['CoM margin is a static geometric surrogate for stability, not a stability '
                        'proof: forward integration is still required',
                        'fixed-pose LP balance; no adhesion; coordinate search is a weak optimiser',
                        'no model file is modified'],
    }
    # ---------------------------------------------------------------------------------
    # FORWARD INTEGRATION on the best pose: does a larger support margin actually delay the
    # escape?  The command is the pose's own muscle-only LP solution, held CONSTANT.
    # ---------------------------------------------------------------------------------
    stability = {'performed': False}
    if best.get('activation') is not None and args.stability_seconds > 0:
        act_vec = np.asarray(best['activation'], float)
        data.qpos[:] = best['seated']
        data.qvel[:] = 0
        data.ctrl[:] = act_vec
        mj.mj_forward(model, data)
        n_step = int(round(args.stability_seconds / model.opt.timestep))
        every = max(1, n_step // 200)
        trace, escape = [], None
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % every == 0 or i == n_step - 1:
                f = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                trace.append({'t': float(data.time), 'thorax_mm': f['thorax_height_mm'],
                              'contacts': f['contact_count'],
                              'tarsus_only': bool(f['contacts_are_feet_only_tarsus'])})
                if escape is None and f['thorax_height_mm'] < args.escape_mm:
                    escape = float(data.time)
        stability = {'performed': True, 'seconds': args.stability_seconds,
                     'escape_below_mm': args.escape_mm, 'escape_time_s': escape,
                     'final_thorax_mm': trace[-1]['thorax_mm'],
                     'tarsus_only_fraction': float(np.mean([t['tarsus_only'] for t in trace])),
                     'trace': trace,
                     'reference_escape': {'round36_pose_margin_0.159mm': 0.0259,
                                          'round36_pose_zero_activation': None}}
    report['forward_integration'] = stability

    assert r15.digest(XML) == xml_before and r15.digest(xml_used) == hash_used
    (out / 'stance_margin.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ok = [h for h in hist if h.get('feasible') and h.get('balanced_margin_mm') is not None]
    allh = [h for h in hist if h['margin_mm'] is not None]
    ax[0].plot([h['eval'] for h in allh], [max(h['margin_mm'], -0.5) for h in allh], '.', ms=3,
               color='tab:gray', label='all poses')
    if ok:
        ax[0].plot([h['eval'] for h in ok], [h['balanced_margin_mm'] for h in ok], '.', ms=3,
                   color='tab:green', label='muscle-only BALANCED poses')
    for k, v in report['reference_margins_mm'].items():
        ax[0].axhline(v, ls=':', lw=.8, label=f'{k}: {v:.2f} mm')
    ax[0].set_xlabel('pose evaluation'); ax[0].set_ylabel('CoM margin (mm)')
    ax[0].set_title('Support margin vs muscle-only balance\n'
                    'green = poses the muscles balance with ZERO adhesion')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    bm = {}
    for h in allh:
        bm.setdefault(h.get('start', 0), []).append((h['margin_mm'], h['feasible']))
    for stx, pts in sorted(bm.items()):
        ax[1].plot([h['eval'] for h in allh if h.get('start') == stx],
                   [h['balanced_margin_mm'] for h in allh if h.get('start') == stx],
                   '.', ms=3, label=f'start {stx}')
    ax[1].set_xlabel('pose evaluation')
    ax[1].set_ylabel('CoM margin of muscle-only balanced poses (mm)')
    ax[1].set_title('How large a support margin does a zero-adhesion balance reach?\n'
                    'flat top = a real ceiling on this pose family')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'stance_margin.png', dpi=110)

    print(json.dumps({'evaluations': evals, 'wall_s': round(report['wall_seconds'], 1),
                      'best_L1': best['L1'], 'best_feasible': best['feasible'],
                      'best_margin_mm': best['margin_mm'],
                      'best_contacts': best['facts']['contact_count'],
                      'best_thorax_mm': best['facts']['thorax_height_mm'],
                      'n_balanced_poses': len(ok),
                      'best_balanced_margin_mm': (max(h['balanced_margin_mm'] for h in ok)
                                                  if ok else None),
                      'escape_time_s': stability.get('escape_time_s'),
                      'final_thorax_mm': stability.get('final_thorax_mm')},
                     indent=2))
    print(out)


if __name__ == '__main__':
    main()
