"""PERTURBATION REJECTION: does a pose that is balanced by MUSCLE ALONE stay put, or run away?

Round 35 showed that the earlier stance claims rested on an LP that was allowed to use adhesion,
and that the same LP solution, held as a constant command, collapses within tens of milliseconds.
It also produced the question that matters: with adhesion forced to ZERO, is there a pose the
muscles can balance?  Round 36's search says yes (L1 = 0.0 with a zero per-leg adhesion budget).

This file asks the next physical question about that pose, and it is the one the goal actually
needs: STABILITY, not balance.

For each case the body is seated at the pose, commanded a fixed muscle activation, and integrated
forward.  The body's own state is then read back as a trajectory, and the file reports
  * the drift over a control window with NO perturbation (is the command a fixed point at all),
  * the response to a small physical perturbation (a root velocity, or one foot lifted by delta),
    measured as whether the state returns to the unperturbed trajectory's neighbourhood.

Nothing here is a controller, a schedule or a target: the command is a constant, the perturbation
is a physical displacement, and the only thing the program does is simulate and measure.

Diagnostic only: no model file is written and no runtime authorisation is granted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'diagnostics'))

import diag_static_support_feasibility as r15  # noqa: E402
import diag_posture_support_search as r16      # noqa: E402

DESIGN = HERE / 'outputs/muscle_only_stance_round36/design_motor_pools.json'
OUT = HERE / 'outputs/stability_round36'
XML = r15.XML


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--design', type=Path, default=DESIGN)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--seconds', type=float, default=0.25)
    ap.add_argument('--penetration-mm', type=float, default=-0.050)
    ap.add_argument('--root-speed-mm-s', type=float, default=5.0)
    ap.add_argument('--foot-lift-mm', type=float, default=0.05)
    ap.add_argument('--settle-window-s', type=float, default=0.05)
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_before = r15.digest(XML)
    model = _load_mjcf(str(XML)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id
    body_weight = float(mj.mj_getTotalmass(model) * abs(float(model.opt.gravity[2])))

    design = json.loads(args.design.read_text())
    best = (design.get('physics', {}).get('design_search_over_poses') or {}).get('best')
    if best is None or 'activation' not in best:
        raise ValueError('this design file has no solved activation pattern')
    muscle_only = float(design['physics']['adhesion_bw_per_leg']) == 0.0

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()

    PARAM_J = ('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll',
               'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll', 'Tibia_pitch')
    params = [(leg, j) for leg in r16.LEGS for j in PARAM_J]
    qpos = key.copy()
    if 'x' in best and len(best['x']) == len(params):
        for k, (leg, j) in enumerate(params):
            jid = model.joint(f'joint_{leg}{j}').id
            lo, hi = float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])
            qpos[model.jnt_qposadr[jid]] = float(np.clip(key[model.jnt_qposadr[jid]] + best['x'][k],
                                                          lo, hi))
    else:
        for jn, v in best.get('joints_rad', {}).items():
            qpos[model.jnt_qposadr[model.joint(jn).id]] = float(v)
    seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor,
                                      args.penetration_mm)
    seated = np.asarray(seated, float)

    ctrl = np.zeros(model.nu)
    for k, v in best['activation'].items():
        if k in names:
            ctrl[names.index(k)] = float(v)
        else:
            for a in names:
                if a.startswith(k + '_a') or a.startswith(k + '_b'):
                    ctrl[names.index(a)] = float(v)
    n_nonzero = int(np.count_nonzero(ctrl))

    n_step = int(round(float(args.seconds) / model.opt.timestep))
    every = max(1, n_step // 400)

    def run(label, root_vel=None, lift_foot=None):
        mj.mj_resetDataKeyframe(model, data, 0)
        data.time = 0.0
        data.qpos[:] = seated
        data.qvel[:] = 0
        if root_vel is not None:
            data.qvel[0:3] = np.asarray(root_vel, float)
        if lift_foot is not None:
            # a pure vertical displacement of ONE foot: move the joint that lifts that tarsus
            jid = model.joint(f'joint_{lift_foot[0]}Tibia_pitch').id
            data.qpos[model.jnt_qposadr[jid]] += float(lift_foot[1])
        mj.mj_forward(model, data)
        data.ctrl[:] = ctrl
        trace = []
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % every == 0 or i == n_step - 1:
                f = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                q = np.asarray(data.qpos[3:7], float)
                trace.append({
                    't': float(data.time),
                    'thorax_mm': f['thorax_height_mm'],
                    'contacts': f['contact_count'],
                    'tarsus_only': bool(f['contacts_are_feet_only_tarsus']),
                    'root_x_model': float(data.qpos[0]),
                    'root_y_model': float(data.qpos[1]),
                    'root_z_model': float(data.qpos[2]),
                    'up_z': float(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)),
                    'qvel_max': float(np.abs(data.qvel).max()),
                })
        return trace

    cases = {}
    base = run('unperturbed')
    cases['unperturbed'] = {'trace': base}
    cases['root_speed'] = {'trace': run('root_speed',
                                        root_vel=(0.0, 0.0, float(args.root_speed_mm_s)))}
    cases['foot_lift'] = {'trace': run('foot_lift',
                                       lift_foot=('LF', float(args.foot_lift_mm)))}

    h0 = seated[2]
    for name, c in cases.items():
        tr = c['trace']
        c['thorax_first_mm'] = tr[0]['thorax_mm']
        c['thorax_final_mm'] = tr[-1]['thorax_mm']
        c['thorax_min_mm'] = min(t['thorax_mm'] for t in tr)
        c['contacts_first'] = tr[0]['contacts']
        c['contacts_final'] = tr[-1]['contacts']
        c['tarsus_only_fraction'] = float(np.mean([t['tarsus_only'] for t in tr]))
        c['up_z_min'] = min(t['up_z'] for t in tr)
        # deviation from the unperturbed trajectory, over the whole window
        dev = [abs(t['thorax_mm'] - b['thorax_mm']) for t, b in zip(tr, base)]
        c['max_deviation_from_unperturbed_mm'] = float(max(dev))
        # rejection ratio: deviation in the second half vs the first half of the window
        half = len(dev) // 2
        first, second = max(dev[1:half] or [0.0]), max(dev[half:] or [0.0])
        c['deviation_first_half_mm'] = float(first)
        c['deviation_second_half_mm'] = float(second)
        c['rejected'] = bool(second < first and second < 0.05)

    report = {
        'status': 'MUSCLE_ONLY_STABILITY_ROUND36',
        'question': 'given a pose that the muscles balance with ZERO adhesion, does the body stay '
                    'in the neighbourhood of that state, and does it reject a small physical '
                    'perturbation?',
        'pose': {'thorax_height_mm': float(r16.pose_facts(model, data, mj, tarsus_names,
                                                          thorax_id)['thorax_height_mm']),
                 'root_z_mm': float(seated[2]), 'penetration_mm': args.penetration_mm},
        'activation': {'n_nonzero_actuators': n_nonzero, 'constant': True,
                       'muscle_only_balance': bool(muscle_only)},
        'perturbations': {'root_speed_mm_s': float(args.root_speed_mm_s),
                          'foot_lift_mm': float(args.foot_lift_mm)},
        'seconds': float(args.seconds), 'body_weight_units': body_weight,
        'cases': cases,
        'source_sha256': {str(XML): r15.digest(XML), str(args.design): r15.digest(args.design)},
        'limitations': [
            'A constant activation is an open-loop command: not a controller, not behaviour.',
            'A single-site perturbation is not a stability proof; it is a measurement.',
            'No model file is modified and no runtime authorisation is granted.'],
    }
    assert r15.digest(XML) == xml_before
    (out / 'stability.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, c in cases.items():
        ax[0].plot([t['t'] for t in c['trace']], [t['thorax_mm'] for t in c['trace']], label=name)
    ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('thorax height (mm)')
    ax[0].set_title('Muscle-only balance held as a CONSTANT command\n'
                    'unperturbed vs small physical perturbations')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for name, c in cases.items():
        if name == 'unperturbed':
            continue
        tr, b0 = c['trace'], base
        ax[1].plot([t['t'] for t in tr],
                   [abs(t['thorax_mm'] - b['thorax_mm']) for t, b in zip(tr, b0)], label=name)
    ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('deviation from unperturbed (mm)')
    ax[1].set_title('Perturbation response: does the deviation shrink?\n'
                    'shrinking = rejection; growth = instability')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'stability.png', dpi=110)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'trace'}
                      for k, v in cases.items()}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
