"""BASELINE: does the design's wiring hold the body when the command is the design's OWN
static solution?

The closed loop with the design overlay sank (thorax 2.03 -> 0.70 mm in 0.3 s).  That has two
possible readings and they must not be confused:

  (a) the WIRING is wrong, so no command the network can express would hold the body; or
  (b) the wiring is fine and the space of commands that holds the body is small, while the
      network's own activity sits outside it.

This file separates them by removing the network entirely: it applies the design's LP activation
pattern as a CONSTANT command, with the body seated at the design's own exact-balance pose, and
integrates.  Nothing here is an animal behaviour or a controller: a constant activation is a
fixed open-loop command, and the file reports only what the body does under it.

  * holds  -> the pose and the pattern are consistent under the engine, so the closed-loop
              collapse is a NETWORK/DYNAMICS statement, not a wiring statement;
  * does not hold -> the static solution is not sustainable even open loop, and the wiring claim
              must be weakened accordingly.

Diagnostic only.  It modifies no model, writes no XML, and grants no runtime authorisation.
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

DESIGN = HERE / 'outputs/design_motor_pools_round34/design_motor_pools.json'
ROUND30 = HERE / 'outputs/stance_joint_feasible_round30/stance_joint_feasible.json'
ADHESION_BW = 2.0
STANCE = HERE / 'outputs/design_runtime_round35_adhesion1636/stance_for_runner.json'
VARIANT = HERE / 'outputs/muscles_six_leg_standing_adhesion/fruitfly_six_leg_standing_adhesion_engine_default.xml'
OUT = HERE / 'outputs/design_openloop_round35'
XML = r15.XML


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--seconds', type=float, default=0.2)
    ap.add_argument('--design', type=Path, default=DESIGN,
                    help='design table JSON whose solved activation is applied as a constant')
    ap.add_argument('--stance-file', type=Path, default=STANCE)
    ap.add_argument('--cases', default='design_solution,zero,pinned_current_pose')
    ap.add_argument('--adhesion-bw', type=float, default=2.0,
                    help='per-leg adhesion budget used by the perfect-balance reference solve')
    ap.add_argument('--xml', type=Path, default=None,
                    help='variant body (default: the canonical one, which has NO adhesion)')
    ap.add_argument('--adhesion-const', type=float, default=0.0,
                    help='constant command on the variant\'s adhesion actuators; it is a declared '
                         'contact property of the body, held constant, never driven by anything')
    args = ap.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    xml_used = args.xml.resolve() if args.xml else XML
    xml_before = r15.digest(XML)
    hash_before = r15.digest(xml_used)
    model = _load_mjcf(str(xml_used)).compile()
    data = mj.MjData(model)
    model.actuator_ctrlrange[:, 0] = 0.0
    muscle_idx = np.array([i for i in range(model.nu)
                           if model.actuator_dyntype[i] == int(mj.mjtDyn.mjDYN_MUSCLE)], int)
    adh_idx = np.array([i for i in range(model.nu) if i not in set(muscle_idx.tolist())], int)
    if len(adh_idx) and args.adhesion_const <= 0.:
        raise ValueError('variant carries adhesion actuators but --adhesion-const is 0')
    names = [model.actuator(i).name or f'act{i}' for i in range(model.nu)]
    model.opt.timestep = float(model.opt.timestep)
    floor = r16.floor_geom_id(model, mj)
    tarsus_geoms = [g for g in range(model.ngeom)
                    if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
    tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
    thorax_id = model.body('Thorax').id

    design = json.loads(args.design.read_text())
    stance = json.loads(args.stance_file.read_text())
    ph = design.get('physics', {})
    best = (ph.get('design_search_over_poses') or {}).get('best') or \
        design.get('best') or ph.get('best')
    if best is None or 'activation' not in best:
        raise ValueError('this design file has no solved activation pattern to apply')

    mj.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mj.mj_forward(model, data)
    key = data.qpos.copy()
    qpos = key.copy()
    for jn, v in stance['joints_rad'].items():
        qpos[model.jnt_qposadr[model.joint(jn).id]] = float(v)

    def ctrl_from_design():
        c = np.zeros(model.nu)
        for k, v in best['activation'].items():
            if k in names:
                c[names.index(k)] = float(v)
            else:
                members = [a for a in names if a.startswith(k + '_a') or a.startswith(k + '_b')]
                for a in members:
                    c[names.index(a)] = float(v)
        return c

    cases = {}
    seconds = float(args.seconds)
    n_step = int(round(seconds / model.opt.timestep))

    # --- the design's constant activation, from the design's own pose -------------------
    def fresh():
        # reset to the model's own keyframe (mj_resetData alone would leave qpos at zero), then
        # zero the time base so every case is plotted from t = 0
        mj.mj_resetDataKeyframe(model, data, 0)
        data.time = 0.0
        return data

    # the SEATED pose: the same seating rule the design, the runner stance file and every earlier
    # round used.  Applying the solved pattern from a differently seated body would test the wrong
    # configuration, so this case is the one that carries the claim.
    seated, root_z, _ = r16.seat_root(model, data, mj, qpos, tarsus_geoms, floor, -0.050)

    if 'design_solution' in args.cases:
        fresh()
        data.qpos[:] = seated; data.qvel[:] = 0
        mj.mj_forward(model, data)
        c = ctrl_from_design()
        full = np.zeros(model.nu); full[muscle_idx] = c[muscle_idx]
        if len(adh_idx): full[adh_idx] = args.adhesion_const
        data.ctrl[:] = full
        trace = []
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % max(1, n_step // 200) == 0:
                facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                trace.append({'t': float(data.time), 'thorax_mm': facts['thorax_height_mm'],
                              'contacts': facts['contact_count'],
                              'tarsus_only': bool(facts['contacts_are_feet_only_tarsus']),
                              'root_z': float(data.qpos[2])})
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        cases['design_solution'] = {
            'ctrl_nonzero': int(np.count_nonzero(c)),
            'h_max_mm': max(t['thorax_mm'] for t in trace),
            'h_final_mm': facts['thorax_height_mm'],
            'h_min_mm': min(t['thorax_mm'] for t in trace),
            'final_tarsus_only': bool(facts['contacts_are_feet_only_tarsus']),
            'trace': trace,
            'reading': 'a constant open-loop command; NOT a controller and NOT behaviour'}

    # --- PERFECT-BALANCE REFERENCE ------------------------------------------------------
    # The strongest possible open-loop command: solve the force balance AT THIS SEATED POSE with
    # all 90 actuators free, and hold that exact solution constant.  If even this falls, the
    # engine/pose combination itself is not a fixed point and every earlier static-balance claim
    # has to be read as "instantaneous balance only".
    if 'perfect_balance' in args.cases:
        fresh()
        data.qpos[:] = seated; data.qvel[:] = 0
        mj.mj_forward(model, data)
        mj.mj_forward(model, data)
        # moment matrix over the MUSCLE actuators only; the adhesion actuators are separate
        # channels, held at a constant ctrl, and their contribution is added to the applied
        # force directly (their force is the contact-normal force, so it is `data.qfrc_actuator`
        # at the declared ctrl with the muscles at zero)
        def moment_arm(cols):
            data.act[:] = 0.0
            mj.mj_forward(model, data)
            base_v = np.asarray(data.qfrc_actuator, float).copy()
            M = np.zeros((model.nv, len(cols)))
            for jj, a in enumerate(cols):
                data.act[:] = 0.0
                data.act[model.actuator_actadr[a]] = 1.0
                mj.mj_forward(model, data)
                M[:, jj] = np.asarray(data.qfrc_actuator, float) - base_v
            data.act[:] = 0.0
            mj.mj_forward(model, data)
            return M, base_v

        Rl, bl = moment_arm(muscle_idx)
        data.ctrl[:] = 0.0
        if len(adh_idx):
            data.ctrl[adh_idx] = args.adhesion_const
        mj.mj_forward(model, data)
        adh_frc = np.asarray(data.qfrc_actuator, float).copy()
        data.ctrl[:] = 0.0
        mj.mj_forward(model, data)
        tg = (np.asarray(data.qfrc_bias, float) - bl
              - np.asarray(data.qfrc_passive, float) - adh_frc)
        tm = r15.contact_terms(model, data, mj)
        lm = r15.limit_terms(model, data, mj)
        Bl = r15.cone_basis(model, data, tm, r15.CONE_DIRECTIONS)
        par = r15.solve_set(Rl, tg, tm, lm, Bl, model.nv, 'perfect balance at seated pose')
        c = np.zeros(model.nu)
        if 'activation' in par:
            c[muscle_idx] = np.asarray(par['activation'], float)
        full = np.zeros(model.nu); full[muscle_idx] = c[muscle_idx]
        if len(adh_idx): full[adh_idx] = args.adhesion_const
        data.ctrl[:] = full
        trace = []
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % max(1, n_step // 200) == 0:
                facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                trace.append({'t': float(data.time), 'thorax_mm': facts['thorax_height_mm'],
                              'contacts': facts['contact_count'],
                              'tarsus_only': bool(facts['contacts_are_feet_only_tarsus'])})
        facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
        cases['perfect_balance'] = {
            'solved_L1_at_seated_pose': float(par['objective_L1_residual']),
            'ctrl_nonzero': int(np.count_nonzero(c)),
            'h_final_mm': facts['thorax_height_mm'],
            'h_max_mm': max(t['thorax_mm'] for t in trace),
            'h_min_mm': min(t['thorax_mm'] for t in trace),
            'trace': trace,
            'reading': 'the best possible constant command at this exact pose; NOT a controller'}

    # --- zero activation, for the scale of the comparison --------------------------------
    if 'zero' in args.cases:
        fresh()
        data.qpos[:] = seated; data.qvel[:] = 0
        mj.mj_forward(model, data)
        full = np.zeros(model.nu)
        if len(adh_idx): full[adh_idx] = args.adhesion_const
        data.ctrl[:] = full
        trace = []
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % max(1, n_step // 200) == 0:
                facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                trace.append({'t': float(data.time), 'thorax_mm': facts['thorax_height_mm'],
                              'contacts': facts['contact_count'],
                              'tarsus_only': bool(facts['contacts_are_feet_only_tarsus'])})
        cases['zero'] = {'h_final_model': float(data.qpos[2]),
                         'h_max_mm': max(t['thorax_mm'] for t in trace),
                         'trace': trace}

    # --- the design's activation, but from the ENGINE's own keyframe ---------------------
    # If the pattern holds from the solved pose but not from the keyframe, the open-loop command
    # is pose-specific, which is exactly what a fixed open-loop command is expected to be.
    if 'pinned_current_pose' in args.cases:
        fresh()
        data.qpos[:] = key; data.qvel[:] = 0
        mj.mj_forward(model, data)
        _c = ctrl_from_design()
        full = np.zeros(model.nu); full[muscle_idx] = _c[muscle_idx]
        if len(adh_idx): full[adh_idx] = args.adhesion_const
        data.ctrl[:] = full
        trace = []
        for i in range(n_step):
            mj.mj_step(model, data)
            if i % max(1, n_step // 200) == 0:
                facts = r16.pose_facts(model, data, mj, tarsus_names, thorax_id)
                trace.append({'t': float(data.time), 'thorax_mm': facts['thorax_height_mm'],
                              'contacts': facts['contact_count']})
        cases['pinned_current_pose'] = {
            'h_final_model': float(data.qpos[2]),
            'h_max_mm': max(t['thorax_mm'] for t in trace), 'trace': trace,
            'note': 'same constant command, started from the XML keyframe instead of the solved '
                    'pose: a fixed open-loop command is expected to be pose specific'}

    report = {
        'status': 'DESIGN_OPENLOOP_BASELINE',
        'body_model': str(xml_used),
        'is_variant_model': bool(args.xml is not None),
        'adhesion_const': float(args.adhesion_const),
        'n_adhesion_actuators': int(len(adh_idx)),
        'question': 'with the network removed, does the design\'s own static solution hold the '
                    'body?',
        'stance': {'source': str(args.stance_file.resolve()), 'thorax_height_mm': stance['thorax_height_mm'],
                   'contact_count': stance['contact_count'], 'tarsus_only': stance['tarsus_only']},
        'design_L1': best['L1'],
        'seconds': seconds, 'timestep': float(model.opt.timestep),
        'cases': cases,
        'source_sha256': {str(XML): r15.digest(XML), str(args.design): r15.digest(args.design),
                          str(args.stance_file): r15.digest(args.stance_file)},
        'limitations': [
            'A constant activation is an open-loop command, not a controller and not behaviour.',
            'Fixed-pose static balance does not imply the pose is dynamically stable.',
            'No model file is modified and no runtime authorisation is granted.'],
    }
    assert r15.digest(XML) == xml_before and r15.digest(xml_used) == hash_before
    (out / 'design_openloop.json').write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {'design_solution': ('tab:green', 'design constant activation (from solved pose)'),
              'perfect_balance': ('tab:red', 'perfect balance solved at the seated pose'),
              'zero': ('tab:gray', 'zero activation'),
              'pinned_current_pose': ('tab:orange', 'design activation, from XML keyframe')}
    for k, c in cases.items():
        if 'trace' not in c:
            continue
        col, lab = styles.get(k, ('tab:blue', k))
        ax.plot([t['t'] for t in c['trace']], [t['thorax_mm'] for t in c['trace']],
                color=col, label=lab)
    ax.axhline(stance['thorax_height_mm'], ls='--', c='k', lw=.8,
               label=f"static solution height {stance['thorax_height_mm']:.3f} mm")
    ax.set_xlabel('time (s)'); ax.set_ylabel('thorax height (mm)')
    ax.set_title('Open loop: the design\'s own static solution as a CONSTANT command\n'
                 'not a controller, not behaviour, not walking')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'design_openloop.png', dpi=110)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'trace'} for k, v in cases.items()},
                     indent=2))
    print(out)


if __name__ == '__main__':
    main()
