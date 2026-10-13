"""Round19: add the engine's documented leg adhesion to a VARIANT model and measure the
REAL dynamic effect.  Diagnostic only: no controller, no target, no schedule.

WHAT IS MEASURED, AND WHAT EACH NUMBER MEANS.
  0. Mechanism check.  A tiny independent model proves that the actuator form used here
     (the same one FlyGym's Fly.add_leg_adhesion produces) really does inject a
     contact-normal attraction: a box held on the floor only by adhesion stays down when
     its own weight would otherwise lift it off, and releases when ctrl goes to 0.
  1. Passive settle, exactly zero muscle command, 0.5 s, three models: canonical, and the
     two adhesion variants.  Reported: thorax height over time, final conctact count, and
     the contact-set COMPOSITION, because "the body is on the floor" is the collapse we
     are trying to move.  A null result here is a valid and important outcome.
  2. Static LP using the project's validated helpers, run on each model.  The LP has no
     notion of adhesion actuators, so its answer is the NO-ADHESION REFERENCE for every
     model; it is reported to show that it is blind to the added actuators and must not
     be double counted against the dynamic result.
  3. Whether the whole-body neural runner can be pointed at a variant.  If it cannot, that
     is reported plainly and the neural-loop outcome is left untested rather than faked by
     editing or forking that runner.

LIMITS.  The adhesion element reproduced here is MuJoCo's own adhesion actuator as used by
FlyGym's FlyBody assets, not pulvillus contact mechanics.  A gain either of the engine's own
documented default (40 = 1.64 body weights per leg) or of the top of the measured biological
range (2.0 body weights per leg) is a CANDIDATE PHYSICAL CORRECTION, not a tuned parameter.
Reduced collapse would NOT be standing, NOT stability, NOT a controller and NOT autonomy.
The canonical asset is deliberately unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import diag_static_support_feasibility as r15  # noqa: E402  validated LP + contact helpers
import diag_posture_support_search as r16      # noqa: E402  contact-set bookkeeping

OUT = ROOT / 'outputs/adhesion_variant_round19'
CANONICAL = ROOT / 'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml'
VARIANTS = ROOT / 'outputs/muscles_six_leg_standing_adhesion'
SETTLE_SECONDS = 0.5
EXTENDED_SETTLE_SECONDS = 2.0
NEURAL_RUNNER = HERE / 'diag_autonomous_sensorimotor.py'


def mechanism_check(mujoco):
    """Independently verify the actuator form AND its gain semantics.

    The box starts RESTING ON the floor, and gravity points UP, so the box's own weight
    pulls it off the floor and only adhesion can hold it.  MuJoCo documents that this
    actuator applies ``ctrl * gain`` of force in the contact-normal direction and that it
    requires ACTIVE contacts, so the box must start in contact.  Sweeping the gain at
    ctrl=1 measures the force in the model's own force units: the box holds exactly when
    gain >= its weight.
    """
    mass = 0.05
    weight = mass * 9.81
    xml = ("<mujoco><option gravity='0 0 9.81'><flag contact='enable'/></option><worldbody>"
           "<geom name='floor' type='plane' size='1 1 0.1'/>"
           f"<body name='box' pos='0 0 0.05'><freejoint/>"
           f"<geom name='box_geom' type='box' size='0.1 0.05 0.05' mass='{mass}'/>"
           "</body></worldbody><actuator>"
           "<adhesion name='box_adhesion' body='box' gain='1.0' ctrlrange='0 1'/>"
           "</actuator></mujoco>")
    if 'flag contact' in xml:  # 'contact' is an enable flag name; drop it if unsupported
        try:
            mujoco.MjModel.from_xml_string(xml)
        except Exception:
            xml = xml.replace("<flag contact='enable'/>", "")
    model = mujoco.MjModel.from_xml_string(xml)
    runs = {}
    for gain, ctrl in ((1.0, 0.0), (0.1, 1.0), (0.3, 1.0), (0.49, 1.0), (0.6, 1.0), (1.0, 1.0)):
        m = mujoco.MjModel.from_xml_string(xml)
        m.actuator_gainprm[:, :] = 0.0
        m.actuator_gainprm[:, 0] = gain
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        z0 = float(data.qpos[2])
        data.ctrl[:] = ctrl
        for _ in range(4000):
            mujoco.mj_step(m, data)
        mujoco.mj_forward(m, data)
        runs[f'gain_{gain:g}_ctrl_{ctrl:g}'] = {
            'applied_force': gain * ctrl,
            'final_height_mm': float(data.qpos[2]) * 1000.0,
            'lifted_mm': (float(data.qpos[2]) - z0) * 1000.0,
            'final_n_contacts': int(data.ncon)}
    holds = {k: v['final_n_contacts'] > 0 for k, v in runs.items()}
    assert holds['gain_1_ctrl_0'] is False, 'with no adhesion the box must leave the floor'
    assert holds['gain_1_ctrl_1'] is True, 'with enough adhesion the box must stay on the floor'
    assert holds['gain_0.6_ctrl_1'] is True and holds['gain_0.3_ctrl_1'] is False, \
        'the holding threshold must track the weight'
    return {'model': 'box resting on a plane with UPWARD gravity (mass 0.05, weight %.4f force '
                     'units) and one adhesion actuator on the box body, ctrlrange (0,1)' % weight,
            'box_weight_force_units': weight,
            'runs': runs,
            'holding_threshold_hint': 'holds for applied force above the weight, releases below it',
            'conclusion': 'the actuator form used in the variants injects a contact-normal '
                          'attraction equal to ctrl*gain, holds the box against its own weight when '
                          'that exceeds the weight, and releases at ctrl=0 - the documented semantics'}


def contact_composition(model, data, mujoco, tarsus_names, floor_id):
    """Ground-contact geoms split into tarsus vs everything else."""
    touching = set()
    plane_ids = {floor_id}
    for c in data.contact:
        g1, g2 = int(c.geom1), int(c.geom2)
        other = g2 if g1 in plane_ids else g1 if g2 in plane_ids else None
        if other is not None:
            touching.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or '')
    feet = sorted(n for n in touching if n in tarsus_names)
    nonfeet = sorted(n for n in touching if n not in tarsus_names)
    return {'ground_geoms': sorted(touching), 'n_ground_geoms': len(touching),
            'tarsus_geoms': feet, 'non_tarsus_geoms': nonfeet,
            'feet_only': bool(touching) and not nonfeet}


def settle(model, data, mujoco, thorax_id, tarsus_names, floor_id, seconds, adhesion_ctrl):
    """Exactly zero muscle command; adhesion ctrl fixed.  Returns trace + final state."""
    n_muscle = 90
    dt = float(model.opt.timestep)
    steps = int(round(seconds / dt))
    assert abs(steps * dt - seconds) < 1e-12
    model.actuator_ctrlrange[:, 0] = 0.0     # memory only: an exactly zero command must be legal
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    data.ctrl[:] = 0.0
    if model.nu > n_muscle:
        data.ctrl[n_muscle:] = adhesion_ctrl
    mujoco.mj_forward(model, data)
    comp0 = contact_composition(model, data, mujoco, tarsus_names, floor_id)
    trace = [{'time_s': 0.0, 'thorax_height_model': float(data.xpos[thorax_id][2]),
              'n_contacts': int(data.ncon)}]
    muscle_ctrl_max = 0.0
    for k in range(steps):
        # re-assert the exact command every step: nothing else may write ctrl
        data.ctrl[:n_muscle] = 0.0
        if model.nu > n_muscle:
            data.ctrl[n_muscle:] = adhesion_ctrl
        mujoco.mj_step(model, data)
        muscle_ctrl_max = max(muscle_ctrl_max, float(np.abs(data.ctrl[:n_muscle]).max()),
                              float(np.abs(data.act[:n_muscle]).max()) if model.na else 0.0)
        if k % 50 == 0:
            trace.append({'time_s': float(data.time), 'thorax_height_model': float(data.xpos[thorax_id][2]),
                          'n_contacts': int(data.ncon)})
    mujoco.mj_forward(model, data)
    trace.append({'time_s': float(data.time), 'thorax_height_model': float(data.xpos[thorax_id][2]),
                  'n_contacts': int(data.ncon)})
    final = contact_composition(model, data, mujoco, tarsus_names, floor_id)
    return {'start': {'thorax_height_mm': trace[0]['thorax_height_mm'], 'n_contacts': comp0['n_ground_geoms'],
                      'feet_only': comp0['feet_only'], 'non_tarsus_geoms': comp0['non_tarsus_geoms']},
            'final': {'thorax_height_mm': trace[-1]['thorax_height_mm'],
                      'n_contacts': int(data.ncon), **final},
            'drop_mm': trace[0]['thorax_height_mm'] - trace[-1]['thorax_height_mm'],
            'min_thorax_height_mm': min(t['thorax_height_mm'] for t in trace),
            'max_abs_muscle_ctrl_or_activation': muscle_ctrl_max,
            'trace': trace,
            'adhesion_ctrl': adhesion_ctrl}


def static_lp(model, data, mujoco, label):
    """LP residual for this model's 90 muscle actuators.  Blind to adhesion by construction."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # r15.actuator_moment_matrix loops over model.nu and writes data.act[j], which is only
    # legal for the muscle actuators (the variants have 96 actuators but 90 activation states).
    # The matrix is therefore built here over the 90 MUSCLE actuators only, using the same
    # (act_j=1) - (act=0) difference convention as the validated helper.
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    base = np.asarray(data.qfrc_actuator, float).copy()
    R = np.zeros((model.nv, 90))
    for j in range(90):
        data.act[:] = 0.0
        data.act[j] = 1.0
        mujoco.mj_forward(model, data)
        R[:, j] = np.asarray(data.qfrc_actuator, float) - base
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    assert R.shape == (model.nv, 90)
    target = np.asarray(data.qfrc_bias, float) - base - np.asarray(data.qfrc_passive, float)
    terms = r15.contact_terms(model, data, mujoco)
    lims = r15.limit_terms(model, data, mujoco)
    B = r15.cone_basis(model, data, terms, r15.CONE_DIRECTIONS)
    sol = r15.solve_set(R, target, terms, lims, B, model.nv, label)
    return {'n_contacts_in_lp': len(terms), 'n_limit_rows': len(lims),
            'all90_L1': float(sol['objective_L1_residual']),
            'max_abs': float(sol['max_abs_residual']),
            'feasible_within_tol': bool(sol['feasible_within_tol']),
            'note': 'the LP has no adhesion term, so this is the no-adhesion reference and must '
                    'not be subtracted from the dynamic result'}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco as mj
    from flygym.compose.fly.musculoskeletal import _load_mjcf

    canon_sha_before = r15.digest(CANONICAL)
    assert canon_sha_before == '65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6'

    gen = json.loads((VARIANTS / 'variant_generation.json').read_text())
    models = [{'label': 'canonical', 'file': CANONICAL, 'sha256': canon_sha_before,
               'adhesion_ctrl': None, 'gain': None, 'bw_per_leg': 0.0}]
    for v in gen['variants']:
        models.append({'label': v['label'], 'file': Path(v['file']), 'sha256': v['sha256'],
                       'adhesion_ctrl': 1.0, 'gain': v['gain'],
                       'bw_per_leg': v['body_weights_per_leg_at_ctrl1']})

    report = {'status': 'ADHESION_VARIANT_DYNAMIC_MEASUREMENT',
              'what_this_is': 'the engine-documented leg adhesion added on a VARIANT model, then '
                              'measured dynamically; no controller, no target, no schedule',
              'canonical': {'file': str(CANONICAL), 'sha256_before': canon_sha_before},
              'mechanism_check': mechanism_check(mj),
              'settle_seconds': SETTLE_SECONDS,
              'body_weight_force_units': gen['body_weight_force_units'],
              'source_sha256': {str(CANONICAL): canon_sha_before},
              'models': [], 'limitations': [
                  'The adhesion element is MuJoCo/FlyGym\'s own actuator, NOT pulvillus contact '
                  'mechanics: no capillarity, peeling, contact area or rate dependence.',
                  'Every actuator activation other than the adhesion control is EXACTLY zero for the '
                  'whole settle; the passivity is asserted every step and the max is reported.',
                  'A changed collapse would be a candidate physical correction, NOT standing, NOT '
                  'stability, NOT controllability and NOT autonomy.',
                  'The static LP is blind to adhesion actuators, so its number is a reference for '
                  'all models and is not evidence about the adhesion variants.']}

    for spec in models:
        model = _load_mjcf(str(spec['file'])).compile()
        data = mj.MjData(model)
        floor_id = r16.floor_geom_id(model, mj)
        tarsus_geoms = [g for g in range(model.ngeom)
                        if 'Tarsus' in (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '')]
        tarsus_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in tarsus_geoms}
        thorax_id = model.body('Thorax').id
        entry = dict(spec)
        entry['file'] = str(spec['file'])
        entry['nu'] = int(model.nu)
        ac = 0.0 if spec['adhesion_ctrl'] is None else spec['adhesion_ctrl']
        entry['passive_settle'] = settle(model, data, mj, thorax_id, tarsus_names, floor_id,
                                         SETTLE_SECONDS, ac)
        entry['passive_settle_extended'] = settle(model, data, mj, thorax_id, tarsus_names, floor_id,
                                                  EXTENDED_SETTLE_SECONDS, ac)
        if model.nu > 90:
            # CONTROL: the same variant with adhesion RELEASED.  If the added actuators have no
            # side effect at ctrl=0 this must reproduce the canonical trajectory, which separates
            # "the fragment changed the model" from "adhesion was applied".
            entry['passive_settle_adhesion_released'] = settle(
                model, data, mj, thorax_id, tarsus_names, floor_id, SETTLE_SECONDS, 0.0)
        entry['static_lp'] = static_lp(model, data, mj, f"{spec['label']}:static")
        report['models'].append(entry)
        print(spec['label'], 'start %.4f -> final %.4f mm, drop %.4f, feet_only %s, nonfeet %d, LP %.4f' % (
            entry['passive_settle']['start']['thorax_height_mm'],
            entry['passive_settle']['final']['thorax_height_mm'],
            entry['passive_settle']['drop_mm'],
            entry['passive_settle']['final']['feet_only'],
            len(entry['passive_settle']['final']['non_tarsus_geoms']),
            entry['static_lp']['all90_L1']), flush=True)

    # (c) can the whole-body neural runner be pointed at a variant?
    runner_text = NEURAL_RUNNER.read_text()
    has_xml_flag = ('--xml' in runner_text) or ("add_argument('--xml" in runner_text)
    report['neural_runner'] = {
        'file': str(NEURAL_RUNNER),
        'accepts_xml_override': bool(has_xml_flag),
        'xml_reference': 'module-level XML constant at the top of the file, not a CLI argument',
        'action_taken': 'none - the runner was NOT edited, forked or monkeypatched',
        'consequence': 'the whole-body NEURAL loop outcome is NOT measured for the adhesion variants; '
                       'the comparison here is the passive trajectory, the contact-set composition and '
                       'the (adhesion-blind) static LP only.  Adding an XML override to that runner is '
                       'its owner\'s decision, not something this diagnostic may do silently.'}

    # CONTROL ASSERTION: with adhesion released the variants must reproduce the canonical
    # trajectory EXACTLY, which separates "the inserted fragment changed the model" from
    # "adhesion was actually applied".
    canon_final = [m for m in report['models'] if m['label'] == 'canonical'][0][
        'passive_settle']['final']['thorax_height_mm']
    released = {m['label']: m['passive_settle_adhesion_released']['final']['thorax_height_mm']
                for m in report['models'] if 'passive_settle_adhesion_released' in m}
    report['adhesion_released_control_neutral'] = {
        'canonical_final_mm': canon_final,
        'variant_final_mm_with_ctrl0': released,
        'bitwise_equal': bool(released) and all(v == canon_final for v in released.values()),
        'meaning': 'the six inserted actuators have no effect at ctrl=0, so the fragment itself is '
                   'neutral and any difference is caused by the adhesion force, not by the edit'}
    assert report['adhesion_released_control_neutral']['bitwise_equal'], \
        'the inserted adhesion fragment must be neutral when released'

    report['canonical']['sha256_after'] = r15.digest(CANONICAL)
    assert report['canonical']['sha256_after'] == canon_sha_before, 'canonical XML changed'
    report['wall_seconds'] = time.time() - t0
    (OUT / 'adhesion_variant.json').write_text(json.dumps(report, indent=2))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for m in report['models']:
        tr = m['passive_settle']['trace']
        axs[0].plot([t['time_s'] for t in tr], [t['thorax_height_mm'] for t in tr], 'o-',
                    ms=3, label=f"{m['label']} ({m['bw_per_leg']:.2f} bw/leg)" if m['gain']
                    else f"{m['label']} (no adhesion)")
    axs[0].set_xlabel('time (s)'); axs[0].set_ylabel('thorax height (mm)')
    axs[0].set_title('Passive settle, exactly zero muscle command')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

    labels = [m['label'] for m in report['models']]
    feet = [len(m['passive_settle']['final']['tarsus_geoms']) for m in report['models']]
    non = [len(m['passive_settle']['final']['non_tarsus_geoms']) for m in report['models']]
    x = np.arange(len(labels))
    axs[1].bar(x - .2, feet, .4, label='tarsus geoms on floor')
    axs[1].bar(x + .2, non, .4, label='non-tarsus geoms on floor')
    axs[1].set_xticks(x); axs[1].set_xticklabels(labels, fontsize=8)
    axs[1].set_ylabel('distinct ground-contact geoms')
    axs[1].set_title('Final ground-contact composition\n(non-tarsus contact = body resting on itself)')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3, axis='y')

    axs[2].bar(x, [m['static_lp']['all90_L1'] for m in report['models']],
               color=['tab:gray', 'tab:blue', 'tab:orange'])
    axs[2].axhline(report['body_weight_force_units'], ls=':', c='k', label='one body weight')
    axs[2].set_xticks(x); axs[2].set_xticklabels(labels, fontsize=8)
    axs[2].set_ylabel('LP L1 residual')
    axs[2].set_title('Static LP (BLIND to adhesion actuators)\nreported as the no-adhesion reference')
    axs[2].legend(fontsize=8); axs[2].grid(alpha=.3, axis='y')
    fig.suptitle('Adhesion variant: candidate physical correction, NOT a controller, NOT standing\n'
                 'canonical asset unchanged')
    fig.tight_layout(); fig.savefig(OUT / 'adhesion_variant.png', dpi=110)

    summary = {'wall_seconds': report['wall_seconds'],
               'mechanism_check': report['mechanism_check']['runs'],
               'models': [{'label': m['label'], 'nu': m['nu'], 'bw_per_leg': m['bw_per_leg'],
                           'start_mm': m['passive_settle']['start']['thorax_height_mm'],
                           'final_mm': m['passive_settle']['final']['thorax_height_mm'],
                           'drop_mm': m['passive_settle']['drop_mm'],
                           'final_feet_only': m['passive_settle']['final']['feet_only'],
                           'final_non_tarsus_geoms': m['passive_settle']['final']['non_tarsus_geoms'],
                           'max_muscle_ctrl_or_act': m['passive_settle']['max_abs_muscle_ctrl_or_activation'],
                           'lp_all90_L1': m['static_lp']['all90_L1']} for m in report['models']],
               'extended_settle_seconds': EXTENDED_SETTLE_SECONDS,
               'extended': [{'label': m['label'],
                             'final_mm': m['passive_settle_extended']['final']['thorax_height_mm'],
                             'non_tarsus_geoms': len(m['passive_settle_extended']['final']['non_tarsus_geoms']),
                             'feet_only': m['passive_settle_extended']['final']['feet_only']}
                            for m in report['models']],
               'adhesion_released_control': [{'label': m['label'],
                                              'final_mm': m['passive_settle_adhesion_released']['final']['thorax_height_mm']}
                                             for m in report['models']
                                             if 'passive_settle_adhesion_released' in m],
               'neural_runner_accepts_xml': report['neural_runner']['accepts_xml_override']}
    print(json.dumps(summary, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
