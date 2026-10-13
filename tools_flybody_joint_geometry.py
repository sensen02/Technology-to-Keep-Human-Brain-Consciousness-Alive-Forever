"""Where the published reference angles fall in the model's own coordinates.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  Geometry measurement only: no neural circuit, no episode
beyond forward kinematics.

The literature gives stimulus quantities in ANATOMICAL terms and the model gives joint coordinates:

  * the femoral chordotonal (claw) population is tonically active as a deviation from a **femur-tibia
    angle of about 80 deg** (Dallmann et al. 2025, Nature 647:445; Mamiya et al. 2018 report a static
    offset near 90 deg), so the posture-reference channel's reference is an ANATOMICAL angle, not the
    seated pose this project has been using;
  * the fly coxa hair plate CxHP8 encodes **coxa rotation** over a span of about 149 deg = 2.601 rad
    (Pratt et al. 2026, Nat Commun 17:2664), so the hair-plate channel's quantity is a swing angle at
    the thorax-coxa joint, not a mean over the leg's joints.

This tool measures, per leg, the map between the model's hinge coordinates and those anatomical
quantities, by forward kinematics only:
  femur-tibia angle = angle between (tibia body origin - femur body origin) and
                      (tarsus body origin - tibia body origin);
  coxa swing angle  = azimuth of the claw about the thorax's vertical axis through the coxa.
It reports the anatomical angle at the seated pose, the hinge value that reproduces the published
reference angle, and the hinge-to-angle gradient, so a sourced reference can be expressed in the
model's coordinates without guessing.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.memory_guard as mg
import run_flybody_body as body_mod
import run_flybody_loop as loop

OUT = loop.OUT
PUBLISHED_CLAW_REFERENCE_DEG = (80.0, 90.0)   # Dallmann 2025 ~80; Mamiya 2018 static offset ~90
LEG_TAG = body_mod.LEG_TAG      # one definition, in run_flybody_body, to avoid a circular import


def femur_tibia_angle(model, data, leg):
    tag, side = LEG_TAG[leg]
    femur = data.xpos[model.body(f'femur_{tag}_{side}').id]
    tibia = data.xpos[model.body(f'tibia_{tag}_{side}').id]
    tarsus = data.xpos[model.body(f'tarsus_{tag}_{side}').id]
    a = tibia - femur
    b = tarsus - tibia
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)),
                                              -1.0, 1.0))))


def coxa_swing_angle(model, data, leg):
    """Azimuth of the claw about the thorax's vertical axis through the coxa, in degrees."""
    tag, side = LEG_TAG[leg]
    coxa = data.xpos[model.body(f'coxa_{tag}_{side}').id]
    claw = data.xpos[model.body(f'claw_{tag}_{side}').id]
    axis = data.xmat[model.body('thorax').id].reshape(3, 3)[:, 2]
    rel = claw - coxa
    rel = rel - axis*np.dot(rel, axis)
    ref = data.xmat[model.body('thorax').id].reshape(3, 3)[:, 0]
    ref = ref - axis*np.dot(ref, axis)
    return float(np.degrees(np.arctan2(np.dot(np.cross(ref, rel), axis), np.dot(ref, rel))))


def sweep(model, hinge, fn, lo, hi, steps=41, seated=None):
    """Forward-kinematics sweep: hinge coordinate -> anatomical angle, with the local gradient."""
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hinge)
    adr = int(model.jnt_qposadr[jid])
    base = np.zeros(model.nq) if seated is None else seated.copy()
    xs, ys = [], []
    for value in np.linspace(lo, hi, steps):
        data = mujoco.MjData(model)
        data.qpos[:] = base
        data.qpos[adr] = value
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        xs.append(float(value))
        ys.append(fn(model, data))
    xs, ys = np.array(xs), np.array(ys)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else None
    return xs, ys, slope


def main(guard_report=None):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(loop.XML_PUBLISHED))
    data = mujoco.MjData(model)
    seated = np.zeros(model.nq)
    seated[2] = body_mod.seat(model, data, gap_model=0.0)
    seated[3] = 1.0
    data.qpos[:] = seated
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    rows = {}
    for leg in LEG_TAG:
        tag, side = LEG_TAG[leg]
        tibia_hinge = f'tibia_{tag}_{side}'
        abduct_hinge = f'coxa_abduct_{tag}_{side}'
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tibia_hinge)
        tibia_adr = int(model.jnt_qposadr[jid])
        lo, hi = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
        xs, ys, slope = sweep(model, tibia_hinge, lambda m, d: femur_tibia_angle(m, d, leg),
                              lo, hi, seated=seated)
        seated_angle = femur_tibia_angle(model, data, leg)
        reference_hinges = {}
        for target in PUBLISHED_CLAW_REFERENCE_DEG:
            index = int(np.argmin(np.abs(ys - target)))
            reference_hinges[f'{target:g}deg'] = dict(
                hinge_value=round(float(xs[index]), 6), achieved_angle_deg=round(float(ys[index]), 3),
                hinge_deviation_from_seated=round(float(xs[index] - seated[tibia_adr]), 6))
        jid_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, abduct_hinge)
        lo_a, hi_a = (float(model.jnt_range[jid_a][0]), float(model.jnt_range[jid_a][1]))
        xa, ya, slope_a = sweep(model, abduct_hinge, lambda m, d: coxa_swing_angle(m, d, leg),
                                lo_a, hi_a, seated=seated)
        rows[leg] = dict(
            tibia_hinge=tibia_hinge, tibia_hinge_range_rad=[lo, hi],
            tibia_hinge_seated_rad=round(float(seated[tibia_adr]), 6),
            anatomical_femur_tibia_angle_at_seated_deg=round(seated_angle, 3),
            femur_tibia_angle_range_deg=[round(float(ys.min()), 3), round(float(ys.max()), 3)],
            hinge_to_anatomical_angle_deg_per_rad=round(slope, 4) if slope else None,
            hinge_value_for_published_reference=reference_hinges,
            coxa_abduct_hinge=abduct_hinge, coxa_abduct_range_rad=[lo_a, hi_a],
            coxa_swing_angle_at_seated_deg=round(coxa_swing_angle(model, data, leg), 3),
            coxa_swing_range_deg=[round(float(ya.min()), 3), round(float(ya.max()), 3)],
            coxa_hinge_to_swing_deg_per_rad=round(slope_a, 4) if slope_a else None)
    seated_angles = [r['anatomical_femur_tibia_angle_at_seated_deg'] for r in rows.values()]
    result = dict(
        status='MEASURED_JOINT_GEOMETRY',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; forward-kinematics measurement only',
        question='Where do the published anatomical reference angles fall in the model\'s own joint '
                 'coordinates?',
        published_references=dict(
            claw_reference_angle_deg=list(PUBLISHED_CLAW_REFERENCE_DEG),
            claw_reference_source='Dallmann et al. 2025, Nature 647:445 (claw neurons encode the '
                                  'femur-tibia joint as a deviation from about 80 deg); Mamiya et al. '
                                  '2018, Neuron 100:636 (static offset near 90 deg)',
            hairplate_span_rad=2.601,
            hairplate_source='Pratt et al. 2026, Nat Commun 17:2664: CxHP8 coxa-rotation response '
                             'span about 149 deg'),
        per_leg=rows,
        seated_angle_summary=dict(
            mean_deg=round(float(np.mean(seated_angles)), 3),
            min_deg=round(float(np.min(seated_angles)), 3),
            max_deg=round(float(np.max(seated_angles)), 3),
            note='the anatomical femur-tibia angle of the published neutral stance, which is the '
                 'reference this project has been using instead of the published one'),
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        xml=str(loop.XML_PUBLISHED),
                        method='forward kinematics only: segment-origin vectors for the femur-tibia '
                               'angle, claw azimuth about the thorax vertical for the coxa swing',
                        memory_guard=guard_report('completed') if guard_report else None),
        what_this_does_not_support=[
            'A geometric angle in this model is not an anatomical measurement of a fly: the model is '
            'a reconstruction, and the published angle comes from a different preparation.',
            'The hinge-to-angle gradient is local and measured at the seated pose; it is not a '
            'constant of the joint.',
            'The coxa swing angle is an azimuth about the thorax vertical, chosen because the '
            'published CxHP8 stimulus is anterior-posterior coxa rotation; that correspondence is '
            'DECLARED, not verified, and the published joint range (199 deg) is wider than any coxa '
            'hinge in this model.',
            'No neural claim and no behaviour: this file measures geometry.',
        ])
    cp_module = __import__('engine.canonical_physics', fromlist=['x'])
    cp_module.write_result(OUT/'flybody_joint_geometry.json', question=result['question'],
                           status='PASS',
                           measured={k: result[k] for k in
                                     ('published_references', 'per_leg', 'seated_angle_summary')},
                           provenance=result['provenance'],
                           units=dict(angle='deg for anatomical angles, rad for hinge coordinates'),
                           label=result['label'],
                           status_meaning='PASS means the sweeps ran',
                           illustrative_vs_measured='Every angle here is measured on the compiled '
                                                    'model; the published reference angles are cited',
                           what_this_does_not_support=result['what_this_does_not_support'])
    print(json.dumps({leg: dict(seated=r['anatomical_femur_tibia_angle_at_seated_deg'],
                                range=r['femur_tibia_angle_range_deg'],
                                grad=r['hinge_to_anatomical_angle_deg_per_rad'],
                                ref=r['hinge_value_for_published_reference'])
                      for leg, r in rows.items()}, indent=1))
    print('seated summary', result['seated_angle_summary'])
    print(OUT/'flybody_joint_geometry.json')
    return result


if __name__ == '__main__':
    with mg.guarded_run('flybody_joint_geometry', need_mb=1600.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
