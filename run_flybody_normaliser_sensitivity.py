"""How much does the loop depend on the ONE normaliser that dominates it?

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

The ablation survey measured that the load channel (campaniform sensilla) single-handedly keeps the
motor drive alive, and that the joint-angle channels contribute about a fifth as much input current.
All channels share one current scale, so their RELATIVE strength is set entirely by the four declared
normalisers the encoder uses: reference_angle_rad, reference_velocity_rad_s, reference_force and
reference_current.  This file asks how the loop responds when those normalisers move, so that the
reach of any published number can be bounded before one is adopted.

It is a SENSITIVITY scan of declared translation parameters.  No value in it is claimed to be
correct, biological or calibrated, and nothing here is tuned to make a body stand: the outcomes are
reported as drive magnitude, per-channel input share, drive/deviation coupling, and the standing
checklist.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.memory_guard as mg
import engine.canonical_physics as cp
import run_flybody_loop as loop
import run_flybody_reference_audit as reference_audit
import run_neural_muscle_closure as closure

OUT = loop.OUT
FORCE_NORMALISERS = (.5, 1.0, 2.0, 4.0, 8.0)
ANGLE_NORMALISERS = (.10, .20, .30, .60)
DURATION = .2


def run_with(built, manifest, groups, seated_height_mm, *, reference_force=None,
             reference_angle_rad=None, duration=DURATION):
    cfg = closure.ChannelConfig()
    if reference_force is not None:
        cfg = closure.ChannelConfig(reference_force=reference_force)
    if reference_angle_rad is not None:
        cfg = closure.ChannelConfig(reference_angle_rad=reference_angle_rad,
                                    reference_force=reference_force if reference_force is not None
                                    else cfg.reference_force)
    row = reference_audit.run('flybody_servo', built, manifest, groups, seated_height_mm,
                              duration=duration, channel_config=cfg)
    row['encoder_config'] = dict(reference_force=cfg.reference_force,
                                 reference_angle_rad=cfg.reference_angle_rad,
                                 reference_velocity_rad_s=cfg.reference_velocity_rad_s,
                                 current_nA_per_unit=cfg.current_nA_per_unit)
    return row


def main(guard_report=None):
    import mujoco
    started = time.time()
    pools = json.loads(closure.POOLS.read_text())
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    probe = cp.load_model()
    names = [mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(probe.nu)]
    manifest = closure.build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    body = loop.build_body('flybody_servo', 'declared')
    data = mujoco.MjData(body['model'])
    data.qpos[:] = body['seated']
    data.qvel[:] = 0.0
    mujoco.mj_forward(body['model'], data)
    seated_height_mm = float(data.subtree_com[body['model'].body('thorax').id][2]
                             - data.geom_xpos[body['floor']][2])*10.0
    rows = {}
    base = run_with(built, manifest, groups, seated_height_mm)
    rows['published'] = base
    shares = base['channel_currents']['per_channel']['share_of_total_current']
    print('published shares', {k: round(v, 3) for k, v in shares.items()}, flush=True)
    # derived, not recommended: what a single normaliser would have to be for the two channels that
    # carry posture to reach the current the load channel already carries
    contact_over_angle = (base['channel_currents']['per_channel']['mean_abs_current_nA']
                          ['campaniform_strain'] /
                          (base['channel_currents']['per_channel']['mean_abs_current_nA']
                           ['claw_position'] +
                           base['channel_currents']['per_channel']['mean_abs_current_nA']
                           ['hairplate_angle']))
    for value in FORCE_NORMALISERS:
        if value == 1.0:
            continue
        key = f'force_{value:g}'
        rows[key] = run_with(built, manifest, groups, seated_height_mm, reference_force=value)
        print(f'  {key:12s} drive {rows[key]["mean_abs_drive"]:.4f} '
              f'median_r {rows[key]["posture_reference"]["median_correlation"]:+.3f} '
              f'standing {rows[key]["standing"]["standing_under_this_projects_criteria"]} '
              f'claws {rows[key]["standing"]["measured"]["n_claw_contacts"]}', flush=True)
    for value in ANGLE_NORMALISERS:
        if value == .30:
            continue
        key = f'angle_{value:g}'
        rows[key] = run_with(built, manifest, groups, seated_height_mm, reference_angle_rad=value)
        print(f'  {key:12s} drive {rows[key]["mean_abs_drive"]:.4f} '
              f'median_r {rows[key]["posture_reference"]["median_correlation"]:+.3f} '
              f'standing {rows[key]["standing"]["standing_under_this_projects_criteria"]} '
              f'claws {rows[key]["standing"]["measured"]["n_claw_contacts"]}', flush=True)
    summary = {}
    for key, row in rows.items():
        summary[key] = dict(
            encoder_config=row['encoder_config'],
            mean_abs_drive=row['mean_abs_drive'],
            drive_share_by_channel=row.get('channel_currents', {}).get('per_channel', {})
            .get('share_of_total_current'),
            median_correlation=row['posture_reference']['median_correlation'],
            n_restoring=row['posture_reference']['n_restoring_negative'],
            escape_time_s=row['escape_time_s'],
            height_mean_last_50ms_mm=row['height_mean_last_50ms_mm'],
            height_std_last_50ms_mm=row['height_std_last_50ms_mm'],
            drift_last_50ms_mm=row['drift_last_50ms_mm'],
            standing=row['standing']['standing_under_this_projects_criteria'],
            failed_checks=row['standing']['failed_checks'],
            standing_measured=row['standing']['measured'])
    result = dict(
        status='MEASURED_NORMALISER_SENSITIVITY',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; declared-normaliser sensitivity scan',
        question='How far does the loop move when the declared normalisers that set the RELATIVE '
                 'strength of the sensory channels change?',
        seated_height_mm=seated_height_mm,
        contact_over_posture_current_ratio=contact_over_angle,
        force_normalisers_scanned=list(FORCE_NORMALISERS),
        angle_normalisers_scanned=list(ANGLE_NORMALISERS),
        summary=summary,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        changed=['declared encoder normalisers (reference_force, reference_angle_rad)'],
                        unchanged=['cells and tier membership', 'connectome weights', 'conductance '
                                   'parameters', 'channel partition', 'muscle pool manifest',
                                   'decoder', 'clocks', 'translation coefficient table',
                                   'current scale (one shared value)'],
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'This is a SENSITIVITY scan of declared parameters.  No scanned value is claimed to be '
            'correct, calibrated, or biological, and none was chosen to make a body stand.',
            'Because all channels share ONE current scale, moving a normaliser changes only the '
            'relative strength of the channels and cannot change which cells exist or how they are '
            'wired.',
            'The derived ratio (contact current over posture current) is an arithmetic observation at '
            'this pose, not a recommendation for a normaliser value.',
            'The standing checklist is this project\'s own definition and the body is the published '
            'FlyBody model with its own servos.',
        ])
    cp.write_result(OUT/'flybody_normaliser_sensitivity.json', question=result['question'],
                    status='PASS',
                    measured={k: result[k] for k in ('seated_height_mm',
                                                     'contact_over_posture_current_ratio',
                                                     'force_normalisers_scanned',
                                                     'angle_normalisers_scanned', 'summary')},
                    provenance=result['provenance'],
                    units=dict(time='s', height='mm', angle='rad',
                               current='nA', normaliser='same units as the quantity it divides'),
                    label=result['label'],
                    status_meaning='PASS means every scan point ran; the scan is not a calibration',
                    illustrative_vs_measured='Drives, shares, couplings and contact states are '
                                             'measured; the normaliser values are DECLARED scan points',
                    what_this_does_not_support=result['what_this_does_not_support'])
    print('done', OUT/'flybody_normaliser_sensitivity.json', flush=True)
    return result


if __name__ == '__main__':
    with mg.guarded_run('flybody_normaliser_scan', need_mb=2200.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
