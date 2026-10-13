"""Run the closure with the normalisers the literature actually supports, and with the quantities it says.

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

Two things change, both declared and both with a citation recorded in
`outputs/model_branches/flybody_loop/sourced_sensory_config.json` (produced by the gate
`tools_apply_sourced_sensory_config.py`, which refuses any parameter without a source):

  NORMALISERS  the angle normaliser becomes half of the femur-tibia range the FeCO studies swept
               (18-180 deg -> 1.4137 rad) and the velocity normaliser becomes half of the swing
               speeds they used (100-800 deg/s -> 6.1087 rad/s).  The FORCE normaliser is unchanged
               and is listed by the gate as REFUSED, because no published load range has been
               obtained yet.
  ROLES        the hair-plate slot stops carrying a mean over the leg's joints and carries the
               thorax-coxa swing instead, because the fly coxa hair plates encode coxa rotation
               (Pratt et al. 2026) and the BANC attachment field for those cells is the coxa or the
               trochanter.  The position slot keeps the femur-tibia angle, which the literature
               supports (Mamiya 2018; Dallmann 2025).

Nothing else moves: same tier, connectome, channel partition, muscle pool manifest, decoder and
clocks, and the same body.
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
DURATION = .2
PUSH = dict(start_s=.15, end_s=.155, weight_fraction=.5)


def main(guard_report=None):
    import mujoco
    started = time.time()
    sourced = json.loads((OUT/'sourced_sensory_config.json').read_text())
    if sourced['reference_force'] is not None:
        raise SystemExit('the gate produced a force normaliser; this script expects it to be refused '
                         'until a published load range is recorded')
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
    published_cfg = closure.ChannelConfig()
    sourced_cfg = closure.ChannelConfig(reference_angle_rad=sourced['reference_angle_rad'],
                                        reference_velocity_rad_s=sourced['reference_velocity_rad_s'])
    def do_run(cfg, roles, **kwargs):
        row = reference_audit.run('flybody_servo', built, manifest, groups, seated_height_mm,
                                  channel_config=cfg, sensory_roles=roles, **kwargs)
        row['encoder_config'] = dict(reference_angle_rad=cfg.reference_angle_rad,
                                     reference_velocity_rad_s=cfg.reference_velocity_rad_s,
                                     reference_force=cfg.reference_force,
                                     current_nA_per_unit=cfg.current_nA_per_unit)
        row['sensory_roles_declared'] = roles
        return row

    literature_roles = {'hairplate': 'coxa'}
    runs = {}
    runs['published_config_historical_roles'] = do_run(published_cfg, None)
    runs['sourced_normalisers_historical_roles'] = do_run(sourced_cfg, None)
    runs['sourced_normalisers_literature_roles'] = do_run(sourced_cfg, literature_roles)
    runs['published_config_historical_roles_with_perturbation'] = do_run(
        published_cfg, None, duration=.4, push=PUSH)
    runs['sourced_normalisers_literature_roles_with_perturbation'] = do_run(
        sourced_cfg, literature_roles, duration=.4, push=PUSH)
    # the sourced filter window: the gate's geometric mean, the window's edges, and the primary 5 ms
    tau_window = sourced.get('filter_tau_window_ms') or [None, None]
    taus = {'primary_5ms': .005,
            'sourced_rule_geometric_mean': sourced.get('filter_tau_s'),
            'window_upper_bound': (tau_window[1]/1000.0 if tau_window[1] else None)}
    tau_runs = {}
    for name, tau in taus.items():
        if tau is None:
            continue
        row = do_run(sourced_cfg, literature_roles, duration=DURATION, filter_tau_s=tau)
        row['filter_tau_s'] = tau
        tau_runs[name] = row
        print(f'  tau {name:28s} {tau*1000:6.2f} ms  h {row["height_mean_last_50ms_mm"]:.4f} '
              f'claws {row["standing"]["measured"]["n_claw_contacts"]} '
              f'standing {row["standing"]["standing_under_this_projects_criteria"]} '
              f'failed {row["standing"]["failed_checks"]}', flush=True)
    summary = {}
    for key, row in runs.items():
        currents = row.get('channel_currents', {}).get('per_channel', {})
        summary[key] = dict(
            encoder_config=row['encoder_config'], sensory_roles=row['sensory_roles_declared'],
            mean_abs_drive=row['mean_abs_drive'],
            channel_currents_nA=currents.get('mean_abs_current_nA'),
            channel_current_shares=currents.get('share_of_total_current'),
            median_correlation=row['posture_reference']['median_correlation'],
            n_restoring=row['posture_reference']['n_restoring_negative'],
            escape_time_s=row['escape_time_s'],
            height_mean_last_50ms_mm=row['height_mean_last_50ms_mm'],
            height_std_last_50ms_mm=row['height_std_last_50ms_mm'],
            returned_to_band=row.get('returned_to_band'),
            standing=row['standing']['standing_under_this_projects_criteria'],
            failed_checks=row['standing']['failed_checks'],
            standing_measured=row['standing']['measured'])
    for key, row in summary.items():
        print(f"  {key:52s} drive {row['mean_abs_drive']:.4f} r {row['median_correlation']:+.3f} "
              f"rest {row['n_restoring']}/36 h {row['height_mean_last_50ms_mm']:.3f} "
              f"standing {row['standing']} fail {row['failed_checks']}", flush=True)
    result = dict(
        status='MEASURED_SOURCED_SENSORY_RUN',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; sourced parameters, one citation each',
        question='What changes when the sensory front end uses the normalisers and the quantity '
                 'mapping the literature actually supports?',
        seated_height_mm=seated_height_mm,
        sourced_config=sourced,
        summary=summary,
        tau_window=dict(published_window_ms=tau_window, sourced_rule_value_s=sourced.get('filter_tau_s'),
                        runs={k: dict(filter_tau_s=v['filter_tau_s'],
                                      height_mean_last_50ms_mm=v['height_mean_last_50ms_mm'],
                                      height_std_last_50ms_mm=v['height_std_last_50ms_mm'],
                                      n_claw_contacts=v['standing']['measured']['n_claw_contacts'],
                                      standing=v['standing']['standing_under_this_projects_criteria'],
                                      failed_checks=v['standing']['failed_checks'])
                               for k, v in tau_runs.items()},
                        reading='the published window is a reading of measured sensory-relay and '
                                'effector-kinetics latencies, not a published rule; where inside it '
                                'the stance survives is measured here'),
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        sourced_config_sha256=__import__('hashlib').sha256(
                            (OUT/'sourced_sensory_config.json').read_bytes()).hexdigest(),
                        literature_parameters=str(HERE/'data/literature_sensory_parameters.json'),
                        tier_neurons=int(built['n']),
                        changed=['angle normaliser (published FeCO swept range)',
                                 'velocity normaliser (published swing speeds)',
                                 'the quantity the hair-plate slot carries (coxa swing, not a joint mean)'],
                        unchanged=['force normaliser (REFUSED by the gate: no published load range)',
                                   'claw/hook/club quantity choices', 'cells and tier membership',
                                   'connectome weights', 'conductance parameters', 'channel partition',
                                   'muscle pool manifest', 'decoder', 'clocks', 'translation table'],
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'A sourced normaliser is a published STIMULUS range, not a published gain: no gain in '
            'spikes per degree or per radian exists for any Drosophila femoral-chordotonal neuron, so '
            'the derivation rule (half the published span) is this project\'s, recorded in the artifact.',
            'The force normaliser is still unsourced and the gate refused it; the load channel keeps a '
            'magnitude nobody has published.',
            'No standing, walking or behaviour claim beyond the reported checklist.',
            'A change in the drive does not become evidence about the fly: the connectome is the same '
            'and the translation is still this project\'s.',
        ])
    cp.write_result(OUT/'flybody_sourced_sensory_run.json', question=result['question'], status='PASS',
                    measured={k: result[k] for k in ('seated_height_mm', 'sourced_config', 'summary',
                                                     'tau_window')},
                    provenance=result['provenance'],
                    units=dict(time='s', height='mm', angle='rad', current='nA',
                               normaliser='published span, converted to rad or rad/s'),
                    label=result['label'],
                    status_meaning='PASS means every configuration ran; it is not a calibration',
                    illustrative_vs_measured='Drives, currents, couplings and contact states are '
                                             'measured; the normalisers and the roles come from the '
                                             'cited literature through the gate',
                    what_this_does_not_support=result['what_this_does_not_support'])
    print('done', OUT/'flybody_sourced_sensory_run.json', flush=True)
    return result


if __name__ == '__main__':
    with mg.guarded_run('flybody_sourced_run', need_mb=2200.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
