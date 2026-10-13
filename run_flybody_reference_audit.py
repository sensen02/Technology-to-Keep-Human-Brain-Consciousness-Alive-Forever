"""Where could a posture REFERENCE live in this loop, and which input carries the drive?

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

The drive audit measured that the connectome's drive correlates POSITIVELY with each hinge's own
deviation from neutral (36 of 36 hinges, median r = +0.76, peak at a +5 ms lag): the loop tracks and
amplifies the deviation instead of opposing it.  A loop without a reference cannot hold a posture, so
this file asks the two questions that decide whether a reference can be placed at all, using
measurements rather than assumptions:

  A. ATTRIBUTION   which sensory channel's input actually carries the drive?  Each channel is ablated
                   on its own (its input current is zeroed) and the drive magnitude, its coupling to
                   the deviation, and the standing checklist are re-measured.  A drive that is carried
                   by the contact-force channel cannot encode posture at all, because contact force is
                   not a posture error.

  B. CONVENTION    the two channels that DO carry posture information are the tibia-deviation channel
                   (`claw_position`) and the multi-joint-mean channel (`hairplate_angle`).  This
                   project records that channel polarity is an UNVALIDATED declared assumption and
                   that every channel is treated as SINGLE-SIGNED, which no evidence supports.  All
                   four sign combinations are therefore run as DECLARED conventions, each judged by a
                   criterion fixed in advance: does the drive become restoring (negative coupling),
                   and does the body pass the standing checklist?

  C. REFERENCE     if some convention makes the loop restoring, the loop is then asked whether it
                   also holds the stance and recovers from the same perturbation the drive audit used.

Nothing here claims the connectome was mis-wired, and no neuron-level evidence is offered for any
sign: a convention that works is a convention that is CONSISTENT with a posture-stabilising reflex,
not evidence about the fly.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import hashlib
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
import run_flybody_drive_audit as drive_audit
import run_neural_muscle_closure as closure

OUT = loop.OUT
CHANNELS = ('claw_position', 'hook_direction', 'hairplate_angle', 'campaniform_strain',
            'bristle_touch')
POSTURE_CHANNELS = ('claw_position', 'hairplate_angle')
DURATION = .2
PUSH = dict(start_s=.15, end_s=.155, weight_fraction=.5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def channel_currents(summary, cfg):
    """Per-channel input current actually produced, from the recorded quantities and the DECLARED
    encoder normalisers.  This is what explains an ablation result quantitatively."""
    rows = {c: [] for c in ('claw_position', 'hook_direction', 'hairplate_angle',
                            'campaniform_strain', 'bristle_touch')}
    for record in summary['records']:
        quantities = record.get('sensory_quantities')
        if not quantities:
            continue
        for leg, q in quantities.items():
            rows['claw_position'].append(abs(q['tibia_joint_dev_rad']/cfg.reference_angle_rad))
            rows['hook_direction'].append(abs(q['tibia_joint_vel_rad_s']/cfg.reference_velocity_rad_s))
            rows['hairplate_angle'].append(abs(q['multi_joint_dev_rad']/cfg.reference_angle_rad))
            rows['campaniform_strain'].append(abs(q['contact_force_norm']/cfg.reference_force))
            rows['bristle_touch'].append(abs(q['foot_contact']))
    means = {c: float(np.mean(v)) if v else None for c, v in rows.items()}
    # MEASURED BUG (fixed): the share divided a current (already scaled by current_nA_per_unit) by an
    # UNSCALED total, so every share came out 1/current_nA_per_unit too small and the shares summed
    # to 0.30 instead of 1.  The denominator is scaled the same way now.
    total = cfg.current_nA_per_unit*sum(v for v in means.values() if v)
    return dict(declared_normalisers=dict(reference_angle_rad=cfg.reference_angle_rad,
                                          reference_velocity_rad_s=cfg.reference_velocity_rad_s,
                                          reference_force=cfg.reference_force,
                                          current_nA_per_unit=cfg.current_nA_per_unit),
                per_channel=dict(mean_abs_current_nA={c: (None if means[c] is None
                                                          else round(cfg.current_nA_per_unit*means[c], 5))
                                                      for c in means},
                                 share_of_total_current={c: (None if means[c] is None else
                                                             round(cfg.current_nA_per_unit*means[c]/total, 4))
                                                         for c in means}),
                derived_what_parity_would_need=dict(
                    current_translation=dict(reference_angle_rad=cfg.reference_angle_rad,
                                             reference_force=cfg.reference_force),
                    contact_over_tibia_deviation_current_ratio=(
                        round(means['campaniform_strain']/means['claw_position'], 4)
                        if means['claw_position'] else None),
                    reference_angle_rad_for_tibia_deviation_parity=(
                        round(cfg.reference_angle_rad*means['claw_position']
                              / means['campaniform_strain'], 5)
                        if means['campaniform_strain'] else None),
                    note='THIS IS A DERIVED NUMBER, NOT A RECOMMENDATION: it says how much the declared '
                         'angle normaliser would have to change for the tibia-deviation channel to '
                         'reach the same input current as the contact-force channel.  It has no '
                         'source in this project, so it is reported and not applied.'),
                note='the four active channels share ONE declared scale, so their RELATIVE strength is '
                     'set entirely by the declared normalisers; the contact-force channel reaches a '
                     'much larger normalised value than the joint-deviation channels at this pose')


def run(body_kind, built, manifest, groups, seated_height_mm, *, channel_signs=None,
        channel_ablation=None, duration=DURATION, push=None, sensory_roles=None, **kwargs):
    body = loop.build_body(body_kind, 'declared')
    extra = lambda d, b=body: drive_audit.assess(b, d, b['weight_per_leg'])
    push_spec = None
    if push is not None:
        push_spec = dict(push, force_model=[push['weight_fraction']*body['weight'], 0.0, 0.0],
                         note="world-frame force on the thorax as a fraction of this body's own weight")
    summary, trajectory, activation, _ = loop.episode(
        body, built, manifest, groups, 'normal', duration, push=push_spec,
        channel_signs=channel_signs, channel_ablation=channel_ablation,
        sensory_roles=sensory_roles, record_extra=extra, **kwargs)
    hinges, drive, deviation = drive_audit.matrices(summary)
    coupling = drive_audit.correlation_report(hinges, drive, deviation)
    heights = np.array([r['height_mm'] for r in summary['records']])
    times = np.array([r['time_start_s'] for r in summary['records']])
    row = dict(sensory_roles=sensory_roles, escape_time_s=summary['escape_time_s'],
               final_height_mm=summary['final_height_mm'],
               final_state=summary['records'][-1].get('extra', {}),
               mean_abs_drive=float(np.mean(np.abs(drive))),
               mean_abs_drive_per_hinge={h: round(float(np.mean(np.abs(drive[:, j]))), 6)
                                         for j, h in enumerate(hinges)},
               posture_reference=coupling,
               height_series_mm=[round(float(x), 4) for x in heights],
               height_series_times_s=[round(float(x), 4) for x in times],
               **drive_audit.stability(summary, trajectory))
    row['standing'] = drive_audit.standing_checklist(row, row['final_state'], seated_height_mm)
    if channel_ablation is None and channel_signs is None:
        row['channel_currents'] = channel_currents(summary, closure.ChannelConfig())
    if push is not None:
        pre = float(np.mean(heights[times < push['start_s']]))
        post = heights[times >= push['start_s']]
        row.update(height_before_push_mm=pre, band_mm=.9*pre,
                   height_min_after_push_mm=float(post.min()),
                   returned_to_band=bool(post[-1] >= .9*pre))
    return row


def attribution(built, manifest, groups, seated_height_mm):
    rows = {'none_ablated': run('flybody_servo', built, manifest, groups, seated_height_mm)}
    for channel in CHANNELS:
        rows[f'ablate_{channel}'] = run('flybody_servo', built, manifest, groups, seated_height_mm,
                                        channel_ablation={channel})
    rows['ablate_all_five'] = run('flybody_servo', built, manifest, groups, seated_height_mm,
                                  channel_ablation=set(CHANNELS))
    base = rows['none_ablated']['mean_abs_drive']
    for name, row in rows.items():
        row['drive_fraction_of_baseline'] = (row['mean_abs_drive']/base if base else None)
    return dict(channels=list(CHANNELS), runs=rows,
                reading='a channel whose ablation barely moves the drive carries almost none of it, '
                        'so a posture reference cannot be placed through it')


def conventions(built, manifest, groups, seated_height_mm):
    rows = {}
    for claw in (1, -1):
        for hair in (1, -1):
            signs = {'claw_position': claw, 'hairplate_angle': hair}
            name = f'claw{claw:+d}_hair{hair:+d}'
            rows[name] = run('flybody_servo', built, manifest, groups, seated_height_mm,
                             channel_signs=signs)
            rows[name]['channel_signs'] = signs
    for name, row in rows.items():
        ref = row['posture_reference']
        row['restoring_verdict'] = dict(
            median_correlation=ref['median_correlation'],
            n_restoring=ref['n_restoring_negative'], n_reinforcing=ref['n_reinforcing_positive'],
            is_restoring=bool(ref['n_restoring_negative'] > ref['n_reinforcing_positive']))
    return dict(posture_channels=list(POSTURE_CHANNELS), runs=rows,
                criterion='DECLARED IN ADVANCE: a convention is "restoring" when more hinges couple '
                          'negatively than positively, and it is only interesting if it ALSO passes '
                          'the standing checklist',
                caveat='every channel here is treated as SINGLE-SIGNED; a real channel probably '
                       'contains cells of both tuning directions, and this front end cannot '
                       'represent that')


def main(guard_report=None):
    started = time.time()
    import mujoco
    pools = json.loads(closure.POOLS.read_text())
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    probe = cp.load_model()
    names = [mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(probe.nu)]
    manifest = closure.build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    probe_body = loop.build_body('flybody_servo', 'declared')
    data = mujoco.MjData(probe_body['model'])
    data.qpos[:] = probe_body['seated']
    data.qvel[:] = 0.0
    mujoco.mj_forward(probe_body['model'], data)
    seated_height_mm = float(data.subtree_com[probe_body['model'].body('thorax').id][2]
                             - data.geom_xpos[probe_body['floor']][2])*10.0
    att = attribution(built, manifest, groups, seated_height_mm)
    fmt = lambda v: ('n/a' if v is None else f'{v:+.3f}')
    for name, row in att['runs'].items():
        print(f"  {name:26s} drive/baseline {row['drive_fraction_of_baseline']:.3f} "
              f"median_r {fmt(row['posture_reference']['median_correlation'])} "
              f"hinges_with_coupling {row['posture_reference']['n_with_defined_correlation']}/36 "
              f"standing {row['standing']['standing_under_this_projects_criteria']} "
              f"failed {row['standing']['failed_checks']}", flush=True)
    conv = conventions(built, manifest, groups, seated_height_mm)
    for name, row in conv['runs'].items():
        print(f"  {name:16s} median_r {fmt(row['posture_reference']['median_correlation'])} "
              f"restoring {row['restoring_verdict']['n_restoring']}/36 "
              f"standing {row['standing']['standing_under_this_projects_criteria']}", flush=True)
    restoring = [n for n, r in conv['runs'].items() if r['restoring_verdict']['is_restoring']
                 and r['standing']['standing_under_this_projects_criteria']]
    reference = {}
    if restoring:
        best = restoring[0]
        reference = dict(convention=best, channel_signs=conv['runs'][best]['channel_signs'],
                         with_perturbation=run('flybody_servo', built, manifest, groups,
                                               seated_height_mm, duration=.4, push=PUSH,
                                               channel_signs=conv['runs'][best]['channel_signs']),
                         note='this convention makes the loop restoring AND passes the standing '
                              'checklist; it is a CONSISTENT convention, not evidence about the fly')
        print('  restoring + standing convention:', best, flush=True)
    else:
        print('  no sign convention was both restoring and standing', flush=True)
    result = dict(
        status='MEASURED_REFERENCE_SURVEY',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; declared channel conventions only',
        question='Which sensory input carries the connectome\'s motor drive, and can a posture '
                 'reference be placed through a declared channel-polarity convention?',
        seated_height_mm=seated_height_mm,
        attribution=att, conventions=conv, reference_convention=reference,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        translation_artifact=str(loop.TRANS),
                        translation_sha256=sha256(loop.TRANS), tier_neurons=int(built['n']),
                        changed=['declared per-channel input sign (channel_signs)',
                                 'declared per-channel input removal (channel_ablation)'],
                        unchanged=['cells and their tier membership', 'connectome weights',
                                   'conductance parameters', 'channel partition',
                                   'muscle pool manifest', 'decoder', 'clocks'],
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'A channel sign is a DECLARED convention with NO neuron-level evidence behind it; this '
            'project records that channel polarity is unvalidated, and every channel here is treated '
            'as single-signed, which a real mixed-tuning channel would violate.',
            'A convention that turns the coupling restoring and passes the standing checklist is a '
            'CONSISTENT convention, not a discovery about the fly and not a corrected connectome.',
            'Ablation removes an input current; it does not remove cells, synapses or weights, and a '
            'channel with no effect may simply be one whose cells have no modelled motor path.',
            'The standing checklist is this project\'s own definition; there is no external '
            'Drosophila benchmark to check it against.',
            'Nothing here is a behaviour claim, and the perturbation is an experimenter force.',
        ])
    cp.write_result(OUT/'flybody_reference_audit.json', question=result['question'], status='PASS',
                    measured={k: result[k] for k in ('seated_height_mm', 'attribution',
                                                     'conventions', 'reference_convention')},
                    provenance=result['provenance'],
                    units=dict(time='s', height='mm', angle='rad',
                               drive='dimensionless signed projection value',
                               correlation='Pearson r, dimensionless'),
                    label=result['label'],
                    status_meaning='PASS means the survey ran; it is a survey of declared conventions',
                    illustrative_vs_measured='Drives, couplings, heights and contact states are '
                                             'measured; the channel signs and the ablation set are '
                                             'DECLARED',
                    what_this_does_not_support=result['what_this_does_not_support'])
    print('done', OUT/'flybody_reference_audit.json', flush=True)
    return result


if __name__ == '__main__':
    with mg.guarded_run('flybody_reference', need_mb=2200.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
