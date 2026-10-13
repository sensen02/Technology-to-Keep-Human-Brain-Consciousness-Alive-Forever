"""Why does the same connectome help the canonical body and hurt FlyBody -- and what holds the
one configuration that stays up?

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.  This file is a diagnostic on the translation layer: it
runs the SAME loop with the SAME connectome and changes only declared translation knobs, so the
candidate explanations can be separated by measurement instead of by argument.

  Q1 MAGNITUDE   if the drive is simply too large, shrinking the declared per-hinge amplitude
                 should delay the collapse.
  Q2 OFFSET      if the drive carries a large constant component away from the published neutral
                 command, removing each hinge's own causal running mean should restore the body.
  Q3 REFERENCE   if the connectome knows where the neutral posture is, a hinge's drive should
                 OPPOSE that hinge's own deviation from neutral (negative correlation).  A drive
                 that reinforces the deviation carries no posture reference.
  Q4 MECHANISM   the one configuration that stays up is then interrogated: if repeating a single
                 interval's command forever (feedback removed from the motor path entirely) also
                 keeps the body up, then what holds it is a STATIC pattern of torques, not
                 feedback control, and no control claim may be made.

Nothing here tunes the loop to stand.  The gain scan is a sensitivity measurement, the running-mean
and frozen-command runs are declared interventions, and every claim is bounded by what Q3 and Q4
measure.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine.memory_guard as mg
import engine.canonical_physics as cp
import run_flybody_body as body_mod
import run_flybody_loop as loop
import run_neural_muscle_closure as closure

OUT = loop.OUT
GAINS = (0.0, .02, .05, .125, .25, loop.DRIVE_FRACTION)
DURATION = .2
FREEZE_AT = (5, 20, 30)          # command intervals, 5 ms each
HOLD_CANDIDATES = (('flybody_torque', 'measured'), ('flybody_torque', 'declared'),
                   ('flybody_servo', 'declared'))


def one_run(body, built, manifest, groups, arm, duration=None, **kwargs):
    return loop.episode(body, built, manifest, groups, arm,
                        DURATION if duration is None else duration, **kwargs)


TAUS_S = (.0025, .005, .01, .02, .04)
PUSH_AT_S, PUSH_LENGTH_S, PUSH_WEIGHT_FRACTION = .15, .005, .5


def bandwidth_scan(kind, tibia_sign, built, manifest, groups, seated_height_mm):
    """Does slowing the drive restore the stance?  The decoder's own filter tau is the knob.

    The harm measured on the published-servo body is in the drive's FLUCTUATION, so widening the
    translation's own population-rate filter should reduce it.  This is a sensitivity scan of a
    declared translation parameter, not a tuned result: the primary runs all use 5 ms.
    """
    rows = {}
    for tau in TAUS_S:
        body = loop.build_body(kind, tibia_sign)
        extra = lambda d, b=body: assess(b, d, b['weight_per_leg'])
        summary, trajectory, _, _ = one_run(body, built, manifest, groups, 'normal',
                                            filter_tau_s=tau, record_extra=extra)
        key = f'tau_{tau*1000:g}ms'
        rows[key] = dict(filter_tau_s=tau, escape_time_s=summary['escape_time_s'],
                         final_height_mm=summary['final_height_mm'],
                         final_state=summary['records'][-1].get('extra', {}),
                         **stability(summary, trajectory))
        rows[key]['standing'] = standing_checklist(rows[key], rows[key]['final_state'],
                                                   seated_height_mm)
    return dict(kind=kind, tibia_sign=tibia_sign, runs=rows, taus_s=list(TAUS_S),
                reading='the primary translation filter is 5 ms; this scan only asks whether the '
                        'harm the drive does scales with its own bandwidth')


def push_recovery(kind, tibia_sign, built, manifest, groups, seated_height_mm):
    """Apply one identical perturbation and ask which arms recover their stance.

    The force is a fraction of the body's own weight, applied to the thorax for 5 ms at 150 ms, in
    every arm identically.  Recovery is measured as: did the body return to its own stance band,
    how far did it fall, and did the feet come back?
    """
    duration = .4
    rows = {}
    arms = [('normal', dict()), ('spike_output_silence', dict()),
            ('static_brace_mean_only', dict(dc_mode='running_mean_only')),
            ('frozen_at_100ms', dict(freeze_after_interval=20))]
    for name, kwargs in arms:
        body = loop.build_body(kind, tibia_sign)
        extra = lambda d, b=body: assess(b, d, b['weight_per_leg'])
        push = dict(start_s=PUSH_AT_S, end_s=PUSH_AT_S+PUSH_LENGTH_S,
                    force_model=[PUSH_WEIGHT_FRACTION*body['weight'], 0.0, 0.0],
                    weight_fraction=PUSH_WEIGHT_FRACTION,
                    note="world-frame force on the thorax as a fraction of this body's own weight")
        # MEASURED BUG (fixed): this loop used to pass neither **kwargs nor a valid arm name, so
        # the "static brace" arm ran as a plain driving run and the "frozen" arm never froze --
        # three arms came back with bit-identical height series.  The intervention now travels with
        # the entry, exactly as it does in held_configuration.
        arm = 'spike_output_silence' if name == 'spike_output_silence' else 'normal'
        summary, trajectory, _, _ = one_run(body, built, manifest, groups, arm, duration=duration,
                                            push=push, record_extra=extra, **kwargs)
        heights = np.array([r['height_mm'] for r in summary['records']])
        times = np.array([r['time_start_s'] for r in summary['records']])
        pre = heights[times < PUSH_AT_S]
        post = heights[times >= PUSH_AT_S]
        band = .9*np.mean(pre)
        recovered = bool(len(post) and post[-1] >= band and np.mean(post[-10:]) >= band)
        rows[name] = dict(kind=kind, tibia_sign=tibia_sign, arm=arm, push=push,
                          height_series_mm=[round(float(h), 4) for h in heights],
                          height_series_times_s=[round(float(t), 4) for t in times],
                          escape_time_s=summary['escape_time_s'],
                          height_before_push_mm=float(np.mean(pre)),
                          height_min_after_push_mm=float(post.min()) if len(post) else None,
                          height_end_mm=float(heights[-1]),
                          returned_to_band=recovered,
                          band_mm=float(band),
                          final_state=summary['records'][-1].get('extra', {}),
                          **stability(summary, trajectory))
        rows[name]['standing'] = standing_checklist(rows[name], rows[name]['final_state'],
                                                    seated_height_mm)
    return dict(kind=kind, tibia_sign=tibia_sign, duration_s=duration,
                push_weight_fraction=PUSH_WEIGHT_FRACTION, runs=rows,
                reading='identical perturbation in every arm; a difference between arms is a '
                        'difference in recovery, and the perturbation is an experimenter force, '
                        'not a command and not a behaviour')


def assess(body, data, per_leg_weight):
    """Foot contacts, body-on-floor contacts, height stability and drift at the final state."""
    import mujoco
    claw = {leg: 0.0 for leg in body_mod.CANONICAL_LEGS}
    nonfoot, n_claw = [], 0
    vec = np.zeros(6)
    floor = body['floor']
    geom_name = lambda g: mujoco.mj_id2name(body['model'], mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        if floor not in (g1, g2):
            continue
        other = g2 if g1 == floor else g1
        mujoco.mj_contactForce(body['model'], data, ci, vec)
        leg = body_mod.leg_of(geom_name(other))
        if leg and geom_name(other).startswith('tarsal_claw'):
            claw[leg] += float(np.linalg.norm(vec[:3]))
            n_claw += 1
        else:
            nonfoot.append(geom_name(other))
    thorax = body['model'].body('thorax').id
    com = float(data.subtree_com[thorax][2] - data.geom_xpos[floor][2])
    return dict(com_height_mm=com*10.0, n_claw_contacts=n_claw,
                per_leg_claw_support={k: round(v/per_leg_weight, 4) for k, v in claw.items()},
                legs_with_contact=sorted(k for k, v in claw.items() if v > 0),
                body_floor_geoms=sorted(set(nonfoot)),
                root_xy_mm=[round(float(data.qpos[0])*10, 4), round(float(data.qpos[1])*10, 4)])


def stability(summary, trajectory=None):
    heights = [r['height_mm'] for r in summary['records']]
    tail = heights[-10:] if len(heights) >= 10 else heights
    h0 = summary['initial_height_mm']
    # MEASURED METRIC PROBLEM (recorded because it changes the reading): `escape_time_s` is a FIRST
    # CROSSING of 90% of the seated height, and a body whose only support is passive springs sags
    # through that line during its initial settle.  It therefore cannot separate a transient dip
    # from a collapse, and every configuration here is additionally reported by where it actually
    # ends up over its last 50 ms.
    drift = None
    if trajectory is not None and len(trajectory) >= 10:
        positions = np.asarray(trajectory)[-10:, :2]
        drift = float(np.max(np.ptp(positions, axis=0)))
    return dict(height_std_last_50ms_mm=float(np.std(tail)),
                drift_last_50ms_mm=drift,
                height_mean_last_50ms_mm=float(np.mean(tail)),
                height_min_mm=float(min(heights)),
                initial_height_mm=h0,
                held_at_the_end=bool(np.mean(tail) >= .9*h0),
                threshold_height_mm=float(.9*h0),
                framing_note='escape_time_s = first crossing of 90% of the seated height (fires on '
                             'the initial sag too); held_at_the_end = mean height over the last '
                             '50 ms at or above that threshold')


SEATED_HEIGHT_MM = None      # filled from the first body built; the published neutral height


def standing_checklist(stability_row, final_state, seated_height_mm):
    """This project's own standing criteria, applied to FlyBody and reported as a checklist.

    There is no external Drosophila kinematic or ground-reaction benchmark, so these five
    conditions are SELF-DEFINED and are reported one by one instead of as a single verdict:
      feet_down      at least five of the six claws carry contact
      only_feet      nothing except claw geoms touches the floor (no leg segment, no abdomen)
      not_drifting   the root moves less than 0.2 mm during the last 50 ms
      quiet          the height varies by less than 0.02 mm over the last 50 ms
      at_height      the height stays within 15% of the published neutral standing height
    """
    claws = final_state.get('n_claw_contacts') or 0
    legs = final_state.get('legs_with_contact') or []
    body_geoms = final_state.get('body_floor_geoms') or []
    drift = stability_row.get('drift_last_50ms_mm')
    checks = dict(
        feet_down=bool(claws >= 5 and len(legs) >= 5),
        only_feet=bool(body_geoms == []),
        not_drifting=bool(drift is not None and drift < .2),
        quiet=bool(stability_row['height_std_last_50ms_mm'] < .02),
        at_height=bool(.85*seated_height_mm <= stability_row['height_mean_last_50ms_mm']
                       <= 1.15*seated_height_mm))
    return dict(criteria='SELF-DEFINED by this project; no external Drosophila benchmark exists',
                measured=dict(n_claw_contacts=claws, legs_with_contact=sorted(legs),
                              non_claw_floor_geoms=sorted(body_geoms),
                              drift_last_50ms_mm=drift,
                              height_std_last_50ms_mm=stability_row['height_std_last_50ms_mm'],
                              height_mean_last_50ms_mm=stability_row['height_mean_last_50ms_mm'],
                              published_neutral_height_mm=seated_height_mm),
                checks=checks, standing_under_this_projects_criteria=all(checks.values()),
                failed_checks=sorted(k for k, v in checks.items() if not v))


def matrices(summary):
    records = [r for r in summary['records'] if r['drive_values']]
    hinges = sorted(records[0]['drive_values'])
    drive = np.array([[r['drive_values'][h] for h in hinges] for r in records])
    deviation = np.array([[r['hinge_deviation_rad'][h] for h in hinges] for r in records])
    if not (np.all(np.isfinite(drive)) and np.all(np.isfinite(deviation))):
        raise FloatingPointError('nonfinite drive or deviation in the audit records')
    return hinges, drive, deviation


LAGS = tuple(range(-10, 11))          # command intervals of 5 ms, so +-50 ms


def lagged_correlation(drive, deviation, lags=LAGS):
    """corr(drive(t), deviation(t + lag)) per lag: where in time the drive is coupled to the pose.

    A drive that opposes the deviation must be coupled at a NEGATIVE or zero lag with a negative
    sign.  A positive peak at a POSITIVE lag means the drive follows the deviation and adds to it,
    which is delayed positive feedback on the deviation, not posture control.
    """
    curve = {}
    for lag in lags:
        a, b = drive, deviation
        if lag > 0:
            a, b = a[:-lag], b[lag:]
        elif lag < 0:
            a, b = a[-lag:], b[:lag]
        if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        curve[lag] = float(np.corrcoef(a, b)[0, 1])
    if not curve:
        # MEASURED BUG (fixed): this early return omitted peak_lag_ms and the caller indexed it, so an
        # ablation that flattened a hinge's drive to a constant crashed the report instead of being
        # reported as "no defined coupling".
        return dict(lags_ms={}, peak_lag_intervals=None, peak_lag_ms=None, r_at_peak=None,
                    r_at_zero=None, reading='no coupling is defined for this hinge')
    peak = max(curve, key=lambda k: abs(curve[k]))
    return dict(lags_ms={str(k*5): round(v, 4) for k, v in sorted(curve.items())},
                peak_lag_intervals=peak, peak_lag_ms=peak*5,
                r_at_peak=round(curve[peak], 4),
                r_at_zero=round(curve.get(0, float('nan')), 4),
                reading='positive lag = the drive follows the deviation (drive(t) against '
                        'deviation(t+lag)); a posture reference would need a NEGATIVE r at zero or '
                        'negative lag')


def correlation_report(hinges, drive, deviation):
    rows = {}
    for j, hinge in enumerate(hinges):
        d, q = drive[:, j], deviation[:, j]
        r = None if (np.std(d) < 1e-12 or np.std(q) < 1e-12) else float(np.corrcoef(d, q)[0, 1])
        rows[hinge] = dict(lagged=lagged_correlation(d, q),
                           deviation_std_rad=float(np.std(q)),
                           deviation_max_abs_rad=float(np.max(np.abs(q))),
                           drive_dc=float(np.mean(d)), drive_ac_std=float(np.std(d)),
                           drive_max_abs=float(np.max(np.abs(d))),
                           correlation_drive_vs_deviation=r)
    values = [r['correlation_drive_vs_deviation'] for r in rows.values()
              if r['correlation_drive_vs_deviation'] is not None]
    peaks = [r['lagged']['peak_lag_ms'] for r in rows.values()
             if r['lagged']['peak_lag_ms'] is not None]
    zeros = [r['lagged']['r_at_zero'] for r in rows.values()
             if r['lagged']['r_at_zero'] is not None and np.isfinite(r['lagged']['r_at_zero'])]
    return dict(per_hinge=rows, n_hinges=len(rows), n_with_defined_correlation=len(values),
                median_peak_lag_ms=float(np.median(peaks)) if peaks else None,
                n_peak_at_positive_lag=sum(1 for p in peaks if p > 0),
                n_peak_at_negative_lag=sum(1 for p in peaks if p < 0),
                median_r_at_zero=float(np.median(zeros)) if zeros else None,
                median_correlation=float(np.median(values)) if values else None,
                n_restoring_negative=sum(1 for v in values if v < 0),
                n_reinforcing_positive=sum(1 for v in values if v > 0),
                reading='restoring (the drive opposes the hinge\'s own deviation from neutral) is a '
                        'NEGATIVE correlation: a positive drive raises that hinge coordinate in '
                        'both translation variants')


def polarity_probe(kind, tibia_sign, built, manifest, groups, seated_height_mm):
    """DIAGNOSTIC PROBE: does flipping the whole drive's sign change the mechanism?

    The measured coupling is a delayed POSITIVE correlation between the drive and each hinge's own
    deviation (the drive reinforces the deviation).  If that is the mechanism, then inverting the
    drive must (a) turn the correlation negative and (b) restore the stance.  If inverting it does
    not restore the stance, the sign is NOT the mechanism.  This is a probe, never a proposed fix:
    only the tibia has a geometric sign check here, so no global sign is known to be anatomically
    right, and a probe that happens to stand is not evidence for the biology.
    """
    rows = {}
    for polarity in (1.0, -1.0):
        body = loop.build_body(kind, tibia_sign)
        extra = lambda d, b=body: assess(b, d, b['weight_per_leg'])
        summary, trajectory, _, _ = one_run(body, built, manifest, groups, 'normal',
                                            drive_polarity=polarity, record_extra=extra)
        hinges, drive, deviation = matrices(summary)
        key = f'polarity_{polarity:+.0f}'
        rows[key] = dict(kind=kind, tibia_sign=tibia_sign, drive_polarity=polarity,
                         escape_time_s=summary['escape_time_s'],
                         final_height_mm=summary['final_height_mm'],
                         final_state=summary['records'][-1].get('extra', {}),
                         posture_reference=correlation_report(hinges, drive, deviation),
                         **stability(summary, trajectory))
        rows[key]['standing'] = standing_checklist(rows[key], rows[key]['final_state'],
                                                   seated_height_mm)
    return dict(kind=kind, tibia_sign=tibia_sign, runs=rows,
                reading='the probe is reported as a mechanism test: sign negative+restoring stance '
                        'would confirm that the harm is a sign/reference defect, and no change would '
                        'refute it')


def gain_and_reference(kind, tibia_sign, built, manifest, groups, seated_height_mm):
    runs, curve = {}, []
    extra = lambda b: (lambda d: assess(b, d, b['weight_per_leg']))
    body = loop.build_body(kind, tibia_sign)
    silence, silence_traj, _, _ = one_run(body, built, manifest, groups,
                                          'spike_output_silence', record_extra=extra(body))
    runs['spike_output_silence'] = dict(
        arm='spike_output_silence', drive_fraction=None, dc_mode='off',
        escape_time_s=silence['escape_time_s'], final_height_mm=silence['final_height_mm'],
        final_state=silence['records'][-1].get('extra', {}),
        **stability(silence, silence_traj))
    runs['spike_output_silence']['standing'] = standing_checklist(
        runs['spike_output_silence'], runs['spike_output_silence']['final_state'], seated_height_mm)
    for gain in GAINS:
        body = loop.build_body(kind, tibia_sign)
        summary, trajectory, activation, _ = one_run(body, built, manifest, groups, 'normal',
                                                     drive_fraction=gain,
                                                     record_extra=extra(body))
        runs[f'normal_gain{gain:g}'] = dict(
            arm='normal', drive_fraction=gain, dc_mode='off',
            escape_time_s=summary['escape_time_s'], final_height_mm=summary['final_height_mm'],
            identical_to_silence=bool(np.array_equal(trajectory, silence_traj)),
            final_state=summary['records'][-1].get('extra', {}),
            **stability(summary, trajectory))
        runs[f'normal_gain{gain:g}']['standing'] = standing_checklist(
            runs[f'normal_gain{gain:g}'], runs[f'normal_gain{gain:g}']['final_state'],
            seated_height_mm)
        curve.append(dict(drive_fraction=gain, escape_time_s=summary['escape_time_s'],
                          final_height_mm=summary['final_height_mm']))
    body = loop.build_body(kind, tibia_sign)
    summary, trajectory, activation, _ = one_run(body, built, manifest, groups, 'normal',
                                                 record_extra=extra(body))
    runs['normal_primary'] = dict(
        arm='normal', drive_fraction=loop.DRIVE_FRACTION, dc_mode='off',
        escape_time_s=summary['escape_time_s'], final_height_mm=summary['final_height_mm'],
        final_state=summary['records'][-1].get('extra', {}),
        **stability(summary, trajectory))
    runs['normal_primary']['standing'] = standing_checklist(
        runs['normal_primary'], runs['normal_primary']['final_state'], seated_height_mm)
    hinges, drive, deviation = matrices(summary)
    plain = correlation_report(hinges, drive, deviation)
    reference = dict(declared_translation=plain, frozen_command=None)
    body = loop.build_body(kind, tibia_sign)
    summary_dc, traj_dc, _, _ = one_run(body, built, manifest, groups, 'normal',
                                        dc_mode='running_mean', record_extra=extra(body))
    runs['normal_running_mean'] = dict(
        arm='normal', drive_fraction=loop.DRIVE_FRACTION, dc_mode='running_mean',
        escape_time_s=summary_dc['escape_time_s'], final_height_mm=summary_dc['final_height_mm'],
        identical_to_silence=bool(np.array_equal(traj_dc, silence_traj)),
        final_state=summary_dc['records'][-1].get('extra', {}),
        **stability(summary_dc, traj_dc))
    runs['normal_running_mean']['standing'] = standing_checklist(
        runs['normal_running_mean'], runs['normal_running_mean']['final_state'], seated_height_mm)
    _, drive2, deviation2 = matrices(summary_dc)
    reference['running_mean_translation'] = correlation_report(hinges, drive2, deviation2)
    if not runs['normal_gain0']['identical_to_silence']:
        raise AssertionError('gain 0 did not reproduce the silenced arm: the amplitude knob is not '
                             'the only difference between those two runs')
    return dict(kind=kind, tibia_sign=tibia_sign, nu=int(body['nu']),
                escape_time_silenced=silence['escape_time_s'], gain_curve=curve, runs=runs,
                posture_reference=reference)


def held_configuration(kind, tibia_sign, built, manifest, groups, seated_height_mm):
    """Q4: for the configuration that stays up, remove feedback from the motor path entirely."""
    import mujoco
    rows, frozen_dc = {}, {}
    # MEASURED BUG (fixed): every entry here used to be run with the arm name hard-coded to
    # 'normal', so the "silenced" row was silently a second driving run and its escape time was a
    # copy of the driving arm's.  The arm name now travels with the entry.
    arms = [('normal', 'normal', {}), ('spike_output_silence', 'spike_output_silence', {})]
    arms += [('static_brace_mean_only', 'normal', dict(dc_mode='running_mean_only'))]
    arms += [(f'frozen_at_{int(i*loop.DT_COMMAND*1000)}ms', 'normal', dict(freeze_after_interval=i))
             for i in FREEZE_AT]
    for name, arm, kwargs in arms:
        body = loop.build_body(kind, tibia_sign)
        summary, trajectory, activation, _ = one_run(
            body, built, manifest, groups, arm,
            record_extra=lambda d, b=body: assess(b, d, b['weight_per_leg']), **kwargs)
        final = summary['records'][-1].get('extra', {})
        rows[name] = dict(kind=kind, tibia_sign=tibia_sign, arm=arm,
                          freeze_after_interval=kwargs.get('freeze_after_interval'),
                          dc_mode=kwargs.get('dc_mode', 'off'),
                          final_state=final,
                          escape_time_s=summary['escape_time_s'],
                          final_height_mm=summary['final_height_mm'],
                          **stability(summary, trajectory))
        rows[name]['standing'] = standing_checklist(rows[name], final, seated_height_mm)
        if kwargs.get('freeze_after_interval') is not None:
            frozen = [r for r in summary['records'] if r['frozen_command']]
            if not frozen:
                raise AssertionError('the freeze never took effect')
            frozen_dc[name] = dict(
                intervals_after_freeze=len(frozen),
                activation_sum_of_the_frozen_command=float(np.sum(activation[
                    kwargs['freeze_after_interval']])),
                note='the same muscle-activation vector is re-applied on every later interval, so '
                     'the sensory->spike->pool path no longer reaches the body')
    return dict(kind=kind, tibia_sign=tibia_sign, runs=rows, freeze_details=frozen_dc)


def main(guard_report=None):
    started = time.time()
    pools = json.loads(closure.POOLS.read_text())
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    import mujoco
    probe = cp.load_model()
    canonical_names = [mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                       for i in range(probe.nu)]
    manifest = closure.build_motor_manifest(pools, built['tier']['root_ids'], canonical_names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    configurations, held = {}, {}
    seated_height_mm = None
    for kind, sign in HOLD_CANDIDATES:
        key = f'{kind}|{sign}'
        probe_body = loop.build_body(kind, sign)
        import mujoco as _mj
        _data = _mj.MjData(probe_body['model'])
        _data.qpos[:] = probe_body['seated']
        _data.qvel[:] = 0.0
        _mj.mj_forward(probe_body['model'], _data)
        seated_height_mm = float(_data.subtree_com[probe_body['model'].body('thorax').id][2]
                                 - _data.geom_xpos[probe_body['floor']][2])*10.0
        row = gain_and_reference(kind, sign, built, manifest, groups, seated_height_mm)
        configurations[key] = row
        print(key, 'published neutral height mm', round(seated_height_mm, 4))
        print(key, 'gain curve',
              [(p['drive_fraction'], None if p['escape_time_s'] is None
                else round(p['escape_time_s']*1000, 1)) for p in row['gain_curve']],
              'running-mean', row['runs']['normal_running_mean']['escape_time_s'],
              'median corr', round(row['posture_reference']['declared_translation']
                                   ['median_correlation'], 3), flush=True)
        for name, r in row['runs'].items():
            print(f"      {name:22s} standing={r['standing']['standing_under_this_projects_criteria']}"
                  f" failed={r['standing']['failed_checks']}", flush=True)
    for kind, sign in (('flybody_torque', 'measured'), ('flybody_torque', 'declared'),
                       ('flybody_servo', 'declared')):
        key = f'{kind}|{sign}'
        held[key] = held_configuration(kind, sign, built, manifest, groups, seated_height_mm)
        print(key, {k: (None if v['escape_time_s'] is None else round(v['escape_time_s']*1000, 1))
                    for k, v in held[key]['runs'].items()}, flush=True)
    probes = {f'{kind}|{sign}': polarity_probe(kind, sign, built, manifest, groups,
                                               seated_height_mm)
              for kind, sign in (('flybody_servo', 'declared'), ('flybody_torque', 'measured'))}
    for key, row in probes.items():
        print('polarity probe', key,
              {k: (None if v['escape_time_s'] is None else round(v['escape_time_s']*1000, 1))
               for k, v in row['runs'].items()},
              {k: (v['standing']['standing_under_this_projects_criteria'],
                   v['posture_reference']['median_r_at_zero'])
               for k, v in row['runs'].items()}, flush=True)
    bandwidth = {f'{k}|{s_}': bandwidth_scan(k, s_, built, manifest, groups, seated_height_mm)
                 for k, s_ in (('flybody_servo', 'declared'),)}
    for key, row in bandwidth.items():
        print('bandwidth', key, {k: (v['standing']['standing_under_this_projects_criteria'],
                                     round(v['height_mean_last_50ms_mm'], 3))
                                 for k, v in row['runs'].items()}, flush=True)
    recovery = {f'{k}|{s_}': push_recovery(k, s_, built, manifest, groups, seated_height_mm)
                for k, s_ in (('flybody_servo', 'declared'),)}
    for key, row in recovery.items():
        print('push recovery', key,
              {k: (v['returned_to_band'], v['standing']['standing_under_this_projects_criteria'],
                   None if v['height_min_after_push_mm'] is None
                   else round(v['height_min_after_push_mm'], 3))
               for k, v in row['runs'].items()}, flush=True)
    result = dict(
        status='MEASURED_DRIVE_AUDIT',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; translation-layer diagnostic',
        question='Is the harm the connectome does to FlyBody caused by the SIZE of its drive, by a '
                 'constant OFFSET in it, or by the absence of a posture REFERENCE -- and what holds '
                 'the one configuration that stays up?',
        duration_s=DURATION, gains_scanned=list(GAINS), freeze_intervals=list(FREEZE_AT),
        published_neutral_height_mm=seated_height_mm,
        configurations=configurations, held_configuration=held, polarity_probe=probes,
        bandwidth_scan=bandwidth, push_recovery=recovery,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        translation_artifact=str(loop.TRANS),
                        translation_sha256=loop.sha256(loop.TRANS),
                        tier_neurons=int(built['n']), usable_muscles=manifest['usable_actuators'],
                        declared_knobs=['per-hinge amplitude fraction of half the declared control '
                                        'range (drive_fraction)',
                                        'causal running-mean removal per hinge (dc_mode)',
                                        'frozen command after one interval (freeze_after_interval)'],
                        unchanged=['tier', 'connectome weights', 'conductance parameters',
                                   'sensory channel partition', 'encoder', 'muscle pool manifest',
                                   'decoder', 'clocks', 'translation coefficient table'],
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'The gain scan is a SENSITIVITY measurement: no gain in it is claimed to be correct, '
            'calibrated, or biological, and the fact that small drives let the PUBLISHED SERVO hold '
            'the body is not evidence that the connectome is controlling anything.',
            'The running-mean and frozen-command runs are declared interventions on the '
            'translation layer, not controllers and not claims about what the connectome would do.',
            'Q3 is a correlation over one 200 ms episode per configuration; it can show that a '
            'posture reference is absent, and it cannot prove that one is present.',
            'The polarity probe reverses the whole drive as a MECHANISM TEST, and it turned out to '
            'have NO discriminating power: the drive/deviation correlation is invariant under a '
            'global sign flip (both the drive and the response change sign), which is why the probe '
            'is reported with that caveat rather than as a result.  A global sign is in any case not '
            'known to be anatomically right: only the tibia has a geometric check, and it passes on '
            'four of six legs.',
            'The bandwidth scan moves a DECLARED translation parameter (the decoder\'s filter tau, '
            '5 ms in every primary run); it is a sensitivity measurement, not a tuned fix.',
            'The perturbation test applies an experimenter force to the thorax.  A difference '
            'between arms under it is a difference in recovery, not a behaviour, and not proof that '
            'the connectome controls anything.',
            'Q4 decides only whether feedback reaches the body in that configuration.  A frozen '
            'command that holds the body up is a STATIC torque pattern: no control, no reflex and '
            'no behaviour may be claimed from it.',
            'No standing, walking or gait claim is made for any configuration in this file, and the '
            'project\'s standing criteria (foot-contact-only support, height stability, no drift) '
            'are reported as measurements rather than as a pass/fail verdict.',
        ])
    cp.write_result(OUT/'flybody_drive_audit.json', question=result['question'], status='PASS',
                    measured={k: result[k] for k in ('duration_s', 'gains_scanned',
                                                     'freeze_intervals',
                                                     'published_neutral_height_mm',
                                                     'configurations', 'held_configuration',
                                                     'polarity_probe', 'bandwidth_scan',
                                                     'push_recovery')},
                    provenance=result['provenance'],
                    units=dict(time='s', height='mm', angle='rad',
                               drive='dimensionless signed projection value before the per-hinge '
                                     'amplitude is applied',
                               correlation='Pearson r, dimensionless'),
                    label=result['label'],
                    status_meaning='PASS means the audit ran and its internal consistency check held '
                                   '(the zero-amplitude run reproduces the silenced arm exactly)',
                    illustrative_vs_measured='Escapes, drives, deviations, correlations and contact '
                                             'states are measured; the amplitude values, the '
                                             'running-mean rule and the freeze points are DECLARED '
                                             'translation variants',
                    what_this_does_not_support=result['what_this_does_not_support'])
    import tools_flybody_audit_figure as audit_figure
    print('figure', audit_figure.figure(result, OUT/'flybody_drive_audit.png'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=DURATION)
    args = parser.parse_args()
    DURATION = args.duration
    with mg.guarded_run('flybody_drive_audit', need_mb=2200.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
