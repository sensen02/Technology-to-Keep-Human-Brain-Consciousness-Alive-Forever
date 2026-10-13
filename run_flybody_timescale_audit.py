"""Does the body itself say what the drive's time scale should be?

MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT.

The drive audit showed that the harm the connectome does to the published-servo body scales with the
drive's BANDWIDTH: slowing the decoder's own filter tau from 5 ms to 20 ms restored a six-foot
stance.  A scanned tau is tuning, so this file asks the body for the number instead:

    1. measure the body's OWN passive time constant -- the whole-body centre-of-mass response to a
       small vertical displacement, and the response of each of the 36 driven hinges to a small
       displacement of that hinge, both with the published neutral command and no neural drive;
    2. cross-check those measurements against the published servo's own gain and the published joint
       damping (tau ~ damping / gain), so a disagreement would be visible;
    3. DECLARE the rule "the drive's filter must not be faster than the body's own response" and run
       the loop at the sourced number -- if the sourced number does not reproduce the standing
       threshold, then the bandwidth that helps is NOT explained by the body's passive dynamics and
       a tuned tau must be reported as tuning.

Nothing here is a fix claim.  The sourced tau is a declared translation rule whose number comes from
the body; whether it works is then measured, and either answer is a result.
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
import run_flybody_body as body_mod
import run_flybody_loop as loop
import run_neural_muscle_closure as closure
import run_flybody_drive_audit as drive_audit

OUT = loop.OUT
STEP_DISPLACEMENT_MM = .05          # whole-body vertical perturbation, as a fraction of the height
JOINT_DISPLACEMENT_RAD = .10
SETTLE_S = .3
TAUS_TO_TEST = (.005, .01, .02, .04)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_constant(times, values, target=None, baseline=None):
    """Time for a response to decay to 1/e of its initial excursion, plus an exponential fit.

    `values` is the deviation from the relaxed state; the 1/e crossing is measured directly and an
    exponential is fitted to the decay envelope, so an oscillation cannot masquerade as a slow
    return without showing up as a poor fit.
    """
    values = np.asarray(values, float)
    times = np.asarray(times, float)
    initial = float(values[0])
    if abs(initial) < 1e-12:
        return dict(tau_one_over_e_s=None, tau_fit_s=None, fit_r2=None, note='no excursion')
    crossing = None
    for i in range(1, len(values)):
        if abs(values[i]) <= abs(initial)/np.e:
            crossing = float(times[i])
            break
    mask = np.abs(values) > 1e-15
    tau_fit = r2 = None
    if mask.sum() > 5:
        slope, intercept = np.polyfit(times[mask], np.log(np.abs(values[mask])), 1)
        if slope < 0:
            tau_fit = float(-1.0/slope)
            predicted = intercept + slope*times[mask]
            observed = np.log(np.abs(values[mask]))
            ss_res = float(np.sum((observed-predicted)**2))
            ss_tot = float(np.sum((observed-observed.mean())**2))
            r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else None
    return dict(tau_one_over_e_s=crossing, tau_fit_s=tau_fit, fit_r2=r2,
                initial_excursion=initial,
                residual_at_end=float(values[-1]),
                note='tau_one_over_e is a direct 1/e crossing; tau_fit is a log-linear fit whose r2 '
                     'shows whether the return is actually exponential')


def com_response(model, seated, data_template=None):
    """Whole-body centre-of-mass response to a small vertical displacement, no neural drive."""
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = seated
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    thorax = model.body('thorax').id
    floor = body_mod.floor_geom(model)
    height = lambda d: float(d.subtree_com[thorax][2] - d.geom_xpos[floor][2])
    relaxed = height(data)
    data.qpos[2] -= STEP_DISPLACEMENT_MM/10.0            # model lengths are cm
    mujoco.mj_forward(model, data)
    displaced = height(data)
    times, values = [], []
    steps = int(round(SETTLE_S/model.opt.timestep))
    for step in range(steps+1):
        if step % 10 == 0:
            times.append(float(data.time))
            values.append(height(data) - relaxed)
        if step < steps:
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    decay = time_constant(times, values)
    return dict(relaxed_height_mm=relaxed*10.0, displaced_height_mm=displaced*10.0,
                displacement_mm=(displaced-relaxed)*10.0, **decay,
                trace_times_s=[round(t, 4) for t in times],
                trace_deviation_mm=[round(v*10.0, 6) for v in values])


def joint_responses(model, seated):
    """Each driven hinge's own response to a small displacement, with the published neutral command."""
    import mujoco
    translation = json.loads(loop.TRANS.read_text())['measured']
    hinges = sorted(translation['compiled_drive_matrix'])
    floor = body_mod.floor_geom(model)
    rows = {}
    for hinge in hinges:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hinge)
        adr = int(model.jnt_qposadr[jid])
        dof = int(model.jnt_dofadr[jid])
        data = mujoco.MjData(model)
        data.qpos[:] = seated
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        relaxed = float(data.qpos[adr])
        data.qpos[adr] = relaxed + JOINT_DISPLACEMENT_RAD
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        times, values = [], []
        steps = int(round(SETTLE_S/model.opt.timestep))
        for step in range(steps+1):
            if step % 10 == 0:
                times.append(float(data.time))
                values.append(float(data.qpos[adr]) - relaxed)
            if step < steps:
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
        decay = time_constant(times, values)
        gain = float(model.actuator_gainprm[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, hinge)][0])
        damping = float(model.dof_damping[dof])
        rows[hinge] = dict(
            joint=hinge, qpos_adr=adr,
            trace_times_s=[round(t, 4) for t in times],
            trace_deviation_rad=[round(v, 6) for v in values],
            published_servo_gain=gain, joint_damping=damping,
            tau_from_gain_and_damping_s=(damping/gain if gain else None),
            initial_excursion_rad=JOINT_DISPLACEMENT_RAD, **decay)
    return rows


def summarise(rows, key):
    values = [r[key] for r in rows.values() if r[key] is not None]
    if not values:
        return dict(n=0)
    values = np.asarray(values, float)
    return dict(n=int(values.size), median_s=float(np.median(values)),
                min_s=float(values.min()), max_s=float(values.max()),
                p25_s=float(np.percentile(values, 25)), p75_s=float(np.percentile(values, 75)))


def main(guard_report=None):
    import mujoco
    started = time.time()
    model = mujoco.MjModel.from_xml_path(str(loop.XML_PUBLISHED))
    data = mujoco.MjData(model)
    seated = np.zeros(model.nq)
    seated[2] = body_mod.seat(model, data, gap_model=0.0)
    seated[3] = 1.0
    com = com_response(model, seated)
    joints = joint_responses(model, seated)
    analytic = summarise(joints, 'tau_from_gain_and_damping_s')
    measured = summarise(joints, 'tau_fit_s')
    crossing = summarise(joints, 'tau_one_over_e_s')
    good_fits = {k: v for k, v in joints.items()
                 if v['fit_r2'] is not None and v['fit_r2'] > .9}
    well_fitted = summarise(good_fits, 'tau_fit_s') if good_fits else dict(n=0)
    # The declared rule: the drive must not be faster than the body's own passive response.  The
    # number is the body's MEDIAN well-fitted hinge time constant, cross-checked against the
    # whole-body CoM mode; both are read off the published model.
    candidates = {}
    if well_fitted.get('n'):
        candidates['median_hinge_tau_fit'] = well_fitted['median_s']
    if crossing.get('n'):
        candidates['median_hinge_tau_one_over_e'] = crossing['median_s']
    if analytic.get('n'):
        candidates['median_hinge_damping_over_gain'] = analytic['median_s']
    if com.get('tau_fit_s'):
        candidates['whole_body_com_tau_fit'] = com['tau_fit_s']
    if not candidates:
        raise RuntimeError('no body time constant could be measured')
    tau_sourced = float(np.median(list(candidates.values())))
    print('measured time constants', {k: round(v, 5) for k, v in candidates.items()},
          '-> sourced tau', round(tau_sourced, 5), flush=True)

    pools = json.loads(closure.POOLS.read_text())
    built = closure.build_neural_tier(closure.TierConfig(background_mean_nA=0, background_sd_nA=0))
    probe = cp.load_model()
    names = [mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(probe.nu)]
    manifest = closure.build_motor_manifest(pools, built['tier']['root_ids'], names)
    groups, _ = closure.channel_index_map(int(built['n']), built['tier']['root_ids'])
    seated_height_mm = com['relaxed_height_mm']
    verdicts = {}
    for tau in sorted(set(TAUS_TO_TEST + (round(tau_sourced, 5),))):
        body = loop.build_body('flybody_servo', 'declared')
        extra = lambda d, b=body: drive_audit.assess(b, d, b['weight_per_leg'])
        summary, trajectory, _, _ = loop.episode(body, built, manifest, groups, 'normal', .2,
                                                 filter_tau_s=tau, record_extra=extra)
        row = dict(filter_tau_s=tau, is_the_sourced_value=bool(abs(tau-tau_sourced) < 1e-9),
                   escape_time_s=summary['escape_time_s'],
                   final_height_mm=summary['final_height_mm'],
                   final_state=summary['records'][-1].get('extra', {}),
                   **drive_audit.stability(summary, trajectory))
        row['standing'] = drive_audit.standing_checklist(row, row['final_state'], seated_height_mm)
        verdicts[f'tau_{tau*1000:g}ms'] = row
        print(f'  tau {tau*1000:6.2f} ms  height {row["height_mean_last_50ms_mm"]:.4f} mm  '
              f'standing {row["standing"]["standing_under_this_projects_criteria"]} '
              f'failed {row["standing"]["failed_checks"]}', flush=True)
    # push recovery at the sourced tau, against the primary 5 ms
    recovery = {}
    for tau in (loop.DRIVE_FRACTION*0 + .005, tau_sourced):
        body = loop.build_body('flybody_servo', 'declared')
        extra = lambda d, b=body: drive_audit.assess(b, d, b['weight_per_leg'])
        push = dict(start_s=.15, end_s=.155,
                    force_model=[.5*body['weight'], 0.0, 0.0], weight_fraction=.5,
                    note="world-frame force on the thorax as a fraction of this body's own weight")
        summary, trajectory, _, _ = loop.episode(body, built, manifest, groups, 'normal', .4,
                                                 filter_tau_s=tau, push=push, record_extra=extra)
        heights = np.array([r['height_mm'] for r in summary['records']])
        times = np.array([r['time_start_s'] for r in summary['records']])
        pre = float(np.mean(heights[times < .15]))
        post = heights[times >= .15]
        recovery[f'tau_{tau*1000:g}ms'] = dict(
            filter_tau_s=tau, height_before_push_mm=pre,
            height_min_after_push_mm=float(post.min()), height_end_mm=float(heights[-1]),
            band_mm=.9*pre, returned_to_band=bool(post[-1] >= .9*pre),
            final_state=summary['records'][-1].get('extra', {}),
            **drive_audit.stability(summary, trajectory))
    result = dict(
        status='MEASURED_BODY_TIMESCALE',
        label='MANUAL RESEARCH PROTOTYPE, NOT A PRODUCT; the body supplies the time scale',
        question='What is the published body\'s own passive response time, and does a drive filtered '
                 'at that sourced time scale behave differently from the scanned 5 ms?',
        declared_rule='the drive must not be faster than the body\'s own passive response, so the '
                      'decoder filter tau is set to the body\'s measured time constant',
        com_response=com, joint_time_constants=joints,
        summaries=dict(hinge_tau_fit=measured, hinge_tau_one_over_e=crossing,
                       hinge_damping_over_gain=analytic, well_fitted_only=well_fitted),
        sourced_candidates={k: round(v, 6) for k, v in candidates.items()},
        tau_sourced_s=tau_sourced,
        tau_grid_s=list(sorted(set(TAUS_TO_TEST + (round(tau_sourced, 5),)))),
        closure_at_each_tau=verdicts, push_recovery=recovery,
        provenance=dict(code=str(Path(__file__).resolve()), interpreter=sys.executable,
                        xml=str(loop.XML_PUBLISHED), xml_sha256=sha256(loop.XML_PUBLISHED),
                        translation_artifact=str(loop.TRANS),
                        translation_sha256=sha256(loop.TRANS),
                        tier_neurons=int(built['n']),
                        measured_on='the published FlyBody model with its own neutral command and no '
                                    'neural drive; the closure runs use the same tier, connectome, '
                                    'encoder, manifest and decoder with only the filter tau moved',
                        memory_guard=guard_report('completed') if guard_report else None,
                        wall_s=time.time()-started),
        what_this_does_not_support=[
            'A sourced time scale is a DECLARED translation rule, not a biological finding: the body '
            'measurement says how fast this model returns, not how fast a fly filters proprioception.',
            'If the sourced tau does not reproduce the standing threshold, then any tau that does is '
            'tuning and must be labelled as such.',
            'The joint time constants come from a small displacement at the seated pose; the response '
            'is pose dependent and only one pose was measured.',
            'No standing, walking or behaviour claim beyond the reported checklist, and the checklist '
            'is this project\'s own definition.',
        ])
    cp.write_result(OUT/'flybody_timescale_audit.json', question=result['question'], status='PASS',
                    measured={k: result[k] for k in
                              ('declared_rule', 'com_response', 'joint_time_constants', 'summaries',
                               'sourced_candidates', 'tau_sourced_s', 'tau_grid_s',
                               'closure_at_each_tau', 'push_recovery')},
                    provenance=result['provenance'],
                    units=dict(time='s', height='mm', angle='rad',
                               tau='s', damping='model torque units per rad/s',
                               gain='model torque units per rad'),
                    label=result['label'],
                    status_meaning='PASS means the body measurement and the closure runs completed',
                    illustrative_vs_measured='Every time constant is measured on the compiled model; '
                                             'the rule that ties the drive filter to it is DECLARED',
                    what_this_does_not_support=result['what_this_does_not_support'])
    print('done', OUT/'flybody_timescale_audit.json', flush=True)
    return result


if __name__ == '__main__':
    with mg.guarded_run('flybody_timescale', need_mb=2200.0, cap_mb=4608.0) as _rep:
        main(guard_report=_rep)
