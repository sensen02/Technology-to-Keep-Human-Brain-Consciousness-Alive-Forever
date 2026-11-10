"""Independent array audit and optional physics replay for neural_walk.py outputs.

Default execution reads saved arrays, then writes VERIFICATION.json, REPORT.md and
SUMMARY.png in the requested run folder. --no-write is strictly read-only.
--replay explicitly opts into full physics integration; it is never automatic.
Tarsus-body-origin motion is reported only as a slip proxy. Exact material-point
velocities require replay and are sampled, not continuous contact instrumentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z'
sys.path.insert(0, str(ROOT / 'tools'))
import dynamic_support_metrics as dsm

DT, CDT, PDT = .0000125, .001, .005
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def runs(mask):
    """Half-open True runs, including censored boundary runs."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def maximum_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b))) if a.size else 0.


def safe_number(value):
    """Serialize absent/nonfinite diagnostic numbers without JSON NaN."""
    if isinstance(value, dict):
        return {k: safe_number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_number(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_number(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def attitude(quat):
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q, axis=1)[:, None]
    w, x, y, z = q.T
    yaw = np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    tilt = np.arccos(np.clip(1-2*(x*x+y*y), -1, 1))
    return yaw, tilt


def sustained_loaded(load, off, on, minimum_frames):
    """Schmitt trigger with consecutive-frame confirmation, backdated on confirmation.

    The first frame establishes a state, not a transition. Unconfirmed excursions
    leave the previous state intact. Boundary swings are later marked censored.
    """
    load = np.asarray(load)
    state = bool(load[0] >= on)
    result = np.full(len(load), state, bool)
    count = 0
    for i, value in enumerate(load):
        switch = value <= off if state else value >= on
        if switch:
            count += 1
            if count >= minimum_frames:
                state = not state
                result[i-count+1:i+1] = state
                count = 0
        else:
            count = 0
        result[i] = state
    return result


def policy_audit(folder, commands, policy, model_path, result, handover, cap):
    """Rebuild model inference from weights; do not import the runtime Policy."""
    w = load_npz(model_path)
    k = int(w['K'])
    lags = np.asarray(w['lags'], int)
    assert k >= 1 and np.all((lags >= 0) & (lags < k))
    assert w['w0'].shape == (128, len(lags)*42)
    assert w['w1'].shape == (128, 128) and w['w2'].shape == (42, 128)
    assert w['center'].shape == (42,) and np.all(w['scale'] > 0)
    targets, actions = policy['targets'], policy['actions']
    assert targets.shape == actions.shape == (len(commands[::5]), 42)
    buf, predictions, warm_history = None, [], None
    mode = result['controller']
    for i, target in enumerate(targets):
        x = (np.asarray(target, np.float32)-w['center'])/w['scale']
        if buf is None:
            buf = np.repeat(x[None], k, axis=0)
        else:
            buf = np.concatenate((buf[1:], x[None]), axis=0)
        tick = 5*i
        if tick == round(handover/CDT):
            warm_history = buf.copy()
        if tick < round(handover/CDT) or mode == 'teacher':
            continue
        if mode in ('hold', 'motor_cut'):
            pred_rate = np.zeros(42)
        else:
            z = buf[lags].reshape(1, -1)
            z = np.tanh(z @ w['w0'].T+w['b0'])
            z = np.tanh(z @ w['w1'].T+w['b1'])
            nxt = w['center']+w['scale']*(z @ w['w2'].T+w['b2'])[0]
            pred_rate = np.clip((nxt-target)/PDT, -cap, cap)
        predictions.append(maximum_error(pred_rate, actions[i]))
    q = commands[:, 1:43]
    rate, perturb = commands[:, 43:85], commands[:, 85:127]
    nxt = np.clip(q[:-1]+CDT*np.clip(rate[:-1]+perturb[:-1], -cap, cap), w['lo'], w['hi'])
    err = max(predictions, default=0.)
    checks = dict(
        target_integrator=maximum_error(nxt, q[1:]) < 1e-10,
        policy_targets_match_commands=maximum_error(targets, q[::5]) < 1e-12,
        policy_actions_match_commands=maximum_error(actions, rate[::5]) < 1e-12,
        rate_held_between_policy_ticks=maximum_error(rate, np.repeat(actions, 5, axis=0)[:len(rate)]) < 1e-12,
        bounded_policy_rate=bool(np.abs(rate).max() <= cap+1e-10),
        bounded_targets=bool(np.all(q >= w['lo']-1e-10) and np.all(q <= w['hi']+1e-10)),
        independently_reconstructed_policy=err < 5e-4,
        model_hash_matches=sha(model_path) == result.get('model_hash'))
    hand = folder/'HANDOVER.npz'
    hand_errors = {}
    if hand.exists() and warm_history is not None:
        saved = load_npz(hand)
        hand_errors = dict(history=maximum_error(saved['history'], warm_history),
                           target=maximum_error(saved['target'], q[round(handover/CDT)]))
        checks['real_warm_history_preserved'] = hand_errors['history'] < 1e-6 and hand_errors['target'] < 1e-12
    else:
        checks['real_warm_history_preserved'] = False
    return dict(passed=all(checks.values()), checks=checks, K=k, lags=lags,
                integration_max_error=maximum_error(nxt, q[1:]),
                policy_rate_max_error=err, handover_errors=hand_errors,
                rate_tolerance_rad_s=5e-4, target_tolerance_rad=1e-10,
                limitation='Warmup teacher labels are not reconstructed. No brain is run in this replacement-ANN arm.')


def gait_audit(native, body, handover, frequency, off_fraction, on_fraction, min_ms):
    bt = body['time']
    per = len(native)//len(bt)
    assert per == 400 and len(native) == len(bt)*per
    mean_force = native[:, 1:7].reshape(len(bt), per, 6).mean(axis=1)
    threshold_off, threshold_on = off_fraction*dsm.WEIGHT, on_fraction*dsm.WEIGHT
    minimum_frames = max(2, int(np.ceil(min_ms/1000/PDT)))
    loaded = np.column_stack([sustained_loaded(mean_force[:, j], threshold_off, threshold_on,
                                               minimum_frames) for j in range(6)])
    # A body row and its force average describe the interval ending at bt.
    valid = bt-PDT >= handover-1e-9
    ids = np.flatnonzero(valid)
    assert len(ids) >= 3, 'insufficient complete release body frames'
    first, end = ids[0], ids[-1]+1
    time = bt[first:end]
    com, feet = body['com'], body['feet']
    yaw, tilt = attitude(body['qpos'][:, 3:7])
    period = 1/frequency
    per_leg, swing_times = [], []
    duration = float(time[-1]-(time[0]-PDT))
    min_lifts = max(1, int(np.floor(duration*frequency*.5)))
    for j, name in enumerate(LEGS):
        local = loaded[first:end, j]
        swings, stance_proxy = [], []
        starts = []
        for a, b in runs(~local):
            # Both transitions must be observed within the whole release.
            if a == 0 or b == len(local) or b-a < minimum_frames:
                continue
            aa, bb = first+a, first+b
            starts.append(float(bt[aa]-PDT))
            swings.append(dict(start_s=float(bt[aa]-PDT), end_s=float(bt[bb-1]),
                               seconds=float((b-a)*PDT),
                               tarsus_origin_peak_z=float(feet[aa:bb, j, 2].max())))
        for a, b in runs(local):
            if b-a < minimum_frames:
                continue
            pts = feet[first+a:first+b, j, :2]
            stance_proxy.append(dict(seconds=float((b-a)*PDT),
                endpoint_2d_drift=float(np.linalg.norm(pts[-1]-pts[0])),
                path_2d_drift=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())))
        intervals = np.diff(starts)
        boundary_gaps = np.diff(np.r_[time[0]-PDT, starts, time[-1]])
        per_leg.append(dict(leg=name, duty=float(local.mean()), complete_swings=len(swings),
            complete_swing_intervals=swings, swing_start_intervals_s=intervals,
            maximal_gap_between_swings_s=float(boundary_gaps.max()),
            median_cycle_s=float(np.median(intervals)) if len(intervals) else None,
            stance_tarsus_origin_drift_proxy=stance_proxy,
            stance_proxy_path_mean=float(np.mean([x['path_2d_drift'] for x in stance_proxy])) if stance_proxy else None,
            stepping_passed=len(swings) >= min_lifts and boundary_gaps.max() <= 2.5*period))
        swing_times.append(np.array(starts))
    # Relative phase from observed LF swing starts, not target-nearest projection.
    lf = swing_times[0]
    for item, starts in zip(per_leg, swing_times):
        phases = []
        for t in starts:
            j = np.searchsorted(lf, t, side='right')-1
            if 0 <= j < len(lf)-1 and .5*period <= lf[j+1]-lf[j] <= 2*period:
                phases.append(float(2*np.pi*(t-lf[j])/(lf[j+1]-lf[j])))
        item['swing_phase_relative_to_observed_LF_cycles_rad'] = phases
    # Valid-window convolution: never pad a path with zeros or repeated endpoints.
    window = max(3, int(round(period/PDT)))
    window += (window % 2 == 0)
    if len(time) < window+2:
        smooth_t, smooth_xy = time, com[first:end, :2]
        smooth_yaw = yaw[first:end]
    else:
        kernel = np.ones(window)/window
        smooth_t = np.convolve(time, kernel, mode='valid')
        smooth_xy = np.column_stack([np.convolve(com[first:end, j], kernel, mode='valid') for j in (0, 1)])
        smooth_yaw = np.convolve(yaw[first:end], kernel, mode='valid')
    # Travel between cycle-spaced means suppresses within-cycle body oscillation.
    picks = np.arange(0, len(smooth_t), max(1, int(round(period/PDT))))
    if len(picks) and picks[-1] != len(smooth_t)-1:
        picks = np.r_[picks, len(smooth_t)-1]
    xy = smooth_xy[picks]
    cy = smooth_yaw[picks]
    increments = np.diff(xy, axis=0)
    midpoint_yaw = (cy[1:]+cy[:-1])/2
    forward = float(np.sum(increments[:, 0]*np.cos(midpoint_yaw)+increments[:, 1]*np.sin(midpoint_yaw)))
    lateral = float(np.sum(-increments[:, 0]*np.sin(midpoint_yaw)+increments[:, 1]*np.cos(midpoint_yaw)))
    path = float(np.linalg.norm(increments, axis=1).sum())
    direct = float(np.linalg.norm(xy[-1]-xy[0])) if len(xy)>1 else 0.
    span = float(smooth_t[-1]-smooth_t[0])
    locomotion = dict(release_body_seconds=duration, smoothing_window_s=window*PDT,
        cycle_mean_path_length=path, cycle_mean_net_displacement=direct,
        cycle_mean_body_forward_travel=forward, cycle_mean_body_lateral_travel=lateral,
        cycle_mean_forward_speed=forward/span if span > 0 else 0.,
        straightness=direct/path if path > 0 else 0.,
        yaw_start_rad=float(yaw[first]), yaw_end_rad=float(yaw[end-1]),
        heading_change_rad=float(yaw[end-1]-yaw[first]),
        heading_range_rad=float(np.ptp(yaw[first:end])),
        max_tilt_deg=float(np.degrees(tilt[first:end]).max()),
        raw_net_xy=(com[end-1, :2]-com[first, :2]),
        raw_origin_path_not_used_for_success=float(np.linalg.norm(np.diff(com[first:end, :2], axis=0), axis=1).sum()),
        sustained_forward_travel=forward > .05 and forward/span > .01 if span>0 else False)
    report = dict(threshold_off_fraction_weight=off_fraction, threshold_on_fraction_weight=on_fraction,
        confirmation_ms=minimum_frames*PDT*1000, minimum_complete_swings_per_leg=min_lifts,
        cadence_reference_hz=frequency, all_legs_stepping=all(x['stepping_passed'] for x in per_leg),
        per_leg=per_leg, locomotion=locomotion,
        limitation='Force is averaged over 5ms. >=10ms unloading is detected with hysteresis; censored boundary swings excluded. '
                   'Foot positions are Tarsus5 BODY ORIGINS, not ground-contact material points; drift and z are proxies, '
                   'not exact slip or sole clearance. Relative phase is measured from observed LF swings. '
                   'Cadence-based gates use the disclosed reference frequency, not an inferred neural phase.')
    plot_data = dict(time=bt, mean_force=mean_force, loaded=loaded, smooth_t=smooth_t,
                     smooth_xy=smooth_xy, cycle_xy=xy)
    return report, plot_data


def contact_row_and_material_velocity(mj, p, d):
    forces = np.zeros(6)
    nf_pos = nf_signed = 0.
    nc = 0
    velocity_sum, velocity_square_sum, weights = np.zeros(6), np.zeros(6), np.zeros(6)
    for i in range(d.ncon):
        co = d.contact[i]
        ga, gb = int(co.geom1), int(co.geom2)
        if not (p.planes[ga] or p.planes[gb]):
            continue
        wrench = np.zeros(6)
        mj.mj_contactForce(p.m, d, i, wrench)
        normal_force = float((co.frame.reshape(3, 3).T @ wrench[:3])[2])*(1 if p.planes[ga] else -1)
        geom = gb if p.planes[ga] else ga
        leg = p.feet[geom]
        if leg < 0:
            nf_pos += max(0., normal_force)
            nf_signed += normal_force
            nc += 1
            continue
        forces[leg] += normal_force
        if normal_force <= 0:
            continue
        bid = int(p.m.geom_bodyid[geom])
        twist = np.zeros(6)
        mj.mj_objectVelocity(p.m, d, mj.mjtObj.mjOBJ_BODY, bid, twist, 0)
        # mj_objectVelocity BODY returns the body inertial-frame origin velocity.
        # The frame origin is xipos; orientation is world when flg_local=0.
        material_velocity = twist[3:]+np.cross(twist[:3], co.pos-d.xipos[bid])
        # An independent point Jacobian guards against confusing BODY's inertial
        # origin with XBODY's body origin in MuJoCo spatial-velocity conventions.
        jac = np.zeros((3, p.m.nv))
        mj.mj_jac(p.m, d, jac, None, co.pos, bid)
        jac_velocity = jac @ d.qvel
        if maximum_error(material_velocity, jac_velocity) > 1e-7:
            raise AssertionError('material point twist/Jacobian velocity disagreement')
        speed = float(np.linalg.norm(material_velocity[:2]))
        velocity_sum[leg] += normal_force*speed
        velocity_square_sum[leg] += normal_force*speed*speed
        weights[leg] += normal_force
    row = np.r_[d.time, forces, nf_pos, nf_signed, nc, d.subtree_com[p.thorax, 2],
                d.qpos[:7], np.mean(d.ctrl**2), np.max(np.abs(d.qfrc_applied[:6]))]
    return row, velocity_sum, velocity_square_sum, weights


def replay(folder, commands, command_arrays, body, native, model_path, result, handover, cap, stride):
    """Independent actuator integration/torque calculation, full unchanged plant.

    No saved qpos/qvel are assigned. p.fresh() is the single initialization.
    Thereafter only controls are assigned; physics advances with mj_step.
    """
    sys.path.insert(0, str(BASE))
    import mujoco as mj
    from plant_probe import Plant
    from incremental_actuator import validate_full_model
    p = Plant()
    m, d = p.m, p.fresh()
    validate_full_model(m)
    assert np.isclose(m.opt.timestep, DT, rtol=0, atol=1e-15)
    w = load_npz(model_path)
    target = np.clip(d.qpos[p.qa].copy(), w['lo'], w['hi'])
    kp, kd, limits = [command_arrays[k] for k in ('kp', 'kd', 'limits')]
    errors = dict(command_target=0., qpos=0., qvel=0., ctrl=0., com=0., feet=0., native_sampled=0.)
    vv, v2, ww = np.zeros(6), np.zeros(6), np.zeros(6)
    samples = np.zeros(6, int)
    nbody, ncontact = 0, 0
    hand_error = None
    hand = load_npz(folder/'HANDOVER.npz') if (folder/'HANDOVER.npz').exists() else {}
    state_key = next((key for key in ('physics_integration', 'physics_state', 'integration') if key in hand), None)
    for k, command in enumerate(commands):
        errors['command_target'] = max(errors['command_target'], maximum_error(target, command[1:43]))
        if k == round(handover/CDT) and state_key:
            state = np.empty(mj.mj_stateSize(m, mj.mjtState.mjSTATE_INTEGRATION))
            mj.mj_getState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
            hand_error = maximum_error(state, hand[state_key])
        target = np.clip(target+CDT*np.clip(command[43:85]+command[85:127], -cap, cap), w['lo'], w['hi'])
        for sub in range(80):
            raw = kp*(target-d.qpos[p.qa])-kd*d.qvel[p.va]
            d.ctrl[:] = 0 if result['controller']=='motor_cut' and k >= round(handover/CDT) else np.clip(raw, -limits, limits)
            assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mj.mj_step(m, d)
            mj.mj_forward(m, d)
            index = k*80+sub
            if (index+1) % stride == 0:
                row, v, v_sq, weight = contact_row_and_material_velocity(mj, p, d)
                errors['native_sampled'] = max(errors['native_sampled'], maximum_error(row, native[index]))
                ncontact += 1
                if index >= round(handover/DT):
                    vv += v
                    v2 += v_sq
                    ww += weight
                    samples += weight > 0
        if k%5 == 4:
            saved = nbody
            values = dict(qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
                          com=d.subtree_com[p.thorax], feet=d.xpos[p.bids])
            for key, value in values.items():
                errors[key] = max(errors[key], maximum_error(value, body[key][saved]))
            nbody += 1
        assert np.isfinite(d.qpos).all() and not d.warning.number.any()
        if (k+1)%1000 == 0:
            print(json.dumps(dict(replay_seconds=(k+1)*CDT)), flush=True)
    mean = np.divide(vv, ww, out=np.full(6, np.nan), where=ww>0)
    rms = np.sqrt(np.divide(v2, ww, out=np.full(6, np.nan), where=ww>0))
    return dict(performed=True, passed=max(errors.values()) < 1e-8 and (hand_error is None or hand_error<1e-8),
        errors=errors, handover_integration_error=hand_error, tolerance=1e-8,
        full_model=dict(nq=m.nq, nv=m.nv, nu=m.nu, nmesh=m.nmesh, neq=m.neq,
                        gravity=m.opt.gravity.copy(), timestep=m.opt.timestep),
        body_frames=nbody, contact_samples=ncontact,
        material_contact_velocity=dict(force_weighted_mean_tangential_speed=mean,
            force_weighted_rms_tangential_speed=rms, loaded_samples=samples,
            sample_interval_s=stride*DT,
            method='Ground-plane tangential velocity at the material point currently contacting ground: '
                   'world BODY twist from mj_objectVelocity plus omega cross (contact.pos-xipos). '
                   'Each point velocity is independently checked against mj_jac @ qvel. '
                   'Force-weighted across all loaded tarsal contacts; no contact-point position differencing.',
            limitation='Sampled instantaneous material velocities; events between samples can be missed. '
                       'No ground compliance deformation model; native force units remain uncalibrated SI.'))


def make_plot(folder, native, body, data, report, handover):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    step = max(1, len(native)//12000)
    sample = native[::step]
    ax = axes[0]
    ax.plot(sample[:, 0], sample[:, 1:7].sum(1)/dsm.WEIGHT, label='foot load / weight', linewidth=.7)
    ax.plot(sample[:, 0], sample[:, 7]/dsm.WEIGHT, label='non-foot positive load / weight', linewidth=.7)
    ax.axvline(handover, color='k', linestyle='--')
    ax.set(ylabel='Native support', title='Support gates use every native sample; display is downsampled')
    ax.legend(loc='upper right')
    ax = axes[1]
    ax.plot(sample[:, 0], sample[:, 10], label='COM height')
    ax.axhline(.8*dsm.H0, color='r', linestyle=':', label='D8 floor')
    ax.axvline(handover, color='k', linestyle='--')
    ax.set(ylabel='Native length')
    ax.legend()
    ax = axes[2]
    ax.plot(body['com'][:, 0], body['com'][:, 1], alpha=.35, linewidth=.6, label='raw COM')
    ax.plot(data['smooth_xy'][:, 0], data['smooth_xy'][:, 1], label='cycle-window mean COM')
    ax.plot(data['cycle_xy'][:, 0], data['cycle_xy'][:, 1], 'o-', markersize=3, label='cycle-spaced mean positions')
    ax.set(xlabel='World x (native)', ylabel='World y (native)', aspect='equal')
    ax.legend()
    ax = axes[3]
    colors = plt.cm.tab10(np.arange(6))
    for j, leg in enumerate(LEGS):
        y = j+np.clip(data['mean_force'][:, j]/dsm.WEIGHT, 0, .5)
        ax.plot(data['time'], y, color=colors[j], linewidth=.7)
        ax.fill_between(data['time'], j, j+.35, where=data['loaded'][:, j], color=colors[j], alpha=.4, step='post')
    ax.axvline(handover, color='k', linestyle='--')
    ax.set(yticks=np.arange(6), yticklabels=LEGS, xlabel='Simulation seconds',
           title='5ms mean foot forces; shading = sustained loaded state, >=10ms confirmation')
    fig.suptitle(f"{folder.name}: support={report['support']['passed']}; stepping={report['gait']['all_legs_stepping']}; "
                 f"locomotion={report['verified_neural_locomotion']}")
    fig.savefig(folder/'SUMMARY.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder', type=Path)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--handover', type=float, default=1.)
    ap.add_argument('--frequency', type=float, default=None)
    ap.add_argument('--cap', type=float, default=30.)
    ap.add_argument('--min-release', type=float, default=10.)
    ap.add_argument('--unload-fraction', type=float, default=.002)
    ap.add_argument('--load-fraction', type=float, default=.01)
    ap.add_argument('--min-contact-ms', type=float, default=10.)
    ap.add_argument('--max-slip-speed-ratio', type=float, default=.5,
                    help='max each-leg force-weighted slip speed / cycle-mean forward speed')
    ap.add_argument('--replay', action='store_true', help='explicitly integrate full physics independently')
    ap.add_argument('--replay-stride', type=int, default=400, help='native substeps per contact-velocity sample')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args(argv)
    assert args.frequency is None or args.frequency > 0
    assert 0 <= args.unload_fraction < args.load_fraction and args.min_contact_ms >= 10
    assert args.replay_stride > 0 and args.min_release > 0
    folder = args.folder.resolve(strict=True)
    result = json.loads((folder/'RESULT.json').read_text())
    native = load_npz(folder/'NATIVE.npz')['rows']
    body = load_npz(folder/'BODY.npz')
    command_arrays = load_npz(folder/'COMMANDS.npz')
    commands = command_arrays['rows']
    policy = load_npz(folder/'POLICY.npz')
    model = args.model or Path(result['model'])
    w = load_npz(model)
    frequency = args.frequency or float(w['freq'])
    assert native.ndim == 2 and native.shape[1] == 20
    assert commands.ndim == 2 and commands.shape[1] == 127
    assert len(native) == 80*len(commands)
    assert all(np.isfinite(a).all() for a in [native, commands, *body.values(), *policy.values()])
    assert np.allclose(np.diff(native[:, 0]), DT, rtol=0, atol=1e-10)
    assert np.allclose(np.diff(commands[:, 0]), CDT, rtol=0, atol=1e-10)
    assert np.allclose(np.diff(body['time']), PDT, rtol=0, atol=1e-10)
    assert abs(commands[0, 0]) < 1e-12 and abs(native[0, 0]-DT) < 1e-12
    assert maximum_error(commands[:, 0]+CDT, native[79::80, 0]) < 1e-9
    assert maximum_error(body['time'], native[399::400, 0]) < 1e-9
    assist = native[(native[:, 0]>=args.handover-.2)&(native[:, 0]<args.handover)]
    release = native[native[:, 0]>=args.handover]
    assert len(assist)>100 and len(release)>100
    ref = dict(foot_support_mean=float(assist[:, 1:7].sum(1).mean()/dsm.WEIGHT),
               com_z_mean=float(assist[:, 10].mean()))
    support = dsm.evaluate_native(ref, release)
    support['checks']['D1_no_nonfoot_ground_force'] = bool(release[:, 7].max() == 0)
    support['passed'] = all(support['checks'].values())
    policy_result = policy_audit(folder, commands, policy, model, result, args.handover, args.cap)
    gait, plot_data = gait_audit(native, body, args.handover, frequency,
                               args.unload_fraction, args.load_fraction, args.min_contact_ms)
    physics = replay(folder, commands, command_arrays, body, native, model, result,
                     args.handover, args.cap, args.replay_stride) if args.replay else dict(performed=False, passed=None)
    seconds = len(commands)*CDT-args.handover
    checks = dict(native_D_gates=bool(support['passed']), saved_policy_and_integrator=bool(policy_result['passed']),
        all_six_legs_sustained_stepping=bool(gait['all_legs_stepping']),
        sustained_body_forward_travel=bool(gait['locomotion']['sustained_forward_travel']),
        whole_release_duration=seconds >= args.min_release-1e-9,
        saved_runtime_no_warnings=result.get('warnings') == 0,
        recorded_no_root_applied_force=bool(np.max(np.abs(release[:, 19])) == 0.),
        saved_body_hash_matches=sha(folder/'BODY.npz') == result.get('body_hash'),
        neural_deployment=result['controller'] == 'ann')
    material_pass = None
    if physics['performed']:
        speed = gait['locomotion']['cycle_mean_forward_speed']
        means = physics['material_contact_velocity']['force_weighted_mean_tangential_speed']
        ratios = np.asarray(means)/max(speed, 1e-12)
        physics['material_contact_velocity']['slip_to_forward_speed_ratio'] = ratios
        material_pass = bool(np.isfinite(ratios).all() and np.max(ratios) <= args.max_slip_speed_ratio and speed > .01)
    replay_ok = bool(physics['performed'] and physics['passed'])
    verified = bool(all(checks.values()) and replay_ok and material_pass)
    status = 'VERIFIED' if verified else 'FAILED' if not all(checks.values()) or (physics['performed'] and (not replay_ok or not material_pass)) else 'NOT_ESTABLISHED_WITHOUT_PHYSICS_REPLAY'
    outputs = [folder/name for name in ('VERIFICATION.json', 'REPORT.md', 'SUMMARY.png')]
    # Read any existing outputs before replacing them; keep their identities in the report.
    old_hashes = {str(p):sha(p) for p in outputs if p.exists()}
    inputs = [folder/name for name in ('RESULT.json', 'BODY.npz', 'NATIVE.npz', 'COMMANDS.npz', 'POLICY.npz')]+[model, Path(__file__), ROOT/'tools/dynamic_support_metrics.py']
    if (folder/'HANDOVER.npz').exists():
        inputs.append(folder/'HANDOVER.npz')
    report = dict(schema='independent-walk-verification-v1', folder=str(folder),
        controller=result['controller'], status=status, verified_neural_locomotion=verified,
        array_engineering_and_motion_checks_passed=all(checks.values()), checks=checks,
        release_seconds=seconds, minimum_release_seconds=args.min_release,
        support=support, policy=policy_result, gait=gait, physics_replay=physics,
        material_slip_passed=material_pass, max_material_slip_to_forward_speed_ratio=args.max_slip_speed_ratio,
        source_hashes={str(p):sha(p) for p in inputs}, previous_outputs=old_hashes,
        limitations=['Default audit does not integrate physics. Recorded forces/state alone do not prove absence of unlogged state writes.',
                     'Standing, displacement, force sign chatter, raw body oscillation and teacher motion are not autonomous locomotion.',
                     'Without replay, tarsus origin drift is only a proxy; planted-foot locomotion remains NOT_ESTABLISHED.',
                     'D gates are unchanged, including native-resolution differentiation and the strict positive-nonfoot override.',
                     'Motion thresholds are disclosed engineering criteria, not biological validation. Original-brain necessity and absolute SI forces NOT_ESTABLISHED.'])
    report = safe_number(report)
    if not args.no_write:
        outputs[0].write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        failed = [name for name, passed in checks.items() if not passed]
        text = (f"# Independent walking verification: {folder.name}\n\n"
                f"Status: **{status}**. Controller: `{result['controller']}`. Full release: {seconds:.3f} s.\n\n"
                f"Unchanged native support gates: {support['passed']}. All six legs sustain stepping: {gait['all_legs_stepping']}. "
                f"Independent target/policy reconstruction: {policy_result['passed']}. Physics replay performed: {physics['performed']}.\n\n"
                f"Cycle-mean body-forward travel: {gait['locomotion']['cycle_mean_body_forward_travel']:.6g} native; "
                f"forward speed: {gait['locomotion']['cycle_mean_forward_speed']:.6g}; "
                f"heading change: {gait['locomotion']['heading_change_rad']:.6g} rad.\n\n"
                f"Failed array/motion checks: {', '.join(failed) or 'none'}.\n\n"
                "Tarsus-origin drift is a proxy, not material-point contact slip. Exact sampled material velocities are available only with `--replay`. "
                "A final locomotion pass requires all array/motion checks, independent replay and the declared material-slip bound.\n\n"
                + '\n'.join('- '+item for item in report['limitations'])+'\n\n'
                + f"Complete evidence and thresholds: `{folder/'VERIFICATION.json'}`. Summary: `{folder/'SUMMARY.png'}`.\n")
        outputs[1].write_text(text)
        make_plot(folder, native, body, plot_data, report, args.handover)
    print(json.dumps(dict(folder=str(folder), status=status, checks=checks,
                          policy_rate_max_error=policy_result['policy_rate_max_error'],
                          support=support['checks'], gait=gait['locomotion'], written=not args.no_write),
                     default=lambda x: np.asarray(x).tolist(), allow_nan=False), flush=True)
    return 0 if verified else 2


if __name__ == '__main__':
    raise SystemExit(main())
