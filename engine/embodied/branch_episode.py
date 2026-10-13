"""Explicit isolated fine-clock copy; canonical episode remains frozen.

Only the required body_dt argument and DT_BODY substitutions differ in the
function. verify_source_copy() checks this exact allowed patch before execution.
Helpers are canonical, not reimplemented. No module-global timestep override.
"""
from __future__ import annotations
import difflib
import inspect
import hashlib
import run_neural_muscle_closure as canonical
from run_neural_muscle_closure import (np, grid_steps, push_steps, DT_COMMAND,
    DT_NEURAL, ChannelSensoryEncoder, ChannelConfig, MotorOutput,
    MotorOutputConfig, ConductanceNetwork, ConductanceParams,
    observe_quantities, cp)


def episode(model, seated, joints, geom_leg, built, manifest, groups, arm, duration, replay=None,
            stride=10, full_detail_intervals=20, push=None, *, body_dt):
    import mujoco
    intervals = grid_steps(duration, DT_COMMAND)
    body_steps = grid_steps(DT_COMMAND, body_dt)
    if not np.isclose(model.opt.timestep, body_dt, rtol=0, atol=1e-15):
        raise ValueError('episode body timestep differs from declared clock')
    neural_steps = grid_steps(DT_COMMAND, DT_NEURAL)
    window = push_steps(push, duration, body_dt)
    if stride < 1:
        raise ValueError('record stride must be >= 1')
    d = mujoco.MjData(model)
    d.qpos[:] = seated
    d.qvel[:] = 0
    d.act[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(model, d)
    thorax = model.body('Thorax').id
    h0 = float(d.xpos[thorax, 2])
    enc = ChannelSensoryEncoder(groups, int(built['n']), ChannelConfig())
    decoder = MotorOutput(manifest, MotorOutputConfig())
    net = ConductanceNetwork(int(built['n']), ConductanceParams(), dt_ms=DT_NEURAL*1000)
    net.set_connectivity(built['edges']['We']*.2, built['edges']['Wi']*.2)
    pending = np.zeros(model.nu)
    source = -1
    records, trajectories, commands, spike_counts = [], [], [], []
    # Compact diagnostic raster: only mapped motor and explicitly stimulated sensory cells.
    # Preserves 0.5ms spike timing without storing a full-tier raster in every arm.
    probe_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']} |
                           {int(i) for i in groups['lf']['bristle_touch']})
    escape = None
    motor_indices = sorted({i for row in manifest['rows'] for i in row['local_indices']})
    cut_s = .015
    for k in range(intervals):
        t = k*DT_COMMAND
        quantities = observe_quantities(model, d, seated, joints, geom_leg)
        current, channels = enc.encode(quantities)
        # Declared experimenter's local tactile stimulus, never motor-targeted.
        # Compare it to a matched sensory-cut arm: zero all input currents there.
        stimulus_on = arm != 'no_external_touch' and .005 <= t < .015
        if stimulus_on:
            for i in groups['lf']['bristle_touch']:
                current[int(i)] += .3
        if arm == 'sensory_cut':
            current[:] = 0
        sp = np.zeros((neural_steps, int(built['n'])), dtype=bool)
        for j in range(neural_steps):
            sp[j] = net.step(current_nA=current)
        probe_raw_events = np.argwhere(sp[:, probe_indices]).tolist()
        if arm == 'spike_output_silence':
            sp[:] = False  # output silence intervention, not a claim of resetting net
        applied = pending.copy()
        if arm == 'motor_cut' and t >= cut_s:
            applied[:] = 0
        if arm == 'replay':
            applied = np.asarray(replay[k], float).copy()
        d.ctrl[:] = applied
        # EXTERNAL PERTURBATION: an experimenter-applied world-frame force on the thorax, applied
        # IDENTICALLY in every arm.  It is not a neural command and never targets a motor neuron
        # pool; it exists so "does the loop change recovery" can be asked at all.  Cleared on
        # every interval so an earlier push cannot leak into a later one.
        push_applied_steps = 0
        for substep in range(body_steps):
            absolute_step = k*body_steps + substep
            push_on = window is not None and window[0] <= absolute_step < window[1]
            d.xfrc_applied[:] = 0.0
            if push_on:
                d.xfrc_applied[thorax, :3] = push['force_model']
                push_applied_steps += 1
            mujoco.mj_step(model, d)
            mujoco.mj_forward(model, d)
            if escape is None and d.xpos[thorax, 2] < .9*h0:
                escape = float(d.time)
        if not all(np.all(np.isfinite(x)) for x in (d.qpos,d.qvel,d.act)):
            raise FloatingPointError(f'nonfinite body in {arm} at {d.time}')
        if abs(d.time-(t+DT_COMMAND)) > 1e-10 or abs(net.t_ms/1000-(t+DT_COMMAND)) > 1e-10:
            raise AssertionError('body/neural clock drift or simulator reset')
        warnings = [int(w.number) for w in d.warning]
        if any(warnings):
            raise RuntimeError(f'MuJoCo warnings in {arm}: {warnings}')
        pending, trace = decoder.consume(sp, spike_dt_s=DT_NEURAL, start_s=t,
                                         end_s=t+DT_COMMAND,
                                         connected=not(arm == 'motor_cut' and t+DT_COMMAND >= cut_s))
        if source >= k:
            raise AssertionError('future spikes applied to body')
        record = dict(time_start_s=t, time_end_s=float(d.time), applied_source_interval=source,
                      source_completed_by_s=t, decoder=trace,
                      probe_raw_spike_events_step_column=probe_raw_events,
                      probe_emitted_spike_events_step_column=np.argwhere(sp[:, probe_indices]).tolist(),
                      command_origin='offline_normal_replay' if arm == 'replay' else 'previous_completed_neural_interval',
                      mujoco_warning_counts=warnings,
                      sensory_current_min_nA=float(current.min()), sensory_current_max_nA=float(current.max()),
                      external_lf_touch_stimulus=stimulus_on,
                      external_push_applied=bool(push_applied_steps),
                      external_push_body_steps=push_applied_steps,
                      external_push_duration_s=push_applied_steps*body_dt,
                      external_push_impulse_model=(np.asarray(push['force_model'])*push_applied_steps*body_dt).tolist() if push else [0.0,0.0,0.0],
                      motor_spikes=int(sp[:,motor_indices].sum()),
                      active_control_count=int(np.count_nonzero(applied)),
                      native_control_floor_clamp_count=int(np.sum((applied < model.actuator_ctrlrange[:,0]) & model.actuator_ctrllimited)),
                      height_model=float(d.xpos[thorax,2]), height_mm_declared=float(cp.model_to_mm(d.xpos[thorax,2])),
                      muscle_force_max_abs_model=float(np.max(np.abs(d.actuator_force))),
                      activation_max=float(d.act.max()))
        # Per-interval channel detail is kept for the first intervals and every `stride` after,
        # so artifact size stays bounded as the episode lengthens instead of growing per interval.
        detailed = k < full_detail_intervals or k % stride == 0
        record['detail'] = 'full' if detailed else 'trimmed'
        if detailed:
            record['channel_record'] = channels
        records.append(record)
        source = k
        trajectories.append(np.r_[d.qpos.copy(), d.qvel.copy(), d.act.copy()])
        commands.append(applied)
        spike_counts.append(sp.sum(axis=0))
    return dict(arm=arm, probe_local_indices=probe_indices,
                probe_event_encoding='[step_index,column_in_probe_local_indices]; spike time=interval_start+(step_index+1)*DT_NEURAL; raw before output silence, emitted after',
                probe_spike_dt_s=DT_NEURAL,
                escape_time_s=escape, initial_height_model=h0,
                final_height_model=float(d.xpos[thorax,2]), records=records), np.array(trajectories), np.array(commands), np.array(spike_counts)


def verify_source_copy():
    original = inspect.getsource(canonical.episode)
    actual = inspect.getsource(episode)
    expected = original.replace('push=None):', 'push=None, *, body_dt):').replace('DT_BODY', 'body_dt')
    if actual != expected:
        raise AssertionError('branch episode exceeds allowed dt-only patch: ' + ''.join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True))))
    return dict(canonical_file=inspect.getsourcefile(canonical.episode),
                canonical_function_sha256=hashlib.sha256(original.encode()).hexdigest(),
                branch_function_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                allowed_patch='required keyword-only body_dt; replace function DT_BODY references only; existing strict assertion retained',
                diff=''.join(difflib.unified_diff(original.splitlines(True), actual.splitlines(True), fromfile='canonical.episode', tofile='branch_episode.episode')))
