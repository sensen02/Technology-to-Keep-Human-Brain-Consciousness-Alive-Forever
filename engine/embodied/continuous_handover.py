"""P3 continuous assisted-walk -> neural-only transition on a caller-owned MjData.

Core reusable API, not a walking controller or evidence of successful behavior.
No MjData allocation/reset and no writes to qpos/qvel/act/time/warm-start state.
The caller owns pool admission and tier construction; pass a manifest built from
load_runtime_pool, not candidate data. No neural steps occur before this call.
"""
from __future__ import annotations
import numpy as np
from .closure_clock import body_clock_steps, grid_steps


def state_snapshot(data):
    """Owned boundary evidence, including persistent integration/user inputs."""
    fields = ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart',
              'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
              'userdata', 'eq_active', 'plugin_state')
    result = {name: np.asarray(getattr(data, name)).copy()
              for name in fields if hasattr(data, name)}
    result['time'] = float(data.time)
    return result


def _make_runtime(built, manifest, groups, neural_dt_s, motor_config, sensory_channel_weights=None, allow_signed_sensory_weights=False):
    # Lazy imports keep dependency-heavy legacy research runners out of API imports.
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    from .sensory_channels import ChannelSensoryEncoder, ChannelConfig
    from .motor_output import MotorOutput, MotorOutputConfig
    net = ConductanceNetwork(int(built['n']), ConductanceParams(),
                             dt_ms=neural_dt_s * 1000)
    net.set_connectivity(built['edges']['We'] * .2, built['edges']['Wi'] * .2)
    encoder = ChannelSensoryEncoder(groups, int(built['n']), ChannelConfig(), per_channel_weight=sensory_channel_weights,
                                   allow_signed_weights=allow_signed_sensory_weights)
    decoder = MotorOutput(manifest, motor_config or MotorOutputConfig())
    return net, encoder, decoder


def run_neural_continuous(model, data, built, manifest, groups, duration_s, *,
                          sensory_reference_qpos, joints, geom_leg,
                          body_dt_s=.0000125, neural_dt_s=.0005,
                          command_dt_s=.005, motor_config=None,
                          body_observer=None, interval_observer=None,
                          retain_currents=False, motor_output_connected=True, sensory_channel_weights=None, descending_command=None,
                          runtime_state=None, assisted_prewarm_callback=None,
                          assisted_neural_closedloop=False, allow_signed_sensory_weights=False):
    """Run neural-only dynamics without losing the warmup's physical state.

    Required reference is the frozen PRE-warmup posture, not the handover pose.
    ``joints``/``geom_leg`` use run_neural_muscle_closure.observe_quantities.
    ``body_observer(model,data)`` is called after withdrawal before integration,
    then after every mj_step/mj_forward. It must be read-only (checked). Existing
    warmup recorders can append both phases to the same arrays; boundary time may
    appear twice, once on either side of withdrawal. ``interval_observer(record)``
    gets a completed neural/physical window. Records use relative decoder time
    AND absolute body time. First command window is zero, not warmup command replay.

    Does not modify the model, enforce a pose, clamp native muscle activations,
    apply touch/push, or certify that the preceding warmup actually walked.
    Zero neural ctrl is not zero muscle activation: the native model's control
    floor and the transferred activation remain intact and must be disclosed.
    Returns owned boundary snapshots, full command-window records and counts;
    full body-state recording is caller-owned via body_observer.
    """
    import mujoco
    from run_neural_muscle_closure import observe_quantities
    per_window, total_steps = body_clock_steps(body_dt_s, neural_dt_s,
                                               command_dt_s, duration_s)
    n_windows = grid_steps(duration_s, command_dt_s)
    n_neural = grid_steps(command_dt_s, neural_dt_s)
    if not np.isclose(model.opt.timestep, body_dt_s, rtol=0, atol=1e-15):
        raise ValueError('body timestep must match the already compiled model')
    reference = np.asarray(sensory_reference_qpos, dtype=float).copy()
    if reference.shape != (model.nq,) or not np.isfinite(reference).all():
        raise ValueError('explicit finite warmup-initial sensory reference required')
    if int(manifest['n_actuators']) != model.nu or int(manifest['n_neurons']) != int(built['n']):
        raise ValueError('tier/manifest/model dimensions disagree')
    if any(w.number for w in data.warning):
        raise RuntimeError('handover received pre-existing MuJoCo warnings')
    before = state_snapshot(data)
    if not all(np.isfinite(before[k]).all() for k in ('qpos', 'qvel', 'act')) or not np.isfinite(before['time']):
        raise ValueError('nonfinite physical handover state')
    if runtime_state is None:
        net, encoder, decoder = _make_runtime(built, manifest, groups, neural_dt_s, motor_config, sensory_channel_weights,
                       allow_signed_sensory_weights=allow_signed_sensory_weights)
        pending = np.zeros(model.nu)
        if abs(float(net.t_ms)) > 1e-12:
            raise AssertionError('new network must start at neural time zero')
    else:
        net, encoder, decoder = runtime_state['runtime']
        pending = np.asarray(runtime_state['pending'], dtype=float).copy()
        if pending.shape != (model.nu,) or not np.isfinite(pending).all():
            raise ValueError('invalid retained neural command')
        if abs(float(data.time)-runtime_state['body_time']) > 1e-10:
            raise ValueError('retained runtime does not match continuous body time')
        if abs(float(net.t_ms)/1000-decoder.elapsed_s) > 1e-10:
            raise ValueError('retained neural/decoder clocks disagree')
    runtime_boundary = {name: np.array(value, copy=True) for name, value in vars(net).items() if isinstance(value, np.ndarray)}
    runtime_boundary['decoder_rates'] = np.asarray(getattr(decoder, 'rates', np.zeros(model.nu))).copy()
    runtime_boundary['pending'] = pending.copy()
    neural_origin_s = float(net.t_ms)/1000
    source_offset = int(round(neural_origin_s/command_dt_s))
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    data.ctrl[:] = pending if motor_output_connected else 0
    after = state_snapshot(data)
    protected = set(before) - {'ctrl', 'qfrc_applied', 'xfrc_applied'}
    if any(not np.array_equal(before[k], after[k]) for k in protected):
        raise AssertionError('withdrawal changed physical/integrator state')
    t0 = float(data.time)
    counts = dict(network_instances=int(runtime_state is None), neural_steps=0, body_steps=0, forward=0)

    def record_body():
        if body_observer is None:
            return
        prior = state_snapshot(data)
        body_observer(model, data)
        now = state_snapshot(data)
        if any(not np.array_equal(prior[k], now[k]) for k in prior):
            raise AssertionError('body_observer mutated live state/inputs')

    # Refresh derived contact/force quantities after removal, never integrate/reset.
    mujoco.mj_forward(model, data)
    counts['forward'] += 1
    record_body()
    motor_indices = sorted({int(i) for row in manifest['rows'] for i in row['local_indices']})
    records, commands, spike_counts = [], [], []
    for k in range(n_windows):
        relative_start = k * command_dt_s
        relative_end = (k + 1) * command_dt_s
        quantities = observe_quantities(model, data, reference, joints, geom_leg)
        current, channels = encoder.encode(quantities)
        current = np.asarray(current, dtype=float).copy()
        if current.shape != (int(built['n']),) or not np.isfinite(current).all():
            raise ValueError('invalid sensory current')
        spikes = np.zeros((n_neural, int(built['n'])), dtype=bool)
        descending_trace = []
        for j in range(n_neural):
            external_current = np.zeros_like(current)
            if descending_command is not None:
                from .descending_command import command_input
                external_current, command_record = command_input(built, descending_command, relative_start + j * neural_dt_s)
                descending_trace.append(dict(relative_s=relative_start + j * neural_dt_s, **command_record))
            spikes[j] = net.step(current_nA=current + external_current)
            counts['neural_steps'] += 1
        applied = pending.copy()
        if not motor_output_connected:
            applied[:] = 0  # causal readout cut only; network and decoded proposals continue
        data.ctrl[:] = applied
        for _ in range(per_window):
            # Neither previous assisted torques nor external forces can leak.
            data.qfrc_applied[:] = 0
            data.xfrc_applied[:] = 0
            if assisted_prewarm_callback is not None:
                assisted_prewarm_callback(model, data)
            mujoco.mj_step(model, data)
            counts['body_steps'] += 1
            # Declared assistance (a callback) is allowed; undeclared external force never is.
            if (assisted_prewarm_callback is None and np.any(data.qfrc_applied)) or np.any(data.xfrc_applied):
                raise AssertionError('external/auxiliary force appeared during neural-only step')
            mujoco.mj_forward(model, data)
            counts['forward'] += 1
            if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.act)):
                raise FloatingPointError('nonfinite neural-only body state')
            if any(w.number for w in data.warning):
                raise RuntimeError('MuJoCo warning during neural-only integration')
            record_body()
        if abs(data.time - (t0 + relative_end)) > 1e-10 or abs(net.t_ms / 1000 - (neural_origin_s + relative_end)) > 1e-10:
            raise AssertionError('body/neural clock drift or state reset')
        pending, trace = decoder.consume(spikes, spike_dt_s=neural_dt_s,
                                         start_s=neural_origin_s + relative_start, end_s=neural_origin_s + relative_end)
        phase_label = ('assisted_neural_closedloop' if assisted_neural_closedloop
                       else 'algorithm_assisted_neural_prewarm') if assisted_prewarm_callback is not None else 'neural_only'
        record = dict(interval=k, phase=phase_label,
                      relative_start_s=relative_start, relative_end_s=relative_end,
                      absolute_start_s=t0 + relative_start, absolute_end_s=float(data.time),
                      applied_source_interval=source_offset+k-1,
                      command_origin='previous_completed_neural_interval',
                      source_completed_by_absolute_s=t0 + relative_start,
                      applied_command=applied.copy(), decoded_next_command=pending.copy(),
                      motor_output_connected=bool(motor_output_connected),
                      decoder=trace, total_network_spikes=int(spikes.sum()),
                      motor_spikes=int(spikes[:, motor_indices].sum()),
                      sensory_current_min_nA=float(current.min()),
                      sensory_current_max_nA=float(current.max()),
                      external_force_max=0., auxiliary_force_max=float(np.max(np.abs(data.qfrc_applied))),
                      assistance_kept_during_neural=bool(assisted_neural_closedloop),
                      channels=channels)
        if descending_command is not None:
            record['descending_command_steps'] = descending_trace
            record['descending_target_spikes'] = int(spikes[:, descending_trace[0]['local_indices']].sum())
        if retain_currents:
            record['sensory_current_nA'] = current.copy()
        records.append(record)
        commands.append(applied)
        spike_counts.append(spikes.sum(axis=0))
        if interval_observer is not None:
            prior = state_snapshot(data)
            interval_observer(record)
            now = state_snapshot(data)
            if any(not np.array_equal(prior[name], now[name]) for name in prior):
                raise AssertionError('interval_observer mutated live body')
    assert counts['body_steps'] == total_steps
    return dict(runtime_boundary=runtime_boundary,
                 runtime_state=dict(runtime=(net, encoder, decoder), pending=pending.copy(), body_time=float(data.time)),
                 neural_origin_s=neural_origin_s,
                 handover_before=before, handover_after_withdrawal=after,
                state_continuity_exact=True, handover_absolute_time_s=t0,
                sensory_reference_qpos=reference, records=records,
                commands=np.asarray(commands), spike_counts=np.asarray(spike_counts),
                counts=counts, final_absolute_time_s=float(data.time),
                neural_initialization='fresh resting network' if runtime_state is None else 'same warmed network and decoder retained',
                support_convention=('ALGORITHMIC ASSIST KEPT: hybrid support, NOT autonomous weight bearing'
                                    if assisted_neural_closedloop else 'assist withdrawn: neural-only'),
                input_convention='Observe current physical body, hold sensory current per command window',
                force_convention='All qfrc_applied/xfrc_applied zero before every neural body step; no qpos writes')
