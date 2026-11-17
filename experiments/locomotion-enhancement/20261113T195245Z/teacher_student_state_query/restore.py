"""ONLYCODE offline saved-state restoration; stdlib-only at import.

Numerical work occurs ONLY when an authorized parent calls restore/snapshot.
No Core is created and no integration/reset/tick operation is performed here.

Authoritative frozen implementation:
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py:56-100,139-164
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/repair_core.py:109-135
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/sources/incremental_actuator.py:156-193

Trace row i is the END of millisecond i+1. At pre-action integer ms, use
row ms-1 and target history [ms-40:ms]. This is offline reconstruction, NOT
new physical behavior or an exact replay of unsaved solver/contact state.
"""
from contextlib import contextmanager
import hashlib
import math
import sys

DT = .001
MAX_RATE = 2.
K = 40
LAGS = (0, 8, 24, 32, 36, 38, 39)
SNAPSHOT_KEYS = ('qpos', 'qvel', 'ctrl', 'xfrc_applied', 'qfrc_applied', 'target', 'buffer')


class RestoreError(ValueError):
    pass


def _array(value, shape, name):
    import numpy as np
    a = np.asarray(value, dtype=np.float64)
    if a.shape != tuple(shape) or not np.isfinite(a).all():
        raise RestoreError('Finite shape ' + str(tuple(shape)) + ' required for ' + name)
    return np.array(a, copy=True, order='C')


def _require(mapping, name):
    if name not in mapping:
        raise RestoreError('Required saved field is missing: ' + name)
    return mapping[name]


def _digest(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def snapshot(c):
    """Seven copied mechanical arrays in the EXACT archived handover schema.

    No measurement, solver call, model creation, or state update is performed.
    Returned arrays are copies, so subsequent offline restoration cannot mutate
    this snapshot. NumPy import is lazy and belongs to the parent numerical run.
    """
    return {name: getattr(c.d, name).copy() for name in SNAPSHOT_KEYS[:5]} | {
        'target': c.actuator.target_rad.copy(), 'buffer': c.buf.copy()}


def handover_check(c, handover):
    """Compare only explicit known source hashes; do NOT invent missing fields.

    Component sources use physical_before_sha256/physical_after_sha256;
    teacher sources use before. Historical walking uses before_sha256, which
    is accepted only when it actually contains one of these mechanical keys.
    This reads state only. It does not raise solely because a comparison fails.
    """
    current = {name: _digest(value) for name, value in snapshot(c).items()}
    comparisons = {}
    for container in ('physical_before_sha256', 'physical_after_sha256',
                      'before', 'before_sha256', 'after_sha256'):
        values = handover.get(container) if isinstance(handover, dict) else None
        if not isinstance(values, dict):
            continue
        known = {name: values[name] for name in SNAPSHOT_KEYS if name in values}
        if not known:
            continue
        for name, digest in known.items():
            if not (isinstance(digest, str) and len(digest) == 64 and
                    all(ch in '0123456789abcdef' for ch in digest)):
                raise RestoreError('Malformed saved handover SHA: ' + container + '.' + name)
        comparisons[container] = {
            'compared_keys': list(known),
            'missing_mechanical_keys': [name for name in SNAPSHOT_KEYS if name not in known],
            'equal_per_key': {name: current[name] == digest for name, digest in known.items()},
            'all_compared_exact': all(current[name] == digest for name, digest in known.items())}
    if not comparisons:
        return dict(status='UNKNOWN_NO_EXPLICIT_MECHANICAL_HASH_SCHEMA', current_hashes=current,
                    all_seven_verified=False, all_compared_exact=None, all_exact=False)
    all_seven = any(not value['missing_mechanical_keys'] for value in comparisons.values())
    exact = all(value['all_compared_exact'] for value in comparisons.values())
    result = dict(status='COMPARED', comparisons=comparisons, current_hashes=current,
                  all_seven_verified=all_seven and exact, all_compared_exact=exact,
                  all_exact=all_seven and exact)
    if isinstance(handover, dict) and isinstance(handover.get('time'), (int, float)):
        result['saved_time_s'] = float(handover['time'])
        result['current_time_s'] = float(c.d.time)
        result['time_error_s'] = float(c.d.time) - float(handover['time'])
    return result


def _force_array(c, trace, initial, ms, name, max_name):
    """Exact full field, or a zero array PROVED by that saved max==0.

    Nonzero maxima cannot identify the underlying array and are rejected;
    neither missing fields nor nonzero arrays are silently replaced by zeros.
    """
    import numpy as np
    shape = getattr(c.d, name).shape
    if ms == 0:
        return _array(_require(initial, name), shape, 'initial.' + name), 'saved_initial_full_array'
    index = ms - 1
    if name in trace:
        values = np.asarray(trace[name])
        if len(values) < ms:
            raise RestoreError('Insufficient rows for trace.' + name)
        value = _array(values[index], shape, 'trace.' + name)
        if max_name in trace:
            recorded = float(np.asarray(trace[max_name])[index])
            if not math.isfinite(recorded) or recorded != float(np.max(np.abs(value))):
                raise RestoreError('Saved full force and maximum disagree: ' + name)
        return value, 'saved_trace_full_array'
    maxima = np.asarray(_require(trace, max_name))
    if len(maxima) < ms:
        raise RestoreError('Insufficient saved force maxima: ' + max_name)
    maximum = float(maxima[index])
    if not math.isfinite(maximum) or maximum != 0.:
        raise RestoreError('Cannot recover full ' + name + ' from missing/nonzero max; no zero fabrication')
    return np.zeros(shape, dtype=np.float64), 'proved_all_elements_zero_by_saved_' + max_name


def _measurement_difference(value, reference):
    import numpy as np
    actual = np.asarray(value, dtype=float)
    saved = np.asarray(reference, dtype=float)
    if actual.shape != saved.shape or not np.isfinite(actual).all() or not np.isfinite(saved).all():
        return dict(status='SHAPE_OR_NONFINITE_MISMATCH', recomputed_shape=list(actual.shape),
                    saved_shape=list(saved.shape), exact=False, max_abs_error=None)
    error = np.abs(actual - saved)
    return dict(status='COMPARED', exact=bool(np.array_equal(actual, saved)),
                max_abs_error=float(error.max()) if error.size else 0.,
                RMS_error=float(np.sqrt(np.mean(error * error))) if error.size else 0.)


_ACTIVE_TRAPS = {}


def trap_neural_calls_if_loaded(state):
    """Patch already-loaded Torch only; never import it or load any checkpoint.

    Call again after Core construction if external imports could have loaded
    Torch. A newly discovered late import leaves the previous interval UNKNOWN:
    canonical forward/load counts become None, rather than fabricated zeros.
    Future attempts are still blocked and observed attempt counters increment.
    All new patches belong to the active install_step_traps finally cleanup.
    """
    record = _ACTIVE_TRAPS.get(id(state))
    if record is None or record['state'] is not state:
        raise RestoreError('Neural traps require an active install_step_traps context')
    torch = sys.modules.get('torch')
    if torch is None:
        state['neural_traps_status'] = 'TORCH_NOT_LOADED_AT_CHECK'
        return state
    if not record['entry_torch_loaded'] and not record['patched_torch']:
        state['neural_tracking_complete'] = False
        state['NN_forward_calls'] = None
        state['torch_load_calls'] = None
    patches = record['patches']

    def patch(obj, name, group):
        if any(old_obj is obj and old_name == name for old_obj, old_name, _, _ in patches):
            return
        original = getattr(obj, name, None)
        if not callable(original):
            raise RestoreError('Loaded Torch neural trap API missing: ' + name)
        existed = name in getattr(obj, '__dict__', {})
        label = 'torch.load' if group == 'torch_load_calls' else 'torch.nn.Module._call_impl'

        def blocked(*args, **kwargs):
            state['calls'][label] = state['calls'].get(label, 0) + 1
            observed = 'observed_' + group
            state[observed] += 1
            if state[group] is not None:
                state[group] += 1
            raise RuntimeError('Offline teacher query forbids ' + label)

        patches.append((obj, name, original, existed))
        setattr(obj, name, blocked)

    patch(torch, 'load', 'torch_load_calls')
    nn = getattr(torch, 'nn', None)
    module_class = getattr(nn, 'Module', None) if nn is not None else None
    if module_class is None:
        raise RestoreError('Torch loaded but Module class unavailable for honest forward trapping')
    patch(module_class, '_call_impl', 'NN_forward_calls')
    record['patched_torch'] = True
    state['neural_traps_status'] = 'PATCHED_LOADED_TORCH' if state['neural_tracking_complete'] else 'PATCHED_LATE_TORCH_PRIOR_INTERVAL_UNKNOWN'
    return state


@contextmanager
def block_core_ticks(core):
    """Optional per-instance tick/reset guard, no Core creation or module import."""
    patches = []
    counts = dict(calls={}, core_tick_or_reset_attempts=0, patches_restored=False)
    try:
        for name in ('tick', 'tick_rate', 'tick_policy', 'reset', 'reset_from'):
            if not hasattr(core, name):
                continue
            existed = name in getattr(core, '__dict__', {})
            original = core.__dict__[name] if existed else getattr(core, name)

            def trap(*args, _name=name, **kwargs):
                counts['core_tick_or_reset_attempts'] += 1
                counts['calls'][_name] = counts['calls'].get(_name, 0) + 1
                raise RuntimeError('Offline teacher query forbids Core.' + _name)

            patches.append((name, original, existed))
            setattr(core, name, trap)
        yield counts
    finally:
        for name, original, existed in reversed(patches):
            if existed:
                setattr(core, name, original)
            else:
                delattr(core, name)
        counts['patches_restored'] = True


@contextmanager
def install_step_traps(mj, core=None):
    """Temporarily forbid integration; use only in this fresh offline process.

    with install_step_traps(mj, core=c) as audit: ...
    patches mj_step/mj_step1/mj_step2 and optional c.tick/tick_rate/tick_policy/
    reset/reset_from. Every attempted call increments a counter and raises.
    mj_forward/FK are allowed and do not advance time. All patched attributes
    are restored in finally, including when an offline query fails. This does
    not intercept a step function copied elsewhere BEFORE installing traps;
    the frozen Core calls mj.mj_step through this module, and Core paths are
    independently trapped when core is supplied.
    """
    counters = dict(integration_calls=0, integration_attempts=0,
                    NN_forward_calls=0, torch_load_calls=0,
                    observed_NN_forward_calls=0, observed_torch_load_calls=0,
                    neural_tracking_complete=True, core_tick_or_reset_attempts=0,
                    calls={}, patches_restored=False, scope='fresh_process_offline_only')
    patches = []
    _ACTIVE_TRAPS[id(counters)] = dict(state=counters, patches=patches,
                                     entry_torch_loaded='torch' in sys.modules,
                                     patched_torch=False)

    def blocked(label, group):
        def trap(*args, **kwargs):
            counters[group] += 1
            if group == 'integration_attempts':
                counters['integration_calls'] += 1
            counters['calls'][label] = counters['calls'].get(label, 0) + 1
            raise RuntimeError('Offline saved-state query forbids ' + label)
        return trap

    try:
        for name in ('mj_step', 'mj_step1', 'mj_step2'):
            if not hasattr(mj, name) or not callable(getattr(mj, name)):
                raise RestoreError('Required integration trap API missing: ' + name)
            original = getattr(mj, name)
            patches.append((mj, name, original, True))
            setattr(mj, name, blocked('mujoco.' + name, 'integration_attempts'))
        if core is not None:
            for name in ('tick', 'tick_rate', 'tick_policy', 'reset', 'reset_from'):
                if not hasattr(core, name):
                    continue
                existed = name in getattr(core, '__dict__', {})
                original = core.__dict__[name] if existed else getattr(core, name)
                patches.append((core, name, original, existed))
                setattr(core, name, blocked('Core.' + name, 'core_tick_or_reset_attempts'))
        trap_neural_calls_if_loaded(counters)
        yield counters
        trap_neural_calls_if_loaded(counters)
    finally:
        for obj, name, original, existed in reversed(patches):
            if existed:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
        _ACTIVE_TRAPS.pop(id(counters), None)
        counters['patches_restored'] = True


def restore(c, trace, initial, physical_ms, *, mujoco_module=None):
    """Restore a saved pre-action state, call ONLY mj_forward, return auditdict.

    c must already be an admitted ORIGINAL 49q/48v/42 finite-PD complete Core,
    adhesion disabled. This helper creates no Core, NN, teacher, optimizer or
    MjData, does not call reset, and never integrates. The passed c is explicitly
    mutated as an OFFLINE reconstruction, not an actual new trajectory.

    physical_ms >= 5800 is the query cohort. General startup support is strict:
    ms0 uses explicit INITIAL_STATE arrays; ms1 requires initial.padz because
    the observed original INITIAL_STATE does not save it. Missing startup padz
    is an error, never an invented zero. At ms>=2 pad derivative is exactly the
    last two stored endpoint padz values divided by the native1ms policy tick.
    """
    import numpy as np
    if type(physical_ms) is not int or physical_ms < 0:
        raise RestoreError('Nonnegative integer physical_ms required (not bool)')
    ms = physical_ms
    if (c.plant.nq, c.plant.nv, c.plant.nu) != (49, 48, 42):
        raise RestoreError('Original full49/48/42 body only')
    if c.adhesion_on:
        raise RestoreError('Saved component cohort requires adhesionFalse')
    if int(getattr(c.plant, 'na', 0)) != 0:
        raise RestoreError('Unrecorded dynamic actuation state is not supported')
    if int(c.K) != K or tuple(int(i) for i in c.lags) != LAGS:
        raise RestoreError('Frozen target history K40/lags contract changed')
    if abs(float(c.actuator.command_dt_s) - DT) > 1e-15 or abs(float(c.plant.opt.timestep) - 12.5e-6) > 1e-15:
        raise RestoreError('Frozen1ms actuator/native12.5us contract changed')
    if not isinstance(initial, dict) or 'time_s' not in initial or float(initial['time_s']) != 0.:
        raise RestoreError('Explicit time_s0 INITIAL_STATE required')
    initial_qpos = _array(_require(initial, 'qpos'), (49,), 'initial.qpos')
    initial_target = _array(_require(initial, 'target'), (42,), 'initial.target')
    initial_buffer = _array(_require(initial, 'buffer'), (K, 42), 'initial.buffer')
    initial_ctrl = _array(_require(initial, 'ctrl'), (42,), 'initial.ctrl')
    initial_qvel = _array(_require(initial, 'qvel'), (48,), 'initial.qvel')
    # Check explicit initial forces even at late query times; they define init identity.
    _array(_require(initial, 'xfrc_applied'), c.d.xfrc_applied.shape, 'initial.xfrc_applied')
    _array(_require(initial, 'qfrc_applied'), c.d.qfrc_applied.shape, 'initial.qfrc_applied')
    times = np.asarray(_require(trace, 'time'), dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or ms > len(times):
        raise RestoreError('Saved time1D prefix must cover physical_ms')
    if not np.allclose(times[:ms], np.arange(1, ms + 1) * DT, atol=1e-8, rtol=0):
        raise RestoreError('Trace row i must be END of millisecond i+1')
    targets = np.asarray(_require(trace, 'target'), dtype=float)
    if targets.shape != (len(times), 42) or not np.isfinite(targets[:ms]).all():
        raise RestoreError('Saved target[time,42] required')
    if ms:
        i = ms - 1
        qpos = _array(_require(trace, 'qpos')[i], (49,), 'trace.qpos[ms-1]')
        qvel = _array(_require(trace, 'qvel')[i], (48,), 'trace.qvel[ms-1]')
        ctrl = _array(_require(trace, 'actualctrl')[i], (42,), 'trace.actualctrl[ms-1]')
        target = _array(targets[i], (42,), 'trace.target[ms-1]')
        # Actual trace value, not nominal ms*.001, retains archived roundoff/time.
        saved_time = float(times[i])
        if ms >= K:
            buffer = np.array(targets[ms-K:ms], copy=True, order='C')
            history_source = 'trace.target[physical_ms-40:physical_ms]'
        else:
            buffer = np.concatenate((initial_buffer[ms:], targets[:ms]), axis=0)
            history_source = 'initial.buffer[physical_ms:] + trace.target[:physical_ms]'
        padz = _array(_require(trace, 'padz')[i], (6,), 'trace.padz[ms-1]')
        previous_padz = _array(trace['padz'][i-1], (6,), 'trace.padz[ms-2]') if ms >= 2 else _array(_require(initial, 'padz'), (6,), 'initial.padz for ms1 derivative')
        pad_rate = (padz - previous_padz) / DT
        previous_target = initial_target if ms == 1 else _array(targets[i-1], (42,), 'trace.target[ms-2]')
        requested_action = _array(_require(trace, 'actual_requested_action')[i], (42,), 'trace.actual_requested_action[ms-1]')
        if np.any(np.abs(requested_action) > 1. + 1e-12):
            raise RestoreError('Saved bounded normalized requested action required')
        requested_rate = requested_action * MAX_RATE
        previous_action = _array(_require(trace, 'action')[i], (42,), 'trace.action[ms-1]')
        if np.any(np.abs(previous_action) > 1. + 1e-12):
            raise RestoreError('Saved actual previous action must remain bounded42')
        previous_action_source = 'actual saved last1ms action, applicable to nextpreaction observation'
    else:
        i = None
        qpos, qvel, ctrl, target = initial_qpos, initial_qvel, initial_ctrl, initial_target
        saved_time = 0.
        buffer = initial_buffer.copy()
        history_source = 'saved INITIAL_STATE.buffer, not synthesized restangles'
        padz = None
        pad_rate = np.zeros(6, dtype=float)
        previous_target = initial_target.copy()
        requested_rate = np.zeros(42, dtype=float)
        previous_action = None
        previous_action_source = 'UNKNOWN_NOT_SAVED_AT_T0; caller must use its explicit producer convention'
    xfrc, xproof = _force_array(c, trace, initial, ms, 'xfrc_applied', 'external_body_force_max')
    qfrc, qproof = _force_array(c, trace, initial, ms, 'qfrc_applied', 'external_generalized_force_max')
    if np.any(xfrc) or np.any(qfrc):
        raise RestoreError('This scoped source cohort requires provedzero extra forces')
    if buffer.shape != (K, 42):
        raise RestoreError('Restored target history must be40x42')
    limits = _array(c.actuator.max_rate_rad_s, (42,), 'actual native actuator rate limits')
    bounds = _array(c.actuator.target_bounds_rad, (42, 2), 'actual native actuator target bounds')
    if np.any(target < bounds[:, 0]-1e-12) or np.any(target > bounds[:, 1]+1e-12):
        raise RestoreError('Saved target outside original actuator bounds')
    limited_rate = np.clip(requested_rate, -limits, limits)
    proposed = previous_target + limited_rate * DT
    # This relation audits source counters/rates, but restoration always uses SAVED target.
    source_target_error = float(np.max(np.abs(np.clip(proposed, bounds[:, 0], bounds[:, 1]) - target))) if ms else 0.
    if ms and source_target_error > 1e-10:
        raise RestoreError('Saved target increment inconsistent with actual requested action and native caps')
    if mujoco_module is None:
        import mujoco as mujoco_module
    mj = mujoco_module
    if not callable(getattr(mj, 'mj_forward', None)):
        raise RestoreError('Explicit usable mujoco_module.mj_forward required')
    solver_before = {}
    for name in ('qacc_warmstart', 'qacc'):
        value = getattr(c.d, name, None)
        if value is not None:
            solver_before[name] = _digest(np.array(value, copy=True))
    # Only now, after all required source validation, mutate the OFFLINE object.
    with install_step_traps(mj, core=c) as traps:
        c.d.qpos[:] = qpos
        c.d.qvel[:] = qvel
        c.d.ctrl[:] = ctrl
        c.d.time = saved_time
        c.d.xfrc_applied[:] = xfrc
        c.d.qfrc_applied[:] = qfrc
        c.actuator.target_rad = target.copy()
        c.buf = np.array(buffer, copy=True, order='C')
        c.t = ms
        c.requested_rate = requested_rate.copy()
        c.actuator.command_ticks = ms
        c.actuator.initial_target_rad = initial_target.copy()
        c.actuator.reset_projection_rad = initial_target - initial_qpos[c.a_qpos]
        c.actuator.last_input_rate_rad_s = requested_rate.copy()
        c.actuator.last_limited_rate_rad_s = limited_rate.copy()
        c.actuator.last_effective_rate_rad_s = (target - previous_target) / DT if ms else np.zeros(42)
        c.actuator.rate_clip_mask = requested_rate != limited_rate
        c.actuator.angle_clip_mask = proposed != np.clip(proposed, bounds[:, 0], bounds[:, 1]) if ms else np.zeros(42, dtype=bool)
        expected = dict(qpos=qpos, qvel=qvel, ctrl=ctrl,
                        xfrc_applied=xfrc, qfrc_applied=qfrc,
                        target=target, buffer=buffer)
        mj.mj_forward(c.plant, c.d)
        if float(c.d.time) != saved_time:
            raise RestoreError('mj_forward advanced time during offline reconstruction')
        restored = snapshot(c)
        exact_state = bool(c.t == ms and float(c.d.time) == saved_time and
                           all(np.array_equal(value, restored[name]) for name, value in expected.items()))
        if not exact_state:
            raise RestoreError('mj_forward altered restored saved mechanical fields')
        measured = c.measure()
        if ms == 0:
            padz = _array(measured['padz'], (6,), 'initialgeometry padz after forward')
        c._padz_prev = padz.copy()
        c._pad_rate = pad_rate.copy()
        if traps['integration_attempts'] or traps['core_tick_or_reset_attempts']:
            raise RestoreError('An integration/reset/tick attempt occurred')
    comparisons = {}
    if ms:
        for source_key, measured_key in (('padz', 'padz'), ('loads', 'loads'),
                                         ('nonfoot', 'nonfoot'), ('speed', 'speed'),
                                         ('relativefoot', 'foot')):
            if source_key in trace and measured_key in measured:
                comparisons[source_key] = _measurement_difference(measured[measured_key], trace[source_key][i])
        if 'com' in trace:
            comparisons['COM'] = _measurement_difference(c.d.subtree_com[c.th], trace['com'][i])
        if 'actuator_force' in trace:
            comparisons['actuator_force'] = _measurement_difference(c.d.actuator_force, trace['actuator_force'][i])
    hashes = {name: _digest(value) for name, value in snapshot(c).items()}
    return dict(kind='OFFLINE_SAVED_PREACTION_RESTORE_V1', physical_ms=ms,
                exact_state=exact_state,
                saved_trace_index=i, saved_time_s=saved_time, current_time_s=float(c.d.time),
                trace_index_rule='row physical_ms-1 is END of that millisecond',
                buffer_source=history_source, buffer_shape=list(c.buf.shape),
                snapshot_hashes=hashes, target_increment_consistency_max_error=source_target_error,
                requested_rate_source='trace.actual_requested_action*sourceMAX_RATE2' if ms else 'sourceCore_reset_measurements zeros at explicittime0',
                actuator_metadata_source='derived from saved1ms targets/requestedrate, not saved original private diagnostics',
                force_restoration_sources=dict(xfrc_applied=xproof, qfrc_applied=qproof),
                pad_derivative_source='trace.padz[ms-1]-trace.padz[ms-2] over .001' if ms >= 2 else ('initial.padz totrace row0' if ms else 'source initialreset rate0, initialFKgeometry'),
                previous_action=None if previous_action is None else previous_action.tolist(),
                previous_action_source=previous_action_source,
                solver_state_note='qacc/qacc_warmstart/contact solver history not stored or restored; mj_forward is a new algebraic solve, not exact nativecontact replay',
                unsaved_solver_state_hashes_before_forward=solver_before,
                recomputed_measurement_vs_saved=comparisons,
                contacts_or_forces_reused_from_saved=False,
                adhesion_cache_note='adhesionFalse; private historical attachment/refresh metadata not reconstructed and must not be used for query acceptance',
                sensory_state_note='sourceproprioception.features is pure; obs238 consumes currentforward loads and restored_pad_rate, no hidden sensory history. Contactload differences must remain reported.',
                initial_heading_note='not overwritten from currentheading; caller initialCore/sourceidentity must retain originalinitialheading',
                traps=dict(traps), physical_integration_steps=0,
                scope='Offline state restoration/FK and contact re-solve only; no neural/teacher query, no new physical trajectory or exact force claim')
