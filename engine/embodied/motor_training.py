"""Bounded offline MN interface fitting (S4 engineering, not biological calibration).

Only recorded binary neural ticks enter runtime features. Targets are offline
labels, never body inputs to the frozen runtime. No bias or new connections.
"""
from dataclasses import dataclass
import math

import numpy as np

from .motor_translation import FiniteMotorTranslator


@dataclass
class TrainingResult:
    translator: FiniteMotorTranslator
    losses: list
    receipt: dict
    features: np.ndarray


def recorded_features(manifest, spikes, connected=None, command_ticks=1):
    """Consume completed blocks; labels apply AFTER each block, never within it.

    command_ticks=10 matches closure's 5ms next-interval command clock.
    Legacy command_ticks=1 is retained for explicit engineering interface tests.
    connected is per command block, matching runtime disconnect semantics.
    """
    if (isinstance(command_ticks, (bool, np.bool_))
            or not isinstance(command_ticks, (int, np.integer))
            or command_ticks not in (1, 10)):
        raise ValueError('command_ticks must be integer 1 (legacy test) or 10 (5ms closure)')
    runtime = FiniteMotorTranslator(manifest)
    sp = np.asarray(spikes)
    if (sp.ndim != 2 or sp.shape[1] != runtime.n or sp.shape[0] < 1
            or not np.all((sp == 0) | (sp == 1))):
        raise ValueError('spikes require nonempty binary finite [T,n]')
    if len(sp) % command_ticks:
        raise ValueError('spikes must contain complete command blocks')
    blocks = len(sp)//command_ticks
    cuts = np.ones(blocks, dtype=bool) if connected is None else np.asarray(connected)
    if cuts.shape != (blocks,) or not np.all((cuts == 0) | (cuts == 1)):
        raise ValueError('connected requires binary finite [command blocks]')
    features = []
    for block in range(blocks):
        start, end = block*command_ticks, (block+1)*command_ticks
        _, trace = runtime.consume(sp[start:end], spike_dt_s=runtime.DT,
                                   start_s=start*runtime.DT, end_s=end*runtime.DT,
                                   connected=bool(cuts[block]))
        features.append(trace['pool_features'])
    return runtime, np.asarray(features).reshape(blocks, len(runtime.pools), 5)


def loss_and_gradient(weights, features, targets, actuator_pool):
    """Mean over T x mapped actuators, including tied-pool multiplicity.

    Clip derivative is one strictly inside (0,1), zero outside and at either
    kink (the chosen valid subgradient). Saturated outputs have zero gradient.
    Inputs are validated by fit; this helper is exposed for finite differences.
    """
    mapped = actuator_pool >= 0
    pool = actuator_pool[mapped]
    if not len(pool):
        raise ValueError('at least one mapped actuator is required')
    raw = np.einsum('tpk,pk->tp', features, weights)
    predictions = np.clip(raw, 0., 1.)
    errors = predictions[:, pool] - targets[:, mapped]
    loss = float(np.mean(errors**2))
    gradient = np.zeros_like(weights)
    for p in range(len(weights)):
        # Summation, not averaging within each shared pool, preserves the
        # mapped-actuator objective when different pools have different fanout.
        residual = errors[:, pool == p].sum(axis=1)
        derivative = ((raw[:, p] > 0) & (raw[:, p] < 1)).astype(float)
        gradient[p] = 2.*np.einsum('t,tk->k', residual*derivative,
                                   features[:, p, :])/errors.size
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite objective or gradient')
    return loss, gradient


def fit(manifest, spikes, targets, connected=None, steps=100, lr=0.01, command_ticks=1):
    """Fit weights in [0,4], returning a reset frozen runtime plus diagnostics.

    steps must be an integer in [1,100]. Labels must be finite [B,nu] in
    [0,1], exactly zero on unmapped rows and identical for shared pools.
    B=T/command_ticks; each label is the command for the interval AFTER its
    completed source block. No future block is used to form that feature.
    Losses contains the initial objective followed by each projected update.
    This API cannot establish label provenance; CLI records INTERFACE_TEST.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 100:
        raise ValueError('steps must be an integer in [1,100]')
    if isinstance(lr, (bool, np.bool_)) or not np.isscalar(lr):
        raise ValueError('learning rate must be positive finite')
    try:
        lr = float(lr)
    except (TypeError, ValueError) as exc:
        raise ValueError('learning rate must be positive finite') from exc
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('learning rate must be positive finite')
    runtime, features = recorded_features(manifest, spikes, connected, command_ticks)
    labels = np.asarray(targets)
    if labels.dtype.kind not in 'biuf':
        raise ValueError('targets must be real numeric labels')
    if not np.isfinite(labels).all():
        raise ValueError('targets must be finite before float conversion')
    with np.errstate(over='ignore', invalid='ignore'):
        labels = labels.astype(float)
    if (labels.shape != (len(features), runtime.nu) or not np.isfinite(labels).all()
            or np.any((labels < 0) | (labels > 1))):
        raise ValueError('targets require finite [T,nu] within [0,1]')
    if np.any(labels[:, runtime.actuator_pool < 0] != 0):
        raise ValueError('unmapped targets must be exactly zero')
    for p in range(len(runtime.pools)):
        group = labels[:, runtime.actuator_pool == p]
        if not np.all(group == group[:, :1]):
            raise ValueError('shared-pool targets must be identical')
    weights = runtime.weights
    initial = weights.copy()
    loss, gradient = loss_and_gradient(weights, features, labels, runtime.actuator_pool)
    losses = [loss]
    max_gradient = float(np.max(np.abs(gradient)))
    for _ in range(steps):
        with np.errstate(over='ignore', invalid='ignore'):
            candidate = weights - lr*gradient
        if not np.isfinite(candidate).all():
            raise ValueError('nonfinite parameter update; reduce learning rate')
        weights = np.clip(candidate, 0., runtime.MAX_WEIGHT)
        loss, gradient = loss_and_gradient(weights, features, labels, runtime.actuator_pool)
        losses.append(loss)
        max_gradient = max(max_gradient, float(np.max(np.abs(gradient))))
    frozen = FiniteMotorTranslator(manifest, weights)
    receipt = dict(version=1, scope='S4_OFFLINE_INTERFACE_ENGINEERING',
                   behavior_claim='NONE', biological_calibration='NOT_ESTABLISHED',
                   optimizer='projected_gradient_descent', objective='mean_squared_mapped_actuator_error',
                   clip_subgradient='1 inside (0,1); 0 outside and at boundaries',
                   steps=int(steps), learning_rate=lr, ticks=len(spikes),
                   command_blocks=len(features), command_ticks=int(command_ticks),
                   command_dt_s=command_ticks*runtime.DT,
                   target_alignment='next_interval_after_completed_source_block',
                   target_valid_from_s=((np.arange(len(features))+1)*command_ticks*runtime.DT).tolist(),
                   target_valid_until_s=((np.arange(len(features))+2)*command_ticks*runtime.DT).tolist(),
                   dt_s=runtime.DT, mapping_digest=runtime.mapping_digest,
                   unique_pools=len(runtime.pools), mapped_actuators=int(np.sum(runtime.actuator_pool >= 0)),
                   unmapped_actuators=int(np.sum(runtime.actuator_pool < 0)),
                   losses=losses, initial_weights=initial.tolist(),
                   final_weights=weights.tolist(), max_abs_gradient=max_gradient,
                   changed_parameters=int(np.count_nonzero(weights != initial)),
                   runtime_inputs='binary MN spikes only; no bias or body input')
    return TrainingResult(frozen, losses, receipt, features)
