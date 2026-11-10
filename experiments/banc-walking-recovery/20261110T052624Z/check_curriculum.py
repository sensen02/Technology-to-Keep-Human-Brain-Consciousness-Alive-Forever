"""Synthetic curriculum checks; no physics simulation or training."""
from pathlib import Path
from dataclasses import FrozenInstanceError
import argparse
import copy
import hashlib
import json
import numpy as np
import curriculum as C


def sample(**changes):
    cfg = C.RewardConfig()
    value = dict(com=np.array([0., 0., 1.4]), com_velocity=np.zeros(3),
                 rotation=np.eye(3), loads=np.full(6, cfg.body_weight_native / 6),
                 speed=np.zeros(6), nonfoot_force_native=0., action=np.zeros(42), time_s=.001)
    value.update(changes)
    return value


def run():
    cfg = C.RewardConfig()
    tests = []
    assert json.loads(json.dumps(cfg.to_dict())) == cfg.to_dict()
    try:
        cfg.min_rate = -10.
        raise AssertionError('Dataclass should be frozen')
    except FrozenInstanceError:
        pass
    tests.append('frozen_serializable_config')
    def evaluate(s=None, target=.1, previous=None, heading=(1., 0.)):
        return C.evaluate_reward(sample() if s is None else s,
                                 np.zeros(42) if previous is None else previous,
                                 heading, target, cfg)
    original = sample()
    before = copy.deepcopy(original)
    assert evaluate(original)['rate'] == 0.
    assert evaluate(original) == evaluate(original)
    for key in original:
        assert np.array_equal(original[key], before[key])
    for velocity in ([0., 0., 0.], [.1, .2, 0.], [-.1, .2, 0.]):
        value = evaluate(sample(com_velocity=np.asarray(velocity)), target=0.)
        assert np.isfinite(value['rate']) and value['components']['forward'] <= 0.
    tests.append('pure_idempotent_and_zero_target_finite')
    air = evaluate(sample(loads=np.zeros(6), speed=np.full(6, 1e6)))
    assert air['components']['slip_cost'] == 0.
    low_speed = evaluate(sample(speed=np.full(6, .1), com_velocity=np.array([.01, 0., 0.])))
    high_speed = evaluate(sample(speed=np.full(6, .1), com_velocity=np.array([5., 0., 0.])))
    assert low_speed['components']['slip_cost'] == high_speed['components']['slip_cost']
    assert np.isclose(low_speed['components']['slip_cost'], 1. / 6.)
    tests.extend(['airborne_high_speed_not_penalized', 'slip_reference_independent_of_instantaneous_velocity'])
    sideways = evaluate(sample(com_velocity=np.array([0., 1., 0.])))
    backwards = evaluate(sample(com_velocity=np.array([-.1, 0., 0.])))
    fast = evaluate(sample(com_velocity=np.array([10., 0., 0.])))
    assert sideways['components']['forward'] == 0.
    assert backwards['components']['forward'] == -1.
    assert fast['components']['forward'] == 1.
    yaw90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    turned = evaluate(sample(rotation=yaw90))
    matching = evaluate(sample(rotation=yaw90, com_velocity=np.array([0., .1, 0.])), heading=(0., 1.))
    assert np.isclose(turned['components']['heading_cost'], 1.)
    assert matching['components']['heading_cost'] == 0.
    assert matching['components']['forward'] == 1.
    tests.append('sideways_backwards_cap_and_initial_heading')
    heavy = evaluate(sample(loads=np.full(6, cfg.body_weight_native)))
    assert heavy['components']['support_risk'] == 0.
    assert heavy['rate'] == 0.
    low = evaluate(sample(loads=np.zeros(6)))
    assert low['components']['support_risk'] == 1.
    above = evaluate(sample(com=np.array([0., 0., 1.35])))
    warning = evaluate(sample(com=np.array([0., 0., 1.25])))
    below = evaluate(sample(com=np.array([0., 0., 1.15])))
    assert above['components']['height_risk'] == 0.
    assert 0. < warning['components']['height_risk'] < below['components']['height_risk']
    assert evaluate(sample(nonfoot_force_native=cfg.body_weight_native / 2))['components']['nonfoot_cost'] == .5
    tests.append('no_highload_bonus_height_prewarning_native_force_scaling')
    #20ms difference reads action(t-20ms); initial fictive history is zero
    # command memory only, never a physical trajectory.
    tracker = C.RewardTracker(horizon_s=.1)
    change_costs, rates, rewards = [], [], []
    for i in range(40):
        s = sample(action=np.ones(42), time_s=(i + 1) * .001,
                   com_velocity=np.array([.1 * np.cos(i * .1), 0., 0.]))
        value = tracker.update(s)
        change_costs.append(value['components']['action_change_cost'])
        rates.append(value['rate'])
        rewards.append(value['reward'])
    assert np.allclose(change_costs[:20], 1.)
    assert np.allclose(change_costs[20:], 0.)
    total_1ms = C.integrate_rates(rates, np.full(40, .001))
    policy_groups = np.asarray(rewards).reshape(-1, 5).sum(1)
    total_5ms = float(policy_groups.sum())
    assert np.isclose(total_1ms, total_5ms, rtol=0, atol=1e-15)
    tracker.reset()
    assert np.array_equal(tracker.history[0], np.zeros(42))
    tests.extend(['fixed_20ms_command_change_window', 'fixed_trajectory_1ms_5ms_partition_equivalence'])
    # Tail exactly equals bounded survival return under declared policy discount.
    tails = []
    for remaining in (0., .001, .005, .013, .1, 1.999):
        n = int(np.floor(remaining / cfg.policy_discount_dt_s + 1e-10))
        durations = [cfg.policy_discount_dt_s] * n
        fraction = remaining - n * cfg.policy_discount_dt_s
        if fraction > 1e-12:
            durations.append(fraction)
        gamma = np.exp(-cfg.policy_discount_dt_s / cfg.discount_time_constant_s)
        worst_survival = sum(cfg.min_rate * dt * gamma ** i for i, dt in enumerate(durations))
        tail = C.absorbing_tail(remaining, cfg)
        assert np.isclose(tail, worst_survival, rtol=0, atol=1e-11)
        # Any allowed bounded-rate continuation has at least this return.
        survival = sum((cfg.min_rate + 1.) * dt * gamma ** i for i, dt in enumerate(durations))
        assert tail <= survival + 1e-11
        tails.append(dict(remaining_s=remaining, tail=tail, worst_survival=worst_survival))
    terminal = C.RewardTracker(horizon_s=2.)
    value = terminal.update(sample(com=np.array([0., 0., .5])))
    assert value['terminated'] and value['remaining_s'] == 1.999
    assert value['absorbing_penalty'] == C.absorbing_tail(1.999, cfg)
    assert value['total_reward'] == value['reward'] + value['absorbing_penalty']
    try:
        terminal.update(sample(time_s=.002))
        raise AssertionError('Postterminal update must reject')
    except RuntimeError:
        pass
    horizon = C.RewardTracker(horizon_s=.001)
    end = horizon.update(sample())
    assert end['truncated'] and not end['terminated'] and end['absorbing_penalty'] == 0.
    tests.append('discounted_absorbing_tail_exact_bounded_survival_no_double_emission')
    tilted = np.array([[np.cos(1.3), 0., np.sin(1.3)], [0., 1., 0.],
                       [-np.sin(1.3), 0., np.cos(1.3)]])
    assert evaluate(sample(rotation=tilted))['terminated']
    assert evaluate(sample(nonfoot_force_native=2 * cfg.body_weight_native))['terminated']
    rng = np.random.default_rng(12)
    for _ in range(100):
        result = evaluate(sample(com=np.array([0., 0., rng.uniform(.1, 2.)]),
                                 com_velocity=rng.uniform(-20., 20., 3),
                                 loads=rng.uniform(0., 10 * cfg.body_weight_native, 6),
                                 speed=rng.uniform(0., 10., 6),
                                 action=rng.uniform(-1., 1., 42)))
        assert cfg.min_rate <= result['rate'] <= cfg.max_rate
        assert np.isfinite(result['rate'])
    try:
        evaluate(sample(speed=np.full(6, np.nan)))
        raise AssertionError('NaN must reject')
    except ValueError:
        pass
    tests.append('terminal_flags_bounded_rates_nonfinite_rejected')
    sources = [Path(C.__file__).resolve(), Path(__file__).resolve()]
    return dict(all_passed=True, status='PASS', tests=tests, config=cfg.to_dict(),
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                partition_error=abs(total_1ms-total_5ms), absorbing_tail_checks=tails,
                label='Synthetic engineering reward checks; no simulation/training or acceptance claim.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(run(), indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)
