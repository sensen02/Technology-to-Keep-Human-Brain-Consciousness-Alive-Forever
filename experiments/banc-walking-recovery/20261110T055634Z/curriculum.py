"""Versioned S2/S5 training reward proxies, separate from walking acceptance.

All physical fields use native units (absolute SI NOT_ESTABLISHED). Reward is a
rate integrated at 1ms. No policy, fake trajectory or acceptance gate lives here.
"""
from collections import deque
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    version: str = 'curriculum-v1'
    dt_s: float = .001
    body_weight_native: float = 24.446354755878005
    velocity_scale_native_per_s: float = .10
    slip_reference_native_per_s: float = .10
    load_gate_fraction: float = .10
    height_warning_native: float = 1.30
    height_floor_native: float = 1.20
    height_risk_scale_native: float = .10
    support_target: float = .90
    tilt_safe_rad: float = .15
    tilt_scale_rad: float = .30
    rate_limit_rad_per_s: float = 2.
    action_change_window_s: float = .020
    action_change_acceleration_ref_rad_per_s2: float = 100.
    height_weight: float = 2.
    nonfoot_weight: float = 2.
    support_weight: float = 1.
    slip_weight: float = .5
    tilt_weight: float = .1
    heading_weight: float = .1
    action_weight: float = .02
    action_change_weight: float = .01
    min_rate: float = -20.
    max_rate: float = 2.
    terminal_com_native: float = .80
    terminal_nonfoot_fraction: float = 1.
    terminal_tilt_rad: float = 1.2
    discount_time_constant_s: float = 1.
    policy_discount_dt_s: float = .005

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) for k, v in values.items() if k != 'version'):
            raise ValueError('Reward constants must be finite')
        positives = ('dt_s', 'body_weight_native', 'velocity_scale_native_per_s',
                     'slip_reference_native_per_s', 'height_risk_scale_native',
                     'support_target', 'tilt_scale_rad', 'rate_limit_rad_per_s',
                     'action_change_window_s', 'action_change_acceleration_ref_rad_per_s2',
                     'discount_time_constant_s', 'policy_discount_dt_s')
        if any(values[k] <= 0 for k in positives):
            raise ValueError('Reward scales must be positive')
        if self.min_rate >= 0 or self.max_rate <= 0:
            raise ValueError('Rate bounds must straddle zero')
        steps = self.action_change_window_s / self.dt_s
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-10):
            raise ValueError('Action change window must contain integer samples')
        if self.height_warning_native <= self.height_floor_native:
            raise ValueError('Height prewarning must be above engineering floor')

    def to_dict(self):
        return asdict(self)


def _array(sample, key, shape):
    value = np.asarray(sample[key], float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{key} must be finite shape {shape}')
    return value


def _heading(value):
    value = np.asarray(value, float)
    if value.shape != (2,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-12:
        raise ValueError('initial_heading must have finite nonzero shape (2,)')
    return value / np.linalg.norm(value)


def evaluate_reward(sample, action_20ms_ago, initial_heading, target_speed=.1, config=None):
    """Pure instantaneous reward-rate evaluation (does not integrate time).

    Required sample keys: com(3), com_velocity(3), rotation(3,3), loads(6),
    speed(6), nonfoot_force_native scalar, action(42) normalized [-1,1].
    action_20ms_ago is normalized commanded action, initialized to zero at reset.
    target_speed is nonnegative, fixed per episode; zero disables progress reward.
    Slip is mean(load/W * (speed/.10)^2) over six legs, gated at load>=.10W.
    Its denominator never depends on achieved body speed. High support earns zero
    support benefit. Heading is initial-heading horizontal projection, not time.
    """
    cfg = config or RewardConfig()
    com = _array(sample, 'com', (3,))
    velocity = _array(sample, 'com_velocity', (3,))
    rotation = _array(sample, 'rotation', (3, 3))
    loads = _array(sample, 'loads', (6,))
    speeds = _array(sample, 'speed', (6,))
    action = _array(sample, 'action', (42,))
    previous = np.asarray(action_20ms_ago, float)
    if previous.shape != (42,) or not np.isfinite(previous).all():
        raise ValueError('action_20ms_ago must be finite shape (42,)')
    nonfoot = float(sample['nonfoot_force_native'])
    if not np.isfinite(nonfoot) or nonfoot < 0 or np.any(loads < 0) or np.any(speeds < 0):
        raise ValueError('Loads, nonfoot and speed must be finite nonnegative')
    if np.any(np.abs(action) > 1. + 1e-12) or np.any(np.abs(previous) > 1. + 1e-12):
        raise ValueError('Actions must be normalized into [-1,1]')
    if not np.isfinite(target_speed) or target_speed < 0:
        raise ValueError('target_speed must be finite nonnegative')
    heading = _heading(initial_heading)
    forward_velocity = float(velocity[:2] @ heading)
    if target_speed == 0:
        # Smooth bounded hold cost, no division by task command.
        forward = -float(np.clip(np.sum(velocity[:2] ** 2) /
                                 cfg.velocity_scale_native_per_s ** 2, 0., 4.))
    else:
        forward = float(np.clip(forward_velocity / cfg.velocity_scale_native_per_s,
                                 -4., target_speed / cfg.velocity_scale_native_per_s))
    warning = max(0., (cfg.height_warning_native - com[2]) / cfg.height_risk_scale_native)
    below_floor = max(0., (cfg.height_floor_native - com[2]) / cfg.height_risk_scale_native)
    height_risk = min(4., warning ** 2 + below_floor ** 2)
    nonfoot_cost = min(4., nonfoot / cfg.body_weight_native)
    support = float(loads.sum() / cfg.body_weight_native)
    support_risk = min(4., (max(0., cfg.support_target - support) / cfg.support_target) ** 2)
    load_fraction = loads / cfg.body_weight_native
    slip_per_leg = np.where(load_fraction >= cfg.load_gate_fraction,
                            load_fraction * (speeds / cfg.slip_reference_native_per_s) ** 2, 0.)
    slip_cost = float(min(4., slip_per_leg.mean()))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1., 1.)))
    tilt_cost = min(4., (max(0., np.hypot(roll, pitch) - cfg.tilt_safe_rad) /
                         cfg.tilt_scale_rad) ** 2)
    current_heading = rotation[:2, 0]
    heading_norm = np.linalg.norm(current_heading)
    heading_cost = 1. if heading_norm <= 1e-12 else float(
        1. - np.clip(current_heading @ heading / heading_norm, -1., 1.))
    action_cost = float(np.mean(action ** 2))
    rate_change = (action - previous) * cfg.rate_limit_rad_per_s
    change_scale = cfg.action_change_window_s * cfg.action_change_acceleration_ref_rad_per_s2
    action_change_cost = float(min(4., np.mean((rate_change / change_scale) ** 2)))
    components = dict(forward=forward, height_risk=height_risk, nonfoot_cost=nonfoot_cost,
                      support_risk=support_risk, slip_cost=slip_cost, tilt_cost=tilt_cost,
                      heading_cost=heading_cost, action_cost=action_cost,
                      action_change_cost=action_change_cost)
    raw_rate = (forward - cfg.height_weight * height_risk - cfg.nonfoot_weight * nonfoot_cost
                - cfg.support_weight * support_risk - cfg.slip_weight * slip_cost
                - cfg.tilt_weight * tilt_cost - cfg.heading_weight * heading_cost
                - cfg.action_weight * action_cost - cfg.action_change_weight * action_change_cost)
    reasons = []
    if com[2] < cfg.terminal_com_native:
        reasons.append('com_below_training_terminal')
    if nonfoot / cfg.body_weight_native > cfg.terminal_nonfoot_fraction:
        reasons.append('nonfoot_above_training_terminal')
    if np.hypot(roll, pitch) > cfg.terminal_tilt_rad:
        reasons.append('tilt_above_training_terminal')
    return dict(components=components, raw_rate=float(raw_rate),
                rate=float(np.clip(raw_rate, cfg.min_rate, cfg.max_rate)),
                terminated=bool(reasons), terminal_reasons=reasons,
                forward_velocity_native_per_s=forward_velocity,
                support_body_weights=support, slip_per_leg=slip_per_leg.tolist(),
                training_proxy_only=True, reward_version=cfg.version)


def integrate_rates(rates, durations_s):
    """Integrate already evaluated rates; regrouping fixed samples is invariant."""
    rates, durations = np.asarray(rates, float), np.asarray(durations_s, float)
    if rates.shape != durations.shape or not np.isfinite(rates).all() or not np.isfinite(durations).all():
        raise ValueError('Finite matching rate/duration shapes required')
    if np.any(durations < 0):
        raise ValueError('Durations must be nonnegative')
    return float(np.sum(rates * durations))


def absorbing_tail(remaining_s, config=None):
    """Worst bounded-rate future return under policy-step left-end discount.

    The next policy interval has discount1, subsequent intervals exp(-dt/tau).
    A final partial interval receives its actual duration. This exactly matches
    sum(gamma**k * min_rate * interval_duration), avoiding an early-exit gain
    under that same return contract. If termination is inside a policy interval,
    the parent groups its physical rewards/tail at that interval's boundary.
    """
    cfg = config or RewardConfig()
    remaining = float(remaining_s)
    if not np.isfinite(remaining) or remaining < 0:
        raise ValueError('remaining_s must be finite nonnegative')
    dt = cfg.policy_discount_dt_s
    steps = int(np.floor(remaining / dt + 1e-10))
    fraction = max(0., remaining - steps * dt)
    gamma = np.exp(-dt / cfg.discount_time_constant_s)
    geometric = -np.expm1(-steps * dt / cfg.discount_time_constant_s) / (-np.expm1(-dt / cfg.discount_time_constant_s))
    return float(cfg.min_rate * (dt * geometric + fraction * gamma ** steps))


class RewardTracker:
    """1ms command-history reward tracker; time is measured from episode origin.

    update receives the END state of one actual 1ms physical interval. Reward is
    that sample's bounded rate *1ms. Hard terminal flags are training decisions,
    independent of engineering/original acceptance. On first terminal, append
    absorbing_tail(remaining_s) as a mathematical absorbing-return tail, using
    the declared policy discount period/time constant. No synthetic samples,
    actions or observations are generated; do not add that tail twice. The
    no-escape comparison applies to these bounded rates and this discount
    contract, and does not establish behavior safety or unclipped-reward bounds.
    """
    def __init__(self, config=None, *, horizon_s=2., target_speed=.1, initial_heading=(1., 0.), start_time_s=0.):
        self.config = config or RewardConfig()
        self.horizon_s = float(horizon_s)
        self.target_speed = float(target_speed)
        self.initial_heading = _heading(initial_heading)
        if not np.isfinite(self.horizon_s) or self.horizon_s <= 0:
            raise ValueError('horizon_s must be finite positive')
        if not np.isfinite(self.target_speed) or self.target_speed < 0:
            raise ValueError('target_speed must be finite nonnegative')
        self.reset(start_time_s=start_time_s)

    def reset(self, *, start_time_s=0., initial_heading=None):
        self.start_time_s = float(start_time_s)
        if not np.isfinite(self.start_time_s):
            raise ValueError('start_time_s must be finite')
        if initial_heading is not None:
            self.initial_heading = _heading(initial_heading)
        self.last_time_s = self.start_time_s
        self.steps = 0
        self.terminated = False
        self.window_steps = int(round(self.config.action_change_window_s / self.config.dt_s))
        self.history = deque([np.zeros(42) for _ in range(self.window_steps)], maxlen=self.window_steps)

    @staticmethod
    def measure(core, action):
        """Copy explicit physical inputs; no core mutation or derivative update."""
        measurement = core.measure()
        return dict(com=core.d.subtree_com[core.th].copy(), com_velocity=core.com_velocity().copy(),
                    rotation=core.d.xmat[core.th].reshape(3, 3).copy(),
                    loads=np.asarray(measurement['loads'], float).copy(),
                    nonfoot_force_native=float(measurement['nonfoot']),
                    speed=np.asarray(measurement['speed'], float).copy(),
                    action=np.asarray(action, float).copy(), time_s=float(core.d.time))

    def update(self, sample):
        if self.terminated:
            raise RuntimeError('RewardTracker requires reset after terminal')
        time_s = float(sample['time_s'])
        expected = self.start_time_s + (self.steps + 1) * self.config.dt_s
        if not np.isfinite(time_s) or not np.isclose(time_s, expected, rtol=0, atol=1e-10):
            raise ValueError('update must receive each consecutive 1ms end sample')
        elapsed = time_s - self.start_time_s
        if elapsed > self.horizon_s + 1e-10:
            raise ValueError('Sample exceeds declared horizon')
        result = evaluate_reward(sample, self.history[0], self.initial_heading,
                                 self.target_speed, self.config)
        remaining = max(0., self.horizon_s - elapsed)
        result['remaining_s'] = remaining
        result['reward'] = result['rate'] * self.config.dt_s
        result['absorbing_penalty'] = absorbing_tail(remaining, self.config) if result['terminated'] else 0.
        result['absorbing_discount_contract'] = dict(
            policy_dt_s=self.config.policy_discount_dt_s,
            time_constant_s=self.config.discount_time_constant_s,
            next_interval_discount=1., mathematical_tail_only=True)
        result['total_reward'] = result['reward'] + result['absorbing_penalty']
        result['truncated'] = bool(not result['terminated'] and remaining <= 1e-10)
        result['elapsed_s'] = elapsed
        self.history.append(np.asarray(sample['action'], float).copy())
        self.steps += 1
        self.last_time_s = time_s
        self.terminated = result['terminated'] or result['truncated']
        return result
