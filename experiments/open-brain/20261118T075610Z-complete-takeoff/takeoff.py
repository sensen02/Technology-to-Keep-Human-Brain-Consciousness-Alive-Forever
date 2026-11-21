"""Continuous FlyBody ground departure through the mapped post-neck DNp11 readout."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PARENT = Path(__file__).resolve().parent.parent / '20261118T072349Z-flight-descending'
sys.path.insert(0, str(PARENT))
from fly import BASELINE, angles, digest, load_policy, make_env
from flybody.tasks.task_utils import com2root
from legged_air import LeggedFlightTeacher
from legged_fixture import build_env, ground_support, set_ground_pose
from transition import kick_direction

COLUMNS = ('time_s', 'x_cm', 'y_cm', 'z_cm', 'vx_cm_s', 'vy_cm_s', 'vz_cm_s',
           'yaw_rad', 'roll_rad', 'pitch_rad', 'feet', 'nonfoot_contacts',
           'external_applied', 'wing_force', 'left_stroke',
           'right_stroke', 'vnc_left', 'vnc_right', 'wing_mix', 'leg_tuck',
           'teacher_mix', 'phase')


@dataclass(frozen=True)
class Settings:
    wing_delay: float = .009
    wing_ramp: float = .005
    teacher_delay: float = .019
    teacher_ramp: float = .015
    teacher_gain: float = 1.
    symmetry_time: float = 0.
    pitch_ramp: float = .035
    speed_ramp: float = .075
    climb_time: float = .135
    climb_cm: float = .95
    flight_pitch_deg: float = 47.5
    kick_rad: float = 1.25
    kick_hold: float = .030
    tuck_tau: float = .006
    seed: int = 17
    duration: float = .40


def smooth(x):
    value = np.clip(x, 0., 1.)
    return value**2 * (3. - 2. * value)


def load_neck(folder=PARENT / 'NECK'):
    path = folder / 'POST_NECK.npz'
    receipt = json.loads((folder / 'IDENTITY.json').read_text())
    assert digest(path) == receipt['data_sha256']
    assert receipt == json.loads((PARENT / 'NECK/IDENTITY.json').read_text())
    with np.load(path, allow_pickle=False) as data:
        cases = tuple(data['conditions'])
        labels = tuple(data['labels'])
        dn = [labels.index(f'takeoff/{side}') for side in ('left', 'right')]
        intact, cut = (cases.index(name) for name in ('takeoff_probe', 'takeoff_cut'))
        emitted = data['emitted']
        arrived = data['arrived']
        assert np.array_equal(emitted[:, intact], emitted[:, cut])
        assert not np.any(arrived[:, cut]) and not np.any(data['levels'][:, cut])
        for column in dn:
            ticks = np.flatnonzero(emitted[:, intact, column])
            assert np.array_equal(ticks + receipt['ports'][labels[column]]['delay_ticks'],
                                  np.flatnonzero(arrived[:, intact, column]))
        levels = {name: data['levels'][:, cases.index(name), :][:, dn]
                  for name in ('takeoff_probe', 'takeoff_cut', 'sham')}
    return receipt, levels


class LaunchMotor:
    """Bounded middle-leg extension, wing opening and contact-gated leg tucking."""

    def __init__(self, task, wing_stow, adapter, dt, settings):
        self.task = task
        self.base_pattern = task._wbpg.step
        task._wbpg.step = self.wing_target
        self.wing_stow, self.adapter, self.dt, self.settings = wing_stow, adapter, dt, settings
        self.started = None
        self.kick_targets = None
        self.kick = np.zeros(2)
        self.wing_mix = 0.
        self.teacher_mix = 0.
        self.tuck = 0.
        self.air_age = 0.
        self.phase = 0
        self.engaged_air = False
        self.asymmetric_mix = 1.
        names = self.adapter.legged_spec.name.split('\t')
        self.wing_columns = np.array([[names.index(f'wing_{axis}_{side}') for axis in ('yaw', 'roll', 'pitch')]
                                      for side in ('left', 'right')])

    def wing_target(self, ctrl_freq):
        goal = self.base_pattern(ctrl_freq)
        return self.wing_stow + self.wing_mix * (goal - self.wing_stow)

    def update(self, vnc, now, direct_support, physics):
        if self.started is None:
            if not np.all(vnc >= .22):
                return
            self.started = now
            self.phase = 1
            self.kick_targets = kick_direction(physics, self.adapter.legged_spec.name.split('\t'))
            self.set_reference(physics)
        age = now - self.started
        p = self.settings
        self.kick += -np.expm1(-self.dt / .001) * ((age < p.kick_hold) - self.kick)
        self.wing_mix = smooth((age - p.wing_delay) / p.wing_ramp)
        self.teacher_mix = self.wing_mix * p.teacher_gain * smooth((age - p.teacher_delay) / p.teacher_ramp)
        self.asymmetric_mix = (1. if p.symmetry_time == 0. else
                               smooth((age - p.symmetry_time) / .025))
        self.air_age = self.air_age + self.dt if not direct_support else 0.
        self.engaged_air = self.engaged_air or self.air_age >= .001
        self.tuck += -np.expm1(-self.dt / p.tuck_tau) * (self.engaged_air - self.tuck)
        if self.engaged_air:
            self.phase = 2

    def set_reference(self, physics):
        p = self.settings
        com = physics.named.data.subtree_com['walker/'].copy()
        age = np.maximum(np.arange(len(self.task._ref_qpos)) * self.dt - self.started, 0.)
        speed = 20. * smooth(age / p.speed_ramp)
        positions = np.repeat(com[None, :], len(age), axis=0)
        positions[:, 0] += np.cumsum(speed) * self.dt
        positions[:, 2] += p.climb_cm * smooth(age / p.climb_time)
        pitch = np.deg2rad(p.flight_pitch_deg) * smooth(age / p.pitch_ramp)
        quaternion = np.column_stack((np.cos(pitch / 2.), np.zeros(len(age)),
                                      -np.sin(pitch / 2.), np.zeros(len(age))))
        self.task._ref_qpos[:] = np.column_stack((com2root(positions, quaternion), quaternion))
        self.task._ref_qvel[:, :3] = np.gradient(positions, self.dt, edge_order=2, axis=0)
        self.task._ref_qvel[:, 3:6] = 0.
        self.task._ref_qvel[:, 4] = -np.gradient(pitch, self.dt, edge_order=2)

    def action(self, policy, observation):
        spec = self.adapter.legged_spec
        action = self.tuck * self.adapter.other_targets
        if self.teacher_mix > 0.:
            teacher = self.adapter.action(policy, observation)
            action[self.adapter.action_columns] = self.teacher_mix * teacher[self.adapter.action_columns]
            if self.asymmetric_mix < 1.:
                wing = action[self.wing_columns]
                common = wing.mean(axis=0)
                action[self.wing_columns] = common + self.asymmetric_mix * (wing - common)
        if self.kick_targets is not None:
            for strength, (indices, direction) in zip(self.kick, self.kick_targets, strict=True):
                action[indices] += (1. - self.tuck) * strength * self.settings.kick_rad * direction
        return np.clip(action, spec.minimum, spec.maximum)


def simulate(policy, stock, stock_observation, commands, dt_ms, settings, *, fluid=True):
    env = build_env()
    task = env.task
    env.random_state.seed(settings.seed)
    original_setup = task.initialize_episode

    def initialize(physics, random):
        original_setup(physics, random)
        set_ground_pose(env)
        task._ref_qpos[:, :3] = task._walker.get_pose(physics)[0]
        task._ref_qpos[:, 3:7] = (1., 0., 0., 0.)
        task._ref_qvel[:] = 0.

    task.initialize_episode = initialize
    # The upstream airborne imitation floor is higher than a standing fly.
    # Our own full native contact and state checks apply through the transition.
    task.check_termination = lambda physics: False
    step = env.reset()
    phy = env.physics
    assert phy.model.body_contype[0] & 1 and phy.model.body_conaffinity[0] & 1
    assert phy.model.opt.density > 0 and phy.model.opt.viscosity > 0
    if not fluid:
        phy.model.opt.density = phy.model.opt.viscosity = 0.
    dt = env.control_timestep()
    substeps = round(dt / phy.timestep())
    stride = round(1000 * dt / dt_ms)
    # Extend the ghost reference through the full episode. No walker state is written here.
    extra = max(0, round(settings.duration / dt) + task._future_steps + 2 - len(task._ref_qpos))
    task._ref_qpos = np.pad(task._ref_qpos, ((0, extra), (0, 0)), mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel, ((0, extra), (0, 0)), mode='edge')
    adapter = LeggedFlightTeacher(stock, stock_observation, env)
    motor = LaunchMotor(task, phy.bind(task._wing_joints).qpos.copy(), adapter, dt, settings)
    ground = phy.model.name2id('groundplane', 'geom')
    wing = [idx for idx in range(phy.model.nu)
            if 'wing_' in (phy.model.id2name(idx, 'actuator') or '')]
    binding = phy.bind(task._wing_joints)
    root = phy.bind(task._root_joint)
    initial = np.concatenate([phy.data.qpos.copy(), phy.data.qvel.copy()])
    initial_time = phy.time()
    wpg = task._wbpg
    initial_wing_phase = float(wpg.traj_ctrl[wpg._freq_idx]['phase'][wpg._step])
    traces, poses, actuators = [], [], []
    nonfoot_bodies = set()
    vnc = np.zeros(2)
    original_sampler = task.after_substep

    def sample(physics, random):
        original_sampler(physics, random)
        feet, nonfeet, bodies = ground_support(physics, ground)
        nonfoot_bodies.update(body for body in bodies if 'tarsus' not in body and 'claw' not in body)
        com = physics.named.data.subtree_com['walker/']
        _, quat = task._walker.get_pose(physics)
        external = max(np.abs(physics.data.qfrc_applied).max(), np.abs(physics.data.xfrc_applied).max())
        traces.append([physics.time(), *com, *root.qvel[:3], *angles(quat), len(feet),
                       nonfeet, external, np.abs(physics.data.actuator_force[wing]).sum(),
                       *binding.qpos[[0, 3]], *vnc, motor.wing_mix, motor.tuck, motor.teacher_mix, motor.phase])

    task.after_substep = sample
    for tick in range(round(settings.duration / dt)):
        now = phy.time()
        vnc = commands[tick * stride] if tick * stride < len(commands) else np.zeros(2)
        if traces:
            supported = traces[-1][10] > 0 or traces[-1][11] > 0
        else:
            supported = True
        motor.update(vnc, now, supported, phy)
        action = motor.action(policy, step.observation)
        step = env.step(action)
        assert abs(phy.time() - initial_time - (tick + 1) * dt) < 1e-8, 'Unexpected reset or time jump'
        poses.append(phy.data.qpos.copy())
        actuators.append(action)
        if step.last():
            break
    return np.asarray(traces), np.asarray(poses), initial, np.asarray(actuators), {
        'settings': asdict(settings), 'trigger_s': motor.started, 'engaged_flight': motor.engaged_air,
        'terminal': bool(step.last()), 'control_steps': len(poses), 'native_steps': len(traces),
        'nonfoot_bodies': sorted(nonfoot_bodies),
        'initial_wing_phase': initial_wing_phase,
        'substeps': substeps, 'dt_s': float(phy.timestep()), 'fluid': fluid,
        'wing_stow_rad': motor.wing_stow.tolist(),
        'actuator_names': adapter.legged_spec.name.split('\t'),
    }


def save(path, result):
    trace, poses, initial, actuators, facts = result
    invalid = np.argwhere(~np.isfinite(trace))
    if len(invalid):
        row, column = invalid[0]
        raise ValueError(f'Nonfinite {COLUMNS[column]} at native row {row}: {trace[row]}')
    path.mkdir()
    for filename, values in [('TRACE.npy', trace), ('QPOS.npy', poses), ('INITIAL.npy', initial),
                             ('ACTION.npy', actuators)]:
        np.save(path / filename, values, allow_pickle=False)
    facts['columns'] = COLUMNS
    facts['files'] = {p.name: digest(p) for p in sorted(path.glob('*.npy'))}
    facts['max_z_mm'] = float(10 * trace[:, 3].max())
    facts['end_xyz_mm'] = (10 * trace[-1, 1:4]).tolist()
    facts['nonfoot_samples'] = int(np.count_nonzero(trace[:, 11]))
    after_flight = (trace[:, 10:12].sum(axis=1) == 0) & (trace[:, 0] > .08)
    facts['air_samples'] = int(after_flight.sum())
    first_air = np.flatnonzero(after_flight)
    facts['first_air_s'] = float(trace[first_air[0], 0]) if len(first_air) else None
    facts['end_angles_deg'] = np.rad2deg(trace[-1, 7:10]).tolist()
    (path / 'RESULT.json').write_text(json.dumps(facts, indent=2, allow_nan=False) + '\n')
    print(path.name, {key: facts[key] for key in ('max_z_mm', 'end_xyz_mm', 'first_air_s',
                                                   'nonfoot_samples', 'air_samples', 'end_angles_deg')},
          flush=True)


def main(output, settings, cases, neck):
    receipt, levels = load_neck(neck)
    stock = make_env()
    stock_observation = stock.reset().observation
    model = BASELINE / 'flight_reference/flight'
    policy = load_policy(model)
    output.mkdir()
    for case in cases:
        commands = levels['takeoff_probe' if case in ('fluid_off', 'legs_only') else case]
        config = replace(settings, wing_delay=settings.duration) if case == 'legs_only' else settings
        save(output / case, simulate(policy, stock, stock_observation, commands,
                                     receipt['dt_ms'], config, fluid=case != 'fluid_off'))
    (output / 'IDENTITY.json').write_text(json.dumps({
        'post_neck_sha256': receipt['data_sha256'],
        'post_neck_source': str(neck.resolve()),
        'policy': {str(p.relative_to(model)): digest(p) for p in sorted(model.rglob('*')) if p.is_file()},
        'wing_pattern_sha256': digest(BASELINE / 'flight_reference/wing_pattern_fmech.npy'),
        'source_sha256': digest(Path(__file__)),
        'mujoco': mujoco.__version__, 'python': sys.version, 'training_updates': 0,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--neck', type=Path, default=PARENT / 'NECK')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cases', nargs='+', choices=('takeoff_probe', 'takeoff_cut', 'sham', 'fluid_off', 'legs_only'),
                        default=('takeoff_probe', 'takeoff_cut', 'sham', 'fluid_off'))
    for name, value in asdict(Settings()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(value))
    args = vars(parser.parse_args())
    dest = args.pop('output')
    cases = args.pop('cases')
    neck = args.pop('neck')
    config = args.pop('config')
    defaults = json.loads(config.read_text()) if config else {}
    defaults.update({key: value for key, value in args.items() if value is not None})
    main(dest, Settings(**defaults), cases, neck)
