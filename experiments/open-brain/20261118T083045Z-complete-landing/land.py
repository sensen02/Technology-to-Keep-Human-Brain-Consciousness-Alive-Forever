"""Fly a single continuous episode from ground through DNp11 and DNp10."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

TAKEOFF = Path(__file__).resolve().parent.parent / '20261118T075610Z-complete-takeoff'
sys.path.insert(0, str(TAKEOFF))
from takeoff import (BASELINE, COLUMNS as TAKEOFF_COLUMNS, LaunchMotor, LeggedFlightTeacher,
                     Settings, angles, build_env, com2root, digest, ground_support,
                     load_policy, make_env, set_ground_pose, smooth)
from flybody.tasks.task_utils import root2com

COLUMNS = (*TAKEOFF_COLUMNS, 'dnp10_left', 'dnp10_right', 'landing_phase',
           'altitude_ref_cm', 'forward_speed_ref_cm_s')
CASES = ('intact', 'cut', 'sham')
DURATION = 1.05


@dataclass(frozen=True)
class Approach:
    brake_s: float = .165
    descent_delay_s: float = .026
    descent_s: float = .260
    final_height_cm: float = .080
    flare_delay_s: float = .070
    flare_s: float = .160
    flare_pitch_deg: float = 8.
    deploy_cm: float = .40
    deploy_tau_s: float = .012
    fold_s: float = .010
    fold_precontact_cm: float = 0.
    neutral_after_contact_s: float = .004
    foot_level: float = 0.
    reach_cm: float = .050
    stance_blend_s: float = .042
    sweep_rad: float = 0.
    sweep_delay_s: float = .035
    near_amplitude: float = 1.
    approach_height_cm: float = 0.
    approach_hold_s: float = .045
    final_descent_s: float = .130
    trim_gain: float = 0.
    synchronized_fold: int = 0
    reference_entry: int = 0
    support_feedback: int = 0
    contact_grip: float = 0.
    rear_hold_s: float = 0.
    settled_height_cm: float = 0.


class Feet:
    """Use the observed body pose and claw Jacobians to reach a level stance."""

    def __init__(self, task, physics, names, reach):
        self.task = task
        self.physics = physics
        self.reach = reach
        self.linear = np.zeros((3, physics.model.nv))
        self.angular = np.zeros_like(self.linear)
        root, _ = task._walker.get_pose(physics)
        root = np.asarray(root).copy()
        self.legs = []
        for side in ('left', 'right'):
            for leg in ('T1', 'T2', 'T3'):
                suffix = f'{leg}_{side}'
                body = physics.model.name2id(f'walker/claw_{suffix}', 'body')
                joints = [f'{part}_{suffix}' for part in ('coxa_abduct', 'coxa', 'femur', 'tibia')]
                ids = [physics.model.name2id(f'walker/{name}', 'joint') for name in joints]
                self.legs.append((body, np.asarray(physics.data.xpos[body]) - root,
                                  [names.index(name) for name in joints],
                                  physics.model.jnt_dofadr[ids], physics.model.jnt_qposadr[ids]))

    def apply(self, targets, mix, support_feedback):
        root, quat = self.task._walker.get_pose(self.physics)
        root = np.asarray(root).copy()
        yaw, roll, pitch = angles(quat)
        rot = np.array(((np.cos(yaw), -np.sin(yaw), 0.), (np.sin(yaw), np.cos(yaw), 0.), (0., 0., 1.)))
        for body, offset, action, dofs, qpos in self.legs:
            desired = root + rot @ offset - np.array([0., 0., self.reach])
            if support_feedback:
                # Increase reach under the lowered side while avoiding a large
                # virtual penetration command at the foot pads.
                tilt_height = np.sin(roll) * offset[1] - np.sin(pitch) * offset[0]
                desired[2] = max(desired[2] + .35 * tilt_height, -.022)
            error = desired - self.physics.data.xpos[body]
            mujoco.mj_jacBody(self.physics.model.ptr, self.physics.data.ptr, self.linear, self.angular, body)
            jac = self.linear[:, dofs]
            correction = jac.T @ np.linalg.solve(jac @ jac.T + .0015 * np.eye(3), error)
            q = self.physics.data.qpos[qpos] + np.clip(.70 * correction, -.35, .35)
            targets[action] = (1. - mix) * targets[action] + mix * q

    def crouch(self, targets, rise):
        for body, _, action, dofs, _ in self.legs:
            mujoco.mj_jacBody(self.physics.model.ptr, self.physics.data.ptr, self.linear, self.angular, body)
            vertical = self.linear[2, dofs]
            shift = rise * vertical / (vertical @ vertical + .0015)
            targets[action] += np.clip(shift, -.23, .23)


def commands_from(folder):
    info = json.loads((folder / 'IDENTITY.json').read_text())
    assert digest(folder / 'COMMANDS.npz') == info['sha256']
    with np.load(folder / 'COMMANDS.npz', allow_pickle=False) as data:
        assert tuple(data['cases']) == CASES
        assert tuple(data['labels']) == ('takeoff/left', 'takeoff/right', 'landing/left', 'landing/right')
        levels = {case: data['levels'][:, column] for column, case in enumerate(CASES)}
    return info, levels


class LandingMotor(LaunchMotor):
    """Track a braking descent, then transfer support to ground contacts."""

    def __init__(self, task, wing_stow, adapter, dt, launch, approach, physics):
        super().__init__(task, wing_stow, adapter, dt, launch)
        self.approach = approach
        self.landing_started = None
        self.fold_started = None
        self.fold_ready = None
        self.fold_from = None
        self.fold_progress = 0.
        self.stance_started = None
        self.support_settle = 0.
        self.crouch_ready_s = 0.
        self.crouch_armed = False
        self.crouch_rise = 0.
        self.since_touch = 0.
        self.deploy = 0.
        self.landing_phase = 0
        self.foot_mix = 0.
        self.stroke_offset = 0.
        self.stroke_scale = 1.
        self.roll_delta = 0.
        self.filtered_pose = None
        self.touched_claws = ()
        names = adapter.legged_spec.name.split('\t')
        self.feet_control = Feet(task, physics, names, approach.reach_cm) if approach.foot_level else None
        self.grip_columns = {name.removeprefix('adhere_'): i
                             for i, name in enumerate(names) if name.startswith('adhere_claw_')}

    def update_episode(self, command, now, feet, nonfeet, physics):
        super().update(command[:2], now, feet > 0 or nonfeet > 0, physics)
        if self.landing_started is None:
            if not self.engaged_air or not np.all(command[2:] >= .18):
                return
            self.landing_started = now
            self.landing_phase = 1
            self.set_approach(physics)
        params = self.approach
        height = physics.named.data.subtree_com['walker/'][2]
        self.stroke_offset = params.sweep_rad * smooth((now - self.landing_started - params.sweep_delay_s) / .050)
        self.stroke_scale = 1. - (1. - params.near_amplitude) * smooth((.52 - height) / .24)
        if params.trim_gain:
            _, q = self.task._walker.get_pose(physics)
            _, roll, pitch = angles(q)
            omega = physics.bind(self.task._root_joint).qvel[3:5]
            current = np.r_[roll, pitch, omega]
            if self.filtered_pose is None:
                self.filtered_pose = current.copy()
            self.filtered_pose += -np.expm1(-self.dt / .006) * (current - self.filtered_pose)
            roll, pitch, wx, wy = self.filtered_pose
            desired = angles(self.task._ref_qpos[self.task._step_counter, 3:7])[2]
            enable = params.trim_gain * smooth((now - self.landing_started - .02) / .04)
            self.roll_delta = float(np.clip(enable * (.21 * roll + .0035 * wx), -.16, .16))
            self.stroke_offset += float(np.clip(enable * (.42 * (desired - pitch) - .0028 * wy), -.18, .34))
        if self.landing_phase >= 2 or height <= params.deploy_cm:
            self.deploy = 1. - (1. - self.deploy) * np.exp(-self.dt / params.deploy_tau_s)
            self.landing_phase = max(self.landing_phase, 2)
        self.since_touch = self.since_touch + self.dt if feet >= 2 and nonfeet == 0 else 0.
        if self.fold_started is None:
            low = params.fold_precontact_cm > 0 and height < params.fold_precontact_cm
            if self.fold_ready is None and (self.since_touch >= params.neutral_after_contact_s or low):
                self.fold_ready = now
            if self.fold_ready is not None:
                wing = physics.bind(self.task._wing_joints).qpos
                swept_back = np.min(wing[[0, 3]]) >= .95
                if not params.synchronized_fold or swept_back or now - self.fold_ready >= .005:
                    self.fold_started = now
                    self.landing_phase = 3
                    if params.synchronized_fold:
                        self.fold_from = wing.copy()
        if self.fold_started is not None:
            folded = smooth((now - self.fold_started - params.rear_hold_s) / params.fold_s)
            self.fold_progress = folded
            if not params.support_feedback:
                self.stance_started = self.fold_started
            elif self.stance_started is None:
                _, orientation = self.task._walker.get_pose(physics)
                _, roll, pitch = angles(orientation)
                speed = np.linalg.norm(physics.bind(self.task._root_joint).qvel[:3])
                steady = feet >= 4 and nonfeet == 0 and speed < 3.5 and max(abs(roll), abs(pitch)) < .42
                self.support_settle = self.support_settle + self.dt if steady else 0.
                if self.support_settle >= .010:
                    self.stance_started = now
            ground_blend = (1. if self.stance_started is None else
                            1. - smooth((now - self.stance_started - params.rear_hold_s) / params.stance_blend_s))
            self.wing_mix *= 1. - folded
            self.teacher_mix *= 1. - folded
            if params.synchronized_fold:
                self.teacher_mix *= 1. - smooth((now - self.fold_started) / .002)
            self.deploy = max(self.deploy, folded)
            if folded == 1.:
                self.landing_phase = 4
        else:
            ground_blend = 1.
        self.foot_mix = params.foot_level * self.deploy * ground_blend
        self.tuck *= 1. - self.deploy
        if params.settled_height_cm and self.landing_phase == 4 and ground_blend == 0.:
            supported = feet == 6 and nonfeet == 0
            self.crouch_ready_s = self.crouch_ready_s + self.dt if supported else 0.
            self.crouch_armed = self.crouch_armed or self.crouch_ready_s >= .022
            if self.crouch_armed and supported:
                self.crouch_rise = float(np.clip(self.crouch_rise
                    + 16. * self.dt * (height - params.settled_height_cm), 0., .030))

    def wing_target(self, ctrl_freq):
        goal = self.base_pattern(ctrl_freq).copy()
        if self.fold_from is not None:
            return self.fold_from + self.fold_progress * (self.wing_stow - self.fold_from)
        if self.stroke_offset != 0. or self.stroke_scale != 1. or self.roll_delta != 0.:
            # Shift the symmetric stroke center; joint actuator bounds remain in effect.
            scales = self.stroke_scale + np.array([-self.roll_delta, self.roll_delta])
            goal[[0, 3]] = np.clip(.288 + scales * (goal[[0, 3]] - .288)
                                  + self.stroke_offset, -1.5, 1.5)
        return self.wing_stow + self.wing_mix * (goal - self.wing_stow)

    def action(self, policy, observation):
        targets = super().action(policy, observation)
        if self.foot_mix > 0.:
            self.feet_control.apply(targets, self.foot_mix, self.approach.support_feedback)
        if self.crouch_rise > 0.:
            self.feet_control.crouch(targets, self.crouch_rise)
        if self.foot_mix > 0. or self.crouch_rise > 0.:
            spec = self.adapter.legged_spec
            targets = np.clip(targets, spec.minimum, spec.maximum)
        if self.landing_phase >= 2 and self.approach.contact_grip:
            for claw in self.touched_claws:
                targets[self.grip_columns[claw]] = self.approach.contact_grip
        return targets

    def set_approach(self, physics):
        params = self.approach
        tick = round(self.landing_started / self.dt)
        # The reference is a target for the published position/orientation policy.
        # It is anchored to the body's current observation and never sets its pose.
        if params.reference_entry:
            previous = self.task._ref_qpos[tick]
            positions = root2com(previous)
            quat = previous[3:7].copy()
            start_vel = self.task._ref_qvel[tick, 0]
        else:
            positions = physics.named.data.subtree_com['walker/'].copy()
            _, quat = self.task._walker.get_pose(physics)
            start_vel = physics.bind(self.task._root_joint).qvel[0]
        start_pitch = -angles(quat)[2]
        tau = np.arange(len(self.task._ref_qpos) - tick) * self.dt
        speed = start_vel * (1. - smooth(tau / params.brake_s))
        ref = np.repeat(positions[None, :], len(tau), axis=0)
        ref[1:, 0] += np.cumsum((speed[1:] + speed[:-1]) * (.5 * self.dt))
        relief = smooth((tau - params.descent_delay_s) / params.descent_s)
        if params.approach_height_cm:
            ref[:, 2] += relief * (params.approach_height_cm - positions[2])
            start_final = params.descent_delay_s + params.descent_s + params.approach_hold_s
            final = smooth((tau - start_final) / params.final_descent_s)
            ref[:, 2] += final * (params.final_height_cm - params.approach_height_cm)
        else:
            ref[:, 2] += relief * (params.final_height_cm - positions[2])
        flare = smooth((tau - params.flare_delay_s) / params.flare_s)
        pitch = start_pitch + flare * (np.deg2rad(params.flare_pitch_deg) - start_pitch)
        q = np.column_stack((np.cos(pitch / 2.), np.zeros(len(tau)),
                             -np.sin(pitch / 2.), np.zeros(len(tau))))
        self.task._ref_qpos[tick:] = np.column_stack((com2root(ref, q), q))
        self.task._ref_qvel[tick:, :3] = np.gradient(ref, self.dt, axis=0, edge_order=2)
        self.task._ref_qvel[tick:, 3:] = 0.
        self.task._ref_qvel[tick:, 4] = -np.gradient(pitch, self.dt, edge_order=2)


def simulate(policy, stock, stock_observation, commands, dt_ms, launch, approach):
    env = build_env()
    env._time_limit = launch.duration + .001
    task = env.task
    env.random_state.seed(launch.seed)
    setup = task.initialize_episode

    def initialize(physics, random):
        setup(physics, random)
        set_ground_pose(env)
        task._ref_qpos[:, :3] = task._walker.get_pose(physics)[0]
        task._ref_qpos[:, 3:7] = (1., 0., 0., 0.)
        task._ref_qvel[:] = 0.

    task.initialize_episode = initialize
    task.check_termination = lambda physics: False
    step = env.reset()
    phy = env.physics
    assert phy.model.body_contype[0] & 1 and phy.model.body_conaffinity[0] & 1
    assert phy.model.opt.density > 0 and phy.model.opt.viscosity > 0
    dt = env.control_timestep()
    native_dt = phy.timestep()
    substeps = round(dt / native_dt)
    stride = round(1000 * dt / dt_ms)
    extra = max(0, round(launch.duration / dt) + task._future_steps + 2 - len(task._ref_qpos))
    task._ref_qpos = np.pad(task._ref_qpos, ((0, extra), (0, 0)), mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel, ((0, extra), (0, 0)), mode='edge')
    adapter = LeggedFlightTeacher(stock, stock_observation, env)
    motor = LandingMotor(task, phy.bind(task._wing_joints).qpos.copy(), adapter, dt, launch, approach, phy)
    ground = phy.model.name2id('groundplane', 'geom')
    wing = [idx for idx in range(phy.model.nu)
            if 'wing_' in (phy.model.id2name(idx, 'actuator') or '')]
    binding = phy.bind(task._wing_joints)
    root = phy.bind(task._root_joint)
    initial = np.concatenate([phy.data.qpos.copy(), phy.data.qvel.copy()])
    traces, poses, actions = [], [], []
    bodies_seen = set()
    seen_contact_feet = set()
    command = np.zeros(4)
    old_sample = task.after_substep
    wpg = task._wbpg
    phase = float(wpg.traj_ctrl[wpg._freq_idx]['phase'][wpg._step])

    def sample(physics, random):
        old_sample(physics, random)
        feet, nonfeet, bodies = ground_support(physics, ground)
        motor.touched_claws = tuple({body.removeprefix('walker/') for body in bodies
                                    if body.startswith('walker/claw_')})
        if motor.landing_started is not None:
            bodies_seen.update(body for body in bodies if 'tarsus' not in body and 'claw' not in body)
            seen_contact_feet.update(feet)
        com = physics.named.data.subtree_com['walker/']
        _, quat = task._walker.get_pose(physics)
        external = max(np.abs(physics.data.qfrc_applied).max(), np.abs(physics.data.xfrc_applied).max())
        ref_idx = task._step_counter
        traces.append([physics.time(), *com, *root.qvel[:3], *angles(quat), len(feet),
                       nonfeet, external, np.abs(physics.data.actuator_force[wing]).sum(),
                       *binding.qpos[[0, 3]], *command[:2], motor.wing_mix, motor.tuck,
                       motor.teacher_mix, motor.phase, *command[2:], motor.landing_phase,
                       task._ref_qpos[ref_idx, 2], task._ref_qvel[ref_idx, 0]])

    task.after_substep = sample
    for tick in range(round(launch.duration / dt)):
        now = phy.time()
        command = commands[tick * stride] if tick * stride < len(commands) else np.zeros(4)
        feet, nonfeet = traces[-1][10:12] if traces else (6, 0)
        motor.update_episode(command, now, feet, nonfeet, phy)
        action = motor.action(policy, step.observation)
        step = env.step(action)
        assert abs(phy.time() - (tick + 1) * dt) < 1e-8
        poses.append(phy.data.qpos.copy())
        actions.append(action)
        if step.last():
            break
    return {
        'TRACE.npy': np.asarray(traces), 'QPOS.npy': np.asarray(poses),
        'ACTION.npy': np.asarray(actions), 'INITIAL.npy': initial,
    }, {
        'columns': COLUMNS, 'native_dt_s': float(native_dt), 'control_dt_s': float(dt),
        'native_steps': len(traces), 'substeps': substeps, 'terminal': bool(step.last()),
        'initial_oscillator_phase': phase, 'launch': asdict(launch), 'approach': asdict(approach),
        'takeoff_trigger_s': motor.started, 'landing_trigger_s': motor.landing_started,
        'fold_trigger_s': motor.fold_started, 'nonfoot_bodies_after_landing_trigger': sorted(bodies_seen),
        'landing_feet': sorted(seen_contact_feet),
        'actuator_names': adapter.legged_spec.name.split('\t'),
    }


def save(folder, result):
    arrays, info = result
    folder.mkdir()
    for name, values in arrays.items():
        assert np.isfinite(values).all(), name
        np.save(folder / name, values, allow_pickle=False)
    info['files'] = {name: digest(folder / name) for name in arrays}
    trace = arrays['TRACE.npy']
    late = trace[trace[:, 0] > .41]
    hits = (late[:, 10] + late[:, 11]) > 0.
    info['landing_observations'] = {
        'first_contact_s': float(late[np.flatnonzero(hits)[0], 0]) if hits.any() else None,
        'late_foot_max': int(late[:, 10].max()),
        'late_nonfoot_samples': int(np.count_nonzero(late[:, 11])),
        'last_100ms_min_feet': int(trace[trace[:, 0] > trace[-1, 0] - .1, 10].min()),
        'end_height_mm': float(10 * trace[-1, 3]),
        'end_speed_mm_s': float(10 * np.linalg.norm(trace[-1, 4:7])),
        'end_pitch_deg': float(np.rad2deg(trace[-1, 9])),
        'end_wing_mix': float(trace[-1, 18]),
        'end_state': int(trace[-1, 24]),
    }
    (folder / 'RESULT.json').write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    print(folder.name, info['landing_observations'], info['nonfoot_bodies_after_landing_trigger'], flush=True)


def setup_policy():
    stock = make_env()
    observation = stock.reset().observation
    model = BASELINE / 'flight_reference/flight'
    return load_policy(model), stock, observation


def launch_settings(seed, duration=DURATION):
    config = json.loads((TAKEOFF / 'CALIBRATED.json').read_text())
    config.update(seed=seed, duration=duration)
    return Settings(**config)


def main(args):
    info, levels = commands_from(args.neck)
    policy, stock, observation = setup_policy()
    launch = launch_settings(args.seed, args.duration)
    approach = Approach(**args.approach)
    args.output.mkdir()
    for case in args.cases:
        save(args.output / case, simulate(policy, stock, observation, levels[case],
                                         info['dt_ms'], launch, approach))
    (args.output / 'IDENTITY.json').write_text(json.dumps({
        'neck_source': str(args.neck.resolve()), 'neck_data_sha256': info['sha256'],
        'takeoff_config_sha256': digest(TAKEOFF / 'CALIBRATED.json'),
        'flight_policy_sha256': digest(BASELINE / 'flight_reference/flight/saved_model.pb'),
        'implementation_sha256': digest(Path(__file__)), 'training_updates': 0,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--duration', type=float, default=DURATION)
    parser.add_argument('--config', type=Path)
    fields = asdict(Approach())
    for key, value in fields.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value))
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    config.update({key: getattr(args, key) for key in fields if getattr(args, key) is not None})
    args.approach = config
    main(args)
