"""Exercise a legged and collidable FlyBody fixture without changing saved flight weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from fly import BASELINE, digest, enable_floor_contacts, load_policy, make_env, tf
from flybody.fly_envs import flight_imitation
from flybody.tasks.constants import _TERMINAL_HEIGHT
from flybody.tasks.trajectory_loaders import constant_speed_trajectory


def build_env():
    env = flight_imitation(wpg_pattern_path=str(BASELINE / 'flight_reference/wing_pattern_fmech.npy'),
                           disable_legs=False, random_state=np.random.RandomState(17),
                           terminal_com_dist=float('inf'))
    trajectory = constant_speed_trajectory(n_steps=1500, speed=20, init_pos=(0, 0, 1),
                                           body_rot_angle_y=-47.5,
                                           control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(*trajectory)
    enable_floor_contacts(env)
    return env


def set_ground_pose(env, lift_cm=0., vertical_speed=0., leg_positions=None):
    physics = env.physics
    task = env.task
    with physics.reset_context():
        # Native XML neutral legs and stock spawn height provide the starting
        # fixture. Put the wings at their published folded spring positions.
        physics.data.qpos[:] = physics.model.qpos0
        pos = task._walker.upright_pose.xpos + np.array([0., 0., lift_cm])
        task._walker.set_pose(physics, pos, [1., 0., 0., 0.])
        for joint in task._wing_joints:
            binding = physics.bind(joint)
            name = binding.element_id
            binding.qpos = physics.model.qpos_spring[physics.model.jnt_qposadr[name]]
        if leg_positions is not None:
            physics.bind(task._leg_joints).qpos = leg_positions
        physics.data.qvel[:] = 0.
        task._walker.set_velocity(physics, (0., 0., vertical_speed))
        physics.data.ctrl[:] = 0.


def ground_support(physics, ground):
    active = []
    for contact in physics.data.contact[:physics.data.ncon]:
        if ground not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = contact.geom2 if contact.geom1 == ground else contact.geom1
        body = physics.model.id2name(physics.model.geom_bodyid[other], 'body')
        if body.startswith('walker/'):
            active.append(body)
    feet = {tuple(body.rsplit('_', 2)[-2:]) for body in active
            if 'tarsus' in body or 'claw' in body}
    nonfeet = sum('tarsus' not in body and 'claw' not in body for body in active)
    return feet, nonfeet, active


def observe_standing(env, output):
    physics = env.physics
    task = env.task
    ground = physics.model.name2id('groundplane', 'geom')
    assert physics.model.body_contype[0] & 1 and physics.model.body_conaffinity[0] & 1
    set_ground_pose(env)
    initial = physics.data.qpos.copy()
    pre_agent = env.action_spec()
    action = np.clip(np.zeros(pre_agent.shape, dtype=pre_agent.dtype),
                     pre_agent.minimum, pre_agent.maximum)
    task._walker.apply_action(physics, action, env.random_state)
    poses, trace = [], []
    pairs = set()
    for _ in range(2400):
        physics.step()
        feet, nonfeet, active = ground_support(physics, ground)
        pairs.update(active)
        position = physics.named.data.subtree_com['walker/'].copy()
        external = max(np.abs(physics.data.xfrc_applied).max(), np.abs(physics.data.qfrc_applied).max())
        trace.append([physics.time(), *position, len(feet), nonfeet, external])
        if len(trace) % 4 == 0:
            poses.append(physics.data.qpos.copy())
    array = np.asarray(trace)
    assert np.isfinite(array).all() and not np.any(array[:, 6])
    output.mkdir()
    np.save(output / 'INITIAL.npy', initial, allow_pickle=False)
    np.save(output / 'QPOS.npy', poses, allow_pickle=False)
    np.save(output / 'TRACE.npy', array, allow_pickle=False)
    last = array[-400:]
    return {
        'initialization': 'stock upright spawn, qpos0 legs, folded wings, zero velocity',
        'controller': 'constant neutral leg commands, no wing or adhesion drive',
        'native_steps': len(trace), 'dt_s': float(physics.timestep()),
        'floor_contact_bodies': sorted(pairs),
        'last_20_ms_min_feet': int(last[:, 4].min()),
        'last_20_ms_nonfoot_samples': int(np.count_nonzero(last[:, 5])),
        'last_20_ms_com_excursion_mm': (10 * np.ptp(last[:, 1:4], axis=0)).tolist(),
        'final_height_mm': float(array[-1, 3] * 10),
        'external_applied_max': float(array[:, 6].max()),
        'trace_sha256': digest(output / 'TRACE.npy'),
        'files': {name: digest(output / name) for name in ('QPOS.npy', 'INITIAL.npy')},
        'trace_columns': ['time_s', 'com_x_cm', 'com_y_cm', 'com_z_cm',
                          'floor_support_feet', 'nonfoot_floor_contact_geoms', 'external_applied_max'],
    }


def main(output):
    stock, legged = make_env(), build_env()
    stock_step, legged_step = stock.reset(), legged.reset()
    policy = load_policy(BASELINE / 'flight_reference/flight')
    result = {'flight_wrapper_cutoff_mm': _TERMINAL_HEIGHT * 10,
              'stock_action_shape': list(stock.action_spec().shape),
              'legged_action_shape': list(legged.action_spec().shape),
              'legged_joint_count': len(legged.task._leg_joints),
              'shape_differences': {}}
    for name in sorted(set(stock_step.observation) | set(legged_step.observation)):
        a = stock_step.observation.get(name)
        b = legged_step.observation.get(name)
        if a is None or b is None or a.shape != b.shape:
            result['shape_differences'][name] = [list(a.shape) if a is not None else None,
                                                 list(b.shape) if b is not None else None]
    def obs(step):
        return {key: tf.convert_to_tensor(np.asarray(value, dtype=np.float32)[None])
                for key, value in step.observation.items()}
    result['stock_teacher_output_shape'] = list(policy(obs(stock_step)).mean().shape)
    try:
        result['legged_teacher_output_shape'] = list(policy(obs(legged_step)).mean().shape)
    except (TypeError, ValueError) as error:
        result['legged_teacher_error'] = f'{type(error).__name__}: {str(error).splitlines()[0]}'
    result['stance'] = observe_standing(legged, output)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
