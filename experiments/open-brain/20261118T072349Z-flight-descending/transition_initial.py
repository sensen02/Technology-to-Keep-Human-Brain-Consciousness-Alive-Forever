"""Calibrate post-neck leg actions at the ground/air boundary on one FlyBody."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from fly import digest, flight_input
from legged_fixture import build_env, ground_support, set_ground_pose


SIDES = ('left', 'right')
CASES = ('takeoff_probe', 'takeoff_cut', 'landing_probe', 'landing_cut')
COLUMNS = ('time_s', 'x_cm', 'y_cm', 'z_cm', 'feet', 'other_contacts',
           'speed_cm_s', 'vertical_cm_s', 'external_applied_max', 'vnc_left', 'vnc_right')


def leg_domains(env, spec):
    names = spec.name.split('\t')
    spring = {}
    for joint, angle in zip(env.task._leg_joints, env.task._leg_springrefs, strict=True):
        if joint.name in names:
            spring[names.index(joint.name)] = (joint.name, angle)
    return names, spring


def kick_direction(physics, names):
    shape = (3, physics.model.nv)
    translation, rotation = np.zeros(shape), np.zeros(shape)
    directions = []
    for side in SIDES:
        claw = physics.model.name2id(f'walker/claw_T2_{side}', 'body')
        joints = [f'{part}_T2_{side}' for part in ('coxa', 'femur', 'tibia')]
        columns = [physics.model.jnt_dofadr[physics.model.name2id(f'walker/{name}', 'joint')]
                   for name in joints]
        mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, translation, rotation, claw)
        gradient = translation[2, columns].copy()
        assert np.linalg.norm(gradient) > .005
        directions.append(([names.index(name) for name in joints], -gradient / np.linalg.norm(gradient)))
    return directions


def play(case, levels, dt_ms, output, kick_rad):
    env = build_env()
    env.reset()
    physics, task = env.physics, env.task
    spec = env.action_spec()
    names, spring = leg_domains(env, spec)
    landing = case.startswith('landing')
    # For this isolated landing stage the fly enters already just above the
    # floor, with folded wings and a gentle descent. No post-reset pose writes.
    if landing:
        set_ground_pose(env, lift_cm=.16, vertical_speed=-.2,
                        leg_positions=np.asarray(task._leg_springrefs))
        levels = levels[700:]
    else:
        set_ground_pose(env)
    ground = physics.model.name2id('groundplane', 'geom')
    root_vel = physics.bind(task._root_joint)
    initial = physics.data.qpos.copy()
    stride = round(env.control_timestep() / (dt_ms * .001))
    substeps = round(env.control_timestep() / physics.timestep())
    columns = (2, 3) if landing else (0, 1)
    traces, poses = [], []
    kicks = None
    landed = np.zeros(2)
    for tick, vnc in enumerate(levels[::stride, columns]):
        raw = np.zeros(spec.shape)
        if landing:
            landed += (1 - np.exp(-env.control_timestep() / .003)) * (np.clip(vnc * 1.8, 0., 1.) - landed)
            for idx, (name, angle) in spring.items():
                side = SIDES.index(name.rsplit('_', 1)[-1])
                raw[idx] = angle * (1. - landed[side])
        else:
            if tick == round(.078 / env.control_timestep()):
                kicks = kick_direction(physics, names)
            if kicks is not None:
                for level, (inds, direction) in zip(vnc, kicks, strict=True):
                    raw[inds] = kick_rad * np.clip(level, 0., 1.) * direction
        action = np.clip(raw, spec.minimum, spec.maximum)
        task._walker.apply_action(physics, action, env.random_state)
        for _ in range(substeps):
            physics.step()
            feet, nonfeet, _ = ground_support(physics, ground)
            position = physics.named.data.subtree_com['walker/'].copy()
            velocity = root_vel.qvel[:3]
            external = max(np.abs(physics.data.xfrc_applied).max(), np.abs(physics.data.qfrc_applied).max())
            traces.append([physics.time(), *position, len(feet), nonfeet,
                           np.linalg.norm(velocity), velocity[2], external, *vnc])
        poses.append(physics.data.qpos.copy())
    trace = np.asarray(traces)
    assert np.isfinite(trace).all() and not np.any(trace[:, 8])
    output.mkdir()
    for name, array in [('TRACE.npy', trace), ('QPOS.npy', poses), ('INITIAL.npy', initial)]:
        np.save(output / name, array, allow_pickle=False)
    first = np.flatnonzero((trace[:, 4] + trace[:, 5]) > 0)
    first_contact = None if not len(first) else float(trace[first[0], 0] * 1000)
    final = trace[-400:]
    unsupported = ((trace[:, 4] + trace[:, 5]) == 0) & (trace[:, 0] >= .08)
    result = {
        'case': case, 'columns': COLUMNS, 'native_steps': len(trace),
        'first_any_floor_contact_ms': first_contact,
        'first_floor_contact_feet': int(trace[first[0], 4]) if len(first) else None,
        'nonfoot_contact_samples': int(np.count_nonzero(trace[:, 5])),
        'last_20ms_min_feet': int(final[:, 4].min()),
        'last_20ms_max_speed_mm_s': float(10 * final[:, 6].max()),
        'peak_height_after_80ms_mm': float(10 * trace[trace[:, 0] >= .08, 3].max()),
        'peak_vertical_speed_after_80ms_mm_s': float(10 * trace[trace[:, 0] >= .08, 7].max()),
        'unsupported_native_steps_after_80ms': int(unsupported.sum()),
        'kick_rad': kick_rad if kicks is not None else None,
        'files': {name: digest(output / name) for name in ('TRACE.npy', 'QPOS.npy', 'INITIAL.npy')},
        'reset_only': True,
        'wing_actuation': 'none; folded spring positions at reset',
    }
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--kick-rad', type=float, default=.6)
    args = parser.parse_args()
    names, commands, facts = flight_input(args.neck, slice(4, 8))
    args.output.mkdir()
    for case in CASES:
        play(case, commands[:, names.index(case)], facts['dt_ms'], args.output / case, args.kick_rad)
    (args.output / 'IDENTITY.json').write_text(json.dumps({
        'post_neck_sha256': facts['data_sha256'],
        'neural_ticks_offset': {'takeoff': 0, 'landing': 700},
        'purpose': 'isolated middle leg thrust and final approach deployment, not sustained powered takeoff or full descent',
    }, indent=2) + '\n')
