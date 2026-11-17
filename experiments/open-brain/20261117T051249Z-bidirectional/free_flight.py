"""Run the published FlyBody flight controller with physically post-neck wing gating."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import type_spec_registry

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[3] / 'flybody_src'
sys.path.insert(0, str(SRC))
from flybody.fly_envs import flight_imitation
from flybody.tasks.trajectory_loaders import constant_speed_trajectory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path):
    # Saved with the historical TFP name. The modern class has the same
    # Independent(Normal) serialization; the network graph remains unchanged.
    distribution = tfp.distributions.Independent(tfp.distributions.Normal([0.], [1.]), reinterpreted_batch_ndims=1)
    registry_name = 'tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec'
    type_spec_registry._NAME_TO_TYPE_SPEC.setdefault(registry_name, type(distribution._type_spec))
    return tf.saved_model.load(str(path))


def make_env():
    source = HERE / 'flight_reference'
    env = flight_imitation(wpg_pattern_path=str(source / 'wing_pattern_fmech.npy'),
                           random_state=np.random.RandomState(17), terminal_com_dist=float('inf'))
    ref_qpos, ref_qvel = constant_speed_trajectory(n_steps=1500, speed=20, init_pos=(0, 0, 1),
                     body_rot_angle_y=-47.5, control_timestep=env.control_timestep())
    env.task._traj_generator.set_next_trajectory(ref_qpos, ref_qvel)
    return env


def run(case, output, gate_dir):
    ref = HERE / 'flight_reference'
    policy = load_policy(ref / 'flight')
    env = make_env()
    step = env.reset()
    phy = env.physics
    gate_file = gate_dir / f'{case}.npy'
    info = json.loads((gate_dir / 'IDENTITY.json').read_text())
    assert sha(gate_file) == info['conditions'][case]['sha256']
    gate = np.load(gate_file, allow_pickle=False)
    wing_actuators = [i for i in range(phy.model.nu) if 'wing_' in (phy.model.id2name(i, 'actuator') or '')]
    assert len(wing_actuators) == 6
    ground = phy.model.name2id('groundplane', 'geom')
    assert phy.model.geom_bodyid[ground] == 0
    world_geoms = {ground}
    nominal_gain = phy.model.actuator_gainprm[wing_actuators].copy()
    spec = env.action_spec()
    samples = []
    poses = []
    native_step = phy.timestep()
    substeps = round(env.control_timestep()/native_step)
    for index in range(len(gate)//2):
        enable = gate[2 * index, 2]
        phy.model.actuator_gainprm[wing_actuators] = nominal_gain * enable
        obs = {name: tf.convert_to_tensor(np.asarray(value, dtype=np.float32)[None])
               for name, value in step.observation.items()}
        canonical = policy(obs).mean()[0].numpy()
        assert canonical.shape == spec.shape and np.isfinite(canonical).all()
        canonical = np.clip(canonical, -1., 1.)
        action = spec.minimum + (spec.maximum-spec.minimum) * (canonical+1) / 2
        step = env.step(action)
        body_pos = phy.named.data.subtree_com['walker/'].copy()
        force = np.sum(np.abs(phy.data.actuator_force[wing_actuators]))
        contact_count = sum(int(c.geom1) in world_geoms or int(c.geom2) in world_geoms
                            for c in phy.data.contact[:phy.data.ncon])
        samples.append([phy.time(), *body_pos, enable, force, contact_count, step.last()])
        poses.append(phy.data.qpos.copy())
        if step.last():
            break
    trace = np.asarray(samples)
    assert np.isfinite(trace).all() and len(trace)
    output.mkdir()
    np.save(output/'TRACE.npy', trace, allow_pickle=False)
    np.save(output/'QPOS.npy', np.asarray(poses), allow_pickle=False)
    details = {'case': case, 'steps': len(trace)*substeps, 'simulated_time': float(trace[-1,0]),
              'final_com': trace[-1,1:4].tolist(), 'displacement': (trace[-1,1:4]-trace[0,1:4]).tolist(),
              'min_height_cm': float(trace[:,3].min()), 'final_height_cm': float(trace[-1,3]),
              'ground_contact_steps': int(np.count_nonzero(trace[:,6])),
              'ground_contact_field': 'sampled contacts involving a collidable geom on world body',
              'world_collision_geoms': [phy.model.id2name(i, 'geom') for i in sorted(world_geoms)],
              'disabled_wing_force_max': float(trace[trace[:,4]==0,5].max(initial=0)),
              'normal_wing_force_max': float(trace[trace[:,4]==1,5].max(initial=0)),
              'controller_source': 'Janelia published frozen FlyBody flight imitation policy and fmech wing trace',
              'network_file_sha256': sha(ref/'flight'/'saved_model.pb'),
              'wing_pattern_sha256': sha(ref/'wing_pattern_fmech.npy'),
              'gate_sha256': sha(gate_file), 'epochs': 0,
              'trace_sha256': sha(output/'TRACE.npy'), 'pose_sha256': sha(output/'QPOS.npy'),
              'terminal_flag': bool(step.last())}
    (output/'RESULT.json').write_text(json.dumps(details, indent=2)+'\n')
    print(json.dumps(details, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('intact','cut','cut_restore'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--gate', type=Path, default=HERE/'FLIGHT_GATE')
    args = parser.parse_args()
    run(args.case, args.out, args.gate)
