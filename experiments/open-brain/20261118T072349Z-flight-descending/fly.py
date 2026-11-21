"""Fly the frozen FlyBody controller with independently measured VNC inputs."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import scipy

from flight_manifest import CASES, COLUMNS, LABELS, ROOT, digest


# The flight runtime is separate from the neural runtime; no neural kernel is
# loaded inside the body process.
BASELINE = ROOT / '20261117T051249Z-bidirectional'
sys.path.insert(0, str(BASELINE))
from free_flight import load_policy, make_env, tf

STROKE = [0, 3]


def flight_input(neck_dir, channels=slice(0, 4)):
    path = neck_dir / 'POST_NECK.npz'
    facts = json.loads((neck_dir / 'IDENTITY.json').read_text())
    if digest(path) != facts['data_sha256']:
        raise ValueError('Post-neck trace does not match its receipt')
    with np.load(path, allow_pickle=False) as data:
        if tuple(data['labels']) != LABELS:
            raise ValueError('Unexpected post-neck channel ordering')
        cases = tuple(data['conditions'])
        commands = data['levels'][:, :, channels]
    return cases, commands, facts


def yaw_rotation(yaw):
    cos, sin = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cos, 0., 0., sin])


def angles(quaternion):
    w, x, y, z = quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1., 1.))
    return yaw, roll, pitch


def enable_floor_contacts(env):
    # Apply this to MJCF before reset compiles it. A runtime geom mask edit
    # would leave MuJoCo's compiled broad-phase body masks disabled.
    for geom in env.task._arena.ground_geoms:
        geom.contype = 1
        geom.conaffinity = 1


class WingReadout:
    """Convert VNC levels to wing motion, keeping the frozen policy in the loop."""

    def __init__(self, task, center, yaw_rate, amplitude_gain):
        self.task = task
        self.pattern = task._wbpg.step
        self.center = center
        self.yaw_rate = yaw_rate
        self.amplitude_gain = amplitude_gain
        self.levels = np.zeros(4)
        self.targets = np.zeros(6)
        self.heading = 0.
        self.steering_started = False
        self.reference = task._ref_qpos.copy()
        self.reference_vel = task._ref_qvel.copy()
        self.anchor = None
        task._wbpg.step = self.step

    def update(self, tick, levels, dt):
        self.levels = levels
        steering = levels[2] - levels[3]
        if not self.steering_started:
            if steering == 0:
                return
            self.steering_started = True
            self.anchor = self.reference[tick, :2].copy()
        # The frozen low-level teacher already tracks position and orientation.
        # Change only the ghost reference at the policy interface. Hold a causal
        # estimate through the preview window using the current VNC observation.
        yaw_rate = self.yaw_rate * steering
        self.heading += yaw_rate * dt
        start = self.anchor.copy()
        for ahead in range(self.task._future_steps + 2):
            index = tick + ahead
            heading = self.heading + yaw_rate * dt * ahead
            velocity = self.reference_vel[index, 0] * np.array([np.cos(heading), np.sin(heading)])
            self.task._ref_qpos[index, :2] = start
            self.task._ref_qpos[index, 3:7] = self.multiply(yaw_rotation(heading), self.reference[index, 3:7])
            self.task._ref_qvel[index, :2] = velocity
            self.task._ref_qvel[index, 5] = yaw_rate
            start = start + velocity * dt
            if ahead == 0:
                self.anchor = start
    @staticmethod
    def multiply(a, b):
        w, x, y, z = a
        s, i, j, k = b
        return np.array([w*s-x*i-y*j-z*k, w*i+x*s+y*k-z*j,
                         w*j-x*k+y*s+z*i, w*k+x*j-y*i+z*s])

    def step(self, ctrl_freq):
        self.targets = self.pattern(ctrl_freq).copy()
        amplitude = self.amplitude_gain * self.levels[:2]
        if np.any(amplitude):
            self.targets[STROKE] += amplitude * (self.targets[STROKE] - self.center)
        return self.targets


def run_one(policy, levels, dt_ms, out, seed, fluid, env=None, extension=None):
    if env is None:
        env = make_env()
    env.random_state.seed(seed)
    enable_floor_contacts(env)
    step = env.reset()
    phy = env.physics
    substeps = round(env.control_timestep() / phy.timestep())
    stride = round(env.control_timestep() / (dt_ms * .001))
    assert abs(env.control_timestep() - stride * dt_ms * .001) < 1e-12
    wing = [i for i in range(phy.model.nu) if 'wing_' in (phy.model.id2name(i, 'actuator') or '')]
    assert len(wing) == 6
    ground = phy.model.name2id('groundplane', 'geom')
    assert phy.model.geom_bodyid[ground] == 0
    assert (phy.model.geom_contype[ground], phy.model.geom_conaffinity[ground]) == (1, 1)
    assert phy.model.body_contype[0] & 1 and phy.model.body_conaffinity[0] & 1
    density_viscosity = [float(phy.model.opt.density), float(phy.model.opt.viscosity)]
    assert all(value > 0 for value in density_viscosity)
    if not fluid:
        phy.model.opt.density = 0.
        phy.model.opt.viscosity = 0.
    task = env.task
    pattern = np.load(BASELINE / 'flight_reference/wing_pattern_fmech.npy', allow_pickle=False)
    center = (pattern[:, 0].min() + pattern[:, 0].max()) / 2
    body = WingReadout(task, center, yaw_rate=4., amplitude_gain=.12)
    spec = env.action_spec()
    native = []
    prior_hook = task.after_substep

    def sample_native(physics, random_state):
        prior_hook(physics, random_state)
        count = sum(ground in (int(c.geom1), int(c.geom2))
                    for c in physics.data.contact[:physics.data.ncon])
        external = max(np.abs(physics.data.xfrc_applied).max(), np.abs(physics.data.qfrc_applied).max())
        native.append((count, external))

    task.after_substep = sample_native
    traces, poses = [], []
    initial = phy.data.qpos.copy()
    assert len(levels) % stride == 0
    for tick in range(len(levels) // stride):
        if extension is None:
            action = teacher_action(policy, step.observation, spec)
        else:
            action = extension.action(policy, step.observation)
        # Use the packet available at the control boundary. No interpolation
        # from future neural time samples.
        current = levels[tick * stride]
        body.update(tick, current, env.control_timestep())
        step = env.step(action)
        assert len(native) == (tick + 1) * substeps, 'Missing native callbacks'
        com = phy.named.data.subtree_com['walker/'].copy()
        _, quaternion = task._walker.get_pose(phy)
        actual = phy.bind(task._wing_joints).qpos[STROKE]
        contacts, external = np.max(native[-substeps:], axis=0)
        traces.append((phy.time(), *com, *angles(quaternion), contacts, *actual,
                       *body.targets[STROKE], *current, np.abs(phy.data.actuator_force[wing]).sum(),
                       external, body.heading))
        poses.append(phy.data.qpos.copy())
        if step.last():
            break
    trace = np.asarray(traces)
    assert np.isfinite(trace).all() and not np.any(trace[:, 17])
    out.mkdir()
    np.save(out / 'TRACE.npy', trace, allow_pickle=False)
    np.save(out / 'QPOS.npy', poses, allow_pickle=False)
    np.save(out / 'NATIVE.npy', native, allow_pickle=False)
    np.save(out / 'INITIAL.npy', initial, allow_pickle=False)
    (out / 'RESULT.json').write_text(json.dumps({
        'columns': COLUMNS, 'simulated_s': float(trace[-1, 0]), 'native_steps': len(native),
        'terminal': bool(step.last()), 'ground_collision_mask': [int(phy.model.geom_contype[ground]),
                                                                 int(phy.model.geom_conaffinity[ground])],
        'ground_contacts': int(np.count_nonzero(np.asarray(native)[:, 0])),
        'density_viscosity': [float(phy.model.opt.density), float(phy.model.opt.viscosity)],
        'max_external_applied': float(trace[:, 17].max()),
        'min_height_mm': float(trace[:, 3].min() * 10), 'final_y_mm': float(trace[-1, 2] * 10),
        'final_yaw_deg': float(np.rad2deg(trace[-1, 4])),
        'trace_sha256': digest(out / 'TRACE.npy'), 'pose_sha256': digest(out / 'QPOS.npy'),
        'native_sha256': digest(out / 'NATIVE.npy'),
        'initial_sha256': digest(out / 'INITIAL.npy'),
    }, indent=2) + '\n')


def teacher_action(policy, observation, spec):
    obs = {name: tf.convert_to_tensor(np.asarray(value, dtype=np.float32)[None])
           for name, value in observation.items()}
    canonical = policy(obs).mean()[0].numpy()
    assert canonical.shape == spec.shape and np.isfinite(canonical).all()
    return spec.minimum + (spec.maximum - spec.minimum) * (np.clip(canonical, -1., 1.) + 1) / 2


def run(args):
    names, levels, facts = flight_input(args.neck)
    policy_path = BASELINE / 'flight_reference/flight'
    policy = load_policy(policy_path)
    args.out.mkdir()
    for case in args.cases:
        run_one(policy, levels[:, names.index(case)], facts['dt_ms'],
                args.out / case, args.seed, not args.no_fluid)
    info = {
        'steering_yaw_rate_rad_s': 4., 'amplitude_gain': .12,
        'post_neck_sha256': facts['data_sha256'],
        'policy_files': {str(path.relative_to(policy_path)): digest(path)
                         for path in sorted(policy_path.rglob('*')) if path.is_file()},
        'wing_pattern_sha256': digest(BASELINE / 'flight_reference/wing_pattern_fmech.npy'),
        'software': {'mujoco': mujoco.__version__, 'tensorflow': tf.__version__,
                     'scipy': scipy.__version__, 'numpy': np.__version__, 'python': sys.version},
        'startup': 'Published frozen FlyBody policy mean with 1 cm air start',
        'seed': args.seed, 'fluid': not args.no_fluid,
        'training_updates': 0,
    }
    (args.out / 'IDENTITY.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('neck', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--no-fluid', action='store_true')
    run(parser.parse_args())
