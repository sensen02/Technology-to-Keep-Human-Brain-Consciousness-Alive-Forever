"""Flight-to-food fixture using a solid fruit, visual landing gate and stock mechanics."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
LANDING = HERE.parent / '20261118T083045Z-complete-landing'
sys.path.insert(0, str(LANDING))
import land
from dm_control.mujoco import wrapper

TOP_CM = .30
RADIUS_CM = .89
CONTACT_TOL_CM = .025
DURATION_S = 1.42
VISION_S = .005
VISUAL_WIDTH = 49
STABLE_S = .070
MOUTH_REACH_CM = .14
EXTRA = ('floor_feet', 'fruit_feet', 'fruit_side_feet', 'fruit_nonfeet',
         'mouth_x_cm', 'mouth_y_cm', 'mouth_z_cm', 'mouth_gap_cm', 'perch_ready',
         'taste_contact')
COLUMNS = (*land.COLUMNS, *EXTRA)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def build_scene(x, y=0.):
    env = land.build_env()
    arena = env.task._arena.mjcf_model
    arena.worldbody.add('geom', name='fruit', type='cylinder',
                        size=(RADIUS_CM, TOP_CM / 2), pos=(x, y, TOP_CM / 2),
                        rgba=(.87, .075, .036, 1), friction=(1.05, .005, .0001),
                        contype=1, conaffinity=1)
    arena.worldbody.add('camera', name='fruit_sensor', fovy=50)
    return env


def support(phy, geom, surface=None):
    top, sides, nonfeet, bodies = set(), set(), 0, set()
    for contact in phy.data.contact[:phy.data.ncon]:
        if geom not in (int(contact.geom1), int(contact.geom2)):
            continue
        other = int(contact.geom2 if int(contact.geom1) == geom else contact.geom1)
        body = phy.model.id2name(phy.model.geom_bodyid[other], 'body')
        if not body.startswith('walker/'):
            continue
        bodies.add(body)
        if 'claw' not in body and 'tarsus' not in body:
            nonfeet += 1
        else:
            flags = top if surface is None or abs(contact.pos[2] - surface) <= CONTACT_TOL_CM else sides
            flags.add(tuple(body.rsplit('_', 2)[-2:]))
    return top, sides, nonfeet, bodies


class FruitEye:
    """A stabilized, heading-relative camera. The trigger sees pixels only."""

    def __init__(self, phy, head, blank):
        self.phy = phy
        self.head = head
        self.camera = phy.model.name2id('fruit_sensor', 'camera')
        self.opt = wrapper.MjvOption()
        self.opt.geomgroup[1:] = 0
        self.blank = blank
        self.samples = []
        self.candidate_s = None
        self.previous = False
        self.last_frame = None

    def sample(self, now, q):
        if self.candidate_s is not None:
            return
        yaw = land.angles(q)[0]
        f = np.array([np.cos(yaw), np.sin(yaw), 0.])
        right = np.array([f[1], -f[0], 0.])
        up = np.array([0., 0., 1.])
        angle = np.deg2rad(12.)
        vertical = np.sin(angle) * f + np.cos(angle) * up
        backward = -np.cos(angle) * f + np.sin(angle) * up
        self.phy.data.cam_xpos[self.camera] = self.phy.data.xpos[self.head] + .08 * f + .055 * up
        self.phy.data.cam_xmat[self.camera] = np.column_stack((right, vertical, backward)).ravel()
        rgb = self.phy.render(width=96, height=64, camera_id=self.camera, scene_option=self.opt)
        if self.blank:
            rgb = np.zeros_like(rgb)
        image = rgb.astype(np.int16)
        red = (image[..., 0] >= 75) & (image[..., 0] - image[..., 1] >= 32) & (image[..., 0] - image[..., 2] >= 32)
        yy, xx = np.nonzero(red)
        width = int(xx.max() - xx.min() + 1) if len(xx) else 0
        midpoint = float(xx.mean()) if len(xx) else -1.
        eligible = width >= VISUAL_WIDTH and len(xx) >= 45 and 30 <= midpoint <= 66
        self.samples.append([now, width, len(xx), midpoint, int(eligible)])
        self.last_frame = rgb
        if self.previous and eligible:
            self.candidate_s = round(now, 8)
        self.previous = eligible


def simulate(policy, stock, observation, commands, dt_ms, seed, x, y, duration,
             descent_delay, blank, vision=True):
    launch = land.launch_settings(seed, duration)
    prior = land.Approach(**json.loads((LANDING / 'CALIBRATED.json').read_text()))
    approach = replace(prior, final_height_cm=prior.final_height_cm + TOP_CM,
                       deploy_cm=prior.deploy_cm + TOP_CM, descent_delay_s=descent_delay,
                       fold_precontact_cm=prior.fold_precontact_cm + TOP_CM,
                       settled_height_cm=prior.settled_height_cm + TOP_CM)
    env = build_scene(x, y)
    env._time_limit = duration + .001
    task = env.task
    env.random_state.seed(seed)
    setup = task.initialize_episode

    def initialize(phy, random):
        setup(phy, random)
        land.set_ground_pose(env)
        task._ref_qpos[:, :3] = task._walker.get_pose(phy)[0]
        task._ref_qpos[:, 3:7] = (1., 0., 0., 0.)
        task._ref_qvel[:] = 0.

    task.initialize_episode = initialize
    task.check_termination = lambda phy: False
    step = env.reset()
    phy = env.physics
    assert phy.model.opt.density > 0 and phy.model.opt.viscosity > 0
    dt = env.control_timestep()
    stride = round(1000 * dt / dt_ms)
    extra = max(0, round(duration / dt) + task._future_steps + 2 - len(task._ref_qpos))
    task._ref_qpos = np.pad(task._ref_qpos, ((0, extra), (0, 0)), mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel, ((0, extra), (0, 0)), mode='edge')
    adapter = land.LeggedFlightTeacher(stock, observation, env)
    motor = land.LandingMotor(task, phy.bind(task._wing_joints).qpos.copy(), adapter,
                              dt, launch, approach, phy)
    floor = phy.model.name2id('groundplane', 'geom')
    fruit = phy.model.name2id('fruit', 'geom')
    assert phy.model.geom_contype[fruit] & 1 and phy.model.geom_conaffinity[fruit] & 1
    assert phy.model.body_contype[0] & 1 and phy.model.body_conaffinity[0] & 1
    head = phy.model.name2id('walker/head', 'body')
    mouth_geoms = [phy.model.name2id(f'walker/labrum_{side}_lower_collision', 'geom') for side in ('left', 'right')]
    assert head >= 0 and min(mouth_geoms) >= 0
    wing = [idx for idx in range(phy.model.nu)
            if 'wing_' in (phy.model.id2name(idx, 'actuator') or '')]
    binding = phy.bind(task._wing_joints)
    root = phy.bind(task._root_joint)
    eye = FruitEye(phy, head, blank)
    initial = np.r_[phy.data.qpos, phy.data.qvel]
    traces, poses, actions = [], [], []
    command = np.zeros(4)
    stable_age = 0.
    native = phy.timestep()
    old_sample = task.after_substep

    def sample(physics, random):
        nonlocal stable_age
        old_sample(physics, random)
        ground_feet, _, ground_hits, ground_bodies = support(physics, floor)
        held_feet, side_feet, fruit_hits, fruit_bodies = support(physics, fruit, TOP_CM)
        feet = ground_feet | held_feet | side_feet
        nonfeet = ground_hits + fruit_hits
        motor.touched_claws = tuple(body.removeprefix('walker/') for body in ground_bodies | fruit_bodies
                                    if body.startswith('walker/claw_'))
        com = physics.named.data.subtree_com['walker/']
        _, quat = task._walker.get_pose(physics)
        yaw, roll, pitch = land.angles(quat)
        external = max(np.abs(physics.data.qfrc_applied).max(), np.abs(physics.data.xfrc_applied).max())
        # The compact feeding action represents short proboscis extension from the real labrum positions.
        mouth = physics.data.geom_xpos[mouth_geoms].mean(axis=0)
        gap = mouth[2] - TOP_CM
        on_patch = np.hypot(mouth[0] - x, mouth[1] - y) <= RADIUS_CM - .015
        settled = (len(held_feet) == 6 and not ground_feet and nonfeet == 0 and
                   motor.landing_phase == 4 and np.linalg.norm(root.qvel[:3]) < .24 and
                   max(abs(roll), abs(pitch)) < .48)
        stable_age = stable_age + native if settled else 0.
        ready = stable_age >= STABLE_S
        taste = ready and on_patch and -.002 <= gap <= MOUTH_REACH_CM
        ref_idx = task._step_counter
        traces.append([physics.time(), *com, *root.qvel[:3], yaw, roll, pitch, len(feet),
                       nonfeet, external, np.abs(physics.data.actuator_force[wing]).sum(),
                       *binding.qpos[[0, 3]], *command[:2], motor.wing_mix, motor.tuck,
                       motor.teacher_mix, motor.phase, *command[2:], motor.landing_phase,
                       task._ref_qpos[ref_idx, 2], task._ref_qvel[ref_idx, 0],
                       len(ground_feet), len(held_feet), len(side_feet), fruit_hits,
                       *mouth, gap, ready, taste])

    task.after_substep = sample
    for tick in range(round(duration / dt)):
        now = phy.time()
        if vision and motor.engaged_air and .22 <= now <= .55 and tick % round(VISION_S / dt) == 0:
            eye.sample(now, task._walker.get_pose(phy)[1])
        command = commands[tick * stride] if tick * stride < len(commands) else np.zeros(4)
        feet, nonfeet = traces[-1][10:12] if traces else (6, 0)
        motor.update_episode(command, now, feet, nonfeet, phy)
        action = motor.action(policy, step.observation)
        step = env.step(action)
        assert abs(phy.time() - (tick + 1) * dt) < 1e-8 and not step.last()
        poses.append(phy.data.qpos.copy())
        actions.append(action)
    return locals()


def save(out, run):
    out.mkdir()
    data = np.array(run['traces'])
    for name, array in {'TRACE': data, 'QPOS': run['poses'], 'ACTION': run['actions'],
                        'INITIAL': run['initial'], 'EYE': run['eye'].samples}.items():
        array = np.asarray(array)
        assert np.isfinite(array).all()
        np.save(out / f'{name}.npy', array, allow_pickle=False)
    if run['eye'].last_frame is not None:
        from PIL import Image
        Image.fromarray(run['eye'].last_frame).save(out / 'EYE_LAST.png')
    z = dict(zip(COLUMNS, data.T, strict=True))

    def first(mask):
        hits = z['time_s'][mask]
        return float(hits[0]) if len(hits) else None

    result = dict(columns=COLUMNS, eye_columns=['time_s', 'red_width_px', 'red_pixels', 'red_mean_x', 'eligible'],
                  geometry={'x_cm': run['x'], 'y_cm': run['y'], 'radius_cm': RADIUS_CM, 'top_cm': TOP_CM},
                  mouth_reach_cm=MOUTH_REACH_CM, stable_threshold_s=STABLE_S,
                  native_dt_s=run['native'], seed=run['seed'], approach=asdict(run['approach']),
                  takeoff_s=run['motor'].started, visual_trigger_s=run['eye'].candidate_s,
                  landing_s=run['motor'].landing_started, fold_s=run['motor'].fold_started,
                  first_fruit_contact_s=first(z['fruit_feet'] > 0),
                  first_all_six_s=first(z['fruit_feet'] == 6),
                  first_perch_s=first(z['perch_ready'] > 0), first_taste_s=first(z['taste_contact'] > 0),
                  perch_total_s=float(z['perch_ready'].sum() * run['native']),
                  taste_total_s=float(z['taste_contact'].sum() * run['native']),
                  fruit_nonfoot_samples=int(np.count_nonzero(z['fruit_nonfeet'])),
                  late_floor_samples=int(np.count_nonzero((z['time_s'] > .2) & (z['floor_feet'] > 0))),
                  final=z['x_cm'][-1:].tolist() + z['z_cm'][-1:].tolist(),
                  final_mouth_gap_cm=float(z['mouth_gap_cm'][-1]), end_fruit_feet=int(z['fruit_feet'][-1]),
                  external_force_max=float(z['external_applied'].max()),
                  files={p.name: land.digest(p) for p in sorted(out.glob('*'))})
    write(out / 'RESULT.json', result)
    print({k: v for k, v in result.items() if k not in ('columns', 'eye_columns', 'files', 'approach')}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--neck', type=Path, default=LANDING / 'NECK')
    parser.add_argument('--case', choices=land.CASES, default='intact')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--x', type=float, default=8.03)
    parser.add_argument('--y', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=DURATION_S)
    parser.add_argument('--descent-delay', type=float, default=.073)
    parser.add_argument('--blank', action='store_true')
    parser.add_argument('--no-vision', action='store_true')
    args = parser.parse_args()
    receipt, commands = land.commands_from(args.neck)
    policy, stock, observation = land.setup_policy()
    run = simulate(policy, stock, observation, commands[args.case], receipt['dt_ms'], args.seed,
                   args.x, args.y, args.duration, args.descent_delay, args.blank, not args.no_vision)
    save(args.output, run)
    write(args.output / 'INPUT.json', {'neck': str(args.neck.resolve()), 'case': args.case,
          'neck_data_sha256': receipt['sha256'], 'visual_mask': args.blank,
          'flight_policy_sha256': land.digest(land.BASELINE / 'flight_reference/flight/saved_model.pb'),
          'source_sha256': land.digest(Path(__file__))})


if __name__ == '__main__':
    main()
