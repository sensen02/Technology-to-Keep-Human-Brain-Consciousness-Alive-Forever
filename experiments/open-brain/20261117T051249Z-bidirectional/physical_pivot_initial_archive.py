"""Connect the same-donor circuit to FlyGym ground and FlyBody airborne fixtures."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from build_catalog import HERE, hash_file, read, record
from circuit import Circuit

PROJECT = HERE.parents[2]
sys.path.append(str(PROJECT / 'venv_body/lib/python3.12/site-packages'))
import mujoco

FLYBODY = PROJECT.parent / 'flybody_src'
FLIGHT_XML = FLYBODY / 'flybody/fruitfly/assets/fruitfly.xml'
DT = 0.0001


class Receiver:
    """Fixed engineering readout of physically post-neck BANC target membrane states."""

    def __init__(self, circuit):
        self.circuit = circuit
        groups = {}
        owners = {}
        gains = {}
        for root in circuit.brain_roots:
            port = circuit.ports[root]
            name = (port['group'], port['side'])
            group = groups.setdefault(name, {})
            gains[name] = port['sign']
            for post, count in port['outgoing']:
                group[post] = max(count, group.get(post, 0))
                owners.setdefault(post, set()).add(name)
        self.probes = {}
        self.filtered = {}
        for name, cells in groups.items():
            exclusive = ((root, count) for root, count in cells.items()
                         if owners[root] == {name} and root not in circuit.allowed_drive_roots)
            roots = [root for root, _ in sorted(exclusive, key=lambda item: (-item[1], item[0]))[:4]]
            assert roots, name
            self.probes[name] = roots, [circuit.index[root] for root in roots], gains[name]
            self.filtered[name] = 0.0

    def update(self, voltage, interval):
        for name, (_, targets, sign) in self.probes.items():
            level = float(np.mean((voltage[0, targets] + 52) * sign))
            old = self.filtered[name]
            self.filtered[name] = old + min(1., interval / .015) * (level - old)

    def level(self, group, side):
        return self.filtered.get((group, side), 0.0)

    def pair(self, group):
        return (self.level(group, 'left') + self.level(group, 'right')) / 2

    def ground(self):
        drive = np.clip(self.pair('forward') / 2.0, 0, 1)
        speed = .47 + .53 * np.clip(self.pair('forward_speed') / .9, 0, 1)
        advance = drive * speed
        reverse = np.clip(self.pair('backward_moonwalker') / .8, 0, 1)
        if reverse > .10:
            advance = -.87 * reverse
        brake = max(self.pair(name) for name in ('halt_bluebell', 'halt_web_a', 'halt_web_b'))
        reduction = 1 - np.clip(brake / 1.2, 0, 1)
        yaw = 0.
        for group, factor in (('walking_steer', .28), ('walking_steer_context', .12), ('walking_outer_stride', .09)):
            yaw += factor * np.clip((self.level(group, 'left') - self.level(group, 'right')) / 2, -1, 1)
        command = np.clip(reduction * np.array([advance - yaw, advance + yaw]), -1, 1)
        return np.where(np.abs(command) >= .035, command, 0.)

    def wings(self):
        power = np.clip(self.pair('wing_amplitude') / .35, 0, 1)
        yaw = .32 * np.clip((self.level('flight_steer', 'left') - self.level('flight_steer', 'right')) / 1.4, -1, 1)
        inhibit = np.clip(self.pair('landing') / .45, 0, .95)
        return np.clip(power * (1 - inhibit) * np.array([1 - yaw, 1 + yaw]), 0., 1.35)

    def takeoff_drive(self):
        return np.clip(self.pair('takeoff') / .8, 0., 1.)

    def manifest(self):
        return {f'{group}/{side}': roots for (group, side), (roots, _, _) in self.probes.items()}


def ground_schedule(case):
    forward = ('forward', None, 2300, 13500)
    if case in ('ground_forward', 'ground_cut', 'ground_taste'):
        return [forward]
    if case == 'ground_speed':
        return [forward, ('forward_speed', None, 2300, 13500)]
    if case == 'ground_halt':
        return [forward, ('halt_web_a', None, 8300, 13500)]
    if case == 'ground_back':
        return [('backward_moonwalker', None, 2300, 13500)]
    if case in ('ground_left', 'ground_right'):
        return [forward, ('walking_steer', case.split('_')[1], 4900, 12000)]
    if case in ('ground_pivot_left', 'ground_pivot_right'):
        return [('walking_steer', case.rsplit('_', 1)[1], 4900, 12000)]
    raise ValueError(case)


def flight_schedule(case):
    flight = [('wing_amplitude', None, 40, 650)]
    if case == 'wing_turn':
        flight.append(('flight_steer', 'left', 120, 650))
    elif case == 'wing_landing':
        flight.append(('landing', None, 320, 650))
    elif case == 'wing_takeoff':
        flight.append(('takeoff', None, 40, 250))
    return flight


def schedule_indices(circuit, schedule):
    output = []
    for group, side, first, last in schedule:
        roots = [root for root, spec in circuit.ports.items()
                 if spec['group'] == group and (side is None or spec['side'] == side)]
        assert roots
        output.append((circuit.validate_inputs(roots, 'brain'), first, last))
    return output


def load_receptors(circuit):
    from collections import defaultdict
    roots = defaultdict(set)
    for root, port in circuit.ports.items():
        group = port['group'].removesuffix('_alt')
        if port['direction'] == 'AN':
            roots[group, port['side']].update(pre for pre, _ in port['incoming_receptors'])
        elif port['direction'] == 'SA':
            roots[group, port['side']].add(root)
    return {key: circuit.validate_inputs(group, 'body') for key, group in roots.items()}


def run_ground(circuit, receiver, case):
    from flygym.anatomy import BodySegment
    from flygym.compose import ActuatorType
    from flygym.compose.world import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain.common import make_locomotion_fly
    from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
    from flygym_demo.complex_terrain.turning_controller import HybridTurningController

    fly = make_locomotion_fly(add_adhesion=False)
    world = FlatGroundWorld()
    world.add_fly(fly, (0, 0, .5), Rotation3D('quat', (1, 0, 0, 0)))
    sim = Simulation(world, timestep=DT)
    dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    motor = HybridTurningController(timestep=DT, enable_adhesion=False, output_dof_order=dofs)
    motor.reset(seed=17)
    head = sim._internal_bodyids_by_fly[fly.name][fly.get_bodysegs_order().index(BodySegment('c_thorax'))]
    legs = fly.get_legs_order()
    fronts = [legs.index(side + 'f') for side in ('l', 'r')]
    knees = [next(index for index, joint in enumerate(fly.get_jointdofs_order())
                  if joint.child.name == side + 'f_tibia' and joint.axis.value == 'pitch')
             for side in ('l', 'r')]
    mj = sim.mj_data
    drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
    cuts = np.array([case == 'ground_cut' and root in circuit.brain_roots for root in circuit.port_roots])
    schedule = schedule_indices(circuit, ground_schedule(case))
    receptors = load_receptors(circuit)
    trace = []
    poses = []
    arrivals = np.zeros(len(circuit.ports), dtype=np.int64)
    contact = np.zeros(2)
    knee_angle = np.zeros(2)
    knee_speed = np.zeros(2)
    acceleration = np.zeros(2)
    for tick in range(17000):
        drive.fill(0)
        if tick % 60 == 0:
            for indices, first, last in schedule:
                if first <= tick < last:
                    drive[0, indices] = 68.75
        if tick % 23 == 0:
            for index, side in enumerate(('left', 'right')):
                # The taste fixture marks the whole floor as a uniform tastant.
                sensors = {'front_touch': contact[index] > .04,
                           'front_load': contact[index] > .20,
                           'joint_position': abs(knee_angle[index]) > .20,
                           'joint_movement': abs(knee_speed[index]) > 1.0,
                           'leg_vibration': abs(acceleration[index]) > 200.,
                           'front_taste': case == 'ground_taste' and contact[index] > .04}
                for group, active in sensors.items():
                    if active:
                        drive[0, receptors.get((group, side), [])] = 68.75
        _, received, _, voltage = circuit.step(drive, cuts)
        arrivals += received[0]
        if tick % 5 == 0:
            receiver.update(voltage, DT * 5)
            command = receiver.ground()
            contact = np.linalg.norm(sim.get_ground_contact_info(fly.name)[1][fronts], axis=1)
            knee_angle = sim.get_joint_angles(fly.name)[knees]
            current_speed = sim.get_joint_velocities(fly.name)[knees]
            acceleration = (current_speed - knee_speed) / (DT * 5)
            knee_speed = current_speed
        obs = HybridControllerObservation.from_sim(sim, fly.name, legs=motor.legs)
        action = motor.step(command, obs)
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, action.joint_angles)
        sim.step()
        if (tick + 1) % 10 == 0:
            yaw = np.arctan2(mj.xmat[head, 3], mj.xmat[head, 0])
            trace.append([mj.time, *mj.xpos[head], yaw, *command, *contact, *knee_angle, *knee_speed, *acceleration])
            poses.append(mj.qpos.copy())
    values = np.asarray(trace)
    assert np.isfinite(values).all() and not mj.warning.number.any()
    return values, arrivals, np.asarray(poses), {'source': 'FlyGym 2.1.0 HybridTurningController', 'steps': 17000,
                             'column_order': ['time', 'thorax_x_mm', 'thorax_y_mm', 'thorax_z_mm', 'yaw_rad',
                                              'left_drive', 'right_drive', 'left_front_force', 'right_front_force',
                                              'left_knee_rad', 'right_knee_rad', 'left_knee_rad_s', 'right_knee_rad_s',
                                              'left_knee_rad_s2', 'right_knee_rad_s2']}


def run_wings(circuit, receiver, case):
    sys.path.append(str(FLYBODY))
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator

    model = mujoco.MjModel.from_xml_path(str(FLIGHT_XML))
    data = mujoco.MjData(model)
    assert np.isclose(model.opt.timestep, DT) and model.opt.density > 0
    model.opt.density *= 0 if case == 'wing_no_air' else 1
    model.opt.viscosity *= 0 if case == 'wing_no_air' else 1
    wing_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'wing_{axis}_{side}')
                for side in ('left', 'right') for axis in ('yaw', 'roll', 'pitch')]
    joint_ids = model.actuator_trnid[wing_ids, 0]
    wing_pos = model.jnt_qposadr[joint_ids]
    halteres = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'haltere_{side}') for side in ('left', 'right')]
    haltere_dofs = model.jnt_dofadr[halteres]
    jumps = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'tibia_T2_{side}') for side in ('left', 'right')]
    assert min(*halteres, *jumps) >= 0
    assert min(wing_ids) >= 0 and len(set(wing_ids)) == 6
    pattern = WingBeatPatternGenerator(dt_ctrl=DT)
    pattern.reset(initial_phase=0.)
    data.qpos[2] += 5.0  # Initial airborne fixture in the upstream model's cm units.
    mujoco.mj_forward(model, data)
    initial = data.qpos[wing_pos].copy()
    drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
    cuts = np.array([case == 'wing_cut' and root in circuit.brain_roots for root in circuit.port_roots])
    schedule = schedule_indices(circuit, flight_schedule(case))
    trace = []
    poses = []
    arrivals = np.zeros(len(circuit.ports), dtype=np.int64)
    receptors = load_receptors(circuit)
    lift_torques = np.zeros(2)
    haltere_speed = np.zeros(2)
    for tick in range(700):
        drive.fill(0)
        if tick % 35 == 0:
            for indices, first, last in schedule:
                if first <= tick < last:
                    drive[0, indices] = 68.75
        if tick % 23 == 0:
            for index, side in enumerate(('left', 'right')):
                if lift_torques[index] > .05:
                    drive[0, receptors['wing_strain', side]] = 68.75
                if abs(haltere_speed[index]) > .02:
                    drive[0, receptors['haltere_strain', side]] = 68.75
        _, received, _, voltage = circuit.step(drive, cuts)
        arrivals += received[0]
        receiver.update(voltage, DT)
        amplitude = receiver.wings()
        angles = pattern.step(pattern.base_beat_freq)
        data.ctrl[wing_ids] = initial + np.repeat(amplitude, 3) * (angles - initial)
        extension = float(receiver.takeoff_drive())
        data.ctrl[jumps] = extension * model.actuator_ctrlrange[jumps, 1]
        mujoco.mj_step(model, data)
        lift_torques = np.linalg.norm(data.actuator_force[wing_ids].reshape(2, 3), axis=1)
        haltere_speed = data.qvel[haltere_dofs]
        trace.append([data.time, *data.qpos[:3], *amplitude, *data.qpos[wing_pos], *data.actuator_force[wing_ids],
                      extension, *lift_torques, *haltere_speed])
        poses.append(data.qpos.copy())
    values = np.asarray(trace)
    assert np.isfinite(values).all() and not data.warning.number.any()
    return values, arrivals, np.asarray(poses), {'source': str(FLIGHT_XML), 'xml_sha256': hash_file(FLIGHT_XML),
                             'density': model.opt.density, 'viscosity': model.opt.viscosity, 'steps': 700,
                             'column_order': ['time','x_cm','y_cm','z_cm','left_amplitude','right_amplitude',
                                              'left_yaw','left_roll','left_pitch','right_yaw','right_roll','right_pitch',
                                              'left_yaw_force','left_roll_force','left_pitch_force',
                                              'right_yaw_force','right_roll_force','right_pitch_force',
                                              'midleg_extension','left_wing_torque','right_wing_torque',
                                              'left_haltere_rad_s','right_haltere_rad_s']}


def main(case, output):
    start = perf_counter()
    catalog = read(HERE / 'CATALOG.json')
    circuit = Circuit(catalog)
    receiver = Receiver(circuit)
    values, arrived, poses, info = (run_ground if case.startswith('ground') else run_wings)(circuit, receiver, case)
    output.mkdir()
    np.save(output / 'TRACE.npy', values, allow_pickle=False)
    np.save(output / 'QPOS.npy', poses, allow_pickle=False)
    record(output / 'RESULT.json', dict(info, case=case, trace_sha256=hash_file(output / 'TRACE.npy'),
           pose_sha256=hash_file(output / 'QPOS.npy'), implementation_sha256=hash_file(HERE / 'physical.py'),
           catalog_sha256=hash_file(HERE / 'CATALOG.json'), probes=receiver.manifest(),
           arrival_counts=dict(zip(circuit.port_roots, map(int, arrived), strict=True)),
           mujoco=mujoco.__version__, seconds=perf_counter()-start, epochs=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('ground_forward', 'ground_speed', 'ground_halt', 'ground_back',
                                       'ground_left', 'ground_right', 'ground_cut', 'ground_taste',
                                       'ground_pivot_left', 'ground_pivot_right', 'wing_power',
                                       'wing_cut', 'wing_turn', 'wing_landing', 'wing_takeoff', 'wing_no_air'))
    parser.add_argument('output', type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    main(arguments.case, arguments.output)
