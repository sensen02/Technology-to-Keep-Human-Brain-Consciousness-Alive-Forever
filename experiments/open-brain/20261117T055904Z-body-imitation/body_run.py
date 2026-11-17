"""Native FlyGym physical imitation with source-qualified neck events."""
from dataclasses import asdict
from hashlib import file_digest
import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from cases import CASES
from model import action, load_policy

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PARENT = HERE.parent / '20261117T051249Z-bidirectional'
sys.path.insert(0, str(PARENT))
from circuit import Circuit
from physical import Receiver

import mujoco
from flygym.anatomy import BodySegment
from flygym.compose import ActuatorType
from flygym.compose.world import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain.common import make_locomotion_fly
from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
from flygym_demo.complex_terrain.turning_controller import HybridTurningController

DT = .0001
HOLD = 10
TICKS = 17000
FLAGS = mujoco.mjtState.mjSTATE_INTEGRATION
GROUPS = ('forward', 'forward_speed', 'halt_web_a', 'front_touch', 'front_load')
MODEL_FILE = HERE / 'FROZEN_INIT.npz'


def sha(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


class Body:
    def __init__(self):
        self.fly = fly = make_locomotion_fly(add_adhesion=False)
        world = FlatGroundWorld()
        world.add_fly(fly, (0, 0, .5), Rotation3D('quat', (1, 0, 0, 0)))
        self.sim = sim = Simulation(world, timestep=DT)
        self.m, self.d = sim.mj_model, sim.mj_data
        self.name = fly.name
        self.dofs = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        asset_dofs = self.fly.get_jointdofs_order()
        self.joint_indices = [asset_dofs.index(dof) for dof in self.dofs]
        self.actuators = sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][self.name]
        limits = self.m.actuator_ctrlrange[self.actuators]
        limited = self.m.actuator_ctrllimited[self.actuators]
        self.minimum = np.where(limited, limits[:, 0], -np.inf)
        self.maximum = np.where(limited, limits[:, 1], np.inf)
        assert len(self.dofs) == len(self.actuators) == 42 and self.m.opt.timestep == DT
        self.thorax = sim._internal_bodyids_by_fly[self.name][fly.get_bodysegs_order().index(BodySegment('c_thorax'))]
        self.fronts = [fly.get_legs_order().index(side+'f') for side in ('l','r')]
        self.ground_geoms = {i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == 0
                             and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
        self.axis_bodies = {i for i in range(1, self.m.nbody)
                            if any(key in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or '')
                                   for key in ('c_head','c_thorax','c_abdomen','wing'))}
        assert self.ground_geoms and self.thorax in self.axis_bodies
        self.weight = float(sum(self.m.body_mass) * abs(self.m.opt.gravity[2]))
        self.new_teacher(17)
        self.neutral = self.teacher(np.zeros(2))

    def new_teacher(self, seed):
        self.motor = HybridTurningController(timestep=DT, enable_adhesion=False, output_dof_order=self.dofs)
        self.motor.reset(seed=seed)

    def teacher(self, command):
        observed = HybridControllerObservation.from_sim(self.sim, self.name, legs=self.motor.legs)
        return self.motor.step(command, observed).joint_angles

    def contact(self):
        found, forces, *other = self.sim.get_ground_contact_info(self.name)
        return found, forces

    def observation(self, command, contacts):
        found, forces = contacts
        joints = self.sim.get_joint_angles(self.name)[self.joint_indices]
        velocity = self.sim.get_joint_velocities(self.name)[self.joint_indices]
        current_setpoint = self.d.ctrl[self.actuators]
        rotation = self.d.xmat[self.thorax].reshape(3, 3)
        rates = self.d.cvel[self.thorax].reshape(2, 3) @ rotation
        return np.concatenate((command, joints-self.neutral, velocity/35., current_setpoint-self.neutral,
                               np.minimum(found, 1.), np.arcsinh(forces.flatten()/max(self.weight/6., 1e-6)),
                               rotation[2], rates[0]/12., rates[1]/15.)).astype(np.float32)

    def clip_action(self, target):
        return np.clip(target, self.minimum, self.maximum)

    def apply(self, target):
        self.sim.set_actuator_inputs(self.name, ActuatorType.POSITION, target)
        self.sim.step()

    def nonfoot(self):
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in self.ground_geoms and self.m.geom_bodyid[c.geom2] in self.axis_bodies
                    or c.geom2 in self.ground_geoms and self.m.geom_bodyid[c.geom1] in self.axis_bodies):
                return True
        return False

    def at_start(self, seed):
        with np.load(MODEL_FILE, allow_pickle=False) as snap:
            assert np.array_equal(snap['neutral'], self.neutral)
            mujoco.mj_setState(self.m, self.d, snap['integration'], FLAGS)
        mujoco.mj_forward(self.m, self.d)
        self.new_teacher(seed)


def freeze():
    body = Body()
    for tick in range(2800):
        body.apply(body.teacher(np.zeros(2)))
    assert not body.d.warning.number.any() and body.d.xmat[body.thorax, 8] > .95
    integration = np.empty(mujoco.mj_stateSize(body.m, FLAGS))
    mujoco.mj_getState(body.m, body.d, integration, FLAGS)
    assert not MODEL_FILE.exists()
    np.savez(MODEL_FILE, integration=integration, neutral=body.neutral)
    mujoco.mj_saveModel(body.m, str(HERE/'BODY.mjb'), None)
    write(HERE/'BODY.json', {'snapshot_sha256': sha(MODEL_FILE), 'compiled_body_sha256': sha(HERE/'BODY.mjb'),
          'actuated_order': [str(dof) for dof in body.dofs], 'boot_ticks': 2800, 'nq': body.m.nq,
          'thorax_body_id': int(body.thorax), 'weight_native': body.weight})


class Neck:
    def __init__(self):
        catalog = json.loads((PARENT/'CATALOG.json').read_text())
        ports = {root: port for root, port in catalog['ports'].items() if port['group'] in GROUPS}
        self.roots = list(ports)
        self.circuit = circuit = Circuit({'ports': ports})
        self.receiver = Receiver(circuit)
        original = json.loads((PARENT/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json').read_text())['probes']
        for name in self.receiver.probes:
            roots = original['/'.join(name)]
            self.receiver.probes[name] = roots, [circuit.index[r] for r in roots], self.receiver.probes[name][2]
        self.forward = circuit.validate_inputs([root for root, spec in ports.items() if spec['group']=='forward'],'brain')
        self.brake = circuit.validate_inputs([root for root, spec in ports.items() if spec['group']=='halt_web_a'],'brain')
        self.receptors = {}
        for port in ports.values():
            if port['direction'] == 'AN':
                group = port['group']
                self.receptors[group, port['side']] = circuit.validate_inputs([r for r, c in port['incoming_receptors']], 'body')
        self.cut = np.array([root in circuit.brain_roots for root in self.roots])
        self.open = np.zeros(len(ports), dtype=bool)
        self.drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
        self.command = np.zeros(2)

    def step(self, tick, case, forces):
        self.drive.fill(0)
        if tick % case.pulse_every == 0 and case.start <= tick < case.stop:
            self.drive[0, self.forward] = 68.75
            if case.brake and tick >= case.brake:
                self.drive[0, self.brake] = 68.75
        if tick % 23 == 0:
            for force, side in zip(forces, ('left', 'right'), strict=True):
                if force > .04:
                    self.drive[0,self.receptors['front_touch',side]] = 68.75
                if force > .2:
                    self.drive[0,self.receptors['front_load',side]] = 68.75
        emitted, arrived, _, voltage = self.circuit.step(self.drive, self.cut if case.cut_start <= tick < case.cut_end else self.open)
        if tick % 5 == 0:
            self.receiver.update(voltage, DT*5)
            self.command = self.receiver.ground()
        return emitted[0], arrived[0]


def score(case, trace):
    tick = np.arange(len(trace))
    pos = trace[:,1:3]
    still_start = float(np.linalg.norm(pos[:1200]-pos[0], axis=1).max(initial=0))
    recent = trace[-2000:]
    end_drift = float(np.linalg.norm(recent[:,1:3]-recent[0,1:3],axis=1).max(initial=0))
    end_rms = float(np.sqrt(np.mean((np.diff(recent[:,1:3],axis=0,prepend=recent[:1,1:3])/DT)**2)*2))
    axis_contact = int(trace[:,8].sum())
    fraction = float(np.mean(trace[:,9:15].sum(1) >= 3))
    fully_cut = case.cut_end >= TICKS and case.cut_start == 0
    effective_start = max(case.start, case.cut_end if case.cut_start == 0 and case.cut_end < TICKS else 0)
    effective_stop = min(t for t in (case.stop, case.brake or TICKS, case.cut_start or TICKS))
    active = trace[(tick >= effective_start+1800) & (tick < min(effective_start+6800,effective_stop))]
    contact = active[:,9:15] > 0
    begins = (contact[1:] & ~contact[:-1]).sum(0).tolist()
    ends = (contact[:-1] & ~contact[1:]).sum(0).tolist()
    speed = float((active[-1,1]-active[0,1])/(DT*(len(active)-1))) if len(active)>1 else 0.
    dx = float(pos[-1,0]-pos[0,0])
    gates = dict(stand=still_start<=.08, height=float(trace[:,3].min())>=.53, upright=float(trace[:,4].min())>=.80,
                 axial_contact=axis_contact==0, calm=end_drift<=.10 and end_rms<=.30)
    if fully_cut:
        gates['no_drive'] = bool(np.max(np.abs(trace[:,6:8]))==0 and abs(dx)<=.16)
    else:
        gates.update(advance=dx>=1.2 and speed>=2.5, cycle=min(begins+ends)>=2)
        if case.cut_end and case.cut_start == 0:
            gates['initial_cut'] = bool(not np.any(trace[:case.cut_end,6:8]))
    return dict(pass_=bool(all(gates.values())), gates=gates, native_steps=len(trace),
                first_still_drift_mm=still_start, last_still_drift_mm=end_drift,last_still_rms_mm_s=end_rms,
                min_height_mm=float(trace[:,3].min()), min_up=float(trace[:,4].min()), net_x_mm=dx,
                active_speed_mm_s=speed, axial_contact_steps=axis_contact, support_fraction=fraction,
                active_contact_starts=begins, active_contact_ends=ends)


@torch.inference_mode()
def run(case, mode, checkpoint, out, mix):
    start = perf_counter()
    body = Body()
    body.at_start(case.seed)
    neck = Neck()
    student = load_policy(checkpoint) if checkpoint else None
    teacher = mode in ('teacher','correct')
    assert (mode == 'teacher') == (student is None)
    # Contacts are sampled on every native step for metrics and receptor input.
    trace = np.zeros((TICKS,15), dtype=np.float64)
    e = np.zeros((TICKS,len(neck.roots)),dtype=np.uint8)
    a = np.zeros_like(e)
    obs, targets, applied, poses = [], [], [], []
    for tick in range(TICKS):
        contacts = body.contact()
        found, forces = contacts
        e[tick], a[tick] = neck.step(tick,case,np.linalg.norm(forces[body.fronts],axis=1))
        command = neck.command
        y = body.clip_action(body.teacher(command)) if teacher else None
        if tick % HOLD == 0:
            x = body.observation(command, contacts)
            if student is not None:
                estimate = body.clip_action(action(student,x)+body.neutral)
            ctrl = y if mode == 'teacher' else ((1-mix)*y+mix*estimate if mode == 'correct' else estimate)
            obs.append(x)
            if teacher:
                targets.append((y-body.neutral).astype(np.float32))
            applied.append(ctrl.copy())
            poses.append(body.d.qpos.copy())
        body.d.xfrc_applied[body.thorax,1] = .12*body.weight*case.push_side if 6500<=tick<6700 else 0
        body.apply(ctrl)
        rotation = body.d.xmat[body.thorax]
        trace[tick] = [(tick+1)*DT, *body.d.xpos[body.thorax], rotation[8], math.atan2(rotation[3],rotation[0]),
                       *command, body.nonfoot(), *found]
    assert not body.d.warning.number.any() and np.isfinite(trace).all()
    out.mkdir()
    np.savez_compressed(out/'TRACE.npz', trace=trace, emitted=e, arrived=a, x=np.asarray(obs),
                        y=np.asarray(targets), ctrl=np.asarray(applied), qpos=np.asarray(poses))
    result = score(case, trace)
    result.update(case=asdict(case), mode=mode, mix=mix, weight_sha256=sha(checkpoint) if checkpoint else None,
                  body_snapshot_sha256=sha(MODEL_FILE), catalog_sha256=sha(PARENT/'CATALOG.json'),
                  arrival_roots=neck.roots, observed_dim=len(obs[0]), trace_sha256=sha(out/'TRACE.npz'),
                  seconds=perf_counter()-start)
    write(out/'RESULT.json',result)
    print(out.name, result['pass_'], {k:round(result[k],4) for k in ('net_x_mm','active_speed_mm_s','last_still_rms_mm_s','min_height_mm')}, result['gates'], flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','teacher','evaluate','correct'))
    parser.add_argument('case',nargs='?',choices=CASES)
    parser.add_argument('out',nargs='?',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--mix',type=float,default=.55)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode=='freeze':
        freeze()
    else:
        run(CASES[args.case],args.mode,args.checkpoint,args.out,args.mix)
