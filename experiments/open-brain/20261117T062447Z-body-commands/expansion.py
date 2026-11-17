"""Wear no task labels on the body: train from BANC VNC activity and local mechanics."""
from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / '20261117T055904Z-body-imitation'
sys.path.insert(0, str(FIRST))
from body_run import Body, Circuit, DT, HOLD, MODEL_FILE, PARENT, Receiver, TICKS, action, load_policy, sha, write
from physical import load_receptors, schedule_indices

FAMILIES = ('cruise', 'fast', 'arc_left', 'arc_right', 'pivot_left', 'pivot_right', 'back', 'brake')
GROUPS = {'forward', 'forward_speed', 'walking_steer', 'backward_moonwalker', 'halt_web_a',
          'front_touch', 'front_load'}


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    seed: int
    pulse_every: int
    start: int
    stop: int
    steer_cut: bool = False
    push_side: int = 0

    def schedule(self):
        left_or_right = self.family.rsplit('_', 1)[-1]
        if self.family == 'back':
            return [('backward_moonwalker', None, self.start, self.stop)]
        if self.family.startswith('pivot'):
            return [('walking_steer', left_or_right, self.start, self.stop)]
        forward = [('forward', None, self.start, self.stop)]
        if self.family == 'fast':
            forward.append(('forward_speed', None, self.start + 400, self.stop))
        if self.family.startswith('arc'):
            forward.append(('walking_steer', left_or_right, self.start + 1400, self.stop))
        if self.family == 'brake':
            forward.append(('halt_web_a', None, self.start + 7100, self.stop))
        return forward

    @property
    def outward(self):
        return 1 if self.family.endswith('left') else -1


TRAIN = tuple(
    Case(f'train_{family}_{variant}', family, 210 + 11*i + variant, period, start, stop,
         push_side=1 if variant == 2 else 0)
    for i, family in enumerate(FAMILIES)
    for variant, (period, start, stop) in enumerate(((56, 2050, 11900), (60, 2310, 12300), (64, 2530, 12500)))
)
DEVELOP = tuple(Case(f'dev_{family}', family, 319+7*i, 58, 2220, 12130) for i, family in enumerate(FAMILIES))
CORRECT = tuple(Case(f'correct_{family}', family, 430+7*i, 62, 2175, 12210, push_side=-1)
                for i, family in enumerate(FAMILIES))
HELD = tuple(Case(f'held_{family}_{variant}', family, 530 + 11*i + variant,
                  period, start, stop, push_side=-1 if variant else 0)
             for i, family in enumerate(FAMILIES)
             for variant, (period, start, stop) in enumerate(((59, 2240, 12250), (63, 2460, 12700))))
CUT = Case('held_cut_steer', 'arc_left', 530 + 11*FAMILIES.index('arc_left'), 59, 2240, 12250, steer_cut=True)
CASES = {case.name: case for case in (*TRAIN, *DEVELOP, *CORRECT, *HELD, CUT)}


class Neck:
    def __init__(self, case):
        catalog = json.loads((PARENT/'CATALOG.json').read_text())
        ports = {root: port for root, port in catalog['ports'].items() if port['group'] in GROUPS}
        self.circuit = circuit = Circuit({'ports': ports})
        self.receiver = Receiver(circuit)
        approved = json.loads((PARENT/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json').read_text())['probes']
        for key, (_, _, sign) in self.receiver.probes.items():
            roots = approved['/'.join(key)]
            self.receiver.probes[key] = roots, [circuit.index[root] for root in roots], sign
        self.schedule = schedule_indices(circuit, case.schedule())
        self.receptors = load_receptors(circuit)
        self.roots = circuit.port_roots
        self.cut = np.array([case.steer_cut and port['group'] == 'walking_steer' and port['side'] == 'left'
                             for port in ports.values()])
        self.drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
        self.command = np.zeros(2)

    def step(self, tick, period, forces):
        self.drive.fill(0)
        if tick % period == 0:
            for columns, start, stop in self.schedule:
                if start <= tick < stop:
                    self.drive[0, columns] = 68.75
        if tick % 23 == 0:
            for force, side in zip(forces, ('left', 'right'), strict=True):
                for group, threshold in (('front_touch', .04), ('front_load', .20)):
                    if force > threshold:
                        self.drive[0, self.receptors[group, side]] = 68.75
        emitted, arrived, _, voltage = self.circuit.step(self.drive, self.cut)
        if tick % 5 == 0:
            self.receiver.update(voltage, DT*5)
            self.command = self.receiver.ground()
        return emitted[0], arrived[0]


def score(case, trace):
    xy = trace[:, 1:3]
    yaw = np.unwrap(trace[:, 5])
    active = slice(case.start+2000, min(case.stop, case.start+7000))
    contact = trace[active, 9:15] > 0
    cycles = np.minimum((contact[1:] & ~contact[:-1]).sum(0),
                        (contact[:-1] & ~contact[1:]).sum(0))
    seconds = DT * (len(contact) - 1)
    pelvis = xy[active]
    directions = np.column_stack((np.cos(yaw[active][:-1]), np.sin(yaw[active][:-1])))
    forward_speed = float(np.sum(np.diff(pelvis, axis=0) * directions) / seconds)
    delta_yaw = float(np.rad2deg(yaw[case.stop-1]-yaw[case.start]))
    peak_excursion = float(np.linalg.norm(xy-xy[0], axis=1).max())
    end_drift = float(np.linalg.norm(xy[-2000:]-xy[-2000], axis=1).max())
    end_rms = float(np.sqrt(np.mean(np.sum(np.diff(xy[-2000:],axis=0)**2,axis=1))) / DT)
    first_drift = float(np.linalg.norm(xy[:1200]-xy[0], axis=1).max())
    minimum_height = float(trace[:, 3].min())
    minimum_up = float(trace[:, 4].min())
    axial_contact = int(trace[:, 8].sum())
    gates = dict(stand=first_drift<=.08, height=minimum_height>=.53, upright=minimum_up>=.80,
                 axial_contact=axial_contact==0, calm=end_drift<=.10 and end_rms<=.30,
                 cycle=bool(min(cycles)>=2))
    if case.steer_cut:
        gates.update(advance=forward_speed>=2.5, turn_absent=abs(delta_yaw)<=12.)
    elif case.family.startswith('pivot'):
        gates.update(yaw=case.outward*delta_yaw>=18., pivot=peak_excursion<=2.25)
    elif case.family.startswith('arc'):
        gates.update(advance=forward_speed>=2.5, yaw=case.outward*delta_yaw>=18.)
    elif case.family == 'back':
        gates.update(reverse=forward_speed<=-2.5, stable_heading=abs(delta_yaw)<=35.)
    else:
        gates.update(advance=forward_speed>=(5.0 if case.family=='fast' else 2.5))
        if case.family == 'brake':
            margin = 1000
            sample = slice(case.start+7100+margin, case.stop)
            gates['active_brake'] = bool(np.abs(trace[sample,6:8]).max(initial=0) <= .035 and
                                         np.linalg.norm(xy[sample]-xy[sample.start],axis=1).max(initial=0)<=.10)
    return dict(pass_=all(gates.values()), gates=gates, native_steps=len(trace),
                net_x_mm=float(xy[-1,0]-xy[0,0]), active_speed_mm_s=forward_speed,
                yaw_degrees=delta_yaw, peak_excursion_mm=peak_excursion,
                last_still_rms_mm_s=end_rms, last_still_drift_mm=end_drift,
                min_height_mm=minimum_height, min_up=minimum_up, axial_contact_steps=axial_contact,
                cycles=cycles.tolist())


@torch.inference_mode()
def run(case, mode, checkpoint, out):
    started = perf_counter()
    body = Body()
    body.at_start(case.seed)
    neck = Neck(case)
    student = load_policy(checkpoint) if checkpoint else None
    teacher = mode != 'evaluate'
    trace = np.empty((TICKS, 15))
    sent = np.empty((TICKS, len(neck.roots)), dtype=np.uint8)
    arrived = np.empty_like(sent)
    observations, labels, commands, poses = [], [], [], []
    for tick in range(TICKS):
        contacts = body.contact()
        found, forces = contacts
        sent[tick], arrived[tick] = neck.step(tick, case.pulse_every, np.linalg.norm(forces[body.fronts],axis=1))
        if teacher:
            target = body.clip_action(body.teacher(neck.command))
        if tick % HOLD == 0:
            obs = body.observation(neck.command, contacts)
            if student is not None:
                estimate = body.clip_action(body.neutral + action(student, obs))
            if teacher:
                labels.append((target-body.neutral).astype(np.float32))
            control = target if mode == 'teacher' else (.48*target+.52*estimate if mode == 'correct' else estimate)
            observations.append(obs)
            commands.append(control.copy())
            poses.append(body.d.qpos.copy())
        body.d.xfrc_applied[body.thorax,1] = .12*body.weight*case.push_side if 6500<=tick<6700 else 0
        body.apply(control)
        rotation = body.d.xmat[body.thorax]
        trace[tick] = [(tick+1)*DT, *body.d.xpos[body.thorax], rotation[8], math.atan2(rotation[3],rotation[0]),
                       *neck.command, body.nonfoot(), *found]
    if body.d.warning.number.any() or not np.isfinite(trace).all():
        raise RuntimeError('native simulation reported a warning or invalid state')
    out.mkdir()
    record = out/'TRACE.npz'
    np.savez_compressed(record, trace=trace, emitted=sent, arrived=arrived, x=np.asarray(observations),
                        y=np.asarray(labels), ctrl=np.asarray(commands), qpos=np.asarray(poses))
    result = score(case, trace)
    result.update(case=asdict(case), mode=mode, weight_sha256=sha(checkpoint) if checkpoint else None,
                  body_snapshot_sha256=sha(MODEL_FILE), catalog_sha256=sha(PARENT/'CATALOG.json'),
                  arrival_roots=neck.roots, probes=neck.receiver.manifest(), schedule=case.schedule(),
                  cut_roots=[root for root, cut in zip(neck.roots, neck.cut, strict=True) if cut],
                  trace_sha256=sha(record), seconds=perf_counter()-started)
    write(out/'RESULT.json', result)
    print(case.name, round(result['active_speed_mm_s'], 2), round(result['yaw_degrees'], 1),
          round(result['peak_excursion_mm'], 2), result['gates'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('teacher','evaluate','correct'))
    parser.add_argument('case', choices=CASES)
    parser.add_argument('out', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.mode == 'teacher'):
        parser.error('checkpoint required for student or corrective collection')
    torch.set_num_threads(1)
    run(CASES[args.case], args.mode, args.checkpoint, args.out)
