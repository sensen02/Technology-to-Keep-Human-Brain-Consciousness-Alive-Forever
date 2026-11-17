"""Evaluate an actual low block on the unchanged finite actuator and native solver."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch

from controller import Controller, HERE, STUMBLE_SOURCE, TURN_SOURCE, turn
from verify_native import Record

sys.path.insert(1, str(STUMBLE_SOURCE))
from physical import Local, LOCAL_FIELDS
from geometry import OBSTACLE, make_obstacle_model

stan, W, LEGS = turn.stand, turn.W, turn.LEGS
INITIAL = turn.STARTS / 'nominal.npz'
LIBRARY = HERE / 'terrain_contacts.so'
COMMAND_UNITS = ('native length/s', 'radian/s')


def build_scene(scenario):
    loader = stan.physics.act_mod.load_full_model
    if scenario is not None:
        model = make_obstacle_model(scenario)
        stan.physics.act_mod.load_full_model = lambda *args, **kwargs: model
    try:
        c = stan.physics.Core(adhesion=False)
    finally:
        stan.physics.act_mod.load_full_model = loader
    terrain = np.asarray(c.planes, dtype=np.int32)
    box = mj.mj_name2id(c.plant, mj.mjtObj.mjOBJ_GEOM, OBSTACLE)
    if box >= 0:
        terrain[box] = 2
    c.planes = terrain > 0
    c.physics_order = stan.physics.rc.SENS_IDX
    sha = turn.read(turn.STARTS / 'BANK.json')['nominal']['sha256']
    if stan.sha(INITIAL) != sha:
        raise RuntimeError('Frozen start snapshot mismatch')
    with np.load(INITIAL, allow_pickle=False) as z:
        mj.mj_setState(c.plant, c.d, z['state'], mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(c.plant, c.d)
        c.t = int(z['t'])
        c.actuator.command_ticks = int(z['ticks'])
        for key in z.files:
            if key.startswith('core_'):
                setattr(c, key[5:], z[key].copy())
            if key.startswith('act_'):
                setattr(c.actuator, key[4:], z[key].copy())
        np.testing.assert_array_equal(c.d.qpos, z['qpos'])
        np.testing.assert_array_equal(c.d.qvel, z['qvel'])
        np.testing.assert_array_equal(stan.observe(c), z['observation'])
    return c, terrain, sha


class BlockRecord(Record):
    def __init__(self, c, finish, terrain):
        super().__init__(c, finish)
        self.terrain = np.ascontiguousarray(terrain, dtype=np.int32)
        self.parts = np.full(c.plant.ngeom, -1, dtype=np.int32)
        for i, leg in enumerate(LEGS):
            for segment in ('Coxa', 'Femur', 'Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5'):
                g = mj.mj_name2id(c.plant, mj.mjtObj.mjOBJ_GEOM, leg + segment + '_geom')
                if g >= 0:
                    self.parts[g] = i
        self.block = np.zeros((finish * 80, 19))
        self.last_native = np.zeros(32)
        lib = ctypes.CDLL(str(LIBRARY))
        self.contact_function = lib.block_totals
        self.contact_function.argtypes = [ctypes.c_void_p] * 7
        self.contact_function.restype = None

    def solved(self, c, before):
        self.contact_function(
            c.plant._address, c.d._address, self.foot_map.ctypes.data, self.parts.ctypes.data,
            self.terrain.ctypes.data, self.velocity.ctypes.data, self.last_native.ctypes.data,
        )
        self.forces[self.count] = self.last_native[:13]
        self.block[self.count] = self.last_native[13:]
        if self.count % 4031 == 0:
            original = stan.physics.NativeRecorder.solved(None, c, {'qvel_before':self.velocity})
            expected = np.r_[original['loads_native'],
                original['loads_native'] * original['contact_speed_native_per_s'],
                original['nonfoot_force_native']]
            np.testing.assert_allclose(self.last_native[:13], expected, atol=1e-9, rtol=1e-12)
            self.cross_checks += 1


def snapshot(c, current, record):
    row = turn.sample(c, current)
    xyz = np.array([c.d.geom_xpos[c.pads[leg]].copy() for leg in LEGS])
    radius = np.array([c.plant.geom_size[c.pads[leg], 0] for leg in LEGS])
    row.update(pad_xyz=xyz, pad_min_x=(xyz[:,:,0]-radius).min(1),
               block_native=record.last_native[13:].copy())
    return row


def consecutive(mask, length):
    diff = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
    return next((int(a) for a,b in zip(starts,stops,strict=True) if b-a>=length), None)


def event(trace, raw, plan, has_box, walk_end):
    xmin = plan['center'][0] - plan['half_size'][0]
    xmax = plan['center'][0] + plan['half_size'][0]
    first_top, last_top, first_face, floor_past = {}, {}, {}, {}
    for i, leg in enumerate(LEGS):
        top = raw[:,i] >= .02 * W
        side = raw[:,6+i] + raw[:,12+i] >= .002 * W
        q = np.flatnonzero(top)
        first_top[leg] = consecutive(top, 800)
        last_top[leg] = int(q[-1]) if len(q) else None
        first_face[leg] = consecutive(side, 16)
        trailing_floor = (
            (trace['pad_min_x'][:, i] > xmax + .01)
            & (trace['loads'][:, i] >= .02 * W)
            & (trace['block_native'][:,i] < .002 * W)
        )
        if last_top[leg] is not None:
            trailing_floor[:last_top[leg]//80+1] = False
        floor_past[leg] = consecutive(trailing_floor, 20)
    com_past = consecutive(trace['com'][:,0] > xmax + .01, plan['anchor_hold_ms'])
    all_pads_past = consecutive((trace['pad_min_x'] > xmax+.01).all(1), plan['anchor_hold_ms'])
    free_floor = consecutive(
        (trace['pad_min_x'] > xmax+.01).all(1)
        & (trace['com'][:,0] > xmax+.01)
        & (trace['block_native'].max(1) < .002 * W),
        plan['remaining_floor_ms'],
    )
    finish_i = min(len(trace['com'])-1, walk_end)
    checks = dict(
        physical_solid_box=has_box,
        initially_ahead_of_every_pad=bool(trace['pad_xyz'][0,:,:,0].max() + .009 < xmin),
        at_least_two_named_top_support=bool(sum(ms is not None for ms in first_top.values()) >= 2),
        hind_leg_floor_reload=any(floor_past[leg] is not None for leg in ('LH', 'RH')),
        each_foot_reloaded_floor=all(ms is not None for ms in floor_past.values()),
        com_and_pads_past=bool(com_past is not None and all_pads_past is not None),
        continued_floor_walk_after_crossing=bool(free_floor is not None and free_floor + plan['remaining_floor_ms'] <= walk_end),
        all_pads_and_com_still_past_at_walk_end=bool(trace['com'][finish_i,0] > xmax+.01 and (trace['pad_min_x'][finish_i] > xmax+.01).all()),
    )
    return dict(checks=checks, passed=all(checks.values()), first_box_top_native_step=first_top,
                last_box_top_native_step=last_top, first_box_face_native_step=first_face,
                first_floor_reload_beyond_ms=floor_past, com_past_ms=com_past,
                all_pads_past_ms=all_pads_past, continued_ground_only_ms=free_floor,
                box_force_on_nonleg_native_peak_W=float(raw[:,18].max()/W),
                box_top_peak_W=dict(zip(LEGS, (raw[:,:6].max(0)/W).tolist())),
                box_face_impulse_W_s=dict(zip(LEGS, ((raw[:,6:12]+raw[:,12:18]).sum(0)*12.5e-6/W).tolist())))


def native_metrics(trace, record, plan, finish):
    end = min(plan['warm_ms'] + plan['course_ms'], finish)
    begin = plan['warm_ms']
    n = record.forces[begin*80:end*80]
    body = record.poses[begin*80:end*80]
    delta = trace['com'][end,:2] - trace['com'][begin,:2]
    reference = np.array([np.cos(trace['yaw'][begin]), np.sin(trace['yaw'][begin])])
    world = np.linalg.norm(delta) / ((end - begin)*.001)
    raw = n[:,6:12].sum(0) / np.maximum(1e-10, n[:,:6].sum(0))
    cycles = turn.PivotCycles(W)
    for i in range(begin-20, end+1):
        cycles.update(trace['time'][i],trace['loads'][i],trace['padz'][i])
    all_cycles = dict(zip(LEGS,map(int,cycles.counts)))
    min_z = float(min(trace['com'][end,2],body[:,2].min()))
    forward = float(delta@reference / ((end-begin)*.001))
    support = float(n[:,:6].sum() / (len(n)*W))
    pressure = float(n[:,12].max()/W)
    slip = float(raw.max()/world) if world > 1e-10 else None
    checks = dict(support=support>=.9, minimum_com=min_z>=1.2, nonfoot=pressure<=.20,
                  forward=forward>=plan['gate_forward_native_s'],
                  native_slip=slip is not None and slip<=plan['gate_raw_slip'],
                  all_legs_complete_cycle=all(all_cycles.values()))
    initial = turn.standing(trace,0,begin)
    stop = turn.standing(trace,end,finish)
    checks.update(initial_stand=bool(initial['passed']),terminal_stand=bool(stop['passed']))
    return dict(checks=checks, passed=all(checks.values()), initial_stand=initial, terminal_stand=stop,
                minimum_com=min_z, mean_support_W=support, nonfoot_peak_W=pressure,
                forward_native_s=forward, world_native_s=float(world), raw_slip_ratio=slip,
                absolute_slide_native_s=dict(zip(LEGS,raw.tolist())), cycles=all_cycles,
                end_yaw_delta_deg=float(np.rad2deg(trace['yaw'][end]-trace['yaw'][begin])))


def run(mode, output, plan_path, probe_ms=None, compare=None, subject=None):
    torch.set_num_threads(1)
    plan = turn.read(plan_path)
    output.mkdir()
    course_end = plan['warm_ms'] + plan['course_ms']
    finish = course_end + plan['stop_ms'] if probe_ms is None else probe_ms
    if not 600 <= finish <= course_end + plan['stop_ms'] or plan['course_ms'] % 800:
        raise ValueError('Require measurable continuous rollout and complete planned six-leg slots')
    c, terrain, initial_sha = build_scene(None if mode == 'flat' else plan)
    recorder = BlockRecord(c, finish, terrain)
    c.recorder = recorder
    controller = None
    sensor = Local(c, plan['course_ms'])
    paths = [plan_path, INITIAL, HERE/'controller.py', HERE/'rollout.py',
             HERE/'terrain_contacts.cpp', LIBRARY, STUMBLE_SOURCE/'physical.py',
             STUMBLE_SOURCE/'geometry.py', TURN_SOURCE/'turning.py', TURN_SOURCE/'verify_native.py',
             turn.TEACHER, Path(turn.StaticTeacher.__module__.replace('.','/')+'.py'),
             turn.PARENT/'walking_bench.py', turn.PARENT/'prototype.py', turn.PARENT/'PROTOTYPES.npz']
    paths[11] = Path(sys.modules[turn.StaticTeacher.__module__].__file__)
    identity = stan.identity.capture(c,dict(stage='S2/S5 body full physical low-block course',mode=mode,
          scope='offline teacher with private terrain geometry' if mode == 'aware' else 'terrain geometry unavailable to gait',
          command_units=COMMAND_UNITS, actual_limit_ms=finish, probe=probe_ms is not None,
          snapshot_sha256=initial_sha),extra_paths=paths)
    stan.put(output/'RUN_IDENTITY.json',identity)
    stock = turn.straight_policy()
    trace, decisions = [], []
    measure = c.measure()
    trace.append(snapshot(c,measure,recorder))
    began = time.monotonic()
    unsafe = 0
    for tick in range(finish):
        if tick % 5 == 0:
            x = turn.features(c, measure)
            local = sensor.sense(measure)
            intent = int(plan['warm_ms'] <= tick < course_end)
            phase = 0 if tick < plan['warm_ms'] else 1 if intent else 2
            foot_ref = np.full((6,3),np.nan)
            if tick < plan['warm_ms']:
                action, command = stock(x, 0), np.zeros(2)
            else:
                if controller is None:
                    controller = Controller(c, plan, mode == 'aware')
                action, command = controller.action(c, (tick-plan['warm_ms'])*.001)
                foot_ref = controller.reference.last_target.copy()
            if subject is not None:
                action = subject(x, intent, local)
            if action.shape!=(42,) or not np.isfinite(action).all() or abs(action).max()>1.:
                raise ValueError('Invalid finite 42 motor rates')
            decisions.append((tick,x.copy(),local.copy(),command.copy(),intent,phase,action.copy(),foot_ref))
        measure = c.tick(action)
        trace.append(snapshot(c,measure,recorder))
        if (c.d.xfrc_applied != 0.).any() or (c.d.qfrc_applied != 0.).any():
            raise RuntimeError('Unapproved additional force')
        unsafe = unsafe + 1 if measure['com']<.8 or measure['nonfoot']>.5*W else 0
        if tick % 1000 == 999:
            print('ms',tick+1,'com',trace[-1]['com'].round(4).tolist(),
                  'block_top_peak',recorder.block[max(0,recorder.count-80000):recorder.count,:6].max().round(5),
                  'wall s',round(time.monotonic()-began,1),flush=True)
        if unsafe >= 100:
            break
    arrays = {k:np.asarray([row[k] for row in trace]) for k in trace[0]}
    d = {key:np.asarray(col) for key,col in zip(
         ('ms','x','local','command','intent','phase','action','private_planned_feet'),zip(*decisions,strict=True),strict=True)}
    actual = len(trace)-1
    natives = recorder.count
    np.testing.assert_array_equal(recorder.endpoints[:len(trace)], arrays['qpos'])
    np.testing.assert_allclose(recorder.poses[:natives,6], arrays['time'][0]+np.arange(natives)*12.5e-6, atol=1.2e-7, rtol=0.)
    if natives != 80*actual or recorder.poses[:natives,7].any():
        raise RuntimeError('Native time or applied force verification failure')
    np.savez_compressed(output/'TRACE.npz',**arrays)
    np.savez_compressed(output/'DECISIONS.npz',**d)
    np.savez(output/'NATIVE.npz', forces=recorder.forces[:natives],
             poses=recorder.poses[:natives], block=recorder.block[:natives])
    metric = native_metrics(arrays, recorder, plan, actual)
    contacts = event(arrays, recorder.block[:natives], plan, mode!='flat', course_end)
    result = dict(mode=mode, actual_ms=actual, expected_finish_ms=course_end+plan['stop_ms'],
                  complete=actual==course_end+plan['stop_ms'], probe=probe_ms is not None, plan=plan,
                  native_integrations=natives, native_force_cross_checks=recorder.cross_checks,
                  native_forces_crosschecked=True, endpoints_bitwise=True, applied_force_zero=True,
                  elapsed_s=time.monotonic()-began, native_engineering=metric, block_passage=contacts,
                  raw_block_columns=[f'{part}_{leg}_native' for part in ('top_pad','side_pad','side_leg') for leg in LEGS]+['other_geoms_normal_force_native'],
                  local_fields=LOCAL_FIELDS, files={f:stan.sha(output/f) for f in ('TRACE.npz','DECISIONS.npz','NATIVE.npz')})
    result['passed'] = bool(result['complete'] and metric['passed'] and contacts['passed'])
    stan.identity.verify_runtime(identity,c)
    if compare is not None:
        ref = turn.read(compare/'RESULT.json')
        if ref['files'] != result['files']:
            raise RuntimeError('Independent physical repeat changed raw outputs')
        result['independent_repeat_bitwise'] = True
    stan.put(output/'RESULT.json', result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('mode',choices=('aware','blind','flat'))
    a.add_argument('output',type=Path)
    a.add_argument('plan',type=Path)
    a.add_argument('--probe-ms',type=int)
    a.add_argument('--compare',type=Path)
    args = a.parse_args()
    run(args.mode,args.output,args.plan,args.probe_ms,args.compare)
