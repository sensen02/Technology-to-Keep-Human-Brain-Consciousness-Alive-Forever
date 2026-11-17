"""Replay the fixed stock body policy on a rigid, inclined floor."""
import argparse
import ctypes
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HISTORY = HERE.parents[1]
BASE = HISTORY / '20261116T000536Z'
TURNS = HISTORY / '20261116T004235Z'
sys.path[:0] = [str(BASE), str(TURNS)]
import walking_bench as bench
from prototype import BANK, CHECKPOINT, policy as stock_policy
from verify_native import Record, SO

STAND = bench.stand
W = bench.W
NATIVE_PER_MS = 80
WARM, WALK, STOP = 500, 9600, 1200
PIVOT = -.5
SLOPE_SO = HERE / 'slope_contact.so'


class Terrain:
    def __init__(self, body, angle_deg):
        self.body = body
        self.angle = float(np.deg2rad(angle_deg))
        model = body.plant
        self.floor = model.geom('floor').id
        radians = self.angle
        self.normal = np.array([-np.sin(radians), 0., np.cos(radians)])
        self.uphill = np.array([np.cos(radians), 0., np.sin(radians)])
        self.point = np.array([0., 0., -PIVOT*np.tan(radians)])
        if model.geom_bodyid[self.floor] != 0 or model.geom_type[self.floor] != mj.mjtGeom.mjGEOM_PLANE:
            raise ValueError('Expected frozen world plane')
        np.testing.assert_array_equal(model.geom_pos[self.floor], np.zeros(3))
        np.testing.assert_array_equal(model.geom_quat[self.floor], [1., 0., 0., 0.])
        model.geom_pos[self.floor] = self.point
        model.geom_quat[self.floor] = [np.cos(.5*radians), 0., -np.sin(.5*radians), 0.]
        if radians != 0.:
            model.geom_sameframe[self.floor] = mj.mjtSameFrame.mjSAMEFRAME_NONE
        self.pads = np.asarray([body.pads[leg] for leg in bench.LEGS])
        self.radii = model.geom_size[self.pads,0].copy()

    def confirm(self):
        d = self.body.d
        normal = d.geom_xmat[self.floor].reshape(3,3)[:,2]
        np.testing.assert_allclose(normal, self.normal, rtol=0., atol=1e-14)
        np.testing.assert_array_equal(d.geom_xpos[self.floor], self.point)

    def clearances(self):
        centers = self.body.d.geom_xpos[self.pads]
        return np.min((centers-self.point) @ self.normal - self.radii, axis=1)

    def reference_contacts(self, velocity, feet):
        body = self.body
        model, d = body.plant, body.d
        jac = np.empty((3,model.nv))
        wrench = np.empty(6)
        totals = np.zeros(13)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom
            if c.efc_address < 0 or self.floor not in (a,b):
                continue
            geom = b if a == self.floor else a
            mj.mj_contactForce(model,d,i,wrench)
            transmitted = c.frame.reshape(3,3).T @ wrench[:3]
            load = max(0., float(self.normal @ transmitted) * (1 if a == self.floor else -1))
            leg = feet[geom]
            if leg < 0:
                totals[12] += load
                continue
            mj.mj_jac(model,d,jac,None,c.pos,model.geom_bodyid[geom])
            speed = jac @ velocity
            tangent = speed - self.normal * (speed @ self.normal)
            totals[leg] += load
            totals[leg+6] += load * np.linalg.norm(tangent)
        return totals


class SlopeRecord(Record):
    def __init__(self, body, duration, terrain):
        super().__init__(body,duration)
        lib = ctypes.CDLL(str(SLOPE_SO))
        self.native_slope = lib.slope_contacts
        self.native_slope.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int] + [ctypes.c_void_p]*3
        self.native_slope.restype = None
        self.normal = np.zeros_like(self.forces)
        self.terrain = terrain

    def solved(self, body, before):
        super().solved(body,before)
        row = self.normal[self.count]
        self.native_slope(body.plant._address,body.d._address,self.terrain.floor,
                          self.foot_map.ctypes.data,self.velocity.ctypes.data,row.ctypes.data)
        if self.count % 4031 == 0:
            reference = self.terrain.reference_contacts(self.velocity,self.foot_map)
            np.testing.assert_allclose(row,reference,rtol=1e-12,atol=1e-9)
            if self.terrain.angle == 0.:
                np.testing.assert_allclose(row,self.forces[self.count],rtol=1e-12,atol=1e-9)


def terrain_window(rec, trace, begin, end):
    terrain = rec.terrain
    weight = W*np.cos(terrain.angle)
    n0,n1 = begin*NATIVE_PER_MS,end*NATIVE_PER_MS
    normal = rec.normal[n0:n1]
    vertical = rec.forces[n0:n1]
    delta = trace['com'][end] - trace['com'][begin]
    tangent = delta - terrain.normal*(terrain.normal @ delta)
    duration = (end-begin)*.001
    speed = float(np.linalg.norm(tangent)/duration)
    load = normal[:,:6].sum(0)
    absolute = np.divide(normal[:,6:12].sum(0),load,out=np.zeros(6),where=load>0.)
    world_load = vertical[:,:6].sum(0)
    world_absolute = np.divide(vertical[:,6:12].sum(0),world_load,out=np.zeros(6),where=world_load>0.)
    horizontal = float(np.linalg.norm(delta[:2])/duration)
    expected_rise = float(delta[0]*np.tan(terrain.angle))
    peak = float(normal[:,12].max())
    clear = min(float(((rec.poses[n0:n1,:3]-terrain.point) @ terrain.normal).min()),
                float((trace['com'][end]-terrain.point) @ terrain.normal))
    return dict(bounds_ms=[begin,end],native_steps=n1-n0,duration_s=duration,
                signed_uphill_progress=float(tangent@terrain.uphill), horizontal_progress=float(delta[0]),
                signed_lateral_drift=float(delta[1]), actual_ascent=float(delta[2]), terrain_ascent=expected_rise,
                slope_tangent_speed=speed, signed_uphill_speed=float(tangent@terrain.uphill/duration),
                terrain_normal_support_mean_expected_fraction=float(load.sum()/len(normal)/weight),
                terrain_normal_support_mean_W=float(load.sum()/len(normal)/W),
                vertical_support_mean_W=float(world_load.sum()/len(vertical)/W),
                min_com_normal_clearance=clear, terrain_normal_nonfoot_peak_W=peak/W,
                absolute_tangent_slide_native_s={leg:float(value) for leg,value in zip(bench.LEGS,absolute)},
                max_normal_slip=float(absolute.max()/speed) if speed>0. and (load>0.).all() else None,
                legacy_vertical_horizontal_slip=float(world_absolute.max()/horizontal) if horizontal>0. else None)


def stand_gate(rec, trace, begin, end):
    terrain = rec.terrain
    weight = W*np.cos(terrain.angle)
    stop = terrain_window(rec,trace,end-200,end)
    low,high = (end-200)*NATIVE_PER_MS, end*NATIVE_PER_MS
    positions = trace['com'][end-200:end+1]
    step_speed = np.linalg.norm(np.diff(positions,axis=0),axis=1)/(.001*STAND.LENGTH)
    loads = rec.normal[low:high,:6]/weight
    conform = dict(enough_time=end-begin>=200, speed_rms=float(np.sqrt(np.square(step_speed).mean()))<=.01,
                   support=stop['terrain_normal_support_mean_expected_fraction']>=.90,
                   three_at_least_95pct=float(((loads>=.02).sum(1)>=3).mean())>=.95,
                   upright=float(rec.poses[low:high,5].min())>=np.cos(np.deg2rad(20.)),
                   clearance=stop['min_com_normal_clearance']>=1.20,nonfoot=stop['terrain_normal_nonfoot_peak_W']<.01)
    return dict(passed=all(conform.values()),gates=conform,speed_rms_bodylength_s=float(np.sqrt(np.square(step_speed).mean())),
                terrain=stop)


def audit(rec, trace, walk_end, duration):
    end = len(trace['qpos'])-1
    assert rec.count == end*NATIVE_PER_MS and not rec.poses[:rec.count,7].any()
    np.testing.assert_array_equal(rec.endpoints[:end+1],trace['qpos'])
    expected_time = trace['time'][0] + np.arange(rec.count)*12.5e-6
    np.testing.assert_allclose(rec.poses[:rec.count,6],expected_time,rtol=0.,atol=5e-8)
    result = dict(native_steps=rec.count, actual_ms=end, requested_ms=duration,
                  no_extra_force=True,native_cross_checks=rec.cross_checks)
    if end < duration:
        return dict(**result,passed=False,complete=False)
    walking = terrain_window(rec,trace,WARM,walk_end)
    feet, _ = bench.cycles(trace,WARM,walk_end,arm_before=20)
    initial = stand_gate(rec,trace,0,WARM)
    final = stand_gate(rec,trace,walk_end,end)
    walking_checks = dict(progress=walking['signed_uphill_progress']>=.5,
        speed=walking['signed_uphill_speed']>=.05, cycles=all(feet.values()),
        support=walking['terrain_normal_support_mean_expected_fraction']>=.90,
        clearance=walking['min_com_normal_clearance']>=1.20, nonfoot=walking['terrain_normal_nonfoot_peak_W']==0.,
        slip=walking['max_normal_slip'] is not None and walking['max_normal_slip']<=.50)
    if rec.terrain.angle>0.:
        walking_checks['ascent'] = walking['actual_ascent']>=.012 and walking['terrain_ascent']>=.012
    elif rec.terrain.angle<0.:
        walking_checks['descent'] = walking['actual_ascent']<=-.012 and walking['terrain_ascent']<=-.012
    else:
        walking_checks['flat_elevation'] = abs(walking['actual_ascent'])<.012
    checks = dict(walking_checks,initial=initial['passed'],terminal=final['passed'])
    return dict(**result,complete=True, walking=walking, cycles=feet, initial_stand=initial, terminal_stand=final,
                gates=checks,passed=all(checks.values()))


def run(output, angle, walk_ms):
    output.mkdir()
    rig = STAND.Bench(output)
    body = rig.body
    body.physics_order = STAND.physics.rc.SENS_IDX
    np.testing.assert_array_equal(body.physics_order,bench.SENSOR_BY_LEG)
    initial = bench.STARTS/'nominal.npz'
    snapshot_hash = bench.read(bench.STARTS/'BANK.json')['nominal']['sha256']
    bench.load_initial(rig,initial,snapshot_hash)
    terrain = Terrain(body,angle)
    if angle != 0.:
        mj.mj_forward(body.plant,body.d)
    with np.load(initial,allow_pickle=False) as source:
        np.testing.assert_array_equal(body.d.qpos,source['qpos'])
        np.testing.assert_array_equal(body.d.qvel,source['qvel'])
    terrain.confirm()
    assert not body.adhesion_on and np.array_equal(body.plant.opt.gravity,[0.,0.,-9801.])
    paths = [Path(__file__),HERE/'slope_contact.cpp',SLOPE_SO,SO,BASE/'prototype.py',BANK,CHECKPOINT,initial]
    walk_end = WARM+walk_ms
    duration = walk_end+STOP
    identity = STAND.identity.capture(body,dict(initial=snapshot_hash,slope_deg=angle,pivot_x=PIVOT,
                    walk_ms=walk_ms, policy='archived instantaneous stock', horizon_ms=duration),extra_paths=paths)
    STAND.put(output/'RUN_IDENTITY.json',identity)
    record = SlopeRecord(body,duration,terrain)
    body.recorder = record
    policy = stock_policy()
    decisions, samples = [], []
    action = rig.zero.copy()
    measured = body.measure()
    clock = time.monotonic()
    bad = 0

    def sample():
        d = body.d
        samples.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),com=d.subtree_com[body.th].copy(),
                            loads=measured['loads'].copy(),nonfoot=float(measured['nonfoot']),padz=terrain.clearances(),
                            padz_world=measured['padz'].copy(),speed=measured['speed'].copy(),foot=measured['foot'].copy(),
                            upright=float(d.xmat[body.th,8]),target=body.actuator.target_rad.copy(),ctrl=d.ctrl.copy(),
                            applied=action.copy(),yaw=float(np.arctan2(d.xmat[body.th,3],d.xmat[body.th,0]))))

    sample()
    for tick in range(duration):
        if tick % 5 == 0:
            intent = int(WARM <= tick < walk_end)
            observation = bench.features(body,measured)
            action = bench.legacy.bounded_action(policy(observation,intent))
            decisions.append((tick,observation.copy(),intent,action.copy()))
        measured = body.tick(action)
        bad = bad+1 if measured['com'] < .8 or measured['nonfoot'] > .5*W else 0
        sample()
        if bad >= 100:
            break
    trace = {field:np.asarray([sample[field] for sample in samples]) for field in samples[0]}
    policy_trace = {key:np.asarray(value) for key,value in zip(('ms','x','intent','action'),zip(*decisions),strict=True)}
    counts = audit(record,trace,walk_end,duration)
    np.savez_compressed(output/'TRACE.npz',**trace)
    np.savez_compressed(output/'DECISIONS.npz',**policy_trace)
    np.savez_compressed(output/'NATIVE.npz',forces=record.forces[:record.count],pose=record.poses[:record.count],
                        surface_normal=record.normal[:record.count])
    STAND.identity.verify_runtime(identity,body)
    result = dict(**counts,walk_ms=walk_ms,angle_deg=angle,plane_point=terrain.point.tolist(),
                  plane_normal=terrain.normal.tolist(),terminal_rule='stop on same slope',
                  declarations='Fixed stock controller and fixed initial joint/root state, only world plane tilt and pivot vary',
                  files={name:STAND.sha(output/name) for name in ('TRACE.npz','DECISIONS.npz','NATIVE.npz','RUN_IDENTITY.json')},
                  elapsed_wall_s=time.monotonic()-clock)
    STAND.put(output/'RESULT.json',result)
    print(result,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--angle',type=float,required=True)
    p.add_argument('--walk-ms',type=int,default=WALK)
    a = p.parse_args()
    if not -4. <= a.angle <= 4. or not 4800 <= a.walk_ms <= 19200 or a.walk_ms%5:
        p.error('Incline and fixed-millisecond course are out of range')
    torch.set_num_threads(1)
    run(a.output.resolve(),a.angle,a.walk_ms)
