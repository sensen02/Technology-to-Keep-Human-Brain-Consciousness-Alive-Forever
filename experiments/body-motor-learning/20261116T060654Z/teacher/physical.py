"""Run one real obstacle/floor pairing with unchanged body and actuator dynamics."""
import argparse
import ctypes
import json
from pathlib import Path
import time
import sys

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree
import torch

from geometry import HERE, LEG, OBSTACLE, SEGMENTS, make_obstacle_model
import walking_bench as bench
from prototype import policy as frozen_policy

BASE = HERE.parents[1] / '20261116T000536Z'
ARCHIVE = BASE / 'PROTO_nominal'
TURNS = BASE.parent / '20261116T004235Z'
sys.path.insert(0, str(TURNS))
from verify_native import Record

W = bench.W
stan = bench.stand
read = bench.read
LEGS = bench.LEGS
RF = LEGS.index(LEG)
RF_JOINTS = slice(RF*7,(RF+1)*7)
OTHER = np.r_[0:RF*7, (RF+1)*7:42]
WARM, STOP = 500, 1200
LOCAL_FIELDS = (*(f'{g}_{axis}_per_W' for g in SEGMENTS for axis in 'xyz'),
                *(f'rf_pad_{axis}_velocity_BL_s' for axis in 'xyz'),
                *(f'rf_nominal_target_departure_{i}_rad' for i in range(7)))


def scenario_read(path):
    if path is None:
        return None
    return read(Path(path))


def ob_contacts(body, ob, target_ids):
    """Contact on the obstacle irrespective of controller sensor classification."""
    force = np.zeros(3)
    hits = np.zeros(len(target_ids), dtype=np.int32)
    other = 0
    resistance = 0.
    a = np.zeros(6)
    heading = body.d.xmat[body.th].reshape(3,3)[:,0]
    for i in range(body.d.ncon):
        c = body.d.contact[i]
        ga, gb = int(c.geom1), int(c.geom2)
        if c.efc_address < 0 or ob not in (ga,gb):
            continue
        name_id = gb if ga == ob else ga
        mj.mj_contactForce(body.plant, body.d, i, a)
        if a[0] <= 0.:
            continue
        f = (c.frame.reshape(3,3).T @ a[:3]) * (1 if ga == ob else -1)
        force += f
        if name_id in target_ids:
            hits[target_ids.index(name_id)] += 1
            resistance += max(0., -float(f @ heading))
        else:
            other += 1
    return np.r_[force / W, resistance/W, hits, other]


class GroundBoxRecord(Record):
    def __init__(self, body, duration, sensor):
        super().__init__(body,duration)
        self.obstacle = np.zeros((duration*80,8))
        self.ob = sensor.ob
        self.sensor_id = np.ascontiguousarray(sensor.ids,dtype=np.int32)
        self.check = sensor.obstacle
        lib=ctypes.CDLL(str(HERE/'obstacle_contacts.so'))
        self.obstacle_contacts=lib.obstacle_contacts
        self.obstacle_contacts.argtypes=[ctypes.c_void_p,ctypes.c_void_p,
                                        ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.obstacle_contacts.restype=None

    def solved(self, body, before):
        super().solved(body, before)
        if self.ob < 0:
            return
        row=self.obstacle[self.count]
        self.obstacle_contacts(body.plant._address,body.d._address,self.ob,body.th,
                               self.sensor_id.ctypes.data,row.ctypes.data)
        row[:4] /= W
        if self.count%4031==0 or (row[4:].any() and self.count%157==0):
            np.testing.assert_allclose(row,self.check(),rtol=1.e-11,atol=1.e-10)


class Local:
    def __init__(self, body, walk_ms):
        self.body = body
        self.ids = tuple(body.plant.geom(n).id for n in SEGMENTS)
        self.leg_map = {g:i for i,g in enumerate(self.ids)}
        self.ob = mj.mj_name2id(body.plant,mj.mjtObj.mjOBJ_GEOM,OBSTACLE)
        self.jac = np.zeros((3,body.plant.nv))
        self.pads = body.pads[LEG]
        with np.load(ARCHIVE/'TRACE.npz',allow_pickle=False) as arr:
            self.nominal = arr['target'].copy()
            self.nominal_pad = arr['padz'][:,RF].copy()
            self.nominal_qpos = arr['qpos'].copy()
        self.ref = self.nominal[WARM:WARM+min(walk_ms,9600):5]
        self.tree = cKDTree(self.ref[:,OTHER])
        self.ob_zero = np.zeros(8)

    def obstacle(self):
        if self.ob < 0:
            return self.ob_zero
        return ob_contacts(self.body,self.ob,self.ids)

    def sense(self, current):
        c = self.body
        m,d = c.plant,c.d
        vectors = np.zeros((3,3))
        force = np.zeros(6)
        for i in range(d.ncon):
            point = d.contact[i]
            if point.efc_address < 0:
                continue
            ga,gb = int(point.geom1), int(point.geom2)
            env_a = c.planes[ga] or ga == self.ob
            env_b = c.planes[gb] or gb == self.ob
            if env_a == env_b:
                continue
            fly = gb if env_a else ga
            k = self.leg_map.get(fly)
            if k is None:
                continue
            mj.mj_contactForce(m,d,i,force)
            vectors[k] += (point.frame.reshape(3,3).T @ force[:3]) * (1 if env_a else -1)
        rot = d.xmat[c.th].reshape(3,3)
        p = d.geom_xpos[self.pads].mean(0)
        mj.mj_jac(m,d,self.jac,None,p,int(m.geom_bodyid[self.pads[0]]))
        velocity = rot.T @ self.jac @ d.qvel / stan.LENGTH
        _, i = self.tree.query(c.actuator.target_rad[OTHER])
        departure = c.actuator.target_rad[RF_JOINTS] - self.ref[i,RF_JOINTS]
        return np.r_[(vectors @ rot / W).ravel(),velocity,departure]


def native_metrics(record, output, result, trace):
    np.testing.assert_array_equal(record.endpoints[:len(trace['qpos'])],trace['qpos'])
    np.testing.assert_allclose(record.poses[:record.count,6],trace['time'][0]+np.arange(record.count)*12.5e-6,
                               atol=5e-8,rtol=0.)
    assert not record.poses[:,7].any()
    end = min(WARM+result['walk_ms'],len(trace['qpos'])-1)
    native = record.forces[WARM*80:end*80]
    per_leg = native[:,:6].sum(0)
    absolute = native[:,6:12].sum(0)/np.maximum(1e-10,per_leg)
    speed = np.linalg.norm((trace['com'][end]-trace['com'][WARM])[:2])/((end-WARM)*.001)
    obs = record.obstacle[:record.count]
    def windows(values):
        starts=np.where(np.diff(np.r_[False,values,False].astype(int))==1)[0]
        stops=np.where(np.diff(np.r_[False,values,False].astype(int))==-1)[0]
        return np.c_[starts,stops].tolist()
    obs_active = obs[:,4:].sum(1)>0.
    unexpected = obs[:,-1]>0
    peak = float(native[:,12].max()/W)
    support = float(per_leg.sum()/len(native)/W)
    com_min=float(min(record.poses[WARM*80:end*80,2].min(),trace['com'][end,2]))
    forward=float((trace['com'][end,0]-trace['com'][WARM,0])/((end-WARM)*.001))
    slip=float(absolute.max()/speed)
    stand_report={}
    for key,a,b in (('initial',WARM-200,WARM),('terminal',end+STOP-200,end+STOP)):
        if b*80 > record.count:
            stand_report[key]=dict(passed=False)
            continue
        forces=record.forces[a*80:b*80]
        poses=record.poses[a*80:b*80]
        g = dict(support=forces[:,:6].sum()/len(forces)/W>=.9,
                 nonfoot=float(forces[:,12].max()/W)<.01,
                 com=min(poses[:,2].min(),trace['com'][b,2])>=1.2,
                 upright=float(poses[:,5].min())>=np.cos(np.deg2rad(20.)),
                 three=float(np.mean((forces[:,:6]>=.02*W).sum(1)>=3))>=.95,
                 sampled=bool(result['initial_stand' if key=='initial' else 'terminal_stand']['passed']))
        g={name:bool(v) for name,v in g.items()}
        stand_report[key]=dict(passed=all(g.values()),gates=g)
    native_gates=dict(support=support>=.9,plane_nonfoot=peak<=.20,com=com_min>=1.2,
                      forward=forward>=.05,slip=slip<=.50,cycles=all(result['cycles'].values()),
                      initial=stand_report['initial']['passed'],terminal=stand_report['terminal']['passed'])
    raw_path=output/'NATIVE.npz'
    np.savez_compressed(raw_path,forces=record.forces[:record.count],pose=record.poses[:record.count],
                        obstacle=obs)
    out=dict(native_integrations=record.count,zero_extra_force=True,endpoint_qpos_exact=True,
             contact_cross_checks=record.cross_checks,native_sha256=stan.sha(raw_path),
             raw_obstacle_names=['Fx_W','Fy_W','Fz_W','backward_resistance_W',*SEGMENTS,'other_geometry'],
             obstacle_contact_intervals_native=windows(obs_active),
             obstacle_other_contact_intervals_native=windows(unexpected),
             obstacle_resisting_impulse_Ws=float(obs[:,3].sum()*12.5e-6),
             obstacle_contact_native_steps=int(obs_active.sum()),obstacle_wrong_geom_native_steps=int(unexpected.sum()),
             ground_foot_support_mean_W=support,ground_nonfoot_peak_W=peak,minimum_com=com_min,
             signed_forward_native_s=forward,absolute_slide_native_s=dict(zip(LEGS,absolute.tolist())),
             max_ground_normalized_slip=slip,native_gates=native_gates,native_joint_pass=all(native_gates.values()),
             standing=stand_report)
    stan.put(output/'NATIVE_AUDIT.json',out)
    return out


def event_metrics(trace, scenario):
    if scenario is None:
        return dict(obstacle_present=False, triggered=False, any_obstacle_contact=False)
    ob=trace['obstacle']
    mask=ob[:,4:].sum(1)>0.
    restrained=np.flatnonzero(ob[:,3]>.002)
    kwargs=dict(obstacle_present=True, any_obstacle_contact=bool(mask.any()),triggered=bool(len(restrained)),
                obstacle_first_contact_ms=int(np.flatnonzero(mask)[0]) if mask.any() else None,
                obstacle_last_contact_ms=int(np.flatnonzero(mask)[-1]) if mask.any() else None,
                obstacle_contact_ms=int(mask.sum()), wrong_geom_contact_ms=int(np.count_nonzero(ob[:,-1])),
                obstacle_impulse_Ws=float(ob[:,3].sum()*.001))
    if not len(restrained):
        return kwargs
    first=int(restrained[0])
    after=first+np.arange(1,min(1200,len(mask)-first))
    loaded=trace['loads'][:,RF]>=.02*W
    supported=loaded[after]& ~mask[after]
    floor_contact=np.flatnonzero(np.convolve(supported,np.ones(20,dtype=int),'valid')==20) if len(after)>=20 else []
    return dict(kwargs, first_resisting_touch_ms=first,
                target_leg_load_at_touch_W=float(trace['loads'][first,RF]/W),
                minimum_other_ground_support_W=float((trace['loads'][first:first+250,OTHER_FEET].sum(1)/W).min()),
                first_target_floor_reload_ms=int(after[floor_contact[0]]) if len(floor_contact) else None,
                target_pad_peak_during_recovery=float(trace['padz'][first:first+550,RF].max()),
                target_pad_xyz_at_reload=trace['rf_pad_xyz'][int(after[floor_contact[0]])].tolist() if len(floor_contact) else None,
                thigh_min_com=float(trace['com'][first:first+550,2].min()),
                correction_accumulated=float(np.abs(trace['correction']).sum()) )


OTHER_FEET=[i for i in range(6) if i!=RF]


def run(output, mode, scenario, policy=None, native=False, walk_ms=9600, *,
        probe_ms=None, teacher_config=None, policy_source=None):
    output = Path(output).resolve()
    scenario = Path(scenario).resolve() if scenario is not None else None
    shape = scenario_read(scenario)
    output.mkdir()
    torch.set_num_threads(1)
    # Wrap original creation without touching any shared source; only this one
    # Core in this process sees the new compile.
    loader=stan.physics.act_mod.load_full_model
    if shape is not None:
        plant=make_obstacle_model(shape)
        stan.physics.act_mod.load_full_model=lambda *args,**kw: plant
    try:
        fixture=stan.Bench(output)
    finally:
        stan.physics.act_mod.load_full_model=loader
    c=fixture.body
    c.physics_order=stan.physics.rc.SENS_IDX
    np.testing.assert_array_equal(c.physics_order,bench.SENSOR_BY_LEG)
    initial=bench.STARTS/'nominal.npz'
    initial_sha=read(bench.STARTS/'BANK.json')['nominal']['sha256']
    bench.load_initial(fixture,initial,initial_sha)
    extra=[Path(__file__),HERE/'geometry.py',BASE/'walking_bench.py',BASE/'prototype.py',
           ARCHIVE/'RESULT.json',initial,bench.STOP_SCRIPT]
    if scenario is not None:
        extra.append(scenario)
    if policy_source is not None:
        extra.append(Path(policy_source))
    if teacher_config is not None:
        extra.extend((HERE/'stumble_demo.py',Path(teacher_config)))
    if native:
        extra.extend((TURNS/'verify_native.py',TURNS/'native_contact.so',TURNS/'native_contact.cpp',
                      HERE/'obstacle_contacts.cpp',HERE/'obstacle_contacts.so'))
    ident=stan.identity.capture(c,dict(scope='single front-leg stumble physical test',mode=mode,
         initial_sha256=initial_sha,scenario=shape,actuator_limit_native_s=stan.physics.MAX_RATE),
         extra_paths=extra)
    stan.put(output/'RUN_IDENTITY.json',ident)
    local=Local(c,walk_ms)
    method=None
    base=frozen_policy()
    if mode == 'teacher':
        from stumble_demo import Teacher
        method=Teacher(local,read(Path(teacher_config)),base)
    if mode == 'student' and policy is None:
        raise ValueError('Student mode requires the loaded current-sensor policy')
    walk_end=WARM+walk_ms
    duration=walk_end+STOP
    if probe_ms is not None:
        duration=min(probe_ms,duration)
    recorder=GroundBoxRecord(c,duration,local) if native else None
    c.recorder=recorder
    zero=np.zeros(42)
    begin=time.monotonic()
    unsafe=0
    trace=[]
    decisions=[]
    def snap(t,m,applied,correction):
        obs = local.obstacle()
        rgeom=np.array([c.d.geom_xpos[g].copy() for g in local.ids])
        pad=c.d.geom_xpos[local.pads].mean(0).copy()
        trace.append(dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),
             com=c.d.subtree_com[c.th].copy(),loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),
             padz=m['padz'].copy(),speed=m['speed'].copy(),foot=m['foot'].copy(),upright=float(c.d.xmat[c.th,8]),
             target=c.actuator.target_rad.copy(),ctrl=c.d.ctrl.copy(),applied=applied.copy(),correction=correction.copy(),
             obstacle=obs,rf_segments_xyz=rgeom,rf_pad_xyz=pad))
    m=c.measure()
    snap(0,m,zero,zero)
    for t in range(duration):
        if t%5==0:
            intent=int(WARM<=t<walk_end)
            x=bench.features(c,m)
            y=local.sense(m)
            b=np.asarray(base(x,intent),dtype=np.float64)
            if method is not None:
                a=method(t,x,intent,y,b,m)
            elif mode=='student':
                a=policy(x,intent,y)
            else:
                a=b.copy()
            action=bench.legacy.bounded_action(a)
            correction=action-b
            decisions.append((t,x.copy(),intent,y.copy(),b.copy(),action.copy()))
        m=c.tick(action)
        if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
            raise RuntimeError("Unexpected external wrench")
        snap(t+1,m,action,correction)
        unsafe=unsafe+1 if m['com']<.8 or m['nonfoot']>.5*W else 0
        if unsafe>=100:
            break
    tr={key:np.asarray([s[key] for s in trace]) for key in trace[0]}
    np.savez_compressed(output/'TRACE.npz',**tr)
    np.savez_compressed(output/'DECISIONS.npz',
        **{name:np.asarray(value) for name,value in zip(('ms','x','intent','local','base','action'),zip(*decisions),strict=True)})
    result=bench.metrics(tr,WARM,walk_ms,walk_end+STOP)
    result.update(start='nominal',scenario=shape,scenario_source=str(scenario) if scenario else None, mode=mode,
                  walk_ms=walk_ms, probe=probe_ms is not None, recipient_leg=LEG,sensor_fields=LOCAL_FIELDS,
                  target_segment_names=SEGMENTS,initial_snapshot_sha256=initial_sha,
                  elapsed_wall_s=time.monotonic()-begin,
                  trace_sha256=stan.sha(output/'TRACE.npz'),decision_sha256=stan.sha(output/'DECISIONS.npz'),
                  physical_native_iterations=(len(tr['time'])-1)*80,raw_native_contacts=bool(native),
                  event=event_metrics(tr,shape),teacher_engagement_ms=method.onset if method else None)
    if native:
        audit=native_metrics(recorder,output,result,tr)
        result['native_audit_sha256']=stan.sha(output/'NATIVE_AUDIT.json')
        result['native_joint_pass']=audit['native_joint_pass']
    stan.identity.verify_runtime(ident,c)
    stan.put(output/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('mode','actual_ms','joint_sampled_pass','event')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['control','teacher'])
    p.add_argument('output',type=Path)
    p.add_argument('--scenario',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--native',action='store_true')
    p.add_argument('--probe-ms',type=int)
    args=p.parse_args()
    run(args.output,args.mode,args.scenario,native=args.native,probe_ms=args.probe_ms,teacher_config=args.plan)
