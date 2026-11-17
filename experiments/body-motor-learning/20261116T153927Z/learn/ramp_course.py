"""Frozen full-body walking from a level approach onto a short physical incline."""
import argparse
import ctypes
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BLOCK=HERE.parent/'block'
sys.path.insert(0,str(BLOCK))
from rollout import BlockRecord, native_metrics, snapshot, stan, turn, W, LEGS
from controller import previous

PLAN=HERE/'ramp_plan.json'
INITIAL=turn.STARTS/'nominal.npz'
RAMP='terrain_ramp_incline'


def surface(x,plan):
    return np.clip(np.asarray(x)-plan['lower_x'],0.,plan['upper_x']-plan['lower_x'])*np.tan(np.deg2rad(plan['slope_degrees']))


def scene(plan):
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    loading=stan.physics.act_mod.load_full_model
    original=loading(stan.physics.PAD_XML)
    spec=_load_mjcf(stan.physics.PAD_XML)
    old_floor=original.geom('floor')
    floor=spec.geom('floor')
    floor.contype=0
    floor.conaffinity=0
    floor.rgba=[0.,0.,0.,0.]
    floor.material=''
    opts=dict(type=mj.mjtGeom.mjGEOM_BOX,contype=int(old_floor.contype[0]),
              conaffinity=int(old_floor.conaffinity[0]),condim=int(old_floor.condim[0]),
              friction=old_floor.friction,solref=old_floor.solref,solimp=old_floor.solimp,
              margin=float(old_floor.margin[0]),gap=float(old_floor.gap[0]),
              solmix=float(old_floor.solmix[0]),priority=int(old_floor.priority[0]))
    x0,x1=plan['lower_x'],plan['upper_x']
    rad=np.deg2rad(plan['slope_degrees'])
    rise=float(surface(x1,plan))
    width=plan['width_half']
    h=.25

    def landing(name,left,right,top,color):
        spec.worldbody.add_geom(name=name,size=[.5*(right-left),width,h],
                                pos=[.5*(left+right),0.,top-h],rgba=color,**opts)

    landing('terrain_ramp_approach',-10.,x0+.00015,0.,[.22,.25,.25,1.])
    landing('terrain_ramp_landing',x1-.00015,10.,rise,[.24,.30,.28,1.])
    normal=np.array([-np.sin(rad),0.,np.cos(rad)])
    top=np.array([.5*(x0+x1),0.,.5*rise])
    spec.worldbody.add_geom(name=RAMP,size=[.5*(x1-x0)/np.cos(rad),width,h],pos=top-normal*h,
                           quat=[np.cos(rad/2.),0.,-np.sin(rad/2.),0.],rgba=[.39,.48,.40,1.],**opts)
    model=spec.compile()
    stan.physics.act_mod.validate_full_model(model)
    model.mat_rgba[:original.nmat]=original.mat_rgba
    for i in range(original.ngeom):
        source=original.geom(i)
        if source.name=='floor':
            continue
        target=model.geom(source.name)
        model.geom_rgba[target.id]=original.geom_rgba[source.id]
        for key in ('type','contype','conaffinity','condim','size','pos','quat','friction','solref','solimp','margin','gap'):
            np.testing.assert_array_equal(getattr(target,key),getattr(source,key),err_msg=source.name+':'+key)
    for key in ('body_mass','body_inertia','body_ipos','body_iquat','jnt_type','jnt_range',
                'jnt_qposadr','jnt_dofadr','actuator_trnid','actuator_gear','dof_damping','dof_armature'):
        np.testing.assert_array_equal(getattr(model,key),getattr(original,key),err_msg=key)
    np.testing.assert_array_equal(model.opt.gravity,original.opt.gravity)
    stan.physics.act_mod.load_full_model=lambda *args,**kwargs:model
    try:
        body=stan.physics.Core(adhesion=False)
    finally:
        stan.physics.act_mod.load_full_model=loading
    role=np.array([2 if model.geom(i).name==RAMP else int(model.geom(i).name.startswith('terrain_ramp_'))
                   for i in range(model.ngeom)],dtype=np.int32)
    body.planes=role>0
    body.physics_order=stan.physics.rc.SENS_IDX
    checksum=turn.read(turn.STARTS/'BANK.json')['nominal']['sha256']
    assert stan.sha(INITIAL)==checksum
    with np.load(INITIAL,allow_pickle=False) as z:
        mj.mj_setState(model,body.d,z['state'],mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(model,body.d)
        body.t=int(z['t'])
        body.actuator.command_ticks=int(z['ticks'])
        for key in z.files:
            if key.startswith('core_'):
                setattr(body,key[5:],z[key].copy())
            if key.startswith('act_'):
                setattr(body.actuator,key[4:],z[key].copy())
        np.testing.assert_array_equal(body.d.qpos,z['qpos'])
        np.testing.assert_array_equal(body.d.qvel,z['qvel'])
        initial_delta=float(np.max(np.abs(stan.observe(body)-z['observation'])))
    return body,role,checksum,initial_delta


class RampRecord(BlockRecord):
    def __init__(self,core,limit,mask):
        super().__init__(core,limit,mask)
        binding=ctypes.CDLL(str(HERE/'slope_contact.so'))
        self.slope=binding.slope_contacts
        self.slope.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*3
        self.slope.restype=None
        self.floor=core.plant.geom(RAMP).id
        self.slope_values=np.zeros((limit*80,13))

    def solved(self,core,before):
        super().solved(core,before)
        if not self.last_native[13:].any():
            return
        self.slope(core.plant._address,core.d._address,self.floor,self.foot_map.ctypes.data,
                   self.velocity.ctypes.data,self.slope_values[self.count].ctypes.data)


class Reference(previous.PivotReference):
    def __init__(self,body,teacher,order,plan):
        super().__init__(body,teacher,'left_pivot',order)
        self.plan=plan
        self.teacher=teacher
        self.duration=plan['course_ms']*.001
        self.speed=plan['speed']
        self.brake=plan['brake_ms']*.001
        self.heading=self.rot0[:2,0]/np.linalg.norm(self.rot0[:2,0])
        self.holding=False

    def center(self,t):
        t=min(t,self.duration)
        distance=self.speed*(previous.ramp_integral(t)-previous.ramp_integral(max(0.,t-self.duration+self.brake),self.brake))
        center=self.origin+np.r_[distance*self.heading,0.]
        # The center is the private Thorax frame; the native COM is farther back.
        center[2]+=float(surface(center[0]-.80,self.plan))
        return center

    def finish_step(self):
        self.anchors[self.order[self.slot%6]]=self.swing_end.copy()

    def sample(self,t):
        if self.last_t is not None and not np.isclose(t-self.last_t,.005,rtol=0.,atol=1e-9):
            raise ValueError('Ramp reference requires fixed 5 ms teacher updates')
        self.last_t=t
        stance=np.ones(6,dtype=bool)
        lift=np.zeros(6)
        if t<self.duration-1e-10:
            slot,leg,u=previous.schedule(t,self.order,self.step_s)
            if slot!=self.slot:
                if self.slot>=0:
                    self.finish_step()
                self.slot=slot
                self.swing_start=self.anchors[leg].copy()
                self.swing_end=self.center((slot+1)*self.step_s+2.)+self.rot0@self.rest[leg]
            progress=previous.blend(u)
            world=self.anchors.copy()
            world[leg]=(1.-progress)*self.swing_start+progress*self.swing_end
            lift[leg]=self.lift*16.*u*u*(1.-u)*(1.-u)
            stance[leg]=False
            stage='ramp_course'
        else:
            if not self.holding:
                self.finish_step()
                self.holding=True
            world=self.anchors.copy()
            stage='landing_stop'
        center=self.center(t)
        height=surface(world[:,0],self.plan)
        self.teacher.rest_pad_bottom[:]=self.bottom0+height-(center[2]-self.origin[2])
        self.last_snapshot=dict(phase=stage,world_foot_z=height.tolist(),center=center.tolist())
        return dict(target_xy=((world-center)@self.rot0)[:,:2],target_z=lift,stance=stance,phase=stage,source_time_s=None)


def audit(trace,record,plan,walk_end,finish):
    actual=len(trace['qpos'])-1
    if actual!=finish:
        return dict(passed=False,complete=False,actual_ms=actual,native_steps=record.count)
    raw=record.slope_values[:record.count]
    block=record.block[:record.count]
    standard=native_metrics(trace,record,plan,finish)
    x0,x1=plan['lower_x'],plan['upper_x']
    moving_end=trace['com'][walk_end]
    start=trace['com'][plan['warm_ms']]
    stats={leg:dict(peak_normal_over_W=float(raw[:,index].max()/W),
                    in_contact_steps=int(np.count_nonzero(raw[:,index]>=.02*W)),
                    load_weighted_tangent_native_s=float(raw[:,index+6].sum()/max(1e-11,raw[:,index].sum())))
           for index,leg in enumerate(LEGS)}
    lo=np.flatnonzero(trace['com'][:,0]>=x0)
    hi=np.flatnonzero(trace['com'][:,0]>=x1)
    contact_legs=[leg for leg,value in stats.items() if value['in_contact_steps']>=800]
    slope_change=float(moving_end[2]-start[2])
    endpoint_delta=float(moving_end[0]-start[0])
    possible=float(surface(moving_end[0],plan)-surface(start[0],plan))
    grade=np.deg2rad(plan['slope_degrees'])
    near_landing=bool((trace['pad_min_x'][-200:] > x1+.005).all())
    checks=dict(native_joint=standard['passed'],initial_feet_on_level=bool(trace['pad_xyz'][0,:,:,0].max()+.012<x0),
                incline_load=len(contact_legs)>=3,com_across=len(lo)>0 and len(hi)>0,
                feet_landed=near_landing, no_nonfoot_slope=float(block[:,18].max())<.01*W,
                measured_rise=slope_change>=plan['minimum_rise_native'] if grade>0 else abs(slope_change)<.012)
    return dict(passed=all(checks.values()),complete=True,actual_ms=actual,native_steps=record.count,checks=checks,
                whole=standard, native_foot_ramp_normal=stats, named_ramp_support=contact_legs,
                ramp_side_impulse_W_s=float(block[:,6:18].sum()*12.5e-6/W),
                signed_forward_native=endpoint_delta,actual_ascent_native=slope_change,surface_rise_native=possible,
                first_com_x0_ms=int(lo[0]) if len(lo) else None,first_com_x1_ms=int(hi[0]) if len(hi) else None,
                terminal_all_feet_above_x1=near_landing, minimum_terminal_pad_min_x=float(trace['pad_min_x'][-200:].min()),
                slope_normal_weight_denominator_W=float(np.cos(grade)),zero_external_force=True,
                whole_slip_definition='unchanged vertical force/horizontal speed; incline has separate load and tangent foot metrics')


def run(output,flat):
    plan=turn.read(PLAN)
    if flat:
        plan['slope_degrees']=0.
    output.mkdir()
    torch.set_num_threads(1)
    start=time.monotonic()
    body,role,sha,delta=scene(plan)
    b=plan['warm_ms']
    e=b+plan['course_ms']
    finish=e+plan['stop_ms']
    record=RampRecord(body,finish,role)
    body.recorder=record
    paths=[Path(__file__),PLAN,BLOCK/'rollout.py',BLOCK/'controller.py',BLOCK/'terrain_contacts.so',
           HERE/'slope_contact.cpp',HERE/'slope_contact.so',turn.TEACHER,INITIAL]
    identity=stan.identity.capture(body,dict(scope='rigid finite gentle ramp and landing',plan=plan,
            initial_snapshot_sha256=sha,initial_observation_max_abs_delta=delta,current_controller='privileged anchored reference'),extra_paths=paths)
    stan.put(output/'RUN_IDENTITY.json',identity)
    stock=turn.straight_policy()
    refs=None
    action=np.zeros(42)
    now=body.measure()
    traces=[]
    decisions=[]
    unsafe=0

    def save():
        row=snapshot(body,now,record)
        z=row['pad_xyz'][:,:,2]-surface(row['pad_xyz'][:,:,0],plan)
        pad_ids=np.asarray([body.pads[leg] for leg in LEGS])
        row['padz']=(z-body.plant.geom_size[pad_ids,0]).min(1)
        row['applied']=action.copy()
        traces.append(row)

    save()
    for t in range(finish):
        if t%5==0:
            x=turn.features(body,now)
            if t<b:
                action=stock(x,0)
            else:
                if refs is None:
                    refs=turn.make_teacher(body,'arc_left',plan['speed'],.115)
                    refs.reference=Reference(body,refs,refs.reference.order,plan)
                action,_=refs.action(body,(t-b)*.001)
            action=np.clip(action,-1.,1.)
            decisions.append((t,x.copy(),np.array([plan['speed'] if b<=t<e else 0.,0.]),action.copy()))
        now=body.tick(action)
        if np.any(body.d.qfrc_applied) or np.any(body.d.xfrc_applied):
            raise RuntimeError('Ramp task has unexpected external force')
        save()
        unsafe=unsafe+1 if now['com']<.8 or now['nonfoot']>.5*W else 0
        if unsafe>=100:
            break
    trace={key:np.asarray([row[key] for row in traces]) for key in traces[0]}
    assert record.count==(len(traces)-1)*80 and not record.poses[:record.count,7].any()
    np.testing.assert_array_equal(record.endpoints[:len(traces)],trace['qpos'])
    report=audit(trace,record,plan,e,finish)
    np.savez_compressed(output/'TRACE.npz',**trace)
    np.savez_compressed(output/'DECISIONS.npz',**dict(zip(('ms','x','command','action'),map(np.asarray,zip(*decisions)),strict=True)))
    np.savez_compressed(output/'NATIVE.npz',forces=record.forces[:record.count],pose=record.poses[:record.count],
                        ramp_contact=record.block[:record.count],ramp_normal=record.slope_values[:record.count])
    stan.identity.verify_runtime(identity,body)
    report.update(plan=plan, angle_deg=plan['slope_degrees'],walk_ms=plan['course_ms'],warm_ms=b,
                  plane_normal=[-np.sin(np.deg2rad(plan['slope_degrees'])),0.,np.cos(np.deg2rad(plan['slope_degrees']))],
                  policy_kind='private geometry and gait phase teacher',initial_snapshot_sha256=sha,
                  elapsed_wall_s=time.monotonic()-start,
                  files={name:stan.sha(output/name) for name in ('TRACE.npz','DECISIONS.npz','NATIVE.npz','RUN_IDENTITY.json')})
    stan.put(output/'RESULT.json',report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--flat',action='store_true')
    a=p.parse_args()
    run(a.output.resolve(),a.flat)
