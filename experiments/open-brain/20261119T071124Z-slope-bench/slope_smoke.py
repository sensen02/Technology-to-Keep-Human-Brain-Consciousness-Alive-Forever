"""Controlled P1 FlyBody incline bench with recorded post-neck drive."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / '20261119T021500Z-orchard-behavior'))
from run_case import SEED, fold_until_launch, land
from steering_recheck import ReversedYawWalk
from legged_fixture import build_env

SOURCE = PARENT / '20261119T052620Z-hunger-foraging/VERIFIED'
LEG_ORDER = [(leg, side) for side in ('left', 'right') for leg in ('T1', 'T2', 'T3')]
WARMUP, WALK_END, STOP_END, END = .20, .94, 1.54, 2.00
ENGINEERING_GOAL_MM = 2.5
TRACE_COLUMNS = ('t', 'com_x', 'com_y', 'com_z', 'vx', 'vy', 'vz', 'up_to_plane',
                 'clearance_cm', 'loaded_feet', 'nonfoot', 'external', 'adhesion_active',
                 'actuator_effort', 'command_cm_s')
ANGLES = (-20, -10, 0, 10, 20)


def inputs(dt):
    with np.load(SOURCE/'HUNGRY/NEURAL.npz') as data:
        active = data['ground']
    with np.load(SOURCE/'FULL/NEURAL.npz') as data:
        rest = data['ground']
    output = []
    origins = []
    for begin, end, origin, series in ((0., WARMUP, .20, rest), (WARMUP,WALK_END, .33,active),
                                        (WALK_END,STOP_END,.3,rest), (STOP_END,END,1.71,active)):
        idx = round(origin/.0001)+round(dt/.0001)*np.arange(round((end-begin)/dt))
        output.append(series[idx])
        origins.extend(idx)
    return np.vstack(output), np.array(origins)


def make_environment(angle, yaw):
    env = build_env()
    env.random_state.seed(SEED)
    env._time_limit = END + .1
    env.task.check_termination = lambda _: False
    plane = env.task._arena.mjcf_model.find('geom', 'groundplane')
    tilt = Rotation.from_euler('y', -angle, degrees=True)
    quat = tilt.as_quat()
    plane.quat = quat[[3,0,1,2]]
    plane.pos = (0., 0., 0.)
    plane.rgba = (.62,.67,.56,1.)
    env.reset()
    task = env.task
    physics = env.physics
    dt = env.control_timestep()
    steps = round(END/dt) + task._future_steps + 3
    task._ref_qpos = np.pad(task._ref_qpos, ((0,steps),(0,0)), mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel, ((0,steps),(0,0)), mode='edge')
    n = physics.model.nq
    start = np.load(SOURCE/'FADE/INITIAL.npy',allow_pickle=False)[:n].copy()
    root_start = Rotation.from_quat(start[[4,5,6,3]])
    relative = Rotation.from_euler('z',yaw,degrees=True) * Rotation.from_euler(
        'z', -land.angles(start[3:7])[0]) * root_start
    start[:3] = tilt.apply([0.,0.,.1378])
    orient = (tilt*relative).as_quat()
    start[3:7] = orient[[3,0,1,2]]
    with physics.reset_context():
        physics.data.qpos[:] = start
        physics.data.qvel[:] = 0.
        physics.data.ctrl[:] = 0.
        physics.data.act[:] = 0.
    fold_until_launch(task,physics)
    upd = env._observation_updater
    upd.reset(env._physics_proxy, env.random_state)
    return env, upd.get_observation(), tilt.apply([0.,0.,1.]), tilt.apply([1.,0.,0.])


class Contacts:
    def __init__(self, physics, normal):
        self.phy = physics
        self.model, self.data = physics.model.ptr, physics.data.ptr
        self.normal = normal
        self.ground = self.model.geom('groundplane').id
        self.th = self.model.body('walker/thorax').id
        self.body_geom = self.model.geom_bodyid
        self.foot = {}
        for j in range(self.model.nbody):
            name = self.model.body(j).name
            if name.startswith('walker/') and ('claw_' in name or 'tarsus' in name):
                ident = tuple(name.rsplit('_',2)[-2:])
                if ident in LEG_ORDER:
                    self.foot[j] = LEG_ORDER.index(ident)
        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.adhesive = [i for i,n in enumerate(names) if '/adhere_claw_' in n]
        assert len(self.adhesive) == 6 and len(set(self.foot.values())) == 6
        self.weight = abs(self.model.opt.gravity[2])*self.model.body_subtreemass[self.th]
        self.f = np.zeros(6)
        self.velocity = np.zeros(6)
        self.native_nonfoot = 0
        self.native_max_external = 0.
        self.count = 0
        self.loads = np.zeros(6)
        self.slip = np.zeros(6)
        self.leg_force = np.zeros((6,3))
        self.last_positions = np.zeros((6,3))

    def update(self):
        m,d=self.model,self.data
        full = self.count % 4 == 0
        self.count += 1
        nonfoot = 0
        if full:
            self.loads.fill(0.)
            self.slip.fill(0.)
            self.leg_force.fill(0.)
        for index,c in enumerate(d.contact[:d.ncon]):
            if c.efc_address < 0:
                continue
            if c.geom1 == self.ground:
                limb = int(self.body_geom[c.geom2])
                sign = 1.
            elif c.geom2 == self.ground:
                limb = int(self.body_geom[c.geom1])
                sign = -1.
            else:
                continue
            leg = self.foot.get(limb)
            if not full and leg is not None:
                continue
            mj.mj_contactForce(m,d,index,self.f)
            if self.f[0] <= 1e-8:
                continue
            if leg is None:
                nonfoot += 1
                continue
            force = sign*c.frame.reshape(3,3).T@self.f[:3]
            load = max(0.,float(force@self.normal))
            if load == 0.:
                continue
            mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_XBODY,limb,self.velocity,0)
            point = c.pos-d.xpos[limb]
            w = self.velocity[:3]
            velocity = self.velocity[3:]+np.array((w[1]*point[2]-w[2]*point[1],
                  w[2]*point[0]-w[0]*point[2],w[0]*point[1]-w[1]*point[0]))
            velocity -= (velocity@self.normal)*self.normal
            self.loads[leg] += load
            self.slip[leg] += load*np.linalg.norm(velocity)
            self.leg_force[leg] += force
            self.last_positions[leg] = c.pos
        self.native_nonfoot += bool(nonfoot)
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            raise RuntimeError('Additional applied force')
        if any(w.number for w in d.warning):
            raise RuntimeError('MuJoCo emitted a physics warning')
        return full, nonfoot

    def feet(self):
        return self.loads >= .012*self.weight


def checks(trace, direction, nontouch):
    progress=10.*(trace[:,1:4]-trace[0,1:4])@direction
    axis_velocity=10.*np.linalg.norm(trace[:,4:7]-(trace[:,4:7]@np.cross(direction,[0.,1.,0.]))[:,None]
                    *np.cross(direction,[0.,1.,0.]),axis=1)
    t=trace[:,0]
    stop=(t>=STOP_END-.2)&(t<STOP_END)
    initial=np.searchsorted(t,WARMUP)
    end_walk=np.searchsorted(t,WALK_END)
    end_stop=np.searchsorted(t,STOP_END)
    walk=float(progress[end_walk]-progress[initial])
    restart=float(progress[-1]-progress[end_stop])
    gates=dict(forward=walk>=ENGINEERING_GOAL_MM,no_nonfoot=nontouch==0,stop=bool(
        np.all((axis_velocity[stop]<=.3)&(trace[stop,9]>=4)&(trace[stop,7]>=.94))),
        restart=restart>=1.)
    return dict(gates=gates,passed=all(gates.values()),advance_mm=walk,restart_mm=restart,
                stop_peak_tangent_mm_s=float(axis_velocity[stop].max()),
                stop_min_feet=int(trace[stop,9].min()),
                min_clearance_mm=float(10*trace[:,8].min()),
                plane_alignment_min=float(trace[:,7].min()),native_nonfoot_samples=nontouch)


def run(angle,yaw,target):
    target.mkdir()
    env,obs,normal,tangent = make_environment(angle,yaw)
    phy,task = env.physics,env.task
    dt,native=env.control_timestep(),phy.timestep()
    walk = ReversedYawWalk(env,END)
    walk.position[:] = phy.data.qpos[:2]
    walk.heading=np.deg2rad(yaw)
    contacts=Contacts(phy,normal)
    original=task.after_substep
    rows, poses, legs, actions = [], [], [], []
    speed = 0.
    control = np.zeros(env.action_spec().shape)
    command, origins = inputs(dt)
    prev_walk = False

    def sample(physics,rng):
        original(physics,rng)
        full, nonfoot=contacts.update()
        if not full:
            return
        d=contacts.data
        th=contacts.th
        position=d.subtree_com[th].copy()
        rows.append([d.time,*position,*d.qvel[:3],d.xmat[th,6:9]@normal,position@normal,
                     np.count_nonzero(contacts.feet()), nonfoot,0.,
                     np.count_nonzero(d.actuator_force[contacts.adhesive]>.001),
                     np.dot(abs(d.actuator_force),abs(d.actuator_velocity)),speed])
        legs.append(np.r_[contacts.loads/contacts.weight,
                          contacts.slip/contacts.weight,contacts.leg_force.ravel()/contacts.weight,
                          d.actuator_force[contacts.adhesive]/contacts.weight])

    task.after_substep = sample
    initial = np.r_[phy.data.qpos.copy(), phy.data.qvel.copy()]
    for tick, drive in enumerate(command):
        desired = .85*np.clip(drive.mean()/.27,0.,1.)
        speed += -np.expm1(-dt/.032)*(desired-speed)
        if tick%10==0:
            active=speed>.03
            if active:
                if not prev_walk:
                    walk.position[:]=phy.data.qpos[:2]
                    walk.heading=np.deg2rad(yaw)
                control=walk.action(phy,speed,0.,obs)
            else:
                control=np.zeros_like(control)
                walk.filtered[:]=0.
            prev_walk=active
        step=env.step(control.copy())
        if step.last():
            raise RuntimeError('Unexpected reset')
        obs=step.observation
        actions.append(control.copy())
        if (tick+1)%25==0:
            poses.append(phy.data.qpos.copy())
    trace=np.array(rows)
    legs=np.array(legs)
    assert contacts.count==round(END/native)
    stance_load=legs[:,:6]
    slip=legs[:,6:12]
    native_slip=np.divide(10.*slip.sum(axis=0),stance_load.sum(axis=0),
        out=np.full(6,np.nan),where=stance_load.sum(axis=0)>0)
    result=checks(trace,tangent,contacts.native_nonfoot)
    result.update(angle_deg=angle,heading_deg=yaw,normal=normal.tolist(), native_steps=contacts.count,
        physics_dt_s=native, measurement_dt_s=4*native, control_dt_s=dt,trace_columns=TRACE_COLUMNS,
        weighted_leg_slip_mm_s=[float(x) if np.isfinite(x) else None for x in native_slip],
        min_native_extra_force=0., ready_to_walk_before=bool(trace[np.searchsorted(trace[:,0],WARMUP)-1,9]>=4))
    np.savez_compressed(target/'TRAJECTORY.npz', trace=trace,pose=poses,feet=legs,actions=actions,
                        drive=command,source_index=origins,initial=initial)
    (target/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(angle,yaw,result['gates'],round(result['advance_mm'],3),round(result['restart_mm'],3),
          result['native_nonfoot_samples'],flush=True)
    walk.env.close()
    env.close()
    return result


def main(args):
    args.out.mkdir()
    all_results = {}
    for angle in args.angles:
        for yaw in args.yaws:
            name=f'{angle:+03d}deg_yaw{yaw:+03d}'
            all_results[name]=run(angle,yaw,args.out/name)
    (args.out/'RESULTS.json').write_text(json.dumps(all_results,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--angles',type=int,nargs='+',default=ANGLES)
    parser.add_argument('--yaws',type=int,nargs='+',default=[0])
    main(parser.parse_args())
