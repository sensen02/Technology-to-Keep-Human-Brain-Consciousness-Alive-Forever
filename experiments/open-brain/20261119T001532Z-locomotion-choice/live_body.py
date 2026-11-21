"""One physical FlyBody run with real sensory feedback, ground motion and takeoff."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'20261119T000140Z-perch-wing-audit'))
sys.path.insert(0,str(HERE.parents[2]))
from perch_wing import ClearAbdomenLanding, STOW_ROUTES, perching, land
from engine.embodied.feeding import FeedingBody, FeedingState, Food
from common_eye import scene, Retina, HeightClassifier, LOW_CM
from walk_adapter import SharedWalk

COLUMNS = ('time_s','x_cm','y_cm','z_cm','floor_feet','fruit_feet','nonfoot',
           'phase','wing_mix','landing_phase','mouth_contact','intake','external',
           'height_probability','forward_cm_s','body_pitch_rad')
FP_BLOCK = .02
NEURAL = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')


class RaisedPerchMotor(ClearAbdomenLanding):
    """Use abdomen joint actuators to keep the rear clear of the fruit."""
    def __init__(self, *args, surface_cm, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics = args[-1]
        self.surface_cm = surface_cm
        model = self.physics.model
        names = self.adapter.legged_spec.name.split('\t')
        axes = ('abdomen', *(f'abdomen_{i}' for i in range(2,8)))
        self.abdomen_action = names.index('abdomen')
        joints = [model.name2id('walker/'+name, 'joint') for name in axes]
        self.abdomen_dofs = model.jnt_dofadr[joints]
        self.abdomen_geoms = [model.name2id(f'walker/abdomen_{i}_collision', 'geom') for i in (6, 7)]
        self.response = 0.
        self.jac = np.zeros((3,model.nv))
        self.foot_contacts_on_fruit = 0

    def action(self, policy, observation):
        action = super().action(policy, observation)
        if self.fold_started is None or self.foot_contacts_on_fruit < 4:
            return action
        model, data = self.physics.model, self.physics.data
        a, b = self.abdomen_geoms
        vertical = abs(data.geom_xmat[a, 8])
        radius, half_length = model.geom_size[a, :2]
        low_a = data.geom_xpos[a, 2] - radius*np.sqrt(max(0.,1.-vertical**2)) - half_length*vertical
        low_b = data.geom_xpos[b, 2] - model.geom_size[b, 0]
        point = a if low_a < low_b else b
        clearance = min(low_a,low_b)-self.surface_cm
        mj.mj_jacGeom(model.ptr,data.ptr,self.jac,None,point)
        # The stock tendon sums these seven joint angles with equal moment arms.
        jac = float(self.jac[2,self.abdomen_dofs].mean())
        error = .012-clearance
        self.response += self.dt*(36.*error*jac/(jac*jac+.00015)-2.5*self.response)
        self.response = np.clip(self.response,-.46,.46)
        spec = self.adapter.legged_spec
        i = self.abdomen_action
        action[i] = np.clip(action[i]+self.response,spec.minimum[i],spec.maximum[i])
        return action


def write_line(pipe, packet):
    pipe.write(json.dumps(packet,separators=(',',':'))+'\n')
    pipe.flush()


def run(args):
    out=args.output.resolve()
    out.mkdir()
    duration=args.duration
    top=perching.TOP_CM if args.raised else LOW_CM
    env=scene(args.x,args.raised)
    task=env.task
    setup=task.initialize_episode
    env._time_limit=duration+.001
    env.random_state.seed(args.seed)

    def start(phy,rng):
        setup(phy,rng)
        land.set_ground_pose(env)
        task._ref_qpos[:,:3]=task._walker.get_pose(phy)[0]
        task._ref_qpos[:,3:7]=(1,0,0,0)
        task._ref_qvel[:]=0.

    task.initialize_episode=start
    task.check_termination=lambda _:False
    step=env.reset()
    phy=env.physics
    dt=env.control_timestep()
    block=round(FP_BLOCK/dt)
    n=round(duration/dt)
    assert n % block == 0
    task._ref_qpos=np.pad(task._ref_qpos,((0,n+task._future_steps),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,n+task._future_steps),(0,0)),mode='edge')
    stock=land.make_env()
    observation=stock.reset().observation
    policy=land.load_policy(land.BASELINE/'flight_reference/flight')
    walk=SharedWalk(env,duration)
    launch=land.launch_settings(args.seed,duration)
    prior=json.loads((perching.LANDING/'CALIBRATED.json').read_text())
    prior.update(final_height_cm=prior['final_height_cm']+top,deploy_cm=prior['deploy_cm']+top,
                 descent_delay_s=.073,fold_precontact_cm=prior['fold_precontact_cm']+top,
                 settled_height_cm=prior['settled_height_cm']+top)
    motor=RaisedPerchMotor(task,phy.bind(task._wing_joints).qpos.copy(),
                              land.LeggedFlightTeacher(stock,observation,env),dt,launch,land.Approach(**prior),
                              phy,waypoints=STOW_ROUTES['pitch'],settle_s=.070,surface_cm=top)
    retina=Retina(phy)
    vision=HeightClassifier(HERE/'COMMON_EYE_FIT.json')
    landing_eye=perching.FruitEye(phy,retina.head,False)
    ground=phy.model.name2id('groundplane','geom')
    ripe=phy.model.name2id('fruit','geom')
    mouth_ids=[phy.model.name2id('walker/labrum_'+side+'_lower_collision','geom') for side in ('left','right')]
    root=phy.bind(task._root_joint)
    native=phy.timestep()
    state=FeedingState(Food(1.8,1.62))
    meal=FeedingBody()
    intake=0.
    first_meal=None
    first_contact=None
    feed=False
    speed=0.
    choice=0.
    max_z=-np.inf
    start_air=None
    stable=0.
    tasted=False
    feet=6
    hits=0
    tour=[]
    poses=[]
    observed=[]
    feet_on_floor=set()
    old_sample=task.after_substep
    action=np.zeros(env.action_spec().shape)
    grounded_meal_started=None
    arrival_action=None
    last_ground_taste=None
    ground_taste_exposure=0.
    initial=np.r_[phy.data.qpos,phy.data.qvel]

    def sample(physics,rng):
        nonlocal feet,hits,stable,tasted,max_z,feet_on_floor,first_contact,start_air
        old_sample(physics,rng)
        feet_on_floor,_,floor_n,floor_bodies=perching.support(physics,ground)
        held,sides,fruit_n,fruit_bodies=perching.support(physics,ripe,top)
        motor.foot_contacts_on_fruit=len(held)
        hits=floor_n+fruit_n
        feet=len(feet_on_floor|held|sides)
        motor.touched_claws=tuple(body.removeprefix('walker/') for body in floor_bodies|fruit_bodies
                                  if body.startswith('walker/claw_'))
        com=physics.named.data.subtree_com['walker/']
        height=com[2]
        max_z=max(max_z,height)
        _,quat=task._walker.get_pose(physics)
        _,roll,pitch=land.angles(quat)
        if held and first_contact is None:
            first_contact=physics.time()
        if motor.started is not None and feet==0 and hits==0 and start_air is None:
            start_air=physics.time()
        support_ok=(len(held)==6 and not feet_on_floor and hits==0 and motor.landing_phase==4
                    and np.linalg.norm(root.qvel[:3])<.24 and max(abs(roll),abs(pitch))<.48)
        stable=stable+native if support_ok else 0.
        oral=physics.data.geom_xpos[mouth_ids].mean(axis=0)
        on_patch=np.hypot(oral[0]-args.x,oral[1])<=perching.RADIUS_CM-.015
        gap=oral[2]-top
        mode_ok=((motor.started is None and bool(feet_on_floor) and hits==0)
                 or stable>=perching.STABLE_S)
        tasted=bool(mode_ok and on_patch and -.002<=gap<=perching.MOUTH_REACH_CM)
        ext=max(np.abs(physics.data.qfrc_applied).max(),np.abs(physics.data.xfrc_applied).max())
        tour.append([physics.time(),*com,len(feet_on_floor),len(held),hits,motor.phase,motor.wing_mix,
                     motor.landing_phase,tasted,intake,ext,choice,speed,pitch])

    task.after_substep=sample
    with (out/'NEURAL.log').open('w') as log, subprocess.Popen(
        [str(NEURAL),'-uB',str(HERE/'live_brain.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=log,text=True,bufsize=1) as neural:
        write_line(neural.stdin,dict(ticks=2*n,adapter=args.adapter,cut=args.cut,
                                    seed=args.seed,stream=str(out/'NEURAL.npz')))
        assert json.loads(neural.stdout.readline())['ready']
        for tick in range(n):
            now=phy.time()
            q=task._walker.get_pose(phy)[1]
            if motor.engaged_air and .21<=now<=.7 and tick%25==0:
                landing_eye.sample(now,q)
            if tick%block==0:
                image,center,width=retina.features(land.angles(q)[0])
                choice=vision.predict(image)
                front=(.08 if ('T1','left') in feet_on_floor else 0.,
                       .08 if ('T1','right') in feet_on_floor else 0.)
                seen=width>0.
                observed.append([now,choice,float(seen),center,width])
                write_line(neural.stdin,dict(height_probability=choice,center=center,seen=seen,
                                            front=front,contact=tasted,landing_eye=landing_eye.candidate_s is not None))
                lines=neural.stdout.readline()
                if not lines:
                    raise RuntimeError('Neural worker failed. See NEURAL.log.')
                packet=json.loads(lines)
                commands=np.array(packet['commands'])
                ground_command=np.array(packet['walk'])
                feed_trace=packet['feed']
            ix=tick%block
            feed=feed_trace[ix]
            motor.update_episode(commands[ix],now,feet,hits,phy)
            wanted=float(2.05*np.clip(ground_command[ix].mean()/.27,0.,1.))
            speed+=-np.expm1(-dt/.032)*(wanted-speed)
            if tasted and motor.started is None:
                last_ground_taste=now
                ground_taste_exposure+=dt
            else:
                ground_taste_exposure=max(0.,ground_taste_exposure-.25*dt)
            if grounded_meal_started is not None and now-last_ground_taste>.16:
                grounded_meal_started=None
            if motor.started is not None:
                action=motor.action(policy,step.observation)
            elif grounded_meal_started is not None:
                action=arrival_action*(1.-land.smooth((now-grounded_meal_started)/.045))
            elif tasted and ground_taste_exposure>=.115:
                grounded_meal_started=now
                arrival_action=action.copy()
            elif tick%10==0:
                action=walk.action(phy,speed,step.observation)
            step=env.step(action.copy())
            assert abs(phy.time()-(tick+1)*dt)<1e-7 and not step.last()
            poses.append(phy.data.qpos.copy())
            if (tick+1)%5==0:
                row=meal.advance(state,5*dt,feed=feed,contact=tasted)
                state=row.state
                intake+=row.ingested.volume
                if row.ingested.volume and first_meal is None:
                    first_meal=phy.time()
            if (tick+1)%round(.5/dt)==0:
                print(round(phy.time(),3),np.round(tour[-1][1:4],3).tolist(),
                      round(choice,3),round(speed,3),round(intake,4),flush=True)
        assert neural.wait()==0
    values=np.asarray(tour)
    assert np.isfinite(values).all()
    np.save(out/'TRACE.npy',values,allow_pickle=False)
    np.save(out/'QPOS.npy',np.asarray(poses),allow_pickle=False)
    np.save(out/'INITIAL.npy',initial,allow_pickle=False)
    np.save(out/'OBS.npy',np.asarray(observed),allow_pickle=False)
    r=np.load(out/'NEURAL.npz',allow_pickle=False)
    fact=dict(columns=COLUMNS,mode='elevated' if args.raised else 'ground',
              adapter=args.adapter,cut=args.cut,geometry=dict(top_cm=top,radius_cm=perching.RADIUS_CM,x_cm=args.x,y_cm=0.),
              output_flystream='live 0.02 s sensory/control blocks on one MuJoCo body',
              native_dt_s=float(native),duration_s=duration,seed=args.seed,
              takeoff_s=motor.started,first_air_s=start_air,visual_trigger_s=landing_eye.candidate_s,landing_s=motor.landing_started,
              fork_rise_s=motor.fold_started,first_fruit_contact_s=first_contact,
              first_all_six_s=float(values[np.flatnonzero(values[:,5]==6)[0],0]) if np.any(values[:,5]==6) else None,
              max_height_cm=float(max_z),source_spikes=r['emitted'].sum(axis=0).tolist(),
              arrivals=r['arrived'].sum(axis=0).tolist(),last_support=values[-1,4:7].tolist(),
              x_end_mm=float(10*values[-1,1]),first_intake_s=first_meal,intake=float(intake),
              grounded_meal_started_s=grounded_meal_started,
              fruit_remaining=float(state.patch.volume),brain_train_updates=0,
              external_force_max=float(values[:,12].max()))
    (out/'RESULT.json').write_text(json.dumps(fact,indent=2,allow_nan=False)+'\n')
    print(fact,flush=True)
    env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--duration',type=float,default=1.62)
    p.add_argument('--x',type=float,default=8.03)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--raised',action='store_true')
    p.add_argument('--no-adapter',action='store_false',dest='adapter')
    p.add_argument('--cut',action='store_true')
    run(p.parse_args())
