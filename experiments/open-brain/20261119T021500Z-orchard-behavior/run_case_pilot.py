"""Paired multi-fruit tests on the P1 neck-coupled FlyBody."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ORCHARD = HERE.parent / '20261119T012530Z-survival-orchard'
CHOICE = HERE.parent / '20261119T001532Z-locomotion-choice'
sys.path[:0] = [str(ORCHARD), str(CHOICE)]
from orchard import build_orchard, RESOURCES
from orchard_eye import OrchardEyes, FOCAL_PX
from live_body import NEURAL, RaisedPerchMotor, STOW_ROUTES, land, perching, write_line
from walk_adapter import SharedWalk, teacher_action
from fly import WingReadout, yaw_rotation
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState

TRACE = ('time_s','x_cm','y_cm','z_cm','floor_feet','fruit_feet','nonfoot',
         'phase','landing_phase','mouth_index','external_force','yaw_rad')
MEAL = ('time_s','mouth_index','ingested','reserve','crop','absorbed','metabolized')
BRANCH = RESOURCES[2]
SEED = 29


class OrchardWalk(SharedWalk):
    def __init__(self, env, duration):
        super().__init__(env, duration)
        self.heading = 0.
        self.position = self.ref[0,:2].copy()

    def action(self, phy, speed, yaw_rate, live_observation):
        ph = self.env.physics
        ph.data.qpos[self.qcopy[0]] = phy.data.qpos[self.qcopy[1]]
        ph.data.qvel[self.vcopy[0]] = phy.data.qvel[self.vcopy[1]]
        ph.forward()
        dt = self.env.control_timestep()
        i = self.step_index
        self.heading += yaw_rate*dt
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += speed*dt*direction
        self.ref[i:,:2] = self.position + np.arange(len(self.ref)-i)[:,None]*dt*speed*direction
        self.ref[i:,3:7] = yaw_rotation(self.heading)
        self.task._step_counter = i
        updater = self.env._observation_updater
        updater.reset(self.env._physics_proxy, self.env.random_state)
        observation = updater.get_observation()
        for name in ('accelerometer','appendages_pos','force','gyro','touch','velocimeter','world_zaxis'):
            key = 'walker/'+name
            if key in live_observation and observation[key].shape == live_observation[key].shape:
                observation[key] = live_observation[key]
        target = teacher_action(self.policy,{k:observation[k] for k in self.keys},self.spec)
        self.filtered += -np.expm1(-dt/.01)*(target-self.filtered)
        self.last[self.columns] = self.filtered
        for act, j in self.action_for_act:
            if act >= 0:
                ph.data.act[act] = self.filtered[j]
        self.step_index += 1
        return self.last.copy()


class OrchardFlight(RaisedPerchMotor):
    def __init__(self, *args, guide_heading, **kwargs):
        super().__init__(*args, **kwargs)
        self.guide_heading = guide_heading

    def orient(self, quaternions):
        yaw = yaw_rotation(self.heading)
        return np.array([WingReadout.multiply(yaw, q) for q in quaternions])

    def set_reference(self, physics):
        anchor = physics.named.data.subtree_com['walker/'][:2].copy()
        _, q = self.task._walker.get_pose(physics)
        self.heading = land.angles(q)[0] if self.guide_heading is None else self.guide_heading
        super().set_reference(physics)
        c,s = np.cos(self.heading),np.sin(self.heading)
        rotation = np.array([[c,-s],[s,c]])
        self.task._ref_qpos[:,:2] = anchor + (self.task._ref_qpos[:,:2]-anchor) @ rotation.T
        self.task._ref_qpos[:,3:7] = self.orient(self.task._ref_qpos[:,3:7])
        self.task._ref_qvel[:,:2] = self.task._ref_qvel[:,:2] @ rotation.T

    def set_approach(self, physics):
        super().set_approach(physics)
        i = round(self.landing_started/self.dt)
        original = self.task._ref_qpos[i:]
        ref = original[:,:3] - land.com2root(np.zeros_like(original[:,:3]),original[:,3:7])
        pos = physics.named.data.subtree_com['walker/'].copy()
        vel = physics.bind(self.task._root_joint).qvel[:2].copy()
        tau = np.arange(len(ref))*self.dt
        attenuation = 1.-land.smooth(tau/self.approach.brake_s)
        distance = np.r_[0.,np.cumsum(.5*(attenuation[1:]+attenuation[:-1])*self.dt)]
        ref[:,:2] = pos[:2] + distance[:,None]*vel
        q = self.orient(self.task._ref_qpos[i:,3:7])
        self.task._ref_qpos[i:] = np.column_stack((land.com2root(ref,q),q))
        self.task._ref_qvel[i:,:2] = attenuation[:,None]*vel


def view_candidates(eye, items):
    _,f,u = eye.head_pose()
    f = f/np.linalg.norm(f)
    r = np.cross(f,u)
    r /= np.linalg.norm(r)
    u = np.cross(r,f)
    result = []
    for item in items:
        x0,y0,x1,y1 = item.box
        depth = item.depth_cm if item.depth_cm is not None else FOCAL_PX/(x1-x0)
        tx = np.tan(np.deg2rad(item.bearing_deg))
        ty = np.tan(np.deg2rad(item.elevation_deg))*np.hypot(1.,tx)
        ray = (f+tx*r+ty*u)*depth
        high = ray[2] > .055 and not item.clipped and y1-y0 >= 4
        result.append(dict(bearing=item.bearing_deg,elevation=item.elevation_deg,box=item.box,
                           stereo=item.depth_cm is not None,horiz=float(np.linalg.norm(ray[:2])),
                           delta_z=float(ray[2]),high=bool(high),clipped=item.clipped))
    return result


class Contacts:
    def __init__(self, phy):
        self.phy = phy
        model = phy.model
        self.fruits = {model.name2id(r.geom,'geom'):r for r in RESOURCES}
        self.floor = {model.name2id(n,'geom') for n in ('groundplane','orchard_terrain')}
        self.mouth = [model.name2id('walker/labrum_'+s+'_lower_collision','geom')
                      for s in ('left','right')]
        self.bodies = []
        for i in range(model.ngeom):
            body = model.id2name(model.geom_bodyid[i],'body') or ''
            foot = tuple(body.rsplit('_',2)[-2:]) if 'claw' in body or 'tarsus' in body else None
            self.bodies.append((body.startswith('walker/'),foot,body))

    def sample(self):
        floor,fruit,feet,claws = set(),{},set(),set()
        nonfoot = 0
        for con in self.phy.data.contact[:self.phy.data.ncon]:
            a,b = int(con.geom1),int(con.geom2)
            if self.bodies[b][0]:
                a,b = b,a
            walker,foot,body = self.bodies[a]
            if not walker or self.bodies[b][0] or con.efc_address < 0:
                continue
            if foot is None:
                nonfoot += 1
                continue
            feet.add(foot)
            if body.startswith('walker/claw_'):
                claws.add(body.removeprefix('walker/'))
            if b in self.floor:
                floor.add(foot)
            if b in self.fruits:
                resource = self.fruits[b]
                if abs(con.pos[2]-resource.top) <= perching.CONTACT_TOL_CM:
                    fruit.setdefault(resource.name,set()).add(foot)
        return floor,fruit,feet,nonfoot,claws


def run(case, out, duration):
    out.mkdir(parents=True)
    guided = case == 'GUIDED'
    env,bank = build_orchard()
    env.random_state.seed(SEED)
    env._time_limit = duration+.01
    task = env.task
    task.check_termination = lambda _:False
    step = env.reset()
    if case == 'EMPTY_GROUND':
        for r in RESOURCES:
            if r.support == 'ground':
                bank.stocks[r.name] = EMPTY
    names = {r.name:i for i,r in enumerate(RESOURCES)}
    before = {name:v.volume for name,v in bank.stocks.items()}
    phy = env.physics
    dt,native = env.control_timestep(),phy.timestep()
    n,block = round(duration/dt),round(.02/dt)
    assert n%block == 0
    task._ref_qpos = np.pad(task._ref_qpos,((0,n+task._future_steps),(0,0)),mode='edge')
    task._ref_qvel = np.pad(task._ref_qvel,((0,n+task._future_steps),(0,0)),mode='edge')
    eye = OrchardEyes(phy)
    walk = OrchardWalk(env,duration)
    motor = None

    def start_motor():
        policy,stock,observation = land.setup_policy()
        offset = BRANCH.top
        prior = json.loads((perching.LANDING/'CALIBRATED.json').read_text())
        for key in ('final_height_cm','deploy_cm','fold_precontact_cm','settled_height_cm'):
            prior[key] += offset
        prior['descent_delay_s'] = .073
        launch = land.launch_settings(SEED,duration)
        launch = replace(launch,climb_cm=launch.climb_cm+offset-.30)
        heading = np.arctan2(BRANCH.y,BRANCH.x) if guided else None
        controller = OrchardFlight(task,phy.bind(task._wing_joints).qpos.copy(),
            land.LeggedFlightTeacher(stock,observation,env),dt,launch,land.Approach(**prior),
            phy,waypoints=STOW_ROUTES['pitch'],settle_s=.070,surface_cm=offset,guide_heading=heading)
        return controller,policy,stock

    contacts = Contacts(phy)
    root = phy.bind(task._root_joint)
    engine,state = FeedingBody(),FeedingState(EMPTY)
    initial = np.r_[phy.data.qpos,phy.data.qvel]
    floor,feet,nonfoot = set(),set(),0
    stable = {r.name:0. for r in RESOURCES if r.support!='ground'}
    longest = stable.copy()
    oral = phy.data.geom_xpos[contacts.mouth].mean(axis=0)
    min_mouth = dict.fromkeys(names,np.inf)
    touched = airborne = descent = None
    oral_attempts = dict.fromkeys(names,0)
    trace,meals,notes,poses = [],[],[],[]
    first_touches,first_intake = {},{}
    intake = dict.fromkeys(names,0.)
    action = np.zeros(env.action_spec().shape)
    speed = turn = exposure = 0.
    parked = last_taste = park_action = None
    previous = task.after_substep

    def sample(physics,rng):
        nonlocal floor,feet,nonfoot,touched,airborne
        previous(physics,rng)
        floor,fruit,feet,nonfoot,claws = contacts.sample()
        com = physics.named.data.subtree_com['walker/']
        _,q = task._walker.get_pose(physics)
        yaw,roll,pitch = land.angles(q)
        launched = motor is not None and motor.started is not None
        if launched:
            motor.foot_contacts_on_fruit = len(fruit.get(BRANCH.name,()))
            motor.touched_claws = claws
            if not feet and not nonfoot and airborne is None:
                airborne = physics.time()
        for name in fruit:
            first_touches.setdefault(name,physics.time())
        for name in stable:
            ok = (launched and motor.landing_phase==4 and not nonfoot and not floor
                  and len(fruit.get(name,()))==6 and np.linalg.norm(root.qvel[:3])<.24
                  and max(abs(roll),abs(pitch))<.48)
            stable[name] = stable[name]+native if ok else 0.
            longest[name] = max(longest[name],stable[name])
        oral = physics.data.geom_xpos[contacts.mouth].mean(axis=0)
        ground_hold = (floor or any(fruit.get(r.name) for r in RESOURCES[:2])) and not nonfoot
        supported = bool(ground_hold) if not launched else any(v>=.07 for v in stable.values())
        touched = bank.contact(oral,supported)
        if touched and ((launched and stable.get(touched,0.)<.07)
                        or (not launched and touched in stable)):
            touched = None
        extra = max(np.abs(physics.data.qfrc_applied).max(),np.abs(physics.data.xfrc_applied).max())
        trace.append((physics.time(),*com,len(floor),max(map(len,fruit.values()),default=0),nonfoot,
                      motor.phase if launched else 0,motor.landing_phase if launched else 0,
                      names.get(touched,-1),extra,yaw))

    task.after_substep = sample
    with (out/'NEURAL.log').open('w') as log, subprocess.Popen(
        [str(NEURAL),'-uB',str(CHOICE/'live_brain.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=log,text=True,bufsize=1) as neural:
        write_line(neural.stdin,dict(ticks=2*n,adapter=True,cut=False,seed=SEED,stream=str(out/'NEURAL.npz')))
        ready = neural.stdout.readline()
        if not ready:
            raise RuntimeError('Neural worker failed; see NEURAL.log')
        assert json.loads(ready)['ready']
        for tick in range(n):
            now = phy.time()
            launched = motor is not None and motor.started is not None
            if tick%block == 0:
                left,right,items,_ = eye.sample()
                options = view_candidates(eye,items)
                chosen = min(options,key=lambda item:item['horiz']+2.5*item['high'],default=None)
                # Diagnostic geometry is recorded separately from the sensory packet.
                oral = phy.data.geom_xpos[contacts.mouth].mean(axis=0)
                for resource in RESOURCES:
                    horizontal = np.hypot(oral[0]-resource.x,oral[1]-resource.y)
                    gap = oral[2]-resource.top
                    min_mouth[resource.name] = min(min_mouth[resource.name],float(np.hypot(horizontal,gap)))
                    reach = resource.radius*(1. if resource.support=='ground' else .72)-.035
                    oral_attempts[resource.name] += int(horizontal<=reach and -.002<=gap<=.14)
                pos = phy.named.data.subtree_com['walker/']
                if guided:
                    _,q = task._walker.get_pose(phy)
                    aim = np.arctan2(BRANCH.y-pos[1],BRANCH.x-pos[0])-land.angles(q)[0]
                    chosen = dict(high=True,bearing=float(-np.rad2deg(aim)),
                                  horiz=float(np.hypot(BRANCH.x-pos[0],BRANCH.y-pos[1])))
                if launched and motor.engaged_air and descent is None and chosen:
                    if abs(chosen['bearing'])<19. and 1.2<chosen['horiz']<=2.65:
                        descent = now
                        Image.fromarray(left).save(out/'eye_descent.png')
                high = float(bool(chosen and chosen['high']))
                center = float(np.clip(chosen['bearing']/35.,-1.,1.)) if chosen else 0.
                front = tuple(.08 if ('T1',side) in feet else 0. for side in ('left','right'))
                row = dict(height_probability=high,center=center,seen=chosen is not None,front=front,
                           contact=touched is not None,landing_eye=descent is not None)
                write_line(neural.stdin,row)
                message = neural.stdout.readline()
                if not message:
                    raise RuntimeError('Neural worker failed; see NEURAL.log')
                packet = json.loads(message)
                commands = np.array(packet['commands'])
                ground = np.array(packet['walk'])
                feed_trace = packet['feed']
                notes.append(dict(time_s=now,selection=chosen,candidates=options,**row))
                if tick == 0:
                    Image.fromarray(left).save(out/'eye_start_left.png')
                    Image.fromarray(right).save(out/'eye_start_right.png')
            ix = tick%block
            if motor is None and np.min(commands[ix,:2])>=.22:
                motor,policy,stock = start_motor()
            if motor is not None:
                motor.update_episode(commands[ix],now,len(feet),nonfoot,phy)
            wanted = float(2.05*np.clip(ground[ix].mean()/.27,0.,1.))
            yaw_command = np.clip(5.*(ground[ix,1]-ground[ix,0]),-2.5,2.5)
            speed += -np.expm1(-dt/.032)*(wanted-speed)
            turn += -np.expm1(-dt/.055)*(yaw_command-turn)
            if touched is not None and not launched:
                last_taste = now
                exposure += dt
            else:
                exposure = max(0.,exposure-.25*dt)
            if parked is not None and now-last_taste>.16:
                parked = None
            if motor is not None and motor.started is not None:
                action = motor.action(policy,step.observation)
            elif parked is not None:
                action = park_action*(1.-land.smooth((now-parked)/.045))
            elif touched is not None and exposure>=.115:
                parked,park_action = now,action.copy()
            elif tick%10==0:
                action = walk.action(phy,speed,turn,step.observation)
            step = env.step(action.copy())
            assert not step.last()
            if (tick+1)%25==0:
                poses.append(phy.data.qpos.copy())
            if (tick+1)%5==0:
                meal = bank.advance(engine,state,5*dt,feed=feed_trace[ix],contact=touched)
                state = meal.state
                if meal.ingested.volume:
                    intake[touched] += meal.ingested.volume
                    first_intake.setdefault(touched,phy.time())
                meals.append((phy.time(),names.get(touched,-1),meal.ingested.volume,state.reserve,
                              state.crop.volume,meal.absorbed,meal.metabolized))
            if (tick+1)%round(.5/dt)==0:
                print(round(phy.time(),3),np.round(trace[-1][1:4],3).tolist(),
                      round(speed,3),round(turn,3),round(sum(intake.values()),4),flush=True)
        assert neural.wait() == 0
    trace,meals = np.array(trace),np.array(meals)
    assert len(trace)==round(duration/native) and np.isfinite(trace).all() and np.isfinite(meals).all()
    for name,value in (('TRACE',trace),('MEAL',meals),('INITIAL',initial),('QPOS',poses)):
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    (out/'EYES.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in notes))
    with np.load(out/'NEURAL.npz',allow_pickle=False) as neural:
        pulses = neural['emitted'].sum(axis=0).tolist()
        arrivals = neural['arrived'].sum(axis=0).tolist()
        ground_peak = float(neural['ground'].max())
    after = {name:v.volume for name,v in bank.stocks.items()}
    report = dict(case=case,seed=SEED,columns=TRACE,meal_columns=MEAL,dt_s=dt,native_dt_s=native,
                  native_steps=len(trace),neural_steps=2*n,brain_train_updates=0,
                  stocks_before=before,stocks_after=after,intake=intake,first_intake=first_intake,
                  fruit_first_touch_s=first_touches,fruit_stable_s=longest,first_air_s=airborne,
                  takeoff_s=motor.started if motor else None,descent_cue_s=descent,
                  landing_s=motor.landing_started if motor else None,
                  max_height_mm=float(10.*trace[:,3].max()),post_neck_ground_peak=ground_peak,
                  emitted=pulses,arrived=arrivals,max_external=float(trace[:,10].max()),
                  mass_error=max(abs(before[k]-after[k]-intake[k]) for k in intake),
                  min_xy_to_fruit_mm={r.name:float(10.*np.hypot(trace[:,1]-r.x,trace[:,2]-r.y).min())
                                      for r in RESOURCES},
                  sampled_mouth_attempts=oral_attempts,
                  min_sampled_mouth_to_fruit_mm={k:10.*v for k,v in min_mouth.items()},
                  reserve_final=state.reserve,crop_final=state.crop.volume,
                  absorbed=float(meals[:,5].sum()),metabolized=float(meals[:,6].sum()),
                  end_xyz_mm=(10.*trace[-1,1:4]).tolist())
    (out/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report,flush=True)
    if motor is not None:
        stock.close()
    walk.env.close()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=['FRESH','EMPTY_GROUND','GUIDED'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--duration',type=float)
    args = parser.parse_args()
    run(args.case,args.output.resolve(),args.duration or (2. if args.case=='GUIDED' else 6.))
