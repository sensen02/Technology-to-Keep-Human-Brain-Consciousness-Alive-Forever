"""P1 orchard energy and feeding experiment with a frozen Eon/BANC loop."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '20261119T021500Z-orchard-behavior'
sys.path.insert(0,str(PREVIOUS))
from run_case import (BRANCH, NEURAL, RESOURCES, SEED, TRACE, Contacts, OrchardEyes,
                      OrchardFlight, build_orchard, fold_until_launch, land, perching,
                      STOW_ROUTES, view_candidates, write_line)
from steering_recheck import ReversedYawWalk
from guided_motor import CurvedBranchApproach
from engine.embodied.feeding import EMPTY, FeedingBody, FeedingState

ENERGY = ('time_s','mouth_index','ingested','reserve','crop','absorbed','metabolized',
          'circulating','midgut_nutrient','crop_nutrient','excreted_nutrient','mode',
          'basal_paid','activity_paid','unmet','hunger','intake_command')
MODE = ('rest','walk','launch','flight')
BASAL_USE = .018
EXTRA_USE = (0., .012, .055, .070)
INITIAL_RESERVE = .24
HUNGER_REFERENCE = .30
EXHAUSTION_TOLERANCE = 1e-12


def run(case, out, duration, *, approach_brake=False, curved_guide=False):
    out.mkdir(parents=True)
    guided = case == 'GUIDED'
    env,bank = build_orchard()
    env.random_state.seed(SEED)
    env._time_limit = duration+.01
    task = env.task
    task.check_termination = lambda _:False
    step = env.reset()
    if case != 'FRESH':
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
    walk = ReversedYawWalk(env,duration)
    motor = None
    pattern,stow = fold_until_launch(task,phy)

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
        task._wbpg.step = pattern
        motor_type = CurvedBranchApproach if curved_guide else OrchardFlight
        controller = motor_type(task,stow,
            land.LeggedFlightTeacher(stock,observation,env),dt,launch,land.Approach(**prior),
            phy,waypoints=STOW_ROUTES['pitch'],settle_s=.070,surface_cm=offset,guide_heading=heading)
        return controller,policy,stock

    contacts = Contacts(phy)
    root = phy.bind(task._root_joint)
    engine = FeedingBody(basal_use=BASAL_USE)
    state = FeedingState(EMPTY,reserve=INITIAL_RESERVE,circulating=INITIAL_RESERVE)
    exhausted = None
    settled = 0.
    activity_time = np.zeros(len(MODE))
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
        wing_error = np.max(np.abs(physics.bind(task._wing_joints).qpos-stow))
        trace.append((physics.time(),*com,len(floor),max(map(len,fruit.values()),default=0),nonfoot,
                      motor.phase if launched else 0,motor.landing_phase if launched else 0,
                      names.get(touched,-1),extra,yaw,wing_error))

    task.after_substep = sample
    with (out/'NEURAL.log').open('w') as log, subprocess.Popen(
        [str(NEURAL),'-uB',str(HERE/'brain_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=log,text=True,bufsize=1) as neural:
        write_line(neural.stdin,dict(ticks=2*n,adapter=True,cut=False,seed=SEED,stream=str(out/'NEURAL.npz')))
        ready = neural.stdout.readline()
        if not ready:
            raise RuntimeError('Neural worker failed; see NEURAL.log')
        assert json.loads(ready)['ready']
        for tick in range(n):
            if exhausted is not None and tick%block==0:
                write_line(neural.stdin, {'finish': True})
                break
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
            if approach_brake and not launched:
                if chosen and not chosen['high']:
                    x0,_,x1,_ = chosen['box']
                    width = x1-x0
                    wanted *= np.interp(width,(165,360),(1.,.43))
                    if width>=280:
                        yaw_command *= .35
                if touched is not None:
                    wanted = 0.
                    yaw_command = 0.
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
            elif touched is not None and exposure>=(.025 if approach_brake else .115):
                parked,park_action = now,action.copy()
            elif tick%10==0:
                action = walk.action(phy,speed,turn,step.observation)
            step = env.step(action.copy())
            assert not step.last()
            if (tick+1)%25==0:
                poses.append(phy.data.qpos.copy())
            if (tick+1)%5==0:
                interval = 5*dt
                if motor is not None and motor.started is not None and motor.landing_phase < 4:
                    mode = 3 if airborne is not None else 2
                elif motor is not None and motor.started is not None:
                    mode = 0
                else:
                    moving = np.linalg.norm(root.qvel[:2]) > .30
                    mode = int(parked is None and (speed > .25 or (bool(feet) and moving)))
                activity_time[mode] += interval
                meal = bank.advance(engine,state,interval,feed=feed_trace[ix],contact=touched,
                                    activity_use=EXTRA_USE[mode])
                state = meal.state
                basal = min(meal.metabolized, BASAL_USE*interval)
                unmet = (BASAL_USE+EXTRA_USE[mode])*interval-meal.metabolized
                if state.reserve <= EXHAUSTION_TOLERANCE and exhausted is None:
                    exhausted = phy.time()
                if not nonfoot and len(feet) == 6 and np.linalg.norm(root.qvel[:3]) < .24:
                    settled += interval
                else:
                    settled = 0.
                if meal.ingested.volume:
                    intake[touched] += meal.ingested.volume
                    first_intake.setdefault(touched,phy.time())
                meals.append((phy.time(),names.get(touched,-1),meal.ingested.volume,state.reserve,
                              state.crop.volume,meal.absorbed,meal.metabolized,state.circulating,
                              state.midgut.nutrient,state.crop.nutrient,meal.excreted.nutrient,mode,
                              basal,meal.metabolized-basal,unmet,
                              1.-min(1.,state.circulating/HUNGER_REFERENCE),feed_trace[ix]))
            if (tick+1)%round(.5/dt)==0:
                print(round(phy.time(),3),np.round(trace[-1][1:4],3).tolist(),
                      round(speed,3),round(turn,3),round(sum(intake.values()),4),flush=True)
        assert neural.wait() == 0
    trace,meals = np.array(trace),np.array(meals)
    end_time = duration if exhausted is None else min(duration, np.ceil((exhausted-1e-8)/.02)*.02)
    assert len(trace)==round(end_time/native)
    assert np.isfinite(trace).all() and np.isfinite(meals).all()
    for name,value in (('TRACE',trace),('MEAL',meals),('INITIAL',initial),('QPOS',poses)):
        np.save(out/(name+'.npy'),value,allow_pickle=False)
    (out/'EYES.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in notes))
    with np.load(out/'NEURAL.npz',allow_pickle=False) as neural:
        pulses = neural['emitted'].sum(axis=0).tolist()
        arrivals = neural['arrived'].sum(axis=0).tolist()
        ground_peak = float(neural['ground'].max())
        neural_steps = len(neural['ground'])
    after = {name:v.volume for name,v in bank.stocks.items()}
    report = dict(case=case,seed=SEED,columns=TRACE,meal_columns=ENERGY,dt_s=dt,native_dt_s=native,
                  native_steps=len(trace),neural_steps=neural_steps,brain_train_updates=0,
                  horizon_s=duration,elapsed_s=end_time,reserve_initial=INITIAL_RESERVE,
                  exhausted_at_s=exhausted,activity_time_s=dict(zip(MODE,activity_time.tolist())),
                  engineering_rates=dict(basal=BASAL_USE,activity=dict(zip(MODE,EXTRA_USE))),
                  hunger_reference=HUNGER_REFERENCE,approach_brake=approach_brake,curved_guide=curved_guide,
                  basal_paid=float(meals[:,12].sum()),activity_paid=float(meals[:,13].sum()),
                  unmet=float(meals[:,14].sum()),settled_final_s=settled,
                  nutrient_mass_error=abs(INITIAL_RESERVE+.9*sum(intake.values())-state.reserve
                    -state.midgut.nutrient-state.crop.nutrient-meals[:,6].sum()-meals[:,10].sum()),
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
    parser.add_argument('--duration',type=float,default=12.)
    parser.add_argument('--approach-brake',action='store_true')
    parser.add_argument('--curved-guide',action='store_true')
    args = parser.parse_args()
    if args.duration <= 0. or abs(args.duration/.02-round(args.duration/.02))>1e-8:
        parser.error('duration must be a positive multiple of 20 ms')
    if args.curved_guide and args.case != 'GUIDED':
        parser.error('the specified branch trajectory is private to the guided control')
    run(args.case,args.output.resolve(),args.duration,
        approach_brake=args.approach_brake,curved_guide=args.curved_guide)
