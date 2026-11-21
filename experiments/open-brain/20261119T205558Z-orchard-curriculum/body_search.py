"""Reward-based body adapter search on an actual recorded post-neck drive."""
import json
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from rollout import HERE,tr

from run_case import land
land.load_policy=lru_cache(maxsize=2)(land.load_policy)
NATIVE_WALK=tr.docking.ReversedYawWalk
CLAW='adhere_claw'
DURATION=1.40

def apply(walk,phy,speed,turn,obs,params):
    pace,yaw,adhere,drive=params
    action=NATIVE_WALK.action(walk,phy,speed*pace,turn*yaw,obs)
    legs=walk.columns
    action[legs]=np.clip(action[legs]*drive,walk.spec.minimum,walk.spec.maximum)
    claws=[i for i,name in zip(legs,walk.commands) if CLAW in name]
    assert len(claws)==6
    action[claws]=np.clip(action[claws]*adhere,0.,1.)
    return action

def make(params):
    class Adapted(NATIVE_WALK):
        def action(self,phy,speed,turn,obs):
            return apply(self,phy,speed,turn,obs,params)
    return Adapted

def run(crop,env,bank,drive,params):
    env.random_state.seed(29)
    env.reset()
    phy=tr.bind_physics(env,bank)
    task=env.task
    dt=env.control_timestep()
    steps=round(DURATION/dt)
    task._ref_qpos=np.pad(task._ref_qpos,((0,steps+task._future_steps),(0,0)),mode='edge')
    task._ref_qvel=np.pad(task._ref_qvel,((0,steps+task._future_steps),(0,0)),mode='edge')
    q=tr.fit_spawn(crop,env,tr.docking.start_pose)
    tr.docking.fold_until_launch(task,phy)
    updater=env._observation_updater
    updater.reset(env._physics_proxy,env.random_state)
    obs=updater.get_observation()
    walk=make(params)(env,DURATION)
    walk.position[:]=q[:2]
    scan=tr.docking.Contacts(phy)
    th=phy.model.name2id('walker/thorax','body')
    pace=turn=0.
    motor=np.zeros(env.action_spec().shape)
    begin=None
    pos=[]; supports=[]; stands=[]; masks=[]; stability=[]; fell=False
    for i in range(steps):
        t=i*dt
        if i%10==0:
            feet,nonfoot,_=scan.scan(phy)
            up=phy.data.xmat[th,8]
            if t>.24 and len(feet)<3 and nonfoot and up<.55:
                fell=True
                break
            xy=phy.data.qpos[:2]
            pos.append(xy.copy())
            stability.append(float(up>.88 and len(feet)>=3))
            supports.append(len(feet))
            if .08<=t<.16:
                stands.append(bool(up>.94 and len(feet)>=4 and np.linalg.norm(phy.data.qvel[:2])<=.08))
            masks.append([int((leg,s) in feet) for leg in ('T1','T2','T3') for s in ('left','right')])
        sp,yw=tr.docking.decode_levels(*drive[2*i])
        sp=min(sp,.86-.25*abs(yw))
        pace+=(sp-pace)*(-np.expm1(-dt/.046))
        turn+=(yw-turn)*(-np.expm1(-dt/.082))
        if i%10==0:
            if abs(pace)+abs(turn)>.021:
                if begin is None:
                    begin=t
                    walk.position[:]=phy.data.qpos[:2]
                    walk.heading=land.angles(phy.data.qpos[3:7])[0]
                ramp=land.smooth((t-begin)/.15)
                tr.docking.bounded_reference(walk,phy)
                motor=ramp*walk.action(phy,pace*ramp,-turn*ramp,obs)
            else:
                begin=None
                walk.filtered[:]=0.
                motor[:]=0.
        step=env.step(motor.copy())
        obs=step.observation
        assert not step.last() and not np.any(phy.data.qfrc_applied) and not np.any(phy.data.xfrc_applied)
    walk.env.close()
    j=round(.16/(dt*10)); k=min(len(pos)-1,round(1.2/(dt*10)))
    delta=10*(np.asarray(pos[k])-pos[j])
    upright=float(np.mean(stability[j:k+1]))
    reloads=0
    for s in np.array(masks[j:k+1]).T:
        idx=np.flatnonzero(s)
        if len(idx)>=2 and not s[idx[0]:idx[-1]].all():
            reloads+=1
    time=float(phy.time())
    # Match the stated standing/moving gate and prefer sustained forward support.
    ok=bool(all(stands) and time>=1.2-1e-6 and np.linalg.norm(delta)>=2.5 and upright>=.9 and reloads>=2)
    reward=2*ok+min(1.4,time)+upright+np.clip(delta[0]/3,-1,1)-1.5*fell
    return dict(time_s=time,gate=ok,return_value=float(reward),advance_x_mm=float(delta[0]),
                net_mm=float(np.linalg.norm(delta)),upright=upright,legs=reloads,
                standing=all(stands),fell=bool(fell),native_steps=round(time/phy.timestep()))

def main():
    rng=np.random.default_rng(11631)
    archive=HERE.parent/'20261119T190005Z-odor-sac/LIVE_DETOUR_ODOR_L/NECK.npz'
    with np.load(archive) as z:
        for j,delay in enumerate(z['delays']):
            assert np.array_equal(z['emitted'][:-delay,j],z['arrived'][delay:,j])
        drive=z['drive']
    out=HERE/'BODY_SEARCH'
    out.mkdir(exist_ok=True)
    scenes=[]
    for site in ('CLEAR','DEBRIS'):
        crop=tr.Crop(site,'mesh')
        env,bank=tr.build(crop)
        env._time_limit=DURATION+.02
        env.task.check_termination=lambda _:False
        scenes.append((crop,env,bank))
    low=np.array([.45,.35,.15,.66]); high=np.array([1.55,1.35,1.12,1.22])
    center=np.ones(4); spread=(high-low)*.32
    history=[]
    for gen in range(2):
        population=np.vstack((center,np.clip(rng.normal(center,spread,(4,4)),low,high)))
        group=[]
        for i,w in enumerate(population):
            evaluations=[run(*scene,drive,w) for scene in scenes]
            scores=[s['return_value'] for s in evaluations]
            row=dict(gen=gen,index=i,weights=w.tolist(),sites=evaluations,
                     score=float(min(scores)+.25*np.mean(scores)))
            group.append(row); history.append(row)
            print(json.dumps(row),flush=True)
            (out/f'{gen}-{i}.json').write_text(json.dumps(row,indent=2)+'\n')
        elite=sorted(group,key=lambda r:r['score'],reverse=True)[:2]
        center=np.mean([s['weights'] for s in elite],axis=0)
        spread=np.maximum(.4*spread,np.std([s['weights'] for s in elite],axis=0))
    best=max(history,key=lambda r:r['score'])
    (out/'SELECTED.json').write_text(json.dumps(best,indent=2)+'\n')
    for _,env,_ in scenes:
        env.close()

if __name__=='__main__':
    main()
