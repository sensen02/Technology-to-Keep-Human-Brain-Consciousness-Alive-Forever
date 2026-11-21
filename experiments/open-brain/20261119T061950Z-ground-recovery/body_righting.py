"""Execute body-only righting with calibrated short limb and wing actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from fixture import HERE, Plant, fold_until_launch, land, mj
from recover import Integrator, TRIALS, start, zero_forces
from orchard_trial import ReversedYawWalk

REFERENCE=HERE.parent/'20261119T052620Z-hunger-foraging/VERIFIED/HUNGRY/NEURAL.npz'
PULSE=.008
STAND_S=.20
CASE_ORDER=tuple(TRIALS)
TRACE_COLUMNS=('elapsed_s','x_cm','y_cm','z_cm','up','linear_speed_cm_s',
    'angular_speed_rad_s','loaded_feet','nonfoot','external','wing_fold_rad','phase')


class RightingController:
    def __init__(self,plant,workers):
        self.p=plant
        self.int=Integrator(plant)
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.library=self.read_library()
        self.trace=[]
        self.poses=[]
        self.actions=[]
        self.t0=0.
        self._force=np.empty(6)

    def read_library(self):
        p=self.p
        commands={}
        for name in ('left_side','right_side','back','orchard_fall'):
            report=json.loads((HERE/f'WING_{name}.json').read_text())
            for _,_,_,values,dwell in report['top'][:10]:
                a=np.array(values)
                assert np.all(a>=p.min) and np.all(a<=p.max)
                commands[(tuple(a),dwell)]=(a,dwell)
                mirror=a.copy()
                mirror[p.act]=a[p.act[[3,4,5,0,1,2]]]
                mirror[p.wing_act]=a[p.wing_act[[3,4,5,0,1,2]]]
                mirror[p.adh]=a[p.adh[[3,4,5,0,1,2]]]
                mirror=np.clip(mirror,p.min,p.max)
                commands[(tuple(mirror),dwell)]=(mirror,dwell)
        return tuple(commands.values())

    def predict(self,current):
        def trial(candidate):
            action,n=candidate
            evaluated=self.int.run(current,[action]*n+[self.p.ctrl_neutral]*12,PULSE)
            if evaluated is None:
                return -np.inf
            y=evaluated[0]
            stood=y[0]>.94 and y[4]>=4 and y[5]==0
            return (10.*stood+4*y[0]+6*y[1]+.25*y[4]-.2*y[5]-.09*y[2]-.03*y[3])
        scores=np.array(tuple(self.pool.map(trial,self.library)))
        if not np.isfinite(scores).any():
            raise RuntimeError('All physical candidates generated warnings')
        best=int(scores.argmax())
        return self.library[best],best

    def observe(self,phase):
        p,m,d=self.p,self.p.model,self.p.data
        contact=np.zeros(6)
        nonfoot=0
        min_load=.012*abs(m.opt.gravity[2])*m.body_subtreemass[p.th]
        for i,c in enumerate(d.contact[:d.ncon]):
            a,b=int(c.geom1),int(c.geom2)
            left,right=p.geo_leg[a],p.geo_leg[b]
            if ((left==-2)==(right==-2)) or c.efc_address<0:
                continue
            support=right if left==-2 else left
            mj.mj_contactForce(m,d,i,self._force)
            if self._force[0]<1e-8:
                continue
            if support<0:
                nonfoot+=1
            else:
                z=np.dot(c.frame.reshape(3,3)[:,2],self._force[:3])
                contact[support]+=max(0.,z if left==-2 else -z)
        max_extra=max(np.abs(d.qfrc_applied).max(),np.abs(d.xfrc_applied).max())
        wing=m.jnt_qposadr[p.wing_joint]
        row=(d.time-self.t0,*d.subtree_com[p.th],d.xmat[p.th,8],
             np.linalg.norm(d.qvel[:2]),np.linalg.norm(d.qvel[3:6]),
             int(np.count_nonzero(contact>=min_load)),nonfoot,max_extra,
             float(np.max(abs(d.qpos[wing]-p.base[wing]))),phase)
        self.trace.append(row)
        self.poses.append(d.qpos.copy())
        return row

    def execute(self,action,seconds,phase):
        p=self.p
        p.data.ctrl[:]=action
        count=round(seconds/.002)
        for _ in range(count):
            mj.mj_step(p.model,p.data,nstep=round(.002/p.model.opt.timestep))
            if (not zero_forces(p.data) or any(w.number for w in p.data.warning)
                    or not np.isfinite(p.data.qpos).all()):
                raise RuntimeError('Unsafe or nonfinite righting execution')
            mj.mj_forward(p.model,p.data)
            self.observe(phase)
        self.actions.append((seconds,phase,action.copy()))

    def handed_off(self):
        p=self.p
        env=p.env
        env._time_limit=10.
        env.task.check_termination=lambda _:False
        duration=.50
        horizon=round((p.data.time+duration)/env.control_timestep())+env.task._future_steps+2
        env.task._ref_qpos=np.pad(env.task._ref_qpos,((0,horizon),(0,0)),mode='edge')
        env.task._ref_qvel=np.pad(env.task._ref_qvel,((0,horizon),(0,0)),mode='edge')
        walk=ReversedYawWalk(env,duration)
        fold_until_launch(env.task,p.phy)
        walk.position[:]=p.data.qpos[:2]
        walk.heading=land.angles(p.data.qpos[3:7])[0]
        updater=env._observation_updater
        updater.reset(env._physics_proxy,env.random_state)
        obs=updater.get_observation()
        with np.load(REFERENCE,allow_pickle=False) as direction:
            # A prior P1 neck output restarts the usual walking executor.
            drives=direction['ground'][800:5800:2]
        dt=env.control_timestep()
        before=p.data.subtree_com[p.th,:2].copy()
        last_action=np.zeros(env.action_spec().shape)
        now=p.data.time-self.t0
        for i,drive in enumerate(drives):
            scale=land.smooth(i*dt/.16)
            speed=scale*np.clip(2.05*drive.mean()/.27,0.,.68)
            yaw=scale*np.clip(5.*(drive[1]-drive[0]),-.18,.18)
            if i%10==0:
                last_action=scale*walk.action(p.phy,float(speed),float(yaw),obs)
            step=env.step(last_action.copy())
            if step.last():
                break
            obs=step.observation
            if (i+1)%10==0:
                mj.mj_forward(p.model,p.data)
                self.observe(3)
        walk.env.close()
        length=np.linalg.norm(p.data.subtree_com[p.th,:2]-before)*10
        samples=np.array(self.trace)
        moving=samples[samples[:,0]>now]
        return dict(actual_displacement_mm=float(length),min_up=float(moving[:,4].min()),
                    max_nonfoot=int(moving[:,8].max()),provenance=str(REFERENCE),
                    passed=bool(length>=1. and moving[:,4].min()>.90 and moving[:,8].max()==0))

    @staticmethod
    def steady(rows):
        needed=round(STAND_S/.002)
        if len(rows)<needed:
            return False
        tail=np.array(rows[-needed:])
        return bool(np.all((tail[:,4]>=.94)&(tail[:,5]<=.03)&(tail[:,6]<=1.)
                           &(tail[:,7]>=4)&(tail[:,8]==0.)&(tail[:,9]==0.)))

    def run(self,name,out,walk=False):
        self.trace,self.poses,self.actions=[],[],[]
        p=self.p
        seed,up,_=start(p,name)
        self.t0=p.data.time
        init=self.observe(0)
        if init[4]>.55 or (init[7]>=3 and init[8]==0):
            raise RuntimeError(f'{name}: did not initialise as fallen ({init})')
        choices=[]
        stood=False
        for _ in range(5):
            (action,ticks),i=self.predict(p.state())
            choices.append(i)
            self.execute(action,ticks*PULSE,1)
            self.execute(p.ctrl_neutral,.10,2)
            top=self.trace[-1]
            if top[4]>.90 and top[7]>=4 and not top[8]:
                self.execute(p.ctrl_neutral,.30,2)
            stood=self.steady(self.trace)
            print(name,'pulse',i,'elapsed',round(self.trace[-1][0],3),
                  'up',round(self.trace[-1][4],3),'loaded',self.trace[-1][7],
                  'stand',stood,flush=True)
            if stood:
                break
        upright_at=self.trace[-1][0] if stood else None
        final_before_walk=p.state()
        handoff=self.handed_off() if stood and walk else None
        out.mkdir()
        np.save(out/'QPOS.npy',np.asarray(self.poses),allow_pickle=False)
        np.save(out/'TRACE.npy',np.asarray(self.trace),allow_pickle=False)
        np.save(out/'INITIAL_STATE.npy',seed,allow_pickle=False)
        np.save(out/'STOOD_STATE.npy',final_before_walk,allow_pickle=False)
        np.savez_compressed(out/'ACTIONS.npz',hold=[v[0] for v in self.actions],
                            phase=[v[1] for v in self.actions],
                            commands=np.array([v[2] for v in self.actions]))
        rec=dict(case=name,passed=stood,initial_up=up,initial_tilt_deg=float(np.rad2deg(np.arccos(up))),
            upright_confirmed_s=upright_at,loaded_at_end=int(self.trace[-1][7]),selected_pulses=choices,
            handoff=handoff,zero_external=not np.any(np.asarray(self.trace)[:,9]),
            trace_columns=TRACE_COLUMNS,posture_samples=len(self.poses),library_size=len(self.library))
        (out/'RESULT.json').write_text(json.dumps(rec,indent=2)+'\n')
        print(rec,flush=True)


def main(args):
    args.root.mkdir()
    plant=Plant()
    controller=RightingController(plant,args.workers)
    for index,name in enumerate(args.cases):
        if index and args.handoff:
            controller.pool.shutdown()
            plant.env.close()
            plant=Plant()
            controller=RightingController(plant,args.workers)
        controller.run(name,args.root/name,walk=args.handoff)
    controller.pool.shutdown()
    plant.env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('cases',nargs='+',choices=CASE_ORDER)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--handoff',action='store_true')
    args=parser.parse_args()
    main(args)
