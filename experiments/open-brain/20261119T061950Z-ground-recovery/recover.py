"""Body-side short-horizon righting on the fixed orchard contact model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
from pathlib import Path
import threading

import numpy as np

from fixture import HERE, Plant, mj

STEP=.04
TRIALS={
    'left_side':(105,0,0,(1,0)),
    'right_side':(-105,0,0,(1,.06)),
    'back':(180,0,0,(1,.12)),
    'orchard_fall':None,
    'test_left':(97,-13,19,(1.8,.43)),
    'test_right':(-102,14,-23,(2.1,.6)),
    'test_back':(173,-11,38,(1.35,-.28)),
    'test_diagonal':(-144,21,34,(1.7,-.4)),
    'test_shallow_left':(89,7,17,(1.95,.28)),
    'test_shallow_right':(-92,-9,-18,(1.67,-.45)),
    'test_dorsal':(179,6,-36,(1.51,.36)),
}
SOURCE=HERE.parent/'20261119T052620Z-hunger-foraging/VERIFIED/FADE/QPOS.npy'


def ptr(array):
    return None if array is None else array.ctypes.data


def zero_forces(d):
    return not (np.any(d.qfrc_applied) or np.any(d.xfrc_applied))


class Integrator:
    def __init__(self,p):
        self.p=p
        self.m=p.model
        self.lib=ct.CDLL(str(HERE/'native.so'))
        self.fn=self.lib.simulate
        self.fn.argtypes=[ct.c_void_p]*4+[ct.c_int]*3+[ct.c_void_p]*4+[ct.c_int]
        self.fn.restype=ct.c_int
        self.local=threading.local()

    def run(self,state,goals,seconds=STEP,save=False):
        if not hasattr(self.local,'d'):
            self.local.d=mj.MjData(self.m)
        d=self.local.d
        steps=round(seconds/self.m.opt.timestep)
        goals=np.ascontiguousarray(goals,dtype=float).reshape(-1,self.m.nu)
        out=np.empty((len(goals)*steps//100,self.m.nq)) if save else None
        next_state=np.empty_like(state)
        metrics=np.empty(8)
        bad=self.fn(self.m._address,d._address,ptr(state),ptr(goals),len(goals),steps,
              self.p.th,ptr(self.p.geo_leg),ptr(metrics),ptr(next_state),ptr(out),100)
        if bad or not zero_forces(d):
            return None
        return metrics,next_state,out


class RightingSearch:
    def __init__(self,p,workers=6):
        self.p=p
        self.int=Integrator(p)
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.d=mj.MjData(p.model)
        self.low=p.min[p.act]
        self.high=p.max[p.act]
        self.library=[]

    def ik(self,state,gen):
        p,d=self.p,self.d
        m=p.model
        p.put(state,d)
        rot=d.xmat[p.th].reshape(3,3).copy()
        centers=d.xpos[p.foot_body].copy()
        center=d.xpos[p.th].copy()
        origin=d.qpos.copy()
        base=np.clip(origin[p.qa],self.low[:,:7],self.high[:,:7])
        shifted=rot[:,0].copy()
        shifted[2]=0.
        shifted/=max(1e-8,np.linalg.norm(shifted))
        transverse=np.cross([0.,0.,1.],shifted)
        lows=np.argsort(centers[:,2])
        variants=[]
        jac=np.zeros((3,m.nv))
        for pair in (lows[:2],lows[:3],np.arange(3),np.arange(3,6),lows[:1]):
            for offset in (-.105,-.035,.035,.105):
                q=base.copy()
                # Reach the low surface with a lateral or sagittal displacement.
                targets=centers.copy()
                targets[pair]+=offset*transverse
                targets[pair]+=gen.uniform(-.05,.05)*shifted
                targets[pair,2]=np.minimum(centers[pair,2]-.022,center[2]-.092)
                for _ in range(9):
                    d.qpos[:]=origin
                    d.qpos[p.qa]=q
                    mj.mj_fwdPosition(m,d)
                    for leg in pair:
                        body=p.foot_body[leg]
                        start=d.xpos[body]
                        mj.mj_jac(m,d,jac,None,start,int(body))
                        j=jac[:,p.dofs[leg]]
                        diff=j.T@np.linalg.solve(j@j.T+.001*np.eye(3),targets[leg]-start)
                        q[leg]=np.clip(q[leg]+np.clip(.85*diff,-.26,.26),self.low[leg,:7],self.high[leg,:7])
                action=p.ctrl_neutral.copy()
                action[p.act[:,:7]]=q
                variants.append(action)
        return variants

    def proposals(self,state,seed,prior,count):
        gen=np.random.default_rng(seed)
        p=self.p
        direct=self.ik(state,gen)
        neutral=p.ctrl_neutral.copy()
        options=[neutral,*direct,*self.library]
        if prior is not None:
            options.append(prior)
        centers=[neutral,*direct] if prior is None else [neutral,prior,prior,*direct]
        for i in range(count):
            action=centers[i%len(centers)].copy()
            varied=action[p.act]+gen.normal(0.,(.25,.55,.95)[i%3],size=self.low.shape)
            if i%4==0:
                varied[3:]=action[p.act[3:]]
            if i%4==1:
                varied[:3]=action[p.act[:3]]
            action[p.act]=np.clip(varied,self.low,self.high)
            if i%5==0:
                # Reuse native adhesive actuators at modest effort on supported legs.
                action[p.adh]=gen.uniform(.1,.5,size=6)
            options.append(action)
        return options

    def best(self,state,seed,prior,count):
        params=self.proposals(state,seed,prior,count)
        neutral=self.p.ctrl_neutral
        def eval_action(a):
            result=self.int.run(state,[a,a,neutral])
            if result is None:
                return -np.inf
            x=result[0]
            return 4.0*x[0]+6*x[1]+.12*x[4]-.06*x[5]-.003*x[3] + .16*x[6]
        values=np.array(list(self.pool.map(eval_action,params)))
        if not np.isfinite(values).any():
            raise RuntimeError('No finite physical rollout')
        return params[int(values.argmax())],float(values.max()),len(values)


def start(p,name):
    config=TRIALS[name]
    if config is None:
        pos=np.load(SOURCE,allow_pickle=False)[round(5.150/.005)-1]
        seed=p.init(0.,0.,0.,(0.,0.),settle=0.,source=pos)
    elif name.startswith(('test_shallow_','test_dorsal')):
        p.init(*config,settle=0.)
        for _ in range(70):
            foot=p.scanner.sample()
            if foot[3]:
                break
            p.data.qpos[2]-=.004
            mj.mj_forward(p.model,p.data)
        assert p.scanner.sample()[3]
        seed=p.state()
    else:
        seed=p.init(*config,settle=.17)
    p.put(seed)
    return seed,float(p.data.xmat[p.th,8]),p.data.qpos.copy()


def save_run(control,case,out,seed,n_candidates):
    p=control.p
    state,up,initial=start(p,case)
    records=[]
    motions=[initial]
    goals=[]
    plan=[]
    counter=0
    landed=None
    prior=None
    neutral=p.ctrl_neutral
    for segment in range(17):
        p.put(state)
        support=p.scanner.sample()
        upright=p.data.xmat[p.th,8]>.92 and len(support[2])>=3 and support[3]==0
        if not upright:
            action,score,n=control.best(state,seed+337*segment,prior,n_candidates)
            counter+=n
            prior=action
        else:
            action,score=neutral,None
        result=control.int.run(state,[action,action,neutral],save=True)
        if result is None:
            raise RuntimeError('Selected rollout hit a physics warning')
        x,state,q=result
        plan.append((segment,up,score,x.tolist()))
        print(case,segment,'up',round(float(x[0]),3),'feet',int(x[4]),'body',int(x[5]),
              'score',score,flush=True)
        records.append(x)
        motions.extend(q)
        goals.extend((action,action,neutral))
        up=float(x[0])
        if up>.95 and x[4]>=4 and not x[5] and x[2]<.32 and x[3]<4.:
            settle=control.int.run(state,[neutral]*8,save=True)
            if settle is None:
                continue
            end,finish,q=settle
            if end[0]>.966 and end[4]>=5 and not end[5] and end[2]<.08 and end[3]<1.:
                landed=float(x[-1])
                motions.extend(q)
                goals.extend([neutral]*8)
                records.append(end)
                state=finish
                break
    out.mkdir()
    np.save(out/'QPOS.npy',np.asarray(motions),allow_pickle=False)
    np.save(out/'ACTION.npy',np.asarray(goals),allow_pickle=False)
    np.save(out/'END.npy',state,allow_pickle=False)
    detail=dict(case=case,passed=landed is not None,initial_world_up=plan[0][1],
                first_settlement_time=landed,last_metrics=records[-1].tolist(),
                eval_branches=counter,execution_actions=len(goals),
                plan=plan,actual_native_dt=p.model.opt.timestep)
    (out/'RESULT.json').write_text(json.dumps(detail,indent=2)+'\n')
    print(detail,flush=True)
    return detail


def main(args):
    p=Plant()
    control=RightingSearch(p,args.workers)
    for idx,case in enumerate(args.cases):
        save_run(control,case,args.root/case,1601+721*idx,args.candidates)
    control.pool.shutdown()
    p.env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('cases',nargs='+',choices=TRIALS)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--candidates',type=int,default=72)
    args=parser.parse_args()
    args.root.mkdir()
    main(args)
