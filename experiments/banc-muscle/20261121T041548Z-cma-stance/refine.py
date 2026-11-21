"""Strict-release CMA blocks, including source-grounded active conductance subsets."""
import json
import multiprocessing as mp
import os
import pickle

import numpy as np
from cmaes import CMA
from bench import BASE, HERE, Bench, sha

OUT = HERE/'strict'
RNG=9713
HORIZON=100.
OBJECTIVE=True
SCALE=1.
MOTOR_DIRECTION=None


def parameters(x):
    out=dict(P)
    value=P[FIELD].copy()
    if FIELD.endswith('fraction'):
        # Tied perturbations over this batch only. Order is pinned in the batch archive.
        value[IDX]=.08*np.tanh(BASE_COORD+x[GROUP])
    elif FIELD=='motor_gain':
        if MOTOR_DIRECTION is None:
            value[IDX] *= np.exp(np.clip(x[GROUP],-5,5))
        else:
            value[IDX] = np.maximum(0., value[IDX] + MOTOR_DIRECTION[IDX]*np.tanh(x[GROUP]))
    elif FIELD in ('feco_gain', 'vnc_gain_uS'):
        value[IDX] += SCALE*np.tanh(x[GROUP])
    out[FIELD]=value
    return out


def evaluate(x):
    return B.evaluate(parameters(x), FIELD=='motor_gain',duration_ms=HORIZON,strict=OBJECTIVE)


def batch(field,index,groups,name,seed,gens,sigma):
    global FIELD,IDX,GROUP,BASE_COORD,P
    FIELD,IDX,GROUP=field,index,groups
    BASE_COORD=np.arctanh(np.clip(P[field][index]/.08,-.999999,.999999)) if field.endswith('fraction') else None
    B.cache=B.prepare(P)
    best=B.evaluate(P,True,HORIZON,strict=OBJECTIVE)
    optimizer=CMA(mean=np.zeros(int(groups.max())+1),sigma=sigma,seed=seed,population_size=10)
    vec=None
    with mp.get_context('fork').Pool(8) as pool, (OUT/(name+'.jsonl')).open('w') as log:
        for i in range(gens):
            xs=[optimizer.ask() for _ in range(optimizer.population_size)]
            scores=pool.map(evaluate,xs)
            optimizer.tell([(x,r[0]) for x,r in zip(xs,scores)])
            winner=np.argmin([r[0] for r in scores])
            if scores[winner][0] < best[0]:
                best=scores[winner];vec=xs[winner]
            log.write(json.dumps(dict(gen=i,scores=scores,vectors=[x.tolist() for x in xs],best=best))+'\n')
            log.flush()
            print(name,i,best,flush=True)
    if vec is not None:
        P=parameters(vec)
    np.savez_compressed(OUT/(name+'.npz'),**P)
    np.savez_compressed(OUT/(name+'_block.npz'),field=field,index=index,groups=groups,
                        selected=np.zeros(optimizer.dim) if vec is None else vec,accepted=vec is not None)
    with (OUT/(name+'_optimizer.pkl')).open('wb') as f:
        pickle.dump(optimizer,f)


def main():
    global B,P
    OUT.mkdir()
    B=Bench()
    P=B.weights
    fit=HERE/'training/warm_motor.npz'
    with np.load(fit) as a:
        fitted=a['motor_gain']
    with np.load(HERE/'training/trained.npz') as a:
        stage1={k:a[k] for k in a.files}
    trials=[]
    # Evaluate interpolation candidates under the stronger continuity objective.
    for state in (P,stage1):
        B.cache=B.prepare(state)
        for t in (0,.25,.5,.75,1.):
            candidate={**state,'motor_gain':state['motor_gain']*(1-t)+fitted*t}
            trials.append((B.evaluate(candidate,True,100.,strict=True),candidate))
    idx=int(np.argmin([r[0][0] for r in trials]))
    P=trials[idx][1]
    print('starts',[r for r,_ in trials],'selected',idx,flush=True)
    np.savez_compressed(OUT/'start.npz',**P)
    arms=B.motor_rows
    leg=arms//15
    family=arms%15
    all_idx=np.arange(len(arms))
    batch('motor_gain',all_idx,family,'family',RNG,6,.6)
    for j,field in enumerate(('exc_fraction','inh_fraction')):
        B.cache=B.prepare(P)
        counts=B.cache[3]
        w=B.io.network[j]
        pre=w.indices
        post=np.repeat(np.arange(w.shape[0]),np.diff(w.indptr))
        score=counts[pre]*w.data*(1+np.isin(post,B.io.motor)*2)
        eligible=np.flatnonzero(score>0)
        order=eligible[np.argsort(-score[eligible],kind='stable')[:4096]]
        # Coordinate groups of anatomically present connections terminating on the same root.
        batch(field,order,post[order]%16,field,RNG+1+j,4,1.1)
    for j in range(3):
        active=np.flatnonzero(leg//2==j)
        group=(leg[active]%2)*15+family[active]
        batch('motor_gain',active,group,'legs'+str(j),RNG+3+j,6,.42)
    np.savez_compressed(OUT/'trained.npz',**P)


if __name__=='__main__':
    main()
