"""Block-coordinate CMA-ES for spike-driven standing, with frozen complement per block."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle

import numpy as np
from cmaes import CMA
from bench import BASE, HERE, Bench, sha

SEED = 7421


def save(path, parameters):
    np.savez_compressed(path, **parameters)
    return sha(path)


def log(record):
    with (OUT/'TRAIN.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')


def warm_fit(params, history, target):
    ids, cols = B.motor_rows, B.motor_cols
    result = params['motor_gain'].copy()
    # Each individual actuator constitutes a separate block. Only reachable columns can affect this loss.
    for arm in range(len(target)):
        idx = np.flatnonzero((ids == arm) & (history[-200:].mean(axis=0)[cols]>.001))
        if not len(idx):
            continue
        h = history[-200:,cols[idx]]
        initial = result[idx].copy()
        scale = np.maximum(.05, .5*target[arm]/np.maximum(.25, h.mean(axis=0)))
        # Smooth nonnegative map admits small gains even on nearly silent tonic channels.
        def decode(x):
            return np.maximum(0, initial+x[:len(idx)]*scale)
        def loss(w):
            err=np.clip(h@w,0,1)-target[arm]
            return float(.55*err[-1]**2+.45*np.mean(err**2))
        best = loss(initial)
        optimizer = CMA(mean=np.zeros(max(2,len(idx))), sigma=.8, seed=SEED+arm, population_size=10)
        for gen in range(22):
            solutions=[]
            for _ in range(optimizer.population_size):
                x=optimizer.ask()
                value=loss(decode(x))
                solutions.append((x,value))
                if value<best:
                    best=value
                    result[idx]=decode(x)
            optimizer.tell(solutions)
        log(dict(stage='warm_motor',arm=arm,indices=idx.tolist(),gains=result[idx].tolist(),
                 generations=22,initial_loss=loss(initial),best_loss=best))
    return {**params,'motor_gain':result}


def decode(x):
    state = dict(CURRENT)
    field, indices = BLOCK
    val = state[field].copy()
    if field.endswith('_fraction'):
        val[indices] = .08*np.tanh(x)
    elif field == 'load_gain':
        val[indices] *= np.exp(np.clip(x,-3,3))
    else:
        # A coordinate rescales only its own actuator's permitted motor pairs.
        val *= np.exp(np.clip(np.bincount(indices, weights=x, minlength=90)[B.motor_rows],-3,3))
    state[field] = val
    return state


def fitness(x):
    return B.evaluate(decode(x), BLOCK[0]=='motor_gain', duration_ms=250.)


def choose_edges(sign, counts):
    w=B.io.network[0 if sign=='exc' else 1]
    pairs=np.flatnonzero(np.isin(np.repeat(np.arange(w.shape[0]),np.diff(w.indptr)), B.io.motor) & (counts[w.indices]>0))
    strength=w.data[pairs]*counts[w.indices[pairs]]
    return pairs[np.argsort(-strength,kind='stable')]


def stage(params, field, indices, tag, seed, generations):
    global CURRENT, BLOCK
    CURRENT=params
    BLOCK=(field,indices)
    B.cache=B.prepare(params)
    baseline=B.evaluate(params, True)
    best=baseline[0]
    chosen=None
    start=np.arctanh(np.clip(params[field][indices]/.08,-.999999,.999999)) if field.endswith('_fraction') else np.zeros(len(indices))
    optimizer=CMA(mean=start,sigma=.8 if field.endswith('_fraction') else .35,seed=seed,population_size=8)
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for gen in range(generations):
            xs=[optimizer.ask() for _ in range(optimizer.population_size)]
            results=pool.map(fitness,xs)
            optimizer.tell([(x,r[0]) for x,r in zip(xs,results)])
            i=int(np.argmin([r[0] for r in results]))
            if results[i][0]<best:
                best=results[i][0]
                chosen=xs[i]
            log(dict(stage=tag,generation=gen,field=field,indices=indices.tolist(),seed=seed,
                     population=[x.tolist() for x in xs],results=results,best=best))
            print(tag,gen,round(best,3),flush=True)
    winner=params if chosen is None else decode(chosen)
    digest=save(OUT/(tag+'.npz'),winner)
    with (OUT/(tag+'_optimizer.pkl')).open('wb') as f:
        pickle.dump(optimizer,f)
    log(dict(stage=tag,base_score=baseline,best=best,changed=chosen is not None,sha256=digest))
    return winner


def main():
    global B
    OUT.mkdir(exist_ok=False)
    B=Bench()
    params=B.weights
    B.cache=B.prepare(params)
    baseline=B.evaluate(params,True)
    fitted=warm_fit(params,B.cache[2],B.target)
    fit=B.evaluate(fitted,True)
    save(OUT/'warm_motor.npz',fitted)
    log(dict(baseline=baseline,warmfit=fit,base_sha=sha(BASE)))
    print('motor prefit',baseline,fit,flush=True)
    if fit[0]<baseline[0]:
        params=fitted
    for j,sign in enumerate(('exc','inh')):
        edges=choose_edges(sign,B.cache[3])[:16]
        params=stage(params,sign+'_fraction',edges,sign,SEED+100+j,3)
    params=stage(params,'load_gain',np.arange(12),'sensory',SEED+200,3)
    for j,indices in enumerate(np.array_split(np.arange(90),3)):
        params=stage(params,'motor_gain',indices,'motor_'+str(j),SEED+300+j,4)
    save(OUT/'trained.npz',params)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    OUT,WORKERS=args.output,args.workers
    main()
