"""Parallel low-priority search across the observed trade-off; never read held-out trials."""
import json
import multiprocessing as mp
import os
import pickle
import time
import numpy as np
import train as t

OUT=t.HERE/'explore'
MIN_NEG=55.
STOPS_AT_PRIMARY=3150.


def quality(metrics):
    nominal,negative,positive=[m[0] for m in metrics]
    floor=min(nominal-141.1,positive-142.4)
    if floor < 0:
        return min(floor,negative-MIN_NEG)+.04*(negative-MIN_NEG)+.03*floor
    return min(negative-MIN_NEG,0)+.03*floor+.15*(negative-MIN_NEG)


def read_weights(digest):
    with np.load(t.HERE/'checkpoints'/(digest+'.npz')) as f:
        return {k:f[k] for k in f.files}


def select_seed():
    records=[]
    for log in sorted(t.HERE.glob('generation_*.json')):
        block=int(log.stem.split('_')[1])
        for g in json.loads(log.read_text()):
            for x,metrics in zip(g['x'],g['metrics']):
                if metrics[1][0]>=58. and min(metrics[0][0],metrics[2][0])>115.:
                    records.append((quality(metrics),block,x,metrics))
    _,block,x,expected=max(records,key=lambda s:s[0])
    with np.load(t.HERE/f'block_{block:04d}.npz') as f:
        t.ANCHOR=read_weights(str(f['anchor']))
        t.FIELD=str(f['field'])
        t.IDS,t.GROUP=f['ids'],f['groups']
        parent=str(f['anchor'])
    result=t.decode(np.asarray(x))
    return result,expected,dict(source_block=block,x=x,metrics=expected,parent=parent)


def search():
    os.nice(8)
    OUT.mkdir(exist_ok=False)
    t.B=t.Bench()
    t.CONDITIONS=t.preparations(t.B)
    with np.load(t.START) as f:
        origin={k:f[k] for k in f.files}
    t.CACHES=t.refresh(origin)
    t.BASE_METRICS=t.evaluate_params(origin,True)
    incumbent,metrics,source=select_seed()
    t.CACHES=t.refresh(incumbent)
    np.testing.assert_allclose(t.evaluate_params(incumbent,True),metrics,rtol=0,atol=1e-10)
    start_digest=t.save_weights(incumbent)
    t.dump(OUT/'seed.json',dict(**source,seed_weight=start_digest))
    status=dict(accepted=None,eligible_metrics=t.BASE_METRICS,incumbent=start_digest,metrics=metrics,
                candidates=0,episodes=0,updates=0,blocks=0,workers=2)
    t.dump(OUT/'progress.json',status)
    begin=time.monotonic()
    while json.loads((t.HERE/'progress.json').read_text())['training_seconds']<STOPS_AT_PRIMARY:
        k=status['blocks']
        t.ANCHOR=incumbent
        t.CACHES=t.refresh(incumbent)
        rng=np.random.default_rng(611042+k)
        ids=np.flatnonzero(incumbent['motor_gain']>0)
        arms=t.B.motor_rows[ids]
        j=k%5
        sigma=(.035,.075,.11,.022,.055)[j]
        if j==0:
            groups=arms%15
        elif j==1:
            groups=rng.permutation(len(t.B.io.motor))[t.B.motor_cols[ids]]%12
        elif j==2:
            ids=rng.choice(ids,16,replace=False)
            groups=np.arange(len(ids))
        elif j==3:
            groups=(arms//15)*2+(arms%2)
        else:
            select=rng.choice(6,2,replace=False)
            ids=ids[np.isin(arms//15,select)]
            groups=t.B.motor_rows[ids]%15
        t.FIELD,t.IDS,t.GROUP='motor_gain',ids,groups
        pop=8
        opt=t.CMA(np.zeros(int(max(groups))+1),sigma=sigma,seed=51811+k,population_size=pop)
        prev=status['incumbent']
        choose=None
        winner=None
        rows=[]
        with mp.get_context('fork').Pool(2) as pool:
            for g in range(6):
                xs=[opt.ask() for _ in range(pop)]
                scores=pool.map(t.worker,xs)
                values=[quality(m) if not t.admissible(metrics) else
                        min(m[0][0]-141.1,m[2][0]-142.4,0)*2.5+m[1][0] for m in scores]
                opt.tell([(x,-v) for x,v in zip(xs,values)])
                for x,m in zip(xs,scores):
                    if t.admissible(m) and t.rank(m)>t.rank(status['eligible_metrics']):
                        status['eligible_metrics']=m
                        winner=x.copy()
                    if t.admissible(status['metrics']):
                        accept=t.admissible(m) and t.rank(m)>t.rank(status['metrics'])
                    else:
                        accept=(t.admissible(m) and m[1][0]>=MIN_NEG) or (m[1][0]>=MIN_NEG and quality(m)>quality(status['metrics']))
                    if accept:
                        choose=x.copy()
                        status['metrics']=m
                rows.append(dict(x=[x.tolist() for x in xs],metrics=scores))
                if json.loads((t.HERE/'progress.json').read_text())['training_seconds']>=STOPS_AT_PRIMARY:
                    break
        if winner is not None:
            status['accepted']=t.save_weights(t.decode(winner))
        if choose is not None:
            incumbent=t.decode(choose)
            metrics=status['metrics']
            status['incumbent']=t.save_weights(incumbent)
        np.savez_compressed(OUT/f'block_{k:04d}.npz',anchor=prev,field=t.FIELD,ids=ids,groups=groups,
                            incumbent=status['incumbent'],accepted=status['accepted'] or '',
                            selected=[] if choose is None else choose,winner=[] if winner is None else winner)
        with (OUT/f'optimizer_{k:04d}.pkl').open('wb') as f:
            pickle.dump(opt,f)
        t.dump(OUT/f'generation_{k:04d}.json',rows)
        status['updates']+=len(rows)
        status['candidates']+=len(rows)*pop
        status['episodes']=status['candidates']*3
        status['blocks']+=1
        status['parallel_wall_s']=time.monotonic()-begin
        t.dump(OUT/'progress.json',status)
        print(k,[m[0] for m in status['metrics']],status['accepted'],flush=True)

if __name__=='__main__':
    search()
