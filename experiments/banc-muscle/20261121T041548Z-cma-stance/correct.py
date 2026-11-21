"""Query the preparation teacher offline on states visited during BANC-only release."""
import copy
import json
import numpy as np
from cmaes import CMA
import refine as r
from bench import Bench, HERE, PARENT, DT_MS, MuscleTeacher, mj


def collect(b,p):
    net,trace,*_=b.prepare(p)
    d=copy.copy(b.handoff)
    pose=np.array(json.loads((PARENT/'stance.json').read_text())['qpos'])
    teacher=MuscleTeacher(b.model,pose,b.target)
    gains=np.zeros_like(b.io.motor_gain)
    gains[b.motor_rows,b.motor_cols]=p['motor_gain']
    drive=np.zeros(net.n)
    traces=[]
    labels=[]
    for k in range(500):
        z=b.obs.measure(d)
        if k and (z['ground_load_w'][-1]>0 or z['torso'][0]<.895*b.height):
            break
        traces.append(trace.copy())
        labels.append(teacher.command(d))
        d.ctrl[:]=np.clip(gains@trace,0,1)
        b.current(b.inputs(z),p,drive)
        trace*=np.exp(-DT_MS/10)
        trace+=net.step(drive)[b.io.motor]
        for _ in range(5):
            mj.mj_step(b.model,d)
            mj.mj_forward(b.model,d)
    return np.array(traces),np.array(labels)


def main():
    b=Bench()
    r.B=b
    r.OUT=HERE/'correction'
    r.OUT.mkdir()
    with np.load(HERE/'sweep1/trained.npz') as f:
        r.P={k:f[k] for k in f.files}
    history,target=collect(b,r.P)
    np.savez_compressed(r.OUT/'teacher_queries.npz',trace=history,target=target)
    prior=r.P['motor_gain']
    fit=prior.copy()
    logs=[]
    for arm in range(90):
        idx=np.flatnonzero(b.motor_rows==arm)
        h=history[:,b.motor_cols[idx]]
        use=h.max(axis=0)>.05
        idx=idx[use]
        h=h[:,use]
        if not len(idx):
            continue
        # Prefer motor roots whose late modulation is distinctive; remaining coefficients stay fixed.
        order=np.argsort(-np.std(h,axis=0))[:12]
        base=np.clip(history[:,b.motor_cols[b.motor_rows==arm]]@prior[b.motor_rows==arm],0,1)
        fixed=history[:,b.motor_cols[b.motor_rows==arm]]@prior[b.motor_rows==arm]-h[:,order]@prior[idx[order]]
        h=h[:,order]
        idx=idx[order]
        initial=prior[idx]
        def decode(x):
            return np.maximum(0, initial+np.maximum(.12,initial*.5)*x[:len(idx)])
        def loss(w):
            predicted=np.clip(fixed+h@w,0,1)
            return float(np.mean((predicted-target[:,arm])**2)+np.mean((predicted[:20]-base[:20])**2))
        c=CMA(mean=np.zeros(max(2,len(idx))),sigma=.7,seed=32900+arm,population_size=8)
        best=loss(initial)
        for gen in range(15):
            xs=[c.ask() for _ in range(c.population_size)]
            ys=[loss(decode(x)) for x in xs]
            winner=np.argmin(ys)
            if ys[winner]<best:
                best=ys[winner]
                fit[idx]=decode(xs[winner])
            c.tell(list(zip(xs,ys)))
        logs.append(dict(arm=arm,old=loss(initial),new=best,indices=idx.tolist(),fitted=fit[idx].tolist(),evaluations=120))
    (r.OUT/'fit.json').write_text(json.dumps(logs)+'\n')
    r.HORIZON=250.
    r.OBJECTIVE='handoff'
    for j in range(3):
        r.MOTOR_DIRECTION=fit-r.P['motor_gain']
        idx=np.flatnonzero(b.motor_rows//30==j)
        r.batch('motor_gain',idx,b.motor_rows[idx]%30,'teacher'+str(j),33200+j,8,.27)
    r.MOTOR_DIRECTION=None
    # Small final fine steps preserve the support attained by larger earlier searches.
    idx=np.arange(len(b.motor_rows))
    r.batch('motor_gain',idx,b.motor_rows%15,'fine',33518,10,.06)
    np.savez_compressed(r.OUT/'trained.npz',**r.P)


if __name__=='__main__':
    main()
