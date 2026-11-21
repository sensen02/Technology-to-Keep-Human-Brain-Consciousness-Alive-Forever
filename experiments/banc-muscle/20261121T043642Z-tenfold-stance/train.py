"""Time-budgeted, multi-start block CMA continuation of the released BANC controller."""
import copy
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '20261121T041548Z-cma-stance'
sys.path[:0] = [str(PRIOR), str(next((PRIOR/'deps').glob('*.whl')))]
from cmaes import CMA
from bench import Bench, PARENT, HANDOFF, SETTLE, MuscleTeacher, load_pose, mj, sha
from warm_release import support

VELOCITIES = (0., -.03, .03)
HELD = (-.045, .015, .045)
START = PRIOR/'CANDIDATE.npz'
TARGET_SECONDS = 3272
POP = 12
GENERATIONS = 8


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp,path)


def save_weights(params):
    tmp=HERE/f'pending_{os.getpid()}.npz'
    np.savez_compressed(tmp, **params)
    digest=sha(tmp)
    path=HERE/'checkpoints'/(digest+'.npz')
    os.replace(tmp,path)
    return digest


def reference_budget():
    stages={}
    for name in ('training','strict','transfer','sweep0','sweep1','correction'):
        log=PRIOR/(name+'.log')
        birth=subprocess.check_output(['stat','-c','%w',str(log)],text=True).strip()
        stages[name]=log.stat().st_mtime-datetime.datetime.fromisoformat(birth).timestamp()
    total=sum(stages.values())
    assert total*10<=TARGET_SECONDS and total*10>TARGET_SECONDS-2
    return dict(previous_log_intervals_s=stages,previous_total_s=total,target_training_s=TARGET_SECONDS,
                workers=8,population=POP,development_vx=VELOCITIES,held_out_vx=HELD,
                parent_sha256=sha(START),cma_wheel_sha256=sha(next((PRIOR/'deps').glob('*.whl'))))


def preparations(b):
    """The warm physical trajectory is independent of the neural parameters."""
    state=json.loads((PARENT/'stance.json').read_text())
    pose=np.array(state['qpos'])
    teacher=MuscleTeacher(b.model,pose,b.target)
    conditions=[(b.handoff,b.preload)]
    for velocity in VELOCITIES[1:]:
        d=load_pose(b.model,pose)
        d.act[:]=d.ctrl[:]=b.target
        d.qvel[0]=velocity
        mj.mj_forward(b.model,d)
        history=[]
        for k in range(HANDOFF):
            if k%5==0:
                d.ctrl[:]=teacher.command(d)
                if k>=SETTLE:
                    history.append(b.inputs(b.obs.measure(d)))
            mj.mj_step(b.model,d)
            mj.mj_forward(b.model,d)
        conditions.append((d,np.array(history)))
    return conditions


def decode(x):
    params=dict(ANCHOR)
    v=ANCHOR[FIELD].copy()
    step=x[GROUP]
    if FIELD=='motor_gain':
        v[IDS]*=np.exp(np.clip(step,-3.,3.))
    elif FIELD=='feco_gain':
        v[IDS]+=.12*np.tanh(step)
    else:
        v[IDS]=.08*np.tanh(np.arctanh(np.clip(v[IDS]/.08,-.999999,.999999))+step)
    params[FIELD]=v
    return params


def evaluate_params(params,cached):
    metrics=[]
    for i,(d,preload) in enumerate(CONDITIONS):
        B.handoff,B.preload=d,preload
        B.cache=CACHES[i] if cached else None
        score,fail,end=B.evaluate(params,cached,duration_ms=250.,strict='handoff')
        metrics.append([250. if fail is None else round(fail,3),round(end,3),score])
    return metrics


def worker(x):
    return evaluate_params(decode(x),FIELD=='motor_gain')


def primary(metrics):
    times=np.array(metrics)[:,0]
    # Development objective emphasizes the weakest initial velocity and retains nominal margin.
    costs=np.array(metrics)[:,2]
    return float(-.70*min(times)-.30*np.mean(times)+.65*max(0.,141.1-times[0])
                 +.03*np.mean(costs))


def admissible(metrics):
    return all(m[0]>=ref[0] for m,ref in zip(metrics,BASE_METRICS))


def rank(metrics):
    t=np.array(metrics)
    return (min(t[:,0]),np.mean(t[:,0]),-np.mean(t[:,2]))


def make_block(k):
    pattern=('muscle_family','leg_units','individual','leg_families','feco',
             'muscle_family','active_units','individual','exc','leg_families','active_units','inh')
    kind=pattern[k%len(pattern)]
    rng=np.random.default_rng(926104+k)
    arms=B.motor_rows
    sigma=[.085,.19,.045,.28][(k//3)%4]
    if kind=='feco':
        ids=np.arange(len(ANCHOR['feco_gain']))
        return 'feco_gain',ids,rng.permutation(len(ids))%12,.35,kind
    if kind in ('exc','inh'):
        field=kind+'_fraction'
        w=B.io.network[int(kind=='inh')]
        counts=CACHES[0][3]
        scores=w.data*counts[w.indices]
        eligible=np.flatnonzero(scores>0)
        ids=rng.choice(eligible,min(2048,len(eligible)),replace=False,
                       p=np.sqrt(scores[eligible])/np.sqrt(scores[eligible]).sum())
        return field,ids,w.indices[ids]%12,.11,kind
    all_ids=np.flatnonzero(ANCHOR['motor_gain']>0)
    if kind=='muscle_family':
        ids=all_ids
        groups=arms[ids]%15
    elif kind=='leg_families':
        ids=all_ids[np.isin(arms[all_ids]//15,[(k//3)%6,(k//3+3)%6])]
        groups=arms[ids]%15
    elif kind=='individual':
        enabled=CACHES[0][2][-200:].mean(axis=0)[B.motor_cols[all_ids]]
        prob=np.sqrt(.005+enabled*ANCHOR['motor_gain'][all_ids])
        ids=rng.choice(all_ids,size=24,replace=False,p=prob/prob.sum())
        groups=np.arange(len(ids))
    else:
        ids=all_ids if kind=='active_units' else all_ids[(arms[all_ids]//15)%3==(k//3)%3]
        groups=rng.permutation(len(B.io.motor))[B.motor_cols[ids]]%16
    return 'motor_gain',ids,groups,sigma,kind


def refresh(params):
    out=[]
    for d,preload in CONDITIONS:
        B.handoff,B.preload=d,preload
        out.append(B.prepare(params))
    return out


def run():
    global B,CONDITIONS,FIELD,ANCHOR,IDS,GROUP,CACHES,BASE_METRICS
    budget=reference_budget()
    assert budget['parent_sha256']=='f0397fa26cb0aa68b0906226eebebb75da3491e8d73a7b58a40618670a66c693'
    dump(HERE/'BUDGET.json',budget)
    (HERE/'checkpoints').mkdir(exist_ok=True)
    B=Bench()
    CONDITIONS=preparations(B)
    with np.load(START) as saved:
        origin={k:saved[k] for k in saved.files}
    CACHES=refresh(origin)
    BASE_METRICS=evaluate_params(origin,True)
    assert [m[0] for m in BASE_METRICS]==[141.1,42.2,142.4],BASE_METRICS
    path=HERE/'progress.json'
    if path.exists():
        status=json.loads(path.read_text())
        assert status['parent']==budget['parent_sha256']
        checkpoint=HERE/'checkpoints'/(status['best']+'.npz')
        assert sha(checkpoint)==status['best']
        with np.load(checkpoint) as saved:
            best={k:saved[k] for k in saved.files}
    else:
        best=origin
        status=dict(parent=budget['parent_sha256'],best=save_weights(best),metrics=BASE_METRICS,
                    completed_blocks=0,training_seconds=0.,candidates=0,episodes=0,updates=0)
        dump(path,status)
    while status['training_seconds'] < TARGET_SECONDS:
        begin=time.monotonic()
        k=status['completed_blocks']
        previous_primary=status['best']
        secondary=HERE/'explore/progress.json'
        if secondary.is_file():
            trial=json.loads(secondary.read_text())
            if trial['accepted'] and rank(trial['eligible_metrics'])>rank(status['metrics']):
                candidate_path=HERE/'checkpoints'/(trial['accepted']+'.npz')
                assert sha(candidate_path)==trial['accepted']
                with np.load(candidate_path) as saved:
                    candidate={name:saved[name] for name in saved.files}
                CACHES=refresh(candidate)
                observed=evaluate_params(candidate,True)
                np.testing.assert_allclose(observed,trial['eligible_metrics'],atol=1e-9,rtol=0)
                assert admissible(observed)
                best=candidate
                status['best']=trial['accepted']
                status['metrics']=observed
        ANCHOR=best
        CACHES=refresh(ANCHOR)
        FIELD,IDS,GROUP,sigma,kind=make_block(k)
        anchor_sha=status['best']
        opt=CMA(mean=np.zeros(int(max(GROUP))+1),sigma=sigma,seed=172073+k,population_size=POP)
        accepted=None
        best_metrics=status['metrics']
        logs=[]
        with mp.get_context('fork').Pool(8) as pool:
            for generation in range(GENERATIONS):
                xs=[opt.ask() for _ in range(POP)]
                measured=pool.map(worker,xs)
                opt.tell([(x,primary(m)) for x,m in zip(xs,measured)])
                for x,m in zip(xs,measured):
                    if admissible(m) and rank(m)>rank(best_metrics):
                        accepted=x.copy()
                        best_metrics=m
                logs.append(dict(generation=generation,x=[x.tolist() for x in xs],metrics=measured))
                if status['training_seconds']+time.monotonic()-begin>=TARGET_SECONDS:
                    break
        if accepted is not None:
            best=decode(accepted)
            status['best']=save_weights(best)
            status['metrics']=best_metrics
        with (HERE/f'block_{k:04d}.npz').open('wb') as stream:
            np.savez_compressed(stream,field=FIELD,ids=IDS,groups=GROUP,anchor=anchor_sha,
                                selected=accepted if accepted is not None else [],
                                candidate=status['best'],kind=kind,sigma=sigma,previous_primary=previous_primary)
        with (HERE/f'optimizer_{k:04d}.pkl').open('wb') as stream:
            pickle.dump(opt,stream)
        dump(HERE/f'generation_{k:04d}.json',logs)
        status['training_seconds']+=time.monotonic()-begin
        status['completed_blocks']+=1
        status['updates']+=len(logs)
        status['candidates']+=len(logs)*POP
        status['episodes']+=len(logs)*POP*len(CONDITIONS)
        dump(path,status)
        print(k,kind,round(status['training_seconds'],1),[m[0] for m in status['metrics']],flush=True)
    (HERE/'CANDIDATE.npz').write_bytes((HERE/'checkpoints'/(status['best']+'.npz')).read_bytes())


if __name__=='__main__':
    run()
