"""Fixed preregistered optimizer arithmetic; independent of episode runner."""
import numpy as np
SEED=20261107
SHAPE=(6,4,7)


def candidates(rng,mean,std,incumbent):
    noise=rng.standard_normal((6,)+SHAPE)
    values=[]
    for pair in range(6):
        for sign in (1,-1):values.append(np.clip(mean+sign*std*noise[pair],-.25,.25))
    values.append(np.array(incumbent,copy=True))
    return noise,np.stack(values)


def choose_update(mean,std,values,results,incumbent_theta,incumbent_reward):
    valid=[i for i,r in enumerate(results) if r.get('valid',False) and np.isfinite(r.get('reward',np.nan))]
    feasible=[i for i in valid if results[i]['feasible']]
    pool=feasible if feasible else valid
    elites=sorted(pool,key=lambda i:(-results[i]['reward'],i))[:4]
    if elites:
        elitevalues=values[elites]
        next_mean=.5*mean+.5*elitevalues.mean(axis=0)
        next_std=np.clip(.7*std+.3*elitevalues.std(axis=0,ddof=0),.004,.05)
    else:next_mean,next_std=mean.copy(),std.copy()
    improved=[i for i in feasible if results[i]['reward']>incumbent_reward]
    winner=min(improved,key=lambda i:(-results[i]['reward'],i)) if improved else None
    next_theta=values[winner].copy() if winner is not None else incumbent_theta.copy()
    next_reward=float(results[winner]['reward']) if winner is not None else float(incumbent_reward)
    return dict(elites=elites,feasible_candidates=feasible,valid_candidates=valid,next_mean=next_mean,next_std=next_std,incumbent_candidate=winner,incumbent_theta=next_theta,incumbent_reward=next_reward)
