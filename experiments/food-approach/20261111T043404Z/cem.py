"""Exact preregistered between-rollout CEM algebra; no motion generation."""
import copy
import numpy as np

def propose(mean,sigma,rng):
 before=copy.deepcopy(rng.bit_generator.state);z=rng.standard_normal((2,3));m=np.asarray(mean);s=np.asarray(sigma)
 deltas=np.clip(np.stack((m+s*z[0],m-s*z[0],m+s*z[1],m-s*z[1])),-.25,.25)
 return deltas,before,copy.deepcopy(rng.bit_generator.state)

def objective(result):
 a=result['autonomous'];criteria=result['criteria']
 constraints=[a['gates'][k] for k in ('support','nonfoot','com','forward')]+[criteria['all_six'],criteria['strict_nonfoot_zero'],criteria['turn_sign_magnitude']]
 slip=float(a['slip_ratio_max'])
 if not np.isfinite(slip):raise ValueError('nonfinite rollout not optimizer measurement')
 failures=sum(not bool(v) for v in constraints)
 return slip if failures==0 else 1000.+10.*failures+min(10.,slip)

def update(mean,sigma,deltas,scores):
 order=sorted(range(4),key=lambda i:(scores[i],i));elite=np.asarray(deltas)[order[:2]]
 return .25*np.asarray(mean)+.75*elite.mean(0),np.clip(.3*np.asarray(sigma)+.7*elite.std(0,ddof=0),.01,.10),order[:2]
