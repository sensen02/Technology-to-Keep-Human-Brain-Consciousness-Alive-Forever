"""Pure NumPy acceptance and whole-decision velocity contracts; no model imports."""
import numpy as np


def validate_secondary(primary,candidate,pad_before,pad_after,bounds,h_before,h_after):
    arrays=[np.asarray(x,dtype=float) for x in (primary,candidate,pad_before,pad_after,bounds)]
    if not all(np.isfinite(x).all() for x in arrays) or not np.isfinite([h_before,h_after]).all():return False,'NONFINITE'
    p,q,a,b,limits=arrays
    if p.shape!=(7,) or q.shape!=(7,) or a.shape!=(3,) or b.shape!=(3,) or limits.shape!=(7,2):return False,'SHAPE'
    if np.max(np.abs(q-p))>.005+1e-12:return False,'DECISION_TRUST'
    if np.any(q<limits[:,0]-1e-12) or np.any(q>limits[:,1]+1e-12):return False,'JOINT_BOUNDS'
    if np.linalg.norm(b-a)>1e-5:return False,'NONLINEAR_PAD_ERROR'
    if h_after-h_before<1e-8:return False,'NO_MEANINGFUL_CLEARANCE_GAIN'
    return True,'ACCEPTED'


def whole_decision_velocity(previous,final,dt=.005,max_rate=2.):
    previous=np.asarray(previous,dtype=float);final=np.asarray(final,dtype=float)
    if previous.shape!=(42,) or final.shape!=(42,) or not np.isfinite([previous,final]).all():raise ValueError('Finite42-angle decision states required')
    if dt!=.005 or max_rate!=2.:raise ValueError('Original .005 decision/MAX_RATE2 must remain unchanged')
    raw=(final-previous)/dt
    return np.clip(raw,-max_rate,max_rate),raw
