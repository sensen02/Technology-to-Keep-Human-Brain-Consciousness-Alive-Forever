"""Offline masked geometric losses for observed/retargeted marker sequences.
No pose writes, motion generator, phase alignment, IK or runtime controller.
Caller must supply already declared common coordinates and dimensional normalizers.
"""
import numpy as np


def _points_and_mask(predicted,target,valid):
    p=np.asarray(predicted,dtype=float);q=np.asarray(target,dtype=float);m=np.asarray(valid)
    if p.shape!=q.shape or p.ndim<3 or p.shape[-1]!=3 or m.shape!=p.shape[:-1] or m.dtype!=bool:
        raise ValueError('Expected matching time-by-markers-by-3 points and boolean marker mask')
    if np.any(m&(~np.isfinite(q).all(axis=-1))):raise ValueError('Target valid mask admits nonfinite coordinates')
    if np.any(m&(~np.isfinite(p).all(axis=-1))):raise ValueError('Prediction is nonfinite on admitted target markers')
    return p,q,m


def masked_cartesian_huber(predicted,target,valid,*,normalizer,delta=1.,weights=None):
    """Huber of each marker's normalized Euclidean distance, mean over admitted weights.
    All missing => evaluable=False and loss=None, never a successful zero.
    """
    p,q,m=_points_and_mask(predicted,target,valid)
    scale=np.asarray(normalizer,dtype=float)
    if not np.isfinite(delta) or delta<=0 or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Positive finite delta and normalizers required')
    try:scale=np.broadcast_to(scale,m.shape)
    except ValueError as e:raise ValueError('Normalizer must broadcast over marker axes') from e
    w=np.ones(m.shape) if weights is None else np.asarray(weights,dtype=float)
    try:w=np.broadcast_to(w,m.shape)
    except ValueError as e:raise ValueError('Weights must broadcast over markers') from e
    if not np.isfinite(w).all() or np.any(w<0):raise ValueError('Finite nonnegative weights required')
    admitted=m&(w>0);n=int(admitted.sum());den=float(w[admitted].sum())
    if not n or not den:return dict(evaluable=False,loss=None,valid_markers=n,total_markers=m.size,weight_sum=den)
    residual=np.linalg.norm(p[admitted]-q[admitted],axis=-1)/scale[admitted]
    penalty=np.where(residual<=delta,.5*residual**2,delta*(residual-.5*delta))
    return dict(evaluable=True,loss=float(np.dot(w[admitted],penalty)/den),valid_markers=n,total_markers=m.size,weight_sum=den,normalized_rms=float(np.sqrt(np.dot(w[admitted],residual**2)/den)),normalized_max=float(residual.max()))


def segment_direction_loss(predicted,target,valid,*,min_segment_length=1e-12):
    """Per-bone squared angular error on adjacent explicitly ordered leg markers.
    Inputs shape(time,leg,marker,3); translation and uniform positive scale invariant.
    """
    p,q,m=_points_and_mask(predicted,target,valid)
    if p.ndim!=4 or p.shape[2]<2 or not np.isfinite(min_segment_length) or min_segment_length<=0:raise ValueError('Expected leg marker chains and positive length cutoff')
    pair=m[:,:,:-1]&m[:,:,1:]
    dp=np.diff(p,axis=2);dq=np.diff(q,axis=2);lp=np.linalg.norm(dp,axis=-1);lq=np.linalg.norm(dq,axis=-1)
    if np.any(pair&((lp<=min_segment_length)|(lq<=min_segment_length))):raise ValueError('Admitted bone has degenerate direction')
    n=int(pair.sum())
    if not n:return dict(evaluable=False,loss=None,valid_bones=0,total_bones=pair.size)
    u=dp[pair]/lp[pair,None];v=dq[pair]/lq[pair,None]
    angle=np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))
    return dict(evaluable=True,loss=float(np.mean(angle**2)),valid_bones=n,total_bones=pair.size,angular_rms_rad=float(np.sqrt(np.mean(angle**2))),angular_max_rad=float(angle.max()))


def geometric_interior_angles(points,valid,*,min_segment_length=1e-12):
    """Unsigned three-marker bending angles; not a named hinge or Euler angle.
    Adjacent length, rigid frame, reflection and positive scale are nuisance gauges.
    Returns values with NaN where any of the three markers is invalid.
    """
    p,_,m=_points_and_mask(points,points,valid)
    if p.ndim!=4 or p.shape[2]<3 or not np.isfinite(min_segment_length) or min_segment_length<=0:raise ValueError('Expected leg chains with >=3markers and positive length cutoff')
    admitted=m[:,:,:-2]&m[:,:,1:-1]&m[:,:,2:]
    left=p[:,:,:-2]-p[:,:,1:-1];right=p[:,:,2:]-p[:,:,1:-1]
    ll=np.linalg.norm(left,axis=-1);lr=np.linalg.norm(right,axis=-1)
    if np.any(admitted&((ll<=min_segment_length)|(lr<=min_segment_length))):raise ValueError('Admitted angle has degenerate segment')
    values=np.full(admitted.shape,np.nan)
    if admitted.any():
        u=left[admitted]/ll[admitted,None];v=right[admitted]/lr[admitted,None]
        values[admitted]=np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))
    return values,admitted


def interior_angle_loss(predicted,target,valid,*,min_segment_length=1e-12):
    """Mean squared bending error in rad²; invariant partial target, not full pose.
    Applies the same target-specified marker mask to prediction and reference.
    """
    p,q,m=_points_and_mask(predicted,target,valid)
    a,ma=geometric_interior_angles(p,m,min_segment_length=min_segment_length)
    b,mb=geometric_interior_angles(q,m,min_segment_length=min_segment_length)
    assert np.array_equal(ma,mb)
    if not ma.any():return dict(evaluable=False,loss=None,valid_angles=0,total_angles=ma.size)
    error=a[ma]-b[ma]
    return dict(evaluable=True,loss=float(np.mean(error**2)),valid_angles=int(ma.sum()),total_angles=ma.size,angular_rms_rad=float(np.sqrt(np.mean(error**2))),angular_max_abs_rad=float(np.max(np.abs(error))))


def masked_velocity_huber(predicted,target,valid,times_s,*,normalizer_per_second,max_gap_s,delta=1.,trial_ids=None):
    """Finite differences only within same trial, valid adjacent points and allowed dt.
    Never interpolate missing data, concatenate trials or warp time.
    """
    p,q,m=_points_and_mask(predicted,target,valid);t=np.asarray(times_s,dtype=float)
    if t.shape!=(p.shape[0],) or not np.isfinite(t).all() or len(t)<2 or not np.isfinite(max_gap_s) or max_gap_s<=0:raise ValueError('Finite per-frame times, >=2frames and positive gap bound required')
    same=np.ones(len(t)-1,dtype=bool)
    if trial_ids is not None:
        ids=np.asarray(trial_ids)
        if ids.shape!=t.shape:raise ValueError('One trial id per frame required')
        same=ids[:-1]==ids[1:]
    dt=np.diff(t)
    if np.any(same&(dt<=0)):raise ValueError('Time must increase strictly within each trial')
    intervals=same&(dt>0)&(dt<=max_gap_s);shape=(len(dt),)+(1,)*(p.ndim-1)
    safe=np.where(intervals,dt,1.).reshape(shape)
    vp=np.diff(p,axis=0)/safe;vq=np.diff(q,axis=0)/safe
    mask=m[:-1]&m[1:]&intervals.reshape((len(dt),)+(1,)*(m.ndim-1))
    result=masked_cartesian_huber(vp,vq,mask,normalizer=normalizer_per_second,delta=delta)
    result.update(eligible_time_intervals=int(intervals.sum()),total_time_intervals=len(dt),excluded_gap_or_trial_intervals=int((~intervals).sum()))
    return result
