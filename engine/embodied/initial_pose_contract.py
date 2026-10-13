"""Describe native scalar joint limits without silently clipping experimental poses."""
import numpy as np

def scalar_limit_report(names,addresses,types,limited,ranges,qpos,tolerance_rad=1e-12):
    # MuJoCo type constants slide=2,hinge=3. Ball limits need different geometry.
    q=np.asarray(qpos,dtype=float);bounds=np.asarray(ranges,dtype=float)
    if q.ndim!=1 or not np.isfinite(q).all():raise ValueError('finite qpos vector required')
    if len(names)!=len(addresses) or len(names)!=len(types) or len(names)!=len(limited) or bounds.shape!=(len(names),2):raise ValueError('joint metadata shapes mismatch')
    failures=[];checked=0
    for name,adr,kind,on,(lo,hi) in zip(names,addresses,types,limited,bounds):
        if not on or kind not in (2,3):continue
        checked+=1
        if not isinstance(adr,(int,np.integer)) or not 0<=adr<len(q):raise ValueError('invalid qpos address')
        value=float(q[adr])
        if value<lo-tolerance_rad or value>hi+tolerance_rad:
            failures.append(dict(joint=name,qpos_address=int(adr),value=value,lower=float(lo),upper=float(hi),outside_by=float(max(lo-value,value-hi)),units='rad' if kind==3 else 'model length'))
    return dict(checked_scalar_joints=checked,within_declared_scalar_ranges=not failures,violations=failures,
                pose_modified=False,interpretation='Strict scalar range diagnostic, separate from contact gate; soft-limit dynamics remain native. Not a claim asset is invalid.')

def model_limit_report(model,qpos):
    return scalar_limit_report([model.joint(j).name for j in range(model.njnt)],model.jnt_qposadr,model.jnt_type,model.jnt_limited,model.jnt_range,qpos)
