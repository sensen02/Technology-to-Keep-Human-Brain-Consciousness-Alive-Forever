"""Offline physical objective for final200ms; no runtime controller or rollout."""
import numpy as np


def score_formal(a, weight_native):
    q = np.asarray(a['qpos'],float); v = np.asarray(a['qvel'],float)
    contacts = np.asarray(a['contacts']); force = np.asarray(a['foot_force'],float)
    depth = np.asarray(a['nonfoot_depth'],float); excess = np.asarray(a['limit_excess'],float)
    if (q.shape != (4001,49) or v.shape != (4001,48) or contacts.shape != (4001,6) or
            force.shape != (4001,6,3) or depth.shape != (4001,) or excess.shape != (4001,)):
        raise ValueError('complete continuous400ms physical records required')
    if (not all(np.isfinite(x).all() for x in (q,v,contacts,force,depth,excess)) or
            not np.isfinite(weight_native) or weight_native<=0 or q[0,2]<=0):
        raise ValueError('finite arrays and positive native reference scales required')
    norms=np.linalg.norm(q[:,3:7],axis=1)
    if np.any(norms==0): raise ValueError('nonzero physical quaternions required')
    height=q[:,2]/q[0,2]
    angle=2*np.arccos(np.clip(abs(q[:,3:7]@q[0,3:7])/(norms*norms[0]),0,1))
    legs=(contacts>0).sum(1);load=force[:,:,2].sum(1)/weight_native
    good=(height>=.85)&(angle<=.35)&(legs>=3)&(load>=.8)&(depth<=1e-12)&(excess<=1e-6)
    series=dict(engineering=10*(~good).astype(float),contactloss=np.maximum(3-legs,0)/3,
        load_deficit=np.maximum(1-load,0)**2,height=2*(np.maximum(1-height,0)/.15)**2,
        descent=(np.maximum(-v[:,2],0)/10)**2,attitude=(angle/.35)**2,
        nonfoot=10*(depth/.01)**2,limit=10*(excess/.01)**2)
    stages={}
    for name,sl in [('preparation',slice(0,2001)),('formal',slice(2000,4001))]:
        parts={k:float(np.trapezoid(x[sl],dx=.0001)/.2) for k,x in series.items()}
        stages[name]=dict(loss=sum(parts.values()),parts=parts,min_height_ratio=float(height[sl].min()),
            max_downward_velocity=float(np.maximum(-v[sl,2],0).max()),max_angle_rad=float(angle[sl].max()),
            max_nonfoot_depth=float(depth[sl].max()),max_joint_limit_excess=float(excess[sl].max()),
            support_ms=float(np.trapezoid(good[sl].astype(float),dx=.1)),all_samples_pass=bool(good[sl].all()))
    bad=np.flatnonzero(~good[2000:])
    prefix=float(max(0,int(bad[0])-1)*.1) if len(bad) else 200.
    out=dict(loss=stages['formal']['loss'],preparation=stages['preparation'],formal=stages['formal'],
        preparation_terminal20ms_pass=bool(good[1800:2001].all()),formal_prefix_ms=prefix,
        first_failures_ms={name:float(np.flatnonzero(fail)[0]*.1) if fail.any() else None for name,fail in
            [('height',height<.85),('load',load<.8),('legs',legs<3),('angle',angle>.35),('nonfoot',depth>1e-12),('limit',excess>1e-6)]},
        physical_units='NOT_ESTABLISHED',objective='FINAL200MS_PHYSICAL_LOSS_ONLY',behavior_claim='NONE')
    out['eligible']=out['preparation_terminal20ms_pass'] and out['formal']['all_samples_pass']
    if not np.isfinite(out['loss']):raise ValueError('nonfinite objective')
    return out


def compare_scores(baseline,candidate,tolerance=1e-12):
    checks={}
    for phase in ('preparation','formal'):
        b,c=baseline[phase],candidate[phase]
        checks[phase]={k:bool(c[k]>=b[k]-tolerance) if k=='min_height_ratio' else bool(c[k]<=b[k]+tolerance)
            for k in ('min_height_ratio','max_downward_velocity','max_angle_rad','max_nonfoot_depth','max_joint_limit_excess')}
    improved=bool(candidate['formal']['support_ms']>baseline['formal']['support_ms']+tolerance)
    accepted=improved and all(all(x.values()) for x in checks.values())
    return dict(safety_nonworsening=checks,formal_support_strict_improvement=improved,
                accepted=accepted,decision='ELIGIBLE_FOR_LATER_VALIDATION' if accepted else 'REJECT_FORMAL_TRAINING_UPDATE',
                note='loss decrease alone insufficient; improvement is not continuous support or walking')
