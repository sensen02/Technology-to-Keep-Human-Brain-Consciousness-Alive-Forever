"""Conditional statistics for parent-loaded saved arrays; NO file/model/physics I/O.
Only standard-library imports at module scope. NumPy is lazy inside public calls.
All action42 statistics are global/actuator-channel statistics, never leg mapping.
"""
import math

LEG_ORDER=('LF','LH','LM','RF','RH','RM')
WINDOWS=((0.,.05),(.05,.5),(.5,2.),(2.,9.6))
GROUPS=(('physical112',0,112),('oldhidden256',112,368),('task2',368,370))
TICK_NAMES={'physical112':'physical','oldhidden256':'oldhidden','task2':'task'}


def _json(value):
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_json(v) for v in value]
    if hasattr(value,'tolist'):return _json(value.tolist())
    if hasattr(value,'item'):return _json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def _array_stats(value):
    import numpy as np
    a=np.asarray(value)
    out=dict(shape=list(a.shape),elements=int(a.size))
    if a.size==0:return dict(out,status='EMPTY',finite_count=0,nonfinite_count=0)
    if a.dtype.kind not in 'biufc':return dict(out,status='NONNUMERIC')
    if a.dtype.kind=='c':return dict(out,status='COMPLEX_NOT_INTERPRETED')
    finite=np.isfinite(a);v=np.asarray(a[finite],dtype=float)
    out.update(finite_count=int(finite.sum()),nonfinite_count=int(a.size-finite.sum()))
    if not v.size:return dict(out,status='NO_FINITE_VALUES')
    out.update(status='MEASURED',minimum=float(v.min()),maximum=float(v.max()),mean=float(v.mean()),RMS=float(np.sqrt(np.mean(v*v))),absmax=float(np.abs(v).max()),abs_p50=float(np.quantile(np.abs(v),.5)),abs_p95=float(np.quantile(np.abs(v),.95)))
    if a.ndim==2 and a.shape[1] and bool(finite.all()):
        out.update(channel_min=a.min(0),channel_max=a.max(0),channel_RMS=np.sqrt(np.mean(a.astype(float)**2,axis=0)),channel_absmax=np.abs(a).max(0))
    return _json(out)


def _time_info(times,origin):
    import numpy as np
    a=np.asarray(times,dtype=float)
    if not len(a):return dict(samples=0,first_time_s=None,last_time_s=None,first_elapsed_s=None,last_elapsed_s=None)
    return dict(samples=len(a),first_time_s=float(a[0]),last_time_s=float(a[-1]),first_elapsed_s=float(a[0]-origin),last_elapsed_s=float(a[-1]-origin))


def _boundary(result,d):
    boundary=result.get('candidate_start')
    if isinstance(boundary,dict) and isinstance(boundary.get('time'),(int,float)):return float(boundary['time']),'result.candidate_start.time'
    if 'time' in d and len(d['time'])>1160:return float(d['time'][1160]),'neural_decision1160.time'
    return None,'UNKNOWN_NO_AUTONOMOUS_BOUNDARY'


def _first(mask,times,origin):
    import numpy as np
    hit=np.flatnonzero(np.asarray(mask,dtype=bool))
    if not len(hit):return dict(observed=False,time_s=None,elapsed_s=None,index_within_autonomous=None)
    i=int(hit[0]);return dict(observed=True,time_s=float(times[i]),elapsed_s=float(times[i]-origin),index_within_autonomous=i)


def _normalizer(identity):
    if isinstance(identity,dict) and isinstance(identity.get('actuator'),dict):return identity['actuator']
    return identity if isinstance(identity,dict) else {}


def _cap_stats(t,indices,actuator):
    import numpy as np
    out=dict(scope='Global42 actuator-order arrays; no action-to-leg map assumed',actual_qvel_cap_mapping='UNKNOWN_NO_EXPLICIT_ACTUATED_DOF_MAPPING',target_rate_note='Finite differences of saved actuator target; not measured physical joint velocity or a fabricated clipping flag')
    if 'actualctrl' in t and 'torque_limit' in actuator:
        ctrl=np.asarray(t['actualctrl'])[indices];limit=np.asarray(actuator['torque_limit'],dtype=float)
        if limit.ndim==1 and ctrl.ndim==2 and ctrl.shape[1]==len(limit) and len(ctrl):
            valid=np.isfinite(limit)&(limit>0)
            if valid.any():
                fraction=np.abs(ctrl[:,valid])/limit[valid]
                out['actual_ctrl_torque_fraction']=dict(valid_channels=int(valid.sum()),fraction=_array_stats(fraction),at_or_above_cap_fraction=float(np.mean(fraction>=1.-1e-8)),strict_exceed_fraction=float(np.mean(fraction>1.+1e-8)),channel_indices=np.flatnonzero(valid).tolist())
    if 'target' in t and 'target_bounds_rad' in actuator:
        target=np.asarray(t['target'])[indices];bounds=np.asarray(actuator['target_bounds_rad'],dtype=float)
        if target.ndim==2 and bounds.shape==(target.shape[1],2) and len(target):
            gap=np.minimum(np.abs(target-bounds[:,0]),np.abs(target-bounds[:,1]));at=gap<=1e-8
            out['target_native_bounds']=dict(at_bound_element_fraction=float(at.mean()),any_channel_at_bound_tick_fraction=float(at.any(1).mean()),per_channel_at_bound_fraction=at.mean(0).tolist(),outside_bound_element_fraction=float(np.mean((target<bounds[:,0]-1e-8)|(target>bounds[:,1]+1e-8))))
    if 'target' in t and 'time' in t and 'max_rate_rad_s' in actuator:
        target=np.asarray(t['target'],dtype=float);times=np.asarray(t['time'],dtype=float);limits=np.asarray(actuator['max_rate_rad_s'],dtype=float)
        use=np.asarray(indices,dtype=int);use=use[use>0]
        if len(use) and target.ndim==2 and limits.shape==(target.shape[1],):
            dt=times[use]-times[use-1];valid_ticks=dt>0
            rates=(target[use[valid_ticks]]-target[use[valid_ticks]-1])/dt[valid_ticks,None]
            valid=np.isfinite(limits)&(limits>0)
            if len(rates) and valid.any():
                fractions=np.abs(rates[:,valid])/limits[valid]
                out['target_rate_to_native_command_cap']=dict(samples=len(rates),rates=_array_stats(rates),fraction=_array_stats(fractions),at_or_above_cap_fraction=float(np.mean(fractions>=1.-1e-8)),strict_exceed_fraction=float(np.mean(fractions>1.+1e-8)),note='Measured target increment/native command cap; does not imply qvel clipping')
    return _json(out)


def raw_stats(t,d,g,result,identity):
    """Return JSONable raw statistics over source6, not warmup, plus fixed windows.
    Inputs are already-loaded dict arrays; this function never opens a file.
    identity may be full IDENTITY or its actuator subdict. Native extrema retain
    JSON provenance and cannot be assigned a substep timestamp from these arrays.
    """
    import numpy as np
    origin,origin_source=_boundary(result,d)
    if origin is None:return dict(status='UNAVAILABLE_NO_AUTONOMOUS_BOUNDARY',condition=result.get('condition'),scope='Saved conditional statistics, no causal or newphysical inference')
    times=np.asarray(t.get('time',[]),dtype=float);source=np.asarray(t.get('control_source',np.zeros(len(times),dtype=int)))
    ti=np.flatnonzero(source==6)
    neural_time=np.asarray(d.get('time',[]),dtype=float);di=np.arange(min(1160,len(neural_time)),len(neural_time),dtype=int)
    if 'control_source' in d:di=di[np.asarray(d['control_source'])[di]==6]
    gt=np.asarray(g.get('time',[]),dtype=float);gi=np.flatnonzero(np.asarray(g.get('control_source',np.zeros(len(gt),dtype=int)))==6)
    W=result.get('weight_native');L=result.get('body_length',{}).get('length_native') if isinstance(result.get('body_length'),dict) else None
    if not isinstance(W,(int,float)) or W<=0:W=None
    if not isinstance(L,(int,float)) or L<=0:L=None
    actuator=_normalizer(identity)
    def collect(tidx,didx,gidx):
        block=dict(trace_time=_time_info(times[tidx],origin),neural_time=_time_info(neural_time[didx],origin),geometry_time=_time_info(gt[gidx],origin),global42={},neural={},feet6={},body={})
        for key in ('mean','rawmean','oldmean','residual','policy_action','actually_applied_action','command'):
            if key in d:block['neural'][key]=_array_stats(np.asarray(d[key])[didx])
        for key in ('hidden_before','hidden_after'):
            if key in d:
                h=np.asarray(d[key])[didx];block['neural'][key]=_array_stats(h)
                if h.ndim==2 and h.shape[1]==352:block['neural'][key+'_old256']=_array_stats(h[:,:256]);block['neural'][key+'_new96']=_array_stats(h[:,256:])
        for key in ('action','actual_requested_action','target','actualctrl','actuator_force'):
            if key in t:block['global42'][key]=_array_stats(np.asarray(t[key])[tidx])
        for key in ('qpos','qvel','com','nonfoot','external_body_force_max','external_generalized_force_max'):
            if key in t:block['body'][key]=_array_stats(np.asarray(t[key])[tidx])
        if 'com' in t and len(tidx):
            com=np.asarray(t['com'])[tidx];b=result.get('candidate_start',{})
            if com.ndim==2 and com.shape[1]==3 and 'com' in b:
                offset=com[:,:2]-np.asarray(b['com'])[:2]
                block['body']['COM_XY_drift_native']=_array_stats(np.linalg.norm(offset,axis=1))
                if L:block['body']['COM_XY_drift_bodylength']=_array_stats(np.linalg.norm(offset,axis=1)/L)
        for key in ('loads','speed','padz'):
            if key in t:
                a=np.asarray(t[key])[tidx]
                if a.ndim==2 and a.shape[1]==6:
                    block['feet6'][key]=dict(order=list(LEG_ORDER),per_leg={leg:_array_stats(a[:,i]) for i,leg in enumerate(LEG_ORDER)})
                    if key=='loads' and W:block['feet6']['loads_over_weight']=dict(order=list(LEG_ORDER),per_leg={leg:_array_stats(a[:,i]/W) for i,leg in enumerate(LEG_ORDER)},summed=_array_stats(a.sum(1)/W))
        if 'thorax_yaw' in g:block['body']['actual_thorax_yaw_raw_rad']=_array_stats(np.asarray(g['thorax_yaw'])[gidx])
        block['actuator_caps']=_cap_stats(t,tidx,actuator)
        return block
    windows=[]
    for lo,hi in WINDOWS:
        def select(indices,clock):
            elapsed=clock[indices]-origin;upper=elapsed<=hi+1e-8 if hi==9.6 else elapsed<hi-1e-8
            return indices[(elapsed>=lo-1e-8)&upper]
        windows.append(dict(start_elapsed_s=lo,end_elapsed_s=hi,interval='[start,end]' if hi==9.6 else '[start,end)',statistics=collect(select(ti,times),select(di,neural_time),select(gi,gt))))
    onsets={}
    if len(ti):
        if 'nonfoot' in t:onsets['nonfoot_1ms_positive']=_first(np.asarray(t['nonfoot'])[ti]>0,times[ti],origin)
        if 'com' in t:onsets['COM_below1p2_native']=_first(np.asarray(t['com'])[ti,2]<1.2,times[ti],origin)
        if 'loads' in t and W:
            loads=np.asarray(t['loads'])[ti]/W
            onsets['summed_footload_below_p9W']=_first(loads.sum(1)<.9,times[ti],origin)
            onsets['any_leg_load_below_p05W']=_first(np.any(loads<.05,axis=1),times[ti],origin)
            onsets['per_leg_load_below_p05W']={leg:_first(loads[:,i]<.05,times[ti],origin) for i,leg in enumerate(LEG_ORDER)}
    handover=dict(status='UNAVAILABLE_NO_NEURAL1160',note='First autonomous decision vs last warm decision, not inferred continuous substep state')
    if len(neural_time)>1160:
        handover=dict(status='MEASURED',warm_index=1159,first_autonomous_index=1160,warm_time_s=float(neural_time[1159]),first_autonomous_time_s=float(neural_time[1160]),deltas={})
        for key in ('residual','mean','rawmean','oldmean','policy_action','actually_applied_action','command'):
            if key in d:
                a=np.asarray(d[key]);handover['deltas'][key]=_array_stats(a[1160]-a[1159])
        for key in ('hidden_before','hidden_after'):
            if key in d:
                a=np.asarray(d[key])
                if a.ndim==2 and a.shape[1]==352:
                    handover[key]=dict(first_old256=_array_stats(a[1160,:256]),first_new96=_array_stats(a[1160,256:]),old256_delta_vs_lastwarm=_array_stats(a[1160,:256]-a[1159,:256]),new96_delta_vs_lastwarm=_array_stats(a[1160,256:]-a[1159,256:]))
        if 'hidden_before' in d and 'hidden_after' in d:
            before=np.asarray(d['hidden_before']);after=np.asarray(d['hidden_after'])
            handover['first_autonomous_state_continuity']=dict(old256_before_vs_lastwarm_after=_array_stats(before[1160,:256]-after[1159,:256]),new96_before_absmax=float(np.abs(before[1160,256:]).max()),first_new96_after_absmax=float(np.abs(after[1160,256:]).max()))
    out=dict(status='MEASURED' if len(ti) else 'NO_SOURCE6_TRACE',condition=result.get('condition'),boundary_time_s=origin,boundary_source=origin_source,leg_order=list(LEG_ORDER),whole_autonomous=collect(ti,di,gi),windows=windows,first5ms_vs_lastwarm=handover,onsets_1ms=onsets,native_summary=result.get('native'),native_peak_time='UNKNOWN: only native summary available; do not assign native peak to a1ms sample',weight_native=W,body_length_native=L,action_leg_mapping='UNKNOWN: no explicit identity joint/actuator-to-leg mapping; action42 global only',scope='Conditional descriptive saved statistics, not causal mechanism or new physical counterfactual')
    return _json(out)


def _compare_arrays(a,b,wanted,times=None):
    import numpy as np
    x=np.asarray(a);y=np.asarray(b)
    if not x.ndim or not y.ndim:return dict(status='UNSUPPORTED_SCALAR_PREFIX',current_shape=list(x.shape),reference_shape=list(y.shape))
    n=min(wanted,len(x),len(y));out=dict(requested_rows=wanted,compared_rows=n,current_available=len(x),reference_available=len(y),complete=n==wanted,current_shape=list(x.shape),reference_shape=list(y.shape))
    if x.shape[1:]!=y.shape[1:]:return dict(out,status='TRAILING_SHAPE_MISMATCH',available_equal=False,max_abs_diff=None,first_difference_time_s=None)
    if not n:return dict(out,status='NO_OVERLAP',available_equal=None,max_abs_diff=None,first_difference_time_s=None)
    xx=x[:n];yy=y[:n];equal=xx==yy
    row_equal=np.all(equal.reshape(n,-1),axis=1) if equal.ndim>1 else equal
    bad=np.flatnonzero(~row_equal);numeric=xx.dtype.kind in 'biuf' and yy.dtype.kind in 'biuf';maximum=None;nonfinite=None
    if numeric:
        xf=xx.astype(float);yf=yy.astype(float);finite=np.isfinite(xf)&np.isfinite(yf);nonfinite=int(finite.size-finite.sum())
        if finite.any():maximum=float(np.max(np.abs(xf[finite]-yf[finite])))
    first=int(bad[0]) if len(bad) else None
    out.update(status='COMPARED',available_equal=bool(np.array_equal(xx,yy)),requested_prefix_equal=bool(n==wanted and np.array_equal(xx,yy)),differing_rows=int(len(bad)),max_abs_diff=maximum,nonfinite_pairs=nonfinite,first_difference_row=first,first_difference_time_s=float(times[first]) if first is not None and times is not None and len(times)>first else None)
    return _json(out)


def _hash_mapping(handover):
    if not isinstance(handover,dict):return None,None
    for key in ('physical_before_sha256','before','before_sha256'):
        value=handover.get(key)
        if isinstance(value,dict):
            accepted={str(k):v for k,v in value.items() if isinstance(v,str) and len(v)==64 and all(ch in '0123456789abcdef' for ch in v)}
            if accepted:return key,accepted
    return None,None


def compare_prefix(current_t,current_d,reference_t,reference_samples,reference_is_teacher,current_handover,reference_handover):
    """Quantify archived-prefix differences without asserting equality or fake state.
    Trace common fields use first5800rows. NNwarm uses actual960 oldmotor ticks;
    teacher samples have no recorded old neural state and remain UNKNOWN there.
    """
    import numpy as np
    common=sorted(set(current_t)&set(reference_t));clock=np.asarray(current_t.get('time',[]))
    trace={key:_compare_arrays(current_t[key],reference_t[key],5800,clock) for key in common}
    out=dict(trace_common_keys=common,trace_current_only=sorted(set(current_t)-set(reference_t)),trace_reference_only=sorted(set(reference_t)-set(current_t)),trace_prefix=trace,scope='Quantification only; mismatching/incomplete prefixes are not silently accepted or causal proof')
    ref_start=0 if reference_is_teacher else 200;current_clock=np.asarray(current_d.get('time',[]))[200:1160]
    if 'X238' in current_d and 'X238' in reference_samples:
        out['warm_X238']=_compare_arrays(np.asarray(current_d['X238'])[200:1160],np.asarray(reference_samples['X238'])[ref_start:ref_start+960],960,current_clock)
    else:out['warm_X238']=dict(status='UNKNOWN_MISSING_X238')
    if reference_is_teacher:out['warm_old256']=dict(status='UNKNOWN_TEACHER_DID_NOT_RECORD_NN_STATE',invented_reference_state=False)
    else:
        states={}
        for key in ('hidden_before','hidden_after'):
            if key in current_d and key in reference_samples:
                a=np.asarray(current_d[key]);b=np.asarray(reference_samples[key])
                if a.ndim==2 and b.ndim==2 and a.shape[1]>=256 and b.shape[1]>=256:
                    states[key]=_compare_arrays(a[200:1160,:256],b[200:1160,:256],960,current_clock)
                    states[key]['compared_old_columns']=256;states[key]['reference_recorded_width']=b.shape[1]
                else:states[key]=dict(status='UNKNOWN_UNSUPPORTED_STATE_SHAPE')
            else:states[key]=dict(status='UNKNOWN_MISSING_RECORDED_STATE')
        out['warm_old256']=states
    ck,cm=_hash_mapping(current_handover);rk,rm=_hash_mapping(reference_handover)
    if cm is None or rm is None:out['handover_hashes']=dict(status='UNKNOWN_NO_COMPARABLE_HASH_MAPPING',current_container=ck,reference_container=rk,note='No guesses about opaque/custom schemas')
    else:
        keys=sorted(set(cm)&set(rm))
        out['handover_hashes']=dict(status='COMPARED' if keys else 'UNKNOWN_NO_SHARED_HASH_KEYS',current_container=ck,reference_container=rk,common_keys=keys,equal_per_key={key:cm[key]==rm[key] for key in keys},target_equal=cm['target']==rm['target'] if 'target' in keys else None,buffer_equal=cm['buffer']==rm['buffer'] if 'buffer' in keys else None,current_only=sorted(set(cm)-set(rm)),reference_only=sorted(set(rm)-set(cm)))
    return _json(out)


def domain_stats(actual_features,training_features,actual_h352):
    """Return (JSONable summary, per_tick numpy arrays) for conditional domain drift.
    Training axes before feature370 are flattened, never treated as extra features.
    Z is defined only on finite nonconstant training columns. Exact min==max
    columns are separately reported even if floating std arithmetic has roundoff.
    """
    import numpy as np
    a=np.asarray(actual_features,dtype=float);tr=np.asarray(training_features,dtype=float);h=np.asarray(actual_h352,dtype=float)
    if a.ndim!=2 or a.shape[1]!=370 or tr.ndim not in (2,3) or tr.shape[-1]!=370 or h.shape!=(len(a),352):raise ValueError('Expected actual[T,370], training[N,370] or[T,8,370], actualhidden[T,352]')
    tr=tr.reshape(-1,370);per_tick={};T=len(a)
    if not len(tr) or not np.isfinite(tr).all() or not np.isfinite(a).all() or not np.isfinite(h).all():
        for name,_,_ in GROUPS:per_tick[TICK_NAMES[name]+'_exceed_fraction']=np.full(T,np.nan);per_tick[TICK_NAMES[name]+'_max_abs_z']=np.full(T,np.nan)
        per_tick['new96_saturation_fraction']=np.where(np.isfinite(h[:,256:]).all(1),np.mean(np.abs(h[:,256:])>.99,axis=1),np.nan) if T else np.empty(0)
        return dict(status='UNKNOWN_EMPTY_OR_NONFINITE_INPUT',actual_ticks=T,training_rows=len(tr),actual_nonfinite_count=int(np.count_nonzero(~np.isfinite(a))),training_nonfinite_count=int(np.count_nonzero(~np.isfinite(tr))),hidden_nonfinite_count=int(np.count_nonzero(~np.isfinite(h))),scope='No epsilon or zero replacement used to fabricate normalized values'),per_tick
    low=tr.min(0);high=tr.max(0);mean=tr.mean(0);std=tr.std(0);constant=low==high;std[constant]=0.;valid=(std>0)&~constant
    groups={}
    for name,start,end in GROUPS:
        x=a[:,start:end];lo=low[start:end];hi=high[start:end];mu=mean[start:end];sd=std[start:end];usable=valid[start:end];const=constant[start:end];outside=(x<lo)|(x>hi)
        exceed=outside.mean(1) if T else np.empty(0);per_tick[TICK_NAMES[name]+'_exceed_fraction']=exceed
        if usable.any():
            z=np.abs((x[:,usable]-mu[usable])/sd[usable]);zmax=z.max(1) if T else np.empty(0);zsummary=_array_stats(z)
        else:zmax=np.full(T,np.nan);zsummary=dict(status='UNKNOWN_NO_NONCONSTANT_COLUMNS')
        per_tick[TICK_NAMES[name]+'_max_abs_z']=zmax
        group=dict(feature_slice=[start,end],feature_count=end-start,training_min=lo.tolist(),training_max=hi.tolist(),training_mean=mu.tolist(),training_std=sd.tolist(),constant_local_columns=np.flatnonzero(const).tolist(),constant_global_columns=(np.flatnonzero(const)+start).tolist(),zero_std_count=int(const.sum()),z_usable_columns=int(usable.sum()),exceed_fraction=_array_stats(exceed),abs_z_on_nonconstant_columns=zsummary,per_feature_exceed_fraction=outside.mean(0).tolist() if T else None,constant_column_exceed_fraction=float(outside[:,const].mean()) if T and const.any() else None,constant_mismatch_tick_fraction=float(outside[:,const].any(1).mean()) if T and const.any() else None)
        groups[name]=group
    saturation=np.mean(np.abs(h[:,256:])>.99,axis=1) if T else np.empty(0);per_tick['new96_saturation_fraction']=saturation
    summary=dict(status='MEASURED' if T else 'EMPTY_ACTUAL',actual_ticks=T,training_rows=len(tr),feature_order=['physical112','oldhidden256','task2'],groups=groups,new96_abs=_array_stats(h[:,256:]),new96_abs_gt_p99_fraction=float(np.mean(np.abs(h[:,256:])>.99)) if T else None,new96_saturation_tick_fraction=_array_stats(saturation),old256_abs=_array_stats(h[:,:256]),constant_column_rule='training min==max => std0, excluded fromz, exact range violations reported separately',scope='Conditional saved feature range/z statistics only; domain excursion is not causal evidence or physicalcounterfactual')
    return _json(summary),per_tick
