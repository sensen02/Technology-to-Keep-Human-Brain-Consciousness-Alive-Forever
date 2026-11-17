"""Parent-called pureNumPy actuator algebra/encoding; no IO/backend/Core/NN."""

def replay_target(actions,initial,bounds,max_rate):
    import numpy as np
    actions=np.asarray(actions,dtype=np.float64);target=np.asarray(initial,dtype=np.float64).copy();bounds=np.asarray(bounds,dtype=np.float64);max_rate=np.asarray(max_rate,dtype=np.float64)
    if actions.ndim!=2 or actions.shape[1]!=42 or target.shape!=(42,) or bounds.shape!=(42,2) or max_rate.shape!=(42,) or not all(np.isfinite(v).all() for v in (actions,target,bounds,max_rate)):raise ValueError('Finiteactual42actuatoralgebra inputs')
    rows=[];ratecap=np.zeros(42,dtype=int);anglecap=np.zeros(42,dtype=int)
    for action in actions:
        rate=np.clip(action*2.,-2.,2.);limited=np.clip(rate,-max_rate,max_rate)
        for _ in range(5):
            proposed=target+limited*.001;after=np.clip(proposed,bounds[:,0],bounds[:,1]);ratecap+=(rate!=limited);anglecap+=(proposed!=after);rows.append(after.copy());target=after
    return np.asarray(rows),dict(rate_clip_count_per_channel=ratecap.tolist(),target_clip_count_per_channel=anglecap.tolist(),rate_clip_count=int(ratecap.sum()),target_clip_count=int(anglecap.sum()))


def actual_replay(trace,actions,sample_ms,bounds,max_rate):
    import numpy as np
    ms=np.asarray(sample_ms,dtype=int)
    if not len(ms) or ms[0]!=5800 or not np.array_equal(ms,5800+np.arange(len(ms))*5):raise ValueError('Exactpreaction5800/5ms clock')
    if abs(float(trace['time'][5799])-5.8)>1e-8:raise ValueError('Tracepreactionindex5799')
    predicted,caps=replay_target(actions,trace['target'][5799],bounds,max_rate);actual=trace['target'][5800:];predicted=predicted[:len(actual)]
    held=np.repeat(actions,5,axis=0)[:len(actual)];delta=predicted-actual
    return dict(max_abs_target_error=float(np.max(np.abs(delta))),target_RMS_error=float(np.sqrt(np.mean(delta**2))),actual_action_hold_exact=bool(np.array_equal(held,trace['action'][5800:])),samples_1ms=len(actual),samples5ms=len(actions),source_preaction_index=5799,source_postslot_first_index=5800,first_post5ms_index=5804,passed=bool(np.max(np.abs(delta))<=1e-12 and np.array_equal(held,trace['action'][5800:])),caps=caps),predicted


def encoding_check(trace,X,sample_ms,mapping):
    import numpy as np
    ms=np.asarray(sample_ms,dtype=int);index=ms-1;qa=np.asarray(mapping['qpos_indices'],dtype=int);q0=np.asarray(mapping['q0_42']);half=np.asarray(mapping['halfspan42']);q=trace['qpos'][index][:,qa];target=trace['target'][index]
    error=np.clip((target-q)/half,-10.,10.).astype(np.float32).astype(np.float64);offset=np.clip((target-q0)/half,-10.,10.).astype(np.float32).astype(np.float64)
    return dict(target_error_float32_exact=bool(np.array_equal(error,X[:,42:84])),target_position_float32_exact=bool(np.array_equal(offset,X[:,126:168])),target_error_max_abs_difference=float(np.max(np.abs(error-X[:,42:84]))),target_position_max_abs_difference=float(np.max(np.abs(offset-X[:,126:168]))),passed=bool(np.array_equal(error,X[:,42:84]) and np.array_equal(offset,X[:,126:168])),source_double_target_not_inverted_from_float32=True)


def compare_targets(predicted,actual,predactions,actualactions):
    import numpy as np
    error=predicted-actual;ae=predactions-actualactions
    return dict(conditional_target_RMS=float(np.sqrt(np.mean(error**2))),conditional_target_absmax=float(np.max(np.abs(error))),conditional_target_final_RMS=float(np.sqrt(np.mean(error[-1]**2))),conditional_target_final_absmax=float(np.max(np.abs(error[-1]))),per_channel_target_RMS=np.sqrt(np.mean(error**2,axis=0)).tolist(),per_channel_target_peak=np.abs(error).max(0).tolist(),per_channel_target_final_signed=error[-1].tolist(),per_channel_signed_mean_action_error=ae.mean(0).tolist(),per_channel_action_error_RMS=np.sqrt(np.mean(ae**2,axis=0)).tolist(),global_action_error_RMS=float(np.sqrt(np.mean(ae**2))),global_signed_action_error_mean=float(ae.mean()),scope='SourcefixedX savedNNprediction conditionalaccumulator only, not target-feedback NN or physicalcounterfactual')
