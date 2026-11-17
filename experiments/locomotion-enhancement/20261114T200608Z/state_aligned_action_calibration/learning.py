"""State-aligned actionlabels only; conditional branch has NO actionlabel MSE."""
from pathlib import Path
import ast,resource,time
ORIGINAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration/learning.py')
_ns={'__name__':'_statealigned_original_pure_learning','__file__':str(ORIGINAL)}
exec(compile(ast.parse(ORIGINAL.read_text()),str(ORIGINAL)+'::EXACT_PURE_HELPERS','exec'),_ns)
for _name in ('cpu1','rng_snapshot','rng_equal','rng_same','restore_rng','adapter_anchor','time_weights'):
    globals()[_name]=_ns[_name]
_original_slice=_ns['slice_data'];_flags=_ns['_flags']

def resource_gate(start,limit_s=7200.):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('Statealigned trainingRSSexceeds1800MiB')
    if time.monotonic()-start>limit_s:raise RuntimeError('Statealigned wall7200sexceeded,noextension')
    return rss

def slice_data(data,lo,hi,T=None):
    result=_original_slice(data,lo,hi,T=T)
    for name in ('basehead','right','left','actual_q'):
        if name in data:result[name]=data[name][:T,lo:hi] if T is not None else data[name][:,lo:hi]
    return result

def tf_forward(scan,data):
    residual,h=scan(data['features']);mean=data['oldmean']+residual
    return mean,residual,h,mean.tanh()

def forward(scan,data,forced=False):
    return scan(data['physical'],data['oldafter'],data['commands'],data['basehead'],data['right'],data['left'],data['gain'],data['initial_target'],data['q0'],data['halfspan'],data['bounds'],data['max_rate'],data['actual_pre_targets'] if forced else None)

def tf_objective(out,data):
    import torch
    mean,residual,h,action=out;y=data['y'];valid=data['valid']
    if action.ndim!=3 or action.shape[-1]!=42 or action.shape!=y.shape or not 1<=action.shape[1]<=12:raise ValueError('Actualsource action/y[T,B,42]')
    if not bool(torch.isfinite(y).all()):raise ValueError('Finiteunchangedrawactionlabelsrequired')
    weights=time_weights(len(action),action,valid)
    term=((action-y).square()*weights[:,:,None]).sum()/(42*12)
    return term,dict(action_MSE_contribution=float(term.detach()),TF_action_MSE_contribution=float(term.detach()),qualified_action_rows=int(valid.sum().detach()),supervised_sequence_mass=action.shape[1]/12.)

def conditional_objective(out,data,m,anchor):
    """No access to y/valid: conditional states have target/prior/anchor supervision."""
    import torch
    mean,residual,h,pretarget,posttarget,action=out
    if residual.ndim!=3 or residual.shape[-1]!=42:raise ValueError('Conditionalresidual[T,B,42]')
    length,batch,_=residual.shape
    if not 1<=batch<=12 or posttarget.shape!=residual.shape or data['actual_post_targets'].shape!=residual.shape:raise ValueError('Alignedrealizedposttargethistoryrequired')
    selector=_flags(data['target_selector'],batch,residual,'target_selector9');v1=_flags(data['v1'],batch,residual,'v1prior6');half=data['halfspan']
    if half.shape!=(42,) or not bool(torch.isfinite(half).all()) or bool((half<.1).any()):raise ValueError('Actualhalfspan42>=.1')
    target_term=((((posttarget-data['actual_post_targets'])/half).square())*selector[None,:,None]).sum()/(length*42*9)
    prior_term=(residual.square()*v1[None,:,None]).sum()/(length*42*6)
    parameters=dict(m.named_parameters());expected={n for n,p in m.named_parameters() if not n.startswith('old.')};count=sum(v.numel() for v in anchor.values())
    if set(anchor)!=expected or len(anchor)!=8 or count<=0:raise ValueError('Exactoriginalfcbd8tensoranchorrequired')
    anchor_term=sum((parameters[k]-v).square().sum() for k,v in anchor.items())/count*(batch/12.)
    loss=target_term+.1*prior_term+.001*anchor_term
    return loss,dict(target_normalized_MSE_contribution=float(target_term.detach()),v1_residual_MSE_contribution=float(prior_term.detach()),newparam_anchor_MSE_contribution=float(anchor_term.detach()),target_supervised_sequences=int(selector.sum().detach()),conditional_action_label_loss_used=False)

def metrics(actualscan,condscan,data):
    """TF sourcefit primary; conditional actiondistance descriptive, never its loss."""
    import torch
    if data['features'].shape[:2]!=(1920,12):raise ValueError('Full12x1920metricsrequired')
    actualrows=[];condrows=[]
    with torch.no_grad():
        for i in range(12):
            part=slice_data(data,i,i+1);tf=tf_forward(actualscan,part);mask=part['valid'][:,0].bool();weights=time_weights(1920,tf[3],part['valid']);squared=(tf[3]-part['y']).square();masked=float(squared[:,0][mask].mean());weighted=float((squared*weights[:,:,None]).sum()/42.)
            identity=data['raw'][i]['entry']['id'];actualrows.append(dict(id=identity,valid_rows=int(mask.sum()),qualified_rows=int(mask.sum()),total_history_rows=1920,masked_action_MSE=masked,masked_action_RMSE=masked**.5,unmasked_action_MSE=float(squared.mean()),unmasked_action_RMSE=float(squared.mean().sqrt()),weighted_action_MSE=weighted,time_weighted_masked_action_MSE=weighted,action_MSE=masked,action_RMSE=masked**.5,equal12_weighted_loss_contribution=weighted/12.,residual_absmax=float(tf[1].abs().max()),residual_RMS=float(tf[1].square().mean().sqrt()),newhidden_absmax=float(tf[2].abs().max()),mean_action_absmax=float(tf[3].abs().max()),scope='ActualrecordedX statealigned supervisedactionfit only, notbodyPASS'))
            del tf,squared
            out=forward(condscan,part);delta=out[4]-part['actual_post_targets'];normalized=delta/part['halfspan'];selected=bool(part['target_selector'][0]);distance=(out[5]-part['y']).square();conditional_distance=float(distance[:,0][mask].mean())
            condrows.append(dict(id=identity,valid_rows=int(mask.sum()),total_history_rows=1920,target_supervision_selected=selected,target_normalized_MSE=float(normalized.square().mean()) if selected else None,target_normalized_RMSE=float(normalized.square().mean().sqrt()) if selected else None,target_absolute_peak_rad=float(delta.abs().max()) if selected else None,target_final_RMSE_rad=float(delta[-1].square().mean().sqrt()) if selected else None,target_final_normalized_RMSE=float(normalized[-1].square().mean().sqrt()) if selected else None,target_final_absolute_peak_rad=float(delta[-1].abs().max()) if selected else None,target_predicted_min_rad=float(out[4].min()),target_predicted_max_rad=float(out[4].max()),residual_RMS=float(out[1].square().mean().sqrt()),residual_absmax=float(out[1].abs().max()),newhidden_absmax=float(out[2].abs().max()),conditional_action_distance_MSE=conditional_distance,conditional_action_distance_RMSE=conditional_distance**.5,conditional_action_distance_unmasked_MSE=float(distance.mean()),conditional_action_distance_weighted_MSE=float((distance*weights[:,:,None]).sum()/42.),masked_action_MSE=conditional_distance,masked_action_RMSE=conditional_distance**.5,unmasked_action_MSE=float(distance.mean()),weighted_action_MSE=float((distance*weights[:,:,None]).sum()/42.),conditional_action_label_loss_used=False,distance_semantics='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION',scope='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION',physical_scope='CONDITIONAL_NOT_PHYSICAL_COUNTERFACTUAL',neural_acceptance=False))
    return dict(actualX_rows=actualrows,conditional_rows=condrows)
