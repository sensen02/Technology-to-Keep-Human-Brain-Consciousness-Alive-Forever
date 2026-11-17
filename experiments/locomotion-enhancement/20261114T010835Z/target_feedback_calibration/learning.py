"""Conditional target-feedback objective/helpers, stdlib-only on import.
No physical simulation or neural execution occurs until an authorized parent
calls these functions. Supervision masks never reset target or hidden history.
"""
from pathlib import Path
import ast
import resource
import time

_UTILS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/learning.py')
_TREE=ast.parse(_UTILS.read_text())
_NAMES={'cpu1','rng_snapshot','rng_equal','restore_rng','adapter_anchor'}
_NODES=[n for n in _TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in _NAMES]
_NS={'__name__':'_targetfeedback_exact_stdlib_rng_utilities','__file__':str(_UTILS)}
exec(compile(ast.Module(body=_NODES,type_ignores=[]),str(_UTILS)+'::CPU_RNG_ANCHOR_ONLY','exec'),_NS)
for _name in _NAMES:
    globals()[_name]=_NS[_name]
rng_same=rng_equal

_TIME_BATCH_KEYS=frozenset(('physical','oldafter','commands','old_nontarget','actual_pre_targets','actual_post_targets','y','valid','features','x','u','oldmean'))
_BATCH_KEYS=frozenset(('initial_target','target_selector','v1'))
_CONSTANT_KEYS=frozenset(('gain','q0','halfspan','bounds','max_rate'))


def resource_gate(start,limit_s=5400.):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('Target-feedback RSS exceeds1800MiB')
    if time.monotonic()-start>limit_s:raise RuntimeError('Target-feedback wall budget exceeded; no extension')
    return rss


def slice_data(data,lo,hi,T=None):
    """View/select a batch, optionally firstT actual ticks, without resetting state.

    Constants42/42x2 are NEVER guessed as batch dimensions. Optional timeprefix
    is an auxiliary numerical probe; full training uses entire1920 history.
    Source dictionaries are not modified and no tensors are deepcopied.
    """
    batch=data['physical'].shape[1]
    if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=batch:
        raise ValueError('Explicit valid batch slice required')
    if T is not None and (type(T) is not int or not 1<=T<=data['physical'].shape[0]):
        raise ValueError('Actual time prefix1..source_length required')
    result={}
    for key,value in data.items():
        if key in _TIME_BATCH_KEYS:
            result[key]=value[:T,lo:hi] if T is not None else value[:,lo:hi]
        elif key in _BATCH_KEYS:
            result[key]=value[lo:hi]
        elif key in _CONSTANT_KEYS:
            result[key]=value
        elif key in ('raw','proofs') and isinstance(value,(tuple,list)) and len(value)==batch:
            result[key]=value[lo:hi]
        elif key=='initial_old' and isinstance(value,(tuple,list)) and len(value)==3:
            result[key]=tuple(v[lo:hi] for v in value)
        else:
            result[key]=value
    return result


def forward(scan,data,forced=False):
    """One stable caller API. forced=True is actual-target parity/value checking.

    Ordinary training uses predicted target feedback. Neither mode changes the
    saved physical observations or simulates a new body trajectory.
    """
    forced_targets=data['actual_pre_targets'] if forced else None
    return scan(data['physical'],data['oldafter'],data['commands'],
                data['old_nontarget'],data['gain'],data['initial_target'],
                data['q0'],data['halfspan'],data['bounds'],data['max_rate'],
                forced_targets)


def _flags(value,batch,reference,name):
    import torch
    if not torch.is_tensor(value) or tuple(value.shape)!=(batch,) or value.device!=reference.device:
        raise ValueError(name+' must be an explicit aligned batch-vector')
    if not bool(torch.isfinite(value).all()) or not bool(((value==0)|(value==1)).all()):
        raise ValueError(name+' must contain only finite0/1 values')
    return value.to(dtype=reference.dtype)


def time_weights(length,reference,valid):
    """Per-sequence masked time mass1: first20 ticks10, all remainingticks1."""
    import torch
    if reference.ndim<2 or reference.shape[0]!=length or length<20:
        raise ValueError('Aligned actual history length>=20 required')
    if not torch.is_tensor(valid) or tuple(valid.shape)!=(length,reference.shape[1]) or valid.device!=reference.device:
        raise ValueError('Explicit aligned valid[T,B] mask required')
    if not bool(torch.isfinite(valid).all()) or not bool(((valid==0)|(valid==1)).all()):
        raise ValueError('Valid mask is finite0/1, not missing-label replacement')
    base=reference.new_ones((length,));base[:20]=10.
    mass=base[:,None]*valid.to(dtype=reference.dtype)
    denominator=mass.sum(0,keepdim=True)
    if not bool((denominator>0).all()):raise ValueError('All-invalid supervised sequence forbidden')
    return mass/denominator


def objective(out,data,m,anchor):
    """Global denominators12/9/6 make fullB12 ==sum of3 complete-history B4.

    The target path term applies ONLY to9 actually realized target sources.
    Its mask is target_selector, not assisted teacher label_qualified. All1920
    target states are supervised on selectedsources, including cap behavior.
    """
    import torch
    mean,residual,h,pretarget,posttarget,action=out
    y=data['y'];valid=data['valid']
    if action.ndim!=3 or action.shape[-1]!=42 or action.shape!=y.shape or residual.shape!=action.shape:
        raise ValueError('Aligned conditional action/residual/y[T,B,42] required')
    length,batch,_=action.shape
    if not 1<=batch<=12 or posttarget.shape!=action.shape or data['actual_post_targets'].shape!=action.shape:
        raise ValueError('Aligned actual post-slot target[T,B,42] required')
    if not bool(torch.isfinite(y).all()):raise ValueError('Raw targetY stays finite even when supervised mask0')
    weights=time_weights(length,action,valid)
    selector=_flags(data['target_selector'],batch,action,'target_selector9')
    v1=_flags(data['v1'],batch,action,'v1prior6')
    halfspan=data['halfspan']
    if halfspan.shape!=(42,) or not bool(torch.isfinite(halfspan).all()) or bool((halfspan<.1).any()):
        raise ValueError('Actual source halfspan42>=.1 required for target normalization')
    action_term=((action-y).square()*weights[:,:,None]).sum()/(42*12)
    target_error=(posttarget-data['actual_post_targets'])/halfspan
    target_term=(target_error.square()*selector[None,:,None]).sum()/(length*42*9)
    retain_term=(residual.square()*v1[None,:,None]).sum()/(length*42*6)
    parameters=dict(m.named_parameters());count=sum(v.numel() for v in anchor.values())
    if not anchor or count<=0:raise ValueError('Parent migrated8tensor anchor required')
    anchor_term=sum((parameters[k]-v).square().sum() for k,v in anchor.items())/count*(batch/12.)
    loss=action_term+target_term+.1*retain_term+.001*anchor_term
    return loss,dict(action_MSE_contribution=float(action_term.detach()),
                     target_normalized_MSE_contribution=float(target_term.detach()),
                     v1_residual_MSE_contribution=float(retain_term.detach()),
                     newparam_anchor_MSE_contribution=float(anchor_term.detach()),
                     qualified_action_rows=int(valid.sum().detach()),
                     target_supervised_sequences=int(selector.sum().detach()),
                     supervised_sequence_mass=batch/12.)


def metrics(scan,data,forced=False):
    """12 training diagnostics, masks exposed; no teacher labels inferred forDEV.

    Returns one row per sequence, compatible with prior calibration metrics.
    Global normalization lives in objective; these per-sequence scores are NOT
    body acceptance. target metrics are None for the3 assisted mixed sources.
    """
    import torch
    if data['physical'].shape[:2]!=(1920,12) or data['valid'].shape!=(1920,12):
        raise ValueError('Metrics require12 full1920-step source histories')
    rows=[]
    with torch.no_grad():
        for i in range(12):
            part=slice_data(data,i,i+1)
            mean,residual,h,pretarget,posttarget,action=forward(scan,part,forced=forced)
            mask=part['valid'][:,0].bool();error=(action-part['y']).square()
            weights=time_weights(1920,action,part['valid'])
            masked=float(error[mask].mean());raw_mse=float(error.mean())
            weighted=float((error*weights[:,:,None]).sum()/42.)
            selected=bool(part['target_selector'][0])
            target_delta=posttarget-part['actual_post_targets']
            normalized=target_delta/part['halfspan']
            entry=data['raw'][i]['entry'] if 'raw' in data else {'id':str(i)}
            row=dict(id=entry['id'],valid_rows=int(mask.sum()),qualified_rows=int(mask.sum()),
                     total_history_rows=1920,masked_action_MSE=masked,
                     masked_action_RMSE=masked**.5,unmasked_action_MSE=raw_mse,
                     unmasked_action_RMSE=raw_mse**.5,weighted_action_MSE=weighted,
                     time_weighted_masked_action_MSE=weighted,
                     equal12_weighted_loss_contribution=weighted/12.,
                     action_MSE=masked,action_RMSE=masked**.5,
                     residual_absmax=float(residual.abs().max()),
                     residual_RMS=float(residual.square().mean().sqrt()),
                     newhidden_absmax=float(h.abs().max()),
                     mean_action_absmax=float(action.abs().max()),
                     target_supervision_selected=selected,
                     target_normalized_MSE=float(normalized.square().mean()) if selected else None,
                     target_normalized_RMSE=float(normalized.square().mean().sqrt()) if selected else None,
                     target_absolute_peak_rad=float(target_delta.abs().max()) if selected else None,
                     target_final_RMSE_rad=float(target_delta[-1].square().mean().sqrt()) if selected else None,
                     target_final_normalized_RMSE=float(normalized[-1].square().mean().sqrt()) if selected else None,
                     target_final_absolute_peak_rad=float(target_delta[-1].abs().max()) if selected else None,
                     target_predicted_min_rad=float(posttarget.min()),
                     target_predicted_max_rad=float(posttarget.max()),
                     target_feedback_mode='forced_actual_pre_targets' if forced else 'predicted_target_register',
                     scope='Conditional saved-physical-history training metrics, not physicalcounterfactual or movementPASS',
                     neural_acceptance=False)
            rows.append(row)
    return rows
