"""Masked equal12-sequence correction loss; parent-only numerical functions."""
from pathlib import Path
import ast
import time
import resource

_UTILS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/learning.py')
_tree=ast.parse(_UTILS.read_text());_names={'cpu1','rng_snapshot','rng_equal','restore_rng','adapter_anchor'}
_nodes=[n for n in _tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in _names]
_namespace={'__name__':'_localmean_immutable_learning_utilities','__file__':str(_UTILS)}
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(_UTILS)+'::EXACT_CPU_RNG_ANCHOR_HELPERS_ONLY','exec'),_namespace)
for _name in _names:globals()[_name]=_namespace[_name]
rng_same=rng_equal


def resource_gate(start,limit_s=3600.):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('RSS budget exceeded1800MiB')
    if time.monotonic()-start>limit_s:raise RuntimeError('Wall budget exceeded; no automatic extension')
    return rss


def _valid_tensor(valid,length,batch,reference):
    import torch
    if valid is None:raise ValueError('Explicit qualified-label mask required; no implicit allvalid fallback')
    if not torch.is_tensor(valid) or tuple(valid.shape)!=(length,batch) or valid.device!=reference.device:raise ValueError('Valid mask must align full time/batch on same device')
    if not bool(torch.isfinite(valid).all()) or not bool(((valid==0)|(valid==1)).all()):raise ValueError('Qualified mask must contain only0/1')
    mask=valid.to(dtype=reference.dtype)
    if not bool((mask.sum(0)>0).all()):raise ValueError('Every sequence requires at leastone qualified label, never fallback')
    return mask


def time_weights(length,reference,valid=None):
    """Return[T,B] normalized per-sequence weights, invalidloss0, historyuntouched.
    The caller must retain every timestep in its recurrent rollout. Length32 is
    allowed for auxiliary numerical checks, not a substitute for full1920BPTT.
    """
    if reference.ndim<2 or length!=reference.shape[0] or length<20:raise ValueError('Aligned time/batch reference and >=20ticks required')
    mask=_valid_tensor(valid,length,reference.shape[1],reference)
    base=reference.new_ones((length,));base[:20]=10.
    weighted=base[:,None]*mask;denominator=weighted.sum(0,keepdim=True)
    if not bool((denominator>0).all()):raise ValueError('No qualified weighted mass in sequence')
    return weighted/denominator


def objective(residual,oldmean,y,v1,m,anchor,valid=None):
    import torch
    if residual.shape!=oldmean.shape or y.shape!=residual.shape or residual.ndim!=3 or residual.shape[2]!=42:raise ValueError('Aligned full-history residual/oldmean/Y42 required')
    T,B,_=residual.shape
    if B<1 or B>12 or v1.shape!=(B,) or not bool(torch.isfinite(v1).all()) or not bool(((v1==0)|(v1==1)).all()):raise ValueError('Microbatch prior flags must be explicit0/1')
    if not bool(torch.isfinite(y).all()):raise ValueError('Actual raw labels must stayfinite even when masked; no NaNzero replacement')
    weights=time_weights(T,residual,valid)
    action=(oldmean+residual).tanh()
    action_term=((action-y).square()*weights[:,:,None]).sum()/(42*12)
    retain_term=(residual.square()*v1[None,:,None]).sum()/(T*42*6)
    parameters=dict(m.named_parameters());total=sum(value.numel() for value in anchor.values())
    if not anchor or total<=0:raise ValueError('Migrated new8 parameter anchor required')
    anchor_term=sum((parameters[name]-value).square().sum() for name,value in anchor.items())/total*(B/12.)
    loss=action_term+.1*retain_term+.001*anchor_term
    return loss,dict(action_MSE_contribution=float(action_term.detach()),v1_residual_MSE_contribution=float(retain_term.detach()),newparam_anchor_MSE_contribution=float(anchor_term.detach()),qualified_rows=int(valid.sum().detach()),supervised_sequence_mass=B/12.)


def metrics(scan,data):
    """Twelve training metrics expose masked and raw fits; never DEV teacherlabels."""
    import torch
    if data['features'].shape[:2]!=(1920,12) or data['valid'].shape!=(1920,12):raise ValueError('Metrics require12 complete maskedtrainingsequences')
    rows=[]
    with torch.no_grad():
        for i,raw in enumerate(data['raw']):
            if raw.get('labels_available') is not True or raw.get('training_input') is not True:raise ValueError('UnlabeledDEV cannot acquire training fit metrics')
            residual,h=scan(data['features'][:,i:i+1]);prediction=(data['oldmean'][:,i:i+1]+residual).tanh();target=data['y'][:,i:i+1]
            mask=_valid_tensor(data['valid'][:,i:i+1],1920,1,prediction);valid_rows=mask[:,0].bool();err=(prediction-target).square()
            masked=float(err[valid_rows].mean());raw_mse=float(err.mean());weights=time_weights(1920,prediction,mask)
            weighted=float((err*weights[:,:,None]).sum()/42.)
            rows.append(dict(id=raw['entry']['id'],valid_rows=int(valid_rows.sum()),qualified_rows=int(valid_rows.sum()),total_history_rows=1920,masked_action_MSE=masked,masked_action_RMSE=masked**.5,unmasked_action_MSE=raw_mse,unmasked_action_RMSE=raw_mse**.5,weighted_action_MSE=weighted,time_weighted_masked_action_MSE=weighted,equal12_weighted_loss_contribution=weighted/12.,action_MSE=masked,action_RMSE=masked**.5,action_MSE_definition='qualified rows only; raw/unmasked andweightedvalues separatelyreported',residual_absmax=float(residual.abs().max()),residual_RMS=float(residual.square().mean().sqrt()),newhidden_absmax=float(h.abs().max()),mean_action_absmax=float(prediction.abs().max()),teacher_assisted=bool(raw['entry'].get('collection') and raw['entry']['teacher']),neural_acceptance=False))
    return rows
