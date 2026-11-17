"""Parent numerical objective helpers; imports Torch only when called."""
import os
import random
import resource
import time


def cpu1():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+'=1 required')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    import torch
    torch.set_num_threads(1)


def rng_snapshot():
    import torch,numpy as np
    return dict(torch=torch.get_rng_state().clone(),numpy=np.random.get_state(),python=random.getstate())


def rng_equal(a,b):
    import torch,numpy as np
    return torch.equal(a['torch'],b['torch']) and a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]


def restore_rng(rng):
    import torch,numpy as np
    torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])


def objective(residual,oldmean,y,v1,m,anchor):
    # Sum globally normalized minibatch contributions; exactly2micro ==fullB8.
    action=(oldmean+residual).tanh();T=action.shape[0]
    action_term=(action-y).square().sum()/(T*42*8)
    retain_term=(residual.square()*v1[None,:,None]).sum()/(T*42*5)
    parameters=dict(m.named_parameters());total=sum(value.numel() for value in anchor.values())
    anchor_term=sum((parameters[name]-value).square().sum() for name,value in anchor.items())/total*(action.shape[1]/8.)
    loss=action_term+.01*retain_term+.001*anchor_term
    return loss,dict(action_MSE_contribution=float(action_term.detach()),v1_residual_MSE_contribution=float(retain_term.detach()),newparam_anchor_MSE_contribution=float(anchor_term.detach()))


def adapter_anchor(m):return {name:p.detach().clone() for name,p in m.named_parameters() if p.requires_grad}


def metrics(scan,data):
    import torch
    rows=[]
    with torch.no_grad():
        for i,raw in enumerate(data['raw']):
            residual,h=scan(data['features'][:,i:i+1]);prediction=(data['oldmean'][:,i:i+1]+residual).tanh();target=data['y'][:,i:i+1]
            rows.append(dict(id=raw['entry']['id'],action_MSE=float((prediction-target).square().mean()),action_RMSE=float((prediction-target).square().mean().sqrt()),residual_absmax=float(residual.abs().max()),residual_RMS=float(residual.square().mean().sqrt()),newhidden_absmax=float(h.abs().max()),mean_action_absmax=float(prediction.abs().max())))
    return rows


def resource_gate(start,limit_s=1800.):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
    if rss>1800:raise RuntimeError('RSS budget exceeded1800MiB')
    if time.monotonic()-start>limit_s:raise RuntimeError('Wall budget exceeded; no extension')
    return rss
