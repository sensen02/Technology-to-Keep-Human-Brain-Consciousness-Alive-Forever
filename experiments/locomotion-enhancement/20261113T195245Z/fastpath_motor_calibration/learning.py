"""Exact utility reuse with explicitly new jointrepair objective; no backendimport."""
from pathlib import Path
import ast
import types

PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration')
g={'__name__':'_fastpath_parent_learning_helpers','__file__':str(PARENT/'learning.py')}
exec(compile(ast.parse((PARENT/'learning.py').read_text()),str(PARENT/'learning.py')+'::same_utilities','exec'),g)
for name in ('cpu1','rng_snapshot','rng_equal','restore_rng','adapter_anchor','metrics','resource_gate'):globals()[name]=g[name]


def time_weights(length,reference):
    if length<20:raise ValueError('Atleast20ticks forjointrepairweightcontract')
    w=reference.new_ones((length,));w[:20]=10.
    return w/w.sum()


def objective(residual,oldmean,y,v1,m,anchor):
    action=(oldmean+residual).tanh();T=action.shape[0];weights=time_weights(T,action)
    action_term=((action-y).square()*weights[:,None,None]).sum()/(42*8)
    retain_term=(residual.square()*v1[None,:,None]).sum()/(T*42*5)
    parameters=dict(m.named_parameters());total=sum(v.numel() for v in anchor.values())
    anchor_term=sum((parameters[k]-v).square().sum() for k,v in anchor.items())/total*(action.shape[1]/8.)
    loss=action_term+.1*retain_term+.001*anchor_term
    return loss,dict(action_MSE_contribution=float(action_term.detach()),v1_residual_MSE_contribution=float(retain_term.detach()),newparam_anchor_MSE_contribution=float(anchor_term.detach()))
