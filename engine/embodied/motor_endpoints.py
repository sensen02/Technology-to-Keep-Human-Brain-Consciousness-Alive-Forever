"""Append verified motor endpoints without reordering existing local neurons.

No edge sign inference: caller uses unchanged strict build_edges afterwards.
"""
import numpy as np
from .neural_command import _ids


def retain_motor_endpoints(data, tier, root_ids, *, max_neurons=20000):
    requested=_ids(root_ids)
    if type(max_neurons) is not int or not 1<=max_neurons<=20000:
        raise ValueError('invalid capacity')
    roots=np.asarray(data['root_ids']);lookup={str(r):i for i,r in enumerate(roots)}
    _ids(roots)
    old=np.asarray(tier['global_index'])
    if old.ndim!=1 or old.dtype.kind not in 'iu' or len(np.unique(old))!=len(old) or np.any(old<0) or np.any(old>=len(roots)):
        raise ValueError('invalid original tier indices')
    added=[]
    for rid in requested:
        if rid not in lookup:raise ValueError('motor endpoint absent from dataset')
        g=lookup[rid]
        if str(data['ann_super_class'][g])!='motor':raise ValueError('target is not annotated motor')
        if g not in set(old.tolist()):added.append(g)
    gi=np.concatenate([old,np.asarray(added,dtype=np.int64)])
    if len(gi)>max_neurons:raise ValueError('endpoint union exceeds capacity; no truncation')
    loc=np.full(len(roots),-1,dtype=np.int64);loc[gi]=np.arange(len(gi))
    pre,post=data['pre'],data['post']
    return {'n':len(gi),'global_index':gi,'local_of_global':loc,
            'inside_rows':(loc[pre]>=0)&(loc[post]>=0),
            'retained_endpoint_root_ids':list(requested),'added_root_ids':[str(roots[g]) for g in added]}
