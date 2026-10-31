"""Opt-in real-edge downstream union; historical tier builder stays unchanged."""
import numpy as np
from .neural_command import _ids


def downstream_union(data, base_global_index, target_root_ids, *, hops=2, max_neurons=20000):
    if type(hops) is not int or not 0 <= hops <= 4:
        raise ValueError('hops must be an integer in [0,4]')
    if type(max_neurons) is not int or not 1 <= max_neurons <= 20000:
        raise ValueError('capacity must be in [1,20000]')
    roots=np.asarray(data['root_ids']); n=len(roots)
    base=np.asarray(base_global_index)
    if base.ndim!=1 or base.dtype.kind not in 'iu' or len(np.unique(base))!=len(base) or np.any(base<0) or np.any(base>=n):
        raise ValueError('base indices must be unique in-range integers')
    _ids(roots)
    requested=_ids(target_root_ids)
    by_id={str(r):i for i,r in enumerate(roots)}
    if any(r not in by_id for r in requested):
        raise ValueError('source target absent from dataset')
    target=np.array([by_id[r] for r in requested],dtype=np.int64)
    pre,post=np.asarray(data['pre']),np.asarray(data['post'])
    if pre.shape!=post.shape or pre.ndim!=1 or pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu' or np.any(pre<0) or np.any(post<0) or np.any(pre>=n) or np.any(post>=n):
        raise ValueError('edge endpoints must be valid dataset indices')
    # Same pure verified presynaptic sign policy as neck_cut_data.build_edges.
    pure=np.isin(data['ann_nt_verified'],['acetylcholine','gaba','glutamate'])
    reached=target.copy(); history=[]
    for h in range(hops):
        selected=np.isin(pre,reached)&pure[pre]
        reached=np.union1d(reached,post[selected])
        history.append({'hop':h+1,'downstream_total':int(len(reached))})
    gi=np.union1d(base,reached)
    if len(gi)>max_neurons:
        raise ValueError(f'union requires {len(gi)} neurons > capacity {max_neurons}; no silent truncation')
    loc=np.full(n,-1,dtype=np.int64);loc[gi]=np.arange(len(gi))
    return {'n':len(gi),'global_index':gi,'local_of_global':loc,
            'inside_rows':(loc[pre]>=0)&(loc[post]>=0),
            'command_expansion':{'target_root_ids':list(requested),'target_local_indices':loc[target].tolist(),
                                 'hops':hops,'history':history,'base_n':int(len(base)),
                                 'added_n':int(len(gi)-len(base)),'max_neurons':max_neurons,
                                 'rule':'union of unchanged base and all pure-verified outgoing partners within hops; no invented edges'}}
