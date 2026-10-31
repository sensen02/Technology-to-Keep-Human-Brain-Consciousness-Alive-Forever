"""Source-real DNp09 augmentation and explicitly declared transient neural input.
No muscle trajectory, new synapse, physical assistance or autonomous-intent claim.
"""
from dataclasses import dataclass
import hashlib,json
import numpy as np

DNP09_ROOTS=(720575941433155799,720575941566493282)
@dataclass(frozen=True)
class DescendingCommand:
    amplitude_nA: float=0.
    duration_s: float=.050
    def validate(self):
        if not np.isfinite(self.amplitude_nA) or self.amplitude_nA<0 or not np.isfinite(self.duration_s) or self.duration_s<0:
            raise ValueError('finite nonnegative declared neuronal command required')
        return self

def command_input(built,config,relative_s):
    """Exact two source-identified DN cells only; identically zero after withdrawal."""
    config.validate();roots=np.asarray(built['tier']['root_ids']);ids=[]
    for r in DNP09_ROOTS:
        found=np.flatnonzero(roots==r)
        if len(found)!=1:raise ValueError('DNp09 identity missing or ambiguous')
        ids.append(int(found[0]))
    value=config.amplitude_nA if relative_s<config.duration_s-1e-12 else 0.
    current=np.zeros(int(built['n']));current[ids]=value
    return current,dict(root_ids=list(DNP09_ROOTS),local_indices=ids,amplitude_nA=float(value),declared_duration_s=config.duration_s,external_neuronal_start_command=True)

def augment_dnp09(built):
    from engine.neck_cut_data import build_edges,TouchTierConfig
    from .adapters import TierConfig
    c=TierConfig();a=built['data'];oldn=int(built['n']);gi=np.asarray(built['global_index']);indices=[]
    for root in DNP09_ROOTS:
        found=np.flatnonzero(a['root_ids']==root)
        if len(found)!=1:raise ValueError('cache root is missing or ambiguous')
        i=int(found[0]);assert i not in set(gi.tolist())
        assert str(a['ann_primary_type'][i])=='DNp09' and str(a['ann_nt_verified'][i])=='acetylcholine'
        indices.append(i)
    newgi=np.r_[gi,indices];loc=np.full(len(a['root_ids']),-1,dtype=np.int64);loc[newgi]=np.arange(len(newgi));keep=loc>=0
    tier=dict(built['tier']);tier.update(n=len(newgi),global_index=newgi,root_ids=a['root_ids'][newgi],local_of_global=loc,keep_mask=keep,inside_rows=keep[a['pre']]&keep[a['post']])
    cfg=TouchTierConfig(max_neurons=len(newgi),partner_min_synapses=c.partner_min_synapses,weight_scale_uS_per_synapse=c.weight_scale_uS_per_synapse)
    edges=build_edges(a,tier,built['cut'],c.sign_mode,'intact',cfg,sign_fill=c.sign_fill)
    for name in ['We','Wi']:
        delta=edges[name][:oldn,:oldn]-built['edges'][name];assert delta.nnz==0,'Existing neural edges changed'
    out=dict(built);out.update(tier=tier,n=len(newgi),global_index=newgi,local_of_global=loc,edges=edges,descending_local=np.r_[built['descending_local'],[oldn,oldn+1]])
    out['coverage']=dict(built['coverage']);out['coverage']['descending_augmentation']=dict(root_ids=list(DNP09_ROOTS),previous_n=oldn,new_n=len(newgi),old_local_order_and_edges_exact=True,source='cache annotated DNp09 acetylcholine, existing annotated rows only')
    out['fingerprint']=hashlib.sha256((built['fingerprint']+json.dumps(list(DNP09_ROOTS))).encode()).hexdigest()
    return out
