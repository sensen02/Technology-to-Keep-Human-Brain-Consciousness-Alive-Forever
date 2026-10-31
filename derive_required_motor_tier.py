"""Append only explicitly required cache nodes; do not regrow neighbors."""
import copy
import numpy as np
from engine.embodied.adapters import build_neural_tier,TierConfig
from engine.neck_cut_data import load_banc,derive_crossing_set,build_cut_set,build_edges,TouchTierConfig
from pathlib import Path
P=Path(__file__).parent

def derive_tier(required_roots=(),base=None):
 cfg=TierConfig(background_mean_nA=0,background_sd_nA=0);base=base if base is not None else build_neural_tier(cfg)
 required=list(required_roots)
 if any(not isinstance(x,(int,np.integer)) for x in required):raise ValueError('integer roots required')
 if len(set(required))!=len(required):raise ValueError('duplicate required root')
 if not required:return base
 a=load_banc(str(P/cfg.data_path));lookup={int(r):i for i,r in enumerate(a['root_ids'])}
 if any(r not in lookup for r in required):raise ValueError('required root missing from cache')
 old=base['tier'];added=[lookup[r] for r in required if r not in set(map(int,old['root_ids']))];gi=np.r_[old['global_index'],added].astype(np.int64);loc=np.full(len(a['root_ids']),-1,dtype=np.int64);loc[gi]=np.arange(len(gi));keep=loc>=0
 tier=dict(old,n=len(gi),global_index=gi,local_of_global=loc,keep_mask=keep,root_ids=a['root_ids'][gi].astype(np.int64),inside_rows=keep[a['pre']]&keep[a['post']])
 cut=build_cut_set(a,derive_crossing_set(a));tcfg=TouchTierConfig(max_neurons=len(gi),partner_min_synapses=cfg.partner_min_synapses,weight_scale_uS_per_synapse=cfg.weight_scale_uS_per_synapse)
 edges=build_edges(a,tier,cut,cfg.sign_mode,'intact',tcfg,sign_fill=cfg.sign_fill);out=dict(base,n=len(gi),tier=tier,edges=edges)
 for name in ('We','Wi'):
  assert (edges[name][:old['n'],:old['n']]!=base['edges'][name]).nnz==0
 return out
