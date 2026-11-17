"""Exact qualified sourcecache and B1 frozencomponents; no new labels or numerics on import."""
from pathlib import Path
import ast
import sources
ORIGINAL=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_calibration')
_ns={'__name__':'_statealigned_original_feedback_data','__file__':str(ORIGINAL/'data.py')}
exec(compile(ast.parse((ORIGINAL/'data.py').read_text()),str(ORIGINAL/'data.py')+'::EXACT_SOURCE_CACHE_NEW_BINDING','exec'),_ns)
_original_prepare=_ns['prepare']
production_sequence=_ns['production_sequence']
dev_sources=_ns['dev_sources']

def prepare(m):
    import torch
    data=_original_prepare(m)
    if data['oldafter'].ndim!=3 or data['oldafter'].shape[2]!=256:raise ValueError('Explicitafterold256history')
    T,B,_=data['oldafter'].shape;base_sequences=[];right_sequences=[];left_sequences=[];proofs=[]
    with torch.no_grad():
        for i in range(B):
            bases=[];rights=[];lefts=[]
            for k in range(T):
                after=data['oldafter'][k,i:i+1]
                bh=after[:,:128];rh=after[:,128:192];lh=after[:,192:256]
                turn=data['commands'][k,i:i+1,1:2];positive=torch.relu(turn);negative=torch.relu(-turn)
                bases.append(m.old.base.head(torch.tanh(m.old.base.hidden(bh)))[0])
                rights.append((negative*m.old.right_branch.bound*torch.tanh(m.old.right_branch.head(rh)))[0])
                lefts.append((positive*m.old.left_branch.bound*torch.tanh(m.old.left_branch.head(lh)))[0])
            base_sequences.append(torch.stack(bases));right_sequences.append(torch.stack(rights));left_sequences.append(torch.stack(lefts))
        data['basehead']=torch.stack(base_sequences,dim=1).detach()
        data['right']=torch.stack(right_sequences,dim=1).detach()
        data['left']=torch.stack(left_sequences,dim=1).detach()
        for i,raw in enumerate(data['raw']):
            offset=data['x'][:,i,126:168]
            reconstructed=data['basehead'][:,i]-data['gain']*offset+data['right'][:,i]+data['left'][:,i]
            exact=torch.equal(reconstructed,data['oldmean'][:,i])
            proof=dict(id=raw['entry']['id'],rows=T,head_batch=1,actual_oldmean_components_exact=exact,max_abs_error=float((reconstructed-data['oldmean'][:,i]).abs().max()),frozen_order='((basehead-gain*offset)+weightedRIGHT)+weightedLEFT',weighted_order='negative*RIGHTbound*correction; positive*LEFTbound*correction')
            if not exact:raise ValueError('ExactsourceB1oldmean reconstruction mismatch: '+raw['entry']['id'])
            proofs.append(proof)
            raw.update(basehead=data['basehead'][:,i],right=data['right'][:,i],left=data['left'][:,i])
    data['exact_component_proofs']=proofs
    binding=data['binding'];sources.verify(binding['source_hashes']);sources.verify(binding['data_hashes'])
    return data
