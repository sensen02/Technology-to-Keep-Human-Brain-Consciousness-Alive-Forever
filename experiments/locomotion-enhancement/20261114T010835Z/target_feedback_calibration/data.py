"""Exact parent12 cache plus readonly realized actuator/normalizer feedback data.
No numeric import on module import; no deploy actor operation is changed here.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import sys
import sources

PARENT=sources.PARENT
namespace={'__name__':'_targetfeedback_exact_parent12_data','__file__':str(PARENT/'data.py')}
exec(compile(ast.parse((PARENT/'data.py').read_text()),str(PARENT/'data.py')+'::EXACT_PARENT12_CACHE_NEW_ROOT','exec'),namespace)
namespace.update(ROOT=sources.ROOT,read=sources.read,verify=sources.verify)
_parent_prepare=namespace['prepare']
production_sequence=namespace['production_sequence']
dev_sources=namespace['dev_sources']


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def prepare(m):
    import numpy as np
    import torch
    data=_parent_prepare(m);binding=data['binding'];mapping=binding['normalizer_mapping'];am=_module('_targetfeedback_saved_normalizer_contract',sources.ACCUMULATOR/'metadata.py')
    q0=np.asarray(mapping['q0_42'],dtype=np.float64);half=np.asarray(mapping['halfspan42'],dtype=np.float64);qa=np.asarray(mapping['qpos_indices'],dtype=int)
    initials=[];pre=[];post=[];q=[];bounds_common=None;rate_common=None;proofs=[]
    for i,entry in enumerate(binding['entries']):
        with np.load(entry['trace_path'],allow_pickle=False) as archive:
            target=archive['target'].copy();qpos=archive['qpos'].copy();time=archive['time'].copy()
        if target.shape!=(15400,42) or qpos.shape!=(15400,49) or not np.allclose(time,np.arange(1,15401)*.001,atol=1e-8,rtol=0):raise ValueError('Exactfull15400 target/qpos/time source')
        identity=am.identity_contract(sources.read(entry['identity_path']))
        for name,value in (('jnt_range',mapping['joint_ranges43']),('key_qpos',[mapping['key_qpos49']])):
            record=identity['compiled_arrays'][name];array=np.asarray(value,dtype=np.dtype(record['dtype']))
            if list(array.shape)!=record['shape'] or hashlib.sha256(array.tobytes()).hexdigest()!=record['sha256']:raise ValueError('ActualcompilednormalizerSHA mismatch: '+name)
        bounds=np.asarray(identity['actuator']['target_bounds_rad'],dtype=np.float64);rate=np.asarray(identity['actuator']['max_rate_rad_s'],dtype=np.float64)
        native=np.asarray([mapping['joint_ranges43'][1+int(address-7)] for address in qa],dtype=np.float64)
        if bounds.shape!=(42,2) or rate.shape!=(42,) or not np.array_equal(native,bounds):raise ValueError('Explicitactual nativebounds/rate42')
        if bounds_common is None:bounds_common=bounds.copy();rate_common=rate.copy()
        elif not np.array_equal(bounds_common,bounds) or not np.array_equal(rate_common,rate):raise ValueError('All12samebody actuatorcontract required')
        indices=5799+np.arange(1920)*5;posts=5804+np.arange(1920)*5
        before=target[indices];after=target[posts];actualq=qpos[indices][:,qa]
        X=data['x'][:,i].numpy();encoded_error=np.clip((before-actualq)/half,-10.,10.).astype(np.float32).astype(np.float64);encoded_offset=np.clip((before-q0)/half,-10.,10.).astype(np.float32).astype(np.float64)
        if not np.array_equal(encoded_error,X[:,42:84]) or not np.array_equal(encoded_offset,X[:,126:168]):raise ValueError('Actualdoubletarget normalization sourceX mismatch')
        initials.append(target[5799]);pre.append(before);post.append(after);q.append(actualq)
        proofs.append(dict(id=entry['id'],preaction_trace_indices='5799+5*k',postaction_trace_indices='5804+5*k',target_selector=entry['target_selector'],compilednormalizer_SHA_exact=True,perceived_target_fields_float32_exact=True,actuator_contract_provenance=identity.get('actuator_contract_provenance','actualIDENTITYnumericlists')))
    with torch.no_grad():
        gain=m.old.base.gain().detach().clone()
        if gain.shape!=(42,) or not bool(torch.isfinite(gain).all()):raise ValueError('Strictfrozen42targetpositionpriorgain')
        data.update(physical=data['features'][:,:,:112].detach(),oldafter=data['features'][:,:,112:368].detach(),commands=data['u'],gain=gain,old_nontarget=(data['oldmean']+gain[None,None,:]*data['x'][:,:,126:168]).detach(),initial_target=torch.tensor(np.stack(initials),dtype=torch.float64),actual_pre_targets=torch.tensor(np.stack(pre,axis=1),dtype=torch.float64),actual_post_targets=torch.tensor(np.stack(post,axis=1),dtype=torch.float64),actual_q=torch.tensor(np.stack(q,axis=1),dtype=torch.float64),target_selector=torch.tensor(binding['target_selector'],dtype=torch.float64),q0_42=torch.tensor(q0,dtype=torch.float64),halfspan42=torch.tensor(half,dtype=torch.float64),bounds=torch.tensor(bounds_common,dtype=torch.float64),max_rate=torch.tensor(rate_common,dtype=torch.float64),target_feedback_source_proofs=proofs)
        data['q0']=data['q0_42'];data['halfspan']=data['halfspan42'];data['bounds42x2']=data['bounds'];data['max_rate42']=data['max_rate']
        for i,raw in enumerate(data['raw']):
            raw.update(actual_q=data['actual_q'][:,i],initial_target=data['initial_target'][i],actual_pre_targets=data['actual_pre_targets'][:,i],actual_post_targets=data['actual_post_targets'][:,i],target_selector=float(data['target_selector'][i]),normalizers=dict(q0=data['q0'],q0_42=data['q0_42'],halfspan=data['halfspan'],halfspan42=data['halfspan42'],bounds=data['bounds'],max_rate=data['max_rate'],gain=gain))
    if int(data['target_selector'].sum())!=9 or data['actual_q'].shape!=(1920,12,42):raise ValueError('Nineonlyrealizedtarget selectors/full12history')
    sources.verify(binding['source_hashes']);sources.verify(binding['data_hashes']);return data
