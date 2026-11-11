"""Authenticated parent outcomes and actual left teacher sequence; no physics."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from model_io import HERE,ROOT,OLD,BASE,REFERENCE,PARENT,PARENT_SHA,sha256,verify_hashes
EVAL=OLD.parent/'two_only_evaluation'
TEACHER=ROOT.parent/'20261110T191449Z/teacher/left_full'


def parent_evidence():
    hashes={str(PARENT):PARENT_SHA}
    spec=importlib.util.spec_from_file_location('_readout_parent_identity_check',BASE/'identity.py')
    identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
    for condition in ('neutral','left','right'):
        directory=EVAL/('EVAL_'+condition);rp=directory/'RESULT.json';ap=EVAL/('VERIFY_'+condition+'.json');ip=directory/'IDENTITY.json'
        result=json.loads(rp.read_text());audit=json.loads(ap.read_text());manifest=json.loads(ip.read_text())
        if result['checkpoint_sha256']!=PARENT_SHA or Path(result['checkpoint']).resolve()!=PARENT:raise ValueError('parent result checkpoint mismatch')
        if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['checkpoint_sha256']!=PARENT_SHA or Path(audit['run']).resolve()!=directory:raise ValueError('parent independent audit mismatch')
        identity.verify(manifest)
        if result['identity_sha256']!=manifest['identity_sha256']:raise ValueError('parent execution identity mismatch')
        if condition in ('neutral','right'):
            if not result['engineering_pass'] or not audit['engineering_pass'] or not all(result['criteria'].values()):raise ValueError('parent qualified function missing')
        else:
            gates=result['autonomous']['gates']
            if result['engineering_pass'] or gates['slip'] or not all(v for k,v in gates.items() if k!='slip') or not result['criteria']['all_six'] or not result['criteria']['strict_nonfoot_zero'] or not result['criteria']['turn_sign_magnitude']:raise ValueError('registered left-only slip failure changed')
        for filename,key in [('trace_1ms.npz','trace'),('neural_decisions.npz','neural_decisions'),('runtime_geometry.npz','runtime_geometry')]:
            p=directory/filename
            if sha256(p)!=result[key]['sha256']:raise ValueError('parent saved artifact changed')
            hashes[str(p)]=sha256(p)
        for p in (rp,ap,ip):hashes[str(p)]=sha256(p)
        for p,r in manifest['files'].items():hashes[p]=r['sha256']
    return hashes


def prepare(model):
    hashes=parent_evidence();rp=TEACHER/'RESULT.json';sp=TEACHER/'SAMPLES_5MS.npz'
    r=json.loads(rp.read_text())
    if not r['all_gates'] or not r['cycles']['all_legs_completed'] or not r['strict_nonfoot_zero'] or not r['requested_sign_agreement']:raise ValueError('original left teacher not qualified')
    if r['samples']['decisions']!=2880 or sha256(sp)!=r['samples']['sha256']:raise ValueError('teacher sample identity mismatch')
    for p in (rp,sp,REFERENCE):hashes[str(p)]=sha256(p)
    with np.load(sp,allow_pickle=False) as z:
        x=torch.tensor(z['X238'].copy(),dtype=torch.float64)
        u=torch.tensor(z['command2'].copy(),dtype=torch.float64)
        y=torch.tensor(z['Y42'].copy(),dtype=torch.float64)
    if x.shape!=(2880,238) or u.shape!=(2880,2) or y.shape!=(2880,42) or not torch.equal(u[:960],torch.zeros_like(u[:960])) or torch.any(u[:,0]!=0):raise ValueError('actual full left sequence required')
    with torch.no_grad():
        mean,states=model.parent.sequence(x[:,None],u[:,None])
    return dict(x=x,u=u,y=y,mean=mean[:,0],base_hidden=states[0][:,0],residual_hidden=states[1][:,0]),hashes


def cached(model,e):return e['mean']+model.correction(e['residual_hidden'],e['u'])


def invariance(model,conditions=('neutral','right')):
    checks={}
    with torch.no_grad():
        for condition in conditions:
            p=EVAL/('EVAL_'+condition)/'neural_decisions.npz'
            with np.load(p,allow_pickle=False) as z:
                x=torch.tensor(z['X238'].copy(),dtype=torch.float64)
                u=torch.tensor(z['command'].copy(),dtype=torch.float64)
                expected_mean=torch.tensor(z['mean'].copy(),dtype=torch.float64)
                expected_action=torch.tensor(z['policy_action'].copy(),dtype=torch.float64)
                expected_state=torch.tensor(z['hidden_after'].copy(),dtype=torch.float64)
            mean,states=model.sequence(x[:,None],u[:,None])
            actual_state=torch.cat(states,dim=-1)[:,0]
            checks[condition+'_mean']=torch.equal(mean[:,0],expected_mean)
            checks[condition+'_action']=torch.equal(mean[:,0].tanh(),expected_action)
            checks[condition+'_dual_state']=torch.equal(actual_state,expected_state)
    return checks
