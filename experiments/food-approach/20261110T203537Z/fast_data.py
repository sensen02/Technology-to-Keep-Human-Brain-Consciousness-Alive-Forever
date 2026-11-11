"""Authenticated actual teacher features. Cached base state is strictly causal."""
import json
from pathlib import Path
import numpy as np
import torch
from fast_motor import HERE,PREVIOUS,BASE,CHECKPOINT,sha256,verify_hashes

REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'


def prepare(model):
    datahashes={str(CHECKPOINT):sha256(CHECKPOINT),str(BASE/'models.py'):sha256(BASE/'models.py'),str(REFERENCE):sha256(REFERENCE)}
    # Completed independent moderate artifact manifest binds actual samples and replay.
    manifest_path=PREVIOUS/'deceleration_moderate/ARTIFACT_SHA256.json'
    manifest=json.loads(manifest_path.read_text())
    if not all(manifest['functional_passes'].values()) or manifest['independent_checks']!=63:raise ValueError('moderate functional evidence not complete')
    verify_hashes({p:r['sha256'] for p,r in manifest['files'].items()})
    datahashes[str(manifest_path)]=sha256(manifest_path)
    entries=[('left_full',PREVIOUS/'teacher/left_full'),('right_full',PREVIOUS/'teacher/right_full'),
             ('straight_085',PREVIOUS/'deceleration_moderate/straight_085'),('straight_090',PREVIOUS/'deceleration_moderate/straight_090')]
    episodes=[]
    for name,directory in entries:
        rp=directory/'RESULT.json';sp=directory/'SAMPLES_5MS.npz'
        r=json.loads(rp.read_text())
        if not r['all_gates'] or r['nonfoot_force_max_native']!=0 or not r['cycles']['all_legs_completed']:raise ValueError('teacher admission failed '+name)
        if not np.isfinite([r['slip_ratio_max'],r['com_height_min'],r['forward_signed_native_per_s']]).all():raise ValueError('invalid teacher measurements')
        if sha256(sp)!=r['samples']['sha256']:raise ValueError('teacher sample hash mismatch')
        if name.endswith('full') and not r['requested_sign_agreement']:raise ValueError('wrong teacher turn direction')
        for p in (rp,sp):datahashes[str(p)]=sha256(p)
        with np.load(sp,allow_pickle=False) as z:
            x=torch.tensor(z['X238'].copy(),dtype=torch.float64)
            y=torch.tensor(z['Y42'].copy(),dtype=torch.float64)
            u=torch.tensor(z['command2'].copy(),dtype=torch.float64)
        if x.shape!=(2880,238) or y.shape!=(2880,42) or u.shape!=(2880,2) or not torch.equal(u[:960],torch.zeros_like(u[:960])):raise ValueError('full actual teacher sequence required')
        state=model.initial_state(1);means=[];hidden=[]
        with torch.no_grad():
            for row in x:
                mean,state=model.base.step(row[None],state);means.append(mean[0]);hidden.append(state[0])
            mean=torch.stack(means);h=torch.stack(hidden);features,gate=model.features(x,h,u)
        episodes.append(dict(name=name,x=x,y=y,u=u,mean=mean,base_hidden=h,features=features,gate=gate))
    return episodes,datahashes


def cached(model,episode):return episode['mean']+model.correction(episode['features'],episode['gate'])


def neutral(model):
    with np.load(REFERENCE,allow_pickle=False) as z,torch.no_grad():
        x=torch.tensor(z['X238'].copy(),dtype=torch.float64)[:,None]
        mean,h=model.sequence(x,x.new_zeros((2880,1,2)))
        result=dict(mean=torch.equal(mean[:,0],torch.tensor(z['mean'])),action=torch.equal(mean[:,0].tanh(),torch.tensor(z['policy_action'])),hidden=torch.equal(h[:,0],torch.tensor(z['hidden_after'])))
    if not all(result.values()):raise ValueError('neutral invariance failed')
    return result


def metrics(model,episodes):
    with torch.no_grad():return {e['name']:float((cached(model,e)[960:].tanh()-e['y'][960:]).square().mean()) for e in episodes}
