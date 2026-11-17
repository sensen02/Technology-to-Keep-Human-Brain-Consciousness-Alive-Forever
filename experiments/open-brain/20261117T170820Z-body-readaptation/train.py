"""Optimize the actual body sensory network and 42-joint phase readout."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from context import HERE,NECK,OLD_WEIGHT,phase,first,hash_file
from tasks import TRAIN,DEVELOP,CORRECT


class AdaptPolicy(phase.Policy):
    def __init__(self):
        super().__init__()
        self.fourier=nn.Parameter(self.fourier.clone())
        self.register_buffer('harmonics',torch.arange(1,33,dtype=torch.float64),persistent=False)

    def forward(self,local,state):
        angles=state[:,:6].double().unsqueeze(-1)*self.harmonics
        design=torch.cat((torch.ones_like(angles[:,:,:1]),torch.cos(angles),torch.sin(angles)),dim=-1)
        values=torch.einsum('nle,lej->nlj',design,self.fourier)
        gait=(values*state[:,6:].double().unsqueeze(-1)).flatten(1)[:,self.order]
        return gait.to(local.dtype)+self.residual(local,state)


def read_policy(checkpoint):
    model=AdaptPolicy()
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['state'])
    return model


def dataset(directory,tasks,qualified):
    x,states,y,weight,identities=[],[],[],[],[]
    for task in tasks:
        result_path=directory/task.name/'RESULT.json'
        result=json.loads(result_path.read_text())
        if qualified and not result['pass_']:
            print('Excluded teacher',task.name,result['gates'],flush=True)
            continue
        trace=directory/task.name/'TRACE.npz'
        assert hash_file(trace)==result['trace_sha256']
        assert result['parameter_sha256']==hash_file(NECK/'PARAMETERS.json')
        assert result['readout_sha256']==hash_file(NECK/'READOUT.json')
        with np.load(trace,allow_pickle=False) as data:
            local,latent,label=data['x'],data['phase'],data['y']
        assert len(local)==len(latent)==len(label) and local.shape[1]==161 and label.shape[1]==42
        assert np.isfinite(local).all() and np.isfinite(latent).all() and np.isfinite(label).all()
        x.append(local)
        states.append(latent)
        y.append(label)
        quiet=np.max(np.abs(local[:,:2]),axis=1)<.04
        weight.append(np.where(quiet,2.,1.).astype(np.float32))
        identities.append(dict(path=str(trace.resolve()),sha256=result['trace_sha256'],task=task.name,
                               source=result['mode']))
    packed=[torch.from_numpy(np.concatenate(rows)) for rows in (x,states,y,weight)]
    return (*packed,identities)


def save(path,model,optimizer,rng,epoch,inputs):
    temp=path.with_suffix('.partial')
    state=dict(state=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),epoch=epoch,inputs=inputs)
    torch.save(state,temp)
    os.replace(temp,path)
    return hash_file(path)


def assess(model,inputs):
    x,state,y,w=inputs[:4]
    loss_sum=0.
    max_error=0.
    with torch.no_grad():
        for ids in torch.arange(len(x)).split(1024):
            error=model(x[ids],state[ids])-y[ids]
            loss_sum+=float(error.square().sum())
            max_error=max(max_error,float(error.abs().max()))
    return dict(rmse_rad=(loss_sum/y.numel())**.5,max_abs_rad=max_error)


def gradient_gate(model,inputs,out):
    # Audit the exact full action objective at actual recorded physical states.
    audit=AdaptPolicy().double()
    audit.load_state_dict(model.state_dict())
    x,s,y,w=inputs[:4]
    ids=torch.arange(300,308)
    local,state,target,weights=(q[ids].double() for q in (x,s,y,w))
    def objective():
        return (((audit(local,state)-target)/.045).square().mean(1)*weights).mean()
    audit.zero_grad(set_to_none=True)
    loss=objective()
    loss.backward()
    checks={}
    for name,param in (('fourier',audit.fourier),('reflex.4.weight',audit.reflex[4].weight)):
        gradient=param.grad.detach().flatten()
        flat=int(gradient.abs().argmax())
        actual=float(gradient[flat])
        assert abs(actual)>1e-10,(name,actual)
        original=float(param.detach().flatten()[flat])
        delta=1e-6
        with torch.no_grad():
            param.flatten()[flat]=original+delta
        above=float(objective().detach())
        with torch.no_grad():
            param.flatten()[flat]=original-delta
        below=float(objective().detach())
        with torch.no_grad():
            param.flatten()[flat]=original
        numeric=(above-below)/(2*delta)
        err=abs(actual-numeric)
        assert err<1e-6+abs(actual)*.001,(name,actual,numeric)
        checks[name]=dict(analytic=actual,numeric=numeric,error=err)
    first.write(out/'GRADIENT.json',checks)
    return checks


def train(out,start,correction):
    torch.set_num_threads(1)
    torch.manual_seed(27183)
    rng=torch.Generator().manual_seed(9529)
    training=dataset(HERE/'TEACHER',TRAIN,True)
    validation=dataset(HERE/'TEACHER_DEV',DEVELOP,True)
    if correction:
        extra=dataset(HERE/'CORRECT',CORRECT,False)
        training=(*[torch.cat((a,b)) for a,b in zip(training[:4],extra[:4],strict=True)],training[4]+extra[4])
    x,state,target,w=training[:4]
    model=read_policy(start)
    reference=phase.read_policy(OLD_WEIGHT).fourier.detach().clone()
    optimizer=torch.optim.AdamW([dict(params=list(model.reflex.parameters()),lr=1e-4),
                                dict(params=[model.fourier],lr=2e-7)],weight_decay=1e-5)
    epochs=20 if correction else 30
    out.mkdir()
    inputs=dict(training=training[4],development=validation[4],train_frames=len(x),dev_frames=len(validation[0]),
           starting_weight=str(start.resolve()),starting_sha256=hash_file(start),
           parameters_sha256=hash_file(NECK/'PARAMETERS.json'),readout_sha256=hash_file(NECK/'READOUT.json'),
           source_sha256=hash_file(HERE/'train.py'),epochs=epochs,optimizer_restored=False,
           trained=['reflex','fourier'],fixed=['neck','readout_calibration','intrinsic_CPG_dynamics'])
    first.write(out/'INPUTS.json',inputs)
    gradient_gate(model,training,out)
    save(out/'epoch000.pt',model,optimizer,rng,0,inputs)
    initial={name:value.detach().clone() for name,value in model.state_dict().items()}
    first.write(out/'INITIAL_FIT.json',assess(model,validation))
    with (out/'JOURNAL.jsonl').open('x',buffering=1) as log:
        for epoch in range(1,epochs+1):
            total=0.
            for ids in torch.randperm(len(x),generator=rng).split(768):
                prediction=model(x[ids],state[ids])
                error=(prediction-target[ids])/.045
                body_loss=(error.square().mean(1)*w[ids]).mean()
                anchor=.002*((model.fourier-reference)/.002).square().mean()
                loss=body_loss+anchor
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2.,error_if_nonfinite=True)
                optimizer.step()
                total+=float(loss.detach())*len(ids)
            metrics=assess(model,validation)
            digest=save(out/f'epoch{epoch:03d}.pt',model,optimizer,rng,epoch,inputs)
            row=dict(epoch=epoch,mean_loss=total/len(x),development=metrics,sha256=digest,
                     last_gradient_norm=float(norm),optimizer_steps=max(int(v['step']) for v in optimizer.state.values()))
            log.write(json.dumps(row)+'\n')
            if epoch%5==0:print(row,flush=True)
    differences={}
    for name,value in model.state_dict().items():
        delta=(value.detach()-initial[name]).double()
        differences[name]=dict(l2=float(delta.norm()),max_abs=float(delta.abs().max()))
    assert differences['fourier']['l2']>0 and sum(v['l2'] for k,v in differences.items() if k.startswith('reflex.'))>0
    first.write(out/'ACTUAL_PARAMETER_UPDATES.json',differences)
    # Deployment uses the standard historical loader, whose phase coefficients
    # are buffers. Verify that the saved trainable readout reproduces that API.
    deployed=phase.read_policy(out/f'epoch{epochs:03d}.pt')
    with torch.no_grad():
        offset=model(x[:10],state[:10]).numpy()
        converted=np.array([deployed.act(row,latent,True) for row,latent in zip(x[:10].numpy(),state[:10].numpy(),strict=True)])
    gap=float(np.max(np.abs(offset-converted)))
    assert gap<1e-6,gap
    first.write(out/'SAVE_RELOAD.json',dict(max_error_rad=gap,standard_loader=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--start',type=Path,default=OLD_WEIGHT)
    parser.add_argument('--correction',action='store_true')
    args=parser.parse_args()
    train(args.out,args.start,args.correction)
