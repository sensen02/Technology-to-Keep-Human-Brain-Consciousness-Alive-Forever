import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from paths import BASE,HERE
from model import BodyModel

torch.set_num_threads(2)
DT=.005

def tensor(a):
    return torch.as_tensor(a,dtype=torch.float32)

def load(seed):
    result=[]
    for folder in sorted((HERE/'data').glob(f'{seed}_*')):
        with np.load(folder/'external.npz') as z:
            pos=tensor(z['markers_mm'])
            force=tensor(z['force_uN'])
        with np.load(folder/'nerve.npz') as z:
            dn=tensor(z['dn'].reshape(-1,50,17).sum(1))
            an=tensor(z['an'].reshape(-1,5,10,2).any(2))
        touch=torch.stack((torch.cat((force[:1],force[:-1])),force),dim=1)
        hist=torch.stack([torch.cat((torch.zeros_like(dn[:i]),dn[:-i])) if i else dn for i in (3,2,1,0)],dim=1)
        result.append((folder.name,pos,touch,hist,an))
    return result

def build(episodes):
    return [torch.stack([row[i] for row in episodes]) for i in range(1,5)]

def loss(net,batch,sens,horizon=5,noise=True):
    pos,touch,dn,an=batch
    prev,now=pos[:,0],pos[:,1]
    if noise:
        now=now+torch.randn_like(now)*.0018
    motion,spikes=0.,0.
    for t in range(horizon):
        nxt,logit=net(prev,now,touch[:,t+1],dn[:,t+1])
        motion+=torch.nn.functional.smooth_l1_loss((nxt-pos[:,t+2])/.035,torch.zeros_like(nxt))
        observed=(logit.sigmoid()*sens).clamp(1e-5,1-1e-5)
        spikes+=torch.nn.functional.binary_cross_entropy(observed,an[:,t+2])
        prev,now=now,nxt
    return (motion+spikes*.8)/horizon

def fit():
    torch.manual_seed(9033)
    net=BodyModel()
    sens=tensor(json.loads((HERE/'calibration.json').read_text())['sensitivity'])
    # Model selection by full seed, followed by a fixed-length refit with both development seeds.
    train=load(10901); val=load(11311)
    horizon=5
    def stage(episodes,epochs,validation):
        nonlocal net
        torch.manual_seed(9033)
        net=BodyModel()
        opt=torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.0005)
        z=build(episodes)
        history=[]
        best=(float('inf'),0)
        if validation:
            v=build(val)
            starts=torch.arange(4,392,6)
            vb=[a[:,starts[:,None]+torch.arange(horizon+2)].flatten(0,1) for a in v]
        for epoch in range(epochs):
            total=0.
            for _ in range(12):
                i=torch.randint(len(episodes),(80,))
                start=torch.randint(4,400-horizon-2,(80,))
                ticks=start[:,None]+torch.arange(horizon+2)
                batch=[a[i[:,None],ticks] for a in z]
                l=loss(net,batch,sens)
                opt.zero_grad();l.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
                opt.step()
                total+=l.item()
            history.append({'epoch':epoch+1,'loss':total/12})
            if validation:
                with torch.no_grad():vscore=loss(net,vb,sens,noise=False).item()
                history[-1]['validation']=vscore
                if vscore<best[0]:best=(vscore,epoch+1)
        return best[1],history
    n,first=stage(train,45,True)
    _,second=stage(train+val,n,False)
    torch.save(net.state_dict(),HERE/'weights.pt')
    (HERE/'training.json').write_text(json.dumps(dict(selected_epochs=n,development=first,refit=second),indent=2))
    print('selected',n,flush=True)

if __name__=='__main__':
    fit()
