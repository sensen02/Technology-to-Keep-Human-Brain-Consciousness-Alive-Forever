"""Initialize the SAC readout on archived physical visual demonstrations."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from course import HERE, UNIFIED, new_brain, representation, command_code, sha
from model import SACBrain

TEACHERS=("straight","left","right","pace")
UPDATES=620


def example(name,result):
    source=UNIFIED/"RUNS"/name/"TRACE.npz"
    assert sha(source)==result["hashes"]["TRACE.npz"] and result["passed"]
    with np.load(source,allow_pickle=False) as z:
        eyes,stamps,dec=(z[k] for k in ("eyes","eye_stamps","decisions"))
    model=new_brain()
    h=torch.zeros((1,1,32))
    features=[]
    labels=[]
    paces=[]
    last_command=(0.,0)
    for image,captured in zip(eyes,stamps,strict=True):
        t=int(captured+20)
        if t//5>=len(dec):
            break
        model.new_frame(image)
        i=t//5
        previous=dec[max(i-4,0):i]
        chosen=np.roll(dec[:,1:3],1,axis=0) if not features else chosen
        prev_command=chosen[max(i-4,0):i]
        gate=np.where((previous[:,15]>.5)&np.any(previous[:,5:7]!=0.,axis=1),4.,-4.)
        x=np.c_[gate,previous[:,7:11],previous[:,12],np.any(prev_command!=0.,axis=1)].astype(np.float32)
        with torch.inference_mode():
            _,h=model.state(torch.from_numpy(x)[None],h)
        visual=representation(model,image,last_command)
        features.append(np.r_[h[0,0].numpy(),visual])
        mode,p=command_code(dec[i,5:7])
        labels.append(mode)
        paces.append(p)
        last_command=tuple(dec[i,5:7])
    return (torch.tensor(np.asarray(features),dtype=torch.float32),torch.tensor(labels),
            torch.tensor(paces,dtype=torch.float32),dict(path=str(source),sha256=sha(source)))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3852)
    archived=json.loads((UNIFIED/"ACCEPTANCE.json").read_text())["cases"]
    examples=[example(case,archived[case]) for case in TEACHERS]
    x,mode,speed=(torch.cat([row[i] for row in examples]) for i in range(3))
    model=SACBrain(initialized=True)
    optimizer=torch.optim.AdamW(model.actor.parameters(),lr=.002,weight_decay=.001)
    weight=(torch.bincount(mode,minlength=4).float().mean()/torch.bincount(mode,minlength=4).float()).sqrt().clamp(.6,2.)
    for update in range(1,UPDATES+1):
        i=torch.randint(len(mode),(160,))
        logs,mu,log_std=model.distribution(x[i])
        moving=mode[i]==1
        # Gaussian variance is left at a calibrated exploration scale.
        degree=((mu.tanh()+1.)*.5)[moving]
        loss=F.nll_loss(logs,mode[i],weight=weight)+F.smooth_l1_loss(degree,speed[i][moving],beta=.1)
        loss+=.025*(log_std+.95).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        logs,mu,_=model.distribution(x)
        per_mode={str(i):float((logs[mode==i].argmax(-1)==i).float().mean()) for i in range(4)}
    out=HERE/"WARM.pt"
    torch.save(dict(model=model.state_dict(),bc_optimizer=optimizer.state_dict(),bc_updates=UPDATES,
                    source=[row[-1] for row in examples],imitation_by_mode=per_mode,frames=len(x)),out.with_suffix(".tmp"))
    out.with_suffix(".tmp").rename(out)
    print(dict(updates=UPDATES,frames=len(x),accuracy_by_mode=per_mode,sha256=sha(out)),flush=True)


if __name__=="__main__":
    main()
