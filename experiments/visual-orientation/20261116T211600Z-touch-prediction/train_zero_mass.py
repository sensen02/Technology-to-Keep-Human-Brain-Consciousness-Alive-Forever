"""Repeat the trained forecast with explicit zero activity and class-imbalance handling."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train import HORIZONS, MODES, Predictor, body_targets, metrics, pack, rows, sha, turn

UPDATES = 300


class ZeroMassForecast(Predictor):
    def __init__(self):
        super().__init__()
        self.nonzero = torch.nn.Linear(24, 8)

    def forward(self, x):
        encoded, _ = self.recurrent(x)
        shape = (*encoded.shape[:-1], len(HORIZONS),2)
        q1, q2 = self.band(encoded).sigmoid().reshape(*shape,2).unbind(-1)
        return (self.events(encoded).reshape(shape), self.nonzero(encoded).reshape(shape),
                torch.stack((torch.minimum(q1,q2),torch.maximum(q1,q2)),-1), self.body(encoded))


def balanced_loss(logits, truth, mask, weight):
    base = F.binary_cross_entropy_with_logits(logits,truth,reduction="none")
    place = torch.where(truth>0,weight,1.)*mask
    return (base*place).sum()/place.sum().clamp_min(1.)


def shift_fit(logits, labels, mask):
    target = (labels*mask).sum((0,1))/mask.sum((0,1))
    lo, hi = torch.full_like(target,-15.),torch.full_like(target,15.)
    for _ in range(24):
        mid = (lo+hi)/2
        value = ((logits+mid).sigmoid()*mask).sum((0,1))/mask.sum((0,1))
        low = value < target
        lo, hi = torch.where(low,mid,lo),torch.where(low,hi,mid)
    return (lo+hi)/2


def train_partition(samples, mode, out):
    physical = np.concatenate([body_targets(r["state"]) for r in samples])
    center = physical.mean(0)
    scale = np.maximum(physical.std(0),.015)
    x, event, strength, mask, body, early = pack(samples,mode,center,scale)
    event_mask = mask[:,:,:,None] * (.30+.70*early[:,:,None,:])
    all_valid = mask[:,:,:,None].expand_as(event)
    strength_bit = (strength>0).float()
    total = event_mask.sum((0,1))
    positive = (event*event_mask).sum((0,1))
    weights = torch.sqrt((total-positive)/positive.clamp_min(1.)).clamp(1.,25.)
    torch.manual_seed(20261117)
    model = ZeroMassForecast()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
    training = []
    for k in range(1,UPDATES+1):
        batch = torch.randint(len(x),(8,))
        events, present, band, predicted = model(x[batch])
        btruth = body[batch]
        event_loss = balanced_loss(events,event[batch],event_mask[batch],weights)
        arrival_loss = balanced_loss(present,strength_bit[batch],all_valid[batch],torch.tensor(5.))
        residual = strength[batch].unsqueeze(-1)-band
        pinball = torch.maximum(torch.tensor([.10,.90])*residual,torch.tensor([-.90,-.10])*residual)
        positive_mask = all_valid[batch]*strength_bit[batch]
        band_loss = (pinball*positive_mask.unsqueeze(-1)).sum()/(2*positive_mask.sum().clamp_min(1.))
        motion = (btruth[...,26]*scale[26]+center[26]>.5).float()
        motion_loss = balanced_loss(predicted[...,26],motion,torch.ones_like(motion),torch.tensor((1-center[26])/center[26]))
        body_loss = F.smooth_l1_loss(predicted[...,:26],btruth[...,:26])
        loss = event_loss+.25*arrival_loss+.30*band_loss+.15*body_loss+.14*motion_loss
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if k in (1,100,200,UPDATES):
            training.append(dict(step=k,loss=loss.item(),event=event_loss.item(),
                                 activity=arrival_loss.item(),body=body_loss.item(),motion=motion_loss.item()))
    path = out / f"{mode}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                    center=torch.tensor(center),scale=torch.tensor(scale),steps=UPDATES),temporary)
    temporary.rename(path)
    turn.stand.put(out/f"{mode}_TRAIN.json",
                  dict(mode=mode,curves=training,weights_sha256=sha(path),train_paths=[r["path"] for r in samples]))
    print(mode,training[-1],flush=True)


@torch.inference_mode()
def evaluate(mode, out, calibration, held):
    path = out/f"{mode}.pt"
    checkpoint = torch.load(path,weights_only=True)
    model = ZeroMassForecast().eval()
    model.load_state_dict(checkpoint["model"])
    center, scale = checkpoint["center"].numpy(),checkpoint["scale"].numpy()
    x,y,strength,valid,_,_ = pack(calibration,mode,center,scale)
    event,active,_,_ = model(x)
    event_offset = shift_fit(event,y,valid[:,:,:,None].expand_as(y))
    active_offset = shift_fit(active,(strength>0).float(),valid[:,:,:,None].expand_as(y))
    x,y,strength,valid,state,early=pack(held,mode,center,scale)
    event,active,quantiles,body=model(x)
    p=(event+event_offset).sigmoid().numpy()
    p_active=(active+active_offset).sigmoid().numpy()
    y=y.numpy().astype(bool)
    categories={}
    for label in ("nominal","external"):
        chosen=np.array([(r["mode"]=="external") == (label=="external") for r in held])
        categories[label]={}
        for index,span in enumerate(HORIZONS):
            clean=valid[:,:,index,None].numpy().astype(bool)&early.numpy().astype(bool)&chosen[:,None,None]
            categories[label][str(span)] = metrics(y[:,:,index],p[:,:,index],clean)
    measured=(state.numpy()[...,26]*scale[26]+center[26])>.5
    inferred=body[...,26].numpy()>0.
    s=strength.numpy()
    q=quantiles.numpy()
    active_positive=s>0
    conditional = (s>=q[...,0])&(s<=q[...,1])
    low=np.where(p_active>.90,q[...,0],0.)
    high=np.where(p_active>.10,q[...,1],0.)
    nominal_mask=valid.numpy().astype(bool)[...,None].repeat(2,axis=-1)
    coverage=(s>=low)&(s<=high)
    return dict(mode=mode,weights_sha256=sha(path),event_before_contact=categories,
                balanced_motion_accuracy=float(.5*(inferred[measured].mean()+(~inferred[~measured]).mean())),
                positive_activity_conditional_coverage=float(conditional[active_positive].mean()),
                zero_aware_band_coverage=float(coverage[nominal_mask].mean()),
                activity_positive_ticks=int(active_positive.sum()),event_offset=event_offset.tolist(),
                activity_offset=active_offset.tolist())


def main(data,out,mode):
    torch.set_num_threads(1)
    if mode=="audit":
        samples=rows(data)
        cal=[r for r in samples if r["case"]["split"]=="calibration"]
        held=[r for r in samples if r["case"]["split"]=="held"]
        result={key:evaluate(key,out,cal,held) for key in MODES}
        compared={key:row["event_before_contact"]["nominal"]["100"] for key,row in result.items()}
        candidate=compared["ordered"]
        result["registered_gate_ap_gain"] = all(candidate["average_precision"]>=compared[m]["average_precision"]+.02
                                                 for m in ("sense","shuffled"))
        result["registered_gate_brier"] = all(candidate["brier"]<=compared[m]["brier"]+.005
                                               for m in ("sense","shuffled"))
        result["held_100ms"] = compared
        result["same_held_groups_as_v1"] = True
        turn.stand.put(out/"EVALUATION.json",result)
        print(compared,flush=True)
    else:
        train_partition([r for r in rows(data,"train") if r["mode"]!="external"],mode,out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("mode",choices=(*MODES,"audit"))
    args=parser.parse_args()
    main(args.data,args.out,args.mode)
