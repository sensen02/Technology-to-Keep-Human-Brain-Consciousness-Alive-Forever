"""Fit a two-head readout from separated contact and load arrivals, never body truth inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bind import BINDING, GRAPH
from check_separation import cases
from replay import WALK_GROUPS
from separation import SCALE
from touch_runtime import sha, turn

HERE = Path(__file__).resolve().parent
HORIZONS = (20,50,100,200)
STRIDE_MS = 5
WINDOW = 136
UPDATES = 420
MODES = ("full","without_load","without_command")


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6,32,batch_first=True)
        self.touch = nn.Linear(32,8)
        self.force_correction = nn.Linear(32,8)
        nn.init.zeros_(self.force_correction.weight)
        nn.init.zeros_(self.force_correction.bias)

    def forward(self, x):
        hidden,_ = self.gru(x)
        shape = (*x.shape[:2],len(HORIZONS),2)
        logits = self.touch(hidden).reshape(shape)
        force = (x[:,:,None,2:4] + self.force_correction(hidden).reshape(shape)).clamp(0.,2.)
        return logits,force


def population(bench, walking):
    for result,path in cases(bench):
        if result["passed"]:
            yield dict(split=result["case"]["split"],path=path,
                       sha256=result["sha256"],kind="factorial",lateral=result["lateral"])
    for family in range(WALK_GROUPS):
        folder = walking / f"group_{family:02d}"
        for mode in ("clear","fixed","external","sham"):
            result = turn.read(folder/f"{mode}.json")
            yield dict(split=result["case"]["split"],path=folder/f"{mode}.npz",
                       sha256=result["replay_sha256"],kind="walking",lateral=mode=="external")


def example(record):
    path=record["path"]
    assert sha(path)==record["sha256"]
    with np.load(path,allow_pickle=False) as z:
        stride=STRIDE_MS if record["kind"]=="factorial" else 1
        received, commands, normal, touch = (z[key] for key in (
            "received","commands","normal_load","truth_tactile"))
        received,commands,normal = received[::stride],commands[::stride],normal[::stride]
        if stride!=1:
            touch=touch.reshape(-1,stride,2).any(1)
    look=max(HORIZONS)//STRIDE_MS
    if len(touch)!=len(received) or len(touch)<WINDOW:
        raise ValueError(f"Unexpected lengths in {path}")
    arms=received[:,:4]
    x=np.c_[arms,received[:,4],commands]
    labels,lengths,forces,availability=[],[],[],[]
    for span_ms in HORIZONS:
        gap=span_ms//STRIDE_MS
        samples=np.lib.stride_tricks.sliding_window_view(touch,gap,axis=0)
        labels.append(samples[1:1+len(x)-look].any(-1))
        forces.append(normal[gap:gap+len(x)-look])
        availability.append(received[gap:gap+len(x)-look,4]>0)
    actual=np.stack(labels,1)
    force=np.stack(forces,1)
    valid=np.stack(availability,1)
    for start in range(0,len(x)-WINDOW+1,WINDOW-look-20):
        stop=start+WINDOW-look
        lengths.append((x[start:stop],actual[start:stop],force[start:stop],
                        valid[start:stop],~touch[start:stop]))
    return lengths


def grouped(records,mode):
    samples=[values for r in records for values in example(r)]
    x,touch,force,valid,early = (torch.as_tensor(np.stack(columns),dtype=torch.float32)
                              for columns in zip(*samples,strict=True))
    if mode=="without_load":
        x[:,:,2:4]=0
    if mode=="without_command":
        x[:,:,5]=0
    return x,touch,force,valid,early


def objective(net, features, touch, force, valid, early, positive_weight):
    logits, normal = net(features)
    valid = valid[:,:,:,None]
    event_loss=nn.functional.binary_cross_entropy_with_logits(logits,touch,reduction="none")
    factors = torch.where(touch>0.,positive_weight,1.)*(.3+.7*early[:,:,None,:])*valid
    event=(event_loss*factors).sum()/factors.sum().clamp_min(1.)
    force_loss=nn.functional.smooth_l1_loss(normal,force/SCALE,reduction="none",beta=.08)
    load=(force_loss*valid).sum()/(valid.sum()*2).clamp_min(1.)
    return event+1.4*load,event,load


def fit(records,mode,folder):
    train=[r for r in records if r["split"]=="train" and not (r["kind"]=="walking" and r["lateral"])]
    features,touch,force,valid,early=grouped(train,mode)
    p=(touch*valid[:,:,:,None]).sum((0,1)).clamp_min(1.)
    weight=torch.sqrt((valid.sum((0,1))[:,None]-p).clamp_min(1.)/p).clamp(1.,25.)
    torch.manual_seed(225901)
    model=Forecast()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    curve=[]
    for step in range(1,UPDATES+1):
        sample=torch.randint(len(features),(12,))
        loss,event,load=objective(model,features[sample],touch[sample],force[sample],valid[sample],early[sample],weight)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step in (1,100,200,300,UPDATES):
            curve.append(dict(update=step,total=loss.item(),touch=event.item(),load=load.item()))
    dest=folder/f"{mode}.pt"
    tmp=dest.with_suffix(".tmp")
    torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                    updates=UPDATES,horizons_ms=HORIZONS),tmp)
    tmp.rename(dest)
    turn.stand.put(folder/f"{mode}_TRAIN.json",dict(curve=curve,weights_sha256=sha(dest),train_windows=len(features),
        train_runs=[dict(path=str(r["path"].resolve()),sha256=r["sha256"]) for r in train],
        separation_acceptance_sha256=sha(HERE/"SEPARATION_ACCEPTANCE.json"),
        source_script_sha256=sha(Path(__file__)),binding_sha256=sha(BINDING),graph_sha256=sha(GRAPH)))
    print(mode,curve[-1],flush=True)


def average_precision(y,p):
    if not y.any():
        return None
    order=np.argsort(-p,kind="stable")
    y,p=y[order],p[order]
    last=np.r_[np.flatnonzero(np.diff(p)!=0),len(y)-1]
    correct=np.cumsum(y)[last]
    return float(np.sum(np.diff(np.r_[0,correct])*correct/(last+1))/correct[-1])


@torch.inference_mode()
def measure(records,filename,mode):
    checkpoint=torch.load(filename,map_location="cpu",weights_only=True)
    model=Forecast().eval()
    model.load_state_dict(checkpoint["weights"])
    x,touch,force,valid,early=grouped(records,mode)
    forecast,prediction=model(x)
    changed=x.clone()
    changed[:,50:]=1-changed[:,50:]
    alternate=model(changed)
    assert torch.equal(forecast[:,:50],alternate[0][:,:50]) and torch.equal(
        prediction[:,:50],alternate[1][:,:50])
    predicted=prediction.numpy()*SCALE
    full,_t,_f,_v,_e=grouped(records,"full")
    persistence=full[:,:,None,2:4].numpy()*SCALE
    results={}
    for col,horizon in enumerate(HORIZONS):
        mask=np.broadcast_to(valid[:,:,col,None].numpy()>0,touch.shape[:2]+(2,)).copy()
        mask[:,:20]=False
        label=touch[:,:,col].numpy()>0
        p=forecast[:,:,col].numpy()
        legal=mask & (early.numpy()>0)
        target=force[:,:,col].numpy()
        error=np.abs(predicted[:,:,col]-target)
        mean=np.abs(persistence[:,:,0]-target)
        loaded=target>=.05
        available=mask & ((target<.02) | loaded)
        class_guess=predicted[:,:,col]>=.035
        balanced=float(.5*(class_guess[available & loaded].mean()+
                         (~class_guess[available & ~loaded]).mean()))
        results[str(horizon)]=dict(contact_ap_before_touch=average_precision(label[legal],p[legal]),
                    contact_rate=float(label[legal].mean()),load_mae_W=float(error[mask].mean()),
                    persistence_mae_W=float(mean[mask].mean()),balanced_support_accuracy=balanced,
                    scored_coverage=int(mask.sum()))
    return dict(model_sha256=sha(filename),causal_prefix_exact=True,episode_count=len(records),horizons=results)


def main(data,walking,folder,variant):
    torch.set_num_threads(1)
    assert turn.read(HERE/"SEPARATION_ACCEPTANCE.json")["passed"], "Physical separation is prerequisite"
    archived=list(population(data,walking))
    if variant!="audit":
        fit(archived,variant,folder)
        return
    report={}
    for domain in ("factorial","walking"):
        held=[r for r in archived if r["split"]=="held" and r["kind"]==domain]
        report[domain]={mode:measure(held,folder/f"{mode}.pt",mode) for mode in MODES}
    turn.stand.put(folder/"FORECAST_EVALUATION.json",report)
    print(json.dumps({domain:{mode:row["horizons"]["50"] for mode,row in reports.items()}
                     for domain,reports in report.items()}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("walking",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("variant",choices=(*MODES,"audit"))
    args=parser.parse_args()
    main(args.data,args.walking,args.output,args.variant)
