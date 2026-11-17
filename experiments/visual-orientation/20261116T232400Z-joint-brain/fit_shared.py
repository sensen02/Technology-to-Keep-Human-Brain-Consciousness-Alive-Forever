"""Jointly supervise one recurrent brain, with a matched action-only ablation."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bind import BINDING, GRAPH
from run_brain import TEACH_CASES, vision_init
from shared_brain import HORIZONS, STATE_DIM, SharedBrain
from touch_runtime import sha, turn

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT.parent/"20261116T211600Z-touch-prediction/DATA"
REPLAY=ROOT.parent/"20261116T225900Z-separated-afferents/WALK_REPLAY"
STEPS=440
STRIDE=5
FUTURE=max(HORIZONS)//STRIDE
BODY_OFFSET=10


@dataclass
class Trial:
    reference: dict
    features: np.ndarray
    intended: np.ndarray
    physical: np.ndarray
    contact: np.ndarray


def demonstrations(path):
    for group in range(len(TEACH_CASES)):
        folder=path/f"g{group}"
        result=turn.read(folder/"RESULT.json")
        assert result["passed"], f"Teacher {group} failed physical test"
        source=folder/"RUN.npz"
        assert sha(source)==result["hashes"]["RUN.npz"]
        with np.load(source,allow_pickle=False) as z:
            arrays={key:z[key] for key in ("features","intended","physical","contact")}
        yield Trial(dict(path=str(source.resolve()),sha256=sha(source),source="visual_physical"),**arrays)


def archived(split):
    for family in range(20):
        directory=REPLAY/f"group_{family:02d}"
        for mode in ("clear","fixed","external","sham"):
            meta=turn.read(directory/f"{mode}.json")
            if meta["case"]["split"]!=split or (split=="train" and mode=="external"):
                continue
            source=directory/f"{mode}.npz"
            physical_path=ARCHIVE/directory.name/mode/"SAMPLES.npz"
            assert sha(source)==meta["replay_sha256"] and sha(physical_path)==meta["native_archive_sha256"]
            with np.load(source,allow_pickle=False) as z:
                arrived,chosen,touch=(z[key] for key in ("received","commands","truth_tactile"))
            with np.load(physical_path,allow_pickle=False) as z:
                raw=z["state"]
            physical=raw[:,[*range(19),25]]
            features=np.c_[np.zeros(len(chosen)),arrived,np.r_[0,chosen[:-1]]]
            yield Trial(dict(path=str(source),sha256=meta["replay_sha256"],physical_path=str(physical_path),
                             physical_sha256=meta["native_archive_sha256"],source="archived_no_visual"),
                        features,chosen,physical,touch)


def aux_targets(trial, center, scale):
    length=len(trial.features)-FUTURE
    beginning=np.arange(length)
    events,loads,valid=[],[],[]
    for horizon in HORIZONS:
        offset=horizon//STRIDE
        contacts=np.lib.stride_tricks.sliding_window_view(trial.contact,offset,axis=0)
        events.append(contacts[1:length+1].any(-1))
        loads.append(trial.physical[beginning+offset][:,[12,15]])
        valid.append(trial.features[beginning+offset,5]>0.)
    later=trial.physical[beginning+BODY_OFFSET]
    velocity=trial.physical[:,18]
    foot_support=later[:,12:18]>=.02
    speed_median=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(velocity,(9,0),mode="edge"),10),axis=1)
    moving=speed_median[beginning+BODY_OFFSET]>=.025
    return tuple(torch.as_tensor(np.array(value),dtype=torch.float32) for value in (
        np.stack(events,1),np.stack(loads,1),np.stack(valid,1),
        (later-center)/scale,np.c_[foot_support,moving]))


def action_loss(logit,target,x):
    paused_with_target=((x[...,0]>0)&(target==0)).float()
    touch_while_visible=((x[...,1:3].amax(-1)>.008)&(x[...,0]>0)).float()
    changed=F.pad((target[:,1:]!=target[:,:-1]).float(),(1,0))
    events=F.max_pool1d(changed[:,None],13,1,6)[:,0]
    weights=1.+3.*paused_with_target+45.*touch_while_visible+10.*events
    return (F.binary_cross_entropy_with_logits(logit,target,reduction="none")*weights).sum()/weights.sum()


def loss_aux(outputs,labels):
    event,load,body,motion=outputs
    cue,normal,valid,state,bits=labels
    weights=torch.where(cue>0,6.,1.)*valid[:,:,:,None]
    evt=(F.binary_cross_entropy_with_logits(event,cue,reduction="none")*weights).sum()/weights.sum().clamp_min(1.)
    f=(F.smooth_l1_loss(load,normal/.45,reduction="none",beta=.08)*valid[:,:,:,None]).sum()/(valid.sum()*2).clamp_min(1)
    q=F.smooth_l1_loss(body,state)
    v=F.binary_cross_entropy_with_logits(motion,bits)
    return .22*evt+.25*f+.17*q+.12*v, dict(contact=evt.item(),load=f.item(),kinematics=q.item(),motion=v.item())


def prepare(trials,center,scale):
    return [(torch.tensor(row.features,dtype=torch.float32),torch.tensor(row.intended,dtype=torch.float32),
             aux_targets(row,center,scale)) for row in trials]


def sequence_batch(items,indices,rng,behavior=False):
    selected=[items[i] for i in indices]
    limit=min(len(row[0]) for row in selected)
    starts=[0 if behavior else int(rng.integers(0,len(row[0])-limit+1)) for row in selected]
    x,y=[torch.stack([row[i][begin:begin+limit] for row,begin in zip(selected,starts,strict=True)]) for i in (0,1)]
    label_steps=limit-FUTURE
    labels=[torch.stack([row[2][tag][begin:begin+label_steps]
                        for row,begin in zip(selected,starts,strict=True)]) for tag in range(5)]
    return x,y,labels


def train(path,out,mode):
    torch.set_num_threads(1)
    teacher=list(demonstrations(path))
    supports=list(archived("train"))
    distribution=np.concatenate([t.physical for t in teacher+supports])
    mean,scale=distribution.mean(0),np.maximum(distribution.std(0),.02)
    assert len(scale)==STATE_DIM
    examples,side_examples=prepare(teacher,mean,scale),prepare(supports,mean,scale)
    torch.manual_seed(232401)
    rng=np.random.default_rng(232401)
    brain=SharedBrain()
    vision_init(brain)
    opt=torch.optim.AdamW((p for p in brain.parameters() if p.requires_grad),lr=.003)
    curves=[]
    for step in range(1,STEPS+1):
        source=rng.choice(len(examples),size=4)
        extra=rng.choice(len(side_examples),size=6)
        x,y,labels=sequence_batch(examples,source,rng,behavior=True)
        xa,ya,la=sequence_batch(side_examples,extra,rng)
        # Action history remains informative, while training also sees mistakes
        # or missing copies that can occur when the policy is run closed loop.
        keep=torch.rand(x.shape[:2])>.24
        x[...,6]=x[...,6]*keep
        logits,h,_=brain.advance(x)
        behavior=action_loss(logits,y,x)
        if mode=="joint":
            _,ha,_=brain.advance(xa)
            main,details=loss_aux(brain.predict(h[:,:-FUTURE],y[:,:-FUTURE]),labels)
            replay,_=loss_aux(brain.predict(ha[:,:-FUTURE],ya[:,:-FUTURE]),la)
            total=behavior+main+.65*replay
        else:
            total=behavior
            details={}
        assert torch.isfinite(total)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(),1.5,error_if_nonfinite=True)
        opt.step()
        if step in (1,100,200,300,STEPS):
            curves.append(dict(step=step,action=behavior.item(),total=total.item(),**details))
    output=out/f"{mode}.pt"
    tmp=output.with_suffix(".tmp")
    torch.save(dict(brain=brain.state_dict(),optimizer=opt.state_dict(),mean=torch.tensor(mean),
                    scale=torch.tensor(scale),updates=STEPS,tensor_rng=torch.get_rng_state(),
                    sampler_rng=json.dumps(rng.bit_generator.state),auxiliary_enabled=mode=="joint"),tmp)
    tmp.rename(output)
    turn.stand.put(out/f"{mode}_TRAIN.json",dict(mode=mode,curves=curves,checkpoint_sha256=sha(output),
                   script_sha256=sha(Path(__file__)),core_sha256=sha(ROOT/"shared_brain.py"),
                   binding=sha(BINDING),graph=sha(GRAPH),trajectories=[t.reference for t in teacher+supports]))
    print(curves[-1],flush=True)


def average_precision(y,p):
    if not y.any():
        return None
    ordered=np.argsort(-p,kind="stable")
    y,p=y[ordered],p[ordered]
    boundaries=np.r_[np.flatnonzero(np.diff(p)!=0),len(y)-1]
    count=np.cumsum(y)[boundaries]
    return float(np.sum(np.diff(np.r_[0,count])*count/(boundaries+1))/count[-1])


def audit(model):
    torch.set_num_threads(1)
    brain=SharedBrain().eval()
    state=torch.load(model,weights_only=True,map_location="cpu")
    brain.load_state_dict(state["brain"])
    held=list(archived("held"))
    all_predictions, all_labels=[],[]
    for row in held:
        x=torch.tensor(row.features,dtype=torch.float32)[None]
        y=torch.tensor(row.intended,dtype=torch.float32)[None]
        cue,load,receive,body,motion=aux_targets(row,state["mean"].numpy(),state["scale"].numpy())
        with torch.inference_mode():
            logits,h,_=brain.advance(x)
            predictions=brain.predict(h[:,:-FUTURE],y[:,:-FUTURE])
            alternate=x.clone()
            alternate[:,100:]=1.-alternate[:,100:]
            new_logits,new_h,_=brain.advance(alternate)
            assert torch.equal(logits[:,:100],new_logits[:,:100]) and torch.equal(h[:,:100],new_h[:,:100])
        all_predictions.append([p[0] for p in predictions])
        all_labels.append((cue,load,receive,body,motion,torch.tensor(row.contact[:-FUTURE])))
    cue,load,valid,body,motion,current=[torch.cat(values).numpy() for values in zip(*all_labels,strict=True)]
    events,loads,physical,move=[torch.cat(values).numpy() for values in zip(*all_predictions,strict=True)]
    errors=np.abs(physical-body)*state["scale"].numpy()
    baseline=np.abs(body)*state["scale"].numpy()
    moving=motion[:,-1]>0
    move_hat=move[:,-1]>0
    positive=valid[:,2,None].astype(bool)&~current
    result=dict(causal_prefix_exact=True,rounds=len(held),touch_100ms_AP=average_precision(
           cue[:,2][positive].astype(bool),events[:,2][positive]),
           load_100ms_MAE_W=float(np.abs(loads[:,2]*.45-load[:,2]).mean()),
           body_50ms_mean_absolute=dict(tibia_rad=float(errors[:,:6].mean()),
                 angular_velocity=float(errors[:,6:12].mean()),forward=float(errors[:,18].mean())),
           body_50ms_training_mean_baseline=dict(tibia_rad=float(baseline[:,:6].mean()),
                 angular_velocity=float(baseline[:,6:12].mean()),forward=float(baseline[:,18].mean())),
           moving_50ms_balanced_accuracy=float(.5*(move_hat[moving].mean()+(~move_hat[~moving]).mean())),
           source_sha256=sha(model))
    if state["auxiliary_enabled"]:
        sample=prepare(held[:1],state["mean"].numpy(),state["scale"].numpy())
        x,y,l=sequence_batch(sample,[0],np.random.default_rng(0))
        decision,h,_=brain.advance(x)
        sensory,details=loss_aux(brain.predict(h[:,:-FUTURE],y[:,:-FUTURE]),l)
        params=tuple(brain.state.parameters())
        aux_gradient=torch.autograd.grad(sensory,params,retain_graph=True)
        act_gradient=torch.autograd.grad(action_loss(decision,y,x),params)
        result["gradient_to_same_gru"]=dict(action=float(sum(t.square().sum() for t in act_gradient).sqrt()),
                   auxiliary=float(sum(t.square().sum() for t in aux_gradient).sqrt()))
        assert min(result["gradient_to_same_gru"].values())>0
    turn.stand.put(model.with_name(f"{model.stem}_OFFLINE.json"),result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("mode",choices=("joint","behavior","audit"))
    a=p.parse_args()
    if a.mode=="audit":
        audit(a.out)
    else:
        train(a.teacher,a.out,a.mode)
