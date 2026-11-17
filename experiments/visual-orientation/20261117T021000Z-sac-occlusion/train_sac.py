"""Hybrid SAC with exact mode expectation and conditional continuous pace."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from course import HERE, GAMMA, sha
from model import SACBrain, CriticPair
from sac_replay import Replay, BURN


def accumulate(optimizer,loss,params):
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


class SAC:
    def __init__(self,initial,seed,anchor_weight):
        torch.manual_seed(seed)
        self.rng=np.random.default_rng(seed)
        data=torch.load(initial,map_location="cpu",weights_only=True)
        self.model=SACBrain()
        self.model.load_state_dict(data["model"])
        self.q=CriticPair()
        self.target=deepcopy(self.q).requires_grad_(False)
        self.actor_params=list(self.model.actor.parameters())
        self.shared_params=list(self.model.brain.state.parameters())
        self.aux_params=list(self.model.intent.parameters())
        for head in (self.model.brain.auxiliary,self.model.brain.future_contacts,self.model.brain.future_load,
                     self.model.brain.future_body,self.model.brain.future_motion):
            self.aux_params.extend(head.parameters())
        self.policy_params=self.actor_params+self.shared_params+self.aux_params
        self.policy_opt=torch.optim.Adam([
            dict(params=self.actor_params,lr=.00010),
            dict(params=self.shared_params,lr=.000025),
            dict(params=self.aux_params,lr=.00010)])
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=.00055)
        self.alpha_log=torch.tensor([np.log(.025),np.log(.006)],dtype=torch.float32,requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.alpha_log],lr=.0008)
        self.index=0
        self.anchor_weight=float(anchor_weight)
        if "critic" in data:
            self.q.load_state_dict(data["critic"])
            self.target.load_state_dict(data["target"])
            self.policy_opt.load_state_dict(data["policy_optimizer"])
            self.q_opt.load_state_dict(data["critic_optimizer"])
            self.alpha_log.data.copy_(data["log_alpha"])
            self.alpha_opt.load_state_dict(data["alpha_optimizer"])
            self.index=data["sac_updates"]
            torch.set_rng_state(data["rng_torch"])
            self.rng.bit_generator.state=json.loads(data["rng_numpy"])

    def target_value(self,state,alpha):
        logs,mu,std=self.model.distribution(state)
        pace,density=self.model.pace_sample(mu,std)
        q1,q2=self.target.all(state,pace)
        probs=logs.exp()
        return (probs*(torch.minimum(q1,q2)-alpha[0]*logs)).sum(-1)-probs[...,1]*alpha[1]*density

    def auxiliary_loss(self,state,mode,pace,b):
        tactile,load,body,motion=self.model.predict(state,mode,pace)
        valid=b["aux_valid"].unsqueeze(-1)
        weights=valid*torch.where(b["aux_touch"]>0.,4.,1.)
        cue=F.binary_cross_entropy_with_logits(tactile,b["aux_touch"],reduction="none")
        tactile_loss=(cue*weights).sum()/weights.sum().clamp_min(1.)
        z=F.smooth_l1_loss(load,b["aux_load"],reduction="none",beta=.1)
        load_loss=(z*valid).sum()/(2*valid.sum()).clamp_min(1.)
        mask=b["aux_valid"][...,1].unsqueeze(-1)
        values=(b["aux_body"]-self.model.body_mean)/self.model.body_scale
        body_error=F.smooth_l1_loss(body,values,reduction="none")
        body_loss=(body_error*mask).sum()/(20*mask.sum()).clamp_min(1.)
        bits=F.binary_cross_entropy_with_logits(motion,b["aux_motion"],reduction="none")
        motion_loss=(bits*mask).sum()/(7*mask.sum()).clamp_min(1.)
        return .35*tactile_loss+.4*load_loss+.65*body_loss+.2*motion_loss

    def step(self,b):
        with torch.no_grad():
            _,memory=self.model.sequence(b["x"][:,:BURN],b["vision"][:,:BURN])
        features,_=self.model.sequence(b["x"][:,BURN:],b["vision"][:,BURN:],memory)
        state=features[:,:-1]
        alpha=self.alpha_log.detach().exp()
        mode,pace=b["mode"],b["pace"].float()
        assert not (b["terminal"] & b["truncated"]).any()
        with torch.no_grad():
            next_value=self.target_value(features[:,1:],alpha)
            target=b["reward"].float()+GAMMA*(~b["terminal"])*next_value
            assert torch.equal(target[b["terminal"]],b["reward"].float()[b["terminal"]])
        q1,q2=self.q.chosen(state.detach(),mode,pace)
        q_loss=F.smooth_l1_loss(q1,target)+F.smooth_l1_loss(q2,target)
        q_norm=accumulate(self.q_opt,q_loss,list(self.q.parameters()))

        self.q.requires_grad_(False)
        logs,mu,std=self.model.distribution(state)
        speed,density=self.model.pace_sample(mu,std)
        a,both=self.q.all(state.detach(),speed)
        p=logs.exp()
        actor=(p*(alpha[0]*logs-torch.minimum(a,both))).sum(-1).mean()
        actor+=(p[...,1]*alpha[1]*density).mean()
        aux=self.auxiliary_loss(state,mode,pace,b)
        weight=b["demo"]
        nll=-logs.gather(-1,mode.unsqueeze(-1)).squeeze(-1)
        demo_loss=(weight*nll).sum()/weight.sum().clamp_min(1.)
        loss=actor+.075*aux+self.anchor_weight*demo_loss
        norm=accumulate(self.policy_opt,loss,self.policy_params)
        self.q.requires_grad_(True)

        entropy=-(p*logs).sum(-1).detach().mean()
        speed_entropy=-density.detach().mean()
        entropy_loss=(self.alpha_log.exp()*torch.stack((entropy-.60,speed_entropy+.70))).sum()
        accumulate(self.alpha_opt,entropy_loss,[self.alpha_log])
        with torch.no_grad():
            self.alpha_log.clamp_(-9.,-.30)
            for t,current in zip(self.target.parameters(),self.q.parameters(),strict=True):
                t.lerp_(current,.01)
        self.index+=1
        aux_base=float(aux.detach())
        return dict(update=self.index,critic=float(q_loss.detach()),actor=float(actor.detach()),aux=aux_base,
                    grad_policy=norm,grad_q=q_norm,demo=float(demo_loss.detach()),anchor_weight=self.anchor_weight,
                    grad_shared=float(sum(p.grad.detach().square().sum().item() for p in self.shared_params)**.5),
                    entropy=float(entropy),speed_entropy=float(speed_entropy),alpha=self.alpha_log.detach().exp().tolist(),
                    mean_target=float(target.mean()),terminals=int(b["terminal"].sum()),truncations=int(b["truncated"].sum()))

    def save(self,path,replay,initial):
        data=dict(model=self.model.state_dict(),critic=self.q.state_dict(),target=self.target.state_dict(),
                  policy_optimizer=self.policy_opt.state_dict(),critic_optimizer=self.q_opt.state_dict(),
                  alpha_optimizer=self.alpha_opt.state_dict(),log_alpha=self.alpha_log.detach(),
                  sac_updates=self.index,anchor_weight=self.anchor_weight,rng_torch=torch.get_rng_state(),
                  rng_numpy=json.dumps(self.rng.bit_generator.state),replay=replay.manifest,
                  source=dict(path=str(initial),sha256=sha(initial)))
        tmp=path.with_suffix(".tmp")
        torch.save(data,tmp)
        tmp.rename(path)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("initial",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--updates",type=int,required=True)
    cli.add_argument("--seed",type=int,default=4621)
    cli.add_argument("--anchor",type=float,default=.20)
    cli.add_argument("episodes",type=Path,nargs="+")
    args=cli.parse_intermixed_args()
    torch.set_num_threads(1)
    args.output.mkdir()
    sac=SAC(args.initial.resolve(),args.seed,args.anchor)
    replay=Replay([p.resolve() for p in args.episodes],sac.rng)
    (args.output/"REPLAY.json").write_text(json.dumps(replay.manifest,indent=2)+"\n")
    with (args.output/"UPDATES.jsonl").open("x") as diary:
        for step in range(1,args.updates+1):
            result=sac.step(replay.sample(12))
            diary.write(json.dumps(result)+"\n")
            diary.flush()
            if step in (1,args.updates) or step%64==0:
                print(result,flush=True)
            if step%128==0 or step==args.updates:
                sac.save(args.output/f"{sac.index:04d}.pt",replay,args.initial.resolve())
    print(dict(actual_updates=args.updates,source_physical_actor_ticks=sum(r["steps"] for r in replay.manifest),
               source_native=sum(r["native_steps"] for r in replay.manifest),finish=sac.index),flush=True)


if __name__=="__main__":
    main()
