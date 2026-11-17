"""Imitate native-validated righting targets with only current body observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from verify import qualifies
from search import ContactLog, HERE, INITIALS, STARTS, load, model, read, stand

UPDATES = 800
DURATION = 3500


class Actor(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(183,96),nn.Tanh(),nn.Linear(96,72),nn.Tanh(),nn.Linear(72,size))

    def forward(self, x):
        return self.layers(x)


def train(out, teachers):
    out.mkdir()
    torch.manual_seed(202611152134)
    generator = torch.Generator().manual_seed(202611152134)
    inputs, labels, goals = [],[],[]
    sources = {}
    for path, runs in teachers:
        result = read(path/"RESULT.json")
        if not result["passed"] or stand.sha(runs)!=result["goal_sha256"]:
            raise ValueError("Teacher has no independently measured standing window")
        with np.load(runs,allow_pickle=False) as archive:
            trace_goals = archive["goals"]
        assert result["handoff_ms"] == 5*len(trace_goals)
        with np.load(path/"TRACE.npz",allow_pickle=False) as record:
            trace_obs = record["observation"]
        starts = [0]
        starts.extend(np.flatnonzero(np.any(trace_goals[1:] != trace_goals[:-1],axis=1))+1)
        idx = np.full(len(trace_obs),-1)
        for start,stop in zip(starts,[*starts[1:],len(trace_goals)],strict=True):
            idx[start:stop] = len(goals)
            goals.append(trace_goals[start])
        inputs.append(trace_obs)
        labels.append(idx)
        for ref in (runs,path/"TRACE.npz",path/"RESULT.json"):
            sources[str(ref.resolve())]=stand.sha(ref)
    goals = np.array(goals)
    x = torch.tensor(np.concatenate(inputs),dtype=torch.double)
    y = torch.tensor(np.concatenate(labels),dtype=torch.long)
    y[y<0]=len(goals)
    center = x.mean(0)
    scale = x.std(0).clamp_min(.006)
    z = ((x-center)/scale).clamp(-10,10)
    net = Actor(len(goals)+1).double()
    opt = torch.optim.Adam(net.parameters(),lr=.002)
    criterion=nn.CrossEntropyLoss()
    with (out/"TRAIN.jsonl").open("x",buffering=1) as journal:
        for k in range(UPDATES+1):
            loss = None
            if k:
                ids = torch.randint(len(z),(192,),generator=generator)
                pred = net(z[ids] + .012*torch.randn((len(ids),183),generator=generator,dtype=torch.double))
                v = criterion(pred,y[ids])
                opt.zero_grad(set_to_none=True)
                v.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),2.,error_if_nonfinite=True)
                opt.step()
                loss=float(v.detach())
            path = out/f"step_{k:04}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(model=net.state_dict(),goals=torch.from_numpy(goals),optimizer=opt.state_dict(),
                            rng=generator.get_state(),center=center,scale=scale,step=k),temporary)
            temporary.rename(path)
            journal.write(json.dumps(dict(update=k,loss=loss,sha256=stand.sha(path)),allow_nan=False)+"\n")
    with torch.no_grad():
        predicted=net(z).argmax(1)
    assert all(int(v["step"]) == UPDATES for v in opt.state.values())
    mismatch = int((predicted != y).sum())
    report=dict(updates=UPDATES, data_rows=len(x), mismatches=mismatch, goals=len(goals),
                checkpoint_sha256=stand.sha(path), sources=sources)
    if any(stand.sha(Path(key)) != v for key,v in sources.items()):
        raise ValueError("Training source changed")
    stand.put(out/"RESULT.json",report)
    print(report,flush=True)


class Policy:
    def __init__(self,checkpoint):
        data=torch.load(checkpoint,map_location="cpu",weights_only=True)
        self.goals=data["goals"].numpy()
        self.center,self.scale=data["center"],data["scale"]
        self.net=Actor(len(self.goals)+1).double()
        self.net.load_state_dict(data["model"])
        self.net.eval()
        self.stand=model()

    @torch.no_grad()
    def command(self,obs,target):
        source=torch.from_numpy(obs)
        z=((source-self.center)/self.scale).clamp(-10,10)
        logits=self.net(z)
        index=int(logits.argmax())
        if index==len(self.goals):
            if -obs[176] >= np.cos(np.deg2rad(20.)) and np.sum(obs[168:174]>=.02)>=3:
                return 2.*self.stand(source).numpy(),index
            index=int(logits[:-1].argmax())
        return np.clip((self.goals[index]-target)/.005,-2.,2.),index


def trial(output,checkpoint,seed):
    output.mkdir()
    bench=stand.Bench(output)
    expected=read(INITIALS/"BANK.json")["seeds"][seed]["snapshot_sha256"]
    load(bench,INITIALS/f"{seed}.npz",expected)
    plant=bench.body
    rec=ContactLog(plant,80*DURATION)
    plant.recorder=rec
    policy=Policy(checkpoint)
    obs, action, branches = [],[],[]
    com=[plant.d.subtree_com[plant.th].copy()]
    targets=[plant.actuator.target_rad.copy()]
    start=float(plant.d.time)
    try:
        for t in range(DURATION):
            if t%5==0:
                x=stand.observe(plant)
                rate,index=policy.command(x,plant.actuator.target_rad)
                obs.append(x)
                action.append(rate.copy()/2.)
                branches.append(index)
                if np.abs(rate).max()>2 or not np.isfinite(rate).all():
                    raise ValueError("Invalid policy rate")
            plant.tick_rate(rate)
            if np.any(plant.d.xfrc_applied) or np.any(plant.d.qfrc_applied):
                raise ValueError("Unexpected external wrench")
            com.append(plant.d.subtree_com[plant.th].copy())
            targets.append(plant.actuator.target_rad.copy())
    finally:
        plant.recorder=None
    assert np.isclose(float(plant.d.time)-start,DURATION*.001,atol=1e-8,rtol=0.)
    native=rec.arrays()
    ms=dict(com=np.asarray(com),target=np.asarray(targets))
    report=qualifies(native,ms,0)
    finished=bool(np.all(np.asarray(branches)[-40:]==len(policy.goals)))
    path=output/"TRACE.npz"
    np.savez_compressed(path,**native,**ms,observation=obs,action=action,branches=branches)
    stand.identity.verify_runtime(bench.manifest,plant)
    info=dict(**report,stand_chosen_entire_tail=finished,full_pass=report["passed"] and finished,seed=seed,
              source_sha256=expected,weights_sha256=stand.sha(checkpoint),trace_sha256=stand.sha(path),
              native_steps=DURATION*80,settle_steps=80000)
    stand.put(output/"RESULT.json",info)
    print(info,flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    sub=a.add_subparsers(dest="mode",required=True)
    train_a=sub.add_parser("train")
    train_a.add_argument("output",type=Path)
    train_a.add_argument("--teacher",action="append",required=True,nargs=2,type=Path)
    val=sub.add_parser("trial")
    val.add_argument("output",type=Path)
    val.add_argument("checkpoint",type=Path)
    val.add_argument("seed",choices=STARTS)
    args=a.parse_args()
    torch.set_num_threads(1)
    if args.mode=="train":
        train(args.output,args.teacher)
    else:
        trial(args.output,args.checkpoint,args.seed)


if __name__=="__main__":
    main()
