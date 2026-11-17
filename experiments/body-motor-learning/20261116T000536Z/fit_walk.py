"""Imitate qualified stance, gait and stop demonstrations with a stateless body."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from walking_bench import HERE, PROJ, INPUTS, LEGS, feature_from_old, read, stand_model, stand

OLD = PROJ/"experiments/banc-walking-recovery/20261110T152714Z"
CONFIRM = OLD.parent/"20261110T152754Z"
STOP = PROJ/"experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/trials/stop"
UPDATES=1200
LAYOUT=(0,2,1,3,5,4)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.leg=nn.Sequential(nn.Linear(26,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh())
        self.whole=nn.Sequential(nn.Linear(201,160),nn.Tanh(),nn.Linear(160,160),nn.Tanh(),nn.Linear(160,42))
        self.register_buffer("names",torch.eye(6,dtype=torch.double))
        self.register_buffer("order",torch.tensor(LAYOUT,dtype=torch.long))

    def forward(self,x):
        sh=x.shape[:-1]
        inputs=torch.cat((x[..., :42].reshape(*sh,6,7),x[...,84:126].reshape(*sh,6,7),
            x[...,168:174][...,self.order].unsqueeze(-1),x[...,183:201].reshape(*sh,6,3),
            x[...,201:207].unsqueeze(-1),x[...,207:213].unsqueeze(-1),
            self.names.expand(*sh,6,6)),dim=-1)
        all_legs=self.leg(inputs).reshape(*sh,192)
        return self.whole(torch.cat((all_legs,x[...,174:183]),dim=-1))


class Net(nn.Module):
    def __init__(self, mean, scale, halfspan):
        super().__init__()
        self.locomotion=Branch()
        self.stop=Branch()
        self.register_buffer("mean",mean)
        self.register_buffer("scale",scale)
        self.register_buffer("halfspan",halfspan)
        self.standing=stand_model().requires_grad_(False)

    def forward(self,x,command):
        z=((x-self.mean)/self.scale).clamp(-8,8)
        correction=8*self.halfspan*(x[...,126:168]-self.mean[126:168])
        walking=(self.locomotion(z)-correction).tanh()
        speed=(x[...,177:183].square().sum(-1)+25*x[...,201:207].clamp_min(0).square().sum(-1))
        activation=-torch.expm1(-speed/.04)
        stopping=(self.standing(x[...,:183])+.45*activation.unsqueeze(-1)*self.stop(z).tanh()).clamp(-1,1)
        return torch.where(command.unsqueeze(-1).bool(),walking,stopping)


def freeze_source(sources,path):
    path=path.resolve()
    sources[str(path)]=stand.sha(path)


def archived(base, stopping, sources):
    freeze_source(sources,base/"RESULT.json")
    trace=base/"trace_1ms.npz"
    observed=base/("teacher_samples_5ms.npz" if stopping else "decisions_5ms.npz")
    freeze_source(sources,trace)
    freeze_source(sources,observed)
    with np.load(observed,allow_pickle=False) as z,np.load(trace,allow_pickle=False) as t:
        at=np.rint((z["time"]-t["time"][0])/.001).astype(int)
        np.testing.assert_allclose(z["time"],t["time"][at],atol=5e-9,rtol=0)
        samples={k:t[k][at] for k in ("relativefoot","padz","speed")}
        x=feature_from_old(z["X238"],samples)
        np.testing.assert_allclose(x[:,201:207]/stand.physics.rc.PAD_H_DIV,
                                   z["X238"][:,183:189],atol=2e-12,rtol=0)
        if stopping:
            use=z["candidate_phase"].astype(bool)
            return x[use],np.zeros(use.sum()),z["Y42"][use]
        return x,np.ones(len(x)),z["action"].copy()


def load_data(sources,predecessor=None):
    moving=[]
    halt=[]
    old_starts=[OLD/"EVAL_ROLE_2500",CONFIRM/"PERTURB_ROLE_2500_SEED_20261111",
                CONFIRM/"PERTURB_ROLE_2500_SEED_20261112"]
    for directory in old_starts:
        r=read(directory/"RESULT.json")
        assert r["autonomous"]["all_gates"] and r["autonomous_cycles"]["all_legs_completed"]
        moving.append(archived(directory,False,sources))
    d=read(STOP/"RESULT.json")
    assert d["full_duration"] and d["candidate_metrics"]["COM_min_native"]>=1.2
    halt.append(archived(STOP,True,sources))
    admissions={}
    for path in sorted(HERE.glob("TEACH_*/RESULT.json")):
        res=read(path)
        walk=res["walk"]
        admitted=bool(walk and res["first_step"] and walk["gates"]["forward"] and walk["gates"]["com"]
                      and walk["gates"]["support"] and walk["nonfoot_force_max_native"]==0.
                      and res["actual_ms"]>=5300)
        admitted_stop=res["terminal_stand"]["passed"] and res["actual_ms"]==6500
        admissions[path.parent.name]=dict(walk=admitted,stop=admitted_stop)
        data=path.parent/"DECISIONS.npz"
        assert stand.sha(data)==res["decision_sha256"]
        freeze_source(sources,path)
        freeze_source(sources,data)
        with np.load(data,allow_pickle=False) as z:
            combo=[z[k] for k in ("x","intent","action")]
            if admitted:
                moving.append(tuple(x[z["intent"]==1] for x in combo))
            quiet=z["ms"]<500
            if admitted_stop:
                quiet |= z["intent"]==0
            if res["initial_stand"]["passed"]:
                halt.append(tuple(x[quiet] for x in combo))
    if not any(r["walk"] for r in admissions.values()):
        raise RuntimeError("No verified newly collected launch")
    rescue=[]
    if predecessor is not None:
        for path in sorted(HERE.glob("RESCUE_*/RESULT.json")):
            r=read(path)
            if not r["admitted"]:
                continue
            data=path.parent/"EXPERT.npz"
            assert stand.sha(data)==r["data_sha256"]
            freeze_source(sources,path)
            freeze_source(sources,data)
            with np.load(data,allow_pickle=False) as z:
                rescue.append(tuple(z[k] for k in ("x","intent","action")))
        if not rescue:
            raise RuntimeError("No admitted corrective rollouts")
    return moving,halt,rescue,admissions


def stack(groups):
    return [torch.tensor(np.concatenate(col),dtype=torch.double) for col in zip(*groups)]


def fit(output,predecessor=None):
    output.mkdir()
    sources={}
    walks,stops,rescues,admitted=load_data(sources,predecessor)
    wx,wi,wa=stack(walks)
    sx,si,sa=stack(stops)
    torch.manual_seed(2026111601)
    rng=torch.Generator().manual_seed(2026111601)
    # Reuse exact native joint ordering, limits and finite actuator geometry.
    c=stand.physics.Core(adhesion=False)
    ca=c.actuator
    half=torch.tensor(np.ptp(ca.target_bounds_rad,axis=1)/2,dtype=torch.double).clamp_min(.1)
    all_inputs=torch.cat((wx,sx))
    policy=Net(wx.mean(0),all_inputs.std(0).clamp_min(.007),half).double()
    opt=torch.optim.Adam((p for p in policy.parameters() if p.requires_grad),lr=.0015)
    if predecessor:
        previous=torch.load(predecessor,map_location="cpu",weights_only=True)
        policy.load_state_dict(previous["model"])
        opt.load_state_dict(previous["optimizer"])
        rng.set_state(previous["rng"])
        opt.param_groups[0]["lr"]=.0006
        rx,ri,ra=stack(rescues)
        freeze_source(sources,predecessor)
    identity=stand.identity.capture(c,dict(train_rows=len(wx)+len(sx),new_demo=admitted))
    stand.put(output/"IDENTITY.json",identity)
    with (output/"JOURNAL.jsonl").open("x",buffering=1) as journal:
        for step in range(UPDATES+1):
            loss=None
            if step:
                ids=torch.randint(len(wx),(192,),generator=rng)
                i2=torch.randint(len(sx),(96,),generator=rng)
                x,z,y = torch.cat((wx[ids],sx[i2])),torch.cat((wi[ids],si[i2])),torch.cat((wa[ids],sa[i2]))
                if predecessor:
                    j=torch.randint(len(rx),(96,),generator=rng)
                    x,z,y=torch.cat((x,rx[j])),torch.cat((z,ri[j])),torch.cat((y,ra[j]))
                # Small local observed-state noise only on the MLP's sensory channels.
                noisy=x.clone()
                s = torch.tensor([*range(42),*range(84,126),*range(174,207)])
                noise=.009*policy.scale[s]*torch.randn((len(x),len(s)),generator=rng,dtype=x.dtype)
                noisy[:,s] += noise
                prediction=policy(noisy,z)
                v=(prediction-y).square().mean()
                opt.zero_grad(set_to_none=True)
                v.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(),2.,error_if_nonfinite=True)
                opt.step()
                loss=float(v.detach())
            path=output/f"step_{step:04}.pt"
            tmp=path.with_suffix(".tmp")
            torch.save(dict(model=policy.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                            step=step, sources=sources),tmp)
            tmp.rename(path)
            journal.write(json.dumps(dict(step=step, loss=loss,sha256=stand.sha(path)),allow_nan=False)+"\n")
    with torch.no_grad():
        errors=[float((policy(a,b)-v).square().mean()) for a,b,v in ((wx,wi,wa),(sx,si,sa))]
    stand.identity.verify_runtime(identity,c)
    assert all(stand.sha(Path(k))==v for k,v in sources.items())
    result=dict(updates=UPDATES,walk_rows=len(wx),stop_rows=len(sx),new_runs=admitted,
                rescue_rows=len(rx) if predecessor else 0, mse=errors, sha256=stand.sha(path), inputs=sources)
    stand.put(output/"RESULT.json",result)
    print({k:v for k,v in result.items() if k!="inputs"},flush=True)


def load_policy(path):
    r=read(path.parent/"RESULT.json")
    assert stand.sha(path)==r["sha256"]
    saved=torch.load(path,map_location="cpu",weights_only=True)
    params=saved["model"]
    net=Net(params["mean"],params["scale"],params["halfspan"]).double()
    net.load_state_dict(params)
    net.eval()

    @torch.no_grad()
    def predict(x, intent):
        return net(torch.from_numpy(x),torch.tensor(intent)).numpy()
    return predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--predecessor",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    fit(a.output,a.predecessor)


if __name__=="__main__":
    main()
