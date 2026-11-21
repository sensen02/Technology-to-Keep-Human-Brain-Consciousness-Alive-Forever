"""Continue the feeding body from the existing candidate with flat replay."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
import numpy as np
import torch

from fit_feeding import OUT,BASE,OLD,PRE,ObservedBody,FlatFloor,T,local_axes

PRIOR=BASE/'20261121T034919Z-feeding-body-observations'
FOOD_PRACTICE=BASE/'20261121T040639Z-feeding-body-practice'
ROUGH=BASE/'20261121T025425Z-mild-terrain-100x'
DN=BASE/'20261121T012712Z-body226-distal-cut'
sys.path.insert(0,str(DN))
from fit_body import load as flat

DEV=(88207,88209)+tuple(range(88300,88480))
HELD=(88211,88224,88225)
HORIZON=40


def demo(seed):
    path=OUT/f'aligned-{seed}'
    if path.exists():return path
    inherited=FOOD_PRACTICE/path.name
    return inherited if inherited.exists() else PRIOR/path.name


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def food(seed):
    folder=demo(seed)
    with np.load(folder/'external.npz') as f:
        p=T(f['markers_mm'])
        force=T(f['force_uN'])
        mouth=T(f['mouth_mm'])
        valid=f['valid']
    with np.load(folder/'electrode.npz') as f:
        counts=T(f['counts'])
    assert len(counts)==len(p) and valid[:2].all()
    hist=torch.stack([torch.cat([torch.zeros_like(counts[:i]),counts[:-i]]) if i else counts
                      for i in (3,2,1,0)],1)
    touch=torch.stack([torch.cat([force[:1],force[:-1]]),force],1)
    cfg=read_json(folder/'RESULT.json')
    goal=T([cfg['geometry']['goal_x']*10,cfg['geometry']['goal_y']*10])
    near=(mouth[:,:2]-goal).norm(dim=1)<5.4
    turn=(local_axes(p)[10:,:2,0]-local_axes(p)[:-10,:2,0]).norm(dim=1)>.04
    turn=torch.nn.functional.pad(turn,(0,10))
    return (p,touch,hist),valid,near,turn,cfg


def build_windows(rows,flags,horizon):
    poses,pressures,commands=zip(*rows)
    sizes=[len(p) for p in poses]
    pad=lambda items:torch.nn.utils.rnn.pad_sequence(items,batch_first=True)
    valid=[];focus=[]
    for lane,(size,flag) in enumerate(zip(sizes,flags)):
        ok,near,turn=flag
        for t in range(1,size-horizon):
            if not ok[t-1:t+horizon+1].all():continue
            pair=(lane,t);valid.append(pair)
            if near[t:t+horizon].any() or turn[t:t+horizon].any():focus.append(pair)
    return tuple(map(pad,(poses,pressures,commands))),torch.tensor(valid),torch.tensor(focus,dtype=torch.long).reshape(-1,2)


def flat_rows(n=96):
    originals=[r[:4] for seed in (10901,11311) for r in flat(seed)]
    bank={r[0]:r[3] for r in originals}
    rows=[r[1:] for r in originals]
    base=np.random.default_rng(94710)
    ids=sorted(map(int,base.choice(np.arange(20,2792),n,replace=False)))
    for ident in ids:
        dest=ROUGH/'episodes'/f'terrain{ident:02d}'
        receipt=read_json(dest/'receipt.json')
        key=f"{receipt['seed']}_{receipt['mode']}"
        with np.load(dest/'external.npz') as z:
            pos=T(z['markers_mm']);force=T(z['force_uN'])
        rows.append((pos,torch.stack((torch.cat([force[:1],force[:-1]]),force),1),bank[key]))
    return rows,ids


def load_floor(p):
    m=FlatFloor()
    m.load_state_dict(torch.load(p,weights_only=True,map_location='cpu'))
    return m


def fit_floor(model, observations, steps):
    inputs=[]
    for (pos,f,_),valid,*_ in observations:
        idx=torch.from_numpy(valid[:-1]&valid[1:]).nonzero().flatten()
        inputs.append((pos[idx],pos[idx+1],f[idx+1,1]))
    p,q,y=[torch.cat(items) for items in zip(*inputs)]
    optim=torch.optim.AdamW(model.parameters(),lr=.0003)
    for _ in range(steps):
        idx=torch.randint(len(y),(96,))
        predicted=model(p[idx],q[idx])
        loss=torch.nn.functional.smooth_l1_loss(predicted,y[idx])
        optim.zero_grad();loss.backward();optim.step()
    return model.eval().requires_grad_(False)


def rollout(net,floor,batch,use_estimate,*,pressure_weight,orientation_weight=0.):
    p,f,dn=batch
    prev,now=p[:,0],p[:,1]
    last=floor(prev,prev)
    loss=0.
    pred_force=floor(prev,now)
    for k in range(1,p.shape[1]-1):
        loads=torch.where(use_estimate,torch.stack((last,pred_force),1),f[:,k])
        predicted,_=net(prev,now,loads,dn[:,k])
        target=p[:,k+1]
        error=torch.nn.functional.smooth_l1_loss((predicted-target)/.035,torch.zeros_like(target))
        center=torch.nn.functional.smooth_l1_loss((predicted[:,0]-target[:,0])/.08,torch.zeros_like(target[:,0]))
        a=torch.nn.functional.normalize(predicted[:,1,:2]-predicted[:,0,:2],dim=1)
        b=torch.nn.functional.normalize(target[:,1,:2]-target[:,0,:2],dim=1)
        loss+=error+.25*center+.75*(a-b).square().sum(-1).mean()
        next_force=floor(now,predicted)
        if pressure_weight:
            surface=f[:,k+1,1]
            pressure_match=torch.nn.functional.smooth_l1_loss(next_force.log1p(),surface.log1p())
            chest_z=(predicted[:,0,2]-target[:,0,2])/.04
            height=torch.nn.functional.smooth_l1_loss(chest_z,torch.zeros_like(chest_z))
            loss+=pressure_weight*pressure_match+.25*height
        if orientation_weight:
            predicted_up=local_axes(predicted)[:,:,2]
            observed_up=local_axes(target)[:,:,2]
            loss+=orientation_weight*(predicted_up-observed_up).square().sum(-1).mean()
        last,pred_force=pred_force,next_force
        prev,now=now,predicted
    return loss/(p.shape[1]-2)


def fit(seeds,updates,variant,initial='adapted',long=False):
    assert not set(seeds)&set(HELD) and set(seeds)<=set(DEV)
    destination=OUT/variant
    destination.mkdir(exist_ok=True)
    origin=FOOD_PRACTICE/'long-rollout' if initial=='long-rollout' else OUT/initial
    for name in ('skeleton.json','mouth_offset.json'):
        shutil.copyfile(origin/name,destination/name)
    torch.manual_seed(882170)
    observations=[food(s) for s in seeds]
    maximum=100 if long else HORIZON
    fdata,fwindows,focus=build_windows([r[0] for r in observations],[r[1:4] for r in observations],maximum)
    startup=fwindows[fwindows[:,1]<=26]
    rows,ids=flat_rows()
    background=[(np.ones(len(p),bool),np.zeros(len(p),bool),np.zeros(len(p),bool)) for p,_,_ in rows]
    bdata,bwindows,_=build_windows(rows,background,HORIZON)
    assert min(len(fwindows),len(focus))>50
    food_floor=fit_floor(load_floor(origin/'floor.pt'),observations,1400)
    flat_floor=load_floor(OLD/'floor.pt').eval().requires_grad_(False)
    torch.save(food_floor.state_dict(),destination/'floor.pt')
    net=ObservedBody(origin/'weights.pt')
    ground=ObservedBody(PRE/'weights.pt')
    ground.raw=net.raw
    net.raw.feedback.requires_grad_(False)
    start_lr=.000055
    opt=torch.optim.AdamW(net.raw.motion.parameters(),lr=start_lr,weight_decay=.0005)
    losses=[[],[]];checkpoints=[]
    for i in range(updates):
        regular=i%4==0
        if regular:
            batch=bwindows[torch.randint(len(bwindows),(40,))]
            horizon=14
        elif long and i%3==1:
            batch=torch.cat([startup[torch.randint(len(startup),(8,))],focus[torch.randint(len(focus),(8,))]])
            horizon=maximum
        else:
            batch=torch.cat([fwindows[torch.randint(len(fwindows),(24,))],focus[torch.randint(len(focus),(16,))]])
            horizon=10 if i<updates//5 else (24 if i<updates*4//5 else HORIZON)
        episode,start=batch.T
        ticks=start[:,None]-1+torch.arange(horizon+2)
        tensors=[x[episode[:,None],ticks] for x in (bdata if regular else fdata)]
        estimate=torch.rand((len(batch),1,1))<(.85 if horizon==100 else .5)
        opt.param_groups[0]['lr']=start_lr*(1.-.7*i/updates)
        loss=rollout(ground if regular else net,flat_floor if regular else food_floor,tensors,estimate,pressure_weight=0. if regular else .35,
                     orientation_weight=.55 if long and not regular else 0.)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.raw.motion.parameters(),2.)
        opt.step()
        losses[int(regular)].append(float(loss.item()))
        if (i+1)%200==0 or i+1==updates:
            point=dict(update=i+1,food_loss=float(np.mean(losses[0])),prior_locomotion_loss=float(np.mean(losses[1])))
            print(point,flush=True)
            checkpoints.append(point)
            losses=[[],[]]
    torch.save(net.raw.state_dict(),destination/'weights.pt')
    assert all(torch.equal(a,b) for a,b in zip(net.raw.feedback.parameters(),
        ObservedBody(origin/'weights.pt').raw.feedback.parameters()))
    save_json(destination/'training.json',dict(updates=updates,pressure_updates=1400,train_seeds=list(seeds),max_rollout_steps=maximum,
        prior_locomotion_episodes=len(rows),prior_heightfield_ids=ids,food_windows=len(fwindows),
        targeted_windows=len(focus),initial_sha256=digest(origin/'weights.pt'),checkpoints=checkpoints,
        weights_sha256=digest(destination/'weights.pt'),teacher_intakes={str(s):r[-1]['intake'] for s,r in zip(seeds,observations)}))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('variant')
    cli.add_argument('seed',type=int,nargs='+')
    cli.add_argument('--updates',type=int,default=2000)
    cli.add_argument('--initial',default='long-rollout')
    cli.add_argument('--long',action='store_true')
    a=cli.parse_args()
    fit(a.seed,a.updates,a.variant,a.initial,a.long)
