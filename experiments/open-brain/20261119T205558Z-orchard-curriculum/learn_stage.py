"""Conservative SAC fine-tuning on measured P1 physical decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from rewards import analyze

HERE=Path(__file__).resolve().parent
ODOR=HERE.parent/'20261119T190005Z-odor-sac'
sys.path.insert(0,str(ODOR))
from train import SAC, Replay, export, actor_mean

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',type=int,choices=[1,2,3])
    p.add_argument('parent',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('episodes',type=Path,nargs='+')
    a=p.parse_args()
    a.out.mkdir()
    seed=20600+a.phase
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    state=torch.load(a.parent,map_location='cpu',weights_only=True)
    agent=SAC(seed)
    for key in ('actor','q1','q2','t1','t2'):
        getattr(agent,key).load_state_dict(state[key])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'])
    # Reset optimizer moments on changing reward scale. Constrain drift from parent
    # on the actual sampled states, which limits extrapolation on small replay sets.
    agent.aopt.param_groups[0]['lr']=.00005
    buf=Replay()
    scores={}
    for folder in a.episodes:
        rows,scores[folder.name]=analyze(folder,a.phase)
        for row in rows:
            buf.add(*row)
    assert buf.count>=8
    obs=torch.from_numpy(buf.rows[0][:buf.count])
    with torch.no_grad():
        anchor=agent.actor(obs,False)[0].clone()
    limit=max(12,min(220,2*buf.count))
    best=None; log=[]
    for step in range(1,limit+1):
        losses=agent.update(buf.batch(rng))
        with torch.no_grad():
            divergence=(agent.actor(obs,False)[0]-anchor).abs().max().item()
        log.append(dict(step=step,losses=losses,action_max_change=divergence))
        if not np.isfinite([*losses,divergence]).all():
            raise RuntimeError('nonfinite SAC step')
        if divergence>.18:
            break
        if step==1 or step%4==0 or step==limit:
            best=step
            agent.save(a.out/'checkpoint.pt',step,dict(phase=a.phase,parent=digest(a.parent)),rng)
    assert best is not None
    save=torch.load(a.out/'checkpoint.pt',weights_only=True)
    agent.actor.load_state_dict(save['actor'])
    assert save['actor_optimizer']['state'] and save['critic_optimizer']['state']
    export(agent.actor,a.out/'POLICY.npz')
    with np.load(a.out/'POLICY.npz') as policy,torch.no_grad():
        error=float(np.max(np.abs(actor_mean(policy,obs.numpy())-agent.actor(obs,False)[0].numpy())))
    assert error<2e-5
    (a.out/'TRAIN.json').write_text(json.dumps(dict(phase=a.phase,episodes=scores,transitions=buf.count,
        attempts=len(log),selected=best,log=log,parent_sha=digest(a.parent),reload_error=error,
        checkpoint_sha256=digest(a.out/'checkpoint.pt'),policy_sha256=digest(a.out/'POLICY.npz')),indent=2)+'\n')
    print(a.phase,buf.count,best,error,flush=True)

if __name__=='__main__':
    main()
