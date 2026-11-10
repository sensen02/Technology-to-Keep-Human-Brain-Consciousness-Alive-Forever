"""Matched-input FF/GRU BC initialization; no imitation score is a walking claim."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from models import ActorCritic
from ppo import PPOTrainer,PPOConfig,save_checkpoint
from physics import Core
from policy_io import evaluate_policy
from run_baseline import seed_all
from identity import atomic_json,capture,verify,sha256


def forward_sequence(model,obs):
    if model.architecture=='gru238':
        return model.actor.sequence(obs,model.actor.initial_state(obs.shape[1]))[0]
    return model.actor(obs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--architecture',choices=['ff238','gru238'],required=True)
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--seed',type=int,default=20261112)
    a=p.parse_args()
    seed_all(a.seed)
    dataset=Path(a.dataset).resolve()
    meta=json.loads((dataset.parent/'DATASET.json').read_text())
    if sha256(dataset)!=meta['dataset_sha256']:
        raise RuntimeError('dataset identity changed')
    config=dict(architecture=a.architecture,steps=a.steps,seed=a.seed,
                dataset=str(dataset),dataset_sha256=sha256(dataset),lr=.001,
                observation='238 engineering sensory; no phase or teacher',
                latent_initial_std=.10,sequence='whole physical reset episode, train episodes weighted by all samples',
                optimizer='BC actor only; reset Adam before PPO',
                teacher_limit='legacy learned policy on derived world; fails slip, this is initialization')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    model=ActorCritic(a.architecture,critic_obs_dim=255,initial_logstd=math.log(.10),dtype=torch.float64)
    core=Core();identity=capture(core,config)
    # Dataset checksum is included in config before the identity digest is computed.
    atomic_json(out/'IDENTITY.json',identity)
    with np.load(dataset,allow_pickle=False) as z:
        train=[(torch.as_tensor(z[f'obs_{i}'],dtype=torch.float64),
                torch.as_tensor(z[f'action_{i}'],dtype=torch.float64)) for i in meta['train_episodes']]
        validation=[(torch.as_tensor(z[f'obs_{i}'],dtype=torch.float64),
                torch.as_tensor(z[f'action_{i}'],dtype=torch.float64)) for i in meta['validation_episodes']]
    # Equal length stored episodes permit T,B,D. Each independent episode starts h=0.
    X=torch.stack([x for x,y in train],dim=1)
    Y=torch.stack([y for x,y in train],dim=1)
    target=torch.atanh(Y.clamp(-.999,.999))
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=config['lr'])
    history=[];start=time.monotonic()
    for step in range(a.steps):
        mean=forward_sequence(model,X)
        loss=(mean-target).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.)
        optimizer.step()
        if not math.isfinite(float(loss.detach())) or not math.isfinite(float(gn)):
            raise FloatingPointError('nonfinite BC')
        if step%20==0 or step+1==a.steps:
            with torch.no_grad():
                train_mse=float((forward_sequence(model,X).tanh()-Y).square().mean())
                vm=[]
                for x,y in validation:
                    pred=forward_sequence(model,x[:,None])[:,0]
                    vm.append(float((pred.tanh()-y).square().mean()))
            row=dict(step=step,loss=float(loss.detach()),train_action_mse=train_mse,
                     validation_action_mse=float(np.mean(vm)),gradient_norm=float(gn))
            history.append(row);print(json.dumps(row),flush=True)
    if a.architecture=='gru238':
        from recurrent import RecurrentPPOTrainer,RecurrentPPOConfig
        trainer=RecurrentPPOTrainer(model,RecurrentPPOConfig())
    else:
        trainer=PPOTrainer(model,PPOConfig(gamma=math.exp(-.005),gae_lambda=math.exp(-.005/.2)))
    # Full BC optimizer preserved as extra; PPO optimizers deliberately new with zero states.
    save_checkpoint(out/'bc.pt',trainer,identity,
                    extra=dict(stage='BC',bc_optimizer=optimizer.state_dict(),curve=history))
    result=evaluate_policy(model,ticks=meta['config']['ticks'],decision_ms=5,target_speed=.10)
    verify(identity)
    atomic_json(out/'BC_RESULT.json',dict(config=config,curve=history,evaluation=result,
           checkpoint_sha256=sha256(out/'bc.pt'),wall_s=time.monotonic()-start,
           parameters=sum(p.numel() for p in model.actor.parameters()),
           conclusion='initialization assessment; teacher-free closed-loop evaluation; not original walking acceptance'))
    print(json.dumps(dict(evaluation=result,wall_s=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    main()
