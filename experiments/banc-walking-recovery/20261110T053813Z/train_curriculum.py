"""Matched 238-feature FF/recurrent physical PPO pilots. Derived-world attempts only."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from models import ActorCritic
from ppo import PPOConfig,PPOTrainer,RolloutBuffer,save_checkpoint,load_checkpoint
from recurrent import RecurrentPPOConfig,RecurrentPPOTrainer
from physics import Core
from curriculum import RewardConfig,RewardTracker,absorbing_tail
from policy_io import actor_observation,critic_observation,initial_state,evaluate_policy
from run_baseline import seed_all,require_checks
from identity import atomic_json,capture,verify,verify_runtime,sha256


def required_reports():
    require_checks()
    for name in ('RECURRENT_CHECKS.json','CURRICULUM_CHECKS.json'):
        data=json.loads((Path(__file__).parent/name).read_text())
        if data['all_passed'] is not True:
            raise RuntimeError('correctness check not passing: '+name)
        for path,digest in data['source_hashes'].items():
            if sha256(path)!=digest:
                raise RuntimeError('stale check '+path)


def collect_episode(core,model,target,ticks,decision_ms,reward_config):
    if not math.isclose(decision_ms*.001,reward_config.policy_discount_dt_s,rel_tol=0,abs_tol=1e-12):
        raise ValueError('reward absorbing discount period must equal decision period')
    core.reset(); previous=np.zeros(42); hidden=initial_state(model)
    tracker=RewardTracker(reward_config,horizon_s=ticks*.001,target_speed=target,
                          initial_heading=core.initial_heading,start_time_s=float(core.d.time))
    keys=('obs','critic_obs','latent','log_prob','mean','logstd','value','reward','terminated','truncated','next_value')
    rows={key:[] for key in keys}; components={}; actual_ticks=0; absorbing_total=0.;reason='deadline'
    for k in range(0,ticks,decision_ms):
        obs=actor_observation(core,model,previous,target)
        cv=critic_observation(core,model,obs,(ticks-k)*.001)
        sample=model.act(obs,cv,state=hidden);hidden=sample['state']
        action=sample['action'][0].cpu().numpy()
        total=0.; terminal=False; requested_block=min(decision_ms,ticks-k)
        for sub in range(requested_block):
            core.tick(action);actual_ticks+=1
            result=tracker.update(tracker.measure(core,action))
            total+=result['reward']
            for name,value in result['components'].items():
                components[name]=components.get(name,0.)+value*.001
            if result['terminated']:
                terminal=True;reason=','.join(result['terminal_reasons'])
                # Complete the reward-only failed portion of this decision interval,
                # then attach discounted worst-case future return at next boundary.
                # No fake physics, observations or policy samples are created.
                remainder=(requested_block-sub-1)*.001
                after_boundary=max(0.,(ticks-k-requested_block)*.001)
                penalty=reward_config.min_rate*remainder+math.exp(-decision_ms*.001/reward_config.discount_time_constant_s)*absorbing_tail(after_boundary,reward_config)
                total+=penalty;absorbing_total+=penalty
                break
        terminal=terminal or k+requested_block>=ticks
        previous=action.copy()
        nextobs=actor_observation(core,model,previous,target)
        nextcv=critic_observation(core,model,nextobs,max(0.,(ticks-k-requested_block)*.001))
        with torch.no_grad():
            nextvalue=0. if terminal else float(model.value(torch.as_tensor(nextcv[None],dtype=torch.float64))[0])
        values=dict(obs=obs,critic_obs=cv,latent=sample['latent'][0].numpy(),
                    log_prob=float(sample['log_prob'][0]),mean=sample['mean'][0].numpy(),
                    logstd=sample['logstd'][0].numpy(),value=float(sample['value'][0]),reward=total,
                    terminated=terminal,truncated=False,next_value=nextvalue)
        for key,value in values.items(): rows[key].append(value)
        if terminal: break
    episode={key:np.asarray(value) for key,value in rows.items()}
    episode['starts_at_reset']=True
    diagnostics=dict(actual_ticks=actual_ticks,decision_steps=len(rows['reward']),return_sum=float(sum(rows['reward'])),
                     absorbing_math_only_penalty=absorbing_total,terminal_reason=reason,component_integrals=components)
    return episode,diagnostics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','replay'])
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--target-speed',type=float,default=None)
    p.add_argument('--updates',type=int,default=3)
    p.add_argument('--rollout-ticks',type=int,default=8000)
    p.add_argument('--ticks',type=int,default=None)
    p.add_argument('--seed',type=int,default=20261113)
    p.add_argument('--native',action='store_true')
    p.add_argument('--adhesion',choices=['on','off'],default=None)
    a=p.parse_args();seed_all(a.seed)
    required_reports()
    checkpoint=Path(a.checkpoint).resolve()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    verify_runtime(payload['identity'],Core(adhesion=payload['identity']['adhesion']['enabled']))
    saved_model=payload['model_identity']
    architecture=saved_model['architecture']
    if architecture not in ('ff238','gru238') or saved_model['critic_obs_dim']!=255 or saved_model['dtype']!='torch.float64':
        raise ValueError('supported model contract is FF/GRU238 with critic255 float64')
    model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64,logstd_bounds=tuple(saved_model['logstd_bounds']))
    if model.identity()!=saved_model:
        raise ValueError('full model metadata mismatch')
    model.load_state_dict(payload['state']['model'])
    saved_task=payload['identity']['config']
    if a.mode=='replay':
        expected_target=saved_task.get('target_speed',.10 if saved_task.get('latent_initial_std') is not None else None)
        expected_ticks=saved_task.get('ticks')
        if expected_ticks is None and saved_task.get('dataset'):
            expected_ticks=json.loads((Path(saved_task['dataset']).parent/'DATASET.json').read_text())['config']['ticks']
        if expected_target is None or expected_ticks is None:
            raise ValueError('checkpoint missing explicit replay task; pass through versioned metadata migration')
        if a.target_speed is not None and a.target_speed!=expected_target or a.ticks is not None and a.ticks!=expected_ticks:
            raise ValueError('replay overrides change saved task; use separately declared evaluation experiment')
        a.target_speed=expected_target;a.ticks=expected_ticks
        expected_adhesion=bool(saved_task.get('adhesion_enabled',payload['identity']['adhesion']['enabled']))
        if a.adhesion is not None and (a.adhesion=='on')!=expected_adhesion:
            raise ValueError('replay adhesion override changes saved physical task')
        a.adhesion=expected_adhesion
    else:
        a.adhesion=False if a.adhesion is None else a.adhesion=='on'
        a.target_speed=0. if a.target_speed is None else a.target_speed
        a.ticks=2000 if a.ticks is None else a.ticks
    if a.ticks<2 or a.ticks%5 or a.updates<1 or a.rollout_ticks<1:
        raise ValueError('positive budget, horizon >=2 and multiple of5ms required')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    reward_cfg=RewardConfig()
    common=dict(gamma=math.exp(-.005),gae_lambda=math.exp(-.005/.2),actor_lr=3e-5,
                critic_lr=1e-4,clip=.1,epochs=4,minibatch=64,target_kl=.03)
    if architecture=='gru238':
        config=RecurrentPPOConfig(**common,episodes_per_minibatch=1)
        trainer=RecurrentPPOTrainer(model,config)
    else:
        config=PPOConfig(**common);trainer=PPOTrainer(model,config)
    if a.mode=='replay':
        # Restore the actual saved state/config, not a new training initialization.
        config=(RecurrentPPOConfig if architecture=='gru238' else PPOConfig)(**payload['config'])
        trainer=(RecurrentPPOTrainer if architecture=='gru238' else PPOTrainer)(model,config)
        load_checkpoint(checkpoint,trainer,payload['identity'])
        result=evaluate_policy(model,a.ticks,5,a.target_speed,out/'native' if a.native else None,adhesion=a.adhesion)
        atomic_json(out/'EVALUATION.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),evaluation=result))
        print(json.dumps(result),flush=True);return
    settings=dict(architecture=architecture,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                  target_speed=a.target_speed,seed=a.seed,updates=a.updates,ticks=a.ticks,rollout_ticks=a.rollout_ticks,
                  decision_ms=5,adhesion_enabled=a.adhesion,ppo=asdict(config),reward=reward_cfg.to_dict(),
                  optimizer='new PPO states after source checkpoint; no inherited BC moments',
                  finite_horizon='2s finite task zero bootstrap, mathematical worst-rate absorbing tail on training failure')
    if not math.isclose(config.gamma,math.exp(-reward_cfg.policy_discount_dt_s/reward_cfg.discount_time_constant_s),rel_tol=0,abs_tol=1e-12):
        raise ValueError('PPO gamma mismatches absorbing reward contract')
    core=Core(adhesion=a.adhesion);identity=capture(core,settings,extra_paths=[checkpoint])
    atomic_json(out/'IDENTITY.json',identity)
    save_checkpoint(out/'initial.pt',trainer,identity,extra=dict(update=-1))
    log=dict(status='running',config=settings,updates=[],initial_evaluation=evaluate_policy(model,a.ticks,5,a.target_speed,adhesion=a.adhesion))
    atomic_json(out/'TRAINING.json',log);start=time.monotonic()
    try:
        for update in range(a.updates):
            episodes=[];diag=[];actual=0
            while actual<a.rollout_ticks:
                episode,d=collect_episode(core,model,a.target_speed,a.ticks,5,reward_cfg)
                episodes.append(episode);diag.append(d);actual+=d['actual_ticks']
            if architecture=='gru238':opt=trainer.update(episodes)
            else:
                buffer=RolloutBuffer()
                for e in episodes:
                    for t in range(len(e['reward'])):
                        buffer.add(**{key:e[key][t] for key in buffer.keys})
                opt=trainer.update(buffer.finish(config.gamma,config.gae_lambda))
            cp=out/'checkpoints'/f'update_{update:04d}.pt'
            save_checkpoint(cp,trainer,identity,extra=dict(update=update))
            ev=evaluate_policy(model,a.ticks,5,a.target_speed,adhesion=a.adhesion)
            row=dict(update=update,checkpoint=str(cp),checkpoint_sha256=sha256(cp),episodes=diag,
                     actual_physics_ticks=actual,optimization=opt,evaluation=ev)
            log['updates'].append(row);verify(identity);atomic_json(out/'TRAINING.json',log)
            print(json.dumps(dict(update=update,kl=opt.get('kl'),
                evaluation={key:ev[key] for key in ('forward_signed_native_per_s','com_height_min','nonfoot_force_max_native','slip_ratio_max','all_gates')})),flush=True)
        save_checkpoint(out/'final.pt',trainer,identity,extra=dict(update=a.updates-1))
        log.update(status='completed',wall_s=time.monotonic()-start)
    except BaseException as exc:
        log.update(status='failed',error=repr(exc),wall_s=time.monotonic()-start)
        atomic_json(out/'TRAINING.json',log);raise
    verify(identity);atomic_json(out/'TRAINING.json',log)


if __name__=='__main__':main()
