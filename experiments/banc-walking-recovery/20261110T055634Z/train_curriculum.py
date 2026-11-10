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
from cycle_reward import CycleConfig,CycleTracker
from policy_io import actor_observation,critic_observation,initial_state,evaluate_policy
from run_baseline import seed_all,require_checks
from identity import atomic_json,capture,verify,verify_runtime,sha256


def required_reports():
    require_checks()
    for name in ('RECURRENT_CHECKS.json','CURRICULUM_CHECKS.json','CYCLE_CHECKS.json'):
        data=json.loads((Path(__file__).parent/name).read_text())
        if data['all_passed'] is not True:
            raise RuntimeError('correctness check not passing: '+name)
        for path,digest in data['source_hashes'].items():
            if sha256(path)!=digest:
                raise RuntimeError('stale check '+path)


def collect_episode(core,model,target,ticks,decision_ms,reward_config,cycle_config=None):
    if not math.isclose(decision_ms*.001,reward_config.policy_discount_dt_s,rel_tol=0,abs_tol=1e-12):
        raise ValueError('reward absorbing discount period must equal decision period')
    core.reset(); previous=np.zeros(42); hidden=initial_state(model)
    tracker=RewardTracker(reward_config,horizon_s=ticks*.001,target_speed=target,
                          initial_heading=core.initial_heading,start_time_s=float(core.d.time))
    cycles=None if cycle_config is None else CycleTracker(cycle_config).reset(start_time=float(core.d.time))
    cycle_events=[];cycle_counts=np.zeros(6,dtype=int);cycle_shaping=0.;cycle_bonus=0.
    keys=('obs','critic_obs','latent','log_prob','mean','logstd','value','reward','terminated','truncated','next_value')
    rows={key:[] for key in keys}; components={}; actual_ticks=0; absorbing_total=0.;reason='deadline'
    for k in range(0,ticks,decision_ms):
        phi_before=0. if cycles is None else cycles.potential
        block_cycle_bonus=0.
        obs=actor_observation(core,model,previous,target)
        cv=critic_observation(core,model,obs,(ticks-k)*.001)
        sample=model.act(obs,cv,state=hidden);hidden=sample['state']
        action=sample['action'][0].cpu().numpy()
        total=0.; terminal=False; requested_block=min(decision_ms,ticks-k)
        for sub in range(requested_block):
            core.tick(action);actual_ticks+=1
            result=tracker.update(tracker.measure(core,action))
            if cycles is not None:
                event_result=cycles.measure(core)
                block_cycle_bonus+=event_result['completion_reward']
                cycle_events.extend(event_result['events'])
                for event in event_result['events']:
                    if event['type']=='complete':
                        from physics import LEG_ORDER
                        cycle_counts[LEG_ORDER.index(event['leg'])]+=1
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
        if cycles is not None:
            if terminal:
                cycle_events.extend(cycles.finalize()['events'])
            phi_after=cycles.potential
            shaped=math.exp(-decision_ms*.001/reward_config.discount_time_constant_s)*phi_after-phi_before
            total+=shaped+block_cycle_bonus
            cycle_shaping+=shaped;cycle_bonus+=block_cycle_bonus
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
                     absorbing_math_only_penalty=absorbing_total,terminal_reason=reason,component_integrals=components,
                     cycle_counts=cycle_counts.tolist(),cycle_events=cycle_events,
                     cycle_potential_shaping=cycle_shaping,cycle_completion_bonus=cycle_bonus)
    return episode,diagnostics


def parse_bool(value):
    if isinstance(value,bool): return value
    if isinstance(value,str) and value.lower() in ('true','false','on','off','1','0'):
        return value.lower() in ('true','on','1')
    if value in (0,1): return bool(value)
    raise ValueError('invalid saved boolean: '+repr(value))


def resolve_task_settings(payload,mode,target_speed=None,ticks=None,adhesion=None,resume=False):
    if resume and mode!='train': raise ValueError('--resume is train-only')
    saved=payload['identity']['config']
    requested=None if adhesion is None else parse_bool(adhesion)
    if mode=='replay' or resume:
        target=saved.get('target_speed',.10 if saved.get('latent_initial_std') is not None else None)
        horizon=saved.get('ticks')
        if horizon is None and saved.get('dataset'):
            horizon=json.loads((Path(saved['dataset']).parent/'DATASET.json').read_text())['config']['ticks']
        if target is None or horizon is None:
            raise ValueError('checkpoint missing explicit saved task; versioned metadata migration required')
        enabled=parse_bool(saved.get('adhesion_enabled',payload['identity']['adhesion']['enabled']))
        if enabled!=parse_bool(payload['identity']['adhesion']['enabled']):
            raise ValueError('saved task and physical adhesion identity disagree')
        if (target_speed is not None and target_speed!=target or ticks is not None and ticks!=horizon
                or requested is not None and requested!=enabled):
            raise ValueError('resume/replay override changes saved task')
        return dict(target_speed=float(target),ticks=int(horizon),adhesion=enabled)
    return dict(target_speed=0. if target_speed is None else target_speed,
                ticks=2000 if ticks is None else ticks,adhesion=False if requested is None else requested)


def build_trainer(payload,restore_config=False):
    saved=payload['model_identity']; architecture=saved['architecture']
    if architecture not in ('ff238','gru238') or saved['critic_obs_dim']!=255 or saved['dtype']!='torch.float64':
        raise ValueError('supported model contract is FF/GRU238 with critic255 float64')
    model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64,logstd_bounds=tuple(saved['logstd_bounds']))
    if model.identity()!=saved: raise ValueError('full model metadata mismatch')
    model.load_state_dict(payload['state']['model'])
    cls=RecurrentPPOConfig if architecture=='gru238' else PPOConfig
    common=dict(gamma=math.exp(-.005),gae_lambda=math.exp(-.005/.2),actor_lr=3e-5,
                critic_lr=1e-4,clip=.1,epochs=4,minibatch=64,target_kl=.03)
    if architecture=='gru238': common['episodes_per_minibatch']=1
    config=cls(**(payload['config'] if restore_config else common))
    trainer=(RecurrentPPOTrainer if architecture=='gru238' else PPOTrainer)(model,config)
    return trainer


def load_source(checkpoint,resume=False):
    checkpoint=Path(checkpoint).resolve()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    # Compatibility is checked against the source physical mode, before any new task mode.
    verify_runtime(payload['identity'],Core(adhesion=parse_bool(payload['identity']['adhesion']['enabled'])))
    trainer=build_trainer(payload,restore_config=resume)
    if resume: load_checkpoint(checkpoint,trainer,payload['identity'])
    return payload,trainer


def state_equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def parent_env_steps(checkpoint,payload):
    extra=payload.get('extra') or {}
    for key in ('cumulative_env_steps','cumulative_physics_ticks'):
        value=extra.get(key)
        if isinstance(value,int) and not isinstance(value,bool) and value>=0: return value
    # Legacy training checkpoints stored a zero-based local update; recover only a complete prefix.
    local=extra.get('local_update',extra.get('update'))
    candidates=[Path(checkpoint).parent/'TRAINING.json',Path(checkpoint).parent.parent/'TRAINING.json']
    if isinstance(local,int):
        for path in candidates:
            if not path.is_file(): continue
            log=json.loads(path.read_text()); rows=log.get('updates',[])
            prefix=rows[:local+1]
            if local<0: return 'UNKNOWN'
            if len(prefix)!=local+1: continue
            if any(r.get('local_update',r.get('update'))!=i for i,r in enumerate(prefix)): continue
            last=prefix[-1]
            if last.get('checkpoint_sha256')!=sha256(checkpoint):
                # A separately saved final.pt can be equivalent to the logged update checkpoint.
                recorded=Path(last.get('checkpoint',''))
                if not recorded.is_file() or sha256(recorded)!=last.get('checkpoint_sha256'): continue
                other=torch.load(recorded,map_location='cpu',weights_only=False)
                if other['identity']!=payload['identity'] or not state_equal(other['state'],payload['state']): continue
            log_config=log.get('config',{})
            if 'cumulative_env_steps_start' in log_config:
                base=log_config['cumulative_env_steps_start']
            elif log_config.get('optimizer')=='new PPO states after source checkpoint; no inherited BC moments':
                base=0  # Legacy runner explicitly began a new training segment.
            else: return 'UNKNOWN'
            if not isinstance(base,int): return 'UNKNOWN'
            if all(isinstance(r.get('actual_physics_ticks'),int) for r in prefix):
                return base+sum(r['actual_physics_ticks'] for r in prefix)
    return 'UNKNOWN'


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
    p.add_argument('--resume',action='store_true',help='train continuation: restore full saved PPO/model/RNG and task')
    p.add_argument('--adhesion',choices=['on','off'],default=None)
    a=p.parse_args();seed_all(a.seed)
    required_reports()
    checkpoint=Path(a.checkpoint).resolve()
    if a.resume and a.mode!='train': p.error('--resume is train-only')
    payload,trainer=load_source(checkpoint,resume=(a.resume or a.mode=='replay'))
    model=trainer.model; architecture=model.architecture; config=trainer.config
    task=resolve_task_settings(payload,a.mode,a.target_speed,a.ticks,a.adhesion,resume=a.resume)
    a.target_speed=task['target_speed'];a.ticks=task['ticks'];a.adhesion=task['adhesion']
    if a.ticks<2 or a.ticks%5 or a.updates<1 or a.rollout_ticks<1:
        raise ValueError('positive budget, horizon >=2 and multiple of5ms required')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    saved_task=payload['identity']['config']
    reward_cfg=RewardConfig(**saved_task['reward']) if a.resume and 'reward' in saved_task else RewardConfig()
    if a.resume or a.mode=='replay':
        cycle_cfg=CycleConfig(**saved_task['cycle_reward']) if saved_task.get('cycle_reward') else None
    else:
        cycle_cfg=CycleConfig()
    if (a.resume or a.mode=='replay') and saved_task.get('decision_ms',5)!=5:
        raise ValueError('saved decision period is not supported by this collector')
    if a.mode=='replay':
        result=evaluate_policy(model,a.ticks,5,a.target_speed,out/'native' if a.native else None,adhesion=a.adhesion,cycle_config=cycle_cfg)
        atomic_json(out/'EVALUATION.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),evaluation=result))
        print(json.dumps(result),flush=True);return
    global_start=trainer.updates
    cumulative=parent_env_steps(checkpoint,payload) if a.resume else 0
    settings=dict(architecture=architecture,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                  parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=sha256(checkpoint),
                  parent_identity_sha256=payload['identity']['identity_sha256'],
                  target_speed=a.target_speed,seed=saved_task.get('seed','UNKNOWN') if a.resume else a.seed,
                  rng_source='parent checkpoint Python/numpy/torch/CUDA states' if a.resume else 'CLI seed',
                  cli_seed_applied=not a.resume,updates=a.updates,ticks=a.ticks,rollout_ticks=a.rollout_ticks,
                  global_update_start=global_start,cumulative_env_steps_start=cumulative,
                  decision_ms=5,adhesion_enabled=a.adhesion,ppo=asdict(config),reward=reward_cfg.to_dict(),
                  cycle_reward=None if cycle_cfg is None else asdict(cycle_cfg),
                  cycle_shaping_contract='policy-boundary gamma*phi_after-phi_before, terminal phi0; completion measured bonus only',
                  optimizer='exact checkpoint state continuation' if a.resume else 'explicit warmstart reset: new PPO optimizers/counters; model only',
                  finite_horizon=f'{a.ticks*.001:g}s finite task zero bootstrap, mathematical worst-rate absorbing tail on training failure')
    if not math.isclose(config.gamma,math.exp(-reward_cfg.policy_discount_dt_s/reward_cfg.discount_time_constant_s),rel_tol=0,abs_tol=1e-12):
        raise ValueError('PPO gamma mismatches absorbing reward contract')
    core=Core(adhesion=a.adhesion);identity=capture(core,settings,extra_paths=[checkpoint])
    atomic_json(out/'IDENTITY.json',identity)
    initial_ev=evaluate_policy(model,a.ticks,5,a.target_speed,adhesion=a.adhesion,cycle_config=cycle_cfg)
    # Evaluation/setup must not consume the continuation RNG stream.
    if a.resume: load_checkpoint(checkpoint,trainer,payload['identity'])
    save_checkpoint(out/'initial.pt',trainer,identity,extra=dict(update=-1,local_update=-1,global_update=trainer.updates,cumulative_env_steps=cumulative))
    log=dict(status='running',config=settings,updates=[],initial_evaluation=initial_ev)
    atomic_json(out/'TRAINING.json',log);start=time.monotonic()
    try:
        for update in range(a.updates):
            episodes=[];diag=[];actual=0
            while actual<a.rollout_ticks:
                episode,d=collect_episode(core,model,a.target_speed,a.ticks,5,reward_cfg,cycle_cfg)
                episodes.append(episode);diag.append(d);actual+=d['actual_ticks']
            if architecture=='gru238':opt=trainer.update(episodes)
            else:
                buffer=RolloutBuffer()
                for e in episodes:
                    for t in range(len(e['reward'])):
                        buffer.add(**{key:e[key][t] for key in buffer.keys})
                opt=trainer.update(buffer.finish(config.gamma,config.gae_lambda))
            cp=out/'checkpoints'/f'update_{update:04d}.pt'
            cumulative=cumulative+actual if isinstance(cumulative,int) else 'UNKNOWN'
            save_checkpoint(cp,trainer,identity,extra=dict(update=update,local_update=update,global_update=trainer.updates,cumulative_env_steps=cumulative))
            ev=evaluate_policy(model,a.ticks,5,a.target_speed,adhesion=a.adhesion,cycle_config=cycle_cfg)
            row=dict(update=update,local_update=update,global_update=trainer.updates,cumulative_env_steps=cumulative,
                     checkpoint=str(cp),checkpoint_sha256=sha256(cp),episodes=diag,
                     actual_physics_ticks=actual,optimization=opt,evaluation=ev)
            log['updates'].append(row);verify(identity);atomic_json(out/'TRAINING.json',log)
            print(json.dumps(dict(update=update,kl=opt.get('kl'),
                evaluation={key:ev[key] for key in ('forward_signed_native_per_s','com_height_min','nonfoot_force_max_native','slip_ratio_max','all_gates')})),flush=True)
        save_checkpoint(out/'final.pt',trainer,identity,extra=dict(update=a.updates-1,local_update=a.updates-1,global_update=trainer.updates,cumulative_env_steps=cumulative))
        log.update(status='completed',wall_s=time.monotonic()-start)
    except BaseException as exc:
        log.update(status='failed',error=repr(exc),wall_s=time.monotonic()-start)
        atomic_json(out/'TRAINING.json',log);raise
    verify(identity);atomic_json(out/'TRAINING.json',log)


if __name__=='__main__':main()
