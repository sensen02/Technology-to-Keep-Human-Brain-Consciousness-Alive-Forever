"""Short physical resume integration; only optimizer updates use synthetic rewards."""
from copy import deepcopy
from dataclasses import asdict
import random
from pathlib import Path
import tempfile
import numpy as np
import torch
from train_curriculum import (collect_episode,resolve_task_settings,build_trainer,load_source,parent_env_steps)
from run_baseline import seed_all
from models import ActorCritic
from ppo import PPOConfig,PPOTrainer,RolloutBuffer,save_checkpoint
from recurrent import RecurrentPPOConfig,RecurrentPPOTrainer
from curriculum import RewardConfig
from physics import Core
from identity import capture,atomic_json,sha256,verify

HERE=Path(__file__).resolve().parent


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def synthetic_update(trainer,episode):
    e=deepcopy(episode)
    e['reward']=np.linspace(-.07,.13,len(e['reward']))
    if trainer.model.architecture=='gru238': return trainer.update([e,e])
    buffer=RolloutBuffer()
    for i in range(len(e['reward'])): buffer.add(**{k:e[k][i] for k in buffer.keys})
    c=trainer.config
    return trainer.update(buffer.finish(c.gamma,c.gae_lambda))


def run():
    torch.set_num_threads(1); checks=[]
    with tempfile.TemporaryDirectory(prefix='resume-check-') as temporary:
        for architecture,adhesion in [('ff238',False),('gru238',True)]:
            seed_all(614)
            model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64)
            common=dict(gamma=np.exp(-.005),gae_lambda=np.exp(-.025),epochs=1,minibatch=2,target_kl=.03,actor_lr=1e-5)
            config=RecurrentPPOConfig(**common,episodes_per_minibatch=2,replay_tolerance=1e-7) if architecture=='gru238' else PPOConfig(**common)
            trainer=(RecurrentPPOTrainer if architecture=='gru238' else PPOTrainer)(model,config)
            core=Core(adhesion=adhesion); reward=RewardConfig()
            first,diag=collect_episode(core,model,.073,10,5,reward)
            assert diag['actual_ticks']==10
            synthetic_update(trainer,first)
            assert trainer.updates==1
            settings=dict(target_speed=.073,ticks=10,adhesion_enabled='true' if adhesion else 'false',
                          decision_ms=5,seed=614,reward=reward.to_dict())
            identity=capture(core,settings)
            path=Path(temporary)/(architecture+'.pt')
            save_checkpoint(path,trainer,identity,extra=dict(update=0,cumulative_env_steps=10))
            payload=torch.load(path,weights_only=False)
            task=resolve_task_settings(payload,'train',resume=True)
            assert task==dict(target_speed=.073,ticks=10,adhesion=adhesion)
            for override in (dict(target_speed=0.),dict(ticks=15),dict(adhesion=not adhesion)):
                try: resolve_task_settings(payload,'train',resume=True,**override)
                except ValueError: pass
                else: raise AssertionError('incompatible override accepted')
            assert resolve_task_settings(payload,'train')['target_speed']==0.
            fresh=build_trainer(payload,restore_config=False)
            assert fresh.updates==0 and not fresh.actor_optimizer.state
            assert equal(fresh.model.state_dict(),trainer.model.state_dict())
            # Model-only constructor above consumes torch RNG; restore the saved stream before reference.
            from ppo import load_checkpoint
            load_checkpoint(path,trainer,identity)
            before=trainer.snapshot()
            python_next=random.random(); numpy_next=np.random.random(); torch_next=torch.rand(3)
            expected,expected_diag=collect_episode(core,model,.073,10,5,reward)
            synthetic_update(trainer,expected); expected_state=trainer.snapshot()
            seed_all(9999)
            restored_payload,resumed=load_source(path,resume=True)
            assert equal(resumed.snapshot(),before)
            assert asdict(resumed.config)==payload['config']
            assert resumed.updates==1 and parent_env_steps(path,payload)==10
            assert random.random()==python_next and np.random.random()==numpy_next and torch.equal(torch.rand(3),torch_next)
            resumed_core=Core(adhesion=task['adhesion'])
            actual,actual_diag=collect_episode(resumed_core,resumed.model,task['target_speed'],task['ticks'],5,reward)
            assert equal(actual,expected) and equal(actual_diag,expected_diag)
            assert np.array_equal(np.tanh(actual['latent']),np.tanh(expected['latent']))
            synthetic_update(resumed,actual)
            assert equal(resumed.snapshot(),expected_state)
            # New-run identity is captured from current paths; parent is preserved and linked.
            new_identity=capture(resumed_core,dict(parent_checkpoint=str(path),parent_checkpoint_sha256=sha256(path),
                parent_identity_sha256=identity['identity_sha256'],global_update_start=1,
                cumulative_env_steps_start=10,rng_source='parent checkpoint',**settings),extra_paths=[path])
            verify(new_identity)
            missing=deepcopy(payload);missing['extra']={}
            assert parent_env_steps(Path(temporary)/'absent.pt',missing)=='UNKNOWN'
            checks.append(dict(architecture=architecture,all_passed=True,physical_ticks_per_episode=10,
                saved_task=task,nondefault_config_exact=True,optimizer_and_model_bitwise_equal=True,
                python_numpy_torch_rng_equal=True,next_physical_episode_bitwise_equal=True,
                next_synthetic_optimizer_update_bitwise_equal=True,override_rejections=3,
                source_runtime_verified=True,new_identity_capture_verified=True))
    report=dict(all_passed=True,checks=checks,scope='10ms actual physical episodes; synthetic-reward optimizer integration only; no learning pilots',
                source_hashes={str(HERE/name):sha256(HERE/name) for name in ('train_curriculum.py','check_resume.py','ppo.py','recurrent.py','identity.py','PREREG.json')})
    atomic_json(HERE/'RESUME_CHECKS.json',report)
    print('RESUME_CHECKS all_passed',report['all_passed'],flush=True)


if __name__=='__main__': run()
