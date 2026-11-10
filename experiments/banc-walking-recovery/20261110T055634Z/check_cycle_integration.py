"""Cycle shaping is a measurement/reward observer, never an action source."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from models import ActorCritic
from physics import Core
from curriculum import RewardConfig
from cycle_reward import CycleConfig
from train_curriculum import collect_episode
from policy_io import evaluate_policy
from identity import HERE,atomic_json


def main():
    torch.set_num_threads(1)
    core=Core(adhesion=False)
    results=[]
    for architecture in ('ff238','gru238'):
        torch.manual_seed(20261116)
        model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64,initial_logstd=np.log(.2))
        state=torch.get_rng_state().clone()
        basic,d0=collect_episode(core,model,.1,50,5,RewardConfig())
        final0=core.d.qpos.copy()
        torch.set_rng_state(state)
        shaped,d1=collect_episode(core,model,.1,50,5,RewardConfig(),CycleConfig())
        assert np.array_equal(core.d.qpos,final0)
        for key in ('obs','critic_obs','latent','log_prob','mean','logstd','terminated','next_value'):
            assert np.array_equal(basic[key],shaped[key]),key
        gamma=np.exp(-.005)
        difference=np.asarray(shaped['reward'])-np.asarray(basic['reward'])
        discounted=float(np.sum(gamma**np.arange(len(difference))*difference))
        assert sum(d1['cycle_counts'])==0
        assert abs(discounted)<1e-12,discounted
        assert d1['cycle_completion_bonus']==0.
        result=evaluate_policy(model,50,5,.1,adhesion=False,cycle_config=CycleConfig())
        assert sum(result['cycle_evidence']['per_leg_complete'].values())==0
        results.append(dict(architecture=architecture,physical_actions_unchanged=True,
                            discounted_shape_no_complete=discounted,complete_events=0))
    paths=[HERE/name for name in ('train_curriculum.py','policy_io.py','cycle_reward.py','check_cycle_integration.py')]
    report=dict(all_passed=True,status='PASS',results=results,
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                limitation='short observation/reward integration check, no behavior success claim')
    atomic_json(HERE/'CYCLE_INTEGRATION_CHECKS.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
