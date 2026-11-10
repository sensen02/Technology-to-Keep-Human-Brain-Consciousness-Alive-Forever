"""Integration-specific contracts before long physical curriculum work."""
from pathlib import Path
import hashlib
import json
import math
import tempfile
import numpy as np
import torch
from models import ActorCritic
from curriculum import RewardConfig,absorbing_tail
from train_curriculum import collect_episode
from physics import Core
from identity import atomic_json,capture,verify,verify_runtime,HERE


def run():
    cfg=RewardConfig();gamma=math.exp(-.005)
    # Failure at every position within a decision and close to horizon. The real
    # interval reward at failure is retained; only future min rates are filled.
    errors=[]
    for horizon in [10,15,2000]:
        for failure in [1,2,3,4,5,horizon-1,horizon]:
            if failure>horizon:continue
            block=((failure-1)//5)*5
            boundary=min(block+5,horizon)
            remainder=(boundary-failure)*.001
            after=(horizon-boundary)*.001
            actual=cfg.min_rate*remainder+gamma*absorbing_tail(after,cfg)
            expected=cfg.min_rate*remainder
            future=after;discount=gamma
            while future>1e-12:
                length=min(.005,future)
                expected+=discount*cfg.min_rate*length
                discount*=gamma;future-=length
            errors.append(abs(actual-expected))
    assert max(errors)<1e-11
    core=Core()
    identity=capture(core,dict(stage='integration checks'))
    verify(identity);verify_runtime(identity,Core())
    tampered=json.loads(json.dumps(identity));tampered['config']['stage']='tampered'
    try:verify(tampered)
    except RuntimeError:pass
    else:raise AssertionError('tampered manifest accepted')
    for architecture in ['ff238','gru238']:
        model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64)
        episode,diag=collect_episode(core,model,.1,10,5,cfg)
        assert episode['obs'].shape==(2,238) and episode['critic_obs'].shape==(2,255)
        assert episode['terminated'][-1] and diag['actual_ticks']==10
        assert episode['starts_at_reset'] is True
        assert np.isfinite(episode['reward']).all()
        try:collect_episode(core,model,.1,10,1,cfg)
        except ValueError:pass
        else:raise AssertionError('timing mismatch accepted')
    paths=[HERE/name for name in ['identity.py','policy_io.py','distill_policy.py','train_curriculum.py','check_integration.py']]
    return dict(all_passed=True,status='PASS',tail_max_abs_error=max(errors),
                tests=['failure_at_first_middle_last_policy_substep','failure_near_horizon',
                       'manifest_digest_tamper_reject','runtime_compiled_parameter_check',
                       'FF_GRU_actual_short_physical_collection','reward_timing_mismatch_reject'],
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                limitation='10ms integration smoke only; no walking conclusion')

if __name__=='__main__':
    torch.set_num_threads(1)
    result=run();atomic_json(HERE/'INTEGRATION_CHECKS.json',result);print(json.dumps(result),flush=True)
