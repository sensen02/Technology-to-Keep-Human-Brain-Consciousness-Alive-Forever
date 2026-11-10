"""Causal policy inference and native-recordable derived-world evaluation."""
from pathlib import Path
import time
import numpy as np
import torch
from physics import Core, NativeRecorder, summarise
from run_baseline import record_row


def actor_observation(core,model,previous,target):
    return core.obs() if model.architecture=='ff495' else core.obs238(previous,target)


def critic_observation(core,model,observation,remaining):
    if model.critic_obs_dim==len(observation):
        return observation
    x=np.r_[observation,core.critic_extras(),remaining]
    if len(x)!=model.critic_obs_dim:
        raise ValueError('critic observation dimension mismatch')
    return x


def initial_state(model):
    return model.actor.initial_state(1) if model.architecture=='gru238' else None


@torch.no_grad()
def evaluate_policy(model,ticks=2000,decision_ms=5,target_speed=.10,native_dir=None,adhesion=True):
    start=time.monotonic()
    recorder=None if native_dir is None else NativeRecorder(native_dir)
    core=Core(recorder=recorder,adhesion=adhesion)
    previous=np.zeros(42)
    state=initial_state(model)
    rec=[]
    try:
        for k in range(0,ticks,decision_ms):
            obs=actor_observation(core,model,previous,target_speed)
            cv=critic_observation(core,model,obs,(ticks-k)*.001)
            sample=model.act(obs,cv,deterministic=True,state=state)
            state=sample['state']
            action=sample['action'][0].cpu().numpy()
            for _ in range(min(decision_ms,ticks-k)):
                m=core.tick(action)
                rec.append(record_row(core,m))
            previous=action.copy()
    finally:
        native=None if recorder is None else recorder.close()
    result=summarise(rec,ticks*80,initial_heading=core.initial_heading,native_summary=native)
    result.update(wall_s=time.monotonic()-start,decision_ms=decision_ms,
                  target_speed=target_speed,adhesion_enabled=bool(adhesion),evaluation_scope='derived engineering; no continuous prewarm handover; original acceptance unverified')
    return result
