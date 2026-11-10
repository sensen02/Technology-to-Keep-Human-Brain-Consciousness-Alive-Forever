"""Declared stable zero-mean policy initializations; actuator hold is not neural skill."""
import math
from pathlib import Path
import json
import torch
import numpy as np
from models import ActorCritic
from ppo import PPOConfig,PPOTrainer,save_checkpoint,load_checkpoint
from recurrent import RecurrentPPOConfig,RecurrentPPOTrainer
from physics import Core
from run_baseline import seed_all
from identity import HERE,capture,verify,verify_runtime,atomic_json,sha256


def main():
    for architecture in ['ff238','gru238']:
        seed_all(20261114)
        out=HERE/('zero_'+architecture)
        out.mkdir(exist_ok=False)
        model=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64,initial_logstd=math.log(.2))
        head=model.actor.head if architecture=='gru238' else model.actor.layers[-1]
        with torch.no_grad():head.weight.zero_();head.bias.zero_()
        core=Core(adhesion=False)
        config=dict(stage='zero mean actuator-hold initialization; not learned gait',architecture=architecture,
                    seed=20261114,target_speed=.10,ticks=2000,decision_ms=5,adhesion_enabled=False,
                    latent_std=.20,head='exact zero; nonhead default Torch initialization')
        identity=capture(core,config)
        atomic_json(out/'IDENTITY.json',identity)
        cfg=RecurrentPPOConfig(episodes_per_minibatch=1) if architecture=='gru238' else PPOConfig()
        trainer=RecurrentPPOTrainer(model,cfg) if architecture=='gru238' else PPOTrainer(model,cfg)
        obs=core.obs238(np.zeros(42),.1)
        tensor=torch.as_tensor(obs[None],dtype=torch.float64)
        with torch.no_grad():
            if architecture=='gru238':mean,_=model.actor.step(tensor,model.actor.initial_state(1))
            else:mean=model.actor(tensor)
        assert torch.count_nonzero(mean)==0
        save_checkpoint(out/'initial.pt',trainer,identity,extra=dict(stage='untrained zero-head',actor_mean_exact_zero=True))
        load_checkpoint(out/'initial.pt',trainer,identity)
        verify_runtime(identity,Core(adhesion=False));verify(identity)
        atomic_json(out/'INITIAL_CHECK.json',dict(all_passed=True,mean_exact_zero=True,std=model.logstd().detach().exp().tolist(),
                    checkpoint_sha256=sha256(out/'initial.pt'),note='safe perturbation verified by prior probe, not multi-seed learning claim'))
        print(architecture,'zero head and reload verified',str(out/'initial.pt'),flush=True)

if __name__=='__main__':main()
