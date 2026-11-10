"""Exact actor input ablation/checkpoint contract, not behavior success."""
from pathlib import Path
from copy import deepcopy
import tempfile
import torch
import numpy as np
from models import ActorCritic
from ppo import PPOTrainer,PPOConfig,save_checkpoint,load_checkpoint
from recurrent import RecurrentPPOTrainer,RecurrentPPOConfig
from identity import HERE,atomic_json,sha256


def main():
    torch.set_num_threads(1);torch.manual_seed(20261110)
    rows=[]
    for architecture in ['ff238','gru238']:
        m=ActorCritic(architecture,critic_obs_dim=255,dtype=torch.float64)
        obs=torch.randn(5,1,238,dtype=torch.float64,requires_grad=True)
        changed=obs.detach().clone();changed[:,:,195:237]+=100.
        def mean(x):
            return m.actor(x) if architecture=='ff238' else m.actor.sequence(x,m.actor.initial_state(1))[0]
        a=mean(obs);b=mean(changed)
        assert torch.equal(a,b)
        grad=torch.autograd.grad(a.square().sum(),obs)[0]
        assert torch.count_nonzero(grad[:,:,195:237])==0
        assert torch.count_nonzero(grad[:,:,:195])>0
        raw=torch.randn(1,255,dtype=torch.float64,requires_grad=True)
        cg=torch.autograd.grad(m.value(raw).sum(),raw)[0]
        assert torch.count_nonzero(cg[:,195:237])>0
        cls=RecurrentPPOTrainer if architecture=='gru238' else PPOTrainer
        cfg=RecurrentPPOConfig() if architecture=='gru238' else PPOConfig()
        trainer=cls(m,cfg)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'masked.pt';identity=dict(test='mask')
            save_checkpoint(path,trainer,identity)
            load_checkpoint(path,trainer,identity)
            bad=torch.load(path,weights_only=False);bad['model_identity'].pop('actor_input_transform')
            torch.save(bad,Path(d)/'bad.pt')
            try:load_checkpoint(Path(d)/'bad.pt',trainer,identity)
            except ValueError:pass
            else:raise AssertionError('raw contract accepted')
        rows.append(dict(architecture=architecture,masked_invariant=True,masked_gradient_zero=True,
                         physical_gradient_nonzero=True,critic_unmasked=True,checkpoint_contract_enforced=True))
    paths=[HERE/name for name in ['models.py','check_mask.py']]
    result=dict(all_passed=True,checks=rows,source_hashes={str(p):sha256(p) for p in paths})
    atomic_json(HERE/'MASK_CHECKS.json',result);print(result)

if __name__=='__main__':main()
