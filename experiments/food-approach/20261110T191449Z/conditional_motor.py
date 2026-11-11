"""New neural command adapter; immutable verified locomotor lineage stays external.

No physics stepping on import. Engineering replacement ANN, not connectome.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
CHECKPOINT = BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt'
BASE_SHA256 = '246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809'
sys.path.insert(0, str(BASE))
import models as legacy_models
if Path(legacy_models.__file__).resolve() != BASE/'models.py':
    raise RuntimeError('wrong legacy models module imported')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_baseline_actor():
    """Training constructor. Runtime physics identity checked separately by runner."""
    if sha256(CHECKPOINT) != BASE_SHA256:
        raise ValueError('verified parent checkpoint changed')
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    mi = payload['model_identity']
    if mi['actor_input_transform'] != legacy_models.SlowNeuralFeedbackGRU.TRANSFORM:
        raise ValueError('wrong baseline architecture')
    actor = legacy_models.SlowNeuralFeedbackGRU(mi['gain_init_H']).double()
    weights = {k[len('actor.'):]: v for k,v in payload['state']['model'].items() if k.startswith('actor.')}
    actor.load_state_dict(weights, strict=True)
    return actor


class ConditionalMotor(nn.Module):
    VERSION = 'slow_gru128_plus_zero_initial_command_encoder2_v1'
    def __init__(self, baseline):
        super().__init__()
        self.base = deepcopy(baseline)
        self.command_encoder = nn.Linear(2,128,bias=False,dtype=torch.float64)
        nn.init.zeros_(self.command_encoder.weight)

    def initial_state(self,batch_size):
        return self.base.initial_state(batch_size)

    def step(self, obs, command, state):
        if obs.shape[-1] != 238 or command.shape != obs.shape[:-1]+(2,) or state.shape != obs.shape[:-1]+(128,):
            raise ValueError('requires matching batch dimensions obs238/command2/state128')
        b = self.base
        drive = b.encoder(obs*b.physical_mask) + self.command_encoder(command)
        candidate = b.gru(torch.tanh(drive),state)
        new_state = (1-b.ALPHA)*state+b.ALPHA*candidate
        mean = b.head(torch.tanh(b.hidden(new_state)))-b.gain()*obs[...,126:168]
        return mean,new_state

    def sequence(self, obs, command, state=None):
        if obs.ndim != 3 or command.shape != obs.shape[:2]+(2,):
            raise ValueError('requires (T,B,238) observations and (T,B,2) commands')
        state = self.initial_state(obs.shape[1]) if state is None else state
        means,states=[],[]
        for x,u in zip(obs,command):
            mean,state = self.step(x,u,state)
            means.append(mean);states.append(state)
        return torch.stack(means),torch.stack(states)

    def set_training_stage(self, stage):
        if stage not in ('adapter','finetune'):
            raise ValueError('stage must be adapter or finetune')
        for p in self.base.parameters():
            p.requires_grad_(stage=='finetune')
        self.command_encoder.weight.requires_grad_(True)

    def identity(self):
        return dict(version=self.VERSION,baseline_checkpoint=str(CHECKPOINT),baseline_sha256=BASE_SHA256,
                    command_order=['forward_modulation','signed_turn_modulation'],neutral=[0.,0.],
                    architecture='238 physical -> encoder128 + biasfree neural command2->128 -> tanh -> slowGRU128 -> decoder42',
                    actor_inputs_exclude=['food_xy','target_distance','target_bearing','time','phase','teacher'],
                    base_identity=self.base.transform_identity())


class OdorBehavior(nn.Module):
    """Recurrent neural behavior module; inputs are two local receptor readings only."""
    VERSION='bilateral_receptors2_slow_gru32_command2_v1'
    ALPHA=-np.expm1(-.005/.1)
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(2,32,dtype=torch.float64)
        self.gru=nn.GRUCell(32,32,dtype=torch.float64)
        self.head=nn.Linear(32,2,dtype=torch.float64)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def initial_state(self,batch_size):
        return self.head.weight.new_zeros((batch_size,32))

    def step(self,receptors,state):
        if receptors.shape[-1] != 2 or state.shape != receptors.shape[:-1]+(32,):
            raise ValueError('behavior accepts only local receptors2 and recurrent state32')
        candidate=self.gru(torch.tanh(self.encoder(receptors)),state)
        state=(1-self.ALPHA)*state+self.ALPHA*candidate
        return torch.tanh(self.head(state)),state


def save_motor(path, model, optimizer, config, curve):
    """Caller chooses a new output path, never overwrites verified parent."""
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload=dict(format_version=1,identity=model.identity(),state_dict=model.state_dict(),
                 optimizer=None if optimizer is None else optimizer.state_dict(),config=config,curve=curve,
                 rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state()),
                 source_sha256={str(HERE/'conditional_motor.py'):sha256(HERE/'conditional_motor.py')})
    torch.save(payload,path)
    return sha256(path)


def load_motor(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=ConditionalMotor(load_baseline_actor())
    if payload['format_version'] != 1 or payload['identity'] != model.identity():
        raise ValueError('conditional motor identity mismatch')
    for p,h in payload['source_sha256'].items():
        if sha256(p) != h:
            raise ValueError('conditional motor source drift')
    model.load_state_dict(payload['state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite model checkpoint')
    return model,payload
