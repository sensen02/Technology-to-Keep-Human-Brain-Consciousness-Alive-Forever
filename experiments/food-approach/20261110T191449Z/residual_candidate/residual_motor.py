"""Isolated offline neural residual candidate. No physics or teacher calls.

The legacy baseline implementation is loaded directly from its immutable path;
conditional_motor.py and the active training branch are neither imported nor edited.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import math
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
CHECKPOINT = BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt'
BASE_SHA256 = '246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_baseline_actor():
    if sha256(CHECKPOINT) != BASE_SHA256:
        raise ValueError('baseline checkpoint drift')
    spec = importlib.util.spec_from_file_location('food_residual_frozen_baseline', BASE/'models.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    mi = payload['model_identity']
    if mi['actor_input_transform'] != legacy.SlowNeuralFeedbackGRU.TRANSFORM:
        raise ValueError('unexpected baseline actor')
    actor = legacy.SlowNeuralFeedbackGRU(mi['gain_init_H']).double()
    actor.load_state_dict({k[6:]:v for k,v in payload['state']['model'].items() if k.startswith('actor.')}, strict=True)
    return actor


class ResidualMotor(nn.Module):
    VERSION = 'frozen_gru128_masked_residual_gru64_command_norm_gate_v1'
    ALPHA = -math.expm1(-.005/.1)
    RESIDUAL_BOUND = .5

    def __init__(self, baseline):
        super().__init__()
        self.base = deepcopy(baseline)
        self.base.requires_grad_(False)
        # Select the 112 permitted channels instead of allocating dead masked weights.
        self.encoder = nn.Linear(112+128+2,64,dtype=torch.float64)
        self.gru = nn.GRUCell(64,64,dtype=torch.float64)
        self.head = nn.Linear(64,42,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.mul_(.01)
            self.head.bias.zero_()

    def initial_state(self, batch_size):
        """Only at physical episode reset, NOT at teacher withdrawal/command change."""
        return (self.base.initial_state(batch_size), self.head.weight.new_zeros((batch_size,64)))

    def adapter_parameters(self):
        return [p for name,p in self.named_parameters() if not name.startswith('base.')]

    def step(self, obs, command, state):
        """Return pre-tanh mean42, explicit (base_hidden128,residual_hidden64)."""
        if obs.ndim != 2 or obs.shape[-1] != 238 or command.shape != (obs.shape[0],2):
            raise ValueError('requires obs(B,238) command(B,2) only')
        if not isinstance(state,(tuple,list)) or len(state)!=2 or state[0].shape!=(obs.shape[0],128) or state[1].shape!=(obs.shape[0],64):
            raise ValueError('requires explicit state tuple(B,128),(B,64)')
        if not torch.isfinite(obs).all() or not torch.isfinite(command).all() or (command.abs()>1).any():
            raise ValueError('finite obs and command within[-1,1] required')
        for s in state:
            if not torch.isfinite(s).all():
                raise ValueError('nonfinite neural state')
        # Execute exact legacy operations; no adapter output feeds baseline hidden.
        mean, bh = self.base.step(obs,state[0])
        features = torch.cat((obs[:,self.base.physical_mask.bool()],bh,command),dim=-1)
        candidate = self.gru(torch.tanh(self.encoder(features)),state[1])
        rh = (1-self.ALPHA)*state[1]+self.ALPHA*candidate
        gate = torch.tanh(command.square().sum(dim=-1,keepdim=True))
        correction = gate*self.RESIDUAL_BOUND*torch.tanh(self.head(rh))
        return mean+correction,(bh,rh)

    def sequence(self, obs, command, state=None):
        """Return (means[T,B,42], (base_states[T,B,128], residual_states[T,B,64])).

        To resume, pass (base_states[-1],residual_states[-1]); no implicit cache.
        Caller excludes priming labels from turn loss if appropriate.
        """
        if obs.ndim!=3 or obs.shape[-1]!=238 or command.shape!=obs.shape[:2]+(2,) or len(obs)==0:
            raise ValueError('requires nonempty obs(T,B,238), command(T,B,2)')
        state=self.initial_state(obs.shape[1]) if state is None else state
        means,bhs,rhs=[],[],[]
        for x,u in zip(obs,command):
            mean,state=self.step(x,u,state)
            means.append(mean);bhs.append(state[0]);rhs.append(state[1])
        return torch.stack(means),(torch.stack(bhs),torch.stack(rhs))

    def identity(self):
        return dict(version=self.VERSION,baseline_checkpoint=str(CHECKPOINT),baseline_sha256=BASE_SHA256,
                    baseline_model_source=str(BASE/'models.py'),baseline_model_source_sha256=sha256(BASE/'models.py'),
                    inputs=238,command_dim=2,command_order=['forward_modulation','signed_turn_modulation'],
                    state_dims=[128,64],residual_bound=self.RESIDUAL_BOUND,alpha=self.ALPHA,
                    neutral=[0.,0.],gate='tanh(sum(command**2))',base_frozen=True,
                    base_identity=self.base.transform_identity())


def save_candidate(path,model,optimizer=None,config=None,runtime_state=None):
    path=Path(path).resolve()
    if not path.is_relative_to(HERE) or path.exists():
        raise ValueError('new output strictly within residual_candidate required')
    torch.save(dict(identity=model.identity(),state_dict=model.state_dict(),
                    optimizer=None if optimizer is None else optimizer.state_dict(),config=config,
                    runtime_state=runtime_state,
                    source_hashes={str(HERE/n):sha256(HERE/n) for n in ('residual_motor.py','PREREG.json')}),path)
    return sha256(path)


def load_candidate(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=ResidualMotor(load_baseline_actor())
    if payload['identity']!=model.identity():
        raise ValueError('candidate identity mismatch')
    for p,h in payload['source_hashes'].items():
        if sha256(p)!=h:
            raise ValueError('candidate source drift')
    model.load_state_dict(payload['state_dict'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite saved weights')
    verified=load_baseline_actor()
    if not all(torch.equal(v,verified.state_dict()[k]) for k,v in model.base.state_dict().items()):
        raise ValueError('frozen baseline weights differ from verified checkpoint')
    return model,payload
