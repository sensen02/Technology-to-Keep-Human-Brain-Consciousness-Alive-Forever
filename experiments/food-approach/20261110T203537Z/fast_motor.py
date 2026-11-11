"""Frozen verified recurrent base + instantaneous neural residual. No physics on import."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'20261110T191449Z'
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
CHECKPOINT=BASE/'BC_GRU_ROLE_3000/checkpoint_002500.pt'
BASE_SHA='246f9fab7d21e2a1332a745fafd4a343b80ad5a70d2cafeeb900e876f7933809'


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load_base():
    if sha256(CHECKPOINT)!=BASE_SHA:raise ValueError('baseline checkpoint drift')
    spec=importlib.util.spec_from_file_location('_food_fast_frozen_base',BASE/'models.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if p['model_identity']['actor_input_transform']!=m.SlowNeuralFeedbackGRU.TRANSFORM:raise ValueError('baseline architecture mismatch')
    actor=m.SlowNeuralFeedbackGRU(p['model_identity']['gain_init_H']).double()
    actor.load_state_dict({k[6:]:v for k,v in p['state']['model'].items() if k.startswith('actor.')},strict=True)
    return actor


def verify_hashes(values):
    for filename,digest in values.items():
        if not Path(filename).is_absolute() or sha256(filename)!=digest:raise ValueError('artifact drift: '+filename)


class FastMotor(nn.Module):
    VERSION='frozen_gru128_fast_neural_residual242_128_128_42_v1'
    def __init__(self,base):
        super().__init__();self.base=deepcopy(base);self.base.requires_grad_(False)
        self.encoder=nn.Linear(242,128,dtype=torch.float64)
        self.hidden=nn.Linear(128,128,dtype=torch.float64)
        self.head=nn.Linear(128,42,dtype=torch.float64)
        self.register_buffer('command_scale',torch.tensor([.15,1.],dtype=torch.float64))
        with torch.no_grad():self.head.weight.mul_(.01);self.head.bias.zero_()

    def initial_state(self,batch_size):return self.base.initial_state(batch_size)

    def adapter_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('base.')]

    def features(self,obs,base_hidden,command):
        normalized=command/self.command_scale
        return torch.cat((obs[...,self.base.physical_mask.bool()],base_hidden,normalized),dim=-1),torch.tanh(normalized.square().sum(-1,keepdim=True))

    def correction(self,features,gate):
        return gate*torch.tanh(self.head(torch.tanh(self.hidden(torch.tanh(self.encoder(features))))))

    def step(self,obs,command,state):
        if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2) or state.shape!=(len(obs),128):raise ValueError('obs238/command2/state128 required')
        if not torch.isfinite(obs).all() or not torch.isfinite(command).all() or not torch.isfinite(state).all():raise ValueError('nonfinite neural input/state')
        if (command[:,0]<-.15-1e-12).any() or (command[:,0]>1e-12).any() or (command[:,1].abs()>1+1e-12).any():raise ValueError('unqualified command outside forward[-.15,0]/turn[-1,1]')
        mean,state=self.base.step(obs,state)
        features,gate=self.features(obs,state,command)
        return mean+self.correction(features,gate),state

    def sequence(self,obs,command,state=None):
        state=self.initial_state(obs.shape[1]) if state is None else state
        means=[];states=[]
        for x,u in zip(obs,command):
            mean,state=self.step(x,u,state);means.append(mean);states.append(state)
        return torch.stack(means),torch.stack(states)

    def identity(self):
        return dict(version=self.VERSION,base_checkpoint=str(CHECKPOINT),base_sha256=BASE_SHA,
                    base_source=str(BASE/'models.py'),base_source_sha256=sha256(BASE/'models.py'),
                    command_order=['forward_scale_minus1','signed_turn'],command_normalization=[.15,1.],
                    command_axis_bounds=[[-.15,0.],[-1.,1.]],state_dims=[128],
                    gate='tanh(sum((command/scale)**2))',residual_bound=1.,base_frozen=True,
                    residual_inputs='112permittedphysical + actualcurrentbasehidden128 + normalizedcommand2',
                    actor_excludes=['clock','phase','food_position','target_bearing','distance','teacher'],base_identity=self.base.transform_identity())


def save(path,model,opt,config,curve,rng):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save(dict(format_version=1,identity=model.identity(),model=model.state_dict(),optimizer=opt.state_dict(),config=config,curve=curve,rng=rng),path)
    return sha256(path)


def load(path,expected_sha=None):
    if expected_sha and sha256(path)!=expected_sha:raise ValueError('selected model digest mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    verify_hashes(p['config']['sources']);verify_hashes(p['config']['data_hashes'])
    model=FastMotor(load_base())
    if p['format_version']!=1 or p['identity']!=model.identity():raise ValueError('fast motor identity mismatch')
    model.load_state_dict(p['model'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('nonfinite saved weights')
    if not all(torch.equal(v,load_base_state) for v,load_base_state in zip(model.base.state_dict().values(),load_base().state_dict().values())):raise ValueError('frozen baseline weights changed')
    return model.eval(),p
