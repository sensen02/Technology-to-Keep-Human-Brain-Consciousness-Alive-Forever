"""Parent-only Torch backend; ONLYCODE preparation must never import this file.
Independent body task commands, not a brain/neck deployment or behavioral proof.
"""
import math
from typing import List, Tuple, Optional
import torch
from torch import nn

FORMAT = 'INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'
OLD_SHA = '789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
STATE_DIMS = (128, 64, 64, 96)
ALPHA = -math.expm1(-.005/.1)
BOUND = 6.
ADAPTER_SHAPES = {
    'encoder.weight': (96, 370), 'encoder.bias': (96,),
    'gru.weight_ih': (288, 96), 'gru.weight_hh': (288, 96),
    'gru.bias_ih': (288,), 'gru.bias_hh': (288,),
    'head.weight': (42, 96), 'head.bias': (42,),
}


def state_digest(state):
    """Stable dtype/shape/name/content digest; CPU numeric work only after release."""
    import hashlib
    import json
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        description = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(description).to_bytes(8, 'little')); digest.update(description)
        raw = value.numpy().tobytes(order='C')
        digest.update(len(raw).to_bytes(8, 'little')); digest.update(raw)
    return digest.hexdigest()


class ResidualScan(nn.Module):
    """Scriptable cache path, shared8 parameters; no frozen motor inside scan.
    forward(features[T,B,370], h96[B,96]) -> residual[T,B,42], hidden[T,B,96].
    Features/cache are constructed only by actual frozen oldmotor, never teacher.
    """
    def __init__(self, encoder, gru, head):
        super().__init__(); self.encoder = encoder; self.gru = gru; self.head = head
        self.alpha = ALPHA; self.bound = BOUND

    def forward(self, features: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = features.new_zeros((features.size(1), 96))
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        states = torch.jit.annotate(List[torch.Tensor], [])
        for k in range(features.size(0)):
            candidate = self.gru(torch.tanh(self.encoder(features[k])), h)
            h = (1.-self.alpha)*h + self.alpha*candidate
            outputs.append(self.bound*torch.tanh(self.head(h))); states.append(h)
        return torch.stack(outputs), torch.stack(states)


class IndependentMotor(nn.Module):
    def __init__(self, oldmotor):
        super().__init__()
        identity = oldmotor.identity()
        if identity.get('format_version') != 'COMBINED_TURN_V1' or identity.get('state_dims') != [128,64,64]:
            raise ValueError('Strict original CombinedTurning256 required')
        if any(p.requires_grad for p in oldmotor.parameters()):
            raise ValueError('Oldmotor must already be frozen, never silently freeze trainable source')
        if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in oldmotor.parameters()):
            raise ValueError('Original CPU float64 frozen motor required')
        mask = oldmotor.base.physical_mask
        if tuple(mask.shape) != (238,) or int(mask.bool().sum()) != 112:
            raise ValueError('Exact old238 mask with112 permitted physical channels required')
        self.old = oldmotor
        self.encoder = nn.Linear(370,96,dtype=torch.float64)
        self.gru = nn.GRUCell(96,96,dtype=torch.float64)
        self.head = nn.Linear(96,42,dtype=torch.float64)
        with torch.no_grad(): self.head.weight.zero_(); self.head.bias.zero_()
        self.frozen_old_state_digest = state_digest(self.old.state_dict())
        self.frozen_old_identity = identity
        self.validate_architecture()

    def adapter_named_parameters(self):
        return [(name,p) for name,p in self.named_parameters() if not name.startswith('old.')]

    def adapter_parameters(self):
        return [p for _,p in self.adapter_named_parameters()]

    def validate_architecture(self):
        pairs = self.adapter_named_parameters()
        if [name for name,_ in pairs] != list(ADAPTER_SHAPES):
            raise ValueError('Exactly ordered encoder2/GRU4/head2 trainable tensors required')
        if any(tuple(p.shape) != ADAPTER_SHAPES[name] or p.dtype != torch.float64 or p.device.type != 'cpu' or not p.requires_grad for name,p in pairs):
            raise ValueError('Adapter8 shapes/CPU/float64/trainability mismatch')
        if any(p.requires_grad for p in self.old.parameters()): raise ValueError('Frozen oldweights became trainable')
        return True

    def verify_old(self):
        if state_digest(self.old.state_dict()) != self.frozen_old_state_digest or self.old.identity() != self.frozen_old_identity:
            raise ValueError('Original frozen oldweights/identity changed')
        if any(p.requires_grad or p.grad is not None for p in self.old.parameters()):
            raise ValueError('Oldweights must stay frozen with no gradients')
        return True

    def initial_state(self, batch_size):
        if type(batch_size) is not int or batch_size <= 0: raise ValueError('Positive integer batch size')
        return (*self.old.initial_state(batch_size), self.head.weight.new_zeros((batch_size,96)))

    def new_state_from_old(self, old_state):
        if not isinstance(old_state,(tuple,list)) or len(old_state) != 3: raise ValueError('Explicit primed old tuple3 required')
        batch = len(old_state[0])
        if any(tuple(s.shape) != (batch,n) for s,n in zip(old_state,STATE_DIMS[:3])): raise ValueError('Primed old state shapes')
        return (*tuple(old_state), self.head.weight.new_zeros((batch,96)))

    def _validate_inputs(self, obs, command):
        if obs.ndim != 2 or obs.shape[1] != 238 or command.shape != (len(obs),2) or len(obs)==0:
            raise ValueError('Actual obs[B,238], independent command[B,2] required')
        if any(t.device.type != 'cpu' or t.dtype != torch.float64 for t in (obs,command)) or not bool(torch.isfinite(obs).all()) or not bool(torch.isfinite(command).all()):
            raise ValueError('Finite CPUfloat64 observations/commands required')
        if bool(((command[:,0]<0.) | (command[:,0]>1.) | (command[:,1].abs()>1.)).any()):
            raise ValueError('Independent speed[0,1], yaw[-1,1] only, no clipping')

    @staticmethod
    def legacy_projection(command):
        turn = command[:,1:2]
        return torch.cat((torch.relu(turn)/3.,turn),dim=-1)

    def cached_features(self, obs, command, old_state):
        """Exactly one oldstep; -> oldmean, features370, afterold tuple3.
        Independent speed enters ONLY new ANN; old strict projection is unmodified.
        """
        self._validate_inputs(obs,command)
        with torch.no_grad():
            oldmean, after = self.old.step(obs,self.legacy_projection(command),old_state)
            physical = obs[:,self.old.base.physical_mask.bool()]
            feature = torch.cat((physical,*after,command),dim=-1)
        return oldmean,feature,tuple(after)

    def prime_old(self, obs, command, state=None):
        """Explicit caller-only teacher warm method, no new GRU operation.
        Call for the ACTUAL960 oldmotor warm ticks (not settle200); newh stays0.
        Returns oldmean,state4. Production step never performs a hidden reset.
        """
        state = self.initial_state(len(obs)) if state is None else state
        if len(state)!=4 or bool(torch.count_nonzero(state[3])): raise ValueError('Warm-only prime_old requires explicit zero newh96')
        oldmean,_,after = self.cached_features(obs,command,state[:3])
        return oldmean,self.new_state_from_old(after)

    def step_details(self, obs, command, state):
        self._validate_inputs(obs,command)
        if not isinstance(state,(tuple,list)) or len(state)!=4 or any(tuple(s.shape)!=(len(obs),n) for s,n in zip(state,STATE_DIMS)):
            raise ValueError('Explicit state tuple128/64/64/96 required')
        if any(s.dtype!=torch.float64 or s.device.type!='cpu' or not bool(torch.isfinite(s).all()) for s in state): raise ValueError('Finite CPUfloat64 state352 required')
        oldmean,feature,after = self.cached_features(obs,command,state[:3])
        candidate = self.gru(torch.tanh(self.encoder(feature)),state[3])
        h = (1.-ALPHA)*state[3] + ALPHA*candidate
        residual = BOUND*torch.tanh(self.head(h))
        return oldmean+residual,(*after,h),residual

    def step(self, obs, command, state):
        mean,state,_ = self.step_details(obs,command,state)
        return mean,state

    def sequence(self, obs, commands, state=None):
        if obs.ndim!=3 or obs.shape[-1]!=238 or commands.shape!=obs.shape[:2]+(2,) or len(obs)==0: raise ValueError('Nonempty T,B,238 and T,B,2 required')
        state = self.initial_state(obs.shape[1]) if state is None else state
        means=[]; rows=[[],[],[],[]]
        for x,u in zip(obs,commands):
            mean,state = self.step(x,u,state); means.append(mean)
            for row,value in zip(rows,state):row.append(value)
        return torch.stack(means),tuple(torch.stack(row) for row in rows)

    def cache_sequence(self, obs, commands, old_state):
        """Parent may cache frozen oldmean/features once; no newGRU execution.
        -> oldmean[T,B,42],features[T,B,370],afterold_states tuple3[T,B,n].
        Start with actual960warm-primed old_state; suffix1920 targets no shift.
        """
        if obs.ndim!=3 or obs.shape[-1]!=238 or commands.shape!=obs.shape[:2]+(2,) or len(obs)==0: raise ValueError('Nonempty aligned suffix sequences required')
        means=[]; features=[]; rows=[[],[],[]]
        for x,u in zip(obs,commands):
            mean,f,old_state=self.cached_features(x,u,old_state);means.append(mean);features.append(f)
            for row,value in zip(rows,old_state):row.append(value)
        return torch.stack(means),torch.stack(features),tuple(torch.stack(row) for row in rows)

    def residual_sequence(self, features, h96=None):
        if features.ndim!=3 or features.shape[-1]!=370 or len(features)==0:raise ValueError('Nonempty cachedfeatures T,B,370')
        h96=features.new_zeros((features.shape[1],96)) if h96 is None else h96
        if h96.shape!=(features.shape[1],96):raise ValueError('Explicit newh B,96')
        return ResidualScan(self.encoder,self.gru,self.head)(features,h96)

    def identity(self):
        return dict(format_version=FORMAT,old_checkpoint_sha256=OLD_SHA,old_identity=self.frozen_old_identity,frozen_old_state_digest=self.frozen_old_state_digest,inputs=238,physical_inputs=112,feature_order=['old_physical112','old_after_base128','old_after_RIGHT64','old_after_LEFT64','independent_speed_yaw2'],feature_dim=370,encoder=[370,96],new_GRU_hidden=96,new_GRU_alpha=ALPHA,new_GRU_tau_s=.1,decision_s=.005,head=[96,42],head_initialization='zero',residual_bound=BOUND,mean='frozen_old_mean+6*tanh(head(newh))',action='tanh(mean)',command_order=['forward','turn'],command_bounds=[[0.,1.],[-1.,1.]],old_projection='[relu(turn)/3,turn]',state_dims=list(STATE_DIMS),state_total=352,new_trainable_tensors={k:list(v) for k,v in ADAPTER_SHAPES.items()},new_trainable_tensor_count=8,new_normalizer=False,actor_clock_phase_caseflags=False,command_reset=False,warm_newh='explicit caller prime_old only, zero96 during actual960warmticks',scope='engineering body command motor, not neck deployment or neural success')


def make_residual_scan(model, script=True):
    scan=ResidualScan(model.encoder,model.gru,model.head)
    expected=dict(model.adapter_named_parameters());actual=dict(scan.named_parameters())
    if expected.keys()!=actual.keys() or any(expected[k] is not actual[k] for k in expected):raise ValueError('Scan must share exactly8 adapter parameters')
    return torch.jit.script(scan) if script else scan


def make_scan(model):
    """Parent public API: scripted scan(features[T,B,370]) -> residual,hidden96.
    Zero h96 by default; dynamic T supports full1920 and auxiliary32 FD probes.
    Optional second h96 argument permits explicit continuation, never implicit reset.
    """
    return make_residual_scan(model, script=True)
