"""Parent-only numerical backend; ONLYCODE preparation never imports this file.
Fast neural readout of current encoded features plus continuous leaky state.
"""
import copy
import math
from typing import List,Tuple,Optional
import torch
from torch import nn

FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_FASTPATH_REPAIR_V1'
PARENT_FORMAT='INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'
PARENT_SHA='454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398'
OLD_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
STATE_DIMS=(128,64,64,96)
ALPHA=-math.expm1(-.005/.1)
BOUND=6.
ADAPTER_SHAPES={'encoder.weight':(96,370),'encoder.bias':(96,),'gru.weight_ih':(288,96),'gru.weight_hh':(288,96),'gru.bias_ih':(288,),'gru.bias_hh':(288,),'head.weight':(42,192),'head.bias':(42,)}


def state_digest(state):
    import hashlib,json
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous();description=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode();raw=value.numpy().tobytes(order='C')
        digest.update(len(description).to_bytes(8,'little'));digest.update(description);digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
    return digest.hexdigest()


class ResidualScan(nn.Module):
    def __init__(self,encoder,gru,head):
        super().__init__();self.encoder=encoder;self.gru=gru;self.head=head;self.alpha=ALPHA;self.bound=BOUND
    def forward(self,features:torch.Tensor,h:Optional[torch.Tensor]=None)->Tuple[torch.Tensor,torch.Tensor]:
        if h is None:h=features.new_zeros((features.size(1),96))
        outputs=torch.jit.annotate(List[torch.Tensor],[]);states=torch.jit.annotate(List[torch.Tensor],[])
        for k in range(features.size(0)):
            encoded=torch.tanh(self.encoder(features[k]));candidate=self.gru(encoded,h);h=(1.-self.alpha)*h+self.alpha*candidate
            outputs.append(self.bound*torch.tanh(self.head(torch.cat((h,encoded),dim=-1))));states.append(h)
        return torch.stack(outputs),torch.stack(states)


class IndependentMotor(nn.Module):
    def __init__(self,parent_motor):
        super().__init__();parent_motor.verify_old();identity=parent_motor.identity()
        if identity.get('format_version')!=PARENT_FORMAT or identity.get('state_dims')!=list(STATE_DIMS) or tuple(parent_motor.head.weight.shape)!=(42,96):raise ValueError('Strict parent residual352/head42x96 required')
        if any(p.requires_grad for p in parent_motor.old.parameters()):raise ValueError('Original789f must already be frozen')
        self.old=copy.deepcopy(parent_motor.old);self.encoder=copy.deepcopy(parent_motor.encoder);self.gru=copy.deepcopy(parent_motor.gru);self.head=nn.Linear(192,42,dtype=torch.float64)
        self.encoder.requires_grad_(True);self.gru.requires_grad_(True)
        with torch.no_grad():self.head.weight[:,:96].copy_(parent_motor.head.weight);self.head.weight[:,96:].zero_();self.head.bias.copy_(parent_motor.head.bias)
        self.frozen_old_state_digest=state_digest(self.old.state_dict());self.frozen_old_identity=copy.deepcopy(self.old.identity());self.parent_identity=copy.deepcopy(identity)
        self.validate_architecture()
    def adapter_named_parameters(self):return [(name,p) for name,p in self.named_parameters() if not name.startswith('old.')]
    def adapter_parameters(self):return [p for _,p in self.adapter_named_parameters()]
    def validate_architecture(self):
        pairs=self.adapter_named_parameters()
        if [name for name,_ in pairs]!=list(ADAPTER_SHAPES):raise ValueError('Exactly ordered encoder2/GRU4/head2 tensors')
        if any(tuple(p.shape)!=ADAPTER_SHAPES[name] or p.dtype!=torch.float64 or p.device.type!='cpu' or not p.requires_grad for name,p in pairs):raise ValueError('Fastpath8 shapes/CPUfloat64/trainability')
        if any(p.requires_grad for p in self.old.parameters()) or tuple(self.old.base.physical_mask.shape)!=(238,) or int(self.old.base.physical_mask.bool().sum())!=112:raise ValueError('Originalfrozenold238mask112 required')
        return True
    def verify_old(self):
        if state_digest(self.old.state_dict())!=self.frozen_old_state_digest or self.old.identity()!=self.frozen_old_identity or any(p.requires_grad or p.grad is not None for p in self.old.parameters()):raise ValueError('Originalfrozenoldweights/identity drift')
        return True
    def initial_state(self,batch_size):
        if type(batch_size) is not int or batch_size<=0:raise ValueError('Positivebatch required')
        return (*self.old.initial_state(batch_size),self.head.weight.new_zeros((batch_size,96)))
    def new_state_from_old(self,old_state):
        if not isinstance(old_state,(tuple,list)) or len(old_state)!=3:raise ValueError('Explicitoldtuple3')
        batch=len(old_state[0])
        if any(tuple(s.shape)!=(batch,n) for s,n in zip(old_state,STATE_DIMS[:3])):raise ValueError('Oldstateshapes')
        return (*tuple(old_state),self.head.weight.new_zeros((batch,96)))
    def _validate_inputs(self,obs,command):
        if obs.ndim!=2 or obs.shape[1]!=238 or command.shape!=(len(obs),2) or not len(obs):raise ValueError('obsB238/independentcommandB2 required')
        if any(t.device.type!='cpu' or t.dtype!=torch.float64 or not bool(torch.isfinite(t).all()) for t in (obs,command)):raise ValueError('FiniteCPUfloat64 inputs')
        if bool(((command[:,0]<0.)|(command[:,0]>1.)|(command[:,1].abs()>1.)).any()):raise ValueError('speed0..1/turn-1..1 independenttask')
    @staticmethod
    def legacy_projection(command):
        turn=command[:,1:2];return torch.cat((torch.relu(turn)/3.,turn),dim=-1)
    def cached_features(self,obs,command,old_state):
        self._validate_inputs(obs,command)
        with torch.no_grad():
            oldmean,after=self.old.step(obs,self.legacy_projection(command),old_state);physical=obs[:,self.old.base.physical_mask.bool()];features=torch.cat((physical,*after,command),dim=-1)
        return oldmean,features,tuple(after)
    def prime_old(self,obs,command,state=None):
        state=self.initial_state(len(obs)) if state is None else state
        if len(state)!=4 or bool(torch.count_nonzero(state[3])):raise ValueError('Explicitwarmonly new96zero')
        oldmean,_,after=self.cached_features(obs,command,state[:3]);return oldmean,self.new_state_from_old(after)
    def step_details(self,obs,command,state):
        self._validate_inputs(obs,command)
        if not isinstance(state,(tuple,list)) or len(state)!=4 or any(tuple(s.shape)!=(len(obs),n) for s,n in zip(state,STATE_DIMS)):raise ValueError('Explicitstate128/64/64/96')
        if any(s.dtype!=torch.float64 or s.device.type!='cpu' or not bool(torch.isfinite(s).all()) for s in state):raise ValueError('FiniteCPUfloat64 state352')
        oldmean,features,after=self.cached_features(obs,command,state[:3]);encoded=torch.tanh(self.encoder(features));candidate=self.gru(encoded,state[3]);h=(1.-ALPHA)*state[3]+ALPHA*candidate;residual=BOUND*torch.tanh(self.head(torch.cat((h,encoded),dim=-1)))
        return oldmean+residual,(*after,h),residual
    def step(self,obs,command,state):
        mean,state,_=self.step_details(obs,command,state);return mean,state
    def sequence(self,obs,commands,state=None):
        if obs.ndim!=3 or obs.shape[-1]!=238 or commands.shape!=obs.shape[:2]+(2,) or not len(obs):raise ValueError('T,B238/T,B2 sequences')
        state=self.initial_state(obs.shape[1]) if state is None else state;means=[];rows=[[],[],[],[]]
        for x,u in zip(obs,commands):
            mean,state=self.step(x,u,state);means.append(mean)
            for row,value in zip(rows,state):row.append(value)
        return torch.stack(means),tuple(torch.stack(row) for row in rows)
    def cache_sequence(self,obs,commands,old_state):
        if obs.ndim!=3 or obs.shape[-1]!=238 or commands.shape!=obs.shape[:2]+(2,) or not len(obs):raise ValueError('Alignedsuffix sequences')
        means=[];features=[];rows=[[],[],[]]
        for x,u in zip(obs,commands):
            mean,f,old_state=self.cached_features(x,u,old_state);means.append(mean);features.append(f)
            for row,value in zip(rows,old_state):row.append(value)
        return torch.stack(means),torch.stack(features),tuple(torch.stack(row) for row in rows)
    def residual_sequence(self,features,h96=None):
        if features.ndim!=3 or features.shape[-1]!=370 or not len(features):raise ValueError('NonemptycachedT,B,370')
        h96=features.new_zeros((features.shape[1],96)) if h96 is None else h96
        if h96.shape!=(features.shape[1],96):raise ValueError('newhB96')
        return ResidualScan(self.encoder,self.gru,self.head)(features,h96)
    def identity(self):
        return dict(format_version=FORMAT,parent_checkpoint_sha256=PARENT_SHA,parent_identity=self.parent_identity,old_checkpoint_sha256=OLD_SHA,old_identity=self.frozen_old_identity,frozen_old_state_digest=self.frozen_old_state_digest,inputs=238,physical_inputs=112,feature_dim=370,feature_order=['oldphysical112','oldafterbase128','oldafterRIGHT64','oldafterLEFT64','independenttask2'],encoder=[370,96],new_GRU_hidden=96,new_GRU_alpha=ALPHA,new_GRU_tau_s=.1,decision_s=.005,head=[192,42],head_input_order=['newh96','currentencoded96'],head_initialization='parentfirst96/currentskip96zero/parentbias',residual_bound=BOUND,mean='oldmean+6*tanh(head(cat(h96,encoded96)))',action='tanh(mean)',state_dims=list(STATE_DIMS),state_total=352,new_trainable_tensors={k:list(v) for k,v in ADAPTER_SHAPES.items()},new_trainable_tensor_count=8,additional_head_parameters=4032,command_order=['forward','turn'],command_bounds=[[0.,1.],[-1.,1.]],old_projection='[relu(turn)/3,turn]',new_normalizer=False,actor_clock_phase_caseflags=False,command_reset=False,scope='finitejointrepair bodymotor only, no brain/neckdeployment or behaviorPASS')


def make_residual_scan(model,script=True):
    scan=ResidualScan(model.encoder,model.gru,model.head);expected=dict(model.adapter_named_parameters());actual=dict(scan.named_parameters())
    if expected.keys()!=actual.keys() or any(expected[k] is not actual[k] for k in expected):raise ValueError('Scanmustsharecanonical8parameters')
    return torch.jit.script(scan) if script else scan

def make_scan(model):return make_residual_scan(model,script=True)
