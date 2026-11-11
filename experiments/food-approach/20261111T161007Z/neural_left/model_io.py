"""Strict LEFT_HEAD only: frozen baseline/transfer features, fresh zero bounded head."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import torch,numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OLD=ROOT/'20261110T191449Z/residual_candidate/residual_motor.py'
TRANSFER=ROOT/'20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt'
TRANSFER_SHA='c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb'
TEACHER=HERE.parent/'left_tarsal_actual_guard/left_yaw_p060'
DATA=TEACHER/'SAMPLES_5MS.npz'
AUDIT=TEACHER.parent/'AUDIT_ACTUAL_GUARD_20261111T183323Z_377784_1794422003628573149.json'
FORMAT='strict_LEFT_HEAD_V1'
LR=.0003

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
legacy=module('_strict_left_head_legacy',OLD)
sha256=legacy.sha256
BASE=legacy.BASE
REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'

def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
 return type(a)==type(b) and a==b

def verify_hashes(values):
 for p,h in values.items():
  if not Path(p).is_absolute() or sha256(p)!=h:raise ValueError('artifact drift '+p)

def commands():
 v=np.clip((np.arange(2880)-960)*.005,0.,1.);b=v*v*(3.-2.*v)
 return np.column_stack((b/3.,b))

class LeftHead(legacy.ResidualMotor):
 VERSION='LEFT_HEAD_coupled_fturn3_positive_bound6_frozen_features_v1'
 RESIDUAL_BOUND=6.
 def __init__(self,baseline):
  super().__init__(baseline);self.encoder.requires_grad_(False);self.gru.requires_grad_(False)
 def adapter_parameters(self):return list(self.head.parameters())
 def step(self,obs,command,state):
  if obs.ndim!=2 or obs.shape[-1]!=238 or command.shape!=(len(obs),2):raise ValueError('actual obs238/command2 required')
  if len(state)!=2 or state[0].shape!=(len(obs),128) or state[1].shape!=(len(obs),64):raise ValueError('explicit128+64 state required')
  if not torch.isfinite(obs).all() or not torch.isfinite(command).all() or not all(torch.isfinite(s).all() for s in state):raise ValueError('nonfinite actor input')
  f,t=command[:,0],command[:,1]
  # Actual teacher uses v/.075-1; retain its float64 values with the declared 1e-12 coupling tolerance, never clip actor inputs.
  if (f < -1e-12).any() or (f > 1./3+1e-12).any() or (t < -1e-12).any() or (t > 1.+1e-12).any() or ((f-t/3).abs()>1e-12).any():raise ValueError('LEFT coupled f=turn/3 domain only')
  mean,bh=self.base.step(obs,state[0])
  features=torch.cat((obs[:,self.base.physical_mask.bool()],bh,command),-1)
  candidate=self.gru(torch.tanh(self.encoder(features)),state[1]);rh=(1-self.ALPHA)*state[1]+self.ALPHA*candidate
  return mean+torch.relu(command[:,1:2])*self.RESIDUAL_BOUND*torch.tanh(self.head(rh)),(bh,rh)
 def identity(self):
  i=super().identity();i.update(version=self.VERSION,kind=FORMAT,residual_bound=6.,gate='relu(command[:,1:2])',command_domain='forward[0,1/3]; turn[0,1]; abs(forward-turn/3)<=1e-12; yaw/.06',transfer=str(TRANSFER),transfer_sha256=TRANSFER_SHA,encoder_frozen=True,gru_frozen=True,head_initialization='new allzero',trainable_parameters=2730,optimizer='fresh Adam head only, 2states',historical_obs_targetspeed=.075);return i

def frozen(m):return {k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('head.')}

def initialize():
 if sha256(TRANSFER)!=TRANSFER_SHA:raise ValueError('transfer SHA mismatch')
 p=torch.load(TRANSFER,map_location='cpu',weights_only=False)
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 if p.get('format_version')!=1 or p['curve'][-1]['step']!=1500:raise ValueError('transfer original1500 format')
 m=LeftHead(legacy.load_baseline_actor())
 for name in ('encoder','gru'):
  getattr(m,name).load_state_dict({k[len(name)+1:]:v for k,v in p['state_dict'].items() if k.startswith(name+'.')},strict=True)
 if not exact(m.base.state_dict(),{k[5:]:v for k,v in p['state_dict'].items() if k.startswith('base.')}):raise ValueError('transfer originalbase differs')
 with torch.no_grad():m.head.weight.zero_();m.head.bias.zero_()
 if sum(p.numel() for p in m.parameters() if p.requires_grad)!=2730:raise ValueError('head-only2730 required')
 return m

def teacher_evidence():
 r=json.loads((TEACHER/'RESULT.json').read_text());a=json.loads(AUDIT.read_text())
 if not a['integrity_all_pass'] or not all(a['checks'].values()) or not a['teacher_eligible']:raise ValueError('teacher independent audit failed')
 if a['result_sha256']!=sha256(TEACHER/'RESULT.json') or a['yaw_deg']!=r['actual_signed_yaw_deg']:raise ValueError('teacher audit/result binding mismatch')
 if not r['all_gates'] or not r['cycles']['all_legs_completed'] or not r['strict_nonfoot_zero_1ms'] or r['actual_signed_yaw_deg']<30:raise ValueError('LEFT teacher behavioral eligibility')
 if r['native']['post_nonfoot_peak_native']!=0 or r['native']['post_native_steps']!=768000 or r['native']['extra_force_max']!=0:raise ValueError('teacher nativeNF0 required')
 if sha256(DATA)!=r['samples']['sha256']:raise ValueError('teacher samples drift')
 # Bind the audit's recorded source and sample hash evidence to files, rather than only its boolean.
 for name in ('source_hashes',):
  if name in a:verify_hashes(a[name])
 return r,a

def dataset():
 teacher_evidence()
 with np.load(DATA,allow_pickle=False) as z:
  x=torch.from_numpy(z['X238'].copy())[:,None];u=torch.from_numpy(z['command2'].copy())[:,None];y=torch.from_numpy(z['Y42'].copy())[:,None]
 if x.shape!=(2880,1,238) or u.shape!=(2880,1,2) or y.shape!=(2880,1,42):raise ValueError('full actual2880 data required')
 if any(v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in (x,u,y)):raise ValueError('finite float64 teacher data')
 if not torch.equal(u[:960],torch.zeros_like(u[:960])) or float((u[:,0]-torch.from_numpy(commands())).abs().max())>1e-12:raise ValueError('LEFT actual coupled command mismatch')
 if not torch.equal(x[:,:,-1],torch.full((2880,1),.075,dtype=torch.float64)) or y.abs().max()>1:raise ValueError('historical target speed or bounded labels changed')
 return x,u,y

def prepare(m):
 x,u,y=dataset()
 with torch.no_grad():means,h=m.sequence(x,u)
 # Frozen recurrent features are independent of all head parameters.
 gate=torch.relu(u[:,:,1:2]);base=[];bh=m.base.initial_state(1)
 with torch.no_grad():
  for row in x:
   b,bh=m.base.step(row,bh);base.append(b)
 return dict(x=x,u=u,y=y,hidden=h[1].detach(),base=torch.stack(base),gate=gate,means=means,base_states=h[0].detach())

def cached_forward(m,c):return c['base']+c['gate']*m.RESIDUAL_BOUND*torch.tanh(m.head(c['hidden']))

def neutral(m):
 with np.load(REFERENCE,allow_pickle=False) as z:
  x=torch.from_numpy(z['X238'].copy())[:,None];ref=torch.from_numpy(z['mean'].copy())[:,None];hb=torch.from_numpy(z['hidden_after'].copy())[:,None];actions=torch.from_numpy(z['policy_action'].copy())[:,None]
 with torch.no_grad():mean,h=m.sequence(x,torch.zeros((2880,1,2),dtype=torch.float64))
 return dict(mean_exact=torch.equal(mean,ref),action_exact=torch.equal(mean.tanh(),actions),base_hidden_exact=torch.equal(h[0],hb))

def source_paths():return [HERE/n for n in ('PREREG.json','model_io.py','train.py','verify_saved.py')]+[OLD,BASE/'models.py']
def data_paths():return [TRANSFER,legacy.CHECKPOINT,DATA,TEACHER/'RESULT.json',AUDIT,REFERENCE]

def load_checked(path,digest,require_final=True):
 if sha256(path)!=digest:raise ValueError('checkpoint SHA mismatch')
 p=torch.load(path,map_location='cpu',weights_only=False)
 if p.get('format_version')!=FORMAT or p['config']['kind']!=FORMAT or p['config']['physical_eval_only_step']!=4000:raise ValueError('strict LEFT_HEAD format required')
 verify_hashes(p['config']['source_hashes']);verify_hashes(p['config']['data_hashes'])
 m=initialize()
 if p['identity']!=m.identity():raise ValueError('LEFT architecture identity mismatch')
 expected=frozen(m);m.load_state_dict(p['state_dict'],strict=True)
 if not exact(expected,frozen(m)) or not all(torch.isfinite(v).all() for v in m.state_dict().values()):raise ValueError('frozen drift or nonfinite weights')
 step=p['curve'][-1]['step']
 if step not in (2000,4000) or (require_final and step!=4000) or len(p['curve'])!=step:raise ValueError('only final4000 physically eligible')
 if p['config']['lr']!=LR or p['config']['seed']!=20261111:raise ValueError('fixed optimizer config changed')
 opt=torch.optim.Adam(m.adapter_parameters(),lr=LR);opt.load_state_dict(p['optimizer'])
 if len(opt.state)!=2 or any(float(v['step'])!=step for v in opt.state.values()) or any(g['lr']!=LR for g in opt.param_groups):raise ValueError('two head Adam states and exact counters/lr required')
 return m.eval(),p
