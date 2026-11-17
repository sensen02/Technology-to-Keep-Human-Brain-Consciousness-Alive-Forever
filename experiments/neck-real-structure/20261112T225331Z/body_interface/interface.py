"""CODE_ONLY default stdlib boundary; all neural imports/construction explicitly lazy.

Engineering decoder/tuning, not native DN motor function or AN sensory physiology.
No neural controller is executed by importing this module.
"""
from pathlib import Path
import hashlib,json,math
BODY_SENSORY_INDICES=tuple(range(0,42))+tuple(range(84,126))+tuple(range(168,195))
EXCLUDED_INDICES=tuple(range(42,84))+tuple(range(126,168))+tuple(range(195,238))
class InterfaceError(ValueError):pass

def require(ok,message):
 if not ok:raise InterfaceError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source_ports(binding):
 """Derive ordered dimensions from a reviewed source inventory, never candidate counts.

This recognizes source inventory only. Separate parent engineering execution
admission remains required; this function cannot admit full-neck reconstruction.
"""
 require(binding.get('schema')=='BODY_SIDE_SOURCE_PORT_BINDING_V1','source binding schema required')
 require(binding.get('scope')=='SCOPED_ENGINEERING_NECK_BODY_INTERFACE','explicit scoped engineering identity required')
 ports=binding.get('ports');require(isinstance(ports,list) and bool(ports),'source-qualified port list required')
 output={'DN':[],'AN':[]};seen=set()
 for p in ports:
  rid=p.get('root_id');direction=p.get('direction')
  require(isinstance(rid,str) and rid.isdecimal() and rid not in seen,'unique original decimal root ID required')
  require(direction in output and p.get('source_reviewed') is True,'reviewed direction required')
  require(p.get('structural_evidence_status')=='SOURCE_GEOMETRY_REVIEWED','candidate annotation alone cannot authorize ports')
  evidence=p.get('evidence');require(isinstance(evidence,list) and bool(evidence),'source evidence required')
  for row in evidence:
   path=row.get('path');digest=row.get('sha256')
   require(isinstance(path,str) and Path(path).is_absolute() and isinstance(digest,str) and len(digest)==64,'absolute source identity required')
   require(Path(path).is_file() and sha(path)==digest,'source identity drift')
  seen.add(rid);output[direction].append(rid)
 require(all(output.values()),'both source DN and AN required')
 return output

def validate_delivery(packet,expected_ids,direction,bindings_sha256,consume_step):
 """Validate already-released raw neck output, no action or task fields accepted."""
 fields={'source_part','target_part','direction','root_ids','values','capture_step','release_step','consume_step','source_bindings_sha256'}
 require(isinstance(packet,dict) and set(packet)==fields,'undeclared packet fields or bypass')
 require(direction in ('DN','AN') and packet['direction']==direction,'wrong direction')
 require(packet['source_part']=='neck' and packet['target_part']==('other' if direction=='DN' else 'brain'),'not a delivered neck-to-side packet')
 require(packet['root_ids']==list(expected_ids),'source ID ordering mismatch')
 require(packet['source_bindings_sha256']==bindings_sha256,'neck binding drift')
 require(type(consume_step) is int and consume_step>=0,'integer consume step required')
 times=[packet[k] for k in ('capture_step','release_step','consume_step')]
 require(all(type(x) is int and x>=0 for x in times) and times==sorted(times) and times[-1]==consume_step,'future/invalid feedback')
 values=packet['values'];require(isinstance(values,list) and len(values)==len(expected_ids),'port activity dimension mismatch')
 require(all(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1 for v in values),'finite bounded activity required')
 return tuple(float(v) for v in values)

def extract_body_sensory(obs238):
 """Pure validation/projection only; this is not a learned AN encoder."""
 require(len(obs238)==238 and all(type(x) in (int,float) and math.isfinite(x) for x in obs238),'finite local observation238 required')
 return tuple(obs238[i] for i in BODY_SENSORY_INDICES)

def make_neural_interfaces(source_binding,*,dtype='float64'):
 """Parent-only numerical construction; no source ports or trained weights guessed.

No numerical call made by this preparation. All learned outputs remain
untrained until parent-authorized finite calibration and independent checks.
"""
 ports=source_ports(source_binding)
 require(source_binding.get('engineering_execution_admission')=='PARENT_PREREGISTERED_SCOPED_EXECUTION','parent scoped execution admission required')
 require(dtype=='float64','explicit CPU float64 architecture only')
 import torch
 from torch import nn
 n=len(ports['DN']);m=len(ports['AN'])
 class BodyDecoder(nn.Module):
  def __init__(self):
   super().__init__();self.layers=nn.Sequential(nn.Linear(n,16,dtype=torch.float64),nn.Tanh(),nn.Linear(16,1,dtype=torch.float64),nn.Tanh())
  def forward(self,delivered_dn):
   if delivered_dn.ndim!=2 or delivered_dn.shape[1]!=n:raise ValueError('delivered source-ordered DN batch required')
   return self.layers(delivered_dn)
  def motor_command(self,delivered_dn):
   # Frozen CombinedTurning ABI only; turn is produced by the learned NN.
   # No fixed relation between raw DN activity and turn exists here.
   turn=self.forward(delivered_dn);return torch.cat((torch.relu(turn)/3.,turn),dim=-1)
 class FeedbackEncoder(nn.Module):
  def __init__(self):
   super().__init__();self.layers=nn.Sequential(nn.Linear(111,32,dtype=torch.float64),nn.Tanh(),nn.Linear(32,m,dtype=torch.float64),nn.Tanh())
   self.register_buffer('allowed_raw_indices',torch.tensor(BODY_SENSORY_INDICES,dtype=torch.int64))
  def forward(self,local_raw238):
   if local_raw238.ndim!=2 or local_raw238.shape[1]!=238:raise ValueError('local body observation238 required, never direct brain observation')
   return self.layers(local_raw238.index_select(1,self.allowed_raw_indices))
 descriptor=dict(kind='UNTRAINED_ENGINEERING_BODY_DECODER_AND_AN_ENCODER',DN_ids=ports['DN'],AN_ids=ports['AN'],body_decoder=[n,16,1],feedback_encoder=[111,32,m],motor_command='f=relu(learned_turn)/3,turn=learned_turn; frozen motor ABI only',feedback_raw_indices=list(BODY_SENSORY_INDICES),no_direct_target_or_brain_hidden_input=True,trained=False,full_neck_reconstruction=False,native_DN_function_established=False,native_AN_tuning_established=False,walking_initiation_or_stopping_control=False)
 return BodyDecoder(),FeedbackEncoder(),descriptor
