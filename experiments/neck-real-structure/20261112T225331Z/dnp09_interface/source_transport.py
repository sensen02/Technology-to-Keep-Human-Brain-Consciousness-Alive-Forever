"""Source-ordered vector same-axon boundary approximation, NOT a neural circuit model.

Startup/final full source hash verification; immutable binding bytes checked each tick.
Engineering gain1 delay2ms, no synthetic neck synapses or biological tuning claim.
"""
from pathlib import Path
import json,math
from api import require,sha,source_ports,validate_delivery
class SourceTransport:
 def __init__(self,path,expected_sha256,*,_allow_synthetic_fixture=False):
  self.path=Path(path);require(sha(self.path)==expected_sha256,'binding hash mismatch');self.digest=expected_sha256;self._bytes=self.path.read_bytes();self.binding=json.loads(self._bytes);require(_allow_synthetic_fixture or self.binding.get('synthetic_fixture') is not True,'synthetic fixture cannot run as real source');self.ports=source_ports(self.binding)
  self._source_files=self.binding.get('source_files',[]);require(bool(self._source_files),'source ledger required');self._verify_sources()
  cfg=self.binding.get('transport_config');require(cfg==dict(dt_s=.001,delay_steps=2,gain=1.,activity_min=-1.,activity_max=1.,evidence='ENGINEERING_ASSUMPTION_NOT_MEASURED_CONDUCTION'),'declared engineering transport config required')
  self.step=0;self._queue=[];self._seen=set();self._cut=set();self._journal=[]
 def _verify_identity(self):require(self.path.read_bytes()==self._bytes and sha(self.path)==self.digest,'binding bytes changed')
 def _verify_sources(self):
  for row in self._source_files:require(sha(row['path'])==row['sha256'],'source file changed '+row['path'])
 def send(self,packet):
  self._verify_identity();fields={'from','to','direction','root_ids','values','capture_step','source_bindings_sha256'};require(isinstance(packet,dict) and set(packet)==fields,'hidden input or missing fields')
  d=packet['direction'];require(d in self.ports,'unknown direction');require((packet['from'],packet['to'])==(('brain','neck') if d=='DN' else ('other','neck')),'brain/body bypass')
  require(packet['root_ids']==self.ports[d],'source identity ordering mismatch');require(packet['source_bindings_sha256']==self.digest,'source packet digest mismatch');require(type(packet['capture_step']) is int and packet['capture_step']==self.step,'packet future/past step')
  vals=packet['values'];require(isinstance(vals,list) and len(vals)==len(self.ports[d]) and all(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1 for v in vals),'bounded source-ordered neural activity required')
  require(d not in self._cut and (self.step,d) not in self._seen,'cut or duplicate packet');item=dict(direction=d,root_ids=list(self.ports[d]),values=[float(v) for v in vals],capture_step=self.step,release_step=self.step+2);self._queue.append(item);self._seen.add((self.step,d));self._journal.append(dict(event='enqueue',**item))
 def advance(self):
  self._verify_identity();self.step+=1;pending=[];out=[]
  for item in self._queue:
   if item['release_step']>self.step:pending.append(item);continue
   require(item['release_step']==self.step,'overdue queue item')
   d=item['direction'];require(d not in self._cut,'cut packet should have been discarded');result=dict(source_part='neck',target_part='other' if d=='DN' else 'brain',**item,consume_step=self.step,source_bindings_sha256=self.digest);validate_delivery(result,self.ports[d],d,self.digest,self.step);out.append(result);self._journal.append(dict(event='release',**result))
  self._queue=pending;self._seen={x for x in self._seen if x[0]>=self.step};return out
 def set_cut(self,direction,cut):
  self._verify_identity();require(direction in self.ports and type(cut) is bool,'known direction/bool cut required')
  if cut:
   self._cut.add(direction);pending=[]
   for item in self._queue:
    if item['direction']==direction:self._journal.append(dict(event='drop_inflight',cut_step=self.step,**item))
    else:pending.append(item)
   self._queue=pending
  else:self._cut.discard(direction)
  self._journal.append(dict(event='cut' if cut else 'restore',direction=direction,step=self.step,clock_reset=False))
 def verify_final(self):
  self._verify_identity();self._verify_sources();return dict(source_files_unchanged=True,binding_sha256=self.digest,actual_steps=self.step,full_neck_admitted=False,physiology_verified=False,neural_training_updates=0)
 def journal(self):self._verify_identity();return json.loads(json.dumps(self._journal))
