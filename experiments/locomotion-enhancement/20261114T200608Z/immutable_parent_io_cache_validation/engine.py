"""Independent IO validation instrumentation; no installation or numerical imports."""
from pathlib import Path
import contextlib,copy,hashlib,json,sys,time
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ALIASES=('model','common','sources','adapter','_statealigned_actual_strict_v4_io','_statealigned_actual_v4_adapter','_statealigned_resume_deployment','_statealigned_exact_v4_storage_definitions')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(ledger):
 for path,digest in ledger.items():
  if not Path(path).is_absolute() or sha(path)!=digest:raise ValueError('Contentidentitychanged:'+str(path))
@contextlib.contextmanager
def namespace_guard():
 sentinel=object();aliases={k:sys.modules.get(k,sentinel) for k in ALIASES};paths=list(sys.path)
 try:yield
 finally:
  sys.path[:]=paths
  for k,v in aliases.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v

class ObservedLedger:
 """Python call observer: originalverify/sha execute unchanged, never returnmock.
 Captures every observed project verify_hashes/verify mapping and successful file
 SHA return. Profile hook is thread-local; validation is CPU1/one Python thread.
 """
 def __init__(self):self.ledger={};self.maps=0;self.sha_returns=0;self.parent_calls=0;self.neural_calls=0;self._pending={}
 def add(self,path,digest):
  path=str(Path(path).resolve())
  if path in self.ledger and self.ledger[path]!=digest:raise ValueError('Observedcontentidentityconflict:'+path)
  self.ledger[path]=digest
 def observe(self,frame,event,arg):
  filename=frame.f_code.co_filename
  if not filename.startswith(str(PROJECT)):return
  name=frame.f_code.co_name
  if event=='call':
   if name in ('step','step_details','forward','cache_sequence','prime_old'):
    self.neural_calls+=1;raise RuntimeError('NNforwardforbidden_in_IO_validation:'+filename+':'+name)
   if name=='load_resume_parent' and 'state_aligned_action_calibration' in filename:self.parent_calls+=1
   if name in ('verify_hashes','verify'):
    for key in ('hashes','ledger','h'):
     mapping=frame.f_locals.get(key)
     if isinstance(mapping,dict) and mapping and all(isinstance(p,str) and Path(p).is_absolute() and isinstance(d,str) and len(d)==64 for p,d in mapping.items()):
      for p,d in mapping.items():self.add(p,d)
      self.maps+=1;break
   if name in ('sha','sha256','content_sha'):
    for key in ('path','p','filename'):
     value=frame.f_locals.get(key)
     if isinstance(value,(str,Path)) and Path(value).is_absolute() and Path(value).is_file():self._pending[id(frame)]=str(value);break
  elif event=='return' and id(frame) in self._pending:
   path=self._pending.pop(id(frame))
   if isinstance(arg,str) and len(arg)==64 and all(c in '0123456789abcdef' for c in arg):self.add(path,arg);self.sha_returns+=1
 @contextlib.contextmanager
 def capture(self):
  previous=sys.getprofile()
  if previous is not None and not getattr(previous,'_io_ledger_observer',False):raise RuntimeError('Preexistingforeignprofilehook_notreplaced')
  def callback(frame,event,arg):
   if previous is not None:previous(frame,event,arg)
   self.observe(frame,event,arg)
  callback._io_ledger_observer=True
  sys.setprofile(callback)
  try:yield self
  finally:sys.setprofile(previous)
 def bind_loaded_sources(self):
  for module in tuple(sys.modules.values()):
   filename=getattr(module,'__file__',None)
   if filename:
    path=Path(filename)
    if path.is_file() and str(path).startswith(str(PROJECT)) and path.suffix=='.py':self.add(path,sha(path))

class ValidatedParent:
 """Cache exactly original producer.load_resume_parent output; no validator bypass.
 Only called inside temporary producer-module binding by this validation program.
 """
 def __init__(self,original,ledger,runtime,rng_context,approve,isolate,fingerprint):
  self.original=original;self.ledger=dict(ledger);self.runtime=copy.deepcopy(runtime);self.rng_context=rng_context;self.approve=approve;self.isolate=isolate;self.fingerprint=fingerprint;self.private=None;self.standard=None;self.misses=0;self.hits=0;self.strict_parent_calls=0;self.capture_report={}
 def __call__(self):
  from checks import runtime_identity
  if runtime_identity()!=self.runtime:raise ValueError('Actualruntimechanged')
  verify(self.ledger)
  with namespace_guard(),self.rng_context():
   try:
    if self.private is None:
     observer=ObservedLedger()
     with observer.capture():
      self.strict_parent_calls+=1;model,payload=self.original()
     observer.bind_loaded_sources()
     if observer.maps<1 or observer.sha_returns<1:raise ValueError('Actualstrictverifiercapturemissing')
     for p,d in observer.ledger.items():
      if p in self.ledger and self.ledger[p]!=d:raise ValueError('Initialandobservedledgerconflict')
      self.ledger[p]=d
     # Original strictloader has already performed actualmodel/Adam/identity/RNG checks.
     self.private=copy.deepcopy((model,payload));self.standard=self.private
     self.isolate((model,payload),self.private);self.approve(model,payload,self.standard);self.misses+=1
     self.capture_report=dict(actual_original_verify_maps=observer.maps,actual_file_sha_returns=observer.sha_returns,observed_ledger_entries=len(observer.ledger),capture_complete_for_this_strict_execution=True)
    else:self.hits+=1
    result=copy.deepcopy(self.private);self.isolate(self.private,result);self.approve(*result,self.standard);verify(self.ledger);return result
   except BaseException:self.private=None;raise
 def identity_key(self):
  return hashlib.sha256(json.dumps(dict(runtime=self.runtime,content_ledger=self.ledger,parent_FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1',parent600_step=600),sort_keys=True,separators=(',',':')).encode()).hexdigest()
