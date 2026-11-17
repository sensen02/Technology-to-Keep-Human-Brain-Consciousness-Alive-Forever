"""Experimental process-local immutable-parent cache. Not installed in any loader.
No numerical imports on import. Caller must supply originalstrictloader and its
original RNG-preserving context; a complete approved content-SHA ledger is mandatory.
"""
from pathlib import Path
import copy,hashlib,json,platform,sys,threading

class CacheInvariantError(RuntimeError):pass

def content_sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_content(ledger):
 if not isinstance(ledger,dict) or not ledger:raise CacheInvariantError('Nonemptycompletecontentledgerrequired')
 for path,digest in ledger.items():
  if not Path(path).is_absolute() or not isinstance(digest,str) or len(digest)!=64 or content_sha(path)!=digest:raise CacheInvariantError('Contentidentitychanged: '+str(path))

def namespace_snapshot():
 sentinel=object();return sentinel,list(sys.path),{k:sys.modules.get(k,sentinel) for k in ('model','common','sources','adapter')}

def restore_namespace(snapshot):
 sentinel,paths,aliases=snapshot;sys.path[:]=paths
 for key,value in aliases.items():
  if value is sentinel:sys.modules.pop(key,None)
  else:sys.modules[key]=value

class ImmutableParentCache:
 """One exact parent per instance. No persistedcache, checkpointloaderpatch or hitfallback.
 The approved ledger MUST contain checkpoint, full ancestral config source/data,
 qualification proofs AND cache/adapter source. Completeness cannot be inferred
 from a checkpoint hash alone: caller audit is a separate admission requirement.
 """
 def __init__(self,*,checkpoint,digest,format_version,approved_ledger,runtime_identity,
              strict_loader,preserve_rng,approve_parent,assert_isolated):
  self._checkpoint=str(Path(checkpoint).resolve());self._digest=digest;self._format=format_version
  self._ledger=copy.deepcopy(approved_ledger);self._runtime=copy.deepcopy(runtime_identity)
  if self._ledger.get(self._checkpoint)!=digest:raise CacheInvariantError('Parentcheckpointmustbecontentbound')
  # Each future integration must bind this exact independent implementation too.
  own=str(Path(__file__).resolve())
  if self._ledger.get(own)!=content_sha(own):raise CacheInvariantError('Independentcacheimplementationmustbecontentbound')
  self._loader=strict_loader;self._rng=preserve_rng;self._approve=approve_parent;self._isolate=assert_isolated
  self._private=None;self._lock=threading.RLock();self._misses=0;self._hits=0
  canonical=dict(checkpoint=self._checkpoint,digest=digest,format_version=format_version,ledger=self._ledger,runtime=self._runtime,python=platform.python_version(),executable=sys.executable)
  self.identity=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 def get(self,*,runtime_identity):
  with self._lock:
   if runtime_identity!=self._runtime:raise CacheInvariantError('Runtimeidentitychanged')
   # Entry content hashing includes all ancestors, never substitutes stat/mtime.
   verify_content(self._ledger);snapshot=namespace_snapshot()
   try:
    with self._rng():
     if self._private is None:
      model,payload=self._loader(self._checkpoint,self._digest,final=True)
      if payload.get('step')!=600 or payload.get('format_version')!=self._format:raise CacheInvariantError('Exactparent600formatrequired')
      self._approve(model,payload)
      private=copy.deepcopy((model,payload));self._isolate((model,payload),private)
      # Do not admit a cache if any source/proof changed during strictloading.
      verify_content(self._ledger);self._private=private;self._misses+=1
     else:self._hits+=1
     result=copy.deepcopy(self._private);self._isolate(self._private,result)
     # Retain all original semantic validation of parent state on every return;
     # this validator must be tensor-only and must not recursively load ancestors.
     self._approve(*result)
     verify_content(self._ledger)
     return result
   except BaseException:
    # A failed postcondition invalidates the slot; no stale fallback is returned.
    self._private=None
    raise
   finally:restore_namespace(snapshot)
 def check_original_anchor(self,anchor,*,runtime_identity):
  """Exact tensor equality against the first strictvalidated payload; no parent NN rebuild.
  Lazy Torch import is for future parent verification only, never used in ONLYCODE.
  Returns no references to the private anchor.
  """
  import torch
  with self._lock:
   if self._private is None:raise CacheInvariantError('Strictmissrequired beforeanchorcomparison')
   if runtime_identity!=self._runtime:raise CacheInvariantError('Runtimeidentitychanged')
   verify_content(self._ledger);expected=self._private[1]['config']['initial_anchor']
   if not isinstance(anchor,dict) or anchor.keys()!=expected.keys():raise CacheInvariantError('Anchorcanonicalnameschanged')
   for name,value in expected.items():
    actual=anchor[name]
    if not torch.is_tensor(actual) or actual.shape!=value.shape or actual.dtype!=value.dtype or actual.device!=value.device or not torch.equal(actual,value):raise CacheInvariantError('Originalanchorchanged: '+name)
   verify_content(self._ledger)
   return True
 @property
 def counts(self):return dict(strict_misses=self._misses,deepcopy_hits=self._hits)
