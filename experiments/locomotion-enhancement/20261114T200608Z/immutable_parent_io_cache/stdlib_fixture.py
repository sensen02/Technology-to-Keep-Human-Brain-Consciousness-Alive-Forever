"""Fake mutable objects only; NEVER imports/loads any numericalmodel or payload."""
from pathlib import Path
from contextlib import contextmanager
import ast,json,random,sys
from cache import ImmutableParentCache,CacheInvariantError,content_sha

def main():
 root=Path(__file__).resolve().parent
 for p in root.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
 fake_checkpoint=root/'EQUIVALENCE.zh-CN.md';own=root/'cache.py';ledger={str(p):content_sha(p) for p in (fake_checkpoint,own)}
 calls=[];state=random.getstate();aliases={k:sys.modules.get(k) for k in ('model','common','sources','adapter')};paths=list(sys.path)
 @contextmanager
 def rng_context():
  before=random.getstate()
  try:yield
  finally:random.setstate(before)
 def loader(path,digest,final):
  assert final and digest==content_sha(path);calls.append(path);random.random()
  return {'weights':[1,2]}, {'step':600,'format_version':'FAKE','config':{'initial_anchor':{'a':[7]}},'optimizer':{'moments':[3,4]},'rng':[9]}
 def approve(model,payload):
  assert model['weights']==[1,2] and payload['step']==600
 def isolated(first,second):
  assert first is not second and first[0] is not second[0] and first[0]['weights'] is not second[0]['weights']
  assert first[1]['optimizer']['moments'] is not second[1]['optimizer']['moments']
  assert first[1]['config']['initial_anchor']['a'] is not second[1]['config']['initial_anchor']['a']
 runtime={'scope':'FAKE_STDLIB_ONLY'}
 cache=ImmutableParentCache(checkpoint=fake_checkpoint,digest=content_sha(fake_checkpoint),format_version='FAKE',approved_ledger=ledger,runtime_identity=runtime,strict_loader=loader,preserve_rng=rng_context,approve_parent=approve,assert_isolated=isolated)
 first=cache.get(runtime_identity=runtime);first[0]['weights'][0]=999;first[1]['optimizer']['moments'][0]=888
 second=cache.get(runtime_identity=runtime);assert second[0]['weights']==[1,2] and second[1]['optimizer']['moments']==[3,4]
 assert len(calls)==1 and cache.counts=={'strict_misses':1,'deepcopy_hits':1}
 assert random.getstate()==state and paths==sys.path and all(sys.modules.get(k) is v for k,v in aliases.items())
 try:cache.get(runtime_identity={'scope':'WRONG'})
 except CacheInvariantError:pass
 else:raise AssertionError('Runtimemutationmustreject')
 cache._ledger[str(fake_checkpoint)]='0'*64
 try:cache.get(runtime_identity=runtime)
 except CacheInvariantError:pass
 else:raise AssertionError('ContentSHAwrongmustreject')
 def failing_loader(path,digest,final):
  random.random();sys.path.insert(0,'FAKE_NAMESPACE_SENTINEL');sys.modules['model']=object();raise RuntimeError('FAKE_STRICT_LOADER_FAILURE')
 bad=ImmutableParentCache(checkpoint=fake_checkpoint,digest=content_sha(fake_checkpoint),format_version='FAKE',approved_ledger=ledger,runtime_identity=runtime,strict_loader=failing_loader,preserve_rng=rng_context,approve_parent=approve,assert_isolated=isolated)
 try:bad.get(runtime_identity=runtime)
 except RuntimeError as e:assert str(e)=='FAKE_STRICT_LOADER_FAILURE'
 else:raise AssertionError('Strictloaderfailuremustpropagate')
 assert random.getstate()==state and paths==sys.path and all(sys.modules.get(k) is v for k,v in aliases.items()) and bad._private is None
 assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym'))
 print(json.dumps(dict(kind='FAKE_STDLIB_FIXTURE_ONLY',one_miss_one_hit=True,mutable_returns_isolated=True,Python_RNG_namespace_restored=True,runtime_and_contentSHA_mismatch_rejected=True,NN_validation_performed=False,numeric_imports=0,model_loads=0,physical_steps=0),indent=2))
if __name__=='__main__':main()
