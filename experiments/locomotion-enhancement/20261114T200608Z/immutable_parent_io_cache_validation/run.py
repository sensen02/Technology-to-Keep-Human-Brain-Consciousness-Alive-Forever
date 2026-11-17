"""Fixednew600 IO-only validation. Parentexec onlyafter144 resources released."""
from pathlib import Path
import argparse,copy,importlib.util,json,os,resource,signal,sys,time
from engine import sha,verify,ObservedLedger,ValidatedParent,namespace_guard,ALIASES
ROOT=Path(__file__).resolve().parent
PRODUCER=ROOT.parent/'state_aligned_action_calibration/wall_replan_9000'
CHECKPOINT=PRODUCER/'TRAIN/checkpoint_000600.pt'
DIGEST='f95d759e1b28aee0a4224615d912bdf0e95490acd9629e4752ce8b89108c26ce'
PYTHON='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python'

def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def metadata():
 proof=read(PRODUCER/'TRAIN/VERIFY_000600.json');binding=read(PRODUCER/'INPUT_BINDINGS.json')
 if sha(CHECKPOINT)!=DIGEST or not proof['integrity_all_pass'] or len(proof['checks'])!=81 or not all(proof['checks'].values()) or proof['checkpoint_sha256']!=DIGEST or proof['step']!=600:raise ValueError('Actualnew60081freshchecksrequired')
 ledger={**binding['source_hashes'],**binding['data_hashes'],str(CHECKPOINT):DIGEST}
 for p in (PRODUCER/'INPUT_BINDINGS.json',PRODUCER/'PREFLIGHT.json',PRODUCER/'TRAIN/VERIFY_000600.json',ROOT/'PREREG.json',ROOT/'engine.py',ROOT/'checks.py',ROOT/'run.py',ROOT/'README.zh-CN.md',ROOT/'CODECHECK.json'):ledger[str(p)]=sha(p)
 verify(ledger);return ledger

def selfcheck():
 import ast,contextlib
 for p in ROOT.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
 sentinel=object();previous=(sentinel,list(sys.path),{k:sys.modules.get(k,sentinel) for k in ALIASES})
 try:
  with namespace_guard():
   sys.path.insert(0,'FAKE_ALIAS_FIXTURE')
   for key in ALIASES:sys.modules[key]=object()
   raise ValueError('fixture')
 except ValueError:pass
 sentinel,paths,aliases=previous
 assert paths==sys.path and all(sys.modules.get(k,sentinel) is v for k,v in aliases.items())
 assert not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','flygym'))
 ledger=metadata()
 observer=ObservedLedger()
 fixtureledger={str(ROOT/'engine.py'):sha(ROOT/'engine.py'),str(ROOT/'PREREG.json'):sha(ROOT/'PREREG.json')}
 with observer.capture():verify(fixtureledger)
 assert observer.maps>=1 and observer.sha_returns>=2 and all(observer.ledger[p]==h for p,h in fixtureledger.items()) and sys.getprofile() is None
 try:verify({str(ROOT/'PREREG.json'):'0'*64})
 except ValueError:pass
 else:raise AssertionError('WrongSHAfixturemustreject')
 print(json.dumps(dict(kind='STD_METADATA_SELFCHECK_ONLY',sourceledger_entries=len(ledger),aliases_exception_fixture=True,numeric_imports=0,NN_calls=0,models_loaded=0)))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--selfcheck',action='store_true');parser.add_argument('--execute-parent-io-validation',action='store_true');args=parser.parse_args()
 if args.selfcheck:selfcheck();return
 if not args.execute_parent_io_validation:parser.error('ParentexclusiveIOvalidationflagrequired')
 if sys.executable!=PYTHON:raise ValueError('Correct125712Pythonrequired')
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
  if os.environ.get(key)!='1':raise ValueError(key+'=1required')
 start=time.monotonic();ledger=metadata();out=ROOT/'RUN_FIXED_F95D600';out.mkdir(exist_ok=False)
 def cap():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
  if rss>1800 or time.monotonic()-start>3600:raise RuntimeError('IOvalidation3600s1800MiBbudgetexceeded')
  return rss
 def deadline(*unused):raise TimeoutError('IOvalidationhardwall3600expired')
 oldalarm=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(.001,3600-(time.monotonic()-start)))
 oldpath=list(sys.path);sentinel=object();oldaliases={k:sys.modules.get(k,sentinel) for k in ALIASES};producer=None;original=None;initialrng=None
 try:
  import torch,numpy as np
  torch.set_num_threads(1);torch.set_num_interop_threads(1);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
  from checks import fingerprint,model_payload_equal,assert_isolated,approve_parent,callable_identity,runtime_identity,rng_snapshot,restore_rng
  runtime=runtime_identity();initialrng=rng_snapshot();spec=importlib.util.spec_from_file_location('_IO_VALIDATION_ORIGINAL_PRODUCER',PRODUCER/'common.py');producer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=producer;spec.loader.exec_module(producer);original=producer.load_resume_parent
  baseline_observer=ObservedLedger();timer=time.monotonic()
  with namespace_guard(),baseline_observer.capture():baseline=producer.load_checkpoint(CHECKPOINT,DIGEST,final=False)
  uncached_seconds=time.monotonic()-timer;cap();baseline_observer.bind_loaded_sources()
  original_functions={name:producer.g[name] for name in ('_check_config','initialize','_validate_payload','load_checkpoint','save_checkpoint')}
  ledger.update(baseline_observer.ledger)
  # Explicit payloadconfig maps are bound inaddition to every actual verifiermap.
  for field in ('source_hashes','data_hashes'):ledger.update(baseline[1]['config'][field])
  verify(ledger);baseline_rng=fingerprint(rng_snapshot());checks={'baseline_rng_preserved':baseline_rng==fingerprint(initialrng),'baseline_noNN':baseline_observer.neural_calls==0}
  cache=ValidatedParent(original,ledger,runtime,producer._preserve_rng,approve_parent,assert_isolated,fingerprint)
  producer.load_resume_parent=cache
  timer=time.monotonic();cached_observer=ObservedLedger()
  # Nested observers chain only our own hooks; everyoriginal verifierstillruns.
  with namespace_guard(),cached_observer.capture():cached=producer.load_checkpoint(CHECKPOINT,DIGEST,final=False)
  cached_seconds=time.monotonic()-timer;cap();ledger_before=dict(cache.ledger);checks['original_validators_initialize_objects_unchanged']=all(producer.g[k] is v for k,v in original_functions.items());checks['cached_noNN_observer']=cached_observer.neural_calls==0;checks['full_new600_payload_model_exact']=model_payload_equal(baseline,cached);checks['public_callable_source_identity']=callable_identity(baseline[0])==callable_identity(cached[0]);checks['runtime_unchanged']=runtime_identity()==runtime
  checks['cached_rng_preserved']=fingerprint(rng_snapshot())==fingerprint(initialrng);assert_isolated(baseline,cached)
  timer=time.monotonic();first=cache();hit_seconds=time.monotonic()-timer
  with torch.no_grad():
   parameter=next(first[0].adapter_parameters().__iter__());parameter.reshape(-1)[0]+=1.
   slot=next(iter(first[1]['optimizer_state_dict']['state'].values()));slot['exp_avg'].reshape(-1)[0]+=1.
   anchor=next(iter(first[1]['config']['initial_anchor'].values()));anchor.reshape(-1)[0]+=1.
   first[1]['curve'][-1]['step']=-999;first[1]['rng']['torch'][0]^=1
  second=cache();checks['mutating_return_does_not_change_private_next_get']=model_payload_equal(second,cache.private);assert_isolated(first,second);assert_isolated(cache.private,second)
  dummy=out/'dummy_source_fixture.txt';dummy.write_text('bound fixture\n');cache.ledger[str(dummy)]=sha(dummy)
  try:
   dummy.write_text('changed fixture\n')
   try:cache()
   except ValueError:checks['dummy_source_change_rejected']=True
   else:checks['dummy_source_change_rejected']=False
  finally:
   dummy.write_text('bound fixture\n');verify(cache.ledger);cache.ledger.pop(str(dummy));dummy.unlink()
  before_rng=fingerprint(rng_snapshot());before_path=list(sys.path);before_alias={k:sys.modules.get(k,sentinel) for k in ALIASES}
  try:
   with namespace_guard(),producer._preserve_rng():
    torch.rand(1);np.random.random();sys.path.insert(0,'EXCEPTION_FIXTURE')
    for key in ALIASES:sys.modules[key]=object()
    raise ValueError('validationfixture')
  except ValueError:pass
  checks['exception_alias_path_RNG_restored']=before_path==sys.path and all(sys.modules.get(k,sentinel) is v for k,v in before_alias.items()) and before_rng==fingerprint(rng_snapshot())
  checks['one_original_strict_parent_miss']=cache.strict_parent_calls==1 and cache.misses==1;checks['noNN_noCore_imports']=not any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave'))
  verify(cache.ledger);verify(ledger);ledger_after={p:sha(p) for p in cache.ledger};checks['source_ancestor_proof_content_before_after']=ledger_before==ledger_after;rss=cap();result=dict(kind='IO_CACHE_FIXED_NEW600_EQUIVALENCE_ONLY',all_passed=bool(all(checks.values())),checks=checks,checkpoint=str(CHECKPOINT),checkpoint_sha256=DIGEST,producer=str(PRODUCER),uncached_parent_calls=baseline_observer.parent_calls,cached_original_parent_calls=cache.strict_parent_calls,cache_misses=cache.misses,cache_hits=cache.hits,cache_identity_key=cache.identity_key(),uncached_strictload_s=uncached_seconds,cached_strictload_s=cached_seconds,isolated_parent_hit_s=hit_seconds,wall_s=time.monotonic()-start,peak_RSS_MiB=rss,runtime=runtime,ledger=cache.ledger,source_before=ledger_before,source_after=ledger_after,actual_verifier_capture=cache.capture_report,NN_forwards=0,optimizer_updates=0,Core_constructions=0,physical_steps=0,qualifies900=False,new81sequence_replay_performed=False)
  save(out/'RESULT.json',result);save(out/'COMPLETION.json',dict(status='PASS' if result['all_passed'] else 'FAIL',result_sha256=sha(out/'RESULT.json'),source_after_verified=True,wall_s=time.monotonic()-start,NN_forwards=0,optimizer_updates=0,qualifies900=False))
  if not result['all_passed']:raise ValueError('IOcacheequivalencefailed')
 except BaseException as exc:
  if not (out/'FAILURE.json').exists():save(out/'FAILURE.json',dict(status='FAIL',exception_type=type(exc).__name__,message=str(exc),wall_s=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,NN_replay_performed=False,optimizer_updates=0,Core_constructions=0,physical_steps=0,qualifies900=False))
  if not (out/'COMPLETION.json').exists():save(out/'COMPLETION.json',dict(status='FAIL',failure_sha256=sha(out/'FAILURE.json'),source_after_verified=False,qualification_not_completed=True,qualifies900=False))
  raise
 finally:
  if initialrng is not None:restore_rng(initialrng)
  if producer is not None and original is not None:producer.load_resume_parent=original
  sys.path[:]=oldpath
  for k,v in oldaliases.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldalarm)
if __name__=='__main__':main()
