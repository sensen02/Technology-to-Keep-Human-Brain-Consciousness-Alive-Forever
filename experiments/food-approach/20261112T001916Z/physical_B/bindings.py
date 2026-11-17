"""Stdlib-only inherited ABv2 proof and fixed candidate3 binding."""
from pathlib import Path
import importlib.util,sys,json,hashlib,types
ROOT=Path(__file__).resolve().parent;CANDIDATE=ROOT.parent
AB=CANDIDATE.parent/'20261111T224234Z'
OLD=CANDIDATE.parent/'20261112T000308Z/physical_B'
FORMAT='ODOR_PAIRED_BATCH_RECOVERY_V1'
KIND='CANDIDATE3_PAIRED_BEHAVIOR_PHYSICAL_B6_V1'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
BASELINE=CANDIDATE.parent/'20261111T161007Z/combined_turning'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def inherited_v2_proof():
 first=read(AB/'FROZEN.json');extra=read(AB/'FROZEN_SUPPLEMENT.json')
 require(extra['old_frozen_sha256']==sha(AB/'FROZEN.json'),'ABv2 original freeze binding drift')
 require(extra['amendment_sha256']==sha(AB/'AMENDMENT_01_BEFORE_ANY_PHYSICS.json'),'ABv2 amendment drift')
 ledger={}
 for proof in (first,extra):
  for p,h in proof['source_hashes'].items():require(Path(p).is_absolute() and sha(p)==h,'ABv2 inherited source/input drift '+p);ledger[p]=h
 for p in (AB/'FROZEN.json',AB/'FROZEN_SUPPLEMENT.json'):ledger[str(p)]=sha(p)
 result_path=BASELINE/'EVAL_neutral/RESULT.json';audit_path=BASELINE/'VERIFY_neutral.json'
 result=read(result_path);audit=read(audit_path)
 require(audit['integrity_all_pass'] is True and all(v is True for v in audit['checks'].values()),'original baseline independent audit invalid')
 require(result['checkpoint_sha256']==audit['checkpoint_sha256']==MOTOR_SHA and result['engineering_pass'] is True and audit['engineering_pass'] is True,'original baseline motor/engineering binding invalid')
 require(audit['condition']=='neutral' and Path(audit['run']).resolve()==BASELINE/'EVAL_neutral','original baseline audit path invalid')
 if 'result_sha256' in audit:require(audit['result_sha256']==sha(result_path),'baseline result drift')
 for p in (result_path,audit_path):require(str(p) in ledger and ledger[str(p)]==sha(p),'baseline proof missing from ABv2 supplement')
 # Retain every original neutral artifact verified by the frozen baseline audit.
 for key,name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
  p=BASELINE/'EVAL_neutral'/name;require(sha(p)==result[key]['sha256'],'baseline artifact drift '+name);ledger[str(p)]=sha(p)
 for name in ('prepare.py','runtime.py','runner.py','audit.py','summarize_B.py','preflight.py','orchestrate.py','check_codeonly.py'):
  p=OLD/name;ledger[str(p)]=sha(p)
 return dict(source_hashes=ledger)
def clone(fn,g):return types.FunctionType(fn.__code__,g,fn.__name__,fn.__defaults__,fn.__closure__)
