"""Stdlib C-admission and fixed metadata; no physics/numerical imports."""
from pathlib import Path
import json,hashlib,importlib.util,sys,types
ROOT=Path(__file__).resolve().parent;B=ROOT.parent/'physical_B';AB=ROOT.parent.parent/'20261111T224234Z'
SENSOR_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034';MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def admission():
 summary=B/'B_FUNCTIONAL_SUMMARY.json';s=read(summary);require(s['all_functional_pass'] is True,'C_NOT_ADMITTED: complete B allfunctionalPASS required')
 hashes={str(summary):sha(summary)};require(len(s['rows'])==6,'all6Brequired')
 for row in s['rows']:
  require(row['functional_pass'] is True,'BfunctionalFAIL');rp=Path(row['result']);ap=Path(row['audit']);r=read(rp);a=read(ap)
  require(r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['behavior_checkpoint_sha256']==a['behavior_checkpoint_sha256']==SENSOR_SHA,'samecandidateBrequired')
  require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(rp),'Bactualauditbinding')
  for p,h in r['source_hashes'].items():require(sha(p)==h,'Bsourceidentitydrift '+p);hashes[p]=h
  for k,n in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
   p=rp.parent/n;require(sha(p)==r[k]['sha256'],'Bartifactdrift');hashes[str(p)]=sha(p)
  hashes[str(rp)]=sha(rp);hashes[str(ap)]=sha(ap)
 for p,h in s['source_hashes'].items():require(sha(p)==h,'Bsummaryidentity');hashes[p]=h
 return hashes

def verify():
 f=read(ROOT/'FROZEN.json')
 for p,h in f['source_hashes'].items():require(sha(p)==h,'Cimmutableinputdrift '+p)
 require(read(ROOT/'INPUT_BINDINGS_C.json')['behavior_sha256']==SENSOR_SHA,'fixedCbehavior')
 return f
