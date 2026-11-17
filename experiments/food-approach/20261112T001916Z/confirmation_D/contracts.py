"""Stdlib-only Dproposal, C6 admission and fixedintervention metadata."""
from pathlib import Path
import importlib.util,types,sys,json,hashlib
ROOT=Path(__file__).resolve().parent;C=ROOT.parent/'spatial_C';B=ROOT.parent/'physical_B';AB=ROOT.parent.parent/'20261111T224234Z'
MOTOR_SHA='789f1a31f81fe4415845b076f9351eea6519b6fdfd4b0712ddb80b6c4a20d0dd';SENSOR_SHA='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load_C_runtime():
 previous=sys.modules.get('contracts');cc=module('_D_frozen_C_contracts',C/'contracts.py');sys.modules['contracts']=cc
 try:m=module('_D_frozen_C_runtime',C/'runtime.py')
 finally:
  if previous is None:sys.modules.pop('contracts',None)
  else:sys.modules['contracts']=previous
 return m

def trials():
 proposal=read(ROOT/'PREREG_PROPOSAL.json');out=[]
 for control in ('normal','cut','swap'):
  for cfg in proposal['layouts']:out.append(dict(cfg,layout_id=cfg['id'],id=control+'_'+cfg['id'],control=control))
 out.append(dict(id='D_noodor',layout_id='NOODOR_METADATA_FRONT',forward=.6,angle=0.,control='noodor'))
 for cfg in proposal['replay_trials']:out.append(dict(cfg,layout_id=cfg['id'],control='replay'))
 return out
def config(name):return next(r for r in trials() if r['id']==name)
def admission():
 summary=C/'C_DEVELOPMENT_SUMMARY.json';s=read(summary);require(s['all_integrity_pass'] and s['all_stable_body'],'D_NOT_ADMITTED_C6integrityandstable required');require(len(s['rows'])==6,'C6complete');hashes={str(summary):sha(summary)}
 for row in s['rows']:
  rp=Path(row['result']);ap=Path(row['audit']);r=read(rp);a=read(ap);require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(rp),'Cauditbinding');require(r['functional']['stable_body'],'Cstableall');require(r['checkpoint_sha256']==a['checkpoint_sha256']==MOTOR_SHA and r['behavior_checkpoint_sha256']==a['behavior_checkpoint_sha256']==SENSOR_SHA,'sameCpolicy')
  hashes[str(rp)]=sha(rp);hashes[str(ap)]=sha(ap)
  for p,h in r['source_hashes'].items():require(sha(p)==h,'Csource drift');hashes[p]=h
 for p,h in s['source_hashes'].items():require(sha(p)==h,'Csummary drift');hashes[p]=h
 screen=ROOT/'BASELINE_LAYOUT_SCREEN.json';r=read(screen);require(r['before_any_D_actual'] is True and r['proposal_sha256']==sha(ROOT/'PREREG_PROPOSAL.json') and r['all8_patch_not_naturally_entered'] is True,'parent baseline-only layoutscreen neededbefore Dfreeze');hashes[str(screen)]=sha(screen)
 return hashes

def replay_binding(trial):
 cfg=config(trial)
 if cfg['control']!='replay':return {}
 b=read(ROOT/('REPLAY_BINDING_'+trial+'.json'));require(b['donor']==cfg['donor'] and b['recipient']==trial,'fixedreplay donorbinding')
 for p,h in b['source_hashes'].items():require(sha(p)==h,'donorartifact drift')
 return {**b['source_hashes'],str(ROOT/('REPLAY_BINDING_'+trial+'.json')):sha(ROOT/('REPLAY_BINDING_'+trial+'.json'))}
def verify():
 f=read(ROOT/'FROZEN.json')
 for p,h in f['source_hashes'].items():require(sha(p)==h,'Dsource drift '+p)
 args=sys.argv;trial=args[args.index('--trial')+1] if '--trial' in args else None
 return dict(f,source_hashes={**f['source_hashes'],**(replay_binding(trial) if trial else {})})
