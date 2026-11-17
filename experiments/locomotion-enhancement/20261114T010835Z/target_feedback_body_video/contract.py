"""Stdlib-only qualification and NPY-header readers. No model/backend imports."""
from pathlib import Path
import ast,hashlib,json,math,struct,zipfile
ROOT=Path(__file__).resolve().parent
BODY=ROOT.parent/'target_feedback_motor_body_v3'
CAL=ROOT.parent/'target_feedback_order_repair_v4'
CASES={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
FORMAT='TARGET_FEEDBACK_ORDER_REPAIR_MOTOR_COMPONENT_BODY_V1'
XML=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(v,message):
 if not v:raise ValueError(message)
def issha(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def verify_map(m):
 require(isinstance(m,dict) and bool(m),'Nonempty source map required')
 for p,h in m.items():require(Path(p).is_absolute() and issha(h) and sha(p)==h,'Hash identity drift: '+p)
def passed(a):return a.get('integrity_all_pass') is True and isinstance(a.get('checks'),dict) and bool(a['checks']) and all(x is True for x in a['checks'].values())
def headers(path):
 out={}
 with zipfile.ZipFile(path) as z:
  for name in z.namelist():
   require(name.endswith('.npy') and '/' not in name,'Only flat NPY members')
   with z.open(name) as f:
    require(f.read(6)==b'\x93NUMPY','NPY magic');version=tuple(f.read(2));size=2 if version==(1,0) else 4
    require(version in ((1,0),(2,0),(3,0)),'NPY version')
    length=int.from_bytes(f.read(size),'little');require(0<length<=65536,'Bounded NPY header')
    h=ast.literal_eval(f.read(length).decode('utf8' if version==(3,0) else 'latin1'))
    require(set(h)=={'descr','fortran_order','shape'} and not isinstance(h['descr'],list) and 'O' not in h['descr'],'No object/structured arrays')
    out[name[:-4]]=dict(shape=list(h['shape']),descr=h['descr'],fortran_order=h['fortran_order'])
 return out

def metadata_contract(r,a,condition,digest):
 require(issha(digest),'Unknown final model SHA rejected')
 require(r['kind']==FORMAT and r['condition']==condition and r['command']==CASES[condition],'Case/command identity')
 require(r['checkpoint_sha256']==a['checkpoint_sha256']==digest and passed(a),'Actual full saved audit required')
 require(r['final_step']==600 and r['motor_format']=='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1','Final600 targetfeedback model')
 require(r['state_dims']==[128,64,64,96] and r['state_total']==a['state_total']==352,'Explicit352 ABI')
 n=r['actual_ms'];require(type(n) is int and 0<n<=15400 and r['actual_native_steps']==n*80==r['native']['actual_native_steps'],'Actual finite duration/native counts')
 require(r['neural_records']==math.ceil(n/5),'Actual neural records')
 require(r['full_duration'] is (n==15400 and r['early_stop'] is None),'Full/short window identity')
 require(n==15400 or r['early_stop'] is not None,'Short window must record actual early stop')
 require(r['teacher_calls_after_handover']==0 and r['physical_component_only'] is True and r['brain_or_sensory_models_loaded'] is False and r['neck_ports_present'] is False and r['training_updates']==0 and r['all71opaque'] is True,'Pure motor component scope')
 for key in ('NN352_shapes','NN352_state_chain','phase_source_labels','native_counts_bound','all71opaque','identity_canonical_digest','finite_PD_actual_torque'):
  require(a['checks'].get(key) is True,'Required saved audit '+key)
 # No task pass or legacy pass requirement. Failed gates and unsafe stops are renderable.
 return dict(samples=n,physical_seconds=n/1000,frames=math.ceil(n/20),last_frame_ms=n%20 or 20,ANN_seconds=max(0,n-5800)/1000)

def qualify():
 ready=read(ROOT/'READY.json');require(ready['status']=='READYSTD_FROZEN_CONDITIONAL_NO_ACTUAL_BODY_READ','Frozen code-only READY required');verify_map(ready['source_hashes']);verify_map(ready['reference_source_hashes'])
 b=read(BODY/'INPUT_BINDINGS.json');digest=b['checkpoint_sha256'];require(issha(digest) and b['unknownSHA_admitted'] is False,'Actual known model SHA binding required')
 require(Path(b['checkpoint'])==CAL/'TRAIN/checkpoint_000600.pt' and sha(b['checkpoint'])==digest and b['final_step']==600 and b['control_source_candidate']==3,'Actualv4 final600/source3')
 for key in ('source_hashes','data_hashes','proof_hashes'):verify_map(b[key])
 caljobs=read(CAL/'JOBS.json');require(len(caljobs)==3 and all(x.get('status')=='COMPLETED' and x.get('exit_code')==0 for x in caljobs),'Actual three calibration stages exit0')
 ev=read(CAL/'EVALUATION_VERIFY.json');stage=read(CAL/'STAGE_VERIFICATION.json')
 require(passed(ev) and ev['checkpoint_sha256']==digest and stage['checkpoint_sha256']==digest and stage['all_stages_exit0'] is True,'Actual final600/stage/finalbatch audit')
 jobs=read(BODY/'JOBS.json');expected=[(c,s) for c in CASES for s in ('PHYSICS','FRESH_NN_SAVED_AUDIT')]+[(None,'SUMMARY')]
 require(len(jobs)==9 and [(x.get('condition'),x['stage']) for x in jobs]==expected and all(x.get('status')=='COMPLETED' and x.get('exit_code')==0 for x in jobs),'Actual all nine body stages required')
 summary=read(BODY/'SUMMARY.json');require(summary['checkpoint_sha256']==digest and jobs[-1]['summary_sha256']==sha(BODY/'SUMMARY.json'),'Actual body summary SHA')
 pre=read(BODY/'PREFLIGHT.json');require(pre['all_passed'] is True and pre['actual_checkpoint_sha256']==digest,'Actual body preflight SHA')
 verify_map(pre['source_hashes']);verify_map(summary['source_hashes'])
 paths=[ROOT/'READY.json',BODY/'INPUT_BINDINGS.json',BODY/'PREFLIGHT.json',BODY/'JOBS.json',BODY/'SUMMARY.json',CAL/'JOBS.json',CAL/'EVALUATION_VERIFY.json',CAL/'STAGE_VERIFICATION.json',Path(b['checkpoint'])]
 records=[];samecompiled=None;all_hashes={**b['source_hashes'],**b['data_hashes'],**b['proof_hashes'],**pre['source_hashes'],**summary['source_hashes']}
 for condition in CASES:
  folder=BODY/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');timing=metadata_contract(r,a,condition,digest)
  require(a['result_sha256']==sha(folder/'RESULT.json') and Path(r['checkpoint'])==Path(b['checkpoint']),'Result audit/model binding')
  verify_map(r['source_hashes']);ident=read(folder/'IDENTITY.json');canonical={k:v for k,v in ident.items() if k!='identity_sha256'}
  require(hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ident['identity_sha256']==r['identity_sha256'],'Canonical identity')
  verify_map({p:x['sha256'] for p,x in ident['files'].items()});all_hashes.update({p:x['sha256'] for p,x in ident['files'].items()});all_hashes.update(r['source_hashes']);require(str(XML) in ident['files'],'Actual derived XML identity required')
  require(ident['physics']['nq']==49 and ident['physics']['nv']==48 and ident['physics']['nu']==42 and ident['adhesion']['enabled'] is False,'Original body ABI/noadhesion')
  require(len(ident['actuator']['kp'])==len(ident['actuator']['kd'])==len(ident['actuator']['torque_limit'])==42,'42 finite PD metadata')
  require(all(math.isfinite(x) and x>0 for x in ident['actuator']['torque_limit']),'Finite positive torque limits')
  require(samecompiled is None or samecompiled==ident['compiled_arrays'],'Four cases same compiled canonical model');samecompiled=ident['compiled_arrays']
  for key in ('trace','neural_decisions','runtime_geometry','initial_state','rng_initial','geometry','handover','body_length_artifact'):
   if key in r:
    item=r[key];p=Path(item['path']);require(p.parent==folder and sha(p)==item['sha256'],'Actual artifact '+key);paths.append(p)
  h=headers(folder/'trace_1ms.npz');n=timing['samples']
  for name,shape in [('qpos',[n,49]),('qvel',[n,48]),('actualctrl',[n,42]),('time',[n]),('com',[n,3]),('loads',[n,6]),('control_source',[n])]:require(h[name]['shape']==shape,'Trace schema '+name)
  paths.extend([folder/'RESULT.json',folder/'VERIFY.json',folder/'IDENTITY.json']);records.append(dict(condition=condition,folder=str(folder),result=r,audit=a,identity=ident,timing=timing))
 require(sum(x['result']['actual_native_steps'] for x in records)==summary['actual_new_native_steps'],'Actual four case native total')
 all_hashes.update({str(p):sha(p) for p in paths});verify_map(all_hashes)
 return dict(checkpoint_sha256=digest,records=records,source_hashes=all_hashes,source_candidate=3,NN_forwards=0,Core_constructions=0,physics_steps=0)
