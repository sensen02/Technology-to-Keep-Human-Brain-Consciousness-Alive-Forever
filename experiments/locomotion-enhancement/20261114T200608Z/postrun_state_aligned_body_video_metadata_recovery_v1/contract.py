"""Stdlib-only qualification and NPY-header readers. No model/backend imports."""
from pathlib import Path
import ast,hashlib,json,math,struct,zipfile
ROOT=Path(__file__).resolve().parent
BODY=ROOT.parent/'state_aligned_motor_body'
CAL=ROOT.parent/'state_aligned_action_calibration/wall_replan_9000'
BATCH=ROOT.parent/'postrun_final_state_aligned_batch_audit'
POSTRUN=ROOT.parent/'postrun_saved900_qualification_ONLYCODE'
BINDER=ROOT.parent/'postrun_body_binding'
POST_MODE='POSTRUN_SAVED_CHECKPOINT_ONLY'
BATCH_MODE='ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED'
CASES={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
FORMAT='STATE_ALIGNED_ACTION_MOTOR_COMPONENT_BODY_V1'
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
 require(r['final_step']==900 and r['motor_format']=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1','Final900 state-aligned action model')
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

def deadline_contract(jobs):
 require(isinstance(jobs,list) and bool(jobs) and all(row.get('status')!='RUNNING' for row in jobs),'Original RUNNING/missing metadata rejected')
 require(jobs[-1].get('status')=='FAILED' and jobs[-1].get('exit_code')==124,'Actual original terminal FAILED124 required; originalsuccess uses original video branch')
 return True

def binding_contract(b,digest,failure,fresh):
 require(b['qualification_mode']==POST_MODE and b['original_pipeline_all3exit0'] is False and b['postrun_all3stages_exit0'] is True and b['postrun_batch72_qualified'] is True and b['postrun_training_updates']==0,'Explicit postrun-only metadata binding required')
 require(b['original_failed_124']==failure and b['checkpoint_sha256']==digest and b['CAL_unchanged']==str(CAL) and b['external_binding_source']==str(BINDER/'prepare.py'),'Original failed evidence and original CAL model domain required')
 require(b['model_source_hashes']==fresh['source_hashes'] and b['model_data_hashes']==fresh['data_hashes'],'Model maps must come from actual fresh900 strict payload config')
 require(all(b['source_hashes'].get(p)==h for p,h in b['model_source_hashes'].items()) and all(b['data_hashes'].get(p)==h for p,h in b['model_data_hashes'].items()),'Wider provenance must preserve actual model domain')
 return True

def verify_batch_ledger(ledgers,tr):
 # Actual postrun audit.py emits four flat ledgers plus a named domain descriptor.
 expected={'final_qualification_before','source_before','qualification_source_domains','source_after','final_qualification_after'}
 require(isinstance(ledgers,dict) and set(ledgers)==expected,'Exact actual batch ledger top-level schema')
 require(ledgers['source_before']==ledgers['source_after'] and ledgers['final_qualification_before']==ledgers['final_qualification_after'],'Batch before/after identities unchanged')
 out={}
 def add(values):
  verify_map(values)
  for p,h in values.items():require(p not in out or out[p]==h,'Conflicting nested SHA leaf: '+p);out[p]=h
 for key in expected-{'qualification_source_domains'}:add(ledgers[key])
 domains=ledgers['qualification_source_domains']
 maps={'actual_model_source_hashes':'source_hashes','actual_model_data_hashes':'data_hashes','qualification_source_hashes':'qualification_source_hashes','qualification_data_hashes':'qualification_data_hashes'}
 require(isinstance(domains,dict) and set(domains)==set(maps)|{'qualification_mode','derived_view_fromproof'},'Exact named source-domain schema')
 for name,target in maps.items():
  require(domains[name]==tr[target],'Actual domain map differs from qualified view: '+name);add(domains[name])
 require(domains['qualification_mode']==tr['qualification_mode']==BATCH_MODE,'Actual descriptive qualification mode')
 descriptor=domains['derived_view_fromproof']
 required={'path','sha256','actual_model_maps_from','actual_model_maps_proof_sha256','qualification_source_domain','actual_model_source_domain'}
 require(isinstance(descriptor,dict) and set(descriptor)==required and descriptor==tr['derived_view_fromproof'],'Exact derived view descriptor matches qualified source')
 require(descriptor['path']==str(POSTRUN/'TRAIN/RESULT.json') and descriptor['actual_model_maps_from']==str(POSTRUN/'TRAIN/VERIFY_000900.json'),'Derived view exact actual proof paths')
 require(descriptor['qualification_source_domain']=='POSTRUN_TOOL_AND_MODEL_PROVENANCE' and descriptor['actual_model_source_domain']=='ORIGINAL_STRICT_PAYLOAD_CONFIG','Explicit model vs qualification descriptors')
 add({descriptor['path']:descriptor['sha256'],descriptor['actual_model_maps_from']:descriptor['actual_model_maps_proof_sha256']})
 return out

def producer_qualification(digest):
 # Import ONLY the frozen standard-library contract, never producer_context/model modules.
 import importlib.util,sys
 path=BATCH/'contracts.py';spec=importlib.util.spec_from_file_location('_state_video_final900_std_contract',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 require(module.PRODUCER==CAL and module.POSTRUN==POSTRUN and module.MODE==BATCH_MODE and module.WALL_S==1800 and module.RSS_MIB==1800 and module.FORMAT=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1' and module.FINAL_STEP==900,'Exact postrun producer contract and1800budget')
 deadline_contract(read(CAL/'JOBS.json'))
 proofs,tr=module.final_qualification(CAL/'TRAIN/checkpoint_000900.pt',digest)
 directory=BATCH/('RUN_'+digest);require(module.accept_run(directory) is True,'Final900 independent batch run accepted')
 result=read(directory/'VERIFY.json');rows=read(directory/'ROWS.json');ledgers=read(directory/'SOURCE_LEDGER.json')
 require(tr['qualification_mode']==BATCH_MODE and result['qualification_basis']==BATCH_MODE and result['checkpoint']==str(CAL/'TRAIN/checkpoint_000900.pt') and result['checkpoint_sha256']==digest and result['checks'].get('all72_conditional_and_actual_comparisons_complete') is True,'Exact actual postrun900 batch72 proof')
 require(len(rows)==12 and all(row['actual_rows']==1920 and row['old256_conditional_exact'] is True and row['old256_actual_exact'] is True for row in rows),'Batch12 fullactual rows')
 for row in rows:
  for branch,count in (('conditional',6),('actualX_TF',4)):
   require(set(row[branch])=={'B1','B12','micro4'},'All batch modes')
   for errors in row[branch].values():require(len(errors)==count and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1e-12 for v in errors.values()),'Actual six/four strict errors')
 proofs.update(verify_batch_ledger(ledgers,tr))
 for name in ('VERIFY.json','RESOURCE_COMPLETION.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md'):
  p=directory/name;proofs[str(p)]=sha(p)
 for values in (tr['source_hashes'],tr['data_hashes'],tr['qualification_source_hashes'],tr['qualification_data_hashes']):verify_map(values);proofs.update(values)
 proofs[str(path)]=sha(path)
 require(not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','PIL','model')),'Producer qualification stays standard-library only')
 return proofs

def qualify():
 ready=read(ROOT/'READY.json');require(ready['status']=='READYSTD_FROZEN_CONDITIONAL_NO_ACTUAL_BODY_READ','Frozen code-only READY required');verify_map(ready['source_hashes']);verify_map(ready['reference_source_hashes'])
 b=read(BODY/'INPUT_BINDINGS.json');digest=b['checkpoint_sha256'];require(issha(digest) and b['unknownSHA_admitted'] is False,'Actual known model SHA binding required')
 require(Path(b['checkpoint'])==CAL/'TRAIN/checkpoint_000900.pt' and sha(b['checkpoint'])==digest and b['final_step']==900 and b['control_source_candidate']==3 and b['kind']==FORMAT and b['motor_format']=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1','Actual final900/source3 body binding')
 for key in ('source_hashes','data_hashes','proof_hashes'):verify_map(b[key])
 producer_proofs=producer_qualification(digest)
 # External frozen binder exports read-only validation methods that remain valid after body execution.
 import importlib.util,sys
 spec=importlib.util.spec_from_file_location('_postrun_video_external_binding',BINDER/'prepare.py');binder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binder;spec.loader.exec_module(binder)
 post=binder.postrun_qualification(digest);batch=binder.batch_qualification(digest)
 binding_contract(b,digest,post['failure'],read(POSTRUN/'TRAIN/VERIFY_000900.json'))
 ev=read(POSTRUN/'EVALUATION_VERIFY.json');stage=read(POSTRUN/'STAGE_VERIFICATION.json')
 require(passed(ev) and ev['checkpoint_sha256']==digest and stage['checkpoint_sha256']==digest and stage['all_postrun_stages_exit0'] is True and stage['original_pipeline_all3exit0'] is False,'Actual postrun stages only; never originalsuccess')
 for group in (post['qualification_source_hashes'],post['qualification_data_hashes'],post['proof_hashes'],batch['source_hashes'],batch['data_hashes'],batch['proof_hashes']):verify_map(group);producer_proofs.update(group)
 jobs=read(BODY/'JOBS.json');expected=[(c,s) for c in CASES for s in ('PHYSICS','FRESH_NN_SAVED_AUDIT')]+[(None,'SUMMARY')]
 require(len(jobs)==9 and [(x.get('condition'),x['stage']) for x in jobs]==expected and all(x.get('status')=='COMPLETED' and x.get('exit_code')==0 for x in jobs),'Actual all nine body stages required')
 summary=read(BODY/'SUMMARY.json');require(summary['checkpoint_sha256']==digest and jobs[-1]['summary_sha256']==sha(BODY/'SUMMARY.json'),'Actual body summary SHA')
 pre=read(BODY/'PREFLIGHT.json');require(pre['all_passed'] is True and pre['actual_checkpoint_sha256']==digest,'Actual body preflight SHA')
 verify_map(pre['source_hashes']);verify_map(summary['source_hashes'])
 paths=[ROOT/'READY.json',BODY/'INPUT_BINDINGS.json',BODY/'PREFLIGHT.json',BODY/'JOBS.json',BODY/'SUMMARY.json',CAL/'JOBS.json',POSTRUN/'JOBS.json',POSTRUN/'EVALUATION_VERIFY.json',POSTRUN/'STAGE_VERIFICATION.json',POSTRUN/'RESOURCE_COMPLETION.json',BINDER/'prepare.py',BINDER/'CODECHECK.json',ROOT.parent/'FUNCTIONAL_MOTION_REPORT_PREREG.json',Path(b['checkpoint'])]
 records=[];samecompiled=None;all_hashes={**b['source_hashes'],**b['data_hashes'],**b['proof_hashes'],**pre['source_hashes'],**summary['source_hashes'],**producer_proofs}
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
