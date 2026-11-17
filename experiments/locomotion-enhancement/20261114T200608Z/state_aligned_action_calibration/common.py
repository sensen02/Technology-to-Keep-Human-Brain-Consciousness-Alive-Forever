"""Strictactual41fa600 resume to900. Stdlib-only import, no numericalchildwork."""
from pathlib import Path
import ast,importlib.util,sys,types
ROOT=Path(__file__).resolve().parent
PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4')
BASE=PARENT.parent
PARENT_CHECKPOINT=PARENT/'TRAIN/checkpoint_000600.pt'
PARENT_SHA='41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b'
PARENT_FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
FORMAT='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
SAVE_STEPS=(600,750,900)
STORAGE_SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration/common.py')
BATCH=BASE/'final_batch_parity_audit'/('RUN_'+PARENT_SHA)
SENSITIVITY=BASE/'local_target_feedback_sensitivity'/('RUN_'+PARENT_SHA)
BODY=BASE/'target_feedback_motor_body_v3'
HANDOVER=BASE/'body_training_handover_parity_recovery_v1'

def _module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
_original=_module('_statealigned_exact_v4_storage_definitions',PARENT/'common.py')
g=dict(_original.g)
REBIND_SCIENCE=('_validate_payload','load_checkpoint','save_checkpoint')
# Exact AST replacement of the sole final-step guard, not unrelated600constants.
_tree=ast.parse(STORAGE_SOURCE.read_text());_selected=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in REBIND_SCIENCE]
if len(_selected)!=3:raise ValueError('Exactlythreecanonicalstoragefunctionsrequired')
_guard=next(n for n in _selected if n.name=='_validate_payload');_changes=0
for _n in ast.walk(_guard):
 if isinstance(_n,ast.Compare) and isinstance(_n.left,ast.Name) and _n.left.id=='step' and len(_n.ops)==1 and isinstance(_n.ops[0],ast.NotEq) and len(_n.comparators)==1 and isinstance(_n.comparators[0],ast.Constant) and _n.comparators[0].value==600:
  _n.comparators[0]=ast.Constant(value=900);_changes+=1
if _changes!=1:raise ValueError('Oneexplicitfinal600guardexpected')
for _n in _selected:
 for _c in ast.walk(_n):
  if isinstance(_c,ast.Constant) and _c.value=='Only0/300/600, final600':_c.value='Only600/750/900, final900'
  if isinstance(_c,ast.Constant) and _c.value=='Only0/300/600save':_c.value='Only600/750/900save'
ast.fix_missing_locations(_tree)
exec(compile(ast.Module(body=_selected,type_ignores=[]),str(STORAGE_SOURCE)+'::ONLY_FINAL_GUARD900_NEW_SCOPE','exec'),g)
sha=g['sha'];verify_hashes=g['verify_hashes'];_read=g['_read'];_preserve_rng=g['_preserve_rng'];restore_rng=g['restore_rng']
for _name in ('_validate_rng','restore_rng','_validate_optimizer_state','_validate_live_optimizer','_digest','sha','_read','verify_hashes','_require_neural_only','_module','_preserve_rng'):
 if g[_name] is not _original.g[_name]:raise ValueError('Pure/importedhelperidentitymustbepreserved:'+_name)

def _checked_artifacts(value,proof):
 for item in value.values():
  if not isinstance(item,dict) or set(('path','sha256'))-item.keys():raise ValueError('Actualartifactpath/digestpairrequired')
  if sha(item['path'])!=item['sha256']:raise ValueError('ActualartifactSHAchanged:'+item['path'])
  proof[item['path']]=item['sha256']

def parent_qualification_hashes():
 """Direct completedproofs, actualfailures kept; no model/NPZpayload execution."""
 proof={};paths=[]
 pre=_read(PARENT/'PREFLIGHT.json')
 if not pre['all_passed'] or len(pre['checks'])!=92 or not all(pre['checks'].values()) or pre['timing_status']!='READY_RESUME300_TO600':raise ValueError('Actual92preflightqualificationrequired')
 if sha(PARENT_CHECKPOINT)!=PARENT_SHA:raise ValueError('Actual41fa600SHArequired')
 jobs=_read(PARENT/'JOBS.json');stage=_read(PARENT/'STAGE_VERIFICATION.json');tr=_read(PARENT/'TRAIN/RESULT.json');ev=_read(PARENT/'EVALUATION.json');audit=_read(PARENT/'EVALUATION_VERIFY.json')
 if len(jobs)!=3 or any(r['exit_code']!=0 or r['status']!='COMPLETED' for r in jobs) or not stage['all_stages_exit0'] or stage['train_updates']!=600 or stage['checkpoint_sha256']!=PARENT_SHA or tr['updates']!=600 or tr['sha256']!=PARENT_SHA:raise ValueError('Actualcompletev4three-stage600required')
 if not audit['integrity_all_pass'] or len(audit['checks'])!=44 or not all(audit['checks'].values()) or audit['evaluation_sha256']!=sha(PARENT/'EVALUATION.json') or ev['checkpoint_sha256']!=PARENT_SHA:raise ValueError('Actual44metricproofrequired')
 paths += [PARENT/n for n in ('JOBS.json','STAGE_VERIFICATION.json','TRAIN/RESULT.json','EVALUATION.json','EVALUATION_VERIFY.json','INPUT_BINDINGS.json','PREFLIGHT.json','PREREG.json','common.py','model.py','adapter.py')]
 paths += [PARENT_CHECKPOINT,STORAGE_SOURCE]
 _checked_artifacts({'evaluation_arrays':ev['arrays']},proof)
 for step in (300,450,600):
  p=PARENT/'TRAIN'/f'VERIFY_{step:06d}.json';v=_read(p)
  if v['step']!=step or not v['integrity_all_pass'] or len(v['checks'])!=49 or not all(v['checks'].values()) or len(v['rows'])!=20 or sha(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('Actual49x3fresh12TRAIN8DEVstorageproofrequired')
  paths.extend((p,Path(v['checkpoint'])));_checked_artifacts({'conditional_arrays':v['conditional_arrays']},proof)
 batch=_read(BATCH/'VERIFY.json')
 if batch['status']!='PASS' or not batch['qualified_final600'] or batch['checkpoint_sha256']!=PARENT_SHA or not batch['all36_comparisons_exact'] or len(batch['checks'])!=48 or not all(batch['checks'].values()) or batch['weights_digest_before']!=batch['weights_digest_after'] or batch['physical_behavior_qualified'] is not False:raise ValueError('Actual48batchproof36exactnotbodyPASSrequired')
 _checked_artifacts(batch['artifacts'],proof);paths.append(BATCH/'VERIFY.json')
 ledger=_read(BATCH/'SOURCE_LEDGER.json')
 for before,after in (('final_qualification_before','final_qualification_after'),('v4_source_before','v4_source_after'),('audit_source_before','audit_source_after')):
  if ledger[before]!=ledger[after]:raise ValueError('Actualbatchsourcebeforeaftermismatch')
  verify_hashes(ledger[after]);proof.update(ledger[after])
 sens=_read(SENSITIVITY/'RESULT.json')
 if sens['status']!='COMPLETED' or not sens['diagnostic_integrity_pass'] or sens['checkpoint_sha256']!=PARENT_SHA or len(sens['checks'])!=31 or not all(sens['checks'].values()) or sens['weights_digest_before']!=sens['weights_digest_after'] or sens['root_cause_established'] is not False or sens['fullbody_stability_established'] is not False or any(row['amplification_flag'] for row in sens['points']):raise ValueError('Actual31localdiagnosticnoamplificationnotcausalityproofrequired')
 if sens['training_updates']!=0 or sens['Core_constructions']!=0 or sens['physical_steps']!=0:raise ValueError('Sensitivityreadonlyscopechanged')
 _checked_artifacts(sens['artifacts'],proof);_checked_artifacts({'FD_arrays':sens['FD_arrays']},proof);paths.extend((SENSITIVITY/'RESULT.json',SENSITIVITY/'REPORT.zh-CN.md'))
 completion=_read(SENSITIVITY/'RESOURCE_COMPLETION.json')
 if not completion['pass'] or not completion['science_and_success_output_included'] or not completion['failure_bookkeeping_never_qualifies'] or completion['wall_s']>240 or completion['RSS_MiB']>512:raise ValueError('Actualsensitivityfinalresourcecommitrequired')
 verify_hashes(completion['artifacts']);proof.update(completion['artifacts']);paths.append(SENSITIVITY/'RESOURCE_COMPLETION.json')
 ledger=_read(SENSITIVITY/'SOURCE_LEDGER.json')
 for before,after in (('proof_before','proof_after'),('source_before','source_after')):
  if ledger[before]!=ledger[after]:raise ValueError('Sensitivitysourcebeforeaftermismatch')
  selected={p:h for p,h in ledger[after].items() if Path(p).parent in (PARENT,SENSITIVITY.parent,BODY,STORAGE_SOURCE.parent)} if after=='source_after' else ledger[after]
  verify_hashes(selected);proof.update(selected)
 summary=_read(BODY/'SUMMARY.json')
 if len(summary['rows'])!=4 or any(r['candidate_metrics']['task_pass_candidate'] for r in summary['rows']):raise ValueError('PreserveactualfourbodyFAIL')
 paths.extend((BODY/'SUMMARY.json',BODY/'JOBS.json'))
 for condition in ('walk','stop','left_pivot','right_pivot'):
  folder=BODY/'trials'/condition;r=_read(folder/'RESULT.json');v=_read(folder/'VERIFY.json')
  if r['checkpoint_sha256']!=PARENT_SHA or r['candidate_metrics']['task_pass_candidate'] is not False or not v['integrity_all_pass'] or not all(v['checks'].values()) or v['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('ActualqualifiedbodyarchiveFAILrequired')
  paths.extend((folder/'RESULT.json',folder/'VERIFY.json',folder/'IDENTITY.json'))
  _checked_artifacts({key:r[key] for key in ('trace','neural_decisions','handover','initial_state')},proof)
 hand=_read(HANDOVER/'RESULT.json');hv=_read(HANDOVER/'VERIFY.json')
 if not hv['integrity_all_pass'] or not all(hv['checks'].values()) or hv['result_sha256']!=sha(HANDOVER/'RESULT.json') or hv['report_sha256']!=sha(HANDOVER/'REPORT.zh-CN.md') or hand['checkpoint_sha256']!=PARENT_SHA or len(hand['rows'])!=4 or hand['NN_forwards']!=0 or hand['physical_steps']!=0:raise ValueError('Actualhandoverreadonlyqualificationrequired')
 paths.extend((HANDOVER/'RESULT.json',HANDOVER/'VERIFY.json',HANDOVER/'REPORT.zh-CN.md'))
 for key in ('source_hashes','data_hashes','proof_hashes'):
  # Bind the full qualifiedhandoverRESULT bySHA; rehash directscience inputs,
  # not thousands of already-qualified environment/ancestor packagefiles.
  selected=hand[key] if key=='proof_hashes' else {p:h for p,h in hand[key].items() if Path(p).parent in (PARENT,HANDOVER,BODY,STORAGE_SOURCE.parent)}
  verify_hashes(selected);proof.update(selected)
 for p in paths:proof[str(p)]=sha(p)
 return proof
resume_qualification_hashes=parent_qualification_hashes

def load_resume_parent():
 parent_qualification_hashes();sentinel=object();aliases={k:sys.modules.get(k,sentinel) for k in ('model','common','sources','adapter')};paths=list(sys.path)
 try:
  sys.path.insert(0,str(PARENT));sys.modules['adapter']=_module('_statealigned_actual_v4_adapter',PARENT/'adapter.py');loader=_module('_statealigned_actual_strict_v4_io',PARENT/'common.py');m,p=loader.load_checkpoint(PARENT_CHECKPOINT,PARENT_SHA,final=True)
 finally:
  sys.path[:]=paths
  for k,v in aliases.items():
   if v is sentinel:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 if p['step']!=600 or p['format_version']!=PARENT_FORMAT or 'initial_anchor' not in p['config']:raise ValueError('Strictactual600originalanchorrequired')
 return m,p

def initialize():
 with _preserve_rng():
  parent,p=load_resume_parent();backend=_module('_statealigned_resume_deployment',ROOT/'model.py');m=backend.make_model(parent)
 return m,p

def _check_config(config):
 if not isinstance(config,dict):raise ValueError('Configdictrequired')
 for key in ('source_hashes','data_hashes'):verify_hashes(config.get(key))
 for path in (ROOT/'model.py',ROOT/'common.py',PARENT/'model.py',PARENT/'common.py',g['OLD_DIR']/'model.py',g['OLD_DIR']/'model_io.py'):
  if config['source_hashes'].get(str(path))!=sha(path):raise ValueError('Newandstrictparentsourcebindingrequired')
 if config['data_hashes'].get(str(PARENT_CHECKPOINT))!=PARENT_SHA or config['data_hashes'].get(str(g['OLD_CHECKPOINT']))!=g['OLD_SHA']:raise ValueError('Actual600and789fbindingsrequired')
 for path,h in parent_qualification_hashes().items():
  if config['data_hashes'].get(path)!=h:raise ValueError('Actualparentqualificationproofmissing:'+path)
 _,p=load_resume_parent()
 import torch
 a=config.get('initial_anchor');b=p['config']['initial_anchor']
 if not isinstance(a,dict) or a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in b):raise ValueError('Originalfcbdanchorchanged')
 return True
g.update(ROOT=ROOT,SAVE_STEPS=SAVE_STEPS,FORMAT=FORMAT,initialize=initialize,_check_config=_check_config)
load_checkpoint=g['load_checkpoint'];save_checkpoint=g['save_checkpoint']
