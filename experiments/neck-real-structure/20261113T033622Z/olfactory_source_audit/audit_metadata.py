"""Stdlib-only existing odor metadata inventory; never imports scientific project modules."""
from pathlib import Path
import ast,hashlib,json,zipfile,struct,sys,os
ROOT=Path(__file__).resolve().parent
PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
FOOD=PROJECT/'experiments/food-approach'
CAND=FOOD/'20261112T001916Z'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def headers(path):
 out={}
 with zipfile.ZipFile(path) as z:
  for entry in z.infolist():
   if not entry.filename.endswith('.npy'):continue
   with z.open(entry) as f:
    if f.read(6)!=b'\x93NUMPY':raise ValueError('Not NPY')
    version=tuple(f.read(2));size=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
    if size>65536:raise ValueError('Header cap')
    h=ast.literal_eval(f.read(size).decode('latin1' if version[0]<3 else 'utf8').strip())
    out[entry.filename[:-4]]=dict(shape=list(h['shape']),dtype=h['descr'],fortran=h['fortran_order'],array_payload_read=False)
 return out

def main():
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});checks={};files={};mismatches=[];hashclaims={}
 def check(k,v):checks[k]=bool(v)
 def bind(path,expected=None):
  path=str(Path(path));actual=files.get(path)
  if actual is None:actual=sha(path);files[path]=actual
  if expected is not None:
   hashclaims[path]=expected
   if actual!=expected:mismatches.append(dict(path=path,expected=expected,actual=actual))
  return actual
 summaries={key:read(CAND/value) for key,value in [('B','physical_B/B_FUNCTIONAL_SUMMARY.json'),('C','spatial_C/C_DEVELOPMENT_SUMMARY.json'),('D','confirmation_D/D_CONFIRMATION_SUMMARY.json')]}
 sourcepaths=[ROOT/'PREREG.json',Path(__file__),FOOD/'20261110T191449Z/steering_behavior.py',FOOD/'20261110T191449Z/odor/food_field.py',FOOD/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json',FOOD/'20261111T224234Z/runtime.py',FOOD/'20261111T224234Z/runner.py',FOOD/'20261111T234150Z/common.py',CAND/'common.py',CAND/'engine.py',CAND/'CURRENT_STATE.json',CAND/'VALIDATION/RESULT.json',CAND/'VALIDATION/VERIFY.json',CAND/'TRAIN/VERIFY_000500.json',CAND/'FROZEN.json',CAND/'physical_B/runtime.py',CAND/'physical_B/bindings.py',CAND/'physical_B/B_FUNCTIONAL_SUMMARY.json',CAND/'spatial_C/spatial.py',CAND/'spatial_C/runtime.py',CAND/'spatial_C/PREREG_PROPOSAL.json',CAND/'spatial_C/C_DEVELOPMENT_SUMMARY.json',CAND/'confirmation_D/contracts.py',CAND/'confirmation_D/runtime.py',CAND/'confirmation_D/PREREG_PROPOSAL.json',CAND/'confirmation_D/D_CONFIRMATION_SUMMARY.json']
 for p in sourcepaths:bind(p)
 ck=CAND/'TRAIN/checkpoint_000500.pt';expected='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034';check('exact_checkpoint_bytes',bind(ck)==expected)
 validation=read(CAND/'VALIDATION/RESULT.json');va=read(CAND/'VALIDATION/VERIFY.json');check('offline_validation_claim_alltargets',validation['all_targets_pass']);check('offline_validation_integrity',va['integrity_all_pass'] and all(va['checks'].values()))
 if 'result_sha256' in va:check('offline_validation_result_SHA',bind(CAND/'VALIDATION/RESULT.json')==va['result_sha256'])
 for name,p in [('candidate_freeze',CAND/'FROZEN.json'),('offline_result',CAND/'VALIDATION/RESULT.json'),('offline_audit',CAND/'VALIDATION/VERIFY.json'),('saved_audit',CAND/'TRAIN/VERIFY_000500.json')]:
  for key in ('source_hashes','data_hashes'):
   for q,h in read(p).get(key,{}).items():bind(q,h)
 for s in summaries.values():
  for p,h in s.get('source_hashes',{}).items():bind(p,h)
 auditpaths={}
 for row in summaries['B']['rows']:auditpaths[row['audit']]='B'
 for row in summaries['C']['rows']:auditpaths[row['audit']]='C'
 for ap in summaries['D']['source_hashes']:
  if ap.endswith('/VERIFY.json'):auditpaths[ap]='D_replay' if '/replay_' in ap else 'D'
 cases=[];inventories={}
 for ap,stage in sorted(auditpaths.items()):
  ap=Path(ap);rp=ap.with_name('RESULT.json');a=read(ap);r=read(rp);bind(ap);bind(rp)
  check('audit_'+str(ap),a['integrity_all_pass'] and all(a['checks'].values()))
  check('result_binding_'+str(ap),a['result_sha256']==bind(rp))
  check('checkpoint_'+str(ap),r.get('behavior_checkpoint_sha256')==expected)
  for q,h in r['source_hashes'].items():bind(q,h)
  artifacts={}
  for key,value in r.items():
   if isinstance(value,dict) and 'path' in value and 'sha256' in value:
    bind(value['path'],value['sha256']);artifacts[key]=dict(value)
  if stage=='B' or r['condition'] in ('normal_L20N','cut_L20N','swap_L20N','D_noodor') or stage=='C':
   for key in ('neural_decisions','geometry'):
    value=r[key];inventories[value['path']]=headers(value['path'])
  cases.append(dict(stage=stage,condition=r['condition'],result=str(rp),audit=str(ap),independent_check_count=len(a['checks']),integrity=a['integrity_all_pass'],source_kind=r.get('source_kind'),engineering_pass=r.get('engineering_pass'),slip=r['autonomous']['slip_ratio_max'],state_total=r.get('state_total'),artifacts=artifacts))
 check('B_six_unique_cases',len(summaries['B']['rows'])==6 and summaries['B']['all_functional_pass'])
 check('C_is_development_not_confirmation',summaries['C']['all_integrity_pass'] and summaries['C']['no_arrival_success_gate_forC'])
 check('D_failure_explicitly_retained',not summaries['D']['all_confirmation_targets_pass'] and summaries['D']['flags']['cut_difference4'] is False)
 check('no_numerical_or_physical_import',not any(k in sys.modules for k in ('torch','numpy','mujoco','physics','student_wave','flygym')))
 check('all_source_and_artifact_claims_match',not mismatches)
 check('all_observed_files_unchanged',all(sha(p)==h for p,h in files.items()))
 report=dict(status='EXISTING_OLFACTORY_METADATA_REVIEW_ONLY_NOT_RESTORED',all_checks_passed=all(checks.values()),checks=checks,checkpoint=dict(path=str(ck),sha256=bind(ck),format='ODOR_PAIRED_BATCH_RECOVERY_V1',step=500,deserialized=False,deployment_qualified=False),interface=dict(source=str(FOOD/'20261110T191449Z/steering_behavior.py'),kind='Learning Linear2to32/tanh/leakyGRU32/head32to1/tanh,notCNN',inputs='actual local left/right receptor response only, [B,2]',hidden32=True,neural_decision_s=.005,neural_leak_tau_s=.1,output='[structural_zero_speed,tanh(neural_turn)]',independent_speed_output=False,old_direct_motor_projection='[relu(turn)/3,turn] from odor output existed, cannot be reused as neck bypass',no_source_geometry_or_phase_as_actor_input=True),transduction=dict(source=str(FOOD/'20261110T191449Z/odor/food_field.py'),head_geometry_source=str(FOOD/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json'),sensor_offsets=read(FOOD/'20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json')['sensor_offsets'],world_field='World-fixed Gaussian evaluated at actual transformed Head proxy XY; environment-only center/amplitude/sigma/onset',world_length_units='native simulation length, SI_NOT_ESTABLISHED',receptor_target='C/(1+C),K1',filter_tau_s=.05,update_s=.005,actual_physical_runner='ClampFilter.advance returns previous pending sample and advances filter; consuming END-interval receptor before interval would be future leak',pending_effective_delay_s=.005,source_onset_ANN_s=.6,source_rise_s=.2,warm_source_off=True,behavior_hidden_warm_zero=True),evidence=dict(offline_validation_status=validation.get('status'),offline_all_targets=validation['all_targets_pass'],B_status=summaries['B']['status'],B_functional_pass=summaries['B']['all_functional_pass'],B_source_kind=summaries['B'].get('source_kind'),B_food_success=summaries['B'].get('food_success'),C_status=summaries['C']['status'],C_arrivals=summaries['C'].get('arrivals'),D_status=summaries['D']['status'],D_all_confirmation_targets_pass=summaries['D']['all_confirmation_targets_pass'],D_flags=summaries['D']['flags'],D_arrival_counts=summaries['D']['arrival_counts'],D_unique_trials=summaries['D']['unique_physical_trials'],D_noodor_one_unique_trial_not_eight_independent_trials=True,D_replay=summaries['D']['replay'],D_no_ingestion_or_parking=summaries['D']['no_food_ingestion_or_parking_claim']),case_reviews=cases,npz_header_inventories=inventories,header_policy='Read ZIP NPY magic/version/header only; never array payload/deserialization',minimal_future_integration=dict(blocked_until='Independent actual visual acceptance; no olfactory restoration/training/physics in current audit',sensory_anatomical_boundary='Head-local antenna receptor sensory signal and frozen odorNN can be brain-side; do not invent ANneck detour',available_scalar='tanh neural_turn from actual receptor2->odorGRU32. Use only neural output as future brain input, not source bearing/gradient/targetposition/time/condition',route='Actual head sampler -> causal receptor filter -> frozen/explicitly-calibrated odorNN brain-side -> artificial brain -> sourceDN2 binding9e88 -> other bodydecoder/motor; body feedback stays sourceAN',new_neck_binding_sha256='9e88a04dd91aa6f9c1c1c68e0562325205e0f36f5cdd9ae41d1dd46c8171dc23',fusion_architecture_and_weights='UNKNOWN_NOT_DEFINED',budget='NOT_DEFINED_NO_CANDIDATE_CREATED',not_a_neural_controller='Gaussian environment/transduction is allowable stimulus model; bilateral concentration gradient steering rule is not a permitted replacement for odorNN or brain'),source_hashes=files,claimed_hashes=hashclaims,hash_mismatches=mismatches,new_model_loads=0,new_training_updates=0,new_physics_steps=0,olfaction_connected=False)
 fresh(ROOT/'OLFACTORY_SOURCE_REVIEW.json',report);fresh(ROOT/'SOURCE_MANIFEST.json',dict(files=files,claimed_hashes=hashclaims,all_bytes_unchanged_at_end=checks['all_observed_files_unchanged'],source_count=len(files)))
 print(json.dumps(dict(status=report['status'],checks=len(checks),all_checks_passed=report['all_checks_passed'],files=len(files),unique_trial_audits=len(cases),npz_headers=len(inventories),mismatches=mismatches),indent=2))
if __name__=='__main__':main()
