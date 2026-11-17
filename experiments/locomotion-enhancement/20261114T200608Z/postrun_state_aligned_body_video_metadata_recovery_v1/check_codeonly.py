"""STD only: metadata fixtures, byte equivalence and recursive function globals."""
from pathlib import Path
import ast,builtins,copy,dis,importlib.util,json,math,sys,types
import contract,render,mp4_timing,deliver,freeze
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent.parent/'20261114T010835Z/target_feedback_body_video'

def globals_check(module):
 names=[]
 def visit(code,env):
  for instruction in dis.get_instructions(code):
   if instruction.opname=='LOAD_GLOBAL':
    name=instruction.argval
    if name not in env and not hasattr(builtins,name):raise AssertionError('Missing global '+module.__name__+':'+name)
  for v in code.co_consts:
   if isinstance(v,types.CodeType):visit(v,env)
 for name,value in vars(module).items():
  if isinstance(value,types.FunctionType) and value.__module__==module.__name__:
   if value.__globals__ is not vars(module):raise AssertionError('Foreign globals')
   visit(value.__code__,value.__globals__);names.append(name)
 return names

def main():
 checks={};negative=0
 for name in ('contract.py','render.py','mp4_timing.py','deliver.py','check_codeonly.py','freeze.py'):
  tree=ast.parse((ROOT/name).read_text())
  for node in tree.body:
   if isinstance(node,(ast.Import,ast.ImportFrom)):
    names=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module]
    assert not any(v.split('.')[0] in ('torch','numpy','mujoco','PIL','physics','flygym','model') for v in names)
 checks['AST_and_no_top_level_heavy_imports']=True
 byteequiv={name:contract.sha(ROOT/name) for name in ('render.py','mp4_timing.py')}
 for name,h in byteequiv.items():assert h==contract.sha(OLD/name)
 checks['render_math_and_MP4_math_exact_bytes_to_actual116']=True
 required={k:True for k in ('NN352_shapes','NN352_state_chain','phase_source_labels','native_counts_bound','all71opaque','identity_canonical_digest','finite_PD_actual_torque')};digest='a'*64
 for condition in contract.CASES:
  for n in (15400,14331,771,5801):
   r=dict(kind=contract.FORMAT,condition=condition,command=contract.CASES[condition],checkpoint_sha256=digest,final_step=900,motor_format='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1',state_dims=[128,64,64,96],state_total=352,actual_ms=n,actual_native_steps=n*80,native={'actual_native_steps':n*80},neural_records=math.ceil(n/5),full_duration=n==15400,early_stop=None if n==15400 else {'reason':'SYNTHETIC_METADATA_ONLY'},teacher_calls_after_handover=0,physical_component_only=True,brain_or_sensory_models_loaded=False,neck_ports_present=False,training_updates=0,all71opaque=True,candidate_metrics={'task_pass_candidate':False},original_five={'all_gates':False})
   a=dict(checkpoint_sha256=digest,integrity_all_pass=True,checks=required.copy(),state_total=352)
   t=contract.metadata_contract(r,a,condition,digest);assert render.indices(n)[-1]==n-1 and t['frames']==math.ceil(n/20) and abs(.02*(t['frames']-1)+t['last_frame_ms']/1000-n/1000)<1e-9
   for key,value in [('final_step',600),('motor_format','INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'),('checkpoint_sha256','b'*64),('state_total',256),('kind','OLD_FORMAT')]:
    bad=copy.deepcopy(r);bad[key]=value
    try:contract.metadata_contract(bad,a,condition,digest)
    except ValueError:negative+=1
    else:raise AssertionError('Wrong900 case admitted')
   for key,value in [('integrity_all_pass',False),('checks',{}),('checkpoint_sha256','b'*64)]:
    bad=copy.deepcopy(a);bad[key]=value
    try:contract.metadata_contract(r,bad,condition,digest)
    except ValueError:negative+=1
    else:raise AssertionError('Wrong audit admitted')
   try:contract.metadata_contract(r,a,condition,'UNKNOWN')
   except ValueError:negative+=1
   else:raise AssertionError('Unknown SHA admitted')
 checks['900_format_SHA_352_and_missing_audit_negative_fixtures']=True;checks['all4_behaviorFAIL_full_earlywarm_earlyANN_supported']=True;checks['last_actual_state_and_exact_short_window_math']=True
 for rows in ([],[{'condition':'walk'}],[{'condition':'walk'}]*4,[{'condition':v} for v in reversed(contract.CASES)]):
  try:deliver.require_four_cases(rows)
  except ValueError:negative+=1
  else:raise AssertionError('Incomplete desktop case set admitted')
 deliver.require_four_cases([{'condition':v} for v in contract.CASES]);checks['independent_review_incomplete_delivery_fixed_and_negative_checked']=True
 # PureSTD batch contract imports, only mocked read/hash; no actual RUN result opened.
 spec=importlib.util.spec_from_file_location('_video_std_batch_mock',contract.BATCH/'contracts.py');batch=importlib.util.module_from_spec(spec);sys.modules[spec.name]=batch;spec.loader.exec_module(batch)
 directory=batch.ROOT/('RUN_'+digest);result={'status':'PASS','qualified_final900':True,'checks':{'72':True},'format_version':batch.FORMAT,'final_step':900,'checkpoint_sha256':digest}
 completion={'pass':True,'wall_s':1799.,'RSS_MiB':1700.,'wall_limit_s':1800,'RSS_limit_MiB':1800,'science_and_success_outputs_included':True,'artifacts':{str(directory/name):digest for name in ('VERIFY.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md')}}
 oldread,oldsha=batch.read,batch.sha
 def mockread(path):return copy.deepcopy(result if path.name=='VERIFY.json' else completion)
 batch.read=mockread;batch.sha=lambda path:digest
 assert batch.accept_run(directory) is True
 for target,key,value in [(result,'final_step',600),(result,'qualified_final900',False),(result,'checkpoint_sha256','b'*64),(result,'checks',{}),(completion,'pass',False),(completion,'wall_s',1801.),(completion,'RSS_MiB',1801.),(completion,'artifacts',{}),(completion,'wall_limit_s',9000)]:
  old=target[key];target[key]=value
  try:batch.accept_run(directory)
  except ValueError:negative+=1
  else:raise AssertionError('Bad batch receipt admitted')
  finally:target[key]=old
 batch.read,batch.sha=oldread,oldsha;checks['finalbatch_resource_completion_exact_artifacts_mock_negative']=True
 contract.deadline_contract([{'status':'FAILED','exit_code':124}])
 for jobs in ([],[{'status':'RUNNING'}],[{'status':'COMPLETED','exit_code':0}],[{'status':'FAILED','exit_code':1}],[{'status':'RUNNING'},{'status':'FAILED','exit_code':124}]):
  try:contract.deadline_contract(jobs)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid original pipeline admitted')
 checks['postrun_reject_RUNNING_unknown_non124_or_originalsuccess']=True
 failure={'path':str(contract.CAL/'JOBS.json'),'sha256':digest,'status':'FAILED','exit_code':124}
 fresh={'source_hashes':{'/mock/modelsource':digest},'data_hashes':{'/mock/modeldata':digest}}
 binding=dict(qualification_mode=contract.POST_MODE,original_pipeline_all3exit0=False,postrun_all3stages_exit0=True,postrun_batch72_qualified=True,postrun_training_updates=0,original_failed_124=failure,checkpoint_sha256=digest,CAL_unchanged=str(contract.CAL),external_binding_source=str(contract.BINDER/'prepare.py'),model_source_hashes=fresh['source_hashes'],model_data_hashes=fresh['data_hashes'],source_hashes={**fresh['source_hashes'],'/mock/tool':digest},data_hashes=fresh['data_hashes'])
 assert contract.binding_contract(binding,digest,failure,fresh) is True
 for key,value in [('qualification_mode','ORIGINAL_SUCCESS'),('original_pipeline_all3exit0',True),('postrun_all3stages_exit0',False),('postrun_batch72_qualified',False),('postrun_training_updates',1),('CAL_unchanged',str(contract.POSTRUN)),('model_source_hashes',{'/mock/tool':digest}),('source_hashes',{'/mock/tool':digest}),('original_failed_124',{})]:
  bad=copy.deepcopy(binding);bad[key]=value
  try:contract.binding_contract(bad,digest,failure,fresh)
  except ValueError:negative+=1
  else:raise AssertionError('Wrong postrun model/tool provenance admitted')
 checks['postrun_mode_originalCAL_modelcfg_vs_wider_toolmaps_negatives']=True
 tr=dict(source_hashes={'/mock/model':digest},data_hashes={'/mock/data':digest},qualification_source_hashes={'/mock/tool':digest},qualification_data_hashes={'/mock/data':digest},qualification_mode=contract.BATCH_MODE,derived_view_fromproof=dict(path=str(contract.POSTRUN/'TRAIN/RESULT.json'),sha256=digest,actual_model_maps_from=str(contract.POSTRUN/'TRAIN/VERIFY_000900.json'),actual_model_maps_proof_sha256=digest,qualification_source_domain='POSTRUN_TOOL_AND_MODEL_PROVENANCE',actual_model_source_domain='ORIGINAL_STRICT_PAYLOAD_CONFIG'))
 nested=dict(final_qualification_before={'/mock/proof':digest},source_before={'/mock/tool':digest},source_after={'/mock/tool':digest},final_qualification_after={'/mock/proof':digest},qualification_source_domains=dict(actual_model_source_hashes=tr['source_hashes'],actual_model_data_hashes=tr['data_hashes'],qualification_source_hashes=tr['qualification_source_hashes'],qualification_data_hashes=tr['qualification_data_hashes'],qualification_mode=contract.BATCH_MODE,derived_view_fromproof=tr['derived_view_fromproof']))
 originalsha=contract.sha;contract.sha=lambda p:digest
 try:
  verified=contract.verify_batch_ledger(nested,tr);assert len(verified)==6
  for field in ('qualification_mode','derived_view_fromproof','actual_model_source_hashes'):
   bad=copy.deepcopy(nested);bad['qualification_source_domains'][field]='BAD'
   try:contract.verify_batch_ledger(bad,tr)
   except (ValueError,TypeError):negative+=1
   else:raise AssertionError('Bad nested descriptor admitted')
  for key in ('source_after','final_qualification_after'):
   bad=copy.deepcopy(nested);bad[key]={'/mock/tampered':'b'*64}
   try:contract.verify_batch_ledger(bad,tr)
   except ValueError:negative+=1
   else:raise AssertionError('Nested hash drift admitted')
  bad=copy.deepcopy(nested);bad['unexpected']={}
  try:contract.verify_batch_ledger(bad,tr)
  except ValueError:negative+=1
  else:raise AssertionError('Unknown schema admitted')
 finally:contract.sha=originalsha
 checks['exact_nested_ledger_all_leaf_SHA_and_descriptors_no_skip']=True
 functions={m.__name__:globals_check(m) for m in (contract,render,mp4_timing,deliver,freeze)};checks['recursive_LOAD_GLOBAL_all_function_and_nested_code']=True
 source=(ROOT/'contract.py').read_text();assert 'module.final_qualification' in source and 'module.accept_run' in source and 'producer_context(' not in source
 assert "result['checks'].get('all72_conditional_and_actual_comparisons_complete') is True" in source
 checks['newproducer_STD_finalqualification_and_batch72_no_context']=True
 assert not any(k in sys.modules for k in ('numpy','torch','mujoco','PIL','physics','flygym','model'))
 checks['no_actual_heavy_import_or_future_actual_read']=True
 contract.save(ROOT/'CODECHECK.json',dict(status='ONLYCODE_STATE_ALIGNED900_PASS_ACTUAL_UNKNOWN',checks=checks,negative_mock_cases=negative,byteequiv_sha256=byteequiv,globals_functions=functions,NN_forwards=0,Core_constructions=0,model_compiles=0,physics_steps=0,render_frames=0,PT_payloads_read=0,NPZ_payloads_read=0,future900_body_inputs_read=0,desktop_published=False,claim='Conditional code readiness only, no actual900/model/body/video qualification.'))
 print(json.dumps(dict(status='ONLYCODE_PASS',checks=len(checks),negative_fixtures=negative,byteequiv=True,heavy_imports=0)))
if __name__=='__main__':main()
