"""Postrun metadata admission only; exact original producer science/globals preserved."""
from pathlib import Path
import ast,hashlib,importlib.util,sys
BASE=Path(__file__).resolve().parent.parent/'final_state_aligned_batch_audit'
BASE_CONTRACTS_SHA='2b3695aef31f1192434b02b2f7308ca077bb6ea300e50f2712c7eaf6de78e0bf'
PRODUCER_PATH=BASE.parent/'state_aligned_action_calibration/wall_replan_9000'
POSTRUN=BASE.parent/'postrun_saved900_qualification_ONLYCODE'
MODE='ORIGINAL_PIPELINE_FAILED124_SAVED900_POSTRUN_QUALIFIED'
def contracts_tree():
    source=BASE/'contracts.py'
    with source.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=BASE_CONTRACTS_SHA:raise ValueError('Frozen base contract source drift')
    tree=ast.parse(source.read_text());changes={key:0 for key in ('PRODUCER','WALL_S','receipt_text','remove_final')};body=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='final_qualification':changes['remove_final']+=1;continue
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key=='PRODUCER':node.value=ast.parse('Path('+repr(str(PRODUCER_PATH))+')',mode='eval').body;changes['PRODUCER']+=1
            elif key=='WALL_S':
                if not isinstance(node.value,ast.Constant) or node.value.value!=600:raise ValueError('Expected base600 budget')
                node.value=ast.Constant(1800);changes['WALL_S']+=1
        for value in ast.walk(node):
            if isinstance(value,ast.Constant) and value.value=='Resource receipt must use the predeclared600s/1800MiB budget':value.value='Resource receipt must use the predeclared postrun1800s/1800MiB budget';changes['receipt_text']+=1
        body.append(node)
    if any(value!=1 for value in changes.values()):raise ValueError('Only registered producerROOT/resource/finalqualification changes allowed')
    return ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
exec(compile(contracts_tree(),str(BASE/'contracts.py')+'::POSTRUN_METADATA_RESOURCE_SCOPE_ONLY','exec'),globals())

POST_STAGES=('FRESH900_SAVED_AUDIT','SAVED900_EVALUATION','INDEPENDENT_SAVED_METRICS')
def post_metadata():
    path=POSTRUN/'metadata.py';spec=importlib.util.spec_from_file_location('_postrun_batch_actual_saved_metadata',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(module.ROOT==POSTRUN and module.ORIGINAL==PRODUCER and module.FORMAT==FORMAT,'Actual postrun metadata producer identity mismatch')
    return module

def final_qualification(checkpoint,digest):
    require(sha_argument(digest),'Explicit actual original900 SHA required, never inferred')
    require(checkpoint==PRODUCER/'TRAIN/checkpoint_000900.pt' and sha(checkpoint)==digest,'Actual original saved900 SHA/path required')
    require(not (POSTRUN/'FAILURE.json').exists(),'Postrun FAILURE prevents admission')
    meta_module=post_metadata();meta=meta_module.qualify(digest)
    require(meta['step']==900 and meta['motor_format']==FORMAT and meta['postrun_training_updates']==meta['postrun_optimizer_operations']==meta['postrun_gradient_operations']==0,'Original saved900-only postrun metadata required')
    require(len(meta['training_rows'])==300 and [row['step'] for row in meta['training_rows']]==list(range(601,901)),'Exactly300 original training rows required')
    binding=read(PRODUCER/'INPUT_BINDINGS.json');pre=read(PRODUCER/'PREFLIGHT.json');proof=reused_preflight_qualification(binding,pre)
    jobs=read(POSTRUN/'JOBS.json');stage=read(POSTRUN/'STAGE_VERIFICATION.json');receipt=read(POSTRUN/'RESOURCE_COMPLETION.json');tr=read(POSTRUN/'TRAIN/RESULT.json')
    require(len(jobs)==3 and tuple(row['stage'] for row in jobs)==POST_STAGES,'All three actual postrun stages required')
    for row in jobs:
        require(row['status']=='COMPLETED' and row['exit_code']==0 and row['command'][:3]==[str(PYTHON),'-B',str(POSTRUN/'runtime.py')] and '--stage' in row['command'] and row['command'][row['command'].index('--stage')+1]==row['stage'] and '--sha256' in row['command'] and row['command'][row['command'].index('--sha256')+1]==digest and '--execute-parent-postrun-stage' in row['command'],'Postrun actual runtime stage/command must complete exit0')
    require(stage['kind']=='POSTRUN_SAVED900_QUALIFICATION_NOT_ORIGINAL_PIPELINE_SUCCESS' and stage['qualification_mode']=='POSTRUN_SAVED_MODEL_ONLY' and stage['all_postrun_stages_exit0'] is True and stage['original_pipeline_all3exit0'] is False,'Never replace original FAILED124 with an all3success claim')
    require(stage['checkpoint']==str(checkpoint) and stage['checkpoint_sha256']==digest and stage['completed_original_total_updates']==900 and stage['completed_original_new_updates']==300 and stage['postrun_training_updates']==stage['optimizer_operations']==stage['physical_steps']==0,'Postrun must not add training or physical steps')
    require(stage['original_deadline_failure']==meta['original_deadline_failure']==tr['original_deadline_failure'],'Original terminal FAILED124 evidence must remain unchanged')
    require(receipt['pass'] is True and receipt['status']=='PASS' and receipt['qualification_mode']=='POSTRUN_SAVED_MODEL_ONLY' and receipt['wall_limit_s']==3600 and receipt['RSS_limit_MiB']==1800 and receipt['wall_s']<3600,'Actual completed postrun3600/RSS1800 receipt required')
    for key in ('all_artifact_hashes_reverified','original_failure_preserved','no_false_original_pipeline_PASS','science_and_success_output_included','failure_bookkeeping_never_qualifies','requires_no_FAILURE_and_actual_run_exit0'):require(receipt[key] is True,'Actual postrun final resource/failure commit missing: '+key)
    require(set(receipt['stage_RSS_MiB'])==set(POST_STAGES) and all(0<=value<=1800 for value in receipt['stage_RSS_MiB'].values()),'All postrun stage RSS values must fit1800')
    require(receipt['postrun_training_updates']==receipt['optimizer_operations']==receipt['Core_constructions']==receipt['physical_steps']==0,'Postrun qualification must be zero-update/noCore')
    require(tr['kind']=='SAVED900_QUALIFIED_NO_NEW_UPDATES' and tr['qualification_mode']=='POSTRUN_SAVED_CHECKPOINT_ONLY' and tr['updates']==tr['total_updates']==900 and tr['new_updates']==300 and tr['already_completed_updates']==600 and tr['qualification_new_updates']==tr['physical_steps']==0 and tr['original_pipeline_success_claim'] is False,'Read wider postrun qualification TRAIN metadata without claiming original pipeline success')
    require(tr['checkpoint']==str(checkpoint) and tr['sha256']==digest and tr['original_training_rows']==300 and tr['original_training_log']==str(PRODUCER/'TRAIN/TRAINING.jsonl') and tr['original_training_log_sha256']==meta['training_journal_sha256'],'Original300 journal and saved900 identity differs')
    observer=tr['strict_payload_observation']
    require(observer['step']==900 and observer['optimizer_states']==8 and observer['all_Adam_counters900'] is True and observer['original_logged_update_rows']==300 and observer['curve_logged_fields_exact'] is True and observer['extra_checkpoint_metrics_only'] is True and observer['strict_payload_identity_and_oldweights'] is True and observer['observer_training_updates']==observer['observer_optimizer_operations']==0 and observer['original_anchor_RNG_validation']=='UNCHANGED_ORIGINAL_STRICT_LOADER','Readonly strictpayload observer must prove curve300/Adam900/format without newoptimizer')
    fresh=read(POSTRUN/'TRAIN/VERIFY_000900.json');expected_model_maps=None
    paths=[POSTRUN/name for name in ('JOBS.json','STAGE_VERIFICATION.json','RESOURCE_COMPLETION.json','TRAIN/RESULT.json','SOURCE_LEDGER.json','INPUT_BINDINGS.json')]
    for step in SAVE_STEPS:
        if step in (600,750):
            v,oldproof=meta_module._saved_proof(step,binding);proof=merge(proof,oldproof)
            require(sha(POSTRUN/'TRAIN'/f'VERIFY_{step:06d}.json')==sha(PRODUCER/'TRAIN'/f'VERIFY_{step:06d}.json'),'Postrun copied original proof must retain exact bytes')
            paths.append(POSTRUN/'TRAIN'/f'VERIFY_{step:06d}.json')
        else:v=fresh
        require(v['step']==step and v['integrity_all_pass'] is True and len(v['checks'])==81 and all_true(v['checks']) and v['checkpoint']==str(PRODUCER/'TRAIN'/f'checkpoint_{step:06d}.pt') and sha(v['checkpoint'])==v['checkpoint_sha256'],'Original600750 and freshpostrun900 strict81 checks required')
        for key in ('strictloader_RNGunchanged','oldweights_frozen','canonical8Adamstates','all_model_weights_unchanged','no_Core_physics_teacherbackend'):require(v['checks'].get(key) is True,'Fresh frozen/RNG/Adam/noCore qualification missing')
        require(len(v['rows'])==20 and sum(row['training'] is True for row in v['rows'])==12,'Actual12TRAIN8DEV saved audit required')
        for i,(row,entry) in enumerate(zip(v['rows'],binding['entries']+binding['dev_entries'])):
            tag=entry['id']+str(i);require(row['id']==entry['id'] and row['actual_rows']==entry['length'] and row['training'] is (i<12) and row['labels_available'] is (i<12),'Fresh saved row identity differs')
            for suffix in ('_production_finite','_actualX_cachedproduction20')+(('_TFproduction','_conditional_fullproduction','_conditional_finite') if i<12 else ()):require(v['checks'].get(tag+suffix) is True,'Actual/conditional per-case proof missing')
            if i<12:require(len(row['conditional_output_errors'])==6 and all(math.isfinite(value) and 0<=value<=GATE for value in row['conditional_output_errors']),'Strict six-output saved parity required')
        if expected_model_maps is None:expected_model_maps=(v['source_hashes'],v['data_hashes'])
        require((v['source_hashes'],v['data_hashes'])==expected_model_maps,'Actual modelconfig source/data must remain identical across600750900')
        item=v['conditional_arrays'];artifact=(POSTRUN if step==900 else PRODUCER)/'TRAIN'/f'CONDITIONAL_{step:06d}.npz'
        require(item['path']==str(artifact) and sha(artifact)==item['sha256'],'Actual fresh conditional/TF artifact path/SHA differs')
        paths.extend((Path(v['checkpoint']),artifact))
    require(fresh['checkpoint_sha256']==digest,'Freshpostrun900 must qualify the explicit original saved900 SHA')
    paths.append(POSTRUN/'TRAIN/VERIFY_000900.json')
    actual_sources,actual_data=fresh['source_hashes'],fresh['data_hashes']
    require(isinstance(actual_sources,dict) and bool(actual_sources) and isinstance(actual_data,dict) and bool(actual_data),'Nonempty actual payload modelconfig maps required')
    for key,actual in (('source_hashes',actual_sources),('data_hashes',actual_data)):
        require(all(tr[key].get(path)==value for path,value in actual.items()),'Wider postrun provenance must retain all actual modelconfig entries')
        verify(tr[key])
    verify(tr['proof_hashes'])
    e=read(POSTRUN/'EVALUATION.json');ev=read(POSTRUN/'EVALUATION_VERIFY.json')
    require(e['kind']=='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1' and e['step']==900 and e['checkpoint']==str(checkpoint) and e['checkpoint_sha256']==digest,'Actual postrun final900 evaluation required')
    require(e['source_hashes']==actual_sources and e['data_hashes']==actual_data,'EVAL must declare original strictpayload maps, not wider qualification-tool maps')
    require(ev['integrity_all_pass'] is True and all_true(ev['checks']) and ev['checkpoint_sha256']==digest and ev['evaluation_sha256']==sha(POSTRUN/'EVALUATION.json'),'Independent saved NumPy final metrics must PASS')
    require(e['arrays']['path']==str(POSTRUN/'EVALUATION_ARRAYS.npz') and sha(e['arrays']['path'])==e['arrays']['sha256'],'Actual postrun evaluation array SHA required')
    require(e['primary_supervision_metric']=='actualX_actionfit' and e['conditional_action_label_distance']=='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION','ActualTF supervision and conditional descriptive-distance semantics required')
    require(len(e['actualX_rows'])==len(e['conditional_rows'])==12 and [row['id'] for row in e['actualX_rows']]==[entry['id'] for entry in binding['entries']]==[row['id'] for row in e['conditional_rows']],'All12 actual/conditional evaluation cases required')
    require(all(row['conditional_action_label_loss_used'] is False and row['distance_semantics']=='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION' for row in e['conditional_rows']),'No conditionalactionlabel supervision claim')
    require(len(e['DEV'])==8 and all(row['labels_available'] is False and row['teacher_MSE'] is None for row in e['DEV']) and e['behavior_pass'] is None and e['new_neck_deployed'] is False,'No new labels/body/neck success')
    paths.extend((POSTRUN/'EVALUATION.json',POSTRUN/'EVALUATION_VERIFY.json',Path(e['arrays']['path'])))
    for name in POST_STAGES:
        p=POSTRUN/(name+'_COMPLETION.json');value=read(p)
        require(value['stage']==name and value['passed'] is True and value['output_ROOT']==str(POSTRUN) and value['source_and_input_unchanged'] is True and value['training_updates']==value['optimizer_operations']==value['Core_constructions']==value['physical_steps']==0 and value['peak_RSS_MiB']<=1800,'All actual postrun stage completion receipts required')
        if name=='FRESH900_SAVED_AUDIT':require(value['AST_changes']=='ONE_FINAL_FALSE_TO_TRUE' and value['strict_payload_observer_readonly'] is True,'Fresh900 must use strengthened original final=True loader')
        paths.append(p)
    expected_outputs={str(POSTRUN/name) for name in ('TRAIN/VERIFY_000900.json','TRAIN/CONDITIONAL_000900.npz','TRAIN/RESULT.json','EVALUATION.json','EVALUATION_ARRAYS.npz','EVALUATION_VERIFY.json','JOBS.json','SOURCE_LEDGER.json')}|{str(POSTRUN/(name+'_COMPLETION.json')) for name in POST_STAGES}
    require(set(stage['outputs'])==expected_outputs and set(receipt['artifacts'])==expected_outputs|{str(POSTRUN/'STAGE_VERIFICATION.json')},'Actual complete postrun commit artifact set required')
    verify(stage['outputs']);verify(receipt['artifacts']);verify(meta['source_hashes']);verify(meta['data_hashes']);verify(meta['proof_hashes'])
    require(not (POSTRUN/'FAILURE.json').exists(),'Postrun failure appeared during qualification')
    # This explicit derived view is in-memory only. Original PT config contains
    # actual model maps; wrapper source is a distinct, wider qualification domain.
    view=dict(tr,source_hashes=actual_sources,data_hashes=actual_data,qualification_source_hashes=tr['source_hashes'],qualification_data_hashes=tr['data_hashes'],qualification_mode=MODE,derived_view_fromproof=dict(path=str(POSTRUN/'TRAIN/RESULT.json'),sha256=sha(POSTRUN/'TRAIN/RESULT.json'),actual_model_maps_from=str(POSTRUN/'TRAIN/VERIFY_000900.json'),actual_model_maps_proof_sha256=sha(POSTRUN/'TRAIN/VERIFY_000900.json'),qualification_source_domain='POSTRUN_TOOL_AND_MODEL_PROVENANCE',actual_model_source_domain='ORIGINAL_STRICT_PAYLOAD_CONFIG'))
    proof=merge(proof,meta['proof_hashes'],ledger(paths),stage['outputs'],receipt['artifacts'],tr['proof_hashes'],tr['source_hashes'],tr['data_hashes'])
    return proof,view
