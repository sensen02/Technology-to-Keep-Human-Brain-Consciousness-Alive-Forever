"""Only STD AST/source-contract/future-metadata fixtures; no scientific imports."""
from pathlib import Path
import ast,builtins,copy,dis,importlib.util,json,sys,types
import contracts as C
ROOT=C.ROOT

def globals_ok(function):
    def visit(code):
        for op in dis.get_instructions(code):
            if op.opname=='LOAD_GLOBAL':assert op.argval in function.__globals__ or op.argval in builtins.__dict__,(function.__name__,op.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):visit(value)
    visit(function.__code__)
    if isinstance(getattr(function,'__wrapped__',None),types.FunctionType):globals_ok(function.__wrapped__)

def equivalence():
    spec=importlib.util.spec_from_file_location('_postrun_audit_noexecute_tree',ROOT/'audit.py');m=importlib.util.module_from_spec(spec)
    # Avoid executing compiled main; extract only wrapper imports/constants/audit_tree.
    tree=ast.parse((ROOT/'audit.py').read_text());tree.body=tree.body[:-1];exec(compile(tree,str(ROOT/'audit.py'),'exec'),m.__dict__)
    effective=m.audit_tree();original=ast.parse((C.BASE/'audit.py').read_text());inverse={v:k for k,v in m.REPLACEMENTS.items()}
    for node in ast.walk(effective):
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in inverse:node.value=inverse[node.value]
    class RemoveMetadata(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].slice,ast.Constant) and node.targets[0].slice.value in ('derived_qualification_view','qualification_source_domains'):return None
            return node
    effective=RemoveMetadata().visit(effective)
    assert ast.dump(effective,include_attributes=False)==ast.dump(original,include_attributes=False)
    return True

def fixtures():
    original=(C.PRODUCER,C.POSTRUN,C.post_metadata,C.read,C.sha,C.verify,C.reused_preflight_qualification)
    C.PRODUCER=Path('/fixture/original');C.POSTRUN=Path('/fixture/postrun');p=C.PRODUCER;q=C.POSTRUN;digest='a'*64;cp=p/'TRAIN/checkpoint_000900.pt'
    records={};hashes={};modelsources={'/fixture/model.py':'1'*64};modeldata={'/fixture/data.npz':'2'*64};wide={**modelsources,'/fixture/postrun_tool.py':'3'*64};wide_data=dict(modeldata)
    fail=dict(path=str(p/'JOBS.json'),sha256='0'*64,status='FAILED',exit_code=124)
    rows=[dict(step=i) for i in range(601,901)]
    meta=dict(step=900,motor_format=C.FORMAT,postrun_training_updates=0,postrun_optimizer_operations=0,postrun_gradient_operations=0,training_rows=rows,original_deadline_failure=fail,training_journal_sha256='0'*64,source_hashes=wide,data_hashes=wide_data,proof_hashes={str(p/'JOBS.json'):'0'*64})
    entries=[dict(id='case'+str(i),length=1920) for i in range(20)];binding=dict(entries=entries[:12],dev_entries=entries[12:]);records[p/'INPUT_BINDINGS.json']=binding;records[p/'PREFLIGHT.json']={}
    records[q/'JOBS.json']=[dict(stage=name,status='COMPLETED',exit_code=0,command=[str(C.PYTHON),'-B',str(q/'runtime.py'),'--stage',name,'--sha256',digest,'--execute-parent-postrun-stage']) for name in C.POST_STAGES]
    observer=dict(step=900,optimizer_states=8,all_Adam_counters900=True,original_logged_update_rows=300,curve_logged_fields_exact=True,extra_checkpoint_metrics_only=True,strict_payload_identity_and_oldweights=True,observer_training_updates=0,observer_optimizer_operations=0,original_anchor_RNG_validation='UNCHANGED_ORIGINAL_STRICT_LOADER')
    records[q/'TRAIN/RESULT.json']=dict(kind='SAVED900_QUALIFIED_NO_NEW_UPDATES',qualification_mode='POSTRUN_SAVED_CHECKPOINT_ONLY',updates=900,total_updates=900,new_updates=300,already_completed_updates=600,qualification_new_updates=0,physical_steps=0,original_pipeline_success_claim=False,checkpoint=str(cp),sha256=digest,original_training_rows=300,original_training_log=str(p/'TRAIN/TRAINING.jsonl'),original_training_log_sha256='0'*64,original_deadline_failure=fail,strict_payload_observation=observer,source_hashes=wide,data_hashes=wide_data,proof_hashes=meta['proof_hashes'])
    for step in C.SAVE_STEPS:
        saved=p/'TRAIN'/f'checkpoint_{step:06d}.pt';hashes[saved]=digest if step==900 else str(step)[0]*64
        checks={};savedrows=[]
        for i,entry in enumerate(entries):
            tag=entry['id']+str(i)
            for suffix in ('_production_finite','_actualX_cachedproduction20')+(('_TFproduction','_conditional_fullproduction','_conditional_finite') if i<12 else ()):checks[tag+suffix]=True
            savedrows.append(dict(id=entry['id'],actual_rows=1920,training=i<12,labels_available=i<12,conditional_output_errors=[0.]*6))
        checks.update({key:True for key in ('strictloader_RNGunchanged','oldweights_frozen','canonical8Adamstates','all_model_weights_unchanged','no_Core_physics_teacherbackend')});assert len(checks)==81
        folder=q if step==900 else p;artifact=folder/'TRAIN'/f'CONDITIONAL_{step:06d}.npz';hashes[artifact]='0'*64
        v=dict(step=step,integrity_all_pass=True,checks=checks,checkpoint=str(saved),checkpoint_sha256=hashes[saved],rows=savedrows,source_hashes=modelsources,data_hashes=modeldata,conditional_arrays=dict(path=str(artifact),sha256='0'*64))
        records[folder/'TRAIN'/f'VERIFY_{step:06d}.json']=v
        if step!=900:records[q/'TRAIN'/f'VERIFY_{step:06d}.json']=v
    records[q/'EVALUATION.json']=dict(kind='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1',step=900,checkpoint=str(cp),checkpoint_sha256=digest,source_hashes=modelsources,data_hashes=modeldata,arrays=dict(path=str(q/'EVALUATION_ARRAYS.npz'),sha256='0'*64),primary_supervision_metric='actualX_actionfit',conditional_action_label_distance='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION',actualX_rows=[dict(id=e['id']) for e in entries[:12]],conditional_rows=[dict(id=e['id'],conditional_action_label_loss_used=False,distance_semantics='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION') for e in entries[:12]],DEV=[dict(labels_available=False,teacher_MSE=None) for _ in range(8)],behavior_pass=None,new_neck_deployed=False)
    records[q/'EVALUATION_VERIFY.json']=dict(integrity_all_pass=True,checks=dict(metrics=True),checkpoint_sha256=digest,evaluation_sha256='0'*64)
    outputs={str(q/name):'0'*64 for name in ('TRAIN/VERIFY_000900.json','TRAIN/CONDITIONAL_000900.npz','TRAIN/RESULT.json','EVALUATION.json','EVALUATION_ARRAYS.npz','EVALUATION_VERIFY.json','JOBS.json','SOURCE_LEDGER.json')}
    for name in C.POST_STAGES:
        outputs[str(q/(name+'_COMPLETION.json'))]='0'*64;records[q/(name+'_COMPLETION.json')]=dict(stage=name,passed=True,output_ROOT=str(q),source_and_input_unchanged=True,training_updates=0,optimizer_operations=0,Core_constructions=0,physical_steps=0,peak_RSS_MiB=500,AST_changes='ONE_FINAL_FALSE_TO_TRUE' if name==C.POST_STAGES[0] else 'NONE',strict_payload_observer_readonly=name==C.POST_STAGES[0])
    records[q/'STAGE_VERIFICATION.json']=dict(kind='POSTRUN_SAVED900_QUALIFICATION_NOT_ORIGINAL_PIPELINE_SUCCESS',qualification_mode='POSTRUN_SAVED_MODEL_ONLY',all_postrun_stages_exit0=True,original_pipeline_all3exit0=False,checkpoint=str(cp),checkpoint_sha256=digest,completed_original_total_updates=900,completed_original_new_updates=300,postrun_training_updates=0,optimizer_operations=0,physical_steps=0,original_deadline_failure=fail,outputs=outputs)
    records[q/'RESOURCE_COMPLETION.json']=dict(**{'pass':True},status='PASS',qualification_mode='POSTRUN_SAVED_MODEL_ONLY',wall_limit_s=3600,RSS_limit_MiB=1800,wall_s=1000,all_artifact_hashes_reverified=True,original_failure_preserved=True,no_false_original_pipeline_PASS=True,science_and_success_output_included=True,failure_bookkeeping_never_qualifies=True,requires_no_FAILURE_and_actual_run_exit0=True,stage_RSS_MiB={name:500 for name in C.POST_STAGES},postrun_training_updates=0,optimizer_operations=0,Core_constructions=0,physical_steps=0,artifacts={**outputs,str(q/'STAGE_VERIFICATION.json'):'0'*64})
    active=copy.deepcopy(records);active_meta=copy.deepcopy(meta)
    def qualify(requested):
        if active_meta['original_deadline_failure']['status']!='FAILED' or active_meta['original_deadline_failure']['exit_code']!=124:raise ValueError('mockoriginalnotterminalfailed124')
        if active_meta.get('missing900'):raise ValueError('mockmissing900')
        if active_meta.get('wrongFORMAT'):raise ValueError('mockwrongFORMAT')
        return copy.deepcopy(active_meta)
    C.post_metadata=lambda:types.SimpleNamespace(qualify=qualify,_saved_proof=lambda step,b:(copy.deepcopy(active[p/'TRAIN'/f'VERIFY_{step:06d}.json']),{str(p/'TRAIN'/f'checkpoint_{step:06d}.pt'):hashes[p/'TRAIN'/f'checkpoint_{step:06d}.pt']}))
    C.read=lambda path:copy.deepcopy(active[Path(path)]);C.sha=lambda path:hashes.get(Path(path),'0'*64);C.verify=lambda value:None;C.reused_preflight_qualification=lambda b,pre:{}
    proof,view=C.final_qualification(cp,digest);assert view['source_hashes']==modelsources and view['qualification_source_hashes']==wide and view['source_hashes']!=view['qualification_source_hashes'] and view['qualification_mode']==C.MODE
    assert all(str(p/'TRAIN'/f'checkpoint_{step:06d}.pt') in proof for step in C.SAVE_STEPS)
    rejected=[]
    def reject(label,change):
        nonlocal active,active_meta
        active=copy.deepcopy(records);active_meta=copy.deepcopy(meta);change(active,active_meta)
        try:C.final_qualification(cp,digest)
        except (ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('Should reject '+label)
    reject('originalRUNNING',lambda r,m:m['original_deadline_failure'].update(status='RUNNING'))
    reject('originalall3success',lambda r,m:m['original_deadline_failure'].update(status='COMPLETED',exit_code=0))
    reject('missing900_only750',lambda r,m:m.update(missing900=True))
    reject('wrongFORMAT',lambda r,m:m.update(wrongFORMAT=True))
    reject('postrun_not_complete',lambda r,m:r[q/'JOBS.json'][1].update(status='RUNNING'))
    reject('missing_completion',lambda r,m:r.pop(q/'RESOURCE_COMPLETION.json'))
    reject('fresh90081_failure',lambda r,m:r[q/'TRAIN/VERIFY_000900.json']['checks'].update(canonical8Adamstates=False))
    reject('Adamcounter_not900',lambda r,m:r[q/'TRAIN/RESULT.json']['strict_payload_observation'].update(all_Adam_counters900=False))
    reject('curve_not300',lambda r,m:r[q/'TRAIN/RESULT.json']['strict_payload_observation'].update(original_logged_update_rows=299))
    reject('false_originalsuccess_claim',lambda r,m:r[q/'STAGE_VERIFICATION.json'].update(original_pipeline_all3exit0=True))
    reject('wider_maps_misused_as_actual',lambda r,m:r[q/'EVALUATION.json'].update(source_hashes=wide))
    reject('completion_false',lambda r,m:r[q/'RESOURCE_COMPLETION.json'].update(**{'pass':False}))
    C.PRODUCER,C.POSTRUN,C.post_metadata,C.read,C.sha,C.verify,C.reused_preflight_qualification=original
    return rejected

def main():
    assert equivalence()
    for value in list(vars(C).values()):
        if isinstance(value,types.FunctionType):globals_ok(value)
    assert C.PRODUCER==C.PRODUCER_PATH and C.WALL_S==1800 and C.RSS_MIB==1800 and C.GATE==1e-12
    prereg=C.read(ROOT/'PREREG.json');assert prereg['resources']['wall_s']==C.WALL_S and prereg['resources']['RSS_MiB']==C.RSS_MIB and prereg['qualification_mode']==C.MODE
    assert C.producer_context.__globals__['__name__']=='contextlib'
    assert C.completion_success(dict(status='PASS',qualified_final900=True),dict(**{'pass':True},wall_s=1799,RSS_MiB=1799))
    assert not C.completion_success(dict(status='PASS',qualified_final900=True),dict(**{'pass':True},wall_s=1801,RSS_MiB=1799))
    rejected=fixtures();assert not any(name in sys.modules for name in ('torch','numpy','physics','mujoco','flygym'))
    print(json.dumps(dict(status='ONLYCODE_STD_PASS',inverse_scienceAST_identical=True,original_producer_not_postrun_model=True,actual_vs_qualification_source_domains=True,metadata_rejections=rejected,budget1800_RSS1800=True,globals_contextmanager_preserved=True,Torch_NumPy_array_PT_model_calls=0),indent=2))
if __name__=='__main__':main()
