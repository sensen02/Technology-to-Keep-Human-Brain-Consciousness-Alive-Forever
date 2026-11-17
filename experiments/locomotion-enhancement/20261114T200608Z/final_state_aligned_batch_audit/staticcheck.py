"""Pure standard-library final900 schema, globals, alias and budget fixtures."""
from pathlib import Path
import ast,builtins,contextlib,copy,dis,importlib.util,json,sys,types
import contracts as C
ROOT=C.ROOT

def unresolved(function):
    errors=[]
    def visit(code,namespace):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in namespace and instruction.argval not in builtins.__dict__:errors.append(instruction.argval)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):visit(item,namespace)
    visit(function.__code__,function.__globals__)
    if isinstance(getattr(function,'__wrapped__',None),types.FunctionType):errors.extend(unresolved(function.__wrapped__))
    return errors

def main():
    tree=ast.parse((ROOT/'audit.py').read_text());compile(tree,str(ROOT/'audit.py'),'exec');compile(ast.parse((ROOT/'contracts.py').read_text()),str(ROOT/'contracts.py'),'exec')
    spec=importlib.util.spec_from_file_location('_onlycode_final900_audit_review',ROOT/'audit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for owner in (C,module):
        for value in vars(owner).values():
            if isinstance(value,types.FunctionType):assert not unresolved(value),(value.__name__,unresolved(value))
    assert C.producer_context.__globals__ is contextlib.__dict__
    assert C.FINAL_STEP==900 and C.SAVE_STEPS==(600,750,900) and C.FORMAT=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
    assert C.WALL_S==600 and C.RSS_MIB==1800 and C.GATE==1e-12
    prereg=C.read(ROOT/'PREREG.json');assert prereg['resources']['wall_s']==C.WALL_S and prereg['resources']['RSS_MiB']==C.RSS_MIB and prereg['final_step']==C.FINAL_STEP and prereg['format']==C.FORMAT
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    strict=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load_checkpoint'];assert len(strict)==1 and any(k.arg=='final' and isinstance(k.value,ast.Constant) and k.value.value is True for k in strict[0].keywords)
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','zero_grad','requires_grad_','save_checkpoint') for n in calls)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('optim','autograd') for n in ast.walk(tree))
    assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='no_grad' for n in calls)
    pin=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='sched_setaffinity'];threads=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='Thread'];assert len(pin)==1 and pin[0]<min(threads)
    alarms=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='alarm'];assert any(n.args and isinstance(n.args[0],ast.Attribute) and n.args[0].attr=='WALL_S' for n in alarms)
    assert C.sha_argument('a'*64) and not C.sha_argument('UNKNOWN') and not C.sha_argument('A'*64)
    # Everything from here through restoration is in-memory fake metadata/hash.
    # No producer preparation, current JOBS, future NPZ or payload is touched.
    old=(C.PRODUCER,C.ORIGINAL,C.read,C.sha,C.equivalence_now);C.PRODUCER=Path('/fixture/state_aligned/wall_replan_9000');C.ORIGINAL=C.PRODUCER.parent;p=C.PRODUCER;o=C.ORIGINAL;digest='a'*64;checkpoint=p/'TRAIN/checkpoint_000900.pt'
    hashes={checkpoint:digest,o/'PREFLIGHT.json':'e'*64,o/'INPUT_BINDINGS.json':'0'*64,o/'PREREG.json':'0'*64};records={};eq=dict(scientific_equivalence=True,fixture=True)
    source={'/fixture/source.py':'1'*64};data={'/fixture/data.npz':'2'*64};correct={f'correct{i}':True for i in range(96)}
    original=dict(all_passed=False,timing_status='BLOCKED_BEFORE_TRAINING_TIMING',wall_limit_s=7200,checks=correct,source_hashes=source,data_hashes=data,predicted300total_withverification_s=8000.,actual_two_branch_micro4x3_update_s=10.,three_fresh_save_allowance_s=3000.,evaluation_allowance_s=1100.)
    records[o/'PREFLIGHT.json']=original;records[o/'INPUT_BINDINGS.json']=dict(source_hashes=source,data_hashes=data);records[o/'PREREG.json']={}
    evidence={str(path):hashes[path] for path in (o/'PREFLIGHT.json',o/'INPUT_BINDINGS.json',o/'PREREG.json')};newsource={**source,**evidence};newdata={**data,**evidence}
    reference=dict(path=str(o/'PREFLIGHT.json'),sha256='e'*64,original_all_passed=False,original_timing_status='BLOCKED_BEFORE_TRAINING_TIMING',actual_correctness_checks=96)
    records[p/'INPUT_BINDINGS.json']=dict(source_hashes=newsource,data_hashes=newdata,resource_wall_replan=dict(science_equivalence=eq,old_wall_s=7200,new_wall_s=9000,no_numeric_rerun=True))
    records[p/'PREFLIGHT.json']=dict(original,all_passed=True,timing_status='READY_STATE_ALIGNED600_TO900',source_hashes=newsource,data_hashes=newdata,wall_limit_s=9000,training_RSS_limit_MiB=1800,preflight_RSS_limit_MiB=3072,resource_replan_pretraining=True,corrected_readiness_against9000=True,original_all_passed_false_preserved=True,new_numeric_preflight_runs=0,new_NN_forwards=0,new_optimizer_probe_updates=0,reused_actual_numeric_preflight=reference,numeric_metrics_origin=reference,science_equivalence=eq,qualification_basis='REUSE_ACTUAL96_CORRECTNESS_AND_EXACT_SCIENCE_SOURCE_EQUIVALENCE_AND_PRETRAINING9000_WALL')
    records[p/'RESOURCE_REPLAN.json']=dict(kind='PRETRAINING_MEASURED_WALL_REPLAN_7200_TO9000',old_numeric_preflight=reference,old_estimate_s=8000.,new_wall_s=9000,NN_forwards=0,model_loads=0,Core_constructions=0,physical_steps=0,science_equivalence=eq)
    records[p/'JOBS.json']=[dict(stage=stage,status='COMPLETED',exit_code=0,command=['python','-B',str(p/script)]) for stage,script in zip(C.STAGES,('train.py','evaluate.py','audit_eval.py'))]
    records[p/'STAGE_VERIFICATION.json']=dict(kind='STATE_ALIGNED_ACTION_RESUME600_TO900',all_stages_exit0=True,train_updates=900,new_updates=300,checkpoint=str(checkpoint),checkpoint_sha256=digest,evaluation_sha256='f'*64)
    records[p/'TRAIN/RESULT.json']=dict(updates=900,total_updates=900,new_updates=300,already_completed_updates=600,checkpoint=str(checkpoint),sha256=digest,source_hashes=newsource,data_hashes=newdata)
    for step in C.SAVE_STEPS:
        saved=p/'TRAIN'/f'checkpoint_{step:06d}.pt';hashes[saved]=digest if step==900 else str(step)[0]*64
        arrays=p/'TRAIN'/f'CONDITIONAL_{step:06d}.npz';hashes[arrays]='3'*64
        checks={f'case{i}_actualX_cachedproduction20':True for i in range(20)}
        for i in range(12):checks[f'case{i}_TFproduction']=True;checks[f'case{i}_conditional_fullproduction']=True
        checks['canonical8Adamstates']=True
        rows=[dict(id=f'case{i}',training=i<12,actual_rows=1920,conditional_output_errors=[0.]*6) for i in range(20)]
        records[p/'TRAIN'/f'VERIFY_{step:06d}.json']=dict(step=step,integrity_all_pass=True,checks=checks,checkpoint=str(saved),checkpoint_sha256=hashes[saved],source_hashes=newsource,data_hashes=newdata,rows=rows,conditional_arrays=dict(path=str(arrays),sha256='3'*64))
    records[p/'EVALUATION.json']=dict(kind='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1',step=900,checkpoint=str(checkpoint),checkpoint_sha256=digest,source_hashes=newsource,data_hashes=newdata,arrays=dict(path=str(p/'EVALUATION_ARRAYS.npz'),sha256='4'*64),primary_supervision_metric='actualX_actionfit',conditional_action_label_distance='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION',actualX_rows=[dict(id=f'case{i}') for i in range(12)],conditional_rows=[dict(id=f'case{i}',conditional_action_label_loss_used=False,distance_semantics='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION') for i in range(12)],DEV=[dict(labels_available=False,teacher_MSE=None) for _ in range(8)],behavior_pass=None,new_neck_deployed=False)
    records[p/'EVALUATION_VERIFY.json']=dict(integrity_all_pass=True,checks=dict(actual_TF=True,conditional=True),checkpoint_sha256=digest,evaluation_sha256='f'*64)
    hashes[p/'EVALUATION.json']='f'*64;hashes[p/'EVALUATION_ARRAYS.npz']='4'*64;hashes[p/'equivalence.py']='0'*64
    for path in records:hashes.setdefault(path,'0'*64)
    active=copy.deepcopy(records);C.read=lambda path:copy.deepcopy(active[Path(path)]);C.sha=lambda path:hashes[Path(path)];C.equivalence_now=lambda:eq
    proof,_=C.final_qualification(checkpoint,digest);assert proof and active[p/'PREFLIGHT.json']['new_numeric_preflight_runs']==0
    assert all(str(p/'TRAIN'/f'checkpoint_{step:06d}.pt') in proof for step in (600,750,900))
    rejected=[]
    def reject(label,change,cp=checkpoint,requested=digest):
        nonlocal active
        active=copy.deepcopy(records);change(active)
        try:C.final_qualification(cp,requested)
        except (ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('Expected refusal: '+label)
    reject('main_stage_running',lambda r:r[p/'JOBS.json'][2].update(status='RUNNING'))
    reject('wrong_actual_final_SHA',lambda r:None,requested='b'*64)
    reject('old600_not_new900',lambda r:None,cp=p/'TRAIN/checkpoint_000600.pt')
    reject('saved750_failed',lambda r:r[p/'TRAIN/VERIFY_000750.json']['checks'].update(canonical8Adamstates=False))
    reject('missing_actualTF_storage_category',lambda r:r[p/'TRAIN/VERIFY_000900.json']['checks'].pop('case0_TFproduction'))
    reject('conditional_precision_exceeds_gate',lambda r:r[p/'TRAIN/VERIFY_000900.json']['rows'][0].update(conditional_output_errors=[2e-12]*6))
    reject('eval_independent_audit_failed',lambda r:r[p/'EVALUATION_VERIFY.json']['checks'].update(actual_TF=False))
    reject('conditional_distance_mislabelled_supervision',lambda r:r[p/'EVALUATION.json'].update(conditional_action_label_distance='SUPERVISION'))
    reject('old_false_evidence_rewritten',lambda r:r[o/'PREFLIGHT.json'].update(all_passed=True))
    reject('one_of96_correctness_failed',lambda r:r[p/'PREFLIGHT.json'].update(checks={f'correct{i}':i!=0 for i in range(96)}))
    reject('resource_equivalence_changed',lambda r:r[p/'PREFLIGHT.json'].update(science_equivalence=dict(scientific_equivalence=False)))
    reject('wrong9000_resource_metadata',lambda r:r[p/'PREFLIGHT.json'].update(wall_limit_s=7200))
    reject('undeclared_numeric_rerun',lambda r:r[p/'PREFLIGHT.json'].update(new_numeric_preflight_runs=1))
    C.PRODUCER,C.ORIGINAL,C.read,C.sha,C.equivalence_now=old
    # Test original contextmanager globals and eight aliases using fake modules.
    importlib_original=C.importlib;sentinel=object();before={name:sys.modules.get(name,sentinel) for name in C.ALIASES};sys_path=list(sys.path);seen=[]
    C.importlib=types.SimpleNamespace(import_module=lambda name:(seen.append(name) or types.SimpleNamespace(__file__=str(C.PRODUCER/(name+'.py')))))
    try:
        with C.producer_context() as modules:assert set(modules)==set(C.MODULES) and 'adapter' not in seen and 'adapter' not in sys.modules
        assert sys.path==sys_path and all(sys.modules.get(name,sentinel) is value for name,value in before.items())
        try:
            with C.producer_context():raise RuntimeError('mock_context_failure')
        except RuntimeError:pass
        assert sys.path==sys_path and all(sys.modules.get(name,sentinel) is value for name,value in before.items())
    finally:C.importlib=importlib_original
    good=dict(status='PASS',qualified_final900=True);commit=dict(**{'pass':True},wall_s=599.,RSS_MiB=1799.)
    assert C.completion_success(good,commit)
    assert not C.completion_success(dict(status='FAILED',qualified_final900=True),commit)
    assert not C.completion_success(good,dict(commit,wall_s=601.)) and not C.completion_success(good,dict(commit,RSS_MiB=1801.))
    assert not C.completion_success(good,dict(commit,**{'pass':False}))
    # Small owned JSON fixture only, cleaned immediately. No producer/model files.
    original_root=C.ROOT
    try:
        with C.tempfile.TemporaryDirectory(prefix='final900_std_commit_') as temporary:
            C.ROOT=Path(temporary);directory=C.ROOT/('RUN_'+'a'*64);directory.mkdir()
            C.save(directory/'ROWS.json',[]);C.save(directory/'SOURCE_LEDGER.json',{});(directory/'REPORT.zh-CN.md').write_text('stdlib fixture\n')
            value=dict(good,format_version=C.FORMAT,final_step=C.FINAL_STEP,checkpoint_sha256='a'*64,checks=dict(mock=True));C.save(directory/'VERIFY.json',value)
            record=dict(commit,wall_limit_s=C.WALL_S,RSS_limit_MiB=C.RSS_MIB,science_and_success_outputs_included=True,artifacts=C.ledger(directory/name for name in ('VERIFY.json','ROWS.json','SOURCE_LEDGER.json','REPORT.zh-CN.md')));C.save(directory/'RESOURCE_COMPLETION.json',record)
            assert C.accept_run(directory)
            (directory/'REPORT.zh-CN.md').write_text('mutated stdlib fixture\n')
            try:C.accept_run(directory)
            except ValueError:pass
            else:raise AssertionError('Artifact drift must reject final commit')
            failed=dict(value,status='FAILED',qualified_final900=False);C.replace_failed_output(directory,'VERIFY.json',failed);C.replace_failed_output(directory,'RESOURCE_COMPLETION.json',dict(record,**{'pass':False}))
            assert C.read(directory/'VERIFY.json')['qualified_final900'] is False and C.read(directory/'RESOURCE_COMPLETION.json')['pass'] is False
            C.save(directory/'FINALIZATION_FAILURE.json',dict(status='FAILED'))
            try:C.accept_run(directory)
            except ValueError:pass
            else:raise AssertionError('Finalization marker must reject')
    finally:C.ROOT=original_root
    assert not any(name in sys.modules for name in ('torch','numpy','mujoco','physics','flygym'))
    print(json.dumps(dict(status='ONLYCODE_STD_REVIEW_PASS',AST_globals=True,original_contextmanager_globals=True,strict_final900_source=True,producer_600750900_schema=True,reused96_new_numeric_runs_zero_accepted=True,rejected_fixtures=rejected,alias_restore_normal_and_exception=True,no_fake_producer_adapter_import=True,pin_before_threads=True,predeclared_audit_wall600_RSS1800_fixture=True,resource_completion_failure_cases=True,Torch_imports=0,NumPy_imports=0,array_loads=0,PT_payload_loads=0,NN_forwards=0,optimizer_operations=0,physical_steps=0),indent=2))
if __name__=='__main__':main()
