"""Stdlib AST, API names, body JSON mock admission and actual ZIP-header check only."""
from pathlib import Path
import ast,builtins,copy,dis,hashlib,importlib.util,json,sys,types
import contracts as C
ROOT=C.ROOT

def global_errors(function):
    errors=[]
    def walk(code,namespace):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in namespace and instruction.argval not in builtins.__dict__:errors.append(instruction.argval)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):walk(item,namespace)
    walk(function.__code__,function.__globals__)
    wrapped=getattr(function,'__wrapped__',None)
    if isinstance(wrapped,types.FunctionType):errors.extend(global_errors(wrapped))
    return errors

def main():
    tree=ast.parse((ROOT/'diagnose.py').read_text());compile(tree,str(ROOT/'diagnose.py'),'exec');compile(ast.parse((ROOT/'contracts.py').read_text()),str(ROOT/'contracts.py'),'exec')
    spec=importlib.util.spec_from_file_location('_local_sensitivity_onlycode_review',ROOT/'diagnose.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for owner in (module,C):
        for value in vars(owner).values():
            if isinstance(value,types.FunctionType):assert not global_errors(value),(value.__name__,global_errors(value))
    prereg=C.read(ROOT/'PREREG.json');assert prereg['checkpoint_sha256']==C.CHECKPOINT_SHA and tuple(prereg['epsilon_native_target_rad'])==C.EPSILONS
    assert tuple((p['case'],p['relative_ANN_s'],p['physical_step_ms'],p['neural_row'],p['preaction_trace_row']) for p in prereg['points'])==C.POINTS
    assert len(C.POINTS)==6 and all(ms-5800==round(relative*1000) and row*5==ms and trace==ms-1 for _,relative,ms,row,trace in C.POINTS)
    assert len(C.POINTS)*(1+len(C.EPSILONS)*42*2)==prereg['expected_calls']['public_step_details']==1014
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','zero_grad','requires_grad_','save_checkpoint') for n in calls)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('optim','autograd') for n in ast.walk(tree))
    strict=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load_checkpoint'];assert len(strict)==1 and any(k.arg=='final' and isinstance(k.value,ast.Constant) and k.value.value is True for k in strict[0].keywords)
    assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='no_grad' for n in calls)
    pin=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='sched_setaffinity'];threads=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='Thread'];assert len(pin)==1 and threads and pin[0]<min(threads)
    assert any(isinstance(n.func,ast.Attribute) and n.func.attr=='step_details' for n in calls) and any(isinstance(n.func,ast.Attribute) and n.func.attr=='legacy_projection' for n in calls)
    assert '_old_command' not in (ROOT/'diagnose.py').read_text()
    actual_headers={case:C.check_headers(case) for case in ('stop','walk')}
    # Entire admission test below uses in-memory JSON and hashes. It does not
    # read current BODY JOBS/SUMMARY, NPZ payload or any Torch checkpoint.
    old=(C.BODY,C.read,C.sha,C.check_headers);C.BODY=Path('/fixture/body');b=C.BODY;records={};hashes={}
    jobs=[dict(condition=case,stage=stage,status='COMPLETED',exit_code=0) for case in ('walk','stop','left_pivot','right_pivot') for stage in ('PHYSICS','FRESH_NN_SAVED_AUDIT')];jobs.append(dict(stage='SUMMARY',status='COMPLETED',exit_code=0,summary_sha256='0'*64))
    fixture_sources={'/fixture/body_science.py':'0'*64};hashes[Path('/fixture/body_science.py')]='0'*64
    records[b/'JOBS.json']=jobs;records[b/'INPUT_BINDINGS.json']=dict(checkpoint=str(C.CHECKPOINT),checkpoint_sha256=C.CHECKPOINT_SHA,source_hashes=fixture_sources)
    summary_rows=[]
    for case in ('walk','stop','left_pivot','right_pivot'):
        folder=b/'trials'/case;rp=folder/'RESULT.json';vp=folder/'VERIFY.json';identity=dict(config=dict(motor_sha256=C.CHECKPOINT_SHA,state_dims=[128,64,64,96],source_hashes=fixture_sources));identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        result=dict(condition=case,checkpoint=str(C.CHECKPOINT),checkpoint_sha256=C.CHECKPOINT_SHA,final_step=600,state_dims=[128,64,64,96],state_total=352,full_duration=True,actual_ms=15400,neural_records=3080,identity_sha256=identity['identity_sha256'],source_hashes=fixture_sources,candidate_metrics=dict(task_pass_candidate=False))
        for key,name in (('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz')):result[key]=dict(path=str(folder/name),sha256='0'*64);hashes[folder/name]='0'*64
        records[rp]=result;records[vp]=dict(integrity_all_pass=True,checks={'fresh':True},result_sha256='0'*64,checkpoint_sha256=C.CHECKPOINT_SHA);records[folder/'IDENTITY.json']=identity
        summary_rows.append(dict(condition=case,result=str(rp),result_sha256='0'*64,audit=str(vp),audit_sha256='0'*64))
    records[b/'SUMMARY.json']=dict(checkpoint_sha256=C.CHECKPOINT_SHA,rows=summary_rows,source_hashes=fixture_sources)
    for path in records:hashes.setdefault(path,'0'*64)
    active=copy.deepcopy(records);C.read=lambda path:copy.deepcopy(active[Path(path)]);C.sha=lambda path:hashes[Path(path)];C.check_headers=lambda case:dict(mock_header=True)
    proof,selected,saved_sources=C.body_qualification();assert set(selected)=={'stop','walk'} and saved_sources==fixture_sources and all(not value['result']['candidate_metrics']['task_pass_candidate'] for value in selected.values())
    rejected=[]
    def reject(label,change):
        nonlocal active
        active=copy.deepcopy(records);change(active)
        try:C.body_qualification()
        except (ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('Should reject '+label)
    reject('body_stage_running',lambda r:r[b/'JOBS.json'][0].update(status='RUNNING'))
    reject('body_missing_ninth_stage',lambda r:r[b/'JOBS.json'].pop())
    reject('body_saved_audit_failure',lambda r:r[b/'trials/left_pivot/VERIFY.json']['checks'].update(fresh=False))
    reject('selected_stop_partial',lambda r:r[b/'trials/stop/RESULT.json'].update(full_duration=False))
    reject('wrong_checkpoint_identity',lambda r:r[b/'INPUT_BINDINGS.json'].update(checkpoint_sha256='f'*64))
    reject('normalizer_body_identity_not_canonical',lambda r:r[b/'trials/stop/IDENTITY.json']['config'].update(state_dims=[352]))
    reject('body_source_binding_missing',lambda r:r[b/'INPUT_BINDINGS.json'].update(source_hashes={}))
    reject('body_result_saved_source_differs',lambda r:r[b/'trials/walk/RESULT.json'].update(source_hashes={}))
    hashes[Path('/fixture/body_science.py')]='1'*64
    reject('body_source_drift_before_diagnostic',lambda r:None)
    hashes[Path('/fixture/body_science.py')]='0'*64
    C.BODY,C.read,C.sha,C.check_headers=old
    good_result=dict(status='COMPLETED',diagnostic_integrity_pass=True);good_completion=dict(**{'pass':True},wall_s=200.,RSS_MiB=700.)
    assert C.completion_success(good_result,good_completion)
    assert not C.completion_success(dict(status='FAILED',diagnostic_integrity_pass=True),good_completion)
    assert not C.completion_success(dict(status='COMPLETED',diagnostic_integrity_pass=False),good_completion)
    assert not C.completion_success(good_result,dict(good_completion,wall_s=301.))
    assert not C.completion_success(good_result,dict(good_completion,RSS_MiB=1801.))
    assert not C.completion_success(good_result,dict(good_completion,**{'pass':False}))
    assert not any(name in sys.modules for name in ('torch','numpy','mujoco','physics','flygym'))
    print(json.dumps(dict(status='ONLYCODE_STD_REVIEW_PASS',AST=True,recursive_globals=True,source_API_names=True,six_points_two_eps_exact=True,registered_call_count=True,strict_finalTrue=True,no_optimizer_autograd=True,actual_saved_ZIP_headers_verified=actual_headers,body_admission_in_memory_positive_including_behaviorFAIL=True,rejected_body_fixtures=rejected,resource_completion_and_failure_state_fixtures=True,Torch_imports=0,NumPy_imports=0,NPZ_array_loads=0,PT_payload_loads=0,neural_evaluations=0,physical_steps=0),indent=2))
if __name__=='__main__':main()
