"""Pure stdlib source/global/metadata fixtures. Never loads actual checkpoint payload."""
from pathlib import Path
import ast,builtins,contextlib,copy,dis,importlib.util,json,sys,types
ROOT=Path(__file__).resolve().parent

def unresolved(function):
    missing=[]
    def visit(code,namespace):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in namespace and instruction.argval not in builtins.__dict__:missing.append(instruction.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):visit(value,namespace)
    visit(function.__code__,function.__globals__)
    wrapped=getattr(function,'__wrapped__',None)
    if isinstance(wrapped,types.FunctionType):missing.extend(unresolved(wrapped))
    return missing

def main():
    tree=ast.parse((ROOT/'audit.py').read_text());compile(tree,str(ROOT/'audit.py'),'exec')
    spec=importlib.util.spec_from_file_location('_stdlib_final_batch_audit_review',ROOT/'audit.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for value in vars(module).values():
        if isinstance(value,types.FunctionType):assert not unresolved(value),(value.__name__,unresolved(value))
    assert module.v4_context.__globals__ is contextlib.__dict__
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr in ('backward','step','zero_grad','save_checkpoint') for node in calls)
    strict=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=='load_checkpoint']
    assert len(strict)==1 and any(k.arg=='final' and isinstance(k.value,ast.Constant) and k.value.value is True for k in strict[0].keywords)
    assert any(isinstance(node.func,ast.Attribute) and node.func.attr=='no_grad' for node in calls)
    assert module.GATE==1e-12 and module.WALL==300 and module.RSS==1800
    # In-memory final JSON fixtures: all hashes are stubbed, no real payload or
    # current v4 JOBS/TRAINING/state is read or deserialized by these checks.
    original=(module.V4,module.read_json,module.sha)
    module.V4=Path('/fixture/final_v4');v4=module.V4;digest='a'*64;checkpoint=v4/'TRAIN/checkpoint_000600.pt'
    source={'/fixture/science.py':'b'*64};data={'/fixture/archive.npz':'c'*64}
    records={v4/'JOBS.json':[dict(stage=stage,status='COMPLETED',exit_code=0,command=['python','-B',str(v4/script)]) for stage,script in zip(module.STAGES,('train.py','evaluate.py','audit_eval.py'))],v4/'STAGE_VERIFICATION.json':dict(all_stages_exit0=True,train_updates=600,new_updates=300,checkpoint=str(checkpoint),checkpoint_sha256=digest,evaluation_sha256='e'*64),v4/'TRAIN/RESULT.json':dict(updates=600,total_updates=600,new_updates=300,already_completed_updates=300,checkpoint=str(checkpoint),sha256=digest,source_hashes=source,data_hashes=data),v4/'EVALUATION.json':dict(step=600,checkpoint=str(checkpoint),checkpoint_sha256=digest,source_hashes=source,data_hashes=data,arrays=dict(path=str(v4/'EVALUATION_ARRAYS.npz'),sha256='f'*64)),v4/'EVALUATION_VERIFY.json':dict(integrity_all_pass=True,checks={'metric':True},checkpoint_sha256=digest,evaluation_sha256='e'*64)}
    hashes={checkpoint:digest,v4/'EVALUATION.json':'e'*64,v4/'EVALUATION_ARRAYS.npz':'f'*64}
    for step in (300,450,600):
        saved=v4/'TRAIN'/f'checkpoint_{step:06d}.pt';hashes[saved]=digest if step==600 else str(step)[0]*64
        records[v4/'TRAIN'/f'VERIFY_{step:06d}.json']=dict(step=step,integrity_all_pass=True,checks={'storage':True},checkpoint=str(saved),checkpoint_sha256=hashes[saved])
    for path in records:hashes.setdefault(path,'0'*64)
    active=copy.deepcopy(records);module.read_json=lambda path:copy.deepcopy(active[Path(path)]);module.sha=lambda path:hashes[Path(path)]
    proof,_=module.final_qualification(checkpoint,digest);assert len(proof)==9
    rejected=[]
    def expect_reject(label,mutate,cp=checkpoint,requested=digest):
        nonlocal active
        active=copy.deepcopy(records);mutate(active)
        try:module.final_qualification(cp,requested)
        except (ValueError,KeyError):rejected.append(label)
        else:raise AssertionError('Fixture should reject '+label)
    expect_reject('main_stage_still_running',lambda r:r[v4/'JOBS.json'][2].update(status='RUNNING'))
    expect_reject('wrong_actual_SHA',lambda r:None,requested='1'*64)
    expect_reject('old450_not_final',lambda r:None,cp=v4/'TRAIN/checkpoint_000450.pt')
    expect_reject('saved450_failure',lambda r:r[v4/'TRAIN/VERIFY_000450.json'].update(integrity_all_pass=False))
    expect_reject('evaluation_audit_failure',lambda r:r[v4/'EVALUATION_VERIFY.json']['checks'].update(metric=False))
    expect_reject('evaluation_provenance_mismatch',lambda r:r[v4/'EVALUATION.json'].update(source_hashes={}))
    module.V4,module.read_json,module.sha=original
    assert not any(name in sys.modules for name in ('torch','numpy','mujoco','physics','flygym'))
    print(json.dumps(dict(status='ONLYCODE_STD_REVIEW_PASS',recursive_globals_resolved=True,contextmanager_original_globals=True,strict_finalTrue_source=True,no_optimizer_gradient_calls=True,metadata_positive_fixture=True,metadata_rejection_fixtures=rejected,Torch_imports=0,NumPy_imports=0,checkpoint_payload_loads=0,neural_forwards=0,physical_steps=0),indent=2))
if __name__=='__main__':main()
