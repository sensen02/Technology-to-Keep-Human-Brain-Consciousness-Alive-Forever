"""STD-only source/global and science-equivalence review; no numerical calls."""
from pathlib import Path
import ast
import builtins
import dis
import hashlib
import importlib.util
import json
import sys
import types
ROOT=Path(__file__).resolve().parent
FILES=('metadata.py','runtime.py','run.py','PREREG.json','README.zh-CN.md','staticcheck.py')


def main():
    checks={};modules=[];forbidden={'numpy','torch','mujoco','physics','flygym','student_wave','static_teacher','scipy'}
    if forbidden.intersection(sys.modules):raise ValueError('CleanSTDprocessrequired')
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('No numerical imports in postrunSTDcheck:'+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guard
    try:
        for name in FILES:
            path=ROOT/name
            if not path.is_file():raise FileNotFoundError(path)
            if name.endswith('.py'):
                compile(ast.parse(path.read_text()),str(path),'exec');spec=importlib.util.spec_from_file_location('_postrun_sourcecheck_'+name.replace('.','_'),path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
        import runtime
        for stage in runtime.STAGES:
            filename,tree,changes=runtime.stage_source(stage);old=ast.parse((runtime.ORIGINAL/filename).read_text())
            if changes:
                for node in ast.walk(tree):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='common' and node.func.attr=='load_checkpoint':
                        for kw in node.keywords:
                            if kw.arg=='final':kw.value=ast.Constant(False)
            checks[stage+'_inverse_final_guard_science_AST_exact']=ast.dump(tree,include_attributes=False)==ast.dump(old,include_attributes=False)
        checks['entry_imports_STD_only']=not forbidden.intersection(sys.modules)
    finally:builtins.__import__=original
    missing={}
    def inspect(code,ns):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in ns and not hasattr(builtins,ins.argval):missing.setdefault(code.co_name,[]).append(ins.argval)
        for val in code.co_consts:
            if isinstance(val,types.CodeType):inspect(val,ns)
    for module in modules:
        for value in vars(module).values():
            if isinstance(value,types.FunctionType):inspect(value.__code__,value.__globals__)
    checks['all_globals_resolve']=not missing
    prereg=json.loads((ROOT/'PREREG.json').read_text());checks['fixed3stage3600_RSS1800_zero_updates']=prereg['stages']==list(runtime.STAGES) and prereg['budget']['shared_wall_s']==3600 and prereg['budget']['RSS_MiB']==1800 and all(prereg['budget'][k]==0 for k in ('new_training_updates','optimizer_operations','gradient_operations','Core_constructions','physical_steps'))
    rtree=ast.parse((ROOT/'runtime.py').read_text())
    checks['no_training_physics_call_sites']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','zero_grad','Core','mj_step') for n in ast.walk(rtree))
    runtext=ast.unparse(ast.parse((ROOT/'run.py').read_text()));runtimetext=ast.unparse(rtree)
    checks['deadline_spawn_and_cleanup_signal_guards']='oldmask = signal.pthread_sigmask(signal.SIG_BLOCK' in runtext and runtext.count('signal.pthread_sigmask(signal.SIG_BLOCK')==2 and 'signal.pthread_sigmask(signal.SIG_UNBLOCK' in runtimetext
    checks['completion_requires_no_FAILURE_and_actual_exit0']='requires_no_FAILURE_and_actual_run_exit0=True' in runtext and 'qualification_pass=False' in runtext
    checks['no_outputs_or_staging_executed']=not any((ROOT/n).exists() for n in ('JOBS.json','INPUT_BINDINGS.json','SOURCE_LEDGER.json','TRAIN','STAGE_VERIFICATION.json','RESOURCE_COMPLETION.json'))
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    print(json.dumps(dict(status='ONLYCODE_CONDITIONAL_POSTRUN_SOURCE_READY',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},child_numeric_imports=0,child_NPZ_payload_reads=0,child_checkpoint_loads=0,child_NN_forwards=0,child_Core_constructions=0,child_physical_steps=0,original_must_terminate_deadline_failed124_and_actual900_required=True,actual_postrun_executed=False),ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('STDpostrunsourcecheckfailed')


if __name__=='__main__':main()
