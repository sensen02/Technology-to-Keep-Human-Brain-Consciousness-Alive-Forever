"""Stdlib source/import/global audit; does not call saved-data qualification."""
from pathlib import Path
import ast
import builtins
import dis
import importlib.util
import json
import sys
import types

ROOT=Path(__file__).resolve().parent
FILES=('PREREG.json','metadata.py','analyze.py','static_check.py','README.zh-CN.md')


def main():
    forbidden={'numpy','torch','mujoco','physics','flygym','student_wave','static_teacher','scipy'};checks={};modules=[]
    if forbidden.intersection(sys.modules):raise ValueError('Fresh stdlib process required')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Numerical import forbidden in staticcheck: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        for name in FILES:
            path=ROOT/name
            if not path.is_file():raise FileNotFoundError(path)
            if name.endswith('.py'):
                compile(ast.parse(path.read_text()),str(path),'exec');spec=importlib.util.spec_from_file_location('_handover_static_'+name.replace('.','_'),path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);modules.append(mod)
                checks[name+'_AST_and_stdlib_import']=not forbidden.intersection(sys.modules)
    finally:builtins.__import__=original
    missing={}
    def inspect_code(code,namespace):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in namespace and not hasattr(builtins,instruction.argval):missing.setdefault(code.co_name,[]).append(instruction.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):inspect_code(value,namespace)
    for mod in modules:
        for value in vars(mod).values():
            if isinstance(value,types.FunctionType) and value.__module__==mod.__name__:inspect_code(value.__code__,value.__globals__)
    checks['all_function_globals_resolve']=not missing
    prereg=json.loads((ROOT/'PREREG.json').read_text());checks['fixed90s1500MiB_zeroNN_Core_physics']=prereg['budget']['wall_s']==90 and prereg['budget']['max_RSS_MiB']==1500 and all(prereg['budget'][key]==0 for key in ('NN_forwards','checkpoint_loads','Core_constructions','physical_steps','training_updates'))
    tree=ast.parse((ROOT/'analyze.py').read_text())
    checks['no_neural_physical_loader_calls']=not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('step','step_details','prime_old','load_checkpoint','Core','mj_step','forward') for node in ast.walk(tree))
    checks['numeric_import_lazy_only']=all(not isinstance(node,ast.Import) or not any(alias.name.split('.')[0] in forbidden for alias in node.names) for node in tree.body)
    checks['explicit_actual_time_indices']=all(any(isinstance(node,ast.Constant) and node.value==value for node in ast.walk(tree)) for value in (5799,5804,5800,1160))
    checks['teacher_hidden_NOT_RECORDED_no_reconstruction']="status='NOT_RECORDED'" in ast.unparse(tree) and 'm.old' not in ast.unparse(tree)
    checks['first_compare_saved_newNN_not_teacherlabel']="evaluation['conditional_mean']" in ast.unparse(tree) and "conditional[tag + '_mean']" in ast.unparse(tree)
    checks['no_existing_analysis_outputs']=not any((ROOT/name).exists() for name in ('RESULT.json','REPORT.zh-CN.md','VERIFY.json'))
    import hashlib
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(status='ONLYCODE_SAVED_HANDOVER_ANALYSIS_READY_FROZEN',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(ROOT/name):sha(ROOT/name) for name in FILES},numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_forwards=0,Core_constructions=0,physical_steps=0,training_updates=0,actual_analysis_executed=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static handover analysis checks failed')


if __name__=='__main__':main()
