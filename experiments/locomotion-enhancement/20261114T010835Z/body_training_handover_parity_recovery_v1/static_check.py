"""Stdlib audit of metadata-only recovery; actual qualify is a separate STD step."""
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
FILES=('PREREG.json','metadata.py','analyze.py','static_check.py','README.zh-CN.md')


def main():
    forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave','static_teacher'};checks={};modules=[]
    if forbidden.intersection(sys.modules):raise ValueError('Clean stdlib process required')
    original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('No numerical imports in STD recovery checks: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guard
    try:
        for name in FILES:
            p=ROOT/name
            if not p.is_file():raise FileNotFoundError(p)
            if name.endswith('.py'):
                compile(ast.parse(p.read_text()),str(p),'exec');spec=importlib.util.spec_from_file_location('_handover_recovery_check_'+name.replace('.','_'),p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
                checks[name+'_AST_stdlib_import']=not forbidden.intersection(sys.modules)
    finally:builtins.__import__=original_import
    missing={}
    def code_globals(code,namespace):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in namespace and not hasattr(builtins,ins.argval):missing.setdefault(code.co_name,[]).append(ins.argval)
        for item in code.co_consts:
            if isinstance(item,types.CodeType):code_globals(item,namespace)
    for module in modules:
        for value in vars(module).values():
            if isinstance(value,types.FunctionType):code_globals(value.__code__,value.__globals__)
    checks['all_compiled_function_globals_resolve']=not missing
    import metadata
    import analyze
    checks['new_ROOT_in_metadata_and_array_functions']=metadata.ROOT==ROOT and analyze.main.__globals__['ROOT']==ROOT and analyze.main.__globals__['qualify'] is metadata.qualify
    oldtree=ast.parse((metadata.FAILED/'metadata.py').read_text());fixed=ast.parse(ast.unparse(metadata._tree))
    class Reverse(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_source_window_length':return ast.copy_location(ast.Subscript(value=ast.Name(id='result',ctx=ast.Load()),slice=ast.Constant('actual_ms'),ctx=ast.Load()),node)
            return self.generic_visit(node)
    checks['only_original_metadata_field_read_changed']=metadata._changes==[103] and ast.dump(Reverse().visit(fixed),include_attributes=False)==ast.dump(oldtree,include_attributes=False)
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    checks['original_array_analysis_source_hash_exact']=sha(metadata.FAILED/'analyze.py')=='b8b3cac78bf43465fb260c93fdca6059e9ab1474d9d72d79dfdb8e2cebb0c1d8'
    checks['original_failure_log_fixed']=sha(metadata.FAILED/'EXECUTION.log')==metadata.FAILED_LOG_SHA
    prereg=json.loads((ROOT/'PREREG.json').read_text());checks['same90s1500MiB_and_zeroNN_Core_physics']=prereg['budget']['wall_s']==90 and prereg['budget']['max_RSS_MiB']==1500 and all(prereg['budget'][key]==0 for key in ('NN_forwards','checkpoint_loads','Core_constructions','physical_steps','training_updates'))
    checks['no_analysis_outputs']=not any((ROOT/n).exists() for n in ('RESULT.json','REPORT.zh-CN.md','VERIFY.json'))
    print(json.dumps(dict(status='STD_SOURCE_CHECK_PASS_METADATA_QUALIFY_STILL_REQUIRED',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_forwards=0,Core_constructions=0,physical_steps=0,actual_array_analysis_executed=False),ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('STD source recovery checks failed')


if __name__=='__main__':main()
