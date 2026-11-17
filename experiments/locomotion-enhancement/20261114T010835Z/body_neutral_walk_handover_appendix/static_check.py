"""Stdlib checks only; actual metadata qualification is a separate explicit step."""
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
FILES=('PREREG.json','appendix.py','static_check.py','README.zh-CN.md')


def main():
    checks={};forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave','static_teacher'}
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Numerical import prohibited in STD appendix check: '+name)
        return original(name,*args,**kwargs)
    builtins.__import__=guard
    try:
        for name in FILES:
            p=ROOT/name
            if not p.is_file():raise FileNotFoundError(p)
            if name.endswith('.py'):compile(ast.parse(p.read_text()),str(p),'exec')
        spec=importlib.util.spec_from_file_location('_neutral_appendix_static_module',ROOT/'appendix.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        sentinel=object();prior=sys.modules.get('metadata',sentinel);metadata,analysis=a.load_frozen_helpers()
        checks['metadata_alias_sentinel_finally_restore']=sys.modules.get('metadata',sentinel) is prior
        checks['ordinary_original_case_report_globals']=analysis.case_report.__globals__ is analysis.__dict__ and analysis.case_report.__code__.co_filename==str(a.ORIGINAL/'analyze.py')
        checks['stdlib_helpers_only']=not forbidden.intersection(sys.modules)
    finally:builtins.__import__=original
    missing={}
    def inspect(code,namespace):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL' and instruction.argval not in namespace and not hasattr(builtins,instruction.argval):missing.setdefault(code.co_name,[]).append(instruction.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):inspect(value,namespace)
    for module in (a,analysis):
        for value in vars(module).values():
            if isinstance(value,types.FunctionType):inspect(value.__code__,value.__globals__)
    checks['all_function_globals_resolve']=not missing
    tree=ast.parse((ROOT/'appendix.py').read_text());calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    checks['single_generic_case_report_call']=sum(isinstance(node.func,ast.Attribute) and node.func.attr=='case_report' for node in calls)==1
    checks['no_NN_Core_or_physical_calls']=not any(isinstance(node.func,ast.Attribute) and node.func.attr in ('step','step_details','prime_old','load_checkpoint','Core','mj_step','forward') for node in calls)
    checks['source_index8_and_conditional_tag']='old_walk_control8' in ast.unparse(tree) and 'training_index=8' in ast.unparse(tree)
    p=json.loads((ROOT/'PREREG.json').read_text());checks['fixed90s1500MiB_CPU1_zeroNN_Core_physics']=p['budget']['wall_s']==90 and p['budget']['RSS_MiB']==1500 and p['budget']['CPU']==1 and all(p['budget'][k]==0 for k in ('NN_forwards','Core_constructions','physical_steps','training_updates'))
    checks['no_array_analysis_outputs']=not any((ROOT/n).exists() for n in ('RESULT.json','REPORT.zh-CN.md','VERIFY.json'))
    def sha(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    print(json.dumps(dict(status='STD_SOURCE_PASS_ACTUAL_METADATA_QUALIFY_REQUIRED',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(ROOT/n):sha(ROOT/n) for n in FILES},numeric_imports=0,NPZ_payload_reads=0,NN_forwards=0,Core_constructions=0,physical_steps=0,actual_array_analysis_executed=False),indent=2,ensure_ascii=False,allow_nan=False))
    if not all(checks.values()):raise ValueError('STD neutral appendix source check failed')


if __name__=='__main__':main()
