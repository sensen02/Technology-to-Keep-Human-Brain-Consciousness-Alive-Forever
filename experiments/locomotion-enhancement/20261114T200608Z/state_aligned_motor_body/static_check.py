"""Stdlib source identity/final900 audit; no numerical or future binding calls."""
from pathlib import Path
import ast
import builtins
import dis
import importlib.util
import json
import sys
import types
import adapter as a
import common as c


def main():
    forbidden={'numpy','torch','mujoco','physics','flygym','student_wave','static_teacher','scipy'};checks={};modules=[]
    if forbidden.intersection(sys.modules):raise ValueError('Fresh stdlib process required')
    for name in a.CORE:
        p=a.ROOT/name
        if not p.is_file():raise FileNotFoundError(p)
        if name.endswith('.py'):compile(ast.parse(p.read_text()),str(p),'exec')
    checks['all_CORE_exist_AST_compile']=True
    for name in a.ENTRIES:
        original=ast.parse((a.OLD/name).read_text());new=a.transformed(name)
        assignments={node.targets[0].id:node.value for node in original.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
        class Reverse(ast.NodeTransformer):
            def visit_Assign(self,node):
                if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ROOT','CAL','TEACHER','FORMAT','FILES'):
                    node.value=assignments[node.targets[0].id];return node
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if type(node.value) is int and node.value==900:return ast.copy_location(ast.Constant(600),node)
                if node.value==a.MOTOR_FORMAT:return ast.copy_location(ast.Constant('INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'),node)
                if isinstance(node.value,str) and 'final900' in node.value:return ast.copy_location(ast.Constant(node.value.replace('final900','final600')),node)
                return node
        inverse=Reverse().visit(new)
        checks[name+'_inverse_final_metadata_AST_exact']=ast.dump(inverse,include_attributes=False)==ast.dump(original,include_attributes=False)
        checks[name+'_all_protocol_numeric_literals_same']=a.numeric(inverse)==a.numeric(original)
        ns=a.namespace(name);checks[name+'_same_newROOT_source3_globals']=ns['ROOT']==a.ROOT and ns['SOURCE_CANDIDATE']==3 and all(value.__globals__.get('ROOT')==a.ROOT and value.__globals__.get('SOURCE_CANDIDATE')==3 for value in ns.values() if isinstance(value,types.FunctionType) and 'SOURCE_CANDIDATE' in value.__globals__)
    original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Numerical import prohibited in STD900check: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guard
    try:
        for name in a.CORE:
            if not name.endswith('.py'):continue
            spec=importlib.util.spec_from_file_location('_state900_sourcecheck_'+name.replace('.','_'),a.ROOT/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
            checks[name+'_stdlib_import']=not forbidden.intersection(sys.modules)
        spec=importlib.util.spec_from_file_location('_state900_strict_storage_STD',a.CAL/'common.py');io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)
        checks['actual_strict_storage_final900_save600750900']=io.FORMAT==a.MOTOR_FORMAT and io.SAVE_STEPS==(600,750,900) and io._changes==1 and 900 in io.g['_validate_payload'].__code__.co_consts and io.g['_validate_payload'].__globals__['ROOT']==a.CAL
        checks['imported_context_and_RNG_helpers_preserved']=all(io.g[name] is io._original.g[name] for name in ('_preserve_rng','restore_rng','_validate_rng','_validate_optimizer_state','_validate_live_optimizer'))
    finally:builtins.__import__=original_import
    missing={}
    def globals_check(code,namespace):
        for ins in dis.get_instructions(code):
            if ins.opname=='LOAD_GLOBAL' and ins.argval not in namespace and not hasattr(builtins,ins.argval):missing.setdefault(code.co_name,[]).append(ins.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):globals_check(value,namespace)
    for module in modules:
        for value in vars(module).values():
            if isinstance(value,types.FunctionType):globals_check(value.__code__,value.__globals__)
    checks['all_function_globals_resolve']=not missing
    prereg=json.loads((a.ROOT/'PREREG.json').read_text())
    checks['correct_CAL_wall9000_and_FORMAT']=c.CAL==a.CAL==a.ROOT.parent/'state_aligned_action_calibration/wall_replan_9000' and a.MOTOR_FORMAT=='INDEPENDENT_MOTOR_STATE_ALIGNED_ACTION_V1'
    checks['original_four_task_commands']=c.CONDITIONS==prereg['conditions']=={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
    checks['only_four_final900_guard_changes']=sum(a.FINAL_GUARD_COUNTS.values())==4 and a.FINAL_STEP==900
    checks['same_body_budget_and_state352']=prereg['budget']['max_native_steps']==4928000 and prereg['budget']['main_stages']==9 and prereg['state_dims']==[128,64,64,96]
    checks['new900_unknown_unbound_no_body_execution']=not any((a.ROOT/n).exists() for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json'))
    print(json.dumps(dict(status='READYSTD_FROZEN_UNBOUND_FINAL900_BEHAVIOR_UNKNOWN',all_passed=all(checks.values()),checks=checks,missing_globals=missing,source_hashes={str(a.ROOT/n):a.sha(a.ROOT/n) for n in a.CORE},scope='BodyANNcomponent_NOTnewbrainneckdeployment',numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_forwards=0,Core_constructions=0,physical_steps=0,training_updates=0,final900_SHA_bound=False),ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('STD900source checks failed')


if __name__=='__main__':main()
