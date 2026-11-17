"""Stdlib static freeze check; no calibration/model/array/physics execution."""
from pathlib import Path
import ast
import builtins
import importlib.util
import json
import sys
import types
import collections
import common as c
import adapter as a


def nums(tree):return collections.Counter((type(n.value).__name__,repr(n.value)) for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float,complex))


def main():
    forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave'};checks={};trees={}
    if forbidden.intersection(sys.modules):raise ValueError('Clean stdlib process required')
    for name in a.CORE:
        path=c.ROOT/name
        if not path.is_file():raise FileNotFoundError(path)
        if name.endswith('.py'):trees[name]=ast.parse(path.read_text());compile(trees[name],str(path),'exec')
    checks['all_core_files_exist_AST_compile']=True
    snap=json.loads((c.ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text());checks['fixed_original7_source_hashes']=all(c.sha(p)==h for p,h in snap['source_hashes'].items())
    oldnames=('common.py','preflight.py','runner.py','audit.py','run.py')
    for name in oldnames:
        old=ast.parse((a.OLD/name).read_text());new=a.transformed(name)
        checks[name+'_numeric_literal_multiset_unchanged']=nums(old)==nums(new)
        compile(new,str(a.OLD/name),'exec')
        orig_assign={n.targets[0].id:n.value for n in old.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
        class Reverse(ast.NodeTransformer):
            def visit_Assign(self,node):
                if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ROOT','CAL','TEACHER','FORMAT','FILES'):
                    node.value=orig_assign[node.targets[0].id];return node
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if node.value==a.MOTOR_FORMAT:return ast.copy_location(ast.Constant('INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'),node)
                return node
        checks[name+'_inverse_metadata_AST_equivalence']=ast.dump(Reverse().visit(new),include_attributes=False)==ast.dump(old,include_attributes=False)
        namespace=a.namespace(name)
        checks[name+'_source3_same_function_globals']=namespace['SOURCE_CANDIDATE']==3 and all(value.__globals__.get('SOURCE_CANDIDATE')==3 for value in namespace.values() if isinstance(value,types.FunctionType) and 'SOURCE_CANDIDATE' in value.__globals__)
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Forbidden numerical import: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        for name in trees:
            spec=importlib.util.spec_from_file_location('_localmean_static_'+name.replace('.','_'),c.ROOT/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            checks[name+'_stdlib_import']=not forbidden.intersection(sys.modules)
            checks[name+'_new_ROOT_globals']=all(value.__globals__.get('ROOT')==c.ROOT for value in vars(module).values() if isinstance(value,types.FunctionType) and 'ROOT' in value.__globals__)
    finally:builtins.__import__=original_import
    p=json.loads((c.ROOT/'PREREG.json').read_text())
    checks['fixed_new_identity']=p['kind']==c.FORMAT+'_PREREG' and p['motor_format']==a.MOTOR_FORMAT and p['source_candidate']==3 and a.SOURCE_ID==3 and c.SOURCE_CANDIDATE==3
    checks['fixed_cases']=p['conditions']=={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
    checks['fixed_budget_and_gates']=p['budget']['max_native_steps']==4928000 and p['budget']['per_case_native_steps']==1232000 and 'all6' in p['gates']['pivot']
    checks['localmean412_explicit']=p['model']['features'].endswith('=412') and p['model']['head']==[192,42] and p['model']['state_dims']==[128,64,64,96]
    checks['no_assisted_mix_or_teacher']=all(token not in ast.unparse(trees['runner.py'])+ast.unparse(trees['preflight.py']) for token in ('student_weight','teacher_weight','mixed_applied_action','teacher_action'))
    checks['strict_entry_source']=not forbidden.intersection(sys.modules)
    checks['no_new_execution_outputs']=not any((c.ROOT/name).exists() for name in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json'))
    hashes={str(c.ROOT/name):c.sha(c.ROOT/name) for name in a.CORE}
    out=dict(status='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN',all_passed=all(checks.values()),checks=checks,source_hashes=hashes,scope='BodyANNcomponent_NOTnewbrainneckdeployment',numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_steps=0,Core_constructions=0,physical_steps=0,training_updates=0,final_checkpoint_SHA_bound=False,new_neck_deployed=False)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static checks failed')


if __name__=='__main__':main()
