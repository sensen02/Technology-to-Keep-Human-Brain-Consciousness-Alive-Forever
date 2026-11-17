"""Read-only stdlib AST/metadata/source freeze check. No numerical entry calls."""
from pathlib import Path
import ast
import builtins
import importlib.util
import json
import sys
import types
import common as c
import adapter as a


def main():
    forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave'};checks={};trees={}
    if forbidden.intersection(sys.modules):raise ValueError('Clean stdlib process required')
    for name in a.CORE:
        path=c.ROOT/name
        if not path.is_file():raise FileNotFoundError(path)
        if name.endswith('.py'):trees[name]=ast.parse(path.read_text());compile(trees[name],str(path),'exec')
    checks['all_core_files_AST_compile']=True
    snapshot=json.loads((c.ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text());checks['fixed_original7_source_hashes']=all(c.sha(p)==h for p,h in snapshot['source_hashes'].items())
    for name in a.ENTRIES:
        old=ast.parse((a.OLD/name).read_text());new=a.transformed(name)
        checks[name+'_numeric_literal_multiset_unchanged']=a.numeric(old)==a.numeric(new)
        assigns={n.targets[0].id:n.value for n in old.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
        class Reverse(ast.NodeTransformer):
            def visit_Assign(self,node):
                if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ROOT','CAL','TEACHER','FORMAT','FILES'):
                    node.value=assigns[node.targets[0].id];return node
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if node.value==a.MOTOR_FORMAT:return ast.copy_location(ast.Constant('INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'),node)
                return node
        checks[name+'_inverse_metadata_AST_exact']=ast.dump(Reverse().visit(new),include_attributes=False)==ast.dump(old,include_attributes=False)
        namespace=a.namespace(name)
        checks[name+'_source3_same_function_globals']=namespace['SOURCE_CANDIDATE']==3 and all(v.__globals__.get('SOURCE_CANDIDATE')==3 for v in namespace.values() if isinstance(v,types.FunctionType) and 'SOURCE_CANDIDATE' in v.__globals__)
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Forbidden numerical import: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        for name in trees:
            spec=importlib.util.spec_from_file_location('_targetfeedback_static_'+name.replace('.','_'),c.ROOT/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            checks[name+'_stdlib_import']=not forbidden.intersection(sys.modules)
            checks[name+'_new_ROOT_globals']=all(v.__globals__.get('ROOT')==c.ROOT for v in vars(module).values() if isinstance(v,types.FunctionType) and 'ROOT' in v.__globals__)
    finally:builtins.__import__=original_import
    p=json.loads((c.ROOT/'PREREG.json').read_text())
    checks['new_kind_and_targetfeedback_format']=p['format']==a.FORMAT and p['motor_format']==a.MOTOR_FORMAT=='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
    checks['CAL_orderrepair_not_failed_envscope']=c.CAL==c.ROOT.parent/'target_feedback_order_repair_v2' and p['CAL']==str(c.CAL)
    checks['fixed_four_tasks']=c.CONDITIONS==p['conditions']=={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
    checks['pure_source3_warm2_settle0']=c.SOURCE_CANDIDATE==p['source_candidate']==3
    checks['fixed_budget_and_9stages']=p['budget']['main_stages']==9 and p['budget']['max_native_steps']==4928000 and p['budget']['per_case_native_steps']==1232000
    checks['same352_API_no_actor_features_copy']=p['state_dims']==[128,64,64,96] and not any(isinstance(n,ast.Constant) and n.value in (370,412) for name in ('runner.py','audit.py','preflight.py') for n in ast.walk(trees[name]))
    checks['actual_source_only_no_new_outputs']=not forbidden.intersection(sys.modules) and not any((c.ROOT/n).exists() for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json'))
    # Producer source is read, never imported: capture exact schemas while the
    # repair remains a separate frozen scientific scope.
    producer_names=('common.py','model.py','sources.py','prepare.py','preflight.py','train.py','run.py','adapter.py')
    producer={str(c.CAL/n):c.sha(c.CAL/n) for n in producer_names}
    original=c.CAL.parent/'target_feedback_calibration'
    for name in ('verify_saved.py','evaluate.py','audit_eval.py'):producer[str(original/name)]=c.sha(original/name)
    hashes={str(c.ROOT/n):c.sha(c.ROOT/n) for n in a.CORE}
    result=dict(status='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN',all_passed=all(checks.values()),checks=checks,source_hashes=hashes,producer_schema_source_hashes=producer,scope='BodyANNcomponent_NOTnewbrainneckdeployment',numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_steps=0,Core_constructions=0,physical_steps=0,training_updates=0,final_checkpoint_SHA_bound=False,new_neck_deployed=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static checks failed')


if __name__=='__main__':main()
