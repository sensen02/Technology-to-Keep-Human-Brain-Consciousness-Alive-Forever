"""Stdlib identity migration checks only; no qualification/main or numeric execution."""
import ast
import builtins
import importlib.util
import json
import sys
import types
import adapter as a
import common as c


def main():
    checks={};forbidden={'torch','numpy','mujoco','physics','flygym','student_wave','scipy'}
    if forbidden.intersection(sys.modules):raise ValueError('Clean stdlib process required')
    for name in a.CORE:
        path=a.ROOT/name
        if name.endswith('.py'):compile(ast.parse(path.read_text()),str(path),'exec')
        if not path.is_file():raise FileNotFoundError(path)
    checks['all_new_CORE_exist_AST_compile']=True
    snapshot=json.loads((a.ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text());checks['original7_and_previous2_fixed_hashes']=all(a.sha(p)==h for p,h in snapshot['source_hashes'].items())
    for name in a.ENTRIES:
        original=ast.parse((a.OLD/name).read_text());migrated=a.transformed(name)
        checks[name+'_original_numeric_literals_unchanged']=a.numeric(original)==a.numeric(migrated)
        # Original body AST differs from previous body only metadata ROOT/CAL/FILES.
        ns=a.namespace(name)
        checks[name+'_new_ROOT_source3_globals']=ns['ROOT']==a.ROOT and ns['SOURCE_CANDIDATE']==3 and all(v.__globals__.get('ROOT')==a.ROOT and v.__globals__.get('SOURCE_CANDIDATE')==3 for v in ns.values() if isinstance(v,types.FunctionType) and 'SOURCE_CANDIDATE' in v.__globals__)
    oldprepare=ast.parse((a.PREVIOUS/'prepare.py').read_text());newprepare=a.relocated('prepare.py')
    checks['prepare_numeric_literals_unchanged']=a.numeric(oldprepare)==a.numeric(newprepare)
    class Reverse(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='target_feedback_order_repair_v4':return ast.copy_location(ast.Constant('target_feedback_order_repair_v2'),node)
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_v4_lineage':return ast.copy_location(ast.Subscript(value=ast.Name(id='b',ctx=ast.Load()),slice=ast.Constant('resume_lineage'),ctx=ast.Load()),node)
            return self.generic_visit(node)
    checks['prepare_inverse_metadata_schema_AST_exact']=ast.dump(Reverse().visit(newprepare),include_attributes=False)==ast.dump(oldprepare,include_attributes=False)
    oldadapter=ast.parse((a.PREVIOUS/'adapter.py').read_text());newadapter=a.relocated('adapter.py')
    checks['adapter_inverse_CAL_identity_AST_exact']=ast.dump(Reverse().visit(newadapter),include_attributes=False)==ast.dump(oldadapter,include_attributes=False)
    orig=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Prohibited numerical import: '+name)
        return orig(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        for name in a.CORE:
            if not name.endswith('.py'):continue
            spec=importlib.util.spec_from_file_location('_v3body_static_'+name.replace('.','_'),a.ROOT/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            checks[name+'_stdlib_import']=not forbidden.intersection(sys.modules)
        p=a.prepare_namespace();checks['prepare_new_ROOT_CAL']=p['ROOT']==a.ROOT and p['CAL']==a.ROOT.parent/'target_feedback_order_repair_v4'
    finally:builtins.__import__=orig
    checks['CAL_v4_only']=c.CAL==a.ROOT.parent/'target_feedback_order_repair_v4'
    checks['source3_fixed_four_tasks']=c.SOURCE_CANDIDATE==3 and c.CONDITIONS=={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
    prereg=json.loads((a.ROOT/'PREREG.json').read_text());checks['fixed_budget4928000_9stages']=prereg['budget']['max_native_steps']==4928000 and prereg['budget']['main_stages']==9
    checks['v4_same_training_storage_stage_producers']=all(a.sha(a.ROOT.parent/'target_feedback_order_repair_v4'/n)==a.sha(a.ROOT.parent/'target_feedback_order_repair_v2'/n) for n in ('train.py','run.py','verify_saved.py','evaluate.py','audit_eval.py'))
    # Only actual JSON schemas, no array payload; do not call metadata qualifier.
    binding=json.loads((a.ROOT.parent/'target_feedback_order_repair_v4/INPUT_BINDINGS.json').read_text())
    checks['actual_v4_schema_resource_not_lineage']='resource_amendment' in binding and 'resume_lineage' not in binding and 'namespace_recovery' not in binding and all(str(a.ROOT.parent/scope/'INPUT_BINDINGS.json') in binding['data_hashes'] for scope in ('target_feedback_order_repair_v3','target_feedback_order_repair_v2'))
    projected=a._v4_lineage(binding)
    checks['actual_v4_metadata_projection_std_only']=projected['start_update']==300 and projected['new_updates']==300 and projected['total_final_update']==600 and projected['save_steps']==[300,450,600]
    checks['reference3072_training1800_separate']=binding['resource_amendment']['reference_preflight_RSS_MiB']==3072 and binding['resource_amendment']['training_RSS_MiB']==1800
    checks['unknown_final_not_bound']=not any((a.ROOT/n).exists() for n in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json'))
    result=dict(status='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN',all_passed=all(checks.values()),checks=checks,source_hashes={str(a.ROOT/n):a.sha(a.ROOT/n) for n in a.CORE},scope='BodyANNcomponent_NOTnewbrainneckdeployment',numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_steps=0,Core_constructions=0,physical_steps=0,training_updates=0,final_checkpoint_SHA_bound=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static migration checks failed')


if __name__=='__main__':main()
