"""Read-only stdlib AST/JSON/sourcehash check. No numeric or entry main calls."""
from pathlib import Path
import ast
import builtins
import importlib.util
import json
import sys
import types
import common as c


def literal_assign(tree,name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def main():
    checks={};forbidden={'numpy','torch','scipy','mujoco','physics','flygym','student_wave'}
    if forbidden.intersection(sys.modules):raise ValueError('Clean stdlib process required')
    trees={}
    for name in c.FILES:
        path=c.ROOT/name
        if not path.is_file():raise FileNotFoundError(path)
        if name.endswith('.py'):trees[name]=ast.parse(path.read_text());compile(trees[name],str(path),'exec')
    checks['all_core_files_exist_and_Python_AST_compile']=True
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Prohibited numerical import: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        for name in trees:
            spec=importlib.util.spec_from_file_location('_collection_static_'+name.replace('.','_'),c.ROOT/name)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            checks[name+'_import_stdlib_only']=not forbidden.intersection(sys.modules)
            checks[name+'_new_ROOT_globals']=all(value.__globals__.get('ROOT')==c.ROOT for value in vars(mod).values() if isinstance(value,types.FunctionType) and 'ROOT' in value.__globals__)
    finally:builtins.__import__=original_import
    checks['common_selected_helpers_newglobals']=all(getattr(c,n).__globals__ is c.__dict__ for n in c._SELECTED)
    checks['common_original_helper_hash_fixed']=c.sha(c._HELPER)=='788577b88d44c79f206e3ef2f003c677537809ec35511f9043048b2598e7d1d1'
    prereg=c.read(c.ROOT/'PREREG.json')
    checks['fixed_four_cases_commands']=c.CASES=={'old_walk_control':('walk',[1.,0.]),'assisted_stop':('stop',[0.,0.]),'assisted_left_pivot':('left_pivot',[0.,1.]),'assisted_right_pivot':('right_pivot',[0.,-1.])} and list(prereg['cases'])==list(c.CASES)
    checks['fixed_budget4times1232000']=prereg['budget']['native_steps_max']==4928000 and prereg['budget']['per_case_native_steps_max']==1232000 and prereg['budget']['settle_s']==1. and prereg['budget']['oldteacherwarm_s']==4.8 and prereg['budget']['collection_s']==9.6
    checks['fixed12stages_no_additional_trajectory']=prereg['budget']['main_stages']==12 and 'len(jobs)!=12' in ast.unparse(trees['run.py']).replace(' ','')
    checks['explicit_non_neural_acceptance_scope']=prereg['scope']==c.SCOPE=='BodyANNcomponent_NOTnewbrainneckdeployment' and prereg['label']==c.LABEL=='TEACHER_ASSISTED_DATA_NOT_NEURAL_ACCEPTANCE'
    checks['actual007d_identity']=c.ACTUAL_SHA=='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd' and prereg['model']['checkpoint_sha256']==c.ACTUAL_SHA
    checks['original11mask_names']=tuple(prereg['label_mask_names'])==c.MASK_NAMES and len(c.MASK_NAMES)==11
    checks['fixed75teacher25student']=prereg['blend']['student_beta']==.25 and prereg['blend']['teacher_weight']==.75 and prereg['blend']['adaptive_beta_or_scan'] is False
    # Compare production new-path formulas structurally to the actual frozen model.
    source_model=ast.parse((c.CAL/'model.py').read_text())
    model_class=next(n for n in source_model.body if isinstance(n,ast.ClassDef) and n.name=='IndependentMotor')
    production=next(n for n in model_class.body if isinstance(n,ast.FunctionDef) and n.name=='step_details')
    shadow=next(n for n in trees['common.py'].body if isinstance(n,ast.FunctionDef) and n.name=='shadow_details')
    def assignments(function):
        return {n.targets[0].id:n.value for n in ast.walk(function) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    pa=assignments(production);sa=assignments(shadow)
    class Canonical(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Name(id='self',ctx=node.ctx),node) if node.id=='m' else node
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='backend' and node.attr in ('ALPHA','BOUND'):return ast.copy_location(ast.Name(id=node.attr,ctx=ast.Load()),node)
            return self.generic_visit(node)
    checks['shadow_formula_matches_actual_model']=all(ast.dump(Canonical().visit(sa[name]),include_attributes=False)==ast.dump(pa[name],include_attributes=False) for name in ('encoded','candidate','h','residual'))
    checks['shadow_one_old_cached_step']=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cached_features' for n in ast.walk(shadow))==1 and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','step_details') for n in ast.walk(shadow))
    collector_text=ast.unparse(trees['collector.py'])
    checks['collector_directold_action_separate']=all(text in collector_text for text in ('oldmean_direct.tanh()','oldmean_direct=oldmean_direct[0]','shadow_details(motor, xt, ut, state)'))
    checks['collector_no_live_root_or_force_assignment']=not any(isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any('c.d.' in ast.unparse(target) for target in (n.targets if isinstance(n,ast.Assign) else [n.target])) for n in ast.walk(trees['collector.py']))
    checks['collector_no_NNreset_after_handover']=collector_text.count('motor.initial_state(')==1 and 'state = motor.initial_state(1)' in collector_text
    checks['teacher_11mask_independent_audit']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='teacher_mask' for n in ast.walk(trees['audit_teacher.py']))
    checks['teacher_audit_standalone_restore_not_old_querymain']='restore_api()' in ast.unparse(trees['audit_teacher.py']) and not any(isinstance(n,ast.Import) and any(a.name=='query' for a in n.names) for n in ast.walk(trees['audit_teacher.py']))
    checks['entry_imports_no_numeric_backend']=not forbidden.intersection(sys.modules)
    checks['new_collection_not_executed']=not any((c.ROOT/name).exists() for name in ('INPUT_BINDINGS.json','PREFLIGHT.json','JOBS.json','SUMMARY.json'))
    hashes={str(c.ROOT/name):c.sha(c.ROOT/name) for name in c.FILES}
    report=dict(status='READYSTD_FROZEN_UNBOUND_ASSISTED_COLLECTION_NOT_EXECUTED',all_passed=all(checks.values()),checks=checks,source_hashes=hashes,label=c.LABEL,scope=c.SCOPE,numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_steps=0,Core_constructions=0,physical_steps=0,training_updates=0)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static collection checks failed')


if __name__=='__main__':main()
