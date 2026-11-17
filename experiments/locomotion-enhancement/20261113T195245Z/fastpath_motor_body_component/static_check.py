"""Stdlib static/read-only checker; prints freeze manifest, never runs numerical code."""
from pathlib import Path
import ast
import builtins
import importlib
import json
import sys
import types
from adapter import ROOT,OLD,CAL,FORMAT,MOTOR_FORMAT,CORE,ENTRIES,sha,transformed,numeric


def main():
    checks={};forbidden=('torch','numpy','mujoco','physics','flygym','student_wave','scipy')
    if any(name in sys.modules for name in forbidden):raise ValueError('Clean stdlib process required')
    for path in ROOT.glob('*.py'):compile(ast.parse(path.read_text()),str(path),'exec')
    checks['all_local_Python_AST_compile']=True
    snap=json.loads((ROOT/'OLD_SOURCE_SNAPSHOT.json').read_text())
    checks['fixed_beforefreeze_original7core_hashes']=len(snap['source_hashes'])==7 and all(sha(p)==h for p,h in snap['source_hashes'].items())
    changes={}
    for name in ENTRIES:
        old=ast.parse((OLD/name).read_text());new=transformed(name)
        checks[name+'_numeric_literal_multiset_unchanged']=numeric(old)==numeric(new)
        # Strong structural equivalence: reverse the declared metadata assignments
        # and exact FORMAT constant replacement, then demand full AST equality.
        orig_assign={n.targets[0].id:n.value for n in old.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
        class Reverse(ast.NodeTransformer):
            def visit_Assign(self,node):
                if name=='common.py' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('ROOT','CAL','TEACHER','FORMAT','FILES'):
                    node.value=orig_assign[node.targets[0].id];return node
                return self.generic_visit(node)
            def visit_Constant(self,node):
                if node.value==MOTOR_FORMAT:return ast.copy_location(ast.Constant('INDEPENDENT_SPEED_YAW_MOTOR_RESIDUAL_V1'),node)
                return node
        checks[name+'_metadata_only_AST_equivalence']=ast.dump(Reverse().visit(new),include_attributes=False)==ast.dump(old,include_attributes=False)
        changes[name]=sha(OLD/name)
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in forbidden:raise RuntimeError('Prohibited numerical import: '+name)
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded
    try:
        import common
        from adapter import namespace
        for name in ENTRIES:
            ns=namespace(name)
            checks[name+'_new_ROOT_FORMAT']=ns['ROOT']==ROOT and ns['FORMAT']==FORMAT
            checks[name+'_all_function_globals_new_ROOT']=all(v.__globals__.get('ROOT')==ROOT for v in ns.values() if isinstance(v,types.FunctionType) and 'ROOT' in v.__globals__)
        checks['common_new_CAL_oldqualified_TEACHER']=common.CAL==CAL and common.TEACHER==OLD.parent/'pivot_teacher_design'
        checks['common_frozen_manifest_includes_adapter_snapshot']=tuple(common.FILES)==CORE and all((ROOT/n).is_file() for n in CORE)
    finally:builtins.__import__=original_import
    checks['imports_stdlib_only']=not any(name in sys.modules for name in forbidden)
    oldpre=json.loads((OLD/'PREREG.json').read_text());pre=json.loads((ROOT/'PREREG.json').read_text())
    excluded={'kind','format','motor_format','explicit_scope','qualification_extension','reuse'}
    checks['all_original_prereg_protocol_fields_identical']=all(pre[k]==v for k,v in oldpre.items() if k not in excluded)
    checks['fixed_four_commands']=common.CONDITIONS=={'walk':[1.,0.],'stop':[0.,0.],'left_pivot':[0.,1.],'right_pivot':[0.,-1.]}
    checks['budget_original4928000']=pre['budget']==oldpre['budget'] and pre['budget']['max_native_steps']==4928000
    b=json.loads((OLD/'INPUT_BINDINGS.json').read_text());summary=json.loads((OLD/'SUMMARY.json').read_text());rows=[]
    for case in common.CONDITIONS:
        folder=OLD/'trials'/case;r=json.loads((folder/'RESULT.json').read_text());v=json.loads((folder/'VERIFY.json').read_text())
        checks[case+'_old_actual42savedchecks']=len(v['checks'])==42 and all(v['checks'].values()) and v['integrity_all_pass'] and v['result_sha256']==sha(folder/'RESULT.json')
        checks[case+'_old_actual_failed_fullwindow']=r['candidate_metrics']['task_pass_candidate'] is False and r['actual_native_steps']==1232000 and r['checkpoint_sha256']==b['checkpoint_sha256']==snap['old_checkpoint_sha256']
        rows.append(dict(condition=case,result_sha256=sha(folder/'RESULT.json'),verify_sha256=sha(folder/'VERIFY.json'),legacy_slip=r['original_five']['slip_ratio_max'],candidate_metrics=r['candidate_metrics'],native=r['native']))
    checks['old_summary_all4_actual_native']=len(summary['rows'])==4 and summary['actual_new_native_steps']==4928000
    checks['new_binding_not_prematurely_written']=not (ROOT/'INPUT_BINDINGS.json').exists()
    checks['newcal_actual_FORMAT_source']=ast.literal_eval(next(n.value for n in ast.parse((CAL/'common.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in n.targets)))==MOTOR_FORMAT
    source={str(ROOT/n):sha(ROOT/n) for n in CORE}
    result=dict(status='READYSTD_FROZEN_UNBOUND_FINAL600_BEHAVIOR_UNKNOWN',checks=checks,all_passed=all(checks.values()),source_hashes=source,original_AST_source_hashes=changes,old_actual_metadata=rows,scope='BodyANNcomponent_NOTnewbrainneckdeployment',numeric_imports=0,NPZ_payload_reads=0,checkpoint_loads=0,NN_steps=0,Core_constructions=0,physical_steps=0,training_updates=0,final_checkpoint_SHA_bound=False,new_neck_deployed=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if not all(checks.values()):raise ValueError('Static checks failed')


if __name__=='__main__':main()
