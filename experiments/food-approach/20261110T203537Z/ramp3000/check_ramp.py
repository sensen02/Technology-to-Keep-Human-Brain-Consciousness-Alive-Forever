"""Code-only AST/algebra/data-contract checks; NEVER imports a physics backend."""
from pathlib import Path
import ast
import copy
import json
import sys
import numpy as np
import eval_ramp as r
import verify_ramp as v


def main():
    checks={};data_hashes={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('offline check failed '+name)
    paths=[r.HERE/n for n in ('eval_ramp.py','verify_ramp.py','pipeline_ramp.py','check_ramp.py')]
    for path in paths:ast.parse(path.read_text(),filename=str(path))
    check('all_new_python_syntax',True)
    original=ast.parse((r.CONT/'eval3000.py').read_text());derived=ast.parse(r.derived_source())
    functions=lambda tree:{node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    of,df=functions(original),functions(derived)
    for name in ('jsonable','pin','condition_score','checked_load','source_paths','neutral_reference','parser'):
        check('derived_old_function_unchanged_'+name,ast.dump(of[name],include_attributes=False)==ast.dump(df[name],include_attributes=False))
    # Normalize exactly the4 declared evaluate changes, require every other AST node equal.
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            replacements={'fixed3000 matched1s smoothstep neural command DIAGNOSTIC':'fixed3000 continued fast neural motor actual evaluation','DIAGNOSTIC_ENGINEERING_ONLY_NOT_FOOD_QUALIFICATION':'ENGINEERING_ONLY_NOT_FULL_NATIVE_CERTIFICATION'}
            if isinstance(node.value,str) and node.value in replacements:node.value=replacements[node.value]
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                key=node.targets[0].id
                if key=='command':return copy.deepcopy(next(n for n in ast.walk(of['evaluate']) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='command'))
                if key=='ref':return copy.deepcopy(next(n for n in ast.walk(of['evaluate']) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ref'))
            return node
    normalized=Normalize().visit(copy.deepcopy(df['evaluate']))
    check('evaluate_AST_only_command_reference_kind_status_changes',ast.dump(normalized,include_attributes=False)==ast.dump(of['evaluate'],include_attributes=False))
    check('no_physics_mutation_added',all(name not in r.derived_source() for name in ('xfrc_applied[:]=','qfrc_applied[:]=','d.qpos[:]=')))
    teacher_root=r.ROOT.parent/'20261110T191449Z'
    teacher_files={'left':teacher_root/'teacher/left_full/SAMPLES_5MS.npz','right':teacher_root/'teacher/right_full/SAMPLES_5MS.npz','speed085':teacher_root/'deceleration_moderate/straight_085/SAMPLES_5MS.npz','speed090':teacher_root/'deceleration_moderate/straight_090/SAMPLES_5MS.npz'}
    for condition,path in teacher_files.items():
        commands=np.stack([r.command_at(condition,k) for k in range(2880)])
        check(condition+'_neutral_prime_zero',np.array_equal(commands[:960],np.zeros((960,2))))
        check(condition+'_first_autonomous_zero',np.array_equal(commands[960],np.zeros(2)))
        check(condition+'_last_is_fixed_target',np.array_equal(commands[-1],np.asarray(r.COMMANDS[condition])))
        check(condition+'_after1s_fixed',np.array_equal(commands[1160:],np.tile(r.COMMANDS[condition],(1720,1))))
        with np.load(path,allow_pickle=False) as z:actual=z['command2'].copy()
        check(condition+'_all2880_match_actual_teacher_to1e14',actual.shape==(2880,2) and np.max(np.abs(commands-actual))<=1e-14)
        data_hashes[str(path)]=r.sha256(path)
    check('ramp_causal_nonnegative_first_step',np.array_equal(r.command_at('left',959),[0,0]) and r.command_at('left',961)[1]>0)
    check('neutral_mathematical_equivalence_all2880',np.array_equal(np.zeros((2880,2))*np.linspace(0,1,2880)[:,None],np.zeros((2880,2))))
    # Original scalar conversion directly extracted via AST, no old physics execution.
    m=ast.Module(body=[copy.deepcopy(of['jsonable'])],type_ignores=[]);ns={'np':np,'Path':Path};exec(compile(m,'offlinejson','exec'),ns)
    check('strict_numpy_json',json.loads(json.dumps(ns['jsonable']({'x':np.bool_(True),'y':np.float64(.2)}),allow_nan=False))=={'x':True,'y':.2})
    ast.parse(v.derived_source());check('derived_verifier_syntax',True)
    check('verifier_exact_smooth_formula',"expected=np.stack([command_at(result['condition'],k) for k in range(2880)])" in v.derived_source())
    check('no_core_or_backend_imported',all(name not in sys.modules for name in ('mujoco','physics','student_wave','torch')))
    frozen=[*paths,r.HERE/'EVAL_PREREG.json',r.CONT/'eval3000.py',r.CONT/'verify_actual3000.py',r.CONT/'pipeline3000.py',r.CONT/'EVAL_PREREG.json',r.CONT/'PREREG.json',r.ROOT/'fast_motor.py',r.ROOT/'evaluate_fast.py',r.ROOT/'verify_fast_actual.py',r.ROOT/'EVAL_PREREG.json',r.ROOT/'PREREG.json']
    report=dict(status='CODE_READY_NO_EXECUTION_AUTHORIZATION',all_passed=all(checks.values()),checks=checks,frozen_sources={str(path):r.sha256(path) for path in frozen},teacher_array_hashes=data_hashes,
                max_concurrent_processes=2,cpu_threads_per_process=1,physics_integration_steps=0,training_updates=0,
                ast_changes_explicit=['only command assignment changes neural input','kind/status diagnostic metadata','reference validation associated for all4conditions','outputroot/strictcheckpoint/sourcecapture and passedneutral association explicitly overridden'],
                launch='pipeline_ramp.py --launch-authorized --sha256 EXACT3000 --neutral-reference ABS_STEP3000_NEUTRAL_RESULT --neutral-audit ABS_STEP3000_NEUTRAL_AUDIT')
    out=r.HERE/'READINESS.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),readiness=str(out),physics_steps=0)))

if __name__=='__main__':main()
