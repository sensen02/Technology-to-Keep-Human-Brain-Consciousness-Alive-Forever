"""Offline binding and unchanged-equation checks; no Core or physical integration."""
from pathlib import Path
import ast
import importlib.util
import json
import sys
import types
import numpy as np
import evaluate_motor as control

HERE=Path(__file__).resolve().parent


def function_ast(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)


def main():
    checks={}
    for name in ('evaluate_motor.py','verify_actual.py','pipeline.py','check_prepared.py'):
        ast.parse((HERE/name).read_text());checks['syntax_'+name]=True
    prereg=control.verify_primary();checks['all_frozen_primary_hashes_match']=True
    primary=(control.PRIMARY/'evaluate_motor.py').read_text();derived=control.derived_source()
    for name in ('command_at','score','pin','flatten','checked_load','parser'):
        checks['original_function_AST_'+name]=function_ast(primary,name)==function_ast(derived,name)
    old_eval=next(n for n in ast.parse(primary).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    new_eval=next(n for n in ast.parse(derived).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    checks['all_physics_inference_loop_AST_identical']=[ast.dump(n,include_attributes=False) for n in ast.walk(old_eval) if isinstance(n,ast.For)]==[ast.dump(n,include_attributes=False) for n in ast.walk(new_eval) if isinstance(n,ast.For)]
    checks['only_one_kind_metadata_change']=primary.replace("kind='round2 fixed1500 slow residual actual physical evaluation'","kind='round2 original-two-only budgetmatched fixed1500 control actual evaluation'")==derived
    checks['actual_control_train_binding']=control._MODULE.TRAIN==control.ROOT/'train_two_only_control' and control._MODULE.CHECKPOINT==control.CHECKPOINT
    paths=control.source_paths()
    checks['source_captures_control_amendment']=control.ROOT/'TWO_TRAJECTORY_CONTROL_AMENDMENT.json' in paths
    checks['source_captures_control_modelio']=control.TRAIN/'model_io.py' in paths
    checks['no_primary_training_sources'] = not any(p.is_relative_to(control.ROOT/'train') for p in paths)
    checks['source_captures_old_evaluator_identity']=all(Path(p) in paths for p in prereg['frozen_primary_sources'])
    class DummyModel:
        def eval(self):return self
    class DummyIO:
        def __init__(self,step):self.step=step;self.calls=0
        def load_checked(self,*a,**kw):
            self.calls+=1
            if kw!={'require_final':True}:raise ValueError('strict final flag missing')
            return DummyModel(),dict(curve=[dict(step=self.step)],config=dict(physical_eval_only_step=1500))
    rejected=False;io=DummyIO(1500)
    try:control.checked_load(io,control.ROOT/'train/TRAIN/checkpoint_001500.pt','a'*64)
    except ValueError:rejected=True
    checks['primary_checkpoint_path_rejected_before_loader']=rejected and io.calls==0
    rejected=False
    try:control.checked_load(DummyIO(1000),control.CHECKPOINT,'a'*64)
    except ValueError:rejected=True
    checks['intermediate1000_rejected']=rejected
    checks['final1500_strict_flag_and_scope']=control.checked_load(DummyIO(1500),control.CHECKPOINT,'a'*64)[1]['curve'][-1]['step']==1500
    auto=dict(all_gates=True,nonfoot_force_max_native=0.);cycles=dict(all_legs_completed=True)
    checks['unchanged_score_accept_and_reject']=control.score('left',auto,cycles,.01)['eligible'] and not control.score('right',auto,cycles,.01)['eligible']
    checks['native_json_numpybool']=json.loads(json.dumps(control.jsonable(dict(flag=np.bool_(True),values=np.array([1.,2.]))),allow_nan=False))==dict(flag=True,values=[1.,2.])
    for condition in ('left','right'):
        actual=np.asarray([control.command_at(condition,k) for k in range(2880)])
        path=control.ROOT.parent/f'20261110T191449Z/teacher/{condition}_full/SAMPLES_5MS.npz'
        with np.load(path,allow_pickle=False) as z:expected=z['command2'].copy()
        checks[condition+'_all2880_teacher_ramp_match']=np.max(np.abs(actual-expected))<=1e-14
        checks[condition+'_neutral_prime_firstANN']=np.array_equal(actual[:961],np.zeros((961,2)))
    checks['pre_model_no_physics_import']='physics' not in sys.modules and 'mujoco' not in sys.modules
    # Public control model_io uses the exact legacy actor; load immutable old500 for tuple API checking only.
    spec=importlib.util.spec_from_file_location('_two_only_offline_public_modelio',control.TRAIN/'model_io.py')
    io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)
    model,payload=io.load_parent()
    import torch
    torch.set_num_threads(1)
    for condition in ('left','right'):
        path=control.ROOT.parent/f'20261110T191449Z/teacher/{condition}_full/SAMPLES_5MS.npz'
        with np.load(path,allow_pickle=False) as z:x=z['X238'].copy();u=z['command2'].copy()
        state=model.initial_state(1);flat=control.flatten(state).clone();means=True;states=True
        with torch.no_grad():
            for xx,uu in zip(x,u):
                obs=torch.tensor(xx[None],dtype=torch.float64);cmd=torch.tensor(uu[None],dtype=torch.float64)
                mean,state=model.step(obs,cmd,state)
                other,otherstate=model.step(obs,cmd,(flat[:,:128],flat[:,128:]));flat=control.flatten(otherstate)
                means=means and torch.equal(mean,other);states=states and torch.equal(control.flatten(state),flat)
        checks[condition+'_all2880_publiccontrol_tuple_means_exact']=means
        checks[condition+'_all2880_publiccontrol_all192states_exact']=states
    checks['tuple_test_zero_physics_import']='physics' not in sys.modules and 'mujoco' not in sys.modules
    sources={str(HERE/n):control.sha256(HERE/n) for n in ('evaluate_motor.py','verify_actual.py','pipeline.py','EVAL_PREREG.json','check_prepared.py')}
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=sources,frozen_primary_sources=prereg['frozen_primary_sources'],
                control_model_io_sha256=control.sha256(control.TRAIN/'model_io.py'),physical_integration_steps=0,optimizer_updates=0,
                scope='CODE_ONLY own control path/identity and immutable actor tuple API; strict final1500 checkpoint not yet physically executed')
    (HERE/'READINESS.json').write_text(json.dumps(control.jsonable(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(control.jsonable(report),indent=2))
    if not report['all_passed']:raise ValueError('control preparation checks failed')

if __name__=='__main__':main()
