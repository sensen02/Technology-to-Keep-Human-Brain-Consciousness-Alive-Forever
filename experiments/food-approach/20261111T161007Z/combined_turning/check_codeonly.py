"""Only stdlib AST/string-adaptation/math mocks. Never import torch/model/Core here."""
from pathlib import Path
import ast,json,hashlib,math,sys
HERE=Path(__file__).resolve().parent

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def functions_only(path,names,namespace):
 tree=ast.parse(path.read_text());nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
 if {node.name for node in nodes}!=set(names):raise ValueError('missing pure source-adaptation helper')
 module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));exec(compile(module,str(path)+'::stdlib_helpers_only','exec'),namespace);return namespace

def main():
 checks={};sources={};trees={}
 for path in sorted(HERE.glob('*.py')):trees[path.name]=ast.parse(path.read_text());sources[str(path)]=digest(path)
 checks['all_python_AST_parsed']=True
 model=trees['model.py'];dims=next(ast.literal_eval(n.value) for n in model.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STATE_DIMS' for t in n.targets));checks['three_states256']=dims==(128,64,64) and sum(dims)==256
 cls=next(n for n in model.body if isinstance(n,ast.ClassDef) and n.name=='CombinedTurning');step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step');calls=[ast.unparse(n.func) for n in ast.walk(step) if isinstance(n,ast.Call)]
 checks['one_sharedbase_step_static']=calls.count('self.base.step')==1;checks['both_neural_branches_advance_static']=calls.count('self.right_branch.advance')==calls.count('self.left_branch.advance')==1;checks['no_step_reset_static']=not any('initial_state' in name or 'reset' in name for name in calls);checks['no_physics_calls_in_model']=not any(key in name for name in calls for key in ('mj_step','tick_rate','teacher.action','reset_from'))
 flat=list(range(256));parts=(flat[:128],flat[128:192],flat[192:]);checks['mock_three_state_layout_roundtrip']=tuple(map(len,parts))==(128,64,64) and sum((list(x) for x in parts),[])==flat
 alpha=-math.expm1(-.005/.1);gate_results=[]
 for turn in (-1.,-.3,0.,.3,1.):
  pos=max(turn,0.);neg=max(-turn,0.);external=[pos/3.,turn];cr=[0.,-neg];cl=[pos/3.,pos];base=.7;rh=(1-alpha)*0.+alpha*math.tanh(.1+sum(cr));lh=(1-alpha)*0.+alpha*math.tanh(.2+sum(cl));mean=base+neg*3.*math.tanh(rh)+pos*6.*math.tanh(lh)
  expected=base if turn==0 else (base+neg*3.*math.tanh(rh) if turn<0 else base+pos*6.*math.tanh(lh));checks['mock_gate_'+str(turn)]=mean==expected and external[0]==max(external[1],0.)/3.;gate_results.append(dict(turn=turn,command=external,cmdR=cr,cmdL=cl,neutral_mean=mean if turn==0 else None))
  if turn==0:checks['mock_both_inactive_memories_still_update']=rh!=0. and lh!=0.
 primary=HERE.parent.parent/'20261110T235830Z/evaluation/evaluate_motor.py';ns=functions_only(HERE/'evaluate_motor.py',['replace_once','replace_function','adapted_source'],{'ast':ast,'PRIMARY':primary});rollout=ns['adapted_source']();ast.parse(rollout);checks['immutable_rollout_exact_replace_counts']=True;checks['rollout_layout_explicit256']=rollout.count('state_layout=[128,64,64]')==2 and rollout.count('state_dims=[128,64,64]')==1 and 'args.composition_ready' in rollout
 verify_primary=HERE.parent.parent/'20261110T235830Z/evaluation/verify_actual.py';audit_ns=functions_only(HERE/'verify_actual.py',['replace_once','adapted_source'],{'PRIMARY':verify_primary});audit=audit_ns['adapted_source']();ast.parse(audit);checks['immutable_saved_audit_exact_replace_counts']=True
 for key in ('full2880_threehidden_decisions','full256_hidden_chain','actual2880_checkpoint_threehidden_exact'):checks['audit_key_'+key]=key in audit
 checks['no_old192_actual_schema']='(2880,192)' not in audit and 'np.zeros(192)' not in audit;checks['explicit256_schema']='(2880,256)' in audit and 'np.zeros(256)' in audit and '[128,64,64]' in audit
 model_io=(HERE/'model_io.py').read_text();prepare=(HERE/'prepare.py').read_text();checks['AMENDMENT03_source_and_bindings']=('AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json' in model_io and 'composition_scope_hashes' in model_io and 'AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json' in prepare)
 checks['AMENDMENT04_source_and_bindings']=('AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json' in model_io and 'AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json' in prepare and 'visual_original_feature_refine' in prepare)
 for name in ('PREREG.json','AMENDMENT_01_LEFT_PENDING_FEATURE_REFINE.json','AMENDMENT_02_CONDITIONAL_RIGHT_FEATURE_REFINE.json','AMENDMENT_03_HONEST_DEMONSTRATION_COMPONENTS.json','AMENDMENT_04_POSTHOC_VISUAL_RIGHT_COMPONENT.json','COMPONENT_PLAN.json','EVAL_PREREG.json'):json.loads((HERE/name).read_text());sources[str(HERE/name)]=digest(HERE/name)
 checks['no_heavy_imports_executed']='torch' not in sys.modules and 'numpy' not in sys.modules and 'mujoco' not in sys.modules;checks['no_composed_artifact_created']=not (HERE/'COMBINED.pt').exists() and not (HERE/'COMPONENT_BINDINGS.json').exists()
 result=dict(status='CODE_ONLY_CHECKS_NOT_MODEL_OR_PHYSICS_VALIDATION',all_passed=all(checks.values()),checks=checks,source_hashes=sources,state_layout=[128,64,64],state_total=256,gating_mocks=gate_results,actual_component_loads=0,Core_constructions=0,model_preflights=0,physics_integration_steps=0,training_performed=False,composed_checkpoint_created=False,LEFT_formal_status='knownangle+slipFAIL perAMENDMENT03',RIGHT_formal_status='b4ed observedslip.503296FAIL, selected POSTHOC_VISUAL_ONLY perAMENDMENT04; qualified replacement rule history retained')
 out=HERE/'CODEONLY.json'
 if out.exists():raise ValueError('CODEONLY artifact immutable; use new amendment/recheck output if needed')
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),state_total=256,actual_component_loads=0,Core_constructions=0)),flush=True)
 if not result['all_passed']:raise ValueError('combined CODEONLY check failed')
if __name__=='__main__':main()
