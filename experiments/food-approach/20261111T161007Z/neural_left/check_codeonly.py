"""Standard-library-only AST/adapter checks; never imports project/Torch/Numpy/physics."""
from pathlib import Path
import ast,json,hashlib,types
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
ROLLOUT=ROOT/'20261110T235830Z/evaluation/evaluate_motor.py'
AUDITOR=ROOT/'20261110T235830Z/evaluation/verify_actual.py'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def function_namespace(path,names,namespace):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('missing static adapter function')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::AST_SELECTED_STDLIB_ONLY','exec'),namespace)
 return namespace

def main():
 checks={};paths=sorted(HERE.glob('*.py'))+sorted(HERE.glob('*PREREG.json'))
 for path in paths:
  if path.suffix=='.py':compile(ast.parse(path.read_text()),str(path),'exec')
  else:json.loads(path.read_text())
 checks['all_local_AST_JSON_parse']=True
 ev=function_namespace(HERE/'evaluate_motor.py',('replace_once','adapted_source'),{'PRIMARY':ROLLOUT})
 rollout=ev['adapted_source']();compile(rollout,str(ROLLOUT)+'::STATIC_LEFT_ADAPTATION','exec');checks['exact_count_rollout_adaptations_compile']=True
 audit=function_namespace(HERE/'verify_actual.py',('adapted_source',),{'PRIMARY':AUDITOR,'ev':types.SimpleNamespace(replace_once=ev['replace_once'])})
 compiled_audit=audit['adapted_source']();compile(compiled_audit,str(AUDITOR)+'::STATIC_LEFT_ADAPTATION','exec');checks['exact_count_independent_audit_adaptations_compile']=True
 # Inspect the actual adapted evaluator function before any future Core construction.
 tree=ast.parse(rollout);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
 writes=[n for n in ast.walk(fn) if isinstance(n,(ast.Assign,ast.AugAssign,ast.AnnAssign)) and any(k in ast.unparse(n).split('=')[0] for k in ('.qpos','.qvel'))]
 resets=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('reset','reset_from')]
 checks['adapted_rollout_no_root_state_writes_or_resets']=not writes and not resets
 checks['LEFT_only_no_RIGHT_format']='RIGHT_ONLY' not in (HERE/'model_io.py').read_text() and 'strict_LEFT_HEAD_V1' in (HERE/'model_io.py').read_text()
 checks['LEFT_4000_and_30_unrounded']='np.degrees(yaw)>=30.' in (HERE/'evaluate_motor.py').read_text() and "checkpoint_004000.pt" in (HERE/'evaluate_motor.py').read_text()
 teacher=HERE.parent/'left_tarsal_actual_guard/left_yaw_p060/RESULT.json';auditpath=teacher.parent.parent/'AUDIT_ACTUAL_GUARD_20261111T183323Z_377784_1794422003628573149.json'
 r=json.loads(teacher.read_text());a=json.loads(auditpath.read_text());checks['teacher_JSON_and_audit_binding']=a['integrity_all_pass'] and a['teacher_eligible'] and a['result_sha256']==sha(teacher) and r['teacher_eligible'] and r['actual_signed_yaw_deg']>=30. and r['native']['post_nonfoot_peak_native']==0
 source=paths+[ROLLOUT,AUDITOR,ROOT/'20261110T191449Z/residual_candidate/residual_motor.py',Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/models.py')]
 report=dict(status='CODEONLY_READY_RESOURCE_PENDING',all_static_checks_passed=all(checks.values()),checks=checks,source_hashes={str(p):sha(p) for p in source},evidence_hashes={str(p):sha(p) for p in (teacher,auditpath)},model_preflight='NOT_RUN',Torch_imports=0,model_forwards=0,array_loads=0,Core_constructions=0,physics_steps=0,training_updates=0)
 print(json.dumps(report,indent=2,allow_nan=False))
 if not all(checks.values()):raise ValueError('CODEONLY static checks failed')
if __name__=='__main__':main()
