"""Stdlib-only proof of targeted final1000 adapter and unchanged physical/sensory constants."""
from pathlib import Path
import ast,json,hashlib,sys
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 out=ROOT/'CODEONLY_CHECK.json'
 if out.exists():raise FileExistsError('exclusive CODE_ONLY proof')
 files=('runtime.py','adapter.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py');checks={}
 for name in files:ast.parse((ROOT/name).read_text());checks[name+'_syntax']=True
 import runtime,adapter
 def numbers(tree):return [n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and type(n.value) in (int,float)]
 for name in ('runner.py','audit.py','summary.py'):
  ns=adapter.namespace(name);checks[name+'_new_ROOT']=ns['ROOT']==ROOT;checks[name+'_new_POLICY']=ns['POLICY']==ROOT.parent/'policy'
  checks[name+'_explicit_original_project']=ns['PROJECT']==Path('/run/media/sensen/Data2/cell_wound_prototype')
  checks[name+'_new_classes_loader']=ns['CameraStream'] is runtime.CameraStream and ns['load_policies'] is runtime.load_policies
  checks[name+'_faranchors_unchanged']=ns['anchor_landmarks'] is runtime.anchor_landmarks
  source=adapter._far_adapter.source(name);change=adapter.CoverageScope(name);adapted=change.visit(ast.parse(ast.unparse(source)))
  checks[name+'_all_numeric_constants_unchanged']=numbers(source)==numbers(adapted)
 for name in ('prepare.py','preflight.py','summary.py','orchestrate.py'):
  ns=adapter.entry_namespace(name);checks[name+'_entry_bound']=ns['ROOT']==ROOT and ns['POLICY']==ROOT.parent/'policy' and ns['CameraStream'] is runtime.CameraStream
  checks[name+'_actual_AST_compiles']=callable(ns['main'])
 checks['capture_function_globals_newROOT']=runtime._capture.__globals__['ROOT']==ROOT and runtime._capture.__globals__['anchor_landmarks'] is runtime.anchor_landmarks
 checks['camera_class_methods_complete_globals']=runtime.CameraStream.__init__.__globals__['ROOT']==ROOT and runtime.CameraStream.__init__.__globals__['_OpticalProxy'] is runtime._OpticalProxy
 checks['cover_methods_complete_globals']=runtime._OpticalProxy.render_eye_pair.__globals__['ROOT']==ROOT and runtime._OpticalProxy.render_eye_pair.__globals__['eye_cover_diagnostics'] is runtime.eye_cover_diagnostics
 checks['actual_camera_capture_same_code']=runtime._capture.__code__ is runtime._far._capture.__code__ and runtime.CameraStream.capture.__code__ is runtime._far.CameraStream.capture.__code__
 checks['closedcovers_same_code']=runtime._OpticalProxy.render_eye_pair.__code__ is runtime._far._OpticalProxy.render_eye_pair.__code__
 checks['loader_targeted_final1000']=1000 in runtime._inherited_load_policies.__code__.co_consts and 500 not in runtime._inherited_load_policies.__code__.co_consts and runtime.FINAL_STEP==1000
 checks['new_format_explicit']=runtime.POLICY_FORMAT=='VISION_COVERAGE_POLICY_V5'
 tree=ast.parse((runtime.POLICY/'common.py').read_text());formats=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in n.targets)]
 checks['actual_common_format_read']=formats==[runtime.POLICY_FORMAT]
 prep=adapter.entry_namespace('prepare.py')['main'];const=prep.__code__.co_consts
 checks['prepare_checkpoint1000_only']='TRAIN/checkpoint_001000.pt' in const and 'TRAIN/checkpoint_000500.pt' not in const and 'TRAIN/VERIFY_001000.json' in const and 'TRAIN/VERIFY_000500.json' not in const
 checks['prepare_new_format']=runtime.POLICY_FORMAT in const and 'VISION_FAR_LANDMARK_POLICY_V3' not in const
 pre=adapter.entry_namespace('preflight.py')['main'];checks['preflight_final1000_newformat']=1000 in pre.__code__.co_consts and 500 not in pre.__code__.co_consts and runtime.POLICY_FORMAT in pre.__code__.co_consts
 orch=adapter.entry_namespace('orchestrate.py')['main'];checks['correct_final_summary_fields']=all(key in orch.__code__.co_consts for key in ('all_declared_targets_pass','normal6_all_functional_pass','control2_all_pass'))
 checks['no_online_mirror_added']='flip(' not in (ROOT/'runtime.py').read_text() and 'mirror_rgb' not in (ROOT/'runtime.py').read_text()
 pr=json.loads((ROOT/'PREREG_PROPOSAL.json').read_text());checks['eight_conditions_final1000_declared']=pr['normal_conditions']==list(runtime.NORMAL_IDS) and pr['control_conditions']==list(runtime.CONTROL_IDS) and len(runtime.IDS)==8 and pr['policy_final_step']==1000
 checks['no_admission_created']=not any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','trials'))
 heavy={name:name in sys.modules for name in ('numpy','torch','mujoco','physics','student_wave','flygym','model')};checks['no_heavy_imports']=not any(heavy.values())
 if not all(checks.values()):raise ValueError('coverage body code-only failed '+str({k:v for k,v in checks.items() if not v}))
 paths=[ROOT/n for n in files]+[ROOT/'PREREG_PROPOSAL.json',runtime.POLICY/'common.py']+[runtime.FAR_BODY/n for n in ('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','PREREG_PROPOSAL.json','check_codeonly.py','CODEONLY_CHECK.json')]+[runtime.BODY2/'adapter.py']+[runtime.BODY1/n for n in ('runtime.py','runner.py','audit.py','summary.py')]
 with out.open('x') as f:json.dump(dict(status='COVERAGE_FINAL1000_FAR_BODY_CODE_ONLY_READY_NOT_ADMITTED',all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,Core_constructions=0,arrays_loaded=0,model_loads=0,forward_calls=0,render_calls=0,physics_steps=0,training_updates=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy,Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
