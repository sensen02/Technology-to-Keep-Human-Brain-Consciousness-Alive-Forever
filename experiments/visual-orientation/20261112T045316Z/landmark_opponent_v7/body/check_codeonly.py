"""Stdlib-only new opponent architecture/path binding and unchanged far-body loop proof."""
from pathlib import Path
import ast,json,hashlib,sys
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 out=ROOT/'CODEONLY_CHECK.json'
 if out.exists():raise FileExistsError('exclusive code-only proof')
 files=('runtime.py','adapter.py','runner.py','audit.py','prepare.py','preflight.py','summary.py','orchestrate.py','check_codeonly.py');checks={}
 for name in files:ast.parse((ROOT/name).read_text());checks[name+'_syntax']=True
 import runtime,adapter
 def nums(t):return [n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and type(n.value) in (int,float)]
 for name in ('runner.py','audit.py','summary.py'):
  ns=adapter.namespace(name);checks[name+'_new_ROOT_POLICY']=ns['ROOT']==ROOT and ns['POLICY']==ROOT.parent/'policy'
  checks[name+'_new_camera_loader']=ns['CameraStream'] is runtime.CameraStream and ns['load_policies'] is runtime.load_policies
  checks[name+'_original_absolute_VIS']=ns['VIS']==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z')
  tree=adapter._coverage_adapter._far_adapter.source(name);adapted=adapter.OpponentScope(name).visit(ast.parse(ast.unparse(tree)))
  checks[name+'_physical_numbers_unchanged']=nums(tree)==nums(adapted)
 for name in ('prepare.py','preflight.py','summary.py','orchestrate.py'):
  ns=adapter.entry_namespace(name);checks[name+'_entry_bound']=ns['ROOT']==ROOT and ns['POLICY']==ROOT.parent/'policy' and ns['CameraStream'] is runtime.CameraStream
 checks['all8conditions']=len(runtime.IDS)==8 and len(runtime.NORMAL_IDS)==6 and len(runtime.CONTROL_IDS)==2
 checks['capture_globals_newROOT']=runtime._capture.__globals__['ROOT']==ROOT and runtime._capture.__globals__['anchor_landmarks'] is runtime.anchor_landmarks
 checks['camera_methods_newROOT']=runtime.CameraStream.__init__.__globals__['ROOT']==ROOT and runtime.CameraStream.__init__.__globals__['_OpticalProxy'] is runtime._OpticalProxy
 checks['closedcover_methods_bound']=runtime._OpticalProxy.render_eye_pair.__globals__['ROOT']==ROOT and runtime._OpticalProxy.render_eye_pair.__globals__['eye_cover_diagnostics'] is runtime.eye_cover_diagnostics
 checks['actualcapturecode_unchanged']=runtime._capture.__code__ is runtime._coverage._capture.__code__ and runtime.CameraStream.capture.__code__ is runtime._coverage.CameraStream.capture.__code__
 checks['actualclosedcovercode_unchanged']=runtime._OpticalProxy.render_eye_pair.__code__ is runtime._coverage._OpticalProxy.render_eye_pair.__code__
 checks['new_model_source_explicit']="POLICY/'model.py'" in (ROOT/'runtime.py').read_text() and "VISION_FIXED_NEURAL_OPPONENT_ARCH_V7" in (ROOT/'runtime.py').read_text()
 tree=ast.parse((runtime.POLICY/'common.py').read_text());formats=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in n.targets)]
 checks['actual_new_common_format']=formats==['VISION_OPPONENT_POLICY_V7']
 architecture=ast.parse((runtime.POLICY/'model.py').read_text());checks['actual_new_model_no_common_import']=not any(isinstance(n,ast.ImportFrom) and n.module=='common' for n in ast.walk(architecture))
 prep=adapter.entry_namespace('prepare.py')['main'];constants=prep.__code__.co_consts
 checks['prepare_new_format_and_final1000']='VISION_OPPONENT_POLICY_V7' in constants and 'TRAIN/checkpoint_001000.pt' in constants and 'TRAIN/VERIFY_001000.json' in constants and 'TRAIN/checkpoint_000500.pt' not in constants
 pre=adapter.entry_namespace('preflight.py')['main'];checks['actualpreflight_format_final1000']='VISION_OPPONENT_POLICY_V7' in pre.__code__.co_consts and 1000 in pre.__code__.co_consts
 checks['loader_final1000']=1000 in runtime.load_policies.__code__.co_consts and 500 not in runtime.load_policies.__code__.co_consts
 orch=adapter.entry_namespace('orchestrate.py')['main'];checks['final_summary_correct_fields']=all(k in orch.__code__.co_consts for k in ('all_declared_targets_pass','normal6_all_functional_pass','control2_all_pass'))
 checks['no_online_mirror_or_rule_added']='flip(' not in (ROOT/'runtime.py').read_text() and 'mirror_rgb' not in (ROOT/'runtime.py').read_text()
 checks['no_admission_created']=not any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','trials'))
 heavy={name:name in sys.modules for name in ('numpy','torch','mujoco','physics','student_wave','flygym','model')};checks['no_heavy_imports']=not any(heavy.values())
 if not all(checks.values()):raise ValueError('opponent body code-only failed '+str({k:v for k,v in checks.items() if not v}))
 paths=[ROOT/n for n in files]+[ROOT/'PREREG_PROPOSAL.json',runtime.POLICY/'common.py',runtime.POLICY/'model.py']+[runtime.COVERAGE_BODY/n for n in ('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py','CODEONLY_CHECK.json','PREREG_PROPOSAL.json')]+[runtime.FAR_BODY/n for n in ('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','CODEONLY_CHECK.json')]+[runtime.BODY2/'adapter.py']+[runtime.BODY1/n for n in ('runtime.py','runner.py','audit.py','summary.py')]
 with out.open('x') as f:json.dump(dict(status='OPPONENT_POLICY_FAR_BODY_CODE_ONLY_READY_NOT_ADMITTED',all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,Core_constructions=0,arrays_loaded=0,model_loads=0,forward_calls=0,render_calls=0,physics_steps=0,training_updates=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy,Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
