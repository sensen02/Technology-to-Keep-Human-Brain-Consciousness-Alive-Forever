"""Stdlib-only exact loop/schema/path checks, never instantiate model, renderer or Core."""
from pathlib import Path
import ast,json,hashlib,sys,importlib.util
ROOT=Path(__file__).resolve().parent

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 out=ROOT/'CODEONLY_CHECK.json'
 if out.exists():raise FileExistsError('code-only proof exclusive; preserve previous proof')
 files=('runtime.py','adapter.py','runner.py','audit.py','summary.py','prepare.py','preflight.py','orchestrate.py','check_codeonly.py')
 checks={}
 for name in files:ast.parse((ROOT/name).read_text());checks[name+'_syntax']=True
 import runtime,adapter
 for name in ('runner.py','audit.py','summary.py'):
  tree=adapter.source(name);compile(tree,str(ROOT/name)+'::codeonly','exec');ns=adapter.namespace(name)
  checks[name+'_actual_AST_compiles']=True
  checks[name+'_explicit_far_ROOT']=ns['ROOT']==ROOT
  checks[name+'_original_camera_root']=ns['VIS']==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/visual-orientation/20261112T045316Z')
  checks[name+'_new_policy_root']=ns['POLICY']==ROOT.parent/'policy'
  checks[name+'_far_anchors_bound']=ns['anchor_landmarks'] is runtime.anchor_landmarks
  checks[name+'_far_camera_controls_bound']=ns['CameraStream'] is runtime.CameraStream and ns['audit_controls'] is runtime.audit_controls
  checks[name+'_strict_loader_bound']=ns['load_policies'] is runtime.load_policies
 checks['absolute_project_stable']=runtime.PROJECT==Path('/run/media/sensen/Data2/cell_wound_prototype')
 checks['capture_globals_far_anchors']=runtime._capture.__globals__['anchor_landmarks'] is runtime.anchor_landmarks and runtime._capture.__globals__['frame_target'] is runtime.frame_target and runtime._capture.__globals__['ROOT']==ROOT
 checks['camera_init_root_same_frozen_optics']=runtime._init.__globals__['VIS']==runtime.VIS and runtime._init.__globals__['ROOT']==ROOT
 checks['all_eight_conditions']=len(runtime.NORMAL_IDS)==6 and len(runtime.CONTROL_IDS)==2 and len(runtime.IDS)==8
 pr=json.loads((ROOT/'PREREG_PROPOSAL.json').read_text());checks['proposal_far_8_and20ms']=pr['world_landmarks']['forward']==2. and pr['world_landmarks']['angles_rad']==[.4,-.4,0.] and pr['normal_conditions']==list(runtime.NORMAL_IDS) and pr['control_conditions']==list(runtime.CONTROL_IDS)
 source=(ROOT/'runtime.py').read_text();checks['closed_two_box_geometry_explicit']="for center in cover_centers(owner.centers)" in source and "owner.cover_basis" in source and "cover_basis_from_head" in source and "[.18,.18,1.4]" in source
 checks['eye_box_clearance_gate_explicit']='min(distance)>.005' in source and "eye_cover_diagnostics(cap['camera_metadata'],centers,basis)" in source
 final_source=(ROOT/'orchestrate.py').read_text();checks['final_fields_correct']=all("result['"+key+"']" in final_source for key in ('all_declared_targets_pass','normal6_all_functional_pass','control2_all_pass')) and "r['all_functional_pass']" not in final_source
 checks['no_admission_or_trials_created']=not any((ROOT/n).exists() for n in ('INPUT_BINDINGS.json','FROZEN.json','PREFLIGHT.json','trials'))
 heavy={name:name in sys.modules for name in ('numpy','torch','mujoco','physics','student_wave','flygym','model')};checks['no_heavy_imports']=not any(heavy.values())
 if not all(checks.values()):raise ValueError('far CODE_ONLY failure '+str({k:v for k,v in checks.items() if not v}))
 paths=[ROOT/n for n in files]+[ROOT/'PREREG_PROPOSAL.json',runtime.BODY2/'adapter.py']+[runtime.BODY1/n for n in ('runtime.py','runner.py','audit.py','summary.py')]
 with out.open('x') as f:json.dump(dict(status='FAR_BODY_CODE_ONLY_READY_NOT_ADMITTED',all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in paths},heavy_imports=heavy,Core_constructions=0,arrays_loaded=0,model_loads=0,forward_calls=0,render_calls=0,physics_steps=0,training_updates=0),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=heavy,Core_constructions=0,physics_steps=0)))
if __name__=='__main__':main()
