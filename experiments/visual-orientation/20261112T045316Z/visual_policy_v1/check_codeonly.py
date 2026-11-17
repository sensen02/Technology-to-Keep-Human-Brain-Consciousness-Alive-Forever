"""Stdlib-only syntax/contract review, never imports actual numerical executables."""
from pathlib import Path
import ast,json,hashlib,sys
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 checks={};files=[HERE/n for n in ('PREREG.json','common.py','model.py','data.py','generate_data.py','preflight.py','train.py','verify_saved.py','evaluate.py','audit_validation.py','run_candidate.py')]
 for path in files:
  if path.suffix=='.py':ast.parse(path.read_text());checks[path.name+'_syntax']=True
 p=json.loads((HERE/'PREREG.json').read_text());checks['fixed500_onlyfinal']=p['training']['updates']==500 and p['training']['validation_only_step']==500 and p['training']['save_steps']==[0,250,500]
 checks['camera20ms_motor5ms']=p['timing']['camera_capture_s']==.02 and p['timing']['camera_release_delay_s']==.02 and p['timing']['neural_decision_s']==.005
 model=(HERE/'model.py').read_text();checks['held_capture_index_and_sharedCNN']='features[k//4]' in model and 'self.conv1=m.conv1' in model and 'self.conv2=m.conv2' in model
 train=(HERE/'train.py').read_text();checks['fixed500_pairedbatch_no_shuffle']='range(1,501)' in train and 'gen.choice(12,2,replace=False)' in train and 'shuffle' not in train
 checks['no_history_detach_in_model']='.detach(' not in model
 checks['no_physical_pipeline']='runner.py' not in (HERE/'run_candidate.py').read_text() and 'physical_status=\'NOT_RUN\'' in (HERE/'run_candidate.py').read_text()
 # Pure Python scene schedule helper only; its actual data module is not imported.
 tree=ast.parse((HERE/'data.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scene_at');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pure_scene_schedule','exec'),ns);cfg=dict(on_s=.6,off_s=3.6,switch_s=4.8)
 checks['scene_off_pulse_switch_boundaries']=ns['scene_at']('left_pulse',.59,cfg)=='off' and ns['scene_at']('left_pulse',3.6,cfg)=='off' and ns['scene_at']('left_to_right',4.8,cfg)=='right'
 checks['occlusion_uses_actual_frame_not_black']=ns['scene_at']('left_occluded',4.,cfg)=='occluded_left'
 checks['no_heavy_imports']=not any(k in sys.modules for k in ('torch','numpy','mujoco','model','common','physics'))
 if not all(checks.values()):raise ValueError(checks)
 path=HERE/'CODEONLY_CHECK.json'
 if path.exists():raise FileExistsError('code-only check proof already exists; use newproof if sourceschange')
 with path.open('x') as f:json.dump(dict(all_passed=True,checks=checks,source_hashes={str(p):sha(p) for p in files},Core_constructions=0,array_loads=0,model_loads=0,forward_calls=0,training_updates=0,heavy_imports=False),f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,checks=len(checks),heavy_imports=False,physics_steps=0,forward_calls=0)))
if __name__=='__main__':main()
