"""Small stdlib source/path/metadata check. No trials, arrays, model or backend read."""
from pathlib import Path
import ast,json,sys
import adapter
import contracts
HERE=Path(__file__).resolve().parent

def main():
 checks={}
 files=('adapter.py','contracts.py','render.py','plot_sync.py','check_codeonly.py')
 for name in files:
  path=HERE/name;tree=ast.parse(path.read_text());compile(tree,str(path),'exec')
  imports=[]
  for node in tree.body:
   if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
   elif isinstance(node,ast.ImportFrom):imports.append(node.module)
  checks[name+'_syntax_and_light_imports']=not any(v and v.split('.')[0] in ('torch','numpy','mujoco','physics','runtime','matplotlib','model','common') for v in imports)
 checks['physical_ROOT_exact']=contracts.ROOT==HERE.parent==Path(adapter.CONFIG['root'])
 checks['original_BASE_exact']=contracts.BASE==adapter.BASE==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
 checks['HEAD_not_relative_wrong_parent']=adapter.HEAD_AUDIT_PATH==Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json')
 checks['new_behaviorSHA_not_bfd6']=contracts.BEHAVIOR_SHA=='66f423df4f00d801b2f0a194f539baa63ce84973765694e3d3111fbeb62aa034'
 checks['B_only_new_matchednoodor']=contracts.IDS==contracts.B_IDS and contracts.B_IDS[0]=='B_noodor'
 for script in ('render.py','plot_sync.py'):
  entry=adapter.visual_entry(script);g=entry.__globals__
  checks[script+'_newROOT_HERE']=g['ROOT']==contracts.ROOT and g['HERE']==HERE
  checks[script+'_new_saved_contract']=g['audited_arrays'] is contracts.audited_arrays and g['bind_functional'] is contracts.bind_functional
  if script=='render.py':checks['render_explicitHEADBASE']=g['HEAD_AUDIT_PATH']==adapter.HEAD_AUDIT_PATH and g['BASE']==adapter.BASE
 h=dict(all_exact=True,no_reset=True,time_s=5.8,state_dims=[128,64,64,32],motor_state_dims=[128,64,64])
 r=dict(kind='STIMULUS_AB_288_V1',condition='B_left_pulse',stage='B',state_layout=[128,64,64,32],state_total=288,checkpoint_sha256=contracts.MOTOR_SHA,behavior_checkpoint_sha256=contracts.BEHAVIOR_SHA,transition=h,engineering_pass=False,teacher_calls_after_handover=0,extra_forces=False,native=dict(actual_native_steps=1232000,post_native_steps=768000))
 a=dict(integrity_all_pass=True,checks={key:True for key in contracts.CHECKS})
 contracts.validate_metadata('B_left_pulse',r,a,h);checks['newSHA_metadata_FALSE_remains_renderable']=True
 r['behavior_checkpoint_sha256']='bfd65a3896d259496114695d7f7f3550272cf4c369895a4ddb34f3ddc3352c06'
 try:contracts.validate_metadata('B_left_pulse',r,a,h)
 except ValueError:checks['old_behavior_rejected']=True
 else:checks['old_behavior_rejected']=False
 checks['no_heavy_imports']=all(v not in sys.modules for v in ('torch','numpy','mujoco','physics','runtime','common','matplotlib','model','model_io'))
 result=dict(status='CANDIDATE3_B_VISUAL_CODE_ONLY_READY',all_passed=all(checks.values()),checks=checks,source_hashes={str(HERE/name):adapter.sha(HERE/name) for name in files+('SOURCE_CONFIG.json','PREREG.json')},inherited_source_hashes=adapter.CONFIG['inherited_visual_sources'],model_loads=0,array_loads=0,forward_calls=0,Core_constructions=0,physics_steps=0,render_frames=0,plot_calls=0,trial_reads=0)
 contracts.save_new(HERE/'CODEONLY_CHECK_FINAL.json',result);print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()
