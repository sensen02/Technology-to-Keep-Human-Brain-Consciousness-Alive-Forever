"""Unchanged historical visual gates plus separately described neural-source cuts."""
import ast
from runtime import *

def legacy_namespace():
 # Historical summary needs2880records from1s; slice an immutable in-memory view only.
 def view_arrays(path):
  value=arrays(path)
  return {k:v[200:] for k,v in value.items()} if Path(path).name=='neural_decisions.npz' else value
 g={k:v for k,v in globals().items() if not k.startswith('_') and k not in ('main','legacy_namespace')};g.update(IDS=NORMAL_IDS,arrays=view_arrays)
 def namespace(name):
  if name!='summary.py':raise ValueError('Only original normal6 summary permitted')
  tree=ast.parse((VIS/'visual_body_v1/summary.py').read_text())
  class OutputName(ast.NodeTransformer):
   def visit_Constant(self,node):return ast.copy_location(ast.Constant('NORMAL6_SUMMARY.json'),node) if node.value=='VISUAL_BODY_SUMMARY.json' else node
  tree=OutputName().visit(tree);ast.fix_missing_locations(tree);ns=dict(g,__name__='_neckvision_historical_normal6');exec(compile(tree,str(ROOT/'summary.py')+'::originalnormal6thresholds','exec'),ns);ns.update(g);return ns
 g.update(NORMAL_IDS=NORMAL_IDS,CONTROL_IDS=CONTROL_IDS,namespace=namespace)
 tree=ast.parse((VIS/'landmark_far_v3/body/summary.py').read_text());tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='adapter')];ns=dict(g,__name__='_neckvision_historical_control2');exec(compile(tree,str(ROOT/'summary.py')+'::originalfarcontrol2thresholds','exec'),ns);ns.update(g);return ns

def main():
 verify();import numpy as np
 # Do not alter stimulus/geometry/threshold labels to inherit a passing conclusion.
 legacy_namespace()['main']();summary=read(ROOT/'VISUAL_BODY_SUMMARY.json');normal=arrays(ROOT/'trials/V_left_pulse/neural_decisions.npz');normalgeom=arrays(ROOT/'trials/V_left_pulse/geometry_1ms.npz');normalyaw=np.unwrap(normalgeom['thorax_yaw']);normalyaw-=normalyaw[5799];cutrows=[];proof=dict(summary['source_hashes'])
 for name in CUT_IDS:
  f=ROOT/'trials'/name;r=read(f/'RESULT.json');a=read(f/'VERIFY.json')
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(f/'RESULT.json'):raise ValueError('Independent cutcase source audit required')
  d=arrays(f/'neural_decisions.npz');g=arrays(f/'geometry_1ms.npz');yaw=np.unwrap(g['thorax_yaw']);yaw-=yaw[5799];prefix=all(np.array_equal(d[k][:1480],normal[k][:1480]) for k in ('X238','full_hidden_before','full_hidden_after','actually_applied_action','held_features','visual_raw_turn'));mask=(d['time']>=7.4-1e-8)&(d['time']<9.4-1e-8)
  if not prefix:raise ValueError('Matched cut prefix differs before intervention')
  cutrows.append(dict(condition=name,normal='V_left_pulse',matched_prefix_before_cut_exact=True,turn_RMS_difference_cutwindow=float(np.sqrt(np.mean((d['raw_turn'][mask]-normal['raw_turn'][mask])**2))),visual_input_RMS_difference_cutwindow=float(np.sqrt(np.mean((d['visual_raw_turn'][mask]-normal['visual_raw_turn'][mask])**2))),final_yaw_difference_deg=float(np.degrees(yaw[-1]-normalyaw[-1])),stable_body=r['functional_stable_body'],original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],descriptive_only_no_new_threshold=True))
  for filename in ('RESULT.json','VERIFY.json','capture_rgb.npz','CAPTURES.json','neural_decisions.npz'):proof[str(f/filename)]=sha(f/filename)
 # Old fixed-frame control tests input replacement, not an assumption ofzero neural output.
 normalresult=read(ROOT/'trials/V_left_to_right/RESULT.json');frozenresult=read(ROOT/'trials/V_flip_fixedframe/RESULT.json');difference=float(np.degrees(normalresult['yaw_change_rad']-frozenresult['yaw_change_rad']))
 comparison=dict(kind='ACTUAL_RGB_SOURCE_NECK_FIXEDFRAME_CAUSAL_COMPARISON',normal='V_left_to_right',control='V_flip_fixedframe',normal_minus_frozen_final_yaw_deg=difference,descriptive_only_no_new_threshold=True,normal_visual_switch_actual_camera_verified=True,frozen_actor_left_image_but_raw_scene_right=True)
 save(ROOT/'FIXED_FRAME_CAUSAL_COMPARISON.json',comparison)
 summary.update(kind=SUMMARY_KIND,scope='Actual redfarlandmark vision restoredbrain-side withsourceidentityneck transport;notnativephysiology/generalnavigation',cuts=cutrows,unique_physical_trials=10,all10_integrity_verified=True,source_hashes=proof,visual_sha256=VISION_SHA,brain_sha256=read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'],neck_binding_sha256=BINDING_SHA,motor_sha256=MOTOR_SHA,state_total=304,neural_records=3080,visual_brain_scalar_only=True,labels_not_actor_inputs=True,whole_neck_or_native_function_verified=False,fixedframe_comparison=comparison)
 save(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json',summary);verify();print(json.dumps(dict(kind=SUMMARY_KIND,normal6_all_functional_pass=summary['normal6_all_functional_pass'],control2_all_pass=summary['control2_all_pass'],cut_rows=cutrows,normal_minus_frozen_yaw_deg=difference),indent=2))
if __name__=='__main__':main()
