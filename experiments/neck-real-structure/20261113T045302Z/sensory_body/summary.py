"""Actual14case summary with unchangedvisual gates and separately preregistered engineering odor gates."""
from runtime import *
def legacy_visual_gate():
 # Immutable known visual threshold code consumes2880-row views; onlynormal6 admitted.
 import ast
 source=PREV/'summary.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='legacy_namespace');ns=dict(globals(),NORMAL_IDS=V_IDS,CONTROL_IDS=())
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/'summary.py')+'::samevisualgates','exec'),ns)
 # Use original normal6 itself to avoid requiringoldocclusion/fixedframecases not among14.
 old=VIS/'visual_body_v1/summary.py';tree=ast.parse(old.read_text())
 class Output(ast.NodeTransformer):
  def visit_Constant(self,node):return ast.copy_location(ast.Constant('VISUAL6_SUMMARY.json'),node) if node.value=='VISUAL_BODY_SUMMARY.json' else node
 tree=Output().visit(tree);ast.fix_missing_locations(tree)
 def view(path):
  z=arrays(path);return {k:v[200:] for k,v in z.items()} if Path(path).name=='neural_decisions.npz' else z
 tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='runtime')];ns=dict(globals(),__name__='_fusion_legacy_visual_thresholds',IDS=V_IDS,arrays=view);exec(compile(tree,str(ROOT/'summary.py')+'::legacyvisualnormal6','exec'),ns);ns.update(ROOT=ROOT,IDS=V_IDS,arrays=view);ns['main']();return read(ROOT/'VISUAL6_SUMMARY.json')
def main():
 verify();import numpy as np
 cases={};proof={}
 for name in IDS:
  p=ROOT/'trials'/name;r=read(p/'RESULT.json');a=read(p/'VERIFY.json')
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(p/'RESULT.json'):raise ValueError('All14actualindependentauditsneeded')
  cases[name]=(r,arrays(p/'neural_decisions.npz'),arrays(p/'geometry_1ms.npz'));proof.update({str(p/n):sha(p/n) for n in ('RESULT.json','VERIFY.json')})
 visual=legacy_visual_gate();base=cases['O_noodor'];byaw=np.unwrap(base[2]['thorax_yaw']);byaw-=byaw[5799];vyaw=np.unwrap(cases['V_empty'][2]['thorax_yaw']);vyaw-=vyaw[5799];rows=[]
 for name in IDS:
  r,d,g=cases[name];elapsed=d['time']-5.8;early=(elapsed>=.82-1e-8)&(elapsed<1.82-1e-8);auto=elapsed>=-1e-8;desired=-1 if name in ('O_right','F_right_agree','O_left_swap') else 1;yaw=np.unwrap(g['thorax_yaw']);yaw-=yaw[5799];delta=np.degrees(yaw-byaw);signfraction=float(np.mean(d['raw_turn'][early]*desired>0));metrics=dict(early_turn_sign_fraction=signfraction,early_turn_mean=float(d['raw_turn'][early].mean()),early_signed_turn_mean=float((d['raw_turn'][early]*desired).mean()),ANN_turn_RMS=float(np.sqrt(np.mean(d['raw_turn'][auto]**2))),body_effect_vs_Onoodor_at2_6_deg=float(delta[8399]),signed_body_effect_at2_6_deg=float(desired*delta[8399]),early_yaw_delta_vs_Onoodor_deg=float(delta[7619]-delta[6619]),final_yaw_deg=float(np.degrees(yaw[-1])));gate=None;gatechecks={}
  if name=='O_noodor':gatechecks=dict(neutral_output_RMS_le_02=metrics['ANN_turn_RMS']<=.02,neutral_yaw_vs_Vempty_le2deg=abs(float(np.degrees(yaw[-1]-vyaw[-1])))<=2.)
  elif name in ('O_left','O_right','O_left_swap','F_left_agree','F_right_agree','F_left_DN_cut'):gatechecks=dict(early_sign_fraction_ge_9=signfraction>=.9,signed_bodyeffect2_6_ge_025deg=metrics['signed_body_effect_at2_6_deg']>=.25,stable_body=r['functional_stable_body'])
  elif name=='O_left_odorcut':
   normal=cases['O_left'];normalyaw=np.unwrap(normal[2]['thorax_yaw']);normalyaw-=normalyaw[5799];window=(elapsed>=2.1-1e-8)&(elapsed<2.6-1e-8);recovery=float(np.sqrt(np.mean(d['raw_turn'][window]**2)));shift=float(np.degrees(yaw[8399]-normalyaw[8399]));metrics.update(odorcut_recovery_RMS_05to1s_after_cut=recovery,odorcut_minus_normal_yaw2_6_deg=shift);gatechecks=dict(cut_recovery_RMS_le_05=recovery<=.05,paired_effect_magnitude_ge_025deg=abs(shift)>=.25,stable_body=r['functional_stable_body'])
  if gatechecks:gate=bool(all(gatechecks.values()))
  rows.append(dict(condition=name,metrics=metrics,orientation_engineering_checks=gatechecks,orientation_engineering_pass=gate,original_five_engineering_pass=r['engineering_pass'],original_slip=r['autonomous']['slip_ratio_max'],body_stable=r['functional_stable_body'],all6cycles=r['autonomous_cycles']['all_legs_completed'],native_nonfoot_peak=r['native']['post_nonfoot_peak_native']))
 result=dict(kind=SUMMARY_KIND,rows=rows,visual_normal6=visual,unique_physical_trials=14,all14_integrity_verified=True,source_hashes=proof,state_total=336,brain_sha256=read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'],visual_sha256=VISION_SHA,odor_sha256=ODOR_SHA,motor_sha256=MOTOR_SHA,neck_binding_sha256=BINDING_SHA,new_odor_thresholds_are_parent_preregistered_engineering_only=True,fusion_same_side_not_same_source=True,original_D_arrival_failure_not_overridden=True,no_multisensory_superiority_or_native_tuning_claim=True,full_neck_or_native_function_verified=False,food_arrival_or_parking_claim=False);save(ROOT/'SENSORY_BODY_SUMMARY.json',result);verify();print(json.dumps(dict(kind=SUMMARY_KIND,all14_integrity=True,odor_fusion_pass={r['condition']:r['orientation_engineering_pass'] for r in rows if r['orientation_engineering_pass'] is not None}),indent=2))
if __name__=='__main__':main()
