"""Actual saved normal/frozen-image matched causal review; no newmodel/physics or thresholds."""
from runtime import *
def main():
 verify();import numpy as np
 summary=read(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json');checks={}
 def check(k,v):
  checks[k]=bool(v)
  if not checks[k]:raise ValueError('Visualneck final saved review '+k)
 check('actual_newsummary_schema',summary['kind']==SUMMARY_KIND and summary['unique_physical_trials']==10 and summary['all10_integrity_verified'])
 for name in IDS:
  f=ROOT/'trials'/name;r=read(f/'RESULT.json');a=read(f/'VERIFY.json');check(name+'_independent_integrity',a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(f/'RESULT.json'));check(name+'_source_identity',r['visual_checkpoint_sha256']==VISION_SHA and r['brain_checkpoint_sha256']==read(ROOT/'INPUT_BINDINGS.json')['brain_sha256'] and r['neck_source_binding_sha256']==BINDING_SHA)
 for p,h in summary['source_hashes'].items():check('summary_source:'+p,sha(p)==h)
 f0=ROOT/'trials/V_left_to_right';f1=ROOT/'trials/V_flip_fixedframe';a=arrays(f0/'neural_decisions.npz');b=arrays(f1/'neural_decisions.npz');t0=arrays(f0/'trace_1ms.npz');t1=arrays(f1/'trace_1ms.npz');r0=read(f0/'RESULT.json');r1=read(f1/'RESULT.json');prefix=a['time']<10.62-1e-8
 check('neural_time_match',np.array_equal(a['time'],b['time']));check('qpos_exact_through_first_new_image_release',np.array_equal(t0['qpos'][:10620],t1['qpos'][:10620]));check('qvel_exact_through_first_new_image_release',np.array_equal(t0['qvel'][:10620],t1['qvel'][:10620]))
 for field in ('full_hidden_before','full_hidden_after','held_features','raw_turn','visual_raw_turn','actually_applied_action'):check(field+'_exact_before_new_image_release',np.array_equal(a[field][prefix],b[field][prefix]))
 after=a['time']>=10.62-1e-8;check('rawcamera_advances_both',np.unique(a['raw_released_image_capture_id'][after]).size>1 and np.unique(b['raw_released_image_capture_id'][after]).size>1);check('actor_images_diverge_only_control',np.all(b['image_capture_id'][after]==529) and np.all(a['image_capture_id'][after]>=530))
 check('motor_actions_match_before_first_new_DN_consumption',np.array_equal(a['actually_applied_action'][a['time']<10.625-1e-8],b['actually_applied_action'][b['time']<10.625-1e-8]))
 for label,records in (('normal',a),('fixed',b)):
  index=int(round(10.625/.005));check(label+'_first_new_visual_DN_release_and_consume',records['DN_capture_step'][index]==10620 and records['DN_release_step'][index]==10622 and records['physical_step_ms'][index]==10625)
 difference=float(np.degrees(r0['yaw_change_rad']-r1['yaw_change_rad']));check('descriptive_final_difference_exact',difference==summary['fixedframe_comparison']['normal_minus_frozen_final_yaw_deg'])
 report=dict(status='ACTUAL_RGB_SOURCE_NECK_TENCASE_INTEGRITY_AND_FIXEDFRAME_COMPARISON_VERIFIED',integrity_all_pass=True,checks=checks,normal_final_yaw_deg=float(np.degrees(r0['yaw_change_rad'])),frozen_actor_final_yaw_deg=float(np.degrees(r1['yaw_change_rad'])),normal_minus_frozen_final_yaw_deg=difference,intervention_ANN_s=4.8,first_new_capture_release_ANN_s=4.82,first_new_visual_DN_release_ANN_s=4.822,first_motor_consumption_of_new_visual_DN_ANN_s=4.825,summary_sha256=sha(ROOT/'NEURAL_VISUAL_BODY_SUMMARY.json'),new_physical_trials=0,thresholds_changed=False,counts_as_additional_independent_trial=False,normal6_functional_pass=summary['normal6_all_functional_pass'],control2_pass=summary['control2_all_pass'],whole_neck_or_general_navigation_verified=False,limitation='Matched pair within this fixed bounded redfarlandmark scene; originalslip retained and cut effects descriptive')
 save(ROOT/'PARENT_READY_REVIEW.json',report);verify();print(json.dumps(dict(integrity_all_pass=True,checks=len(checks),normal6_functional_pass=report['normal6_functional_pass'],control2_pass=report['control2_pass'],normal_minus_frozen_final_yaw_deg=difference)),flush=True)
if __name__=='__main__':main()
