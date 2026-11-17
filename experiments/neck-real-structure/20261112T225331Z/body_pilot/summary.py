"""Actual six-case outputs and matched interventions; no invented functional allPASS gate."""
from runtime import *
def main():
 verify();import numpy as np
 cases={};hashes={}
 for condition in IDS:
  folder=ROOT/'trials'/condition;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json')
  if not a['integrity_all_pass'] or not all(a['checks'].values()) or a['result_sha256']!=sha(folder/'RESULT.json'):raise ValueError('All actualcase audits required')
  d=arrays(folder/'neural_decisions.npz');g=arrays(folder/'geometry_1ms.npz');cases[condition]=(r,d,g);hashes[str(folder/'RESULT.json')]=sha(folder/'RESULT.json');hashes[str(folder/'VERIFY.json')]=sha(folder/'VERIFY.json')
 base_r,base_d,base_g=cases['P_empty'];base_yaw=np.unwrap(base_g['thorax_yaw']);base_yaw-=base_yaw[5799];rows=[]
 for condition in IDS:
  r,d,g=cases[condition];e=d['time']-5.8;auto=e>=-1e-8;plateau=(e>=1.6-1e-8)&(e<2.6-1e-8);recovery=(e>=4.12-1e-8)&(e<4.62-1e-8);yaw=np.unwrap(g['thorax_yaw']);yaw-=yaw[5799];difference=np.degrees(yaw-base_yaw);sign=1 if condition!='P_right_pulse' else -1;turn=d['raw_turn'];metrics=dict(ANN_turn_RMS=float(np.sqrt(np.mean(turn[auto]**2))),plateau_signed_fraction=float(np.mean(turn[plateau]*sign>0)),plateau_turn_mean=float(np.mean(turn[plateau])),recovery_turn_RMS_descriptive=float(np.sqrt(np.mean(turn[recovery]**2))),body_effect_at4s_vs_empty_deg=float(difference[9799]),final_yaw_deg=float(np.degrees(yaw[-1])),final_body_effect_vs_empty_deg=float(difference[-1]))
  rows.append(dict(condition=condition,stable_body=r['functional_stable_body'],original_engineering_pass=r['engineering_pass'],original_slip=r['autonomous']['slip_ratio_max'],all6cycles=r['autonomous_cycles']['all_legs_completed'],native_nonfoot_peak=r['native']['post_nonfoot_peak_native'],metrics=metrics,counts_as_new_whole_neck_or_brain_walking_PASS=False))
 comparisons=[]
 normal_r,normal_d,normal_g=cases['P_left_pulse']
 for condition in ('P_left_DN_cut','P_left_AN_cut','P_left_cuezero'):
  r,d,g=cases[condition];expected_prefix=1480 if condition!='P_left_cuezero' else 0
  prefix=True if expected_prefix==0 else all(np.array_equal(d[key][:expected_prefix],normal_d[key][:expected_prefix]) for key in ('X238','raw_turn','full_hidden_before','full_hidden_after','actually_applied_action'))
  mask=(d['time']>=7.4-1e-8)&(d['time']<9.4-1e-8);normal_yaw=np.unwrap(normal_g['thorax_yaw']);actual_yaw=np.unwrap(g['thorax_yaw']);normal_yaw-=normal_yaw[5799];actual_yaw-=actual_yaw[5799]
  comparisons.append(dict(condition=condition,normal='P_left_pulse',matched_neural_prefix_before_cut_exact=prefix if expected_prefix else 'NOT_APPLICABLE_CUEZERO_VARIES_FROM_CUE_ON',turn_RMS_difference_ANN1_6to3_6=float(np.sqrt(np.mean((d['raw_turn'][mask]-normal_d['raw_turn'][mask])**2))),final_yaw_difference_from_normal_deg=float(np.degrees(actual_yaw[-1]-normal_yaw[-1])),body_yaw4s_difference_from_normal_deg=float(np.degrees(actual_yaw[9799]-normal_yaw[9799])),descriptive_only_no_causal_success_threshold=True))
 cue_r,cue_d,cue_g=cases['P_left_cuezero'];cuezero_empty_exact=all(np.array_equal(cue_d[k],base_d[k]) for k in ('X238','raw_turn','full_hidden_before','full_hidden_after','actually_applied_action')) and np.array_equal(cue_g['thorax_yaw'],base_g['thorax_yaw'])
 result=dict(kind='ARTIFICIAL_BRAIN_SOURCE_NECK_SIX_CASE_BODY_SUMMARY',rows=rows,comparisons=comparisons,cuezero_and_empty_actual_neural_yaw_exact=cuezero_empty_exact,unique_physical_trials=6,all6_integrity_verified=True,stable_body_count=sum(row['stable_body'] for row in rows),original_engineering_pass_count=sum(row['original_engineering_pass'] for row in rows),all6functionalPASS_claim=False,source_hashes=hashes,brain_sha256=BRIDGE_SHA,neck_binding_sha256=BINDING_SHA,motor_sha256=MOTOR_SHA,whole_neck_or_native_tuning_verified=False,lowlevel_basewalking_not_brain_startstop=True,no_visual_input=True)
 verify();save(ROOT/'BODY_SUMMARY.json',result);print(json.dumps(dict(integrity_all6=True,stable_body_count=result['stable_body_count'],engineering_pass_count=result['original_engineering_pass_count'],comparisons=comparisons),indent=2))
if __name__=='__main__':main()
